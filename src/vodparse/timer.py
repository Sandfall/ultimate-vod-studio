"""Timer localization + reading.

Locates the match clock in a frame with the timer YOLO model, crops it, and reads it
with EasyOCR via the preprocessing/parsing in `ocr_timer`. The model and OCR reader are
passed in so callers control loading/device. Shared by the detection pipeline and the
match extractor (which probes the timer to anchor game boundaries).

`TimerRecovery` / `detect_timer_with_recovery` add a fallback for frames where the detector
loses the clock (a transient overlay throwing off the YOLO box) even though the clock is still
on screen: OCR the remembered timer location directly so a match's reading isn't dropped, cutting
its clip short. An accepted recovery re-arms the fallback, so a run of consecutive dropped frames
is recovered as long as the clock keeps plausibly ticking down (chaining); a junk reading (out of
range, risen, or frozen) fails that test, is thrown out, and counts toward giving up."""

from collections import deque
from collections.abc import Sequence
from statistics import median

import cv2
import numpy as np

from vodparse.bbox import crop_frame
from vodparse.ocr_timer import preprocess_variants, parse_time
from vodparse.schema import TimerReading

MAX_CLOCK_SECONDS        = 420   # 7:00: a timer read outside [0, this] is an OCR mistake
TIMER_DROP_TOLERANCE_SEC = 5     # a recovered read may exceed elapsed wall time by this much (jitter)
TIMER_RECOVERY_FRAMES    = 2     # give up after this many consecutive frames with no recoverable clock (0 disables)
TIMER_BBOX_WINDOW        = 8     # remember this many recent detector boxes for the median location
TIMER_RECOVERY_MAX_CHAIN = 6     # cap on consecutive recovered reads: a backstop so a plausibly-falling
                                 # false clock at the remembered box can't extend a clip indefinitely
MAX_RECORDED_REJECTIONS  = 20000 # backstop so a pathological VOD can't grow the debug rejection log
                                 # unbounded (the writer reports the count, so hitting it is visible)


def _record_rejection(rejections: list | None, record: dict) -> None:
    """Append one implausible/garbage timer reading to the per-run debug log (capped). Callers pass
    `rejections=None` to opt out: recording is purely a debug side-channel, never required."""
    if rejections is not None and len(rejections) < MAX_RECORDED_REJECTIONS:
        rejections.append(record)


def read_timer_at_bbox(ocr, frame: np.ndarray, bbox: Sequence[float] | Sequence[int], *,
                       confidence: float,
                       rejections: list | None = None, sample_ts: float | None = None,
                       stage: str = "ocr_parse") -> TimerReading | None:
    """OCR the match clock at a known bounding box (source-video pixels [x1, y1, x2, y2]) and parse
    it. Shared by detect_timer (box from the YOLO detector) and the dropped-read recovery path (box
    remembered from recent detections). Returns None if the crop is empty or nothing parses.

    Pass `rejections` (a list) to record garbage: when a box was localized but no variant yields a
    parseable clock, the raw OCR text + a reason are appended for the debug/rejected_timers.csv dump.
    `sample_ts`/`stage` tag the row; an empty crop (no box) is not recorded: that's a non-detection,
    not a misread."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    # Clamped crop: a detector/median box nudged past a frame edge reads the visible part instead
    # of numpy-wrapping to the wrong region. The reading keeps the unclamped box so TimerRecovery's
    # median memory sees what the detector actually reported.
    crop = crop_frame(frame, (x1, y1, x2, y2))
    if crop is None:
        return None
    seen_raw: list[str] = []   # distinct non-empty OCR strings, kept only to explain a rejection
    for _, proc in preprocess_variants(crop):
        # The crop is already the localized timer, so skip CRAFT text *detection* (the
        # expensive half of readtext) and run the recognizer directly. With no box list,
        # recognize() treats the whole crop as one horizontal line; the variants are
        # already single-channel, which is what it expects.
        # Each EasyOCR result is (bbox_points, text, confidence). Join the confident
        # pieces left-to-right (sorted by the bbox's leftmost x) into one string.
        raw = ocr.recognize(proc, allowlist="0123456789:. ", paragraph=False)
        raw.sort(key=lambda r: r[0][0][0])
        combined = "".join(text for _bbox, text, conf in raw if conf > 0.2)
        if combined and combined not in seen_raw:
            seen_raw.append(combined)
        parsed = parse_time(combined)
        if parsed:
            return {**parsed, "bbox": [x1, y1, x2, y2], "confidence": round(confidence, 3)}
    # Nothing parsed across every preprocessing variant. low_confidence: every variant came back
    # empty (all OCR pieces below the 0.2 floor); unparseable: text was read but no clock pattern
    # matched (or seconds >= 60). The raw strings (joined) let you tell which after the fact.
    _record_rejection(rejections, {
        "stage": stage,
        "rejection_reason": "unparseable" if seen_raw else "low_confidence",
        "timestamp_sec": sample_ts,
        "raw_ocr_string": "|".join(seen_raw),
        "parsed_value": None,
        "prev_accepted_value": None,
        "prev_accepted_ts": None,
        "bbox": [x1, y1, x2, y2],
    })
    return None


def detect_timer(timer_model, ocr, frame: np.ndarray, device: str,
                 infer_size: int = 640, *,
                 rejections: list | None = None, sample_ts: float | None = None) -> TimerReading | None:
    """Localize the match clock with the YOLO detector and OCR it (via read_timer_at_bbox).

    Only detections whose center sits in the top-right quadrant of the frame are considered:
    the in-game timer is always top-center/top-right, while lower boxes are post-game results
    screens (e.g. "Out at 1:44") that share a similar digit pattern and would otherwise be read
    as a clock. The highest-confidence surviving box is cropped and read. Returns None when no
    box is detected, none falls in the top-right quadrant, or nothing parses."""
    h, w = frame.shape[:2]
    scale = infer_size / max(h, w)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    results = timer_model(small, conf=0.3, verbose=False, device=device)
    if not results[0].boxes:
        return None
    # Only consider detections whose center is in the top-right quarter of the frame.
    # The in-game timer is always top-center/top-right; lower detections are post-game
    # results screens (e.g. "Out at 1:44") which share a similar digit pattern.
    sh, sw = small.shape[:2]
    top_right = [b for b in results[0].boxes
                 if b.xywh[0][0].item() > sw / 2 and b.xywh[0][1].item() < sh / 2]
    if not top_right:
        return None
    best_box = max(top_right, key=lambda b: b.conf.item())
    x1, y1, x2, y2 = (int(c / scale) for c in best_box.xyxy[0].tolist())
    return read_timer_at_bbox(ocr, frame, [x1, y1, x2, y2], confidence=best_box.conf.item(),
                              rejections=rejections, sample_ts=sample_ts, stage="ocr_parse_detector")


class TimerRecovery:
    """Per-run state for recovering dropped timer reads. The clock sits in a near-fixed spot, so we
    remember recent *detector* boxes; when the detector misses, OCR the median box directly and accept
    a reading only if it plausibly continues the countdown. An accepted recovery re-arms the window, so
    a run of consecutive dropped frames is recovered as long as the clock keeps falling, but only a
    read that passes `accept` chains; a junk read (out of range, risen, or frozen) is thrown out and
    counts toward giving up, and a chain continuing a previous recovery must *strictly* fall (a frozen
    or jittered-up reading can't sustain it). `max_chain` caps one uninterrupted chain as a backstop.
    Recovered reads never feed the box history, so the remembered location stays anchored to real
    detections and recovery never drifts into the next game."""

    def __init__(self, recovery_frames: int = TIMER_RECOVERY_FRAMES, window: int = TIMER_BBOX_WINDOW,
                 max_chain: int = TIMER_RECOVERY_MAX_CHAIN):
        self.recovery_frames = recovery_frames
        self.max_chain = max_chain
        self._bboxes: deque[list[int]] = deque(maxlen=window)
        self.detector_misses = 0
        self.recovery_run = 0                       # consecutive recovered reads since the last detector hit
        self.last_secs: float | None = None
        self.last_ts: float | None = None
        self.last_was_recovered = False             # provenance of the current anchor (detector vs recovery)
        self.rejections: list[dict] = []            # implausible/garbage reads dropped this run (debug log)

    def _reject(self, reason: str, total_seconds: float, ts: float, bbox) -> None:
        """Record a recovery reading `accept` refused (see below) so debug/rejected_timers.csv can show
        what the fallback threw out and why. Pure side-channel: state is left untouched."""
        _record_rejection(self.rejections, {
            "stage": "recovery_accept",
            "rejection_reason": reason,
            "timestamp_sec": ts,
            "raw_ocr_string": "",
            "parsed_value": total_seconds,
            "prev_accepted_value": self.last_secs,
            "prev_accepted_ts": self.last_ts,
            "bbox": list(bbox) if bbox is not None else None,
        })

    def record_hit(self, bbox, total_seconds: float, ts: float) -> None:
        """A real detector reading: remember its box, clear the miss/chain streaks, set the anchor."""
        self._bboxes.append([int(v) for v in bbox])
        self.detector_misses = 0
        self.recovery_run = 0
        self.last_secs = total_seconds
        self.last_ts = ts
        self.last_was_recovered = False

    def note_miss(self) -> bool:
        """Count a detector miss; return True if the remembered box should be OCR'd this frame. An
        accepted recovery resets the counter (see accept), so it bounds *consecutive un-recovered*
        frames: a chain of good recoveries keeps the fallback armed, a real gap exhausts it."""
        self.detector_misses += 1
        return bool(self._bboxes) and self.detector_misses <= self.recovery_frames

    def median_bbox(self) -> list[int]:
        """The per-coordinate median of the remembered boxes: robust to a one-off misdetection."""
        return [int(median(coord)) for coord in zip(*self._bboxes)]

    def accept(self, total_seconds: float, ts: float, bbox=None) -> bool:
        """If `total_seconds` plausibly continues the countdown, record it as the new anchor, re-arm the
        recovery window, and return True; otherwise leave state untouched and return False (a junk read,
        thrown out: and logged to self.rejections). Plausible means: in range; within the per-chain cap;
        and fallen a believable amount since the last accepted read. A read continuing a *previous
        recovery* must strictly fall, so a frozen or jittered-up reading can't sustain a chain; the first
        recovery after a real detection tolerates OCR jitter in either direction: a flat reading, or one
        up to TIMER_DROP_TOLERANCE_SEC above the last value, still counts across a single dropped frame (a
        next game's clock resets far higher than that, so it can't slip through this window)."""
        if not (0 <= total_seconds <= MAX_CLOCK_SECONDS):
            self._reject("out_of_range", total_seconds, ts, bbox)
            return False
        if self.recovery_run >= self.max_chain:     # backstop: an uninterrupted chain can't run forever
            self._reject("chain_cap_exceeded", total_seconds, ts, bbox)
            return False
        if self.last_secs is not None:
            # `drop` is positive when the clock fell (total < last), negative when it rose. A valid
            # continuation falls by no more than the wall time elapsed (+ jitter): a chained recovery
            # must strictly fall (drop > 0); a first recovery may read flat or jitter slightly up.
            drop    = self.last_secs - total_seconds
            elapsed = ts - self.last_ts if self.last_ts is not None else 0.0
            lower_ok = drop > 0 if self.last_was_recovered else drop >= -TIMER_DROP_TOLERANCE_SEC
            if not (lower_ok and drop <= elapsed + TIMER_DROP_TOLERANCE_SEC):
                # Too-fast a fall vs. wall time is an implausible_drop; anything else here is a rise (or
                # a frozen reading on a chain that must strictly fall): an implausible_rise.
                reason = "implausible_drop" if drop > elapsed + TIMER_DROP_TOLERANCE_SEC else "implausible_rise"
                self._reject(reason, total_seconds, ts, bbox)
                return False
        self.last_secs = total_seconds
        self.last_ts = ts
        self.last_was_recovered = True
        self.detector_misses = 0                    # an accepted recovery re-arms the window → chaining
        self.recovery_run += 1
        return True


def detect_timer_with_recovery(state: TimerRecovery, timer_model, ocr, frame: np.ndarray, device: str,
                               infer_size: int, sample_ts: float) -> TimerReading | None:
    """detect_timer, plus a dropped-read fallback: when the detector misses but recently had the
    clock, OCR the remembered median box and accept a plausible countdown continuation (flagged
    `recovered=True`). An accepted recovery re-arms the fallback, so it chains across consecutive
    dropped frames while the clock keeps falling: bounded by `recovery_frames` consecutive failures
    and the per-chain cap. See TimerRecovery."""
    timer = detect_timer(timer_model, ocr, frame, device, infer_size,
                         rejections=state.rejections, sample_ts=sample_ts)
    if timer is not None:
        state.record_hit(timer["bbox"], timer["total_seconds"], sample_ts)
        return timer
    if state.note_miss():
        recovered = read_timer_at_bbox(ocr, frame, state.median_bbox(), confidence=0.0,
                                       rejections=state.rejections, sample_ts=sample_ts,
                                       stage="ocr_parse_recovery")
        if recovered is not None and state.accept(recovered["total_seconds"], sample_ts,
                                                  bbox=recovered.get("bbox")):
            recovered["recovered"] = True
            return recovered
    return None
