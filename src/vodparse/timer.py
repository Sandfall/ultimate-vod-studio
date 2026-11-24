"""Timer localization + reading.

Locates the match clock in a frame with the timer YOLO model, crops it, and reads it
with EasyOCR via the preprocessing/parsing in `ocr_timer`. The model and OCR reader are
passed in so callers control loading/device. Shared by the detection pipeline and the
match extractor (which probes the timer to anchor game boundaries).

Where a game draws its clock and how its digits come back from the recognizer differ per
game (Ultimate top-right with a literal colon; Melee and Rivals 2 top-center, colon often
returned as "."), so the localize/read path takes the game's `vodparse.games.ClockHud`:
it filters the detector's boxes to the game's clock span and sets the parse rules.

`TimerRecovery` / `detect_timer_with_recovery` add a fallback for frames where the detector
loses the clock (a transient overlay throwing off the YOLO box) even though the clock is still
on screen: OCR the remembered timer location directly so a match's reading isn't dropped, cutting
its clip short. An accepted recovery re-arms the fallback, so a run of consecutive dropped frames
is recovered as long as the clock keeps plausibly ticking down (chaining); a junk reading (out of
range, risen, or frozen) fails that test, is thrown out, and counts toward giving up."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from vodparse.bbox import crop_frame
from vodparse.games import ClockHud
from vodparse.layout_reads import pad_box
from vodparse.match_extraction import FULL_GAME_SECONDS
from vodparse.ocr_timer import MISREAD_MAP, preprocess_variants, parse_time
from vodparse.schema import ParsedTime, TimerReading

if TYPE_CHECKING:   # annotation-only: the model and reader are loaded by the callers
    import easyocr
    from ultralytics import YOLO

CLOCK_CAP_TOLERANCE_SEC  = 5     # OCR can read a full clock a few seconds high (e.g. a real 7:00
                                 # as 7:02); same magnitude as TIMER_DROP_TOLERANCE_SEC below
TIMER_DROP_TOLERANCE_SEC = 5     # a recovered read may exceed elapsed wall time by this much (jitter)
TIMER_RECOVERY_FRAMES    = 2     # give up after this many consecutive frames with no recoverable clock (0 disables)
TIMER_BBOX_WINDOW        = 8     # remember this many recent detector boxes for the median location
TIMER_RECOVERY_MAX_CHAIN = 6     # cap on consecutive recovered reads: a backstop so a plausibly-falling
                                 # false clock at the remembered box can't extend a clip indefinitely
MAX_RECORDED_REJECTIONS  = 20000 # backstop so a pathological VOD can't grow the debug rejection log
                                 # unbounded (the writer reports the count, so hitting it is visible)
OCR_PIECE_CONF_FLOOR     = 0.2   # recognizer pieces at or below this confidence are dropped before parsing
TIMER_DETECT_CONF        = 0.3   # YOLO confidence floor for a clock box (one big, high-contrast object)
READ_TRACE_NAME          = "timer_read_trace.txt"   # per-variant trace written into a read's debug_dir


def max_plausible_clock(full_game_seconds: float) -> float:
    """The highest clock reading `TimerRecovery.accept` treats as plausible for a game whose full
    clock is `full_game_seconds`: the clock itself plus `CLOCK_CAP_TOLERANCE_SEC` of OCR jitter
    headroom, so a real full clock is never rejected as out-of-range."""
    return full_game_seconds + CLOCK_CAP_TOLERANCE_SEC


def _record_rejection(rejections: list[dict] | None, record: dict) -> None:
    """Append one implausible/garbage timer reading to the per-run debug log (capped). Callers pass
    `rejections=None` to opt out: recording is purely a debug side-channel, never required."""
    if rejections is not None and len(rejections) < MAX_RECORDED_REJECTIONS:
        rejections.append(record)


# A user-drawn timer box (a custom overlay layout's, vodparse.overlay_layout) is parsed with
# BOTH of parse_time's accommodations on, whatever the game: the user has said where the clock
# is, so the parser accepts every clock shape the games draw (a dot for the colon, unseparated
# centiseconds) and a false parse inside that box is unlikely. The pipeline's drawn-box read
# (cli.pipeline._sample_timer) and the editor's test read (cli.overlay_read.read_timer_region)
# both use these, so the editor previews exactly what analysis reads. The detector and
# calibrated-box paths use the game's own profile instead (games.ClockHud).
DRAWN_BOX_DOT_AS_COLON = True
DRAWN_BOX_UNSEPARATED_CENTISECONDS = True


def read_timer_at_bbox(ocr: easyocr.Reader, frame: np.ndarray,
                       bbox: Sequence[float] | Sequence[int], *,
                       confidence: float,
                       rejections: list[dict] | None = None, sample_ts: float | None = None,
                       stage: str = "ocr_parse",
                       debug_dir: Path | None = None,
                       dot_as_colon: bool = False,
                       unseparated_centiseconds: bool = False,
                       max_variants: int | None = None) -> TimerReading | None:
    """OCR the match clock at a known bounding box (source-video pixels [x1, y1, x2, y2]) and parse
    it. Shared by detect_timer (box from the YOLO detector) and the dropped-read recovery path (box
    remembered from recent detections). Returns None if the crop is empty or nothing parses.

    Pass `rejections` (a list) to record garbage: when a box was localized but no variant yields a
    parseable clock, the raw OCR text + a reason are appended for the debug/rejected_timers.csv dump.
    `sample_ts`/`stage` tag the row; an empty crop (no box) is not recorded: that's a non-detection,
    not a misread. `debug_dir` (a directory) makes preprocess_variants write every variant image
    it OCRs there (crop_<variant>.jpg) and adds a per-variant trace of the selection
    (READ_TRACE_NAME: what the recognizer returned, what was parsed, which variant won), so
    what the recognizer actually saw can be inspected. `dot_as_colon` lets "7.28" parse as
    7:28 and `unseparated_centiseconds` lets "7:2884" parse as 7:28.84 (ocr_timer.parse_time):
    the game's clock profile (ClockHud) sets both on the detector path, and a user-drawn timer
    box turns both on regardless of game; Ultimate's detector path leaves both off.

    `max_variants` caps how many preprocessing variants are tried, in preprocess_variants'
    order (the first is the one that reads most real clocks); None, the default and what
    every pipeline read uses, tries them all. Each variant is one recognizer call, so a
    benchmark or tool can cap the sweep to measure a subset."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    # Clamped crop: a detector/median box nudged past a frame edge reads the visible part instead
    # of numpy-wrapping to the wrong region. The reading keeps the unclamped box so TimerRecovery's
    # median memory sees what the detector actually reported.
    crop = crop_frame(frame, (x1, y1, x2, y2))
    if crop is None:
        return None
    seen_raw: list[str] = []   # distinct non-empty OCR strings, kept only to explain a rejection
    trace: list[str] = []      # one line per variant tried, for the debug trace file
    variants = preprocess_variants(crop, debug_dir)
    if max_variants is not None:
        variants = variants[:max_variants]
    for name, proc in variants:
        # The crop is already the localized timer, so skip CRAFT text *detection* (the
        # expensive half of readtext) and run the recognizer directly. With no box list,
        # recognize() treats the whole crop as one horizontal line; the variants are
        # already single-channel, which is what it expects.
        # Each EasyOCR result is (bbox_points, text, confidence). Join the confident
        # pieces left-to-right (sorted by the bbox's leftmost x) into one string.
        raw = ocr.recognize(proc, allowlist="0123456789:. ", paragraph=False)
        raw.sort(key=lambda r: r[0][0][0])
        combined = "".join(text for _bbox, text, conf in raw if conf > OCR_PIECE_CONF_FLOOR)
        if combined and combined not in seen_raw:
            seen_raw.append(combined)
        parsed = parse_time(combined, dot_as_colon=dot_as_colon,
                            unseparated_centiseconds=unseparated_centiseconds)
        if debug_dir is not None:
            trace.append(_trace_line(name, raw, combined, parsed))
        if parsed:
            if debug_dir is not None:
                _write_read_trace(debug_dir, [x1, y1, x2, y2], crop, trace,
                                  f"parsed {parsed['formatted']} from variant {name}; "
                                  "later variants were not tried", dot_as_colon,
                                  unseparated_centiseconds)
            return {**parsed, "bbox": [x1, y1, x2, y2], "confidence": round(confidence, 3)}
    if debug_dir is not None:
        _write_read_trace(debug_dir, [x1, y1, x2, y2], crop, trace, "no variant parsed as a clock",
                          dot_as_colon, unseparated_centiseconds)
    # Nothing parsed across every preprocessing variant. low_confidence: every variant came back
    # empty (all OCR pieces at or below OCR_PIECE_CONF_FLOOR); unparseable: text was read but no
    # clock pattern matched (or seconds >= 60). The raw strings (joined) let you tell which after
    # the fact.
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


def _trace_line(variant: str, raw: list, combined: str, parsed: ParsedTime | None) -> str:
    """One trace line for a preprocessing variant: every recognizer piece with its confidence
    (marking the ones the floor dropped), the joined string, the string after the misread map,
    and the parse outcome."""
    if raw:
        pieces = "  ".join(
            f"'{text}'@{conf:.2f}" + ("" if conf > OCR_PIECE_CONF_FLOOR else " (dropped: below floor)")
            for _bbox, text, conf in raw)
    else:
        pieces = "(recognizer returned nothing)"
    cleaned = combined.translate(MISREAD_MAP)
    outcome = f"parsed {parsed['formatted']}" if parsed else "no clock pattern"
    return (f"{variant:<10} pieces: {pieces}\n"
            f"{'':<10} joined: '{combined}'   after misread map: '{cleaned}'   -> {outcome}")


def _write_read_trace(directory: Path, bbox: list[int], crop: np.ndarray, lines: list[str],
                      outcome: str, dot_as_colon: bool, unseparated_centiseconds: bool) -> None:
    """Write the per-variant selection trace of one read as READ_TRACE_NAME in ``directory``
    (overwriting). Best-effort: a trace failure must never break the read."""
    crop_h, crop_w = crop.shape[:2]
    header = [
        "Timer read trace: variants are tried in this order and the first one whose text parses",
        "as a clock wins. Pieces are the recognizer's results left to right; pieces at or below",
        f"the {OCR_PIECE_CONF_FLOOR} confidence floor are dropped before joining.",
        f"box: {bbox}   crop: {crop_w}x{crop_h} px   dot-as-colon fallback: "
        + ("on" if dot_as_colon else "off")
        + "   unseparated-centiseconds fallback: "
        + ("on" if unseparated_centiseconds else "off"),
        "",
    ]
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / READ_TRACE_NAME).write_text(
            "\n".join(header + lines + ["", f"result: {outcome}", ""]), encoding="utf-8")
    except OSError:
        pass


def localize_timer(timer_model: YOLO, frame: np.ndarray, device: str, infer_size: int = 640, *,
                   clock_hud: ClockHud,
                   conf: float = TIMER_DETECT_CONF) -> tuple[list[int], float] | None:
    """Find the match clock's box with the YOLO detector, without reading it: the frame is
    scaled so its long side is `infer_size`, the detector runs at `conf`, and only boxes whose
    center falls in the game's clock span (`clock_hud.contains`) count: a post-game results
    screen ("Out at 1:44") or a stream scoreboard carries clock-like digits elsewhere on the
    frame and would otherwise be read as a live clock. Returns the highest-confidence
    surviving box in source-frame pixels ([x1, y1, x2, y2]) with its confidence, or None when
    nothing was detected inside the span. The read half is `detect_timer`; tools that only
    need the box (cutout harvest, detector A/B) call this directly."""
    h, w = frame.shape[:2]
    scale = infer_size / max(h, w)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    # Typed Any on purpose: ultralytics' own result types don't index or iterate cleanly
    # (see the note above `detect_logos` in cli/pipeline.py).
    results: Any = timer_model(small, conf=conf, verbose=False, device=device)
    if not results[0].boxes:
        return None
    sh, sw = small.shape[:2]
    in_span = [b for b in results[0].boxes
               if clock_hud.contains(b.xywh[0][0].item() / sw, b.xywh[0][1].item() / sh)]
    if not in_span:
        return None
    best_box = max(in_span, key=lambda b: b.conf.item())
    x1, y1, x2, y2 = (int(c / scale) for c in best_box.xyxy[0].tolist())
    return [x1, y1, x2, y2], float(best_box.conf.item())


def detect_timer(timer_model: YOLO, ocr: easyocr.Reader, frame: np.ndarray, device: str,
                 infer_size: int = 640, *, clock_hud: ClockHud,
                 rejections: list[dict] | None = None,
                 sample_ts: float | None = None) -> TimerReading | None:
    """Localize the match clock with the YOLO detector (`localize_timer`, filtered to the
    game's clock span) and OCR it (`read_timer_at_bbox`, with the game's digit rules:
    `clock_hud.dot_as_colon` / `unseparated_centiseconds`). Returns None when no box is
    detected, none falls in the span, or nothing parses."""
    located = localize_timer(timer_model, frame, device, infer_size, clock_hud=clock_hud)
    if located is None:
        return None
    bbox, confidence = located
    return read_timer_at_bbox(ocr, frame, bbox, confidence=confidence,
                              rejections=rejections, sample_ts=sample_ts, stage="ocr_parse_detector",
                              dot_as_colon=clock_hud.dot_as_colon,
                              unseparated_centiseconds=clock_hud.unseparated_centiseconds)


CALIBRATED_TIMER_STAGE = "ocr_parse_calibrated"
"""Rejection stage for a read of the run's calibrated clock box (clock_calibration) that did
not parse, so debug/rejected_timers.csv tells it apart from a detector or recovery read."""


def read_calibrated_clock(ocr: easyocr.Reader, frame: np.ndarray, bbox: Sequence[int], *,
                          clock_hud: ClockHud, max_clock_seconds: float,
                          rejections: list[dict] | None = None,
                          sample_ts: float | None = None) -> TimerReading | None:
    """Read the run's calibrated clock box (vodparse.clock_calibration) on one frame, the
    way a user-drawn layout box is read: padded like a drawn box, the game's digit rules,
    and any in-range parse accepted, rises included. It deliberately bypasses
    TimerRecovery's rise rejection: that machinery re-finds a box the detector LOST a few
    frames ago and must reject a rising read so junk cannot chain into the next game; the
    calibrated box is the VOD's proven clock spot, so a rising read there is a new game
    starting, and extract already handles a fixed box's stream of reads (`valid_secs`,
    `_clean_timer_frames`). The caller still hands an accepted read to
    `TimerRecovery.note_anchor`, so a later recovery read is judged against this clock
    and not against whatever the detector last saw (cli.pipeline._sample_timer).
    The reading is flagged ``calibrated``; its confidence is 0.0 (no detector scored it), as
    a recovered read's is. Out-of-range parses are dropped and logged under the stage.

    The full preprocessing sweep is tried, as the detector path tries it. This read runs on
    every sample, clock or not, so capping the sweep at its first two variants looked like
    a saving for the clock-less ones; measured on two of our own captures (2026-09-22, an
    Ultimate capture and a Rivals 2 Gauntlet) it saved nothing, because every real clock
    the short sweep missed fell through to the detector path and its own full sweep."""
    reading = read_timer_at_bbox(ocr, frame, pad_box(list(bbox)), confidence=0.0,
                                 rejections=rejections, sample_ts=sample_ts,
                                 stage=CALIBRATED_TIMER_STAGE,
                                 dot_as_colon=clock_hud.dot_as_colon,
                                 unseparated_centiseconds=clock_hud.unseparated_centiseconds)
    if reading is None:
        return None
    if not (0 <= reading["total_seconds"] <= max_clock_seconds):
        _record_rejection(rejections, {
            "stage": CALIBRATED_TIMER_STAGE,
            "rejection_reason": "out_of_range",
            "timestamp_sec": sample_ts,
            "raw_ocr_string": reading.get("raw", ""),
            "parsed_value": reading["total_seconds"],
            "prev_accepted_value": None,
            "prev_accepted_ts": None,
            "bbox": list(bbox),
        })
        return None
    reading["calibrated"] = True
    return reading


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
                 max_chain: int = TIMER_RECOVERY_MAX_CHAIN, *,
                 max_clock_seconds: float = max_plausible_clock(FULL_GAME_SECONDS)):
        self.recovery_frames = recovery_frames
        self.max_chain = max_chain
        self.max_clock_seconds = max_clock_seconds  # accept() rejects any read above this (see accept)
        self._bboxes: deque[list[int]] = deque(maxlen=window)
        self.detector_misses = 0
        self.recovery_run = 0                       # consecutive recovered reads since the last detector hit
        self.last_secs: float | None = None
        self.last_ts: float | None = None
        self.last_was_recovered = False             # provenance of the current anchor (detector vs recovery)
        self.rejections: list[dict] = []            # implausible/garbage reads dropped this run (debug log)

    def _reject(self, reason: str, total_seconds: float, ts: float,
                bbox: Sequence[int] | None) -> None:
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

    def record_hit(self, bbox: Sequence[int] | Sequence[float], total_seconds: float,
                   ts: float) -> None:
        """A real detector reading: remember its box, clear the miss/chain streaks, set the anchor."""
        self._bboxes.append([int(v) for v in bbox])
        self.note_anchor(total_seconds, ts)

    def note_anchor(self, total_seconds: float, ts: float) -> None:
        """A reading the run trusts without a detector box (the calibrated clock box read by
        cli.pipeline._sample_timer): clear the miss/chain streaks and set the anchor, exactly as
        a detector hit does, but remember no box. The box history stays detector-only (a
        calibrated box is padded, and the median must not drift onto it); the anchor must
        move, though: a recovery read is judged against the LAST TRUSTED clock, and a
        detector hit from a game ago is so far back that `accept` would wave nearly any
        lower value through as a plausible fall."""
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

    def accept(self, total_seconds: float, ts: float, bbox: Sequence[int] | None = None) -> bool:
        """If `total_seconds` plausibly continues the countdown, record it as the new anchor, re-arm the
        recovery window, and return True; otherwise leave state untouched and return False (a junk read,
        thrown out: and logged to self.rejections). Plausible means: in range; within the per-chain cap;
        and fallen a believable amount since the last accepted read. A read continuing a *previous
        recovery* must strictly fall, so a frozen or jittered-up reading can't sustain a chain; the first
        recovery after a real detection tolerates OCR jitter in either direction: a flat reading, or one
        up to TIMER_DROP_TOLERANCE_SEC above the last value, still counts across a single dropped frame (a
        next game's clock resets far higher than that, so it can't slip through this window)."""
        if not (0 <= total_seconds <= self.max_clock_seconds):
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


def detect_timer_with_recovery(state: TimerRecovery, timer_model: YOLO, ocr: easyocr.Reader,
                               frame: np.ndarray, device: str, infer_size: int, sample_ts: float, *,
                               clock_hud: ClockHud) -> TimerReading | None:
    """detect_timer, plus a dropped-read fallback: when the detector misses but recently had the
    clock, OCR the remembered median box and accept a plausible countdown continuation (flagged
    `recovered=True`). An accepted recovery re-arms the fallback, so it chains across consecutive
    dropped frames while the clock keeps falling: bounded by `recovery_frames` consecutive failures
    and the per-chain cap. See TimerRecovery. The recovery read applies the same digit rules as
    the detector read (`clock_hud`): it is the same clock font at the same spot."""
    timer = detect_timer(timer_model, ocr, frame, device, infer_size, clock_hud=clock_hud,
                         rejections=state.rejections, sample_ts=sample_ts)
    if timer is not None:
        state.record_hit(timer["bbox"], timer["total_seconds"], sample_ts)
        return timer
    if state.note_miss():
        recovered = read_timer_at_bbox(ocr, frame, state.median_bbox(), confidence=0.0,
                                       rejections=state.rejections, sample_ts=sample_ts,
                                       stage="ocr_parse_recovery",
                                       dot_as_colon=clock_hud.dot_as_colon,
                                       unseparated_centiseconds=clock_hud.unseparated_centiseconds)
        if recovered is not None and state.accept(recovered["total_seconds"], sample_ts,
                                                  bbox=recovered.get("bbox")):
            recovered["recovered"] = True
            return recovered
    return None
