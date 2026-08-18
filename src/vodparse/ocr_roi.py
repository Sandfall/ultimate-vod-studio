"""Per-VOD text-region calibration for recognizer-only full-frame OCR.

The expensive half of the full-frame text pass is CRAFT text DETECTION (measured at 85-89%
of the pass on CPU), and within one VOD it keeps re-discovering the same answer: stream
overlays and in-game nameplates sit at fixed positions for the whole broadcast; only the
STRINGS change (and those change at game boundaries). The timer stage already exploits this
shape (YOLO localizes once per frame, EasyOCR runs recognizer-only on the crop); this module
generalizes it to the tag bands:

  * CALIBRATE: the first few OCR events run the normal full CRAFT pass; each band's text
    boxes go into a rolling window. Boxes recurring at the same spot form the band's STABLE
    set: the evidence that this band's layout is cacheable at all.
  * STEADY STATE: once a band has a stable set, subsequent events skip CRAFT entirely: the
    recognizer re-reads every DISTINCT box position seen across the window's recent full
    passes (newest occurrence wins; deduped by position) via easyocr's recognize(). The
    union matters: CRAFT flakes on individual nameplates frame to frame, and a box that
    happened to be missed by the latest pass must not vanish from the read set for a whole
    game. Recognizer-only on those tight boxes reproduces the full pass exactly (measured);
    a cached region whose text has since vanished or moved reads as background/other digits
    and is handled by the same confidence threshold + downstream numeric/ignored filters the
    full pass relies on.
  * REFRESH: the cache is never trusted forever. Full CRAFT passes are forced (a) every
    REFRESH_EVERY-th OCR event, (b) as a short BURST per detected new game (clock gap/reset:
    that is when strings and, rarely, layouts change; a burst rather than a single pass
    because the first clock-valid sample of a game is the intro moment, where the HUD is
    still animating in and even the full pass under-detects), and (c) when a previously
    productive band reads empty twice in a row. Every full read feeds the same rolling
    window; if it stops matching the cache, the window is cleared and calibration starts
    over (handles the rare mid-VOD layout change). A band whose window is all-empty (no
    overlay) is skipped outright.

The cache operates ONLY on clock-valid frames (the caller checks the timer before touching
should_run_full/observe_full/read_cached; timer-less frames take the normal full pass).
That is where the cacheable text lives (the in-game HUD and the overlay), and it keeps
intermission menus (character select, versus screens) from poisoning the calibration
window: a first A/B run without this rule let periodic refreshes land on menu frames,
clearing the cache against menu text and starving the tag votes of the whole late bracket.

Downstream contract: emitted boxes are the UNPADDED stable hulls (where the text has actually
been), not the padded read regions: vodparse.tags does real geometry on these quads (mirror
symmetry, same-line sponsor tests, sublabel shadows), and padded extents would distort it.
Every sampled frame still yields a text read attempt, so tag-vote cardinality (and the
splitter, which never reads texts at all) is unchanged; the failure mode this trades into is
a text element at a genuinely NEW position going unread until the next refresh, which the
per-game majority vote absorbs.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from vodparse.match_extraction import INTERMISSION_MIN_SAMPLES, RESET_JUMP_SEC
from vodparse.schema import TextBox
from vodparse.tags import OVERLAY_BOTTOM_FRAC, TAG_TOP_FRAC

if TYPE_CHECKING:
    import easyocr

# ── Tuning constants ───────────────────────────────────────────────────────────

CAL_FRAMES    = 4     # full-CRAFT reads a band's rolling window holds
STABLE_MIN    = 3     # a box must recur in this many window frames to count as stable
REFRESH_EVERY = 12    # periodic full-CRAFT sanity read, every Nth OCR event (~6 min at 30s)
NEW_GAME_BURST = 2    # full passes per detected new game (game-intro frames under-detect)
EMPTY_STREAK_LIMIT = 2   # productive band reading empty this many times in a row → refresh
VERIFY_KEEP_FRAC   = 0.5 # a full read matching less than this fraction of ROIs clears the window

# Read regions are EXACTLY the boxes CRAFT last produced: zero padding. Measured on a
# dense reference frame (CPU), recognizer-only keeps 11/11 baseline texts at 0px pad,
# 8/11 at 3px, and fabricates glyph-swallowing junk from 6px up ('layerone'→'Playerone'); the
# recognizer's tolerance cliff is that steep. String-length changes between games don't
# need padding either: the per-game refresh burst re-detects each game's extents before
# the cached reads of that game.
PAD_PX = 0

# Two boxes are "the same spot" when their centers agree within these fractions of the larger
# box's size: loose on purpose: CRAFT extents wobble with the rendered string.
MATCH_X_FRAC = 0.5
MATCH_Y_FRAC = 0.7


Rect = tuple[int, int, int, int]   # axis-aligned (x_min, y_min, x_max, y_max), full-frame px


def _hull(box: list[list[int]]) -> Rect:
    """Axis-aligned hull of a TextBox quad."""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def _same_spot(a: Rect, b: Rect) -> bool:
    """True when two hulls plausibly outline the same on-screen text element."""
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    dx = abs((a[0] + a[2]) - (b[0] + b[2])) / 2
    dy = abs((a[1] + a[3]) - (b[1] + b[3])) / 2
    return dx <= MATCH_X_FRAC * max(aw, bw) and dy <= MATCH_Y_FRAC * max(ah, bh)


@dataclass
class _Roi:
    """One cached text region: the newest matched box of its cluster (the current string's
    true extent: reported downstream, where tags.py geometry runs on it, AND read verbatim;
    see PAD_PX). Medians/unions of a cluster's extents were tried and fail here: overlay tags
    change strings per set at the same anchor, so mixed-extent aggregates fit no actual
    string ('PLAYERONE44' read as 'LAYERONE4Z')."""
    box: Rect             # newest matched box in the window
    read: Rect = (0, 0, 0, 0)


@dataclass
class _BandState:
    """Rolling calibration state for one horizontal band (top overlay / bottom nameplates)."""
    y0_frac: float                    # band bounds as frame-height fractions, mirroring
    y1_frac: float                    # pipeline.detect_text's crop exactly
    window: deque[list[Rect]] = field(default_factory=lambda: deque(maxlen=CAL_FRAMES))
    stable: list[Rect] = field(default_factory=list)   # boxes recurring across the window
    rois: list[_Roi] = field(default_factory=list)     # read set = the newest pass's boxes
    empty_streak: int = 0             # consecutive steady reads with zero texts

    @property
    def calibrated(self) -> bool:
        return len(self.window) == CAL_FRAMES

    @property
    def empty(self) -> bool:
        """All window frames found nothing → this VOD has no text in this band; skip it."""
        return self.calibrated and not any(self.window)

    @property
    def steady(self) -> bool:
        """Cacheable: enough full reads seen, and some text position actually recurs. A band
        whose boxes churn frame-to-frame (no stable set) keeps today's full pass forever."""
        return self.calibrated and bool(self.stable)

    def rebuild(self, frame_height: int) -> None:
        """Recompute the band's state from the rolling window. The STABLE set (positions
        recurring in >= STABLE_MIN window frames) decides cacheability and anchors the
        layout-change check. The READ set is every distinct box position across the window:
        neither filtered by stability (a flaky nameplate box must not drop out for a whole
        game) nor limited to the latest pass (which may itself have flaked on it). Each
        cluster is represented by its NEWEST matched box (see _Roi on why not an aggregate)."""
        clusters: list[tuple[Rect, set[int]]] = []   # (newest matched box, frames seen in)
        for frame_idx, boxes in enumerate(self.window):   # deque order = oldest → newest
            for box in boxes:
                for i, (rep, seen) in enumerate(clusters):
                    if _same_spot(rep, box):
                        clusters[i] = (box, seen | {frame_idx})
                        break
                else:
                    clusters.append((box, {frame_idx}))
        self.stable = [rep for rep, seen in clusters if len(seen) >= STABLE_MIN]
        rois = [_Roi(rep) for rep, _ in clusters]

        y_lo, y_hi = int(frame_height * self.y0_frac), int(frame_height * self.y1_frac)
        for roi in rois:
            x0, y0, x1, y1 = roi.box
            roi.read = (x0 - PAD_PX, max(y_lo, y0 - PAD_PX),
                        x1 + PAD_PX, min(y_hi, y1 + PAD_PX))
        # Same-row neighbours (e.g. [sponsor][gamertag]): stop each pad at the gap midpoint
        # so one region's read crop can't include the neighbour's text.
        rois.sort(key=lambda r: r.box[0])
        for left, right in zip(rois, rois[1:]):
            overlap = min(left.box[3], right.box[3]) - max(left.box[1], right.box[1])
            same_row = overlap > 0.5 * min(left.box[3] - left.box[1],
                                           right.box[3] - right.box[1])
            if same_row and right.box[0] > left.box[2]:
                mid = (left.box[2] + right.box[0]) // 2
                left.read = (left.read[0], left.read[1], min(left.read[2], mid - 1), left.read[3])
                right.read = (max(right.read[0], mid + 1), right.read[1],
                              right.read[2], right.read[3])
        self.rois = rois

    def observe(self, boxes: list[Rect], frame_height: int) -> None:
        """Feed one full-CRAFT read into the window. A calibrated band whose STABLE positions
        no longer show up (fewer than VERIFY_KEEP_FRAC re-found) has changed layout: drop the
        window and recalibrate from scratch rather than serving a stale cache. The stable set
        is the anchor here on purpose: transient gameplay text churns between full reads and
        must not be able to fake a layout change."""
        if self.steady:
            matched = sum(any(_same_spot(s, b) for b in boxes) for s in self.stable)
            if matched < VERIFY_KEEP_FRAC * len(self.stable):
                self.window.clear()
                self.stable = []
                self.rois = []
        self.window.append(boxes)
        self.empty_streak = 0
        self.rebuild(frame_height)


class OcrRoiCache:
    """Decides, per periodic-OCR event, whether to run the full CRAFT pass or serve the
    cached-region recognizer-only read, and owns the calibration state either way.

    Call order per event (see pipeline._periodic_ocr_event):
      1. note_timer(seconds, timestamp): ALWAYS, even when the OCR gate will skip the read
         or the frame is clock-less; this drives new-game detection from the clock stream.
      2. should_run_full(): only when OCR runs AND the frame is clock-valid (see module
         docstring); advances the periodic-refresh cadence.
      3. observe_full(texts, frame_height) after a clock-valid full pass, or
         read_cached(...) instead. Clock-less frames never touch the window or the cache.
    """

    def __init__(self, bottom_frac: float = OVERLAY_BOTTOM_FRAC) -> None:
        # `bottom_frac` mirrors pipeline.detect_text's bottom crop, which a shrunk capture
        # widens below the default 0.92 to reach lifted in-game nametags (see
        # tags.INGAME_TAG_BAND_FRAC_DEFAULT); the cache must cache the same region it OCRs.
        self.bands = {"top": _BandState(0.0, TAG_TOP_FRAC),
                      "bottom": _BandState(bottom_frac, 1.0)}
        self._events_since_full = 0
        self._force_remaining = 0      # pending forced full passes (new game / empty-streak)
        # Clock-transition tracking (mirrors the splitter's own thresholds).
        self._last_valid_secs: float | None = None
        self._last_valid_ts: float | None = None
        self._invalid_streak = 0

    # ── New-game detection (the refresh trigger that tracks string changes) ──

    def note_timer(self, seconds: float | None, timestamp_sec: float,
                   sample_interval_sec: float) -> None:
        """Watch the per-sample clock stream for game boundaries: a valid clock after a dead
        gap of >= INTERMISSION_MIN_SAMPLES samples (intermission), or an upward jump of more
        than RESET_JUMP_SEC between valid readings (in-place reset). Debounced so a single
        missed timer read (detector flake under --gate-ocr-on-timer) can't force a refresh.
        A replay-induced clock rewind can fire this too: the cost is one spare full pass."""
        if seconds is None:
            self._invalid_streak += 1
            return
        came_back = (self._invalid_streak >= INTERMISSION_MIN_SAMPLES
                     or (self._last_valid_ts is not None and self._last_valid_secs is not None
                         and timestamp_sec - self._last_valid_ts
                         >= INTERMISSION_MIN_SAMPLES * sample_interval_sec))
        # An upward jump vs the last valid reading marks a new game whether or not a short
        # dead gap sat in between: a 1-2 sample intermission (quick rematch) satisfies
        # neither the gap rule above nor a zero-gap requirement, and missing the transition
        # would serve the previous game's regions for most of the new one.
        jumped_up = (self._last_valid_secs is not None
                     and seconds - self._last_valid_secs > RESET_JUMP_SEC)
        if came_back or jumped_up:
            self._force_remaining = NEW_GAME_BURST
        self._invalid_streak = 0
        self._last_valid_secs, self._last_valid_ts = seconds, timestamp_sec

    # ── Full-vs-cached decision ──

    def should_run_full(self) -> bool:
        """True when this event must pay for the full CRAFT pass: any band not yet settled
        (still calibrating, or full of churning boxes that never stabilize: that band simply
        keeps today's full-pass behavior), forced refreshes pending, or the periodic sanity
        cadence coming due."""
        self._events_since_full += 1
        unsettled = any(not (b.steady or b.empty) for b in self.bands.values())
        if unsettled or self._force_remaining > 0 or self._events_since_full >= REFRESH_EVERY:
            self._force_remaining = max(0, self._force_remaining - 1)
            self._events_since_full = 0
            return True
        return False

    def observe_full(self, texts: list[TextBox], frame_height: int) -> None:
        """Feed a full pass's detections (full-frame coords, both bands mixed) into the
        per-band windows. detect_text's band crop guarantees every box lands in exactly one."""
        for band in self.bands.values():
            y_lo = frame_height * band.y0_frac
            y_hi = frame_height * band.y1_frac
            boxes = [_hull(t["bbox"]) for t in texts
                     if y_lo <= sum(p[1] for p in t["bbox"]) / len(t["bbox"]) < y_hi]
            band.observe(boxes, frame_height)

    # ── Steady-state read ──

    def read_cached(self, ocr: "easyocr.Reader", frame: np.ndarray, conf_threshold: float,
                    decoder: str = "greedy") -> list[TextBox]:
        """Recognizer-only read of every band's cached regions (CRAFT skipped: the same
        trick timer.read_timer_at_bbox uses on the clock crop). Empty bands cost nothing.
        Reported bboxes are the stable hulls, not the padded read regions (see module doc)."""
        frame_height = frame.shape[0]
        out: list[TextBox] = []
        for band in self.bands.values():
            if not band.steady:
                continue    # empty band: nothing has ever been here to read
            y_off = int(frame_height * band.y0_frac)
            grey = cv2.cvtColor(frame[y_off:int(frame_height * band.y1_frac)],
                                cv2.COLOR_BGR2GRAY)
            # easyocr horizontal_list entries are [x_min, x_max, y_min, y_max], band-local.
            boxes = [[r.read[0], r.read[2], r.read[1] - y_off, r.read[3] - y_off]
                     for r in band.rois]
            found = 0
            for roi, (_, text, conf) in zip(
                    band.rois, ocr.recognize(grey, horizontal_list=boxes, free_list=[],
                                             decoder=decoder, paragraph=False)):
                if conf < conf_threshold or not text.strip():
                    continue
                found += 1
                x0, y0, x1, y1 = roi.box
                out.append({"text": text.strip(), "confidence": round(float(conf), 3),
                            "bbox": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]})
            # A band that reliably produced text going silent means the strings moved or the
            # overlay went away: schedule a full pass to find out instead of trusting it.
            band.empty_streak = band.empty_streak + 1 if found == 0 else 0
            if band.empty_streak >= EMPTY_STREAK_LIMIT:
                band.empty_streak = 0
                self._force_remaining = max(self._force_remaining, 1)
        return out
