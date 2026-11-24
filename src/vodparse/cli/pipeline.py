"""
Detection Pipeline
------------------
Main loop that scans a VOD. By default it runs in *seek mode* (sample a frame every
--seek-every seconds, group matches by timer resets): the same mode the GUI always uses.

The legacy *go/game mode* (--seek-every 0) runs the logo detector on every frame and hands
control to on_logo_detected() to decide whether to run OCR and where to seek next. The logo
weight (weights/logo.pt) is no longer distributed, so go/game mode requires you to supply your
own; without it the pipeline exits with a clear message.

Usage:
    python -m vodparse.cli.pipeline --video other/vods/input.mp4               # seek mode (default)
    python -m vodparse.cli.pipeline --video other/vods/input.mp4 --seek-every 0  # go/game mode (needs weights/logo.pt)
"""

import contextlib
import cv2
import os
import queue
import argparse
import sys
import time
import warnings
import numpy as np
from dataclasses import dataclass, replace
from pathlib import Path

# EasyOCR hardcodes pin_memory=True in its recognition DataLoader; on MPS that
# optimization doesn't exist, so torch warns and ignores it (results unchanged).
# Scoped to that one message so nothing else is hidden.
warnings.filterwarnings("ignore", message=".*pin_memory.*not supported on MPS.*")

# Import torch BEFORE ultralytics/easyocr. Whichever import loads torch first triggers its
# thread-pool init, and ultralytics pins OMP_NUM_THREADS=1 at its own import; so if we let
# ultralytics pull torch in, torch initializes under that pin and EasyOCR's readtext can slow
# down dramatically (the OpenMP/import-order effect documented for this pipeline). Loading torch
# here first keeps the thread config (and OCR throughput) matching how it's always been. The
# name isn't used directly (device selection lives in vodparse.device); the import order is.
import torch  # noqa: F401 - imported for import-order/thread-config side effect, see above

from ultralytics import YOLO
import easyocr

from vodparse import LOGO_MODEL_PATH
from vodparse.device import resolve_device
from vodparse.games import (DEFAULT_INFER_SIZE, GAME_DEFAULT, GAMES, INGAME_BAND_NONE,
                            ULTIMATE_CLOCK_HUD, ClockHud, Game, StockHud, full_game_seconds_for,
                            game_by_id, ingame_band_engine_for, ingame_band_top_for,
                            weight_path)
from vodparse.cli.stage_io import prepare_stage_streams
from vodparse.jsonio import write_json_atomic
from vodparse.paths import output_dir, strip_converted_suffix, vod_stem
from vodparse.frame_reader import FrameReader, VideoFrameSource, open_capture
# The progress-file values and the --ingame-band-engine choices live in schema (the GUI reads
# or sends them too); pipeline.PROGRESS_STAGE_* and pipeline.CLOCK_CALIBRATION_* stay
# importable from here.
from vodparse.schema import (CLOCK_CALIBRATION_FOUND, CLOCK_CALIBRATION_NOT_FOUND,
                             CLOCK_CALIBRATION_SKIPPED, INGAME_BAND_ENGINE_OVERLAY,
                             INGAME_BAND_ENGINE_PINNED, PROGRESS_STAGE_CALIBRATING,
                             PROGRESS_STAGE_LOCATING_OVERLAY, BBox, Event, LayoutRead,
                             PipelineData, PipelineProgress, SideCharacters, StockSlot, TextBox,
                             TimerReading)
from vodparse.stock_icons import detect_stock_icons
from vodparse.timer import (DRAWN_BOX_DOT_AS_COLON, DRAWN_BOX_UNSEPARATED_CENTISECONDS,
                            TimerRecovery, detect_timer_with_recovery, localize_timer,
                            read_timer_at_bbox,
                            read_calibrated_clock, MAX_RECORDED_REJECTIONS, max_plausible_clock)
from vodparse.clock_calibration import ClockCalibration, FrameCallback, calibrate_clock_box
from vodparse.side_overlay import (SIDE_OVERLAY_SKIPPED, IngameCheck, SideOverlayFrameCallback,
                                   FeedSpan, SideOverlayScan, feed_span_for, locate_side_overlay)
from vodparse.prepass_handover import PrepassHandover, read_handover
from vodparse.layout_reads import pad_box, read_region
from vodparse.sample_preview import PreviewMark, draw_preview_marks, sample_preview_marks
from vodparse.overlay_regions import (OverlayRegions, SideOverlayRecord,
                                      layout_overlay_regions)
from vodparse.overlay_layout import (INGAME_REGION_KEYS, LAYOUT_STORE_NAME,
                                     CustomOverlayLayout, ingame_band_mode_for, layout_by_name,
                                     layout_to_dict, layouts_path, load_layouts)
from vodparse.debug import flush_rejected_timers
from vodparse.characters import HUD_STRIP_TOP, detect_characters
from vodparse.m_char import MCharClassifier, load_m_char_classifier, refine_characters
from vodparse.ocr_engine import (DOCTR_NOT_INSTALLED_MESSAGE, DOCTR_WEIGHTS_MB_APPROX,
                                 FONT_STYLE_STYLIZED, OCR_CANVAS_DEFAULT,
                                 OCR_ENGINE_DEFAULT, OCR_ENGINE_DOCTR, OCR_ENGINE_LABELS,
                                 EasyOcrOverlayEngine, OverlayOcr, doctr_available,
                                 make_doctr_engine, make_layout_box_engine,
                                 make_overlay_engine, validate_engine_lang)
from vodparse.ocr_langs import OCR_LANG_DEFAULT, OCR_LANG_LABELS, make_reader
from vodparse.ocr_roi import OcrRoiCache
from vodparse.overlay_bands import (BAND_BOTTOM, BAND_TOP, BOTH_BANDS, BandOcrPlan, band_boxes,
                                    band_ocr_plan)
from vodparse.tags import (INGAME_TAG_BAND_FRAC_DEFAULT, INGAME_TAG_BAND_FRAC_FLOOR,
                           OVERLAY_BOTTOM_FRAC, parse_ingame_tag_band_frac)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class LogoDetection:
    timestamp_sec: float
    frame_number: int
    label: str
    confidence: float
    bbox: list[int]             # [x1, y1, x2, y2]


@dataclass
class TextDetection:
    timestamp_sec: float
    frame_number: int
    text: str
    confidence: float
    bbox: list[list[int]]       # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]


def texts_to_payload(texts: list[TextDetection]) -> list[TextBox]:
    """Project TextDetections into the JSON-serialisable {text, confidence, bbox} dicts
    stored on an event's "texts" field."""
    return [{"text": t.text, "confidence": t.confidence, "bbox": t.bbox} for t in texts]


@dataclass
class Decision:
    run_ocr: bool = False
    seek_to: float | None = None  # absolute timestamp in seconds
    seek_by: float | None = None  # relative offset (+30 = forward, -10 = back)
    # seek_to takes priority over seek_by if both are set


# ── Seek helper ───────────────────────────────────────────────────────────────

def apply_seek(decision: Decision, reader: FrameReader, frame_number: int, fps: float) -> int:
    """Execute a seek from a Decision. Returns the new frame_count reset value (always 0)."""
    if decision.seek_to is not None:
        target = int(decision.seek_to * fps)
        reader.seek(target)
        print(f"  → seeking to {decision.seek_to:.1f}s (frame {target})")
    elif decision.seek_by is not None:
        target = max(0, frame_number + int(decision.seek_by * fps))
        reader.seek(target)
        direction = "forward" if decision.seek_by > 0 else "back"
        print(f"  → seeking {direction} {abs(decision.seek_by):.1f}s to {target / fps:.1f}s")
    return 0


# ── Detector / display functions ──────────────────────────────────────────────

# `model` is intentionally left untyped (like detect_timer/detect_characters): annotating it
# `YOLO` makes mypy resolve ultralytics' own result types for `model(...)`, whose __call__ now
# returns an un-indexable union and whose Boxes isn't declared iterable; so `results[0].boxes`
# below stops type-checking. Keeping the model untyped lets that result chain stay Any, matching
# the sibling detectors. Don't re-add the annotation without also fixing the result handling.
def detect_logos(model, frame: np.ndarray, timestamp_sec: float, frame_number: int,
                 conf: float = 0.5, infer_size: int = 640,
                 device: str = "cpu") -> list[LogoDetection]:
    """Run the go/game logo YOLO on a frame and return its detections above `conf`."""
    h, w  = frame.shape[:2]
    scale = infer_size / max(h, w)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    results = model(small, conf=conf, verbose=False, device=device)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = (c / scale for c in box.xyxy[0].tolist())
        detections.append(LogoDetection(
            timestamp_sec=round(timestamp_sec, 3),
            frame_number=frame_number,
            label=model.names[int(box.cls)],
            confidence=round(box.conf.item(), 3),
            bbox=[int(x1), int(y1), int(x2), int(y2)],
        ))
    return detections


def _write_preview_frame(jpeg_bytes: bytes, preview_file: str) -> None:
    """Write `jpeg_bytes` to `preview_file` via write-then-replace, so the GUI's poller never
    reads a half-written frame. On Windows the replace can lose a race against the GUI, which
    briefly holds the destination open while polling it (CPython opens files without
    FILE_SHARE_DELETE); that raises PermissionError. Treat it as a dropped frame; invisible to
    the user; rather than let it kill an hours-long analysis run."""
    tmp = preview_file + ".tmp"
    with open(tmp, "wb") as f:
        f.write(jpeg_bytes)
    try:
        os.replace(tmp, preview_file)
    except OSError:
        with contextlib.suppress(OSError):
            os.remove(tmp)   # best-effort: don't leak the orphaned tmp file


def _write_preview_image(display: np.ndarray, preview_file: str) -> None:
    """Encode the finished preview frame `display` as a JPEG and write it to the GUI's
    `preview_file` (_write_preview_frame). A frame that fails to encode is dropped.
    (imencode picks the codec from the extension, not the temp filename.)"""
    ok, buf = cv2.imencode(".jpg", display)
    if ok:
        _write_preview_frame(buf.tobytes(), preview_file)


def render_preview(frame: np.ndarray, logos: list[LogoDetection], texts: list[TextDetection],
                   timestamp_sec: float, duration_sec: float,
                   preview_file: str | None = None,
                   marks: list[PreviewMark] | None = None) -> bool:
    """Draw boxes on a copy of frame and either show it in an OpenCV window or write it to
    `preview_file` (for an embedded GUI preview). `marks` are a seek-mode sample's reads
    (sample_preview.sample_preview_marks): seek mode has no `logos` and passes no `texts`,
    so they are what its preview shows. Returns True if the user pressed Q (window mode
    only)."""
    display = frame.copy()
    if marks:
        draw_preview_marks(display, marks)
    for logo in logos:
        x1, y1, x2, y2 = logo.bbox
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, f"{logo.label} {logo.confidence:.0%}",
                    (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    for t in texts:
        pts = np.array(t.bbox, dtype="int32").reshape(-1, 1, 2)
        cv2.polylines(display, [pts], True, (0, 120, 255), 2)
        cv2.putText(display, t.text, tuple(pts[0][0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 120, 255), 3)
    if preview_file is not None:
        # No position text on the GUI's frames: its progress line already shows the percent
        # and ETA, and text in the frame's corner runs into the stream's own overlay.
        _write_preview_image(display, preview_file)
        return False

    # The OpenCV window has no other progress display, so the frame carries its position.
    pct = (timestamp_sec / duration_sec * 100) if duration_sec > 0 else 0
    cv2.putText(display, f"{timestamp_sec:.1f}s  ({pct:.1f}%)", (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    cv2.imshow("Pipeline", display)
    return cv2.waitKey(1) & 0xFF == ord("q")


def write_opening_frame(video_path: str, preview_file: str) -> None:
    """Write the video's first frame to `preview_file`, with nothing drawn on it, before
    the run has read anything: the GUI's Preview box otherwise sits blank from the Analyze
    click through model loading and into the clock pre-pass (whose first sampled frame is
    the first thing the run would show). One cv2 seek to 0, no decode toward a target, so
    it costs nothing worth measuring. A video that won't open or yields no frame is left
    to the run's own opening of it, which reports the error; this writes nothing."""
    try:
        with VideoFrameSource(video_path) as source:
            frame = source.grab_at(0.0)
    except ValueError:
        return
    if frame is not None:
        _write_preview_image(frame, preview_file)


def write_progress(progress_file: str, timestamp_sec: float, duration_sec: float,
                   stage: str | None = None, clock_calibration: str | None = None,
                   side_overlay: str | None = None) -> None:
    """Report how far through the video the main loop has reached, for the GUI's
    progress bar; `stage` names a pre-loop phase instead (PipelineProgress.stage),
    `clock_calibration` the clock pre-pass's outcome (PipelineProgress.clock_calibration)
    and `side_overlay` the side-bar pre-pass's (PipelineProgress.side_overlay).
    Atomic for the same reason as the preview frames: the GUI polls the file and must
    never read a half-written one.

    Same Windows sharing-violation race as the preview frame can hit the replace inside
    write_json_atomic (the GUI's poll can have the file open at that instant); one skipped
    ~250ms progress tick is invisible, so it's dropped rather than allowed to kill the run.
    write_json_atomic already cleans up its own tmp file before re-raising, so there's nothing
    left to do here but swallow the error."""
    progress: PipelineProgress = {"timestamp_sec": round(timestamp_sec, 1),
                                  "duration_sec": round(duration_sec, 1)}
    if stage is not None:
        progress["stage"] = stage
    if clock_calibration is not None:
        progress["clock_calibration"] = clock_calibration
    if side_overlay is not None:
        progress["side_overlay"] = side_overlay
    try:
        write_json_atomic(progress_file, progress)
    except OSError:
        pass


def detect_text(ocr: OverlayOcr, frame: np.ndarray, timestamp_sec: float, frame_number: int,
                conf_threshold: float = 0.5, beamsearch: bool = False,
                canvas_size: int = OCR_CANVAS_DEFAULT,
                ingame_tag_frac: float = INGAME_TAG_BAND_FRAC_DEFAULT,
                bottom_ocr: OverlayOcr | None = None,
                bands: BandOcrPlan = BOTH_BANDS) -> list[TextDetection]:
    """Run the overlay OCR engine on the top/bottom bands of a frame and return text
    detections above `conf_threshold`. `ingame_tag_frac` lowers the bottom band's top edge
    so a shrunk capture's lifted in-game nametags still get OCR'd (see
    tags.INGAME_TAG_BAND_FRAC_DEFAULT); it defaults to the fixed 0.92 band. `bottom_ocr`
    reads the bottom band in place of `ocr` when the game's HUD pins its own engine
    (_Detectors.band_ocr); None reads both bands with `ocr`. `bands` (see band_ocr_plan)
    drops a band whose readers this run has no use for: the OCR of a dropped band is what
    the plan saves, so it is skipped here rather than filtered afterwards. The default reads
    both bands, as every run without a custom overlay layout does."""
    # OCR only the top and bottom bands of the frame, not the middle. Every text consumer in
    # vodparse.tags reads a box only if its center is in the outer bands (tournament tags/
    # round/best-of above h*TAG_TOP_FRAC; in-game names and the at-bottom overlay below
    # h*OVERLAY_BOTTOM_FRAC: the game renders its nametags at a fixed HUD position inside
    # that strip, see the constant's comment), so the rest is dead weight; running CRAFT over
    # it only costs time and invites false reads (stage text, the "Combatants" panel, the
    # 3-2-1 countdown, spell menus, damage-counter misreads). band_boxes (vodparse.overlay_bands)
    # turns those exact fractions into pixel boxes, so the crop and the gates can never drift
    # apart. The bands are asymmetric: the top is a third to fit the round/best-of second row
    # (see TAG_TOP_FRAC).
    h, w = frame.shape[:2]
    # The crop must cover every bottom-band gate, so it reaches up to the *highest* of them:
    # the tunable in-game nametag band (INGAME_TAG_BAND_FRAC_DEFAULT), which a shrunk capture
    # lowers below the fixed 0.92 the tournament reader/row learner still use. min() keeps the
    # invariant that no gate references a pixel the crop never OCR'd (tags' constant comment).
    boxes = band_boxes(w, h, top=bands.top, bottom=bands.bottom,
                       bottom_frac=min(ingame_tag_frac, OVERLAY_BOTTOM_FRAC))
    # (y_offset, band): adding y_offset maps a band-local point back to full-frame coordinates,
    # so every stored bbox is identical to what full-frame OCR would yield and the position math
    # in vodparse.tags is unaffected. Bands span the full width, so x needs no offset.
    crops = []
    if bands.top:
        _, y1, _, y2 = boxes[BAND_TOP]
        crops.append((y1, frame[y1:y2], ocr))
    if bands.bottom:
        _, y1, _, y2 = boxes[BAND_BOTTOM]
        crops.append((y1, frame[y1:y2], bottom_ocr or ocr))

    detections = []
    for y_off, band, engine in crops:
        for (bbox_pts, text, conf) in engine.read_band(band, beamsearch=beamsearch,
                                                       canvas_size=canvas_size):
            if conf < conf_threshold or not text.strip():
                continue
            detections.append(TextDetection(
                timestamp_sec=round(timestamp_sec, 3),
                frame_number=frame_number,
                text=text.strip(),
                confidence=round(conf, 3),
                bbox=[[int(p[0]), int(p[1]) + y_off] for p in bbox_pts],
            ))
    return detections


def format_characters(characters: SideCharacters | None) -> str:
    """One-line console summary of a detect_characters() result."""
    if not characters:
        return "not detected"
    parts = []
    for side, entry in (("left", characters["left"]), ("right", characters["right"])):
        parts.append(f"{side} {entry['name']} {entry['confidence']:.0%}" if entry
                     else f"{side} -")
    return "  ".join(parts)


def format_stock_slots(slots: list[StockSlot]) -> str:
    """One-line console summary of a stock-icon read: each HUD slot left to right with its
    character and stocks left ("slot 1 Fox x3  slot 2 Marth x1")."""
    if not slots:
        return "no icons"
    return "  ".join(f"slot {i} {slot['name']} x{slot['stocks']}"
                     for i, slot in enumerate(slots, start=1))


# ── User logic ─────────────────────────────────────────────────────────────────

def on_logo_detected(logos: list[LogoDetection], _frame: np.ndarray,
                     timestamp_sec: float, events: list[Event]) -> Decision:
    """Record "go" (match start) and "game" (match end) splash detections into `events`
    (the run's event log). Either one triggers OCR on this frame and a 10s jump past
    the splash screen."""
    for logo in logos:
        if logo.label in ("go", "game"):
            events.append({
                "frame_number":  logo.frame_number,
                "timestamp_sec": logo.timestamp_sec,
                "label":         logo.label,
                "confidence":    logo.confidence,
                "bbox":          logo.bbox,
            })
            print(f"  → [{logo.label.upper()}] recorded at {timestamp_sec:.2f}s")
            return Decision(run_ocr=True, seek_by=10.0)

    return Decision()


# ── Main loop ──────────────────────────────────────────────────────────────────

PREVIEW_WRITE_INTERVAL  = 0.1  # seconds: cap embedded-preview frame writes to ~10/s
PROGRESS_WRITE_INTERVAL = 0.5  # seconds: cap progress-file writes to ~2/s


# Where a pre-pass preview's caption is drawn: the text's baseline, one line down from
# the frame's top-left corner.
PREVIEW_CAPTION_ORIGIN = (10, 36)
PREVIEW_CAPTION_SCALE = 1.1
PREVIEW_CAPTION_THICKNESS = 3

CALIBRATION_BOX_COLOUR = (0, 255, 0)      # BGR: the detector's box on a pre-pass preview frame
CALIBRATION_BOX_THICKNESS = 3
CALIBRATION_PREVIEW_CAPTION = "finding the clock"

SIDE_OVERLAY_BOX_COLOUR = (0, 165, 255)   # BGR orange: the bars and the rows the pre-pass reads
SIDE_OVERLAY_BOX_THICKNESS = 2
SIDE_OVERLAY_PREVIEW_CAPTION = "finding player tags"


def _boxes_preview(preview_file: str, *, colour: tuple[int, int, int],
                   box_thickness: int, caption: str,
                   write_interval_sec: float) -> SideOverlayFrameCallback:
    """A pre-pass's live preview: a callback that draws the given boxes on each sampled
    frame in ``colour``, captions it, and writes it to the GUI's preview file, throttled
    like the main loop's frames. Cheap next to the seek that produced the frame. The shared
    body of calibration_preview and side_overlay_preview. The callback takes the frame's
    timestamp because the pre-passes pass one, but the GUI's frames don't show it."""
    last_write = 0.0

    def show(frame: np.ndarray, boxes: list[BBox], timestamp_sec: float) -> None:
        nonlocal last_write
        now = time.time()
        if now - last_write < write_interval_sec:
            return
        display = frame.copy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(display, (x1, y1), (x2, y2), colour, box_thickness)
        cv2.putText(display, caption, PREVIEW_CAPTION_ORIGIN, cv2.FONT_HERSHEY_SIMPLEX,
                    PREVIEW_CAPTION_SCALE, colour, PREVIEW_CAPTION_THICKNESS)
        _write_preview_image(display, preview_file)
        last_write = now

    return show


def calibration_preview(preview_file: str,
                        write_interval_sec: float = PREVIEW_WRITE_INTERVAL) -> FrameCallback:
    """The clock pre-pass's live preview: the detector's box (when it fired) on each
    sampled frame, in green (``_boxes_preview``)."""
    show_boxes = _boxes_preview(preview_file, colour=CALIBRATION_BOX_COLOUR,
                                box_thickness=CALIBRATION_BOX_THICKNESS,
                                caption=CALIBRATION_PREVIEW_CAPTION,
                                write_interval_sec=write_interval_sec)

    def show(frame: np.ndarray, box: BBox | None, timestamp_sec: float) -> None:
        show_boxes(frame, [box] if box is not None else [], timestamp_sec)

    return show


class ClockFireCounter:
    """A clock pre-pass frame callback (``calibrate_clock_box``'s ``on_frame``) that counts
    the sampled frames the detector fired on, passing each frame on to the live preview
    (``show``) when there is one. A pre-pass that saw no clock at all means the side-bar
    pre-pass has no game frame to read either (it reads only frames with a clock)."""

    def __init__(self, show: FrameCallback | None = None) -> None:
        self.fired = 0
        self._show = show

    def __call__(self, frame: np.ndarray, box: BBox | None, timestamp_sec: float) -> None:
        if box is not None:
            self.fired += 1
        if self._show is not None:
            self._show(frame, box, timestamp_sec)


def side_overlay_preview(preview_file: str,
                         write_interval_sec: float = PREVIEW_WRITE_INTERVAL
                         ) -> SideOverlayFrameCallback:
    """The side-bar pre-pass's live preview: the bars and that frame's candidate text boxes
    on each sampled frame, in orange (``_boxes_preview``)."""
    return _boxes_preview(preview_file, colour=SIDE_OVERLAY_BOX_COLOUR,
                          box_thickness=SIDE_OVERLAY_BOX_THICKNESS,
                          caption=SIDE_OVERLAY_PREVIEW_CAPTION,
                          write_interval_sec=write_interval_sec)


@dataclass(frozen=True)
class _Detectors:
    """The models one run loads, bundled so the per-frame OCR helpers don't each take five
    model parameters. Any member (except device) may be None: absent weights degrade that
    detector to a no-op, and `ocr` is None under --no-ocr.

    `ocr` is the overlay-band engine (EasyOCR or docTR, per --ocr-engine); `overlay_reader`
    is the concrete easyocr.Reader behind it when the engine is EasyOCR, for the one path
    that stays EasyOCR-specific (the timer-reader reuse below); it is None under the docTR
    engine or --no-ocr.

    `box_ocr` reads a custom overlay layout's drawn boxes (ocr_engine.make_layout_box_engine):
    the same object as `ocr` under EasyOCR, a sibling docTR engine with a small detector
    canvas under docTR (a drawn box at the pretrained canvas misreads and costs a full
    band). None when there is no layout to read, or under --no-ocr.

    `band_ocr` reads the BOTTOM band (the game's in-game nameplate strip) when the game's
    HUD pins an engine different from `ocr` (games.IngameHud.ocr_engine: Rivals 2's names
    read with docTR); None means `ocr` reads both bands, as it always did. `ingame_box_ocr`
    is the matching drawn-box engine for a layout's nameplate boxes (REGION_INGAME_P1/P2),
    built on `band_ocr` the way `box_ocr` is built on `ocr`; None means `box_ocr` reads them.

    `timer_ocr` is the reader used only for the clock crop, and is **always EasyOCR**
    regardless of the overlay engine (its allowlisted digit read has no docTR equivalent).
    Clock digits are always Latin regardless of overlay language, so under a non-English
    `ocr_lang` sharing the overlay reader with the timer misreads the colon (e.g. "6:32" ->
    "6.32"), which parse_time can't parse. `timer_ocr` is None exactly when `ocr` is None
    (--no-ocr); when the overlay reader is EasyOCR-English it's that same reader object (no
    extra cost); otherwise it's a dedicated English-only reader."""
    logo: YOLO | None
    timer: YOLO | None
    character: YOLO | None
    m_char: MCharClassifier | None
    ocr: OverlayOcr | None
    overlay_reader: easyocr.Reader | None
    timer_ocr: easyocr.Reader | None
    device: str
    box_ocr: OverlayOcr | None = None
    band_ocr: OverlayOcr | None = None
    ingame_box_ocr: OverlayOcr | None = None
    hud_strip_top: float = HUD_STRIP_TOP
    """The bottom-HUD line `character` accepts portrait tiles below, as a fraction of frame
    height: the loaded game's GameWeights.hud_strip_top (see vodparse.games). It travels with
    the detector because the two are measured together, and is simply unused when `character`
    is None; the default is Ultimate's line, the only game with a character weight today."""
    clock_hud: ClockHud = ULTIMATE_CLOCK_HUD
    """Where the loaded game draws its clock and how its digits read (games.Game.clock_hud):
    what `timer` is filtered and parsed with. Unused when `timer` is None (a layout's timer
    box is read at its drawn spot with both digit rules on); the default is Ultimate's."""
    timer_infer_size: int | None = None
    """The inference size `timer` runs at (games.GameWeights.timer_infer_size), or None to
    run it at the general `infer_size` like every other detector; Ultimate's declares none."""
    calibrated_clock: BBox | None = None
    """The run's calibrated clock box (vodparse.clock_calibration), set by run() after the
    pre-pass when the detector agreed on one spot: _sample_timer reads it first on every
    sample, with `timer` as the fallback. None = no calibration (a drawn timer box, no
    agreeing spot, or --no-calibrate-clock): the detector path alone, as before."""
    stock_icons: YOLO | None = None
    """The stock-icon detector (games.GameWeights.stock_icons): the character read of a game
    with no portrait tiles (Melee), run by vodparse.stock_icons over the bottom strip. A game
    has this or `character`, never both (validated on the profile)."""
    stock_hud: StockHud | None = None
    """The stock row's geometry for `stock_icons` (games.GameWeights.stock_hud); set together
    with it and unused without it."""


def _declared_weight_path(game: Game, which: str) -> str | None:
    """The bundle-resolved path of the `which` weight ("timer", "character" or
    "stock_icons") when the game declares one, else None: what the run's
    inference_settings record for it."""
    if game.weights is None:
        return None
    relative: str | None = getattr(game.weights, which)
    return weight_path(relative) if relative is not None else None


def _load_detectors(game: Game, seek_every: float, use_ocr: bool, ocr_lang: str,
                    ocr_engine: str, layout_boxes_wanted: bool = False,
                    band_engine: str | None = None,
                    layout_timer_box: bool = False,
                    band_on_overlay_engine: bool = False) -> _Detectors:
    """Pick the device and load every model the run needs for `game` (its YOLO detectors and
    m_char classifier cascade, the overlay OCR engine, the EasyOCR timer reader, and, with
    `layout_boxes_wanted`, the engine for a custom layout's drawn boxes). A game that ships no
    weights (see vodparse.games.Game.weights) loads no YOLO at all: its clock comes from a
    custom overlay layout's timer box and it exports without characters. A game whose declared
    clock detector is missing from this build is refused (SystemExit with the fix) unless
    `layout_timer_box` says a drawn timer box stands in for it: without either the run would
    silently produce one giant match. Other missing weights degrade to no-ops. `band_engine` (see
    games.ingame_band_engine_for) pins the engine for the bottom nameplate band: when it
    differs from `ocr_engine` a second reader is loaded for that band alone (English: the
    names it reads are Latin); when that engine is not installed the run stops with how to
    install it (SystemExit), never reading the band with `ocr_engine` in its place unasked.
    `band_on_overlay_engine` (--ingame-band-engine overlay) reads the band with `ocr_engine`
    on request: the GUI's "No" when the user declines docTR's model download."""
    validate_engine_lang(ocr_engine, ocr_lang)
    # Prefer Apple GPU (mps), then NVIDIA (cuda), then CPU; see vodparse.device.
    device = resolve_device()

    print(f"Loading YOLO on {device}...")
    # The go/game logo model only matters in go/game mode (seek_every <= 0); seek mode ignores
    # logo events entirely (timer-only grouping), so skip loading and running it. logo.pt is no
    # longer distributed; if go/game mode is requested without it, fail loudly with guidance
    # rather than crashing deep in YOLO (drop a logo.pt into weights/ to revive the mode).
    if seek_every <= 0:
        if not Path(LOGO_MODEL_PATH).exists():
            raise SystemExit(
                "go/game mode needs weights/logo.pt, which is no longer distributed.\n"
                "Run seek mode (the default, or --seek-every N), or supply your own logo "
                "weight at weights/logo.pt."
            )
        model = YOLO(LOGO_MODEL_PATH)
    else:
        model = None
    timer_model: YOLO | None = None
    character_model: YOLO | None = None
    m_char_classifier: MCharClassifier | None = None
    hud_strip_top = HUD_STRIP_TOP
    stock_model: YOLO | None = None
    stock_hud: StockHud | None = None
    if game.weights is None:
        # Melee, Rivals 2, "Other": no detectors of their own, so nothing is loaded here.
        # main() refuses such a run without a custom overlay layout, so the clock always has
        # a source; the character columns simply stay empty.
        print(f"  models: none ({game.display_name} has no detectors)")
    else:
        # A game declares only the detectors it has (GameWeights: each field optional). A
        # declared weight missing from this build degrades that detector to a no-op, exactly
        # as it always has (an incomplete weights/ folder still runs).
        if game.weights.timer is not None:
            timer_path = weight_path(game.weights.timer)
            if Path(timer_path).exists():
                timer_model = YOLO(timer_path)
            elif not layout_timer_box:
                raise SystemExit(
                    f"{game.display_name}'s clock detector is not in this build "
                    f"({timer_path} is missing), so the run has no clock to split on.\n"
                    "Fetch the weights with `python -m vodparse.cli.download_weights`, or pick "
                    "an overlay layout with a timer box.")
        if game.weights.character is not None:
            character_path = weight_path(game.weights.character)
            character_model = YOLO(character_path) if Path(character_path).exists() else None
            if game.weights.m_char is not None:
                # Cascades after the character detector to name the three m_char classes; no-op if absent.
                m_char_classifier = load_m_char_classifier(weight_path(game.weights.m_char), device)
            if game.weights.hud_strip_top is not None:   # always set with character (validated)
                hud_strip_top = game.weights.hud_strip_top
        if game.weights.stock_icons is not None:
            stock_path = weight_path(game.weights.stock_icons)
            stock_model = YOLO(stock_path) if Path(stock_path).exists() else None
            stock_hud = game.weights.stock_hud        # always set with stock_icons (validated)
        loaded = [name for name, m in (("logo", model), ("timer", timer_model),
                                       ("character", character_model),
                                       ("m_char", m_char_classifier),
                                       ("stock_icons", stock_model)) if m is not None]
        print(f"  models: {', '.join(loaded) if loaded else 'none'}")

    ocr: OverlayOcr | None = None
    overlay_reader: easyocr.Reader | None = None
    box_ocr: OverlayOcr | None = None
    band_ocr: OverlayOcr | None = None
    ingame_box_ocr: OverlayOcr | None = None
    if use_ocr:
        if ocr_engine == OCR_ENGINE_DOCTR:
            ocr = make_doctr_engine(device)
        else:
            overlay_reader = make_reader(ocr_lang, gpu=True)
            ocr = EasyOcrOverlayEngine(overlay_reader)
        if layout_boxes_wanted:
            box_ocr = make_layout_box_engine(ocr_engine, ocr)
        if band_engine is not None and band_engine != ocr_engine:
            if band_on_overlay_engine:
                print(f"Note: reading {game.display_name}'s nameplate band with {ocr_engine} "
                      f"instead of {band_engine}, as asked (--ingame-band-engine "
                      f"{INGAME_BAND_ENGINE_OVERLAY}).")
            elif band_engine == OCR_ENGINE_DOCTR and not doctr_available():
                raise SystemExit(
                    f"{game.display_name} reads its nameplates with docTR ({FONT_STYLE_STYLIZED}). "
                    f"{DOCTR_NOT_INSTALLED_MESSAGE}\n"
                    f"To read them with {ocr_engine} instead, pass --ingame-band-engine "
                    f"{INGAME_BAND_ENGINE_OVERLAY}.")
            else:
                print(f"Loading {band_engine} for {game.display_name}'s ingame "
                      f"nameplates...")
                band_ocr, _ = make_overlay_engine(band_engine, OCR_LANG_DEFAULT, device)
                if layout_boxes_wanted:
                    ingame_box_ocr = make_layout_box_engine(band_engine, band_ocr)
    # The clock crop is always read with EasyOCR, and always as Latin digits (see the
    # _Detectors docstring). Reuse the overlay reader when it's already EasyOCR-English;
    # building a second reader would be pure waste; a docTR run or a non-English overlay
    # reader needs its own plain English reader.
    if ocr is None:
        timer_ocr = None
    elif overlay_reader is not None and ocr_lang == OCR_LANG_DEFAULT:
        timer_ocr = overlay_reader
    else:
        print("Loading dedicated English timer reader (the clock is always read with EasyOCR)...")
        timer_ocr = make_reader(OCR_LANG_DEFAULT, gpu=True)
    # A game with no clock profile (Other, or one still unmeasured) has no timer weight
    # either, so its clock_hud is never consulted; Ultimate's stands in as the default.
    clock_hud = game.clock_hud if game.clock_hud is not None else ULTIMATE_CLOCK_HUD
    timer_infer_size = game.weights.timer_infer_size if game.weights is not None else None
    return _Detectors(model, timer_model, character_model, m_char_classifier,
                      ocr, overlay_reader, timer_ocr, device, box_ocr, band_ocr, ingame_box_ocr,
                      hud_strip_top, clock_hud, timer_infer_size,
                      stock_icons=stock_model, stock_hud=stock_hud)


# ── Custom overlay layout reads ──────────────────────────────────────────────────
# A layout (vodparse.overlay_layout) is the user's own map of where the overlay's text
# sits. Its text boxes are read per OCR sample into event["layout_reads"] (extract votes
# them instead of running the position heuristics), and its timer box, when drawn, is
# where the clock is read — replacing the game's YOLO clock detector, which only knows
# its own game's clock graphics. That substitution is what lets the seek-splitter run on
# a game without a clock detector ("Other").

# The rejection-log stage tag for clock reads at a layout's fixed box (see
# timer.read_timer_at_bbox / debug.flush_rejected_timers).
LAYOUT_TIMER_STAGE = "ocr_parse_layout"


@dataclass(frozen=True)
class _LayoutBoxes:
    """A custom overlay layout resolved to this video's pixels, once per run: the text
    regions to read per OCR sample (region key → box) and the timer box, if drawn."""
    regions: dict[str, BBox]
    timer: BBox | None


def _resolve_layout_boxes(layout: CustomOverlayLayout, frame_width: int,
                          frame_height: int) -> _LayoutBoxes:
    """The layout's boxes in this video's pixels: the text regions through
    overlay_regions.layout_overlay_regions (the one place a layout's rects become the
    per-sample read boxes, so extract votes exactly what was read) and the timer box."""
    text_regions = layout_overlay_regions(layout, frame_width, frame_height)
    regions = text_regions.boxes if text_regions is not None else {}
    timer = (layout.timer.to_pixels(frame_width, frame_height)
             if layout.timer is not None else None)
    return _LayoutBoxes(regions=regions, timer=timer)


# ── The side-bar lock (vodparse.side_overlay) ──────────────────────────────────────
# A 4:3 game's stream puts the player names in the bars beside the feed, where the band
# OCR never looks. When the game declares side bars (games.Game.has_side_bars) and the run
# has no custom layout, an OCR pre-pass locks the rows the names sit on, and the main pass
# reads those rows on every sample exactly like a drawn layout's boxes.

def _ingame_check(det: _Detectors, infer_size: int) -> IngameCheck:
    """Whether a sampled frame is in-game, by the clock detector firing on it: what the
    side-bar pre-pass counts row support over. Only called with a clock detector loaded
    (_side_overlay_wanted requires one)."""
    timer_model = det.timer
    assert timer_model is not None   # _side_overlay_wanted
    timer_infer_size = det.timer_infer_size if det.timer_infer_size is not None else infer_size

    def is_ingame(frame: np.ndarray) -> bool:
        return localize_timer(timer_model, frame, det.device, timer_infer_size,
                              clock_hud=det.clock_hud) is not None

    return is_ingame


def _side_overlay_wanted(game: Game, overlay_layout: CustomOverlayLayout | None, det: _Detectors,
                         locate_overlay: bool) -> bool:
    """Whether this run runs the side-bar pre-pass: only for a game that declares side bars,
    only without a custom layout (the user's boxes already say where the names are), only
    with the overlay OCR and the drawn-box engine loaded (the lock is read like drawn
    boxes), only with the clock detector loaded (it is how the pre-pass tells a game frame
    from a break: a frame without a clock is never read, and without the detector nothing
    would be), and not on --no-locate-overlay."""
    return (locate_overlay and game.has_side_bars and overlay_layout is None
            and det.ocr is not None and det.box_ocr is not None and det.timer is not None)


def _run_feed_span(det: _Detectors, calibration: ClockCalibration | None,
                   frame_size: tuple[int, int]) -> FeedSpan:
    """Where this run places the game feed, and so the side bars: from the calibrated
    clock box when the pre-pass found one, assumed centred otherwise (side_overlay.
    feed_span_for). The side-bar pre-pass searches these bars, and rows handed over by
    the layout preview are taken only when they were found in the same ones."""
    frame_width, frame_height = frame_size
    placement = det.clock_hud.feed_placement
    assert placement is not None   # the caller checked game.has_side_bars, which is this
    return feed_span_for(calibration.bbox if calibration is not None else None,
                         frame_width, frame_height, placement)


def _side_overlay_prepass(video_path: str, det: _Detectors, calibration: ClockCalibration | None,
                          frame_size: tuple[int, int], duration_sec: float, ocr_canvas: int,
                          infer_size: int, preview_file: str | None,
                          progress_file: str | None) -> SideOverlayScan | None:
    """Run the side-bar pre-pass for this video: the feed span from the calibrated clock
    (assumed centred without one), then vodparse.side_overlay.locate_side_overlay on the
    band engine with the clock detector as the in-game check. Announced on the progress
    file first (stage "locating_overlay") like the clock pre-pass, previewed the same way.
    None when the video could not be sampled. The caller decides what to do with the scan;
    this only runs it and reports."""
    feed = _run_feed_span(det, calibration, frame_size)
    if progress_file:
        write_progress(progress_file, 0.0, duration_sec, stage=PROGRESS_STAGE_LOCATING_OVERLAY)
    print("Locating the player tags in the side bars (OCR pre-pass)...")
    print(f"  feed x {feed.x0}-{feed.x1} ({feed.source})")
    return locate_side_overlay(
        video_path, det.ocr, ocr_canvas, feed, is_ingame=_ingame_check(det, infer_size),
        on_frame=side_overlay_preview(preview_file) if preview_file else None)


def _handed_over_calibration(handover: PrepassHandover | None, video_path: str, game: Game,
                             frame_size: tuple[int, int]) -> ClockCalibration | None:
    """The calibrated clock the layout preview's search handed this run
    (vodparse.prepass_handover), or None when there is none for this video and game, or
    its box does not fit this video's frame: the run then runs the pre-pass itself."""
    if handover is None:
        return None
    return ClockCalibration.from_config(
        handover.clock_calibration_for(video_path, game.id), frame_size)


def _handed_over_clock_searched(handover: PrepassHandover | None, video_path: str,
                                game: Game) -> bool:
    """Whether the layout preview's clock search already ran for this video and game and
    calibrated nothing (vodparse.prepass_handover): the run then skips its pre-pass and
    reads every sample with the detector, as it would after its own refusal."""
    return handover is not None and handover.clock_searched_for(video_path, game.id)


def _handed_over_clock_never_fired(handover: PrepassHandover | None, video_path: str,
                                   game: Game) -> bool:
    """Whether that preview clock search also found the detector firing on none of its
    frames (vodparse.prepass_handover): the run then skips the side-bar pre-pass too."""
    return handover is not None and handover.clock_never_fired_for(video_path, game.id)


def _handed_over_side_overlay(handover: PrepassHandover | None, video_path: str, game: Game,
                              ocr_engine: str, ocr_lang: str, frame_size: tuple[int, int],
                              feed: FeedSpan) -> SideOverlayRecord | None:
    """The side-bar search's outcome the layout preview handed this run, or None when there
    is none for this video, game, Font style and language, its bars were not placed from
    the same feed span this run places them from (``feed``: the preview's search may have
    had no clock box while this run's pre-pass found one, or the other way round), or a
    locked box does not fit this video's frame: the run then runs the pre-pass itself."""
    if handover is None:
        return None
    record = SideOverlayRecord.from_config(
        handover.side_overlay_for(video_path, game.id, ocr_engine, ocr_lang))
    if record is None:
        return None
    if record.feed_edges() != (feed.x0, feed.x1):
        return None
    frame_width, frame_height = frame_size
    for x1, y1, x2, y2 in (record.boxes or {}).values():
        if not (0 <= x1 and x2 <= frame_width and 0 <= y1 and y2 <= frame_height):
            return None
    return record


def _sample_timer(det: _Detectors, timer_recovery: TimerRecovery, frame: np.ndarray,
                  timestamp: float, infer_size: int,
                  layout_boxes: _LayoutBoxes | None) -> TimerReading | None:
    """The clock reading for one sample: at the layout's fixed timer box when one is
    drawn, else via the YOLO detector with dropped-box recovery; None when neither
    source is available (no timer weights and no timer box).

    The fixed-box read deliberately bypasses TimerRecovery: that machinery reconstructs
    a box the detector *lost*, but a drawn box is known on every frame, and its read is
    as authoritative as a detector hit — so no `recovered` flag either. Its parse
    rejections still flow into the run's debug dump through the shared list.

    The calibrated-box read (`read_calibrated_clock`) bypasses only TimerRecovery's
    plausibility gate; a hit is handed to `TimerRecovery.note_anchor` so the recovery
    state's anchor is this sample's clock. Without that, a later sample the box AND the
    detector both miss would judge its recovery read against a detector hit from a game
    ago, and any lower value would pass as a plausible fall over that long an interval.

    The fixed-box read accepts "7.28" as 7:28 (`dot_as_colon`) and "7:2884" as 7:28.84
    (`unseparated_centiseconds`) whatever the game: a user draws the box on any clock font,
    and one whose colon the recognizer returns as "." or whose centiseconds follow a space
    would otherwise never parse. A false parse the rules let through (a "76.57" misread as
    76:57) is dropped at extract time, where every clock is checked against the layout's own
    clock length (`match_extraction.valid_secs`). The detector path applies the loaded game's
    own rules (`det.clock_hud`)."""
    if layout_boxes is not None and layout_boxes.timer is not None:
        if det.timer_ocr is None:   # --no-ocr: no reader to read the clock with
            return None
        return read_timer_at_bbox(det.timer_ocr, frame, pad_box(layout_boxes.timer),
                                  confidence=1.0, rejections=timer_recovery.rejections,
                                  sample_ts=timestamp, stage=LAYOUT_TIMER_STAGE,
                                  dot_as_colon=DRAWN_BOX_DOT_AS_COLON,
                                  unseparated_centiseconds=DRAWN_BOX_UNSEPARATED_CENTISECONDS)
    if det.timer is None or det.timer_ocr is None:   # no detector, or --no-ocr: nothing reads the box
        return None
    # The calibrated box (the spot the detector agreed on across this VOD) is read first,
    # like a drawn box; the detector only runs on a sample it does not parse. So a frame the
    # detector would have missed, or fired on the wrong digits for, still reads the clock.
    if det.calibrated_clock is not None:
        calibrated = read_calibrated_clock(det.timer_ocr, frame, det.calibrated_clock,
                                           clock_hud=det.clock_hud,
                                           max_clock_seconds=timer_recovery.max_clock_seconds,
                                           rejections=timer_recovery.rejections,
                                           sample_ts=timestamp)
        if calibrated is not None:
            timer_recovery.note_anchor(calibrated["total_seconds"], timestamp)
            return calibrated
    timer_infer_size = det.timer_infer_size if det.timer_infer_size is not None else infer_size
    return detect_timer_with_recovery(timer_recovery, det.timer, det.timer_ocr,
                                      frame, det.device, timer_infer_size, timestamp,
                                      clock_hud=det.clock_hud)


def _read_layout_regions(box_ocr: OverlayOcr, frame: np.ndarray, boxes: dict[str, BBox],
                         canvas_size: int,
                         ingame_box_ocr: OverlayOcr | None = None) -> dict[str, LayoutRead]:
    """Read every one of the run's overlay regions on this frame (a drawn layout's text
    boxes, or the rows the side-bar lock locked) with `box_ocr` (_Detectors.box_ocr: the
    full detect+recognize read the editor's test previews, on the drawn-box engine), keyed
    by region. The nameplate boxes (INGAME_REGION_KEYS) go to `ingame_box_ocr` when
    the game pins its own engine for them (_Detectors.ingame_box_ocr), else to `box_ocr`.
    Every sample pays the full read: a per-box ROI cache was tried and dropped (see
    vodparse.ocr_roi's module docstring). Each read is folded for its region (a player box
    into gamertag + sponsor, see layout_reads.layout_read_from_texts), the same way the
    editor's test read folds it. No confidence threshold: tags.attach_overlay_regions and
    tags.attach_layout_ingame vote the reads."""
    reads: dict[str, LayoutRead] = {}
    for key, box in boxes.items():
        engine = ingame_box_ocr if key in INGAME_REGION_KEYS and ingame_box_ocr is not None else box_ocr
        reads[key] = read_region(engine, frame, box, canvas_size, region=key)
    return reads


def _format_layout_reads(reads: dict[str, LayoutRead]) -> str:
    """One console line of a sample's box reads: ``player1='blargh257' (sponsor 'Rob jr')``
    for a player box that split a sponsor off, ``round='Winners Final'`` for the rest."""
    parts = []
    for key, read in reads.items():
        sponsor = read.get("sponsor")
        suffix = f" (sponsor {sponsor!r})" if sponsor else ""
        parts.append(f"{key}={read['text']!r}{suffix}")
    return ", ".join(parts)


# ── The per-sample character read ────────────────────────────────────────────────

def _read_characters(det: _Detectors, frame: np.ndarray,
                     infer_size: int) -> tuple[SideCharacters | None, list[StockSlot] | None]:
    """The per-sample character read, whichever detector the game ships: the portrait
    detector (with the m_char cascade after it) when `det.character` is loaded, the
    stock-icon detector when `det.stock_icons` is, neither otherwise. Returns the per-side
    read both produce alike, and the stock slots (every slot with its stock count) only from
    the stock path: None from the portrait path, so an Ultimate event carries no such key."""
    if det.character is not None:
        characters = detect_characters(det.character, frame, det.device, infer_size,
                                       hud_strip_top=det.hud_strip_top)
        return refine_characters(characters, frame, det.m_char), None
    if det.stock_icons is not None:
        assert det.stock_hud is not None, "stock_hud is loaded together with stock_icons"
        read = detect_stock_icons(det.stock_icons, frame, det.device, det.stock_hud)
        return read.characters, read.slots
    return None, None


def _reads_characters(det: _Detectors) -> bool:
    """Whether the run has any character detector at all (the console prints a character
    line only then)."""
    return det.character is not None or det.stock_icons is not None


def _loaded_character_model_path(game: Game, det: _Detectors) -> str | None:
    """The portrait-detector weight the run actually read with, for inference_settings: the
    declared path only when the detector loaded. A missing character.pt degrades the run to
    no character read, and extract decides whether the splitter may ask a frame for a
    detected fighter on this record (extract_matches.has_character_detector), so it must not
    name a weight that never ran."""
    if det.character is None:
        return None
    return _declared_weight_path(game, "character")


def _loaded_stock_model_path(game: Game, det: _Detectors) -> str | None:
    """The stock-icon weight the run actually read with, for inference_settings: the
    declared path only when the detector loaded. A missing weight degrades the run to no
    stock read, and extract's "characters read from stock icons" note keys on this record,
    so it must not name a weight that never ran."""
    if det.stock_icons is None:
        return None
    return _declared_weight_path(game, "stock_icons")


def _splash_frame_ocr(det: _Detectors, timer_recovery: TimerRecovery, events: list[Event],
                      frame: np.ndarray, timestamp: float, frame_number: int, *,
                      infer_size: int, text_conf: float,
                      ocr_beamsearch: bool, ocr_canvas: int,
                      ingame_tag_frac: float,
                      layout_boxes: _LayoutBoxes | None = None,
                      bands: BandOcrPlan = BOTH_BANDS) -> list[TextDetection]:
    """The full-frame OCR pass on a go/game splash frame: read texts, the clock, and the
    character portraits, enriching the just-appended splash event in place. Returns the
    detected texts (the preview overlay draws them). `layout_boxes` (a custom overlay
    layout, see _sample_timer) adds its box reads and can supply the clock. `bands` is the
    run's band plan (band_ocr_plan), the same one the periodic samples read with."""
    if det.ocr is None:
        print("  [warning] OCR requested but --no-ocr was set, skipping")
        return []
    texts = detect_text(det.ocr, frame, timestamp, frame_number,
                        conf_threshold=text_conf, beamsearch=ocr_beamsearch,
                        canvas_size=ocr_canvas, ingame_tag_frac=ingame_tag_frac,
                        bottom_ocr=det.band_ocr, bands=bands)
    if events:
        events[-1]["texts"] = texts_to_payload(texts)
    has_timer_source = (det.timer is not None
                        or (layout_boxes is not None and layout_boxes.timer is not None))
    if has_timer_source:
        timer = _sample_timer(det, timer_recovery, frame, timestamp, infer_size, layout_boxes)
        if events:
            events[-1]["timer"] = timer
        print(f"    timer: {timer['formatted']}" if timer else "    timer: not detected")
    if layout_boxes is not None and layout_boxes.regions and events and det.box_ocr is not None:
        reads = _read_layout_regions(det.box_ocr, frame, layout_boxes.regions, ocr_canvas,
                                     ingame_box_ocr=det.ingame_box_ocr)
        events[-1]["layout_reads"] = reads
        print(f"    layout: {_format_layout_reads(reads)}")
    if _reads_characters(det):
        characters, slots = _read_characters(det, frame, infer_size)
        if events:
            events[-1]["characters"] = characters
            if slots is not None:
                events[-1]["stock_slots"] = slots
        print(f"    characters: {format_characters(characters)}")
        if slots is not None:
            print(f"    stocks: {format_stock_slots(slots)}")
    if texts:
        print(f"    OCR: {[t.text for t in texts]}")
    return texts


def _periodic_ocr_event(det: _Detectors, timer_recovery: TimerRecovery,
                        frame: np.ndarray, timestamp: float, frame_number: int, *,
                        infer_size: int, text_conf: float, ocr_beamsearch: bool,
                        ocr_canvas: int, ocr_interval_sec: float, gate_ocr_on_timer: bool,
                        roi_cache: OcrRoiCache | None, ingame_tag_frac: float,
                        layout_boxes: _LayoutBoxes | None = None,
                        bands: BandOcrPlan = BOTH_BANDS) -> Event:
    """One periodic-OCR sample: read the clock, (unless gated) the full-frame texts, and the
    character portraits, and return the label-less Event to append. The caller guarantees
    det.ocr is loaded (periodic OCR never fires under --no-ocr). `layout_boxes` (a custom
    overlay layout, see _sample_timer) adds its box reads (on det.box_ocr) and can supply
    the clock. `bands` (see band_ocr_plan) is the run's band plan: a band the layout's own
    boxes have made redundant is not OCR'd, and with neither band left the event carries no
    band texts at all, only the clock and the layout's box reads."""
    assert det.ocr is not None
    # Timer first: it always runs (its own cheap localized digit OCR, separate from
    # the full-frame pass) and feeds the recovery state machine every sample.
    timer = _sample_timer(det, timer_recovery, frame, timestamp, infer_size, layout_boxes)
    # The ROI cache watches the clock stream on EVERY sample (including ones the gate will
    # skip below); a clock coming back after a dead gap; or resetting upward, marks a new
    # game: exactly when tags/nameplates change and the cache must re-verify its regions.
    if roi_cache is not None:
        roi_cache.note_timer(timer["total_seconds"] if timer else None,
                             timestamp, ocr_interval_sec)
    # Timer gate: on a "dead" frame (no valid timer from the model OR the
    # recovery chain), skip the expensive full-frame OCR (detect_text). The frame is
    # still recorded and flagged ocr_gated so grouping keeps it as a clock-less sample
    # (intermission splitting + first-game start depend on its presence).
    gated = gate_ocr_on_timer and timer is None
    # The ROI cache only operates on clock-valid frames; that's where the cacheable text
    # (HUD nameplates, overlay) lives, and it keeps intermission menus from poisoning the
    # calibration window. Clock-less frames take the normal full pass, exactly as today.
    roi = roi_cache if (roi_cache is not None and timer is not None) else None
    roi_cached = False
    if gated or not bands.reads_anything:
        texts_payload: list[TextBox] = []
    elif roi is not None and not roi.should_run_full():
        # ROI cache, steady state: recognizer-only over the calibrated text regions; the
        # text-detection pass (85-89% of the full-pass cost) is skipped. The bands' reads
        # are flattened: downstream never cared which band a text came from.
        per_band = roi.read_cached(det.ocr, frame, conf_threshold=text_conf,
                                   beamsearch=ocr_beamsearch)
        texts_payload = [text for band_texts in per_band.values() for text in band_texts]
        roi_cached = True
    else:
        texts_payload = texts_to_payload(
            detect_text(det.ocr, frame, timestamp, frame_number,
                        conf_threshold=text_conf, beamsearch=ocr_beamsearch,
                        canvas_size=ocr_canvas, ingame_tag_frac=ingame_tag_frac,
                        bottom_ocr=det.band_ocr, bands=bands))
        if roi is not None:
            roi.observe_full(texts_payload)
    characters, stock_slots = _read_characters(det, frame, infer_size)
    event: Event = {
        "frame_number":  frame_number,
        "timestamp_sec": round(timestamp, 3),
        "label":         None,
        "texts": texts_payload,
        "timer": timer,
        "characters": characters,
    }
    if stock_slots is not None:
        event["stock_slots"] = stock_slots
    if gated:
        event["ocr_gated"] = True
    elif layout_boxes is not None and layout_boxes.regions and det.box_ocr is not None:
        # The layout's own boxes, read the way the editor previewed them. Gated frames
        # skip these too: with no clock on screen there is no overlay to read.
        event["layout_reads"] = _read_layout_regions(det.box_ocr, frame, layout_boxes.regions,
                                                     ocr_canvas,
                                                     ingame_box_ocr=det.ingame_box_ocr)
    if gated:
        summary = "gated (no timer)"
    elif not bands.reads_anything:
        summary = "bands skipped (the layout covers every band reader)"
    else:
        summary = f"{len(texts_payload)} text(s)" + (" (roi-cached)" if roi_cached else "")
    print(f"  [periodic OCR] @ {timestamp:.2f}s: {summary}")
    print(f"    timer: {timer['formatted']}" if timer else "    timer: not detected")
    if "layout_reads" in event:
        print(f"    layout: {_format_layout_reads(event['layout_reads'])}")
    if _reads_characters(det):
        print(f"    characters: {format_characters(event.get('characters'))}")
    if stock_slots is not None:
        print(f"    stocks: {format_stock_slots(stock_slots)}")
    if texts_payload:
        print(f"    OCR: {[t['text'] for t in texts_payload]}")
    return event


def run(video_path: str, game: Game, conf: float = 0.5,
        infer_size: int = DEFAULT_INFER_SIZE, every_n: int = 1,
        text_conf: float = 0.5, use_ocr: bool = True,
        show_preview: bool = False, seek_every: float = 30.0,
        ocr_interval_sec: float = 30.0, ocr_beamsearch: bool = False,
        ocr_canvas: int = OCR_CANVAS_DEFAULT, gate_ocr_on_timer: bool = False,
        ocr_roi_cache: bool = False, ocr_lang: str = OCR_LANG_DEFAULT,
        ocr_engine: str = OCR_ENGINE_DEFAULT,
        ingame_tag_frac: float = INGAME_TAG_BAND_FRAC_DEFAULT,
        preview_file: str | None = None, progress_file: str | None = None,
        overlay_layout: CustomOverlayLayout | None = None,
        calibrate_clock: bool = True, locate_overlay: bool = True,
        band_on_overlay_engine: bool = False,
        prepass_handover: PrepassHandover | None = None) -> None:

    # Something in the GUI's Preview box right away: the models below take a while to
    # load, and the first frame the run itself draws comes later still.
    if preview_file:
        write_opening_frame(video_path, preview_file)

    # What the game's nameplate band holds this run (see overlay_layout.ingame_band_mode_for),
    # which OCR bands still have a reader (band_ocr_plan: a layout's boxes can retire either),
    # and the bottom band's engine, the game's call when its HUD pins one (Rivals 2: docTR).
    ingame_band = ingame_band_mode_for(game, overlay_layout)
    bands = band_ocr_plan(game, overlay_layout)
    band_engine = ingame_band_engine_for(game, ingame_band)
    det = _load_detectors(game, seek_every, use_ocr, ocr_lang, ocr_engine,
                          layout_boxes_wanted=overlay_layout is not None or game.has_side_bars,
                          band_engine=band_engine,
                          layout_timer_box=(overlay_layout is not None
                                            and overlay_layout.timer is not None),
                          band_on_overlay_engine=band_on_overlay_engine)
    # The engine the pinned reader stands in for, when it loaded (a missing docTR, or
    # --ingame-band-engine overlay, falls back to the overlay engine, see _load_detectors):
    # it reads the nameplates wherever they are
    # read this run, the band or the layout's nameplate boxes.
    nameplate_engine: str = ocr_engine
    if det.band_ocr is not None and band_engine is not None:
        nameplate_engine = band_engine
    # What actually read the BOTTOM BAND, recorded for extract (which word-merges a docTR
    # band's reads): the pinned engine only when the band was read at all; a plan that retired
    # the band read it with nothing, so the overlay engine is recorded and extract does no
    # band merge for it.
    band_ocr_engine = nameplate_engine if bands.bottom else ocr_engine
    # Per-VOD text-region calibration: full CRAFT on a few frames, then recognizer-only on
    # the cached regions (see vodparse.ocr_roi). Seek-mode periodic OCR only; the go/game
    # splash path below always runs the full pass (splashes are rare). EasyOCR-only: under
    # docTR the cache's recognizer-only reads regressed the tag votes on every VOD of a
    # 2026-09-03 A/B (the ocr_roi module docstring has the why), so every docTR periodic
    # sample keeps the full pass.
    # A pinned band engine rules the cache out only while the bottom band is read: the cache
    # re-reads its calibrated regions with the EasyOCR reader, which would put the wrong engine
    # on the nameplate band. Once the plan has retired that band (the layout's nameplate boxes
    # are read instead, with the pinned engine, outside the cache) the cache only ever holds
    # the top band, and is safe again.
    band_is_pinned = det.band_ocr is not None and bands.bottom
    if ocr_roi_cache and det.ocr is not None and det.overlay_reader is None:
        print("Note: --ocr-roi-cache is EasyOCR-only; disabled for the docTR engine.")
    elif ocr_roi_cache and band_is_pinned:
        print(f"Note: --ocr-roi-cache disabled: this game's nameplate band is read "
              f"with {band_ocr_engine}.")
    band_cache_wanted = ocr_roi_cache and det.overlay_reader is not None and not band_is_pinned

    try:
        reader = FrameReader(video_path)
    except (FileNotFoundError, ValueError) as err:
        # e.g. the drive was unplugged while the models above loaded: one clear line, no traceback.
        sys.exit(f"Error: {err}")
    reader.start()

    fps          = reader.fps
    total_frames = reader.total_frames
    duration_sec = total_frames / fps
    frame_width  = reader.width
    frame_height = reader.height
    # The layout is normalized (fractions of the frame), so its boxes resolve to pixels
    # once here and stay constant for the run.
    layout_boxes = (_resolve_layout_boxes(overlay_layout, frame_width, frame_height)
                    if overlay_layout is not None else None)

    # The calibrated clock box (vodparse.clock_calibration): a detector-only pre-pass over a
    # few dozen frames; when they agree on one spot, every sample reads that box first and
    # the detector is the fallback. Skipped when a layout draws the timer box (the user's
    # box already is that), when there is no clock detector, or on --no-calibrate-clock.
    # Announced on the progress file first (stage "calibrating"): the GUI's bar has no
    # position to show for a pre-pass, and its label would otherwise sit on "Starting…".
    calibration: ClockCalibration | None = None
    clock_status = CLOCK_CALIBRATION_SKIPPED       # what every progress tick reports (schema)
    clock_never_fired = False   # the pre-pass (this run's or the preview's) saw no clock at all
    if calibrate_clock and det.timer is not None and (layout_boxes is None or layout_boxes.timer is None):
        # A box the layout preview's own search already found for this video and game is
        # used as it is (vodparse.prepass_handover): the same pre-pass, already run.
        calibration = _handed_over_calibration(prepass_handover, video_path, game,
                                               (frame_width, frame_height))
        if calibration is not None:
            print("Using the clock box the preview's search found (no pre-pass)...")
        elif _handed_over_clock_searched(prepass_handover, video_path, game):
            # The preview's search sampled the same frames by the same rules and found no
            # steady spot: repeating it would conclude the same, sixty seeks later.
            print("The preview's search found no steady clock spot (no pre-pass)...")
            clock_never_fired = _handed_over_clock_never_fired(prepass_handover, video_path, game)
        else:
            timer_infer_size = (det.timer_infer_size if det.timer_infer_size is not None
                                else infer_size)
            if progress_file:
                write_progress(progress_file, 0.0, duration_sec, stage=PROGRESS_STAGE_CALIBRATING)
            print("Calibrating the clock box (detector-only pre-pass)...")
            fires = ClockFireCounter(
                calibration_preview(preview_file) if preview_file else None)
            calibration = calibrate_clock_box(
                video_path, det.timer, det.device, timer_infer_size, det.clock_hud,
                on_frame=fires)
            clock_never_fired = fires.fired == 0
        if calibration is None:
            clock_status = CLOCK_CALIBRATION_NOT_FOUND
            print("  no steady clock spot: the detector reads every sample on its own")
        else:
            clock_status = CLOCK_CALIBRATION_FOUND
            print(f"  calibrated clock box {calibration.bbox} from {calibration.hits} of "
                  f"{calibration.samples} sampled frames")
            det = replace(det, calibrated_clock=calibration.bbox)

    # The overlay text regions this run reads per sample (vodparse.overlay_regions): the
    # layout's drawn boxes, or the rows the side-bar pre-pass locks below. Recorded in the
    # JSON so extract votes exactly what was read.
    overlay_regions: OverlayRegions | None = (
        layout_overlay_regions(overlay_layout, frame_width, frame_height)
        if overlay_layout is not None else None)
    # The side-bar lock (see the section above _side_overlay_prepass; _side_overlay_wanted
    # says when it runs). A lock becomes this run's layout boxes; any other outcome leaves
    # the built-in band reading as it was.
    side_record: SideOverlayRecord | None = None
    side_status = SIDE_OVERLAY_SKIPPED             # what every progress tick reports (schema)
    side_overlay_wanted = _side_overlay_wanted(game, overlay_layout, det, locate_overlay)
    if side_overlay_wanted and clock_never_fired:
        print("No clock found in the clock pre-pass: skipping the side-bar search for player "
              "tags (it reads only frames with a clock)")
    elif side_overlay_wanted:
        # As with the clock: rows the layout preview's search already locked for this
        # video, game, Font style and language are used as they are.
        side_record = _handed_over_side_overlay(
            prepass_handover, video_path, game, ocr_engine, ocr_lang, (frame_width, frame_height),
            _run_feed_span(det, calibration, (frame_width, frame_height)))
        if side_record is not None:
            print("Using the player tag rows the preview's search found (no pre-pass)...")
        else:
            side_scan = _side_overlay_prepass(
                video_path, det, calibration, (frame_width, frame_height), duration_sec,
                ocr_canvas, infer_size, preview_file, progress_file)
            if side_scan is not None:
                side_record = SideOverlayRecord.from_config(side_scan.as_config())
        if side_record is None:
            print("  the video could not be sampled; reading the bands the built-in way")
        else:
            side_status = side_record.status
            print(f"  {side_record.note}")
            lock = side_record.overlay_regions()
            if lock is not None:
                layout_boxes = _LayoutBoxes(regions=dict(lock.boxes), timer=None)
                overlay_regions = lock
    elif game.has_side_bars and overlay_layout is None and det.timer is None:
        print("No clock detector for this game: skipping the side-bar search for player "
              "tags (it needs the clock to tell game frames from breaks)")
    # The cache needs the frame size: its bands mirror detect_text's crops (bottom band
    # widened the same way), and it caches only the bands this run actually reads
    # (`bands`, above). No band pass at all: nothing to calibrate on, and for_bands refuses
    # an empty plan, so the guard here is load-bearing, not belt-and-braces.
    roi_cache = (OcrRoiCache.for_bands(frame_width, frame_height,
                                       min(ingame_tag_frac, OVERLAY_BOTTOM_FRAC),
                                       top=bands.top, bottom=bands.bottom)
                 if band_cache_wanted and bands.reads_anything else None)

    print(f"\nVideo: {video_path}")
    print(f"  {total_frames} frames @ {fps:.1f}fps ({duration_sec:.1f}s)")
    print(f"  Game: {game.display_name}")
    if bands.bottom:
        # The bottom OCR band's top edge: the game's nameplate band (or the Ultimate widening
        # knob), resolved by main() via games.ingame_band_top_for; and the engine reading it.
        print(f"  Ingame nameplate band: from {ingame_tag_frac:.2f} of frame height down, "
              f"read with {band_ocr_engine}")
    elif ingame_band != INGAME_BAND_NONE:
        # The band would be read, but the layout draws the nameplates as boxes: those box
        # reads replace the band search, so the whole band pass is skipped (band_ocr_plan).
        print(f"  Ingame nameplate band: not read; the layout's nameplate boxes are read "
              f"instead, with {nameplate_engine}")
    if not bands.top:
        print("  Tournament-overlay band: not read; the layout's boxes cover it")
    print(f"  Inference every {every_n} frame(s)")
    if overlay_layout is not None:
        clock_source = "the layout's timer box" if overlay_layout.timer is not None else "the timer detector"
        clock = overlay_layout.clock_seconds
        print(f"  Custom overlay layout: {overlay_layout.name!r} "
              f"(boxes: {', '.join(overlay_layout.regions())}); clock from {clock_source}, "
              f"full clock {clock // 60}:{clock % 60:02d}")
    print()

    events: list[Event] = []   # the run's event log, saved as the pipeline JSON's "events"
    frame_count          = 0
    start_time           = time.time()
    last_preview_write   = 0.0                           # throttle embedded-preview writes
    last_progress_write  = 0.0                           # throttle progress-file writes
    ocr_interval         = int(fps * ocr_interval_sec)   # periodic OCR cadence
    next_periodic_ocr    = ocr_interval                  # first trigger one interval in
    # The detector path runs the game's own clock detector, and a layout's own clock length
    # (when one is set) still bounds what a recovered read can plausibly be.
    full_game_seconds = full_game_seconds_for(
        game, overlay_layout.clock_seconds if overlay_layout is not None else None)
    timer_recovery = TimerRecovery(                      # recover clock reads the detector drops
        max_clock_seconds=max_plausible_clock(full_game_seconds))

    # Guarantee the decode thread and its VideoCapture are released even if inference,
    # OCR, or a disk write raises mid-loop; otherwise the FrameReader thread leaks it.
    try:
        while not reader.stopped:
            try:
                frame_number, frame = reader.read()
            except queue.Empty:
                break

            frame_count += 1
            timestamp    = frame_number / fps

            # Progress heartbeat for the GUI's progress bar; before the frame skip, so it
            # keeps beating whatever the inference cadence.
            if progress_file:
                now = time.time()
                if now - last_progress_write >= PROGRESS_WRITE_INTERVAL:
                    write_progress(progress_file, timestamp, duration_sec,
                                   clock_calibration=clock_status, side_overlay=side_status)
                    last_progress_write = now

            # ── Step 1: Skip frames ────────────────────────────────────────────
            if frame_count % every_n != 0:
                continue

            # ── Step 2: Logo detection (go/game mode only) ─────────────────────
            logos    = []
            texts    = []
            decision = None
            sampled: Event | None = None   # this frame's periodic-OCR event, for the preview

            if det.logo is not None:
                logos = detect_logos(det.logo, frame, timestamp, frame_number,
                                     conf=conf, infer_size=infer_size, device=det.device)
                logos = [l for l in logos if l.confidence >= 0.9]

            if logos:
                for logo in logos:
                    print(f"  {logo.label} @ {timestamp:.2f}s  conf={logo.confidence:.0%}")

                # ── Step 3: User logic ─────────────────────────────────────────
                decision = on_logo_detected(logos, frame, timestamp, events)

                # ── Step 4: OCR ────────────────────────────────────────────────
                if decision.run_ocr:
                    texts = _splash_frame_ocr(det, timer_recovery, events,
                                              frame, timestamp, frame_number,
                                              infer_size=infer_size, text_conf=text_conf,
                                              ocr_beamsearch=ocr_beamsearch,
                                              ocr_canvas=ocr_canvas,
                                              ingame_tag_frac=ingame_tag_frac,
                                              layout_boxes=layout_boxes, bands=bands)

            # ── Step 4b: Periodic OCR (every ocr_interval_sec; skip if logo OCR just ran) ──
            if det.ocr is not None and frame_number >= next_periodic_ocr:
                next_periodic_ocr = frame_number + ocr_interval
                ocr_ran_this_frame = decision is not None and decision.run_ocr
                if not ocr_ran_this_frame:
                    sampled = _periodic_ocr_event(
                        det, timer_recovery, frame, timestamp, frame_number,
                        infer_size=infer_size, text_conf=text_conf,
                        ocr_beamsearch=ocr_beamsearch, ocr_canvas=ocr_canvas,
                        ocr_interval_sec=ocr_interval_sec,
                        gate_ocr_on_timer=gate_ocr_on_timer, roi_cache=roi_cache,
                        ingame_tag_frac=ingame_tag_frac, layout_boxes=layout_boxes,
                        bands=bands)
                    events.append(sampled)

            # ── Step 5: Preview ────────────────────────────────────────────────
            if preview_file:
                now = time.time()
                if now - last_preview_write >= PREVIEW_WRITE_INTERVAL:
                    # What this sample was read from, boxed; a layout region that read
                    # nothing is outlined dotted (nothing on a frame no sample was taken on).
                    marks = (sample_preview_marks(
                                 sampled, layout_boxes.regions if layout_boxes is not None else {})
                             if sampled is not None else None)
                    render_preview(frame, logos, texts, timestamp, duration_sec, preview_file,
                                   marks=marks)
                    last_preview_write = now
            elif show_preview and render_preview(frame, logos, texts, timestamp, duration_sec):
                break

            # ── Step 6: Seek ───────────────────────────────────────────────────
            if seek_every > 0:
                target = frame_number + int(seek_every * fps)
                reader.seek(target)
                frame_count = 0
            elif decision is not None and (decision.seek_to is not None or decision.seek_by is not None):
                frame_count = apply_seek(decision, reader, frame_number, fps)
        if progress_file:
            # Land the bar on 100%; the loop usually ends mid-seek, short of the last heartbeat.
            write_progress(progress_file, duration_sec, duration_sec,
                           clock_calibration=clock_status, side_overlay=side_status)
    finally:
        reader.stop()
        cv2.destroyAllWindows()
    elapsed = time.time() - start_time

    mode = "seek" if seek_every > 0 else "go/game"
    # Recorded for the diagnostics bundle and for anyone re-deriving what a run loaded; None
    # for a game with no detectors, which loaded neither. The character and stock paths are
    # the LOADED weights (None when a declared weight was missing), since extract keys the
    # splitter's fighter check and its stock-icon note on them.
    timer_model_path = _declared_weight_path(game, "timer")
    character_model_path = _loaded_character_model_path(game, det)
    stock_model_path = _loaded_stock_model_path(game, det)
    output_data: PipelineData = {
        "config": {
            "mode": mode,
            "vod_filename": Path(video_path).name,
            "video_path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration_sec,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "elapsed_sec": round(elapsed, 1),   # wall time of the whole run (the "Done in"
                                               # line), so GUI runs, which keep no console
                                               # log, can be timed against each other
            "clock_calibration": calibration.as_config() if calibration is not None else None,
            "side_overlay": side_record.config if side_record is not None else None,
            "inference_settings": {
                "logo_conf": conf,
                "infer_size": infer_size,
                "every_n": every_n,
                "text_conf": text_conf,
                "use_ocr": use_ocr,
                "ocr_interval_sec": ocr_interval_sec,
                "ocr_beamsearch": ocr_beamsearch,
                "ocr_canvas": ocr_canvas,
                "gate_ocr_on_timer": gate_ocr_on_timer,
                "ocr_roi_cache": ocr_roi_cache,
                "ocr_lang": ocr_lang,
                "ocr_engine": ocr_engine,
                "ingame_band_ocr_engine": band_ocr_engine,  # what read the bottom band (extract
                                                           # word-merges that band's docTR reads)
                "ingame_tag_band_frac": ingame_tag_frac,   # extract_matches reads this back so its
                                                           # nametag gate matches this run's OCR crop
                "band_ocr_top": bands.top,                 # which of the two overlay OCR bands this
                "band_ocr_bottom": bands.bottom,           # run read (band_ocr_plan): a band with no
                                                           # consumer is skipped whole. extract_matches
                                                           # reads the bottom one back to skip its
                                                           # at-bottom tag reading
                "game": game.id,                           # extract_matches reads this back too, for
                                                           # the game's in-game nameplate profile
                "timer_model_path": timer_model_path,
                "character_model_path": character_model_path,
                "stock_model_path": stock_model_path,
                "seek_every": seek_every,
            },
        },
        "events": events,
    }
    if overlay_layout is not None:
        # Recorded so extract_matches reads the same layout back (see
        # extract_matches.resolve_overlay_layout), the way ingame_tag_band_frac is, and the
        # text regions it read per sample beside it (resolve_overlay_regions votes those).
        output_data["config"]["inference_settings"]["overlay_layout"] = layout_to_dict(overlay_layout)
    if overlay_regions is not None:
        # The text regions read per sample, whether a layout drew them or the side-bar lock
        # found them (extract_matches.resolve_overlay_regions votes those).
        output_data["config"]["inference_settings"]["overlay_regions"] = overlay_regions.as_config()

    output_path = output_dir(vod_stem(video_path)) / "pipeline.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, output_data)

    # Dump the implausible/garbage clock reads the run threw out (OCR that wouldn't parse, plus
    # recovery reads that failed the plausibility gate) so they can be inspected; they're otherwise
    # nulled out and gone. Single flush point: the TimerRecovery accumulated them across the run.
    # flush_rejected_timers clears any stale file first, so a clean re-run never ships an old log.
    rej_path = flush_rejected_timers(timer_recovery.rejections, vod_stem(video_path))
    if rej_path is not None:
        n = len(timer_recovery.rejections)
        capped = " (capped; there were more)" if n >= MAX_RECORDED_REJECTIONS else ""
        print(f"  {n} rejected timer reading(s){capped} → {rej_path}")

    print(f"\nDone in {elapsed:.1f}s: {len(events)} events saved to {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def artifact_folder_note(video_path: str) -> str | None:
    """The note to print when a video's artifact folder is not named after the file, or the
    folder looks misnamed; None when it is. vod_stem drops the ".converted" of the GUI's MP4
    conversion, so the conversion shares the original's folder, and it sanitizes filename
    oddities (e.g. a trailing space) that Windows can't address as a folder name."""
    file_stem = Path(video_path).stem
    folder_stem = vod_stem(video_path)
    if strip_converted_suffix(file_stem) != file_stem:
        return (f"Note: this is a converted MP4, so it shares the original video's artifacts "
                f"in json_output/{folder_stem}/.")
    if folder_stem != file_stem:
        return (f"Note: filename sanitized for folder use; artifacts for this video "
                f"land in json_output/{folder_stem}/.")
    return None


def main() -> None:
    """Parse CLI args and run the detection pipeline on one video."""
    prepare_stage_streams()

    parser = argparse.ArgumentParser()
    parser.add_argument("--video",          required=True,             help="Path to input video")
    parser.add_argument("--conf",           type=float, default=0.5,   help="Logo confidence threshold")
    parser.add_argument("--size",           type=int,   default=DEFAULT_INFER_SIZE,
                        help="Inference resolution (long side, px) for every detector; a game "
                             "may set its own for its clock detector")
    parser.add_argument("--every",          type=int,   default=1,     help="Run inference every N frames")
    parser.add_argument("--text-conf",      type=float, default=0.5,   help="OCR confidence threshold")
    parser.add_argument("--no-ocr",         action="store_true",        help="Skip loading easyocr")
    parser.add_argument("--preview",        action="store_true",        help="Show preview window")
    parser.add_argument("--preview-file",   default=None,               help="Write annotated preview frames here (for an embedded GUI preview) instead of a window")
    parser.add_argument("--progress-file",  default=None,               help="Write run progress JSON here (for the GUI's progress bar)")
    parser.add_argument("--seek-every",     type=float, default=30.0,  help="Seconds to jump forward between sampled frames (seek mode, the default). Use 0 for the old per-frame logo detection (needs weights/logo.pt)")
    parser.add_argument("--ocr-interval",   type=float, default=30.0,  help="Run periodic OCR every N seconds of video (default: 30)")
    parser.add_argument("--ocr-beamsearch", action="store_true",        help="Use beamsearch decoder for OCR (more accurate, ~20%% slower; EasyOCR engine only)")
    parser.add_argument("--ocr-canvas",     type=int,   default=OCR_CANVAS_DEFAULT, help="Cap the text-detection input's longest side, in pixels; lower is faster but misses small text (the recognizer always reads full resolution). Default 2560 = no downscaling at 1080p. EasyOCR only")
    parser.add_argument("--gate-ocr-on-timer", action="store_true",      help="Skip full-frame OCR on frames with no readable timer (this means we don't run on sections where there's no gameplay). Those frames still count for game splitting, but their tags and banners go unread")
    parser.add_argument("--ocr-roi-cache",  action="store_true",         help="Learn where this VOD's text sits, then read only those spots, rechecking periodically and at game boundaries. Much faster, but text that appears somewhere new is missed until the next recheck")
    parser.add_argument("--ingame-tag-band-frac", type=parse_ingame_tag_band_frac, default=None,
                        help="Ultimate only: fraction of frame height where the ingame nametag "
                             f"OCR band starts (default: {INGAME_TAG_BAND_FRAC_DEFAULT}, the bottom "
                             "8%%). Lower it to about 0.88 for captures shrunk by the Switch's "
                             "'Adjust Screen Size' slider, which lifts the nametags out of the "
                             "default band and costs some OCR speed. Floor "
                             f"{INGAME_TAG_BAND_FRAC_FLOOR}: below that the band reaches the damage "
                             "counters. Other games always read at their own nameplate band "
                             "(vodparse.games) and ignore this flag")
    parser.add_argument("--ocr-lang",       choices=sorted(OCR_LANG_LABELS), default=OCR_LANG_DEFAULT,
                        help="Also read one non-English script alongside English (default: en, English only). Picking another language downloads its recognition model (~60 MB) the first time")
    parser.add_argument("--ocr-engine",     choices=sorted(OCR_ENGINE_LABELS), default=OCR_ENGINE_DEFAULT,
                        help="Which engine reads the overlay text bands (default: easyocr). "
                             "doctr (DBNet + PARSeq) is often better at stylized overlay fonts, but reads "
                             "English/Latin text only, and downloads its models "
                             f"(~{DOCTR_WEIGHTS_MB_APPROX} MB) on first use. "
                             "The match clock is always read with EasyOCR")
    parser.add_argument("--ingame-band-engine",
                        choices=[INGAME_BAND_ENGINE_PINNED, INGAME_BAND_ENGINE_OVERLAY],
                        default=INGAME_BAND_ENGINE_PINNED,
                        help="Which engine reads the ingame nameplate band of a game that pins "
                             "its own (Rivals 2 reads its names with doctr). "
                             f"{INGAME_BAND_ENGINE_PINNED} (default) loads the game's engine, "
                             f"downloading its models on first use; {INGAME_BAND_ENGINE_OVERLAY} "
                             "reads the band with --ocr-engine instead. No effect on a game that "
                             "pins no engine")
    parser.add_argument("--game",           choices=[g.id for g in GAMES], default=GAME_DEFAULT,
                        help="Which game the VOD is of (default: "
                             f"{GAME_DEFAULT}). A game without a shipped clock detector needs "
                             "--overlay-layout naming a layout with a timer box, which then "
                             "supplies the clock that splits the VOD into games")
    parser.add_argument("--no-calibrate-clock", action="store_true",
                        help="Skip the pre-pass that finds where this VOD draws its clock; "
                             "the clock detector then reads every sample on its own")
    parser.add_argument("--no-locate-overlay", action="store_true",
                        help="Skip the pre-pass that finds where a 4:3 game's stream shows "
                             "the player tags in its side bars; the built-in band reading "
                             "then runs alone")
    parser.add_argument("--prepass-handover", type=Path, default=None, metavar="PATH",
                        help="A JSON file of what the GUI's preview already found for this "
                             "video (the clock box, the side-bar player tag rows). Each "
                             "answer that matches this run replaces its pre-pass; a missing "
                             "or mismatched file is ignored")
    parser.add_argument("--overlay-layout", default=None, metavar="NAME",
                        help="Read the overlay through a saved custom layout (drawn in the GUI: "
                             "the main screen's Overlay layout row → Edit…). Its "
                             "player-tag / round / best-of boxes are OCR'd directly instead of "
                             "located by the built-in heuristics, and its "
                             "timer box, when drawn, replaces the timer detector as the clock source "
                             "(what lets game splitting work for a game without a clock "
                             "detector). Recorded in pipeline.json, so extract_matches reads "
                             "the same layout back")
    parser.add_argument("--overlay-layout-file", type=Path, default=None, metavar="PATH",
                        help=f"The layout store to look --overlay-layout up in (default: "
                             f"{LAYOUT_STORE_NAME} in the workspace root, where the GUI saves it)")
    args = parser.parse_args()
    # argparse restricted --game to a registry id, so the lookup always yields a Game.
    game = next(g for g in GAMES if g.id == args.game)
    # The bottom OCR band starts at the game's own nameplate band; the widening flag is an
    # Ultimate calibration and only applies there (see games.ingame_band_top_for). Recorded
    # in the pipeline JSON, so extract gates on the exact band this run cropped.
    ingame_tag_frac = ingame_band_top_for(game, args.ingame_tag_band_frac)
    overlay_layout = resolve_overlay_layout_arg(args.overlay_layout, args.overlay_layout_file,
                                                game)
    if not game.has_timer_model and overlay_layout is None:
        # Without a clock there is nothing to split the VOD on, and the built-in reading only
        # knows Ultimate's graphics: refuse rather than hand back one giant "match".
        parser.error(f"{game.display_name} has no clock detector: pass --overlay-layout NAME "
                     "naming a layout with a timer box (create one in the app: the main "
                     "screen's Overlay layout row → Edit…).")

    # Fail fast on a missing or undecodable video: FrameReader inside run() would raise the
    # same complaint, but only after the detector and OCR models spend a while loading.
    try:
        open_capture(args.video).release()
    except (FileNotFoundError, ValueError) as err:
        sys.exit(f"Error: {err}")

    note = artifact_folder_note(args.video)
    if note is not None:
        print(note)

    run(
        video_path=args.video,
        game=game,
        conf=args.conf,
        infer_size=args.size,
        every_n=args.every,
        text_conf=args.text_conf,
        use_ocr=not args.no_ocr,
        show_preview=args.preview,
        seek_every=args.seek_every,
        ocr_interval_sec=args.ocr_interval,
        ocr_beamsearch=args.ocr_beamsearch,
        ocr_canvas=args.ocr_canvas,
        gate_ocr_on_timer=args.gate_ocr_on_timer,
        ocr_roi_cache=args.ocr_roi_cache,
        ocr_lang=args.ocr_lang,
        ocr_engine=args.ocr_engine,
        ingame_tag_frac=ingame_tag_frac,
        preview_file=args.preview_file,
        progress_file=args.progress_file,
        overlay_layout=overlay_layout,
        calibrate_clock=not args.no_calibrate_clock,
        locate_overlay=not args.no_locate_overlay,
        band_on_overlay_engine=args.ingame_band_engine == INGAME_BAND_ENGINE_OVERLAY,
        prepass_handover=(read_handover(args.prepass_handover)
                          if args.prepass_handover is not None else None),
    )


def resolve_overlay_layout_arg(name: str | None, store: Path | None,
                               game: Game) -> CustomOverlayLayout | None:
    """The layout ``--overlay-layout NAME`` names, looked up in ``store`` (the GUI's shared
    store when None) and checked to be a preset for ``game`` (``--game``). A name that
    isn't there, or a preset drawn for another game, is a hard error with guidance:
    silently running the built-in reading instead would hand the user an analysis that
    ignores the layout they drew, and a Melee preset read as Ultimate would
    split on the wrong clock."""
    if name is None:
        return None
    path = store if store is not None else layouts_path()
    layout = layout_by_name(load_layouts(path), name)
    if layout is None:
        raise SystemExit(f"error: no custom overlay layout named {name!r} in {path}. Save one "
                         "in the GUI (the main screen's Overlay layout row → Edit…) "
                         "or point --overlay-layout-file at the store that has it.")
    if layout.game != game.id:
        layout_game = game_by_id(layout.game)
        layout_game_name = layout_game.display_name if layout_game is not None else layout.game
        raise SystemExit(f"error: the overlay layout {layout.name!r} was saved for "
                         f"{layout_game_name}, but --game is {game.display_name}. Pick a layout "
                         f"saved for {game.display_name}, or run with the game the layout is for.")
    return layout


if __name__ == "__main__":
    main()
