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
from dataclasses import dataclass
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

from vodparse import (CHARACTER_MODEL_PATH, LOGO_MODEL_PATH, M_CHAR_CLASSIFIER_PATH,
                      TIMER_MODEL_PATH)
from vodparse.device import resolve_device
from vodparse.cli.stage_io import prepare_stage_streams
from vodparse.jsonio import write_json_atomic
from vodparse.paths import output_dir, vod_stem
from vodparse.frame_reader import FrameReader, open_capture
from vodparse.schema import Event, PipelineData, PipelineProgress, SideCharacters, TextBox
from vodparse.timer import TimerRecovery, detect_timer_with_recovery, MAX_RECORDED_REJECTIONS
from vodparse.debug import flush_rejected_timers
from vodparse.characters import detect_characters
from vodparse.m_char import MCharClassifier, load_m_char_classifier, refine_characters
from vodparse.ocr_engine import (DOCTR_WEIGHTS_MB_APPROX, OCR_ENGINE_DEFAULT, OCR_ENGINE_DOCTR,
                                 OCR_ENGINE_LABELS, EasyOcrOverlayEngine, OverlayOcr,
                                 make_doctr_engine, validate_engine_lang)
from vodparse.ocr_langs import OCR_LANG_DEFAULT, OCR_LANG_LABELS, make_reader
from vodparse.ocr_roi import OcrRoiCache
from vodparse.tags import (INGAME_TAG_BAND_FRAC_DEFAULT, INGAME_TAG_BAND_FRAC_FLOOR,
                           TAG_TOP_FRAC, OVERLAY_BOTTOM_FRAC, parse_ingame_tag_band_frac)


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


def render_preview(frame: np.ndarray, logos: list[LogoDetection], texts: list[TextDetection],
                   timestamp_sec: float, duration_sec: float,
                   preview_file: str | None = None) -> bool:
    """Draw boxes on a copy of frame and either show it in an OpenCV window or write it to
    `preview_file` (for an embedded GUI preview). Returns True if the user pressed Q (window
    mode only)."""
    display = frame.copy()
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
    pct = (timestamp_sec / duration_sec * 100) if duration_sec > 0 else 0
    cv2.putText(display, f"{timestamp_sec:.1f}s  ({pct:.1f}%)", (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

    if preview_file is not None:
        # Encode to JPEG bytes and write+rename atomically so the GUI never reads a half-written
        # frame. (imencode picks the codec from the extension, not the temp filename.)
        ok, buf = cv2.imencode(".jpg", display)
        if ok:
            _write_preview_frame(buf.tobytes(), preview_file)
        return False

    cv2.imshow("Pipeline", display)
    return cv2.waitKey(1) & 0xFF == ord("q")


def write_progress(progress_file: str, timestamp_sec: float, duration_sec: float) -> None:
    """Report how far through the video the main loop has reached, for the GUI's
    progress bar. Atomic for the same reason as the preview frames: the GUI polls
    the file and must never read a half-written one.

    Same Windows sharing-violation race as the preview frame can hit the replace inside
    write_json_atomic (the GUI's poll can have the file open at that instant); one skipped
    ~250ms progress tick is invisible, so it's dropped rather than allowed to kill the run.
    write_json_atomic already cleans up its own tmp file before re-raising, so there's nothing
    left to do here but swallow the error."""
    progress: PipelineProgress = {"timestamp_sec": round(timestamp_sec, 1),
                                  "duration_sec": round(duration_sec, 1)}
    try:
        write_json_atomic(progress_file, progress)
    except OSError:
        pass


# Cap on the CRAFT text-DETECTION input: EasyOCR downscales a band until its longest side fits
# canvas_size before running the detector, then maps the found boxes back to source pixels and
# recognizes from the ORIGINAL-resolution crops. So lowering the cap makes the expensive
# detection pass cheaper (cost ~ pixels) without touching recognition quality; the trade is
# purely detection recall on small text. 2560 is EasyOCR's own default: a 1920-wide 1080p band
# passes through unscaled, i.e. today's behavior.
OCR_CANVAS_DEFAULT = 2560


def detect_text(ocr: OverlayOcr, frame: np.ndarray, timestamp_sec: float, frame_number: int,
                conf_threshold: float = 0.5, beamsearch: bool = False,
                canvas_size: int = OCR_CANVAS_DEFAULT,
                ingame_tag_frac: float = INGAME_TAG_BAND_FRAC_DEFAULT) -> list[TextDetection]:
    """Run the overlay OCR engine on the top/bottom bands of a frame and return text
    detections above `conf_threshold`. `ingame_tag_frac` lowers the bottom band's top edge
    so a shrunk capture's lifted in-game nametags still get OCR'd (see
    tags.INGAME_TAG_BAND_FRAC_DEFAULT); it defaults to the fixed 0.92 band."""
    # OCR only the top and bottom bands of the frame, not the middle. Every text consumer in
    # vodparse.tags reads a box only if its center is in the outer bands (tournament tags/
    # round/best-of above h*TAG_TOP_FRAC; in-game names and the at-bottom overlay below
    # h*OVERLAY_BOTTOM_FRAC: the game renders its nametags at a fixed HUD position inside
    # that strip, see the constant's comment), so the rest is dead weight; running CRAFT over
    # it only costs time and invites false reads (stage text, the "Combatants" panel, the
    # 3-2-1 countdown, spell menus, damage-counter misreads). We crop to those exact fractions
    # (imported so the crop and the gates can never drift apart). The bands are asymmetric: the
    # top is a third to fit the round/best-of second row (see TAG_TOP_FRAC).
    h = frame.shape[0]
    top_h = int(h * TAG_TOP_FRAC)
    # The crop must cover every bottom-band gate, so it reaches up to the *highest* of them:
    # the tunable in-game nametag band (INGAME_TAG_BAND_FRAC_DEFAULT), which a shrunk capture
    # lowers below the fixed 0.92 the tournament reader/row learner still use. min() keeps the
    # invariant that no gate references a pixel the crop never OCR'd (tags' constant comment).
    bot_y = int(h * min(ingame_tag_frac, OVERLAY_BOTTOM_FRAC))
    # (y_offset, band): adding y_offset maps a band-local point back to full-frame coordinates,
    # so every stored bbox is identical to what full-frame OCR would yield and the position math
    # in vodparse.tags is unaffected. Bands span the full width, so x needs no offset.
    bands = [(0, frame[:top_h]), (bot_y, frame[bot_y:])]

    detections = []
    for y_off, band in bands:
        for (bbox_pts, text, conf) in ocr.read_band(band, beamsearch=beamsearch,
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


@dataclass(frozen=True)
class _Detectors:
    """The models one run loads, bundled so the per-frame OCR helpers don't each take five
    model parameters. Any member (except device) may be None: absent weights degrade that
    detector to a no-op, and `ocr` is None under --no-ocr.

    `ocr` is the overlay-band engine (EasyOCR or docTR, per --ocr-engine); `overlay_reader`
    is the concrete easyocr.Reader behind it when the engine is EasyOCR, for the two paths
    that stay EasyOCR-specific (the OCR ROI cache and the timer-reader reuse below); it is
    None under the docTR engine or --no-ocr.

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


def _load_detectors(seek_every: float, use_ocr: bool, ocr_lang: str,
                    ocr_engine: str) -> _Detectors:
    """Pick the device and load every model the run needs (YOLO detectors, the m_char
    classifier cascade, the overlay OCR engine, the EasyOCR timer reader)."""
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
    timer_model = YOLO(TIMER_MODEL_PATH) if Path(TIMER_MODEL_PATH).exists() else None
    character_model = YOLO(CHARACTER_MODEL_PATH) if Path(CHARACTER_MODEL_PATH).exists() else None
    # Cascades after the character detector to name the three m_char classes; no-op if absent.
    m_char_classifier = load_m_char_classifier(M_CHAR_CLASSIFIER_PATH, device)
    loaded = [name for name, m in (("logo", model), ("timer", timer_model),
                                   ("character", character_model),
                                   ("m_char", m_char_classifier)) if m is not None]
    print(f"  models: {', '.join(loaded) if loaded else 'none'}")

    ocr: OverlayOcr | None = None
    overlay_reader: easyocr.Reader | None = None
    if use_ocr:
        if ocr_engine == OCR_ENGINE_DOCTR:
            ocr = make_doctr_engine(device)
        else:
            overlay_reader = make_reader(ocr_lang, gpu=True)
            ocr = EasyOcrOverlayEngine(overlay_reader)
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
    return _Detectors(model, timer_model, character_model, m_char_classifier,
                      ocr, overlay_reader, timer_ocr, device)


def _splash_frame_ocr(det: _Detectors, timer_recovery: TimerRecovery, events: list[Event],
                      frame: np.ndarray, timestamp: float, frame_number: int, *,
                      infer_size: int, text_conf: float,
                      ocr_beamsearch: bool, ocr_canvas: int,
                      ingame_tag_frac: float) -> list[TextDetection]:
    """The full-frame OCR pass on a go/game splash frame: read texts, the clock, and the
    character portraits, enriching the just-appended splash event in place. Returns the
    detected texts (the preview overlay draws them)."""
    if det.ocr is None:
        print("  [warning] OCR requested but --no-ocr was set, skipping")
        return []
    texts = detect_text(det.ocr, frame, timestamp, frame_number,
                        conf_threshold=text_conf, beamsearch=ocr_beamsearch,
                        canvas_size=ocr_canvas, ingame_tag_frac=ingame_tag_frac)
    if events:
        events[-1]["texts"] = texts_to_payload(texts)
    if det.timer is not None:
        timer = detect_timer_with_recovery(timer_recovery, det.timer, det.timer_ocr,
                                           frame, det.device, infer_size, timestamp)
        if events:
            events[-1]["timer"] = timer
        print(f"    timer: {timer['formatted']}" if timer else "    timer: not detected")
    if det.character is not None:
        characters = detect_characters(det.character, frame, det.device, infer_size)
        characters = refine_characters(characters, frame, det.m_char)
        if events:
            events[-1]["characters"] = characters
        print(f"    characters: {format_characters(characters)}")
    if texts:
        print(f"    OCR: {[t.text for t in texts]}")
    return texts


def _periodic_ocr_event(det: _Detectors, timer_recovery: TimerRecovery,
                        frame: np.ndarray, timestamp: float, frame_number: int, *,
                        infer_size: int, text_conf: float, ocr_beamsearch: bool,
                        ocr_canvas: int, ocr_interval_sec: float, gate_ocr_on_timer: bool,
                        roi_cache: OcrRoiCache | None, ingame_tag_frac: float) -> Event:
    """One periodic-OCR sample: read the clock, (unless gated) the full-frame texts, and the
    character portraits, and return the label-less Event to append. The caller guarantees
    det.ocr is loaded (periodic OCR never fires under --no-ocr)."""
    assert det.ocr is not None
    # Timer first: it always runs (its own cheap localized digit OCR, separate from
    # the full-frame pass) and feeds the recovery state machine every sample.
    timer = (detect_timer_with_recovery(timer_recovery, det.timer, det.timer_ocr,
                                        frame, det.device, infer_size, timestamp)
             if det.timer is not None else None)
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
    if gated:
        texts_payload: list[TextBox] = []
    elif roi is not None and not roi.should_run_full():
        # ROI cache, steady state: recognizer-only over the calibrated
        # text regions; the CRAFT detection pass (85-89% of the full-pass cost) is skipped.
        assert det.overlay_reader is not None  # run() creates the cache only for EasyOCR
        texts_payload = roi.read_cached(
            det.overlay_reader, frame, conf_threshold=text_conf,
            decoder="beamsearch" if ocr_beamsearch else "greedy")
        roi_cached = True
    else:
        texts_payload = texts_to_payload(
            detect_text(det.ocr, frame, timestamp, frame_number,
                        conf_threshold=text_conf, beamsearch=ocr_beamsearch,
                        canvas_size=ocr_canvas, ingame_tag_frac=ingame_tag_frac))
        if roi is not None:
            roi.observe_full(texts_payload, frame.shape[0])
    event: Event = {
        "frame_number":  frame_number,
        "timestamp_sec": round(timestamp, 3),
        "label":         None,
        "texts": texts_payload,
        "timer": timer,
        "characters": refine_characters(
            detect_characters(det.character, frame, det.device, infer_size)
            if det.character is not None else None,
            frame, det.m_char),
    }
    if gated:
        event["ocr_gated"] = True
    summary = ("gated (no timer)" if gated else
               f"{len(texts_payload)} text(s)" + (" (roi-cached)" if roi_cached else ""))
    print(f"  [periodic OCR] @ {timestamp:.2f}s: {summary}")
    print(f"    timer: {timer['formatted']}" if timer else "    timer: not detected")
    if det.character is not None:
        print(f"    characters: {format_characters(event.get('characters'))}")
    if texts_payload:
        print(f"    OCR: {[t['text'] for t in texts_payload]}")
    return event


def run(video_path: str, conf: float = 0.5,
        infer_size: int = 320, every_n: int = 1,   # 320 matches the CLI --size default, i.e. what
                                                   # every GUI run uses (the GUI passes no --size)
        text_conf: float = 0.5, use_ocr: bool = True,
        show_preview: bool = False, seek_every: float = 30.0,
        ocr_interval_sec: float = 30.0, ocr_beamsearch: bool = False,
        ocr_canvas: int = OCR_CANVAS_DEFAULT, gate_ocr_on_timer: bool = False,
        ocr_roi_cache: bool = False, ocr_lang: str = OCR_LANG_DEFAULT,
        ocr_engine: str = OCR_ENGINE_DEFAULT,
        ingame_tag_frac: float = INGAME_TAG_BAND_FRAC_DEFAULT,
        preview_file: str | None = None, progress_file: str | None = None) -> None:

    det = _load_detectors(seek_every, use_ocr, ocr_lang, ocr_engine)
    # Per-VOD text-region calibration: full CRAFT on a few frames, then
    # recognizer-only on the cached regions (see vodparse.ocr_roi). Seek-mode periodic OCR
    # only; the go/game splash path below always runs the full pass (splashes are rare).
    # EasyOCR-only: the cache's recognizer-only reads reuse EasyOCR's CRAFT boxes, which
    # the docTR engine doesn't have; under docTR every periodic sample takes the full pass.
    if ocr_roi_cache and det.ocr is not None and det.overlay_reader is None:
        print("Note: --ocr-roi-cache is EasyOCR-specific; disabled for the docTR engine.")
    # The cache's bottom band mirrors detect_text's crop, so widen it the same way.
    roi_bottom_frac = min(ingame_tag_frac, OVERLAY_BOTTOM_FRAC)
    roi_cache = (OcrRoiCache(bottom_frac=roi_bottom_frac)
                 if (ocr_roi_cache and det.overlay_reader is not None) else None)

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

    print(f"\nVideo: {video_path}")
    print(f"  {total_frames} frames @ {fps:.1f}fps ({duration_sec:.1f}s)")
    print(f"  Inference every {every_n} frame(s)\n")

    events: list[Event] = []   # the run's event log, saved as the pipeline JSON's "events"
    frame_count          = 0
    start_time           = time.time()
    last_preview_write   = 0.0                           # throttle embedded-preview writes
    last_progress_write  = 0.0                           # throttle progress-file writes
    ocr_interval         = int(fps * ocr_interval_sec)   # periodic OCR cadence
    next_periodic_ocr    = ocr_interval                  # first trigger one interval in
    timer_recovery       = TimerRecovery()               # recover clock reads the detector drops

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
                    write_progress(progress_file, timestamp, duration_sec)
                    last_progress_write = now

            # ── Step 1: Skip frames ────────────────────────────────────────────
            if frame_count % every_n != 0:
                continue

            # ── Step 2: Logo detection (go/game mode only) ─────────────────────
            logos    = []
            texts    = []
            decision = None

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
                                              ingame_tag_frac=ingame_tag_frac)

            # ── Step 4b: Periodic OCR (every ocr_interval_sec; skip if logo OCR just ran) ──
            if det.ocr is not None and frame_number >= next_periodic_ocr:
                next_periodic_ocr = frame_number + ocr_interval
                ocr_ran_this_frame = decision is not None and decision.run_ocr
                if not ocr_ran_this_frame:
                    events.append(_periodic_ocr_event(
                        det, timer_recovery, frame, timestamp, frame_number,
                        infer_size=infer_size, text_conf=text_conf,
                        ocr_beamsearch=ocr_beamsearch, ocr_canvas=ocr_canvas,
                        ocr_interval_sec=ocr_interval_sec,
                        gate_ocr_on_timer=gate_ocr_on_timer, roi_cache=roi_cache,
                        ingame_tag_frac=ingame_tag_frac))

            # ── Step 5: Preview ────────────────────────────────────────────────
            if preview_file:
                now = time.time()
                if now - last_preview_write >= PREVIEW_WRITE_INTERVAL:
                    render_preview(frame, logos, texts, timestamp, duration_sec, preview_file)
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
            write_progress(progress_file, duration_sec, duration_sec)
    finally:
        reader.stop()
        cv2.destroyAllWindows()
    elapsed = time.time() - start_time

    mode = "seek" if seek_every > 0 else "go/game"
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
                "ingame_tag_band_frac": ingame_tag_frac,   # extract_matches reads this back so its
                                                           # nametag gate matches this run's OCR crop
                "timer_model_path": TIMER_MODEL_PATH,
                "character_model_path": CHARACTER_MODEL_PATH,
                "seek_every": seek_every,
            },
        },
        "events": events,
    }

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

def main() -> None:
    """Parse CLI args and run the detection pipeline on one video."""
    prepare_stage_streams()

    parser = argparse.ArgumentParser()
    parser.add_argument("--video",          required=True,             help="Path to input video")
    parser.add_argument("--conf",           type=float, default=0.5,   help="Logo confidence threshold")
    parser.add_argument("--size",           type=int,   default=320,   help="Inference resolution")
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
    parser.add_argument("--ingame-tag-band-frac", type=parse_ingame_tag_band_frac, default=INGAME_TAG_BAND_FRAC_DEFAULT,
                        help="Fraction of frame height where the ingame nametag OCR band starts "
                             f"(default: {INGAME_TAG_BAND_FRAC_DEFAULT}, the bottom 8%%). Lower it to "
                             "about 0.88 for captures shrunk by the Switch's 'Adjust Screen Size' "
                             "slider, which lifts the nametags out of the default band and costs "
                             f"some OCR speed. Floor {INGAME_TAG_BAND_FRAC_FLOOR}: below that the "
                             "band reaches the damage counters")
    parser.add_argument("--ocr-lang",       choices=sorted(OCR_LANG_LABELS), default=OCR_LANG_DEFAULT,
                        help="Also read one non-English script alongside English (default: en, English only). Picking another language downloads its recognition model (~60 MB) the first time")
    parser.add_argument("--ocr-engine",     choices=sorted(OCR_ENGINE_LABELS), default=OCR_ENGINE_DEFAULT,
                        help="Which engine reads the overlay text bands (default: easyocr). "
                             "doctr (DBNet + PARSeq) is often better at stylized overlay fonts, but reads "
                             "English/Latin text only, and downloads its models "
                             f"(~{DOCTR_WEIGHTS_MB_APPROX} MB) on first use. "
                             "The match clock is always read with EasyOCR")
    args = parser.parse_args()

    # Fail fast on a missing or undecodable video: FrameReader inside run() would raise the
    # same complaint, but only after the detector and OCR models spend a while loading.
    try:
        open_capture(args.video).release()
    except (FileNotFoundError, ValueError) as err:
        sys.exit(f"Error: {err}")

    # vod_stem sanitizes filename oddities (e.g. a trailing space) that Windows can't
    # address as a folder name; say so, or the artifact folder looks misnamed.
    sanitized = vod_stem(args.video)
    if sanitized != Path(args.video).stem:
        print(f"Note: filename sanitized for folder use; artifacts for this video "
              f"land in json_output/{sanitized}/.")

    run(
        video_path=args.video,
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
        ingame_tag_frac=args.ingame_tag_band_frac,
        preview_file=args.preview_file,
        progress_file=args.progress_file,
    )


if __name__ == "__main__":
    main()
