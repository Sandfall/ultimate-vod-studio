"""Selectable OCR engine for the overlay text bands (player names, sponsors, tags).

Two engines can read the top/bottom overlay bands that ``vodparse.cli.pipeline``'s
``detect_text`` crops out of each sampled frame:

* **EasyOCR** (:data:`OCR_ENGINE_EASYOCR`, the default): the reader built by
  :func:`vodparse.ocr_langs.make_reader`, wrapped in :class:`EasyOcrOverlayEngine`.
* **docTR** (:data:`OCR_ENGINE_DOCTR`): DBNet detection + the PARSeq recognizer,
  which handles stylized overlay fonts better. Ships with the app (a core
  dependency; the frozen build also bundles its models: see ``freeze_entry.py``).
  English/Latin text only (see :data:`DOCTR_SUPPORTED_LANGS`).

The engine choice covers the overlay bands and a custom layout's boxes, including the
OCR ROI cache's recognizer-only reads (``vodparse.ocr_roi``, through
``OverlayOcr.recognize_boxes``). The match-clock crop (``vodparse.timer``) is always
read with EasyOCR: its recognizer-only ``recognize(allowlist=...)`` digit read has no
docTR equivalent.

Both engines return EasyOCR-shaped ``(bbox_points, text, confidence)`` triples
(:data:`OcrResult`), so ``detect_text``'s filtering and band-offset math is
engine-agnostic. Qt-free; easyocr/doctr are imported lazily so the GUI can import
this module for :func:`doctr_available` / :func:`doctr_weights_cached` /
:func:`engine_unusable_reason` without pulling in torch. When docTR can't run, nothing
reads with EasyOCR in its place: callers stop and show :func:`engine_unusable_reason`.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from vodparse.ocr_langs import make_reader

if TYPE_CHECKING:
    import easyocr
    import numpy as np

# Engine identifiers: the values behind the GUI's "Font style" choice and the CLI's
# --ocr-engine flag, and what pipeline.json records as inference_settings.ocr_engine.
OCR_ENGINE_EASYOCR = "easyocr"
OCR_ENGINE_DOCTR = "doctr"
OCR_ENGINE_DEFAULT = OCR_ENGINE_EASYOCR

# Display label for each engine, and the source of truth for what --ocr-engine accepts.
# Keep in sync with vodparse.gui.settings.OCR_ENGINE_CHOICES (GUI-specific labels).
OCR_ENGINE_LABELS: dict[str, str] = {
    OCR_ENGINE_EASYOCR: "EasyOCR",
    OCR_ENGINE_DOCTR: "docTR (PARSeq)",
}

# The Font style dropdown's name for each engine. The dropdown is built from this table
# (vodparse.gui.settings.OCR_ENGINE_CHOICES) and every message that names a style takes the
# name from here, so a renamed style can't leave a message naming the old one.
FONT_STYLE_NAMES: dict[str, str] = {
    OCR_ENGINE_EASYOCR: "Standard",
    OCR_ENGINE_DOCTR: "Stylized",
}
FONT_STYLE_STANDARD = FONT_STYLE_NAMES[OCR_ENGINE_EASYOCR]
FONT_STYLE_STYLIZED = FONT_STYLE_NAMES[OCR_ENGINE_DOCTR]

# The docTR architectures this app pins by name, so a doctr upgrade can never silently
# swap models (and so the weight-cache check below knows which files to look for).
DOCTR_DET_ARCH = "db_resnet50"
DOCTR_RECO_ARCH = "parseq"
DOCTR_DET_INPUT_DEFAULT = 1024   # the pretrained detector's square input canvas, px
# The detector canvas for reading a custom overlay layout's drawn boxes under docTR. A drawn
# box is a small crop; at the pretrained 1024 canvas it is upscaled several times over and
# DBNet finds nothing in it (17 of 21 real tag boxes read empty on the reference screenshot),
# while costing as much as a whole band. At 256 the same boxes read correctly at ~1/13 the
# cost (measured 2026-09-03).
DOCTR_BOX_DET_INPUT = 256

# The pretrained PARSeq recognizer reads Latin script only, so the docTR engine is
# valid solely for the default (English) overlay language.
DOCTR_SUPPORTED_LANGS: tuple[str, ...] = ("en",)

# Approximate one-time download for the two pinned docTR models, quoted wherever the
# download is offered (GUI dialog, CLI notes).
DOCTR_WEIGHTS_MB_APPROX = 190

# Cap on the CRAFT text-DETECTION input: EasyOCR downscales a band until its longest side fits
# canvas_size before running the detector, then maps the found boxes back to source pixels and
# recognizes from the ORIGINAL-resolution crops. So lowering the cap makes the expensive
# detection pass cheaper (cost ~ pixels) without touching recognition quality; the trade is
# purely detection recall on small text. 2560 is EasyOCR's own default: a 1920-wide 1080p band
# passes through unscaled, i.e. today's behavior. (Lives here, in a torch-free module, so the
# editor's helper can import it without paying for cli.pipeline's module-level torch import.)
OCR_CANVAS_DEFAULT = 2560

# One overlay text read in EasyOCR's readtext shape: 4 corner points (clockwise from
# top-left, band-local pixels), the recognized text, and the confidence in [0, 1].
OcrResult = tuple[list[list[float]], str, float]

# An axis-aligned (x0, y0, x1, y1) pixel box, local to the image it was found in.
Rect = tuple[int, int, int, int]


def _rect_quad(rect: Rect) -> list[list[float]]:
    """A Rect as the 4-corner quad OcrResult carries (clockwise from top-left)."""
    x0, y0, x1, y1 = rect
    return [[float(x0), float(y0)], [float(x1), float(y0)], [float(x1), float(y1)], [float(x0), float(y1)]]


class OverlayOcr(Protocol):
    """What ``detect_text`` needs from an overlay OCR engine: read one band image."""

    def read_band(self, band: np.ndarray, *, beamsearch: bool,
                  canvas_size: int) -> list[OcrResult]:
        """Read all text in `band` (a BGR frame crop), returning OcrResult triples.

        ``beamsearch`` and ``canvas_size`` are EasyOCR decoding/detection knobs
        (see ``--ocr-beamsearch`` / ``--ocr-canvas``); engines without those
        concepts ignore them."""
        ...

    def recognize_boxes(self, image: np.ndarray, boxes: list[Rect], *,
                        beamsearch: bool) -> list[OcrResult]:
        """Recognizer-only read of `boxes` (image-local Rects, each already tight around
        one text element, as a previous detection pass found them) in `image` (a BGR
        band or crop), skipping text detection: the expensive half of a read (docTR's
        detector resizes every input to a fixed 1024x1024 canvas, so even a tiny crop
        costs as much as a whole band). Returns exactly one OcrResult per box, in the
        boxes' order, whose quad is the box itself; the text is "" when nothing was read.
        Backs the ROI cache's steady state (vodparse.ocr_roi)."""
        ...


class EasyOcrOverlayEngine:
    """The default engine: a thin adapter over an ``easyocr.Reader``'s readtext.

    The wrapped reader stays public (``.reader``) because one pipeline path remains
    EasyOCR-specific and needs the concrete object: reusing the overlay reader for the
    timer crop (the clock's allowlisted digit read)."""

    def __init__(self, reader: easyocr.Reader) -> None:
        self.reader = reader

    def read_band(self, band: np.ndarray, *, beamsearch: bool,
                  canvas_size: int) -> list[OcrResult]:
        decoder = "beamsearch" if beamsearch else "greedy"
        return self.reader.readtext(band, decoder=decoder, canvas_size=canvas_size)

    def recognize_boxes(self, image: np.ndarray, boxes: list[Rect], *,
                        beamsearch: bool) -> list[OcrResult]:
        import cv2

        if not boxes:
            return []
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # easyocr's horizontal_list entries are [x_min, x_max, y_min, y_max]. Its results
        # come back sorted by position, not in request order (utils.get_image_list sorts
        # them), so each is matched back to its box by the quad it reports.
        horizontal = [[x0, x1, y0, y1] for x0, y0, x1, y1 in boxes]
        found = self.reader.recognize(grey, horizontal_list=horizontal, free_list=[],
                                      decoder="beamsearch" if beamsearch else "greedy",
                                      paragraph=False)
        by_corner = {(int(quad[0][0]), int(quad[0][1])): (text, conf) for quad, text, conf in found}
        out: list[OcrResult] = []
        for rect in boxes:
            text, conf = by_corner.get((rect[0], rect[1]), ("", 0.0))
            out.append((_rect_quad(rect), text, float(conf)))
        return out


class DoctrOverlayEngine:
    """The docTR engine: DBNet text detection + PARSeq recognition over one band.

    ``beamsearch``/``canvas_size`` are EasyOCR concepts (its CTC decoder and CRAFT
    input cap) with no docTR counterpart, so they are accepted and ignored."""

    def __init__(self, predictor: Any) -> None:
        # An OCRPredictor from doctr.models.ocr_predictor. Typed Any so this module
        # imports (and unit-tests with fakes) without doctr installed.
        self.predictor = predictor

    def with_detector_canvas(self, size: int) -> "DoctrOverlayEngine":
        """A second engine whose text DETECTOR resizes its input to a `size` px square
        canvas instead of the model's own (1024 for the pretrained weights). The detection
        and recognition models are shared with this engine, so it costs no extra weights
        or memory: only the detection preprocessor differs. The detector's cost goes with
        its canvas, not its input, and its boxes come back as fractions of the input, so
        nothing else changes. Used for drawn overlay-layout boxes (DOCTR_BOX_DET_INPUT)."""
        from doctr.models.detection.predictor import DetectionPredictor
        from doctr.models.predictor import OCRPredictor
        from doctr.models.preprocessor import PreProcessor

        base = self.predictor
        det = base.det_predictor
        small_det = DetectionPredictor(
            PreProcessor((size, size), batch_size=det.pre_processor.batch_size,
                         mean=det.model.cfg["mean"], std=det.model.cfg["std"],
                         preserve_aspect_ratio=True, symmetric_pad=True),
            det.model)
        derived = OCRPredictor(small_det, base.reco_predictor,
                               assume_straight_pages=base.assume_straight_pages,
                               straighten_pages=base.straighten_pages,
                               preserve_aspect_ratio=True, symmetric_pad=True)
        return DoctrOverlayEngine(derived)

    def read_band(self, band: np.ndarray, *, beamsearch: bool,
                  canvas_size: int) -> list[OcrResult]:
        import numpy as np

        # OpenCV frames are BGR; docTR expects RGB. Copy into a contiguous array:
        # a negative-stride view from ::-1 alone can trip strict tensor conversion.
        rgb = np.ascontiguousarray(band[:, :, ::-1])
        result = self.predictor([rgb])
        height, width = band.shape[:2]
        return doctr_result_to_readtext(result, width, height)

    def recognize_boxes(self, image: np.ndarray, boxes: list[Rect], *,
                        beamsearch: bool) -> list[OcrResult]:
        import numpy as np

        # The OCRPredictor's own recognition stage (PARSeq) on each box's sub-crop; it
        # returns one (text, confidence) per crop in request order, splitting and
        # re-joining very wide crops itself. `beamsearch` is an EasyOCR knob: ignored.
        height, width = image.shape[:2]
        crops: list[np.ndarray] = []
        readable: list[int] = []   # indices of boxes with a non-empty sub-crop
        for index, (x0, y0, x1, y1) in enumerate(boxes):
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            crops.append(np.ascontiguousarray(image[y0:y1, x0:x1, ::-1]))   # BGR → RGB
            readable.append(index)
        texts: list[tuple[str, float]] = [("", 0.0)] * len(boxes)
        if crops:
            for index, (text, conf) in zip(readable, self.predictor.reco_predictor(crops)):
                texts[index] = (text, float(conf))
        return [(_rect_quad(rect), text, conf) for rect, (text, conf) in zip(boxes, texts)]


def doctr_result_to_readtext(result: Any, width: int, height: int) -> list[OcrResult]:
    """Flatten a docTR OCR result into EasyOCR-shaped OcrResult triples.

    docTR nests words under pages → blocks → lines and gives each word a *relative*
    axis-aligned geometry ``((x0, y0), (x1, y1))`` in [0, 1] (its default
    straight-pages mode). This walks that tree in reading order and scales each box
    to absolute pixels as the 4 corner points ``detect_text`` stores. Duck-typed on
    purpose: unit tests exercise the geometry math with plain fakes, no doctr
    install needed."""
    reads: list[OcrResult] = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    (rel_x0, rel_y0), (rel_x1, rel_y1) = word.geometry
                    x0, y0 = rel_x0 * width, rel_y0 * height
                    x1, y1 = rel_x1 * width, rel_y1 * height
                    bbox = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                    reads.append((bbox, word.value, float(word.confidence)))
    return reads


def doctr_available() -> bool:
    """Whether the python-doctr package is installed (metadata check only:
    nothing is imported, so this is safe to call from the GUI)."""
    return importlib.util.find_spec("doctr") is not None


def doctr_cache_dir() -> Path:
    """docTR's model-cache directory: ``$DOCTR_CACHE_DIR`` when set, else docTR's own
    ``~/.cache/doctr`` default. The env var is also the hook a future frozen build
    can use to point docTR at a per-user workspace directory."""
    override = os.environ.get("DOCTR_CACHE_DIR")
    return Path(override) if override else Path.home() / ".cache" / "doctr"


def doctr_weights_cached() -> bool:
    """Whether both pinned docTR models are already in the cache, judged by filename
    (docTR names each downloaded weight after its architecture). Stdlib-only, so the
    GUI can decide whether to offer the download without importing doctr/torch."""
    models_dir = doctr_cache_dir() / "models"
    if not models_dir.is_dir():
        return False
    filenames = [path.name for path in models_dir.iterdir()]
    return all(any(arch in name for name in filenames)
               for arch in (DOCTR_DET_ARCH, DOCTR_RECO_ARCH))


# Why Stylized (docTR) can't read yet, in words the user can act on. Nothing ever reads
# with Standard in docTR's place: a read or a run that needs docTR stops with one of these
# instead, so a result is never quietly the other engine's.
DOCTR_NOT_INSTALLED_MESSAGE = (
    f"{FONT_STYLE_STYLIZED} needs the python-doctr package, which isn't installed in this "
    "environment. "
    "Install it with: pip install python-doctr")
DOCTR_MODELS_MISSING_MESSAGE = (
    f"{FONT_STYLE_STYLIZED}'s text models aren't downloaded yet. The app will ask to download "
    "them if you click Analyze VOD.")


def engine_unusable_reason(engine_name: str) -> str | None:
    """Why `engine_name` can't read a frame right now (one of the DOCTR_*_MESSAGE texts),
    or None when it can. EasyOCR always can; docTR needs its package installed and its
    models downloaded, since a preview or test read never downloads them itself (only the
    Analyze page's offer may). Stdlib-only, like the two checks it combines."""
    if engine_name != OCR_ENGINE_DOCTR:
        return None
    if not doctr_available():
        return DOCTR_NOT_INSTALLED_MESSAGE
    if not doctr_weights_cached():
        return DOCTR_MODELS_MISSING_MESSAGE
    return None


def make_doctr_engine(device: str, det_input_size: int | None = None) -> DoctrOverlayEngine:
    """Construct the docTR overlay engine on `device`. Imports doctr lazily; exits
    with an actionable message when the optional install is missing (mirroring how
    other absent pipeline pieces fail loudly rather than deep in a traceback).

    `det_input_size` returns the engine with its text detector at that square canvas
    instead of the pretrained 1024 (see with_detector_canvas); None keeps the model's own.
    The pipeline builds the band engine plain and derives the drawn-box engine from it
    (make_layout_box_engine); the option exists for tools that want one directly."""
    try:
        from doctr.models import ocr_predictor
    except ImportError:
        raise SystemExit(
            "error: --ocr-engine doctr needs the python-doctr package, which is not "
            "installed in this environment.\n"
            "       Install it with: pip install python-doctr"
        ) from None

    download_note = ("" if doctr_weights_cached()
                     else f" (first use downloads the models, ~{DOCTR_WEIGHTS_MB_APPROX} MB, one time)")
    print(f"Loading docTR ({DOCTR_DET_ARCH} + {DOCTR_RECO_ARCH})...{download_note}")
    predictor = ocr_predictor(det_arch=DOCTR_DET_ARCH, reco_arch=DOCTR_RECO_ARCH,
                              pretrained=True).to(device)
    engine = DoctrOverlayEngine(predictor)
    return engine if det_input_size is None else engine.with_detector_canvas(det_input_size)


def make_layout_box_engine(engine_name: str, overlay: OverlayOcr) -> OverlayOcr:
    """The engine that reads a custom overlay layout's drawn boxes, derived from the
    overlay-band engine `overlay` (built by make_overlay_engine for `engine_name`).
    EasyOCR reads the boxes as it reads the bands (CRAFT's cost scales with the crop, so a
    small box is cheap). docTR gets a sibling engine with its detector at the small
    DOCTR_BOX_DET_INPUT canvas, sharing the models: at the pretrained canvas a drawn box
    both misreads and costs a full band (see the constant's comment)."""
    if engine_name == OCR_ENGINE_DOCTR:
        assert isinstance(overlay, DoctrOverlayEngine)
        return overlay.with_detector_canvas(DOCTR_BOX_DET_INPUT)
    return overlay


def validate_engine_lang(engine: str, lang: str) -> None:
    """Reject engine/language combinations that cannot work: today just docTR with a
    non-Latin overlay language (PARSeq's pretrained vocabulary is Latin-only). Called
    at pipeline start so both CLI and GUI-spawned runs fail up front with guidance."""
    if engine == OCR_ENGINE_DOCTR and lang not in DOCTR_SUPPORTED_LANGS:
        raise SystemExit(
            f"error: --ocr-engine doctr reads English/Latin text only and cannot be "
            f"combined with --ocr-lang {lang}.\n"
            f"       Use the default EasyOCR engine for non-English overlays."
        )


def make_overlay_engine(engine: str, lang: str,
                        device: str) -> tuple[OverlayOcr, "easyocr.Reader | None"]:
    """Build one overlay OCR engine, plus the concrete EasyOCR reader behind it when there
    is one (`_Detectors.overlay_reader` mirrors this, None under docTR, whose predictor is
    not an easyocr.Reader). Validates the engine/language pair first, so an unusable combo
    (docTR + a non-Latin language) fails loudly here rather than mid-OCR."""
    validate_engine_lang(engine, lang)
    if engine == OCR_ENGINE_DOCTR:
        return make_doctr_engine(device), None
    reader = make_reader(lang, gpu=True)
    return EasyOcrOverlayEngine(reader), reader
