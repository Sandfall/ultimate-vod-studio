"""Selectable OCR engine for the overlay text bands (player names, sponsors, tags).

Two engines can read the top/bottom overlay bands that ``vodparse.cli.pipeline``'s
``detect_text`` crops out of each sampled frame:

* **EasyOCR** (:data:`OCR_ENGINE_EASYOCR`, the default): the reader built by
  :func:`vodparse.ocr_langs.make_reader`, wrapped in :class:`EasyOcrOverlayEngine`.
* **docTR** (:data:`OCR_ENGINE_DOCTR`): DBNet detection + the PARSeq recognizer,
  which handles stylized overlay fonts better. Ships with the app (a core
  dependency; the frozen build also bundles its models: see ``freeze_entry.py``).
  English/Latin text only (see :data:`DOCTR_SUPPORTED_LANGS`).

The engine choice covers **only** the overlay bands. The match-clock crop
(``vodparse.timer``) is always read with EasyOCR (its recognizer-only
``recognize(allowlist=...)`` digit read has no docTR equivalent), and the OCR ROI
cache (``vodparse.ocr_roi``) is likewise EasyOCR-specific, so the pipeline disables
it under docTR.

Both engines return EasyOCR-shaped ``(bbox_points, text, confidence)`` triples
(:data:`OcrResult`), so ``detect_text``'s filtering and band-offset math is
engine-agnostic. Qt-free; easyocr/doctr are imported lazily so the GUI can import
this module for :func:`doctr_available` / :func:`doctr_weights_cached` without
pulling in torch.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

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

# The docTR architectures this app pins by name, so a doctr upgrade can never silently
# swap models (and so the weight-cache check below knows which files to look for).
DOCTR_DET_ARCH = "db_resnet50"
DOCTR_RECO_ARCH = "parseq"

# The pretrained PARSeq recognizer reads Latin script only, so the docTR engine is
# valid solely for the default (English) overlay language.
DOCTR_SUPPORTED_LANGS: tuple[str, ...] = ("en",)

# Approximate one-time download for the two pinned docTR models, quoted wherever the
# download is offered (GUI dialog, CLI notes).
DOCTR_WEIGHTS_MB_APPROX = 190

# One overlay text read in EasyOCR's readtext shape: 4 corner points (clockwise from
# top-left, band-local pixels), the recognized text, and the confidence in [0, 1].
OcrResult = tuple[list[list[float]], str, float]


class OverlayOcr(Protocol):
    """What ``detect_text`` needs from an overlay OCR engine: read one band image."""

    def read_band(self, band: np.ndarray, *, beamsearch: bool,
                  canvas_size: int) -> list[OcrResult]:
        """Read all text in `band` (a BGR frame crop), returning OcrResult triples.

        ``beamsearch`` and ``canvas_size`` are EasyOCR decoding/detection knobs
        (see ``--ocr-beamsearch`` / ``--ocr-canvas``); engines without those
        concepts ignore them."""
        ...


class EasyOcrOverlayEngine:
    """The default engine: a thin adapter over an ``easyocr.Reader``'s readtext.

    The wrapped reader stays public (``.reader``) because two pipeline paths remain
    EasyOCR-specific and need the concrete object: the OCR ROI cache's
    recognizer-only reads, and reusing the overlay reader for the timer crop."""

    def __init__(self, reader: easyocr.Reader) -> None:
        self.reader = reader

    def read_band(self, band: np.ndarray, *, beamsearch: bool,
                  canvas_size: int) -> list[OcrResult]:
        decoder = "beamsearch" if beamsearch else "greedy"
        return self.reader.readtext(band, decoder=decoder, canvas_size=canvas_size)


class DoctrOverlayEngine:
    """The docTR engine: DBNet text detection + PARSeq recognition over one band.

    ``beamsearch``/``canvas_size`` are EasyOCR concepts (its CTC decoder and CRAFT
    input cap) with no docTR counterpart, so they are accepted and ignored."""

    def __init__(self, predictor: Any) -> None:
        # An OCRPredictor from doctr.models.ocr_predictor. Typed Any so this module
        # imports (and unit-tests with fakes) without doctr installed.
        self.predictor = predictor

    def read_band(self, band: np.ndarray, *, beamsearch: bool,
                  canvas_size: int) -> list[OcrResult]:
        import numpy as np

        # OpenCV frames are BGR; docTR expects RGB. Copy into a contiguous array:
        # a negative-stride view from ::-1 alone can trip strict tensor conversion.
        rgb = np.ascontiguousarray(band[:, :, ::-1])
        result = self.predictor([rgb])
        height, width = band.shape[:2]
        return doctr_result_to_readtext(result, width, height)


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


def make_doctr_engine(device: str) -> DoctrOverlayEngine:
    """Construct the docTR overlay engine on `device`. Imports doctr lazily; exits
    with an actionable message when the optional install is missing (mirroring how
    other absent pipeline pieces fail loudly rather than deep in a traceback)."""
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
    return DoctrOverlayEngine(predictor)


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
