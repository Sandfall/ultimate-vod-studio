"""OCR language configuration: which languages EasyOCR reads, and where each language's
recognition model is stored.

Two distinct concepts:

* **Language choice**: :data:`OCR_LANG_LABELS` / :data:`OCR_LANG_DEFAULT` /
  :func:`reader_langs`. EasyOCR pairs exactly one non-Latin script with English per
  ``Reader`` instance, so the overlay is always read as English plus (optionally) one
  other language. Adding a language later is a two-line edit: one entry in
  ``OCR_LANG_LABELS`` here, one in ``vodparse.gui.settings.OCR_LANG_CHOICES``.
* **Model storage**: :func:`easyocr_model_dir`. In a source checkout, EasyOCR's own
  default (``~/.EasyOCR``) is used for every language. In a frozen (PyInstaller) build,
  English keeps using the bundled offline models via the ``EASYOCR_MODULE_PATH``
  environment variable that ``freeze_entry.py`` already sets, so this module stays out of
  that mechanism's way. A frozen build asking for a non-English language has no bundled
  weights for it, so its recognizer model is downloaded on first use into a writable
  per-user directory instead. Frozen builds also redirect EasyOCR's (never-used)
  custom-network directory to that writable per-user area
  (:func:`easyocr_user_network_dir`), because the bundle itself may be read-only.

:func:`make_reader` ties both together as the single place that constructs an
``easyocr.Reader``.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from vodparse.runtime_paths import bundle_dir, is_frozen, workspace_dir

if TYPE_CHECKING:
    import easyocr

# Display label for each supported language, shown in the GUI's language picker.
OCR_LANG_LABELS: dict[str, str] = {
    "en": "English",
    "ja": "Japanese + English",
}

OCR_LANG_DEFAULT = "en"

# The text-detector weight shared by every language's Reader. Seeding it into a
# newly-created model directory (see easyocr_model_dir) avoids re-downloading it just
# because a non-English language's recognizer file hasn't been fetched yet.
_CRAFT_DETECTOR_FILENAME = "craft_mlt_25k.pth"


def reader_langs(lang: str) -> list[str]:
    """The EasyOCR language list for `lang` ("en" or a non-English code like "ja").
    EasyOCR wants the non-English language listed first, with English always included."""
    return ["en"] if lang == "en" else [lang, "en"]


def easyocr_model_dir(lang: str) -> str | None:
    """Where EasyOCR should read/write its recognition models for `lang`, or None to use
    EasyOCR's own default.

    A source checkout always returns None (the user's ``~/.EasyOCR`` cache). A frozen
    build reading English also returns None, so it keeps using the bundled offline
    models via the ``EASYOCR_MODULE_PATH`` environment variable set in
    ``freeze_entry.py``: this function must not interfere with that.

    A frozen build reading a non-English language has no bundled weights for it, so it
    gets its own writable directory under the workspace root. The shared CRAFT
    detector weight is seeded into that directory from the bundle when available, so
    only the language-specific recognizer file needs a first-use download.
    """
    if not is_frozen() or lang == "en":
        return None

    model_dir = workspace_dir() / "easyocr_models" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    bundled_craft = bundle_dir() / "easyocr_models" / "model" / _CRAFT_DETECTOR_FILENAME
    seeded_craft = model_dir / _CRAFT_DETECTOR_FILENAME
    if bundled_craft.exists() and not seeded_craft.exists():
        shutil.copy2(bundled_craft, seeded_craft)

    return str(model_dir)


def easyocr_user_network_dir() -> str | None:
    """Where EasyOCR should look for user-defined custom recognition networks, or None
    to use EasyOCR's own default.

    The app never installs a custom network, so this directory stays empty forever —
    but ``easyocr.Reader.__init__`` unconditionally ``mkdir``s it. Left at its default
    in a frozen build it resolves inside the bundle (via ``EASYOCR_MODULE_PATH``), and
    the bundle is not always writable: macOS runs a quarantined download from a
    read-only App Translocation mount, and a Windows install dir can be read-only too.
    That mkdir then crashes every Reader construction. So a frozen build redirects it
    to the per-user workspace, keeping EasyOCR's writes out of the bundle entirely
    (EasyOCR itself creates the directory). A source checkout returns None: the
    ``~/.EasyOCR`` default is already writable.
    """
    if not is_frozen():
        return None
    return str(workspace_dir() / "easyocr_models" / "user_network")


def make_reader(lang: str, gpu: bool = True) -> easyocr.Reader:
    """Construct the EasyOCR reader for `lang`. Imports easyocr lazily so that merely
    importing this module (e.g. from the GUI or a lightweight test) never pulls in
    torch/easyocr."""
    import easyocr

    download_note = "" if lang == "en" else " (first use downloads the recognition model, ~60 MB, one time)"
    print(f"Loading easyocr ({OCR_LANG_LABELS[lang]})...{download_note}")
    return easyocr.Reader(
        reader_langs(lang),
        gpu=gpu,
        model_storage_directory=easyocr_model_dir(lang),
        user_network_directory=easyocr_user_network_dir(),
    )
