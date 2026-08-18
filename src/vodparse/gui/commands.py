"""Subprocess argv builders for the GUI's Main page (vodparse.cli.pipeline, vodparse.cli.extract_matches).

MainWindow's ``_on_run``/``_start_extract`` (app.py) resolve widget/QSettings state (the
chosen OCR cadence, the OCR speed-up toggles, the Advanced Extract Options) into
plain arguments and hand them to the two builders below, which own the argv shape (flag
names, order, and when a flag is omitted). Kept Qt-free so the argv logic is unit-testable
without a QApplication.
"""

from pathlib import Path
from typing import Literal

from vodparse.gui.settings import OCR_ENGINE_DEFAULT_GUI, OCR_LANG_DEFAULT_GUI
from vodparse.tags import INGAME_TAG_BAND_FRAC_DEFAULT


def pipeline_argv(video: str, *, ocr_interval: int, gate_ocr: bool, roi_cache: bool,
                  ocr_lang: str, ocr_engine: str, ingame_tag_frac: float, preview_file: Path,
                  progress_file: Path) -> list[str]:
    """The vodparse.cli.pipeline argv for Analyze VOD.

    The GUI always runs the pipeline in seek mode (go/game mode remains CLI-only), so
    ``ocr_interval`` drives both the periodic OCR interval (--ocr-interval) and the seek
    step (--seek-every), keeping the two in sync (see OCR_INTERVAL_CHOICES in
    settings.py). ``--ocr-lang``, ``--ocr-engine`` and ``--ingame-tag-band-frac`` are only
    emitted when they differ from their defaults, so a default run matches
    vodparse.cli.pipeline's own defaults rather than repeating them on the command line.
    The band fraction goes only on the pipeline (it is baked into the pipeline JSON, which
    extract_matches reads back); the extract argv needs no matching flag.
    """
    argv = ["-m", "vodparse.cli.pipeline", "--video", video,
            "--ocr-interval", str(ocr_interval),
            "--seek-every", str(ocr_interval)]
    if gate_ocr:
        argv += ["--gate-ocr-on-timer"]
    if roi_cache:
        argv += ["--ocr-roi-cache"]
    if ocr_lang != OCR_LANG_DEFAULT_GUI:
        argv += ["--ocr-lang", ocr_lang]
    if ocr_engine != OCR_ENGINE_DEFAULT_GUI:
        argv += ["--ocr-engine", ocr_engine]
    if ingame_tag_frac != INGAME_TAG_BAND_FRAC_DEFAULT:
        argv += ["--ingame-tag-band-frac", str(ingame_tag_frac)]
    argv += ["--preview-file", str(preview_file)]
    argv += ["--progress-file", str(progress_file)]
    return argv


def extract_argv(video: str, pipeline_json: Path, *, mode: Literal["match", "tournament_set"],
                 group_by: str, set_characters: str, use_sponsors: bool, tag_flags: list[str],
                 debug: bool, out_root: Path) -> list[str]:
    """The vodparse.cli.extract_matches argv shared by Analyze VOD (after the pipeline), Use existing
    analysis, and Re-read tags.

    ``--csv-only``: write the matches JSON but don't cut clips: clips are exported from the
    Review screen after corrections. ``out_root`` is the resolved export root (see
    vodparse.gui.export_root); inert while --csv-only is always passed, but kept so the two
    export-capable stages get a uniform argv. ``tag_flags`` is the already-resolved
    tag-reading flag list (either the Advanced Extract Options via tag_reading_flags, or
    Re-read tags' own per-VOD choice): both callers reduce to "resolved tag flags, then
    optional --debug", so this builder takes them separately and appends --debug itself.
    ``tag_flags`` is read, never mutated.
    """
    argv = [
        "-m", "vodparse.cli.extract_matches",
        "--input", str(pipeline_json),
        "--video", video,
        "--mode", mode,
        "--csv-only",
        "--out-root", str(out_root),
    ]
    if mode == "tournament_set":
        argv += ["--group-by", group_by, "--set-characters", set_characters]
    if not use_sponsors:
        argv.append("--no-sponsors")
    argv += tag_flags
    if debug:
        argv.append("--debug")
    return argv
