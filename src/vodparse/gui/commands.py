"""Subprocess argv builders for the GUI's Main page (vodparse.cli.pipeline,
vodparse.cli.extract_matches, vodparse.cli.clock_locate, vodparse.cli.side_overlay_locate).

MainWindow's ``_on_run``/``_start_extract`` (app.py) resolve widget/QSettings state (the
chosen OCR cadence, the OCR speed-up toggles, the Advanced Analysis Options) into
plain arguments and hand them to the builders below, which own the argv shape (flag
names, order, and when a flag is omitted); the Preview pane's clock search
(vodparse.gui.clock_locate) and player-name search (vodparse.gui.side_overlay_locate) get
their argv the same way. Kept Qt-free so the argv logic is
unit-testable without a QApplication.
"""

from pathlib import Path
from typing import Literal

from vodparse.games import GAME_DEFAULT
from vodparse.gui.settings import (OCR_ENGINE_DEFAULT_GUI, OCR_LANG_DEFAULT_GUI,
                                   OVERLAY_LAYOUT_DEFAULT)
from vodparse.schema import INGAME_BAND_ENGINE_OVERLAY, BBox
from vodparse.tags import INGAME_TAG_BAND_FRAC_DEFAULT


def pipeline_argv(video: str, *, ocr_interval: int, gate_ocr: bool, roi_cache: bool,
                  ocr_lang: str, ocr_engine: str, ingame_tag_frac: float,
                  game: str = GAME_DEFAULT,
                  overlay_layout: str = OVERLAY_LAYOUT_DEFAULT,
                  band_on_overlay_engine: bool = False, preview_file: Path,
                  progress_file: Path, prepass_handover: Path | None = None) -> list[str]:
    """The vodparse.cli.pipeline argv for Analyze VOD.

    The GUI always runs the pipeline in seek mode (go/game mode remains CLI-only), so
    ``ocr_interval`` drives both the periodic OCR interval (--ocr-interval) and the seek
    step (--seek-every), keeping the two in sync (see OCR_INTERVAL_CHOICES in
    settings.py). ``--ocr-lang``, ``--ocr-engine``, ``--ingame-tag-band-frac``, ``--game``
    and ``--overlay-layout`` are only emitted when they differ from their defaults, so a
    default run matches vodparse.cli.pipeline's own defaults rather than repeating them
    on the command line. The band fraction, the game and the overlay layout all go only on
    the pipeline (each is baked into the pipeline JSON, which extract_matches reads back);
    the extract argv needs no matching flag for any of them. ``band_on_overlay_engine``
    adds ``--ingame-band-engine overlay``: a game that pins its own nameplate-band engine
    (Rivals 2: docTR) reads the band with ``ocr_engine`` instead, for when the user declined
    that engine's model download. ``prepass_handover`` names the file holding what the
    layout preview's searches already found for this video (vodparse.prepass_handover);
    None when there is nothing to hand over, and the run searches for itself.
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
    if game != GAME_DEFAULT:
        argv += ["--game", game]
    if overlay_layout != OVERLAY_LAYOUT_DEFAULT:
        argv += ["--overlay-layout", overlay_layout]
    if band_on_overlay_engine:
        argv += ["--ingame-band-engine", INGAME_BAND_ENGINE_OVERLAY]
    argv += ["--preview-file", str(preview_file)]
    argv += ["--progress-file", str(progress_file)]
    if prepass_handover is not None:
        argv += ["--prepass-handover", str(prepass_handover)]
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
    tag-reading flag list (either the Advanced Analysis Options via tag_reading_flags, or
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


def clock_locate_argv(video: str, game: str, *, samples: int | None = None) -> list[str]:
    """The vodparse.cli.clock_locate argv for the Preview pane's clock search.

    ``--game`` is always emitted: the stage is per game by nature (which clock detector
    to load), and reading the argv back should say which one it searched with, unlike
    the pipeline's omit-at-default rule for a flag that merely restates a default.
    ``samples`` overrides the pre-pass's own sample count; None leaves the stage on the
    pipeline's tunable, so the preview finds exactly what an Auto run will lock."""
    argv = ["-m", "vodparse.cli.clock_locate", video, "--game", game]
    if samples is not None:
        argv += ["--samples", str(samples)]
    return argv


def side_overlay_locate_argv(video: str, game: str, *, ocr_engine: str, ocr_lang: str,
                             clock_bbox: BBox | None) -> list[str]:
    """The vodparse.cli.side_overlay_locate argv for the Preview pane's player-name search.

    ``--game``, ``--ocr-engine`` and ``--ocr-lang`` are always emitted, as clock_locate_argv
    always emits ``--game``: the answer depends on all three, and reading the argv back
    should say what it searched with. ``clock_bbox`` is the calibrated clock box the bars
    are placed from; None leaves the flag off, and the stage assumes a centred picture."""
    argv = ["-m", "vodparse.cli.side_overlay_locate", video, "--game", game,
            "--ocr-engine", ocr_engine, "--ocr-lang", ocr_lang]
    if clock_bbox is not None:
        argv += ["--clock-bbox", *(str(value) for value in clock_bbox)]
    return argv
