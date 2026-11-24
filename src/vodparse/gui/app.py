"""Desktop GUI for running the VOD analysis pipeline.

Two screens in a QStackedWidget share one subprocess log:
  Screen 1 (Main) picks a vod and runs the analysis → extract chain:
    ``vodparse.cli.pipeline`` (logo/timer detection → ``json_output/<stem>/pipeline.json``) followed by
    ``vodparse.cli.extract_matches --csv-only`` (group into matches/sets → ``json_output/<stem>/matches.json``,
    no clips yet). "Analyze VOD" runs both back to back; the second button reopens a saved Review
    draft when one exists ("Resume edits"), otherwise reuses an existing pipeline.json and only
    extracts ("Open existing analysis"). Re-extracting while a draft exists happens from Review
    (Re-read tags, or Reset all edits → Original analysis). Both buttons land on:
  Screen 2 (Review) lets the user correct tags/characters and trim each clip's start/end,
    then exports the clips via ``vodparse.cli.export_reviewed``.
The subprocess stages run via QProcess so the UI stays responsive; their stdout collects into an
off-screen buffer, viewable any time via the "Show log…" popup.

Usage (from the repo root):

    python -m vodparse.gui.app

    # testing / GUI automation: a known starting state, settings kept apart from the user's
    python -m vodparse.gui.app --settings-profile test --game rivals2 --video "other/vods/<vod>.mp4"

    # straight to Review on a vod whose analysis exists (--debug-resume: its saved edits)
    python -m vodparse.gui.app --debug "other/vods/<vod>.mp4"
"""

import argparse
import json
import shutil
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

from vodparse.gui.qt_plugins import configure_qt_plugins

configure_qt_plugins()  # must run before any PyQt6 widget module is imported

from PyQt6.QtCore import (  # noqa: E402  (must follow plugin setup)
    QEvent, Qt, QPoint, QProcess, QSettings, QTimer, QUrl,
)
from PyQt6.QtGui import (QAction, QCloseEvent, QDesktopServices, QFont, QIcon, QKeySequence,  # noqa: E402
                         QScreen)
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from vodparse.gui import crash_log, windows_theme  # noqa: E402
from vodparse.gui.commands import extract_argv, pipeline_argv  # noqa: E402
from vodparse.gui.export_root import export_root, export_root_reachable  # noqa: E402
from vodparse.games import (GAME_DEFAULT, GAMES, Game, default_game, ingame_band_engine_for,  # noqa: E402  (Qt-free, torch-free)
                            game_by_id, resolve_saved_game)
from vodparse.overlay_layout import (AUTO_LAYOUT_NAME, CustomOverlayLayout,  # noqa: E402  (Qt-free, cv2/torch-free)
                                     ingame_band_mode_for, layout_by_name, layouts_for_game,
                                     layouts_path, load_layouts)
from vodparse.tags import INGAME_TAG_BAND_FRAC_DEFAULT  # noqa: E402  (cv2/torch-free)
from vodparse.gui.common import (VIDEO_FILTER, BusyDialog, ElidingLabel,  # noqa: E402
                        apply_font_scale, apply_system_theme, muted_text_color,
                        notice_color, stage_environment,
                        warning_color)
from vodparse.gui.form_columns import FormColumns  # noqa: E402
from vodparse.gui.dialogs import (AboutDialog, ExtractOptionsDialog, LogDialog,  # noqa: E402
                                  RereadTagsDialog, SettingsDialog, TournamentNameDialog)
from vodparse.gui.paths import (PROJECT_ROOT, analysis_settings_differences,  # noqa: E402
                       app_icon_path, default_tournament_name, existing_analysis,
                       existing_analysis_summary, last_export_info,
                       other_vods_to_confirm, outdated_clip_files, output_dir_for,
                       vod_clip_filenames,
                       pipeline_json_for, resolve_export_csv, review_draft_for,
                       same_physical_dir, stale_export_dir, video_metadata_mismatches,
                       vod_picker_start_dir)
from vodparse.gui.settings import (EXTRACT_OPT_DEFAULTS, FONT_SCALE_DEFAULT,  # noqa: E402
                          OCR_INTERVAL_CHOICES, SMART_TAGS_FLAG, review_mode_note,
                          seed_tag_options, tag_reading_flags,
                          OCR_ENGINE_CHOICES, OCR_ENGINE_DEFAULT_GUI, resolve_saved_engine,
                          OCR_INTERVAL_DEFAULT_SEC, OCR_LANG_DEFAULT_GUI, lang_display_name,
                          EXPORT_MODE_DEFAULT, SET_CHARACTERS_DEFAULT,
                          SETTING_CONFIRM_REUSE_SETTINGS,
                          SETTING_FONT_SCALE, SETTING_GPU_SKIP_FIRST_RUN_OFFER,
                          SETTING_DEFAULT_VOD_DIR, SETTING_GAME, SETTING_GATE_OCR_ON_TIMER,
                          SETTING_GEOMETRY_MAIN, SETTING_GEOMETRY_REVIEW,
                          SETTING_INGAME_TAG_BAND_FRAC, SETTING_OCR_ENGINE,
                          SETTING_OCR_INTERVAL_SEC, SETTING_OCR_LANG, SETTING_OCR_ROI_CACHE,
                          SETTING_SHOW_LAYOUT_PREVIEW, SHOW_LAYOUT_PREVIEW_DEFAULT,
                          SETTING_ASK_CONVERT_MKV, ASK_CONVERT_MKV_DEFAULT,
                          OVERLAY_LAYOUT_DEFAULT, overlay_layout_setting_key,
                          GATE_OCR_ON_TIMER_DEFAULT, OCR_ROI_CACHE_DEFAULT,
                          SETTING_SHOW_DEBUG_MENU, SETTING_SHOW_FONT_STYLE_NOTE,
                          SETTING_SHOW_SMART_TAGS_POPUP,
                          SETTING_TITLE_CHAR_LIMIT,
                          SETTING_TITLE_FILENAMES_MATCHES, SETTING_TITLE_FILENAMES_SETS,
                          SITUATIONAL_EXTRACT_OPTS,
                          TITLE_CHAR_LIMIT_DEFAULT)
from vodparse.gui.convert import Mp4Converter, is_mp4_conversion_of, opencv_can_read  # noqa: E402
from vodparse import gpu_support  # noqa: E402  (stdlib-only; no torch/Qt at import)
from vodparse.gui.gpu_support import (  # noqa: E402
    GpuSupportDialog, GpuSupportManager, connect_site_swap)
from vodparse.gui.updates import UpdateChecker  # noqa: E402
from vodparse.ocr_engine import (DOCTR_NOT_INSTALLED_MESSAGE,  # noqa: E402  (stdlib-only; no torch/Qt at import)
                                 DOCTR_WEIGHTS_MB_APPROX, FONT_STYLE_STANDARD,
                                 FONT_STYLE_STYLIZED, OCR_ENGINE_DOCTR, doctr_available,
                                 doctr_weights_cached)
from vodparse.runtime_paths import is_frozen  # noqa: E402
from vodparse.gui.clock_locate import ClockLocator  # noqa: E402
from vodparse.gui.side_overlay_locate import SideOverlayLocator  # noqa: E402
from vodparse.cli.side_overlay_locate import SideOverlayLocateResult  # noqa: E402
from vodparse.prepass_handover import PrepassHandover, write_handover  # noqa: E402
from vodparse.gui.preview import PreviewProgressPane  # noqa: E402
from vodparse.gui.widgets import (FittingStack, PopupPickComboBox,  # noqa: E402
                                  combo_width_for_chars, make_icon_button,
                                  widen_popup_to_fit_items)
from vodparse.extract_flags import EXTRACT_FLAGS_NAME, read_sidecar  # noqa: E402
from vodparse.smart_tags import SMART_TAGS_REPORT_NAME  # noqa: E402
from vodparse.gui.review_tools import (  # noqa: E402  (Tools-menu pieces shared with the vod grabber)
    BUSY_EXPORT_NOTE, RENAME_EXPORT_NOTE, confirm_reread_discard,
    rename_leaves_exported_clips_behind, show_smart_tags_summary,
)
from vodparse.gui.review.screen import ReviewScreen  # noqa: E402  (QtMultimedia; after plugin setup)
from vodparse.gui.upload_assistant import (  # noqa: E402  (PyQt6 widgets; after plugin setup)
    UploadAssistant, forget_saved_geometry as forget_upload_assistant_geometry)
from vodparse.diagnostics import available_stems, default_zip_name  # noqa: E402
from vodparse.export import (CSV_READ_ERRORS, build_chapters, clip_unit,  # noqa: E402
                             count_saved_lines, export_basename, planned_clip_filename)
from vodparse.ffmpeg_runtime import offers_conversion  # noqa: E402
from vodparse.jsonio import write_json_atomic  # noqa: E402
from vodparse.paths import vod_stem  # noqa: E402
from vodparse.schema import BBox, ReviewedData  # noqa: E402
from vodparse.cli.clock_locate import ClockLocateResult  # noqa: E402  (import-light: no torch)


def _dir_size_gb(path: Path) -> float:
    """Total size of the files under `path` in GB (decimal, matching how drives are
    labelled: the MP4 converter's size warning uses the same convention). 0 if unreadable."""
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
    except OSError:
        return 0.0


def _outdated_clips_tooltip(clips: list[Path]) -> str:
    """The names a tick of the "remove outdated clips" checkbox deletes, one per line,
    capped at OUTDATED_CLIP_TOOLTIP_NAMES with a count of the rest."""
    names = [clip.name for clip in clips[:OUTDATED_CLIP_TOOLTIP_NAMES]]
    rest = len(clips) - len(names)
    if rest:
        names.append(f"… and {rest} more")
    return "\n".join(names)


def _files_size_gb(paths: list[Path]) -> float:
    """Total size of the files `paths` in GB (the same convention as _dir_size_gb); a file
    that can't be stat'd counts as 0."""
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total / 1e9


# The signature QProcess.finished delivers: (exit_code, exit_status).
FinishHandler = Callable[[int, QProcess.ExitStatus], None]

# How long _stop_process waits for the child to exit: after the polite terminate(),
# then after the follow-up kill(). Windows skips the polite step (see _stop_process).
STOP_TERM_WAIT_MS = 3000
STOP_KILL_WAIT_MS = 1000

# How often the no-op timer wakes the interpreter so Ctrl+C can be handled (see main()).
SIGINT_PUMP_INTERVAL_MS = 200

# Default window size per stacked screen (keyed by stack index), used on first launch and
# by "Reset window layouts" — a screen's own saved geometry always wins once the user has
# resized it. The Review default deliberately
# over-asks: it exceeds a 13-14" laptop display, and _fit_to_screen trims it to the
# available screen area, so Review opens as large as the display allows, up to this cap.
DEFAULT_WINDOW_SIZE_MAIN = (1150, 820)     # the analysis page (stack index 0)
DEFAULT_WINDOW_SIZE_REVIEW = (1400, 920)   # the Review page (stack index 1)

# The narrowest the Analyze page's Preview box may get: half the page's default window
# width. The option boxes beside it are as wide as their contents ask, so without a floor
# a wide one (a long layout name, a large Text size) takes its extra width out of the
# preview. With it, the window grows instead once the preview is down to this.
PREVIEW_MIN_WIDTH_PX = DEFAULT_WINDOW_SIZE_MAIN[0] // 2

# The longest name, in characters, the Overlay layout combo grows to show in full. A
# longer one is cut short with "…" in the closed combo (its list still shows it in full).
# Room for a typical event-and-stage name ("Genesis 9 main stage"), while one very long
# name cannot widen the whole option column and take the width out of the preview.
OVERLAY_LAYOUT_COMBO_MAX_CHARS = 24

# The extract's --group-by value (which tags identify a set). Always "auto" from the GUI:
# the "Group sets by:" picker was dropped because ingame-tag grouping only works reliably
# as a *fallback*. The --group-by plumbing (commands.extract_argv, the Review mode note)
# stays live, so a picker can return without re-threading it.
GROUP_BY_DEFAULT = "auto"

# The muted note under the main screen's "Font style:" combo, one per style: what that
# style is good at, so the trade-off is visible without opening the info button beside it.
# Which one shows follows the combo; both are hidden together when the user turns the note
# off in Settings → Analysis. See _refresh_engine_note.
FONT_STYLE_NOTE_STANDARD = (f"{FONT_STYLE_STANDARD} reads most plain text overlays. If you want "
                            f"to read fancier fonts (or italics), try using {FONT_STYLE_STYLIZED}.")
FONT_STYLE_NOTE_STYLIZED = (f"{FONT_STYLE_STYLIZED} reads a wider range of fonts, but VOD analysis "
                            "is slower.")
# The button beside the Overlay layout combo that opens the layout editor (see
# _on_edit_layouts). The messages that send the user to it (the game note below, the two
# "Overlay layout …" warnings in _on_run) call it "Edit", so change them together.
EDIT_LAYOUTS_BUTTON_TEXT = "Edit…"

# Teal note under the Game combo while a game with no clock detector of its own is selected
# (only "Other" today, by design): its run needs a custom overlay layout to supply the
# clock, so say so before Analyze VOD is pressed. See
# vodparse.games.Game.has_timer_model.
GAME_NOTE_NO_TIMER_MODEL = ("{game} has no clock detector model, click \"Edit\" to make a "
                            "layout for it.")

# Shown under the Game combo (in the warning colour) when the preview's clock search for
# the selected game found no steady clock: its detector fired on none of the sampled
# frames, or fired only here and there (another game's clock can look enough like it,
# e.g. Melee's under Ultimate; a VOD with little game footage among the samples can do
# it too) without settling on one spot. The likeliest cause is a VOD of another game with
# last session's game still selected, and the place to fix it is the combo right above,
# which is all the note says (the user's own words, 2026-10-19: the search's reasons are
# on the Preview box's status line already). The same doubt is raised once more on
# Analyze (GAME_CONFIRM_NO_CLOCK), the moment the mistake would cost a long run.
GAME_NOTE_NO_CLOCK = "Check to make sure the correct game is selected."
GAME_CONFIRM_NO_CLOCK = ("Check to make sure the correct game is selected.\n\n"
                         "Analyze it as {game} anyway?")

# How many clip names the "remove outdated clips" checkbox lists in its tooltip before
# summarising the rest: a re-export that retitled a whole bracket would otherwise list
# more than fits on screen.
OUTDATED_CLIP_TOOLTIP_NAMES = 12

# The Overlay layout combo's last item, under a separator: not a choice but an action,
# opening the layout editor straight on a new layout (see _on_overlay_layout_activated).
# For a game with nothing else to list (no Auto, no preset yet) it is the only item, with
# no separator, and the one current. It is told apart by a flag in its own item-data role
# rather than by its data, since any string could also be a preset's name.
OVERLAY_LAYOUT_NEW_ITEM = "New layout…"
OVERLAY_LAYOUT_NEW_ROLE = Qt.ItemDataRole.UserRole + 1

# Teal reminder above the main page's run buttons while the "Export as" choice in the
# Advanced Analysis Options popup is Games rather than the default Sets: the popup is closed
# by then, and the choice decides what Review and Export produce. The same words as the
# popup's own note beside the radios (dialogs.EXPORT_MODE_GAMES_NOTE). See
# _refresh_export_mode_note.
EXPORT_MODE_GAMES_REMINDER = "Each game will be exported as its own clip."

# The file a run is handed what the Preview box's searches already found in
# (vodparse.prepass_handover), written beside the run's progress file.
PREPASS_HANDOVER_FILE_NAME = "vodparse_prepass_handover.json"


def _calibrated_bbox(clock: ClockLocateResult | None) -> BBox | None:
    """The calibrated clock box of a clock search's answer; None when the search had
    nothing to find, could not run, or found no steady spot."""
    if clock is None or clock.error is not None:
        return None
    return clock.bbox


def _gpu_offer_preview_applies() -> bool:
    """Whether the Debug menu carries "Run first-run GPU offer…". Not on macOS: the Apple
    GPU is used automatically there, so the offer can only end in its "No compatible NVIDIA
    GPU … on the CPU" notice, which contradicts Settings' "Using: Apple GPU (Metal/MPS)"."""
    return sys.platform != "darwin"


def _colour_note(label: QLabel, colour: str, *, bold: bool = False) -> None:
    """Give one of the main window's notes its theme colour (``colour`` is a
    muted_text_color / notice_color / warning_color value, resolved against the current
    palette). The one place the notes' style sheet is written: each note's creation site
    and MainWindow._restyle_themed_notes (a light/dark switch) both go through it, so a
    note's look can't drift between the two."""
    weight = " font-weight: bold;" if bold else ""
    label.setStyleSheet(f"color: {colour};{weight}")


class MainWindow(QWidget):
    """The app's main window: a two-page QStackedWidget (Main → Review) sharing one subprocess
    log. The Main page picks a VOD and runs the analysis→extract chain via QProcess; the Review
    page (ReviewScreen) corrects and trims the results before export. See the module docstring."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Ultimate VOD Studio")

        self._video_path: str | None = None
        self._proc: QProcess | None = None
        # Whether a subprocess run is in flight, per _set_running: which flips this BEFORE
        # the QProcess object exists, when _process_running() would still report idle.
        self._running = False
        self._proc_output = ""
        self._upload_assistant: UploadAssistant | None = None  # reused across exports; see _open_upload_assistant
        self._exporting = False  # gate the Review export-progress line counting in _read_output
        # The now-superseded previous export folder, captured at export kickoff (a tournament
        # rename moves the export elsewhere); offered for deletion in _on_export_finished.
        self._stale_export_dir: Path | None = None
        # This VOD's clip filenames in the target CSV before the export replaced its rows,
        # captured at the same kickoff: the files a re-export no longer lists are offered
        # for removal in _on_export_finished (see gui.paths.outdated_clip_files).
        self._previous_clip_names: frozenset[str] = frozenset()
        # The export root captured at the same kickoff, so the in-flight export's folder,
        # the stale-folder vetting above, and the post-export delete guard all agree even
        # if the paths/export_dir setting changes mid-export.
        self._export_root: Path | None = None
        self._stopping = False  # set while a user-requested Stop is in flight; the child exit is deliberate
        self._shutting_down = False  # the window is quitting: start no background work
        self._extract_was_smart = False  # whether the in-flight extract ran --smart-tags; set at kickoff

        # Persistent app settings (see SettingsDialog); QSettings keys off the org/app names set in
        # main(), so values survive restarts on every platform.
        self._settings = QSettings()

        # Advanced extract flags, edited via the Advanced Analysis Options popup (see ExtractOptionsDialog). Seeded
        # from their last-used values (persisted in QSettings), falling back to EXTRACT_OPT_DEFAULTS.
        self._extract_opts = self._load_extract_opts()

        # Embedded layout preview + progress bar: the chosen video scrubbed with the selected
        # overlay layout's read regions drawn on the frame, swapped for the pipeline's live
        # annotated frames (plus a progress bar/ETA) while a run is active. See preview.py
        # for the file-polling scheme and layout_preview.py for the scrubber.
        self._preview_pane = PreviewProgressPane(self._settings)
        # The preview's clock search (see clock_locate.py): one answer per video and game,
        # from a saved analysis or a torch sidecar process. Asked by _refresh_layout_preview
        # whenever the video, game or layout choice changes; its answers land in the pane.
        self._clock_locator = ClockLocator(self)
        # (video path, game id) pairs the user agreed to analyze although the clock search
        # found no steady clock (_confirm_no_clock), so the question is asked once per pair.
        self._no_clock_confirmed: set[tuple[str, str]] = set()
        self._clock_locator.searching.connect(self._preview_pane.set_clock_status)
        self._clock_locator.located.connect(self._on_clock_located)
        # The preview's player-name search (see side_overlay_locate.py), for a game whose
        # names sit in bars beside the picture: asked once the clock has answered (the
        # bars are placed from its box), by _request_side_overlay.
        self._side_overlay_locator = SideOverlayLocator(self)
        self._side_overlay_locator.searching.connect(self._on_side_overlay_searching)
        self._side_overlay_locator.located.connect(self._preview_pane.set_side_overlay)

        # MKV/AVI/WebM → MP4 conversion offers (see convert.py): whatever file the user ends up
        # with: converted, reused, or the source as-is: is adopted like any picked file.
        self._converter = Mp4Converter(self)
        self._converter.fileReady.connect(self._finish_selection)

        # Optional CUDA-PyTorch GPU acceleration (Windows). The manager drives the download and
        # the compute-status readout; a one-time offer is made after the window is up (see
        # _maybe_offer_gpu_on_first_run), scheduled at the end of __init__.
        self._gpu = GpuSupportManager(self)
        self._gpu.declined.connect(self._suppress_gpu_first_run_offer)
        self._gpu.no_gpu_found.connect(self._suppress_gpu_first_run_offer)
        # The preview's OCR helper imports torch, and once GPU support is installed that is the
        # CUDA build, whose folder an install or uninstall (from any manager, Settings' included)
        # must rename or delete: the helper stops and holds its reads until that is over, as
        # Windows refuses while it holds the DLLs.
        connect_site_swap(self._preview_pane.release_for_site_swap,
                          self._preview_pane.read_after_site_swap)
        # Drives Help ▸ Check for Updates (compares __version__ against the latest GitHub
        # Release off the UI thread). Manual only: no automatic phone-home on launch.
        self._updates = UpdateChecker(self)

        self._review = ReviewScreen()
        self._review.backRequested.connect(self._goto_main)
        self._review.exportRequested.connect(self._on_review_export)
        self._review.exportOneRequested.connect(self._on_review_export_one)
        self._review.chaptersRequested.connect(self._on_review_chapters)
        self._review.uploadAssistantRequested.connect(self._on_open_upload_assistant)
        self._review.openExportFolderRequested.connect(self._on_open_export_folder)

        self._stack = FittingStack()
        self._stack.addWidget(self._build_main_page())   # index 0
        self._stack.addWidget(self._review)              # index 1

        # Heading shows the app name and which stage of the pipeline is on screen;
        # it tracks the stack so every navigation path keeps it in sync. Eliding
        # (natural-width mode) because the Review heading carries the tournament name:
        # a long one must squeeze down with the window, not pin its minimum width.
        self._title = ElidingLabel(natural_width=True)
        title_font = QFont()
        # 18pt is an absolute size (not inherited), so scale it by the Text-size preference by hand
        # to keep the heading in step with the rest of the UI (exactly 18pt at 100%).
        title_font.setPointSizeF(18.0 * self._font_scale())
        title_font.setBold(True)
        self._title.setFont(title_font)
        # Keep at least ~12 heading characters visible however narrow the window gets;
        # set after the font so the floor is measured at the 18pt size.
        self._title.setMinimumWidth(self._title.fontMetrics().averageCharWidth() * 12)
        # The muted parenthetical after the heading (the Review mode note; see _update_title).
        # Its own label, not rich text inside the heading, so the heading can elide.
        self._title_note = QLabel()
        note_font = QFont(title_font)
        note_font.setPointSizeF(round(title_font.pointSizeF() * 0.6))
        note_font.setBold(False)
        self._title_note.setFont(note_font)
        self._title_note.hide()
        self._stack.currentChanged.connect(self._update_title)

        # The subprocess transcript. It's never shown inline: _append accumulates it in this
        # hidden buffer (parented + hidden so it never renders), and the "Show log…" popup
        # (LogDialog) reads/streams from it: _append keeps an open popup live and the finish
        # handlers auto-open it on failure. It's opened from the Debug menu's "Show log…" action
        # (see _build_menu_bar), so it's reachable from any page.
        self._log = QPlainTextEdit(self)
        self._log.setReadOnly(True)
        self._log.hide()
        self._log_dialog: LogDialog | None = None

        layout = QVBoxLayout(self)
        layout.setMenuBar(self._build_menu_bar())  # global menu bar on macOS, in-window elsewhere
        layout.addLayout(self._build_title_row())  # title + the Preferences/Tools buttons (uses the menu's actions)
        layout.addWidget(self._stack, stretch=1)

        self.resize(*DEFAULT_WINDOW_SIZE_MAIN)   # the window starts on the analysis page
        # The window minimum tracks the *current* page's content minimum (see _apply_min_size),
        # so a page can't be dragged small enough to overlap its own controls; but it's capped to
        # the screen so a tall page still fits a small display. Applied after the first layout
        # settles (see main) and on every page switch.
        # A page switch (_show_page) swaps in that page's own remembered window geometry;
        # what follows the visible page itself is updated on currentChanged.
        self._stack.currentChanged.connect(self._on_screen_changed)
        self._update_title(self._stack.currentIndex())
        # currentChanged doesn't fire at construction, so seed the Tools surfaces' visibility for
        # the starting page (Analyze) explicitly: both surfaces exist now (_build_title_row ran).
        self._apply_tools_visibility(self._stack.currentIndex())
        self._refresh_nav()
        self._refresh_export_mode_note()  # surface a Games export mode persisted from a past session
        self._refresh_opts_warning()   # surface a situational toggle persisted from a past session
        # Offer the GPU download once, after the window is up (0-delay = "once the event loop runs").
        QTimer.singleShot(0, self._maybe_offer_gpu_on_first_run)

    def _geometry_key(self, index: int) -> str:
        """The QSettings key holding a stacked screen's saved window geometry, so the analysis
        page and the Review page each remember their own size independently."""
        return SETTING_GEOMETRY_REVIEW if index == 1 else SETTING_GEOMETRY_MAIN

    def _default_size(self, index: int) -> tuple[int, int]:
        """A stacked screen's default window size, used whenever that screen has no saved
        geometry to restore (first launch, restore off, after a layout reset)."""
        return DEFAULT_WINDOW_SIZE_REVIEW if index == 1 else DEFAULT_WINDOW_SIZE_MAIN

    def _show_page(self, index: int) -> None:
        """Switch the stack to page ``index`` (0 Analyze, 1 Review), swapping the window's
        geometry with it: stash the size of the page being left, switch, then restore the
        entering page's own size (_restore_page_geometry). Every page switch goes through
        here.

        The stash has to come before the switch. When the entering page needs more height
        than the window has, the stack grows the window inside setCurrentIndex, before
        currentChanged reaches any slot; stashed after that, the Analyze page's size would be
        Review's minimum height, and Back would bring the Analyze page back that tall (VM
        screenshot round, 2026-10-27)."""
        leaving = self._stack.currentIndex()
        if index == leaving:
            return
        self._settings.setValue(self._geometry_key(leaving), self.saveGeometry())
        self._stack.setCurrentIndex(index)
        self._restore_page_geometry(index)

    def _on_screen_changed(self, new_index: int) -> None:
        """What follows the visible page: the Review-only Tools surfaces, and on entering
        Review the export-output buttons. The window geometry is _show_page's."""
        self._apply_tools_visibility(new_index)
        if new_index == 1:  # entering Review: light up the export-output buttons by what's on disk
            self._refresh_export_outputs()

    def _restore_page_geometry(self, new_index: int) -> None:
        """Restore the window geometry page ``new_index`` last had (or its default size the
        first time), floored at its content minimum and clamped on-screen. Called by
        _show_page once the stack shows that page."""
        # Re-floor the window for the screen we're entering BEFORE restoring its geometry:
        # minimumSizeHint() already reflects the new page. If we restored first, a wider screen's
        # leftover minimum (e.g. Review's ~1116) would clamp a narrower screen's restored width and
        # silently clobber it (the deferred re-floor afterwards couldn't bring the lost size back).
        self._apply_min_size()
        saved = self._settings.value(self._geometry_key(new_index))
        if saved:
            # The saved geometry carries the page's last position too, which can be on
            # another display (Review last used on an external monitor): the page keeps its
            # remembered size but stays on the display the window is on now.
            screen_before = self.screen()
            top_left_before = self.frameGeometry().topLeft()
            if self.restoreGeometry(saved):
                self._keep_on_screen(screen_before, top_left_before)
            else:
                # Saved rect no longer lands on any available screen (e.g. an external
                # monitor was unplugged since the last save): fall back to the default
                # size. No centering here — the deferred _fit_to_screen below already
                # clamps the window back on-screen for a page switch.
                self.resize(*self._default_size(new_index))
        elif not saved:
            # First-ever visit (or post-reset): open the screen at its own default size.
            # Review's default deliberately exceeds small displays — the deferred fit
            # below trims it, so Review opens as large as the screen allows.
            self.resize(*self._default_size(new_index))
        # else: geometry restore is off — keep the current, manually-sized window as-is.
        QTimer.singleShot(0, self._fit_to_screen)

    def _keep_on_screen(self, screen: QScreen | None, top_left: QPoint) -> None:
        """After a page switch restored that page's saved geometry: when the restored window
        sits mostly off ``screen`` (the display the window was on before the switch), move it
        back to ``top_left``, where it was, keeping the restored size. A restored maximized or
        full-screen state is cleared first: a window manager ignores move() on such a window,
        so it would stay on the other display. The deferred _fit_to_screen then trims anything
        that overhangs that display's edges. (For a top-level widget pos() already includes
        the frame, so ``top_left``, taken from frameGeometry(), is what move() wants.)"""
        if screen is None:
            return
        if screen.availableGeometry().contains(self.frameGeometry().center()):
            return
        if self.isMaximized() or self.isFullScreen():
            self.setWindowState(Qt.WindowState.WindowNoState)
        self.move(top_left)

    def _apply_min_size(self) -> None:
        """Set the window minimum to the current page's natural content minimum, so the page
        can't be squeezed small enough to overlap its controls. Capped to the available screen
        (minus window-manager chrome) so the floor never exceeds the display and shoves a tall
        page off a small screen."""
        hint = self.minimumSizeHint()   # current page's composed minimum (via FittingStack)
        w, h = hint.width(), hint.height()
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            chrome_w = self.frameGeometry().width() - self.width()
            chrome_h = self.frameGeometry().height() - self.height()
            w = min(w, avail.width() - chrome_w)
            h = min(h, avail.height() - chrome_h)
        self.setMinimumSize(w, h)

    def _fit_to_screen(self) -> None:
        """Clamp the window to the screen's available area (size and position), so a page
        switch can never leave it clipping past the screen edges. Runs only on page changes:
        manual drag-resizing is untouched."""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        # Window-manager chrome (title bar) counts toward the on-screen footprint.
        chrome_w = self.frameGeometry().width() - self.width()
        chrome_h = self.frameGeometry().height() - self.height()
        w = min(self.width(), avail.width() - chrome_w)
        h = min(self.height(), avail.height() - chrome_h)
        if (w, h) != (self.width(), self.height()):
            self.resize(w, h)
        frame = self.frameGeometry()
        x = min(max(frame.x(), avail.x()), avail.x() + avail.width() - frame.width())
        y = min(max(frame.y(), avail.y()), avail.y() + avail.height() - frame.height())
        if (x, y) != (frame.x(), frame.y()):
            self.move(self.x() + (x - frame.x()), self.y() + (y - frame.y()))

    def _center_on_screen(self) -> None:
        """Center the window (frame included) on its current screen, falling back to the primary
        one. Used on a first-run launch and when the layout is reset."""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        frame = self.frameGeometry()
        frame.moveCenter(screen.availableGeometry().center())
        self.move(frame.topLeft())

    def restore_startup_geometry(self) -> None:
        """Restore the analysis page's saved geometry (the window opens on that screen), or size
        it to that page's default and center on first run / after a layout reset / when the saved
        rect no longer lands on any available screen (e.g. an external monitor was unplugged since
        the last save), then floor the window at its content minimum and clamp it onto the screen.
        Call once after show(): the window must be assigned to a screen so restoreGeometry() can
        validate the saved rect still lands on an available one; the min-size and the fit are
        deferred a turn so the layout has settled (page switches re-apply both for the Review
        page)."""
        saved = self._settings.value(self._geometry_key(0))
        if not (saved and self.restoreGeometry(saved)):
            self.resize(*self._default_size(0))
            self._center_on_screen()
        QTimer.singleShot(0, self._apply_min_size)
        # Deferred fit (after the min-size settles): a default larger than the display
        # must be trimmed on the very first paint, not only on the first page switch.
        QTimer.singleShot(0, self._fit_to_screen)

    def reset_window_layout(self) -> None:
        """Forget both saved per-screen geometries and snap the live window back to the current
        page's default size (clamped to the screen), centered. Invoked by the Preferences
        'Reset window layouts' button; the next save re-stores this reset state. The Settings
        dialog resets its own size and position alongside this (see SettingsDialog)."""
        self._settings.remove(SETTING_GEOMETRY_MAIN)
        self._settings.remove(SETTING_GEOMETRY_REVIEW)
        # The upload assistant keeps its geometry under its own key. An open window resets itself
        # (it would otherwise re-save the old geometry from closeEvent); one that was never opened
        # this session has no object to ask, so just clear the key.
        if self._upload_assistant is not None:
            self._upload_assistant.reset_layout()
        else:
            forget_upload_assistant_geometry(self._settings)
        self._apply_min_size()
        self.resize(*self._default_size(self._stack.currentIndex()))
        self._center_on_screen()
        QTimer.singleShot(0, self._fit_to_screen)

    # ── Settings / menu bar ────────────────────────────────────────────────────
    def _build_menu_bar(self) -> QMenuBar:
        menubar = QMenuBar(self)
        # Preferences is decoupled from the (hideable) Debug menu: the only link between them is the
        # show/hide-Debug toggle in the Preferences pane. It lives in its own always-visible "File"
        # menu so that on macOS PreferencesRole relocates it into the application menu (⌘,) and keeps
        # it ENABLED regardless of Debug. (A bare menubar action doesn't relocate; it vanishes; a
        # role action in a HIDDEN menu greys out: so the host has to be a permanent, visible menu.)
        # The title-row gear button stays the always-present fallback on every platform.
        file_menu = menubar.addMenu("File")
        assert file_menu is not None
        prefs = QAction("Preferences…", self)
        prefs.setMenuRole(QAction.MenuRole.PreferencesRole)
        prefs.setShortcut(QKeySequence.StandardKey.Preferences)
        prefs.triggered.connect(self._open_settings)
        file_menu.addAction(prefs)

        # A conversion the user asks for, for a chosen video that loads but won't show (a
        # black picture gives the app nothing to notice, so there is no automatic offer for
        # it). Enable-gated like the export action below: Analyze page only, with a video
        # chosen and no run in progress (see _refresh_convert_action). NoRole keeps it in
        # this menu on macOS.
        self._convert_action = QAction("Convert to MP4…", self)
        self._convert_action.setMenuRole(QAction.MenuRole.NoRole)
        self._convert_action.setEnabled(False)  # no video is chosen yet
        self._convert_action.triggered.connect(self._on_convert_to_mp4)
        file_menu.addAction(self._convert_action)

        # The Review tree's single-clip export (same as its right-click action), on the current
        # selection. Unlike the Tools/Edit menus, File is visible on every page, so this action
        # is enable-gated instead of hidden: Review page only, and not during a subprocess run
        # (see _refresh_export_selected_action): which also keeps its shortcut inert there,
        # since a disabled action never fires, and relabels "selected" to "selected set"/
        # "selected game" once a review is loaded. NoRole keeps it in this menu on macOS.
        self._export_selected_action = QAction("Export selected to a file…", self)
        self._export_selected_action.setMenuRole(QAction.MenuRole.NoRole)
        self._export_selected_action.setShortcut("Ctrl+E")  # ⌘E on macOS
        self._export_selected_action.setEnabled(False)  # starts on the Analyze page
        self._export_selected_action.triggered.connect(self._review.export_selected_clip)
        file_menu.addAction(self._export_selected_action)
        file_menu.addSeparator()

        # One-click route to the (standalone) Upload Assistant: no need to walk through
        # analyze/review/export first; it opens empty and its Open… button picks an export
        # CSV. NoRole keeps it in this menu on macOS.
        self._assistant_action = QAction("Upload Assistant…", self)
        self._assistant_action.setMenuRole(QAction.MenuRole.NoRole)
        self._assistant_action.triggered.connect(self._show_upload_assistant)
        file_menu.addAction(self._assistant_action)

        # Compute status + (on Windows) the GPU-acceleration download. Reachable here and in
        # Settings → General; both open the same status panel. NoRole keeps it in this menu on macOS.
        gpu = QAction("GPU Acceleration…", self)
        gpu.setMenuRole(QAction.MenuRole.NoRole)
        gpu.triggered.connect(self._on_setup_gpu)
        file_menu.addAction(gpu)

        menu = menubar.addMenu("Debug")
        assert menu is not None
        self._debug_menu = menu  # whole menu is shown/hidden by the "Show Debug in menu bar" preference

        # The run-transcript popup, opened from here rather than a loose per-page button. NoRole keeps
        # it in this menu on macOS (unlike Preferences, which the role relocates to the app menu). The
        # transcript itself is the hidden self._log buffer set up in __init__.
        show_log = QAction("Show log…", self)
        show_log.setMenuRole(QAction.MenuRole.NoRole)
        show_log.triggered.connect(self._show_log)
        menu.addAction(show_log)

        # The crash log (vodparse.gui.crash_log), which is a different thing from "Show log…":
        # that one is this run's subprocess transcript, this one is the persistent file that
        # survives the process dying. Opened with the OS default handler rather than shown in a
        # dialog, since the interesting case is reading it *after* a crash took the window down.
        crash = QAction("Show crash log…", self)
        crash.setMenuRole(QAction.MenuRole.NoRole)
        crash.triggered.connect(self._on_show_crash_log)
        menu.addAction(crash)

        # Test hook for the Windows live theme-flip crash: flips the OS light/dark setting from
        # inside the app, so the real colorSchemeChanged path runs. Windows Server has no
        # Personalization → Colors page to click, and PowerToys-style schedulers flip this same
        # registry value, so this reproduces the real-world trigger. Windows-only.
        if sys.platform == "win32":
            flip = QAction("Flip OS dark mode now…", self)
            flip.setMenuRole(QAction.MenuRole.NoRole)
            flip.setToolTip("Toggles the Windows apps light/dark registry setting, "
                            "reproducing a live OS theme flip (see the crash log).")
            flip.triggered.connect(self._on_flip_os_theme)
            menu.addAction(flip)

        # Writes the diagnostics zip for the analyzed VODs. A utility action like Show log, kept in
        # this menu rather than Tools; NoRole keeps it here on macOS.
        diag = QAction("Export Diagnostics…", self)
        diag.setMenuRole(QAction.MenuRole.NoRole)
        diag.triggered.connect(self._on_export_diagnostics)
        menu.addAction(diag)

        # Test hook: run the real GPU-support download while bypassing the NVIDIA-GPU check, so the
        # whole download→unpack→activate→probe path can be exercised on a machine without an NVIDIA
        # card (the platform guard still applies: CUDA wheels are Windows-only). NoRole keeps it here.
        force_gpu = QAction("Force download GPU support (ignore GPU check)…", self)
        force_gpu.setMenuRole(QAction.MenuRole.NoRole)
        force_gpu.triggered.connect(lambda: self._gpu.offer_install(force=True))
        menu.addAction(force_gpu)

        # Runs the real first-run flow ("Checking for GPU…" → offer / no-GPU notice) without
        # the installed/skip-flag guards, so it can be previewed on any build. It exercises
        # the shipped behavior for real: a no-GPU result persists the skip flag and "Don't ask
        # again" is honored; on a source checkout it ends in offer_install's pip hint. Left
        # out on macOS (see _gpu_offer_preview_applies), where there is nothing to offer.
        if _gpu_offer_preview_applies():
            first_run_gpu = QAction("Run first-run GPU offer…", self)
            first_run_gpu.setMenuRole(QAction.MenuRole.NoRole)
            first_run_gpu.triggered.connect(self._gpu.check_then_offer)
            menu.addAction(first_run_gpu)

        # A persisted checkable toggle for the smart-tag pick popup shown after a --smart-tags
        # run / re-read. Off by default: the full ranking always goes to the log (Show log…);
        # flip this on to also get the winning-combo popup. NoRole keeps it in this menu on macOS.
        menu.setToolTipsVisible(True)  # action tooltips don't show in a QMenu otherwise
        menu.addSeparator()
        self._smart_tags_popup_action = QAction("Show popup when using smart tags", self)
        self._smart_tags_popup_action.setMenuRole(QAction.MenuRole.NoRole)
        self._smart_tags_popup_action.setCheckable(True)
        self._smart_tags_popup_action.setChecked(self._show_smart_tags_popup())

        self._smart_tags_popup_action.toggled.connect(
            lambda on: self._settings.setValue(SETTING_SHOW_SMART_TAGS_POPUP, on))
        menu.addAction(self._smart_tags_popup_action)

        tools = menubar.addMenu("Tools")
        assert tools is not None
        self._tools_menu = tools  # whole menu (and the wrench button) shows only on Review; see
                                  # _apply_tools_visibility
        # The Tools actions. Stored on self so the in-window Tools button (see _build_title_row)
        # reuses the very same actions. They act on the Review screen's clips, so the whole Tools
        # surface is shown only on the Review page (toggled in _on_screen_changed): no enable gating
        # needed, since it's unreachable elsewhere.
        self._split_sponsor_action = QAction("Split sponsor from tag…", self)
        self._split_sponsor_action.triggered.connect(self._review.split_sponsors_from_tags)
        # Ctrl+Shift+T (mnemonic: tag) joins the Review Ctrl+Shift+<key> family; the Tools menu
        # is hidden off Review, which suppresses the shortcut there just like the Edit-menu keys.
        self._split_sponsor_action.setShortcut("Ctrl+Shift+T")
        self._tools_menu.addAction(self._split_sponsor_action)
        self._fix_caps_action = QAction("Fix capitalization across all games…", self)
        self._fix_caps_action.triggered.connect(self._review.fix_capitalization)
        self._tools_menu.addAction(self._fix_caps_action)
        # Fix a forgotten/wrong tournament name without leaving Review; the current name is
        # always visible in the Review heading (see _update_title).
        self._rename_tournament_action = QAction("Change tournament name…", self)
        self._rename_tournament_action.triggered.connect(self._on_change_tournament_name)
        self._tools_menu.addAction(self._rename_tournament_action)
        # Re-read tags: regenerate this VOD's matches under different tag-reading flags. It runs
        # a subprocess (unlike the other Tools actions, which edit in place), so unlike them it's
        # disabled during a run (see _set_running). Review-only visibility comes free with Tools.
        self._tools_menu.addSeparator()
        self._reread_action = QAction("Reread tags…", self)
        self._reread_action.triggered.connect(self._on_reread_tags)
        self._tools_menu.addAction(self._reread_action)

        # Edit / Playback / View act on the Review screen, so: like Tools: the whole menus are
        # shown only on the Review page (see _apply_tools_visibility). Hiding a menu suppresses its
        # actions' keyboard shortcuts, which is what keeps these editing keys inert off Review. The
        # actions themselves (and their per-selection enablement) live in ReviewScreen.
        def _add_menu(title: str, actions: list) -> QMenu:
            menu = menubar.addMenu(title)
            assert menu is not None
            for action in actions:
                if action is None:
                    menu.addSeparator()
                else:
                    menu.addAction(action)
            return menu

        self._edit_menu = _add_menu("Edit", self._review.edit_menu_actions())
        self._playback_menu = _add_menu("Playback", self._review.playback_menu_actions())
        self._view_menu = _add_menu("View", self._review.view_menu_actions())

        # Help is a top-level, always-visible menu (unlike the Review-only Edit/Tools menus)
        # so AboutRole's macOS relocation works on every page. On macOS the role pulls the
        # action out of Help into the application menu (Ultimate VOD Studio ▸ About …); on
        # Windows/Linux it stays here, the conventional home for About. Same role mechanism as
        # Preferences above, which PreferencesRole relocates to the app menu.
        help_menu = menubar.addMenu("Help")
        assert help_menu is not None
        about = QAction("About Ultimate VOD Studio…", self)
        about.setMenuRole(QAction.MenuRole.AboutRole)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

        # Check for Updates: compares this build against the latest GitHub Release and, if newer,
        # links to the download page (it never updates in place). ApplicationSpecificRole puts it
        # in the macOS application menu next to About (the platform-conventional home for it);
        # on Windows/Linux it stays here in Help.
        updates = QAction("Check for Updates…", self)
        updates.setMenuRole(QAction.MenuRole.ApplicationSpecificRole)
        updates.triggered.connect(self._updates.check_interactive)
        help_menu.addAction(updates)

        self._apply_debug_menu_visibility()  # honour the saved "Show Debug in menu bar" preference
        return menubar

    def _open_settings(self) -> None:
        SettingsDialog(self._settings, parent=self,
                      on_reset_layout=self.reset_window_layout).exec()
        # Re-apply settings with a live UI effect (a no-op when the dialog was cancelled).
        # A font-scale change only takes hold on the next launch; the hint under the
        # Text size control in Settings says so, so there's no modal here.
        self._review.refresh_flag_highlights()    # re-reads the flag toggles/limits and re-styles rows
        self._refresh_lang_note()                 # reflect a just-changed overlay language on the main screen
        self._refresh_engine_note()               # show/hide the font style note per the toggle
        self._apply_debug_menu_visibility()       # show/hide the Debug menu per the toggle
        self._apply_layout_preview_setting()      # layout preview or first-frame still per the toggle
        # "Reset all preferences" clears the game key and the per-game layout keys; rebuilding
        # the Game combo re-reads both, since restoring a game refills the layout combo.
        self._populate_game_combo()

    def _apply_layout_preview_setting(self) -> None:
        """Settings closed: switch the Preview box to the look the toggle names (see
        _show_layout_preview). Off stops both searches first, since the box no longer shows
        answers, and takes the wrong-game doubt down with the note refresh; on previews the
        chosen video again and asks for its clock (_refresh_layout_preview, which a freshly
        built layout preview needs; a no-op without a video, and off it asks nothing)."""
        if not self._show_layout_preview():
            self._clock_locator.cancel()
            self._side_overlay_locator.cancel()
        self._preview_pane.apply_layout_preview_setting()
        self._refresh_layout_preview()
        self._refresh_game_note()

    def _show_about(self) -> None:
        """Open the About box (Help menu, or the macOS application menu via AboutRole)."""
        AboutDialog(self).exec()

    def _on_setup_gpu(self) -> None:
        """Open the GPU Acceleration window (compute status + the download, on Windows)."""
        GpuSupportDialog(self).exec()

    def _maybe_offer_gpu_on_first_run(self) -> None:
        """Consider the one-time CUDA-download offer: only on Windows, nothing installed yet,
        and not already resolved. The check itself (a torch-importing compute probe) takes
        seconds, so the manager shows a visible "Checking for GPU…" dialog while it runs and
        then offers, reports "no GPU", or drops out on cancel (see check_then_offer). The
        guards here are cheap (no hardware calls)."""
        if not gpu_support.is_supported_platform():
            return
        if not is_frozen():
            # A source checkout never activates the downloaded site (activate_cuda_site() runs
            # only from the frozen entry), so an offer here would push a multi-gigabyte download
            # that can't work. Source users with an NVIDIA GPU install the CUDA torch into their
            # venv instead.
            return
        if gpu_support.active_site_dir() is not None:
            return
        if self._settings.value(SETTING_GPU_SKIP_FIRST_RUN_OFFER, False, type=bool):
            return
        self._gpu.check_then_offer()

    def _suppress_gpu_first_run_offer(self) -> None:
        """Stop the first-run GPU check on later launches: the user chose 'Don't ask again' on
        the offer (``declined``), or the check found no NVIDIA GPU (``no_gpu_found``). Either
        way there is nothing left to ask. A cancelled check or a plain dismissal of the offer
        reaches neither signal, so those retry next launch."""
        self._settings.setValue(SETTING_GPU_SKIP_FIRST_RUN_OFFER, True)

    def _apply_debug_menu_visibility(self) -> None:
        """Show or hide the whole Debug menu per the Preferences toggle. Preferences itself stays
        reachable via the in-window gear button, so hiding the menu never locks the user out."""
        action = self._debug_menu.menuAction()
        if action is not None:
            action.setVisible(self._show_debug_menu())

    def _apply_tools_visibility(self, index: int) -> None:
        """Show the Review-only menu surfaces only on the Review page (index 1): the in-window
        wrench button and the Tools/Edit/Playback/View menus. Their actions act on Review's clips,
        so off Review the whole surface is hidden rather than shown-but-disabled: and hiding the
        Edit/Playback/View menus also suppresses their keyboard shortcuts off Review."""
        visible = index == 1
        for menu in (self._tools_menu, self._edit_menu, self._playback_menu, self._view_menu):
            action = menu.menuAction()
            if action is not None:
                action.setVisible(visible)
        self._tools_btn.setVisible(visible)
        self._refresh_export_selected_action()
        self._refresh_convert_action()

    def _refresh_export_selected_action(self) -> None:
        """Enable File → "Export selected set/game to a file…" only where it can act: on the
        Review page, with no subprocess running. The run state comes from self._running (not
        a live QProcess check) because _set_running(True) fires before the process object
        exists, when that check would still say idle. One combined check (re-run on page
        switches and run-state flips) so the two gates can't stomp each other's setEnabled."""
        self._export_selected_action.setEnabled(
            self._stack.currentIndex() == 1 and not self._running)

    def _refresh_convert_action(self) -> None:
        """Enable File → "Convert to MP4…" only where it can act: on the Analyze page
        (index 0), with a video chosen and no subprocess running. Re-run on page switches,
        run-state flips and whenever the chosen video changes (_refresh_nav), as one
        combined check, like _refresh_export_selected_action. A conversion already in
        progress needs no gate here: its progress dialog is modal over this window, and
        Mp4Converter.convert_on_request ignores a second request anyway."""
        self._convert_action.setEnabled(
            self._stack.currentIndex() == 0 and self._video_path is not None
            and not self._running)

    def _on_convert_to_mp4(self) -> None:
        """File → "Convert to MP4…": convert the chosen video on request. The converter's
        fileReady lands in _finish_selection, so the MP4 becomes the chosen video and the
        Preview reloads once the conversion is done; cancelling leaves the selection alone."""
        if self._video_path is not None:
            self._converter.convert_on_request(self._video_path)

    def _relabel_export_selected_action(self) -> None:
        """Match the File export action's label to the loaded review's unit: "Export
        selected set/game to a file…". Called where a review model loads (fresh load and
        draft resume); the menu is built before any review exists, so the generic
        "selected" label holds until then, where the action is disabled anyway."""
        target = self._review.export_target()
        unit = "" if target is None else f" {self._review.unit_label().lower()}"
        self._export_selected_action.setText(f"Export selected{unit} to a file…")

    # ── Settings accessors ────────────────────────────────────────────────────
    def _title_char_limit(self) -> int:
        """Max clip-title length (default 255, the OS filename limit); longer titles get their round
        abbreviated/dropped on export."""
        return self._settings.value(SETTING_TITLE_CHAR_LIMIT, TITLE_CHAR_LIMIT_DEFAULT, type=int)

    def _title_filenames(self, mode: str) -> bool:
        """Whether exported clip files are named after their titles for this export mode
        (defaults: on for sets, off for matches)."""
        if mode == "match":
            return self._settings.value(SETTING_TITLE_FILENAMES_MATCHES, False, type=bool)
        return self._settings.value(SETTING_TITLE_FILENAMES_SETS, True, type=bool)

    def _show_debug_menu(self) -> bool:
        """Whether the Debug menu (Preferences / Show log / Export Diagnostics) appears in the
        menu bar (default False: hidden until enabled). The gear button is always the way into
        Preferences, so it's reachable to turn this back on."""
        return self._settings.value(SETTING_SHOW_DEBUG_MENU, False, type=bool)

    def _ask_convert_mkv(self) -> bool:
        """Whether choosing an .mkv brings up the "convert to MP4" offer where the player
        shows an MKV directly (Windows; default False). Other platforms always make the
        offer and don't consult this (see ffmpeg_runtime.offers_conversion)."""
        return self._settings.value(
            SETTING_ASK_CONVERT_MKV, ASK_CONVERT_MKV_DEFAULT, type=bool)

    def _show_layout_preview(self) -> bool:
        """Whether the Preview box is the layout preview, with its clock and player-name
        searches (default True). Off, the box shows a still of the chosen video's first
        frame (the first release's look) and this window asks no search and uses no answer:
        no wrong-game doubt, nothing handed to the run, which does its own pre-passes."""
        return self._settings.value(
            SETTING_SHOW_LAYOUT_PREVIEW, SHOW_LAYOUT_PREVIEW_DEFAULT, type=bool)

    def _show_smart_tags_popup(self) -> bool:
        """Whether the smart-tag pick popup appears after a --smart-tags run or re-read
        (default False). Toggled from the Debug menu; the full ranking always goes to the
        log transcript (Show log…) regardless."""
        return self._settings.value(SETTING_SHOW_SMART_TAGS_POPUP, False, type=bool)

    def _font_scale(self) -> float:
        """Whole-UI text-size factor (default 1.0). Applied once at startup in main() via
        apply_font_scale; the title font (which pins an absolute size) reads this to scale in step."""
        return self._settings.value(SETTING_FONT_SCALE, FONT_SCALE_DEFAULT, type=float)

    def _default_vod_dir(self) -> str:
        """The folder the 'Choose input video' picker opens to (blank = wherever last browsed)."""
        return self._settings.value(SETTING_DEFAULT_VOD_DIR, "", type=str)

    # ── Stage heading ─────────────────────────────────────────────────────────
    _STAGE_NAMES = ("Analyze VOD", "Review")

    def _review_mode_note(self) -> str:
        """The Review-title parenthetical for the current VOD's resolved tag source: read from
        the per-VOD extract_flags sidecar (the flags that actually produced matches.json),
        falling back to the live main-page selection for a pre-sidecar VOD. Empty when the tag
        source is the default (see review_mode_note)."""
        video = self._video_path
        if video is None:
            return ""
        sidecar = read_sidecar(output_dir_for(video) / EXTRACT_FLAGS_NAME)
        if sidecar is not None:
            return review_mode_note(sidecar["group_by"],
                                    bool(sidecar["flags"]["overlay_at_bottom"]))
        return review_mode_note(self._selected_group_by(),
                                self._extract_opts["overlay_at_bottom"])

    def _update_title(self, index: int) -> None:
        stage = self._STAGE_NAMES[index] if 0 <= index < len(self._STAGE_NAMES) else ""
        text = stage or "Ultimate VOD Studio"
        if index == 1 and self._review.tournament_name():
            # The Review heading carries the tournament name so a forgotten/wrong name is
            # visible before export (fixable via Tools → Change tournament name…).
            text = f"{text}: {self._review.tournament_name()}"
        self._title.setText(text)
        # The mode note rides in its own smaller, muted label so the heading elides cleanly;
        # the color re-resolves here (not at construction) to track a light/dark theme flip.
        note = self._review_mode_note() if index == 1 else ""
        self._title_note.setText(f"({note})")
        _colour_note(self._title_note, muted_text_color())
        self._title_note.setVisible(bool(note))

    def _build_title_row(self) -> QHBoxLayout:
        """The heading row: the stage title on the left, then in-window Preferences (gear) and
        Tools (wrench) buttons pushed to the right. These mirror the macOS global menu bar, which
        is detached from the window and easy to miss: the menu bar itself stays for the native
        ⌘, shortcut and app-menu placement. Must run after _build_menu_bar, which creates the
        Tools action the wrench dropdown reuses."""
        self._prefs_btn = make_icon_button(
            "⚙", "Preferences…", QStyle.StandardPixmap.SP_FileDialogDetailedView)
        self._prefs_btn.clicked.connect(self._open_settings)

        # Reuse the menu bar's own Tools action so the two stay in lockstep: notably the
        # Review-only enable toggle on _fix_caps_action reaches this dropdown for free.
        tools_menu = QMenu(self)
        tools_menu.addAction(self._split_sponsor_action)
        tools_menu.addAction(self._fix_caps_action)
        tools_menu.addAction(self._rename_tournament_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self._reread_action)
        self._tools_btn = make_icon_button(
            "⚒", "Tools", QStyle.StandardPixmap.SP_FileDialogContentsView)
        self._tools_btn.setMenu(tools_menu)
        self._tools_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        # Reread tags lives only under Tools now (see _reread_action): it's not used often
        # enough to warrant its own title-row button next to the wrench.
        row = QHBoxLayout()
        row.addWidget(self._title)
        row.addWidget(self._title_note)
        row.addStretch(1)
        row.addWidget(self._tools_btn)
        row.addWidget(self._prefs_btn)   # Preferences sits on the far right
        return row

    def _make_info_button(self, title: str, body: str) -> QToolButton:
        """A flat circled-i button that pops a short explanatory dialog (and shows the
        same text on hover). Reusable for any section that wants an inline help affordance."""
        btn = QToolButton()
        btn.setAutoRaise(True)           # flat until hovered
        btn.setToolTip(body)             # hover also works, for free
        # Prefer the compact "ⓘ" glyph; fall back to the native style icon only on the
        # rare platform whose font lacks U+24D8 (else it renders as a tofu box).
        if btn.fontMetrics().inFont("ⓘ"):
            btn.setText("ⓘ")
        else:
            style = self.style()
            if style is not None:
                btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation))
        btn.clicked.connect(lambda: QMessageBox.information(self, title, body))
        return btn

    # ── Screen 1: main (analyze → extract) ────────────────────────────────────
    def _build_main_page(self) -> QWidget:
        """The single entry screen: pick a vod, set analysis/set-grouping options, then run the
        analyze→extract chain (or reuse a prior analysis / resume saved edits). Controls are grouped
        into Source / Game / Analysis boxes beside the live preview pane; each box (and the
        action row and status rows) is assembled by its own builder below."""
        page = QWidget()
        # The three boxes lay their rows on one shared set of columns (see form_columns), so
        # their labels, fields and info buttons line up from box to box.
        form_columns = FormColumns()
        source_box   = self._build_source_box(form_columns)
        game_box     = self._build_game_box(form_columns)
        analysis_box = self._build_analysis_box(form_columns)
        form_columns.align()   # once every box's rows are in
        action_row   = self._build_action_row()
        self._build_status_rows()
        preview_col_widget = self._preview_pane
        preview_col_widget.setMinimumWidth(PREVIEW_MIN_WIDTH_PX)

        # Left column: the option boxes stacked, with a trailing stretch that pins them to the top.
        # It's wrapped in a real widget (not a bare sub-layout) so QHBoxLayout lets it *expand* to
        # fill the row exactly like the preview widget does: both then start at the row's top, so
        # Source's top lines up with the Preview box's top. (An AlignTop'd or bare-sub-layout column
        # is instead handed a cell inset ~15px from the row top, which is what misaligned it.)
        controls = QVBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(source_box)
        controls.addWidget(game_box)
        controls.addWidget(analysis_box)
        controls.addStretch(1)
        controls_col = QWidget()
        controls_col.setLayout(controls)

        # Top: option column beside the preview, both expanding to fill the row from its top.
        top_row = QHBoxLayout()
        top_row.addWidget(controls_col, stretch=0)
        top_row.addWidget(preview_col_widget, stretch=1)

        layout = QVBoxLayout(page)
        layout.addLayout(top_row, stretch=1)
        layout.addWidget(self._export_mode_note)
        layout.addWidget(self._opts_warning_row)
        layout.addWidget(self._existing_analysis_label)
        layout.addLayout(action_row)
        return page

    def _build_source_box(self, form_columns: FormColumns) -> QGroupBox:
        """Source: pick the vod and name the tournament. Its rows go on `form_columns`,
        the columns the Game and Analysis boxes share with it."""
        # Kept on self: a run locks it (see _set_running), since the run's later stages
        # read the chosen video when they start.
        self._choose_btn = QPushButton("Choose video…")
        self._choose_btn.clicked.connect(self._on_choose)
        # ElideMiddle + a 22-char floor: long filenames elide in the middle (extension stays
        # visible) without widening the Source column and squeezing the preview pane beside it.
        self._file_label = ElidingLabel("No video selected",
                                        mode=Qt.TextElideMode.ElideMiddle, min_chars=22)

        # Prefixed to every clip title ("<tournament> <round> - A vs B"). Seeded from the vod's
        # filename when a video is chosen (see _on_choose); edits stick for that video. The
        # placeholder is set only once a video is chosen (see _seed_tournament_name), so an empty
        # field reads as blank: not "omitted from titles": before there's anything to analyze.
        self._tournament_edit = QLineEdit()
        self._tournament_prefilled_for: str | None = None

        source_box = QGroupBox("Source")
        source_grid = form_columns.new_grid(source_box)
        # The button sits in the label column, so the file name beside it starts where the
        # fields of every other row start.
        form_columns.add_row(source_grid, 0, self._choose_btn, self._file_label)
        form_columns.add_row(source_grid, 1, QLabel("Tournament name:"), self._tournament_edit)
        return source_box

    def _build_game_box(self, form_columns: FormColumns) -> QGroupBox:
        """Game: which game the vod is of, and the overlay layout it is read through. Both
        choices are per game (the layout list is rebuilt whenever the game changes), which is
        why they share a box apart from the Analysis options below them. Its rows go on
        `form_columns`, the columns the Source and Analysis boxes share with it."""
        # Teal note shown while a game with no clock detector of its own is selected, so its
        # need for an overlay layout is visible before Analyze VOD is pressed. Mirrors the
        # Analysis box's _lang_note style. Built before the combo: the populate at the end of
        # this method already refreshes it (see _refresh_game_note).
        self._game_note = QLabel()
        _colour_note(self._game_note, notice_color())  # teal, theme-aware
        self._game_note.setWordWrap(True)
        self._game_note.setVisible(False)

        # Which game the VOD is of (vodparse.games): it picks the detectors the pipeline
        # loads, the clock length the split assumes, and which overlay-layout presets the
        # combo below it offers. Restored from QSettings by _populate_game_combo and
        # persisted on change. Populated at the end of this method rather than here, because
        # selecting a game rebuilds the Overlay layout combo, which does not exist yet.
        self._game_combo = QComboBox()
        game_info_btn = self._make_info_button(
            "About games",
            "Select the game you are analyzing. If your game is not listed here, you can "
            "select \"Other\" and make a custom overlay for it.")

        # Custom overlay layout: which saved preset (if any) the pipeline reads player tags,
        # round, best-of and the match clock from, instead of the built-in overlay heuristics.
        # A per-VOD choice, so it lives here rather than in Settings. Its contents depend on the
        # Game combo above (only that game's presets, and "Auto" only when the game has a clock
        # detector), so it too is populated at the end of this method, through the Game combo.
        # A PopupPickComboBox: its "New layout…" action item opens the editor only on a pick
        # in the popup (see _on_overlay_layout_activated).
        self._overlay_layout_combo = PopupPickComboBox()
        # Presets are named in the editor, so the combo grows to fit a longer name, but only
        # as far as OVERLAY_LAYOUT_COMBO_MAX_CHARS: past that the closed combo cuts the name
        # short with "…" instead of widening the whole column and squeezing the preview.
        self._overlay_layout_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._overlay_layout_combo.setMaximumWidth(
            combo_width_for_chars(self._overlay_layout_combo, OVERLAY_LAYOUT_COMBO_MAX_CHARS))
        layout_info_btn = self._make_info_button(
            "About overlay layouts",
            "Define a custom layout for reading player tags and the timer. Do this if you "
            "have a particularly difficult overlay that isn't being read by \"Auto\".")

        # Opens the layout editor on the combo's choice (see _on_edit_layouts). It sits
        # right beside the combo, in the Overlay layout row, so its short name needs no
        # "layouts" to say what it edits.
        edit_layouts_btn = QPushButton(EDIT_LAYOUTS_BUTTON_TEXT)
        edit_layouts_btn.clicked.connect(self._on_edit_layouts)

        game_box = QGroupBox("Game")
        game_grid = form_columns.new_grid(game_box)
        form_columns.add_row(game_grid, 0, QLabel("Game:"), self._game_combo,
                             info=game_info_btn)
        form_columns.add_full_width(game_grid, 1, self._game_note)
        form_columns.add_row(game_grid, 2, QLabel("Overlay layout:"), self._overlay_layout_combo,
                             side_button=edit_layouts_btn, info=layout_info_btn)
        # The Game and Overlay layout combos are populated here, together and in this order:
        # restoring the saved game runs _on_game_selected, which refreshes the game note and
        # fills the layout combo for that game. Each handler is connected only after its combo
        # has been restored (the order the Analysis box's _ocr_engine_combo uses too), so
        # restoring never counts as a change.
        self._populate_game_combo()
        self._game_combo.currentIndexChanged.connect(self._on_game_selected)
        self._overlay_layout_combo.currentIndexChanged.connect(self._on_overlay_layout_selected)
        self._overlay_layout_combo.activated.connect(self._on_overlay_layout_activated)
        return game_box

    def _build_analysis_box(self, form_columns: FormColumns) -> QGroupBox:
        """Analysis: how the pipeline scans the vod (text cadence, font style), plus the
        Advanced Analysis Options button for the extract's rarer knobs ("Export as" sets or
        games, tag reading, set characters, debug tables). Its rows go on `form_columns`,
        the columns the Source and Game boxes share with it."""
        # How often OCR fires (and, in seek mode, how far each step jumps). Restores the last-used
        # cadence from QSettings; the seconds for each choice ride along as item data.
        self._ocr_interval_combo = QComboBox()
        for label, secs in OCR_INTERVAL_CHOICES.items():
            self._ocr_interval_combo.addItem(label, secs)
        saved_interval = self._settings.value(
            SETTING_OCR_INTERVAL_SEC, OCR_INTERVAL_DEFAULT_SEC, type=int)
        saved_idx = self._ocr_interval_combo.findData(saved_interval)
        self._ocr_interval_combo.setCurrentIndex(saved_idx if saved_idx >= 0 else
                                                 self._ocr_interval_combo.findData(OCR_INTERVAL_DEFAULT_SEC))
        ocr_info_btn = self._make_info_button(
            "About analysis",
            "How often the video is checked for tags, characters and the clock.\n\n"
            "Shorter intervals are more accurate, but slower.")

        # Which engine reads the overlay text (see vodparse.ocr_engine). docTR ships with
        # the app; in an environment without it Stylized stays listed, selectable and
        # selected when saved, marked "not installed": choosing it, the Preview and
        # Analyze VOD then say what's missing, and nothing reads with Standard in its place.
        # The selection handler offers the one-time model download (a no-op when the frozen
        # bundle carries the models): connected after the restore below, so merely
        # restoring a saved docTR choice never pops the offer.
        self._ocr_engine_combo = QComboBox()
        for label, engine_id in OCR_ENGINE_CHOICES.items():
            self._ocr_engine_combo.addItem(label, engine_id)
        if not doctr_available():
            doctr_idx = self._ocr_engine_combo.findData(OCR_ENGINE_DOCTR)
            self._ocr_engine_combo.setItemText(
                doctr_idx, self._ocr_engine_combo.itemText(doctr_idx) + ": not installed")
        saved_engine = self._settings.value(SETTING_OCR_ENGINE, OCR_ENGINE_DEFAULT_GUI, type=str)
        self._ocr_engine_combo.setCurrentIndex(self._ocr_engine_combo.findData(
            resolve_saved_engine(saved_engine)))
        self._ocr_engine_combo.currentIndexChanged.connect(self._on_ocr_engine_selected)
        self._ocr_engine_combo.currentIndexChanged.connect(self._refresh_engine_note)
        # Standard is EasyOCR; Stylized is docTR (PARSeq), which reads English/Latin only and
        # downloads its models (~DOCTR_WEIGHTS_MB_APPROX MB) once if missing. The match clock is
        # always read with EasyOCR, regardless of this choice.
        engine_info_btn = self._make_info_button(
            "About font styles",
            f"How the text on screen is read.\n\n{FONT_STYLE_STANDARD} is faster and works "
            f"with most overlays. {FONT_STYLE_STYLIZED} is slower but handles \"fancy\" fonts "
            f"(italics, comic sans and so on).\n\nIf {FONT_STYLE_STANDARD} causes the program "
            f"to miss text, try {FONT_STYLE_STYLIZED}.")

        # Muted note describing whichever font style is selected (see FONT_STYLE_NOTE_*): an
        # aside rather than something that needs acting on, so it stays quieter than the
        # _game_note / _lang_note (teal; the game note turns amber for the wrong-game doubt).
        self._engine_note = QLabel()
        _colour_note(self._engine_note, muted_text_color())  # grey, theme-aware
        self._engine_note.setWordWrap(True)
        self._engine_note.setVisible(False)

        # Teal note naming the overlay language, shown only while a non-default (non-English)
        # one is selected in Preferences → Analysis. _refresh_lang_note reflects a saved choice
        # here and a change applied via the Settings dialog. Mirrors _game_note's teal style.
        self._lang_note = QLabel()
        _colour_note(self._lang_note, notice_color())  # teal, theme-aware
        self._lang_note.setWordWrap(True)
        self._lang_note.setVisible(False)

        # The extract's rarer knobs live in a popup (see ExtractOptionsDialog), read back from
        # self._extract_opts by _selected_mode / _selected_set_characters. Kept on self: a run
        # locks it (see _set_running). It takes the whole last row of the box, as wide as
        # the box itself.
        self._options_btn = QPushButton("Advanced Analysis Options…")
        self._options_btn.clicked.connect(self._edit_extract_options)

        analysis_box = QGroupBox("Analysis")
        analysis_grid = form_columns.new_grid(analysis_box)
        form_columns.add_row(analysis_grid, 0, QLabel("Detect Text:"), self._ocr_interval_combo,
                             info=ocr_info_btn)
        form_columns.add_row(analysis_grid, 1, QLabel("Font style:"), self._ocr_engine_combo,
                             info=engine_info_btn)
        form_columns.add_full_width(analysis_grid, 2, self._engine_note)  # under its own combo
        form_columns.add_full_width(analysis_grid, 3, self._lang_note)
        form_columns.add_full_width(analysis_grid, 4, self._options_btn)
        self._refresh_engine_note()  # reflect the restored font style on first build
        self._refresh_lang_note()    # reflect a restored non-English overlay language on first build
        return analysis_box

    def _build_action_row(self) -> QHBoxLayout:
        """The run buttons: Analyze VOD runs pipeline→extract→Review; the second button reopens
        the saved Review draft when one exists ("Resume edits"), otherwise reuses an existing
        pipeline.json and only extracts→Review ("Open existing analysis"); Stop kills the
        in-flight stage."""
        self._analyze_btn = QPushButton("Analyze VOD")
        self._analyze_btn.clicked.connect(self._on_run)
        # Mark it the primary action so the style gives it the native "default button"
        # emphasis (and Enter triggers it): it's the main "go" button of the page.
        self._analyze_btn.setDefault(True)
        self._analyze_btn.setAutoDefault(True)
        # The no-draft default label; _refresh_nav sets the real label once a video is chosen.
        self._resume_btn = QPushButton("Open existing analysis")
        self._resume_btn.clicked.connect(self._on_resume_or_reuse)
        # Stop is live only during a run (see _set_running); it kills the in-flight stage.
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)
        # The full-transcript popup (LogDialog) is reached from the Debug menu's "Show log…"
        # (see _build_menu_bar); the transcript itself lives in the hidden self._log buffer.
        action_row = QHBoxLayout()
        action_row.addWidget(self._analyze_btn)
        action_row.addWidget(self._resume_btn)
        action_row.addStretch(1)
        action_row.addWidget(self._stop_btn)
        return action_row

    def _build_status_rows(self) -> None:
        """The rows that sit just above the run buttons: the Games export-mode reminder, the
        situational-options warning and the saved-analysis note. All start hidden;
        _refresh_export_mode_note / _refresh_opts_warning / _refresh_nav toggle them."""
        # Shown while the popup's Export as choice is Games (see EXPORT_MODE_GAMES_REMINDER).
        # Notice color: the choice is deliberate, so calmer than the amber warning below it.
        self._export_mode_note = QLabel(EXPORT_MODE_GAMES_REMINDER)
        _colour_note(self._export_mode_note, notice_color())  # teal, theme-aware
        self._export_mode_note.setWordWrap(True)
        self._export_mode_note.setVisible(False)

        # Appears only when a situational extract toggle (see SITUATIONAL_EXTRACT_OPTS) is
        # left on, so it can't silently carry into the next VOD.
        self._opts_warning = QLabel()
        _colour_note(self._opts_warning, warning_color(), bold=True)  # amber, theme-aware
        self._opts_warning_clear = QPushButton("Clear")
        self._opts_warning_clear.clicked.connect(self._clear_situational_opts)
        warning_layout = QHBoxLayout()
        warning_layout.setContentsMargins(0, 0, 0, 0)
        warning_layout.addWidget(self._opts_warning)
        warning_layout.addStretch(1)
        warning_layout.addWidget(self._opts_warning_clear)
        self._opts_warning_row = QWidget()
        self._opts_warning_row.setLayout(warning_layout)
        self._opts_warning_row.setVisible(False)

        # Sits just above the run buttons and shows the settings the saved analysis was run with
        # (see paths.existing_analysis_summary); visible (set in _refresh_nav) whenever a saved
        # pipeline.json exists, so it's clear what "Open existing analysis" (or a later re-extract)
        # would reuse.
        self._existing_analysis_label = QLabel()
        note_font = self._existing_analysis_label.font()
        note_font.setItalic(True)
        self._existing_analysis_label.setFont(note_font)
        self._existing_analysis_label.setVisible(False)

    # ── Navigation ────────────────────────────────────────────────────────────
    def _reuse_warning_ok(self, *, title: str, text: str, issues: list[str],
                          suppress_key: str | None = None) -> bool:
        """Warn before reusing a saved analysis, listing `issues` as bullets. Returns True to go
        ahead with the reuse; False when the user cancelled or chose Reanalyze (which this method
        starts itself, so the caller only has to stop).

        With a `suppress_key`, the dialog carries a "Don't show this again" checkbox and is skipped
        outright once that key is stored. The key is written only when the user chose Continue:
        ticking the box and then cancelling (or reanalyzing) must not silence a warning they never
        accepted. Restorable from Settings, which clears the whole SETTING_CONFIRM_GROUP."""
        if suppress_key is not None and self._settings.value(suppress_key, False, type=bool):
            return True  # silenced via "Don't show this again"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(text)
        box.setInformativeText("\n".join(f"• {issue}" for issue in issues))
        dont_show = QCheckBox("Don't show this again") if suppress_key is not None else None
        if dont_show is not None:
            box.setCheckBox(dont_show)
        cont = box.addButton("Continue anyway", QMessageBox.ButtonRole.AcceptRole)
        rerun = box.addButton("Reanalyze", QMessageBox.ButtonRole.DestructiveRole)
        # The explicit Cancel is also the escape button, so Esc aborts rather than launching a
        # full re-analysis (which is what a lone RejectRole button would have done).
        box.addButton(QMessageBox.StandardButton.Cancel)
        # Enter takes the reuse the user already asked for; neither Enter nor Esc should be able
        # to kick off a full re-analysis, which is the one choice here that costs real time.
        box.setDefaultButton(cont)
        box.exec()
        clicked = box.clickedButton()
        if clicked is rerun:
            self._on_run()  # full re-analysis, with its own overwrite-confirm guard
            return False
        if clicked is not cont:
            return False  # Cancel, or the dialog was dismissed
        if suppress_key is not None and dont_show is not None and dont_show.isChecked():
            self._settings.setValue(suppress_key, True)  # only once they actually go ahead
        return True

    def _on_resume_or_reuse(self) -> None:
        """The draft-aware second run button. With a saved Review draft ("Resume edits") it reopens
        the draft; without one ("Open existing analysis") it extracts from the saved pipeline.json
        instead. Both branches first warn if the video on disk no longer matches what was analyzed:
        the draft and the reused analysis are both timestamp-bearing, so a re-encoded/trimmed/
        swapped file would silently produce garbage clips."""
        video = self._video_path
        if video is None:
            return
        if not review_draft_for(video).exists():
            self._reuse_existing_analysis(video)
            return
        if not self._video_unchanged_ok(video):
            return
        if not self._resume_review():
            # The draft vanished between the exists() check and the load (or was unreadable).
            self._append("\n> No saved edits to resume.")
            self._refresh_nav()

    def _video_unchanged_ok(self, video: str) -> bool:
        """Warn when the video on disk no longer matches the metadata its analysis recorded (a
        re-encoded/trimmed/swapped file would silently produce garbage). Returns True to go ahead;
        never suppressible: a changed video means garbage rather than merely stale results. A
        missing or unreadable pipeline.json passes silently (nothing to compare against)."""
        issues = video_metadata_mismatches(video)
        return not issues or self._reuse_warning_ok(
            title="Video may have changed",
            text=f"{Path(video).name} looks different from when it was analyzed. "
                 "It's recommended that you reanalyze it, or the results may be inaccurate.",
            issues=issues)

    def _reuse_existing_analysis(self, video: str) -> None:
        """Open existing analysis: extract sets from the saved pipeline.json (skipping re-analysis),
        then open Review. Two warnings guard it first: the video on disk may no longer match what
        was analyzed, and the saved analysis may have been read with different text-reading settings
        than are selected now. Only the second is suppressible (see _video_unchanged_ok)."""
        if not self._video_unchanged_ok(video):
            return
        saved = existing_analysis(video)
        differences = [] if saved is None else analysis_settings_differences(
            saved, ocr_engine=self._ocr_engine_combo.currentData(),
            ocr_lang=self._settings.value(SETTING_OCR_LANG, OCR_LANG_DEFAULT_GUI, type=str),
            # "New layout…" (data None) is not a layout choice, so compare as Auto:
            # reusing an analysis reads the layout it recorded, not the combo.
            overlay_layout=self._overlay_layout_combo.currentData() or "",
            game=self._game_combo.currentData())
        if differences and not self._reuse_warning_ok(
                title="Analysis settings changed",
                text=f"{Path(video).name} was analyzed with different settings. "
                     "Reusing it keeps the old ones.\n\nChanged settings:",
                issues=differences,
                suppress_key=SETTING_CONFIRM_REUSE_SETTINGS):
            return
        self._start_extract()

    def _seed_tournament_name(self) -> None:
        """Seed the tournament name from the vod's filename, but only once per video so the
        user's edits survive (e.g. coming back from Review)."""
        video = self._video_path
        if video is None:
            return
        # Now that a video is chosen, show the placeholder so a cleared field explains itself.
        self._tournament_edit.setPlaceholderText("No tournament name (omitted from titles)")
        if self._tournament_prefilled_for != video:
            self._tournament_edit.setText(default_tournament_name(video))
            self._tournament_prefilled_for = video

    def _goto_main(self) -> None:
        """Return to the main page (e.g. Review's Back button)."""
        self._refresh_nav()  # the Resume-edits button depends on a draft existing for this vod
        self._show_page(0)
        self._request_clock()  # an Analyze run held the Preview box's clock search

    def _refresh_nav(self) -> None:
        """Enable the action buttons based on whether a video is chosen and its artifacts exist.
        The second button doubles up: it resumes the saved Review draft when one exists, and
        falls back to reusing the saved analysis otherwise: the label tracks which it would do."""
        video = self._video_path
        self._analyze_btn.setEnabled(video is not None)
        has_analysis = video is not None and pipeline_json_for(video).exists()
        has_draft = video is not None and review_draft_for(video).exists()
        self._resume_btn.setEnabled(has_draft or has_analysis)
        self._resume_btn.setText("Resume edits" if has_draft else "Open existing analysis")
        # Surface what the saved analysis was read with (cadence, font style, non-default
        # language) above the buttons whenever one exists on disk.
        if video is not None and has_analysis:
            self._existing_analysis_label.setText(existing_analysis_summary(video))
        self._existing_analysis_label.setVisible(has_analysis)
        self._refresh_convert_action()   # it needs a chosen video too

    def _set_running(self, running: bool) -> None:
        """Disable the action buttons during a subprocess run; restore them after.
        Choose video… is one of them: the extract stage and the Review load that follow
        the pipeline read the chosen video when they start, so a video picked mid-run
        would be extracted and loaded in place of the one being analyzed."""
        self._running = running
        for btn in (self._choose_btn, self._analyze_btn, self._resume_btn, self._options_btn):
            btn.setEnabled(not running)
        # Reread tags also kicks off a subprocess, so lock its Tools action during a run.
        self._reread_action.setEnabled(not running)
        # The Upload Assistant rewrites the whole export CSV on a tournament rename, and an
        # export running underneath it rewrites the same file when it finishes: whichever
        # writes last wins, silently. Lock the menu entry while an export runs (the Review
        # screen's own Upload button is locked separately, see set_upload_available).
        self._assistant_action.setEnabled(not (running and self._exporting))
        self._refresh_export_selected_action()  # also Review-only; see the combined check
        self._refresh_convert_action()          # Analyze-only; the same combined check
        self._stop_btn.setEnabled(running)  # inverse of the action buttons
        if not running:
            self._refresh_nav()  # reuse/resume buttons depend on artifacts existing

    # ── Export-mode state ──────────────────────────────────────────────────────
    def _selected_mode(self) -> Literal["match", "tournament_set"]:
        """The --mode value chosen as "Export as" in the Advanced Analysis Options popup
        (defaults to whole sets)."""
        if self._extract_opts.get("export_mode", EXPORT_MODE_DEFAULT) == "match":
            return "match"
        return "tournament_set"

    def _selected_group_by(self) -> str:
        """The --group-by value the extract and Review run with: always GROUP_BY_DEFAULT
        now that the picker is gone (see that constant)."""
        return GROUP_BY_DEFAULT

    def _selected_set_characters(self) -> str:
        """The --set-characters value chosen in the Advanced Analysis Options popup (defaults to 'join')."""
        return self._extract_opts.get("set_characters", SET_CHARACTERS_DEFAULT)

    def _edit_extract_options(self) -> None:
        """Open the advanced-flags popup; it mutates self._extract_opts in place on OK. Persist the
        chosen flags so they're remembered as the defaults next time (and next launch)."""
        if ExtractOptionsDialog(self._extract_opts, self).exec() == QDialog.DialogCode.Accepted:
            self._save_extract_opts()
            self._refresh_export_mode_note()
            self._refresh_opts_warning()

    def _load_extract_opts(self) -> dict:
        """Last-used extract options flags from QSettings, defaulting to EXTRACT_OPT_DEFAULTS
        (plus the strings set_characters and export_mode, tracked separately from the booleans)."""
        opts = {key: self._settings.value(f"extract/opt_{key}", default, type=bool)
                for key, default in EXTRACT_OPT_DEFAULTS.items()}
        opts["set_characters"] = self._settings.value(
            "extract/opt_set_characters", SET_CHARACTERS_DEFAULT, type=str)
        opts["export_mode"] = self._settings.value(
            "extract/opt_export_mode", EXPORT_MODE_DEFAULT, type=str)
        return opts

    def _save_extract_opts(self) -> None:
        for key, value in self._extract_opts.items():
            self._settings.setValue(f"extract/opt_{key}", value)

    def _refresh_export_mode_note(self) -> None:
        """Show the Games reminder above the run buttons only while the popup's Export as
        choice is Games (see EXPORT_MODE_GAMES_REMINDER)."""
        self._export_mode_note.setVisible(self._selected_mode() == "match")

    def _refresh_opts_warning(self) -> None:
        """Show/hide the run-time warning naming any situational extract toggle that differs from
        its default: in either direction and showing its current state, so the check stays correct
        even if a default is later flipped. Smart tag reading ignores the situational
        toggles entirely, so a stale value can't corrupt the run: no warning then."""
        if self._extract_opts.get("smart_tags"):
            self._opts_warning_row.setVisible(False)
            return
        active = [f"{label} ({'on' if self._extract_opts[key] else 'off'})"
                  for key, label in SITUATIONAL_EXTRACT_OPTS.items()
                  if self._extract_opts[key] != EXTRACT_OPT_DEFAULTS[key]]
        if active:
            self._opts_warning.setText("⚠ Non-default option selected: " + ", ".join(active))
        self._opts_warning_row.setVisible(bool(active))

    def _clear_situational_opts(self) -> None:
        """Reset the situational toggles to their defaults (the warning's one-click fix)."""
        for key in SITUATIONAL_EXTRACT_OPTS:
            self._extract_opts[key] = EXTRACT_OPT_DEFAULTS[key]
        self._save_extract_opts()
        self._refresh_opts_warning()

    # ── Run handlers ──────────────────────────────────────────────────────────
    def _on_choose(self) -> None:
        # Open to the user's configured default VOD folder when set; otherwise the project root
        # in a source checkout, or wherever the picker last landed on a bundled build.
        configured = self._default_vod_dir()
        if configured and not Path(configured).is_dir():
            # Configured but gone (renamed, deleted, or on an unplugged drive): say so instead
            # of silently opening the picker somewhere unexpected.
            QMessageBox.warning(
                self, "Default VOD folder not found",
                f"The default VOD folder no longer exists:\n\n{configured}\n\n"
                "You can update it in Settings")
            configured = ""
        if configured:
            start_dir = configured
        else:
            fallback = vod_picker_start_dir()
            start_dir = str(fallback) if fallback.is_dir() else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose input video", start_dir, VIDEO_FILTER
        )
        if not path:
            return
        self.choose_video(path)

    def choose_video(self, path: str) -> None:
        """Take ``path`` as the chosen video, exactly as Choose video… does once its dialog
        returns: a container the player can't open is offered for conversion (as is an MKV
        on Windows, once the offer is turned on in Settings), a file the analyzer can't
        decode for re-encoding, anything else is adopted at once. Public for the ``--video``
        launch option, which skips the dialog."""
        if offers_conversion(path, ask_for_mkv=self._ask_convert_mkv()):
            self._converter.offer_convert(path)   # finishes the selection via fileReady once resolved
        elif not opencv_can_read(path):
            # Fail at selection time instead of minutes into a run: the pipeline reads frames
            # through the same cv2.VideoCapture path, so a file this probe rejects would only
            # die later inside Analyze VOD. The offer's re-encode rescues the common cause
            # (HEVC/10-bit) and finishes the selection via fileReady once resolved.
            self._converter.offer_reencode(path)
        else:
            self._finish_selection(path)

    def _finish_selection(self, path: str) -> None:
        """Adopt `path` as the current vod and refresh the Main page for it. Refused
        while a run is in flight (see _set_running): Choose video… is locked then, but
        an MP4 conversion started before the run can still finish during it.

        The MP4 made from the current vod (File → "Convert to MP4…", or a conversion
        offer for the same file chosen again) is the same VOD, so the tournament name
        seeded or typed for it carries over rather than being reseeded from the new
        file name."""
        if self._running:
            self._append(f"\n> {Path(path).name} is ready, but a run is in progress. "
                         "Choose it again once the run ends.")
            return
        current = self._video_path
        if (current is not None and self._tournament_prefilled_for == current
                and is_mp4_conversion_of(path, current)):
            self._tournament_prefilled_for = path   # _seed_tournament_name keeps the text
        self._video_path = path
        self._file_label.setText(Path(path).name)
        self._seed_tournament_name()  # once per vod; survives a later return from Review
        self._load_video_into_preview(path)
        self._refresh_nav()

    def _confirm_overwrite_analysis(self) -> bool:
        """Before a full re-analysis (Analyze VOD), confirm overwriting any existing analysis and/or
        discarding saved edits for the current vod. Returns True to proceed; nothing to overwrite
        means no prompt and True."""
        video = self._video_path
        if video is None:
            return False
        has_analysis = pipeline_json_for(video).exists()
        has_draft = review_draft_for(video).exists()
        if not has_analysis and not has_draft:
            return True
        lines = [f"Re-analyzing {Path(video).name} will:"]
        if has_analysis:
            lines.append("• overwrite the saved VOD analysis")
        if has_draft:
            lines.append("• discard your saved Review edits (you can use “Resume edits” instead to reopen them)")
        lines.append("\nContinue?")
        reply = QMessageBox.question(self, "Reanalyze this video?", "\n".join(lines))
        return reply == QMessageBox.StandardButton.Yes

    def _confirm_reread_discard(self) -> bool:
        """Confirm discarding Review edits before a Re-read tags regeneration (the shared
        review_tools.confirm_reread_discard: asks only when there are edits to lose)."""
        video = self._video_path
        if video is None:
            return False
        return confirm_reread_discard(self, self._review, vod_stem(video), Path(video).name)

    def _on_ocr_engine_selected(self) -> None:
        """The Text-engine combo changed: the Preview box reads with the new choice at
        once (the setting itself is saved only at Run). Choosing docTR without its package
        says how to install it; choosing it before its models are downloaded offers the
        one-time download right away rather than surprising the user at Run. Either way
        the combo stays on Stylized: until docTR can read, the Preview says why it can't
        and Analyze VOD asks again, but nothing reads with Standard in its place."""
        self._preview_pane.set_engine(self._ocr_engine_combo.currentData())
        self._request_side_overlay()   # the rows are found with the Font style's engine
        if self._ocr_engine_combo.currentData() != OCR_ENGINE_DOCTR:
            return
        if not doctr_available():
            QMessageBox.warning(self, f"{FONT_STYLE_STYLIZED} isn't installed",
                                DOCTR_NOT_INSTALLED_MESSAGE)
            return
        if not doctr_weights_cached():
            self._offer_doctr_fetch(self._on_doctr_offer_done)

    def _on_doctr_offer_done(self, downloaded: bool) -> None:
        """The download offered on choosing docTR ended: a completed one hands the choice
        to the Preview box again now that it can read with it. A declined, cancelled or
        failed one changes nothing (see _on_ocr_engine_selected's docstring)."""
        if downloaded:
            self._preview_pane.set_engine(self._ocr_engine_combo.currentData())
            self._request_side_overlay()

    def _refresh_engine_note(self) -> None:
        """Describe the selected font style in the muted note under the combo, unless the notes
        are turned off in Settings → Analysis (in which case neither style shows one)."""
        if not self._settings.value(SETTING_SHOW_FONT_STYLE_NOTE, True, type=bool):
            self._engine_note.setVisible(False)
            return
        is_stylized = self._ocr_engine_combo.currentData() == OCR_ENGINE_DOCTR
        self._engine_note.setText(
            FONT_STYLE_NOTE_STYLIZED if is_stylized else FONT_STYLE_NOTE_STANDARD)
        self._engine_note.setVisible(True)

    def _refresh_lang_note(self) -> None:
        """Name the selected overlay language while a non-default (non-English) one is chosen in
        Preferences; English (the default) shows nothing."""
        lang = self._settings.value(SETTING_OCR_LANG, OCR_LANG_DEFAULT_GUI, type=str)
        is_default = lang == OCR_LANG_DEFAULT_GUI
        label = lang_display_name(lang)
        self._lang_note.setText("" if is_default else f"The overlay language is set to {label}.")
        self._lang_note.setVisible(not is_default)

    def changeEvent(self, a0: QEvent | None) -> None:
        """Follow a live light/dark switch. Qt re-colours ordinary widgets from the new
        palette by itself, but the notes restyled here carry an explicit colour in a style
        sheet, resolved when it was set; without this a light-to-dark switch leaves the font
        style note in light mode's dark grey on the dark background."""
        if a0 is not None and a0.type() == QEvent.Type.PaletteChange:
            self._restyle_themed_notes()
        super().changeEvent(a0)

    def _restyle_themed_notes(self) -> None:
        """Re-resolve every main-window note's theme colour (muted grey, notice teal, warning
        amber) against the current palette. Safe to repeat: setting an unchanged style sheet
        on a child label sends no PaletteChange back to this window."""
        # A PaletteChange can arrive while __init__ is still building the pages; the title
        # note is the last of these labels to exist, so until then there is nothing to do.
        if getattr(self, "_title_note", None) is None:
            return
        self._update_title(self._stack.currentIndex())   # the title note re-resolves there
        _colour_note(self._engine_note, muted_text_color())
        _colour_note(self._lang_note, notice_color())
        _colour_note(self._export_mode_note, notice_color())
        _colour_note(self._opts_warning, warning_color(), bold=True)
        self._refresh_game_note()   # notice or warning colour, by what the note says

    def _selected_game(self) -> Game:
        """The registry entry for the Game combo's current selection. The combo only ever
        holds registry ids (see _populate_game_combo), so the default is unreachable in
        practice; it keeps the callers below total rather than Optional."""
        return game_by_id(self._game_combo.currentData()) or default_game()

    def _add_overlay_layout_new_item(self) -> None:
        """Append the OVERLAY_LAYOUT_NEW_ITEM action item to the Overlay layout combo,
        flagged in OVERLAY_LAYOUT_NEW_ROLE, under a separator when there are choices above
        it. Its data is ``None``, so the rebuild's restore (findData of a saved name) never
        lands on it, and _on_run stops on it when it is all a game has; picking it opens
        the editor instead of becoming the selection.

        With nothing above it (a game with no Auto and no preset yet) it gets no
        separator: it is then item 0, the item every fallback to the first item lands on."""
        combo = self._overlay_layout_combo
        if combo.count() > 0:
            combo.insertSeparator(combo.count())
        combo.addItem(OVERLAY_LAYOUT_NEW_ITEM, None)
        combo.setItemData(combo.count() - 1, True, OVERLAY_LAYOUT_NEW_ROLE)

    def _populate_overlay_layout_combo(self) -> None:
        """(Re)build the Overlay layout combo for the game the Game combo currently shows:
        "Auto" (item data ``""``, meaning the built-in overlay reading) followed by one item
        per saved preset of that game, name as both text and data, then a separator and the
        "New layout…" action item (see _add_overlay_layout_new_item).

        Auto is offered only when the game ships a clock detector (every game but Other
        today), since the built-in reading has no other clock source; a game without one
        that has no preset yet therefore has nothing to list at all, and shows "New
        layout…" as its only item, current. Called on first build, whenever the Game combo
        changes, after the editor closes (a preset may have been added/renamed/deleted) and
        after the Settings dialog closes ("Reset all preferences" clears the keys).

        The selection to restore is always read from that game's own QSettings key
        (overlay_layout_setting_key): the key is the single source of truth, written on every
        change by _on_overlay_layout_selected, and one key per game means switching games and
        back restores each game's own choice. A saved name that is no longer in the list (the
        preset was deleted, or it was saved for another game) falls back to the first item —
        Auto, else the first preset, else "New layout…" — and the write-back at the end
        records that fallback so the key and the combo never disagree; after Reset-all this
        recreates the key with "" on purpose. Signals are blocked while rebuilding because
        QComboBox.clear() emits currentIndexChanged(-1), which would otherwise persist None
        over the user's choice mid-repopulate."""
        game = self._selected_game()
        saved = self._settings.value(
            overlay_layout_setting_key(game.id), OVERLAY_LAYOUT_DEFAULT, type=str)
        combo = self._overlay_layout_combo
        combo.blockSignals(True)
        combo.clear()
        if game.has_timer_model:
            combo.addItem(AUTO_LAYOUT_NAME, OVERLAY_LAYOUT_DEFAULT)
        for layout in layouts_for_game(load_layouts(layouts_path()), game.id):
            combo.addItem(layout.name, layout.name)
        self._add_overlay_layout_new_item()
        # The closed combo cuts a very long name short (see OVERLAY_LAYOUT_COMBO_MAX_CHARS);
        # its list shows every name in full.
        widen_popup_to_fit_items(combo)
        idx = combo.findData(saved)
        # Item 0: Auto, else the first preset, else "New layout…" (then the only item).
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)
        self._on_overlay_layout_selected()  # write back a deleted-preset fallback (key == combo)

    def _on_overlay_layout_selected(self) -> None:
        """Persist the Overlay layout choice for the selected game as soon as it changes.
        Unlike the cadence and font-style combos (written in _on_run), this value is also read
        by the reuse-analysis warning and cleared by "Reset all preferences", and the Settings
        dialog used to persist it on OK; writing on change keeps every reader on one value.

        "New layout…" (data ``None``) is never written: it is not a choice. While it is a
        game's only item (no Auto, no preset yet) it is simply what the combo shows, and the
        game keeps whatever it had saved, ready for the moment a preset for it exists.

        Where there are choices above it, landing on "New layout…" is either a pick in the
        popup, which _on_overlay_layout_activated (connected to ``activated``, emitted right
        after this) answers by opening the editor, or an arrow key, the mouse wheel or
        type-ahead on the closed combo (Qt skips the separator above it). That is not a
        request for the editor, so the combo goes back to the game's saved choice instead."""
        combo = self._overlay_layout_combo
        on_new_layout = bool(combo.currentData(OVERLAY_LAYOUT_NEW_ROLE))
        new_layout_is_the_only_item = combo.count() == 1
        if on_new_layout and not new_layout_is_the_only_item:
            if not combo.picked_from_popup():
                self._restore_overlay_layout_choice()
            return
        choice = combo.currentData()
        if choice is not None:
            self._settings.setValue(overlay_layout_setting_key(self._selected_game().id), choice)
        # Every path that changes what the run would read with ends here: a layout pick, a
        # game change, the editor closing and a Settings reset all funnel through
        # _populate_overlay_layout_combo, which calls this last.
        self._refresh_layout_preview()

    def _on_overlay_layout_activated(self, index: int) -> None:
        """Open the layout editor on a new layout when "New layout…" is picked in the
        Overlay layout combo's popup. Connected to ``activated``, which Qt emits after every
        pick in the popup, also of the item already current: that is how a game whose only
        item is "New layout…" (no Auto, no preset yet) opens the editor at all, since
        picking the current item changes no index. Only a pick in the popup counts
        (picked_from_popup): ``activated`` also follows an arrow key, the wheel or
        type-ahead, and those never open the editor.

        Picking it writes nothing: the combo is rebuilt when the editor closes (see
        _open_layout_editor), showing the layout saved there, else the game's previous
        choice again ("New layout…" itself for a game that still has nothing else)."""
        combo = self._overlay_layout_combo
        if combo.itemData(index, OVERLAY_LAYOUT_NEW_ROLE) and combo.picked_from_popup():
            self._open_layout_editor(initial_layout=None)

    def _restore_overlay_layout_choice(self) -> None:
        """Put the Overlay layout combo back on the selected game's saved choice (the key
        _on_overlay_layout_selected writes on every real change), silently: nothing about
        what the run reads with has changed. A saved name no longer listed falls back to
        the first item, as in _populate_overlay_layout_combo."""
        combo = self._overlay_layout_combo
        saved = self._settings.value(
            overlay_layout_setting_key(self._selected_game().id), OVERLAY_LAYOUT_DEFAULT, type=str)
        idx = combo.findData(saved)
        combo.blockSignals(True)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _load_video_into_preview(self, path: str) -> None:
        """Open the newly chosen ``path`` in the Preview box and ask for its clock. The
        clock search is cancelled first: the preview forgets its clock on every set_video,
        so a request for the pair already asked for (the same video chosen again) must be
        delivered again (from the cache, or searched afresh after a failure) rather than
        skipped as a repeat. The note under the Game combo is refreshed too: a wrong-game
        doubt raised for the previous video says nothing about this one."""
        self._clock_locator.cancel()
        self._side_overlay_locator.cancel()
        self._preview_pane.set_video(path)
        self._refresh_layout_preview()
        self._refresh_game_note()

    def _refresh_layout_preview(self) -> None:
        """Tell the Preview box what the run would read with (the selected game, the
        selected overlay layout resolved to its preset, the in-game band widening, the
        Font style) and ask
        the clock search for the video's clock under that game (not while a run is in
        flight, see _request_clock). A no-op until a video is chosen; _finish_selection
        calls it once there is one. "New layout…" as a game's only item (no layout to
        choose) previews like Auto: the run itself refuses to start there.

        The preset is resolved the way _on_run resolves it (a name no longer in the store
        reads as no layout; _on_run says so when the run is started)."""
        if self._video_path is None:
            return
        game = self._selected_game()
        layout = self._selected_layout()
        ingame_tag_frac = self._settings.value(
            SETTING_INGAME_TAG_BAND_FRAC, INGAME_TAG_BAND_FRAC_DEFAULT, type=float)
        self._preview_pane.set_engine(self._ocr_engine_combo.currentData())
        self._preview_pane.set_context(game, layout, ingame_tag_frac)
        self._request_clock()

    def _request_clock(self) -> None:
        """Ask the clock search for the chosen video's clock under the selected game. A
        no-op with no video, while a run is in flight, off the main page, while the
        window quits, and while the layout preview is turned off in Settings (the box
        shows a still then, see _show_layout_preview). A pipeline run cancels the search
        (no torch sidecar competing with
        it), and it is asked for again where the Preview box comes back into view: a failed
        or stopped run, or a Review that failed to load, leaves the main page showing
        (_on_pipeline_finished, _on_extract_finished, _consume_stop), and a finished one
        goes to Review, whose Back button returns here (_goto_main). Asking at the end of
        every run instead would start a torch search while Review loads."""
        video = self._video_path
        if (video is None or self._running or self._shutting_down
                or self._stack.currentIndex() != 0   # 0: the main page
                or not self._show_layout_preview()):
            return
        self._clock_locator.request(video, self._selected_game())
        self._request_side_overlay()

    def _selected_layout(self) -> CustomOverlayLayout | None:
        """The Overlay layout choice resolved to its preset: None for Auto, for "New
        layout…" (a game's only item while it has no preset), and for a name no longer in
        the store."""
        layout_name = self._overlay_layout_combo.currentData()
        return layout_by_name(load_layouts(layouts_path()), layout_name) if layout_name else None

    def _preview_ocr(self) -> tuple[str, str]:
        """The OCR engine and language the Preview box's searches read with: the Font style
        choice as it stands (a search refuses, saying why, while docTR can't read yet) and
        the saved overlay language."""
        engine = self._ocr_engine_combo.currentData()
        lang = self._settings.value(SETTING_OCR_LANG, OCR_LANG_DEFAULT_GUI, type=str)
        return engine, lang

    def _on_clock_located(self, result: ClockLocateResult | None) -> None:
        """The clock search answered: hand the answer to the Preview box, refresh the note
        under the Game combo (no steady clock raises the wrong-game doubt there),
        then ask for the player-name rows, whose bars are placed from the clock's box."""
        self._preview_pane.set_clock(result)
        self._refresh_game_note()
        self._request_side_overlay()

    def _wrong_game_suspected(self) -> bool:
        """Whether the clock search for the chosen video and game has answered that it
        found no steady clock: the detector fired on none of its sampled frames, or fired
        but calibration refused a box (the hits disagreed, or there were too few to trust:
        what another game's clock tends to produce, and what a VOD with little game
        footage among the samples can produce too, so the question this raises is only a
        confirm). Those are the outcomes that point at the wrong game being selected. A
        search still running, a failed one (weights missing), a game without a detector,
        and a steady clock that could not be read are all False: those say something else.
        So is the layout preview being off: an answer found while it was on is not used
        (see _show_layout_preview)."""
        video = self._video_path
        if video is None or not self._show_layout_preview():
            return False
        game = self._selected_game()
        if not game.has_timer_model:
            return False
        answered, clock = self._clock_locator.answer_for(video, game)
        if not answered or clock is None or clock.error is not None:
            return False
        never_fired = clock.first_hit_sec is None
        fired_without_a_steady_spot = clock.bbox is None
        return never_fired or fired_without_a_steady_spot

    def _confirm_no_clock(self) -> bool:
        """Ask whether to analyze as the selected game although no steady clock of it was
        found (GAME_CONFIRM_NO_CLOCK); True to go ahead. A Yes is remembered for this
        video and game for the session: the doubt stands until the search finds a clock,
        and a run that failed or was stopped should not raise the same question again."""
        assert self._video_path is not None   # _on_run returns before this without a video
        key = (self._video_path, self._selected_game().id)
        if key in self._no_clock_confirmed:
            return True
        reply = QMessageBox.question(
            self, "Wrong game?",
            GAME_CONFIRM_NO_CLOCK.format(game=self._selected_game().display_name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return False
        self._no_clock_confirmed.add(key)
        return True

    def _request_side_overlay(self) -> None:
        """Ask the player-name search for the rows an Auto run would lock on the chosen
        video (see side_overlay_locate.py). A no-op in the situations _request_clock is,
        and until the clock search has answered for this video and game (its answer
        arrives through _on_clock_located, which asks again). Where the search does not
        apply (a game with no side bars, or a custom overlay layout, whose own boxes say
        where the names are) one in flight is stopped and its progress text taken off
        the Preview box (a cancelled search answers nothing, so nothing else would);
        the box ignores the rows there itself."""
        video = self._video_path
        if (video is None or self._running or self._shutting_down
                or self._stack.currentIndex() != 0   # 0: the main page
                or not self._show_layout_preview()):
            return
        game = self._selected_game()
        if not game.has_side_bars or self._selected_layout() is not None:
            if self._side_overlay_locator.cancel():
                self._preview_pane.set_side_overlay_status("")
            return
        answered, clock = self._clock_locator.answer_for(video, game)
        if not answered:
            return
        if clock is not None and clock.error is None and clock.first_hit_sec is None:
            # The clock detector fired on none of the clock search's frames. The name search
            # reads only frames with a clock, so it would read its frames for nothing:
            # answer for it instead (the run skips its pre-pass the same way), with nothing
            # to add to the clock search's own "No clock was found" on the status line.
            self._side_overlay_locator.cancel()
            self._preview_pane.set_side_overlay(SideOverlayLocateResult.not_searched())
            return
        engine, lang = self._preview_ocr()
        self._side_overlay_locator.request(video, game, engine, lang, _calibrated_bbox(clock))

    def _on_side_overlay_searching(self, text: str) -> None:
        """A player-name search is running: whatever rows the Preview box drew were found
        for something else (another Font style, another clock box), so they are dropped
        while the progress text shows."""
        self._preview_pane.set_side_overlay(None)
        self._preview_pane.set_side_overlay_status(text)

    def _write_prepass_handover(self, video: str, game: Game, ocr_engine: str, ocr_lang: str,
                                layout: CustomOverlayLayout | None) -> Path | None:
        """Write what the Preview box's searches already found for this run (the calibrated
        clock, the player-name rows) to the handover file, and return its path for the
        pipeline argv; None when there is nothing to hand over or the file could not be
        written, and the run then searches for itself (see vodparse.prepass_handover). The
        rows are looked up under the run's own Font style and language, so rows found
        with another engine are never handed over. Nor is anything with the layout
        preview off: an answer cached while it was on is not used (see
        _show_layout_preview), and the run searches as it did before the preview existed."""
        if not self._show_layout_preview():
            return None
        answered, clock = self._clock_locator.answer_for(video, game)
        if not answered:
            return None
        calibration = clock.calibration if clock is not None and clock.error is None else None
        # The search ran and calibrated nothing: the run skips its own pre-pass, which
        # would sample the same frames by the same rules (None is a game with no detector,
        # which has no search to hand over).
        clock_searched = clock is not None and clock.error is None and calibration is None
        # ...and the detector fired on none of its frames: the run skips its side-bar
        # pre-pass too, as the Preview box skipped its own name search.
        clock_never_fired = clock_searched and clock is not None and clock.first_hit_sec is None
        side_record = None
        if layout is None and game.has_side_bars:
            side = self._side_overlay_locator.answer_for(
                video, game, ocr_engine, ocr_lang, _calibrated_bbox(clock))
            if side is not None and side.record is not None:
                side_record = side.record.config
        handover = PrepassHandover(video=video, game=game.id, clock_calibration=calibration,
                                   clock_searched=clock_searched,
                                   clock_never_fired=clock_never_fired,
                                   side_overlay_record=side_record,
                                   ocr_engine=ocr_engine, ocr_lang=ocr_lang)
        if handover.is_empty:
            return None
        path = self._preview_pane.progress_file.with_name(PREPASS_HANDOVER_FILE_NAME)
        return path if write_handover(path, handover) else None

    def _on_edit_layouts(self) -> None:
        """The Edit… button beside the Overlay layout combo: open the editor on that
        choice (see _open_layout_editor). Auto's "" and the None of "New layout…" (a
        game's only item while it has no preset) both open on a new layout."""
        self._open_layout_editor(initial_layout=self._overlay_layout_combo.currentData())

    def _open_layout_editor(self, initial_layout: str | None) -> None:
        """Open the overlay-layout editor already scrubbing the chosen VOD (if any) at the
        Preview box's playhead, on the main screen's game and on ``initial_layout`` (a
        preset name; None or "" opens on a new layout), then rebuild the combo so a preset added/renamed/deleted in the editor
        shows up. A preset saved in the editor (new or edited) becomes its game's selection:
        it is what the user means to run with next, and the chooser would otherwise keep
        that game's earlier pick (which is what it does when the editor closes without
        saving)."""
        # Imported lazily: OverlayLayoutEditorDialog pulls in cv2 and QtMultimedia, which are
        # slow to import and unwanted on every launch, so it's loaded only when the user
        # actually opens the editor (matches the repo's lazy-import idiom elsewhere).
        from vodparse.gui.overlay_editor import OverlayLayoutEditorDialog
        # The editor opens on the main screen's game (its own dropdowns can switch from
        # there without touching the choices here).
        # The editor's scrubber opens where the Preview box's stands, so the frame on
        # screen is the one the boxes get drawn over. The video is the one this window
        # already settled (past its own conversion offer), so the editor makes no second
        # offer for it.
        editor = OverlayLayoutEditorDialog(self._settings, self, initial_video=self._video_path,
                                           initial_video_resolved=True,
                                           initial_sec=self._preview_pane.playhead_sec(),
                                           initial_game=self._game_combo.currentData(),
                                           initial_layout=initial_layout)
        editor.exec()
        saved = editor.last_saved_layout
        if saved is not None:
            # Written to the saved preset's own game key: the rebuild below shows it when
            # that is the game on screen, and switching to that game later shows it too.
            self._settings.setValue(overlay_layout_setting_key(saved.game), saved.name)
        self._populate_overlay_layout_combo()  # pick up any preset added/renamed/deleted

    def _populate_game_combo(self) -> None:
        """(Re)build the Game combo from the registry (vodparse.games.GAMES), display name as
        text and id as data, and preselect the saved id (an unknown one falls back to the
        default, see resolve_saved_game). Called on first build and after the Settings dialog
        closes ("Reset all preferences" clears the key). Signals are blocked while rebuilding
        for the same reason as _populate_overlay_layout_combo; the write-back at the end keeps
        the key and the combo on one value, refreshes the note under the combo and fills the
        Overlay layout combo for the restored game."""
        saved = self._settings.value(SETTING_GAME, GAME_DEFAULT, type=str)
        combo = self._game_combo
        combo.blockSignals(True)
        combo.clear()
        for game in GAMES:
            combo.addItem(game.display_name, game.id)
        combo.setCurrentIndex(combo.findData(resolve_saved_game(saved)))
        combo.blockSignals(False)
        self._on_game_selected()

    def select_game(self, game_id: str) -> None:
        """Select the game with registry id ``game_id`` in the Game combo, as a user pick
        would (it is saved, and the note and Overlay layout combo follow). Public for the
        ``--game`` launch option; an id the registry doesn't hold raises ValueError."""
        index = self._game_combo.findData(game_id)
        if index < 0:
            raise ValueError(f"unknown game id {game_id!r}")
        self._game_combo.setCurrentIndex(index)

    def _on_game_selected(self) -> None:
        """Persist the Game choice as soon as it changes, refresh its note, and rebuild the
        Overlay layout combo: which presets it lists, and whether it offers Auto at all, are
        both per game."""
        self._settings.setValue(SETTING_GAME, self._game_combo.currentData())
        self._refresh_game_note()
        self._populate_overlay_layout_combo()

    def _refresh_game_note(self) -> None:
        """The note under the Game combo: the no-clock-detector note
        (GAME_NOTE_NO_TIMER_MODEL, notice colour) while a game without a timer model of its
        own is selected; the wrong-game doubt (GAME_NOTE_NO_CLOCK, warning colour) while the
        clock search has found no steady clock for the selected game on the chosen video; nothing
        otherwise. Refreshed on a game change, on every clock answer and on every video
        choice."""
        game = self._selected_game()
        if not game.has_timer_model:
            _colour_note(self._game_note, notice_color())
            self._game_note.setText(GAME_NOTE_NO_TIMER_MODEL.format(game=game.display_name))
            self._game_note.setVisible(True)
            return
        if self._wrong_game_suspected():
            _colour_note(self._game_note, warning_color())
            self._game_note.setText(GAME_NOTE_NO_CLOCK)
            self._game_note.setVisible(True)
            return
        self._game_note.setVisible(False)

    def _offer_doctr_fetch(self, on_done: Callable[[bool], None]) -> None:
        """Ask before the one-time docTR model download, then run it. ``on_done`` receives
        True only when the models are in the cache afterwards: False on decline, cancel,
        or failure: so callers can revert the engine choice or continue into a run."""
        reply = QMessageBox.question(
            self, "Download text models?",
            f"{FONT_STYLE_STYLIZED} needs to download text models (docTR) before it can be used. "
            f"Download them now? (~{DOCTR_WEIGHTS_MB_APPROX} MB)")
        if reply != QMessageBox.StandardButton.Yes:
            on_done(False)
            return
        self._start_doctr_fetch(on_done)

    def _offer_band_engine_fetch(self, game: Game, run: Callable[[bool], None]) -> None:
        """Ask before a run whose game reads its nameplate band with docTR (Rivals 2) while
        the Font style is another engine and docTR's models are not downloaded. Three
        answers: Yes downloads them, then runs with docTR reading the band; No runs at once
        with the band read by the user's own engine; Cancel runs nothing. ``run`` receives
        whether the band falls back to the user's engine. A download that fails or is
        cancelled runs nothing either: pressing Analyze VOD again asks again."""
        reply = QMessageBox.question(
            self, "Download text models?",
            f"{game.display_name} needs an additional text model (docTR) to detect "
            f"characters (~{DOCTR_WEIGHTS_MB_APPROX} MB).\n\n"
            "Download it now? (\"No\" will attempt to run the analysis with the default "
            "models).",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes)
        if reply == QMessageBox.StandardButton.Yes:
            self._start_doctr_fetch(lambda ok: run(False) if ok else None)
        elif reply == QMessageBox.StandardButton.No:
            run(True)

    def _start_doctr_fetch(self, on_done: Callable[[bool], None]) -> None:
        """Run ``vodparse.cli.doctr_support fetch`` on a QProcess behind a modal, cancelable
        progress dialog: indeterminate, since docTR's downloader reports no byte progress.
        A failure (not a user cancel) pops a warning with the downloader output's tail;
        either way ``on_done`` gets whether the models are now in the cache."""
        # Constructing the BusyDialog shows it, which has to happen before start() below:
        # errorOccurred can fire synchronously inside start(), running finish (which closes and
        # deleteLater()s the dialog): a show() after that would flash the dead dialog again
        # once the failure warning is dismissed.
        dialog = BusyDialog(
            f"Downloading the docTR text models (~{DOCTR_WEIGHTS_MB_APPROX} MB)…",
            "Downloading text models", self)

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.setProcessEnvironment(stage_environment())
        fired = False  # finished + errorOccurred can both arrive; run the completion once

        def on_cancel() -> None:
            proc.kill()  # → finished fires and the completion below runs

        def finish(code: int, status: QProcess.ExitStatus) -> None:
            nonlocal fired
            if fired:
                return
            fired = True
            # A programmatic close, which BusyDialog keeps from reading as a user cancel: left
            # to itself QProgressDialog emits canceled() from its own closeEvent, and a
            # finished download would then be reported as a cancel.
            dialog.close_quietly()
            cancelled = dialog.was_cancelled
            output = proc.readAllStandardOutput().data().decode("utf-8", "ignore")
            proc.deleteLater()
            ok = (not cancelled and code == 0
                  and status == QProcess.ExitStatus.NormalExit and doctr_weights_cached())
            if not ok and not cancelled:
                tail = "\n".join(output.strip().splitlines()[-8:])
                QMessageBox.warning(
                    self, "Text-model download failed",
                    "The docTR model download didn't complete.\n\n"
                    f"{tail or 'The downloader exited with an error.'}")
            on_done(ok)

        dialog.cancelled.connect(on_cancel)
        proc.finished.connect(finish)
        # FailedToStart never gets a finished signal: without this the dialog would hang open.
        proc.errorOccurred.connect(
            lambda err: finish(-1, QProcess.ExitStatus.CrashExit)
            if err == QProcess.ProcessError.FailedToStart else None)
        proc.start(sys.executable, ["-m", "vodparse.cli.doctr_support", "fetch"])

    def _on_run(self) -> None:
        """Analyze VOD: run the pipeline, then (on success) extract sets and open Review. The
        wrong-game doubt is raised first (there is no point agreeing to overwrite an analysis
        for a run the user then declines), then the one confirm that covers overwriting an
        existing analysis and/or saved edits; after that the pipeline→extract→Review chain
        runs unattended. The argv itself is built by commands.pipeline_argv; this method just
        resolves the widget/QSettings state it needs."""
        video = self._video_path
        if video is None:
            return
        if self._wrong_game_suspected() and not self._confirm_no_clock():
            return
        if not self._confirm_overwrite_analysis():
            return

        ocr_engine = self._ocr_engine_combo.currentData()
        ocr_lang = self._settings.value(SETTING_OCR_LANG, OCR_LANG_DEFAULT_GUI, type=str)
        if ocr_engine == OCR_ENGINE_DOCTR and ocr_lang != OCR_LANG_DEFAULT_GUI:
            # The pipeline's own validate_engine_lang would stop the subprocess anyway;
            # catching it here turns a failed run into an actionable dialog.
            QMessageBox.warning(
                self, f"{FONT_STYLE_STYLIZED} reads English only",
                f"The {FONT_STYLE_STYLIZED} font style reads English/Latin text only. Switch "
                "the overlay language back to English in Settings → Analysis, or set the Font style "
                f"to {FONT_STYLE_STANDARD}.")
            return

        overlay_layout = self._overlay_layout_combo.currentData()
        if overlay_layout is None:
            # "New layout…" as the only item: the selected game has no clock detector and no
            # preset saved for it yet, so this run would have no clock at all and nothing to
            # split on. The pipeline refuses it too; catch it here with a dialog that says what
            # to do.
            QMessageBox.warning(
                self, "Overlay layout needed",
                f"{self._selected_game().display_name} needs a custom overlay layout with a "
                "timer box. Click \"Edit\" to create one.")
            return

        layout = (layout_by_name(load_layouts(layouts_path()), overlay_layout)
                  if overlay_layout else None)
        if overlay_layout and layout is None:
            # The preset was renamed or deleted since it was selected (the store can change on
            # disk while the app runs); the pipeline itself would refuse it too, so catch it
            # here with an actionable dialog instead of a failed run.
            QMessageBox.warning(
                self, "Overlay layout not found",
                f"The overlay layout {overlay_layout!r} no longer exists. Pick another one "
                "from the Overlay layout list, or click \"Edit\" to recreate it.")
            self._populate_overlay_layout_combo()  # fall back to Auto now that we've said so
            return

        # The game's own nameplate band may pin docTR whatever the Font style says (Rivals 2
        # reads its character names with it: games.IngameHud.ocr_engine).
        game = self._selected_game()
        band_engine = ingame_band_engine_for(game, ingame_band_mode_for(game, layout))
        needs_doctr = OCR_ENGINE_DOCTR in (ocr_engine, band_engine)
        if needs_doctr and not doctr_available():
            # Nothing reads with Standard in docTR's place: without the package the run
            # can't do what was asked, so say how to fix it rather than run with another engine.
            why = (DOCTR_NOT_INSTALLED_MESSAGE if ocr_engine == OCR_ENGINE_DOCTR else
                   f"{game.display_name} reads its nameplates with {FONT_STYLE_STYLIZED}. "
                   f"{DOCTR_NOT_INSTALLED_MESSAGE}")
            QMessageBox.warning(self, f"{FONT_STYLE_STYLIZED} isn't installed", why)
            return

        self._clear_log()
        ocr_interval = self._ocr_interval_combo.currentData()
        self._settings.setValue(SETTING_OCR_INTERVAL_SEC, ocr_interval)  # remember for next run
        self._settings.setValue(SETTING_OCR_ENGINE, ocr_engine)          # remember for next run
        # The overlay layout is not written here: _on_overlay_layout_selected persists it on change.
        # The OCR speed-ups and the in-game nametag band live in Preferences (SettingsDialog);
        # read saved values. The band frac rides only on the pipeline argv: it is baked into the
        # pipeline JSON, which extract_matches reads back (see resolve_ingame_tag_band_frac).
        gate_ocr = self._settings.value(SETTING_GATE_OCR_ON_TIMER, GATE_OCR_ON_TIMER_DEFAULT,
                                        type=bool)
        roi_cache = self._settings.value(SETTING_OCR_ROI_CACHE, OCR_ROI_CACHE_DEFAULT, type=bool)
        ingame_tag_frac = self._settings.value(
            SETTING_INGAME_TAG_BAND_FRAC, INGAME_TAG_BAND_FRAC_DEFAULT, type=float)
        game_id = self._game_combo.currentData()   # read now: the argv may be built after a download

        def run_argv(band_on_overlay_engine: bool = False) -> list[str]:
            """This run's pipeline argv; see commands.pipeline_argv for the flag. Built
            when the run starts (after a model download, when there is one), so the
            handover carries what the Preview box's searches have found by then."""
            return pipeline_argv(
                video, ocr_interval=ocr_interval, gate_ocr=gate_ocr, roi_cache=roi_cache,
                ocr_lang=ocr_lang, ocr_engine=ocr_engine, ingame_tag_frac=ingame_tag_frac,
                game=game_id, overlay_layout=overlay_layout,
                band_on_overlay_engine=band_on_overlay_engine,
                preview_file=self._preview_pane.preview_file,
                progress_file=self._preview_pane.progress_file,
                prepass_handover=self._write_prepass_handover(
                    video, self._selected_game(), ocr_engine, ocr_lang, layout),
            )

        if needs_doctr and not doctr_weights_cached():
            # Stylized without its models: chosen with the download declined, saved from a
            # session that had them, or only the game's band wanting docTR (then this is the
            # one place the offer can happen). Asked again on every Analyze until downloaded.
            if ocr_engine == OCR_ENGINE_DOCTR:
                # The user's own Font style is docTR: without its models there is nothing to
                # read the overlay with, so a declined (or failed) download runs nothing.
                self._offer_doctr_fetch(
                    lambda ok: self._start_pipeline_run(run_argv()) if ok else None)
            else:
                # Only the game's band wants docTR: the user may also run without it, the band
                # then read with their own engine (--ingame-band-engine overlay).
                self._offer_band_engine_fetch(
                    game, lambda on_overlay: self._start_pipeline_run(
                        run_argv(band_on_overlay_engine=on_overlay)))
            return
        self._start_pipeline_run(run_argv())

    def _start_pipeline_run(self, argv: list[str]) -> None:
        """Launch the pipeline stage: the tail of _on_run, split out so a docTR model
        download can finish first and then continue straight into the run."""
        self._clock_locator.cancel()   # no torch sidecar competing with the run
        self._side_overlay_locator.cancel()
        self._preview_pane.start_run()

        self._append("> Running pipeline…\n")
        self._set_running(True)
        self._start_process(argv, self._on_pipeline_finished)

    def _stage_failed(self, exit_code: int, exit_status: QProcess.ExitStatus, label: str) -> bool:
        """Shared failure test for the QProcess finish handlers: on a non-clean exit, append the
        '> {label} failed (exit code {n})' line and surface the log, returning True. Returns False
        on a clean exit so the caller continues. The caller still handles running-state teardown
        (each finish handler stops differently)."""
        if exit_code != 0 or exit_status != QProcess.ExitStatus.NormalExit:
            self._append(f"\n> {label} failed (exit code {exit_code}).")
            self._show_log()  # surface the failure without making the user open it
            return True
        return False

    def _on_pipeline_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        """Pipeline stage done: stop the live preview, unwind to idle on a user Stop or a
        failure, else chain straight into the extract stage (staying in the running state)
        with the progress pane holding "Finishing up…" until Review opens."""
        if self._consume_stop():
            return
        if exit_code != 0 or exit_status != QProcess.ExitStatus.NormalExit:
            self._preview_pane.stop_run()
            self._set_running(False)
            self._request_clock()  # still on the main page: the run held the clock search
            self._append(f"\n> Pipeline failed (exit code {exit_code}). Stopping.")
            self._show_log()  # surface the failure without making the user open it
            return
        # Success → continue straight into extract (stays in the running state; no prompt).
        # The pane keeps showing "Finishing up…" through the extract subprocess and the
        # synchronous Review load; _on_extract_finished hides it on every exit.
        self._preview_pane.show_finishing()
        self._start_extract()

    def _start_extract(self, *, tag_flags: list[str] | None = None) -> None:
        """Run vodparse.cli.extract_matches --csv-only for the current vod, then (on success) open Review.
        Shared by Analyze VOD (after the pipeline), Open existing analysis, and Re-read tags; the
        caller is responsible for confirming any saved-edit discard first. `tag_flags` overrides
        the tag-reading flags for this one run (Re-read tags passes its per-VOD choice); when None
        the Advanced Analysis Options are used. The global --debug preference is appended either way.
        The argv itself is built by commands.extract_argv."""
        video = self._video_path
        if video is None:
            return
        pipeline_json = pipeline_json_for(video)
        mode = self._selected_mode()
        # Re-read tags supplies its own per-VOD tag flags; otherwise use the Advanced Analysis
        # Options. Either way the global --debug preference rides along.
        resolved_tag_flags = (tag_reading_flags(self._extract_opts) if tag_flags is None
                              else tag_flags)
        argv = extract_argv(
            video, pipeline_json, mode=mode, group_by=self._selected_group_by(),
            set_characters=self._selected_set_characters(), use_sponsors=True,
            tag_flags=resolved_tag_flags, debug=self._extract_opts["debug"],
            out_root=export_root(),
        )
        # Remembered at kickoff (not read back from _extract_opts on finish) so the summary
        # popup matches the run that actually happened, even if the options change meanwhile.
        self._extract_was_smart = SMART_TAGS_FLAG in argv

        self._append("\n> Extracting games…\n")
        self._set_running(True)
        self._start_process(argv, self._on_extract_finished)

    def _on_extract_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        """Extract stage done: back to idle; on success drop the now-stale Review draft, show
        the smart-tags summary popup if its Debug-menu toggle is on, and open Review."""
        if self._consume_stop():
            return
        self._set_running(False)
        # Every exit below ends the Analyze-page "Finishing up…" hold (a no-op for the
        # Open-existing/Re-read paths, where the progress pane was never shown).
        if self._stage_failed(exit_code, exit_status, "Game extraction"):
            self._preview_pane.hide_progress()
            self._request_clock()  # still on the main page: the run held the clock search
            return
        # A fresh extract supersedes any prior edits (the user was warned before the run); drop the
        # draft so Review loads clean. Were a stale draft to survive, its protection would be
        # illusory anyway: the first edit in the freshly loaded Review overwrites it.
        video = self._video_path
        if video is None:
            self._preview_pane.hide_progress()
            return
        review_draft_for(video).unlink(missing_ok=True)
        # Show the smart-tags outcome only when its Debug-menu toggle is on; the full ranking is
        # always in the log transcript (Debug → Show log…), so normal users aren't interrupted.
        if self._extract_was_smart and self._show_smart_tags_popup():
            self._show_smart_tags_summary(video)
        self._load_review()
        self._preview_pane.hide_progress()  # after the switch, so the hold spans the whole load
        self._request_clock()  # a Review that failed to load leaves the main page showing

    def _show_smart_tags_summary(self, video: str) -> None:
        """Pop the smart-tags outcome: the winning flag combination and the runner-up, with
        their scores: read from the report the extract run just wrote. (The full 16-row
        ranking table went to the log transcript: Debug → Show log….) Defensive on a
        missing/malformed report: the popup is informational and must never block Review."""
        show_smart_tags_summary(self, output_dir_for(video) / SMART_TAGS_REPORT_NAME,
                                log_menu_path="Debug → Show log…")

    def _on_reread_tags(self) -> None:
        """Re-read tags (Review screen): re-run the cheap --csv-only extraction for THIS vod under
        different tag-reading flags and refresh Review in place. The dialog is seeded from the flags
        that produced the current matches.json (the per-VOD sidecar), not the global options, and
        its choice drives this one run only: the global extract/opt_* settings are left untouched
        (contrast _edit_extract_options, which persists). Mode and set-characters keep coming
        from the Advanced Analysis Options and group-by from the Main page, the same sources
        _load_review reads when it re-renders."""
        video = self._video_path
        if video is None:
            return
        if self._process_running():
            return
        if not self._confirm_reread_discard():
            return
        options = seed_tag_options(
            read_sidecar(output_dir_for(video) / EXTRACT_FLAGS_NAME), self._extract_opts)
        if RereadTagsDialog(options, Path(video).name, self).exec() != QDialog.DialogCode.Accepted:
            return
        self._start_extract(tag_flags=tag_reading_flags(options))
        self._review.show_status("Rereading tags…")

    def _on_change_tournament_name(self) -> None:
        """Change tournament name (Tools, Review screen): fix a name the user forgot to set
        before extracting, without leaving Review. On accept the new name flows to the review
        session's meta (export titles / CSV tournament column / export folder naming all read
        it), the Main-page tournament field (which seeds any later fresh load), the Review
        heading, and the export-output buttons (whose paths derive from the name)."""
        dialog = TournamentNameDialog(self._review.tournament_name(), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name = dialog.name()
        self._review.set_tournament_name(name)
        self._tournament_edit.setText(name)
        self._update_title(self._stack.currentIndex())
        self._refresh_export_outputs()
        # If clips were already exported, the export-output buttons now fall back to that
        # older export (resolve_export_csv): say so, since the new name only reaches the
        # clips/CSV/titles through a fresh export.
        if rename_leaves_exported_clips_behind(self._review, export_root()):
            self._review.show_status(RENAME_EXPORT_NOTE)

    def _load_review(self) -> None:
        video = self._video_path
        if video is None:
            return
        stem = vod_stem(video)
        matches_json = output_dir_for(video) / "matches.json"
        try:
            matches = json.loads(matches_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self._append(f"\n> Could not load {matches_json.name}: {e}")
            return
        mode = self._selected_mode()
        self._review.load_model(matches, mode, self._selected_group_by(),
                                video, stem,
                                tournament_name=self._tournament_edit.text().strip(),
                                set_characters=self._selected_set_characters())
        self._append("\n> Loaded games. Review, then Export clips.")
        self._show_page(1)
        # Refresh the mode note in case a re-read kept us on the Review page.
        self._update_title(1)
        self._relabel_export_selected_action()

    def _resume_review(self) -> bool:
        """Reopen the saved Review edits for the current vod (no extract) and switch to Review.
        Returns whether a draft was actually loaded. The draft holds the full editing state, so
        this works even without matches.json."""
        video = self._video_path
        if video is None:
            return False
        if not self._review.load_from_draft(video, vod_stem(video),
                                            set_characters=self._selected_set_characters()):
            return False
        self._append("\n> Resumed saved edits. Review, then Export clips.")
        self._show_page(1)
        self._relabel_export_selected_action()
        return True

    def load_debug_vod(self, vod: str, resume: bool = False) -> None:
        """--debug entry point: preselect the vod and jump to Review. By default it loads fresh
        from the vod's matches JSON; with `resume` (--debug-resume) it reopens the saved edit draft
        instead, falling back to a fresh load if no draft exists."""
        self._video_path = vod
        self._file_label.setText(Path(vod).name)
        self._load_video_into_preview(vod)
        self._seed_tournament_name()  # --debug sets _video_path directly, bypassing _on_choose
        self._refresh_nav()
        if resume:
            self._append(f"--debug (resume): {Path(vod).name}")
            # Straight to _resume_review: the video-changed warning is intentionally skipped here.
            if self._resume_review():
                return
            self._append("  no saved edits, loading fresh from the extract instead")
        else:
            self._append(f"--debug: {Path(vod).name}")
        self._load_review()

    def _on_review_export(self, data: dict) -> None:
        if self._process_running():   # a reread (or the run itself) is still going
            self._review.show_status(BUSY_EXPORT_NOTE)
            return
        data["title_char_limit"] = self._title_char_limit()
        data["title_filenames"] = self._title_filenames(data["mode"])
        # Written explicitly (not omitted) so overwriting a reviewed.json that an older build
        # saved with use_sponsors=false flips it back on: the exporter honours a stored false.
        data["use_sponsors"] = True
        out = PROJECT_ROOT / "json_output" / data["stem"] / "reviewed.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Capture the export root once for this run: the folder below, the stale-folder
        # vetting, the subprocess argv, and the post-export delete guard must all agree
        # even if the paths/export_dir setting changes while the export runs.
        self._export_root = export_root()
        # Refuse an unreachable root (typically an unplugged export drive) BEFORE the
        # write_json_atomic below: probing a dead root finds no previous export, so the
        # rewrite would silently drop the export_folder/export_csv records that let the
        # export-output buttons find the old clips once the drive returns.
        if not export_root_reachable(self._export_root):
            self._warn_export_root_unavailable(self._export_root)
            return
        # Same folder vodparse.cli.export_reviewed derives from the reviewed JSON, for the dialog.
        self._export_dir = self._export_root / export_basename(
            data.get("tournament_name", ""), data["stem"], data["mode"] != "match")
        # The previous export, looked up BEFORE write_json_atomic below replaces reviewed.json
        # (destroying the export_folder/export_csv records that point at it).
        previous = last_export_info(data["stem"], self._export_root)
        # The tournament's CSV may already list other VODs (a stream restart splits one
        # tournament across several files). Confirm the first time this VOD joins one, and
        # back out before anything is written if the user cancels.
        export_csv = self._export_dir / f"{self._export_dir.name}.csv"
        if not self._confirm_shared_export_csv(export_csv, data["stem"]):
            return
        self._previous_clip_names = vod_clip_filenames(export_csv, data["stem"])
        if previous is not None and previous.csv.exists():
            # Carry the previous export records through the rewrite: if this export fails, the
            # export-output buttons can still fall back to the old clips (resolve_export_csv);
            # a successful export writes the new records over them (see vodparse.cli.export_reviewed).
            data["export_folder"] = previous.csv.parent.name
            data["export_csv"] = previous.csv.name
        # A rename moves the export to a new folder, silently duplicating the clips' disk
        # space: remember the superseded folder and offer to delete it on success.
        self._stale_export_dir = stale_export_dir(previous, self._export_dir, data["mode"],
                                                  self._export_root, stem=data["stem"])
        write_json_atomic(out, data)
        self._append(f"\n> Exporting clips from {out.name}…\n")
        self._exporting = True
        self._set_running(True)
        # Size the bar from the payload that actually gets exported: build_reviewed_data drops
        # excluded sets, and the subprocess prints one "  Saved:" line per surviving clip, so
        # len(data["clips"]) is exactly how many lines will arrive (finding S2).
        self._review.set_exporting(True, len(data["clips"]))
        self._review.set_upload_available(False)  # don't open a half-written CSV mid-export
        self._start_process(["-m", "vodparse.cli.export_reviewed", "--input", str(out),
                             "--out-root", str(self._export_root)],
                            self._on_export_finished)

    def _confirm_shared_export_csv(self, csv_path: Path, stem: str) -> bool:
        """Whether the export of VOD `stem` may go ahead into the tournament CSV at
        `csv_path`. When this VOD would join a CSV that already lists other VODs for the
        first time (see gui.paths.other_vods_to_confirm), ask: "Add to tournament" goes
        ahead, anything else cancels. An unreadable CSV is handed to
        _confirm_overwrite_unreadable_csv: the export would refuse it, after the user waited
        for nothing."""
        try:
            other_vods = other_vods_to_confirm(csv_path, stem)
        except CSV_READ_ERRORS as err:
            return self._confirm_overwrite_unreadable_csv(csv_path, err)
        if not other_vods:
            return True
        lines = [f"• {other_vod}: {count} clip{'s' if count != 1 else ''}"
                 for other_vod, count in other_vods.items()]
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Add to an existing tournament export?")
        source = "another VOD" if len(other_vods) == 1 else "other VODs"
        box.setText(f"“{csv_path.parent.name}” already has clips from {source}:\n\n"
                    + "\n".join(lines))
        box.setInformativeText(
            "Adding this VOD's clips puts the whole tournament in one CSV for the Upload "
            "Assistant.\n\n"
            "If this is a different event, cancel and change the tournament name.")
        add_btn = box.addButton("Add to tournament", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(add_btn)
        self._keep_above_assistant(box)
        box.exec()
        return box.clickedButton() is add_btn

    def _confirm_overwrite_unreadable_csv(self, csv_path: Path, err: Exception) -> bool:
        """The tournament CSV at `csv_path` exists but can't be read (`err` says why), so the
        export would refuse it: it may list other VODs' clips, and writing this VOD's rows
        over it would lose them. Ask. Overwrite deletes the file, and the export then
        writes a new CSV of this VOD's clips alone; Cancel, the default, backs out and
        leaves the file for the user to move or fix. A file that can't be deleted (open in
        a spreadsheet app, on Windows) is warned about and backs out too."""
        box = QMessageBox(
            QMessageBox.Icon.Warning, "Can't read the export's CSV",
            f"{csv_path}\n\ncouldn't be read ({err}).\n\n"
            "Move it out of the folder and export again.\n\n"
            "Overwrite replaces it with a CSV of this VOD's clips only.",
            parent=self)
        overwrite_btn = box.addButton("Overwrite", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        self._keep_above_assistant(box)
        box.exec()
        if box.clickedButton() is not overwrite_btn:
            return False
        try:
            csv_path.unlink(missing_ok=True)
        except OSError as exc:
            QMessageBox.warning(
                self, "Couldn't overwrite the CSV",
                f"{csv_path}\n\ncouldn't be removed ({exc}).\n\n"
                "Close it in any program that has it open and export again.")
            return False
        self._append(f"\n> Removed the unreadable {csv_path.name}; the export writes a new one.")
        return True

    def _warn_export_root_unavailable(self, root: Path) -> None:
        """Tell the user an export can't proceed because its destination root is missing
        (see gui.export_root.export_root_reachable): the one message every export entry
        point shows before backing out. A dialog, not a status line, because the fix is
        outside the app: plug the drive back in, or change the setting."""
        QMessageBox.warning(
            self, "Export folder not available",
            f"The export folder isn't available:\n\n{root}\n\n"
            "If it's on an external drive, connect the drive and try again, "
            "or choose a different export folder in Settings")

    def _export_dialog_dir(self, data: dict) -> tuple[Path, bool] | None:
        """The folder export-adjacent save dialogs (single-clip export, chapter timestamps)
        open into, and whether this call created it: the current-name export folder, or
        after a rename with no re-export yet: the previous export's folder where the clips
        actually are (see resolve_export_csv). Created only when nothing was ever exported,
        so the dialog still has a sensible place to open without strewing empty folders
        around post-rename. The single-clip export also fits its suggested filename against
        this folder; after a cross-mode or pre-rename history the resolved folder can
        differ from the one a full export would write to, which can skew the suggestion
        only through the Windows path-length budget: an accepted, extreme-depth-only
        divergence, since the user sees and confirms the name.

        None (with the warning already shown) when the export root is unreachable —
        typically an unplugged export drive, where mkdir(parents=True) would otherwise
        silently rebuild the missing mount point as a plain folder — or when the mkdir
        itself fails (drive yanked between check and mkdir, read-only location): callers
        skip their save flow, and there's never a folder to clean up."""
        is_set = data["mode"] != "match"
        root = export_root()
        if not export_root_reachable(root):
            self._warn_export_root_unavailable(root)
            return None
        clip_dir = resolve_export_csv(data.get("tournament_name", ""), data["stem"],
                                      is_set, root).parent
        created = not clip_dir.exists()
        if created:
            try:
                clip_dir.mkdir(parents=True)
            except OSError:
                self._warn_export_root_unavailable(root)
                return None
        return clip_dir, created

    def _on_review_export_one(self, data: dict, clip_number: int) -> None:
        """The Review tree's right-click "Export this set/game to a file…": ask where to save,
        then cut just that clip (with the current edits: `data` is the same reviewed
        projection a full export uses) via the export subprocess. Unlike a full export this
        produces one file wherever the user chose and nothing else: the projection goes to
        a sibling of reviewed.json, so the canonical file, its export records, the
        export-output buttons, and the delete-superseded-folder offer are all left alone."""
        if self._process_running():
            # The File-menu twin is enable-gated during runs, but the tree's right-click can
            # still land here: explain instead of silently swallowing the click.
            self._review.show_status("Busy: wait for the current run to finish before exporting")
            return
        reviewed = cast(ReviewedData, data)  # the dict IS the projection; typed view for helpers
        unit = clip_unit(reviewed)
        # The same Settings injection as a full export, so the suggested filename below
        # resolves title naming identically.
        data["title_char_limit"] = self._title_char_limit()
        data["title_filenames"] = self._title_filenames(data["mode"])
        data["use_sponsors"] = True   # explicit for the same reason as _on_review_export
        # If this call created the default folder, it's removed again: on cancel below,
        # else in the finish handler: unless the export lands a file in it, so no path
        # leaves an empty folder that lights up "Open export folder".
        dialog_dir = self._export_dialog_dir(data)
        if dialog_dir is None:
            return  # export root unavailable; warned there, and nothing was created
        clip_dir, created = dialog_dir
        # Suggest exactly the filename a full "Export clips" run would give this clip.
        default = clip_dir / planned_clip_filename(reviewed, clip_number, clip_dir)
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export {unit} {clip_number} to", str(default), "Videos (*.mp4)")
        if path and Path(path).suffix.lower() != ".mp4":
            # ffmpeg derives the container from the extension. Testing for ".mp4"
            # specifically (not just any suffix) matters: a dotted player tag like
            # "Mr.Tag vs Rival" has a truthy-but-bogus suffix that would otherwise
            # slip through. On macOS this block never runs: the native save panel
            # appends .mp4 and vets the overwrite itself (verified live 2026-07-18):
            # so it's the fallback for platforms whose dialog returns the name as typed.
            path += ".mp4"
            # The dialog's overwrite prompt vetted the name as typed, without the appended
            # extension: re-check the real destination ourselves.
            if Path(path).exists():
                answer = QMessageBox.question(
                    self, "Replace file?", f"{Path(path).name} already exists. Replace it?")
                if answer != QMessageBox.StandardButton.Yes:
                    path = ""  # treated as a cancel below, including the folder cleanup
        if not path:
            if created:
                # Cancelled with the just-created folder still untouched: remove it again.
                try:
                    clip_dir.rmdir()
                except OSError:
                    pass
            return
        path = str(Path(path))  # Qt's dialog spells Windows paths with "/": show them natively
        # The subprocess reads the projection from a sibling file, NOT reviewed.json: a
        # single-clip cut is not a full export, and rewriting the canonical file would
        # replace its mode/tournament fields while old export records carry through:
        # falsifying what last_export_info derives (up to letting the delete-previous-
        # export offer target the other mode's clips).
        out = PROJECT_ROOT / "json_output" / data["stem"] / "reviewed_single.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(out, data)
        self._append(f"\n> Exporting {unit} {clip_number} to {path}…\n")
        self._set_running(True)
        self._exporting = True  # count the one "  Saved:" line into the progress bar
        self._review.set_exporting(True, 1)
        # The destination (and the dialog folder, if this export created it) ride into
        # the finish handler via this closure, not window state: there is no attribute
        # to go stale between exports.
        created_dir = clip_dir if created else None

        def finished(code: int, status: QProcess.ExitStatus) -> None:
            self._on_export_one_finished(code, status, path, out, created_dir)

        self._start_process(["-m", "vodparse.cli.export_reviewed", "--input", str(out),
                             "--only-clip", str(clip_number), "--out-file", path],
                            finished)

    def _on_export_one_finished(self, exit_code: int, exit_status: QProcess.ExitStatus,
                                path: str, projection: Path,
                                created_dir: Path | None) -> None:
        """Single-clip export done: back to idle; on success note the saved file (`path`,
        bound at kickoff) in the log and the Review status line, refresh the
        export-output buttons: the clip may have just created the default export folder
        their enablement keys on: and delete the consumed `projection` sibling file (on
        failure it stays behind so a diagnostics bundle captures what the subprocess
        read). No completion dialog: the user picked the destination themselves.

        `created_dir` is the dialog folder _export_dialog_dir made for this export (None
        if it already existed). It's removed here, after the subprocess is done: never
        before, so a running ffmpeg can't lose its destination: and rmdir succeeds only
        while the folder is empty, i.e. exactly when the export failed, was stopped, or
        landed elsewhere and would otherwise strand a folder that lights up "Open export
        folder"."""
        if created_dir is not None:
            try:
                created_dir.rmdir()
            except OSError:
                pass  # not empty: a file landed in it; it's a real export folder now
        if self._consume_stop():
            return
        self._exporting = False
        self._set_running(False)
        self._review.set_exporting(False)
        if self._stage_failed(exit_code, exit_status, "Clip export"):
            return
        projection.unlink(missing_ok=True)
        self._refresh_export_outputs()
        self._append(f"\n> Saved {path}.")
        self._review.show_status(f"Saved {Path(path).name}")

    def _on_review_chapters(self, data: dict) -> None:
        """Put the chapter timestamps on the clipboard (the Review screen's "Copy timestamps"
        button). Copying is the only delivery: pasting into a description is what these are
        for, and anyone wanting a file can paste into one. There was once a save-to-file mode
        behind an Export-tab preference; it was removed as surface area nobody needed."""
        text = build_chapters(cast(ReviewedData, data), use_sponsors=True)
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return
        clipboard.setText(text)
        message = "Chapter timestamps copied to the clipboard."
        self._append(f"\n> {message}")
        self._review.show_status(message)

    def _on_export_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        """Clip export done: back to idle and refresh the export-output buttons; on success
        show the completion dialog (repeating any title-length warnings from the log) with
        Open-folder / Upload-Assistant shortcuts, plus an opt-in checkbox to delete the
        previous export's folder after a rename, or to remove the clips a re-export left
        behind."""
        if self._consume_stop():
            return
        self._exporting = False
        # Consume the superseded-folder capture from kickoff (failure discards it: the old
        # export is then still the only complete one, so it must not be offered for deletion).
        stale = self._stale_export_dir
        self._stale_export_dir = None
        previous_clip_names = self._previous_clip_names
        self._previous_clip_names = frozenset()
        self._set_running(False)
        self._review.set_exporting(False)
        self._refresh_export_outputs()  # exported clips/folder now (likely) exist; reflect on buttons
        self._reload_assistant_showing(self._export_dir / f"{self._export_dir.name}.csv")
        if self._stage_failed(exit_code, exit_status, "Clip export"):
            return
        self._append(f"\n> Clips exported to {self._export_dir}.")
        # Title-length warnings from vodparse.cli.export_reviewed would scroll by in the log under this
        # dialog, so repeat them where they'll be seen.
        warnings = [line.strip() for line in self._proc_output.splitlines()
                    if (line.strip().startswith("Warning:") and "title(s)" in line)
                    or line.strip().startswith("Title report:")]
        message = (f"All clips have been saved to:\n\n"
                   f"{self._export_dir}")
        if warnings:
            message += "\n\n⚠ " + "\n⚠ ".join(warnings)
        # Offer to open the folder or the Upload Assistant; any button dismisses (Done stays put).
        box = QMessageBox(self)
        self._keep_above_assistant(box)  # don't hide under an already-open assistant window
        box.setWindowTitle("Export complete")
        box.setText(message)
        open_folder_btn = box.addButton("Open folder", QMessageBox.ButtonRole.ActionRole)
        assistant_btn = box.addButton("Open Upload Assistant", QMessageBox.ButtonRole.ActionRole)
        done_btn = box.addButton("Done", QMessageBox.ButtonRole.AcceptRole)
        box.setDefaultButton(done_btn)
        # A rename moved this export to a new folder, leaving the old one as a full duplicate
        # of the clips: offer to reclaim the space. Opt-in (unchecked by default), applied on
        # whichever button dismisses the dialog.
        delete_box: QCheckBox | None = None
        # same_physical_dir: on a case-insensitive filesystem a case-only rename lands the
        # new export in the very folder captured as "stale": offering it would delete the
        # clips that just finished writing. Checked here (not at kickoff) because the new
        # export dir doesn't exist until the CLI creates it.
        if (stale is not None and stale.is_dir()
                and not same_physical_dir(stale, self._export_dir)):
            # Show the full path, not just the folder name: the target came from a JSON
            # record on disk, and the user should see exactly what a tick deletes.
            delete_box = QCheckBox(
                f"Delete the previous export ({_dir_size_gb(stale):.1f} GB):\n{stale}")
            box.setCheckBox(delete_box)
        # A re-export into the same folder replaced this VOD's CSV rows, but the clips it no
        # longer lists (merged, split, retitled, excluded) are still on disk: offer to remove
        # them, opt-in like the folder offer. A QMessageBox holds one checkbox, so the folder
        # offer wins on the rare export that qualifies for both.
        outdated: list[Path] = []
        outdated_box: QCheckBox | None = None
        if delete_box is None and previous_clip_names:
            outdated = outdated_clip_files(
                self._export_dir, self._export_dir / f"{self._export_dir.name}.csv",
                previous_clip_names)
        if outdated:
            outdated_box = QCheckBox(
                f"Remove {len(outdated)} outdated clip{'s' if len(outdated) != 1 else ''} "
                f"from this VOD's last export ({_files_size_gb(outdated):.1f} GB)")
            outdated_box.setToolTip(_outdated_clips_tooltip(outdated))
            box.setCheckBox(outdated_box)
        box.exec()
        clicked = box.clickedButton()
        if stale is not None and delete_box is not None and delete_box.isChecked():
            if self._delete_stale_export(stale):
                self._retarget_assistant_after_delete(stale)
        if outdated_box is not None and outdated_box.isChecked():
            self._delete_outdated_clips(outdated)
        if clicked is open_folder_btn:
            self._open_folder(self._export_dir)
        elif clicked is assistant_btn:
            self._open_upload_assistant(self._export_dir / f"{self._export_dir.name}.csv")
        # Done / closed: stay on the Review screen.

    def _open_folder(self, path: Path) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _delete_stale_export(self, folder: Path) -> bool:
        """Remove a superseded export folder (the ticked offer in the Export-complete dialog).
        The target was vetted at export kickoff (see gui.paths.stale_export_dir); the cheap
        invariants are re-checked here in case the disk changed under us during the export.
        Windows refuses while a clip inside is open in another program: warn, don't crash.
        The guard uses the export root captured at kickoff (the same root the vetting ran
        against), not a live read: a mid-export settings change must not re-aim it.
        Returns whether the folder was actually deleted."""
        if (self._export_root is None or folder.is_symlink()
                or folder.resolve().parent != self._export_root.resolve()):
            self._append(f"\n> Not deleting {folder}: it no longer looks like an export folder.")
            return False
        if same_physical_dir(folder, self._export_dir):
            self._append(f"\n> Not deleting {folder}: it is the folder the new export "
                         "just landed in.")
            return False
        try:
            shutil.rmtree(folder)
        except OSError as err:
            box = QMessageBox(
                QMessageBox.Icon.Warning, "Couldn't delete previous export",
                f"“{folder.name}” couldn't be fully deleted:\n{err}\n\n"
                "A file inside may be open in another program.",
                parent=self)
            self._keep_above_assistant(box)  # the export may have finished under an open assistant
            box.exec()
            return False
        self._append(f"\n> Deleted previous export {folder}.")
        return True

    def _delete_outdated_clips(self, clips: list[Path]) -> None:
        """Remove the clip files a re-export left behind (the ticked offer in the
        Export-complete dialog; vetted by gui.paths.outdated_clip_files). Windows refuses
        while a clip is open in another program: the rest are still removed, and the ones
        that couldn't be are listed in one warning."""
        failed: list[str] = []
        for clip in clips:
            try:
                clip.unlink()
            except FileNotFoundError:
                continue  # already gone: nothing left to remove
            except OSError as err:
                failed.append(f"{clip.name}: {err}")
                continue
            self._append(f"\n> Removed outdated clip {clip}.")
        if failed:
            box = QMessageBox(
                QMessageBox.Icon.Warning, "Couldn't remove some outdated clips",
                "These clips couldn't be removed:\n\n" + "\n".join(failed)
                + "\n\nA file may be open in another program.",
                parent=self)
            self._keep_above_assistant(box)
            box.exec()

    def _reload_assistant_showing(self, csv: Path) -> None:
        """Reload an open Upload Assistant that has `csv` loaded, which an export just
        rewrote: the CSV can be shared by a tournament's VODs, so the export may have
        added or replaced rows the assistant doesn't have, and its next save would write
        its stale rows back over them (see UploadAssistant.reload_csv). Runs after a
        failed export too: harmless when the CSV didn't change."""
        if self._upload_assistant is None:
            return
        loaded = self._upload_assistant.loaded_csv()
        if loaded is None:
            return
        try:
            showing = loaded.samefile(csv)
        except OSError:
            showing = False  # either file is gone: nothing loaded to go stale
        if showing:
            self._upload_assistant.reload_csv()

    def _retarget_assistant_after_delete(self, deleted: Path) -> None:
        """After deleting a superseded export folder, reload an open Upload Assistant that
        still had a CSV from inside it: its save-back path just vanished, so its next save
        would only raise a generic OSError and strand the edits. Reloading onto the fresh
        export's CSV keeps it pointed at the clips that exist."""
        if self._upload_assistant is None:
            return
        loaded = self._upload_assistant.loaded_csv()
        if loaded is not None and loaded.is_relative_to(deleted):
            self._upload_assistant.load_csv(self._export_dir / f"{self._export_dir.name}.csv")

    def _current_export_csv(self) -> Path | None:
        """The metadata CSV for the review screen's current VOD, or None if nothing is loaded.
        Prefers the CSV the current settings would export; when it doesn't exist yet (e.g. the
        tournament was renamed after exporting) this falls back to the last completed export's
        CSV, so the Upload Assistant / export-folder buttons keep opening the clips that are
        actually on disk (see gui.paths.resolve_export_csv)."""
        target = self._review.export_target()
        if target is None:
            return None
        return resolve_export_csv(*target, export_root())

    def _refresh_export_outputs(self) -> None:
        """Enable the Review screen's export-output buttons by what's on disk: the Upload Assistant
        once the metadata CSV exists, Open export folder once the folder exists."""
        csv = self._current_export_csv()
        self._review.set_upload_available(csv is not None and csv.exists())
        self._review.set_export_folder_available(csv is not None and csv.parent.exists())

    def _on_open_upload_assistant(self) -> None:
        """Review-screen button: open the assistant for the current VOD's exported clips."""
        csv = self._current_export_csv()
        if csv is not None:
            self._open_upload_assistant(csv)

    def _on_open_export_folder(self) -> None:
        """Review-screen button: open the current VOD's export folder in the file manager."""
        csv = self._current_export_csv()
        if csv is not None and csv.parent.exists():
            self._open_folder(csv.parent)

    def _keep_above_assistant(self, dialog: QWidget) -> None:
        """Lift a modal dialog into the always-on-top window band when the Upload Assistant is
        showing. The assistant's always-on-top window otherwise covers an application-modal
        dialog on Windows while the dialog blocks all input: the app looks frozen with nothing
        visibly open. Both on top, the dialog shows last and lands above. Call on any modal that
        can appear while the assistant is open, before its exec()."""
        if self._upload_assistant is not None and self._upload_assistant.isVisible():
            dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

    def _open_upload_assistant(self, csv: Path) -> None:
        """Open (or reuse) the Upload Assistant window, preloaded with `csv`."""
        if not csv.exists():  # defensive: fall back to any CSV in the export folder
            others = sorted(csv.parent.glob("*.csv"))
            if not others:
                box = QMessageBox(
                    QMessageBox.Icon.Warning, "No CSV found",
                    f"No metadata CSV in {csv.parent}.\n\n"
                    "Export again, then reopen the Upload Assistant.",
                    parent=self)
                self._keep_above_assistant(box)
                box.exec()
                return
            csv = others[0]
        self._show_upload_assistant().load_csv(csv)  # same call upload_assistant.main() uses

    def _show_upload_assistant(self) -> UploadAssistant:
        """Create (once) and surface the Upload Assistant window without touching its loaded
        CSV. The direct target of File → Upload Assistant…, which opens the assistant empty:
        its own Open… button picks an export CSV: or showing whatever it last had loaded."""
        if self._upload_assistant is None:
            self._upload_assistant = UploadAssistant()  # top-level window; kept on self so it lives
            self._upload_assistant.closed.connect(lambda: self.setEnabled(True))
        self._upload_assistant.show()
        self._upload_assistant.raise_()
        self._upload_assistant.activateWindow()
        # Lock the main window (greyed, input-dead) until the assistant closes: modal in
        # effect, like Preferences. A live main window under the (default-on) always-on-top
        # assistant was the source of a whole class of Windows bugs: modal dialogs opening
        # invisibly UNDERNEATH it while blocking all input. Emulated with setEnabled rather
        # than real Qt modality because platform modality overrides the window level the
        # assistant's Stay-on-top needs to float over the browser. Top-level child dialogs
        # of a disabled window stay interactive, so the one path this can't close: an
        # export already running when the assistant opens: still pops its completion box
        # (which _keep_above_assistant lifts over the assistant).
        self.setEnabled(False)
        return self._upload_assistant

    # ── Diagnostics export ───────────────────────────────────────────────────
    def _on_export_diagnostics(self) -> None:
        """Bundle one vod's JSON/CSV outputs into a verifiable zip the user can send for
        troubleshooting. Self-contained: the user picks any analyzed vod, independent of
        whatever video is loaded; the bundling is done by vodparse.cli.export_diagnostics."""
        if self._process_running():
            return
        stems = available_stems(PROJECT_ROOT)
        if not stems:
            QMessageBox.information(
                self, "Export Diagnostics",
                "No analyzed vods found! Run Analyze VOD on a video first.")
            return
        stem, ok = QInputDialog.getItem(
            self, "Export Diagnostics", "Choose a vod to bundle:", stems, 0, False)
        if not ok or not stem:
            return
        suggested = str(Path.home() / default_zip_name(stem))
        path, _ = QFileDialog.getSaveFileName(
            self, "Save diagnostics bundle", suggested, "Zip archives (*.zip)")
        if not path:
            return
        path = str(Path(path))  # Qt's dialog spells Windows paths with "/": show them natively
        # Snapshot the user's settings into the manifest (all inert preferences: title limits,
        # toggles, intervals). Passed as one JSON argv element; QProcess needs no shell quoting.
        settings = {k: str(self._settings.value(k)) for k in self._settings.allKeys()}
        self._diag_zip = path
        self._append(f"\n> Exporting diagnostics for {stem}…\n")
        self._set_running(True)
        self._start_process(
            ["-m", "vodparse.cli.export_diagnostics", "--stem", stem, "--output", path,
             "--settings-json", json.dumps(settings), "--out-root", str(export_root())],
            self._on_diagnostics_finished)

    def _on_diagnostics_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        """Diagnostics export done: back to idle; on success confirm where the zip was saved."""
        if self._consume_stop():
            return
        self._set_running(False)
        if self._stage_failed(exit_code, exit_status, "Diagnostics export"):
            return
        self._append(f"\n> Diagnostics saved to {self._diag_zip}.")
        QMessageBox.information(
            self, "Diagnostics exported",
            f"A diagnostics bundle was saved to:\n\n{self._diag_zip}")

    # ── QProcess plumbing ────────────────────────────────────────────────────
    def _process_running(self) -> bool:
        """Whether a child process (pipeline/extract/export/diagnostics) is currently in flight."""
        return self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning

    def _on_stop(self) -> None:
        """Stop button: kill the in-flight stage after a confirmation. Re-running starts over;
        nothing is resumed or persisted."""
        if not self._process_running():
            return
        answer = QMessageBox.question(
            self, "Stop?", "Stop the current run? Any progress on the analysis so far will be discarded.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        if not self._process_running():
            # The run finished while the confirmation was open (the modal dialog spins a
            # nested event loop, so the finish handler already ran): nothing to stop.
            # Setting _stopping here would mark the NEXT run's finish as a Stop.
            return
        self._stopping = True  # the resulting child exit is deliberate: _consume_stop handles it
        self._stop_process()

    def _consume_stop(self) -> bool:
        """If the just-finished child process was killed by a user Stop, reset to idle and report it.
        Returns True when the finish was a deliberate stop, so the caller should return early
        instead of treating the non-zero exit as a failure."""
        if not self._stopping:
            return False
        self._stopping = False
        self._exporting = False
        self._stale_export_dir = None          # the kickoff captures live for one export
        self._previous_clip_names = frozenset()
        self._review.set_exporting(False)
        self._preview_pane.stop_run()
        self._set_running(False)
        self._request_clock()  # a stopped Analyze run held the Preview box's clock search
        self._append("\n⏹ Stopped.")
        return True

    def _stop_process(self) -> None:
        """Shut down the child process, so quitting never orphans it: politely first where a
        polite stop can reach it, else by force."""
        if not self._process_running():
            return
        assert self._proc is not None  # implied by _process_running()
        if sys.platform == "win32":
            # On Windows terminate() only posts WM_CLOSE, which a windowless stage process
            # never reads: it always sat out the full STOP_TERM_WAIT_MS (a frozen UI) before
            # the kill() below. No stage handles a polite stop anyway, so kill at once.
            self._proc.kill()
            self._proc.waitForFinished(STOP_KILL_WAIT_MS)
            return
        self._proc.terminate()
        if not self._proc.waitForFinished(STOP_TERM_WAIT_MS):
            self._proc.kill()
            self._proc.waitForFinished(STOP_KILL_WAIT_MS)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self._settings.setValue(self._geometry_key(self._stack.currentIndex()), self.saveGeometry())
        # Before anything that spins a nested event loop (the prompt below): a resolving
        # update check would otherwise deliver its queued verdict inside that loop and pop
        # a dialog over the quit confirmation. If the user then answers No, the dismissed
        # check revives on the next Help ▸ Check for Updates click.
        self._updates.dismiss()
        if self._process_running():
            box = QMessageBox(
                QMessageBox.Icon.Question, "Quit?",
                "A run is still in progress. Quit anyway?\n\n"
                "The current analysis will be stopped and its progress lost.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                parent=self)
            box.setDefaultButton(QMessageBox.StandardButton.No)
            # An export started before the assistant opened is a run too, and Alt+F4 still
            # reaches the locked main window (see the assistant's close below).
            self._keep_above_assistant(box)
            box.exec()
            if box.clickedButton() is not box.button(QMessageBox.StandardButton.Yes):
                if event is not None:
                    event.ignore()
                return
            self._stopping = True  # route the synchronous finish through _consume_stop, not the failure path
        self._shutting_down = True
        # The Upload Assistant is its own top-level window. Left open, it kept the app running
        # after the (locked) main window closed, with no way back to it: Alt+F4 and the
        # taskbar's Close both reach a disabled window on Windows. Quitting closes it too.
        if self._upload_assistant is not None:
            self._upload_assistant.close()
        # Background work first: stopping the run below finishes it synchronously, and its
        # finish handler would otherwise restart the clock search and the preview's OCR
        # helper just before the teardown kills them.
        self._teardown_background_work()
        self._stop_process()   # a no-op when no run is going
        if event is not None:
            event.accept()

    def _teardown_background_work(self) -> None:
        """Stop the window's background workers before Qt teardown. Shared by ``closeEvent``
        and ``force_quit``: a teardown listed in only one quit path resurfaces as a shutdown
        crash (a ``QThread`` destroyed while running aborts the app)."""
        self._converter.cancel_active()   # don't leave an ffmpeg conversion running past the window
        self._gpu.cancel_active_probes()  # nor a torch-importing GPU probe
        self._clock_locator.cancel()      # nor the preview's clock-search sidecar
        self._side_overlay_locator.cancel()   # nor its player-name search
        self._preview_pane.shutdown()     # nor its ffmpeg frame grab / media player
        self._updates.shutdown()          # a check thread destroyed while running aborts teardown

    def force_quit(self) -> None:
        """Quit without the confirmation prompt (Ctrl+C in the terminal). The background work
        is torn down before the run is stopped, as in closeEvent."""
        self._shutting_down = True
        self._teardown_background_work()
        self._stop_process()
        QApplication.quit()

    def _start_process(self, argv: list[str], on_finished: FinishHandler) -> None:
        """Run ``[python] argv`` from the project root, streaming output to the log."""
        proc = QProcess(self)
        # PROJECT_ROOT is the writable workspace root for json_output/; making it the
        # subprocess CWD means the relative paths the pipeline scripts write land there too.
        # Exported clips are different: the export-capable stages get the resolved export
        # root as an explicit --out-root argv (see vodparse.gui.export_root).
        proc.setWorkingDirectory(str(PROJECT_ROOT))
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._read_output)
        proc.finished.connect(on_finished)
        # finished never fires if the process fails to start, so unwind via the same finish
        # handler in that one case (every other error still emits finished).
        proc.errorOccurred.connect(
            lambda err, cb=on_finished: self._on_process_error(err, cb))
        # Free the previous run's QProcess. Each is parented to the window, so without this they
        # accumulate for the window's lifetime (one per stage/run). The prior finish handler has
        # already run by the time we start a new process, so deleteLater is safe here.
        if self._proc is not None:
            self._proc.deleteLater()
        # Keep a reference so the process isn't garbage-collected mid-run.
        self._proc = proc
        # Per-run transcript, so finish handlers can pull warnings out of it (the popup log
        # would bury them under the completion dialog, and it isn't open by default).
        self._proc_output = ""
        # Belt-and-suspenders UTF-8 for third-party output the stage main can't guard itself;
        # see stage_environment's docstring.
        proc.setProcessEnvironment(stage_environment())
        proc.start(sys.executable, argv)

    def _on_process_error(self, error: QProcess.ProcessError,
                          on_finished: FinishHandler) -> None:
        """Handle the one error QProcess.finished doesn't cover: a process that never starts.
        Delegate to the normal finish handler with a failure code so it unwinds running state
        (re-enables buttons, clears the export busy state, surfaces the log). Other errors
        still emit finished and are handled there, so they're ignored here."""
        if error != QProcess.ProcessError.FailedToStart:
            return
        self._append(f"\n> Could not start: {sys.executable} {self._proc.arguments() if self._proc else ''}")
        on_finished(-1, QProcess.ExitStatus.CrashExit)

    def _read_output(self) -> None:
        if self._proc is None:
            return
        data = self._proc.readAllStandardOutput().data().decode(errors="replace")
        self._proc_output += data
        self._append(data, newline=False)
        if self._exporting:
            # export_reviewed prints one "  Saved:" line per clip; count_saved_lines counts them
            # over the accumulated buffer (robust to reads splitting a line) to drive the Review
            # bar. Shared with the writer via export.SAVED_LINE_PREFIX so the seam can't drift.
            self._review.set_export_progress(count_saved_lines(self._proc_output))

    def _append(self, text: str, newline: bool = True) -> None:
        chunk = text + ("\n" if newline else "")
        self._log.moveCursor(self._log.textCursor().MoveOperation.End)
        self._log.insertPlainText(chunk)
        self._log.moveCursor(self._log.textCursor().MoveOperation.End)
        if self._log_dialog is not None:  # keep the open full-transcript popup live
            self._log_dialog.append(chunk)

    def _clear_log(self) -> None:
        """Reset the log buffer (and the open popup, if any) at the start of a run."""
        self._log.clear()
        if self._log_dialog is not None:
            self._log_dialog.clear()

    def _show_log(self) -> None:
        """Open the full-transcript popup (or resurface it if already open). _append mirrors new
        output into it while it's open; the finish handlers call this to surface failures."""
        if self._log_dialog is None:
            self._log_dialog = LogDialog(self._log.toPlainText(), self)
            self._log_dialog.finished.connect(self._on_log_dialog_closed)
            self._log_dialog.show()
        else:
            self._log_dialog.raise_()
            self._log_dialog.activateWindow()

    def _on_log_dialog_closed(self, _result: int) -> None:
        """Drop the popup reference on close so _append stops mirroring and the next _show_log
        rebuilds it from the current transcript."""
        self._log_dialog = None

    # ----- crash-log debug hooks -------------------------------------------
    def _on_show_crash_log(self) -> None:
        """Open the crash log in the OS default text handler, or say where it would be.

        The file only exists once something has been logged (a session banner is written at
        startup, so in practice it exists), and logging can be switched off entirely with
        $VODPARSE_CRASH_LOG=0: both cases get a message naming the path rather than silence.

        Working out where the log lives can itself fail — frozen, it resolves under the app-data
        workspace, which may be on a drive that has gone away since launch. That is caught here
        because an exception escaping a slot is not an error dialog but an abort: PyQt6 calls
        qFatal() on the way out.
        """
        try:
            path = crash_log.crash_log_path()
        except OSError as exc:
            QMessageBox.warning(self, "Crash log",
                                f"Could not work out where the crash log lives:\n{exc}")
            return
        if path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            return
        state = "no crash log has been written yet" if crash_log.logging_enabled() else (
            f"crash logging is off ({crash_log.CRASH_LOG_ENV} is set to an off value)")
        QMessageBox.information(self, "Crash log", f"{state}.\n\nIt would be written to:\n{path}")

    def _on_flip_os_theme(self) -> None:
        """Flip the Windows apps light/dark registry setting, reproducing a live OS theme flip.

        Windows-only (the action is not created elsewhere). The flip itself is what is under
        test: if the app dies here, the crash log's last lines say which step it died on. The
        setting is left flipped: it is the user's own OS preference, and silently flipping it
        back would fire a second colorSchemeChanged and muddy the log.
        """
        crash_log.trace("theme: Debug → Flip OS dark mode requested")
        try:
            light = windows_theme.flip_apps_theme()
        except OSError as exc:
            QMessageBox.warning(self, "Could not flip the theme",
                                f"Writing {windows_theme.THEME_KEY} failed:\n{exc}")
            return
        # Naming the log is a convenience on top of a flip that already happened, and resolving
        # its path can raise (see _on_show_crash_log): report the flip either way rather than
        # letting an exception escape the slot and abort the process.
        try:
            log_note = f" (crash log: {crash_log.crash_log_path()})"
        except OSError as exc:
            log_note = ""
            QMessageBox.warning(self, "Crash log",
                                f"The theme was flipped, but the crash log's location could not "
                                f"be worked out:\n{exc}")
        self._append(f"Flipped the OS theme to {'light' if light else 'dark'}{log_note}")


APPLICATION_NAME = "Ultimate VOD Studio"


def settings_application_name(profile: str | None) -> str:
    """The application name QSettings files the settings under: the app's own, or for a
    ``--settings-profile`` run a separate store beside it ("Ultimate VOD Studio (test)")."""
    if not profile:
        return APPLICATION_NAME
    return f"{APPLICATION_NAME} ({profile})"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultimate VOD Studio GUI")
    parser.add_argument("--debug", metavar="VOD", default=None,
                        help="Load this vod and jump straight to the Review screen if its "
                             "matches JSON already exists (otherwise opens normally).")
    parser.add_argument("--debug-resume", action="store_true",
                        help="With --debug, reopen the saved Review edit draft instead of loading "
                             "fresh from the extract (falls back to fresh if no draft exists).")
    # Launch options for testing and GUI automation: start in a known state without the file
    # dialog, and keep a test run's choices out of the user's own settings.
    parser.add_argument("--game", choices=[game.id for game in GAMES], default=None,
                        help="Select this game at startup (saved like a pick in the Game combo).")
    parser.add_argument("--video", metavar="PATH", default=None,
                        help="Choose this video at startup, as Choose video… would (conversion "
                             "and re-encode offers included). Applied after --game.")
    parser.add_argument("--settings-profile", metavar="NAME", default=None,
                        help="Keep settings in a separate store named after NAME instead of the "
                             "user's own, so a test run's choices never change theirs.")
    args, _ = parser.parse_known_args()  # ignore Qt's own args
    return args


def main() -> None:
    args = _parse_args()
    # Capture crashes to a file from here on: before the QApplication, so a startup crash is
    # covered too. Frozen and windowed, this is the only record a crash leaves at all (see
    # vodparse.gui.crash_log); the path is reachable from Debug → Show crash log….
    crash_log.install("vodparse.gui.app")
    # On Windows, give the process an explicit app id so the taskbar shows our icon
    # instead of grouping the window under pythonw.exe's generic icon. No-op elsewhere.
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ultimatevodstudio.app")
    app = QApplication(sys.argv)
    # Log the OS theme-change messages as they arrive, ahead of Qt's own handling — but only
    # when VODPARSE_NATIVE_TRACE=1 asks for it. The filter is a debugging aid for the theme-flip
    # crash and is *itself* able to crash the process (see crash_log.install_native_event_trace),
    # so a normal launch must not carry it.
    crash_log.install_native_event_trace(app)
    # Follow the OS light/dark setting on Windows/Linux (no-op on macOS, where Cocoa
    # already does this). Before any widget is built, so everything paints themed once.
    apply_system_theme(app)
    icon_path = app_icon_path()
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    # Cross-platform persistent settings: a plain .ini on every OS, keyed by the app identity below
    # (%APPDATA%\... on Windows, ~/.config/... on macOS/Linux). Set before any QSettings is created.
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    app.setOrganizationName("Ultimate VOD Studio")
    app.setOrganizationDomain("ultimatevodstudio.app")
    app.setApplicationName(settings_application_name(args.settings_profile))
    # Scale the whole UI's text per the saved "Text size" preference, before any widget is built
    # (see apply_font_scale). QSettings() needs the org/app identity above to resolve the right file.
    apply_font_scale(app, QSettings().value(SETTING_FONT_SCALE, FONT_SCALE_DEFAULT, type=float))
    window = MainWindow()
    # Ctrl+C: stop any running stage before quitting (no confirmation: Ctrl+C is explicit).
    # Python signal handlers only run while the interpreter executes bytecode, so the no-op
    # timer wakes it periodically while Qt's C++ event loop is in charge.
    signal.signal(signal.SIGINT, lambda *_: window.force_quit())
    sigint_timer = QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(SIGINT_PUMP_INTERVAL_MS)
    window.show()
    window.restore_startup_geometry()
    if args.game:
        window.select_game(args.game)
    if args.video:
        window.choose_video(args.video)
    if args.debug:
        window.load_debug_vod(args.debug, resume=args.debug_resume)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
