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
    Qt, QProcess, QSettings, QTimer, QUrl,
)
from PyQt6.QtGui import (QAction, QCloseEvent, QDesktopServices, QFont, QIcon, QKeySequence,  # noqa: E402
                         QStandardItemModel)
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QButtonGroup,
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
    QRadioButton,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from vodparse.gui import crash_log, windows_theme  # noqa: E402
from vodparse.gui.commands import extract_argv, pipeline_argv  # noqa: E402
from vodparse.gui.export_root import export_root, export_root_reachable  # noqa: E402
from vodparse.tags import INGAME_TAG_BAND_FRAC_DEFAULT  # noqa: E402  (cv2/torch-free)
from vodparse.gui.common import (BusyDialog, ElidingLabel, apply_font_scale,  # noqa: E402
                        apply_system_theme, muted_text_color, notice_color,
                        selected_radio_value, stage_environment, warning_color)
from vodparse.gui.dialogs import (AboutDialog, ExtractOptionsDialog, LogDialog,  # noqa: E402
                                  RereadTagsDialog, SettingsDialog, TournamentNameDialog)
from vodparse.gui.paths import (PROJECT_ROOT, analysis_settings_differences,  # noqa: E402
                       app_icon_path, default_tournament_name, existing_analysis,
                       existing_analysis_summary, export_csv_for, last_export_info, output_dir_for,
                       pipeline_json_for, resolve_export_csv, review_draft_for,
                       same_physical_dir, stale_export_dir, video_metadata_mismatches,
                       vod_picker_start_dir)
from vodparse.gui.settings import (EXTRACT_OPT_DEFAULTS, FONT_SCALE_DEFAULT,  # noqa: E402
                          OCR_INTERVAL_CHOICES, review_mode_note, seed_tag_options,
                          tag_reading_flags,
                          OCR_ENGINE_CHOICES, OCR_ENGINE_DEFAULT_GUI, resolve_saved_engine,
                          OCR_INTERVAL_DEFAULT_SEC, OCR_LANG_DEFAULT_GUI, lang_display_name,
                          SET_CHARACTERS_DEFAULT,
                          SETTING_CONFIRM_REUSE_SETTINGS,
                          SETTING_FONT_SCALE, SETTING_GPU_SKIP_FIRST_RUN_OFFER,
                          SETTING_DEFAULT_VOD_DIR, SETTING_GATE_OCR_ON_TIMER,
                          SETTING_GEOMETRY_MAIN, SETTING_GEOMETRY_REVIEW,
                          SETTING_INGAME_TAG_BAND_FRAC, SETTING_OCR_ENGINE,
                          SETTING_OCR_INTERVAL_SEC, SETTING_OCR_LANG, SETTING_OCR_ROI_CACHE,
                          SETTING_SHOW_DEBUG_MENU, SETTING_SHOW_FONT_STYLE_NOTE,
                          SETTING_SHOW_SMART_TAGS_POPUP,
                          SETTING_TITLE_CHAR_LIMIT,
                          SETTING_TITLE_FILENAMES_MATCHES, SETTING_TITLE_FILENAMES_SETS,
                          SITUATIONAL_EXTRACT_OPTS,
                          TITLE_CHAR_LIMIT_DEFAULT)
from vodparse.gui.convert import Mp4Converter, opencv_can_read  # noqa: E402
from vodparse import gpu_support  # noqa: E402  (stdlib-only; no torch/Qt at import)
from vodparse.gui.gpu_support import GpuSupportManager, GpuSupportDialog  # noqa: E402
from vodparse.gui.updates import UpdateChecker  # noqa: E402
from vodparse.ocr_engine import (DOCTR_WEIGHTS_MB_APPROX, OCR_ENGINE_DOCTR,  # noqa: E402  (stdlib-only; no torch/Qt at import)
                                 doctr_available, doctr_weights_cached)
from vodparse.runtime_paths import is_frozen  # noqa: E402
from vodparse.gui.preview import PreviewProgressPane  # noqa: E402
from vodparse.gui.widgets import FittingStack, make_icon_button  # noqa: E402
from vodparse.extract_flags import EXTRACT_FLAGS_NAME, read_sidecar  # noqa: E402
from vodparse.smart_tags import SMART_TAGS_REPORT_NAME  # noqa: E402
from vodparse.gui.review.screen import ReviewScreen  # noqa: E402  (QtMultimedia; after plugin setup)
from vodparse.gui.upload_assistant import (  # noqa: E402  (PyQt6 widgets; after plugin setup)
    UploadAssistant, forget_saved_geometry as forget_upload_assistant_geometry)
from vodparse.diagnostics import available_stems, default_zip_name  # noqa: E402
from vodparse.export import (build_chapters, clip_unit, count_saved_lines,  # noqa: E402
                             export_basename, planned_clip_filename)
from vodparse.ffmpeg_runtime import UNPLAYABLE_CONTAINER_EXTS  # noqa: E402
from vodparse.jsonio import write_json_atomic  # noqa: E402
from vodparse.paths import vod_stem  # noqa: E402
from vodparse.schema import ReviewedData  # noqa: E402

VIDEO_FILTER = "Videos (*.mp4 *.mov *.mkv *.avi);;All files (*)"


def _dir_size_gb(path: Path) -> float:
    """Total size of the files under `path` in GB (decimal, matching how drives are
    labelled: the MP4 converter's size warning uses the same convention). 0 if unreadable."""
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
    except OSError:
        return 0.0

# The signature QProcess.finished delivers: (exit_code, exit_status).
FinishHandler = Callable[[int, QProcess.ExitStatus], None]

# How long _stop_process waits for the child to exit: after the polite terminate(),
# then after the follow-up kill().
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

# Whether the Sets box shows the "Group sets by:" source picker. Hidden for the OSS
# release: ingame-tag grouping only works reliably as a *fallback*, so the choice stays
# pinned to its "Auto" default for now. All the wiring (radios, note, --group-by plumbing)
# stays live behind this flag: flip to True to restore the row once ingame grouping works.
SHOW_GROUP_BY_CONTROLS = False

# The muted note under the main screen's "Font style:" combo, one per style: what that
# style is good at, so the trade-off is visible without opening the info button beside it.
# Which one shows follows the combo; both are hidden together when the user turns the note
# off in Settings → Analysis. See _refresh_engine_note.
FONT_STYLE_NOTE_STANDARD = ("Standard reads most plain text overlays. If you want to read "
                            "fancier fonts (or italics), try using Stylized.")
FONT_STYLE_NOTE_STYLIZED = "Stylized reads a wider range of fonts, but VOD analysis is slower."



def _smart_entry_text(title: str, entry: dict) -> str:
    """One smart-tags report entry ("Picked"/"Runner-up") for the summary popup: the flag
    label plus each scorer's value ("-" for the no-tags variance sentinel, stored as null),
    plus the heavy-fallback promotion reason on a winner the correction moved to the front."""
    scores = ", ".join(f"{name} {value:g}" if isinstance(value, (int, float)) else f"{name} -"
                       for name, value in entry["scores"].items())
    text = f"{title}: {entry['label']}\n{scores}"
    if entry.get("promotion"):
        text += f"\n{entry['promotion']}"
    return text


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
        # The export root captured at the same kickoff, so the in-flight export's folder,
        # the stale-folder vetting above, and the post-export delete guard all agree even
        # if the paths/export_dir setting changes mid-export.
        self._export_root: Path | None = None
        self._stopping = False  # set while a user-requested Stop is in flight; the child exit is deliberate
        self._extract_was_smart = False  # whether the in-flight extract ran --smart-tags; set at kickoff

        # Persistent app settings (see SettingsDialog); QSettings keys off the org/app names set in
        # main(), so values survive restarts on every platform.
        self._settings = QSettings()

        # Advanced extract flags, edited via the Advanced Extract Options popup (see ExtractOptionsDialog). Seeded
        # from their last-used values (persisted in QSettings), falling back to EXTRACT_OPT_DEFAULTS.
        self._extract_opts = self._load_extract_opts()

        # Embedded pipeline preview + progress bar: the chosen video's first frame, swapped for
        # the pipeline's live annotated frames (plus a progress bar/ETA) while a run is active.
        # See preview.py for the file-polling scheme.
        self._preview_pane = PreviewProgressPane(self._settings)

        # MKV/AVI → MP4 conversion offers (see convert.py): whatever file the user ends up
        # with: converted, reused, or the source as-is: is adopted like any picked file.
        self._converter = Mp4Converter(self)
        self._converter.fileReady.connect(self._finish_selection)

        # Optional CUDA-PyTorch GPU acceleration (Windows). The manager drives the download and
        # the compute-status readout; a one-time offer is made after the window is up (see
        # _maybe_offer_gpu_on_first_run), scheduled at the end of __init__.
        self._gpu = GpuSupportManager(self)
        self._gpu.declined.connect(self._suppress_gpu_first_run_offer)
        self._gpu.no_gpu_found.connect(self._suppress_gpu_first_run_offer)
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
        # The screen currently on top, so a page switch (and close) knows which screen's geometry
        # to save. The window starts on the analysis page.
        self._active_screen = 0
        # On a page switch, swap in that screen's own remembered window geometry (and clamp it
        # back on-screen: entering Review grows the minimum, which can push it past the edge).
        self._stack.currentChanged.connect(self._on_screen_changed)
        self._update_title(self._stack.currentIndex())
        # currentChanged doesn't fire at construction, so seed the Tools surfaces' visibility for
        # the starting page (Analyze) explicitly: both surfaces exist now (_build_title_row ran).
        self._apply_tools_visibility(self._stack.currentIndex())
        self._refresh_nav()
        self._update_groupby_enabled()
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

    def _on_screen_changed(self, new_index: int) -> None:
        """On a page switch, stash the geometry of the screen we're leaving and restore the one
        we're entering (if it has a remembered size yet), then clamp on-screen."""
        # The Review-only Tools surfaces follow the visible page (set before the early-return so a
        # redundant signal still leaves them in the right state).
        self._apply_tools_visibility(new_index)
        if new_index == self._active_screen:
            return
        self._settings.setValue(self._geometry_key(self._active_screen), self.saveGeometry())
        self._active_screen = new_index
        # Re-floor the window for the screen we're entering BEFORE restoring its geometry:
        # minimumSizeHint() already reflects the new page. If we restored first, a wider screen's
        # leftover minimum (e.g. Review's ~1116) would clamp a narrower screen's restored width and
        # silently clobber it (the deferred re-floor afterwards couldn't bring the lost size back).
        self._apply_min_size()
        saved = self._settings.value(self._geometry_key(new_index))
        if saved:
            if not self.restoreGeometry(saved):
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
        if new_index == 1:  # entering Review: light up the export-output buttons by what's on disk
            self._refresh_export_outputs()

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
        self.resize(*self._default_size(self._active_screen))
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
        assistant = QAction("Upload Assistant…", self)
        assistant.setMenuRole(QAction.MenuRole.NoRole)
        assistant.triggered.connect(self._show_upload_assistant)
        file_menu.addAction(assistant)

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
        # again" is honored; on a source checkout it ends in offer_install's pip hint.
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

    def _refresh_export_selected_action(self) -> None:
        """Enable File → "Export selected set/game to a file…" only where it can act: on the
        Review page, with no subprocess running. The run state comes from self._running (not
        a live QProcess check) because _set_running(True) fires before the process object
        exists, when that check would still say idle. One combined check (re-run on page
        switches and run-state flips) so the two gates can't stomp each other's setEnabled."""
        self._export_selected_action.setEnabled(
            self._stack.currentIndex() == 1 and not self._running)

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
        self._title_note.setStyleSheet(f"color: {muted_text_color()};")
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
        into Source / Analysis / Sets boxes beside the live preview pane; each box (and the action
        row and status rows) is assembled by its own builder below."""
        page = QWidget()
        source_box   = self._build_source_box()
        analysis_box = self._build_analysis_box()
        sets_box     = self._build_sets_box()
        action_row   = self._build_action_row()
        self._build_status_rows()
        preview_col_widget = self._preview_pane

        # Left column: the option boxes stacked, with a trailing stretch that pins them to the top.
        # It's wrapped in a real widget (not a bare sub-layout) so QHBoxLayout lets it *expand* to
        # fill the row exactly like the preview widget does: both then start at the row's top, so
        # Source's top lines up with the Preview box's top. (An AlignTop'd or bare-sub-layout column
        # is instead handed a cell inset ~15px from the row top, which is what misaligned it.)
        controls = QVBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(source_box)
        controls.addWidget(analysis_box)
        controls.addWidget(sets_box)
        controls.addStretch(1)
        controls_col = QWidget()
        controls_col.setLayout(controls)

        # Top: option column beside the preview, both expanding to fill the row from its top.
        top_row = QHBoxLayout()
        top_row.addWidget(controls_col, stretch=0)
        top_row.addWidget(preview_col_widget, stretch=1)

        layout = QVBoxLayout(page)
        layout.addLayout(top_row, stretch=1)
        layout.addWidget(self._opts_warning_row)
        layout.addWidget(self._existing_analysis_label)
        layout.addLayout(action_row)
        return page

    def _build_source_box(self) -> QGroupBox:
        """Source: pick the vod and name the tournament."""
        choose_btn = QPushButton("Choose video…")
        choose_btn.clicked.connect(self._on_choose)
        # ElideMiddle + a 22-char floor: long filenames elide in the middle (extension stays
        # visible) without widening the Source column and squeezing the preview pane beside it.
        self._file_label = ElidingLabel("No video selected",
                                        mode=Qt.TextElideMode.ElideMiddle, min_chars=22)
        file_row = QHBoxLayout()
        file_row.addWidget(choose_btn)
        file_row.addWidget(self._file_label, stretch=1)

        # Prefixed to every clip title ("<tournament> <round> - A vs B"). Seeded from the vod's
        # filename when a video is chosen (see _on_choose); edits stick for that video. The
        # placeholder is set only once a video is chosen (see _seed_tournament_name), so an empty
        # field reads as blank: not "omitted from titles": before there's anything to analyze.
        self._tournament_edit = QLineEdit()
        self._tournament_prefilled_for: str | None = None
        tournament_row = QHBoxLayout()
        tournament_row.addWidget(QLabel("Tournament name:"))
        tournament_row.addWidget(self._tournament_edit, 1)

        source_box = QGroupBox("Source")
        source_layout = QVBoxLayout(source_box)
        source_layout.addLayout(file_row)
        source_layout.addLayout(tournament_row)
        return source_box

    def _build_analysis_box(self) -> QGroupBox:
        """Analysis: how the pipeline scans the vod."""
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
        ocr_row = QHBoxLayout()
        ocr_row.addWidget(QLabel("Detect Text:"))
        ocr_row.addWidget(self._ocr_interval_combo)
        ocr_row.addStretch(1)
        ocr_row.addWidget(self._make_info_button(
            "About analysis",
            "How often the analyzer models detect tags, characters, the timer, etc.\n\n"
            "Shorter intervals are more accurate, but the analysis is slower."))

        # Which engine reads the overlay text (see vodparse.ocr_engine). docTR ships with
        # the app, but stay robust to an environment without it: its item is greyed out
        # when the package isn't importable, and resolve_saved_engine keeps a stale saved
        # choice from preselecting it then. The selection handler offers the one-time model
        # download (a no-op when the frozen bundle carries the models): connected after
        # the restore below, so merely restoring a saved docTR choice never pops the offer.
        self._ocr_engine_combo = QComboBox()
        for label, engine_id in OCR_ENGINE_CHOICES.items():
            self._ocr_engine_combo.addItem(label, engine_id)
        if not doctr_available():
            doctr_idx = self._ocr_engine_combo.findData(OCR_ENGINE_DOCTR)
            engine_model = self._ocr_engine_combo.model()
            assert isinstance(engine_model, QStandardItemModel)  # QComboBox's default model
            doctr_item = engine_model.item(doctr_idx)
            if doctr_item is not None:
                doctr_item.setEnabled(False)
            self._ocr_engine_combo.setItemText(
                doctr_idx, self._ocr_engine_combo.itemText(doctr_idx) + ": not installed")
        saved_engine = self._settings.value(SETTING_OCR_ENGINE, OCR_ENGINE_DEFAULT_GUI, type=str)
        self._ocr_engine_combo.setCurrentIndex(self._ocr_engine_combo.findData(
            resolve_saved_engine(saved_engine, doctr_available())))
        self._ocr_engine_combo.currentIndexChanged.connect(self._on_ocr_engine_selected)
        self._ocr_engine_combo.currentIndexChanged.connect(self._refresh_engine_note)
        engine_row = QHBoxLayout()
        engine_row.addWidget(QLabel("Font style:"))
        engine_row.addWidget(self._ocr_engine_combo)
        engine_row.addStretch(1)
        # Standard is EasyOCR; Stylized is docTR (PARSeq), which reads English/Latin only and
        # downloads its models (~DOCTR_WEIGHTS_MB_APPROX MB) once if missing. The match clock is
        # always read with EasyOCR, regardless of this choice.
        engine_row.addWidget(self._make_info_button(
            "About font styles",
            "How onscreen text is read.\n\nStandard is faster and works with most overlays.\n\n"
            "Stylized is slower but works with \"fancy\" fonts (italics, comic sans, etc.).\n\n"
            "If it looks like you're missing text when using Standard, try Stylized."))

        # Muted note describing whichever font style is selected (see FONT_STYLE_NOTE_*): an
        # aside rather than something that needs acting on, so it stays quieter than the teal
        # _sets_note / _lang_note.
        self._engine_note = QLabel()
        self._engine_note.setStyleSheet(f"color: {muted_text_color()};")  # grey, theme-aware
        self._engine_note.setWordWrap(True)
        self._engine_note.setVisible(False)

        # Teal note naming the overlay language, shown only while a non-default (non-English)
        # one is selected in Preferences → Analysis. _refresh_lang_note reflects a saved choice
        # here and a change applied via the Settings dialog. Mirrors _sets_note's style.
        self._lang_note = QLabel()
        self._lang_note.setStyleSheet(f"color: {notice_color()};")  # teal, theme-aware
        self._lang_note.setWordWrap(True)
        self._lang_note.setVisible(False)

        analysis_box = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_box)
        analysis_layout.addLayout(ocr_row)
        analysis_layout.addLayout(engine_row)
        analysis_layout.addWidget(self._engine_note)
        analysis_layout.addWidget(self._lang_note)
        self._refresh_engine_note()  # reflect the restored font style on first build
        self._refresh_lang_note()    # reflect a restored non-English overlay language on first build
        return analysis_box

    def _build_sets_box(self) -> QGroupBox:
        """Sets: how matches group into exported sets."""
        self._match_radio = QRadioButton("Games")
        self._set_radio = QRadioButton("Sets")
        self._set_radio.setChecked(True)  # mirrors extract_matches default (tournament_set)
        self._mode_choice = QButtonGroup(self)
        self._mode_choice.addButton(self._match_radio)
        self._mode_choice.addButton(self._set_radio)
        self._set_radio.toggled.connect(self._update_groupby_enabled)
        choice_row = QHBoxLayout()
        choice_row.addWidget(QLabel("Export as:"))
        choice_row.addWidget(self._set_radio)
        choice_row.addWidget(self._match_radio)
        choice_row.addStretch(1)

        # Maps each label to the extract_matches --group-by value.
        self._groupby_radios = {
            "auto": QRadioButton("Auto"),
            "overlay": QRadioButton("Stream overlays"),
            "ingame": QRadioButton("Ingame tags"),
        }
        self._groupby_radios["auto"].setChecked(True)  # preserve current auto behavior
        self._groupby_choice = QButtonGroup(self)
        self._groupby_label = QLabel("Group sets by:")
        groupby_row = QHBoxLayout()
        groupby_row.addWidget(self._groupby_label)
        for radio in self._groupby_radios.values():
            self._groupby_choice.addButton(radio)
            groupby_row.addWidget(radio)
        groupby_row.addStretch(1)

        # Teal note shown while a non-default choice here is active (Game mode or ingame-tag
        # grouping): both change what gets exported, so the selection shouldn't go unnoticed.
        # Notice color, deliberately calmer than _opts_warning's bold amber "⚠".
        self._sets_note = QLabel()
        self._sets_note.setStyleSheet(f"color: {notice_color()};")  # teal, theme-aware
        self._sets_note.setWordWrap(True)
        self._sets_note.setVisible(False)
        self._match_radio.toggled.connect(self._refresh_sets_note)
        self._groupby_radios["ingame"].toggled.connect(self._refresh_sets_note)

        # "Set characters" (--set-characters) lives in the Advanced Extract Options popup (see ExtractOptionsDialog);
        # _selected_set_characters reads it from self._extract_opts.
        self._options_btn = QPushButton("Advanced Extract Options…")
        self._options_btn.clicked.connect(self._edit_extract_options)
        options_row = QHBoxLayout()
        options_row.addStretch(1)
        options_row.addWidget(self._options_btn)

        sets_box = QGroupBox("Sets")
        sets_layout = QVBoxLayout(sets_box)
        sets_layout.addLayout(choice_row)
        if SHOW_GROUP_BY_CONTROLS:
            sets_layout.addLayout(groupby_row)
        else:
            # The row stays out of the UI, so the never-shown radios keep their "Auto"
            # default and _selected_group_by() always resolves to "auto".
            for widget in (self._groupby_label, *self._groupby_radios.values()):
                widget.setVisible(False)
        sets_layout.addWidget(self._sets_note)
        sets_layout.addLayout(options_row)
        return sets_box

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
        """The two rows that sit just above the run buttons: the situational-options warning
        and the saved-analysis note. Both start hidden; _refresh_nav / the extract handlers
        toggle them."""
        # Appears only when a situational extract toggle (see SITUATIONAL_EXTRACT_OPTS) is
        # left on, so it can't silently carry into the next VOD.
        self._opts_warning = QLabel()
        self._opts_warning.setStyleSheet(f"color: {warning_color()}; font-weight: bold;")  # amber, theme-aware
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
            ocr_lang=self._settings.value(SETTING_OCR_LANG, OCR_LANG_DEFAULT_GUI, type=str))
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
        self._stack.setCurrentIndex(0)

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

    def _set_running(self, running: bool) -> None:
        """Disable the action buttons during a subprocess run; restore them after."""
        self._running = running
        for btn in (self._analyze_btn, self._resume_btn, self._options_btn):
            btn.setEnabled(not running)
        # Reread tags also kicks off a subprocess, so lock its Tools action during a run.
        self._reread_action.setEnabled(not running)
        self._refresh_export_selected_action()  # also Review-only; see the combined check
        self._stop_btn.setEnabled(running)  # inverse of the action buttons
        if not running:
            self._refresh_nav()  # reuse/resume buttons depend on artifacts existing

    # ── Set-grouping control state ─────────────────────────────────────────────
    def _update_groupby_enabled(self) -> None:
        """Set-grouping source only applies when exporting sets, not individual matches.
        (Set characters is likewise set-only but lives in the Advanced Extract Options popup now.)"""
        enabled = self._set_radio.isChecked()
        self._groupby_label.setEnabled(enabled)
        for radio in self._groupby_radios.values():
            radio.setEnabled(enabled)

    def _refresh_sets_note(self) -> None:
        """Show/hide the amber note under the Sets controls naming the non-default choice in
        effect: Game mode, or ingame-tag grouping (only mentioned in Set mode: Game mode greys
        the group-by radios out, so their value doesn't apply)."""
        if self._match_radio.isChecked():
            note = "Each game will be exported as its own clip (instead of grouping them into sets)"
        elif self._groupby_radios["ingame"].isChecked():
            note = "Sets will be grouped by the ingame tags instead of the stream overlay."
        else:
            note = ""
        self._sets_note.setText(note)
        self._sets_note.setVisible(bool(note))

    def _selected_group_by(self) -> str:
        """The --group-by value for the checked source radio (defaults to 'auto')."""
        return selected_radio_value(self._groupby_radios, "auto")

    def _selected_set_characters(self) -> str:
        """The --set-characters value chosen in the Advanced Extract Options popup (defaults to 'join')."""
        return self._extract_opts.get("set_characters", SET_CHARACTERS_DEFAULT)

    def _edit_extract_options(self) -> None:
        """Open the advanced-flags popup; it mutates self._extract_opts in place on OK. Persist the
        chosen flags so they're remembered as the defaults next time (and next launch)."""
        if ExtractOptionsDialog(self._extract_opts, self).exec() == QDialog.DialogCode.Accepted:
            self._save_extract_opts()
            self._refresh_opts_warning()

    def _load_extract_opts(self) -> dict:
        """Last-used extract options flags from QSettings, defaulting to EXTRACT_OPT_DEFAULTS
        (plus the string set_characters, tracked separately from the booleans)."""
        opts = {key: self._settings.value(f"extract/opt_{key}", default, type=bool)
                for key, default in EXTRACT_OPT_DEFAULTS.items()}
        opts["set_characters"] = self._settings.value(
            "extract/opt_set_characters", SET_CHARACTERS_DEFAULT, type=str)
        return opts

    def _save_extract_opts(self) -> None:
        for key, value in self._extract_opts.items():
            self._settings.setValue(f"extract/opt_{key}", value)

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
        if Path(path).suffix.lower() in UNPLAYABLE_CONTAINER_EXTS:
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
        """Adopt `path` as the current vod and refresh the Main page for it."""
        self._video_path = path
        self._file_label.setText(Path(path).name)
        self._seed_tournament_name()  # once per vod; survives a later return from Review
        self._preview_pane.set_video(self._video_path)
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
        """Confirm discarding Review edits before a Re-read tags regeneration. Prompts when there
        are edits to lose: either live edits not yet flushed to the draft (has_unsaved_edits) or a
        saved draft on disk: with re-read wording; returns True to proceed. Nothing to lose means
        no prompt and True. The plain draft-file check alone would miss edits made inside the
        debounced-save window, so the live-edit check is needed too."""
        video = self._video_path
        if video is None:
            return False
        if not self._review.has_unsaved_edits() and not review_draft_for(video).exists():
            return True
        reply = QMessageBox.question(
            self, "Reread tags?",
            f"Rereading the tags for {Path(video).name} regenerates its games and discards your "
            "current Review edits. Continue?",
        )
        return reply == QMessageBox.StandardButton.Yes

    def _on_ocr_engine_selected(self) -> None:
        """The Text-engine combo changed: choosing docTR before its models are downloaded
        offers the one-time download right away rather than surprising the user at Run.
        Declining, cancelling, or a failed download reverts the combo to EasyOCR so the
        choice never silently points at an engine that can't read anything yet."""
        if self._ocr_engine_combo.currentData() != OCR_ENGINE_DOCTR or doctr_weights_cached():
            return
        self._offer_doctr_fetch(lambda ok: None if ok else self._revert_text_engine())

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

    def _revert_text_engine(self) -> None:
        """Put the Text-engine combo back on the default (EasyOCR)."""
        self._ocr_engine_combo.setCurrentIndex(
            self._ocr_engine_combo.findData(OCR_ENGINE_DEFAULT_GUI))

    def _offer_doctr_fetch(self, on_done: Callable[[bool], None]) -> None:
        """Ask before the one-time docTR model download, then run it. ``on_done`` receives
        True only when the models are in the cache afterwards: False on decline, cancel,
        or failure: so callers can revert the engine choice or continue into a run."""
        reply = QMessageBox.question(
            self, "Download text models?",
            f"docTR needs its text models (a one-time ~{DOCTR_WEIGHTS_MB_APPROX} MB download) "
            "before it can read overlays. Download now?")
        if reply != QMessageBox.StandardButton.Yes:
            on_done(False)
            return
        self._start_doctr_fetch(on_done)

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
        """Analyze VOD: run the pipeline, then (on success) extract sets and open Review. The one
        up-front confirm covers overwriting an existing analysis and/or saved edits; after that the
        pipeline→extract→Review chain runs unattended. The argv itself is built by
        commands.pipeline_argv; this method just resolves the widget/QSettings state it needs."""
        video = self._video_path
        if video is None:
            return
        if not self._confirm_overwrite_analysis():
            return

        ocr_engine = self._ocr_engine_combo.currentData()
        ocr_lang = self._settings.value(SETTING_OCR_LANG, OCR_LANG_DEFAULT_GUI, type=str)
        if ocr_engine == OCR_ENGINE_DOCTR and ocr_lang != OCR_LANG_DEFAULT_GUI:
            # The pipeline's own validate_engine_lang would stop the subprocess anyway;
            # catching it here turns a failed run into an actionable dialog.
            QMessageBox.warning(
                self, "Stylized reads English only",
                "The Stylized font style reads English/Latin text only. Switch "
                "the overlay language back to English in Settings → Analysis, or set the Font style "
                "to Standard.")
            return

        self._clear_log()
        ocr_interval = self._ocr_interval_combo.currentData()
        self._settings.setValue(SETTING_OCR_INTERVAL_SEC, ocr_interval)  # remember for next run
        self._settings.setValue(SETTING_OCR_ENGINE, ocr_engine)          # remember for next run
        # The OCR speed-ups and the in-game nametag band live in Preferences (SettingsDialog);
        # read saved values. The band frac rides only on the pipeline argv: it is baked into the
        # pipeline JSON, which extract_matches reads back (see resolve_ingame_tag_band_frac).
        gate_ocr = self._settings.value(SETTING_GATE_OCR_ON_TIMER, True, type=bool)
        roi_cache = self._settings.value(SETTING_OCR_ROI_CACHE, True, type=bool)
        ingame_tag_frac = self._settings.value(
            SETTING_INGAME_TAG_BAND_FRAC, INGAME_TAG_BAND_FRAC_DEFAULT, type=float)
        argv = pipeline_argv(
            video, ocr_interval=ocr_interval, gate_ocr=gate_ocr, roi_cache=roi_cache,
            ocr_lang=ocr_lang, ocr_engine=ocr_engine, ingame_tag_frac=ingame_tag_frac,
            preview_file=self._preview_pane.preview_file,
            progress_file=self._preview_pane.progress_file,
        )
        if ocr_engine == OCR_ENGINE_DOCTR and not doctr_weights_cached():
            # Backstop for a saved docTR choice whose models were never (or no longer are)
            # downloaded: the combo's own handler normally offered this at selection time.
            self._offer_doctr_fetch(lambda ok: self._start_pipeline_run(argv) if ok else None)
            return
        self._start_pipeline_run(argv)

    def _start_pipeline_run(self, argv: list[str]) -> None:
        """Launch the pipeline stage: the tail of _on_run, split out so a docTR model
        download can finish first and then continue straight into the run."""
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
        the Advanced Extract Options are used. The global --debug preference is appended either way.
        The argv itself is built by commands.extract_argv."""
        video = self._video_path
        if video is None:
            return
        pipeline_json = pipeline_json_for(video)
        mode: Literal["match", "tournament_set"] = (
            "match" if self._match_radio.isChecked() else "tournament_set")
        # Re-read tags supplies its own per-VOD tag flags; otherwise use the Advanced Extract
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
        self._extract_was_smart = "--smart-tags" in argv

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

    def _show_smart_tags_summary(self, video: str) -> None:
        """Pop the smart-tags outcome: the winning flag combination and the runner-up, with
        their scores: read from the report the extract run just wrote. (The full 16-row
        ranking table went to the log transcript: Debug → Show log….) Defensive on a
        missing/malformed report: the popup is informational and must never block Review."""
        try:
            report = json.loads(
                (output_dir_for(video) / SMART_TAGS_REPORT_NAME).read_text(encoding="utf-8"),
            )
            lines = [_smart_entry_text("Picked", report["winner"])]
            if report.get("runner_up"):
                lines.append(_smart_entry_text("Runner-up", report["runner_up"]))
        except (OSError, ValueError, KeyError, TypeError):
            return
        QMessageBox.information(self, "Smart tag reading",
                                "\n\n".join(lines) + "\n\nFull toggle ranking: Debug → Show log…")

    def _on_reread_tags(self) -> None:
        """Re-read tags (Review screen): re-run the cheap --csv-only extraction for THIS vod under
        different tag-reading flags and refresh Review in place. The dialog is seeded from the flags
        that produced the current matches.json (the per-VOD sidecar), not the global options, and
        its choice drives this one run only: the global extract/opt_* settings are left untouched
        (contrast _edit_extract_options, which persists). Mode/group-by/set-characters keep
        coming from the Main-page widgets, the same source _load_review reads when it re-renders."""
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
        target = self._review.export_target()
        if target is not None and self._current_export_csv() != export_csv_for(*target,
                                                                              export_root()):
            self._review.show_status(
                "Already-exported clips keep the previous name. Export again to apply the new one.")

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
        mode = "match" if self._match_radio.isChecked() else "tournament_set"
        self._review.load_model(matches, mode, self._selected_group_by(),
                                video, stem,
                                tournament_name=self._tournament_edit.text().strip(),
                                set_characters=self._selected_set_characters())
        self._append("\n> Loaded games. Review, then Export clips.")
        self._stack.setCurrentIndex(1)
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
        self._stack.setCurrentIndex(1)
        self._relabel_export_selected_action()
        return True

    def load_debug_vod(self, vod: str, resume: bool = False) -> None:
        """--debug entry point: preselect the vod and jump to Review. By default it loads fresh
        from the vod's matches JSON; with `resume` (--debug-resume) it reopens the saved edit draft
        instead, falling back to a fresh load if no draft exists."""
        self._video_path = vod
        self._file_label.setText(Path(vod).name)
        self._preview_pane.set_video(self._video_path)
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
        if self._process_running():
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
        if previous is not None and previous.csv.exists():
            # Carry the previous export records through the rewrite: if this export fails, the
            # export-output buttons can still fall back to the old clips (resolve_export_csv);
            # a successful export writes the new records over them (see vodparse.cli.export_reviewed).
            data["export_folder"] = previous.csv.parent.name
            data["export_csv"] = previous.csv.name
        # A rename moves the export to a new folder, silently duplicating the clips' disk
        # space: remember the superseded folder and offer to delete it on success.
        self._stale_export_dir = stale_export_dir(previous, self._export_dir, data["mode"],
                                                  self._export_root)
        write_json_atomic(out, data)
        self._append(f"\n> Exporting clips from {out.name}…\n")
        self._set_running(True)
        self._exporting = True
        # Size the bar from the payload that actually gets exported: build_reviewed_data drops
        # excluded sets, and the subprocess prints one "  Saved:" line per surviving clip, so
        # len(data["clips"]) is exactly how many lines will arrive (finding S2).
        self._review.set_exporting(True, len(data["clips"]))
        self._review.set_upload_available(False)  # don't open a half-written CSV mid-export
        self._start_process(["-m", "vodparse.cli.export_reviewed", "--input", str(out),
                             "--out-root", str(self._export_root)],
                            self._on_export_finished)

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
        Open-folder / Upload-Assistant shortcuts."""
        if self._consume_stop():
            return
        self._exporting = False
        # Consume the superseded-folder capture from kickoff (failure discards it: the old
        # export is then still the only complete one, so it must not be offered for deletion).
        stale = self._stale_export_dir
        self._stale_export_dir = None
        self._set_running(False)
        self._review.set_exporting(False)
        self._refresh_export_outputs()  # exported clips/folder now (likely) exist; reflect on buttons
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
        box.exec()
        clicked = box.clickedButton()
        if stale is not None and delete_box is not None and delete_box.isChecked():
            if self._delete_stale_export(stale):
                self._retarget_assistant_after_delete(stale)
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
            QMessageBox.warning(
                self, "Couldn't delete previous export",
                f"“{folder.name}” couldn't be fully deleted:\n{err}\n\n"
                "A file inside may be open in another program.")
            return False
        self._append(f"\n> Deleted previous export {folder}.")
        return True

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
        self._review.set_exporting(False)
        self._preview_pane.stop_run()
        self._set_running(False)
        self._append("\n⏹ Stopped.")
        return True

    def _stop_process(self) -> None:
        """Shut down the child process, politely first, so quitting never orphans it."""
        if not self._process_running():
            return
        assert self._proc is not None  # implied by _process_running()
        self._proc.terminate()
        if not self._proc.waitForFinished(STOP_TERM_WAIT_MS):
            self._proc.kill()
            self._proc.waitForFinished(STOP_KILL_WAIT_MS)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self._settings.setValue(self._geometry_key(self._active_screen), self.saveGeometry())
        # Before anything that spins a nested event loop (the prompt below): a resolving
        # update check would otherwise deliver its queued verdict inside that loop and pop
        # a dialog over the quit confirmation. If the user then answers No, the dismissed
        # check revives on the next Help ▸ Check for Updates click.
        self._updates.dismiss()
        if self._process_running():
            answer = QMessageBox.question(
                self, "Quit?",
                "A run is still in progress. Quit anyway?\n\n"
                "The current analysis will be stopped and its progress lost.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                if event is not None:
                    event.ignore()
                return
            self._stopping = True  # route the synchronous finish through _consume_stop, not the failure path
            self._stop_process()
        self._teardown_background_work()
        if event is not None:
            event.accept()

    def _teardown_background_work(self) -> None:
        """Stop the window's background workers before Qt teardown. Shared by ``closeEvent``
        and ``force_quit``: a teardown listed in only one quit path resurfaces as a shutdown
        crash (a ``QThread`` destroyed while running aborts the app)."""
        self._converter.cancel_active()   # don't leave an ffmpeg conversion running past the window
        self._gpu.cancel_active_probes()  # nor a torch-importing GPU probe
        self._updates.shutdown()          # a check thread destroyed while running aborts teardown

    def force_quit(self) -> None:
        """Quit without the confirmation prompt (Ctrl+C in the terminal)."""
        self._stop_process()
        self._teardown_background_work()
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultimate VOD Studio GUI")
    parser.add_argument("--debug", metavar="VOD", default=None,
                        help="Load this vod and jump straight to the Review screen if its "
                             "matches JSON already exists (otherwise opens normally).")
    parser.add_argument("--debug-resume", action="store_true",
                        help="With --debug, reopen the saved Review edit draft instead of loading "
                             "fresh from the extract (falls back to fresh if no draft exists).")
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
    app.setApplicationName("Ultimate VOD Studio")
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
    if args.debug:
        window.load_debug_vod(args.debug, resume=args.debug_resume)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
