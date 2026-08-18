"""Dialogs for the desktop GUI (app.py): Advanced Extract Options, the log popup, and Settings.

Import these only after ``qt_plugins.configure_qt_plugins()`` has run (every app entry
module does so at its top), since this module imports PyQt6 widgets at load time.
"""

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, QTime, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from vodparse import PUBLIC_REPO, __version__
from vodparse.gui.common import (
    bold_label,
    copy_with_feedback,
    muted_text_color,
    selected_radio_value,
)
from vodparse.gui.export_root import default_export_root
from vodparse.gui.gpu_support import GpuStatusPanel
from vodparse.gui.paths import app_icon_path, third_party_licenses_path, vod_picker_start_dir
from vodparse.gui.settings import (
    FONT_SCALE_CHOICES,
    FONT_SCALE_DEFAULT,
    OCR_LANG_CHOICES,
    OCR_LANG_DEFAULT_GUI,
    SET_CHARACTERS_DEFAULT,
    SETTING_CONFIRM_GROUP,
    SETTING_DEFAULT_VOD_DIR,
    SETTING_EXPORT_DIR,
    SETTING_FONT_SCALE,
    SETTING_GATE_OCR_ON_TIMER,
    SETTING_GPU_SKIP_FIRST_RUN_OFFER,
    SETTING_INGAME_TAG_BAND_FRAC,
    SETTING_OCR_LANG,
    SETTING_OCR_ROI_CACHE,
    SETTING_SHOW_DEBUG_MENU,
    SETTING_SHOW_ETA,
    SETTING_SHOW_FONT_STYLE_NOTE,
    SETTING_TITLE_CHAR_LIMIT,
    SETTING_TITLE_FILENAMES_MATCHES,
    SETTING_TITLE_FILENAMES_SETS,
    SETTING_UPLOAD_DESC_TEMPLATE,
    SETTING_UPLOAD_TITLE_TEMPLATE,
    TITLE_CHAR_LIMIT_DEFAULT,
)
from vodparse.gui.review.flags import (
    LONG_LEADIN_INTERVALS_DEFAULT,
    MAX_MATCH_SEC_DEFAULT,
    MIN_MATCH_SEC_DEFAULT,
    SETTING_FLAG_LONG_LEADIN,
    SETTING_FLAG_LONG_MATCHES,
    SETTING_FLAG_SHORT_MATCHES,
    SETTING_FLAG_WEIRD_TIMER,
    SETTING_LONG_LEADIN_INTERVALS,
    SETTING_MAX_MATCH_SEC,
    SETTING_MIN_MATCH_SEC,
)
from vodparse.gui.review.model import (
    SETTING_MARK_INGAME_FALLBACK,
)
from vodparse.runtime_paths import workspace_dir
from vodparse.tags import INGAME_TAG_BAND_FRAC_DEFAULT, INGAME_TAG_BAND_FRAC_FLOOR


def _band_pct_from_frac(frac: float) -> int:
    """The in-game nametag band's height as a whole-number percent of frame height, for the
    Settings spinbox: the stored top-edge fraction 0.92 (bottom 8%) shows as 8, 0.86 as 14."""
    return round((1.0 - frac) * 100)


def _band_frac_from_pct(pct: int) -> float:
    """Inverse of _band_pct_from_frac: the spinbox's height-percent back to the stored top-edge
    fraction the pipeline/CLI take (8 % -> 0.92). Rounded to 2 dp to match --ingame-tag-band-frac."""
    return round(1.0 - pct / 100, 2)


class TagReadingOptions(QWidget):
    """The Smart/Manual tag-reading control: the two-radio choice plus the four manual
    overlay-reading toggles, with the toggles greyed out under Smart.

    Shared verbatim by the Advanced Extract Options dialog and the Review screen's reread
    tags dialog, so the carefully-worded checkbox labels and tooltips live in exactly one
    place and can't drift between the two. ``options`` seeds the initial state and
    ``values()`` reads it back: both keyed by the five settings.TAG_OPTION_KEYS
    (``smart_tags`` plus the four flag keys). The widget carries no margins, so it embeds
    flush in a host dialog's layout.

    ``optionsChanged`` fires whenever the user changes any of the five controls, for hosts
    that react live to edits (the overlay spot-check tool rereads its screenshot on every
    toggle); the OK/Cancel dialogs here just ignore it and read ``values()`` on accept.
    """

    optionsChanged = pyqtSignal()

    def __init__(self, options: dict, parent=None) -> None:
        super().__init__(parent)

        # Smart vs Manual tag reading. Smart runs extract_matches --smart-tags, which tries
        # every combination of the four flag toggles below over the same analysis and keeps
        # the best-scoring result: so the toggles are inert (greyed out) while it's selected.
        self._smart_tags = QRadioButton("Try every combination and keep the best one")
        self._smart_tags.setToolTip("Can sometimes pick the wrong combination")
        self._manual_tags = QRadioButton("Manual")
        self._tagmode_choice = QButtonGroup(self)
        for radio in (self._smart_tags, self._manual_tags):
            self._tagmode_choice.addButton(radio)
        (self._smart_tags if options.get("smart_tags") else self._manual_tags).setChecked(True)

        self._filter_banners = QCheckBox("Filter stream banners")
        self._filter_banners.setChecked(options["filter_banners"])

        # The static-overlay filter keeps only tag text detected at the same spot in at least
        # half a match's OCR frames (dropping drifting in-game scenery). On busy/animated
        # overlays with sparse OCR the real gamertag flickers in and out of detection and gets
        # dropped too: so this checkbox turns the filter off (--no-overlay-filter).
        self._no_overlay_filter = QCheckBox("Keep tags that are inconsistently detected")
        self._no_overlay_filter.setChecked(options["no_overlay_filter"])
        self._no_overlay_filter.setToolTip(
            "The overlay filter drops tags not detected in at least half of a "
            "game's frames. Some busy or animated overlays can mess with this. "
            "Turn this on if it keeps missing tags on the overlay")

        # Rare layout: some tournaments render their overlay (gamertags + sponsors) over the
        # in-game HUD at the bottom instead of the top. When ticked, the overlay is read from
        # the bottom of the frame so its sponsors are captured.
        self._overlay_at_bottom = QCheckBox("Tournament overlay is at the bottom of the screen")
        self._overlay_at_bottom.setChecked(options["overlay_at_bottom"])

        # Rare layout: some overlays stack a metadata line (location / Twitter handle / seed)
        # above each gamertag. When ticked, the gamertag is read from the line below it instead
        # of the top line (which is otherwise mistaken for the tag).
        self._overlay_stacked = QCheckBox("Stacked overlay")
        self._overlay_stacked.setChecked(options["overlay_stacked"])
        self._overlay_stacked.setToolTip(
            "Some overlays have extra data (handles, locations, seeds, etc.)  "
            "above each tag. When checked, the reader tries to read around that extra data "
            "(which is sometimes mistaken for the tag).")

        # The manual toggles only apply in Manual mode; grey them out under Smart so it's
        # visible that the sweep ignores them. Wired after all four exist.
        self._smart_tags.toggled.connect(self._apply_tag_mode)
        self._apply_tag_mode(self._smart_tags.isChecked())

        # Any user edit → optionsChanged. Only the Smart radio of the exclusive pair is
        # connected, or every mode flip would fire twice (once per radio).
        self._smart_tags.toggled.connect(self.optionsChanged)
        for box in (self._filter_banners, self._overlay_at_bottom,
                    self._overlay_stacked, self._no_overlay_filter):
            box.toggled.connect(self.optionsChanged)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)   # embed flush in the host dialog's layout
        layout.addWidget(self._smart_tags)
        layout.addWidget(self._manual_tags)
        # The four toggles sit indented under the Manual radio they belong to.
        manual_toggles = QVBoxLayout()
        manual_toggles.setContentsMargins(24, 0, 0, 0)    # indent under the Manual radio
        for box in (self._filter_banners, self._overlay_at_bottom,
                    self._overlay_stacked, self._no_overlay_filter):
            manual_toggles.addWidget(box)
        layout.addLayout(manual_toggles)

    def _apply_tag_mode(self, smart: bool) -> None:
        """Enable/disable the four manual tag toggles to match the Smart/Manual choice."""
        for box in (self._filter_banners, self._overlay_at_bottom,
                    self._overlay_stacked, self._no_overlay_filter):
            box.setEnabled(not smart)

    def values(self) -> dict:
        """The chosen tag options as the five TAG_OPTION_KEYS (smart_tags + the four flags)."""
        return {
            "smart_tags": self._smart_tags.isChecked(),
            "filter_banners": self._filter_banners.isChecked(),
            "overlay_at_bottom": self._overlay_at_bottom.isChecked(),
            "overlay_stacked": self._overlay_stacked.isChecked(),
            "no_overlay_filter": self._no_overlay_filter.isChecked(),
        }


class ExtractOptionsDialog(QDialog):
    """Advanced options for the extract step, edited in a popup.

    Most options here change the matches JSON that extract writes (tag voting and match
    boundaries). The clip-export-only flags (start/end buffers, tournament name, output
    folder) do nothing in the GUI's ``--csv-only`` extract and are applied later at
    Review/Export, so they're intentionally absent. The exception is "Set characters"
    (--set-characters), a set-grouping choice that's also applied at Review/Export but lives
    here next to the other extract knobs; it's a string ("join"/"majority"), not a bool.
    ``options`` (a flat dict keyed by the extract_matches flag name) is mutated in place when
    the dialog is accepted.
    """

    def __init__(self, options: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Extract Options")
        self._options = options

        # The Smart/Manual tag-reading control (radios + four toggles + greying) is shared
        # with the Review screen's reread tags dialog, so it lives in one widget.
        self._tag_options = TagReadingOptions(options, self)

        # How a set's per-player character field reads when they switched characters between
        # games. Maps each radio to the extract_matches --set-characters value. Only affects
        # set exports, but harmless in match mode (the flag is dropped there).
        self._setchars_radios = {
            "join": QRadioButton("Show all played characters in the set"),
            "majority": QRadioButton("Show only the most played character in the set"),
        }
        self._setchars_choice = QButtonGroup(self)
        for radio in self._setchars_radios.values():
            self._setchars_choice.addButton(radio)
        self._setchars_radios.get(
            options.get("set_characters", SET_CHARACTERS_DEFAULT),
            self._setchars_radios["join"]).setChecked(True)

        self._debug = QCheckBox("Save debug tables")
        self._debug.setChecked(options["debug"])

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._section("Tag reading"))
        layout.addWidget(self._tag_options)
        layout.addWidget(self._section("Character Detection"))
        for radio in self._setchars_radios.values():
            layout.addWidget(radio)
        layout.addWidget(self._section("Diagnostics"))
        layout.addWidget(self._debug)
        layout.addSpacing(8)
        layout.addWidget(buttons)

    @staticmethod
    def _section(text: str) -> QLabel:
        """A small bold heading separating the checkbox groups."""
        return bold_label(text)

    def accept(self) -> None:
        self._options.update(
            **self._tag_options.values(),
            debug=self._debug.isChecked(),
            set_characters=self._selected_setchars(),
        )
        super().accept()

    def _selected_setchars(self) -> str:
        """The --set-characters value for the checked radio (defaults to 'join')."""
        return selected_radio_value(self._setchars_radios, SET_CHARACTERS_DEFAULT)


class TournamentNameDialog(QDialog):
    """Change the tournament name after the fact (Review's Tools menu, and the upload
    assistant's Options menu).

    One text box, prefilled with the current name. The dialog only collects the new name:
    each caller applies it to its own world (the review session's meta, or the assistant's
    loaded CSV rows). `caption` overrides the greyed hint line for callers whose scope
    differs from the Review default.
    """

    def __init__(self, current_name: str, parent: QWidget | None = None, *,
                 caption: str = "Used in clip titles, the export folder name, "
                                "and the metadata CSV.") -> None:
        super().__init__(parent)
        self.setWindowTitle("Change tournament name")
        self.setMinimumWidth(420)

        self._edit = QLineEdit(current_name)
        self._edit.selectAll()  # the common case is replacing the whole name

        caption_label = QLabel(caption)
        caption_label.setWordWrap(True)
        caption_label.setEnabled(False)   # greyed, so it reads as a hint rather than a control

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Tournament name"))
        layout.addWidget(self._edit)
        layout.addWidget(caption_label)
        layout.addWidget(buttons)

    def name(self) -> str:
        """The entered tournament name, whitespace-trimmed."""
        return self._edit.text().strip()


# App-icon edge length in the About dialog, in logical px (scaled with a smooth transform
# so it stays crisp on HiDPI displays).
_ABOUT_LOGO_PX = 96


class AboutDialog(QDialog):
    """The Help ▸ About box (opened from app.py's ``_show_about``; on macOS the menu
    action's AboutRole relocates it into the application menu).

    A read-only splash — the app icon, name, version (from ``vodparse.__version__``), and
    clickable repo/license links. Nothing here mutates state, so it just closes.
    """

    _REPO_URL = f"https://github.com/{PUBLIC_REPO}"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About Ultimate VOD Studio")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # The same icon app.py sets as the window/taskbar icon. Skipped when assets/ isn't
        # present (a source checkout without the bundled icon); see app_icon_path's guard note.
        icon_path = app_icon_path()
        if icon_path.exists():
            pixmap = QPixmap(str(icon_path)).scaled(
                _ABOUT_LOGO_PX, _ABOUT_LOGO_PX,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            logo = QLabel()
            logo.setPixmap(pixmap)
            logo.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            layout.addWidget(logo)

        title = bold_label("Ultimate VOD Studio")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(title)

        version = QLabel(f"Version {__version__}")
        version.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        version.setStyleSheet(f"color: {muted_text_color()};")
        layout.addWidget(version)

        # Rich-text anchors; setOpenExternalLinks hands the click straight to the OS
        # browser (cross-platform), so no per-link click wiring is needed.
        links = QLabel(
            f'<a href="{self._REPO_URL}">Project page</a> &nbsp;·&nbsp; '
            f'<a href="{self._REPO_URL}/blob/main/LICENSE">License (AGPL-3.0-or-later)</a>')
        links.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        links.setOpenExternalLinks(True)
        layout.addWidget(links)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        # The bundled third-party license texts (GPL-3.0 for PyQt6, LGPL for Qt 6,
        # python-bidi and OpenCV's video backend) must not just ship with the app but be
        # reachable from it. Shown only when the document is actually present: a source
        # checkout that never ran `license_report.py bundle` has none, and a dead button is
        # worse than none. Same existence guard as the icon above.
        if third_party_licenses_path().is_file():
            licenses_btn = buttons.addButton("Licenses",
                                             QDialogButtonBox.ButtonRole.ActionRole)
            assert licenses_btn is not None  # addButton(text, role) always returns one
            licenses_btn.setAutoDefault(False)  # Close stays the Enter/Escape action
            licenses_btn.clicked.connect(self._open_licenses)
        layout.addWidget(buttons)

    @staticmethod
    def _open_licenses() -> None:
        """Open the bundled third-party license texts in the OS text viewer."""
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(third_party_licenses_path())))


class RereadTagsDialog(QDialog):
    """Reread the tournament tags for the video being reviewed under different flags.

    A Review-only editor. It seeds the same Smart/Manual control as the Advanced Extract
    Options popup (see TagReadingOptions) from the flags that actually produced the current
    matches.json, and on OK those flags drive a one-off regeneration of THIS video's matches
    only. The choices are **never persisted**: unlike the Advanced Extract Options popup,
    they don't touch the global ``extract/opt_*`` settings used for the next VOD. ``options``
    (the five TAG_OPTION_KEYS) is mutated in place on accept, matching ExtractOptionsDialog's
    in-place contract; the set-grouping/character/mode knobs stay on the Main page
    and are deliberately not editable here.
    """

    def __init__(self, options: dict, vod_name: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Reread tags: {vod_name}")
        self._options = options
        self._tag_options = TagReadingOptions(options, self)

        caption = QLabel("Regenerates the sets/games for this video and reloads the review window.")
        caption.setWordWrap(True)
        caption.setEnabled(False)   # greyed, so it reads as a hint rather than a control

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._tag_options)
        layout.addWidget(caption)
        layout.addSpacing(8)
        layout.addWidget(buttons)

    def accept(self) -> None:
        self._options.update(self._tag_options.values())
        super().accept()


class LogDialog(QDialog):
    """Resizable, non-modal popup showing the full run transcript: the only place the log is
    viewed now. Stays live during a run: MainWindow._append mirrors new output here via append()
    while it's open."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Log")
        self.resize(800, 500)
        self._view = QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setPlainText(text)
        self._scroll_to_end()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        # ActionRole keeps Copy from triggering accept/reject; the box places it appropriately.
        copy_btn = buttons.addButton("Copy", QDialogButtonBox.ButtonRole.ActionRole)
        assert copy_btn is not None
        self._copy_btn: QPushButton = copy_btn
        self._copy_btn.clicked.connect(self._copy)

        layout = QVBoxLayout(self)
        layout.addWidget(self._view)
        layout.addWidget(buttons)

    def _scroll_to_end(self) -> None:
        self._view.moveCursor(self._view.textCursor().MoveOperation.End)

    def _copy(self) -> None:
        """Copy the full transcript to the clipboard, with brief 'Copied' feedback on the button."""
        copy_with_feedback(self._view.toPlainText(), self._copy_btn)

    def append(self, text: str) -> None:
        self._view.moveCursor(self._view.textCursor().MoveOperation.End)
        self._view.insertPlainText(text)
        self._scroll_to_end()

    def clear(self) -> None:
        self._view.clear()


def _duration_edit(default_sec: float, min_sec: int, max_sec: int) -> QTimeEdit:
    """A mm:ss field for a duration setting. The value is entered as minutes + seconds (more
    intuitive than raw seconds) but stored back as seconds: see _duration_seconds.

    Durations must be under an hour: _qtime packs the whole minute count into QTime's minute field,
    which only accepts 0-59, so a >= 3600s bound would build an INVALID QTime that setMinimumTime /
    setTime then silently ignore (leaving the wrong 00:00 bounds). The mm:ss display format assumes
    the same. Assert it so a future caller passing a >= 1h limit fails loudly instead of misconfiguring."""
    assert max_sec < 3600, "_duration_edit is mm:ss only (durations under an hour)"

    def _qtime(total: int) -> QTime:
        return QTime(0, total // 60, total % 60)

    edit = QTimeEdit()
    edit.setDisplayFormat("mm:ss")
    edit.setMinimumTime(_qtime(min_sec))
    edit.setMaximumTime(_qtime(max_sec))
    edit.setTime(_qtime(int(default_sec)))
    return edit


def _duration_seconds(edit: QTimeEdit) -> int:
    """The mm:ss field's value as whole seconds (the form persisted to QSettings)."""
    t = edit.time()
    return t.minute() * 60 + t.second()


# Fixed opening size for the Settings dialog. sizeHint() is unreliable once the tabs hold
# word-wrapped hint labels, and sizing to the content would let the tallest tab inflate every
# tab. The dialog stays resizable, and each tab scrolls (see _scrollable), so a tab whose
# content outgrows this height scrolls rather than truncating its hint text.
_SETTINGS_DIALOG_WIDTH = 600
_SETTINGS_DIALOG_HEIGHT = 640

# The whole-UI "Text size" control is built but kept off the General tab: it works, but nothing
# has needed it yet, so it isn't worth the surface area until someone asks for it. Flip this to
# True to show the row again — the control, its saved value (SETTING_FONT_SCALE) and the startup
# code that applies it (apply_font_scale in app.py) are all still wired up and unchanged.
_SHOW_FONT_SCALE_SETTING = False

# Vertical gap between the titled groups on a tab. A QVBoxLayout's default (6 px) leaves adjacent
# group frames looking joined together, so each page sets this once via _page rather than pushing
# the groups apart with per-gap addSpacing() calls, which is how the tabs drifted apart before.
_TAB_GROUP_SPACING = 18


class SettingsDialog(QDialog):
    """Persistent app settings, grouped into a tab per GUI page. Toggles are read from / written
    to QSettings (cross-platform)."""

    def __init__(self, settings: QSettings, parent=None, *,
                on_reset_layout: Callable[[], None] | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self._settings = settings
        self._on_reset_layout = on_reset_layout

        tabs = QTabWidget()
        # No Extract tab currently: its only setting (clear-draft-on-extract) was removed in favor
        # of unconditional clearing behind the discard-edits warnings (see _confirm_reread_discard /
        # _confirm_overwrite_analysis). To bring it back for future settings:
        # tabs.addTab(self._extract_tab(), "Extract") plus the usual SETTING_* constant / control /
        # accept() persistence / MainWindow reader.
        tabs.addTab(self._scrollable(self._general_tab()), "General")
        tabs.addTab(self._scrollable(self._analysis_tab()), "Analysis")
        tabs.addTab(self._scrollable(self._review_tab()), "Review")
        tabs.addTab(self._scrollable(self._export_tab()), "Export")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

        self.resize(_SETTINGS_DIALOG_WIDTH, _SETTINGS_DIALOG_HEIGHT)

    @staticmethod
    def _scrollable(page: QWidget) -> QScrollArea:
        """Wrap a tab page so its content is never squeezed shorter than it needs to be.

        A word-wrapped QLabel reports a minimum height of roughly one line: it can't tell the
        layout how tall it actually needs to be at its final width. So whenever a tab's content
        wants more room than this fixed-height dialog gives it, the QVBoxLayout is free to shrink
        the hint rows below their wrapped height and the last line is silently cut off. It bites
        first on the tallest tabs and on platforms/text sizes with roomier font metrics (macOS,
        and anything the Text size control scales up), which is exactly where it's hardest to
        notice while developing.

        Inside a widget-resizable scroll area the page always gets its full preferred height and
        the dialog scrolls instead of truncating. With the content shorter than the dialog (the
        common case) there's no scrollbar, and the rest of the body keeps the wrapper itself from
        showing: NoFrame drops its border, and clearing autoFillBackground on both the viewport
        and the page stops them painting an opaque Window-coloured rectangle over the tab pane.
        That fill is invisible under Fusion, where the pane happens to be the same flat colour,
        but macOS draws its tab pane differently and the rectangle reads as a box around the
        whole tab. Transparent instead, the pane shows through and the tab looks unwrapped.
        """
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.Shape.NoFrame)
        viewport = area.viewport()
        assert viewport is not None      # QAbstractScrollArea always builds one in its ctor
        viewport.setAutoFillBackground(False)
        area.setWidget(page)
        # AFTER setWidget, which force-enables autoFillBackground on the widget it adopts:
        # clearing it beforehand is silently undone.
        page.setAutoFillBackground(False)
        return area

    @staticmethod
    def _page(*groups: QWidget) -> QWidget:
        """A tab page: the titled groups stacked top-down, separated by _TAB_GROUP_SPACING, with
        a trailing stretch so a short tab sits at the top instead of spreading its groups down
        the page. Every tab builds its page through this, so the gaps can't drift per tab."""
        page = QWidget()
        box = QVBoxLayout(page)
        box.setSpacing(_TAB_GROUP_SPACING)
        for group in groups:
            box.addWidget(group)
        box.addStretch(1)
        return page

    def _general_tab(self) -> QWidget:
        # Default VOD folder: the "Choose input video" picker opens here when set.
        self._default_vod_dir = QLineEdit(
            self._settings.value(SETTING_DEFAULT_VOD_DIR, "", type=str))
        self._default_vod_dir.setPlaceholderText("Last opened folder (default)")
  
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_default_vod_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Default VOD folder:"))
        dir_row.addWidget(self._default_vod_dir, 1)
        dir_row.addWidget(browse)

        # One-click access to the app's data folder: analyses live there but (frozen) it's
        # a semi-hidden app-data directory, so cleanup needs a way in.
        open_app_data = QPushButton("Open app data folder")
        open_app_data.clicked.connect(self._open_app_data_folder)
        open_app_data_row = QHBoxLayout()
        open_app_data_row.addWidget(open_app_data)
        open_app_data_row.addStretch(1)

        self._show_debug_menu = QCheckBox("Show the Debug menu in the menu bar")
        self._show_debug_menu.setChecked(
            self._settings.value(SETTING_SHOW_DEBUG_MENU, False, type=bool))

        # Whole-UI text size: each choice's scale factor rides along as item data (mirrors the
        # OCR-cadence combo). Applied at startup, so a note flags that it takes effect on restart.
        # Built unconditionally even while _SHOW_FONT_SCALE_SETTING hides the row, so accept()
        # keeps round-tripping the stored value instead of dropping it.
        self._font_scale = QComboBox()
        for label, factor in FONT_SCALE_CHOICES.items():
            self._font_scale.addItem(label, factor)
        saved_scale = self._settings.value(SETTING_FONT_SCALE, FONT_SCALE_DEFAULT, type=float)
        saved_scale_idx = self._font_scale.findData(saved_scale)
        self._font_scale.setCurrentIndex(saved_scale_idx if saved_scale_idx >= 0 else
                                         self._font_scale.findData(FONT_SCALE_DEFAULT))

        # Window size/position: the window always reopens at its last-used geometry (each page
        # remembers its own), so there's no toggle — this button is the way back if a saved size
        # ends up wrong, which is why it sits with the other fix-it actions below.
        reset_layout = QPushButton("Reset window layouts")
        # macOS draws the native rounded button only at EXACTLY the standard control height; any
        # other height falls back to a square bezel (Qt's macOS style, cocoaControlType). A
        # word-wrapped hint label above this button once shaved its row a few pixels short on
        # macOS and turned it square; the hint is gone, but pinning the minimum to the style's
        # own hint keeps any future neighbour from doing it again.
        reset_layout.setMinimumHeight(reset_layout.sizeHint().height())
        reset_layout.setEnabled(self._on_reset_layout is not None)
        reset_layout.clicked.connect(self._reset_window_layout)

        # Bring back any warnings the user silenced with "Don't show this again".
        restore_prompts = QPushButton("Restore confirmation prompts")
        restore_prompts.clicked.connect(self._restore_confirmation_prompts)

        # The two targeted, reversible resets share a row. Stacking every button vertically left
        # awkward gaps on macOS, whose style reserves focus-ring space above and below each
        # button: two stacked buttons pay that twice, so the space between them read as much
        # wider than the space under the checkbox above. One row means one such gap.
        targeted_resets_row = QHBoxLayout()
        targeted_resets_row.addWidget(reset_layout)
        targeted_resets_row.addWidget(restore_prompts)
        targeted_resets_row.addStretch(1)

        # Wipe every preference back to its shipped default. Kept on its own row below the
        # targeted pair: it's the only one of the three that isn't narrowly scoped.
        reset_all = QPushButton("Reset all preferences to defaults…")
        reset_all.clicked.connect(self._reset_all_to_defaults)
        reset_all_row = QHBoxLayout()
        reset_all_row.addWidget(reset_all)
        reset_all_row.addStretch(1)

        # Compute status + (on Windows) the optional GPU-acceleration download. The panel is
        # self-contained (probes for the active device, drives its own install/remove); the same
        # panel is also reachable from the File-menu "GPU Acceleration…" action.
        gpu_panel = GpuStatusPanel()

        groups = [self._group("Files", dir_row, open_app_data_row)]
        # Appearance holds only the text size, so the whole group comes and goes with it.
        if _SHOW_FONT_SCALE_SETTING:
            font_scale_row = QHBoxLayout()
            font_scale_row.addWidget(QLabel("Text size:"))
            font_scale_row.addWidget(self._font_scale)
            font_scale_row.addStretch(1)
            groups.append(self._group(
                "Appearance", font_scale_row,
                self._hint_row("Restart to apply any text size changes.")))
        groups.append(self._group("Hardware acceleration", gpu_panel))
        # The buttons run in ascending scope: one window, the silenced prompts, everything.
        groups.append(self._group(
            "Troubleshooting", self._show_debug_menu, targeted_resets_row, reset_all_row))
        return self._page(*groups)

    def _browse_default_vod_dir(self) -> None:
        current = self._default_vod_dir.text().strip()
        start = current if current and Path(current).is_dir() else str(vod_picker_start_dir())
        chosen = QFileDialog.getExistingDirectory(self, "Choose default VOD folder", start)
        if chosen:
            self._default_vod_dir.setText(chosen)

    def _open_app_data_folder(self) -> None:
        """Reveal the writable workspace (analyses, drafts) in Finder/Explorer."""
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(workspace_dir())))

    def _reset_window_layout(self) -> None:
        """Forget the saved per-screen window geometry and recenter the live window now, then snap
        this dialog back to its own opening size and position.

        The host callback owns the main window's geometry keys and the centering/fit logic. This
        dialog resets itself as well because it persists no geometry of its own: a Settings window
        the user had dragged or resized would otherwise sit untouched until it was closed and
        reopened, which reads as the button not having worked.
        """
        if self._on_reset_layout is None:
            return
        self._on_reset_layout()
        self.resize(_SETTINGS_DIALOG_WIDTH, _SETTINGS_DIALOG_HEIGHT)
        # Re-center on the next turn: the host defers its own fit-to-screen the same way, so
        # centering now would centre over the position the main window is about to leave.
        QTimer.singleShot(0, self._center_on_parent)

    def _center_on_parent(self) -> None:
        """Center the dialog over the window that opened it (or on its screen if it has none),
        matching where Qt places it when it first opens."""
        parent = self.parentWidget()
        if parent is not None:
            reference = parent.frameGeometry()
        else:
            screen = self.screen()
            if screen is None:
                return
            reference = screen.availableGeometry()
        frame = self.frameGeometry()
        frame.moveCenter(reference.center())
        self.move(frame.topLeft())

    def _restore_confirmation_prompts(self) -> None:
        """Clear the "Don't show this again" flags so the suppressible warnings show again. One
        targeted, reversible undo of those checkboxes — unlike Reset all, it touches nothing else.
        Removing the whole group covers every key filed under it (see SETTING_CONFIRM_GROUP), plus
        the GPU first-run offer's opt-out (SETTING_GPU_SKIP_FIRST_RUN_OFFER), which lives outside
        that group since it predates it.

        The button carries no hint line, so this confirmation is also what explains the action:
        it states the effect before doing anything. Defaults to Yes — restoring a prompt is
        harmless and undone by ticking "don't ask again" once more."""
        if QMessageBox.question(
            self, "Restore confirmation prompts?",
            "If you clicked \"don't ask again\" on any popups, this will make the app ask you "
            "again. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        ) != QMessageBox.StandardButton.Yes:
            return
        self._settings.remove(SETTING_CONFIRM_GROUP)
        self._settings.remove(SETTING_GPU_SKIP_FIRST_RUN_OFFER)
        self._settings.sync()

    def _reset_all_to_defaults(self) -> None:
        """Clear every stored preference so each falls back to its default, then close without
        running accept() (which would re-persist this dialog's current control values). The
        upload assistant's title/description templates survive the wipe: they're the user's
        own writing, with their own reset button in that window."""
        if QMessageBox.question(
            self, "Reset all preferences?",
            "Reset every preference on every tab to its default? This can't be undone.",
            QMessageBox.StandardButton.Reset | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        ) != QMessageBox.StandardButton.Reset:
            return
        spared = {key: self._settings.value(key)
                  for key in (SETTING_UPLOAD_TITLE_TEMPLATE, SETTING_UPLOAD_DESC_TEMPLATE)
                  if self._settings.contains(key)}
        self._settings.clear()
        for key, value in spared.items():
            self._settings.setValue(key, value)
        self._settings.sync()
        # reject(), not accept(): accept() would write the dialog's (now stale) control values
        # straight back over the defaults we just cleared. _open_settings re-applies the live UI.
        self.reject()

    def _export_tab(self) -> QWidget:
        self._title_limit = QSpinBox()
        self._title_limit.setRange(20, 500)
        self._title_limit.setValue(
            self._settings.value(SETTING_TITLE_CHAR_LIMIT, TITLE_CHAR_LIMIT_DEFAULT, type=int))
        row = QHBoxLayout()
        row.addWidget(QLabel("Title character limit:"))
        row.addWidget(self._title_limit)
        row.addStretch(1)
        title_limit_hint = self._hint_row(
            "Longer titles get trimmed on export. 255 is the filename limit for Mac/Windows.")

        self._title_filenames_sets = QCheckBox("Name set clips after their titles")
        self._title_filenames_sets.setChecked(
            self._settings.value(SETTING_TITLE_FILENAMES_SETS, True, type=bool))
        title_filenames_sets_hint = self._hint_row(
            "If unchecked, each set's filename will be named set1, set2, etc.")
        self._title_filenames_matches = QCheckBox("Name game clips after their titles")
        self._title_filenames_matches.setChecked(
            self._settings.value(SETTING_TITLE_FILENAMES_MATCHES, False, type=bool))
        title_filenames_matches_hint = self._hint_row(
            "If unchecked, each game's filename will be named game1, game2, etc. This "
            "is only relevant when you're exporting single games.")

        # Export folder: where exported clips land (see vodparse.gui.export_root). Empty means
        # the platform default, shown as the placeholder so the user sees where clips will go.
        self._export_dir_edit = QLineEdit(
            self._settings.value(SETTING_EXPORT_DIR, "", type=str))
        self._export_dir_edit.setPlaceholderText(f"{default_export_root()} (default)")
        export_dir_browse = QPushButton("Browse…")
        export_dir_browse.clicked.connect(self._browse_export_dir)
        export_dir_row = QHBoxLayout()
        export_dir_row.addWidget(QLabel("Export folder:"))
        export_dir_row.addWidget(self._export_dir_edit, 1)
        export_dir_row.addWidget(export_dir_browse)
        export_dir_hint = self._hint_row(
            "Where exported clips are saved. Each export creates its own subfolder here.")

        # No timestamp-delivery choice: the Review screen's button always copies to the
        # clipboard, which is what the timestamps are pasted into anyway.

        return self._page(
            self._group("Export folder", export_dir_row, export_dir_hint),
            self._group(
                "Clip titles", row, title_limit_hint,
                self._title_filenames_sets, title_filenames_sets_hint,
                self._title_filenames_matches, title_filenames_matches_hint))

    def _browse_export_dir(self) -> None:
        current = self._export_dir_edit.text().strip()
        start = current if current and Path(current).is_dir() else str(default_export_root())
        chosen = QFileDialog.getExistingDirectory(self, "Choose export folder", start)
        if chosen:
            self._export_dir_edit.setText(chosen)

    def _analysis_tab(self) -> QWidget:
        self._show_eta = QCheckBox("Show estimated time remaining when analyzing a VOD")
        self._show_eta.setChecked(
            self._settings.value(SETTING_SHOW_ETA, True, type=bool))
       
        self._gate_ocr_on_timer = QCheckBox("Skip analysis on frames with no timer (faster)")
        self._gate_ocr_on_timer.setChecked(
            self._settings.value(SETTING_GATE_OCR_ON_TIMER, True, type=bool))
        gate_ocr_hint = self._hint_row(
            "Only read onscreen text on frames where the game clock is "
            "visible. This skips menus, commentator clips, etc. Turn this off if "
            "it seems like you're missing tags.")

        self._ocr_roi_cache = QCheckBox("Analyzer remembers where text lives (faster)")
        self._ocr_roi_cache.setChecked(
            self._settings.value(SETTING_OCR_ROI_CACHE, True, type=bool))
        roi_cache_hint = self._hint_row(
            "This speeds up analysis by having the OCR models not have to \"find\" the "
            "text every time they run. Try turning this off if you're getting weird tag reads.")

        # In-game nametag band, shown as its height (% of frame height) so higher = reads
        # further up: more intuitive than the stored top-edge fraction, where lower = taller.
        # Stored as the fraction (SETTING_INGAME_TAG_BAND_FRAC) to match the pipeline's
        # --ingame-tag-band-frac and its JSON; default 0.92 = 8% (the bottom band), widening to
        # the floor 0.86 = 14%. See _band_pct_from_frac / _band_frac_from_pct for the mapping.
        self._ingame_band_pct = QSpinBox()
        self._ingame_band_pct.setRange(_band_pct_from_frac(INGAME_TAG_BAND_FRAC_DEFAULT),
                                       _band_pct_from_frac(INGAME_TAG_BAND_FRAC_FLOOR))
        self._ingame_band_pct.setSuffix(" %")
        saved_frac = self._settings.value(
            SETTING_INGAME_TAG_BAND_FRAC, INGAME_TAG_BAND_FRAC_DEFAULT, type=float)
        self._ingame_band_pct.setValue(_band_pct_from_frac(saved_frac))
        ingame_band_default_pct = _band_pct_from_frac(INGAME_TAG_BAND_FRAC_DEFAULT)
        ingame_band_revert = QPushButton("Revert to default")
        ingame_band_revert.clicked.connect(
            lambda: self._ingame_band_pct.setValue(ingame_band_default_pct))
        ingame_band_row = QHBoxLayout()
        ingame_band_row.addWidget(QLabel("Ingame tag reading height:"))
        ingame_band_row.addWidget(self._ingame_band_pct)
        ingame_band_row.addWidget(ingame_band_revert)
        ingame_band_row.addStretch(1)
        ingame_band_hint = self._hint_row(
            "How far up from the bottom of the screen to look for ingame tags. Higher percents "
            "are slightly slower. Bump this up to 12 or 14% if you have a VOD that isn't fit to "
            "your screen (this can happen if you used the \"Adjust Screen Size\" setting on your "
            "console and set it to less than 100%).")

        # Overlay language: each choice's language code rides along as item data (mirrors the
        # font-scale and OCR-cadence combos).
        self._ocr_lang = QComboBox()
        for label, code in OCR_LANG_CHOICES.items():
            self._ocr_lang.addItem(label, code)
        saved_lang = self._settings.value(SETTING_OCR_LANG, OCR_LANG_DEFAULT_GUI, type=str)
        saved_lang_idx = self._ocr_lang.findData(saved_lang)
        self._ocr_lang.setCurrentIndex(saved_lang_idx if saved_lang_idx >= 0 else
                                       self._ocr_lang.findData(OCR_LANG_DEFAULT_GUI))
        # Top-level control in its own group (below), so no _indented_row indent: that indent
        # is for nesting a control under the checkbox it qualifies.
        ocr_lang_row = QHBoxLayout()
        ocr_lang_row.addWidget(QLabel("Overlay language:"))
        ocr_lang_row.addWidget(self._ocr_lang)
        ocr_lang_row.addStretch(1)
        ocr_lang_hint = self._hint_row(
            "English should work for most alphanumeric tags. Use the Japanese option "
            "only for Japanese vods (it doesn't work well on English ones). The "
            "Japanese option downloads a ~60 MB model on first use.")

        # No hint row: the label says what the setting does, and the note it controls is
        # visible on the main screen while you decide.
        self._show_font_style_note = QCheckBox("Explain the font styles on the main screen")
        self._show_font_style_note.setChecked(
            self._settings.value(SETTING_SHOW_FONT_STYLE_NOTE, True, type=bool))

        return self._page(
            self._group("Progress", self._show_eta),
            self._group(
                "Text reading (OCR)", self._gate_ocr_on_timer, gate_ocr_hint,
                self._ocr_roi_cache, roi_cache_hint, ingame_band_row, ingame_band_hint,
                self._show_font_style_note),
            self._group("Overlay settings", ocr_lang_row, ocr_lang_hint))

    @staticmethod
    def _indented_row(label_text: str, widget: QWidget) -> QHBoxLayout:
        """An indented "<label> <widget>" row, indented under the checkbox it qualifies."""
        row = QHBoxLayout()
        row.addSpacing(24)                                # indent under the checkbox it qualifies
        row.addWidget(QLabel(label_text))
        row.addWidget(widget)
        row.addStretch(1)
        return row

    @staticmethod
    def _group(title: str, *items: QWidget | QLayout) -> QGroupBox:
        """A titled box stacking `items` vertically: matching the Analyze VOD screen's
        `_build_*_box` idiom (see app.py). Each item is added by type: widgets via addWidget,
        nested rows/layouts via addLayout."""
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        for item in items:
            if isinstance(item, QWidget):
                layout.addWidget(item)
            else:
                layout.addLayout(item)
        return box

    @staticmethod
    def _hint_row(text: str) -> QHBoxLayout:
        """An indented, word-wrapped hint line in the muted text color: the shared style for
        the small explanatory notes under a control (mirrors _indented_row's 24px indent)."""
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {muted_text_color()};")
        row = QHBoxLayout()
        row.addSpacing(24)                                # indent under the control it qualifies
        row.addWidget(label, 1)
        return row

    @staticmethod
    def _bind_enabled(checkbox: QCheckBox, widget: QWidget) -> None:
        """Grey out `widget` unless `checkbox` is ticked (the qualifier only matters when on)."""
        widget.setEnabled(checkbox.isChecked())
        checkbox.toggled.connect(widget.setEnabled)

    def _checkbox(self, setting: str, default: bool, label: str, tip: str = "") -> QCheckBox:
        """A settings-backed checkbox: labelled, initialized from `setting`, with an optional
        tooltip (most explanations are visible _hint_row lines instead)."""
        cb = QCheckBox(label)
        cb.setChecked(self._settings.value(setting, default, type=bool))
        if tip:
            cb.setToolTip(tip)
        return cb

    def _flag_row(self, checkbox: QCheckBox, row_label: str, widget: QWidget) -> QHBoxLayout:
        """The qualifier row under a review-flag checkbox: grey `widget` out unless the flag is
        on, laid out indented beneath it."""
        self._bind_enabled(checkbox, widget)
        return self._indented_row(row_label, widget)

    def _review_tab(self) -> QWidget:
        self._flag_long = self._checkbox(
            SETTING_FLAG_LONG_MATCHES, True, "Flag unusually long games for review")
        self._max_match = _duration_edit(                 # mm:ss; 1:00 .. 30:00
            self._settings.value(SETTING_MAX_MATCH_SEC, MAX_MATCH_SEC_DEFAULT, type=float), 60, 1800)
        max_row = self._flag_row(self._flag_long, "Flag games longer than:", self._max_match)
        flag_long_hint = self._hint_row(
            "Games that run long usually happen when there's a replay on the stream, handwarmer or " 
            "the analyzer somehow misses the end of the game.")

        self._flag_short = self._checkbox(
            SETTING_FLAG_SHORT_MATCHES, True, "Flag very short games for review")
        self._min_match = _duration_edit(                 # mm:ss; 0:05 .. 5:00
            self._settings.value(SETTING_MIN_MATCH_SEC, MIN_MATCH_SEC_DEFAULT, type=float), 5, 300)
        min_row = self._flag_row(self._flag_short, "Flag games shorter than:", self._min_match)
        flag_short_hint = self._hint_row("Short games are usually handwarmers or restarts.")

        self._flag_weird = self._checkbox(
            SETTING_FLAG_WEIRD_TIMER, True, "Flag games with a bad start timer for review")
        flag_weird_hint = self._hint_row(
            "The clip starts before the game clock says it began. Usually a restart, "
            "replay, or pause.")

        self._flag_long_leadin = self._checkbox(
            SETTING_FLAG_LONG_LEADIN, True, "Flag games with a long lead-in for review")
        self._leadin_intervals = QDoubleSpinBox()
        self._leadin_intervals.setRange(1.0, 20.0)
        self._leadin_intervals.setSingleStep(0.5)
        self._leadin_intervals.setDecimals(1)
        self._leadin_intervals.setSuffix(" × scan interval")
        self._leadin_intervals.setValue(
            self._settings.value(SETTING_LONG_LEADIN_INTERVALS, LONG_LEADIN_INTERVALS_DEFAULT,
                                 type=float))
      
        leadin_row = self._flag_row(self._flag_long_leadin, "Flag a long lead-in past:",
                                    self._leadin_intervals)
        leadin_hint = self._hint_row(
            "The game starts further back than a single analysis step could have skipped. That's "
            "often a replay on the stream, the streamer forgetting to switch the scene on time, "
            "or the analyzer missing the clock in the beginning.")

        self._mark_ingame_fallback = self._checkbox(
            SETTING_MARK_INGAME_FALLBACK, True, "Mark ingame fallback tags")
        ingame_fallback_hint = self._hint_row(
            "When in 'auto' mode, mark tags that were read from the ingame HUD instead of the overlay.")

        return self._page(self._group(
            "Review flags", self._flag_long, max_row, flag_long_hint, self._flag_short,
            min_row, flag_short_hint, self._flag_weird, flag_weird_hint,
            self._flag_long_leadin, leadin_row, leadin_hint,
            self._mark_ingame_fallback, ingame_fallback_hint))

    def accept(self) -> None:
        self._settings.setValue(SETTING_TITLE_CHAR_LIMIT, self._title_limit.value())
        self._settings.setValue(SETTING_TITLE_FILENAMES_SETS,
                                self._title_filenames_sets.isChecked())
        self._settings.setValue(SETTING_TITLE_FILENAMES_MATCHES,
                                self._title_filenames_matches.isChecked())
        self._settings.setValue(SETTING_FLAG_LONG_MATCHES, self._flag_long.isChecked())
        self._settings.setValue(SETTING_MAX_MATCH_SEC, float(_duration_seconds(self._max_match)))
        self._settings.setValue(SETTING_FLAG_SHORT_MATCHES, self._flag_short.isChecked())
        self._settings.setValue(SETTING_MIN_MATCH_SEC, float(_duration_seconds(self._min_match)))
        self._settings.setValue(SETTING_FLAG_WEIRD_TIMER, self._flag_weird.isChecked())
        self._settings.setValue(SETTING_FLAG_LONG_LEADIN, self._flag_long_leadin.isChecked())
        self._settings.setValue(SETTING_LONG_LEADIN_INTERVALS, float(self._leadin_intervals.value()))
        self._settings.setValue(SETTING_MARK_INGAME_FALLBACK,
                                self._mark_ingame_fallback.isChecked())
        self._settings.setValue(SETTING_SHOW_ETA, self._show_eta.isChecked())
        self._settings.setValue(SETTING_GATE_OCR_ON_TIMER, self._gate_ocr_on_timer.isChecked())
        self._settings.setValue(SETTING_OCR_ROI_CACHE, self._ocr_roi_cache.isChecked())
        self._settings.setValue(SETTING_INGAME_TAG_BAND_FRAC,
                                _band_frac_from_pct(self._ingame_band_pct.value()))
        self._settings.setValue(SETTING_OCR_LANG, self._ocr_lang.currentData())
        self._settings.setValue(SETTING_SHOW_FONT_STYLE_NOTE,
                                self._show_font_style_note.isChecked())
        self._settings.setValue(SETTING_SHOW_DEBUG_MENU, self._show_debug_menu.isChecked())
        self._settings.setValue(SETTING_FONT_SCALE, float(self._font_scale.currentData()))
        self._settings.setValue(SETTING_DEFAULT_VOD_DIR, self._default_vod_dir.text().strip())
        self._settings.setValue(SETTING_EXPORT_DIR, self._export_dir_edit.text().strip())
        super().accept()
