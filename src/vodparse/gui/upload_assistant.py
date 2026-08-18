"""Standalone upload assistant.

A one-screen wizard that steps through the clips of a single export (the CSV + ``.mp4``s the
main GUI writes into ``video_out/<basename>/``). For each clip it shows:

  * a draggable thumbnail you drag straight out of the app and into the video upload page:
    the browser treats it as a real file drop, the same as dragging from a file manager;
  * the title (with a Copy button), built from an editable title template over the clip's
    parts (round, sponsors, tags, characters) and auto-shortened to a configurable character
    limit; the template-level analogue of ``titles.fit_title`` (round abbreviated, then
    sponsors and round dropped); a title that can't be shortened enough is kept in full and
    flagged, hand-edits past the limit show a live character count, and a clip with no data
    for a player slot gets a warning too;
  * an auto-generated description built from an editable template (configured in Settings,
    persisted via QSettings), with a Copy button. Its ``{title}`` placeholder tracks the
    title box above (including hand-edits) for as long as the description itself is
    untouched generated text.

Back / Next move between clips. Run with ``python -m vodparse.gui.upload_assistant [path/to/export.csv]``.
"""

import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from vodparse.gui.qt_plugins import configure_qt_plugins

configure_qt_plugins()  # must run before any PyQt6 widget module is imported

from PyQt6.QtCore import (  # noqa: E402
    QByteArray, QEvent, Qt, QMimeData, QPoint, QSettings, QUrl, pyqtSignal,
)
from PyQt6.QtGui import (  # noqa: E402
    QAction, QActionGroup, QCloseEvent, QDrag, QKeyEvent, QMouseEvent, QPixmap,
    QResizeEvent,
)
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QDialogButtonBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenuBar,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from vodparse.gui.common import (ElidingLabel, apply_font_scale, apply_system_theme,  # noqa: E402
                        copy_with_feedback, muted_text_color, warning_color)
from vodparse.gui import crash_log  # noqa: E402
from vodparse.gui.dialogs import TournamentNameDialog  # noqa: E402
from vodparse.gui.export_root import export_root  # noqa: E402
from vodparse.gui.settings import (  # noqa: E402
    FONT_SCALE_DEFAULT, SETTING_FONT_SCALE,
    SETTING_UPLOAD_DESC_TEMPLATE, SETTING_UPLOAD_TITLE_TEMPLATE,
)
from vodparse.ffmpeg_runtime import NO_CONSOLE_WINDOW, ffmpeg_binary  # noqa: E402
from vodparse.titles import abbreviate_round  # noqa: E402

# --- theme-aware colors ------------------------------------------------------
# Secondary text and the warning use explicit colors that must track the system light/dark setting
# or they go unreadable (dark amber / mid-grey on a dark background). The muted_text_color /
# warning_color helpers (in common.py) read the live app palette; widgets that use them re-apply
# on ApplicationPaletteChange (below) to follow a theme switch while open.

# The per-widget palette change Qt delivers to changeEvent() when the OS theme flips. (The
# app-wide ApplicationPaletteChange goes to event(), not changeEvent(), so it can't be used here.)
# Our own setStyleSheet() also emits PaletteChange, which would re-enter changeEvent endlessly:
# so the themed _apply_theme() methods are idempotent (they skip setStyleSheet when the value is
# unchanged), which both breaks that recursion and makes a real theme switch settle in one step.
_THEME_EVENTS = (QEvent.Type.PaletteChange,)


# Persisted settings, sharing the main GUI's app identity (see main()). The two template keys
# live in vodparse.gui.settings so the Settings dialog's "Reset all preferences" can spare them.
SETTING_DESC_TEMPLATE = SETTING_UPLOAD_DESC_TEMPLATE
SETTING_TITLE_TEMPLATE = SETTING_UPLOAD_TITLE_TEMPLATE
SETTING_TITLE_CHAR_LIMIT = "upload/title_char_limit"
SETTING_STAY_ON_TOP = "upload/stay_on_top"  # default True; handy while dragging into the browser
SETTING_COPY_ON_LEFT = "upload/copy_on_left"  # Copy buttons left of the fields; default False (right)
SETTING_GEOMETRY = "upload/geometry"  # window size+position; absent → default to the left edge

# Opening size before the user has resized the window (SETTING_GEOMETRY overrides it). Named
# because "Reset window layouts" restores it too, so the two must not drift apart.
DEFAULT_WINDOW_SIZE = (560, 640)


def forget_saved_geometry(settings: QSettings) -> None:
    """Clear the assistant's remembered size/position, so it next opens at its defaults.

    Module-level because the Settings dialog's "Reset window layouts" must clear the key even
    when the assistant was never opened this session and no window object exists; an open window
    goes through UploadAssistant.reset_layout instead, which also moves it.
    """
    settings.remove(SETTING_GEOMETRY)


# Template placeholders, shared by the title and description templates. {p1_tag}/{p2_tag}
# resolve to the tag the exported title used (the player_N_tag column, i.e. export.py's
# title_tags = grouping_tags or tournament_tags or tags), falling back through tournament ->
# ingame -> legacy columns for older CSVs; the explicit *_ingame_tag / *_tournament_tag
# placeholders pull a specific column. {tournament} is the export-wide tournament name column
# (empty for CSVs from before it existed). Unknown placeholders the user types also render
# empty (never raise), so a template is always safe to apply to any export-era CSV. {p1}/{p2} are the composed player blocks as export titles render
# them; "SPONSOR | TAG (CHARACTER)", each separator present only when its part is; so a
# template built from them never shows a dangling " | " when a sponsor drops out. {title}
# differs between the two templates: in the title template it is the raw CSV title column; in
# the description template it is whatever the title box currently shows (the generated title,
# or the user's hand-edit of it).
AVAILABLE_PLACEHOLDERS = [
    "title", "filename", "round", "best_of", "match_count", "matches",
    "start_time", "end_time", "tournament",
    "p1", "p2",
    "p1_tag", "p2_tag",
    "p1_ingame_tag", "p2_ingame_tag",
    "p1_tournament_tag", "p2_tournament_tag",
    "p1_sponsor", "p2_sponsor",
    "p1_character", "p2_character",
]

# {p1}/{p2} are the composed "SPONSOR | TAG (CHARACTER)" blocks (see _render_context), so
# sponsors and characters survive; one fact per line means an empty placeholder leaves a
# blank line rather than a stray fragment.
DEFAULT_DESC_TEMPLATE = (
    "{p1} vs {p2}\n"
    "{round} {best_of}\n"
    "{tournament}"
)

# Mirrors the export title's shape, built from editable parts so the length-fitting stages
# have something to shorten. {tournament} renders empty (and its spacing tidies away) for
# CSVs from before the tournament column existed.
DEFAULT_TITLE_TEMPLATE = "{tournament} - {round} - {p1} vs {p2}"
DEFAULT_TITLE_CHAR_LIMIT = 100  # YouTube's title cap; matches titles.fit_title's default

# Under-title note shown when render_fitted_title had to act. "too_long" is the only warning
# (the title still exceeds the limit); the others just explain why the title differs from the
# CSV. A row-level missing-player warning (_missing_player_note) can join either kind.
_TITLE_FIT_NOTES = {
    "abbreviated":      "Round abbreviated to fit the {limit}-character title limit",
    "sponsors_dropped": "Sponsors dropped to fit the {limit}-character title limit",
    "round_dropped":    "Round dropped to fit the {limit}-character title limit",
    "too_long":         "Title is {length} characters; over the {limit}-character limit; shorten by hand",
}


class _SafeDict(dict):
    """format_map backing dict: an unknown {placeholder} renders empty instead of raising."""

    def __missing__(self, key: str) -> str:
        return ""


def _render_context(row: dict[str, str]) -> dict[str, str]:
    """Map a CSV row to the template placeholders, tolerating missing columns (empty string)."""
    def tag(prefix: str) -> str:
        # Preferred display tag: the one the exported title used (player_N_tag: export's
        # grouping/tournament/ingame pick, including review corrections), then the specific
        # columns for CSVs from before that column existed, then the legacy plain column.
        return (row.get(f"{prefix}_tag")
                or row.get(f"{prefix}_tournament_tag")
                or row.get(f"{prefix}_ingame_tag")
                or row.get(prefix)
                or "")

    def player(prefix: str) -> str:
        # The composed player block as export titles render it (mirrors titles.build_title):
        # "SPONSOR | TAG (CHARACTER)", each separator only when its part exists.
        name = tag(prefix)
        sponsor = row.get(f"{prefix}_sponsor", "")
        if sponsor:
            name = f"{sponsor} | {name}"
        character = row.get(f"{prefix}_character", "")
        return f"{name} ({character})" if character else name

    return {
        "p1":                player("player_1"),
        "p2":                player("player_2"),
        "title":             row.get("title", ""),
        "filename":          row.get("filename", ""),
        "round":             row.get("round", ""),
        "best_of":           row.get("best_of", ""),
        "match_count":       row.get("match_count", ""),
        "matches":           row.get("matches", ""),
        "start_time":        row.get("start_time", ""),
        "end_time":          row.get("end_time", ""),
        "tournament":        row.get("tournament", ""),  # column absent in pre-2026-07 CSVs
        "p1_tag":            tag("player_1"),
        "p2_tag":            tag("player_2"),
        "p1_ingame_tag":     row.get("player_1_ingame_tag", ""),
        "p2_ingame_tag":     row.get("player_2_ingame_tag", ""),
        "p1_tournament_tag": row.get("player_1_tournament_tag", ""),
        "p2_tournament_tag": row.get("player_2_tournament_tag", ""),
        "p1_sponsor":        row.get("player_1_sponsor", ""),
        "p2_sponsor":        row.get("player_2_sponsor", ""),
        "p1_character":      row.get("player_1_character", ""),
        "p2_character":      row.get("player_2_character", ""),
    }


def render_description(template: str, row: dict[str, str], title: str | None = None) -> str:
    """Fill `template` from a CSV `row`. Missing columns and unknown placeholders render empty.

    When `title` is given it overrides the {title} placeholder; the assistant passes the final
    generated title (see render_fitted_title) so the description echoes what is actually shown,
    not the raw CSV column."""
    context = _render_context(row)
    if title is not None:
        context["title"] = title
    return template.format_map(_SafeDict(context))


def _flatten_newlines(text: str) -> str:
    """Collapse line breaks to spaces; a video title is a single line. (Render-time counterpart
    of TitleEdit.insertFromMimeData, which flattens pasted newlines the same way.)"""
    return text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


# Dash separators left dangling by an empty placeholder; e.g. the "- " head of the default
# template on a clip with no tournament, or the doubled "- - " it leaves in the middle when
# only the round is missing. A dash only counts as a separator with a space on both sides
# (or an end of the title), so hyphens inside real tags ("Player-One", "-PlayerTwo-") survive;
# pipes and colons are never eaten. The blind spot: a tag that is itself a bare dash next
# to a separator loses it. An edge artifact from a custom template using other separators
# is left visible for the user to edit.
_DANGLING_EDGE_SEPARATORS = re.compile(r"^(?:[-—–]+ )+|(?: [-—–]+)+$")
_DOUBLED_SEPARATORS = re.compile(r"(?:[-—–]+ )+(?=[-—–]+ )")


def _tidy_title(text: str) -> str:
    """Make a rendered title presentable after placeholders may have rendered empty: newlines
    flatten to spaces, whitespace runs collapse to one space, ends are stripped, and dash
    separators left dangling at either end or doubled up mid-title are removed (see
    _DANGLING_EDGE_SEPARATORS / _DOUBLED_SEPARATORS)."""
    text = " ".join(_flatten_newlines(text).split())
    text = _DOUBLED_SEPARATORS.sub("", text)
    return _DANGLING_EDGE_SEPARATORS.sub("", text)


def _missing_player_note(row: dict[str, str]) -> str:
    """A warning when a clip has no data at all for one (or both) player slots; the composed
    title renders a broken matchup like "Round 1 - vs PlayerTwo", which the length-fitting notes
    would never flag. A slot with *any* part (tag, sponsor, or character) counts as present;
    "" means both slots have something."""
    context = _render_context(row)
    missing = [label for label, key in (("Player 1", "p1"), ("Player 2", "p2"))
               if not context[key]]
    if len(missing) == 2:
        return "No player data for this clip; check the export CSV"
    if missing:
        return f"{missing[0]} data missing for this clip; check the export CSV"
    return ""


def render_fitted_title(template: str, row: dict[str, str],
                        limit: int = DEFAULT_TITLE_CHAR_LIMIT) -> tuple[str, str | None]:
    """Render the title template and shorten it to `limit` with the least intervention,
    mirroring titles.fit_title at template level: abbreviate the round, then also drop the
    sponsors, then also drop the round. Each stage patches the CSV *row* (not the rendered
    text) and re-renders, so composed placeholders like {p1} rebuild without their sponsor
    prefix rather than keeping a dangling " | ". A stage only counts when it actually changed
    the rendered title, so templates that don't use the round or sponsors fall through
    cleanly. Returns (title, action), action one of None | "abbreviated" | "sponsors_dropped"
    | "round_dropped" | "too_long". A title that can't fit even with everything dropped is
    returned in FULL ("too_long"); the caller warns instead of hard-truncating. Output is
    always single-line and tidied (see _tidy_title).

    Accepted limitation: literal text around a RAW placeholder stays when it renders empty:
    "{p1_sponsor} | {p1_tag}" keeps its mid-string " | " when the sponsor drops (only dangling
    ends are tidied). Templates built from {p1}/{p2} don't have this problem."""
    def render(patched_row: dict[str, str]) -> str:
        return _tidy_title(template.format_map(_SafeDict(_render_context(patched_row))))

    full = render(row)
    if len(full) <= limit:
        return full, None
    abbreviated_row = row | {"round": abbreviate_round(row.get("round", ""))}
    abbreviated = render(abbreviated_row)
    if abbreviated != full and len(abbreviated) <= limit:
        return abbreviated, "abbreviated"
    unsponsored_row = abbreviated_row | {"player_1_sponsor": "", "player_2_sponsor": ""}
    unsponsored = render(unsponsored_row)
    if unsponsored != full and len(unsponsored) <= limit:
        return unsponsored, "sponsors_dropped"
    bare_row = unsponsored_row | {"round": ""}
    bare = render(bare_row)
    if bare != full and len(bare) <= limit:
        return bare, "round_dropped"
    return full, "too_long"


def _thumbnail(clip_path: Path) -> QPixmap | None:
    """A still ~1s into the clip via the ffmpeg binary, as a QPixmap. None on any failure:
    the caller falls back to a generic icon. The drag never depends on this; it always carries
    the real file path."""
    # A unique temp file per call, so two assistant instances can't clobber each other's thumbnail.
    with tempfile.NamedTemporaryFile(suffix=".jpg", prefix="upload_assistant_thumb_",
                                     delete=False) as f:
        out = Path(f.name)
    try:
        subprocess.run(
            [ffmpeg_binary(), "-y", "-ss", "1", "-i", str(clip_path),
             "-frames:v", "1", "-vf", "scale=320:-1", str(out)],
            check=True, capture_output=True, timeout=15,
            creationflags=NO_CONSOLE_WINDOW,  # no console flash per thumbnail on Windows
        )
        pixmap = QPixmap(str(out))
    except (subprocess.SubprocessError, OSError):
        return None
    finally:
        out.unlink(missing_ok=True)
    return pixmap if not pixmap.isNull() else None


class ClipDragLabel(QLabel):
    """The clip thumbnail, and a drag *source*: dragging it out of the app into a browser's
    upload zone drops the clip's real ``.mp4``.

    The drag carries the file's URL via ``QMimeData.setUrls``; Qt exports that as
    ``text/uri-list`` plus the platform-native file format, which browser drop targets accept as
    a real file drop. (Needs a real display; can't be exercised headlessly.)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._clip_path: Path | None = None
        self._press_pos: QPoint | None = None
        self.setMinimumSize(320, 180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._apply_theme()

    def _apply_theme(self) -> None:
        # Idempotent: skip setStyleSheet when unchanged, so the PaletteChange it would emit can't
        # re-enter changeEvent → _apply_theme forever (see _THEME_EVENTS).
        style = f"QLabel {{ border: 1px dashed palette(mid); color: {muted_text_color()}; }}"
        if self.styleSheet() != style:
            self.setStyleSheet(style)

    def changeEvent(self, a0: QEvent | None) -> None:
        if a0 is not None and a0.type() in _THEME_EVENTS:
            self._apply_theme()
        super().changeEvent(a0)

    def set_clip(self, path: Path | None, pixmap: QPixmap | None) -> None:
        """Show `path`'s thumbnail (or a fallback). `path` None ⇒ nothing to drag."""
        self._clip_path = path
        if pixmap is not None and not pixmap.isNull():
            self.setPixmap(pixmap)
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.clear()
            if path is None:
                self.setText("Open the CSV in your clip export folder to begin")
                self.setCursor(Qt.CursorShape.ArrowCursor)
            else:
                self.setText(f"{path.name}\n(drag me to the upload screen)")
                self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, ev: QMouseEvent | None) -> None:
        if ev is not None and ev.button() == Qt.MouseButton.LeftButton:
            self._press_pos = ev.position().toPoint()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent | None) -> None:
        if (ev is None or self._press_pos is None or self._clip_path is None
                or not (ev.buttons() & Qt.MouseButton.LeftButton)):
            return
        moved = (ev.position().toPoint() - self._press_pos).manhattanLength()
        if moved < QApplication.startDragDistance():
            return
        if not self._clip_path.exists():
            return  # the window already shows a missing-file warning

        drag = QDrag(self)
        mime = QMimeData()
        url = QUrl.fromLocalFile(str(self._clip_path))
        mime.setUrls([url])           # text/uri-list + native file mime; the line browsers need
        mime.setText(url.toString())  # plain-text fallback (e.g. Firefox x-moz-url drop zones)
        drag.setMimeData(mime)
        pixmap = self.pixmap()
        if pixmap is not None and not pixmap.isNull():
            drag.setPixmap(pixmap.scaled(
                160, 90, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        drag.exec(Qt.DropAction.CopyAction)  # Copy, never Move; never removes the source file


class TitleEdit(QPlainTextEdit):
    """The title field: wraps and grows vertically to fit a long title, but stays a single
    logical line; Return/Enter and pasted newlines are rejected, since a video title has none.

    Exposes setText/text so callers treat it like the QLineEdit it replaced.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        doc = self.document()
        layout = doc.documentLayout() if doc is not None else None
        if layout is not None:
            layout.documentSizeChanged.connect(lambda _size: self._fit_height())
        self._fit_height()

    def setText(self, text: str) -> None:
        self.setPlainText(text)

    def text(self) -> str:
        return self.toPlainText()

    def keyPressEvent(self, e: QKeyEvent | None) -> None:
        if e is not None and e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            return  # a title is single-line; swallow newline keys
        super().keyPressEvent(e)

    def resizeEvent(self, e: QResizeEvent | None) -> None:
        # A width change re-wraps the text (more/fewer lines) but doesn't emit
        # documentSizeChanged, so re-fit here too. This covers the first show()
        #; the title is set while the widget still has its default width; and
        # any later window resize.
        super().resizeEvent(e)
        self._fit_height()

    def insertFromMimeData(self, source: QMimeData | None) -> None:
        # Flatten newlines in pasted text to spaces so a paste can't sneak in a line break.
        if source is not None and source.hasText():
            flat = QMimeData()
            flat.setText(" ".join(source.text().split("\n")).replace("\r", " "))
            super().insertFromMimeData(flat)
        else:
            super().insertFromMimeData(source)

    def _fit_height(self) -> None:
        """Size the widget to its wrapped content so it grows/shrinks with the text, starting at a
        one-line baseline like the QLineEdit it replaced. QPlainTextDocumentLayout reports the
        document height in *lines* (incl. wrapped ones), so convert to pixels via line spacing."""
        doc = self.document()
        if doc is None:
            return
        lines = max(1, round(doc.size().height()))
        margins = self.contentsMargins()
        height = (lines * self.fontMetrics().lineSpacing()
                  + 2 * doc.documentMargin()
                  + margins.top() + margins.bottom()
                  + 2 * self.frameWidth())
        self.setFixedHeight(int(height))


class UploadSettingsDialog(QDialog):
    """Edits and persists the title template, title character limit, and description template
    (all three via QSettings)."""

    def __init__(self, settings: QSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Upload settings")
        self.resize(560, 480)
        self._settings = settings

        self._title_template = QLineEdit(
            settings.value(SETTING_TITLE_TEMPLATE, DEFAULT_TITLE_TEMPLATE, type=str))

        self._title_limit = QSpinBox()
        self._title_limit.setRange(20, 500)  # same bounds as the export title-length spinbox
        self._title_limit.setValue(
            settings.value(SETTING_TITLE_CHAR_LIMIT, DEFAULT_TITLE_CHAR_LIMIT, type=int))
        title_limit_row = QHBoxLayout()
        title_limit_row.addWidget(QLabel("Title character limit"))
        title_limit_row.addWidget(self._title_limit)
        title_limit_row.addStretch(1)

        self._desc_edit = QPlainTextEdit()
        self._desc_edit.setPlainText(
            settings.value(SETTING_DESC_TEMPLATE, DEFAULT_DESC_TEMPLATE, type=str))

        placeholders = QLabel(
            "Available placeholders (both templates):\n"
            + "  ".join(f"{{{name}}}" for name in AVAILABLE_PLACEHOLDERS)
            + "\nIn the description template, {title} is the current contents of the title box.")
        placeholders.setWordWrap(True)
        placeholders.setStyleSheet(f"color: {muted_text_color()};")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        reset = buttons.addButton("Reset to defaults", QDialogButtonBox.ButtonRole.ResetRole)
        assert reset is not None
        reset.clicked.connect(self._reset_to_defaults)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Title template"))
        layout.addWidget(self._title_template)
        layout.addLayout(title_limit_row)
        layout.addWidget(QLabel("Description template"))
        layout.addWidget(self._desc_edit)
        layout.addWidget(placeholders)
        layout.addWidget(buttons)

    def _reset_to_defaults(self) -> None:
        """Restore all three fields to their built-in defaults (does not persist until OK)."""
        self._title_template.setText(DEFAULT_TITLE_TEMPLATE)
        self._title_limit.setValue(DEFAULT_TITLE_CHAR_LIMIT)
        self._desc_edit.setPlainText(DEFAULT_DESC_TEMPLATE)

    def accept(self) -> None:
        self._settings.setValue(SETTING_TITLE_TEMPLATE, self._title_template.text())
        self._settings.setValue(SETTING_TITLE_CHAR_LIMIT, self._title_limit.value())
        self._settings.setValue(SETTING_DESC_TEMPLATE, self._desc_edit.toPlainText())
        super().accept()


class UploadAssistant(QWidget):
    """Single-screen wizard stepping through one export's clips."""

    # Emitted when the window closes. The main GUI locks itself (setEnabled(False)) while
    # this window is open: modality in effect, without the platform modality that would
    # break Stay-on-top; and listens to this to unlock. (Caveat for visibility-derived
    # listeners: the emit happens inside closeEvent, BEFORE Qt hides the window.)
    closed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Upload Assistant")
        self.resize(*DEFAULT_WINDOW_SIZE)
        self._settings = QSettings()
        self._rows: list[dict[str, str]] = []
        self._csv_dir: Path | None = None
        self._csv_path: Path | None = None  # the loaded CSV file, for save-back rewrites
        self._csv_fields: list[str] = []  # its original column order
        self._index = 0
        self._title_note_kind: str | None = None  # which _TITLE_FIT_NOTES key is showing, if any
        self._missing_player_note = ""  # row-level warning, set per clip in _show_clip
        self._title_char_limit = DEFAULT_TITLE_CHAR_LIMIT  # refreshed from settings in _show_clip
        self._generated_desc = ""  # last auto-rendered description; edits make the box "owned"
        self._syncing_title = False  # _show_clip is setting the title; skip the live-edit handler
        self._build_ui()
        self._show_clip()
        self._on_top.setChecked(self._settings.value(SETTING_STAY_ON_TOP, True, type=bool))
        self._restore_geometry()

    def _restore_geometry(self) -> None:
        """Reopen where the user last left the window; first run defaults to the left edge so the
        tool sits beside the browser it drags into. Restore last so it overrides the initial size."""
        saved = self._settings.value(SETTING_GEOMETRY, QByteArray(), type=QByteArray)
        if saved.isEmpty() or not self.restoreGeometry(saved):
            self._move_to_default_position()

    def _move_to_default_position(self) -> None:
        """Park the window against the left edge of its screen, vertically centered: where it
        opens before the user has ever moved it (see _restore_geometry)."""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        y = area.top() + max(0, (area.height() - self.height()) // 2)
        self.move(area.left(), y)

    def reset_layout(self) -> None:
        """Forget this window's remembered size/position and snap it back to its defaults now.
        Driven by the Settings dialog's "Reset window layouts" button, via MainWindow.

        Resetting the live window matters, not just the stored key: closeEvent writes the current
        geometry back, so clearing the key under an open, still-misplaced window would simply
        re-save the bad geometry on close and undo the reset.
        """
        forget_saved_geometry(self._settings)
        self.resize(*DEFAULT_WINDOW_SIZE)
        self._move_to_default_position()

    def closeEvent(self, e: QCloseEvent | None) -> None:
        self._settings.setValue(SETTING_GEOMETRY, self.saveGeometry())
        self.closed.emit()
        super().closeEvent(e)

    # ----- UI construction -------------------------------------------------
    def _build_ui(self) -> None:
        self._on_top = QAction("Stay on top", self)
        self._on_top.setCheckable(True)
        self._on_top.toggled.connect(self._set_stay_on_top)
        menu_bar = QMenuBar()
        # Not titled "View": macOS auto-injects an "Enter Full Screen" item into a menu named
        # "View", which would hide the browser this tool is meant to sit on top of.
        options_menu = menu_bar.addMenu("Options")
        assert options_menu is not None
        options_menu.addAction(self._on_top)

        # Fix a wrong tournament name noticed only at upload time: rewrites the {tournament}
        # column of every loaded row and saves the CSV (see _change_tournament_name).
        self._rename_tournament = QAction("Change tournament name…", self)
        self._rename_tournament.setEnabled(False)  # needs a loaded CSV; _show_clip toggles it
        self._rename_tournament.triggered.connect(self._change_tournament_name)
        options_menu.addAction(self._rename_tournament)

        copy_menu = options_menu.addMenu("Copy buttons")
        assert copy_menu is not None
        copy_side = QActionGroup(self)  # exclusive: exactly one of left/right is checked
        left_act = QAction("On the left", self)
        right_act = QAction("On the right", self)
        for act in (left_act, right_act):
            act.setCheckable(True)
            copy_side.addAction(act)
            copy_menu.addAction(act)
        left = self._settings.value(SETTING_COPY_ON_LEFT, False, type=bool)
        (left_act if left else right_act).setChecked(True)
        # Connect after the initial check so it doesn't fire during construction; _build_ui's
        # closing _apply_copy_side() does the first layout pass.
        left_act.toggled.connect(self._set_copy_on_left)

        open_btn = QPushButton("Open…")
        open_btn.clicked.connect(self._open_csv)
        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self._open_settings)
        top = QHBoxLayout()
        top.addWidget(open_btn)
        top.addStretch(1)
        top.addWidget(settings_btn)

        self._counter = QLabel("")  # empty until a CSV loads; the thumbnail box shows the hint
        self._counter.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._thumb = ClipDragLabel()

        self._caption = QLabel("")  # shown only when a draggable clip is loaded
        self._caption.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Eliding (middle, like a filename) so a long clip name in "Clip file missing: …"
        # can't pin the window's minimum width; the full name rides along as the tooltip.
        self._warning = ElidingLabel("", mode=Qt.TextElideMode.ElideMiddle)
        self._warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_note = QLabel("")
        self._apply_theme()

        self._title = TitleEdit()  # editable, single line, grows vertically to fit a long title
        self._title.textChanged.connect(self._on_title_edited)
        self._title_copy = QPushButton("Copy")
        self._title_copy.clicked.connect(self._copy_title)
        self._title_row = QHBoxLayout()
        self._title_row.addWidget(self._title)
        self._title_row.addWidget(self._title_copy, 0, Qt.AlignmentFlag.AlignTop)

        self._desc = QPlainTextEdit()  # editable, so per-clip tweaks are possible before copying
        self._desc_copy = QPushButton("Copy")
        self._desc_copy.clicked.connect(self._copy_description)
        self._desc_row = QHBoxLayout()
        self._desc_row.addWidget(self._desc, 1)
        self._desc_row.addWidget(self._desc_copy, 0, Qt.AlignmentFlag.AlignTop)

        self._back = QPushButton("◀ Back")
        self._back.clicked.connect(self._prev)
        self._next = QPushButton("Next ▶")
        self._next.clicked.connect(self._next_clip)
        nav = QHBoxLayout()
        nav.addWidget(self._back)
        nav.addStretch(1)
        nav.addWidget(self._next)

        layout = QVBoxLayout(self)
        layout.setMenuBar(menu_bar)
        layout.addLayout(top)
        layout.addWidget(self._counter)
        layout.addWidget(self._thumb, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._caption)
        layout.addWidget(self._warning)
        layout.addWidget(QLabel("Title"))
        layout.addLayout(self._title_row)
        layout.addWidget(self._title_note)
        layout.addWidget(QLabel("Description"))
        layout.addLayout(self._desc_row, 1)
        layout.addLayout(nav)

        self._apply_copy_side()  # order the Copy buttons per the saved preference

    # ----- loading & navigation -------------------------------------------
    def _open_csv(self) -> None:
        # Fall back to the app's export root when nothing is loaded yet; the cwd of a
        # frozen app is meaningless to the user. A previously loaded CSV's dir wins.
        start = str(self._csv_dir or export_root())
        path, _ = QFileDialog.getOpenFileName(
            self, "Open export CSV", start, "CSV files (*.csv);;All files (*)")
        if path:
            self.load_csv(Path(path))

    def loaded_csv(self) -> Path | None:
        """The CSV currently loaded (the file _save_csv writes back to), or None when the
        assistant is empty. Lets the main window detect that an open assistant still points
        inside an old export folder it just deleted, and reload it onto the fresh export."""
        return self._csv_path

    def load_csv(self, csv_path: Path) -> None:
        try:
            with csv_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fields = list(reader.fieldnames or [])
        except OSError as exc:
            QMessageBox.warning(self, "Could not open CSV", str(exc))
            return
        if not rows:
            QMessageBox.warning(self, "Empty CSV", f"{csv_path.name} has no clips.")
            return
        self._rows = rows
        self._csv_dir = csv_path.parent
        self._csv_path = csv_path
        self._csv_fields = fields  # original column order, kept for save-back rewrites
        self._index = 0
        self._show_clip()

    def _show_clip(self) -> None:
        """Render the current clip (or the empty state when nothing is loaded)."""
        has_rows = bool(self._rows)
        self._back.setEnabled(has_rows and self._index > 0)
        self._next.setEnabled(has_rows and self._index < len(self._rows) - 1)
        self._title_copy.setEnabled(has_rows)
        self._desc_copy.setEnabled(has_rows)
        self._rename_tournament.setEnabled(has_rows)

        if not has_rows:
            self._counter.setText("")  # hint lives in the thumbnail box instead
            self._thumb.set_clip(None, None)
            self._caption.setText("")
            self._warning.setText("")
            self._syncing_title = True
            self._title.clear()
            self._syncing_title = False
            self._desc.setPlainText("")
            self._generated_desc = ""
            self._missing_player_note = ""
            self._set_title_note(None, 0, 0)
            return

        row = self._rows[self._index]
        self._counter.setText(f"Clip {self._index + 1} of {len(self._rows)}")
        title_template = self._settings.value(
            SETTING_TITLE_TEMPLATE, DEFAULT_TITLE_TEMPLATE, type=str)
        self._title_char_limit = self._settings.value(
            SETTING_TITLE_CHAR_LIMIT, DEFAULT_TITLE_CHAR_LIMIT, type=int)
        self._missing_player_note = _missing_player_note(row)
        title, action = render_fitted_title(title_template, row, self._title_char_limit)
        self._syncing_title = True  # a programmatic set, not a hand-edit
        self._title.setText(title)
        self._syncing_title = False
        self._set_title_note(action, len(title), self._title_char_limit)
        desc_template = self._settings.value(
            SETTING_DESC_TEMPLATE, DEFAULT_DESC_TEMPLATE, type=str)
        self._generated_desc = render_description(desc_template, row, title=title)
        self._desc.setPlainText(self._generated_desc)

        assert self._csv_dir is not None
        clip_path = self._csv_dir / row.get("filename", "")
        if row.get("filename") and clip_path.exists():
            self._warning.setText("")
            self._thumb.set_clip(clip_path, _thumbnail(clip_path) or self._fallback_icon())
            self._caption.setText("Drag the clip onto the video upload screen")
        else:
            self._warning.setText(f"Clip file missing: {row.get('filename', '(no filename)')}")
            self._thumb.set_clip(None, None)  # non-draggable; title/description still usable
            self._caption.setText("")  # nothing to drag

    def _fallback_icon(self) -> QPixmap:
        style = self.style()
        assert style is not None
        return style.standardIcon(QStyle.StandardPixmap.SP_FileIcon).pixmap(96, 96)

    def _prev(self) -> None:
        if self._index > 0:
            self._index -= 1
            self._show_clip()

    def _next_clip(self) -> None:
        if self._index < len(self._rows) - 1:
            self._index += 1
            self._show_clip()

    # ----- misc ------------------------------------------------------------
    def _apply_theme(self) -> None:
        # getattr guard: a PaletteChange can arrive during construction, before _build_ui runs.
        # Idempotent (skip-if-unchanged) for the same anti-recursion reason as ClipDragLabel.
        warning = getattr(self, "_warning", None)
        if warning is None:
            return
        style = f"color: {warning_color()}; font-weight: bold;"
        if warning.styleSheet() != style:
            warning.setStyleSheet(style)

        title_note = getattr(self, "_title_note", None)
        if title_note is None:
            return
        # "too_long" (the title still exceeds the limit) and a missing player slot both warn;
        # the other reasons just explain why the title differs from the CSV, so they're muted.
        warn = self._title_note_kind == "too_long" or bool(self._missing_player_note)
        note_color = warning_color() if warn else muted_text_color()
        note_style = f"color: {note_color};"
        if title_note.styleSheet() != note_style:
            title_note.setStyleSheet(note_style)

    def _copy_title(self) -> None:
        """Copy the title and focus its box; a visible cue for which field is on the clipboard."""
        copy_with_feedback(self._title.text(), self._title_copy)
        self._title.setFocus(Qt.FocusReason.OtherFocusReason)

    def _copy_description(self) -> None:
        """Copy the description and focus its box; a visible cue for which field is on the
        clipboard."""
        copy_with_feedback(self._desc.toPlainText(), self._desc_copy)
        self._desc.setFocus(Qt.FocusReason.OtherFocusReason)

    def _set_title_note(self, action: str | None, length: int, limit: int) -> None:
        """Show why the generated title was shortened (muted) or that it's still too long
        (warning), plus the current clip's missing-player warning when there is one (see
        _missing_player_note, refreshed per clip in _show_clip); clear the note when neither
        has anything to say."""
        self._title_note_kind = action
        parts = []
        if action is not None:
            parts.append(_TITLE_FIT_NOTES[action].format(length=length, limit=limit))
        if self._missing_player_note:
            parts.append(self._missing_player_note)
        self._title_note.setText("\n".join(parts))
        self._apply_theme()

    def _on_title_edited(self) -> None:
        """Live feedback while the user edits the title: a character count (as the "too_long"
        warning) whenever the text exceeds the limit, cleared once it fits again. The
        description re-renders so its {title} tracks the box; but only while the description
        is still untouched generated text, so a hand-tweaked description is never clobbered."""
        if self._syncing_title or not self._rows:
            return  # a programmatic update (_show_clip), or the empty state clearing the box
        title = self._title.text()
        if len(title) > self._title_char_limit:
            self._set_title_note("too_long", len(title), self._title_char_limit)
        else:
            self._set_title_note(None, 0, 0)
        if self._desc.toPlainText() == self._generated_desc:
            desc_template = self._settings.value(
                SETTING_DESC_TEMPLATE, DEFAULT_DESC_TEMPLATE, type=str)
            self._generated_desc = render_description(
                desc_template, self._rows[self._index], title=title)
            self._desc.setPlainText(self._generated_desc)

    def changeEvent(self, a0: QEvent | None) -> None:
        if a0 is not None and a0.type() in _THEME_EVENTS:
            self._apply_theme()
        super().changeEvent(a0)

    def _set_copy_on_left(self, on: bool) -> None:
        self._settings.setValue(SETTING_COPY_ON_LEFT, on)
        self._apply_copy_side()

    def _apply_copy_side(self) -> None:
        """Place each Copy button left or right of its field per the saved preference."""
        left = self._settings.value(SETTING_COPY_ON_LEFT, False, type=bool)
        for title_widget in (self._title, self._title_copy):
            self._title_row.removeWidget(title_widget)
        for desc_widget in (self._desc, self._desc_copy):
            self._desc_row.removeWidget(desc_widget)
        top = Qt.AlignmentFlag.AlignTop  # Copy buttons anchor to the top as the fields grow
        if left:
            self._title_row.addWidget(self._title_copy, 0, top)
            self._title_row.addWidget(self._title)
            self._desc_row.addWidget(self._desc_copy, 0, top)
            self._desc_row.addWidget(self._desc, 1)
        else:
            self._title_row.addWidget(self._title)
            self._title_row.addWidget(self._title_copy, 0, top)
            self._desc_row.addWidget(self._desc, 1)
            self._desc_row.addWidget(self._desc_copy, 0, top)

    def _set_stay_on_top(self, on: bool) -> None:
        """Toggle always-on-top and persist it. setWindowFlag hides the window, so capture
        visibility first, then re-show and bring it to the front so it doesn't vanish."""
        self._settings.setValue(SETTING_STAY_ON_TOP, on)
        was_visible = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, on)
        if was_visible:
            self.show()
            self.raise_()
            self.activateWindow()

    def _open_settings(self) -> None:
        dialog = UploadSettingsDialog(self._settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._show_clip()  # re-render with the new template

    def _change_tournament_name(self) -> None:
        """Options → Change tournament name: fix the {tournament} value for every clip of this
        export, then save the CSV so reopening the assistant keeps it. Generated titles and
        descriptions pick the new name up immediately; the clip files and export folder keep
        their old name (they never reach the upload); re-export from the main app if those
        matter too."""
        if not self._rows:
            return
        dialog = TournamentNameDialog(
            self._rows[self._index].get("tournament", ""), self,
            caption="Renames the tournament in the titles, descriptions, and output CSV. "
                    "Clip and folder names keep the old name. Export again from the Review "
                    "screen if you want to change those.")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._apply_tournament_name(dialog.name())

    def _apply_tournament_name(self, name: str) -> None:
        """Set the tournament column on every loaded row, persist the CSV, and re-render the
        current clip so the title/description show the new name."""
        for row in self._rows:
            row["tournament"] = name
        self._save_csv()
        self._show_clip()

    def _save_csv(self) -> None:
        """Write the loaded rows back to the CSV they came from, keeping the original column
        order (plus a tournament column appended for CSVs from before it existed)."""
        assert self._csv_path is not None
        fields = list(self._csv_fields)
        if "tournament" not in fields:
            fields.append("tournament")
        try:
            with self._csv_path.open("w", newline="", encoding="utf-8") as f:
                # extrasaction="ignore" tolerates the None key DictReader adds when a row had
                # more values than the header; never written back, same as it read.
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self._rows)
        except OSError as exc:
            QMessageBox.warning(self, "Could not save CSV", str(exc))



def main() -> None:
    # Crash capture before the QApplication, same as app.py main() (see vodparse.gui.crash_log).
    crash_log.install("vodparse.gui.upload_assistant")
    app = QApplication(sys.argv)
    # Follow the OS light/dark setting on Windows/Linux, same as app.py main().
    apply_system_theme(app)
    # Share the main GUI's settings store (same identity); see app.py main().
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    app.setOrganizationName("Ultimate VOD Studio")
    app.setOrganizationDomain("ultimatevodstudio.app")
    app.setApplicationName("Ultimate VOD Studio")
    # Honour the same "Text size" preference as the main app (shared settings store).
    apply_font_scale(app, QSettings().value(SETTING_FONT_SCALE, FONT_SCALE_DEFAULT, type=float))
    window = UploadAssistant()
    if len(sys.argv) > 1:  # optional: python -m vodparse.gui.upload_assistant path/to/export.csv
        window.load_csv(Path(sys.argv[1]))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
