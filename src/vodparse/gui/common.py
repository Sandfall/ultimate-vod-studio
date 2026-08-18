"""Small Qt helpers shared by the desktop apps (app.py, vodparse.gui.review.screen,
upload_assistant.py) and their background-task front ends (gui.gpu_support, gui.updates).

Import these only after ``qt_plugins.configure_qt_plugins()`` has run (every app does so at the
top of its entry module), since this module imports PyQt6 widgets at load time.
"""

import sys
from typing import override

from PyQt6 import sip
from PyQt6.QtCore import (QObject, QProcessEnvironment, QSize, Qt, QTimer,
                          pyqtSignal, pyqtSlot)
from PyQt6.QtGui import QColor, QFont, QPalette, QResizeEvent
from PyQt6.QtWidgets import (QApplication, QLabel, QProgressDialog, QPushButton,
                             QRadioButton, QSizePolicy, QWidget)


def qobject_alive(obj: QObject) -> bool:
    """True while a Qt object's underlying C++ instance still exists.

    A fire-and-forget QProcess, QThread or dialog is routinely torn down before the slot that
    was waiting on it runs: the Settings dialog closed with Escape while a compute probe is
    mid-flight, a busy dialog cancelled before its background check returns. Touching the
    Python wrapper afterwards raises ``RuntimeError: wrapped C/C++ object ... has been
    deleted``, so guard every deferred touch of a Qt object whose owner may already be gone
    with this.
    """
    return not sip.isdeleted(obj)


def stage_environment() -> QProcessEnvironment:
    """The environment for a GUI-spawned stage subprocess: the inherited system environment
    plus PYTHONUTF8=1. Stage children print over a pipe, and on a Windows source checkout the
    pipe's locale codec is cp1252; the stage mains reconfigure their own streams to UTF-8
    (``vodparse.cli.stage_io.prepare_stage_streams``), but third-party code can print before or
    around that reconfiguration: EasyOCR's "█" model-download progress bar on first run is the
    known case. PYTHONUTF8=1 forces UTF-8 mode for the whole interpreter, covering that gap and
    any grandchild processes the stage spawns. The frozen build has no such gap: it never touches
    a locale-dependent codec in the first place (``freeze_entry._force_utf8_streams``).

    Shared by the pipeline/extract/export runner (app._start_process) and the GPU-support
    installer/probe (gui.gpu_support) so the two can't drift."""
    env = QProcessEnvironment.systemEnvironment()
    env.insert("PYTHONUTF8", "1")
    return env


def apply_font_scale(app: QApplication, scale: float) -> None:
    """Scale every widget's text by `scale`, keeping the platform font family and weight.

    Call once at startup, right after the QApplication is created and before any widget is
    built: Qt's default font then cascades to nearly the whole UI by inheritance, and the
    fontMetrics-derived sizes throughout the app compute against the scaled font from the
    start (so nothing is left half-scaled). The chosen scale comes from the Settings → General
    "Text size" preset (FONT_SCALE_CHOICES); changing it takes effect on the next launch.
    """
    if scale == 1.0:
        return  # platform default: leave the app font untouched
    font = QFont(app.font())
    if font.pointSizeF() > 0:                       # desktop default fonts are point-sized
        font.setPointSizeF(font.pointSizeF() * scale)
    else:                                           # rare pixel-sized font: scale that instead
        font.setPixelSize(round(font.pixelSize() * scale))
    app.setFont(font)


# The standard dark-Fusion palette (the widely-used Qt dark theme recipe).
_DARK_WINDOW = QColor(53, 53, 53)         # window background, buttons, alternate rows
_DARK_BASE = QColor(35, 35, 35)           # text-entry and item-view background
_DARK_TOOLTIP = QColor(25, 25, 25)
_DARK_DISABLED_TEXT = QColor(127, 127, 127)
_ACCENT_BLUE = QColor(42, 130, 218)       # links and selection highlight


def _dark_fusion_palette() -> QPalette:
    """The standard Qt dark-Fusion palette, built explicitly so the whole palette-driven
    color system (is_dark_theme + the *_color helpers) reads dark."""
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, _DARK_WINDOW)
    p.setColor(QPalette.ColorRole.WindowText, QColor(Qt.GlobalColor.white))
    p.setColor(QPalette.ColorRole.Base, _DARK_BASE)
    p.setColor(QPalette.ColorRole.AlternateBase, _DARK_WINDOW)
    p.setColor(QPalette.ColorRole.ToolTipBase, _DARK_TOOLTIP)
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(Qt.GlobalColor.white))
    p.setColor(QPalette.ColorRole.Text, QColor(Qt.GlobalColor.white))
    p.setColor(QPalette.ColorRole.Button, _DARK_WINDOW)
    p.setColor(QPalette.ColorRole.ButtonText, QColor(Qt.GlobalColor.white))
    p.setColor(QPalette.ColorRole.BrightText, QColor(Qt.GlobalColor.red))
    p.setColor(QPalette.ColorRole.Link, _ACCENT_BLUE)
    p.setColor(QPalette.ColorRole.Highlight, _ACCENT_BLUE)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(Qt.GlobalColor.black))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText,
                 QPalette.ColorRole.WindowText):
        p.setColor(QPalette.ColorGroup.Disabled, role, _DARK_DISABLED_TEXT)
    return p


class SystemThemeFollower(QObject):
    """Applies the OS light/dark choice to the application, and re-applies it when it changes.

    Created by :func:`apply_system_theme`, which owns the one instance. **This being a QObject
    with a decorated slot is the fix for the Windows live theme-flip crash**, not a matter of
    style: connecting ``colorSchemeChanged`` to a plain local closure instead kills the process
    with an access violation the first time the OS setting is flipped under a running app.

    The fault is in *delivering* the signal, not in the work the slot does. Windows broadcasts
    the theme change as ``WM_SETTINGCHANGE``, which each window handles inside its window
    procedure, so the emission re-enters Python on a deep native callback stack; the crash logs
    stop before any widget sees the resulting ``PaletteChange``, and the app dies before the slot
    body runs at all. A QObject with an explicit parent, a known thread affinity and a
    ``@pyqtSlot``-decorated method is the receiver shape Qt documents for a connection meant to
    outlive the call that made it, and it survives that delivery where a closure — whose receiver
    PyQt synthesizes and whose lifetime it manages on your behalf — does not. Isolated by
    experiment; do not reduce this back to a closure or a lambda.

    The startup style and palette are captured at construction, so light mode can restore exactly
    the appearance the platform gave us rather than an approximation of it.
    """

    def __init__(self, app: QApplication) -> None:
        super().__init__(app)  # parented to the app: lives exactly as long as the connection
        self._app = app
        style = app.style()
        assert style is not None  # a QApplication always has a style
        self._native_style_name = style.objectName()
        self._native_palette = QPalette(app.palette())
        hints = app.styleHints()
        assert hints is not None  # a QGuiApplication always has style hints
        self._hints = hints

    @pyqtSlot()
    def sync(self) -> None:
        """Apply the OS scheme as it stands right now.

        Re-reads ``colorScheme()`` rather than trusting a signal argument, so rapid flips settle
        on the final state instead of replaying every intermediate one.
        """
        if self._hints.colorScheme() == Qt.ColorScheme.Dark:
            self._app.setStyle("Fusion")
            self._app.setPalette(_dark_fusion_palette())
        else:  # Light: or Unknown, where today's light appearance is the safe default
            self._app.setStyle(self._native_style_name)
            self._app.setPalette(self._native_palette)


def apply_system_theme(app: QApplication) -> None:
    """Follow the OS light/dark setting on Windows and Linux.

    macOS needs nothing: the Cocoa plugin mirrors the system appearance into the app
    palette on its own: so this returns immediately there (forcing a style would regress
    the native look). Elsewhere the platform plugin hands Qt a light palette regardless of
    the OS setting, and the native Windows style draws its chrome in light colors even
    under a dark palette; so in dark mode this switches to the Fusion style plus an
    explicit dark palette (the one combination Qt renders dark reliably), and in light
    mode restores the untouched native style and startup palette, keeping today's light
    appearance exactly. Re-syncs live when the OS setting changes.

    Call once at startup, right after the QApplication is created and before any widget
    is built. is_dark_theme() and the *_color() helpers follow automatically, since they
    read the live app palette.

    The work itself lives in :class:`SystemThemeFollower`, whose docstring explains why this
    connects a QObject's slot rather than a local function.
    """
    if sys.platform == "darwin":
        return
    follower = SystemThemeFollower(app)
    hints = app.styleHints()
    assert hints is not None  # a QGuiApplication always has style hints
    hints.colorSchemeChanged.connect(follower.sync)
    follower.sync()


def is_dark_theme() -> bool:
    """True when the app is running under a dark system palette.

    Uses the ``window()`` brush accessor rather than ``color(ColorRole.Window)``: the latter's
    overload resolution is inconsistent across PyQt6 versions and raises TypeError on some macOS
    builds; ``window().color()`` is unambiguous everywhere.
    """
    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app.palette().window().color().lightness() < 128
    return False


def muted_text_color() -> str:
    """Secondary-text color that stays readable in the current light/dark theme."""
    return "#a8a8a8" if is_dark_theme() else "#5a5a5a"


def warning_color() -> str:
    """Amber warning color, brightened on dark themes so it stays visible."""
    return "#ffb454" if is_dark_theme() else "#b35c00"


def notice_color() -> str:
    """Teal notice color for "worth your attention, nothing's wrong" notes: deliberately
    calmer than warning_color(). Brightened on dark themes so it stays visible."""
    return "#4dd0c4" if is_dark_theme() else "#0f766e"


def bold_label(text: str) -> QLabel:
    """A QLabel with bold (heading) weight at the default text color. Bold reads as a heading
    without the greyed-out look of PlaceholderText, and the default color role keeps it following
    the light/dark switch live."""
    label = QLabel(text)
    font = label.font()
    font.setBold(True)
    label.setFont(font)
    return label


class ElidingLabel(QLabel):
    """A single-line QLabel that elides long text (…) to its current width instead of widening
    the window to fit it. The full text is kept: text() still returns it whole (so callers
    comparing the label text keep working) and it rides along as the tooltip, while only the
    displayed string is clipped.

    `mode` picks where the ellipsis lands (ElideMiddle suits filenames, ElideRight running text).
    `min_chars` reserves a text-independent minimum width of that many average characters so
    short/medium strings show in full; at the default 0 the label never drives the layout's size.

    Two sizing behaviors:
      * default (fill): the Ignored horizontal policy leaves sizing entirely to the
        surrounding controls: the label takes whatever width the layout gives it
        (status lines, filename rows).
      * `natural_width=True` (hug): the label asks for its full text's width, so a widget
        placed right after it in the layout sits against the text, but it still squeezes
        down to the min_chars floor (eliding the display) instead of blocking the window
        from shrinking. Suits headings."""

    def __init__(self, text: str = "", *,
                 mode: Qt.TextElideMode = Qt.TextElideMode.ElideRight,
                 min_chars: int = 0,
                 natural_width: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full = text
        self._mode = mode
        self._natural_width = natural_width
        if not natural_width:
            self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(self.fontMetrics().averageCharWidth() * min_chars)
        self.setText(text)

    @override
    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        if self._natural_width:
            # Ask for the FULL text's width, not the currently-displayed (possibly elided)
            # text's: otherwise a once-squeezed label would never get its width back when
            # the window regrows.
            margins = self.contentsMargins()
            hint.setWidth(self.fontMetrics().size(0, self._full).width()
                          + margins.left() + margins.right())
        return hint

    @override
    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        if self._natural_width:
            # QLabel's own minimum is the full text width, which would pin the window that
            # wide; the explicit min_chars floor (minimumWidth) is the real lower bound.
            hint.setWidth(self.minimumWidth())
        return hint

    @override
    def setText(self, text: str | None) -> None:
        self._full = text or ""
        self.setToolTip(self._full)
        self._elide()

    @override
    def text(self) -> str:
        return self._full  # the full string, not the elided display text

    @override
    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        elided = self.fontMetrics().elidedText(self._full, self._mode, self.width())
        super().setText(elided)


def copy_with_feedback(text: str, button: QPushButton) -> None:
    """Copy `text` to the clipboard and flash 'Copied' on `button` for 1.2s (the copy is
    otherwise silent)."""
    clipboard = QApplication.clipboard()
    assert clipboard is not None
    clipboard.setText(text)
    button.setText("Copied")
    QTimer.singleShot(1200, lambda: button.setText("Copy"))


def selected_radio_value(radios: dict[str, QRadioButton], default: str) -> str:
    """The value whose radio isChecked(), else `default`."""
    for value, radio in radios.items():
        if radio.isChecked():
            return value
    return default


# ── Busy dialog over a background task ───────────────────────────────────────────────
# A QProgressDialog whose range is empty (0, 0) draws the indeterminate "busy" bar instead of a
# percentage: the right shape for work whose progress can't be measured (a network GET, a torch
# cold-load, a downloader that reports no bytes).
_BUSY_RANGE_MIN = 0
_BUSY_RANGE_MAX = 0
_BUSY_CANCEL_TEXT = "Cancel"


class BusyDialog(QObject):
    """Modal, cancelable, indeterminate "working…" dialog over a background task.

    Constructing one shows it, so the caller's next statement can start the work. That
    also puts the dialog on screen *before* the caller has connected :attr:`cancelled` —
    safe, because ``show()`` spins no event loop, so no cancel can arrive until the
    caller's wiring after the constructor is done. The user
    cancelling it (Cancel button, window close, Escape) emits :attr:`cancelled` once and
    retires the dialog; :meth:`close_quietly` retires it programmatically without that ever
    reading as a user cancel. Cancel *policy* stays with the caller: what backing out should
    do (kill the QProcess, set a flag) and how the finishing work reads the outcome
    (:attr:`was_cancelled`).

    Two Qt traps live here so that no caller has to rediscover them:

    * ``QProgressDialog`` emits ``canceled()`` from its own ``closeEvent``, which makes a
      programmatic ``close()`` indistinguishable from the user backing out. Every close path
      below drops that connection *before* closing. (Missing this is what once made a
      *successful* docTR model download report itself as cancelled.)
    * An indeterminate dialog never gets a ``setValue`` call, so Qt's automatic show doesn't
      trigger until its ~4-second force timer: far too late for work that usually resolves in
      well under a second. The constructor shows the dialog outright instead, which is also
      why ``setMinimumDuration(0)`` (the usual workaround) is deliberately not set: with an
      explicit show it does nothing.

    Single-use: once cancelled or closed, the dialog is gone and nothing more is ever emitted.
    Build another BusyDialog for another wait (the update check's cancel-then-re-click revival
    does exactly that). The helper is parented to `parent`, so it lives no longer than the
    window it belongs to, and :attr:`was_cancelled` / :meth:`close_quietly` stay safe to call
    even after that teardown.
    """

    cancelled = pyqtSignal()  # the user pressed Cancel, or closed the dialog

    def __init__(self, message: str, title: str, parent: QWidget) -> None:
        super().__init__(parent)
        dialog = QProgressDialog(message, _BUSY_CANCEL_TEXT,
                                 _BUSY_RANGE_MIN, _BUSY_RANGE_MAX, parent)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._dialog: QProgressDialog | None = dialog
        self._was_cancelled = False
        dialog.canceled.connect(self._on_canceled)
        dialog.show()  # see the class docstring: Qt's own auto-show comes ~4 seconds too late

    @property
    def was_cancelled(self) -> bool:
        """True once the user has cancelled, and sticky from then on. Background work usually
        can't be interrupted mid-flight, so its completion handler reads this to decide
        whether to present its result or drop it."""
        return self._was_cancelled

    def close_quietly(self) -> None:
        """Retire the dialog programmatically, without it reading as a user cancel.

        Idempotent, and safe once the parent window (and with it the dialog) has already been
        destroyed: both are no-ops."""
        dialog = self._dialog
        self._dialog = None
        if dialog is None or not qobject_alive(dialog):
            return
        # Drop the cancel connection BEFORE closing: QProgressDialog emits canceled() from its
        # closeEvent, so leaving it connected turns this close into a phantom user cancel.
        dialog.canceled.disconnect(self._on_canceled)
        dialog.close()
        dialog.deleteLater()

    def _on_canceled(self) -> None:
        """The user backed out: record it, release the dialog, and tell the caller."""
        self._was_cancelled = True
        dialog = self._dialog
        self._dialog = None
        if dialog is not None and qobject_alive(dialog):
            dialog.deleteLater()  # QProgressDialog hides itself on cancel; just release it
        self.cancelled.emit()
