"""Small shared widgets/helpers for the Review screen (screen.py).

Import these only after ``qt_plugins.configure_qt_plugins()`` has run (every app entry
module does so at its top), since this module imports PyQt6 widgets at load time.
"""

from collections.abc import Callable
from typing import override

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFocusEvent, QKeyEvent
from PyQt6.QtWidgets import QFrame, QLabel, QLineEdit

from vodparse.gui.common import bold_label


def group_label(text: str) -> QLabel:
    """Section label naming a control group (ops-bar clusters, field columns). Thin alias for
    the shared bold_label helper."""
    return bold_label(text)


def vline_separator() -> QFrame:
    """A thin sunken vertical divider between control groups (ops-bar columns, detail panel)."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def nn[T](value: T | None) -> T:
    """Unwrap a Qt return that the stubs mark Optional but is never None at this call site."""
    assert value is not None
    return value


class FieldLineEdit(QLineEdit):
    """A detail-panel edit field that releases focus on Enter or Escape and ends the undo
    typing burst whenever focus leaves it.

    The field edits the selected clip live (every keystroke syncs through _on_field_changed),
    so Enter/Escape have nothing to commit or revert; they just release focus (on_release,
    which hands it to the set tree) so the value is "left" and keyboard navigation (and the
    window-level Undo shortcut) resume. Ctrl+Z stays QLineEdit's built-in field-local undo
    while the field is focused, by design; document-level undo is the Undo button, or the
    keyboard shortcut once focus is elsewhere.

    Ending the burst (on_leave) on focus-out (Enter/Escape/click-away/window blur) means
    re-entering a field always starts a fresh undo entry instead of coalescing with the
    pre-blur edit."""

    def __init__(self, *, on_release: Callable[[], None], on_leave: Callable[[], None]) -> None:
        super().__init__()
        self._on_release = on_release
        self._on_leave = on_leave

    @override
    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is not None and event.key() in (
                Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape):
            self._on_release()   # value is already live; just leave the field
            event.accept()
            return
        super().keyPressEvent(event)

    @override
    def focusOutEvent(self, event: QFocusEvent | None) -> None:
        super().focusOutEvent(event)
        self._on_leave()
