"""Options popup for the Review screen's Tools ▸ "Fix capitalization across all games…" batch
op.

Three mode radios (Title Case / Sentence case / all lowercase) pick the ``CapMode`` the sweep
runs under; the shared excludable-preview base (see ``preview_dialog.ExcludablePreviewDialog``)
supplies the live Before/After table, the per-row "Exclude from fix" toggle, in-place editing
of the After cell (the Before cell is the row's identity and stays read-only), the header
count, and the Apply gate. Apply persists the chosen mode (so the popup reopens the way the
user left it) and doubles as the confirm: the caller reads ``values()``, ``excluded_tags()``
and ``tag_overrides()`` on an ``Accepted`` exec and runs ``tag_fixes.capitalization_fixes``
itself. Exclusions and edits are per-open only, keyed on each row's pre-sweep tag text, and
never persisted.
"""

from typing import get_args

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from vodparse.gui.review.model import ReviewClip, SETTING_FIX_CAPS_MODE
from vodparse.gui.review.preview_dialog import ExcludablePreviewDialog, PreviewRow
from vodparse.gui.review.tag_fixes import CAP_MODE_DEFAULT, CapMode, capitalization_preview


class FixCapsOptionsDialog(ExcludablePreviewDialog):
    """Options for re-casing every single-case gamertag across the VOD.

    Three mutually-exclusive radios pick the ``CapMode`` that configures
    ``capitalization_fixes`` (see tag_fixes.py) before the sweep runs; the base class renders
    the live Before/After preview, the per-row exclude toggle, and in-place edits of the After
    cell. Accept persists the chosen mode to ``settings`` under ``SETTING_FIX_CAPS_MODE``; the
    caller pulls the final mode back out via ``values()``, the excluded tags via
    ``excluded_tags()``, and the hand-edited results via ``tag_overrides()``.
    """

    def __init__(self, settings: QSettings, clips: list[ReviewClip],
                 parent: QWidget | None = None) -> None:
        self._settings = settings
        super().__init__(
            clips, parent,
            window_title="Fix capitalization",
            header_labels=["Before", "After"],
            empty_preview_text="No all caps tags to fix!",
            exclude_noun="fix",
            editable_columns=(1,))   # After only; Before is the row's identity key

    def _add_controls(self, layout: QVBoxLayout) -> None:
        # Validate against CapMode's own members (the single source of truth in tag_fixes), so a
        # future mode can't be silently rejected here as an "unknown" stored value.
        stored_mode = self._settings.value(SETTING_FIX_CAPS_MODE, CAP_MODE_DEFAULT, type=str)
        if stored_mode not in get_args(CapMode):
            stored_mode = CAP_MODE_DEFAULT

        # Mutually exclusive via QButtonGroup; each radio is also kept on self so values()
        # can read the checked one directly without walking the group.
        self._mode_group = QButtonGroup(self)
        self._mode_title = QRadioButton("Title Case")
        self._mode_sentence = QRadioButton("Sentence case")
        self._mode_lower = QRadioButton("all lowercase")
        for radio in (self._mode_title, self._mode_sentence, self._mode_lower):
            self._mode_group.addButton(radio)
        self._mode_title.setChecked(stored_mode == "title")
        self._mode_sentence.setChecked(stored_mode == "sentence")
        self._mode_lower.setChecked(stored_mode == "lower")
        # Seed above, then wire the live preview (so seeding doesn't fire a premature refresh).
        for radio in (self._mode_title, self._mode_sentence, self._mode_lower):
            radio.toggled.connect(self._refresh_preview)

        mode_row = QHBoxLayout()
        mode_row.addWidget(self._mode_title)
        mode_row.addWidget(self._mode_sentence)
        mode_row.addWidget(self._mode_lower)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

    def _preview_rows(self) -> list[PreviewRow]:
        # original is (Before, Before): the pre-sweep slot showed its tag verbatim in both the
        # read-only Before column and (conceptually) as the not-yet-re-cased After, so typing the
        # Before text back into After is the row's no-op state.
        return [PreviewRow(before=row.before, col0=row.before, col1=row.after, slots=row.slots,
                           original=(row.before, row.before))
                for row in capitalization_preview(self._clips, mode=self.values())]

    def accept(self) -> None:
        self._settings.setValue(SETTING_FIX_CAPS_MODE, self.values())
        super().accept()

    def values(self) -> CapMode:
        """The chosen capitalization mode, read live from the radios."""
        if self._mode_sentence.isChecked():
            return "sentence"
        if self._mode_lower.isChecked():
            return "lower"
        return "title"

    def tag_overrides(self) -> dict[str, str]:
        """The base overrides reshaped for ``capitalization_fixes``: pre-sweep tag text -> the
        hand-edited final tag text (the After cell; the Before cell is never editable)."""
        return {before: pair[1] for before, pair in self.overrides().items()}
