"""Options popup for the Review screen's Tools ▸ "Split sponsor from tag…" batch op.

Two length-bound spinboxes (kept min ≤ max), a max-words spinbox, and an all-caps toggle
configure ``split_leading_sponsor`` before the sweep runs; the shared excludable-preview base
(see ``preview_dialog.ExcludablePreviewDialog``) supplies the live Sponsor/Tag table, the
per-row "Exclude from split" toggle, in-place editing of both cells, the header count, and the
Apply gate. Apply persists the four control values (so the popup reopens the way the user left
it) and doubles as the confirm: the caller reads ``values()``, ``excluded_tags()`` and
``overrides()`` on an ``Accepted`` exec and runs ``tag_fixes.sponsor_splits`` itself.
Exclusions and edits are per-open only, keyed on each row's pre-sweep tag text, and never
persisted.
"""

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vodparse.gui.review.model import (
    ReviewClip,
    SETTING_SPLIT_SPONSOR_ALL_CAPS,
    SETTING_SPLIT_SPONSOR_MAX_LEN,
    SETTING_SPLIT_SPONSOR_MAX_WORDS,
    SETTING_SPLIT_SPONSOR_MIN_LEN,
)
from vodparse.gui.review.preview_dialog import ExcludablePreviewDialog, PreviewRow
from vodparse.gui.review.tag_fixes import (
    SPONSOR_MAX_LEN_DEFAULT,
    SPONSOR_MAX_WORDS_DEFAULT,
    SPONSOR_MIN_LEN_DEFAULT,
    sponsor_split_preview,
)

# Sponsor tokens are usually short acronyms ("ORG", "FAM"), but full-word sponsor names
# turn up too, so the max is effectively uncapped (a QSpinBox still needs some ceiling).
_SPONSOR_LEN_MIN = 1
_SPONSOR_LEN_MAX = 99

# Multi-sponsor handles ("ORG SP PlayerFour") span a few words at most; 9 is far past real use.
_SPONSOR_WORDS_MIN = 1
_SPONSOR_WORDS_MAX = 9

# Wide enough that the spinbox value never looks cramped, whatever the platform style.
_LEN_SPINBOX_WIDTH = 64


class SponsorSplitOptionsDialog(ExcludablePreviewDialog):
    """Options for peeling a colour-merged sponsor off the front of a gamertag.

    Two length-bound spinboxes (kept min ≤ max), a max-words spinbox, and an all-caps toggle
    configure ``split_leading_sponsor`` (see tag_fixes.py) before the sweep runs; the base class
    renders the live Sponsor/Tag preview, the per-row exclude toggle, and in-place edits of both
    cells. Accept persists the four spinbox/checkbox values to ``settings`` under the
    SETTING_SPLIT_SPONSOR_* keys; the caller pulls the final values back out via ``values()``,
    the excluded tags via ``excluded_tags()``, and the hand-edited (sponsor, tag) pairs via the
    base's ``overrides()`` (already sweep-shaped for ``sponsor_splits``).
    """

    def __init__(self, settings: QSettings, clips: list[ReviewClip],
                 parent: QWidget | None = None) -> None:
        self._settings = settings
        super().__init__(
            clips, parent,
            window_title="Split sponsor from tag",
            header_labels=["Sponsor", "Tag"],
            empty_preview_text="Nothing to split",
            exclude_noun="split",
            editable_columns=(0, 1))   # both the Sponsor and Tag results are hand-editable

    def _add_controls(self, layout: QVBoxLayout) -> None:
        self._min_len = QSpinBox()
        self._min_len.setRange(_SPONSOR_LEN_MIN, _SPONSOR_LEN_MAX)
        self._min_len.setMinimumWidth(_LEN_SPINBOX_WIDTH)
        self._min_len.setValue(
            self._settings.value(SETTING_SPLIT_SPONSOR_MIN_LEN, SPONSOR_MIN_LEN_DEFAULT, type=int))
        min_row = QHBoxLayout()
        min_row.addWidget(QLabel("Min word length:"))
        min_row.addWidget(self._min_len)
        min_row.addStretch(1)

        self._max_len = QSpinBox()
        self._max_len.setRange(_SPONSOR_LEN_MIN, _SPONSOR_LEN_MAX)
        self._max_len.setMinimumWidth(_LEN_SPINBOX_WIDTH)
        self._max_len.setValue(
            self._settings.value(SETTING_SPLIT_SPONSOR_MAX_LEN, SPONSOR_MAX_LEN_DEFAULT, type=int))
        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max word length:"))
        max_row.addWidget(self._max_len)
        max_row.addStretch(1)

        # Keep max >= min: nudging one past the other drags its partner along instead of
        # allowing an inverted range. Wired after both are seeded above.
        self._min_len.valueChanged.connect(self._raise_max_to_min)
        self._max_len.valueChanged.connect(self._lower_min_to_max)

        # How many leading words one sponsor may span; the peel always keeps the tag's
        # last word regardless, so a high value can't empty a tag.
        self._max_words = QSpinBox()
        self._max_words.setRange(_SPONSOR_WORDS_MIN, _SPONSOR_WORDS_MAX)
        self._max_words.setMinimumWidth(_LEN_SPINBOX_WIDTH)
        self._max_words.setValue(
            self._settings.value(SETTING_SPLIT_SPONSOR_MAX_WORDS, SPONSOR_MAX_WORDS_DEFAULT,
                                 type=int))
        words_row = QHBoxLayout()
        words_row.addWidget(QLabel("Max words in sponsor:"))
        words_row.addWidget(self._max_words)
        words_row.addStretch(1)

        self._all_caps = QCheckBox("Only take fully UPPERCASE sponsors")
        self._all_caps.setChecked(
            self._settings.value(SETTING_SPLIT_SPONSOR_ALL_CAPS, False, type=bool))

        # Every control refreshes the preview; the min/max clamp above may re-enter via the
        # partner spinbox's valueChanged, which just refreshes again with the settled values.
        self._min_len.valueChanged.connect(self._refresh_preview)
        self._max_len.valueChanged.connect(self._refresh_preview)
        self._max_words.valueChanged.connect(self._refresh_preview)
        self._all_caps.toggled.connect(self._refresh_preview)

        layout.addLayout(min_row)
        layout.addLayout(max_row)
        layout.addLayout(words_row)
        layout.addWidget(self._all_caps)

    def _raise_max_to_min(self, min_value: int) -> None:
        """Bump the max spinbox up if the min was raised past it."""
        if min_value > self._max_len.value():
            self._max_len.setValue(min_value)

    def _lower_min_to_max(self, max_value: int) -> None:
        """Pull the min spinbox down if the max was lowered past it."""
        if max_value < self._min_len.value():
            self._min_len.setValue(max_value)

    def _preview_rows(self) -> list[PreviewRow]:
        min_len, max_len, max_words, all_caps = self.values()
        # original is the slot's pre-sweep (Sponsor, Tag): the sponsor it already carried and its
        # tag before any peel, so typing that pair back over the computed split is the row's
        # no-op state.
        return [PreviewRow(before=row.tag_before, col0=row.sponsor, col1=row.tag, slots=row.slots,
                           original=(row.sponsor_before, row.tag_before))
                for row in sponsor_split_preview(self._clips, min_len=min_len, max_len=max_len,
                                                 max_words=max_words, all_caps=all_caps)]

    def accept(self) -> None:
        self._settings.setValue(SETTING_SPLIT_SPONSOR_MIN_LEN, self._min_len.value())
        self._settings.setValue(SETTING_SPLIT_SPONSOR_MAX_LEN, self._max_len.value())
        self._settings.setValue(SETTING_SPLIT_SPONSOR_MAX_WORDS, self._max_words.value())
        self._settings.setValue(SETTING_SPLIT_SPONSOR_ALL_CAPS, self._all_caps.isChecked())
        super().accept()

    def values(self) -> tuple[int, int, int, bool]:
        """The chosen (min_len, max_len, max_words, all_caps), read live from the widgets."""
        return (self._min_len.value(), self._max_len.value(), self._max_words.value(),
                self._all_caps.isChecked())
