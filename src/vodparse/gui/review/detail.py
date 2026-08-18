"""The Review screen's editable detail form (see screen.py): one column per player
(sponsor + tag row, character), the ⇄ swap column between them, a per-player "Options"
button + menu, and the shared Round info column.

Pure presentation: the panel owns the widgets and layout; ReviewScreen owns the clip model,
the undo bursts, and the option menus' contents; it reads/writes fields via ``field()`` /
``set_clip_fields()`` and wires the exposed swap button and menus to its own handlers.
Every field edit (user keystroke or programmatic ``setText``) emits ``fieldEdited(key)``;
the caller's loading flag distinguishes the two, so the panel never blocks signals itself.

Import this only after ``qt_plugins.configure_qt_plugins()`` has run (every app entry
module does so at its top), since it imports PyQt6 widgets at load time.
"""

from typing import cast

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vodparse.gui.review.model import ReviewClip, nth_or_blank
from vodparse.gui.review.widgets import FieldLineEdit, group_label, vline_separator

# The detail panel's group headers ("Player 1", "Round info") get a small size bump over the
# ops-bar group labels: they head editable content, not button clusters.
DETAIL_HEADER_PT_BUMP = 3.0

# Every editable field, by key: the contract between the panel and ReviewScreen's
# write-back/populate code (see field() / set_clip_fields()).
FIELD_KEYS = ("tag_1", "tag_2", "sponsor_1", "sponsor_2",
              "char_1", "char_2", "round", "best_of")


class ReviewDetailPanel(QWidget):
    """See the module docstring.

    Signals: ``fieldEdited(key)`` on any field text change (user or programmatic, the
    caller's loading flag distinguishes them); ``fieldReleased()`` when Enter/Escape is
    pressed in a field (the caller moves focus on); ``fieldFocusLeft()`` when focus leaves
    a field (the caller ends its undo typing burst)."""

    fieldEdited = pyqtSignal(str)
    fieldReleased = pyqtSignal()
    fieldFocusLeft = pyqtSignal()

    def __init__(self) -> None:
        """Build the whole fields row: the editable fields grouped by object, one column per
        player plus the shared round info, mirroring the ops bar's grouping (bold group labels,
        separators). The columns sit side by side so the field block stays shallow and the
        video below keeps the vertical space."""
        super().__init__()
        self._fields: dict[str, QLineEdit] = {}

        def field(key: str) -> QLineEdit:
            edit = FieldLineEdit(on_release=self.fieldReleased.emit,
                                 on_leave=self.fieldFocusLeft.emit)
            # The key tells the caller which field is being typed in, so consecutive
            # keystrokes there coalesce into one undo entry.
            edit.textChanged.connect(lambda _="", k=key: self.fieldEdited.emit(k))
            self._fields[key] = edit
            return edit

        def tag_row(sponsor_key: str, tag_key: str) -> QHBoxLayout:
            """Sponsor prefix + tag on one row, mirroring start.gg's "PRO | ABC".
            The "Tag" label lives inside the row (the row spans the form, see group()) so the
            sponsor box starts right after it instead of at the shared label column, tags
            run long, so the tag field claims the dead space a column-aligned label leaves."""
            sponsor = field(sponsor_key)
            sponsor.setPlaceholderText("Spons.")
            # Sized for typical 2–5 character sponsors ("PRO", "ORG"); the tag takes the rest.
            # Fixed (not just max) so the box can't shrink and shift the tag's left edge. Derived
            # from the font (the "Spons." placeholder plus ~2 chars of frame/cursor room) so it
            # grows with the Text-size preference instead of clipping.
            sfm = sponsor.fontMetrics()
            sponsor.setFixedWidth(sfm.horizontalAdvance("Spons.") + 2 * sfm.averageCharWidth())
            label = QLabel("Tag")
            tag = field(tag_key)
            # A width floor so the tag stays usable as the panel narrows, it has all the row's
            # stretch and would otherwise collapse toward the sponsor box. ~10 chars fits a
            # typical gamertag while keeping the Review window's minimum width comfortably inside
            # narrow laptop screens; derived from the font so it holds up across DPIs/fonts.
            tag.setMinimumWidth(tag.fontMetrics().horizontalAdvance("0" * 10))
            row = QHBoxLayout()
            row.addWidget(label)
            row.addWidget(sponsor)
            row.addWidget(tag, 1)
            return row

        def group(title: str, rows: list[tuple[str | None, str | QLayout]],
                  header_pt_bump: float = 0.0) -> QVBoxLayout:
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            # macOS style defaults to FieldsStayAtSizeHint, which leaves single fields
            # (Character, Round) narrower than layout rows like sponsor+tag, grow them all
            # to the same column width instead.
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            # The macOS style also centers the form within its column and right-aligns the
            # labels ("Tag" floats away from its left edge once the sponsor box widens the
            # field column), pin both left.
            form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
            for label, value in rows:
                if label is None:
                    # spanning row that carries its own label (tag_row), always a layout here
                    form.addRow(cast(QLayout, value))
                else:
                    form.addRow(label, field(value) if isinstance(value, str) else value)
            header = group_label(title)
            if header_pt_bump:
                font = header.font()
                font.setPointSizeF(font.pointSizeF() + header_pt_bump)
                header.setFont(font)
            box = QVBoxLayout()
            box.addWidget(header)
            box.addLayout(form)
            box.addStretch(1)  # top-align the shorter Round info column
            return box

        p1_col = group("Player 1", [(None,        tag_row("sponsor_1", "tag_1")),
                                    ("Character", "char_1")], DETAIL_HEADER_PT_BUMP)
        p2_col = group("Player 2", [(None,        tag_row("sponsor_2", "tag_2")),
                                    ("Character", "char_2")], DETAIL_HEADER_PT_BUMP)
        swap_col = self._build_swap_column()

        # Player columns get double the Round info column's width: they hold the
        # sponsor+tag pair, while Round/Best of are short values.
        fields_row = QHBoxLayout(self)
        fields_row.setContentsMargins(0, 0, 0, 0)
        fields_row.addLayout(p1_col, 2)
        fields_row.addLayout(swap_col)
        fields_row.addLayout(p2_col, 2)
        self._build_player_option_menus(p1_col, p2_col)

        round_col = group("Round info", [("Round",   "round"),
                                         ("Best of", "best_of")], DETAIL_HEADER_PT_BUMP)
        fields_row.addWidget(vline_separator())
        fields_row.addLayout(round_col, 1)

    # ── Field access (keys: FIELD_KEYS) ──────────────────────────────────────

    def field(self, key: str) -> QLineEdit:
        """The live edit widget for `key`: for reads (the caller's write-back), writes
        (populate/swap), and focus checks."""
        return self._fields[key]

    def set_clip_fields(self, clip: ReviewClip) -> None:
        """Write all eight fields from the clip dict (the populate half of selection and of the
        whole-VOD batch ops). Each setText fires fieldEdited, so the caller wraps this in its
        loading guard."""
        self._fields["tag_1"].setText(nth_or_blank(clip["grouping_tags"], 0))
        self._fields["tag_2"].setText(nth_or_blank(clip["grouping_tags"], 1))
        self._fields["sponsor_1"].setText(nth_or_blank(clip.get("tournament_sponsors"), 0))
        self._fields["sponsor_2"].setText(nth_or_blank(clip.get("tournament_sponsors"), 1))
        self._fields["char_1"].setText(nth_or_blank(clip["characters"], 0))
        self._fields["char_2"].setText(nth_or_blank(clip["characters"], 1))
        self._fields["round"].setText(clip["round"])
        self._fields["best_of"].setText(clip["best_of"])

    def set_fallback_note(self, tip: str) -> None:
        """Set (or clear, with "") the in-game-fallback tooltip on both Tag fields: the caller
        computes the note from the clip so it always agrees with the row tint."""
        self._fields["tag_1"].setToolTip(tip)
        self._fields["tag_2"].setToolTip(tip)

    # ── Widgets ReviewScreen wires into its ops bar / menus ──────────────────
    # Exposed as read-only properties rather than relay signals: the screen's _ops dict
    # registers the buttons for enablement/visibility refresh, and the menus' contents are
    # rebuilt from screen-owned clip data on each open.

    @property
    def swap_button(self) -> QPushButton:
        """The ⇄ swap-characters button; the screen connects clicked and drives enablement."""
        return self._swap_btn

    @property
    def player_option_buttons(self) -> tuple[QPushButton, QPushButton]:
        """The per-player "Options" buttons, registered in the screen's _ops dict."""
        return self._player_option_btns[0], self._player_option_btns[1]

    @property
    def player_menus(self) -> tuple[QMenu, QMenu]:
        """The per-player Options menus; the screen connects aboutToShow and fills them."""
        return self._player_menus[0], self._player_menus[1]

    # ── Construction helpers ──────────────────────────────────────────────────

    def _build_swap_column(self) -> QVBoxLayout:
        """The ⇄ swap-characters button in its own column between the two players. Stream overlays
        sometimes show the players on the wrong sides, so the side-keyed character detection lands
        on the wrong player; one click swaps them back. The column mirrors the player columns'
        structure (blank header, a Tag-row-high spacer, then the button as the Character row) so ⇄
        lands beside the Character fields no matter how tall the fields row gets."""
        self._swap_btn = swap_btn = QPushButton("⇄")
        swap_btn.setToolTip("Swap the two players' characters")
        # Just wide enough for the ⇄ glyph plus button padding, derived from the font so it grows
        # with the Text-size preference instead of clipping.
        swap_fm = swap_btn.fontMetrics()
        swap_btn.setFixedWidth(swap_fm.horizontalAdvance("⇄") + 2 * swap_fm.averageCharWidth())
        # Spacer height tuned by eye: the cocoa button's rendered bezel is taller than its
        # sizeHint reports, so this trims more than the pure geometry would suggest.
        char_h = self._fields["char_1"].sizeHint().height()
        extra  = max(0, swap_btn.sizeHint().height() - char_h // 2)
        swap_spacer = QWidget()
        swap_spacer.setFixedHeight(self._fields["tag_1"].sizeHint().height() - extra // 2)

        swap_form = QFormLayout()
        swap_form.setContentsMargins(0, 0, 0, 0)
        swap_form.addRow(swap_spacer)
        swap_form.addRow(swap_btn)
        swap_col = QVBoxLayout()
        swap_col.addWidget(group_label(" "))   # blank header, same top offset as Player 1/2
        swap_col.addLayout(swap_form)
        swap_col.addStretch(1)
        return swap_col

    def _build_player_option_menus(self, p1_col: QVBoxLayout, p2_col: QVBoxLayout) -> None:
        """Add a per-player "Options" button + menu to the bottom of each player's column: every
        propagation action for that player in one menu, rebuilt with live candidacy on each open
        (the screen's _rebuild_player_menu, via the aboutToShow connection it makes). Registered
        in the screen's _ops by _build_ops_bar, which handles enablement refresh and match-mode
        hiding. Future per-player actions belong in the same menu."""
        self._player_option_btns: list[QPushButton] = []
        self._player_menus: list[QMenu] = []
        for i, col in enumerate((p1_col, p2_col)):
            btn = QPushButton(f"Player {i + 1} options")
            menu = QMenu(btn)
            menu.setToolTipsVisible(True)
            btn.setMenu(menu)
            btn.setEnabled(False)
            col.insertWidget(2, btn)   # after the form, before the stretch
            self._player_option_btns.append(btn)
            self._player_menus.append(menu)
