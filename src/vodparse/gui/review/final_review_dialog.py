"""Final Review dialog: a table over the whole ReviewSession, one row per set, with a
character-swap button and its own Undo/Redo, for a single-glance pass across every set before
exporting. Reached from the Review screen's "Review and export" button (screen.py), which
sits alongside the existing "Export clips" button rather than replacing it. Sets excluded
from export stay listed but greyed + struck out, like the Review tree (see ``_style_row``):
keeping one row per clip preserves the row index == clip index assumption every table
lookup here relies on.

The dialog opens in **Compact** mode: a "Player 1"/"Player 2"/"Round" view with one
title-style composite per player per cell (``ORG | PlayerOne (Char1)`` — the same string the
exported title is built from, see titles.player_composite), the two sides in adjacent
columns so characters line up vertically for scanning. Compact is
read-only: rows stay selectable so Swap/Undo/Redo/Export all work, but editing requires
switching to **Detailed**, the original seven-column inline-editable table. A pair of radio
buttons switches between the two.

Rather than morphing one QTreeWidget's column count between the two layouts (the earlier
design: visibly janky, since the header dragged columns around live on every switch), the
dialog builds **two** QTreeWidgets up front, ``self._compact_table`` and
``self._detailed_table``, each with its own permanent headers/widths/hidden-flags, stacked
in a ``QStackedWidget``. Both tables are always fully populated, so a mode switch is a pure
widget flip (``self._stack.setCurrentWidget(...)``) with no repopulate and no per-mode
width bookkeeping to leak across a column-count change. The mode is a run-time choice only,
never persisted, so the dialog always reopens in Compact.

The dialog is a thin view over session.ReviewSession: every edit and swap goes through the
same edit_fields/swap_characters ops the detail panel uses, so undo/redo and the
manual-edit carry-across-restructure machinery keep working unchanged: this module adds
no editing logic of its own beyond translating table events into session calls. It mutates
the session it's given in place; the caller (ReviewScreen._on_review_and_export) resyncs its
own tree and undo buttons after the dialog closes.
"""

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QBrush, QKeySequence, QPalette, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vodparse.gui.review.model import (
    SETTING_FINAL_REVIEW_SHOW_ROUND,
    SETTING_FINAL_REVIEW_SHOW_SPONSORS,
    ReviewClip,
    nth_or_blank,
)
from vodparse.gui.review.session import ReviewSession, Selection
from vodparse.titles import player_composite

# Column → clip-field mapping (see the plan's table): P1/P2 sponsor, tag, and character,
# then the shared Round column. best_of has no column: edit_fields is always called with
# the clip's own best_of unchanged, so a cell commit here never touches it.
_HEADERS = [
    "P1 Sponsor", "P1 Tag", "P1 Character", "P2 Sponsor", "P2 Tag", "P2 Character", "Round",
]
(_COL_P1_SPONSOR, _COL_P1_TAG, _COL_P1_CHAR,
 _COL_P2_SPONSOR, _COL_P2_TAG, _COL_P2_CHAR, _COL_ROUND) = range(len(_HEADERS))

# Compact mode's columns: one title-style composite per player ("SPONSOR | Tag (Char)",
# see titles.player_composite) side by side so the two sides line up vertically for
# scanning, and the Round column (reusing the same "Show round" checkbox/machinery as
# detailed mode's Round column).
_COMPACT_HEADERS = ["Player 1", "Player 2", "Round"]
_COL_COMPACT_P1, _COL_COMPACT_P2, _COL_COMPACT_ROUND = range(len(_COMPACT_HEADERS))

# Player 1's cells and header label are right-justified (Player 2 stays left-justified),
# so the two composites meet at the column seam like a scoreboard and the characters —
# the ends nearest the seam — line up vertically.
_P1_ALIGNMENT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

# Display-mode identifiers, used both as the QButtonGroup keys and as self._mode's value.
_MODE_COMPACT = "compact"
_MODE_DETAILED = "detailed"

# Opening size: wide enough for seven columns of tag-length text without eliding, tall
# enough to show a useful number of sets at once; the dialog stays resizable.
_DIALOG_WIDTH = 900
_DIALOG_HEIGHT = 520


class FinalReviewDialog(QDialog):
    """Every set at a glance, laid out as a flat table (the house ``QTreeWidget`` pattern,
    see ``sponsor_split_dialog.SponsorSplitOptionsDialog`` for the same structure), with a
    Compact/Detailed radio switch over how each row is displayed.

    **Compact** (the mode the dialog always opens in, see ``self._mode``) shows
    "Player 1", "Player 2", and "Round" columns, one glanceable ``S | Tag (Char)`` composite
    per player via ``vodparse.titles.player_composite``: the same building block the
    exported clip title uses — with the two sides in adjacent columns so a character on the
    wrong side jumps out when scanning down either column. Compact rows are
    read-only: no cell opens an editor, but rows stay selectable so Swap/Undo/Redo/Export all
    keep working. **Detailed** is the original seven-column table with every cell directly
    editable; committing a changed cell calls ``session.edit_fields`` exactly as the detail
    panel's fields do, so it's one undoable step per commit. A same-value commit is a no-op
    and pushes nothing (see ``_on_cell_edited``): editing a cell without changing its text
    shouldn't clutter Undo. The mode is a run-time UI choice only, never persisted to
    ``settings``, so the dialog is always Compact on the next open.

    Compact and Detailed are two separate ``QTreeWidget``s (``self._compact_table``,
    ``self._detailed_table``), stacked in ``self._stack``, both kept fully populated at all
    times. A mode switch is nothing more than ``self._stack.setCurrentWidget(...)`` plus
    carrying the current selection across: there is no shared header, so each table's
    column widths and hidden flags are entirely its own and can never leak into the other's.
    An edit committed in Detailed refreshes the corresponding row in Compact immediately
    (see ``_on_cell_edited``), and vice versa a swap refreshes both, so neither table can go
    stale relative to the session.

    "Swap characters" swaps the two players' characters for every selected row (see
    ``session.swap_characters``: it deliberately leaves sponsors/tags in place, since
    those follow the stream overlay, which is exactly what's backwards when a swap is
    needed). Undo/Redo buttons plus dialog-local ``QShortcut``s mirror the Review screen's
    Edit-menu actions (screen.py:346-349): a modal dialog blocks those menu-bar QActions,
    so the same physical keys are re-declared here.

    In detailed mode, the sponsor and Round columns start hidden: tags and characters are
    what a final pass eyeballs, and seven columns of text crowd the table, with "Show
    sponsors" / "Show round" checkboxes to bring them back; in compact mode the same
    checkboxes fold the sponsor into the matchup string and add/remove the Round column.
    Each toggle persists to ``settings`` immediately (the common way out is Close, which is
    ``reject``: an on-accept save would lose the choice), so the table reopens the way the
    user left it (mode aside, which never persists).

    Close (``reject``) leaves the session as edited; Export (``accept`` with
    ``wants_export`` set) tells the caller to run the normal export path once the dialog
    closes. Either way the caller resyncs its own tree/undo buttons afterward; this dialog
    never touches them.
    """

    def __init__(self, session: ReviewSession, settings: QSettings,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review and export")
        self.resize(_DIALOG_WIDTH, _DIALOG_HEIGHT)
        self._session = session
        self._settings = settings
        self._wants_export = False
        # Guards _on_cell_edited (and the character-cell refill after a swap, and the
        # compact-table refill after a detailed commit) against itemChanged firing while a
        # programmatic write is in progress: the same guard idiom as ReviewScreen._loading.
        self._loading = False
        # Always Compact on open: this is a run-time display choice, not a persisted one.
        self._mode = _MODE_COMPACT

        self._compact_table = self._build_table(_COMPACT_HEADERS)
        self._detailed_table = self._build_table(_HEADERS)
        # The two player columns each stretch to half the table (last-section stretch off,
        # so Round — when shown — keeps its own hand-sizable width instead). With P1
        # right-justified per _P1_ALIGNMENT and P2 left-justified, all the text gravitates
        # to the shared column edge, which the equal stretch pins to the center of the
        # dialog — rather than the whole slate huddling in the top-left.
        compact_header = self._compact_table.header()
        assert compact_header is not None
        compact_header.setStretchLastSection(False)
        compact_header.setSectionResizeMode(_COL_COMPACT_P1, QHeaderView.ResizeMode.Stretch)
        compact_header.setSectionResizeMode(_COL_COMPACT_P2, QHeaderView.ResizeMode.Stretch)
        compact_header_item = self._compact_table.headerItem()
        assert compact_header_item is not None
        compact_header_item.setTextAlignment(_COL_COMPACT_P1, _P1_ALIGNMENT)
        # Only the detailed table's cells are ever editable, so only it needs to react to a
        # committed edit; both tables report selection changes so Swap/Undo/Redo stay
        # enabled correctly regardless of which one is on screen.
        self._detailed_table.itemChanged.connect(self._on_cell_edited)
        self._compact_table.itemSelectionChanged.connect(self._refresh_actions)
        self._detailed_table.itemSelectionChanged.connect(self._refresh_actions)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._compact_table)
        self._stack.addWidget(self._detailed_table)
        self._stack.setCurrentWidget(self._compact_table)

        # Compact/Detailed radios (house dict-of-radios + QButtonGroup pattern, see
        # dialogs.py:228-231). Only the Detailed radio's toggled(bool) is connected: with
        # exactly two mutually exclusive radios, that one bool unambiguously determines the
        # new mode, both directions of the switch fire it, and it avoids depending on
        # QButtonGroup's signal-ordering between the two buttons.
        self._mode_radios = {
            _MODE_COMPACT: QRadioButton("Compact"),
            _MODE_DETAILED: QRadioButton("Detailed"),
        }
        self._mode_radios[_MODE_COMPACT].setChecked(True)
        self._mode_group = QButtonGroup(self)
        for radio in self._mode_radios.values():
            self._mode_group.addButton(radio)
        self._mode_radios[_MODE_DETAILED].toggled.connect(self._on_mode_toggled)

        # Column toggles, seeded from the persisted preference (hidden by default). Their
        # initial effect is applied once below, directly on both tables, there's no
        # per-mode header morph left to route through, so a plain setColumnHidden call on
        # each table is all that's needed.
        self._show_sponsors = QCheckBox("Show sponsors")
        self._show_sponsors.setChecked(
            settings.value(SETTING_FINAL_REVIEW_SHOW_SPONSORS, False, type=bool))
        self._show_sponsors.toggled.connect(self._on_show_sponsors_toggled)
        self._show_round = QCheckBox("Show round")
        self._show_round.setChecked(
            settings.value(SETTING_FINAL_REVIEW_SHOW_ROUND, False, type=bool))
        self._show_round.toggled.connect(self._on_show_round_toggled)

        show_sponsors = self._show_sponsors.isChecked()
        show_round = self._show_round.isChecked()
        self._compact_table.setColumnHidden(_COL_COMPACT_ROUND, not show_round)
        self._detailed_table.setColumnHidden(_COL_P1_SPONSOR, not show_sponsors)
        self._detailed_table.setColumnHidden(_COL_P2_SPONSOR, not show_sponsors)
        self._detailed_table.setColumnHidden(_COL_ROUND, not show_round)

        toggles_row = QHBoxLayout()
        toggles_row.addWidget(self._mode_radios[_MODE_COMPACT])
        toggles_row.addWidget(self._mode_radios[_MODE_DETAILED])
        toggles_row.addSpacing(16)
        toggles_row.addWidget(self._show_sponsors)
        toggles_row.addWidget(self._show_round)
        toggles_row.addStretch(1)

        self._swap_btn = QPushButton("Swap characters")
        self._swap_btn.clicked.connect(self._on_swap)
        self._undo_btn = QPushButton("Undo")
        self._undo_btn.clicked.connect(self._on_undo)
        self._redo_btn = QPushButton("Redo")
        self._redo_btn.clicked.connect(self._on_redo)

        # A modal dialog blocks the MainWindow's menu-bar Undo/Redo QActions (they hang off
        # the Review menus, hidden off this dialog), so Undo/Redo need their own
        # dialog-local shortcuts here: the platform's standard key, same as the menu
        # actions use (QKeySequence.StandardKey.Undo/Redo, not a literal "Ctrl+Z").
        undo_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Undo), self)
        undo_shortcut.activated.connect(self._on_undo)
        redo_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Redo), self)
        redo_shortcut.activated.connect(self._on_redo)

        edit_row = QHBoxLayout()
        edit_row.addWidget(self._swap_btn)
        edit_row.addWidget(self._undo_btn)
        edit_row.addWidget(self._redo_btn)
        edit_row.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        export_btn = buttons.addButton("Export", QDialogButtonBox.ButtonRole.AcceptRole)
        assert export_btn is not None
        export_btn.clicked.connect(self._on_export_clicked)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(toggles_row)
        layout.addWidget(self._stack, 1)   # the table soaks up any extra height
        layout.addLayout(edit_row)
        layout.addWidget(buttons)

        self._populate()
        # One-time auto-fit on first showing: each table keeps whatever widths the user
        # drags afterward, permanently, since nothing ever rebuilds its header again.
        for table in (self._compact_table, self._detailed_table):
            for col in range(table.columnCount()):
                table.resizeColumnToContents(col)
        self._refresh_actions()

    @staticmethod
    def _build_table(headers: list[str]) -> QTreeWidget:
        """Construct one of the dialog's two permanent tables: flat rows, multi-row
        selection, and a stretched last column. Headers are set once at construction and
        never change again: each table's column count, widths, and hidden flags are its
        own for the table's entire lifetime."""
        table = QTreeWidget()
        table.setHeaderLabels(headers)
        table.setRootIsDecorated(False)   # flat rows: no expander gutter
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # Zebra striping, from the palette's AlternateBase: the dark-Fusion palette
        # (common.py) and macOS's native palette both define it, so this tracks the theme.
        table.setAlternatingRowColors(True)
        header = table.header()
        assert header is not None
        header.setStretchLastSection(True)
        return table

    def _active_table(self) -> QTreeWidget:
        """The table for the current display mode: the one on top of ``self._stack``."""
        return self._detailed_table if self._mode == _MODE_DETAILED else self._compact_table

    @property
    def wants_export(self) -> bool:
        """True once the Export button has been clicked; the caller (screen.py) reads this
        after ``exec()`` returns to decide whether to run the normal export path."""
        return self._wants_export

    def _on_export_clicked(self) -> None:
        self._wants_export = True
        self.accept()

    # ── Display mode ─────────────────────────────────────────────────────────────

    def _on_mode_toggled(self, detailed_checked: bool) -> None:
        """Switch the visible table between Compact and Detailed. Connected only to the
        Detailed radio's ``toggled`` signal (see the radio setup in ``__init__`` for why one
        bool is enough); an early return on a same-mode call guards against that signal
        firing for reasons other than an actual mode change.

        Both tables are always fully populated, so switching is nothing more than carrying
        the outgoing table's selection onto the incoming one and flipping the stack, no
        repopulate, no column-width bookkeeping."""
        mode = _MODE_DETAILED if detailed_checked else _MODE_COMPACT
        if mode == self._mode:
            return
        outgoing = self._active_table()
        selected_rows = {self._row_index(item) for item in outgoing.selectedItems()}
        current_row = self._current_row()
        self._mode = mode
        incoming = self._active_table()
        # The incoming table still holds whatever selection it had when it was last the
        # active one; without clearing it first the carried-over selection would UNION with
        # that stale one, and Swap acts on every selected row, so a stale extra row would
        # get swapped too.
        incoming.clearSelection()
        self._restore_selection(incoming, selected_rows, current_row)
        self._stack.setCurrentWidget(incoming)
        self._refresh_actions()

    # ── Column toggles ───────────────────────────────────────────────────────────

    def _on_show_sponsors_toggled(self, show: bool) -> None:
        """Persist the choice, then apply it to both tables: detailed hides/shows its two
        dedicated sponsor columns (the hidden cells keep their text, so
        _on_cell_edited's whole-row gather works unchanged either way); compact has no
        sponsor column of its own: the sponsor prefix lives inside the matchup string, so
        every compact row is rebuilt instead. The rebuild runs under ``_loading`` even
        though only the detailed table's ``itemChanged`` is connected, since it's cheap
        insurance against future wiring changes."""
        self._settings.setValue(SETTING_FINAL_REVIEW_SHOW_SPONSORS, show)
        self._loading = True
        try:
            self._detailed_table.setColumnHidden(_COL_P1_SPONSOR, not show)
            self._detailed_table.setColumnHidden(_COL_P2_SPONSOR, not show)
            for row, clip in enumerate(self._session.clips):
                item = self._compact_table.topLevelItem(row)
                assert item is not None
                self._fill_compact_row(item, clip)
        finally:
            self._loading = False

    def _on_show_round_toggled(self, show: bool) -> None:
        """Persist the choice, then show/hide whichever column means Round on each table."""
        self._settings.setValue(SETTING_FINAL_REVIEW_SHOW_ROUND, show)
        self._compact_table.setColumnHidden(_COL_COMPACT_ROUND, not show)
        self._detailed_table.setColumnHidden(_COL_ROUND, not show)

    # ── Table population ─────────────────────────────────────────────────────────

    def _populate(self) -> None:
        """(Re)fill both tables, one row per ``session.clips`` entry — excluded sets
        included, greyed + struck out (see ``_style_row``), so every table row keeps the
        same index as its clip. Called on open and
        again after Undo/Redo (which can rebuild the whole clip list). Selection and the
        current row are captured from the active table before the rebuild and restored onto
        both tables afterward (see ``_restore_selection``), so this never drops what was
        selected in either view."""
        active = self._active_table()
        selected_rows = {self._row_index(item) for item in active.selectedItems()}
        current_row = self._current_row()
        self._loading = True
        try:
            self._compact_table.clear()
            self._detailed_table.clear()
            for clip in self._session.clips:
                detailed_item = QTreeWidgetItem(self._detailed_table)
                detailed_item.setFlags(detailed_item.flags() | Qt.ItemFlag.ItemIsEditable)
                self._fill_detailed_row(detailed_item, clip)
                self._style_row(detailed_item, clip)

                # Compact rows are read-only by omission: ItemIsEditable is simply never
                # added to the default flags, no bit-masking needed.
                compact_item = QTreeWidgetItem(self._compact_table)
                self._fill_compact_row(compact_item, clip)
                self._style_row(compact_item, clip)
        finally:
            self._loading = False
        self._restore_selection(self._compact_table, selected_rows, current_row)
        self._restore_selection(self._detailed_table, selected_rows, current_row)

    @staticmethod
    def _restore_selection(table: QTreeWidget, selected_rows: set[int],
                            current_row: int | None) -> None:
        """Reselect the rows and re-focus the current row that survived a ``_populate()``
        rebuild (or a mode switch) on `table`, clamped to its row count: a mode switch keeps
        the same clips so nothing is lost, but Undo/Redo can shrink or grow the clip list out
        from under a stale row index."""
        row_count = table.topLevelItemCount()
        for row in selected_rows:
            if row < row_count:
                item = table.topLevelItem(row)
                assert item is not None
                item.setSelected(True)
        if current_row is not None and current_row < row_count:
            item = table.topLevelItem(current_row)
            assert item is not None
            table.setCurrentItem(item)

    @staticmethod
    def _fill_detailed_row(item: QTreeWidgetItem, clip: ReviewClip) -> None:
        """Write one clip's editable fields into `item`'s columns, per the column mapping."""
        item.setText(_COL_P1_SPONSOR, nth_or_blank(clip["tournament_sponsors"], 0))
        item.setText(_COL_P1_TAG, nth_or_blank(clip["grouping_tags"], 0))
        item.setText(_COL_P1_CHAR, nth_or_blank(clip["characters"], 0))
        item.setText(_COL_P2_SPONSOR, nth_or_blank(clip["tournament_sponsors"], 1))
        item.setText(_COL_P2_TAG, nth_or_blank(clip["grouping_tags"], 1))
        item.setText(_COL_P2_CHAR, nth_or_blank(clip["characters"], 1))
        item.setText(_COL_ROUND, clip["round"])

    def _fill_compact_row(self, item: QTreeWidgetItem, clip: ReviewClip) -> None:
        """Write one clip's per-player composite strings and round into `item`'s compact
        columns. Reads "Show sponsors" directly (unlike ``_fill_detailed_row``, this isn't
        static) since the sponsor prefix is folded into each composite string rather than
        living in a dedicated column."""
        show_sponsors = self._show_sponsors.isChecked()
        s1 = nth_or_blank(clip["tournament_sponsors"], 0) if show_sponsors else ""
        s2 = nth_or_blank(clip["tournament_sponsors"], 1) if show_sponsors else ""
        item.setTextAlignment(_COL_COMPACT_P1, _P1_ALIGNMENT)
        item.setText(_COL_COMPACT_P1, player_composite(
            nth_or_blank(clip["grouping_tags"], 0), nth_or_blank(clip["characters"], 0), s1))
        item.setText(_COL_COMPACT_P2, player_composite(
            nth_or_blank(clip["grouping_tags"], 1), nth_or_blank(clip["characters"], 1), s2))
        item.setText(_COL_COMPACT_ROUND, clip["round"])

    def _style_row(self, item: QTreeWidgetItem, clip: ReviewClip) -> None:
        """Grey + strike out every column of `item` when `clip` is excluded from export,
        mirroring the Review tree's styling (see ReviewScreen._style_clip_row). Excluded
        sets stay listed — the whole dialog assumes row index == clip index (see
        ``_row_index``) — but the styling shows at a glance which rows won't be cut.
        Called right after a row is filled, so ``item.columnCount()`` is final; the flag
        can only change under an open dialog via Undo/Redo, which reruns ``_populate``
        (and thus this) for every row."""
        excluded = clip.get("exclude_from_export", False)
        dim = self.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
        for col in range(item.columnCount()):
            font = item.font(col)
            font.setStrikeOut(excluded)
            item.setFont(col, font)
            item.setForeground(col, QBrush(dim) if excluded else QBrush())

    @staticmethod
    def _row_index(item: QTreeWidgetItem) -> int:
        """The clip index a top-level `item` corresponds to (rows are flat and never
        reordered relative to ``session.clips``, so this is just its sibling position).
        Resolves against whichever table `item` actually belongs to, via
        ``QTreeWidgetItem.treeWidget()``, so callers never need to track which of the two
        tables an item came from."""
        tree = item.treeWidget()
        assert tree is not None
        root = tree.invisibleRootItem()
        assert root is not None
        return root.indexOfChild(item)

    # ── Inline edit ──────────────────────────────────────────────────────────────

    def _on_cell_edited(self, item: QTreeWidgetItem, _column: int) -> None:
        """Commit every column of the edited row into the session as one field edit, then
        refresh the compact table's corresponding row so it never goes stale relative to a
        detailed-mode edit, even without a mode switch in between. `burst_key=None` makes
        each commit its own undo entry: correct for a discrete cell edit, unlike the detail
        panel's per-keystroke bursts. edit_fields itself skips a commit whose normalized values
        are unchanged (returning None, e.g. clicking into a cell and pressing Enter without
        typing), so a no-op edit doesn't push a spurious undo entry or need a refresh here.

        Only the detailed table's ``itemChanged`` is connected (see ``__init__``), so `item`
        is always one of its rows and the ``_COL_*`` indices below always apply: the
        ``self._loading`` guard exists purely to ignore this table's own programmatic writes
        (``_populate``, undo/redo, a swap's character-cell rewrite)."""
        if self._loading:
            return
        idx = self._row_index(item)
        clip = self._session.clips[idx]

        p1_sponsor, p2_sponsor = item.text(_COL_P1_SPONSOR), item.text(_COL_P2_SPONSOR)
        p1_tag, p2_tag = item.text(_COL_P1_TAG), item.text(_COL_P2_TAG)
        p1_char, p2_char = item.text(_COL_P1_CHAR), item.text(_COL_P2_CHAR)
        round_text = item.text(_COL_ROUND)

        result = self._session.edit_fields(
            idx, tags=(p1_tag, p2_tag), sponsors=(p1_sponsor, p2_sponsor),
            characters=(p1_char, p2_char), round_text=round_text, best_of=clip["best_of"],
            burst_key=None, selection=Selection(idx, None))
        if result is None:   # a no-op commit: edit_fields pushed nothing, nothing to refresh
            return

        self._loading = True
        try:
            compact_item = self._compact_table.topLevelItem(idx)
            assert compact_item is not None
            self._fill_compact_row(compact_item, self._session.clips[idx])
        finally:
            self._loading = False
        self._refresh_actions()

    # ── Swap ─────────────────────────────────────────────────────────────────────

    def _on_swap(self) -> None:
        """Swap P1/P2 characters for every row selected in the active table as a SINGLE
        undoable step (see ``session.swap_characters_bulk``, one button press swaps every
        selected row, so one Undo reverses the whole press), then refill both tables' rows:
        detailed only needs its two character cells rewritten (sponsor/tag/round cells are
        untouched: the swap deliberately leaves those fields alone); compact's character info
        lives inside the composed matchup string, so the whole row is rebuilt via
        ``_fill_compact_row``."""
        indices = [self._row_index(item) for item in self._active_table().selectedItems()]
        if not indices:
            return
        self._session.swap_characters_bulk(indices, Selection(indices[0], None))
        self._loading = True
        try:
            for idx in indices:
                clip = self._session.clips[idx]
                detailed_item = self._detailed_table.topLevelItem(idx)
                assert detailed_item is not None
                detailed_item.setText(_COL_P1_CHAR, nth_or_blank(clip["characters"], 0))
                detailed_item.setText(_COL_P2_CHAR, nth_or_blank(clip["characters"], 1))

                compact_item = self._compact_table.topLevelItem(idx)
                assert compact_item is not None
                self._fill_compact_row(compact_item, clip)
        finally:
            self._loading = False
        self._refresh_actions()

    # ── Undo / redo ──────────────────────────────────────────────────────────────

    def _current_row(self) -> int | None:
        """The clip index of the active table's current row, or None with nothing current
        (an empty table, or before any row has been focused)."""
        item = self._active_table().currentItem()
        return None if item is None else self._row_index(item)

    def _on_undo(self) -> None:
        result = self._session.undo(Selection(self._current_row(), None))
        if result is None:
            return
        self._populate()   # undo can rebuild the whole clip list
        self._refresh_actions()

    def _on_redo(self) -> None:
        result = self._session.redo(Selection(self._current_row(), None))
        if result is None:
            return
        self._populate()
        self._refresh_actions()

    def _refresh_actions(self) -> None:
        """Enable/disable Undo/Redo from the session's stacks, and Swap from whether any
        row is currently selected in the active table."""
        self._undo_btn.setEnabled(self._session.can_undo())
        self._redo_btn.setEnabled(self._session.can_redo())
        self._swap_btn.setEnabled(bool(self._active_table().selectedItems()))
