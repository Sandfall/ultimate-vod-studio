"""Shared base for the Review screen's two "options + live preview" batch-op dialogs
(Tools ▸ "Fix capitalization…" and Tools ▸ "Split sponsor from tag…").

Both dialogs share one interaction: op-specific controls on top, and below them a two-column
preview table listing exactly what the sweep would change. Preview rows can be selected and
toggled out of the run: "Exclude from fix"/"…split" greys the row (strikethrough,
disabled-text colour), drops it from the header count and the Apply gate, and a second click on
the same selection restores it. Rows can also be hand-edited in place (on the columns the
subclass opts in via ``editable_columns``): the typed texts become a per-row *override* the
caller applies instead of the computed result, shown italic in the table. That entire
excludable/editable-preview half lives here; each subclass supplies only its own controls, how
it reads them back (``values`` / ``accept``), and how it turns the clips into preview rows
(``_preview_rows``).

The dialog is a pure options popup: Apply is ``accept`` (the subclass persists its control
values there), Cancel is ``reject``; the caller reads ``values()``, ``excluded_tags()`` and
``overrides()`` after an ``Accepted`` exec and runs the sweep itself. This module never imports
the sweep logic.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import NamedTuple

from PyQt6.QtCore import QModelIndex, Qt
from PyQt6.QtGui import QBrush, QPalette
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vodparse.gui.review.model import ReviewClip

# Opening size: tall enough for the controls plus a usefully deep preview pane; the dialog
# stays resizable for VODs with many distinct preview rows.
_DIALOG_WIDTH = 480
_DIALOG_HEIGHT = 420

# Per-item data roles on the preview rows: the pre-sweep tag text is the exclusion/override key
# the session matches on, the slot count feeds the header total, the sweep-computed (col0, col1)
# pair backs override bookkeeping (no-op detection, blank-edit revert), and the pre-sweep
# (col0, col1) pair lets _recount_header drop a row hand-edited back to its original state. All
# live off UserRole so they don't collide with the display text.
_TAG_BEFORE_ROLE = Qt.ItemDataRole.UserRole
_SLOTS_ROLE = Qt.ItemDataRole.UserRole + 1
_COMPUTED_ROLE = Qt.ItemDataRole.UserRole + 2
_ORIGINAL_ROLE = Qt.ItemDataRole.UserRole + 3


class _NoEditDelegate(QStyledItemDelegate):
    """Blocks in-place editing on the column it's installed on by never creating an editor.
    ItemIsEditable is a per-item flag, so a read-only *column* on an otherwise editable row
    needs this standard Qt idiom instead."""

    def createEditor(self, parent: QWidget | None, option: QStyleOptionViewItem,
                     index: QModelIndex) -> QWidget | None:
        return None


class PreviewRow(NamedTuple):
    """One row of the live preview as the base table renders it: ``col0``/``col1`` are the two
    column texts, ``before`` is the pre-sweep tag text that keys exclusion (and the session's
    ``excluded_tags`` skip), ``slots`` is how many tag slots across the VOD this row stands
    for (its contribution to the header total), and ``original`` is the ``(col0, col1)`` pair the
    slot showed *before* the sweep, so a row hand-edited back to that pair can be recognised as a
    no-op and dropped from the count. Each subclass maps its own op preview row (CapPreviewRow /
    SponsorSplitPreviewRow) onto this shape in ``_preview_rows``."""
    before: str
    col0: str
    col1: str
    slots: int
    original: tuple[str, str]


class ExcludablePreviewDialog(QDialog):
    """Base for an options dialog with a live, per-row excludable and editable preview table.

    Subclasses build their own controls (``_add_controls``), read them back
    (``values`` / ``accept``), and turn the clips into ``PreviewRow``s (``_preview_rows``);
    everything about the preview table, the exclude toggle, in-place cell editing, the header
    count, the Apply gate, ``excluded_tags`` and ``overrides`` lives here. ``self._clips``,
    ``self._excluded`` and ``self._overrides`` are owned here too.

    Not an ABC: QDialog's Qt metaclass rules out mixing in ``ABCMeta``, so the two subclass
    hooks raise ``NotImplementedError`` rather than using ``@abstractmethod``.
    """

    def __init__(self, clips: list[ReviewClip], parent: QWidget | None, *,
                 window_title: str, header_labels: list[str], empty_preview_text: str,
                 exclude_noun: str, editable_columns: tuple[int, ...] = ()) -> None:
        super().__init__(parent)
        self.setWindowTitle(window_title)
        self.resize(_DIALOG_WIDTH, _DIALOG_HEIGHT)
        self._clips = clips
        # Shown by _recount_header when nothing matches; the "N tag(s) will change" count form
        # is shared, only the empty-state wording differs per op.
        self._empty_preview_text = empty_preview_text
        # The op noun for the exclude button's "Exclude from <noun>" / "Include in <noun>" label.
        self._exclude_noun = exclude_noun
        # Which preview columns the user may edit in place; empty = a read-only table.
        self._editable_columns = editable_columns
        # Pre-sweep tag texts the user has toggled out of this run; per-open only, never
        # persisted. Entries that match no current preview row sit dormant (other settings
        # may bring the row back), so this is never pruned against the live table.
        self._excluded: set[str] = set()
        # Per-row hand edits: pre-sweep tag text -> the (col0, col1) texts the user typed over
        # the computed result. Same contract as _excluded: per-open only, never persisted, and
        # a key matching no current row sits dormant until a rebuild brings the row back.
        self._overrides: dict[str, tuple[str, str]] = {}
        # Gates _on_item_edited off while the dialog writes items itself; every such writer runs
        # inside the _suppress_item_signals() guard, which explains the mechanism in full.
        self._updating = False

        # Live preview: a two-column table so the pair lines up for at-a-glance scanning
        # (QTreeWidget is the codebase's multi-column widget: see the Review tree in
        # screen.py; nothing here nests). Rows are multi-selectable so the user can grey out
        # a batch of them with the button below.
        self._preview_header = QLabel()
        self._preview = QTreeWidget()
        self._preview.setHeaderLabels(header_labels)
        self._preview.setRootIsDecorated(False)   # flat rows: no expander gutter
        self._preview.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        header = self._preview.header()
        assert header is not None
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self._preview.itemChanged.connect(self._on_item_edited)
        if self._editable_columns:
            # ItemIsEditable is per-item, so the read-only columns each get a no-editor
            # delegate (parented to the dialog to keep it alive for the table's lifetime).
            no_edit = _NoEditDelegate(self)
            for column in (0, 1):
                if column not in self._editable_columns:
                    self._preview.setItemDelegateForColumn(column, no_edit)

        # Toggles the selected rows in/out of the sweep; disabled until something is selected,
        # and its label flips to "Include in <noun>" once the whole selection is already
        # excluded (see _on_selection_changed).
        self._exclude_btn = QPushButton(f"Exclude from {exclude_noun}")
        self._exclude_btn.setEnabled(False)
        self._exclude_btn.clicked.connect(self._toggle_selected_exclusion)
        self._preview.itemSelectionChanged.connect(self._on_selection_changed)
        exclude_row = QHBoxLayout()
        exclude_row.addWidget(self._exclude_btn)
        exclude_row.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        apply_btn = buttons.addButton("Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        assert apply_btn is not None
        self._apply_btn: QPushButton = apply_btn
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        self._add_controls(layout)           # subclass: build its controls, wire them, place them
        layout.addSpacing(8)
        layout.addWidget(self._preview_header)
        layout.addWidget(self._preview, 1)   # the preview soaks up any extra height
        layout.addLayout(exclude_row)
        layout.addWidget(buttons)

        self._refresh_preview()

    @contextmanager
    def _suppress_item_signals(self) -> Iterator[None]:
        """Gate _on_item_edited off for the duration of the block so it ignores the dialog's own
        item writes. QTreeWidget.itemChanged fires on every programmatic write
        (setText/setFont/setForeground/setData/construction), not just user edits, so every such
        writer runs inside this guard. It saves and restores the flag rather than setting then
        clearing it, because the writers nest: styling runs inside the guarded rebuild. This is
        deliberately narrower than QWidget.blockSignals, which would also silence the tree's other
        signals; only _on_item_edited must be gated."""
        previous = self._updating
        self._updating = True
        try:
            yield
        finally:
            self._updating = previous

    def _style_item(self, item: QTreeWidgetItem, excluded: bool) -> None:
        """Grey out (or restore) a preview row across both columns. Excluded rows get the
        disabled-palette text colour and a strikethrough font; included rows are reset to the
        view's defaults so the styling is fully reversible on re-include. Only strikeout and
        foreground are touched, so the override italic (see _mark_override) composes freely."""
        with self._suppress_item_signals():
            excluded_color = self.palette().color(
                QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
            for column in (0, 1):
                font = item.font(column)
                font.setStrikeOut(excluded)
                item.setFont(column, font)
                # An empty QBrush restores the view default; a coloured one greys the text.
                item.setForeground(column, QBrush(excluded_color) if excluded else QBrush())

    def _mark_override(self, item: QTreeWidgetItem) -> None:
        """Italicize each cell whose text differs from the sweep-computed value, so hand-edited
        results stand out; a cell back at its computed text loses the marker. Only the italic
        bit is touched, composing with _style_item's strikeout/foreground."""
        with self._suppress_item_signals():
            computed: tuple[str, str] = item.data(0, _COMPUTED_ROLE)
            for column in (0, 1):
                font = item.font(column)
                font.setItalic(item.text(column) != computed[column])
                item.setFont(column, font)

    def _set_text_guarded(self, item: QTreeWidgetItem, column: int, text: str) -> None:
        """Write a cell's text without _on_item_edited mistaking it for a user edit."""
        with self._suppress_item_signals():
            item.setText(column, text)

    def _recount_header(self) -> None:
        """Refresh the preview header and the Apply gate from the current rows, counting only
        the ones that would actually change. An empty table shows the op's empty-state wording;
        a table whose rows would all no-op shows the count form with 0 and disables Apply.

        A row is skipped (no slots counted, no Apply) when it is excluded, or when its current
        displayed pair equals its pre-sweep original: a hand-edit back to the original state
        produces nothing, so it mustn't inflate the count or arm an Apply the sweep would no-op."""
        root = self._preview.invisibleRootItem()
        assert root is not None
        if root.childCount() == 0:
            self._preview_header.setText(self._empty_preview_text)
            self._apply_btn.setEnabled(False)
            return
        total = 0
        included_any = False
        for index in range(root.childCount()):
            item = root.child(index)
            assert item is not None
            if item.data(0, _TAG_BEFORE_ROLE) in self._excluded:
                continue
            original: tuple[str, str] = item.data(0, _ORIGINAL_ROLE)
            if (item.text(0), item.text(1)) == original:
                continue   # hand-edited back to the pre-sweep state: this row won't change
            total += int(item.data(0, _SLOTS_ROLE))
            included_any = True
        self._preview_header.setText(f"{total} tag(s) will change:")
        self._apply_btn.setEnabled(included_any)

    def _on_selection_changed(self) -> None:
        """Enable the exclude button only when rows are selected, and flip its label to
        "Include in <noun>" once every selected row is already excluded (so the same click
        restores them)."""
        selected = self._preview.selectedItems()
        self._exclude_btn.setEnabled(bool(selected))
        all_excluded = bool(selected) and all(
            item.data(0, _TAG_BEFORE_ROLE) in self._excluded for item in selected)
        verb = "Include in" if all_excluded else "Exclude from"
        self._exclude_btn.setText(f"{verb} {self._exclude_noun}")

    def _toggle_selected_exclusion(self) -> None:
        """Exclude every selected row from the sweep, or, if they're all already excluded,
        re-include them all. Restyles and recounts in place (the table is not rebuilt) so the
        selection survives the toggle.

        The restyling follows the *before-keys* the selection carries rather than the selected
        items, because one before-key can back several rows at once (sponsor rows that dedupe
        apart on their differing pre-existing sponsors still share one tag-text key) while
        exclusion is keyed on that text alone. Styling only the selection would leave an
        unselected sibling showing the opposite state to the one Apply acts on."""
        selected = self._preview.selectedItems()
        if not selected:
            return
        toggled_before: set[str] = {item.data(0, _TAG_BEFORE_ROLE) for item in selected}
        all_excluded = toggled_before.issubset(self._excluded)
        if all_excluded:
            self._excluded.difference_update(toggled_before)
        else:
            self._excluded.update(toggled_before)
        root = self._preview.invisibleRootItem()
        assert root is not None
        for index in range(root.childCount()):
            item = root.child(index)
            assert item is not None
            if item.data(0, _TAG_BEFORE_ROLE) in toggled_before:
                self._style_item(item, excluded=not all_excluded)
        self._recount_header()
        self._on_selection_changed()   # relabel the button for the new exclusion state

    def _on_item_edited(self, item: QTreeWidgetItem, column: int) -> None:
        """React to a user's in-place cell edit (itemChanged with the guard down): record the
        row's texts as its override, or drop the override when the edit lands back on the
        computed values. An edit on a read-only column (reachable programmatically even though
        the delegate blocks the UI) reverts, as does one that blanks the tag cell (column 1 in
        both dialogs): the sweep never empties a tag and an edit can't either. Editing an
        excluded row re-includes it (an edit is a stronger statement than "skip"). The result
        is propagated to every row sharing this row's before-key, since sponsor rows that
        dedupe apart on differing pre-existing sponsors still share one override entry."""
        if self._updating:
            return
        before: str = item.data(0, _TAG_BEFORE_ROLE)
        computed: tuple[str, str] = item.data(0, _COMPUTED_ROLE)
        prior = self._overrides.get(before, computed)   # the cell texts before this edit
        if column not in self._editable_columns:
            self._set_text_guarded(item, column, prior[column])
            return
        text = item.text(column).strip()
        if column == 1 and not text:
            self._set_text_guarded(item, 1, prior[1])
            return
        if text != item.text(column):
            self._set_text_guarded(item, column, text)   # normalize stray whitespace
        new_pair = (item.text(0), item.text(1))
        if new_pair == computed:
            self._overrides.pop(before, None)
        else:
            self._overrides[before] = new_pair
        re_included = before in self._excluded
        self._excluded.discard(before)
        # Sync every row keyed on this before-text: each shows the shared override when one
        # exists, else its own computed pair (siblings' computed col0 can differ, so a removed
        # override must restore each row's own values, not this row's).
        override = self._overrides.get(before)
        root = self._preview.invisibleRootItem()
        assert root is not None
        for index in range(root.childCount()):
            sibling = root.child(index)
            assert sibling is not None
            if sibling.data(0, _TAG_BEFORE_ROLE) != before:
                continue
            sibling_texts = override or sibling.data(0, _COMPUTED_ROLE)
            for sync_column in (0, 1):
                if sibling.text(sync_column) != sibling_texts[sync_column]:
                    self._set_text_guarded(sibling, sync_column, sibling_texts[sync_column])
            if re_included:
                self._style_item(sibling, excluded=False)
            self._mark_override(sibling)
        self._recount_header()
        self._on_selection_changed()   # relabel the exclude button if the row was re-included

    def _refresh_preview(self) -> None:
        """Recompute the pending changes for the current controls (via the subclass's
        ``_preview_rows``) and refill the table, re-applying any live exclusions and overrides
        (matched by pre-sweep tag text) to the rebuilt rows; an overridden row displays its
        override so the table always shows exactly what Apply will do. An override the new
        computed values have caught up with is pruned (a no-op edit leaves no residue). The
        header total and Apply gate then count only the rows that would still change.

        Pruning is sibling-aware: one before-key can back several rows at once (sponsor rows that
        dedupe apart on their differing pre-existing sponsors still share the tag-text override
        key), and the override is a *single* shared entry. So an override is pruned only when it
        equals the computed pair of **every** row under that key; matching just one sibling is a
        real edit that must survive, else the shared entry would vanish while a sibling still
        displays it as an italic override the table couldn't back."""
        rows = self._preview_rows()
        # Prune only overrides the sweep has fully caught up with. Group each key's computed
        # pairs so a key with siblings keeps its override unless all of them now match it; a
        # lone-row key keeps the original no-op-prune behaviour (its single pair must match).
        computed_by_before: dict[str, list[tuple[str, str]]] = {}
        for row in rows:
            computed_by_before.setdefault(row.before, []).append((row.col0, row.col1))
        for before, computed_pairs in computed_by_before.items():
            override = self._overrides.get(before)
            if override is not None and all(pair == override for pair in computed_pairs):
                del self._overrides[before]

        with self._suppress_item_signals():
            self._preview.clear()
            for row in rows:
                computed = (row.col0, row.col1)
                override = self._overrides.get(row.before)
                item = QTreeWidgetItem(self._preview, list(override or computed))
                item.setData(0, _TAG_BEFORE_ROLE, row.before)
                item.setData(0, _SLOTS_ROLE, row.slots)
                item.setData(0, _COMPUTED_ROLE, computed)
                item.setData(0, _ORIGINAL_ROLE, row.original)
                if self._editable_columns:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                self._style_item(item, excluded=row.before in self._excluded)
                self._mark_override(item)
        self._recount_header()
        self._on_selection_changed()   # the rebuilt table starts with nothing selected

    def excluded_tags(self) -> frozenset[str]:
        """The pre-sweep tag texts the user toggled out of this run (empty if none)."""
        return frozenset(self._excluded)

    def overrides(self) -> dict[str, tuple[str, str]]:
        """The per-row hand edits from this run: pre-sweep tag text -> the (col0, col1) texts
        the user typed over the computed result (empty if none). Keys the sweep can't reach
        (a row a later control change removed) are included and sit dormant there too."""
        return dict(self._overrides)

    # ── Subclass hooks ───────────────────────────────────────────────────────────

    def _add_controls(self, layout: QVBoxLayout) -> None:
        """Build this op's controls, seed them from settings, wire each to ``_refresh_preview``,
        and add them to the top of ``layout``. Seed before connecting so seeding doesn't fire a
        premature refresh (the base calls ``_refresh_preview`` once after this returns)."""
        raise NotImplementedError

    def _preview_rows(self) -> list[PreviewRow]:
        """The distinct before/after rows the sweep would produce for the current controls,
        mapped onto ``PreviewRow``: the live preview's contents."""
        raise NotImplementedError
