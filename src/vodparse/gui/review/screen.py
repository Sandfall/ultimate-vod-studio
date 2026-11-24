"""Review screen: inspect extracted clips, correct their tags/characters/round, and trim each
clip's start/end by eye before exporting.

A clip is the unit that gets exported: in "match" mode one clip = one match; in "tournament_set"
mode one clip = one set (its member matches show as read-only children for context). Boundaries
are seeded by applying the export buffers to the detected markers, then edited freely in the
TrimPlayer: export uses the edited absolute seconds verbatim (see vodparse.cli.export_reviewed).

This module is the view + wiring: the editing model (clips, undo/redo, and every edit
operation) lives in session.ReviewSession (Qt-free). Widget events are resolved into
session calls, and each returned OpResult is applied by _apply_result (status line, tree
repopulate or in-place relabel/restyle, selection restore).
"""

import json
from pathlib import Path
from collections.abc import Callable
from typing import Literal, override

# The pure, Qt-free clip/match domain logic lives in model.py, and the review-flag
# guards (toggle keys, thresholds, row tints) in flags.py; draft (de)serialization
# and file I/O live in draft.py. The small model.py helpers are imported under their
# original private names so the call sites in this file read unchanged.
from vodparse.gui.review.flags import (
    FlagSettings,
    LONG_LEADIN_INTERVALS_DEFAULT,
    LONG_MATCH_HEADROOM_SEC_DEFAULT,
    MIN_MATCH_SEC_DEFAULT,
    RETIRED_SETTING_MAX_MATCH_SEC,
    REVIEW_FLAG_COLOR,
    REVIEW_FLAG_COLOR_DARK,
    REVIEW_LEADIN_COLOR,
    REVIEW_LEADIN_COLOR_DARK,
    SETTING_FLAG_LONG_LEADIN,
    SETTING_FLAG_LONG_MATCHES,
    SETTING_FLAG_SHORT_MATCHES,
    SETTING_FLAG_WEIRD_TIMER,
    SETTING_LONG_LEADIN_INTERVALS,
    SETTING_LONG_MATCH_HEADROOM_SEC,
    SETTING_MIN_MATCH_SEC,
    flag_reason,
)
from vodparse.gui.review.model import (
    INGAME_FALLBACK_TOOLTIP,
    MATCH_SEEK_OFFSET_SEC_DEFAULT,
    ReviewClip,
    REVIEW_FALLBACK_COLOR,
    REVIEW_FALLBACK_COLOR_DARK,
    REVIEW_MISSING_TAG_COLOR,
    REVIEW_MISSING_TAG_COLOR_DARK,
    REVIEW_SINGLE_GAME_BG_COLOR,
    REVIEW_SINGLE_GAME_BG_COLOR_DARK,
    SETTING_MARK_INGAME_FALLBACK,
    SETTING_MATCH_SEEK_OFFSET_SEC,
    SINGLE_GAME_SET_TOOLTIP,
    export_status_text,
    is_single_game_set,
    match_seek_target,
    missing_tag_message,
    missing_tag_players,
    overlay_proven_for_clips,
    used_ingame_fallback,
    display_tags as _display_tags,
    hms as _hms,
)
from vodparse.gui.review.draft import (
    DraftPayload,
    clear_draft_file,
    read_draft,
    write_draft_file,
)
# The small shared widgets/helpers live in widgets.py; imported under their original
# private names so the call sites in this file read unchanged.
from vodparse.gui.review.widgets import (
    FieldLineEdit as _FieldLineEdit,
    group_label as _group_label,
    nn as _nn,
    vline_separator as _vline_separator,
)
from vodparse.gui.review.detail import ReviewDetailPanel
from vodparse.gui.review.final_review_dialog import FinalReviewDialog
from vodparse.gui.review.session import OpRejected, OpResult, ReviewSession, Selection
from vodparse.gui.review.fix_caps_dialog import FixCapsOptionsDialog
from vodparse.gui.review.sponsor_split_dialog import SponsorSplitOptionsDialog
from vodparse.gui.common import ElidingLabel, is_dark_theme
from vodparse.gui.paths import draft_path, recorded_clock_seconds
from vodparse.schema import Match, ReviewedData

from PyQt6.QtCore import QEvent, QObject, QPoint, QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QKeySequence,
    QPalette,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vodparse.gui.player.trim_player import TrimPlayer

# Auto-saved edits live next to matches.json in the per-vod folder, under the writable workspace
# root (the repo in a checkout, a per-user folder when frozen) so it doesn't depend on the process
# cwd and matches where the GUI writes reviewed.json.


# Toggle for the ops-bar's less-used precision columns (Split set/match, Clip-bounds marks).
# Off by default so the bar leads with the common merge/move/exclude actions; persisted so the
# user's choice sticks across sessions (see _toggle_more / _update_ops_visibility).
SETTING_SHOW_ADVANCED_OPS = "review/show_advanced_ops"


_CLIP_ROLE = Qt.ItemDataRole.UserRole              # clip index (top-level + child items)
_MATCH_START_ROLE = Qt.ItemDataRole.UserRole + 1   # a child match's start (sec), for fine seeking
_MATCH_LOCAL_ROLE = Qt.ItemDataRole.UserRole + 2   # a child match's index within its clip's _matches


class ReviewScreen(QWidget):
    """Emits backRequested(), exportRequested(reviewed_data: dict),
    exportOneRequested(reviewed_data: dict, clip_number: int),
    chaptersRequested(reviewed_data: dict), uploadAssistantRequested(), and
    openExportFolderRequested()."""

    backRequested = pyqtSignal()
    exportRequested = pyqtSignal(dict)
    # The tree's right-click "Export this set/game to a file…": the same reviewed
    # projection as exportRequested plus the clip_number of the one clip to cut.
    exportOneRequested = pyqtSignal(dict, int)
    chaptersRequested = pyqtSignal(dict)
    uploadAssistantRequested = pyqtSignal()
    openExportFolderRequested = pyqtSignal()

    # ── Construction ──────────────────────────────────────────────────────────
    def __init__(self) -> None:
        super().__init__()
        self._init_state()
        self._build_tree()

        detail = self._build_detail()   # also creates self._detail, self._player, and the option buttons

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._tree)
        split.addWidget(detail)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([300, 840])

        self._ops_bar = self._build_ops_bar()
        self._status = ElidingLabel("")   # defaults: ElideRight, no reserved width
        # Full-strength text colour (palette-driven, so it follows the macOS light/dark
        # switch live) in italic, so the status reads as a status without being dim.
        status_font = self._status.font()
        status_font.setItalic(True)
        self._status.setFont(status_font)

        # Reset sits at the right end of the player's transport row, apart from the
        # per-selection editing ops in the ops bar, because it acts on the whole review
        # (every clip, every kind of edit).
        self._reset_btn = QPushButton("Reset all edits")
        self._reset_btn.clicked.connect(self._reset_all)
        self._reset_btn.setEnabled(False)
        self._player.add_transport_widget(self._reset_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(split, stretch=1)
        layout.addWidget(self._ops_bar)
        layout.addWidget(self._status)
        layout.addLayout(self._build_header_actions())

        # Undo/Redo (and every other editing op) get their shortcuts from the Edit/Playback/View
        # menu QActions built in _build_menu_actions; the menus live in MainWindow's menu bar and
        # are hidden off the Review page, which is what gates the shortcuts. (Standalone QShortcuts
        # here would double-register those keys and ambiguous-overload the QActions.)
        self._build_menu_actions()

        # On macOS push buttons don't take keyboard focus on click (the platform convention), so
        # clicking an editing button leaves a text field focused: where Ctrl+Z is QLineEdit's
        # field-local undo, not document undo. Filter every button's press to simply drop the
        # field's focus (no widget gains it), so the window-level Undo shortcut is document undo
        # again. Enter/Escape, by contrast, hand focus to the set tree (see _FieldLineEdit).
        for btn in self.findChildren(QPushButton):
            btn.installEventFilter(self)

    def _init_state(self) -> None:
        """Initialise the non-widget state: the editing model (a session.ReviewSession),
        the view-side caches, the cached settings, and the debounced auto-save timer. No widgets
        are built here."""
        # The editing model (clips, meta, undo/redo, dirtiness) lives in the session; this
        # class resolves widget events into session calls and applies the returned OpResults
        # (see _apply_result). An empty session stands in until load_model/load_from_draft
        # replaces it, so is_dirty/export queries work before any load.
        self._session = ReviewSession(meta={}, clips=[], synth_counter=0,
                                      set_characters="join", resumed=False)
        # Bumped every time the session above is replaced (load_model / load_from_draft).
        # Menus and modal confirms spin nested event loops, so a background Reread run can
        # finish and rebuild the review while one is open; actions built against the old
        # model capture this counter and bail if it moved (see _model_still_current).
        self._model_generation = 0
        self._current: int | None = None
        self._loading = False
        # Selecting a flagged row shows its reason in the status line; see _update_flag_status. We track
        # the reason we put there so we only clear it on click-away (never an action's own message), and
        # suppress it on programmatic reselection (an action owns the status then) via _select_item.
        self._flag_status_text = ""
        self._programmatic_select = False

        # Ops-bar "⋯ More" toggle: reveal the Split and Clip-bounds columns. Off by default and
        # remembered across sessions (see _toggle_more). _mode starts as set mode until a model
        # loads; both feed _update_ops_visibility, which composes the two filters.
        self._settings = QSettings()
        # The long-game flag's old absolute limit, superseded by a headroom past the clock: drop
        # a value v0.9.x saved, unmigrated (see RETIRED_SETTING_MAX_MATCH_SEC). Done here because
        # both the main GUI and the vod grabber build this screen at startup, each on its own store.
        self._settings.remove(RETIRED_SETTING_MAX_MATCH_SEC)
        self._advanced_on = self._settings.value(
            SETTING_SHOW_ADVANCED_OPS, False, type=bool)
        # Over-long / too-short match flagging (settings owned by the SettingsDialog; cached per
        # render). _load_flag_settings builds self._flag_settings (a flags.FlagSettings)
        # from QSettings; refresh_flag_highlights re-reads it on a settings change; the styling
        # lives in _style_clip_row.
        # The one per-VOD flag input: the match clock's length, read from the analysis when
        # a VOD loads (load_model / load_from_draft); Ultimate's 7:00 until then.
        self._full_game_seconds = FlagSettings().full_game_seconds
        self._load_flag_settings()
        self._mode = "set"

        # Edits are auto-saved to a draft JSON, debounced so per-keystroke field edits coalesce
        # into one write (see _schedule_save/_write_draft).
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(700)
        self._save_timer.timeout.connect(self._write_draft)

    def _build_tree(self) -> None:
        """Build the set/clip tree (left pane): hidden header, extended selection for multi-row
        Move, and the selection / right-click-menu wiring."""
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setMinimumWidth(260)
        # Extended selection lets a contiguous run of match rows be moved together; the
        # current-item-driven ops (field edits, merge, split) are unaffected, and the Move
        # buttons re-evaluate their block as the selection changes.
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._tree.currentItemChanged.connect(self._on_select)
        self._tree.itemSelectionChanged.connect(self._update_op_buttons)
        # Right-click any game of a set → exclude just that game from export (isolate it into its
        # own excluded clip): mirrors the Exclude button, reachable from the clicked row.
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_tree_menu)

    def _build_header_actions(self) -> QHBoxLayout:
        """The bottom row of whole-review actions: Back, Open export folder / Upload Assistant
        (both enabled post-export by MainWindow), then the export-progress bar and the Chapters /
        Export buttons. (Reset lives on the player transport instead: see __init__.)"""
        back_btn = QPushButton("← Back")
        back_btn.clicked.connect(self._on_back)
        # Opens the Upload Assistant (MainWindow owns it). Disabled until exported clips exist;
        # MainWindow toggles this via set_upload_available on entering Review and after export.
        self._assistant_btn = QPushButton("Open Upload Assistant")
        self._assistant_btn.setEnabled(False)
        self._assistant_btn.clicked.connect(self.uploadAssistantRequested)
        # Opens this VOD's export folder in the OS file manager (MainWindow handles it).
        self._open_folder_btn = QPushButton("Open export folder")
        self._open_folder_btn.setEnabled(False)
        self._open_folder_btn.clicked.connect(self.openExportFolderRequested)
        # Copying is the only delivery, so the label is fixed (it used to track an Export-tab
        # preference that also offered saving to a file).
        self._chapters_btn = QPushButton("Copy timestamps")
        self._chapters_btn.clicked.connect(self._on_chapters)
        # Opens the Final Review dialog (every set at a glance, inline-editable) before
        # exporting; sits alongside, not instead of, the direct Export clips button.
        self._review_export_btn = QPushButton("Review and export")
        self._review_export_btn.clicked.connect(self._on_review_and_export)
        self._export_btn = QPushButton("Export clips")
        self._export_btn.clicked.connect(self._on_export)
        # Fills clip-by-clip while the export subprocess runs (the GUI counts "Saved:" lines
        # and feeds set_export_progress); hidden otherwise, together with the caption above
        # it. The unit ("games"/"sets") is captured by set_exporting so set_export_progress
        # can word the caption. The count lives in its own small caption right on top of the
        # bar, not in the general status line, which row-selection flag messages also write
        # to and would clobber mid-export.
        self._export_unit = "clips"
        self._export_label = QLabel("")
        self._export_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        caption_font = self._export_label.font()
        if caption_font.pointSizeF() > 0:   # rare pixel-sized fonts: keep the default size
            caption_font.setPointSizeF(caption_font.pointSizeF() * 0.85)
        self._export_label.setFont(caption_font)
        self._export_label.setVisible(False)
        self._export_progress = QProgressBar()
        self._export_progress.setTextVisible(False)
        self._export_progress.setFixedWidth(160)
        self._export_progress.setVisible(False)
        # Stretches center the snug caption+bar pair vertically against the taller buttons.
        progress_col = QVBoxLayout()
        progress_col.setContentsMargins(0, 0, 0, 0)
        progress_col.setSpacing(2)
        progress_col.addStretch(1)
        progress_col.addWidget(self._export_label)
        progress_col.addWidget(self._export_progress, alignment=Qt.AlignmentFlag.AlignHCenter)
        progress_col.addStretch(1)
        actions = QHBoxLayout()
        actions.addWidget(back_btn)
        actions.addWidget(self._open_folder_btn)
        actions.addWidget(self._assistant_btn)
        actions.addStretch(1)
        actions.addLayout(progress_col)
        actions.addWidget(self._chapters_btn)
        actions.addWidget(self._review_export_btn)
        actions.addWidget(self._export_btn)
        return actions

    @override
    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        # A button click leaves a focused field focused on macOS, keeping Ctrl+Z field-local. Just
        # unfocus the field on press (nothing else gains focus) so Ctrl+Z is document undo again
        # (focusOutEvent then ends the typing burst). The press still reaches the button.
        if event is not None and event.type() == QEvent.Type.MouseButtonPress:
            focused = QApplication.focusWidget()
            if isinstance(focused, _FieldLineEdit):
                focused.clearFocus()
        return super().eventFilter(obj, event)

    # ── Edit / Playback / View menu actions ──────────────────────────────────────
    def _build_menu_actions(self) -> None:
        """Build the QActions backing the Edit/Playback/View menus that MainWindow adds to the
        menu bar. Each pairs with an existing ops-bar/transport control and runs the same handler;
        _sync_menu_actions mirrors the buttons' enabled/checked state onto them. The menus are
        hidden off the Review page, and hiding a menu suppresses its actions' shortcuts: that's
        the gate that keeps these keys inert elsewhere."""

        def act(label: str, shortcut: "QKeySequence.StandardKey | str | None",
                handler: Callable[..., object] | None, *,
                checkable: bool = False) -> QAction:
            a = QAction(label, self)
            if shortcut is not None:
                a.setShortcut(shortcut)
            a.setCheckable(checkable)
            if handler is not None:
                a.triggered.connect(handler)
            self.addAction(a)
            return a

        sk = QKeySequence.StandardKey
        self._edit_actions: dict[str, QAction] = {
            "undo": act("Undo", sk.Undo, self._undo),
            "redo": act("Redo", sk.Redo, self._redo),
            "move_prev": act("Move game to previous set", "Ctrl+Shift+Up", self._move_prev),
            "move_next": act("Move game to next set", "Ctrl+Shift+Down", self._move_next),
            "merge_prev": act("Merge with previous set", "Ctrl+Shift+[", self._merge_prev),
            "merge_next": act("Merge with next set", "Ctrl+Shift+]", self._merge_next),
            # Ctrl+Shift+B ("break" the set), not S; Cmd+Shift+S reads as Save As on macOS.
            "split": act("Split set at selected game", "Ctrl+Shift+B", self._split_here),
            "split_match": act("Split game at playhead", "Ctrl+Shift+D", self._split_match),
            "swap": act("Swap characters", "Ctrl+Shift+P", self._swap_characters),
            "exclude": act("Exclude from export", "Ctrl+Shift+X", self._exclude_selected,
                           checkable=True),
            "mark_start": act("Mark clip start at playhead", "Ctrl+Shift+I",
                              self._player.stamp_start),
            "mark_end": act("Mark clip end at playhead", "Ctrl+Shift+O",
                            self._player.stamp_end),
            "reset": act("Reset all edits…", None, self._reset_all),
        }

        self._playback_actions: dict[str, QAction] = {
            "play": act("Play / Pause", "Space", self._player.toggle_play),
            "jump_start": act("Jump to clip start", "Ctrl+Home", self._player.jump_to_start),
            "jump_end": act("Jump to clip end", "Ctrl+End", self._player.jump_to_end),
        }

        # Two checkable view toggles mirror an existing button/checkbox both ways. setChecked is
        # wired AFTER seeding the initial state so the seed doesn't fire the handler; the value
        # guards in _toggle_more / _on_toggle_full keep the mutual updates from looping.
        advanced = act("Advanced controls", "Ctrl+Shift+A", None, checkable=True)
        advanced.setChecked(self._advanced_on)
        advanced.toggled.connect(self._toggle_more)
        advanced.toggled.connect(self._more_btn.setChecked)
        self._more_btn.toggled.connect(advanced.setChecked)

        whole_vod = act("Show whole VOD", "Ctrl+Shift+F", None, checkable=True)
        whole_vod.setChecked(self._player.is_full_view())
        whole_vod.toggled.connect(self._player.set_full_view)
        self._player.fullViewChanged.connect(whole_vod.setChecked)

        self._view_actions: dict[str, QAction] = {
            "advanced": advanced,
            "whole_vod": whole_vod,
            "bounds": act("Edit preview bounds…", None, self._player.edit_preview_bounds),
        }

    def _sync_menu_actions(self) -> None:
        """Mirror the canonical ops-bar buttons' enabled/checked state onto the Edit-menu actions,
        so the menu items track the buttons exactly. Playback/View actions stay enabled (their
        menus are hidden off the Review page); their checked state is bound separately."""
        if not hasattr(self, "_edit_actions"):
            return   # an enablement refresh can fire during construction, before the actions exist
        e = self._edit_actions
        e["undo"].setEnabled(self._undo_btn.isEnabled())
        e["redo"].setEnabled(self._redo_btn.isEnabled())
        e["reset"].setEnabled(self._reset_btn.isEnabled())
        e["mark_start"].setEnabled(self._trim["set_start"].isEnabled())
        e["mark_end"].setEnabled(self._trim["set_end"].isEnabled())
        e["exclude"].setEnabled(self._exclude_btn.isEnabled())
        e["exclude"].setChecked(self._exclude_btn.isChecked())
        e["swap"].setEnabled(self._detail.swap_button.isEnabled())
        # The restructuring ops only apply in set mode, where _update_op_buttons keeps the buttons
        # current; in match mode it early-returns and leaves them stale, so gate on the mode too.
        set_mode = self._session.meta.get("mode") != "match"
        for key in ("move_prev", "move_next", "merge_prev", "merge_next", "split", "split_match"):
            e[key].setEnabled(set_mode and self._ops[key].isEnabled())

    def edit_menu_actions(self) -> list[QAction | None]:
        """Ordered Edit-menu entries; None marks a separator (see MainWindow._build_menu_bar)."""
        e = self._edit_actions
        return [e["undo"], e["redo"], None,
                e["move_prev"], e["move_next"], None,
                e["merge_prev"], e["merge_next"], e["split"], e["split_match"], None,
                e["swap"], e["exclude"], None,
                e["mark_start"], e["mark_end"], None,
                e["reset"]]

    def playback_menu_actions(self) -> list[QAction | None]:
        p = self._playback_actions
        return [p["play"], None, p["jump_start"], p["jump_end"]]

    def view_menu_actions(self) -> list[QAction | None]:
        v = self._view_actions
        return [v["advanced"], v["whole_vod"], None, v["bounds"]]

    # ── Ops bar ───────────────────────────────────────────────────────────────
    def _build_ops_bar(self) -> QWidget:
        """Toolbar of editing actions, grouped by what they act on: the current clip's export
        bounds, the selected set, the selected match. Each group is a column: a bold label naming
        the group's object sits above that group's row of buttons, so the short captions stay
        unambiguous. Buttons enable per the current tree selection. Clip bounds and Undo/Redo apply
        in any mode; the Set and Match columns (and their separators) show only in set mode (see
        _apply_mode_visibility). Reset all edits lives in the header row, not here."""
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)

        def button(label: str, tip: str, slot, target: QHBoxLayout) -> QPushButton:
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            btn.setEnabled(False)
            # Floor the width at the caption's text plus a little padding, so a narrow window can
            # squeeze the buttons a touch but never far enough to elide the label to an ellipsis
            # (the Split set/match buttons were the first to disappear). The roomier sizeHint stays
            # the preferred width, so they sit comfortably when there's space.
            btn.setMinimumWidth(btn.fontMetrics().horizontalAdvance(label) + 16)
            target.addWidget(btn)
            return btn

        def group(title: str) -> tuple[QWidget, QHBoxLayout]:
            """A category column: a bold label on top, that group's buttons in a row beneath it."""
            col = QWidget()
            box = QVBoxLayout(col)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(3)
            box.addWidget(_group_label(title))
            btn_row = QHBoxLayout()
            box.addLayout(btn_row)
            return col, btn_row

        # Match: act on the selected match row. Leads the restructuring columns (no separator
        # before it): nudging a stray match into its neighbour set is the most common fix, and it
        # sits nearest the tree where the match was just selected.
        match_col, match_row = group("Game:")
        self._ops: dict[str, QPushButton] = {}
        self._ops["move_prev"] = button(
            "▲ Move", "Move the selected game into the previous set", self._move_prev, match_row)
        self._ops["move_next"] = button(
            "▼ Move", "Move the selected game into the next set", self._move_next, match_row)
        row.addWidget(match_col)

        # Set: restructure the selected set relative to its neighbours.
        set_sep = _vline_separator()
        row.addWidget(set_sep)
        set_col, set_row = group("Set:")
        self._ops["merge_prev"] = button(
            "▲ Merge", "Merge this set with the previous set.", self._merge_prev, set_row)
        self._ops["merge_next"] = button(
            "Merge ▼", "Merge this set with the next set.", self._merge_next, set_row)
        row.addWidget(set_col)
        # The per-player Options buttons live up in the detail panel, under the player
        # columns they act on (see _build_detail), but join _ops so enablement and
        # match-mode hiding treat them like the other set ops.
        self._ops["p1_options"] = self._detail.player_option_buttons[0]
        self._ops["p2_options"] = self._detail.player_option_buttons[1]

        # Exclude: flag the selected set to be left out of the export. Anchored right after the
        # Match moves so it stays put when Advanced reveals the Split/Clip-bounds columns to its
        # right. Visible in any mode (not in self._structural), like Undo/Redo, since exclusion
        # applies to a set or a match-clip alike. Checkable: pressed when the set is excluded.
        row.addWidget(_vline_separator())
        export_col, export_row = group("Exporting:")
        self._exclude_btn = button(
            "Exclude", "Exclude the highlighted row from the export.",
            self._exclude_selected, export_row)
        self._exclude_btn.setCheckable(True)
        row.addWidget(export_col)

        # Split: divide a set or match at the current position.
        split_sep = _vline_separator()
        row.addWidget(split_sep)
        split_col, split_row = group("Split:")
        self._ops["split"] = button(
            "Split set", "Split this set into two at the selected game.", self._split_here, split_row)
        self._ops["split_match"] = button(
            "Split game", "Split the selected game in two at the playhead.", self._split_match, split_row)
        row.addWidget(split_col)

        # Clip bounds: stamp the current clip's export start/end at the playhead (any mode).
        clip_sep = _vline_separator()
        row.addWidget(clip_sep)
        clip_col, clip_row = group("Clip bounds:")
        self._trim = {
            "set_start": button("⇥ Mark start",
                                "Move this clip's export start to the playhead.",
                                self._player.stamp_start, clip_row),
            "set_end":   button("⇤ Mark end",
                                "Move this clip's export end to the playhead.",
                                self._player.stamp_end, clip_row),
        }
        row.addWidget(clip_col)

        # Undo/Redo: apply in any mode. A blank header keeps its buttons on the same line as the
        # other columns' button rows.
        row.addWidget(_vline_separator())
        edit_col, edit_row = group(" ")
        self._undo_btn = button("Undo", "", self._undo, edit_row)
        self._redo_btn = button("Redo", "", self._redo, edit_row)
        row.addWidget(edit_col)

        row.addStretch(1)

        # "⋯ More": view toggle (not an edit) pinned to the far right, revealing the Split and
        # Clip-bounds columns. Checkable like Exclude; starts enabled (it never depends on a
        # selection) and reflects the persisted preference.
        more_col, more_row = group(" ")
        self._more_btn = button(
            "⋯ Advanced", "Show the Split and Clip-bounds controls.",
            self._toggle_more, more_row)
        self._more_btn.setCheckable(True)
        self._more_btn.setEnabled(True)
        self._more_btn.setChecked(self._advanced_on)
        row.addWidget(more_col)

        # Set/Match restructuring only makes sense when clips are sets; hide those columns whole
        # (each container hides its label + buttons; their separators hide alongside) in match
        # mode. The per-player Options buttons live in the detail panel, so hide them explicitly.
        self._structural = [match_col, set_sep, set_col, split_sep, split_col, clip_sep,
                            self._ops["p1_options"], self._ops["p2_options"]]
        # The "⋯ More" toggle hides these precision columns by default. Their separators ride
        # along so no stray dividers are left behind. clip_col is here but not in _structural:
        # Clip bounds applies in any mode, so only the toggle (not the set/match mode) gates it.
        self._advanced = [split_sep, split_col, clip_sep, clip_col]
        # Union (order-preserving) of every widget whose visibility the two filters drive.
        self._ops_toggleable = list(dict.fromkeys(self._structural + self._advanced))
        self._update_ops_visibility()   # apply the persisted toggle before the first load
        return bar

    def _toggle_more(self, on: bool) -> None:
        # Guard so the button <-> View-menu-action mirror settles instead of looping.
        if on == self._advanced_on:
            return
        self._advanced_on = on
        self._settings.setValue(SETTING_SHOW_ADVANCED_OPS, on)
        self._update_ops_visibility()

    def _apply_mode_visibility(self, mode: str) -> None:
        self._mode = mode
        self._update_ops_visibility()

    def _update_ops_visibility(self) -> None:
        """A widget shows only if both filters allow it: the set/match mode (structural columns
        hide in match mode) and the "⋯ More" toggle (advanced columns hide when it's off)."""
        for w in self._ops_toggleable:
            mode_ok = w not in self._structural or self._mode != "match"
            adv_ok = w not in self._advanced or self._advanced_on
            w.setVisible(mode_ok and adv_ok)

    # ── Status line (GUI-applied) ─────────────────────────────────────────────
    def show_status(self, message: str) -> None:
        """Show a one-line message in the Review status line (used by the GUI to report the
        chapter-timestamp copy, which it performs)."""
        self._status.setText(message)

    # ── Review-flag settings ──────────────────────────────────────────────────
    def _adopt_clock_length(self, video_path: str) -> None:
        """Take the match clock's length from the VOD's saved analysis (a custom overlay
        layout's own clock, else the game's, else 7:00) and rebuild the flag settings around
        it, so the bad-start flag back-dates game starts the way the extraction did and the
        over-long flag measures its headroom from this VOD's clock."""
        self._full_game_seconds = recorded_clock_seconds(video_path)
        self._load_flag_settings()

    def _load_flag_settings(self) -> None:
        """Cache the flag toggles and limits from settings (read once per render)."""
        # Bad-start-timer flags: compare each match's stored start to the consensus of its own clock
        # readings. Two independent toggles (see model.py): the severe pair (amber) and the softer
        # long-lead-in (yellow), the latter with a tunable threshold in OCR scan steps (intervals, not
        # seconds, so it holds across cadences).
        self._flag_settings = FlagSettings(
            flag_long_enabled=self._settings.value(SETTING_FLAG_LONG_MATCHES, True, type=bool),
            long_match_headroom_sec=self._settings.value(
                SETTING_LONG_MATCH_HEADROOM_SEC, LONG_MATCH_HEADROOM_SEC_DEFAULT, type=float),
            flag_short_enabled=self._settings.value(SETTING_FLAG_SHORT_MATCHES, True, type=bool),
            min_match_sec=self._settings.value(
                SETTING_MIN_MATCH_SEC, MIN_MATCH_SEC_DEFAULT, type=float),
            flag_weird_timer_enabled=self._settings.value(
                SETTING_FLAG_WEIRD_TIMER, True, type=bool),
            flag_long_leadin_enabled=self._settings.value(
                SETTING_FLAG_LONG_LEADIN, True, type=bool),
            long_leadin_intervals=self._settings.value(
                SETTING_LONG_LEADIN_INTERVALS, LONG_LEADIN_INTERVALS_DEFAULT, type=float),
            full_game_seconds=self._full_game_seconds,
        )
        # Not a flag, but a review-screen marker gated the same way: tint sets/matches whose tags
        # fell back to the in-game nameplates (see _is_fallback_row / used_ingame_fallback).
        self._mark_ingame_fallback_enabled = self._settings.value(
            SETTING_MARK_INGAME_FALLBACK, True, type=bool)

    def refresh_flag_highlights(self) -> None:
        """Re-read the flag toggles/limits and re-style the tree in place. Called by the GUI when
        Settings change, so toggling or retuning a limit updates the highlights live without
        rebuilding the tree (which would drop the selection)."""
        self._load_flag_settings()
        for i in range(self._tree.topLevelItemCount()):
            item = _nn(self._tree.topLevelItem(i))
            idx = item.data(0, _CLIP_ROLE)
            if idx is not None and idx < len(self._session.clips):
                self._style_clip_row(item, self._session.clips[idx])
        # Re-styling refreshed each row's reason tooltip; follow it on the selected row so a flag the
        # user just toggled off/on (or retuned past the threshold) updates the status line live.
        self._update_flag_status(self._tree.currentItem())
        # The tag-field tooltips aren't part of the tree styling, so refresh the open clip's here too:
        # toggling the fallback marker then updates the detail panel live, not just on reselection.
        if self._current is not None:
            self._apply_fallback_field_tooltips(self._session.clips[self._current])

    # ── Scrubber regions ──────────────────────────────────────────────────────
    def _apply_core_regions(self, clip: ReviewClip) -> None:
        """Tell the scrubber which spans are match coverage (main green). Each match is its
        own span so the dead space between them shows in the lighter buffer green."""
        matches = clip["_matches"]
        core = [(m["start"]["timestamp_sec"], m["end"]["timestamp_sec"]) for m in matches]
        self._player.set_core_regions(core)

    def _other_set_regions(self, current: int) -> list[tuple[float, float]]:
        """Export spans of every set except the selected one (excluded sets included, since the
        scrubber is a navigation aid, not an export preview). The scrubber clips these to its
        visible window, so neighbouring sets show in a deeper red without any recomputation."""
        return [(c["start_sec"], c["end_sec"])
                for i, c in enumerate(self._session.clips) if i != current]

    # ── Detail panel & player ─────────────────────────────────────────────────
    def _build_detail(self) -> QWidget:
        """The detail area: the editable fields panel (ReviewDetailPanel: see detail.py),
        the bounds label, and the player below. The panel owns the field widgets and layout;
        this wires its signals to the handlers here, which own the clip model and undo."""
        panel = QWidget()

        self._detail = ReviewDetailPanel()
        self._detail.fieldEdited.connect(self._on_field_changed)
        self._detail.fieldReleased.connect(lambda: self._tree.setFocus())
        self._detail.fieldFocusLeft.connect(self._end_field_burst)
        self._detail.swap_button.clicked.connect(self._swap_characters)
        for i, menu in enumerate(self._detail.player_menus):
            # Rebuilt with live candidacy on each open; the menus' contents are QActions bound
            # to handlers here over screen-owned clip data (see _rebuild_player_menu).
            menu.aboutToShow.connect(lambda p=i: self._rebuild_player_menu(p))

        self._bounds_label = QLabel("Start: -   End: -")

        self._player = TrimPlayer(settings=self._settings)
        self._player.startChanged.connect(self._on_start_changed)
        self._player.endChanged.connect(self._on_end_changed)
        self._player.scrubbed.connect(self._on_scrub)

        layout = QVBoxLayout(panel)
        layout.addWidget(self._detail)
        layout.addWidget(self._bounds_label)
        layout.addWidget(self._player, stretch=1)
        return panel

    # ── Model ─────────────────────────────────────────────────────────────────
    def load_model(self, matches: list[Match], mode: str, group_by: str,
                   video_path: str, stem: str, tournament_name: str = "",
                   set_characters: str = "join") -> None:
        """Build a fresh review from a just-extracted match list. Any saved edit draft is left to
        the caller; re-running extract regenerates the matches, so prior edits no longer line up
        and the GUI clears the draft before calling this. To reopen saved edits instead, the GUI
        calls load_from_draft (the Resume edits button), bypassing this."""
        self._model_generation += 1  # invalidate menus/confirms opened on the old model
        self._session = ReviewSession.from_matches(
            matches, mode, group_by, video_path, stem,
            tournament_name=tournament_name, set_characters=set_characters)
        self._adopt_clock_length(video_path)
        self._apply_mode_visibility(mode)
        self._player.set_source(video_path)
        self._populate_tree()
        self._refresh_undo_buttons()
        if self._session.clips:
            self._select_item(self._tree.topLevelItem(0))

    def _match_start_markers(self, clip: ReviewClip) -> list[float]:
        """Where each match (after the first) starts, in seconds: the internal split points of
        the set. The first match's start is skipped since it's already the green/red edge. Empty
        in match mode (a clip is a single match: no internal splits)."""
        if self._session.meta.get("mode") == "match":
            return []
        return [m["start"]["timestamp_sec"] for m in clip["_matches"][1:]]

    # ── Tree ──────────────────────────────────────────────────────────────────
    def _populate_tree(self) -> None:
        self._load_flag_settings()   # reflect current flag settings on every (re)build
        is_set = self._session.meta["mode"] != "match"
        unit = "Set" if is_set else "Game"
        # Block signals while the tree is mid-rebuild: clear()/addTopLevelItem emit
        # itemSelectionChanged, and letting _update_op_buttons read the half-built tree
        # against the already-swapped self._session.clips is unsafe (re-entrant Qt access).
        self._tree.blockSignals(True)
        try:
            self._tree.clear()
            for idx, clip in enumerate(self._session.clips):
                item = QTreeWidgetItem([self._clip_label(unit, clip)])
                item.setData(0, _CLIP_ROLE, idx)
                if is_set:
                    for n, m in enumerate(clip["_matches"], start=1):
                        child = QTreeWidgetItem([
                            f"Game {n} ({m['match_number']}): "
                            f"{_hms(m['start']['timestamp_sec'])}–{_hms(m['end']['timestamp_sec'])}"
                        ])
                        child.setData(0, _CLIP_ROLE, idx)
                        child.setData(0, _MATCH_START_ROLE, m["start"]["timestamp_sec"])
                        child.setData(0, _MATCH_LOCAL_ROLE, n - 1)
                        item.addChild(child)
                self._tree.addTopLevelItem(item)
                self._style_clip_row(item, clip)  # grey + strike a set excluded from export
            self._tree.expandAll()
        finally:
            self._tree.blockSignals(False)

    @staticmethod
    def _clip_label(unit: str, clip: ReviewClip) -> str:
        p1, p2 = _display_tags(clip)
        return f"{unit} {clip['clip_number']}: {p1} vs {p2}"

    def _style_clip_row(self, item: QTreeWidgetItem, clip: ReviewClip) -> None:
        """Style a set/match row and its game children: grey + strike when excluded from the
        export (so the tree shows at a glance which won't be cut), and (when the relevant flags are
        on) tint amber, with an explanatory tooltip, any match flag_reason fires on (a possible
        replay/missed split, a warm-up/false positive, or a missed start, to check). A whole
        excluded set greys the parent and every child (an excluded single game lives in its own
        excluded set, so it's greyed the same way). Excluded takes precedence over the amber flag,
        row by row, and the flag in turn over the in-game-fallback tint: one meaning per row colour,
        priority excluded > missing tag > flagged > fallback > default. Independently of that text
        colour, a set holding a single game gets a muted-red *background* band on both the set row
        and its one game child (a separate channel, so it coexists with an amber flag), suppressed
        only when the row is excluded or missing-tag. All colours track the light/dark theme."""
        set_excluded = clip.get("exclude_from_export", False)
        dim = self.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
        # Two flag tints: amber for the severe cases, a softer yellow for a long lead-in (worth a glance).
        # Each picks its theme variant so the tint stays legible on both a light and a dark background.
        dark = is_dark_theme()
        flag = REVIEW_FLAG_COLOR_DARK if dark else REVIEW_FLAG_COLOR
        leadin = REVIEW_LEADIN_COLOR_DARK if dark else REVIEW_LEADIN_COLOR
        brush = {"flag": QBrush(QColor(flag)), "leadin": QBrush(QColor(leadin))}
        # The in-game-fallback marker is a clip-level property, so it tints the set/match (parent) row
        # only, and sits one tier below the flags: a flagged row keeps its amber (more urgent), and the
        # fallback still surfaces in the detail panel.
        fallback = self._is_fallback_row(clip)
        fallback_brush = QBrush(QColor(REVIEW_FALLBACK_COLOR_DARK if dark else REVIEW_FALLBACK_COLOR))
        # The missing-tag marker is also clip-level (a blank/"none" grouping_tags slot), so it
        # tints the parent row only too, and outranks the flags: a set that can't be identified
        # is a harder blocker than a possible replay/short/bad-start review flag.
        missing_msg = missing_tag_message(missing_tag_players(clip))
        missing_brush = QBrush(QColor(REVIEW_MISSING_TAG_COLOR_DARK if dark else REVIEW_MISSING_TAG_COLOR))
        # A lone-game set gets a background band rather than a text tint: a separate channel from the
        # foreground priority above, so it can show under an amber flag; only excluded/missing-tag mute it.
        single_game = is_single_game_set(clip, self._session.meta.get("mode", ""))
        single_game_bg = QBrush(QColor(
            REVIEW_SINGLE_GAME_BG_COLOR_DARK if dark else REVIEW_SINGLE_GAME_BG_COLOR))
        is_set = self._session.meta.get("mode") != "match"
        matches = clip.get("_matches", [])

        # Each row carries (item, excluded, result) where result is (reason, severity) or None. In set
        # mode the children are the games; in match mode the top-level item is the single match. A flagged
        # child also marks its (possibly collapsed) parent set: amber if any child is severe, else the
        # softer yellow when only long lead-ins fired. A whole excluded set greys its parent and every
        # child; a single excluded game now lives in its own excluded set, so per-game greying is just
        # set_excluded (no per-game span to honour since first/last trimming was retired).
        if is_set:
            child_results = [flag_reason(matches[i], i == 0, self._flag_settings) if i < len(matches) else None
                             for i in range(item.childCount())]
            flagged = [(i, res[1]) for i, res in enumerate(child_results) if res]
            if flagged:
                names = ", ".join(f"Game {i + 1}" for i, _ in flagged)
                severity = "flag" if any(sev == "flag" for _, sev in flagged) else "leadin"
                parent_result: tuple[str, str] | None = (
                    f"Flagged for review: {names}.", severity)
            else:
                parent_result = None
            # The trailing flag marks the set (parent) row: the only row the fallback tint may paint.
            rows = [(item, set_excluded, parent_result, True)]
            rows += [(_nn(item.child(i)), set_excluded, child_results[i], False)
                     for i in range(item.childCount())]
        else:
            # Match mode: each clip is one standalone match, so its start is the editable trim point.
            rows = [(item, set_excluded,
                     flag_reason(matches[0] if matches else None, True, self._flag_settings), True)]

        for r, excluded, result, is_parent in rows:
            f = r.font(0)
            f.setStrikeOut(excluded)
            r.setFont(0, f)
            # Background band on both the set row and its single game child, independent of the
            # foreground tint. QBrush() clears it, so a merge/split that drops the game count
            # re-styles the rows back to no band for free.
            band = single_game and not excluded and not missing_msg
            r.setBackground(0, single_game_bg if band else QBrush())
            if excluded:
                r.setForeground(0, QBrush(dim))
                r.setToolTip(0, "")
            elif is_parent and missing_msg:
                r.setForeground(0, missing_brush)
                r.setToolTip(0, missing_msg)
            elif result:
                reason, severity = result
                r.setForeground(0, brush[severity])
                r.setToolTip(0, reason)
            elif is_parent and fallback:
                r.setForeground(0, fallback_brush)
                r.setToolTip(0, INGAME_FALLBACK_TOOLTIP)
            else:
                r.setForeground(0, QBrush())   # default colour
                # The band's own note, but only when no more-specific tooltip above claimed the row.
                r.setToolTip(0, SINGLE_GAME_SET_TOOLTIP if band else "")

    def _is_fallback_row(self, clip: ReviewClip) -> bool:
        """Whether this clip's identity tags came from the in-game fallback *and* the marker is on:
        the single gate both the row tint and the tag-field tooltips read (so they never disagree)."""
        return self._mark_ingame_fallback_enabled and used_ingame_fallback(
            clip, self._session.meta.get("mode", ""), self._session.meta.get("group_by", "auto"),
            trust_half_overlay=overlay_proven_for_clips(self._session.clips))

    def _apply_fallback_field_tooltips(self, clip: ReviewClip) -> None:
        """Give the Tag fields the in-game-fallback note when this clip fell back, else clear it."""
        tip = INGAME_FALLBACK_TOOLTIP if self._is_fallback_row(clip) else ""
        self._detail.set_fallback_note(tip)

    # ── Selection / editing ─────────────────────────────────────────────────--
    def _on_select(self, item: QTreeWidgetItem | None, _prev) -> None:
        if item is None:
            return
        idx = item.data(0, _CLIP_ROLE)
        if idx is None:
            return
        self._current = idx
        self._session.end_field_burst()   # a field edit after navigating starts a fresh undo entry
        clip = self._session.clips[idx]

        self._loading = True
        self._detail.set_clip_fields(clip)
        self._apply_fallback_field_tooltips(clip)
        self._loading = False

        self._update_bounds_label()
        self._player.set_region(clip["start_sec"], clip["end_sec"])
        self._player.set_markers(self._match_start_markers(clip))
        self._apply_core_regions(clip)
        self._player.set_other_regions(self._other_set_regions(self._current))

        # Clicking a child match row seeks within the set to that match (the set window and
        # detail panel stay put: just finer navigation than the set's start). The playhead lands
        # the user's seek offset into the game, since production usually cuts to the game a
        # beat after the detected start. Read live so a Settings change applies on the next click.
        match_start = item.data(0, _MATCH_START_ROLE)
        if match_start is not None:
            local = item.data(0, _MATCH_LOCAL_ROLE)
            match_end = clip["_matches"][local]["end"]["timestamp_sec"]
            offset = self._settings.value(
                SETTING_MATCH_SEEK_OFFSET_SEC, MATCH_SEEK_OFFSET_SEC_DEFAULT, type=float)
            self._player.seek_to(match_seek_target(match_start, match_end, offset))

        self._update_op_buttons(item)
        self._update_flag_status(item)

    def _select_item(self, item: QTreeWidgetItem | None) -> None:
        """Programmatically select a row without letting it hijack the status line with a flag reason:
        that's reserved for the user navigating to a flagged row; the calling action owns the status.
        _on_select still runs (loads the clip); only the flag-status step is suppressed."""
        self._programmatic_select = True
        try:
            self._tree.setCurrentItem(item)
        finally:
            self._programmatic_select = False

    def _update_flag_status(self, item: QTreeWidgetItem | None) -> None:
        """Show a flagged row's reason (its column-0 tooltip, set by _style_clip_row) in the status line;
        when the selection moves to an unflagged row, clear it, but only if our flag message is still
        what's showing, so an action's own status message is never clobbered."""
        if self._programmatic_select:
            return
        reason = item.toolTip(0) if item is not None else ""
        if reason:
            self._status.setText(reason)
            self._flag_status_text = reason
        else:
            if self._flag_status_text and self._status.text() == self._flag_status_text:
                self._status.setText("")
            self._flag_status_text = ""

    def _on_scrub(self, sec: float) -> None:
        """User dragged/clicked the scrubber: highlight the match the playhead is over, within
        the current set only. Does nothing in the red ±margin context (other sets) and never
        moves the playhead or resets the view; only the sidebar selection follows."""
        if self._current is None:
            return
        clip = self._session.clips[self._current]
        if not (clip["start_sec"] <= sec <= clip["end_sec"]):
            return  # red region: leave the selection where it is
        # Bucket by match start (matches the amber ticks): the last match begun at or before sec,
        # or the first match when sec is still in the lead-in.
        idx = 0
        for i, m in enumerate(clip["_matches"]):
            if m["start"]["timestamp_sec"] <= sec:
                idx = i
            else:
                break
        child = _nn(self._tree.topLevelItem(self._current)).child(idx)
        if child is None or child is self._tree.currentItem():
            return  # match mode (no child rows) or already selected
        # Select without running _on_select (which would reset the view and seek the playhead).
        self._tree.blockSignals(True)
        try:
            self._tree.setCurrentItem(child)
        finally:
            self._tree.blockSignals(False)
        self._update_op_buttons()
        self._update_flag_status(child)   # scrubbing onto a match is user navigation: show its flag

    def _on_field_changed(self, key: str | None = None) -> None:
        if self._loading or self._current is None:
            return
        # The per-source tournament/in-game tags stay as detected: only the identity tags,
        # sponsors, and characters are editable (the sources feed the CSV posterity columns).
        # The session handles the burst coalescing (one undo entry per typing burst).
        result = self._session.edit_fields(
            self._current,
            tags=(self._detail.field("tag_1").text(), self._detail.field("tag_2").text()),
            sponsors=(self._detail.field("sponsor_1").text(),
                      self._detail.field("sponsor_2").text()),
            characters=(self._detail.field("char_1").text(),
                        self._detail.field("char_2").text()),
            round_text=self._detail.field("round").text(),
            best_of=self._detail.field("best_of").text(),
            burst_key=key, selection=self._current_selection())
        self._apply_result(result)
        # A tag edit can flip the missing-tag state (blank/"none" ↔ a real tag): also refresh
        # the status-line prompt so it clears the instant both tags look real (_apply_result
        # already restyled the row).
        if key in ("tag_1", "tag_2"):
            parent = self._tree.topLevelItem(self._current)   # same lookup _propagate_tags uses
            if parent is not None:
                self._update_flag_status(parent)

    def _end_field_burst(self) -> None:
        """End the current field-typing burst so the next field edit starts a fresh undo entry.
        Called when focus leaves a field (Enter/Escape/click-away/window blur); harmless when no
        burst is open or when a selection change already cleared it."""
        self._session.end_field_burst()

    def _swap_characters(self) -> None:
        """Swap the port-anchored data (characters and in-game tags) between the players: see
        ReviewSession.swap_characters. The tag/sponsor fields stay put: they follow the stream
        overlay, which is exactly what's backwards when this button is needed. The panel then
        reloads from the swapped clip (under _loading, so the writeback doesn't re-fire)."""
        if self._loading or self._current is None:
            return
        result = self._session.swap_characters(self._current, self._current_selection())
        self._loading = True
        self._detail.set_clip_fields(self._session.clips[self._current])
        self._loading = False
        self._apply_result(result)

    def _on_start_changed(self, sec: float) -> None:
        self._set_bound("start_sec", sec, "start")

    def _on_end_changed(self, sec: float) -> None:
        self._set_bound("end_sec", sec, "end")

    def _set_bound(self, key: Literal["start_sec", "end_sec"], sec: float, label: str) -> None:
        """Stamp a clip's start/end (see ReviewSession.set_bound; a no-op restamp returns
        None and is ignored) and refresh the bounds label."""
        if self._current is None:
            return
        result = self._session.set_bound(self._current, key, sec, label,
                                         self._current_selection())
        if result is None:
            return
        self._update_bounds_label()
        self._apply_result(result)

    def _toggle_exclude(self, _checked: bool = False, *, clip_idx: int | None = None) -> None:
        """Flag/unflag a set (or match-clip) so it's left out of the export (see
        ReviewSession.toggle_exclude). `clip_idx` defaults to the current selection (the Exclude
        button); the right-click menu passes the clicked clip. (`_checked` just absorbs the
        checkable button's clicked(bool) argument.)"""
        if clip_idx is None:
            clip_idx, _ = self._selection()
        if clip_idx is None:
            return
        self._apply_result(self._session.toggle_exclude(clip_idx, self._current_selection()))

    def _exclude_selected(self, _checked: bool = False) -> None:
        """The Exclude button / Ctrl+Shift+X handler: act on whatever tree row is highlighted. A set
        row (or any row in match mode: there are no game rows) toggles the whole set. A game row of a
        multi-game set excludes just that game by splitting it into its own one-game set and excluding
        it (_isolate_and_exclude_game), the same for first, middle, or last. Pressing Exclude on any
        row of an already-excluded set re-includes the whole set. (`_checked` absorbs the checkable
        widget's clicked(bool) arg.)"""
        clip_idx, local = self._selection()
        if clip_idx is None:
            return
        clip = self._session.clips[clip_idx]
        # Set row, match mode (local is None), or any row of an already-excluded set → whole-set
        # toggle. The excluded-set case wins over the per-game branch so the press brings it back.
        if local is None or clip.get("exclude_from_export", False):
            self._toggle_exclude(clip_idx=clip_idx)
        elif len(clip["_matches"]) <= 1:    # single-game set: the game IS the set
            self._toggle_exclude(clip_idx=clip_idx)
        else:
            self._isolate_and_exclude_game(clip_idx, local)   # first / middle / last all isolate

    def _game_isolate_choice(self, item: QTreeWidgetItem | None) -> tuple[int, int] | None:
        """`(clip_idx, local)` when `item` is a game row eligible for isolate-and-exclude: a game of
        a multi-game set still in the export (any game: first, middle, or last), else None. A
        single-game set has no game to isolate (the game IS the set → whole-set Exclude). Pure
        decision split out of _on_tree_menu so it's testable without exec()-ing a modal menu."""
        if self._session.meta.get("mode") == "match" or item is None:
            return None
        clip_idx = item.data(0, _CLIP_ROLE)
        local = item.data(0, _MATCH_LOCAL_ROLE)
        if clip_idx is None or local is None:
            return None                                # a set/top-level row, not a game
        clip = self._session.clips[clip_idx]
        if clip.get("exclude_from_export", False) or len(clip["_matches"]) <= 1:
            return None                                # excluded set, or a lone game (use whole-set)
        return clip_idx, local

    def _export_one_choice(self, item: QTreeWidgetItem | None) -> int | None:
        """The clip_number to offer "Export … to a file…" for when `item` is any row of a
        clip still in the export, else None. An excluded clip isn't offered; it's dropped
        from the reviewed projection the single export cuts from (re-include it first).
        Pure decision split out of _on_tree_menu so it's testable without exec()-ing a
        modal menu."""
        if item is None:
            return None
        clip_idx = item.data(0, _CLIP_ROLE)
        if clip_idx is None:
            return None
        clip = self._session.clips[clip_idx]
        if clip.get("exclude_from_export", False):
            return None
        return clip["clip_number"]

    def export_selected_clip(self) -> None:
        """File → "Export Selected to a File…": the single-clip export for the tree's current
        row: the same action the tree's right-click menu offers, so a keyboard/menu-driven
        review doesn't need the mouse gesture. Quietly explains itself in the status line
        when the selection isn't exportable (nothing selected, or the clip is excluded)."""
        clip_number = self._export_one_choice(self._tree.currentItem())
        if clip_number is None:
            unit = self._session.unit_label().lower()
            self.show_status(f"Select a {unit} that isn't excluded from export first")
            return
        self.exportOneRequested.emit(self.build_reviewed_data(), clip_number)

    def _model_still_current(self, generation: int) -> bool:
        """Whether the review model is still the one `generation` was captured against.
        Menus and modal confirms run nested event loops, so a background Reread run can
        finish and rebuild the review while one is open: an action carrying row indices
        from the old model must be dropped, not applied to whatever row now holds that
        number. Explains the ignored click in the status line so it doesn't read as a
        dead button."""
        if generation == self._model_generation:
            return True
        self.show_status("The window was reloaded, try that action again")
        return False

    def _drop_if_reloaded(self, generation: int,
                          action: Callable[[], None]) -> Callable[[bool], None]:
        """A triggered-slot for `action` that drops the click when the model was reloaded
        after `generation` was captured (see _model_still_current): for menu actions
        whose meaning was fixed when their menu opened, not when the click lands."""
        def run(_checked: bool = False) -> None:
            if self._model_still_current(generation):
                action()
        return run

    def _on_tree_menu(self, pos: QPoint) -> None:
        """Right-click menu on the tree: exclude the whole set/match from export (any row, mirroring
        the Exclude button) and (on any game of a multi-game set) exclude just that game (split it
        into its own excluded clip). Reachable from wherever the cursor is, so a game right-click
        still offers the set exclusion below. Non-excluded rows also offer exporting just that
        set/match to a file of the user's choosing."""
        item = self._tree.itemAt(pos)
        if item is None:
            return
        clip_idx = item.data(0, _CLIP_ROLE)
        if clip_idx is None:
            return
        clip = self._session.clips[clip_idx]
        menu = QMenu(self._tree)
        menu.setToolTipsVisible(True)

        # menu.exec below spins a nested event loop, and the row indices baked into the
        # actions only mean anything for the model the menu was built from: guard every
        # action against the model being reloaded while the menu is open.
        generation = self._model_generation

        def guarded(action: Callable[[], None]) -> Callable[[bool], None]:
            return self._drop_if_reloaded(generation, action)

        # On a game row, the game-specific action leads (it's why you'd right-click that row); the
        # whole-set/match exclusion follows below a separator.
        choice = self._game_isolate_choice(item)
        if choice is not None:
            ci, lo = choice
            g_action = _nn(menu.addAction("Exclude this game from export"))
            g_action.triggered.connect(
                guarded(lambda: self._isolate_and_exclude_game(ci, lo)))
            menu.addSeparator()

        unit = self._session.unit_label().lower()
        s_action = _nn(menu.addAction(f"Exclude {unit} from export"))
        s_action.setCheckable(True)
        s_action.setChecked(bool(clip.get("exclude_from_export")))
        s_action.triggered.connect(
            guarded(lambda: self._toggle_exclude(clip_idx=clip_idx)))

        # One-off export of just this clip, with the current edits: on a game row of a set
        # this still exports the whole containing set (games aren't standalone clips).
        clip_number = self._export_one_choice(item)
        if clip_number is not None:
            menu.addSeparator()
            e_action = _nn(menu.addAction(f"Export this {unit} to a file…"))
            e_action.triggered.connect(
                guarded(lambda: self.exportOneRequested.emit(
                    self.build_reviewed_data(), clip_number)))

        menu.exec(_nn(self._tree.viewport()).mapToGlobal(pos))

    def _update_bounds_label(self) -> None:
        if self._current is None:
            return
        clip = self._session.clips[self._current]
        dur = max(0.0, clip["end_sec"] - clip["start_sec"])
        self._bounds_label.setText(
            f"Start: {_hms(clip['start_sec'])}   End: {_hms(clip['end_sec'])}   ({_hms(dur)})"
        )

    # ── Set restructuring (merge / split / move) ───────────────────────────────
    def _match_item(self, match_number: int) -> QTreeWidgetItem | None:
        """The child row for a given match (by match_number), or None if not present."""
        for i, clip in enumerate(self._session.clips):
            for j, m in enumerate(clip["_matches"]):
                if m["match_number"] == match_number:
                    return _nn(self._tree.topLevelItem(i)).child(j)
        return None

    def _select_match(self, match_number: int) -> bool:
        """Select the child row of a given match (by match_number); False if not present."""
        item = self._match_item(match_number)
        if item is None:
            return False
        self._select_item(item)
        return True

    def _selection(self) -> tuple[int | None, int | None]:
        """(clip index, child-match local index or None) for the current tree row."""
        item = self._tree.currentItem()
        if item is None:
            return None, None
        return item.data(0, _CLIP_ROLE), item.data(0, _MATCH_LOCAL_ROLE)

    def _current_match_number(self) -> int | None:
        """The match number of the highlighted child row, or None if a set row is selected."""
        clip, local = self._selection()
        if clip is None or local is None:
            return None
        return self._session.clips[clip]["_matches"][local]["match_number"]

    def _merge_prev(self) -> None:
        clip, _ = self._selection()
        if clip is not None and clip > 0:
            self._apply_result(self._session.merge(
                clip - 1, focus_match=self._current_match_number(),
                selection=self._current_selection()))

    def _merge_next(self) -> None:
        clip, _ = self._selection()
        if clip is not None and clip < len(self._session.clips) - 1:
            self._apply_result(self._session.merge(
                clip, focus_match=self._current_match_number(),
                selection=self._current_selection()))

    def _split_here(self) -> None:
        """Split the selected clip into two sets at the selected match (see
        ReviewSession.split_set)."""
        clip, local = self._selection()
        if clip is None or local is None:
            return
        self._apply_result(self._session.split_set(clip, local, self._current_selection()))

    def _confirm_single_game_clip(self, message: str) -> bool:
        """Modal confirm shown when an exclusion would leave a single-game clip (not a full set):
        when isolating a game (first, middle, or last) would leave a surviving side that is a lone
        game. Returns True to proceed. Split out (and message-driven) so it's reused and tests can
        stub it without a real dialog."""
        return QMessageBox.question(
            self, "Exclude game", message) == QMessageBox.StandardButton.Yes

    def _isolate_and_exclude_game(self, clip: int, local: int) -> None:
        """Exclude a single game, first, middle, or last, by splitting it into its own one-game set
        flagged out of the export, with the games before/after it forming their own sets (see
        ReviewSession.isolate_and_exclude_game). If a surviving side would be a single-game clip
        (not a full set), the user is asked to confirm first."""
        if self._session.isolate_creates_single_game_clip(clip, local):
            match_no = self._session.match_number_at(clip, local)
            generation = self._model_generation
            if not self._confirm_single_game_clip(
                    f"Excluding game {match_no} would make a set with just one game. Exclude it anyway?"):
                return
            # The confirm is modal: the model can be reloaded underneath it (a Reread
            # finishing), leaving clip/local pointing into the wrong grouping.
            if not self._model_still_current(generation):
                return
        self._apply_result(
            self._session.isolate_and_exclude_game(clip, local, self._current_selection()))

    def _rebuild_player_menu(self, player: int) -> None:
        """Refill one player's Options menu: every propagation action for that player, each
        enabled per its own candidacy at open time. Entries are player-scoped: the button
        lives under the player's own column, so they don't repeat the name."""
        menu = self._detail.player_menus[player]
        menu.clear()
        # The open popup spins a nested event loop, and the actions propagate FROM the
        # set the menu was opened on (self._current at open time): if a Reread rebuilds
        # the model underneath, _current snaps to the new model's first row and a click
        # would propagate the wrong set. Guard like the tree menu (_on_tree_menu).
        generation = self._model_generation
        tag_action = _nn(menu.addAction(
            "Copy tag/sponsor to their other sets",
            self._drop_if_reloaded(generation, lambda: self._propagate_tags(player))))
        tag_action.setEnabled(bool(self._session.propagation_targets(self._current, player)))
        menu.addSeparator()
        fill_action = _nn(menu.addAction(
            "Fill blank characters",
            self._drop_if_reloaded(generation,
                                   lambda: self._propagate_characters(False, player))))
        fill_action.setToolTip(
            "Copy this player's character to blank Character slots in their other sets")
        fill_action.setEnabled(
            bool(self._session.character_targets(self._current, False, player)))
        over_action = _nn(menu.addAction(
            "Overwrite all characters",
            self._drop_if_reloaded(generation,
                                   lambda: self._propagate_characters(True, player))))
        over_action.setToolTip(
            "Replace this player's character in all their sets.")
        over_action.setEnabled(
            bool(self._session.character_targets(self._current, True, player)))

    def _propagate_tags(self, player: int | None = None) -> None:
        """Apply the current set's tag/sponsor corrections (one player's, via `player`, or both)
        to every other set with the same player (see ReviewSession.propagate_tags). Tree labels
        refresh in place so the selection stays; the current clip is the source of the
        propagation, never a target, so its own panel never needs reloading."""
        if self._current is None:
            return  # Guard only: the menu entries grey out individually when they have no targets.
        self._apply_result(
            self._session.propagate_tags(self._current, player, self._current_selection()))

    def _propagate_characters(self, overwrite: bool = False,
                              player: int | None = None) -> None:
        """Apply the current set's characters (one player's, via `player`, or both) to every
        other set with the same player: only blank slots unless `overwrite` (see
        ReviewSession.propagate_characters). Tree labels refresh in place; the current clip is
        the source, never a target, so its own panel never needs reloading."""
        if self._current is None:
            return  # Guard only: the menu entries grey out individually when they have no targets.
        self._apply_result(self._session.propagate_characters(
            self._current, overwrite, player, self._current_selection()))

    def fix_capitalization(self) -> None:
        """Host Tools-menu batch op: re-case every single-case player tag across all clips,
        all-caps (`PLAYERONE`) and all-lowercase (`playertwo`) overlays, leaving intentional mixed-case
        styling (`xPLAYERTHREE`) alone (see tag_fixes.apply_capitalization). Opens the options dialog first
        (Title / Sentence / lowercase mode + a live preview with per-row Exclude and hand-editable
        After cells; Apply doubles as the confirm, so there's no separate confirmation popup); a
        no-op run is skipped (see ReviewSession.fix_capitalization for the undoable step itself).
        Tree labels refresh in place and, if the current clip changed, its Tag fields reload so
        the panel stays in sync."""
        # The dialog is modal: the model can be reloaded underneath it (a Reread finishing),
        # so re-check before mutating; same guard as the other modal-then-mutate handlers.
        generation = self._model_generation
        dialog = FixCapsOptionsDialog(self._settings, self._session.clips, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if not self._model_still_current(generation):
            return
        mode = dialog.values()
        excluded_tags = dialog.excluded_tags()
        overrides = dialog.tag_overrides()
        if not self._session.capitalization_changes(mode=mode, excluded_tags=excluded_tags,
                                                    overrides=overrides):
            self._status.setText("No capitalization to fix!")
            return
        result = self._session.fix_capitalization(
            self._current_selection(), mode=mode, excluded_tags=excluded_tags,
            overrides=overrides)
        self._apply_result(result)
        # The current clip's tags may have just changed under the Tag fields; reload the panel
        # from the clip (guarded by _loading so the writeback doesn't fire and re-snapshot).
        if self._current is not None and self._current in result.changed_indices:
            self._loading = True
            self._detail.set_clip_fields(self._session.clips[self._current])
            self._loading = False

    def split_sponsors_from_tags(self) -> None:
        """Host Tools-menu batch op: across all clips, move a colour-merged sponsor off the
        front of each player tag into the sponsor field; some overlays colour the sponsor apart
        from the gamertag, which OCR can't see, so "ORG Player" lands as one tag (see
        tag_fixes.split_leading_sponsor). Slots that already carry a sponsor are append-only,
        a bigger max-words re-run extends them, never rewrites them (see tag_fixes.sponsor_splits).
        Opens the options dialog first (length bounds, max words, all-caps mode, plus a live
        preview with per-row Exclude and hand-editable Sponsor/Tag cells; Apply doubles as the
        confirm, so there's no separate confirmation popup); a no-op run is skipped (see
        ReviewSession.split_sponsors_from_tags for the undoable step itself). Tree labels refresh
        and, if the current clip changed, its Tag/Sponsor fields reload so the panel stays in
        sync."""
        # The dialog is modal: the model can be reloaded underneath it (a Reread finishing),
        # so re-check before mutating; same guard as the other modal-then-mutate handlers.
        generation = self._model_generation
        dialog = SponsorSplitOptionsDialog(self._settings, self._session.clips, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if not self._model_still_current(generation):
            return
        min_len, max_len, max_words, all_caps = dialog.values()
        excluded_tags = dialog.excluded_tags()
        overrides = dialog.overrides()
        if not self._session.sponsor_split_changes(
                min_len=min_len, max_len=max_len, max_words=max_words, all_caps=all_caps,
                excluded_tags=excluded_tags, overrides=overrides):
            self._status.setText("No matching sponsor acronyms to split")
            return
        result = self._session.split_sponsors_from_tags(
            self._current_selection(), min_len=min_len, max_len=max_len,
            max_words=max_words, all_caps=all_caps, excluded_tags=excluded_tags,
            overrides=overrides)
        self._apply_result(result)
        # The current clip's tag/sponsor may have just changed under the fields; reload the panel
        # from the clip (guarded by _loading so the writeback doesn't fire and re-snapshot).
        if self._current is not None and self._current in result.changed_indices:
            self._loading = True
            self._detail.set_clip_fields(self._session.clips[self._current])
            self._loading = False

    def _split_match(self) -> None:
        """Split at the playhead, acting on wherever the playhead actually is (see
        ReviewSession.split_match). The scrubber view is frozen across a successful rebuild
        (both the in-match cut and the gap-fill paths), exactly as it was before the op moved
        into the session, so the newly split match doesn't cause a visible jump before the
        selection lands on it; a rejection needs no freeze."""
        clip, _ = self._selection()
        if clip is None:
            return
        result = self._session.split_match(
            clip, self._player.playhead_sec(), self._current_selection())
        if isinstance(result, OpRejected):
            self._apply_result(result)
            return
        self._player.freeze_view(True)
        try:
            self._apply_result(result)
        finally:
            self._player.freeze_view(False)

    def _move_block(self) -> tuple[int, list[int]] | None:
        """The selected run of match rows as (clip index, sorted local indices), or None
        unless every selected row is a child match in the *same* set with *contiguous*
        local indices. A single selected match yields a one-element block."""
        items = self._tree.selectedItems()
        if not items:
            return None
        clip = items[0].data(0, _CLIP_ROLE)
        local_indices = []
        for it in items:
            local = it.data(0, _MATCH_LOCAL_ROLE)
            if local is None or it.data(0, _CLIP_ROLE) != clip:
                return None  # a set row, or rows spanning sets: not a movable block
            local_indices.append(local)
        local_indices.sort()
        if local_indices != list(range(local_indices[0], local_indices[0] + len(local_indices))):
            return None  # gap in the selection
        return clip, local_indices

    def _move_prev(self) -> None:
        block = self._move_block()
        if block is None:
            return
        clip, local_indices = block
        if clip > 0 and local_indices[0] == 0:
            self._apply_result(self._session.move_block(
                clip, local_indices, clip - 1, self._current_selection()))

    def _move_next(self) -> None:
        block = self._move_block()
        if block is None:
            return
        clip, local_indices = block
        if clip < len(self._session.clips) - 1 and local_indices[-1] == len(self._session.clips[clip]["_matches"]) - 1:
            self._apply_result(self._session.move_block(
                clip, local_indices, clip + 1, self._current_selection()))

    def _selection_excluded(self) -> bool:
        """Whether the Exclude button should read as pressed for the current row: whether the selected
        clip is excluded. A game in a non-excluded set is never individually excluded now (excluding it
        isolates it into its own separate set), so only the whole-set flag matters: any row of an
        excluded set (including the now-isolated game's own set) reads as pressed."""
        clip, _ = self._selection()
        return clip is not None and self._session.clips[clip].get("exclude_from_export", False)

    def _update_op_buttons(self, item: QTreeWidgetItem | None = None) -> None:
        clip, local = self._selection()
        # Trim, Exclude and Swap apply whenever a clip is selected, in any mode. Exclude additionally
        # needs a single highlighted row: it acts on one set/game (a middle-game exclude even
        # restructures), so a multi-row selection has no unambiguous target; grey it out then.
        single_row = len(self._tree.selectedItems()) <= 1
        self._trim["set_start"].setEnabled(clip is not None)
        self._trim["set_end"].setEnabled(clip is not None)
        self._detail.swap_button.setEnabled(clip is not None)
        self._exclude_btn.setEnabled(clip is not None and single_row)
        self._exclude_btn.setChecked(single_row and self._selection_excluded())
        if self._session.meta.get("mode") == "match":
            self._sync_menu_actions()
            return
        n = len(self._session.clips)
        is_child = local is not None
        match_count = len(self._session.clips[clip]["_matches"]) if (clip is not None and is_child) else 0
        self._ops["merge_prev"].setEnabled(clip is not None and clip > 0)
        self._ops["merge_next"].setEnabled(clip is not None and clip < n - 1)
        # Splittable from any match in a multi-match set: the first splits after itself, later
        # matches before themselves. A lone match has nothing to split off.
        self._ops["split"].setEnabled(is_child and match_count > 1)
        # Each player's Options button lights when that player has anything to propagate.
        # Overwrite character targets are a superset of fill targets (a blank also
        # "differs"); the menu entries grey out individually when the menu opens.
        for i, key in enumerate(("p1_options", "p2_options")):
            self._ops[key].setEnabled(
                bool(self._session.propagation_targets(self._current, i))
                or bool(self._session.character_targets(self._current, True, i)))
        self._ops["split_match"].setEnabled(is_child)
        # Move acts on the selected run: enabled when it's a contiguous block touching the
        # set edge nearest the move direction, with a neighbor set to land in.
        block = self._move_block()
        if block is None:
            self._ops["move_prev"].setEnabled(False)
            self._ops["move_next"].setEnabled(False)
        else:
            b_clip, b_locals = block
            last = len(self._session.clips[b_clip]["_matches"]) - 1
            self._ops["move_prev"].setEnabled(b_clip > 0 and b_locals[0] == 0)
            self._ops["move_next"].setEnabled(b_clip < n - 1 and b_locals[-1] == last)
        self._sync_menu_actions()

    # ── Undo / reset ────────────────────────────────────────────────────────────
    def _current_selection(self) -> Selection:
        """The acting tree selection, resolved for the session: the current clip index and,
        in set mode, the highlighted child match's number."""
        return Selection(self._current, self._current_match_number())

    def _apply_result(self, result: OpResult | OpRejected | None) -> None:
        """Apply a session op's outcome: status line, tree repopulate or in-place
        relabel/restyle, undo-button refresh, draft save, selection restore."""
        if result is None:
            return
        if isinstance(result, OpRejected):
            if result.warn:
                QMessageBox.warning(self, "Game too short", result.message)
            else:
                self._status.setText(result.message)
            return
        if result.rebuilt:
            # A rebuild can shrink the clip list, leaving self._current pointing past the end.
            # setCurrentItem (in the selection restore below) emits itemSelectionChanged before
            # _on_select refreshes _current, so _update_op_buttons would index the old position
            # and raise, which PyQt escalates to a fatal abort. Clear it; the tree has no
            # selection until _select_match/_select_item restores it.
            self._current = None
            self._populate_tree()
        else:
            for idx in result.changed_indices:
                item = _nn(self._tree.topLevelItem(idx))
                item.setText(0, self._clip_label(self._session.unit_label(),
                                                 self._session.clips[idx]))
                self._style_clip_row(item, self._session.clips[idx])
        if result.status:
            self._status.setText(result.status)
        self._refresh_undo_buttons()
        self._schedule_save()
        # Prefer the exact child match row the user was on (this also seeks the playhead back
        # to that game, via _on_select); fall back to the set/clip row when there was no match
        # selected or it's gone (e.g. match mode, or the match no longer exists).
        if result.rebuilt:
            if ((result.focus_match is None or not self._select_match(result.focus_match))
                    and result.select is not None):
                self._select_item(self._tree.topLevelItem(result.select))
            if result.extra_select_matches:
                # Move-block multi-select restore: the anchor is already current (via
                # focus_match above); extend the highlight to the rest of the moved block.
                self._tree.blockSignals(True)
                try:
                    for n in result.extra_select_matches:
                        sibling = self._match_item(n)
                        if sibling is not None:
                            sibling.setSelected(True)
                finally:
                    self._tree.blockSignals(False)
        elif result.select is not None:
            self._select_item(self._tree.topLevelItem(result.select))
        self._update_op_buttons()

    def _undo(self) -> None:
        self._apply_result(self._session.undo(self._current_selection()))

    def _redo(self) -> None:
        self._apply_result(self._session.redo(self._current_selection()))

    def _reset_all(self) -> None:
        """Discard every edit made since load: grouping (merges/splits/moves), trims, and
        tag/character/round changes. A screen loaded fresh from an extract restores that
        extract (and removes the now-stale draft). A resumed draft offers a choice: rebuild
        from the original matches.json, or just fall back to the state the draft was resumed
        with (kept on disk, since it still holds edits worth resuming again)."""
        if not self.has_unsaved_edits():
            return
        # Both confirms below are modal: the model can be reloaded underneath them (a
        # Reread finishing), making the dirty/resumed premise the dialog asked about
        # stale; re-check after each before acting (see _model_still_current).
        generation = self._model_generation
        if not self._session.resumed:
            if QMessageBox.question(
                    self, "Reset all edits",
                    "Discard ALL edits and restore the original analysis?"
            ) != QMessageBox.StandardButton.Yes:
                return
            if not self._model_still_current(generation):
                return
            self._apply_result(self._session.reset_all())
            self._flush_draft()  # clean again → removes the saved draft immediately
            return

        box = QMessageBox(self)
        box.setWindowTitle("Reset all edits")
        box.setText("Discard ALL edits and restore the original analysis?")
        box.setInformativeText("These edits were resumed from a saved draft. Reset all the way "
                               "to the original analysis, or back to the resumed draft?")
        extract_btn = _nn(box.addButton("Original analysis", QMessageBox.ButtonRole.DestructiveRole))
        draft_btn = box.addButton("Resumed draft", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        matches = self._read_matches()
        if matches is None:
            extract_btn.setEnabled(False)
            # This should only happen if the user deleted matches.json by hand.
            extract_btn.setToolTip("matches.json is missing! Only the resumed draft "
                                   "is available.")
        box.exec()
        if not self._model_still_current(generation):
            return
        if box.clickedButton() is extract_btn and matches is not None:
            self._reset_to_extract(matches)
        elif box.clickedButton() is draft_btn:
            self._reset_to_resumed()

    def _reset_to_resumed(self) -> None:
        """Fall back to the state the draft was resumed with (see ReviewSession.reset_to_resumed;
        the draft stays on disk since the restored state IS its content)."""
        self._apply_result(self._session.reset_to_resumed())

    def _reset_to_extract(self, matches: list[Match]) -> None:
        """Rebuild from the original matches.json (see ReviewSession.reset_to_extract) and
        remove the now-stale draft, since the screen matches the extract again."""
        self._apply_result(self._session.reset_to_extract(matches))
        self._flush_draft()  # clean again → removes the saved draft immediately

    def _read_matches(self) -> list[Match] | None:
        """The original extract (matches.json, next to the draft), or None if unreadable,
        resuming works without it, so a draft-only folder simply can't offer the deeper
        reset."""
        try:
            data = json.loads(
                self._draft_path().with_name("matches.json").read_text(encoding="utf-8"),
            )
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, list) else None

    def _refresh_undo_buttons(self) -> None:
        self._undo_btn.setEnabled(self._session.can_undo())
        self._redo_btn.setEnabled(self._session.can_redo())
        self._reset_btn.setEnabled(self._session.is_dirty())
        self._sync_menu_actions()

    # ── Export ──────────────────────────────────────────────────────────────--
    def build_reviewed_data(self) -> ReviewedData:
        """The reviewed.json projection of the current edits: excluded sets dropped, each surviving
        set reduced to schema.Clip's fields. Emitted to vodparse.gui.app on Export/Chapters, and called
        directly by the headless vod_grabber pipeline. The projection itself is pure; see
        model.build_reviewed_data."""
        return self._session.reviewed_data()

    def _on_export(self) -> None:
        self.exportRequested.emit(self.build_reviewed_data())

    def _on_review_and_export(self) -> None:
        """Open the Final Review dialog for a whole-slate pass before exporting. The dialog
        mutates the shared session in place (edits, swaps, undo/redo all go straight through
        session ops), so once it closes the tree and undo buttons are resynced to match, and
        the edits are queued for the auto-save draft same as any other change. If the user hit
        Export inside the dialog, run the normal export path; its inline progress bar shows
        here on the Review screen, behind the now-closed dialog."""
        prev_row = self._current
        dlg = FinalReviewDialog(self._session, self._settings, self)
        dlg.exec()
        # The dialog edited the session in place, and its Undo/Redo can rebuild the whole clip
        # list, so self._current and the detail panel are now stale. Clear the index before
        # repopulating: a stale one indexing self._session.clips in a later Qt slot is a fatal
        # abort (same hazard _apply_result guards against), then reselect the row the user was
        # on if it survived, so the panel reloads from the (possibly mutated) clip instead of
        # keeping pre-dialog text that the next field edit would write back over the changes.
        self._current = None
        self._populate_tree()
        if prev_row is not None and prev_row < len(self._session.clips):
            self._select_item(self._tree.topLevelItem(prev_row))
        self._refresh_undo_buttons()
        # Reselecting above refreshes the op buttons via _on_select; when nothing is reselected
        # (no prior row, or it was dropped by the dialog's Undo) they'd otherwise keep their
        # pre-dialog enabled/checked state over a tree with no selection, so refresh explicitly
        # here too, mirroring _apply_result's rebuild branch.
        self._update_op_buttons()
        self._schedule_save()
        if dlg.wants_export:
            self._on_export()

    def _on_chapters(self) -> None:
        self.chaptersRequested.emit(self.build_reviewed_data())

    def set_exporting(self, active: bool, total: int = 0) -> None:
        """Grey out Export clips, relabel it "Exporting…", and show the progress bar with its
        clips-saved caption while the export subprocess runs; the GUI flips this around the
        export's start and finish (including the failure path).

        `total` is the number of clips the export will actually write: the non-excluded
        count the GUI takes from the reviewed payload (``len(data["clips"])``). It must NOT
        be recomputed from ``self._session.clips`` here: excluded sets are dropped by
        build_reviewed_data and never reach the exporter, so that count overshoots the
        "  Saved:" lines the subprocess emits and the bar would never fill (finding S2)."""
        self._export_btn.setEnabled(not active)
        # Disabled in lockstep with Export clips: opening the Final Review dialog mid-export
        # would let a second export start (its own Export button) while this one still runs.
        self._review_export_btn.setEnabled(not active)
        if active:
            self._export_btn.setText("Exporting…")
            self._export_progress.setRange(0, max(1, total))
            self._export_progress.setValue(0)
            if self._session.meta.get("mode") != "match":
                self._export_unit = "set" if total == 1 else "sets"
            else:
                self._export_unit = "game" if total == 1 else "games"
            self._export_label.setText(export_status_text(0, total, self._export_unit))
            self._export_label.setVisible(True)
            self._export_progress.setVisible(True)
        else:
            self._export_btn.setText("Export clips")
            self._export_label.setVisible(False)
            self._export_progress.setVisible(False)

    def set_export_progress(self, done: int) -> None:
        """Advance the export bar and its caption to `done` clips saved (clamped to the
        total); the GUI calls this as it counts "Saved:" lines streaming from the export
        subprocess."""
        total = self._export_progress.maximum()
        done  = min(done, total)
        self._export_progress.setValue(done)
        self._export_label.setText(export_status_text(done, total, self._export_unit))

    def set_upload_available(self, available: bool) -> None:
        """Enable the Open Upload Assistant button once this VOD's clips have been exported."""
        self._assistant_btn.setEnabled(available)

    def set_export_folder_available(self, available: bool) -> None:
        """Enable the Open export folder button once this VOD's export folder exists."""
        self._open_folder_btn.setEnabled(available)

    def export_target(self) -> tuple[str, str, bool] | None:
        """(tournament_name, stem, is_set) naming this review's export folder/CSV, or None if no
        model is loaded yet. Mirrors how MainWindow derives the export basename."""
        if not self._session.meta or "stem" not in self._session.meta or "mode" not in self._session.meta:
            return None
        return (self._session.meta.get("tournament_name", ""), self._session.meta["stem"],
                self._session.meta["mode"] != "match")

    def unit_label(self) -> str:
        """The loaded review's word for one top-level row: "Set" or "Game" (see
        ReviewSession.unit_label). Only meaningful once export_target() is non-None:
        with no model loaded it defaults to "Set"."""
        return self._session.unit_label()

    def tournament_name(self) -> str:
        """The tournament name this review will export under ("" until a model is loaded)."""
        return self._session.meta.get("tournament_name", "")

    def set_tournament_name(self, name: str) -> None:
        """Change the tournament name mid-review (Tools → Change tournament name…). The session
        meta is the single source downstream: export clip titles, the CSV "tournament" column,
        and the export folder/CSV basename all read it (build_reviewed_data / export_target).
        The rename marks the session dirty (see ReviewSession.set_tournament_name), so it
        persists in the draft and survives an app restart via Resume edits."""
        if not self._session.meta:
            return  # nothing loaded yet; the Tools surface is Review-only so this is belt+braces
        self._session.set_tournament_name(name)
        self._schedule_save()

    # ── Draft auto-save / resume ────────────────────────────────────────────────
    def _draft_path(self) -> Path:
        return draft_path(self._session.meta["stem"])

    def _schedule_save(self) -> None:
        """Queue a debounced draft write after an edit. No-op until a model is loaded; the timer
        coalesces bursts (e.g. per-character field edits) into a single write."""
        if self._loading or not self._session.meta:
            return
        self._save_timer.start()

    def _flush_draft(self) -> None:
        """Write any pending (debounced) draft immediately: used before navigating away."""
        if self._save_timer.isActive():
            self._save_timer.stop()
            self._write_draft()

    def _write_draft(self) -> None:
        """Persist the current edits, or remove the draft if the screen is back to its unedited
        baseline (so a now-clean screen never leaves a stale draft to resume)."""
        if not self._session.meta:
            return
        if not self.has_unsaved_edits():
            self.clear_draft()
            return
        try:
            write_draft_file(self._draft_path(), self._session.draft_payload())
        except OSError as e:
            self._status.setText(f"Couldn't save draft: {e}")

    def _read_draft(self) -> DraftPayload | None:
        return read_draft(self._draft_path())

    def clear_draft(self) -> None:
        if self._session.meta:
            # Cancel any pending debounced save first, or it would fire after the unlink
            # and resurrect the draft moments after it was cleared.
            self._save_timer.stop()
            clear_draft_file(self._draft_path())

    def load_from_draft(self, video_path: str, stem: str,
                        set_characters: str = "join") -> bool:
        """Reopen the saved edit draft for `stem` (the Resume edits button), rebuilding the editing
        state straight from the draft without an extract. The draft carries its own mode and
        tournament name; `video_path` is the current on-disk location (used in case the file moved).
        `set_characters` only matters if Reset later rebuilds from matches.json (the draft's clips
        already have their characters collapsed). Returns False if no draft is present."""
        self._model_generation += 1  # invalidate menus/confirms opened on the old model
        # Enough meta for _draft_path to locate the draft, and to keep clear_draft working
        # if the read fails and no full session ever loads.
        self._session = ReviewSession(meta={"stem": stem}, clips=[], synth_counter=0,
                                      set_characters=set_characters, resumed=False)
        draft = self._read_draft()
        if not draft:
            return False
        self._session = ReviewSession.from_draft(draft, video_path, stem, set_characters)
        self._adopt_clock_length(video_path)
        self._apply_mode_visibility(self._session.meta["mode"])
        self._player.set_source(self._session.meta["video_path"])
        self._populate_tree()
        self._refresh_undo_buttons()
        if self._session.clips:
            self._select_item(self._tree.topLevelItem(0))
        return True

    # ── Back (auto-saved; no discard prompt) ────────────────────────────────────
    def has_unsaved_edits(self) -> bool:
        """Any review edits since load (see ReviewSession.is_dirty)."""
        return self._session.is_dirty()

    def _on_back(self) -> None:
        # Edits are auto-saved to the draft, so going back loses nothing; flush any pending save
        # (an edit made within the debounce window) and leave without a warning.
        self._flush_draft()
        self._player.stop()        # don't leave a clip playing behind the analysis page
        self._status.setText("")   # don't carry this session's status line back into the panel
        self.backRequested.emit()
