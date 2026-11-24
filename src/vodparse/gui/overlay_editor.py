"""
Custom overlay layout editor
------------------------------
The main window's layout editor popup ("Edit…" in the Overlay layout row of the Game box):
define a named custom overlay layout by drawing boxes directly on a frame of a VOD. The
user scrubs to a frame where the stream overlay is visible (the Review screen's
``TrimPlayer`` reused as the scrubbing surface),
then arms one region at a time (player 1 tag, player 2 tag, and optionally round /
best-of / the match clock) and click-drags its box over the captured still. The frame
under the playhead is captured by itself whenever the playhead settles (the scrubber let
go or playback paused, after ``SETTLE_DELAY_MS``, as on the Analyze page's preview), and
also on picking a saved preset (so its boxes show at once) or pressing Draw over the live
video; there is no capture button to press. Boxes
are stored as fractions of the frame size (``vodparse.overlay_layout.NormRect``), so a
preset drawn on a 1080p VOD applies unchanged to a 720p capture of the same stream.

The captured still is shown on a ``RegionDrawPane`` (``vodparse.gui.region_pane``: an
``ImagePane`` extended with a drag-to-draw mode) installed as the player's alternate
display page; any scrub or play auto-dismisses it back to the live video, exactly like
the overlay spot-check tool's video mode. The frame itself is grabbed off the UI thread
by ``vodparse.gui.frame_grabber.FrameGrabber`` (an ffmpeg process). Both helpers are
shared with the Analyze page's layout preview (``vodparse.gui.layout_preview``), which
shows a saved layout's boxes the same way without editing them. Drawn boxes live in
dialog state as normalized rects, so they survive the dismissal and re-appear on the next
capture — including a capture from a different VOD. A drawn box can be adjusted on the
still: click it, then drag it to move it, drag a corner to resize it, or nudge it with the
arrow keys (the editor's pane is built ``adjustable``; the Analyze preview's is not).

Presets are saved to the shared JSON store (``vodparse.overlay_layout``); closing the
dialog with unsaved box, band or clock edits offers to save them first (Save / Discard /
Cancel; see ``_has_unsaved_changes`` for what counts). **Save and continue** saves and
closes in one click, and the main screen then selects that layout. Escape does not close
the dialog; it cancels an armed Draw instead (see ``keyPressEvent``). The drawn boxes are
read by themselves, like the Analyze page's preview reads its frame: shortly after every
capture and every box drawn, moved, resized, nudged or cleared (``READ_DELAY_MS``, so a run
of nudges ends in one read), each box is OCR'd — per-box crop + read, exactly the semantics
the pipeline will apply — using the OCR engine/language from Settings, so the readings match
a real run. The OCR itself runs in a helper *process* (``vodparse.cli.overlay_read``, kept
warm between reads), never in this process: the GUI is torch-free by design, and
running torch on an in-process Qt thread fatally crashed the interpreter on macOS
(``_PyThreadState_Attach``: torch + PyQt threads + Python 3.13). Every
capture/draw/preset action bumps a generation token that requests echo back, so
results for a superseded frame are dropped (the spot-check tool's token scheme).

Import this only after ``qt_plugins.configure_qt_plugins()`` has run
(``MainWindow._on_edit_layouts`` imports it lazily, from inside an already-running app).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QSettings, Qt, QTime, QTimer
from PyQt6.QtGui import QCloseEvent, QKeyEvent, QStandardItemModel
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFrame,
                             QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QMessageBox, QPushButton, QScrollArea, QTimeEdit, QVBoxLayout,
                             QWidget)

# Import-light by contract (see its module docstring): only the protocol constant, so
# the seam between this client and the helper can't drift.
from vodparse.ffmpeg_runtime import offers_conversion
from vodparse.frame_reader import VideoFrameSource
from vodparse.games import (GAME_DEFAULT, GAMES, INGAME_BAND_CHARACTER,
                            INGAME_BAND_NONE, INGAME_BAND_TAGS, Game, game_by_id,
                            ingame_band_engine_for, resolve_saved_game)
from vodparse.gui import crash_log
from vodparse.gui.common import VIDEO_FILTER, ElidingLabel
from vodparse.gui.convert import Mp4Converter
from vodparse.gui.frame_grabber import SETTLE_DELAY_MS, FrameGrabber
from vodparse.gui.image_pane import make_zoom_buttons
from vodparse.gui.overlay_read_client import (OverlayReadProcess, RegionReading,
                                              tag_display_text)
from vodparse.gui.player.trim_player import TrimPlayer
from vodparse.gui.region_pane import (NUDGE_BIG_PX, REGION_COLORS, REGION_LABELS,
                                     RegionDrawPane)
from vodparse.gui.review.model import hms
from vodparse.gui.settings import (ASK_CONVERT_MKV_DEFAULT, OCR_ENGINE_CHOICES,
                                   OCR_ENGINE_DEFAULT_GUI, OCR_LANG_DEFAULT_GUI,
                                   SETTING_ASK_CONVERT_MKV, SETTING_OCR_ENGINE,
                                   SETTING_OCR_LANG, resolve_saved_engine)
from vodparse.ocr_engine import OCR_ENGINE_DOCTR, doctr_available, doctr_weights_cached
from vodparse.overlay_layout import (DEFAULT_CLOCK_SECONDS, AUTO_LAYOUT_NAME,
                                     INGAME_REGION_KEYS, REGION_BEST_OF, REGION_INGAME_P1,
                                     REGION_INGAME_P2, REGION_KEYS, REGION_PLAYER1,
                                     REGION_PLAYER2, REGION_ROUND, REGION_TIMER,
                                     CustomOverlayLayout, NormRect, layout_by_name,
                                     layouts_for_game, layouts_path, load_layouts,
                                     save_layouts, with_layout_saved)

# ── Region presentation ──────────────────────────────────────────────────────────

# REGION_LABELS (display labels for the drawable regions) and REGION_COLORS (their box
# colors) live in vodparse.gui.region_pane now: RegionDrawPane needs both to draw a box,
# and this dialog reuses them for its region rows' swatches/names/tooltips.

# The two nameplate rows say what a drawn box would be read as, following the Bottom of
# screen combo: the players' in-game tags, the characters' names, or nothing (rows
# disabled). Keyed by mode, then by region key. REGION_LABELS keeps the mode-neutral text
# for the readings and tooltips.
INGAME_ROW_LABELS: dict[str, dict[str, str]] = {
    INGAME_BAND_TAGS: {REGION_INGAME_P1: "Ingame tag P1 (optional)",
                       REGION_INGAME_P2: "Ingame tag P2 (optional)"},
    INGAME_BAND_CHARACTER: {REGION_INGAME_P1: "Character name P1 (optional)",
                            REGION_INGAME_P2: "Character name P2 (optional)"},
    INGAME_BAND_NONE: {REGION_INGAME_P1: "Ingame P1 (bottom not read)",
                       REGION_INGAME_P2: "Ingame P2 (bottom not read)"},
}
# Display labels for the in-game band's three meanings, keyed by the shared mode
# constants (vodparse.games.INGAME_BAND_MODES). The band is the strip where the *game*
# itself draws its nameplates (Ultimate's player tags, Rivals 2's character names): it
# is read left and right by screen side even under a custom layout, so it is never one of
# the boxes drawn here. Its position belongs to the game (vodparse.games.IngameHud); only
# what the text in it means is chosen per preset, with the selected game supplying the
# default (vodparse.games.Game.ingame_band_default).
INGAME_BAND_LABELS: dict[str, str] = {
    INGAME_BAND_NONE: "None",
    INGAME_BAND_TAGS: "Ingame tags",
    INGAME_BAND_CHARACTER: "Character name",
}

# The timer row's label for a game with no clock detector (vodparse.games.Game.
# has_timer_model): there the drawn box is the run's only clock source, so a preset for
# such a game must carry one. Games with a detector keep the plain REGION_LABELS text.
TIMER_LABEL_REQUIRED = "Timer (required)"

# The preset combo's width, in average characters (see OverlayLayoutEditorDialog.__init__).
PRESET_COMBO_MIN_CHARS = 24
# The columns of the "Layout presets" group's grid: label | field | button.
PRESET_LABEL_COLUMN = 0
PRESET_FIELD_COLUMN = 1
PRESET_BUTTON_COLUMN = 2
PRESET_COLUMN_COUNT = 3


def _all_labels_for(key: str) -> list[str]:
    """Every text a region row's name label can show over the dialog's life: the plain
    label, the timer's "required" form, and the nameplate rows' per-mode forms."""
    labels = [REGION_LABELS[key]]
    if key == REGION_TIMER:
        labels.append(TIMER_LABEL_REQUIRED)
    for per_mode in INGAME_ROW_LABELS.values():
        if key in per_mode:
            labels.append(per_mode[key])
    return labels

# The preset combo's first entry: no preset selected, working on an unsaved layout.
NEW_LAYOUT_ITEM = "(new layout)"

# The button beside Close that saves the layout and closes the editor in one click.
SAVE_AND_CONTINUE_LABEL = "Save and continue"

# Result-line placeholder before a read (and after anything invalidates one).
NO_READING = "—"
# Width the test-read result labels reserve (in average characters): enough to show a typical
# tag or clock read in full before eliding kicks in (see ElidingLabel).
RESULT_MIN_CHARS = 24

# The still pane's placeholder before any frame is captured, and the status line shown
# while the live video is up. The frame is captured whenever the playhead settles (and
# Draw captures it on its own).
NO_FRAME_HINT = ("Choose a video, scrub to a frame where the overlay is visible,\n"
                 "then press the Draw button to draw regions for timers, tags, etc.")
LIVE_STATUS = "Scrubbing…"


def _selected_box_status(key: str) -> str:
    """The status line while a drawn box is selected: how to adjust it."""
    return (f"{REGION_LABELS[key]} box selected. Drag it to move it, drag a corner to resize "
            f"it, or nudge it with the arrow keys (Shift: {NUDGE_BIG_PX} px).")


# How long the boxes and frame must stay put before they are read: long enough that a run of
# arrow-key nudges (or a drag's last moves) ends in one read rather than one per change.
READ_DELAY_MS = 400
# The readings group's title, the Analyze page preview's wording for the same readout.
READINGS_TITLE = "Tags read on this frame"
# The readings group's own status line, kept apart from the dialog's status line so a read
# finishing never overwrites a hint the user is acting on (such as a selected box's): a read
# in progress, then its outcome, how many boxes read some text of those drawn.
READ_STARTED_STATUS = "Reading text… The first read can take a while."
READ_DONE_STATUS = "Read {read} of {drawn} boxes."
# The captured frame couldn't be written for the helper (see _report_frame_save_failure).
FRAME_SAVE_FAILED_STATUS = "Couldn't save the captured frame, try again."
FRAME_SAVE_FAILED_TITLE = "Couldn't save the captured frame"
FRAME_SAVE_FAILED_MESSAGE = (
    "The frame couldn't be saved for reading:\n\n{reason}\n\n"
    "Try again. If it keeps happening, please report it with the crash log (Debug → Show "
    "crash log…; turn on \"Show the Debug menu in the menu bar\" in Settings → General if "
    "the menu is hidden).")

WINDOW_TITLE = "Custom overlay layouts"


# ── The editor dialog ────────────────────────────────────────────────────────────

class OverlayLayoutEditorDialog(QDialog):
    """The custom-overlay-layout editor popup (see the module docstring for the flow).

    Owns one ``VideoFrameSource`` for the current video (released on video change and on
    close) and the drawn boxes as normalized rects; the ``TrimPlayer`` handles all
    scrubbing/seeking, the ``RegionDrawPane`` all drawing, and the ``OverlayReadProcess``
    helper (a warm child process) all OCR.

    ``initial_game`` (a ``vodparse.games`` id) preselects the dialog's own Game
    dropdown, normally from the main screen's Game combo; an id this build doesn't know
    falls back to the default game.

    ``initial_layout`` names a preset of that game to open on, normally the main screen's
    Overlay layout choice: it is selected in the preset combo once the video is loaded, so
    its boxes, band and clock load exactly as picking it by hand would, and the video's
    first frame is captured so the boxes show over the VOD straight away (boxes are only
    drawn on a captured still). None, or a name with no preset for that game (the main
    screen's "Auto" is ``""``), opens on '(new layout)' over the live video.

    ``initial_video`` opens at ``initial_sec``. It goes through the MP4 conversion offer
    first, as Choose video… does (see _load_video), unless ``initial_video_resolved`` says
    the caller already settled that file: the main window passes its chosen video, which
    went through its own offer, so a second one would only ask again (and a conversion
    made here would lose the playhead and never reach the main window).

    ``start_worker=False`` builds the dialog with no helper client, for offscreen tests:
    readings are injected straight into ``_on_regions_read``, the spot-check window's
    testing scheme. ``layouts_file`` overrides the shared preset store path (tests point
    it at a tmp file)."""

    def __init__(self, settings: QSettings, parent: QWidget | None = None, *,
                 initial_video: str | None = None, initial_video_resolved: bool = False,
                 initial_sec: float = 0.0, initial_game: str = GAME_DEFAULT,
                 initial_layout: str | None = None, start_worker: bool = True,
                 layouts_file: Path | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(WINDOW_TITLE)
        self._settings = settings   # OCR engine/language for the reads; the MKV offer
        self._video_path: str | None = None
        self._source: VideoFrameSource | None = None
        self._captured: np.ndarray | None = None
        self._regions: dict[str, NormRect] = {}
        self._region_texts: dict[str, str] = {}   # last read's text per region
        self._layouts_file = layouts_file if layouts_file is not None else layouts_path()
        self._layouts = load_layouts(self._layouts_file)
        self._generation = 0        # bumped whenever the frame or the boxes change
        self._read_in_flight = False
        # Restarted by every capture and box edit (_schedule_read): the read starts once the
        # frame and the boxes have stayed put for READ_DELAY_MS. Made before the widgets,
        # since building them already bumps the generation (which stops this timer).
        self._read_timer = QTimer(self)
        self._read_timer.setSingleShot(True)
        self._read_timer.setInterval(READ_DELAY_MS)
        self._read_timer.timeout.connect(self._start_read)
        self._read_note = ""        # the helper's note on the read in flight, if any
        self._torn_down = False
        # The preset most recently saved in this dialog session (None until a Save succeeds;
        # cleared again if that preset is deleted): the main screen adopts it as the chooser's
        # selection once the dialog closes (see last_saved_layout).
        self._last_saved: CustomOverlayLayout | None = None
        # The boxes as last loaded from or saved to a preset (none yet): on '(new layout)',
        # closing compares against these to tell freshly drawn boxes from a preset's boxes
        # carried over (see _has_unsaved_changes).
        self._baseline_regions: dict[str, NormRect] = {}
        # The preset the boxes in the editor belong to (None: '(new layout)'): what the
        # preset combo showed when its selection was last taken up. The combo has already
        # moved on by the time a new pick is handled, so unsaved edits are judged against
        # this (see _on_preset_selected).
        self._selected_preset: str | None = None

        # Unplayable containers (mkv/avi/webm) are offered for MP4 conversion exactly as the
        # main window's Choose video does; fileReady adopts whichever file results.
        self._converter = Mp4Converter(self)
        self._converter.fileReady.connect(self._open_video)

        # ── Left column: scrubbing player with the draw pane as its display page ──
        # whole_vod_only: there is no clip here, so the scrubber spans the whole video and
        # the Review screen's clip-window controls are left out.
        self._player = TrimPlayer(whole_vod_only=True)
        self._pane = RegionDrawPane(adjustable=True)   # boxes can be moved and resized here
        self._pane.set_hint_text(NO_FRAME_HINT)
        self._player.set_display_page(self._pane)
        self._player.displayPageDismissed.connect(self._on_live_resumed)
        self._pane.regionDrawn.connect(self._on_region_drawn)
        self._pane.regionSelected.connect(self._on_region_selected)
        # The status line a box selection replaced, put back when the selection ends (see
        # _on_region_selected).
        self._status_before_selection = ""

        # The zoom buttons share the Play row, rather than taking a row of their own.
        for button in make_zoom_buttons(self._pane):
            self._player.add_transport_widget(button)

        left = QVBoxLayout()
        left.addWidget(self._player, stretch=1)

        # ── Right column: video row, presets, region rows, readings, status ──
        self._video_label = ElidingLabel("No video chosen", mode=Qt.TextElideMode.ElideMiddle)
        choose_button = QPushButton("Choose video…")
        choose_button.clicked.connect(self._on_choose_video)
        video_row = QHBoxLayout()
        video_row.addWidget(self._video_label, stretch=1)
        video_row.addWidget(choose_button)

        # Which game the presets below belong to. Local to this dialog on purpose: it
        # never writes the main screen's Game setting, so a user can edit layouts for any
        # game here without changing what the next run analyses. Connected only after the
        # initial selection, so building the combo can't reload a half-built dialog.
        self._game_combo = QComboBox()
        for game in GAMES:
            self._game_combo.addItem(game.display_name, game.id)
        self._game_combo.setCurrentIndex(
            self._game_combo.findData(resolve_saved_game(initial_game)))
        self._game_combo.currentIndexChanged.connect(self._on_game_changed)

        self._preset_combo = QComboBox()
        # A long preset name must not widen the controls column (it scrolls vertically only):
        # size the combo for a fixed number of characters and let its popup show the rest.
        self._preset_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._preset_combo.setMinimumContentsLength(PRESET_COMBO_MIN_CHARS)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        self._delete_button = QPushButton("Delete")
        self._delete_button.clicked.connect(self._on_delete)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Layout name")
        save_button = QPushButton("Save")
        save_button.clicked.connect(self._on_save)
        # The layout's full match clock: the splitter back-dates every game's start from the
        # clock, so a 6:00 or 8:00 game needs its real length, not Ultimate's 7:00.
        self._clock_edit = QTimeEdit()
        self._clock_edit.setDisplayFormat("m:ss")
        self._clock_edit.setTimeRange(QTime(0, 0, 1), QTime(0, 59, 59))
        self._clock_edit.setToolTip("Length of the game clock for this game.")
        self._set_clock_seconds(self._new_layout_clock_seconds())
        clock_row = QHBoxLayout()
        clock_row.addWidget(QLabel("Game clock:"))
        clock_row.addWidget(self._clock_edit)
        clock_row.addStretch(1)
        # One grid of "label | field | button" columns, so the three fields start at the
        # same place, the Layout combo and the Name field end at the same place, and
        # Delete and Save are the same width. The Game combo has no button, so it runs
        # on through the button column. The Game clock row keeps its own short layout.
        preset_group = QGroupBox("Layout presets")
        preset_grid = QGridLayout(preset_group)
        preset_grid.setColumnStretch(PRESET_FIELD_COLUMN, 1)
        preset_grid.addWidget(QLabel("Game:"), 0, PRESET_LABEL_COLUMN)
        preset_grid.addWidget(self._game_combo, 0, PRESET_FIELD_COLUMN, 1, 2)
        preset_grid.addWidget(QLabel("Layout:"), 1, PRESET_LABEL_COLUMN)
        preset_grid.addWidget(self._preset_combo, 1, PRESET_FIELD_COLUMN)
        preset_grid.addWidget(self._delete_button, 1, PRESET_BUTTON_COLUMN)
        preset_grid.addWidget(QLabel("Name:"), 2, PRESET_LABEL_COLUMN)
        preset_grid.addWidget(self._name_edit, 2, PRESET_FIELD_COLUMN)
        preset_grid.addWidget(save_button, 2, PRESET_BUTTON_COLUMN)
        preset_grid.addLayout(clock_row, 3, PRESET_LABEL_COLUMN, 1, PRESET_COLUMN_COUNT)

        self._draw_buttons: dict[str, QPushButton] = {}
        self._clear_buttons: dict[str, QPushButton] = {}
        self._coord_labels: dict[str, QLabel] = {}
        self._region_labels: dict[str, QLabel] = {}   # row names; the timer's varies
        # What the game's own nameplate band holds for this stream. It heads the group
        # although it draws no box: the band's position is the game's, never drawn here
        # (see INGAME_BAND_LABELS). Seeded from the selected game and re-seeded on every
        # game switch; no signal is connected because nothing else reacts to the choice.
        self._ingame_band_combo = QComboBox()
        for band_mode, band_label in INGAME_BAND_LABELS.items():
            self._ingame_band_combo.addItem(band_label, band_mode)
        self._ingame_band_combo.setToolTip(
            "Does the bottom of the screen (under the character icons) have tags or "
            "character names?")
        self._set_ingame_band(self._current_game().ingame_band_default)
        band_row = QHBoxLayout()
        band_row.addWidget(QLabel("Bottom of screen:"))
        band_row.addWidget(self._ingame_band_combo)
        band_row.addStretch(1)
        regions_group = QGroupBox("Overlay regions")
        regions_layout = QVBoxLayout(regions_group)
        regions_layout.addLayout(band_row)
        for key in REGION_KEYS:
            regions_layout.addLayout(self._build_region_row(key))
        # The two nameplate rows follow the band combo (labels + enabled state); connected
        # only now that the rows exist. Their first refresh happens with the rest of the UI
        # state once every widget is built (the _reload_preset_combo call below).
        self._ingame_band_combo.currentIndexChanged.connect(self._on_ingame_band_changed)

        # Which engine reads the boxes for the test, seeded from the Settings choice and
        # switchable here so both font styles can be compared on the same boxes without
        # a trip through Settings. The analysis run itself still uses the Settings
        # choice. Like the Analyze page's Font-style combo, Stylized stays listed,
        # selectable and selected when saved while docTR can't read (marked why: not
        # installed, or its text models not downloaded, which only the Analyze page's own
        # offer downloads, never a read here). A read with it then fails with
        # the reason; nothing is read with Standard in its place.
        self._engine_combo = QComboBox()
        for engine_label, engine_id in OCR_ENGINE_CHOICES.items():
            self._engine_combo.addItem(engine_label, engine_id)
        if not (doctr_available() and doctr_weights_cached()):
            doctr_idx = self._engine_combo.findData(OCR_ENGINE_DOCTR)
            why = ": not installed" if not doctr_available() else ": text models not downloaded"
            self._engine_combo.setItemText(doctr_idx, self._engine_combo.itemText(doctr_idx) + why)
        saved_engine = settings.value(SETTING_OCR_ENGINE, OCR_ENGINE_DEFAULT_GUI, type=str)
        self._engine_combo.setCurrentIndex(self._engine_combo.findData(
            resolve_saved_engine(saved_engine)))
        self._engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        engine_row = QHBoxLayout()
        engine_row.addWidget(QLabel("Font style:"))
        engine_row.addWidget(self._engine_combo)
        engine_row.addStretch(1)

        self._result_labels: dict[str, QLabel] = {}
        test_group = QGroupBox(READINGS_TITLE)
        test_layout = QVBoxLayout(test_group)
        test_layout.addLayout(engine_row)
        for key in REGION_KEYS:
            # Eliding, so a long read never widens the column past the scroll area's
            # viewport (which would let the trackpad pan sideways); the full text rides
            # along as the tooltip.
            result = ElidingLabel(NO_READING, min_chars=RESULT_MIN_CHARS)
            self._result_labels[key] = result
            result_row = QHBoxLayout()
            result_row.addWidget(QLabel(f"{REGION_LABELS[key]}:"))
            result_row.addWidget(result, stretch=1)
            test_layout.addLayout(result_row)
        self._read_status_label = QLabel()
        self._read_status_label.setWordWrap(True)
        test_layout.addWidget(self._read_status_label)

        # Empty until a video is loaded: there is nothing to scrub yet, and the draw
        # pane's hint (NO_FRAME_HINT) already says how to start.
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        save_and_continue_button = QPushButton(SAVE_AND_CONTINUE_LABEL)
        save_and_continue_button.setToolTip("Save and go back to the analysis page")
        save_and_continue_button.clicked.connect(self._on_save_and_continue)
        buttons.addButton(save_and_continue_button, QDialogButtonBox.ButtonRole.ActionRole)

        # The controls column outgrows a laptop screen (seven region rows plus their test
        # readouts), so everything above the status line scrolls; the status line and the
        # Close button stay put below the scroll area.
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addLayout(video_row)
        controls_layout.addWidget(preset_group)
        controls_layout.addWidget(regions_group)
        controls_layout.addWidget(test_group)
        controls_layout.addStretch(1)
        self._controls_scroll = QScrollArea()
        self._controls_scroll.setWidget(controls)
        self._controls_scroll.setWidgetResizable(True)
        self._controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right = QVBoxLayout()
        right.addWidget(self._controls_scroll, stretch=1)
        right.addWidget(self._status_label)
        right.addWidget(buttons)

        layout = QHBoxLayout(self)
        layout.addLayout(left, stretch=1)
        layout.addLayout(right)
        self.resize(*self._default_size())
        # No button here answers Enter. A QDialog makes every push button auto-default, so
        # Enter in the name field (or the Enter that picks a preset in the macOS popup, when
        # the switch asks about unsaved changes) pressed the player's Play button: the video
        # started and the captured still was thrown away (Mac, 2026-10-25). Save and continue
        # must not fire on Enter either: it would close the editor.
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)

        self._reload_preset_combo(select_name=None)
        self._refresh_timer_required_label()
        self._refresh_ingame_rows()
        # Measured last, once every row wears its final label: wide enough for the column's
        # minimum width plus the scrollbar, so nothing is ever clipped sideways and only the
        # height gives. Everything whose text can grow later (test-read results, the video
        # name, preset names) elides or is fixed-width, so this minimum holds for the life
        # of the dialog.
        scrollbar = self._controls_scroll.verticalScrollBar()
        scrollbar_width = scrollbar.sizeHint().width() if scrollbar is not None else 0
        self._controls_scroll.setMinimumWidth(controls.minimumSizeHint().width() + scrollbar_width)

        # ── OCR helper client + frame grabber (both QProcess-backed) ──
        # start_worker=False builds neither, for offscreen tests: grabs and readings are
        # injected straight into _on_frame_grabbed/_on_regions_read.
        self._reader: OverlayReadProcess | None = None
        self._grabber: FrameGrabber | None = None
        self._scratch_dir: Path | None = None   # temp dir for the frames handed to the helper
        self._scratch_frames: dict[int, Path] = {}   # those frames not yet answered, by read token
        self._grab_token = 0          # bumped per grab request; stale grabs are dropped
        self._grab_in_flight = False
        self._pending_draw_key: str | None = None   # arm this region once the grab lands
        if start_worker:
            self._reader = OverlayReadProcess(self)
            self._reader.regions_read.connect(self._on_regions_read)
            self._reader.read_noted.connect(self._on_read_noted)
            self._reader.failed.connect(self._on_read_failed)
            self._grabber = FrameGrabber(self)
            self._grabber.frame_grabbed.connect(self._on_frame_grabbed)
            self._grabber.grab_failed.connect(self._on_grab_failed)
        # A scrub moves the playhead off whatever a grab in flight was aimed at; letting
        # the scrubber go or pausing playback settles it, and the settle timer then
        # captures the frame under it (the Analyze page's preview works the same way).
        self._player.scrubbed.connect(self._on_scrubbed)
        self._player.scrubReleased.connect(self._on_scrub_released)
        self._player.playingChanged.connect(self._on_playing_changed)
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(SETTLE_DELAY_MS)
        self._settle_timer.timeout.connect(self._on_settled)

        if initial_video is not None:
            # Opened at the Analyze page's own playhead, so the editor continues from the
            # frame the preview showed rather than rewinding to the start.
            if initial_video_resolved:
                self._open_video(initial_video, start_sec=initial_sec)
            else:
                self._load_video(initial_video, start_sec=initial_sec)
        if self._source is None:
            # No video yet (none was given, or the given one couldn't be opened): swap the
            # (empty) draw pane in front of the black video widget so its hint text
            # explains the first step. Loading a video dismisses it back to live.
            self._player.show_display_page(True)
        # After the video: the capture needs its frame source. findData misses Auto's ""
        # and '(new layout)' is index 0, so only a real preset of this game is selected.
        if initial_layout is not None:
            index = self._preset_combo.findData(initial_layout)
            if index > 0:
                # Runs _on_preset_selected, which grabs the frame at the playhead (the
                # start) so the preset's boxes show at once.
                self._preset_combo.setCurrentIndex(index)
        self._update_ui_state()

    # ── Video loading ───────────────────────────────────────────────────────────

    def _on_choose_video(self) -> None:
        start_dir = str(Path(self._video_path).parent) if self._video_path else ""
        path, _ = QFileDialog.getOpenFileName(self, "Choose video", start_dir, VIDEO_FILTER)
        if path:
            self._load_video(path)

    def _load_video(self, path: str, start_sec: float = 0.0) -> None:
        """Open a video for scrubbing, routing unplayable containers through the MP4
        conversion offer first (its fileReady lands in _open_video either way, at the
        start: the converted file is a different video). An MKV on Windows gets the offer
        too once it is turned on in Settings, as in the main window. ``start_sec`` is where
        the playhead opens."""
        ask_for_mkv = self._settings.value(
            SETTING_ASK_CONVERT_MKV, ASK_CONVERT_MKV_DEFAULT, type=bool)
        if offers_conversion(path, ask_for_mkv=ask_for_mkv):
            self._converter.offer_convert(path)
        else:
            self._open_video(path, start_sec)

    def _open_video(self, path: str, start_sec: float = 0.0) -> None:
        """Adopt ``path``: build its frame source (an unopenable file warns and keeps the
        current video usable), point the player at it (at ``start_sec``), and return to
        the live state. The drawn boxes are kept: they're normalized, so they apply to
        the new video too."""
        try:
            source = VideoFrameSource(path)
        except ValueError as exc:
            QMessageBox.warning(self, WINDOW_TITLE, str(exc))
            return
        if self._source is not None:
            self._source.release()
        self._source = source
        self._video_path = path
        self._captured = None
        self._grab_token += 1   # a grab still in flight belongs to the old video
        self._grab_in_flight = False
        self._pending_draw_key = None
        if self._grabber is not None:
            self._grabber.cancel()
        self._bump_generation()
        self._video_label.setText(Path(path).name)
        self._video_label.setToolTip(path)

        self._player.set_source(path)
        self._player.set_region(0.0, source.duration_sec())   # scrub the whole video
        # After the region is set: set_region moves the playhead to the region's start,
        # so a seek placed before it would be undone.
        if start_sec > 0.0:
            self._player.seek_to(min(start_sec, source.duration_sec()))
        self._set_status(LIVE_STATUS)
        # Capture the opening frame once the player has painted it, as letting go of the
        # scrubber does: the editor opens paused, so no scrub or pause would otherwise come
        # along to capture it, and a new layout sat on "Scrubbing…" until the first Draw.
        self._settle_timer.start()
        self._update_ui_state()

    # ── Capture / draw flow ─────────────────────────────────────────────────────

    def _request_capture(self) -> None:
        """Start grabbing the playhead frame asynchronously (see FrameGrabber — never on
        the UI thread, and via ffmpeg's fast index seek). _on_frame_grabbed finishes the
        capture when the frame lands; until then capture and drawing are disabled. The
        in-flight flag is raised before the pause so the pause's own playingChanged
        doesn't queue a second capture (see _on_playing_changed)."""
        if self._source is None or self._video_path is None:
            return
        self._settle_timer.stop()
        self._grab_token += 1
        self._grab_in_flight = True
        self._player.pause()
        sec = self._player.playhead_sec()
        self._set_status(f"Grabbing the frame at {hms(sec)}…")
        if self._grabber is not None:
            self._grabber.grab(self._video_path, sec, self._grab_token)
        self._update_ui_state()

    def _on_frame_grabbed(self, frame: np.ndarray, sec: float, token: int) -> None:
        """A grab landed: freeze it onto the draw pane, arm a Draw that was waiting on it,
        and read the drawn boxes on it. The grabbed frame — not the player's displayed
        one — is what boxes are drawn on and what the read OCRs, so what you mark
        is exactly what gets read; the two can differ by ~a frame (different demuxers),
        which is fine for box drawing. A stale token means the user scrubbed on or
        re-captured while this grab was in flight; drop it."""
        if token != self._grab_token:
            return
        self._grab_in_flight = False
        self._captured = frame
        self._bump_generation()   # a new frame invalidates any in-flight/old read
        self._pane.set_frame(frame)
        self._pane.set_regions(self._regions)
        self._player.show_display_page(True)
        if self._regions:
            # The read that follows reports in the readings group, not here.
            self._set_status(f"Captured the frame at {hms(sec)}.")
        else:
            # Nothing to read yet, so the status line says how to start drawing.
            self._set_status(f"Captured the frame at {hms(sec)}. "
                             "Press a region's Draw button, then drag its box on the frame.")
        pending_draw, self._pending_draw_key = self._pending_draw_key, None
        if pending_draw is not None:
            self._pane.set_armed_region(pending_draw)   # its Draw button stayed checked
        self._schedule_read()
        self._update_ui_state()

    def _on_grab_failed(self, message: str, token: int) -> None:
        if token != self._grab_token:
            return
        self._grab_in_flight = False
        self._pending_draw_key = None
        self._uncheck_draw_buttons()
        self._set_status(f"Couldn't grab a frame: {message}")
        self._update_ui_state()

    def _on_scrubbed(self, _sec: float) -> None:
        """The scrubber moved: no capture is due until it is let go. (A still on show is
        dismissed by the player itself, on the press.)"""
        self._settle_timer.stop()
        self._cancel_grab()

    def _on_scrub_released(self, _sec: float) -> None:
        """The scrubber was let go: capture the frame there once the player has painted
        it (see SETTLE_DELAY_MS)."""
        self._settle_timer.start()

    def _on_playing_changed(self, playing: bool) -> None:
        """Playback started (the playhead is moving: nothing to capture) or paused (the
        playhead settled: capture after the delay). A pause that is a capture's own
        (``_request_capture`` pauses before grabbing) or one arriving while a capture is
        already due must not queue another capture."""
        if playing:
            self._settle_timer.stop()
            self._cancel_grab()
            return
        if self._grab_in_flight or self._settle_timer.isActive():
            return
        self._settle_timer.start()

    def _on_settled(self) -> None:
        """The settle delay ran out: capture the frame under the playhead, unless a still
        is already up or a Draw started a grab meanwhile."""
        if self._captured is None and not self._grab_in_flight:
            self._request_capture()

    def _cancel_grab(self) -> None:
        """The playhead moved while a grab was in flight: that grab is aimed at a frame
        the playhead has left, so cancel it (with a Draw waiting on it)
        and return to the live state. A no-op when no grab is in flight."""
        if not self._grab_in_flight:
            return
        self._grab_token += 1   # anything already grabbed comes back stale
        if self._grabber is not None:
            self._grabber.cancel()
        self._grab_in_flight = False
        self._pending_draw_key = None
        self._uncheck_draw_buttons()
        self._set_status(LIVE_STATUS)
        self._update_ui_state()

    def _on_live_resumed(self) -> None:
        """The player auto-dismissed the captured still (a scrub/play moved the playhead,
        or a new video loaded): drop the still and disarm drawing. The boxes themselves
        persist — they're normalized rects, not tied to that still.

        Without a video there is no live state to return to: Play or the scrubber was
        pressed before one was chosen, so the hint pane goes back up and the status line
        stays as it was."""
        if self._source is None:
            self._player.show_display_page(True)
            return
        self._captured = None
        self._bump_generation()
        self._pane.set_armed_region(None)
        self._uncheck_draw_buttons()
        self._set_status(LIVE_STATUS)
        self._update_ui_state()

    def _build_region_row(self, key: str) -> QHBoxLayout:
        """One region's control row: color swatch, name, coords readout, Draw, Clear."""
        swatch = QLabel()
        swatch.setFixedSize(12, 12)
        swatch.setStyleSheet(
            f"background-color: {REGION_COLORS[key].name()}; border-radius: 2px;")
        name = QLabel(REGION_LABELS[key])
        # Rows that relabel at run time (the timer's "required" form, the nameplate rows
        # following the band combo) reserve their widest text now, so relabeling never
        # changes the column's minimum width after it has been measured.
        widest = max((self.fontMetrics().horizontalAdvance(text)
                      for text in _all_labels_for(key)), default=0)
        name.setMinimumWidth(widest)
        # The readout grows from a dash to "x 88%–100%  y 88%–100%" once a box is drawn, so it
        # reserves its longest form now too; otherwise a drawn box outgrows the measured
        # column and the readout is squeezed over the name and clipped.
        coords = QLabel(self._widest_coords_text())
        coords.setMinimumWidth(coords.sizeHint().width())   # the label's own measure of it
        coords.setText(self._coords_text(None))
        coords.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        draw = QPushButton("Draw")
        draw.setCheckable(True)
        draw.setToolTip(f"Drag a box around the {REGION_LABELS[key]} on the captured frame")
        draw.toggled.connect(lambda checked, k=key: self._on_draw_toggled(k, checked))
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda _=False, k=key: self._on_clear_region(k))
        self._draw_buttons[key] = draw
        self._clear_buttons[key] = clear
        self._coord_labels[key] = coords
        self._region_labels[key] = name

        row = QHBoxLayout()
        row.addWidget(swatch)
        row.addWidget(name)
        row.addStretch(1)
        row.addWidget(coords)
        row.addWidget(draw)
        row.addWidget(clear)
        return row

    def _on_draw_toggled(self, key: str, checked: bool) -> None:
        """A region's Draw button: checking it arms that region (and unchecks the rest —
        one region draws at a time); unchecking disarms. Pressed over the live video, it
        starts capturing the playhead frame and arms once the frame lands (the button
        stays checked meanwhile), so drawing is always one click away."""
        if checked:
            for other, button in self._draw_buttons.items():
                if other != key and button.isChecked():
                    button.blockSignals(True)
                    button.setChecked(False)
                    button.blockSignals(False)
            if self._captured is None:
                self._pending_draw_key = key
                self._request_capture()
                return
            self._pane.set_armed_region(key)
        else:
            if self._pending_draw_key == key:
                self._pending_draw_key = None
            if self._pane.armed_region() == key:
                self._pane.set_armed_region(None)

    def _cancel_draw(self) -> None:
        """Disarm a Draw (Escape): uncheck its button, drop a Draw still waiting on its
        frame grab (the grab goes on and lands as a plain capture), and cancel a drag in
        progress, leaving that region's box as it was. A no-op when no Draw is armed; a
        checked Draw button marks both the armed and the waiting state."""
        if not any(button.isChecked() for button in self._draw_buttons.values()):
            return
        self._pending_draw_key = None
        self._pane.set_armed_region(None)   # also drops a half-dragged rubber band
        self._uncheck_draw_buttons()
        self._set_status("Draw cancelled.")
        self._update_ui_state()

    def _on_region_drawn(self, key: str, rect: NormRect) -> None:
        """A box was drawn on the pane, or an existing one moved, resized or nudged: store
        it and refresh the visualization. After a draw the pane disarmed itself already;
        sync the Draw buttons to match."""
        self._regions[key] = rect
        self._bump_generation()   # old readings no longer describe these boxes
        self._uncheck_draw_buttons()
        self._pane.set_regions(self._regions)
        self._coord_labels[key].setText(self._coords_text(rect))
        self._schedule_read()
        self._update_ui_state()

    def _on_region_selected(self, key: str | None) -> None:
        """A box was selected on the pane: say how to adjust it. When the selection ends
        (empty frame clicked, Escape, another Draw armed), put back the line the selection
        message replaced. If anything else has written the status line since (a save, a
        capture), that newer line stays."""
        showing_selection = self._status_label.text() in (
            _selected_box_status(region) for region in REGION_KEYS)
        if key is not None:
            if not showing_selection:   # moving from one box to another keeps the first line
                self._status_before_selection = self._status_label.text()
            self._set_status(_selected_box_status(key))
        elif showing_selection:
            self._set_status(self._status_before_selection)

    def _on_clear_region(self, key: str) -> None:
        self._regions.pop(key, None)
        self._bump_generation()
        self._pane.set_regions(self._regions)
        self._coord_labels[key].setText(self._coords_text(None))
        self._schedule_read()
        self._update_ui_state()

    # ── Presets ─────────────────────────────────────────────────────────────────

    def _current_game(self) -> Game:
        """The registry entry for the Game combo's selection. Every combo entry carries a
        registry id as its data, so the lookup always succeeds."""
        game = game_by_id(str(self._game_combo.currentData()))
        assert game is not None, "the Game combo only holds registry ids"
        return game

    def _new_layout_clock_seconds(self) -> int:
        """The Game clock to seed an unsaved layout with: the selected game's own full
        clock (Ultimate 7:00, Melee and Rivals 2 8:00), or DEFAULT_CLOCK_SECONDS for a
        game with no clock of its own ('Other')."""
        full_game_seconds = self._current_game().full_game_seconds
        if full_game_seconds is None:
            return DEFAULT_CLOCK_SECONDS
        return full_game_seconds

    def _refresh_timer_required_label(self) -> None:
        """Point the timer row's label at the rule for the selected game: required when
        the game has no clock detector, plain otherwise (see TIMER_LABEL_REQUIRED)."""
        label = (REGION_LABELS[REGION_TIMER] if self._current_game().has_timer_model
                 else TIMER_LABEL_REQUIRED)
        self._region_labels[REGION_TIMER].setText(label)

    def _on_game_changed(self, _index: int) -> None:
        """The Game combo changed: the preset list, the seeded Game clock, the seeded
        Bottom of screen mode and the timer row's label all follow the selected game, and the
        preset combo lands on '(new layout)'.

        The drawn boxes deliberately survive the switch: loading an Ultimate preset,
        switching the game here and saving under a new name is how a user copies a layout
        to another game."""
        self._reload_preset_combo(select_name=None)   # always lands on '(new layout)'
        self._set_clock_seconds(self._new_layout_clock_seconds())
        self._set_ingame_band(self._current_game().ingame_band_default)
        self._refresh_timer_required_label()
        self._set_status("Showing the layout presets for "
                         f"{self._current_game().display_name}.")

    def _reload_preset_combo(self, select_name: str | None) -> None:
        """Rebuild the preset combo from the loaded layouts *of the selected game*: the
        '(new layout)' entry (data None) first, then one entry per preset (name as data).
        Selects ``select_name`` when it exists, else '(new layout)' — which also covers a
        just-deleted preset and every game switch."""
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        self._preset_combo.addItem(NEW_LAYOUT_ITEM, None)
        for layout in layouts_for_game(self._layouts, self._current_game().id):
            self._preset_combo.addItem(layout.name, layout.name)
        index = self._preset_combo.findData(select_name) if select_name is not None else 0
        self._preset_combo.setCurrentIndex(max(index, 0))
        self._preset_combo.blockSignals(False)
        self._selected_preset = self._preset_combo.currentData()
        self._update_ui_state()

    def _show_preset_silently(self, name: str | None) -> None:
        """Put the preset combo on ``name`` ('(new layout)' for None, or for a name no
        longer listed) without loading anything: for going back to the layout being
        edited when a switch away from it is cancelled."""
        index = self._preset_combo.findData(name) if name is not None else 0
        self._preset_combo.blockSignals(True)
        self._preset_combo.setCurrentIndex(max(index, 0))
        self._preset_combo.blockSignals(False)
        self._update_ui_state()

    def _on_preset_selected(self, _index: int) -> None:
        """Load the picked preset's boxes into the editor and show them on a captured
        still (or, for '(new layout)', just clear the name — the drawn boxes stay, ready
        to be saved under a new name).

        Loading a preset replaces the boxes in the editor, so unsaved edits to the layout
        being left are offered for saving first (_confirm_preset_switch); with nothing
        unsaved the preset loads at once. Cancel there puts the combo back."""
        name = self._preset_combo.currentData()
        previous = self._selected_preset
        if name is not None and not self._confirm_preset_switch(previous, name):
            self._show_preset_silently(previous)
            return
        self._selected_preset = name
        if name is None:
            self._name_edit.setText("")
            self._set_clock_seconds(self._new_layout_clock_seconds())
            self._set_ingame_band(self._current_game().ingame_band_default)
            return
        layout = layout_by_name(self._layouts, name)
        if layout is None:   # deleted meanwhile; the combo rebuild will drop the entry
            return
        self._regions = dict(layout.regions())
        self._baseline_regions = dict(self._regions)
        self._name_edit.setText(layout.name)
        self._set_clock_seconds(layout.clock_seconds)
        self._set_ingame_band(layout.ingame_band)
        self._bump_generation()
        self._pane.set_regions(self._regions)
        for key in REGION_KEYS:
            self._coord_labels[key].setText(self._coords_text(self._regions.get(key)))
        self._set_status(f"Loaded layout {layout.name!r}.")
        self._schedule_read()
        self._update_ui_state()
        # Show the boxes over the VOD at once, without waiting for a scrub: over the
        # live video, grab the playhead frame (a no-op without a video). A still already
        # on screen has just redrawn them, and a grab in flight lands with them.
        if self._captured is None and not self._grab_in_flight:
            self._request_capture()

    def _on_save(self) -> bool:
        """Save the drawn boxes under the typed name: a new preset for a new name, an
        in-place update for an existing one (matched case-insensitively). Validation
        problems land in the status line rather than a modal — they're routine editing
        states (no name yet, boxes not drawn yet), not errors. Returns whether the
        layout was written (the Save button ignores it; the two save prompts act on it)."""
        name = self._name_edit.text().strip()
        if not name:
            self._set_status("Type a name for the layout, then press Save.")
            return False
        if name.casefold() == AUTO_LAYOUT_NAME.casefold():
            self._set_status(f"{AUTO_LAYOUT_NAME!r} is reserved for the built-in "
                             "overlay reading. Pick another name.")
            return False
        game = self._current_game()
        existing = layout_by_name(self._layouts, name)
        if existing is not None and existing.game != game.id:
            # Names are unique across the whole store (the chooser, the reuse warning and
            # --overlay-layout all identify a preset by name alone), so saving under another
            # game's name would silently replace that game's preset: refuse instead.
            other = game_by_id(existing.game)
            other_game = other.display_name if other is not None else existing.game
            self._set_status(f"{name!r} is already taken by a layout for {other_game}. Pick "
                             f"another name for this {game.display_name} layout.")
            return False
        try:
            layout = CustomOverlayLayout(
                name=name,
                game=game.id,
                ingame_band=self._ingame_band(),
                player1=self._regions.get(REGION_PLAYER1),
                player2=self._regions.get(REGION_PLAYER2),
                round_label=self._regions.get(REGION_ROUND),
                best_of=self._regions.get(REGION_BEST_OF),
                timer=self._regions.get(REGION_TIMER),
                ingame_p1=self._regions.get(REGION_INGAME_P1),
                ingame_p2=self._regions.get(REGION_INGAME_P2),
                clock_seconds=self._clock_seconds(),
            )
        except ValueError as exc:
            # The model owns the rules (a game without a clock detector needs a timer box;
            # the gamertag boxes come as a pair or not at all; at least one box must be
            # drawn), and its message is the user-facing wording for them.
            self._set_status(str(exc))
            return False
        updated = existing is not None
        saved = with_layout_saved(self._layouts, layout)
        # The in-memory store changes only once the file has: a failed write must leave
        # the edit counting as unsaved, or the close prompt would close over it.
        try:
            save_layouts(self._layouts_file, saved)
        except OSError as exc:
            QMessageBox.warning(self, WINDOW_TITLE, f"Couldn't save the layout: {exc}")
            return False
        self._layouts = saved
        self._last_saved = layout
        self._baseline_regions = dict(self._regions)
        self._reload_preset_combo(select_name=name)
        self._set_status(f"{'Updated' if updated else 'Saved'} layout {name!r}.")
        return True

    def _on_save_and_continue(self) -> None:
        """Save the layout exactly as the Save button does, then close the editor (the
        Save and continue button). The main screen selects the layout just saved (see
        last_saved_layout), so one click goes from the drawn boxes to a run with them. A
        layout Save can't pass yet (no name, a required box missing) keeps the editor
        open with Save's own message in the status line."""
        if not self._on_save():
            if not self._name_edit.text().strip():
                self._name_edit.setFocus()   # the usual missing piece on a new layout
            return
        # Nothing is unsaved now, so done() closes without its save prompt.
        self.accept()

    def _on_delete(self) -> None:
        """Delete the selected preset. Preset names are unique across the whole store
        (not per game), so the store is filtered by name alone; the combo then reloads
        for the selected game."""
        name = self._preset_combo.currentData()
        if name is None:
            return
        answer = QMessageBox.question(self, WINDOW_TITLE,
                                      f"Delete the layout {name!r}?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._layouts = [layout for layout in self._layouts
                         if layout.name.casefold() != name.casefold()]
        try:
            save_layouts(self._layouts_file, self._layouts)
        except OSError as exc:
            QMessageBox.warning(self, WINDOW_TITLE, f"Couldn't save the change: {exc}")
            return
        if self._last_saved is not None and self._last_saved.name.casefold() == name.casefold():
            self._last_saved = None   # nothing for the main screen to select any more
        self._reload_preset_combo(select_name=None)
        self._set_status(f"Deleted layout {name!r}.")

    @property
    def last_saved_layout(self) -> CustomOverlayLayout | None:
        """The preset most recently saved (new or updated) while this dialog was open, or
        None when nothing was saved or that preset was deleted again. Read by the main
        screen after the dialog closes: a preset the user just saved is the one they mean
        to run with next, so it becomes the Overlay layout chooser's selection for its game
        (the chooser otherwise keeps each game's remembered pick)."""
        return self._last_saved

    # ── Reading the boxes ──────────────────────────────────────────────────────

    def _schedule_read(self) -> None:
        """(Re)start the wait before the drawn boxes are read on the captured frame: every
        capture and box edit calls this, and the read starts once nothing has changed for
        READ_DELAY_MS. Nothing is due without a still on screen or a box drawn."""
        if self._captured is None or not self._regions:
            self._read_timer.stop()
            return
        self._read_timer.start()

    def _start_read(self) -> None:
        """Send the drawn boxes off to the OCR helper: the frame goes as a lossless PNG
        in this dialog's temp dir, so the helper reads exactly the captured pixels. The
        engine comes from the dialog's Font-style combo (seeded from Settings) and the
        language from Settings, so the readings preview what a real run reads; the in-game
        nameplate boxes are read with the engine the selected game pins for its band
        (Rivals 2: docTR) when the band is read at all, as a real run reads them."""
        if self._captured is None or not self._regions:
            return
        frame_h, frame_w = self._captured.shape[:2]
        rects_px = {key: rect.to_pixels(frame_w, frame_h)
                    for key, rect in self._regions.items()}
        engine = self._engine_combo.currentData()
        lang = self._settings.value(SETTING_OCR_LANG, OCR_LANG_DEFAULT_GUI, type=str)
        ingame_engine = ingame_band_engine_for(self._current_game(), self._ingame_band())
        self._read_in_flight = True
        self._read_note = ""
        self._read_status_label.setText(READ_STARTED_STATUS)
        if self._reader is not None:
            try:
                frame_path = self._write_frame_png(self._generation)
            except OSError as exc:
                self._read_in_flight = False
                self._read_status_label.clear()
                self._report_frame_save_failure(exc)
            else:
                self._reader.request(engine, lang, str(frame_path), rects_px,
                                     self._generation, ingame_engine=ingame_engine)
        self._update_ui_state()

    def _report_frame_save_failure(self, exc: OSError) -> None:
        """The captured frame couldn't be written for the helper (a full disk, an unwritable
        temp folder): rare enough that it warrants a dialog naming the reason, and a line in
        the crash log for a bug report, rather than only the status line."""
        crash_log.log_exception("layout editor: saving the captured frame for a read", exc)
        self._set_status(FRAME_SAVE_FAILED_STATUS)
        QMessageBox.warning(self, FRAME_SAVE_FAILED_TITLE,
                            FRAME_SAVE_FAILED_MESSAGE.format(reason=exc))

    def _write_frame_png(self, token: int) -> Path:
        """Save the captured frame for the helper, token-named so a superseded request
        that the helper is still reading can never see a half-overwritten file. Each
        frame is deleted once its request is answered (see _delete_answered_frames); the
        whole temp dir goes away in _teardown. Raises OSError, its message saying why,
        when the frame can't be written."""
        assert self._captured is not None   # _start_read returns before this without one
        if self._scratch_dir is None:
            self._scratch_dir = Path(tempfile.mkdtemp(prefix="vodparse-overlay-editor-"))
        path = self._scratch_dir / f"frame_{token}.png"
        try:
            written = cv2.imwrite(str(path), self._captured)
        except cv2.error as exc:
            raise OSError(f"OpenCV couldn't write {path}: {exc}") from exc
        if not written:
            raise OSError(f"OpenCV couldn't write {path}")
        self._scratch_frames[token] = path
        return path

    def _delete_answered_frames(self, token: int) -> None:
        """Delete the frame of every request up to ``token``, whose answer (or failure)
        just came in. A read starts after every capture and box edit, each on its own
        full-resolution PNG (tens of MB for a 4K frame), so a session of edits would
        otherwise pile them up until the dialog closes. The helper reads a request's
        frame before answering it and answers strictly in request order, so no request at
        or before ``token`` will open its frame again; a helper killed or crashed on the
        way reads nothing more at all. Best-effort: _teardown removes the whole temp dir
        anyway. Same scheme as LayoutPreviewPane._delete_answered_frames."""
        answered = [written for written in self._scratch_frames if written <= token]
        for written in answered:
            path = self._scratch_frames.pop(written)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _on_regions_read(self, readings: dict[str, RegionReading], token: int) -> None:
        """A read came back from the helper. Results for a superseded frame/boxes
        (stale token) are dropped, their frame deleted all the same: their request's
        in-flight flag was already cleared by the generation bump that superseded them."""
        self._delete_answered_frames(token)
        if token != self._generation:
            return
        self._read_in_flight = False
        # A player's sponsor rides along as "Sponsor | Tag" (tag_display_text), on the
        # result label and on the box label drawn on the frame alike.
        self._region_texts = {key: tag_display_text(reading)
                              for key, reading in readings.items()}
        for key, reading in readings.items():
            # The timer goes through the pipeline's clock parser, not a generic text
            # read (see cli.overlay_read), so its result is pass/fail: either a clock
            # parsed there — clock-based splitting will work — or it didn't. A failure
            # shows what OCR actually saw (when it saw anything), so a colon misread
            # as "." or a hallucinated digit is diagnosable at a glance.
            if key == REGION_TIMER:
                if reading.text:
                    self._result_labels[key].setText(reading.text)
                elif reading.raw:
                    self._result_labels[key].setText(
                        f"No clock found (reading: {reading.raw!r})")
                else:
                    self._result_labels[key].setText("No clock found")
            elif reading.text:
                self._result_labels[key].setText(
                    f"{self._region_texts[key]}  (confidence {reading.confidence:.2f})")
            else:
                self._result_labels[key].setText("(nothing read)")
        self._pane.set_regions(self._regions, texts=self._region_texts)
        boxes_read = sum(1 for reading in readings.values() if reading.text)
        status = READ_DONE_STATUS.format(read=boxes_read, drawn=len(readings))
        if self._read_note:
            status += f" {self._read_note}"
        self._read_status_label.setText(status)
        self._update_ui_state()

    def _on_read_noted(self, note: str, token: int) -> None:
        """The helper attached a note to the read whose readings follow (it left some
        boxes unread and says why: the game reads its in-game names with an engine that
        can't read here); shown with the readings group's status line."""
        if token == self._generation:
            self._read_note = note

    def _on_engine_changed(self, _index: int) -> None:
        """The reading engine changed: the shown readings came from the previous engine,
        so they no longer describe what this engine reads. Read again with this one."""
        self._bump_generation()
        self._schedule_read()
        self._update_ui_state()

    def _on_read_failed(self, message: str, token: int) -> None:
        self._delete_answered_frames(token)
        if token != self._generation:
            return
        self._read_in_flight = False
        self._read_status_label.clear()
        self._set_status(message)
        self._update_ui_state()

    def _bump_generation(self) -> None:
        """The frame or the boxes changed: invalidate any in-flight or shown read, and any
        read still waiting to start (the caller schedules a fresh one when there is
        something to read). A superseded request still finishes inside the helper (a few
        wasted seconds of OCR); its response is dropped here by the token mismatch."""
        self._read_timer.stop()
        self._generation += 1
        self._read_in_flight = False
        self._region_texts = {}
        for label in self._result_labels.values():
            label.setText(NO_READING)
        self._read_status_label.clear()

    def _uncheck_draw_buttons(self) -> None:
        for button in self._draw_buttons.values():
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)

    def _clock_seconds(self) -> int:
        """The Game clock control's value in seconds (the control shows m:ss)."""
        time = self._clock_edit.time()
        return time.minute() * 60 + time.second()

    def _set_clock_seconds(self, seconds: int) -> None:
        self._clock_edit.setTime(QTime(0, seconds // 60, seconds % 60))

    # The dialog's preferred size, and how much of the screen it may take before it shrinks
    # to fit (the controls column scrolls, so a smaller height costs nothing but scrolling).
    PREFERRED_SIZE = (1240, 720)
    SCREEN_FILL_FRAC = 0.9

    def _default_size(self) -> tuple[int, int]:
        """PREFERRED_SIZE, shrunk to SCREEN_FILL_FRAC of the available screen on a small
        display (a 13-inch laptop minus its menu bar and dock) so the dialog opens fully on
        screen with its Close button reachable."""
        preferred_width, preferred_height = self.PREFERRED_SIZE
        screen = self.screen()
        if screen is None:
            return preferred_width, preferred_height
        available = screen.availableGeometry()
        return (min(preferred_width, int(available.width() * self.SCREEN_FILL_FRAC)),
                min(preferred_height, int(available.height() * self.SCREEN_FILL_FRAC)))

    def _ingame_band(self) -> str:
        """The Bottom of screen combo's selection: one of vodparse.games.INGAME_BAND_MODES
        (every entry carries a mode as its data)."""
        return str(self._ingame_band_combo.currentData())

    def _on_ingame_band_changed(self, _index: int) -> None:
        """The band mode decides whether, and with which engine, the in-game boxes are
        read: the shown readings no longer hold, so read again."""
        self._refresh_ingame_rows()
        self._bump_generation()
        self._schedule_read()

    def _refresh_ingame_rows(self) -> None:
        """Relabel the two nameplate rows for what a drawn box would be read as, and grey
        them out when the band is not read at all (a box drawn there would feed nothing;
        the boxes themselves are kept, so switching the mode back restores them)."""
        mode = self._ingame_band()
        for key in INGAME_REGION_KEYS:
            self._region_labels[key].setText(INGAME_ROW_LABELS[mode][key])
        self._update_ui_state()

    def _set_ingame_band(self, mode: str) -> None:
        """Select ``mode`` on the Bottom of screen combo, offering only the modes the selected
        game can read its band as (``Game.ingame_band_choices``): the others are greyed
        out, and the combo is locked when there is one choice (Melee: "None"). A mode the
        game does not offer (a layout saved under another game, or by an older build)
        selects "None", which is what a run reads it as."""
        choices = self._current_game().ingame_band_choices
        model = self._ingame_band_combo.model()
        assert isinstance(model, QStandardItemModel)   # a QComboBox's default model
        for index in range(self._ingame_band_combo.count()):
            item = model.item(index)
            if item is not None:
                item.setEnabled(self._ingame_band_combo.itemData(index) in choices)
        self._ingame_band_combo.setEnabled(len(choices) > 1)
        if mode not in choices:
            mode = INGAME_BAND_NONE
        self._ingame_band_combo.setCurrentIndex(self._ingame_band_combo.findData(mode))

    def _widest_coords_text(self) -> str:
        """The longest coordinate readout this font draws: a two-digit start and a 100% end
        on both axes, compared over every repeated digit (11%, 22%, ... 99%) because a UI
        font needn't draw all digits at one width."""
        candidates = [self._coords_text(NormRect(n / 100, n / 100, 1.0, 1.0))
                      for n in range(11, 100, 11)]
        return max(candidates, key=self.fontMetrics().horizontalAdvance)

    @staticmethod
    def _coords_text(rect: NormRect | None) -> str:
        """A box's readout as frame percentages ('x 5–30%  y 90–97%'), or a dash."""
        if rect is None:
            return "—"
        return (f"x {rect.x1:.0%}–{rect.x2:.0%}  y {rect.y1:.0%}–{rect.y2:.0%}")

    # ── Shared state upkeep ─────────────────────────────────────────────────────

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def _update_ui_state(self) -> None:
        """Sync enabled states: drawing needs a video (Draw captures the
        playhead frame itself when none is up); Clear needs a drawn box; Delete needs a
        saved preset selected."""
        ready = self._source is not None and not self._grab_in_flight
        band_read = self._ingame_band() != INGAME_BAND_NONE
        for key in REGION_KEYS:
            # The nameplate boxes only mean something while the band is read (see
            # _refresh_ingame_rows); every other row follows the video alone.
            row_active = band_read or key not in INGAME_REGION_KEYS
            self._draw_buttons[key].setEnabled(ready and row_active)
            self._clear_buttons[key].setEnabled(key in self._regions and row_active)
        self._delete_button.setEnabled(self._preset_combo.currentData() is not None)

    # ── Unsaved changes ─────────────────────────────────────────────────────────

    def _has_unsaved_changes(self) -> bool:
        """Whether closing now would lose layout work (see _unsaved_against), judged
        against the preset the combo shows."""
        return self._unsaved_against(self._preset_combo.currentData())

    def _unsaved_against(self, name: str | None) -> bool:
        """Whether the editor holds layout work that the preset ``name`` (None:
        '(new layout)') does not.

        Against a preset: the boxes, Bottom of screen mode or Game clock differ from what the
        store holds. Against '(new layout)': boxes are drawn that differ from the ones last
        loaded or saved, so a preset's boxes carried over by a game switch or a delete
        don't count, and neither do the band and clock alone (there's nothing to save
        without a box). The name field is not layout content and never counts."""
        if name is None:
            return bool(self._regions) and self._regions != self._baseline_regions
        saved = layout_by_name(self._layouts, name)
        if saved is None:   # deleted meanwhile; the combo rebuild will drop the entry
            return bool(self._regions)
        return (self._regions != saved.regions()
                or self._ingame_band() != saved.ingame_band
                or self._clock_seconds() != saved.clock_seconds)

    def _confirm_close(self) -> bool:
        """Offer to save unsaved layout work before the dialog closes. Returns True to go
        ahead and close: nothing was unsaved, the user chose Discard, or the save went
        through. Cancel keeps the dialog open, and so does a Save the layout can't pass
        yet (no name, a required box missing): the status line then says what's missing,
        exactly as the Save button would."""
        if not self._has_unsaved_changes():
            return True
        name = self._preset_combo.currentData()
        text = (f"Save your changes to the layout {name!r} before closing?"
                if name is not None else "Save this new layout before closing?")
        answer = QMessageBox.question(
            self, WINDOW_TITLE, text,
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save)
        if answer == QMessageBox.StandardButton.Discard:
            return True
        if answer != QMessageBox.StandardButton.Save:
            return False   # Cancel, or the prompt dismissed with Escape
        saved = self._on_save()
        if not saved and not self._name_edit.text().strip():
            self._name_edit.setFocus()   # the usual missing piece on a new layout
        return saved

    def _confirm_preset_switch(self, previous: str | None, target: str) -> bool:
        """Offer to save unsaved layout work before the preset ``target`` is loaded over
        it. ``previous`` is the layout the work belongs to (None: '(new layout)'), since
        the combo already shows ``target``. Returns True to go ahead and load: nothing
        was unsaved, the user chose Discard, or the save went through. Cancel keeps the
        boxes, and so does a Save the layout can't pass yet (no name, a required box
        missing): the status line then says what's missing, as the Save button would."""
        if not self._unsaved_against(previous):
            return True
        # Save writes under the typed name, as the Save button does: say so when that is
        # not the layout being left (a retyped name saves a copy and leaves it as it was).
        typed = self._name_edit.text().strip()
        if previous is not None and typed.casefold() == previous.casefold():
            text = f"Save your changes to the layout {previous!r} before switching to {target!r}?"
        elif typed:
            text = f"Save your changes as the layout {typed!r} before switching to {target!r}?"
        else:
            text = f"Save this new layout before switching to {target!r}?"
        answer = QMessageBox.question(
            self, WINDOW_TITLE, text,
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save)
        if answer == QMessageBox.StandardButton.Discard:
            return True
        if answer != QMessageBox.StandardButton.Save:
            return False   # Cancel, or the prompt dismissed with Escape
        if not self._on_save():
            if not self._name_edit.text().strip():
                self._name_edit.setFocus()   # the usual missing piece on a new layout
            return False
        # The save reloaded the combo onto the layout just saved; go on to the picked one.
        self._show_preset_silently(target)
        return True

    # ── Teardown ────────────────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent | None) -> None:   # noqa: N802 (Qt override)
        """Escape cancels an armed Draw (see _cancel_draw) and never closes the dialog,
        which QDialog's own binding (Escape -> reject()) would: a stray Escape mid-drawing
        shouldn't end an editing session. The Close button and the window's close button
        still close it (through the save prompt)."""
        if event is not None and event.key() == Qt.Key.Key_Escape:
            self._cancel_draw()
            event.accept()
            return
        super().keyPressEvent(event)

    def done(self, result: int) -> None:
        """Every dialog exit (Close button, Save and continue, window close) funnels through
        done(): offer to save unsaved layout work first (Cancel keeps the dialog open),
        then release the frame source and stop playback exactly once here."""
        if not self._confirm_close():
            return
        self._teardown()
        super().done(result)

    def closeEvent(self, event: QCloseEvent | None) -> None:   # noqa: N802 (Qt override)
        """The window's own close button. QDialog routes a visible dialog's close through
        reject() and so done(), whose save prompt can keep it open (Cancel), so tear down
        only once the close has actually gone through."""
        super().closeEvent(event)
        if event is None or event.isAccepted():
            self._teardown()

    def _teardown(self) -> None:
        if self._torn_down:
            return
        self._torn_down = True
        self._settle_timer.stop()
        # A read still due would otherwise start after teardown: _start_read's guard passes
        # (the frame and boxes are still set), so it would respawn the OCR helper and make a
        # new scratch folder that nothing removes any more.
        self._read_timer.stop()
        if self._grabber is not None:
            self._grabber.cancel()   # don't leave an ffmpeg grab running
        if self._reader is not None:
            self._reader.stop()   # don't leave the OCR helper process running
        if self._scratch_dir is not None:
            shutil.rmtree(self._scratch_dir, ignore_errors=True)
            self._scratch_dir = None
            self._scratch_frames.clear()
        self._converter.cancel_active()   # don't leave an ffmpeg conversion running
        self._player.stop()
        if self._source is not None:
            self._source.release()
            self._source = None
