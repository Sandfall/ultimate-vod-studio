"""The Analyze page's layout preview: scrub the chosen VOD and see what the run will read.

The idle state of the Preview box (``vodparse.gui.preview.PreviewProgressPane``) is this
pane: a ``TrimPlayer`` scrubbing the whole VOD, with the layout editor's drawing pane
(``gui.region_pane.RegionDrawPane``) installed as its alternate display page. Whenever
the playhead settles, on the scrubber being let go or playback pausing, the frame under
it is grabbed asynchronously (``gui.frame_grabber.FrameGrabber``, ffmpeg off the UI
thread) and shown as a still. Scrubbing or playing again dismisses the still back to
live video, so checking a layout against a stream is a matter of dragging the scrubber
and looking.

Every still is then *read* the way analysis would read it, through the warm OCR helper
process (``gui.overlay_read_client.OverlayReadProcess``, torch never in this process):
the preset's boxes under a custom layout, or the pipeline's own band read and tag vote
under Auto. What each region read as appears in the readout under the player (the
"details", which a button on the Play row shows and hides), and a box is drawn on the
frame wherever something was read: the box the finder chose for an Auto read, a locked
row, the clock's box when a clock parsed. Under Auto a region that read nothing gets no
box, and nothing is drawn before the read lands, so the frame carries only what analysis
would find on it. A drawn layout's own boxes are the exception: every one of them is
on the frame from the moment the still is up, solid once it read something and a
sparse dotted outline while it has not (before the read lands, or when the read came
back empty), so a layout that sits on nothing is seen to sit on nothing rather than
look unread. A run under a drawn layout also keeps reading, with the built-in readers,
every field the layout draws no box for (``preview_regions.preview_background_keys``:
the tags without player boxes, the round and best-of without theirs, the in-game names
while the bottom band is read, the clock at its calibrated box without a timer box);
the still is read for those too, in the same request, and the readout lists them
with an asterisk on the row name (explained by a note under the rows), boxed on the
frame where they were found,
so what the readout shows is what Review will show after the run. The OCR engine
follows the Analyze page's Font style combo live
(``set_engine``; the saved setting only changes when a run starts), the language is
the run's own (Settings).

A freshly chosen VOD is captured at its first frame the moment it loads (the live
player paused at zero paints nothing on macOS, and the clock search below can take a
while), so the box is never blank once a video is in. The clock box, and the first
in-game timestamp the preview lands on, come from the clock search
(``gui.clock_locate.ClockLocator``, owned by the main window and fed in through
``set_clock``): when its answer arrives and the user has not moved the playhead
themselves, the player seeks to the first sampled frame where the search *read* the
clock at its calibrated box and captures it, so the VOD then opens on an in-game frame
with the clock marked and read. A search that detected a clock but read it on none of
its sampled frames (or found no steady spot to read it at) lands nowhere: the preview
stays on the first frame, and the status line says why.

For a game whose player names sit in bars beside the picture (Melee), the rows an Auto
run would lock come from the player-name search (``gui.side_overlay_locate.
SideOverlayLocator``, also owned by the main window, fed in through ``set_side_overlay``):
once it answers, the locked rows are read on every still, as the run reads them, and
drawn in the players' colours where they read a name.

A read waits for the searches it depends on. The still itself is shown the moment it is
captured (with a drawn layout's dotted boxes), but it is not sent to the helper until
the clock search has answered and, for a side-bar game under Auto, the player-name
search has too: an Auto read of an intro frame with no clock box and no rows is the
tag finder voting on a title card, and the readout would show those "tags" until the
answers arrived and the still was read again. Both answers re-read the still when they
land, so a held read goes out then; a clock search that fails releases the hold (no rows
will come), and another game forgets both answers and holds again until its own land.
The readout's values show a bare "…" meanwhile (READOUT_CALIBRATING), under the bar above
the player that says what is going on.

The capture flow mirrors the editor's (``gui.overlay_editor``: a token per grab, stale
grabs dropped, a scrub cancelling a grab in flight), with the Capture button replaced by
the settle timer: a short delay after the release/pause lets the player paint before
the grab, and lets a slider release that is immediately followed by another drag start
cancel cleanly. Import this lazily (it pulls in cv2 through the drawing pane and
QtMultimedia through the player), as the editor is.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPalette, QPen, QPixmap, QShowEvent
from PyQt6.QtWidgets import QGridLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

from vodparse.cli.clock_locate import ClockLocateResult
from vodparse.cli.side_overlay_locate import SideOverlayLocateResult
from vodparse.frame_reader import VideoFrameSource
from vodparse.games import INGAME_BAND_NONE, Game, ingame_band_engine_for
from vodparse.gui.common import ElidingLabel
from vodparse.gui.frame_grabber import SETTLE_DELAY_MS, FrameGrabber
from vodparse.gui.gpu_support import site_swap_in_flight
from vodparse.gui.image_pane import make_zoom_buttons
from vodparse.gui.overlay_read_client import OverlayReadProcess, RegionReading, tag_display_text
from vodparse.gui.paths import pipeline_json_for
from vodparse.gui.player.trim_player import TrimPlayer
from vodparse.gui.region_pane import RegionDrawPane
from vodparse.gui.settings import (OCR_ENGINE_DEFAULT_GUI, OCR_LANG_DEFAULT_GUI,
                                   PREVIEW_SHOW_DETAILS_DEFAULT, SETTING_OCR_LANG,
                                   SETTING_PREVIEW_SHOW_DETAILS)
from vodparse.ocr_engine import engine_unusable_reason
from vodparse.overlay_layout import (REGION_INGAME_P1, REGION_INGAME_P2, REGION_KEYS,
                                     REGION_PLAYER1, REGION_PLAYER2, REGION_TIMER,
                                     CustomOverlayLayout, ingame_band_mode_for)
from vodparse.overlay_regions import SPONSOR_REGION_FOR_PLAYER
from vodparse.preview_regions import (PREVIEW_LABELS, REGION_RGB, clock_read_key,
                                      preview_background_keys, preview_drawn_keys,
                                      preview_label, preview_read_keys)
from vodparse.schema import BBox
from vodparse.side_overlay import SIDE_OVERLAY_NOT_FOUND

# The drawing pane's own floor (ImagePane: 640x360) is lowered to what the Preview box's
# plain label had, so the main window's minimum size does not grow with this pane.
PREVIEW_MIN_WIDTH = 480
PREVIEW_MIN_HEIGHT = 270

# The readout lays its swatch/name/value triples out this many to a row, so it takes a
# couple of lines under the video rather than one per region. Each value asks for its full
# text's width, so a long value (a round banner, a sponsor and tag) takes the room its short
# neighbours leave instead of an equal third of the line; only when the line is truly out
# of room does a value elide, never below this many characters (the full text rides along
# as the tooltip), so a long read never widens the Preview box.
#
# The floor also sets the readout's share of the main window's minimum width, three values
# to a line: at 12 characters, showing the readout pushed that minimum past the window's
# 1150 px opening width on the Mac (1193 px under Ultimate, 1239 px under Rivals 2,
# measured 2026-10-18), so the window grew whenever it was shown.
READOUT_PAIRS_PER_ROW = 3
READOUT_COLUMNS_PER_PAIR = 3   # swatch, name, value
READOUT_MIN_CHARS = 6

# The two sides' rows, one readout line each, Player 1's side first: the overlay tags,
# then the in-game names under them, so each in-game name sits right below its side's tag
# (see readout_cells). Every other region fills the slots these leave.
READOUT_SIDE_LINES: tuple[tuple[str, str], ...] = (
    (REGION_PLAYER1, REGION_PLAYER2),
    (REGION_INGAME_P1, REGION_INGAME_P2),
)

# Each row's swatch: a filled square in the colour of the box that region is drawn in on
# the frame (the boxes carry no captions, so the colour is what ties a row to its box),
# with a faint border in the text colour so a light swatch still shows on a light theme.
READOUT_SWATCH_PX = 10
READOUT_SWATCH_BORDER_ALPHA = 120

NO_VIDEO_HINT = "Choose a video to preview the overlay layout on it"

# The one status line under the player. It shows only what the user needs telling: a
# capture or read that failed, or a search whose outcome changes how the run will read
# the video. An ordinary capture, read or found clock says nothing: the readout and the
# boxes on the frame already show it.
# A running search's progress is shown on a bar above the player instead
# (SEARCH_BANNER_STYLE): the line under the player is easy to miss, and a user who has
# just chosen a video is looking at the picture. The bar sits above the video in the
# layout rather than over it (the picture shrinks a little while it shows), because the
# live video widget wraps a native window that no ordinary widget can be drawn over.
# Only the progress goes there; a failure or an outcome note is read, not glanced at,
# and belongs on the line.
STATUS_READ_FAILED = "Couldn't read the tags: {message}"
STATUS_GRAB_FAILED = "Couldn't grab a frame: {message}"
STATUS_UNREADABLE = "Couldn't read the video."
STATUS_CLOCK_UNSTEADY = "Couldn't consistently detect the clock."
STATUS_CLOCK_NOT_FOUND = "No clock was found in the VOD."
STATUS_CLOCK_UNREAD = ("The clock was detected by the clock detector model, but its digits "
                       "couldn't be read (this shouldn't happen, file a bug report if you see "
                       "this message).")
STATUS_NAMES_NOT_FOUND = ("No player tags found in the sidebars. Assuming a \"standard\" "
                          "stream overlay (top of the screen).")
# Dark with bold white text whatever the theme, so it stands out from the controls around
# it; hidden (and taking no room) while no search runs.
SEARCH_BANNER_STYLE = "QLabel { background: #202020; color: white; padding: 6px 10px; }"
# What the bar says: a fixed lead-in (the user's words) and then the search's own
# progress text ("Finding the clock… 12/40") after a colon.
SEARCH_BANNER_TEXT = "Calibrating preview: {progress}"

READOUT_TITLE = "Tags read on this frame"
READOUT_EMPTY = "—"          # no still on show (or the read did not answer for the row)
READOUT_PENDING = "reading…"
# The row-name mark for a field a drawn layout has no box for, read in the background by
# the run's built-in readers (preview_background_keys), and the note under the rows that
# explains it, shown while any marked row is. Under Auto every row is read that way and
# none is marked. A single character on purpose: names never elide, and a word-long
# suffix (" (auto)", ~40 px a name) pushed three pairs to a line past the window's
# 1150 px opening width on the Mac (measured 2026-10-25).
# The note wraps, so its length never widens the pane.
READOUT_AUTO_MARK = "*"
READOUT_AUTO_NOTE = "* Read automatically: this layout has no box for it."
# A read held until the searches have answered. The bar above the player says why
# ("Calibrating preview: …"), so each value only marks the wait, and short enough to show in
# full at the READOUT_MIN_CHARS floor: "calibrating…" elided to "calibrati…" in every cell at
# the window's 1150 px opening width (Mac, 2026-10-25).
READOUT_CALIBRATING = "…"
READOUT_NOT_DETECTED = "(not detected)"

# The button on the Play row that shows and hides the readout box; its text names what a
# press does.
DETAILS_HIDE_TEXT = "Hide details"
DETAILS_SHOW_TEXT = "Show details"

class LayoutPreviewPane(QWidget):
    """The scrubbing preview with the run's read regions drawn on the settled frame and
    read out under it (see the module docstring for the flow).

    Fed from the main window: ``set_video`` when a VOD is chosen, ``set_context`` whenever
    the Game or Overlay layout choice changes, ``set_engine`` whenever the Font style
    choice changes, ``set_clock`` / ``set_clock_status`` from the clock search,
    ``set_side_overlay`` / ``set_side_overlay_status`` from the player-name search;
    ``suspend``/``resume`` around a pipeline run (the Preview box shows the run's own
    frames then) and ``shutdown`` when the window closes. ``settings`` supplies the run's
    OCR language for the reads.

    ``start_worker=False`` builds the pane with no frame grabber and no OCR helper, for
    offscreen tests: grabs are injected straight into ``_on_frame_grabbed`` and readings
    into ``_on_regions_read``, the editor's testing scheme."""

    def __init__(self, settings: QSettings, *, start_worker: bool = True) -> None:
        super().__init__()
        self._settings = settings
        self._video_path: str | None = None
        self._captured: np.ndarray | None = None   # the still on show, if any
        self._captured_sec = 0.0
        self._game: Game | None = None
        self._layout: CustomOverlayLayout | None = None
        self._ingame_tag_frac: float | None = None
        self._engine: str = OCR_ENGINE_DEFAULT_GUI   # the Font style choice, see set_engine
        # That choice and why it couldn't read when last handed over (see _engine_state_now)
        self._engine_state: tuple[str, str | None] = (OCR_ENGINE_DEFAULT_GUI, None)
        self._clock: ClockLocateResult | None = None
        # Whether the clock search has answered for the video on show (None is an answer
        # too: a game with no clock detector). Lets set_clock tell a repeat of the answer
        # it has from the first one.
        self._clock_answered = False
        # The games whose clock answer has already landed the player on a readable frame
        # for the video on show (set_clock lands once per game).
        self._landed_games: set[str] = set()
        # The player-name search's answer for the video on show (None: not answered, or
        # nothing to search for under the current game and layout).
        self._side_overlay: SideOverlayLocateResult | None = None
        self._readings: dict[str, RegionReading] = {}   # the still's reads, per region
        # What the search bar and the status line are built from (see _refresh_status): each
        # search's progress text while it runs (the bar), and the last capture or read
        # failure (the status line).
        self._clock_progress = ""
        self._side_overlay_progress = ""
        self._capture_error = ""
        self._read_note = ""          # the helper's note on the last read (regions it left unread)
        self._grab_token = 0          # bumped per grab request; stale grabs are dropped
        self._grab_in_flight = False
        self._read_token = 0          # bumped per read request and whenever the still,
        self._read_in_flight = False  # the boxes or the clock change; stale reads dropped
        self._suspended = False       # a pipeline run is showing its own frames instead
        self._reread_when_shown = False   # a run ended: read the still again once on screen
        self._read_after_swap = False     # a read waited out a GPU-support install
        self._closed = False          # shutdown ran: nothing is read or grabbed any more
        # Whether the user has moved the playhead (a scrub, a play) since the video loaded:
        # the clock search's landing seek must never hijack a playhead the user placed.
        self._user_moved_playhead = False

        # ── The player with the drawing pane as its display page ──
        self._player = TrimPlayer(whole_vod_only=True)
        self._pane = RegionDrawPane()
        self._pane.setMinimumSize(PREVIEW_MIN_WIDTH, PREVIEW_MIN_HEIGHT)
        self._pane.set_hint_text(NO_VIDEO_HINT)
        self._player.set_display_page(self._pane)
        self._player.displayPageDismissed.connect(self._on_live_resumed)
        self._player.scrubbed.connect(self._on_scrubbed)
        self._player.scrubReleased.connect(self._on_scrub_released)
        self._player.playingChanged.connect(self._on_playing_changed)

        # The settle delay: armed by a release/pause, disarmed by any playhead move.
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(SETTLE_DELAY_MS)
        self._settle_timer.timeout.connect(self._on_settled)

        # The search bar above the player (see the STATUS_* notes above): a running
        # search's progress text, hidden otherwise.
        self._search_banner = QLabel()
        self._search_banner.setStyleSheet(SEARCH_BANNER_STYLE)
        banner_font = self._search_banner.font()
        banner_font.setBold(True)
        self._search_banner.setFont(banner_font)
        self._search_banner.hide()

        # One status line (see the STATUS_* notes above), hidden while it has nothing to say.
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.hide()

        # The readout: its own titled box under the player, holding a colour swatch, name
        # and value per region the run reads, READOUT_PAIRS_PER_ROW to a line with each
        # in-game name under its side's tag (readout_cells), filled from the still's read.
        # Always on show, with the rows the current game and layout would answer for
        # (preview_read_keys) and a dash where there is nothing yet: a readout that came
        # and went with each still made the pane grow and shrink under the user's
        # scrubbing. A row exists for every region key; _lay_out_readout places the
        # current context's, in order; the swatch matches the box on the frame.
        self._readout_box = QGroupBox(READOUT_TITLE)
        readout_layout = QVBoxLayout(self._readout_box)
        self._readout_grid = QGridLayout()
        readout_layout.addLayout(self._readout_grid)
        self._readout_auto_note = QLabel(READOUT_AUTO_NOTE)
        self._readout_auto_note.setWordWrap(True)
        self._readout_auto_note.hide()
        readout_layout.addWidget(self._readout_auto_note)
        self._readout_swatches: dict[str, QLabel] = {}
        self._readout_names: dict[str, QLabel] = {}
        self._readout_values: dict[str, ElidingLabel] = {}
        for key in REGION_KEYS:
            self._readout_swatches[key] = QLabel()
            self._readout_names[key] = QLabel(f"{PREVIEW_LABELS[key]}:")
            self._readout_values[key] = ElidingLabel(READOUT_EMPTY, min_chars=READOUT_MIN_CHARS,
                                                     natural_width=True)
        # The value columns take their text's width (natural_width above) and the room left
        # over goes to an empty column past the last pair: equal stretch on the value columns
        # held a long value (a round banner) to a third of the line while its short
        # neighbours sat half empty, eliding it early.
        spare_column = READOUT_PAIRS_PER_ROW * READOUT_COLUMNS_PER_PAIR
        self._readout_grid.setColumnStretch(spare_column, 1)
        self._lay_out_readout(list(REGION_KEYS))
        self._refresh_readout_swatches()

        # The zoom buttons sit on the player's Play row rather than a row of their own,
        # and the details button beside them. The readout starts hidden, and its last
        # shown/hidden state is remembered across launches; frames are read either way, since the boxes on the
        # frame come from the read.
        for button in make_zoom_buttons(self._pane):
            self._player.add_transport_widget(button)
        self._details_button = QPushButton()
        self._details_button.clicked.connect(self._toggle_details)
        self._player.add_transport_widget(self._details_button)
        self._show_details(self._settings.value(
            SETTING_PREVIEW_SHOW_DETAILS, PREVIEW_SHOW_DETAILS_DEFAULT, type=bool))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._search_banner)
        layout.addWidget(self._player, stretch=1)
        layout.addWidget(self._status_label)
        layout.addWidget(self._readout_box)

        # ── Frame grabber + OCR helper client (both QProcess-backed) ──
        self._grabber: FrameGrabber | None = None
        self._reader: OverlayReadProcess | None = None
        self._scratch_dir: Path | None = None   # temp dir for the frames handed to the helper
        self._scratch_frames: dict[int, Path] = {}   # those frames not yet answered, by read token
        if start_worker:
            self._grabber = FrameGrabber(self)
            self._grabber.frame_grabbed.connect(self._on_frame_grabbed)
            self._grabber.grab_failed.connect(self._on_grab_failed)
            self._reader = OverlayReadProcess(self)
            self._reader.regions_read.connect(self._on_regions_read)
            self._reader.failed.connect(self._on_read_failed)
            self._reader.read_noted.connect(self._on_read_noted)

        # No video yet: the (empty) drawing pane's hint explains the first step; loading
        # a video dismisses it back to live.
        self._player.show_display_page(True)

    # ── Inputs from the main window ─────────────────────────────────────────────────

    def set_video(self, path: str) -> None:
        """Load ``path`` for scrubbing and capture its first frame straight away, so the
        box shows the video at once rather than whatever the media backend paints for a
        player paused at zero (on macOS, nothing) until the clock search lands it on an
        in-game frame. An unopenable file leaves the previous video (if any) in place and
        says so in the status line. A video handed over during a run is not captured:
        the box belongs to the run until resume."""
        try:
            source = VideoFrameSource(path)
        except ValueError:
            self._set_capture_error(STATUS_UNREADABLE)
            return
        try:
            duration = source.duration_sec()
        finally:
            source.release()
        self._video_path = path
        self._cancel_grab()
        self._drop_still()
        self._forget_searches()
        self._landed_games = set()
        self._user_moved_playhead = False
        self._player.set_source(path)
        self._player.set_region(0.0, duration)   # scrub the whole video
        self._clock_progress = ""
        self._side_overlay_progress = ""
        self._set_capture_error("")
        if not self._suspended:
            self._request_capture()   # the playhead stands at 0: the opening frame

    def set_context(self, game: Game, layout: CustomOverlayLayout | None,
                    ingame_tag_frac: float | None) -> None:
        """What the run would read with: the selected game, its selected overlay layout
        (None = the built-in "Auto" reading) and the Settings knob that widens Ultimate's
        bottom band. The still on show, if any, is redrawn and read again with them,
        without a re-grab. Another game forgets the searches' answers: its clock is
        another detector's, and until its own search answers (the main window asks at
        once) the read is held rather than made with the previous game's clock box."""
        if self._game is not None and game.id != self._game.id:
            self._forget_searches()
        self._game, self._layout, self._ingame_tag_frac = game, layout, ingame_tag_frac
        self._lay_out_readout(self._readout_keys())
        self._refresh_readout_swatches()
        self._refresh_still()
        self._refresh_status()   # the player-name search applies under Auto only

    def set_engine(self, engine: str) -> None:
        """The Font style the run would read with: the Analyze page's combo as it stands,
        not the saved setting (which is written only when a run starts, so it names the
        engine of the *last* run). The still on show is read again when the engine or its
        readiness changes (see _engine_state_now): so also when the same docTR choice is
        handed over again once its models have been downloaded."""
        self._engine = engine
        state = self._engine_state_now()
        if state != self._engine_state:
            self._engine_state = state
            self._refresh_still()

    def set_clock(self, result: ClockLocateResult | None) -> None:
        """The clock search's answer for the current video and game (None for a game with
        no clock detector). When the user has not moved the playhead since the video
        loaded and the search read the clock somewhere, lands the player on that frame
        and captures it (that capture is the still that gets read: the one on show, the
        opening frame or wherever the search left it, is not read first, which would
        flash its tags). Otherwise draws the box on the still on show and reads the
        clock there.

        The answer this pane already has is delivered again whenever the search is asked
        anew for the same video and game: a run cancels the search, and the main window
        asks again when the Preview box comes back. Nothing changed then, so nothing is
        redrawn, read or re-landed (it used to re-read the still and re-grab the very
        frame on show)."""
        self._clock_progress = ""   # the search's "Finding the clock…" may be showing
        if self._clock_answered and result == self._clock:
            self._refresh_status()
            return
        self._clock = result
        self._clock_answered = True
        self._refresh_readout_swatches()
        self._refresh_status()
        landing_sec: float | None = None
        game = self._game
        if (result is not None and self._video_path is not None and game is not None
                and game.id not in self._landed_games
                and not self._user_moved_playhead and not self._suspended):
            landing_sec = result.first_read_sec
        if landing_sec is None or game is None:
            self._refresh_still()
            return
        # Once per game and video: switching games and back brings this answer again
        # (set_context forgets it), and the frame landed on then is still the right one.
        self._landed_games.add(game.id)
        self._player.seek_to(landing_sec)
        self._request_capture()

    def _forget_searches(self) -> None:
        """Forget the clock and player-name answers: a new video, or another game, whose
        own searches have yet to answer (a read is held until they do)."""
        self._clock = None
        self._clock_answered = False
        self._side_overlay = None

    def set_clock_status(self, text: str) -> None:
        """Show the clock search's progress text ("Finding the clock…") on the search bar
        above the player until its answer arrives through set_clock."""
        self._clock_progress = text
        self._refresh_status()

    def set_side_overlay(self, result: SideOverlayLocateResult | None) -> None:
        """The player-name search's answer for the current video, game and Font style
        (None: no answer, because the search is running anew or there is nothing to
        search for under the current game and layout). The rows it locked are drawn on
        the still on show and read there. The answer this pane already has changes
        nothing, as in set_clock."""
        self._side_overlay_progress = ""   # a search's progress text may be showing
        if result == self._side_overlay:
            self._refresh_status()
            return
        self._side_overlay = result
        self._refresh_still()
        self._refresh_status()

    def set_side_overlay_status(self, text: str) -> None:
        """Show the player-name search's progress text ("Finding player tags…") on
        the search bar above the player until its answer arrives through set_side_overlay."""
        self._side_overlay_progress = text
        self._refresh_status()

    def suspend(self) -> None:
        """A pipeline run is starting: hold the playhead still and stop capturing and
        reading (the Preview box shows the run's own frames until resume). A read in
        flight comes back stale, and the OCR helper is stopped so its models leave memory
        for the run; the first read after resume starts it again."""
        self._suspended = True
        self._settle_timer.stop()
        self._cancel_grab()
        self._player.pause()
        self.release_reader()

    def release_reader(self) -> None:
        """Stop the OCR helper so its process lets go of the models it loaded, for a run
        that needs the memory. A read in flight comes back stale; the next read starts a
        fresh helper."""
        self._read_token += 1
        self._read_in_flight = False
        if self._reader is not None:
            self._reader.stop()
            self._delete_answered_frames(self._read_token)   # a stopped helper reads none

    def release_for_site_swap(self) -> None:
        """A GPU-support install or uninstall is starting: stop the OCR helper, which holds
        the torch that install replaces (Windows refuses while a DLL from it is loaded: see
        gui.gpu_support.connect_site_swap). Reads wait until read_after_site_swap; a read
        this drops in flight is one of them."""
        if self._read_in_flight:
            self._read_after_swap = True
        self.release_reader()

    def read_after_site_swap(self) -> None:
        """The install or uninstall has ended: read the still whose read waited for it."""
        if self._read_after_swap:
            self._read_after_swap = False
            self._read_still_or_wait()

    def resume(self) -> None:
        """The run is over: capturing works again, and the still on show, if any, is read
        again (its read in flight was dropped at suspend, and the game or layout may have
        changed during the run), but only once this pane is on screen: a finished run
        lands on Review, and reading there would start the OCR helper for a pane nobody
        sees (showEvent reads it when the main page comes back). A no-op when not
        suspended: the Preview box calls this whenever it shows the preview, run or not."""
        if not self._suspended:
            return
        self._suspended = False
        self._reread_when_shown = True
        self._reread_if_shown()
        if self._captured is None and not self._grab_in_flight and not self._closed:
            # No still to re-read: the run started before the opening or landing grab
            # landed (suspend cancelled it), so capture the frame on show now. The read
            # of that still waits for the pane to be on screen, like the re-read above.
            self._request_capture()

    def showEvent(self, event: QShowEvent | None) -> None:
        """Read the still a finished run left unread, now that the pane is on screen."""
        super().showEvent(event)
        self._reread_if_shown()

    def _reread_if_shown(self) -> None:
        if self._reread_when_shown and self.isVisible():
            self._reread_when_shown = False
            self._refresh_still()

    def shutdown(self) -> None:
        """The window is closing: stop the grab, the OCR helper and the player, and drop
        the frames handed to the helper. Nothing is read after this, so a run stopped by
        the quit that ends in resume() starts no helper on the way out."""
        self._closed = True
        self._settle_timer.stop()
        self._cancel_grab()
        if self._reader is not None:
            self._reader.stop()
        if self._scratch_dir is not None:
            shutil.rmtree(self._scratch_dir, ignore_errors=True)
            self._scratch_dir = None
        self._scratch_frames.clear()
        self._player.stop()

    @property
    def player(self) -> TrimPlayer:
        """The scrubbing player, for the main window's playback shortcuts and tests."""
        return self._player

    # ── The settle → grab → still flow ──────────────────────────────────────────────

    def _on_scrubbed(self, _sec: float) -> None:
        """The scrubber moved: whatever grab was aimed at the old playhead is stale, and
        no capture is due until the drag ends (the still, if shown, is dismissed by the
        player itself on the press)."""
        self._user_moved_playhead = True
        self._settle_timer.stop()
        if self._grab_in_flight:
            self._cancel_grab()

    def _on_scrub_released(self, _sec: float) -> None:
        if self._suspended:
            return
        self._settle_timer.start()

    def _on_playing_changed(self, playing: bool) -> None:
        """Playback started (the playhead is moving: nothing to capture) or paused (the
        playhead settled: capture after the delay). A pause that is the capture's own
        (``_request_capture`` pauses before grabbing) or one arriving while a capture is
        already pending must not queue another capture."""
        if self._suspended:
            return
        if playing:
            self._user_moved_playhead = True
            self._settle_timer.stop()
            self._cancel_grab()
            return
        if self._grab_in_flight or self._settle_timer.isActive():
            return
        self._settle_timer.start()

    def _on_settled(self) -> None:
        if self._suspended:
            return
        self._request_capture()

    def _request_capture(self) -> None:
        """Start grabbing the playhead frame asynchronously (see FrameGrabber); the still
        appears when it lands in _on_frame_grabbed. The in-flight flag is raised before
        the pause so the pause's own playingChanged is ignored (see _on_playing_changed)."""
        if self._video_path is None:
            return
        self._grab_token += 1
        self._grab_in_flight = True
        self._player.pause()
        sec = self._player.playhead_sec()
        self._set_capture_error("")
        if self._grabber is not None:
            self._grabber.grab(self._video_path, sec, self._grab_token)

    def _on_frame_grabbed(self, frame: np.ndarray, sec: float, token: int) -> None:
        """A grab landed: show it with the regions drawn on, then read it. A stale token
        means the playhead moved on (or the video changed) while the grab was in flight;
        drop it."""
        if token != self._grab_token:
            return
        self._grab_in_flight = False
        self._captured = frame
        self._captured_sec = sec
        self._readings = {}
        self._pane.set_frame(frame)
        self._draw_boxes()   # a drawn layout's boxes, dotted until the read fills them
        self._player.show_display_page(True)
        self._read_still_or_wait()

    def _on_grab_failed(self, message: str, token: int) -> None:
        if token != self._grab_token:
            return
        self._grab_in_flight = False
        self._set_capture_error(STATUS_GRAB_FAILED.format(message=message))

    def _on_live_resumed(self) -> None:
        """The player dismissed the still (a scrub/play/seek moved the playhead, or a new
        video loaded): the boxes are redrawn on the next capture."""
        self._drop_still()
        self._set_capture_error("")

    def _cancel_grab(self) -> None:
        """Forget a grab in flight: its result comes back stale, and ffmpeg is stopped."""
        self._grab_token += 1
        self._grab_in_flight = False
        if self._grabber is not None:
            self._grabber.cancel()

    def _drop_still(self) -> None:
        """Forget the still and its reads (a read in flight comes back stale)."""
        self._captured = None
        self._readings = {}
        self._read_token += 1
        self._read_in_flight = False
        self._fill_readout(READOUT_EMPTY)

    def _refresh_still(self) -> None:
        """The boxes or the clock changed under the still on show: redraw and read again.
        While another page of the window is showing (Review, where Preferences… can still
        change what the run reads with) the read waits for showEvent, as a finished run's
        re-read does (see resume): reading at once would start the OCR helper, and load
        its models, for a pane nobody sees."""
        if self._captured is None:
            return
        self._readings = {}
        self._draw_boxes()
        self._read_still_or_wait()

    def _read_still_or_wait(self) -> None:
        """Read the still on show, unless another page of the window is showing: then
        the read waits for showEvent (a read in flight is dropped: it would answer for the
        old boxes). The one gate for every path that ends in a read, so a grab that lands
        while Review is up (the clock search's landing, say) starts no OCR helper there."""
        if self._is_behind_another_page():
            self._read_token += 1
            self._read_in_flight = False
            self._reread_when_shown = True
            return
        self._start_read()

    def _is_behind_another_page(self) -> bool:
        """Whether the window is on screen while this pane is not: another page of a
        stack above it is the one showing. A window not shown at all (start-up, before
        the first show) is not that case."""
        window = self.window()
        return window is not None and window.isVisible() and not self.isVisible()

    # ── Reading the still ───────────────────────────────────────────────────────────

    def _start_read(self) -> None:
        """Send the still off to the OCR helper, as the drawn boxes of the preset or as
        the Auto reading (see OverlayReadProcess). The frame goes as a lossless PNG in
        this pane's temp dir, so the helper reads exactly the captured pixels. Nothing is
        read before the main window has said which game the run is for, nor while a
        pipeline run holds the pane (resume reads the still then), nor after shutdown,
        nor while a search the read depends on is still running (see the module
        docstring; set_clock and set_side_overlay read the still when they land)."""
        if self._captured is None or self._game is None or self._suspended or self._closed:
            return
        if site_swap_in_flight():
            # A new helper would load the torch the install is replacing (a search landing
            # mid-download asks for a read): wait, showing the read as pending.
            self._read_token += 1
            self._read_in_flight = False
            self._read_after_swap = True
            self._show_readout_pending()
            return
        if self._searches_pending():
            # A read already in flight was made with answers now withdrawn (a search
            # running anew): dropped, or it would land over the held "…" with them.
            self._read_token += 1
            self._read_in_flight = False
            self._fill_readout(READOUT_CALIBRATING)
            return
        self._read_token += 1
        self._read_in_flight = True
        self._read_note = ""
        self._show_readout_pending()
        if self._reader is None:
            return
        frame_path = self._write_frame_png(self._read_token)
        if frame_path is None:
            self._read_in_flight = False
            self._set_capture_error(STATUS_READ_FAILED.format(
                message="couldn't save the frame for reading"))
            return
        engine = self._engine
        lang = self._settings.value(SETTING_OCR_LANG, OCR_LANG_DEFAULT_GUI, type=str)
        clock_bbox = (self._clock.bbox if self._clock is not None and self._clock.error is None
                      else None)
        if self._layout is None:
            self._reader.request_auto(engine, lang, str(frame_path), game=self._game.id,
                                      ingame_tag_frac=self._ingame_tag_frac,
                                      clock_bbox=clock_bbox, token=self._read_token,
                                      locked=self._locked_boxes(),
                                      pipe_evidence_run=self._stored_run_path())
        else:
            frame_height, frame_width = self._captured.shape[:2]
            drawn = preview_drawn_keys(self._layout)
            rects_px = {key: rect.to_pixels(frame_width, frame_height)
                        for key, rect in self._layout.regions().items() if key in drawn}
            # The in-game boxes are read with the game's pinned nameplate engine, as the run
            # reads them (Rivals 2: docTR).
            band_mode = ingame_band_mode_for(self._game, self._layout)
            ingame_engine = ingame_band_engine_for(self._game, band_mode)
            # The fields the run reads in the background for this layout (see the module
            # docstring), answered in the same request from the bands and the clock box.
            # The band mode goes along: the helper plans its band passes from it, as the run
            # plans them from the layout (a Melee or Other layout may read a band the game's
            # own Auto reading does not).
            background_keys = preview_background_keys(self._game, self._layout)
            background: dict[str, Any] | None = None
            if background_keys:
                background = {"game": self._game.id, "ingame_tag_frac": self._ingame_tag_frac,
                              "ingame_band": band_mode, "clock_bbox": clock_bbox,
                              "keys": background_keys}
            self._reader.request(engine, lang, str(frame_path), rects_px, self._read_token,
                                 ingame_engine=ingame_engine, background=background)

    def _stored_run_path(self) -> str | None:
        """The video's pipeline.json when it has an analysis: the Auto read borrows that
        run's pipe evidence (see cli.overlay_read.run_pipe_evidence). None otherwise."""
        if self._video_path is None:
            return None
        path = pipeline_json_for(self._video_path)
        return str(path) if path.exists() else None

    def _searches_pending(self) -> bool:
        """Whether a search this still's read depends on has yet to answer: the clock
        search (every video is asked for its clock; a game without a detector is answered
        at once with None), and the player-name search where it applies (the main window
        asks it once the clock has answered; its answer, found or not, is never None)."""
        if not self._clock_answered:
            return True
        if self._clock is not None and self._clock.error is not None:
            # The clock search could not run (it died, or the stage refused). The main
            # window asks the player-name search only after a clock answer it can place
            # the bars from, so nothing more is coming: the still is read without rows
            # rather than held on "…" for good.
            return False
        return self._side_overlay_applies() and self._side_overlay is None

    def _side_overlay_applies(self) -> bool:
        """Whether a run would search the side bars at all: under Auto, for a game with
        side bars. A custom layout's own boxes say where the names are instead."""
        return self._layout is None and self._game is not None and self._game.has_side_bars

    def _locked_boxes(self) -> dict[str, BBox] | None:
        """The rows the player-name search locked, when the run would read them: where
        the search applies, once it has answered with a lock. None otherwise (the players
        are then read from the top band, or from the layout's boxes)."""
        if (not self._side_overlay_applies() or self._side_overlay is None
                or self._side_overlay.record is None):
            return None
        return self._side_overlay.record.boxes

    def _engine_state_now(self) -> tuple[str, str | None]:
        """The Font style choice and why it can't read yet (None when it can: see
        ocr_engine.engine_unusable_reason). A read always uses the choice itself; while it
        can't read, the helper answers with the reason, shown on the status line, and
        nothing is read with Standard in its place."""
        return self._engine, engine_unusable_reason(self._engine)

    def _write_frame_png(self, token: int) -> Path | None:
        """Save the still for the helper, token-named so a superseded request the helper
        is still reading can never see a half-overwritten file. Each frame is deleted once
        its request is answered (see _delete_answered_frames); the whole temp dir goes
        away in shutdown."""
        if self._captured is None:
            return None
        if self._scratch_dir is None:
            self._scratch_dir = Path(tempfile.mkdtemp(prefix="vodparse-layout-preview-"))
        path = self._scratch_dir / f"frame_{token}.png"
        try:
            if not cv2.imwrite(str(path), self._captured):
                return None
        except cv2.error:
            return None
        self._scratch_frames[token] = path
        return path

    def _delete_answered_frames(self, token: int) -> None:
        """Delete the frame of every request up to ``token``, whose answer (or failure)
        just came in. The helper reads a request's frame before answering it and answers
        strictly in request order, one line each, so no request at or before ``token``
        will open its frame again; a helper killed or crashed on the way reads nothing
        more at all. Best-effort: shutdown removes the whole temp dir anyway."""
        answered = [written for written in self._scratch_frames if written <= token]
        for written in answered:
            path = self._scratch_frames.pop(written)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _on_regions_read(self, readings: dict[str, RegionReading], token: int) -> None:
        """The still's read came back: show each region's text in the readout and mark
        where it was read on the frame. A stale token (the still, the boxes or the clock
        changed while the helper was reading) is dropped, its frame deleted all the same."""
        self._delete_answered_frames(token)
        if token != self._read_token:
            return
        self._read_in_flight = False
        self._readings = readings
        self._draw_boxes()
        self._show_readout(readings)
        self._set_capture_error("")

    def _on_read_failed(self, message: str, token: int) -> None:
        self._delete_answered_frames(token)
        if token != self._read_token:
            return
        self._read_in_flight = False
        self._fill_readout(READOUT_EMPTY)
        self._set_capture_error(STATUS_READ_FAILED.format(message=message))

    def _on_read_noted(self, note: str, token: int) -> None:
        """The helper attached a note to the read whose readings follow: it left some
        regions unread and says why (the game reads its in-game names with an engine that
        can't read here). Shown on the status line until the next read."""
        if token == self._read_token:
            self._read_note = note
            self._refresh_status()

    # ── Drawing and status ──────────────────────────────────────────────────────────

    def _draw_boxes(self) -> None:
        """Draw a box on the still wherever its read found something (see the module
        docstring): each region whose reading has text, at the box the read reported (an
        Auto find, a locked row, the clock) or at the layout's own box for a drawn layout,
        whose reads carry none. A drawn layout's boxes the run reads are all on the frame
        regardless (preview_drawn_keys: not the in-game boxes a layout keeps under band
        mode "none", which nothing reads), dotted until their read has text. The clock's
        box takes the colour of whichever
        box the clock is read at, like its readout swatch. The boxes carry no captions:
        the readout says what was read where, and its swatches carry the boxes' colours.
        A no-op with no still, or before the main window has said which game the run is
        for."""
        if self._captured is None or self._game is None:
            return
        frame_height, frame_width = self._captured.shape[:2]
        layout_boxes: dict[str, BBox] = {}
        if self._layout is not None:
            drawn = preview_drawn_keys(self._layout)
            layout_boxes = {key: rect.to_pixels(frame_width, frame_height)
                            for key, rect in self._layout.regions().items() if key in drawn}
        locked = self._locked_boxes() or {}
        # Every drawn box starts dotted; a reading with text makes it solid below.
        boxes: dict[str, BBox] = dict(layout_boxes)
        dotted: set[str] = set(layout_boxes)
        for key, reading in self._readings.items():
            if not reading.text:
                continue
            bbox = reading.bbox if reading.bbox is not None else layout_boxes.get(key)
            if bbox is None:
                continue
            draw_key = clock_read_key(self._layout, self._clock) if key == REGION_TIMER else key
            boxes[draw_key] = bbox
            dotted.discard(draw_key)
            # A locked sponsor row is read into its player's reading: draw it when that
            # reading carries a sponsor.
            sponsor_key = SPONSOR_REGION_FOR_PLAYER.get(key)
            if sponsor_key is not None and sponsor_key in locked and reading.sponsor:
                boxes[sponsor_key] = locked[sponsor_key]
        self._pane.set_boxes(boxes, captions={}, dotted=dotted)

    def _readout_keys(self) -> list[str]:
        """The rows the readout shows: what a read would answer for under the current
        game and layout (every region before the main window has said which game)."""
        if self._game is None:
            return list(REGION_KEYS)
        return preview_read_keys(self._game, self._layout)

    def _fill_readout(self, text: str) -> None:
        """Put ``text`` in every row's value (a dash between stills, "reading…" during)."""
        for key in REGION_KEYS:
            self._readout_values[key].setText(text)

    def _show_readout_pending(self) -> None:
        self._fill_readout(READOUT_PENDING)

    def _show_readout(self, readings: dict[str, RegionReading]) -> None:
        """Fill the readout from the read; a row the read did not answer for (the clock
        before the search lands) shows a dash. The rows themselves stay as laid out for
        the context, so the pane's height does not move."""
        for key in REGION_KEYS:
            reading = readings.get(key)
            self._readout_values[key].setText(
                readout_text(key, reading) if reading is not None else READOUT_EMPTY)

    def _lay_out_readout(self, keys: list[str]) -> None:
        """Place the rows for ``keys`` in the grid where readout_cells puts them, and
        hide every other row; re-placing (rather than hiding in fixed cells) keeps the
        layout tight whatever subset the context reads. A row the run reads in the
        background under the current layout is named with READOUT_AUTO_MARK, and the note
        that explains the mark shows while any such row does; the two in-game rows are
        named after the band mode in force, as the layout editor names its rows
        (preview_regions.preview_label)."""
        background: set[str] = set()
        ingame_band = INGAME_BAND_NONE
        if self._game is not None:   # before the main window has said which game: no rows read
            background = set(preview_background_keys(self._game, self._layout))
            ingame_band = ingame_band_mode_for(self._game, self._layout)
        for key in REGION_KEYS:
            shown = key in keys
            mark = READOUT_AUTO_MARK if key in background else ""
            self._readout_names[key].setText(f"{preview_label(key, ingame_band)}{mark}:")
            for widget in self._readout_row(key):
                self._readout_grid.removeWidget(widget)
                widget.setVisible(shown)
        self._readout_auto_note.setVisible(any(key in background for key in keys))
        for key, (line, pair) in readout_cells(keys).items():
            first_column = pair * READOUT_COLUMNS_PER_PAIR
            for offset, widget in enumerate(self._readout_row(key)):
                self._readout_grid.addWidget(widget, line, first_column + offset)

    def _readout_row(self, key: str) -> tuple[QLabel, QLabel, ElidingLabel]:
        """One region's widgets in grid order: swatch, name, value."""
        return (self._readout_swatches[key], self._readout_names[key],
                self._readout_values[key])

    def _refresh_readout_swatches(self) -> None:
        """Colour each row's swatch like the box its region is drawn in on the frame:
        the region's own colour, except the clock row, which follows whichever box the
        clock is read at (the layout's timer box or the clock search's box)."""
        border = self.palette().color(QPalette.ColorRole.WindowText)
        border.setAlpha(READOUT_SWATCH_BORDER_ALPHA)
        for key in REGION_KEYS:
            colour_key = clock_read_key(self._layout, self._clock) if key == REGION_TIMER else key
            self._readout_swatches[key].setPixmap(
                swatch_pixmap(REGION_RGB[colour_key], border, READOUT_SWATCH_PX))

    def _set_capture_error(self, text: str) -> None:
        """Remember the last capture or read failure ("" clears it) and show it."""
        self._capture_error = text
        self._refresh_status()

    def _refresh_status(self) -> None:
        """Fill the bar above the player with a running search's progress text (hidden
        otherwise), and the one status line (see the STATUS_* notes) with a capture or
        read failure, else what a search found that the user should know (nothing for
        an ordinary find; hidden when empty). The player-name search's progress counts
        only where that search applies: the main window cancels the search when a custom
        layout or a game without side bars is chosen, and nothing comes back from a
        cancelled search to clear its text."""
        side_progress = self._side_overlay_progress if self._side_overlay_applies() else ""
        progress = self._clock_progress or side_progress
        self._search_banner.setText(SEARCH_BANNER_TEXT.format(progress=progress) if progress else "")
        self._search_banner.setVisible(bool(progress))
        if self._capture_error:
            text = self._capture_error
        else:
            side_overlay = self._side_overlay if self._side_overlay_applies() else None
            notes = [clock_status_text(self._clock), side_overlay_status_text(side_overlay),
                     self._read_note]
            text = " ".join(note for note in notes if note)
        self._status_label.setText(text)
        self._status_label.setVisible(bool(text))

    # ── The details (readout) toggle ────────────────────────────────────────────────

    def _toggle_details(self) -> None:
        """The details button: flip the readout box and remember the choice."""
        shown = self._readout_box.isHidden()
        self._show_details(shown)
        self._settings.setValue(SETTING_PREVIEW_SHOW_DETAILS, shown)

    def _show_details(self, shown: bool) -> None:
        """Show or hide the readout box; the button's text names what a press does."""
        self._readout_box.setVisible(shown)
        self._details_button.setText(DETAILS_HIDE_TEXT if shown else DETAILS_SHOW_TEXT)


def swatch_pixmap(rgb: tuple[int, int, int], border: QColor, size_px: int) -> QPixmap:
    """A ``size_px`` square filled with ``rgb`` and outlined in ``border``."""
    pixmap = QPixmap(size_px, size_px)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setPen(QPen(border, 1))
    painter.setBrush(QColor(*rgb))
    painter.drawRect(0, 0, size_px - 1, size_px - 1)
    painter.end()
    return pixmap


def readout_cells(keys: list[str]) -> dict[str, tuple[int, int]]:
    """Where each of ``keys`` sits in the readout, as (line, pair slot) with
    READOUT_PAIRS_PER_ROW slots to a line. Each of READOUT_SIDE_LINES that ``keys`` has
    any of gets a line, Player 1's side in the first slot and Player 2's in the second,
    so an in-game name always sits right under its side's tag; a side the context does
    not read leaves its slot empty rather than pulling the other side across. The other
    regions (round, best-of, clock), in ``keys`` order, fill the slot after the sides on
    those lines, then flow gaplessly onto lines of their own below. Pure, so the
    arrangement is testable without a widget."""
    cells: dict[str, tuple[int, int]] = {}
    side_lines = [sides for sides in READOUT_SIDE_LINES if any(key in keys for key in sides)]
    free_slots: list[tuple[int, int]] = []
    for line, sides in enumerate(side_lines):
        for pair, key in enumerate(sides):
            if key in keys:
                cells[key] = (line, pair)
        free_slots.extend((line, pair) for pair in range(len(sides), READOUT_PAIRS_PER_ROW))
    side_keys = {key for sides in READOUT_SIDE_LINES for key in sides}
    others = [key for key in keys if key not in side_keys]
    for index, key in enumerate(others):
        if index < len(free_slots):
            cells[key] = free_slots[index]
        else:
            overflow_line, pair = divmod(index - len(free_slots), READOUT_PAIRS_PER_ROW)
            cells[key] = (len(side_lines) + overflow_line, pair)
    return cells


def readout_text(key: str, reading: RegionReading) -> str:
    """One readout row's text: what was read (a player row shows the sponsor split off
    the tag as "Sponsor | Tag", tag_display_text; the clock row shows the parsed clock),
    or READOUT_NOT_DETECTED when the region read nothing or no clock parsed. Pure, so
    the wording is testable without a widget."""
    if not reading.text:
        return READOUT_NOT_DETECTED
    if key == REGION_TIMER:
        return reading.text
    return tag_display_text(reading)


def clock_status_text(result: ClockLocateResult | None) -> str:
    """What the status line says about a clock search's answer: nothing for a game with
    no clock detector and for an ordinary find; the stage's own reason when it could not
    run; the two outcomes that leave the run without a fixed clock spot; and a spot that
    was found but read on no sampled frame (the preview then stays on the first frame).
    Pure, so the wording is testable without a widget."""
    if result is None:
        return ""
    if result.error is not None:
        return result.error
    if result.first_hit_sec is None:
        return STATUS_CLOCK_NOT_FOUND
    if result.bbox is None:
        return STATUS_CLOCK_UNSTEADY
    if result.first_read_sec is None:
        return STATUS_CLOCK_UNREAD
    return ""


def side_overlay_status_text(result: SideOverlayLocateResult | None) -> str:
    """What the status line says about a player-name search's answer: nothing when there
    is none and when the names were found (in the side bars or the top band); the
    stage's own reason when it could not run; and that nothing was found, since the run
    then reads the top band. Pure, so the wording is testable without a widget."""
    if result is None:
        return ""
    if result.error is not None:
        return result.error
    if result.record is not None and result.record.status == SIDE_OVERLAY_NOT_FOUND:
        return STATUS_NAMES_NOT_FOUND
    return ""
