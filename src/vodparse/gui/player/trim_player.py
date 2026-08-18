"""A lightweight video trim player for the Review screen.

Wraps QMediaPlayer + QVideoWidget over the *source* vod and lets the user scrub to find a
clip's true start/end and stamp them from the playhead. Boundaries are absolute source-video
seconds (the same space the detector and exporter use), so seeking is a direct setPosition().
Stream-copy/keyframe seeking is ~1s-granular, which is fine for human trimming.
"""

import io
import sys
import time
from os import environ
from os.path import basename
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QShowEvent
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from vodparse.gui.player.seek_slider import SeekSlider
from vodparse.gui.review.model import hms
from vodparse.gui.widgets import pin_windows_symbol_font

# ── Diagnostic trace (opt-in) ───────────────────────────────────────────────────
# The intermittent black-frame / scrub-lag behaviour only reproduces on the macOS AVFoundation
# backend, which a headless CI environment can't run (no working video backend). Set
# TRIMPLAYER_TRACE to record the media-player signal sequence + state on every load, seek, and
# scrub, so the real ordering can be captured on the user's computer and compared black-run vs
# working-run. Values:
#   TRIMPLAYER_TRACE=1        → append to debug_traces/trim_trace.log next to this file
#   TRIMPLAYER_TRACE=stderr   → write lines to stderr
#   TRIMPLAYER_TRACE=<path>   → append to that file
# Unset (the default) makes tracing a no-op, so normal runs pay nothing.
_TRACE_SETTING = environ.get("TRIMPLAYER_TRACE", "").strip()
_TRACE_ENABLED = bool(_TRACE_SETTING)
_TRACE_TO_STDERR = _TRACE_SETTING.lower() == "stderr"
# Repo-relative default so the trace lands somewhere predictable, not the system temp dir.
_TRACE_DEFAULT_FILE = Path(__file__).resolve().parent / "debug_traces" / "trim_trace.log"
_TRACE_FILE: Path | None = None
if _TRACE_ENABLED and not _TRACE_TO_STDERR:
    _TRACE_FILE = (_TRACE_DEFAULT_FILE
                   if _TRACE_SETTING.lower() in ("1", "true", "on", "yes", "file")
                   else Path(_TRACE_SETTING))
# High-frequency seams (playback position, scrubbing) throttle their trace lines so a fast drag
# or steady playback doesn't flood the log; or, more importantly, doesn't lag the UI. The priming
# kick is logged unthrottled, since its few ticks are the point.
_POS_TRACE_MIN_INTERVAL_SEC = 0.2
_SCRUB_TRACE_MIN_INTERVAL_SEC = 0.1

_trace_handle: io.TextIOWrapper | None = None   # cached append handle for _TRACE_FILE


def _trace(event: str, **fields: object) -> None:
    """Append one timestamped diagnostic line when TRIMPLAYER_TRACE is set; otherwise do nothing.

    The timestamp is a monotonic clock in seconds; relative timing is what matters for reading
    the sequence. ``event`` names the seam; ``fields`` are the values to record. The file handle
    is opened once and kept open: opening/closing (and mkdir-checking) per line made a fast scrub
    lag the UI. Each write is flushed so the file stays readable live."""
    if not _TRACE_ENABLED:
        return
    parts = [f"{time.monotonic():10.3f}", f"{event:<16}"]
    parts += [f"{key}={value}" for key, value in fields.items()]
    line = " ".join(parts)
    if _TRACE_FILE is None:
        print(line, file=sys.stderr, flush=True)
        return
    global _trace_handle
    if _trace_handle is None:
        _TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _trace_handle = _TRACE_FILE.open("a", encoding="utf-8")
    _trace_handle.write(line + "\n")
    _trace_handle.flush()


# A session marker so multiple app launches appended to the same file stay separable.
if _TRACE_ENABLED:
    _trace("session.start", target=str(_TRACE_FILE) if _TRACE_FILE is not None else "stderr")


# How much context to show on each side of the current clip when not viewing the whole vod:
# the fallback default used until the user saves their own (see SETTING_CLIP_VIEW_MARGIN_SEC).
CLIP_VIEW_MARGIN_SEC = 120.0
# Persisted override for the above, written by the Edit-preview-bounds "Save as default" tick so
# a chosen margin carries to new vods/sessions; absent until then, leaving the constant in force.
SETTING_CLIP_VIEW_MARGIN_SEC = "review/clip_view_margin_sec"


def _fmt(ms: int) -> str:
    """Milliseconds → HH:MM:SS for the transport labels (negatives clamp to 0)."""
    return hms(max(0, ms) // 1000)


class TrimPlayer(QWidget):
    """Emits startChanged/endChanged (absolute seconds) when the user stamps a boundary."""

    startChanged = pyqtSignal(float)
    endChanged = pyqtSignal(float)
    scrubbed = pyqtSignal(float)   # absolute seconds, only on user click/drag of the scrubber
    fullViewChanged = pyqtSignal(bool)   # whole-VOD view toggled (keeps the View-menu item in sync)
    displayPageDismissed = pyqtSignal()   # the alternate display page was auto-hidden by a
    #                                       playhead move (scrub/jump/play, or a new
    #                                       source/region/seek); see set_display_page

    def __init__(self, view_margin_sec: float = CLIP_VIEW_MARGIN_SEC,
                 settings: QSettings | None = None) -> None:
        super().__init__()
        self._source: str | None = None
        self._start_sec = 0.0
        self._end_sec = 0.0

        # The per-side preview margin: a saved default (Edit preview bounds → "Save as default")
        # wins over the built-in fallback, so a new vod opens with the user's preferred context.
        self._settings = settings
        if settings is not None:
            # QSettings.value(..., type=float) only falls back to the default when the key is
            # MISSING. A present-but-unparseable value (an empty string, a hand-corrupted INI line,
            # a half-written setting) coerces to 0.0 instead; which would collapse the preview to
            # zero context (the clip with no lead-in/tail) with no error. Read raw and cast ourselves
            # so a corrupt stored value keeps the fallback default.
            raw = settings.value(SETTING_CLIP_VIEW_MARGIN_SEC, view_margin_sec)
            try:
                view_margin_sec = float(raw)
            except (TypeError, ValueError):
                pass
        self._margin_ms = int(view_margin_sec * 1000)
        self._duration_ms = 0
        self._view_start_ms = 0
        self._view_end_ms = 0
        self._freeze_view = False   # keep the current extent across a rebuild (Split match)
        self._full_view = False
        self._markers_sec: list[float] = []   # current set's match boundaries (hidden in full view)
        # Paused-frame rendering. The macOS backend only paints a seeked frame while the player is
        # in PausedState; after priming it idles PausedState → StoppedState after ~1.3s, and in
        # StoppedState a seek moves the position but paints nothing. So every seek (select/scrub)
        # goes through _render_at, which re-primes (a brief play→pause "kick") back to Paused when
        # the player has gone Stopped, then lands on the target once Paused.
        self._primed = False    # has at least one prime completed for this source?
        self._priming = False   # a play→pause kick is in flight
        self._pending_seek_ms = 0   # position to land on once the player is renderable (Paused)
        self._last_pos_trace = 0.0     # monotonic time of the last throttled position trace line
        self._last_scrub_trace = 0.0   # monotonic time of the last throttled scrub trace line

        self._video = QVideoWidget()
        # A small floor so the video never collapses to nothing, kept modest so it doesn't crowd
        # small screens. This is safe now that the Review window's minimum tracks its content (see
        # gui._apply_min_size): the floor just raises that minimum a little, so the window stops
        # shrinking before the video would overlap the scrubber/transport rows below it; rather
        # than pinning the video tall and letting its native surface paint over them.
        self._video.setMinimumSize(320, 120)
        self._build_player()

        # A host can install one alternate page for the display area (set_display_page)
        # that swaps in place of the video; the spot-check tool shows its analyzed still
        # there; while the scrubber/transport rows below stay visible and usable in both
        # states. The QStackedWidget that makes the swap possible is created lazily, on
        # the first set_display_page call, so a player that never hosts a page (the
        # shipped Review screen) keeps the video widget directly in the layout; the
        # exact pre-stack widget tree, leaving no chance of the extra container touching
        # the macOS video backend's compositing/priming behavior.
        self._display_page: QWidget | None = None
        self._display_stack: QStackedWidget | None = None

        self._slider = SeekSlider(Qt.Orientation.Horizontal)
        self._slider.sliderMoved.connect(self._on_scrub)
        # Ensure the player can paint seeks once, when the scrubber is grabbed; keeps the
        # per-move scrub (sliderMoved → _on_scrub) a plain, lag-free setPosition.
        self._slider.sliderPressed.connect(self._on_scrub_pressed)
        self._time_label = QLabel("00:00:00 / 00:00:00")
        # Fixed width + right align so the slider doesn't resize as the digits change while scrubbing.
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._time_label.setFixedWidth(
            self._time_label.fontMetrics().horizontalAdvance("00:00:00 / 00:00:00") + 10)

        self._play_btn = QPushButton("▶ Play")
        self._play_btn.clicked.connect(self.toggle_play)
        # ⏸ included for the button's Pause state (toggle_play swaps its text). On Windows
        # these glyphs would otherwise fall back to multicolor Segoe UI Emoji; pin the flat
        # symbol font, matching the header icon buttons (see widgets.pin_windows_symbol_font).
        # Jumping to the clip's start/end stayed keyboard-only (Ctrl+Home/Ctrl+End actions on
        # the Review screen call jump_to_start/jump_to_end); the transport buttons for them
        # were dropped as clutter.
        pin_windows_symbol_font([self._play_btn], "▶⏸")
        self._full_check = QCheckBox("Show whole VOD")
        self._full_check.toggled.connect(self._on_toggle_full)
        bounds_btn = QPushButton("Edit preview bounds…")
        bounds_btn.clicked.connect(self.edit_preview_bounds)

        transport = self._transport = QHBoxLayout()
        transport.addWidget(self._play_btn)
        transport.addWidget(self._full_check)
        transport.addWidget(bounds_btn)
        transport.addStretch(1)

        scrub = QHBoxLayout()
        scrub.addWidget(self._slider, stretch=1)
        scrub.addWidget(self._time_label)

        layout = self._root_layout = QVBoxLayout(self)
        layout.addWidget(self._video, stretch=1)
        layout.addLayout(scrub)
        layout.addLayout(transport)

    def showEvent(self, event: QShowEvent | None) -> None:
        super().showEvent(event)
        self._trace_state("showEvent")
        # Prime the pending frame once we're visible whenever the player can't currently paint one
        #; the cold start (media loaded while this widget was on a hidden stack page) and re-entry
        # (same vod, but the backend has idled the player to Stopped) both land here in Stopped.
        # _kick is a no-op if already renderable, mid-kick, unloaded, or off screen.
        if not self._is_renderable():
            self._kick()

    # ── Player lifecycle ────────────────────────────────────────────────────────
    def _build_player(self) -> None:
        """Create the QMediaPlayer (+ audio out), point it at the video widget, and wire its
        signals to our slots. Factored out of __init__ so set_source can hand each vod a fresh
        player rather than reusing one across sources (see _rebuild_player)."""
        self._player = QMediaPlayer()
        self._audio = QAudioOutput()
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._video)
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.mediaStatusChanged.connect(self._on_media_status)
        self._player.playbackStateChanged.connect(self._on_playback_state)

    def _rebuild_player(self) -> None:
        """Replace the current player with a fresh one before loading a different vod.

        The macOS 'darwin' media backend attaches a single AVPlayerItemVideoOutput to the
        player's item; loading a second source through the same QMediaPlayer re-attaches the
        still-attached output and aborts the app with an uncaught NSException
        (-[AVPlayerItem addOutput:] 'Cannot attach an output that is already attached'). Stopping,
        detaching the video output, and clearing the source release it before the new player
        claims the widget."""
        old = self._player
        old.stop()
        old.setVideoOutput(None)
        old.setSource(QUrl())
        old.deleteLater()
        self._build_player()

    # ── Public API ────────────────────────────────────────────────────────────
    def set_source(self, video_path: str) -> None:
        """Load the source vod; no-op if it's already loaded. Switching to a *different* vod
        rebuilds the player first (the darwin backend can't reuse one player across sources).

        Dismisses any alternate display page: the page belongs to a frame of the *old* vod, so
        it must not stay painted over the new one (see _dismiss_display_page)."""
        if video_path == self._source:
            return
        _trace("set_source", file=basename(video_path))
        self._dismiss_display_page()
        if self._source is not None:
            self._rebuild_player()
        self._source = video_path
        self._primed = False
        self._priming = False
        self._pending_seek_ms = 0
        self._player.setSource(QUrl.fromLocalFile(video_path))
        self._trace_state("set_source.done")

    def set_region(self, start_sec: float, end_sec: float) -> None:
        """Point the player at a clip's [start, end], window the scrubber, and show the start frame.

        Routes through _render_at, so it renders live when the player is already paused and
        re-primes first when the backend has idled it to Stopped. Moving to the start frame is a
        playhead move, so any alternate display page is dismissed first."""
        self._dismiss_display_page()
        self._start_sec = start_sec
        self._end_sec = end_sec
        self._apply_view()
        self._trace_state("set_region", start=start_sec, end=end_sec)
        self._render_at(int(start_sec * 1000))

    def stop(self) -> None:
        self._player.stop()

    def playhead_sec(self) -> float:
        """The current playhead position in absolute source-video seconds. Tracked by us
        (_pending_seek_ms) rather than read from the player, which reports a stale/0 position
        once the backend has idled the source to Stopped."""
        return self._pending_seek_ms / 1000.0

    def seek_to(self, sec: float) -> None:
        """Move the playhead to an absolute second within the current window (no region change).
        Dismisses any alternate display page first, like every other playhead move."""
        self._dismiss_display_page()
        self._seek_sec(sec)

    def add_transport_widget(self, widget: QWidget) -> None:
        """Place an external control at the right end of the transport (playback) row, so a
        caller can fold a related action into the player's button row instead of giving it a
        row of its own. The Review screen uses this for its Reset-all-edits button."""
        self._transport.addWidget(widget)

    # ── Alternate display page ───────────────────────────────────────────────────
    # The video area can host one caller-supplied page (the spot-check tool's analyzed still).
    # Any playhead move auto-dismisses it back to the live video and emits displayPageDismissed,
    # so the host can drop state tied to the old frame. That covers the user-driven moves
    # (scrubber grab, jump, play) and the programmatic ones (set_source, set_region, seek_to):
    # a new vod or a new region leaves the page showing a frame that is no longer on screen.
    # The dismissal seams are all one-shot events; the per-move scrub hot path never touches
    # this. The Review screen installs no page, making all of it a no-op there.
    def set_display_page(self, widget: QWidget) -> None:
        """Install an alternate page for the video display area. One page is supported;
        calling again replaces the previous page. The page starts hidden; call
        show_display_page(True) to swap it in front of the live video.

        The first call wraps the video widget in the QStackedWidget that makes the swap
        possible; until then the video sits directly in the layout (see __init__), so a
        player that never hosts a page carries no stack at all."""
        if self._display_stack is None:
            self._display_stack = QStackedWidget()
            self._root_layout.removeWidget(self._video)
            self._display_stack.addWidget(self._video)
            self._root_layout.insertWidget(0, self._display_stack, stretch=1)
        if self._display_page is not None:
            self._display_stack.removeWidget(self._display_page)
        self._display_page = widget
        self._display_stack.addWidget(widget)

    def show_display_page(self, on: bool) -> None:
        """Show the installed alternate page in place of the live video (True) or restore the
        video (False). A host-initiated flip; unlike the auto-dismiss on a playhead move, it
        does not emit displayPageDismissed. No-op when no page is installed."""
        if self._display_page is None or self._display_stack is None:
            return
        self._display_stack.setCurrentWidget(self._display_page if on else self._video)

    def _dismiss_display_page(self) -> None:
        """Auto-hide the alternate page because the playhead is about to move (scrub/jump/play,
        or a new source/region/seek), restoring the live video and emitting displayPageDismissed.
        No-op when no page is installed (the shipped Review screen; two attribute reads, off the
        scrub hot path's cost radar) or when the video is already showing; callers can therefore
        call it unconditionally."""
        if (self._display_page is None or self._display_stack is None
                or self._display_stack.currentWidget() is self._video):
            return
        self._display_stack.setCurrentWidget(self._video)
        self.displayPageDismissed.emit()

    def set_markers(self, markers_sec: list[float]) -> None:
        """Match-boundary ticks (absolute seconds) for the current set. Hidden while the whole
        vod is shown, since they'd only apply to this one set."""
        self._markers_sec = list(markers_sec)
        self._apply_markers()

    def set_core_regions(self, regions_sec: list[tuple[float, float]]) -> None:
        """Match coverage (absolute seconds) inside the current set as (start, end) ranges; the
        lead-in/tail buffer; and mid-set gaps in debug; show a shade lighter."""
        self._slider.set_core_regions([(int(s * 1000), int(e * 1000)) for s, e in regions_sec])

    def set_other_regions(self, regions_sec: list[tuple[float, float]]) -> None:
        """Spans (absolute seconds) of other sets to mark in a deeper muted red. The scrubber
        clips them to whatever range it's currently showing, so pass them all; no need to
        recompute when the view margin or whole-VOD toggle changes."""
        self._slider.set_other_regions([(int(s * 1000), int(e * 1000)) for s, e in regions_sec])

    def _apply_markers(self) -> None:
        ms = [] if self._full_view else [int(s * 1000) for s in self._markers_sec]
        self._slider.set_markers(ms)

    # ── Paused-frame rendering (prime / re-prime) ───────────────────────────────
    # The macOS backend paints a seeked frame only while the player is in PausedState. After a
    # prime it idles PausedState → StoppedState after ~1.3s, and in StoppedState a seek moves the
    # position but paints nothing. So every seek goes through _render_at: it seeks live when the
    # player is renderable, and otherwise kicks it (a brief play→pause) back to Paused first,
    # landing on the target once there. _pending_seek_ms is the single source of truth for the
    # playhead; set by every seek/scrub and tracked through playback.
    def _is_playing(self) -> bool:
        return self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def _is_renderable(self) -> bool:
        """Whether a seek will actually paint a frame; true while Paused or Playing, false while
        Stopped (where the backend moves the position but renders nothing)."""
        return self._player.playbackState() in (
            QMediaPlayer.PlaybackState.PausedState,
            QMediaPlayer.PlaybackState.PlayingState,
        )

    def _media_ready(self) -> bool:
        """Whether the source is loaded enough to seek or prime."""
        return self._player.mediaStatus() in (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
            QMediaPlayer.MediaStatus.BufferingMedia,
        )

    def _render_at(self, ms: int) -> None:
        """Show the frame at absolute position ``ms``.

        Seeks live when the player is already paused/playing. When the backend has idled it to
        Stopped; where a seek paints nothing; it kicks the player back to a paused, renderable
        state first and lands on ``ms`` once there. When the media hasn't loaded yet, ``ms`` is
        just remembered and applied by the load-time prime."""
        self._pending_seek_ms = ms
        if self._priming:
            return   # a kick is in flight; it will land on this latest pending position
        if self._is_renderable():
            self._player.setPosition(ms)
        elif self._media_ready():
            self._kick()
        # else: media not loaded yet: _on_media_status / showEvent will prime and apply pending

    def _kick(self) -> None:
        """Bring the player into a paused, renderable state with a brief play→pause. Completion
        (seek to the pending position + unmute) lands in _on_playback_state when the Playing→
        Paused transition arrives. No-op while already kicking, unloaded, or off screen."""
        if self._priming or not self._media_ready() or not self.isVisible():
            self._trace_state("prime.skip")
            return
        self._trace_state("prime.kick")
        self._priming = True
        self._audio.setMuted(True)   # silence the brief kick
        self._player.play()

    def _on_scrub_pressed(self) -> None:
        """The scrubber was grabbed: kick the player out of an idle Stopped state so seeks paint,
        and dismiss any alternate display page (a grab precedes every click/drag seek, so the
        analyzed still is on its way out the moment the user touches the slider). Both are done
        once per grab (not per move), so the scrub hot path stays a plain setPosition."""
        self._trace_state("scrub.press")
        self._dismiss_display_page()
        if not self._is_renderable():
            self._kick()

    def _on_scrub(self, ms: int) -> None:
        """User scrubbed (click/drag): seek live and notify the review screen. A plain setPosition
        with no per-move state work; renderability and display-page dismissal are handled once,
        on press (_on_scrub_pressed), so this matches the original's smooth native scrubbing with
        no lag. _pending_seek_ms is kept current so an in-flight press-kick lands on the latest
        scrub position."""
        self._pending_seek_ms = ms
        self._player.setPosition(ms)
        self.scrubbed.emit(ms / 1000.0)
        self._trace_scrub(ms)

    def _on_media_status(self, status: QMediaPlayer.MediaStatus) -> None:
        self._trace_state("mediaStatus", status=status.name)
        # Cold start only: once the freshly-loaded media is ready, prime the first frame at the
        # pending position. Guarded by `not primed` because the backend's own idle Paused→Stopped
        # also fires mediaStatus; auto-re-priming on that would loop; re-priming after an idle
        # instead happens on the next user interaction, via _render_at.
        if not self._primed and self._media_ready():
            self._kick()

    def _on_playback_state(self, state: QMediaPlayer.PlaybackState) -> None:
        self._trace_state("playbackState", state=state.name)
        if self._priming:
            if state == QMediaPlayer.PlaybackState.PlayingState:
                self._player.pause()   # reached Playing; settle into a renderable PausedState
            elif state == QMediaPlayer.PlaybackState.PausedState:
                # Genuinely paused now: land on the pending position, which paints the frame.
                self._trace_state("prime.finish")
                self._priming = False
                self._primed = True
                self._player.setPosition(self._pending_seek_ms)
                self._audio.setMuted(False)
        self._update_play_btn()

    # ── View window ───────────────────────────────────────────────────────────
    def _apply_view(self) -> None:
        """Set the scrubber range to the current clip ± margin (or the whole vod). While
        _freeze_view is set the extent is held as-is (the clip region still redraws), so a
        Split-match rebuild doesn't refill a margin the user shrank with Mark start/end."""
        if not self._freeze_view:
            if self._full_view or self._duration_ms <= 0:
                self._view_start_ms = 0
                self._view_end_ms = self._duration_ms
            else:
                self._view_start_ms = max(0, int(self._start_sec * 1000) - self._margin_ms)
                self._view_end_ms = int(self._end_sec * 1000) + self._margin_ms
                if self._duration_ms > 0:
                    self._view_end_ms = min(self._duration_ms, self._view_end_ms)
        self._slider.setRange(self._view_start_ms, self._view_end_ms)
        self._sync_clip_region()

    def freeze_view(self, on: bool) -> None:
        """Toggle holding the scrubber's view extent across an _apply_view (used to wrap a
        Split-match rebuild so the visible margin doesn't snap back to full)."""
        self._freeze_view = on

    def _sync_clip_region(self) -> None:
        """Tell the scrubber where the set is (drawn green) within its current window."""
        self._slider.set_clip_region(int(self._start_sec * 1000), int(self._end_sec * 1000))

    def _on_toggle_full(self, checked: bool) -> None:
        # Guard so set_full_view -> checkbox -> here -> fullViewChanged -> menu action -> set_full_view
        # settles instead of looping (the menu action and the checkbox mirror each other).
        if checked == self._full_view:
            return
        self._full_view = checked
        self._apply_view()
        self._apply_markers()   # hide match ticks in whole-vod view, restore them when windowed
        if self._pending_seek_ms < self._view_start_ms or self._pending_seek_ms > self._view_end_ms:
            self._render_at(self._view_start_ms)
        self.fullViewChanged.emit(checked)

    def set_view_margin(self, seconds: float) -> None:
        """Set how much context to show on each side of the clip and re-window live. Ignored
        visually while the whole vod is shown, but takes effect once that's unchecked."""
        self._margin_ms = int(seconds * 1000)
        self._apply_view()
        if self._pending_seek_ms < self._view_start_ms or self._pending_seek_ms > self._view_end_ms:
            self._render_at(self._view_start_ms)

    def _edit_view_margin(self) -> None:
        """Popup to set the per-side preview margin in minutes/seconds, seeded from the current.
        'Save as default' (off on every open, so persisting is a deliberate opt-in) also writes
        the value to settings, so new vods/sessions open with it instead of the built-in fallback."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit preview bounds")
        minutes = QSpinBox()
        minutes.setRange(0, 99)
        minutes.setValue(self._margin_ms // 60000)
        seconds = QSpinBox()
        seconds.setRange(0, 59)
        seconds.setValue((self._margin_ms // 1000) % 60)
        save_default = QCheckBox("Save as default")
        save_default.setEnabled(self._settings is not None)   # nothing to persist to otherwise

        form = QFormLayout(dialog)
        form.addRow(QLabel("Preview margin on each side"))
        form.addRow("Minutes", minutes)
        form.addRow("Seconds", seconds)
        form.addRow(save_default)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            total = minutes.value() * 60 + seconds.value()
            self.set_view_margin(total)
            if save_default.isChecked() and self._settings is not None:
                self._settings.setValue(SETTING_CLIP_VIEW_MARGIN_SEC, float(total))

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def _player_state(self) -> dict[str, object]:
        """The media-player fields the trace records; the ones that reveal whether a seek could
        paint a frame (status/state/hasVideo) and whether the surface was on screen (vis/vwVis)."""
        return {
            "pos": self._player.position(),
            "dur": self._player.duration(),
            "status": self._player.mediaStatus().name,
            "state": self._player.playbackState().name,
            "hasVideo": self._player.hasVideo(),
            "vis": int(self.isVisible()),
            "vwVis": int(self._video.isVisible()),
            "primed": int(self._primed),
            "priming": int(self._priming),
        }

    def _trace_state(self, event: str, **extra: object) -> None:
        """Trace ``event`` with the full player-state snapshot, computing the snapshot only when
        tracing is enabled (so the high-frequency seams cost nothing on a normal run). Explicit
        ``extra`` fields override the snapshot's, so an event can pass e.g. its signal argument
        without colliding with the same-named snapshot key."""
        if _TRACE_ENABLED:
            fields = self._player_state()
            fields.update(extra)
            _trace(event, **fields)

    def _trace_scrub(self, ms: int) -> None:
        """Record scrubber moves (throttled) with player state. Throttled so a fast drag neither
        floods the log nor lags the UI; any re-prime a scrub triggers is still captured by the
        un-throttled prime.kick / playbackState lines."""
        if not _TRACE_ENABLED:
            return
        now = time.monotonic()
        if now - self._last_scrub_trace >= _SCRUB_TRACE_MIN_INTERVAL_SEC:
            self._last_scrub_trace = now
            self._trace_state("scrub", ms=ms)

    # ── Internals ─────────────────────────────────────────────────────────────
    def _seek_sec(self, sec: float) -> None:
        _trace("seek", ms=int(sec * 1000))
        self._render_at(int(sec * 1000))

    def _toggle_play(self) -> None:
        self._trace_state("toggle_play")
        if self._is_playing():
            self._player.pause()
        else:
            self._dismiss_display_page()   # playback moves the playhead off the analyzed frame
            # Starting from a Stopped/idled player would play from its reset position, so restore
            # the tracked playhead first.
            if not self._is_renderable():
                self._player.setPosition(self._pending_seek_ms)
            self._player.play()

    def stamp_start(self) -> None:
        """Set the clip start to the current playhead and emit startChanged."""
        self._start_sec = self._pending_seek_ms / 1000.0
        self._sync_clip_region()
        self.startChanged.emit(self._start_sec)

    def stamp_end(self) -> None:
        """Set the clip end to the current playhead and emit endChanged."""
        self._end_sec = self._pending_seek_ms / 1000.0
        self._sync_clip_region()
        self.endChanged.emit(self._end_sec)

    # Public transport API: the same operations as the transport buttons, exposed so the
    # Playback/View menus (built in ReviewScreen) can drive them.
    def toggle_play(self) -> None:
        """Toggle play/pause."""
        self._toggle_play()

    def pause(self) -> None:
        """Pause playback; a deterministic pause (unlike toggle_play), for callers that need
        the playhead to hold still regardless of the current playback state."""
        self._player.pause()

    def jump_to_start(self) -> None:
        """Seek the playhead to the clip's export start."""
        self._dismiss_display_page()
        self._seek_sec(self._start_sec)

    def jump_to_end(self) -> None:
        """Seek the playhead to the clip's export end."""
        self._dismiss_display_page()
        self._seek_sec(self._end_sec)

    def is_full_view(self) -> bool:
        """Whether the whole VOD is shown (vs the windowed clip ± margin)."""
        return self._full_view

    def set_full_view(self, on: bool) -> None:
        """Show the whole VOD or the windowed clip. Drives the checkbox, which runs the toggle
        handler and emits fullViewChanged; a no-op when already in that view."""
        self._full_check.setChecked(on)

    def edit_preview_bounds(self) -> None:
        """Open the preview-margin dialog."""
        self._edit_view_margin()

    def _on_position(self, ms: int) -> None:
        if _TRACE_ENABLED:
            # Log during the priming kick unthrottled (its few ticks answer "does position advance
            # after a paused seek?"); throttle steady playback so the trace stays readable.
            now = time.monotonic()
            if self._priming or now - self._last_pos_trace >= _POS_TRACE_MIN_INTERVAL_SEC:
                self._last_pos_trace = now
                _trace("position", ms=ms, priming=int(self._priming),
                       state=self._player.playbackState().name, hasVideo=self._player.hasVideo())
        if self._priming:
            return   # transient positions from the play→pause kick; don't move the scrubber
        self._pending_seek_ms = ms   # track the live playhead through playback (see _render_at)
        # When windowed, stop at the window end so playback doesn't bleed into the next set.
        if (not self._full_view and self._view_end_ms > 0 and ms >= self._view_end_ms
                and self._is_playing()):
            self._player.pause()
            self._player.setPosition(self._view_end_ms)
            self._pending_seek_ms = self._view_end_ms
            ms = self._view_end_ms
        if not self._slider.isSliderDown():
            self._slider.setValue(ms)
        self._time_label.setText(f"{_fmt(ms)} / {_fmt(self._view_end_ms)}")
        self._update_play_btn()

    def _update_play_btn(self) -> None:
        self._play_btn.setText("⏸ Pause" if self._is_playing() else "▶ Play")

    def _on_duration(self, ms: int) -> None:
        self._duration_ms = ms
        self._apply_view()
