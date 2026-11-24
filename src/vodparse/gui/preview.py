"""The Analyze page's preview + progress pane (app.py).

Two pages in one titled box. Idle, it is the layout preview (``gui.layout_preview.
LayoutPreviewPane``): the chosen video scrubbed live, with the regions the run will read
drawn on the settled frame. While a run is active it shows the pipeline's live annotated
frames plus a progress bar and ETA. The pipeline subprocess writes both files; frames to
``--preview-file``, ``{"timestamp_sec", "duration_sec"}`` JSON to ``--progress-file``
(plus a pre-pass ``stage`` and the clock calibration's outcome, see
schema.PipelineProgress); and two QTimers poll them into the UI (no OpenCV popup window).

The layout preview is built on the first ``set_video`` and imported only then: it pulls
in cv2 and QtMultimedia, which app startup (and a window that never gets a video) should
not pay for. Before that the run page's label doubles as the "No video selected" idle
look. With the layout preview turned off in Settings (``SETTING_SHOW_LAYOUT_PREVIEW``) the
idle look is instead the first release's: a still of the video's first frame on that same
label, grabbed through ``gui.frame_grabber`` (imported lazily for the same reason), and
nothing is searched or read before a run. Import this only after
``qt_plugins.configure_qt_plugins()`` has run (every app entry module does so at its top),
since it imports PyQt6 widgets at load time.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QSettings, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QGroupBox, QLabel, QProgressBar, QStackedWidget, QVBoxLayout, QWidget

from vodparse.cli.clock_locate import ClockLocateResult
from vodparse.cli.side_overlay_locate import SideOverlayLocateResult
from vodparse.games import Game
from vodparse.gui.settings import (SETTING_SHOW_ETA, SETTING_SHOW_LAYOUT_PREVIEW,
                                   SHOW_LAYOUT_PREVIEW_DEFAULT)
from vodparse.overlay_layout import CustomOverlayLayout
from vodparse.schema import (CLOCK_CALIBRATION_NOT_FOUND, PROGRESS_STAGE_CALIBRATING,
                             PROGRESS_STAGE_LOCATING_OVERLAY)

if TYPE_CHECKING:   # imported lazily at run time, see the module docstring
    import numpy as np

    from vodparse.gui.frame_grabber import FrameGrabber
    from vodparse.gui.layout_preview import LayoutPreviewPane

# How often the preview frame and the progress file are polled while a run is active.
PREVIEW_POLL_MS = 150
PROGRESS_POLL_MS = 250

# Shown once the bar tops out: the pipeline still tears down after its final 100% write,
# and the extract stage + the synchronous Review load follow before the screen switches;
# a frozen "100%" through that pause looks hung. Same wording as gpu_support's
# "finishing" stage label.
FINISHING_TEXT = "Finishing up…"

# Shown instead of the bare "0%" while a pre-pass runs before the main loop (the progress
# file's `stage`): the bar measures position in the video, which a pre-pass doesn't advance.
CALIBRATING_TEXT = "Finding the clock…"
LOCATING_OVERLAY_TEXT = "Finding player tags…"
STAGE_TEXTS: dict[str, str] = {
    PROGRESS_STAGE_CALIBRATING: CALIBRATING_TEXT,
    PROGRESS_STAGE_LOCATING_OVERLAY: LOCATING_OVERLAY_TEXT,
}

# Appended to the percent line once the main loop runs, when the clock pre-pass found no
# fixed spot: the run then detects the clock on every sample, which is worth knowing. A
# clock that was found (the ordinary case), a skipped calibration, and an older pipeline
# that doesn't report it all add nothing.
CLOCK_CALIBRATION_NOTES: dict[str, str] = {
    CLOCK_CALIBRATION_NOT_FOUND: "no steady clock spot",
}

# The still that stands in for the layout preview when it is turned off: the video's
# opening frame, as the first release showed; and the run page's text when that frame
# could not be grabbed (ffmpeg missing, a file it can't open).
FIRST_FRAME_SEC = 0.0
FIRST_FRAME_FAILED_TEXT = "Couldn't read the video's first frame"

# Default file locations the pipeline subprocess is told to write to (see start_run's callers
# in app.py, which pass these as --preview-file/--progress-file). Overridable in the constructor
# so tests can point them at a tmp_path instead of the shared system tempdir.
DEFAULT_PREVIEW_FILE = Path(tempfile.gettempdir()) / "vodparse_pipeline_preview.jpg"
DEFAULT_PROGRESS_FILE = Path(tempfile.gettempdir()) / "vodparse_pipeline_progress.json"


def _unlink_quietly(path: Path) -> None:
    """Delete `path` if it exists, swallowing PermissionError too: on Windows the pipeline
    subprocess can hold the file open at this exact instant (a sharing-violation race, not a
    real failure), and missing_ok=True alone doesn't cover that case."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def format_remaining(seconds: float) -> str:
    """An ETA in minutes to one decimal, e.g. "1.5 mins", "1.0 min". Minutes (not seconds)
    so the estimate's own second-to-second wobble doesn't make the text twitch."""
    minutes = round(seconds / 60.0, 1)
    return f"{minutes:.1f} min" if minutes == 1.0 else f"{minutes:.1f} mins"


def progress_eta_seconds(anchor_time: float, anchor_frac: float,
                         now: float, frac: float) -> float | None:
    """Seconds remaining projected from the average rate since the first-progress anchor,
    or None when nothing has advanced. Throughput (video-seconds processed per wall-second)
    is roughly steady, so the average rate since the first real-progress sample projects the
    rest. Anchoring to that first sample rather than the run start keeps model-load time:
    which elapses before any progress is written; out of the rate."""
    elapsed, advanced = now - anchor_time, frac - anchor_frac
    if elapsed <= 0.0 or advanced <= 0.0:
        return None
    return elapsed / advanced * (1.0 - frac)


class PreviewProgressPane(QWidget):
    """The titled "Preview" box on the Analyze page: the layout preview of the chosen video
    (scrub it, see the read regions), or a still of its first frame when the layout
    preview is turned off in Settings, swapped for the live detection frames plus the
    progress bar while the pipeline runs.

    ``settings`` supplies that toggle and the layout preview's OCR language (the run's
    own; its engine follows the Font style combo through ``set_engine``);
    ``start_worker`` is handed to it too (False builds it with no frame grabber and no
    OCR helper, for offscreen tests, and grabs no still either)."""

    def __init__(self, settings: QSettings, *,
                 preview_file: Path = DEFAULT_PREVIEW_FILE,
                 progress_file: Path = DEFAULT_PROGRESS_FILE,
                 start_worker: bool = True) -> None:
        super().__init__()
        self._settings = settings
        self._preview_file = preview_file
        self._progress_file = progress_file
        self._start_worker = start_worker
        self._video_path: str | None = None
        # The layout preview and what the main window last told it, replayed onto it when
        # it is built (a game/layout choice can precede the first video).
        self._scrubber: LayoutPreviewPane | None = None
        self._context: tuple[Game, CustomOverlayLayout | None, float | None] | None = None
        self._engine: str | None = None
        # The still shown instead (the layout preview off): its grabber, built on the first
        # still; the frame, unscaled, kept so a run's end can show it again without a
        # re-grab; and the token the grab in flight echoes, so a video chosen while it runs
        # supersedes it.
        self._first_frame_grabber: FrameGrabber | None = None
        self._first_frame: QPixmap | None = None
        self._first_frame_token = 0

        # Embedded pipeline preview: the pipeline subprocess writes annotated frames to this file
        # and a timer pumps them into a label (no OpenCV popup).
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(PREVIEW_POLL_MS)
        self._preview_timer.timeout.connect(self._update_preview)

        # Pipeline progress bar: same file-polling scheme as the preview, shown for the whole run.
        self._progress_anchor: tuple[float, float] | None = None  # set per run by start_run
        # A run holds the box from start_run until hide_progress (so through the
        # "Finishing up…" hold too, when the polling timers have already stopped).
        self._run_active = False
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(PROGRESS_POLL_MS)
        self._progress_timer.timeout.connect(self._update_progress)

        # The run page: the pipeline's live frames while it runs, and the idle text
        # before any video is chosen (the scrubber page takes over from the first video).
        self._preview_label = QLabel("No video selected")
        self._preview_label.setMinimumSize(480, 270)
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pages = QStackedWidget()
        self._pages.addWidget(self._preview_label)

        # How far through the video the run has gotten. Visible only while the pipeline
        # runs; both fed by _update_progress polling the pipeline's progress file. The
        # label carries the percent + ETA text (too cramped on the bar itself); the bar
        # uses range 0–1000 so it moves smoothly on hours-long vods.
        self._progress_label = QLabel()
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.hide()
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.hide()

        # Wrapped in a titled box to match the Source/Analysis/Sets groupings on the left.
        preview_box = QGroupBox("Preview")
        preview_col = QVBoxLayout(preview_box)
        preview_col.addWidget(self._pages, stretch=1)
        preview_col.addWidget(self._progress_label)
        preview_col.addWidget(self._progress_bar)
        # ...then wrapped again in a plain widget so it's the *same kind* of layout item as the
        # left column. A QGroupBox added straight to a box layout overhangs ~15px into the title
        # space the layout reserves at the top; a plain-widget-wrapped one doesn't; so wrapping
        # both keeps their tops on the same line.
        preview_wrap = QVBoxLayout()
        preview_wrap.setContentsMargins(0, 0, 0, 0)
        preview_wrap.addWidget(preview_box)
        self.setLayout(preview_wrap)

    @property
    def preview_file(self) -> Path:
        """The file the pipeline subprocess is handed as --preview-file."""
        return self._preview_file

    @property
    def progress_file(self) -> Path:
        """The file the pipeline subprocess is handed as --progress-file."""
        return self._progress_file

    def _show_eta(self) -> bool:
        """Whether the analysis progress label appends an "~N mins left" estimate to the
        percent (default True)."""
        return self._settings.value(SETTING_SHOW_ETA, True, type=bool)

    def _show_layout_preview(self) -> bool:
        """Whether the idle box is the layout preview (default True) or a still of the
        chosen video's first frame (SETTING_SHOW_LAYOUT_PREVIEW off)."""
        return self._settings.value(
            SETTING_SHOW_LAYOUT_PREVIEW, SHOW_LAYOUT_PREVIEW_DEFAULT, type=bool)

    def set_video(self, path: str | None) -> None:
        """Remember `path` as the current vod and show it in the idle box: opened in the
        layout preview (built on the first video, see the module docstring), or, with the
        layout preview turned off in Settings, as a still of its first frame."""
        self._video_path = path
        if path is None:
            return
        if self._show_layout_preview():
            self._open_in_layout_preview(path)
        else:
            self._show_first_frame(path)

    def apply_layout_preview_setting(self) -> None:
        """Settings closed: if the toggle no longer matches the idle look, switch it. Off
        tears the layout preview down and grabs the still; on opens the video in a fresh
        layout preview (a shut-down one can't be reused, see _drop_scrubber). A no-op
        when the look already matches, and before a video there is nothing to show either
        way: only a layout preview left over from a video since dropped goes."""
        if self._show_layout_preview() == (self._scrubber is not None):
            return
        if self._video_path is None:
            self._drop_scrubber()
            return
        self.set_video(self._video_path)

    def _open_in_layout_preview(self, path: str) -> None:
        """Open ``path`` in the layout preview, building it on the first video, and show
        it unless a run holds the box: the preview then stays behind the run page, and
        the run's own end (hide_progress) shows it."""
        if self._scrubber is None:
            self._scrubber = self._build_scrubber()
        self._scrubber.set_video(path)
        if not self._run_active:
            self._show_idle_page()

    def _build_scrubber(self) -> LayoutPreviewPane:
        """Build the layout preview as the box's second page, replaying the game/layout
        choice and Font style the main window told this pane before it existed. Built
        while a run holds the box (the Settings toggle can do that) it starts suspended,
        as start_run leaves one: no capture behind the run page."""
        from vodparse.gui.layout_preview import LayoutPreviewPane   # lazy: cv2 + QtMultimedia

        scrubber = LayoutPreviewPane(self._settings, start_worker=self._start_worker)
        self._pages.addWidget(scrubber)
        if self._engine is not None:
            scrubber.set_engine(self._engine)
        if self._context is not None:
            scrubber.set_context(*self._context)
        if self._run_active:
            scrubber.suspend()
        return scrubber

    def _drop_scrubber(self) -> None:
        """Tear the layout preview down and take it out of the box. Its shutdown is final
        (see LayoutPreviewPane.shutdown: nothing is read after it), so the widget goes
        and the next video under the layout preview builds a fresh one. A no-op when
        none was built."""
        if self._scrubber is None:
            return
        self._scrubber.shutdown()
        self._pages.removeWidget(self._scrubber)
        self._scrubber.deleteLater()
        self._scrubber = None

    def _show_first_frame(self, path: str) -> None:
        """Grab the first frame of ``path`` for the still (the layout preview off; one
        built for an earlier video goes first). The grab runs through FrameGrabber
        (ffmpeg off the UI thread, the path the layout preview captures with) and answers
        into _on_first_frame_grabbed. Without a worker (offscreen tests) nothing is
        grabbed: tests hand the frame to that slot."""
        self._drop_scrubber()
        self._first_frame = None
        self._first_frame_token += 1
        if not self._run_active:
            # Blank until the still lands: neither the idle "No video selected" text nor
            # the previous video's still belongs to the video just chosen.
            self._preview_label.clear()
            self._pages.setCurrentWidget(self._preview_label)
        if not self._start_worker:
            return
        if self._first_frame_grabber is None:
            from vodparse.gui.frame_grabber import FrameGrabber   # lazy: cv2 + numpy

            self._first_frame_grabber = FrameGrabber(self)
            self._first_frame_grabber.frame_grabbed.connect(self._on_first_frame_grabbed)
            self._first_frame_grabber.grab_failed.connect(self._on_first_frame_failed)
        self._first_frame_grabber.grab(path, FIRST_FRAME_SEC, self._first_frame_token)

    def _on_first_frame_grabbed(self, frame: np.ndarray, _sec: float, token: int) -> None:
        """The still's frame arrived (BGR, as FrameGrabber emits it): keep it as the
        pixmap the idle box shows, and show it now unless a run holds the box."""
        if token != self._first_frame_token:
            return   # a video chosen since: its own grab answers for it
        from vodparse.gui.image_pane import ndarray_to_pixmap   # lazy: cv2 + numpy

        self._first_frame = ndarray_to_pixmap(frame)
        if not self._run_active:
            self._show_idle_page()

    def _on_first_frame_failed(self, _message: str, token: int) -> None:
        """The still's grab failed: say so on the run page unless a run holds the box."""
        if token != self._first_frame_token:
            return
        if not self._run_active:
            self._pages.setCurrentWidget(self._preview_label)
            self._preview_label.setText(FIRST_FRAME_FAILED_TEXT)

    def set_context(self, game: Game, layout: CustomOverlayLayout | None,
                    ingame_tag_frac: float | None) -> None:
        """Hand the layout preview what the run would read with (see
        LayoutPreviewPane.set_context); remembered for a preview not built yet."""
        self._context = (game, layout, ingame_tag_frac)
        if self._scrubber is not None:
            self._scrubber.set_context(game, layout, ingame_tag_frac)

    def set_engine(self, engine: str) -> None:
        """Hand the layout preview the Font style choice (see LayoutPreviewPane.set_engine);
        remembered for a preview not built yet."""
        self._engine = engine
        if self._scrubber is not None:
            self._scrubber.set_engine(engine)

    def playhead_sec(self) -> float:
        """Where the layout preview's scrubber stands, in source-video seconds: what the
        layout editor opens on, so editing continues from the frame on screen. 0
        before a video (no preview has been built yet)."""
        if self._scrubber is None:
            return 0.0
        return self._scrubber.player.playhead_sec()

    def set_clock(self, result: ClockLocateResult | None) -> None:
        """Hand the layout preview the clock search's answer (a no-op before a video)."""
        if self._scrubber is not None:
            self._scrubber.set_clock(result)

    def set_clock_status(self, text: str) -> None:
        """Hand the layout preview the clock search's progress text (a no-op before a video)."""
        if self._scrubber is not None:
            self._scrubber.set_clock_status(text)

    def set_side_overlay(self, result: SideOverlayLocateResult | None) -> None:
        """Hand the layout preview the player-name search's answer (a no-op before a video)."""
        if self._scrubber is not None:
            self._scrubber.set_side_overlay(result)

    def set_side_overlay_status(self, text: str) -> None:
        """Hand the layout preview the player-name search's progress text (a no-op before
        a video)."""
        if self._scrubber is not None:
            self._scrubber.set_side_overlay_status(text)

    def shutdown(self) -> None:
        """The window is closing: stop the layout preview's player and grab, and the
        still's grab."""
        if self._scrubber is not None:
            self._scrubber.shutdown()
        if self._first_frame_grabber is not None:
            self._first_frame_grabber.cancel()

    def release_for_site_swap(self) -> None:
        """A GPU-support install or uninstall is starting: the layout preview stops its OCR
        helper and holds its reads (see LayoutPreviewPane.release_for_site_swap; a no-op
        before a video)."""
        if self._scrubber is not None:
            self._scrubber.release_for_site_swap()

    def read_after_site_swap(self) -> None:
        """The install or uninstall has ended: the layout preview reads what it held back
        (a no-op before a video)."""
        if self._scrubber is not None:
            self._scrubber.read_after_site_swap()

    def start_run(self) -> None:
        """Begin polling both files for a fresh pipeline run: drop any stale frame/progress
        left over from a previous run, then start the timers. The layout preview is held
        (no capture while the run has the box) behind the run page."""
        self._run_active = True
        if self._scrubber is not None:
            self._scrubber.suspend()
        # Blank until the run's first frame lands (the pipeline writes the video's opening
        # frame as soon as it starts, before its models load): the label otherwise still
        # shows its idle "No video selected" text (a first run) or the last frame of the
        # run before.
        self._preview_label.clear()
        self._pages.setCurrentWidget(self._preview_label)
        _unlink_quietly(self._preview_file)   # drop any stale frame
        self._preview_timer.start()

        _unlink_quietly(self._progress_file)   # drop any stale progress
        self._progress_anchor = None   # (monotonic_time, frac) of the first real progress, for the ETA
        self._progress_bar.setValue(0)
        self._progress_label.setText("Starting…")
        self._progress_label.show()
        self._progress_bar.show()
        self._progress_timer.start()

    def stop_run(self) -> None:
        """End a pipeline run: stop polling and restore the pane to its idle look."""
        self._stop_polling()
        self.hide_progress()

    def show_finishing(self) -> None:
        """End the pipeline's polling like stop_run, but keep the bar pinned at 100% under
        FINISHING_TEXT instead of hiding: the extract stage and the synchronous Review load
        still run before the screen switches, and a vanished bar there looks hung.
        hide_progress() ends the hold."""
        self._stop_polling()
        # Pinned explicitly: the finished signal can beat the poll of the final 100% write,
        # and success (the only caller) means the run really did complete.
        self._progress_bar.setValue(self._progress_bar.maximum())
        self._progress_label.setText(FINISHING_TEXT)

    def hide_progress(self) -> None:
        """Hide the progress bar and label: ends a show_finishing hold (no-op when idle) and
        gives the box back to the layout preview."""
        self._run_active = False
        self._progress_bar.hide()
        self._progress_label.hide()
        self._show_idle_page()

    def _show_idle_page(self) -> None:
        """Show the idle look: the layout preview (once a video has built it), let capture
        again; otherwise the run page's label, with the first-frame still back in place
        of a run's last live frame once one has been grabbed."""
        if self._scrubber is not None:
            self._scrubber.resume()
            self._pages.setCurrentWidget(self._scrubber)
            return
        self._pages.setCurrentWidget(self._preview_label)
        if self._first_frame is not None:
            self._set_preview_pixmap(self._first_frame)

    def _stop_polling(self) -> None:
        """Stop both file polls and drop their files: shared teardown for stop_run and
        show_finishing. Leaves the bar/label visibility untouched."""
        self._preview_timer.stop()
        _unlink_quietly(self._preview_file)

        self._update_progress()  # catch the pipeline's final 100% write
        self._progress_timer.stop()
        _unlink_quietly(self._progress_file)

    def _update_preview(self) -> None:
        pix = QPixmap(str(self._preview_file))
        if not pix.isNull():
            self._set_preview_pixmap(pix)

    def _update_progress(self) -> None:
        """Read the pipeline's progress file into the bar and its label: a pre-pass's own
        text while one runs, then the percent (with the ETA and the clock calibration's
        outcome) once the main loop is under way."""
        try:
            data = json.loads(self._progress_file.read_text(encoding="utf-8"))
            stage_text = STAGE_TEXTS.get(data.get("stage"))
            clock_note = CLOCK_CALIBRATION_NOTES.get(data.get("clock_calibration"))
        except (OSError, json.JSONDecodeError, AttributeError, TypeError):
            # Not written yet (model loading takes a while), or not the shape the pipeline
            # writes (a list, an unhashable stage): keep the last value.
            return
        # A pre-pass's text needs no fraction: it shows even when the video reported no
        # duration (the percent below would divide by it).
        if stage_text is not None:
            self._progress_label.setText(stage_text)
            return
        try:
            frac = data["timestamp_sec"] / data["duration_sec"]
        except (KeyError, TypeError, ZeroDivisionError):
            return  # a tick without a usable position; keep the last value
        frac = min(max(frac, 0.0), 1.0)
        self._progress_bar.setValue(round(frac * 1000))
        if frac >= 1.0:
            # The pipeline keeps tearing down (readers, pipeline.json) after its last
            # write; "100% (~0.0 mins left)" through that stretch looks hung.
            self._progress_label.setText(FINISHING_TEXT)
            return
        pct = round(frac * 100)
        eta = self._progress_eta(frac) if self._show_eta() else None
        text = f"{pct}% (~{eta} left)" if eta else f"{pct}%"
        if clock_note is not None:
            text += f" · {clock_note}"
        self._progress_label.setText(text)

    def _progress_eta(self, frac: float) -> str | None:
        """Time remaining as an h/m/s string, or None until there's enough to estimate.
        Does the anchor bookkeeping (remembering the first real-progress sample) and
        delegates the actual rate math to the pure progress_eta_seconds."""
        now = time.monotonic()
        if self._progress_anchor is None:
            if frac > 0.0:
                self._progress_anchor = (now, frac)
            return None
        anchor_time, anchor_frac = self._progress_anchor
        seconds = progress_eta_seconds(anchor_time, anchor_frac, now, frac)
        return format_remaining(seconds) if seconds is not None else None

    def _set_preview_pixmap(self, pix: QPixmap) -> None:
        self._preview_label.setPixmap(pix.scaled(
            self._preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
