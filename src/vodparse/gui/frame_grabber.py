"""
Async video frame grabber
---------------------------
``FrameGrabber``: grabs the frame a paused player shows at a timestamp via ffmpeg
``QProcess``es, asynchronously.

Used in place of ``VideoFrameSource.grab_at`` because that runs cv2's CAP_PROP_POS_MSEC
seek on the caller's thread — on long-keyframe VODs (Twitch downloads especially) cv2
can spend tens of seconds decoding toward the target, freezing the GUI for the whole
ride. ffmpeg with ``-ss`` *before* ``-i`` seeks via the container index straight to the
nearest keyframe and decodes only the short tail up to the exact frame — typically well
under a second on the same files — and the QProcess keeps even a slow grab off the UI
thread. The frame comes back as a lossless PNG on stdout (stderr stays separate to carry
ffmpeg's error text).

Which frame: the one a paused Qt player shows at that position, i.e. the last frame
whose timestamp is at or *before* it (measured on a real VOD). ffmpeg's ``-ss`` instead
emits the first frame at or *after* the seek, and a playhead at millisecond resolution
nearly always falls between two frames, so a one-step grab came back one frame ahead of
the live picture under it. Snapping the seek to a nominal frame grid did not fix it
either: a stream whose first frame starts 59 ms after the container's zero (common in
stream recordings) has its frames at 59 ms + k/fps, not k/fps. So the grab runs in two
stages, neither assuming a frame rate:

1. **List** the video packets around the playhead with ``-c copy`` (no decoding) into
   the ``framecrc`` muxer, which prints each packet's timestamp. The seek lands on the
   keyframe at or before the playhead and the listing runs a short window past it, so
   the frame under the playhead is always in it, whatever the stream's timing.
2. **Grab** the frame whose timestamp is the last at or before the playhead, with a
   ``-ss`` a hair before that timestamp (see ``last_frame_at_or_before``).

One grab at a time: a new ``grab`` cancels any in-flight one, at either stage. Results
echo the caller's token so a superseded grab (the user scrubbed on) can be dropped.

Shared by the custom-overlay-layout editor (``vodparse.gui.overlay_editor``) and the
main screen's layout preview pane.

Import this only after ``qt_plugins.configure_qt_plugins()`` has run: like
``vodparse.gui.overlay_editor``, this module imports cv2 and numpy at load time, not
lazily.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QProcess, pyqtSignal

from vodparse.ffmpeg_runtime import ensure_ffmpeg

# How long after the scrubber is let go (or playback pauses) before a host grabs the
# settled frame: enough for the media backend to paint the seeked frame and for a release
# that is really a pause between two drags to be cancelled by the next press. Shared by
# the layout editor and the Analyze page's layout preview, which both capture that way.
SETTLE_DELAY_MS = 350

# How far past the playhead the packet listing runs. Packets come in decode order, so a
# frame shown before the playhead can be listed after one shown past it (B-frame
# reordering); half a second covers any reordering depth in practice.
LISTING_WINDOW_SEC = 0.5

# How far before the chosen frame's timestamp the grab seeks: ffmpeg emits the first
# frame at or after ``-ss``, so seeking exactly on the timestamp could skip the frame if
# the formatted value rounds a hair later than the stream's own. Half a millisecond is
# far less than any frame period, so the previous frame is never reached.
GRAB_LEAD_SEC = 0.0005

# What the listing's packet lines start with: the framecrc muxer's stream index column.
PACKET_LINE_PREFIX = "0,"
TIME_BASE_LINE_PREFIX = "#tb 0: "

NO_FRAME_MESSAGE = ("ffmpeg produced no frame (maybe it tried seeking past the end of the "
                    "video?)")


def last_frame_at_or_before(listing: str, seek_sec: float, target_sec: float) -> float | None:
    """The timestamp (in the player's seconds) of the frame a paused player shows at
    ``target_sec``, read off a ``framecrc`` listing made with ``-ss seek_sec`` and no
    ``-copyts``: the last packet timestamp at or before the target. The listing's
    timestamps are relative to the seek, so ``seek_sec`` is added back.

    A target before the stream's first frame gets that first frame (as the player
    shows it). None when the listing holds no packets (a seek past the end). Pure, so
    the parse is testable on a canned listing."""
    time_base = Fraction(1, 90000)   # the muxer always prints one; this is just a type seed
    stamps: list[float] = []
    for line in listing.splitlines():
        if line.startswith(TIME_BASE_LINE_PREFIX):
            time_base = Fraction(line[len(TIME_BASE_LINE_PREFIX):].strip())
        elif line.startswith(PACKET_LINE_PREFIX):
            # "0,  <dts>,  <pts>,  <duration>,  <size>, <crc>"
            pts_ticks = int(line.split(",")[2])
            stamps.append(seek_sec + float(pts_ticks * time_base))
    if not stamps:
        return None
    at_or_before = [stamp for stamp in stamps if stamp <= target_sec]
    return max(at_or_before) if at_or_before else min(stamps)


class FrameGrabber(QObject):
    """Grabs, asynchronously via ffmpeg QProcesses, the frame a paused player shows at a
    timestamp: a no-decode packet listing to find that frame's own timestamp, then the
    one-frame grab at it. See the module docstring for why two stages and why no frame
    rate is assumed.

    One grab at a time: a new ``grab`` cancels any in-flight one, at either stage.
    Results echo the caller's token so a superseded grab (the user scrubbed on) can be
    dropped."""

    frame_grabbed = pyqtSignal(object, float, int)   # BGR frame, sec, token
    grab_failed   = pyqtSignal(str, int)             # status message, token

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._proc: QProcess | None = None   # whichever stage is in flight
        self._ffmpeg = ""
        self._video_path = ""
        self._sec = 0.0
        self._token = -1

    def grab(self, video_path: str, sec: float, token: int) -> None:
        """Start grabbing the frame shown at ``sec``; ``frame_grabbed`` or ``grab_failed``
        arrives later carrying ``token``. Cancels any grab still in flight."""
        self.cancel()
        try:
            self._ffmpeg = ensure_ffmpeg()   # cached after the first call; raises with guidance
        except RuntimeError as exc:
            self.grab_failed.emit(str(exc), token)
            return
        self._video_path = video_path
        self._sec = max(sec, 0.0)
        self._token = token
        self._start_listing()

    def cancel(self) -> None:
        """Kill any in-flight stage silently (a no-op when idle)."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        proc.blockSignals(True)
        proc.kill()
        proc.waitForFinished(1000)
        proc.deleteLater()

    # ── Stage 1: list the packets around the playhead ───────────────────────────
    def _start_listing(self) -> None:
        """Seek to the keyframe at or before the playhead and list the video packets from
        there to LISTING_WINDOW_SEC past it, copied (not decoded) into framecrc."""
        self._proc = self._start_ffmpeg(self._on_listing_finished, [
            "-ss", f"{self._sec:.3f}", "-i", self._video_path,
            "-t", f"{LISTING_WINDOW_SEC}", "-map", "0:v:0", "-c", "copy",
            "-f", "framecrc", "-"])

    def _on_listing_finished(self, code: int, _status: QProcess.ExitStatus) -> None:
        proc = self._take_finished_process()
        if proc is None:
            return   # already cancelled
        listing = proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        error = proc.readAllStandardError().data().decode("utf-8", errors="replace").strip()
        proc.deleteLater()
        frame_sec = last_frame_at_or_before(listing, self._sec, self._sec) if code == 0 else None
        if frame_sec is None:
            self.grab_failed.emit(error or NO_FRAME_MESSAGE, self._token)
            return
        self._start_grab(frame_sec)

    # ── Stage 2: grab that frame ────────────────────────────────────────────────
    def _start_grab(self, frame_sec: float) -> None:
        """Decode the one frame at ``frame_sec`` (seeking a hair before it, see
        GRAB_LEAD_SEC) to a PNG on stdout."""
        seek_sec = max(frame_sec - GRAB_LEAD_SEC, 0.0)
        self._proc = self._start_ffmpeg(self._on_grab_finished, [
            "-ss", f"{seek_sec:.6f}", "-i", self._video_path,
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"])

    def _on_grab_finished(self, code: int, _status: QProcess.ExitStatus) -> None:
        proc = self._take_finished_process()
        if proc is None:
            return   # already cancelled
        png = proc.readAllStandardOutput().data()
        error = proc.readAllStandardError().data().decode("utf-8", errors="replace").strip()
        proc.deleteLater()
        frame = (cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
                 if png else None)
        if code == 0 and frame is not None:
            self.frame_grabbed.emit(frame, self._sec, self._token)
        else:
            self.grab_failed.emit(error or NO_FRAME_MESSAGE, self._token)

    # ── Process plumbing ────────────────────────────────────────────────────────
    def _start_ffmpeg(self, on_finished: Callable[[int, QProcess.ExitStatus], None],
                      args: list[str]) -> QProcess:
        """Launch ffmpeg with the common quiet flags plus ``args``, finishing into
        ``on_finished``."""
        proc = QProcess(self)
        proc.finished.connect(on_finished)
        # finished never fires when the process fails to start; errorOccurred covers it.
        proc.errorOccurred.connect(self._on_error)
        proc.start(self._ffmpeg, ["-hide_banner", "-loglevel", "error", *args])
        return proc

    def _take_finished_process(self) -> QProcess | None:
        """The stage that just finished, cleared from ``_proc``; None if it was cancelled
        meanwhile (its signals are blocked then, but a finish already queued can land)."""
        proc, self._proc = self._proc, None
        return proc

    def _on_error(self, error: QProcess.ProcessError) -> None:
        """Only a start failure is reported here; a crash still reaches finished()."""
        if error != QProcess.ProcessError.FailedToStart or self._proc is None:
            return
        self._proc.deleteLater()
        self._proc = None
        self.grab_failed.emit("Couldn't start ffmpeg to grab the frame", self._token)
