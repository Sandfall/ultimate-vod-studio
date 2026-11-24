"""Threaded video frame reader with seek support.

Decodes frames in a background thread so the caller's YOLO inference and
frame decode overlap: eliminating the serial decode→infer→decode→infer stall.

The thread applies backpressure: it stays at most ~one frame ahead of the consumer
rather than racing on. That overlap is the point in sequential reads, but it also
stops the reader from decoding (and discarding) every frame between two seeks, which,
in seek mode, was ~1000 wasted decodes per seek that stole CPU from inference on
machines without a spare core.

Also the single-frame path: ``VideoFrameSource`` grabs one frame at a timestamp (the
video mode's playhead, the pre-passes' seeks), and ``sample_timestamps`` spreads a
pre-pass's few dozen seeks over a video."""

import queue
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Self, override

import cv2
import numpy as np


def open_capture(source: str) -> cv2.VideoCapture:
    """A ``cv2.VideoCapture`` on `source`, or an accurate complaint: FileNotFoundError when
    the file is missing, ValueError when it exists but OpenCV can't decode it (corrupted,
    incomplete, or an unsupported codec). Callers wanting a fail-fast pre-flight (before
    loading models, say) can open and immediately release."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        if not Path(source).exists():
            raise FileNotFoundError(f"Video file not found: {source}")
        raise ValueError(f"Can't decode {source}: the file may be corrupted or incomplete, "
                         f"or use an unsupported codec.")
    return cap


class FrameReader(threading.Thread):
    """Background thread that decodes video frames into a 1-slot queue, with seek support."""

    def __init__(self, source: str, start_frame: int = 0) -> None:
        super().__init__(daemon=True)
        self.cap = open_capture(source)
        self.fps          = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width        = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_number = start_frame
        self.stopped      = False
        self._queue: queue.Queue[tuple[int, int, Any]] = queue.Queue(maxsize=1)
        self._seek_target: int | None = start_frame if start_frame > 0 else None
        self._seek_lock   = threading.Lock()
        # Bumped on every seek and stamped onto each queued frame, so read() can tell a
        # frame decoded before the latest seek from one decoded after it. Guarded by
        # _seek_lock together with _seek_target.
        self._generation  = 0

    def seek(self, frame_number: int) -> None:
        """Reposition the decode thread to `frame_number` (0-based). A target at or past
        the end of the video is treated as reaching EOF: the reader stops and delivers no
        further frames (``stopped`` becomes True; a subsequent read() times out).

        Why not just let OpenCV handle it: OpenCV clamps a seek to the reported frame count
        and converts that index to a time using the AVERAGE frame rate. On a variable-
        frame-rate file (dropped frames in a recording, say) that time lands SHORT of the
        real last frame, so read() keeps succeeding after every "past the end" seek and a
        caller stepping forward in fixed increments never sees EOF: it loops forever on the
        same tail frame. Stopping here is what makes the pipeline's seek loop terminate."""
        if self._is_past_end(frame_number):
            self.stopped = True
            return
        with self._seek_lock:
            self._seek_target = frame_number
            self._generation += 1
        # Drop any frame already queued. A pre-seek frame the reader thread queues *after*
        # this drain still carries the old generation, so read() discards it.
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

    def _is_past_end(self, frame_number: int) -> bool:
        """True when `frame_number` is at or beyond the last frame (indices run 0 to
        total_frames - 1). Always False when the container reports no frame count, since
        then there's nothing to compare against and a sequential read is the only way
        to find the end."""
        return self.total_frames > 0 and frame_number >= self.total_frames

    @override
    def run(self) -> None:
        try:
            while not self.stopped:
                with self._seek_lock:
                    target     = self._seek_target
                    generation = self._generation
                    if target is not None:
                        self._seek_target = None
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                        self.frame_number = target

                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.frame_number += 1
                # Backpressure: wait for the consumer to take the previous frame instead of
                # dropping it and racing on. The reader stays ~one frame ahead: enough to
                # overlap decode with the consumer's work, but without free-running through
                # (and discarding) every frame between two seeks. Bail out of the wait if we
                # are asked to stop, or if a new seek arrives; this frame is then stale, so
                # drop it and loop up to serve the seek.
                item = (generation, self.frame_number, frame)
                while not self.stopped:
                    with self._seek_lock:
                        if self._seek_target is not None:
                            break
                    try:
                        self._queue.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        finally:
            self.cap.release()

    def read(self):
        """Next (frame_number, frame), skipping frames decoded before the latest seek.
        Raises queue.Empty if no fresh frame arrives within the timeout."""
        while True:
            generation, frame_number, frame = self._queue.get(timeout=2.0)
            with self._seek_lock:
                current = self._generation
            if generation == current:
                return frame_number, frame

    def stop(self) -> None:
        """Signal the decode thread to finish and release the capture. Joins the thread so
        its run()-finally release completes; if the thread was never started, releases the
        capture directly here. Safe to call more than once (release is idempotent)."""
        self.stopped = True
        if self.is_alive():
            self.join(timeout=2.0)
        self.cap.release()

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


# ── Single-frame grabbing ────────────────────────────────────────────────────────

# FrameReader falls back to 30fps when CAP_PROP_FPS reads falsy (`self.fps =
# self.cap.get(cv2.CAP_PROP_FPS) or 30`); VideoFrameSource mirrors that same fallback
# value under its own name since it also treats a negative fps (some containers report
# -1 for an unknown rate) as unusable, which `or` alone wouldn't catch.
FPS_FALLBACK = 30.0


class VideoFrameSource:
    """cv2.VideoCapture wrapper for grabbing single frames of a video at a timestamp.

    Used by the video mode to analyze the frame under the player's playhead, and by the
    pre-passes that seek a few dozen frames across a VOD before the main pass. The grab
    here and the player's displayed frame may differ by ~a frame (different demuxers /
    seek rounding) accepted, because analysis, annotation, and the displayed still all
    use the SAME grabbed frame, so what you analyze is exactly what's annotated. (The
    GUI's async grabber finds the frame under the playhead by its real timestamp instead;
    see gui.frame_grabber.last_frame_at_or_before.)

    A context manager: ``with VideoFrameSource(path) as source:`` releases the capture on
    the way out."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            self._cap.release()
            raise ValueError(f"Couldn't open the video: {path}")
        # (width, height) of the first frame decoded; see frame_size().
        self._frame_size: tuple[int, int] | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def fps(self) -> float:
        """The container's nominal frame rate, falling back to FPS_FALLBACK when it
        doesn't report a usable one. Nominal: a variable-frame-rate file reports an
        average, so frame arithmetic on it is approximate."""
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else FPS_FALLBACK

    def duration_sec(self) -> float:
        """Video length in seconds (frame count / fps, see fps())."""
        frame_count = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return frame_count / self.fps()

    def grab_at(self, sec: float) -> np.ndarray | None:
        """Seek to `sec` and decode one BGR frame (OpenCV's native layout, matching
        load_image). Returns None when the read fails: a seek past end-of-file or a
        decode error, both of which cv2 reports as `ok=False` rather than raising.
        Negative timestamps clamp to the first frame.

        Runs on the caller's thread: a keyframe-sparse source can take a noticeable
        moment to seek. Moving the grab onto a worker thread is a deliberate escape
        hatch left for later, not built here: fine for a terminal-launched dev tool."""
        sec = max(sec, 0.0)
        self._cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        if self._frame_size is None:
            self._frame_size = (frame.shape[1], frame.shape[0])
        return frame

    def frames_at(self, timestamps: Iterable[float]) -> Iterator[tuple[float, np.ndarray]]:
        """``(timestamp, frame)`` for each of ``timestamps`` in order, via ``grab_at``. A
        timestamp the decoder cannot deliver a frame for is skipped, not yielded as None."""
        for timestamp in timestamps:
            frame = self.grab_at(timestamp)
            if frame is not None:
                yield timestamp, frame

    def frame_size(self) -> tuple[int, int] | None:
        """The video's ``(width, height)`` as decoded, never the container's metadata (which
        some files report as 0x0). Taken from the first frame this source grabbed; when it
        has grabbed none yet, the first frame is decoded for it. None when the video yields
        no frame at all."""
        if self._frame_size is None:
            self.grab_at(0.0)
        return self._frame_size

    def release(self) -> None:
        """Release the underlying capture. Safe to call more than once: cv2.VideoCapture
        tolerates releasing an already-released capture."""
        self._cap.release()


# ── Spreading samples over a video ───────────────────────────────────────────────

MIN_SAMPLE_STEP_SEC = 1.0
"""The closest two ``sample_timestamps`` ever land: a short video yields fewer samples
rather than several seeks onto (nearly) the same frame."""


def sample_timestamps(duration_sec: float, count: int,
                      span: tuple[float, float]) -> list[float]:
    """``count`` timestamps spread evenly over the ``span`` stretch of the video, given as
    (first, last) fractions of its duration: ``(0.05, 0.95)`` keeps 5 % clear of both
    ends. A very short video yields fewer, never two within ``MIN_SAMPLE_STEP_SEC`` of each
    other, and one shorter than that step yields the single mid-span timestamp. Rounded
    to the millisecond. Empty when there is no duration or nothing is asked for.

    The pre-passes that seek a few dozen frames before the main pass
    (``clock_calibration``, ``side_overlay``) all sample through this."""
    if duration_sec <= 0 or count <= 0:
        return []
    first = duration_sec * span[0]
    last = duration_sec * span[1]
    if count == 1 or last - first < MIN_SAMPLE_STEP_SEC:
        return [round((first + last) / 2, 3)]
    step = (last - first) / (count - 1)
    if step < MIN_SAMPLE_STEP_SEC:
        count = int((last - first) // MIN_SAMPLE_STEP_SEC) + 1
        step = (last - first) / max(count - 1, 1)
    return [round(first + i * step, 3) for i in range(count)]
