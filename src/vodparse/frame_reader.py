"""Threaded video frame reader with seek support.

Decodes frames in a background thread so the caller's YOLO inference and
frame decode overlap: eliminating the serial decode→infer→decode→infer stall.

The thread applies backpressure: it stays at most ~one frame ahead of the consumer
rather than racing on. That overlap is the point in sequential reads, but it also
stops the reader from decoding (and discarding) every frame between two seeks, which,
in seek mode, was ~1000 wasted decodes per seek that stole CPU from inference on
machines without a spare core."""

import queue
import threading
from pathlib import Path
from typing import Any, override

import cv2


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
        with self._seek_lock:
            self._seek_target = frame_number
            self._generation += 1
        # Drop any frame already queued. A pre-seek frame the reader thread queues *after*
        # this drain still carries the old generation, so read() discards it.
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

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
