"""Pause-aware start refinement: an opt-in postprocess, kept out of the main extraction path.

The timer-only extractor back-calculates each game's start from its clock (7:00 - remaining).
That is wrong when the game was *paused* before the first clock reading we saw: the real start
sits further back than the clock implies. This module reads the live video clock a few seconds
before a computed start and walks the start earlier to cover any pause.

It needs the video and the timer model loaded (TimerProbe), so it is deliberately separate from
the JSON-only extraction in match_extraction.py. It is NOT currently wired into the pipeline:
pauses before the first visible clock reading are rare. To enable it, run it as a postprocess
over the extracted matches (see the pointer comment in vodparse.cli.extract_matches)."""

from __future__ import annotations

from .match_extraction import FULL_GAME_SECONDS, valid_secs

PRECHECK_GAP_SEC = 5     # probe the video this many seconds before a computed start
MAX_REFINE_ITERS = 5     # cap on the pause walk-back (avoid an infinite loop)


def refine_start_for_pause(start_ts: float, probe: TimerProbe) -> float:
    """Walk a computed game-start timestamp earlier to cover a pause. Step back PRECHECK_GAP_SEC
    at a time, read the live clock there, and if the game was already running (clock shown),
    move the start to where that earlier reading implies the game began. Stops after
    MAX_REFINE_ITERS or once the clock is no longer visible (before the game / on a menu)."""
    start = start_ts
    for _ in range(MAX_REFINE_ITERS):
        probe_ts = start - PRECHECK_GAP_SEC
        earlier_remaining = probe.seconds_at(probe_ts)
        if earlier_remaining is None:
            break
        start = probe_ts - (FULL_GAME_SECONDS - earlier_remaining)
    return start


class TimerProbe:
    """Loads the timer model and OCR so the match clock can be read at any point in the
    video, used to check a calculated game start against a possible pause. Created only when
    both a video and a timer model are given, so runs that only read JSON don't need the
    machine-learning libraries."""

    def __init__(self, video_path: str, timer_model_path: str):
        # Imports are local so JSON-only runs never need cv2/torch/easyocr/ultralytics.
        import cv2
        import easyocr
        from ultralytics import YOLO
        from vodparse.device import resolve_device
        from vodparse.timer import detect_timer

        self._cv2 = cv2
        self._detect_timer = detect_timer
        # Prefer Apple GPU (mps), then NVIDIA (cuda), then CPU: see vodparse.device.
        self.device = resolve_device()
        print(f"Loading timer model on {self.device} for pause verification...")
        self.model = YOLO(timer_model_path)
        self.ocr = easyocr.Reader(["en"], gpu=True)
        # Hold the video open for the lifetime of the probe so seconds_at can seek into it.
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.cap.release()
            raise FileNotFoundError(f"Cannot open video: {video_path}")

    def seconds_at(self, timestamp_sec: float) -> float | None:
        """Timer 'total_seconds' at the given video time, or None if no clock is shown."""
        self.cap.set(self._cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
        ok, frame = self.cap.read()
        if not ok:
            return None
        res = self._detect_timer(self.model, self.ocr, frame, self.device)
        return valid_secs(res.get("total_seconds")) if res else None

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "TimerProbe":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
