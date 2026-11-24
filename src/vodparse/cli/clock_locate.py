"""Find a VOD's match clock for the main screen's layout preview: the sidecar behind
"Finding the clock…".

The Analyze page's Preview pane wants two things from the pipeline's clock pre-pass
(``vodparse.clock_calibration.calibrate_clock_box``): the calibrated clock box, to draw
over the frame, and a sampled timestamp where the clock is *read* at that box, to land
the scrubber on an in-game frame. This stage runs that same pre-pass alone (same samples,
same agreement test, same detector at the game's own inference size), so what the
preview shows is what an Auto run will lock; then it seeks back to the hits that sit on
the calibrated spot, earliest first, and reads the clock there the way the run reads its
calibrated box, until one parses (``first_read_sec``). A detector fire alone is not a
landing: it fires on the game-end freeze, where the digits do not read, and on a game
shown in a corner window during a break, where the calibrated box holds nothing. The
first fire is still reported (``first_hit_sec``) for the status line, and its box
(``first_hit_bbox``) picks the readout swatch when the calibration refuses.

It is a separate process, like ``vodparse.cli.overlay_read``, because the GUI process
never imports torch: running inference on an in-process Qt thread fatally crashed the
interpreter on macOS. One-shot: it prints progress lines while seeking and exactly one
result line, then exits. Both are prefixed so the GUI can tell them from model-loading
chatter (the ``overlay_read.RESULT_LINE_PREFIX`` idiom)::

    CLOCK_LOCATE_PROGRESS {"sample": 12, "samples": 60}
    CLOCK_LOCATE_RESULT {"first_hit_sec": 312.5, "first_hit_bbox": [...], "bbox": [...],
                         "calibration": {...}, "hits": 41, "samples": 60, "error": null,
                         "first_read_sec": 312.5}

Every failure (a game with no clock detector, a missing weight, an unopenable video, an
unexpected exception) comes back as a result line carrying ``error`` with exit status 0,
so the GUI has one thing to parse. Try it by hand from a checkout::

    PYTHONPATH=src python -m vodparse.cli.clock_locate path/to/vod.mp4 --game melee

Module-level imports stay light (argparse/json + the Qt-free games registry): the GUI
imports this module for the protocol constants and the result model, and must never pay
for torch or cv2 by doing so. The heavy imports happen inside :func:`locate`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vodparse.games import GAME_DEFAULT, GAMES, Game, game_by_id, weight_path
from vodparse.schema import BBox

# ── Protocol ────────────────────────────────────────────────────────────────────────

RESULT_LINE_PREFIX = "CLOCK_LOCATE_RESULT "
"""The stdout marker for the one result line; shared with vodparse.gui.clock_locate."""

PROGRESS_LINE_PREFIX = "CLOCK_LOCATE_PROGRESS "
"""The stdout marker for a progress line, ``{"sample": i, "samples": n}`` (1-based)."""

ProgressCallback = Callable[[int, int], None]
"""What ``locate`` calls after each sampled frame: (frames seen so far, frames planned)."""

# The stage's own wording for the one failure the user can fix; the same guidance the
# pipeline gives when it refuses a run for the same reason (cli.pipeline._load_detectors).
WEIGHTS_HINT = "Fetch the weights with `python -m vodparse.cli.download_weights`."

CLOCK_READ_ATTEMPTS = 8
"""How many agreeing hits are seeked back to and read before the search gives up on a
readable frame (``first_read_sec`` stays None and the preview stays on the first frame).
Each attempt is one seek and one OCR of the clock box; the first or second usually reads."""


# ── Result ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClockLocateResult:
    """What the search found, in source-frame pixels.

    ``first_hit_sec``/``first_hit_bbox`` are the earliest sampled frame the detector fired
    on and its box there (None when it never fired). ``bbox`` is the calibrated box (the
    median of the agreeing hits) and ``calibration`` its ``ClockCalibration.as_config()``
    record; both None when the hits were too few or disagreed. ``first_read_sec`` is the
    earliest sampled frame where the clock was read at the calibrated box (see the module
    docstring): where the preview lands; None when there is no calibrated box or no hit
    read within CLOCK_READ_ATTEMPTS. ``hits``/``samples`` feed the status line.
    ``error`` is a user-facing reason nothing ran, with every other field at its empty
    value."""

    first_hit_sec: float | None
    first_hit_bbox: BBox | None
    bbox: BBox | None
    calibration: dict[str, Any] | None
    hits: int
    samples: int
    error: str | None = None
    first_read_sec: float | None = None

    @classmethod
    def failure(cls, error: str) -> ClockLocateResult:
        """A result that says why the search could not run."""
        return cls(first_hit_sec=None, first_hit_bbox=None, bbox=None, calibration=None,
                   hits=0, samples=0, error=error)

    def to_json(self) -> dict[str, Any]:
        """The JSON object printed after ``RESULT_LINE_PREFIX``."""
        return {
            "first_hit_sec": self.first_hit_sec,
            "first_hit_bbox": None if self.first_hit_bbox is None else list(self.first_hit_bbox),
            "bbox": None if self.bbox is None else list(self.bbox),
            "calibration": self.calibration,
            "hits": self.hits,
            "samples": self.samples,
            "error": self.error,
            "first_read_sec": self.first_read_sec,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ClockLocateResult:
        """Rebuild a result from its JSON object, defensively: a field of the wrong shape
        (a hand-edited record, a build mismatch) reads as absent rather than raising, so
        the GUI never crashes on a line it did not write itself."""
        calibration = payload.get("calibration")
        error = payload.get("error")
        return cls(
            first_hit_sec=_as_float(payload.get("first_hit_sec")),
            first_hit_bbox=_as_bbox(payload.get("first_hit_bbox")),
            bbox=_as_bbox(payload.get("bbox")),
            calibration=calibration if isinstance(calibration, dict) else None,
            hits=_as_count(payload.get("hits")),
            samples=_as_count(payload.get("samples")),
            error=error if isinstance(error, str) else None,
            first_read_sec=_as_float(payload.get("first_read_sec")),
        )


def _is_number(value: Any) -> bool:
    """True for an int or float, never for a bool (which json would also accept)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_float(value: Any) -> float | None:
    """A number as a float, or None for anything else."""
    return float(value) if _is_number(value) else None


def _as_count(value: Any) -> int:
    """A number as an int, or 0 for anything else."""
    return int(value) if _is_number(value) else 0


def _as_bbox(value: Any) -> BBox | None:
    """A ``[x1, y1, x2, y2]`` list of ints, or None for anything else."""
    if isinstance(value, list) and len(value) == 4 and all(_is_number(v) for v in value):
        return [int(v) for v in value]
    return None


def parse_result_line(line: str) -> ClockLocateResult | None:
    """The result carried by a stdout line, or None when the line is not a result line
    (progress, model-loading chatter) or its JSON is unreadable."""
    line = line.strip()
    if not line.startswith(RESULT_LINE_PREFIX):
        return None
    try:
        payload = json.loads(line[len(RESULT_LINE_PREFIX):])
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    return ClockLocateResult.from_json(payload)


def parse_progress_line(line: str) -> tuple[int, int] | None:
    """The ``(sample, samples)`` pair carried by a progress line, or None for any other
    line or a malformed one."""
    line = line.strip()
    if not line.startswith(PROGRESS_LINE_PREFIX):
        return None
    try:
        payload = json.loads(line[len(PROGRESS_LINE_PREFIX):])
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    sample, samples = payload.get("sample"), payload.get("samples")
    if not (isinstance(sample, int) and isinstance(samples, int)):
        return None
    return sample, samples


# ── The search ──────────────────────────────────────────────────────────────────────

def _load_timer_model(game: Game) -> Any:
    """The game's clock detector, loaded the way the pipeline loads it. The test seam:
    tests replace this with a fake detector so no weights or torch are needed."""
    from ultralytics import YOLO

    assert game.weights is not None and game.weights.timer is not None  # checked by locate
    return YOLO(weight_path(game.weights.timer))


def _video_duration_sec(video_path: str) -> float | None:
    """The video's duration, or None when it cannot be opened. Read up front so the
    planned sample count is known for progress before the pre-pass starts; read the same
    way ``calibrate_clock_box`` reads it (``VideoFrameSource.duration_sec``), so the plan
    and the pre-pass sample the same timestamps."""
    from vodparse.frame_reader import VideoFrameSource

    try:
        source = VideoFrameSource(video_path)
    except ValueError:
        return None
    with source:
        return source.duration_sec()


def locate(video_path: str, game: Game, *, samples: int | None = None,
           min_hits: int | None = None,
           on_progress: ProgressCallback | None = None) -> ClockLocateResult:
    """Run the clock pre-pass on ``video_path`` for ``game`` and report what it found.

    ``samples``/``min_hits`` default to the pre-pass's own tunables
    (``clock_calibration.CLOCK_CALIBRATION_SAMPLES`` / ``_MIN_HITS``), resolved in here
    rather than in the signature because that module imports cv2 at load time and this
    one must stay import-light. The heavy imports (cv2, torch via the detector) happen
    here. Never raises for the expected failures: a game with no clock detector, a weight
    missing from this build, or a video that will not open each return a result whose
    ``error`` says so.

    The first hit is taken from the pre-pass's per-frame callback, so it is known even
    when the calibration itself refuses (too few hits, or a clock that moves between
    scenes): the preview can still draw the box seen there. The landing frame
    (``first_read_sec``) is found afterwards by ``_first_readable_hit``, from the hits
    the callback kept."""
    from vodparse import clock_calibration
    from vodparse.device import resolve_device
    from vodparse.games import timer_infer_size_for

    if game.weights is None or game.weights.timer is None or game.clock_hud is None:
        return ClockLocateResult.failure(
            f"{game.display_name} has no clock detector, so there is no clock to find.")
    timer_path = weight_path(game.weights.timer)
    if not Path(timer_path).exists():
        return ClockLocateResult.failure(
            f"{game.display_name}'s clock detector is not in this build "
            f"({timer_path} is missing). {WEIGHTS_HINT}")
    duration = _video_duration_sec(video_path)
    if duration is None:
        return ClockLocateResult.failure(f"Couldn't open the video: {video_path}")

    samples = samples if samples is not None else clock_calibration.CLOCK_CALIBRATION_SAMPLES
    min_hits = min_hits if min_hits is not None else clock_calibration.CLOCK_CALIBRATION_MIN_HITS
    planned = len(clock_calibration.clock_sample_timestamps(duration, samples))

    # The per-frame hook records what the calibration's own return value does not carry:
    # the first hit, and the hit count even when the calibration refuses.
    seen = 0
    hit_frames: list[tuple[float, BBox]] = []   # (timestamp, box) of every fire, in order
    frame_size: tuple[int, int] | None = None

    def on_frame(frame: Any, box: BBox | None, timestamp_sec: float) -> None:
        nonlocal seen, frame_size
        seen += 1
        if frame_size is None:
            frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        if box is not None:
            hit_frames.append((timestamp_sec, box))
        if on_progress is not None:
            on_progress(seen, planned)

    calibration = clock_calibration.calibrate_clock_box(
        video_path, _load_timer_model(game), resolve_device(), timer_infer_size_for(game),
        game.clock_hud, samples=samples, min_hits=min_hits, on_frame=on_frame)
    first_read_sec: float | None = None
    if calibration is not None and frame_size is not None:
        agreeing = [timestamp for timestamp, box in hit_frames
                    if clock_calibration.box_agrees(box, calibration.bbox, frame_size)]
        first_read_sec = _first_readable_hit(video_path, game, calibration.bbox, agreeing)
    first_hit_sec, first_hit_bbox = hit_frames[0] if hit_frames else (None, None)
    return ClockLocateResult(
        first_hit_sec=first_hit_sec,
        first_hit_bbox=first_hit_bbox,
        bbox=None if calibration is None else calibration.bbox,
        calibration=None if calibration is None else calibration.as_config(),
        hits=len(hit_frames),
        samples=planned,
        first_read_sec=first_read_sec,
    )


def _first_readable_hit(video_path: str, game: Game, calibrated: BBox,
                        hit_secs: list[float]) -> float | None:
    """The earliest of ``hit_secs`` (sampled timestamps whose detector box sits on the
    calibrated spot) where the clock reads at ``calibrated``, or None when none of the
    first CLOCK_READ_ATTEMPTS does. Seeks each frame afresh (the pre-pass keeps no frames)
    and reads it as the run reads its calibrated box, with the same always-English
    EasyOCR reader (see cli.overlay_read._timer_reader for why English).

    The landing is a convenience on top of a calibration that already succeeded, so
    anything that goes wrong here (the reader failing to load, a seek or a read raising)
    is noted on stderr and answered with None: the preview then stays on the frame it
    shows and reports the clock as found but not read, instead of main() turning the
    whole result into a failure and throwing the calibrated box away."""
    from vodparse.frame_reader import VideoFrameSource

    if not hit_secs:
        return None
    try:
        reader = _load_clock_reader()
        with VideoFrameSource(video_path) as source:
            for timestamp in hit_secs[:CLOCK_READ_ATTEMPTS]:
                frame = source.grab_at(timestamp)
                if frame is not None and _read_clock(reader, frame, calibrated, game) is not None:
                    return timestamp
    except Exception as exc:  # noqa: BLE001  (the calibration must survive any landing error)
        print(f"clock landing skipped: {exc}", file=sys.stderr)
    return None


def _load_clock_reader() -> Any:
    """The EasyOCR reader the clock is read with (a seam the tests replace)."""
    from vodparse.ocr_langs import make_reader

    return make_reader("en")


def _read_clock(reader: Any, frame: Any, calibrated: BBox, game: Game) -> Any:
    """One clock read at the calibrated box, as the run and the preview read it
    (``timer.read_calibrated_clock``); the parsed reading or None. A seam the tests
    replace, so no OCR model runs in them."""
    from vodparse.games import full_game_seconds_for
    from vodparse.timer import read_calibrated_clock

    assert game.clock_hud is not None   # locate() refused the game without one
    return read_calibrated_clock(reader, frame, calibrated, clock_hud=game.clock_hud,
                                 max_clock_seconds=float(full_game_seconds_for(game, None)))


# ── CLI ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the clock pre-pass on a VOD and print where the match clock is and "
                    "when it first appears (one prefixed JSON line); used by the GUI's "
                    "layout preview. See the module docstring for the protocol.")
    parser.add_argument("video", help="The VOD to search")
    parser.add_argument("--game", default=GAME_DEFAULT, choices=[game.id for game in GAMES],
                        help="Which game's clock detector to use (default: %(default)s)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Frames to sample across the VOD (default: the pipeline's "
                             "clock_calibration.CLOCK_CALIBRATION_SAMPLES)")
    parser.add_argument("--min-hits", type=int, default=None,
                        help="Detector hits that must agree before a box is trusted "
                             "(default: clock_calibration.CLOCK_CALIBRATION_MIN_HITS)")
    return parser


def _print_progress(sample: int, samples: int) -> None:
    print(PROGRESS_LINE_PREFIX + json.dumps({"sample": sample, "samples": samples}), flush=True)


def main(argv: list[str] | None = None) -> int:
    """Parse the argv, run the search, and print exactly one result line. An unexpected
    exception is reported the same way (as an ``error`` result) so the GUI's one parser
    covers it; the exit status is 0 either way because the line, not the status, is the
    contract."""
    from vodparse.cli.stage_io import prepare_stage_streams

    prepare_stage_streams()
    args = build_arg_parser().parse_args(argv)
    game = game_by_id(args.game)
    assert game is not None  # argparse's choices are the registry ids
    try:
        result = locate(args.video, game, samples=args.samples, min_hits=args.min_hits,
                        on_progress=_print_progress)
    except Exception as exc:  # noqa: BLE001 - one parser on the GUI side; report, don't crash
        result = ClockLocateResult.failure(f"The clock search failed: {exc}")
    print(RESULT_LINE_PREFIX + json.dumps(result.to_json()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
