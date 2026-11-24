"""Find where a VOD's stream prints the player names in its side bars, for the main
screen's layout preview: the sidecar behind "Finding player tags…".

A game whose picture leaves bars beside it (Melee) has its player names read from rows in
those bars, and where the rows are is only known after a search over many frames
(``vodparse.side_overlay``). An Auto run does that search as a pre-pass; this stage runs
the same search alone (same samples, same rules, same OCR engine at the same canvas) so
the preview can draw and read the rows the run will lock, and so the run can be handed the
answer instead of searching again (``vodparse.prepass_handover``).

A separate one-shot process, like ``vodparse.cli.clock_locate``, because the GUI process
never imports torch. It prints progress lines while seeking and exactly one result line,
then exits; both are prefixed so the GUI can tell them from model-loading chatter::

    SIDE_OVERLAY_LOCATE_PROGRESS {"sample": 12, "samples": 30}
    SIDE_OVERLAY_LOCATE_RESULT {"side_overlay": {...SideOverlayScan.as_config()...},
                                "error": null}

Every failure (a game with no side bars, docTR's models not downloaded, an unopenable
video, an unexpected exception) comes back as a result line carrying ``error`` with exit
status 0, so the GUI has one thing to parse. Try it by hand from a checkout::

    PYTHONPATH=src python -m vodparse.cli.side_overlay_locate path/to/vod.mp4 \\
        --game melee --ocr-engine doctr --clock-bbox 797 108 1162 186

``--clock-bbox`` is the calibrated clock box (what ``cli.clock_locate`` printed as
``bbox``): the bars are placed from it. Without one the picture is assumed centred, as a
run assumes when its own clock pre-pass finds no steady spot.

Module-level imports stay light (argparse/json + the Qt-free registries): the GUI imports
this module for the protocol constants and the result model, and must never pay for torch
or cv2 by doing so. The heavy imports happen inside :func:`locate`.
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
from vodparse.ocr_engine import OCR_ENGINE_DEFAULT, OCR_ENGINE_LABELS
from vodparse.ocr_langs import OCR_LANG_DEFAULT, OCR_LANG_LABELS
from vodparse.overlay_regions import SideOverlayRecord
from vodparse.schema import BBox

# ── Protocol ────────────────────────────────────────────────────────────────────────

RESULT_LINE_PREFIX = "SIDE_OVERLAY_LOCATE_RESULT "
"""The stdout marker for the one result line; shared with vodparse.gui.side_overlay_locate."""

PROGRESS_LINE_PREFIX = "SIDE_OVERLAY_LOCATE_PROGRESS "
"""The stdout marker for a progress line, ``{"sample": i, "samples": n}`` (1-based)."""

ProgressCallback = Callable[[int, int], None]
"""What ``locate`` calls after each sampled frame: (frames seen so far, frames planned)."""

NO_CLOCK_DETECTOR_ERROR = "No clock detector for this game."


# ── Result ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SideOverlayLocateResult:
    """What the search concluded: its record (``overlay_regions.SideOverlayRecord``: the
    status, the locked boxes when it locked any, and the whole JSON record a run would
    save), or ``error``, a user-facing reason nothing ran. Neither when the search was
    skipped with nothing to say (``not_searched``)."""

    record: SideOverlayRecord | None
    error: str | None = None

    @classmethod
    def failure(cls, error: str) -> SideOverlayLocateResult:
        """A result that says why the search could not run."""
        return cls(record=None, error=error)

    @classmethod
    def not_searched(cls) -> SideOverlayLocateResult:
        """The answer the GUI gives in the search's place when the clock search found the
        detector firing on none of its frames: only frames with a clock count, so the
        search would read its frames for nothing (the run skips its pre-pass the same
        way). No error: the clock search's own status line already says no clock was found."""
        return cls(record=None, error=None)

    def to_json(self) -> dict[str, Any]:
        """The JSON object printed after ``RESULT_LINE_PREFIX``."""
        return {"side_overlay": None if self.record is None else self.record.config,
                "error": self.error}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> SideOverlayLocateResult:
        """Rebuild a result from its JSON object, defensively: a record of the wrong shape
        (a build mismatch) reads as absent rather than raising."""
        error = payload.get("error")
        return cls(record=SideOverlayRecord.from_config(payload.get("side_overlay")),
                   error=error if isinstance(error, str) else None)


def parse_result_line(line: str) -> SideOverlayLocateResult | None:
    """The result carried by a stdout line, or None when the line is not a result line
    (progress, model-loading chatter) or its JSON is unreadable."""
    payload = _prefixed_json(line, RESULT_LINE_PREFIX)
    if payload is None:
        return None
    return SideOverlayLocateResult.from_json(payload)


def parse_progress_line(line: str) -> tuple[int, int] | None:
    """The ``(sample, samples)`` pair carried by a progress line, or None for any other
    line or a malformed one."""
    payload = _prefixed_json(line, PROGRESS_LINE_PREFIX)
    if payload is None:
        return None
    sample, samples = payload.get("sample"), payload.get("samples")
    if not (isinstance(sample, int) and isinstance(samples, int)):
        return None
    return sample, samples


def _prefixed_json(line: str, prefix: str) -> dict[str, Any] | None:
    """The JSON object after ``prefix`` on ``line``, or None when the line does not start
    with it or what follows is not a JSON object."""
    line = line.strip()
    if not line.startswith(prefix):
        return None
    try:
        payload = json.loads(line[len(prefix):])
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


# ── The search ──────────────────────────────────────────────────────────────────────

def _load_band_engine(engine_name: str, lang: str) -> Any:
    """The overlay engine that reads whole bands, built the way a run builds it. A test
    seam: tests replace this with a fake engine so no models or torch are needed."""
    from vodparse.device import resolve_device
    from vodparse.ocr_engine import make_overlay_engine

    engine, _reader = make_overlay_engine(engine_name, lang, resolve_device())
    return engine


def _ingame_check(game: Game) -> Callable[[Any], bool] | None:
    """Whether a sampled frame is in-game, by the game's clock detector firing on it, as
    the run's own pre-pass checks (``cli.pipeline._ingame_check``). None when the detector's
    weight is not in this build: the search then refuses (NO_CLOCK_DETECTOR_ERROR), as a
    run without it skips its pre-pass. A test seam beside ``_load_band_engine``."""
    assert game.weights is not None and game.clock_hud is not None   # checked by locate
    if game.weights.timer is None or not Path(weight_path(game.weights.timer)).exists():
        return None
    from vodparse.cli.clock_locate import _load_timer_model
    from vodparse.device import resolve_device
    from vodparse.games import timer_infer_size_for
    from vodparse.timer import localize_timer

    timer_model = _load_timer_model(game)
    device = resolve_device()
    infer_size = timer_infer_size_for(game)
    clock_hud = game.clock_hud

    def is_ingame(frame: Any) -> bool:
        return localize_timer(timer_model, frame, device, infer_size,
                              clock_hud=clock_hud) is not None

    return is_ingame


def _video_shape(video_path: str) -> tuple[float, tuple[int, int]] | None:
    """The video's duration and its decoded frame size (width, height), or None when it
    cannot be opened or decodes no frame. Read up front: the frame size places the bars,
    and the duration gives the planned sample count for progress."""
    from vodparse.frame_reader import VideoFrameSource

    try:
        source = VideoFrameSource(video_path)
    except ValueError:
        return None
    with source:
        frame_size = source.frame_size()
        if frame_size is None:
            return None
        return source.duration_sec(), frame_size


def locate(video_path: str, game: Game, ocr_engine: str, ocr_lang: str, *,
           clock_bbox: BBox | None = None,
           on_progress: ProgressCallback | None = None) -> SideOverlayLocateResult:
    """Run the side-bar search on ``video_path`` for ``game``, reading with ``ocr_engine``
    in ``ocr_lang``, and report what it concluded. ``clock_bbox`` is the calibrated clock
    box the bars are placed from (None: the picture is assumed centred).

    Never raises for the expected failures: a game with no side bars, docTR named before
    its models are downloaded or python-doctr not installed (this stage never downloads
    the models, only the Analyze page's own offer may, and never searches with another
    engine instead), or a video that will not open each return a result whose ``error``
    says so."""
    from vodparse import side_overlay
    from vodparse.ocr_engine import OCR_CANVAS_DEFAULT, engine_unusable_reason

    if not game.has_side_bars or game.clock_hud is None or game.weights is None:
        return SideOverlayLocateResult.failure(
            f"{game.display_name} has no side bars to search for player tags.")
    placement = game.clock_hud.feed_placement
    assert placement is not None   # has_side_bars is exactly this
    unusable = engine_unusable_reason(ocr_engine)
    if unusable is not None:
        return SideOverlayLocateResult.failure(unusable)
    shape = _video_shape(video_path)
    if shape is None:
        return SideOverlayLocateResult.failure(f"Couldn't open the video: {video_path}")
    duration, (frame_width, frame_height) = shape

    feed = side_overlay.feed_span_for(clock_bbox, frame_width, frame_height, placement)
    # The search samples round by round and stops once enough frames were in-game
    # (side_overlay.sample_rounds), so the total is one round's until a frame of the next
    # arrives: "31 / 60", never "31 / 30".
    round_sizes = [len(timestamps) for timestamps in side_overlay.sample_rounds(duration)]
    planned = round_sizes[0]
    rounds_started = 1
    seen = 0

    def on_frame(_frame: Any, _boxes: list[BBox], _timestamp_sec: float) -> None:
        nonlocal seen, planned, rounds_started
        seen += 1
        while seen > planned and rounds_started < len(round_sizes):
            planned += round_sizes[rounds_started]
            rounds_started += 1
        if on_progress is not None:
            on_progress(seen, planned)

    is_ingame = _ingame_check(game)
    if is_ingame is None:
        return SideOverlayLocateResult.failure(NO_CLOCK_DETECTOR_ERROR)
    scan = side_overlay.locate_side_overlay(
        video_path, _load_band_engine(ocr_engine, ocr_lang), OCR_CANVAS_DEFAULT, feed,
        is_ingame=is_ingame, on_frame=on_frame)
    if scan is None:
        return SideOverlayLocateResult.failure(f"Couldn't sample the video: {video_path}")
    return SideOverlayLocateResult(record=SideOverlayRecord.from_config(scan.as_config()))


# ── CLI ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the side-bar player tag search on a VOD and print what it locked "
                    "(one prefixed JSON line); used by the GUI's layout preview. See the "
                    "module docstring for the protocol.")
    parser.add_argument("video", help="The VOD to search")
    parser.add_argument("--game", default=GAME_DEFAULT, choices=[game.id for game in GAMES],
                        help="Which game the VOD is of (default: %(default)s)")
    parser.add_argument("--ocr-engine", default=OCR_ENGINE_DEFAULT,
                        choices=sorted(OCR_ENGINE_LABELS),
                        help="Which engine reads the bars (default: %(default)s)")
    parser.add_argument("--ocr-lang", default=OCR_LANG_DEFAULT, choices=sorted(OCR_LANG_LABELS),
                        help="The overlay language to read (default: %(default)s)")
    parser.add_argument("--clock-bbox", type=int, nargs=4, default=None,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="The calibrated clock box the bars are placed from (default: "
                             "the picture is assumed centred)")
    return parser


def _print_progress(sample: int, samples: int) -> None:
    print(PROGRESS_LINE_PREFIX + json.dumps({"sample": sample, "samples": samples}), flush=True)


def main(argv: list[str] | None = None) -> int:
    """Parse the argv, run the search, and print exactly one result line. An unexpected
    exception, and an engine/language pair that cannot be built (``SystemExit`` from the
    engine factory), are reported the same way (as an ``error`` result) so the GUI's one
    parser covers them; the exit status is 0 either way because the line, not the status,
    is the contract."""
    from vodparse.cli.stage_io import prepare_stage_streams

    prepare_stage_streams()
    args = build_arg_parser().parse_args(argv)
    game = game_by_id(args.game)
    assert game is not None  # argparse's choices are the registry ids
    try:
        result = locate(args.video, game, args.ocr_engine, args.ocr_lang,
                        clock_bbox=args.clock_bbox, on_progress=_print_progress)
    except SystemExit as exc:   # make_overlay_engine's actionable "not usable" message
        result = SideOverlayLocateResult.failure(str(exc))
    except Exception as exc:  # noqa: BLE001 - one parser on the GUI side; report, don't crash
        result = SideOverlayLocateResult.failure(f"The player tag search failed: {exc}")
    print(RESULT_LINE_PREFIX + json.dumps(result.to_json()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
