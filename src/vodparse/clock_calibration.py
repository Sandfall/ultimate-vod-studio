"""Calibrating a VOD's clock box before the main pass (the "calibrated clock box").

A game like Rivals 2 draws its match clock at one screen spot for the whole
VOD, and the clock detector finds that spot with high confidence on most frames of any
stream it is pointed at. What it does not do reliably is fire on EVERY frame of a stream
it did not train on: on a second stream it went blank for whole games, and a game that
opens with a miss is lost because the dropped-read recovery (timer.TimerRecovery) only
re-reads its remembered box for a couple of frames and rejects a rising clock. A box the
user draws in a custom overlay layout has none of that problem: the pipeline reads it on
every sample and the same VOD yields every match.

So: before the main pass, seek a few dozen frames spread across the VOD, run ONLY the
clock detector on them (no OCR, no other detector), and if enough of its boxes agree on
one spot, take their median as the run's calibrated clock box. The main pass then reads
that box on every sample the way it reads a drawn one, with the detector as the fallback
for a frame the box does not parse (vodparse.cli.pipeline._sample_timer). A stream that
moves its clock (a zoomed feed scene, two layouts) never agrees closely enough and gets
today's detector-only behaviour.

Cost: the pre-pass is seek-bound (YOLOv8-nano is milliseconds a frame), about
CLOCK_CALIBRATION_SAMPLES seeks, a few percent of a full run.

Qt-free and video-library-light: the pure parts (`clock_sample_timestamps`,
`calibrate_from_boxes`) are unit-tested on hand-made boxes; `calibrate_clock_box` is the
seeking wrapper, on a `frame_reader.VideoFrameSource`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from vodparse.frame_reader import VideoFrameSource, sample_timestamps
from vodparse.games import ClockHud
from vodparse.schema import BBox
from vodparse.timer import localize_timer

# ── Tunables ────────────────────────────────────────────────────────────────────────

CLOCK_CALIBRATION_SAMPLES = 60
"""Frames the pre-pass looks at, spread evenly across the VOD so every game contributes
and a long pre-show cannot starve it."""

CLOCK_CALIBRATION_EDGE_FRAC = 0.05
"""Skip this fraction of the VOD at either end (stream intro/outro cards)."""

CLOCK_CALIBRATION_SPAN = (CLOCK_CALIBRATION_EDGE_FRAC, 1.0 - CLOCK_CALIBRATION_EDGE_FRAC)
"""The stretch of the VOD the samples are spread over, as fractions of its duration."""

CLOCK_CALIBRATION_MIN_HITS = 12
"""Detector boxes that must agree before a spot is trusted. Fewer means the detector is
not finding this stream's clock often enough to say where it is."""

CLOCK_CALIBRATION_MAX_CENTER_SPREAD_FRAC = 0.015
"""The median absolute deviation of the hits' box centres, each axis, as a fraction of
the frame's long side (about 29 px at 1080p). A clock that moves between scenes spreads
wider than this and is refused."""

CLOCK_CALIBRATION_MAX_SIZE_SPREAD = 0.25
"""The same on box width and height, relative to their medians."""

CLOCK_LANDING_MAX_CENTER_OFFSET_FRAC = 0.03
"""How far one hit's box centre may sit from the calibrated box's, each axis, as a
fraction of the frame's long side (about 58 px at 1080p), for that hit to count as the
clock at its spot (``box_agrees``). Twice the calibration's own spread limit: one box is
judged alone here, with no median to absorb the detector's jitter. A hit on a shrunken
game window (a stream showing the game in a corner during a break) sits hundreds of
pixels off and is left out."""


# ── Result ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClockCalibration:
    """The calibrated clock box and how it was earned, in source-frame pixels."""

    bbox: BBox
    hits: int
    samples: int
    center_spread_frac: float
    size_spread: float

    def as_config(self) -> dict[str, Any]:
        """The JSON-ready record written under the run's config."""
        return {
            "bbox": list(self.bbox),
            "hits": self.hits,
            "samples": self.samples,
            "center_spread_frac": round(self.center_spread_frac, 4),
            "size_spread": round(self.size_spread, 3),
        }

    @classmethod
    def from_config(cls, record: object,
                    frame_size: tuple[int, int]) -> ClockCalibration | None:
        """A calibration read back from its ``as_config`` record, for a run handed the box
        the layout preview already found (``vodparse.prepass_handover``). None when the
        record is absent, malformed, or its box does not sit inside a frame of
        ``frame_size`` (width, height): the run then calibrates for itself."""
        if not isinstance(record, dict):
            return None
        bbox = record.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4
                and all(isinstance(v, int) and not isinstance(v, bool) for v in bbox)):
            return None
        frame_width, frame_height = frame_size
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= frame_width and 0 <= y1 < y2 <= frame_height):
            return None
        hits, samples = record.get("hits"), record.get("samples")
        center_spread, size_spread = record.get("center_spread_frac"), record.get("size_spread")
        if not (isinstance(hits, int) and isinstance(samples, int)
                and isinstance(center_spread, (int, float))
                and isinstance(size_spread, (int, float))):
            return None
        return cls(bbox=list(bbox), hits=hits, samples=samples,
                   center_spread_frac=float(center_spread), size_spread=float(size_spread))


# ── Pure parts ──────────────────────────────────────────────────────────────────────

def clock_sample_timestamps(duration_sec: float,
                            count: int = CLOCK_CALIBRATION_SAMPLES) -> list[float]:
    """The timestamps the pre-pass seeks: ``count`` spread evenly over
    ``CLOCK_CALIBRATION_SPAN`` (``frame_reader.sample_timestamps``). The one place that
    spreading is decided, so a caller counting the samples ahead of the pre-pass (the
    layout preview's progress, ``cli.clock_locate``) plans exactly the frames
    ``calibrate_clock_box`` will seek."""
    return sample_timestamps(duration_sec, count, CLOCK_CALIBRATION_SPAN)


def _mad(values: Sequence[float]) -> float:
    """Median absolute deviation: how far a typical value sits from the median."""
    centre = median(values)
    return median(abs(v - centre) for v in values)


def calibrate_from_boxes(boxes: Sequence[BBox], frame_size: tuple[int, int], *,
                         samples: int,
                         min_hits: int = CLOCK_CALIBRATION_MIN_HITS,
                         max_center_spread_frac: float = CLOCK_CALIBRATION_MAX_CENTER_SPREAD_FRAC,
                         max_size_spread: float = CLOCK_CALIBRATION_MAX_SIZE_SPREAD
                         ) -> ClockCalibration | None:
    """The calibrated box from the detector's hits, or None when they are too few or do
    not agree on one spot.

    ``boxes`` are [x1, y1, x2, y2] in source pixels; ``frame_size`` is (width, height).
    Agreement is judged by the median absolute deviation of the box centres on each axis
    (against the frame's long side) and of the widths and heights (against their medians),
    so one stray box cannot spoil a run and one stray box cannot fake one either. The
    returned box is the per-coordinate median of every hit.
    """
    if len(boxes) < min_hits:
        return None
    long_side = float(max(frame_size))
    xs = [(b[0] + b[2]) / 2 for b in boxes]
    ys = [(b[1] + b[3]) / 2 for b in boxes]
    widths = [float(b[2] - b[0]) for b in boxes]
    heights = [float(b[3] - b[1]) for b in boxes]
    center_spread = max(_mad(xs), _mad(ys)) / long_side
    median_w, median_h = median(widths), median(heights)
    if median_w <= 0 or median_h <= 0:
        return None
    size_spread = max(_mad(widths) / median_w, _mad(heights) / median_h)
    if center_spread > max_center_spread_frac or size_spread > max_size_spread:
        return None
    bbox = [int(round(median(coord))) for coord in zip(*boxes)]
    return ClockCalibration(bbox=bbox, hits=len(boxes), samples=samples,
                            center_spread_frac=center_spread, size_spread=size_spread)


def box_agrees(box: BBox, calibrated: BBox, frame_size: tuple[int, int], *,
               max_center_offset_frac: float = CLOCK_LANDING_MAX_CENTER_OFFSET_FRAC,
               max_size_spread: float = CLOCK_CALIBRATION_MAX_SIZE_SPREAD) -> bool:
    """Whether one detector hit sits on the calibrated clock spot: its centre within
    ``max_center_offset_frac`` of the frame's long side on each axis, and its width and
    height within ``max_size_spread`` of the calibrated box's. Used to pick the sampled
    frames the clock search reads the clock on (``cli.clock_locate``): a fire elsewhere on
    the frame is a clock the run would not read at the calibrated box."""
    long_side = float(max(frame_size))
    dx = abs((box[0] + box[2]) / 2 - (calibrated[0] + calibrated[2]) / 2)
    dy = abs((box[1] + box[3]) / 2 - (calibrated[1] + calibrated[3]) / 2)
    if max(dx, dy) > max_center_offset_frac * long_side:
        return False
    width, height = float(box[2] - box[0]), float(box[3] - box[1])
    cal_width, cal_height = float(calibrated[2] - calibrated[0]), float(calibrated[3] - calibrated[1])
    if cal_width <= 0 or cal_height <= 0:
        return False
    return (abs(width - cal_width) / cal_width <= max_size_spread
            and abs(height - cal_height) / cal_height <= max_size_spread)


# ── The seeking wrapper ─────────────────────────────────────────────────────────────

FrameCallback = Callable[[np.ndarray, BBox | None, float], None]
"""What ``calibrate_clock_box`` calls per sampled frame: the frame, the detector's box on it
(None for a miss) and the timestamp. The pipeline uses it to show the frames it is looking
at in the GUI's preview."""


def calibrate_clock_box(video_path: str | Path, timer_model: Any, device: str,
                        infer_size: int, clock_hud: ClockHud, *,
                        samples: int = CLOCK_CALIBRATION_SAMPLES,
                        min_hits: int = CLOCK_CALIBRATION_MIN_HITS,
                        on_frame: FrameCallback | None = None) -> ClockCalibration | None:
    """Seek ``samples`` frames across the video (``clock_sample_timestamps``), run the clock
    detector alone on each, and calibrate from the boxes it returns
    (``calibrate_from_boxes``, against the decoded frames' size). Opens its own capture and
    releases it; the main pass opens the video afresh afterwards. None when the video
    cannot be opened, has no duration, decodes no frame, or the hits do not agree.
    ``on_frame`` sees every sampled frame with its box, for a live preview."""
    try:
        source = VideoFrameSource(str(video_path))
    except ValueError:
        return None
    with source:
        timestamps = clock_sample_timestamps(source.duration_sec(), samples)
        boxes: list[BBox] = []
        for ts, frame in source.frames_at(timestamps):
            located = localize_timer(timer_model, frame, device, infer_size, clock_hud=clock_hud)
            if located is not None:
                boxes.append(located[0])
            if on_frame is not None:
                on_frame(frame, None if located is None else located[0], ts)
        frame_size = source.frame_size()
    if frame_size is None:
        return None
    return calibrate_from_boxes(boxes, frame_size, samples=len(timestamps), min_hits=min_hits)
