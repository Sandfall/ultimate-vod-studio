"""What the pipeline draws on a sampled frame for the run's live preview: a box around
everything that frame was read from.

During analysis the pipeline writes each sampled frame to a file the GUI's Preview pane
polls (``cli.pipeline.render_preview``). This module turns that sample's Event into the
marks drawn on it, so a person watching a run can see where the tags, the round and the
clock are being read. Boxes only: the reads themselves are not drawn (text over the frame
is clutter at the Preview pane's size):

* the band texts (the built-in "Auto" reading's top and bottom bands), one box per text;
* the run's overlay regions (a drawn layout's boxes, or the rows the side-bar pre-pass
  locked), each in its region colour;
* the clock, at the box it was read in.

A solid box marks a read that found something. A layout region whose read came back
empty is still shown, as a sparse dotted outline in its colour: the box is where the run
is looking, and a person watching can see at once that a drawn box sits on nothing (a
layout drawn for another stream, say) rather than wonder whether it is being read at all.
Nothing is drawn for a sample the timer gate skipped (no clock on screen: the overlay
reads never ran), nor for a band text or a clock that was not read: those have no box
of their own to point at.

Qt-free and model-free; the colours are ``preview_regions.REGION_RGB``, the palette the
Analyze page's own preview and the layout editor draw the same regions in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from vodparse.bbox import quad_to_bbox
from vodparse.preview_regions import PREVIEW_BAND_TOP, PREVIEW_CLOCK_CALIBRATED, REGION_RGB
from vodparse.schema import BBox, Event

# ── Drawing ─────────────────────────────────────────────────────────────────────────

MARK_BOX_THICKNESS = 2

# A dotted outline is a row of filled dots along each edge, one every MARK_DOT_SPACING
# source pixels, each a filled circle of MARK_DOT_RADIUS (about 2r + 1 px across). Sparse
# on purpose: the Preview pane shows a 1080p frame at about a third of its size, where a
# tight dotted line reads as solid, and the outline is meant to look fainter than a box
# that read something.
MARK_DOT_SPACING = 14
MARK_DOT_RADIUS = 2

# "solid": the region read something. "dotted": a layout region whose read came back empty.
MarkStyle = Literal["solid", "dotted"]


@dataclass(frozen=True)
class PreviewMark:
    """One box to draw on a preview frame: where (source-frame pixels), its colour as RGB,
    and whether it is a solid box (something was read there) or a dotted outline (a layout
    region that read nothing)."""

    bbox: BBox
    rgb: tuple[int, int, int]
    style: MarkStyle = "solid"


def sample_preview_marks(event: Event, region_boxes: dict[str, BBox]) -> list[PreviewMark]:
    """The marks for one sample, in drawing order: band texts, then the run's overlay
    regions (`region_boxes`: region key → box, empty when the run has none), then the
    clock. A region that was read but came back empty is marked dotted; everything else
    is marked only where something was read (see the module docstring)."""
    marks: list[PreviewMark] = []
    for text in event["texts"]:
        marks.append(PreviewMark(quad_to_bbox(text), REGION_RGB[PREVIEW_BAND_TOP]))
    for key, read in event.get("layout_reads", {}).items():
        if key not in region_boxes:
            continue
        style: MarkStyle = "solid" if read["text"] else "dotted"
        marks.append(PreviewMark(region_boxes[key], REGION_RGB[key], style))
    timer = event["timer"]
    if timer is not None:
        marks.append(PreviewMark(timer["bbox"], REGION_RGB[PREVIEW_CLOCK_CALIBRATED]))
    return marks


def draw_preview_marks(display: np.ndarray, marks: list[PreviewMark]) -> None:
    """Draw `marks` onto `display` in place (the caller passes a copy of the frame)."""
    for mark in marks:
        x1, y1, x2, y2 = mark.bbox
        red, green, blue = mark.rgb
        colour = (blue, green, red)   # OpenCV draws in BGR
        if mark.style == "dotted":
            _draw_dotted_rectangle(display, mark.bbox, colour)
        else:
            cv2.rectangle(display, (x1, y1), (x2, y2), colour, MARK_BOX_THICKNESS)


def _draw_dotted_rectangle(display: np.ndarray, bbox: BBox,
                           colour: tuple[int, int, int]) -> None:
    """A dotted outline of `bbox`: filled dots every MARK_DOT_SPACING pixels along each
    edge, starting at the corners (OpenCV has no dotted line style). `colour` is BGR."""
    x1, y1, x2, y2 = bbox
    for x in range(x1, x2 + 1, MARK_DOT_SPACING):      # top and bottom edges
        cv2.circle(display, (x, y1), MARK_DOT_RADIUS, colour, cv2.FILLED)
        cv2.circle(display, (x, y2), MARK_DOT_RADIUS, colour, cv2.FILLED)
    for y in range(y1, y2 + 1, MARK_DOT_SPACING):      # left and right edges
        cv2.circle(display, (x1, y), MARK_DOT_RADIUS, colour, cv2.FILLED)
        cv2.circle(display, (x2, y), MARK_DOT_RADIUS, colour, cv2.FILLED)
