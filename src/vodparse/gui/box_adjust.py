"""
Box adjustment geometry for the overlay layout editor
------------------------------------------------------
The pure math behind dragging an already-drawn overlay box in the layout editor
(``vodparse.gui.region_pane.RegionDrawPane``): which part of which box sits under the
cursor, and where a box ends up after a move, a corner drag or an arrow-key nudge. Kept
Qt-free so it is unit-testable on plain numbers; the pane only translates mouse and key
events into these calls.

Everything is in frame pixels (the pane's scene coordinates), as floats. Boxes are
resized by their corners only; dragging inside a box moves it. Every result stays inside
the frame and at least ``min_size`` wide and tall, and a corner dragged past the opposite
one stops there rather than flipping the box.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Grab(Enum):
    """The part of a box a drag took hold of."""
    MOVE = "move"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


CORNER_GRABS = (Grab.TOP_LEFT, Grab.TOP_RIGHT, Grab.BOTTOM_LEFT, Grab.BOTTOM_RIGHT)

# A corner's grab zone is at most this fraction of the box's shorter side, so on a box
# only a few screen pixels tall the corners never swallow the whole box and it can still
# be grabbed in the middle to move it.
MAX_CORNER_ZONE_FRAC = 1 / 3


@dataclass(frozen=True)
class PixelBox:
    """A box in frame pixels: ``x1 < x2`` and ``y1 < y2``."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """The box's horizontal extent, in frame pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """The box's vertical extent, in frame pixels."""
        return self.y2 - self.y1

    def contains(self, x: float, y: float) -> bool:
        """Whether the point (x, y) lies inside the box, its edges included."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def corner(self, grab: Grab) -> tuple[float, float]:
        """The (x, y) of one of the four corners (``grab`` must be a corner grab)."""
        corners = {Grab.TOP_LEFT: (self.x1, self.y1), Grab.TOP_RIGHT: (self.x2, self.y1),
                   Grab.BOTTOM_LEFT: (self.x1, self.y2), Grab.BOTTOM_RIGHT: (self.x2, self.y2)}
        return corners[grab]


def hit_test(boxes: dict[str, PixelBox], x: float, y: float,
             corner_zone: float) -> tuple[str, Grab] | None:
    """What a press at (x, y) takes hold of, or None for empty frame.

    Corners win over insides, so a small box's corner can be grabbed even where it
    overlaps a neighbour: the corner nearest the point, among every box whose corner zone
    (a square of half-width ``corner_zone``, capped by MAX_CORNER_ZONE_FRAC) holds it.
    Failing that, the smallest box containing the point is moved: a small box drawn
    inside a larger one stays reachable."""
    best_corner: tuple[float, str, Grab] | None = None
    for key, box in boxes.items():
        zone = min(corner_zone, MAX_CORNER_ZONE_FRAC * min(box.width, box.height))
        for grab in CORNER_GRABS:
            cx, cy = box.corner(grab)
            dx, dy = abs(x - cx), abs(y - cy)
            if dx <= zone and dy <= zone:
                distance = max(dx, dy)
                if best_corner is None or distance < best_corner[0]:
                    best_corner = (distance, key, grab)
    if best_corner is not None:
        _distance, key, grab = best_corner
        return key, grab

    containing = [(box.width * box.height, key) for key, box in boxes.items()
                  if box.contains(x, y)]
    if not containing:
        return None
    _area, key = min(containing)
    return key, Grab.MOVE


def adjust(box: PixelBox, grab: Grab, dx: float, dy: float, frame_w: float,
           frame_h: float, min_size: float) -> PixelBox:
    """``box`` after dragging its ``grab`` part by (dx, dy): a move keeps the size and
    stops at the frame edges; a corner drag moves that corner only, clamped into the frame
    and kept at least ``min_size`` from the opposite corner on both axes."""
    if grab is Grab.MOVE:
        x1 = min(max(box.x1 + dx, 0.0), frame_w - box.width)
        y1 = min(max(box.y1 + dy, 0.0), frame_h - box.height)
        return PixelBox(x1, y1, x1 + box.width, y1 + box.height)

    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    if grab in (Grab.TOP_LEFT, Grab.BOTTOM_LEFT):
        x1 = min(max(x1 + dx, 0.0), x2 - min_size)
    else:
        x2 = max(min(x2 + dx, frame_w), x1 + min_size)
    if grab in (Grab.TOP_LEFT, Grab.TOP_RIGHT):
        y1 = min(max(y1 + dy, 0.0), y2 - min_size)
    else:
        y2 = max(min(y2 + dy, frame_h), y1 + min_size)
    return PixelBox(x1, y1, x2, y2)
