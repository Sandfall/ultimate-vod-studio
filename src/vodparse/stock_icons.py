"""Stock-icon character reading.

Melee shows no character names and no portrait tiles a detector could read; the one stable,
character-specific element of its HUD is the row of **stock icons** (one small head sprite per
remaining stock, drawn above each player's damage %). This module runs a YOLO detector trained
on those icons over the bottom strip of a frame and turns the boxes it finds into the same
per-side character read the portrait detector produces (`characters.detect_characters`), so the
per-match vote (`tags.attach_characters`) and everything after it stay untouched. It also keeps
the icon COUNT per slot, which is the number of stocks that player has left: recorded on every
sample as a future game-end / winner signal, with no consumer yet.

Two facts of the Melee HUD shape the geometry here (`games.StockHud` carries the numbers):

* **Slots follow controller ports.** The HUD spreads four slots across the game feed and a
  player's icons sit in the slot of their port, so a 1v1 may show its two rows on ports 1+4,
  1+2 (both in the left half of the frame) or 3+4. Streams also place the feed itself
  differently. Icons are therefore clustered into slots by the gaps between them and the
  slots ordered left to right: slot i is player i, the same "left tile = player 1" meaning
  Ultimate's portrait read has, and nothing about x is ever stored.
* **The row's height is fixed.** Every stream measured draws the feed to the bottom edge, so
  the icons' centers sit in a narrow vertical band; anything outside it (a stream's own
  character badges beside the player cams, look-alikes on results screens) is rejected on
  position, then a stray detection off the row is dropped in favour of the row that holds
  the most icons.

Exactly two slots fill the character read (player 1 left, player 2 right). One slot alone is
port-ambiguous and three or four is not a 1v1 (doubles, free-for-all), so those samples carry
their slots but no character read. Zelda and Sheik have distinct icons and stay distinct
classes: a transform mid-game splits the vote and the majority wins, as with any misread.

The model is passed in so callers control loading and device, and this module stays free of
torch and Qt (the unit tests drive it with a stub model, see the FakeModel pattern in the
portrait tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

import cv2
import numpy as np

from vodparse.games import StockHud
from vodparse.schema import BBox, CharacterDetection, SideCharacters, StockSlot
from vodparse.voting import vote_weighted

# Detector confidence floor: the same bar the portrait detector runs at.
STOCK_ICON_CONF_DEFAULT = 0.5

# Two icons whose centers are further apart vertically than this many median icon heights sit
# on different rows (see keep_majority_row). The row itself is level, so anything a whole icon
# height away is not part of it.
ROW_SPLIT_ICON_HEIGHTS = 1.0

# How many slots a 1v1 shows: the only count the character read is defined for.
SLOTS_IN_A_1V1 = 2


@dataclass(frozen=True)
class IconBox:
    """One detected stock icon, in source-frame pixels."""

    name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def bbox(self) -> BBox:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class StockRead:
    """What one frame's stock-icon read produced: every slot seen, left to right, and the
    per-side character read when exactly two slots were seen (see slots_to_side_characters)."""

    slots: list[StockSlot]
    characters: SideCharacters | None


# ── Detection: the only part that touches the model ───────────────────────────────────

def crop_strip(frame: np.ndarray, hud: StockHud) -> tuple[np.ndarray, int]:
    """The bottom strip the detector runs on and its top edge in frame pixels (the y offset
    that maps strip coordinates back to the frame)."""
    height = frame.shape[0]
    y_offset = int(height * hud.strip_top)
    return frame[y_offset:], y_offset


def strip_scale(strip_width: int, infer_width: int) -> float:
    """The factor the strip is resized by so its width becomes `infer_width`."""
    return infer_width / strip_width


def detect_icon_boxes(model: Any, frame: np.ndarray, device: str, hud: StockHud,
                      conf: float = STOCK_ICON_CONF_DEFAULT) -> list[IconBox]:
    """Run the stock-icon detector over the frame's bottom strip and return every box it
    found, mapped back to source-frame pixels. The strip is resized to `hud.infer_width`
    wide and the detector is run at that size, so the ~45 px icons of a 1080p frame keep
    about 30 px instead of the smudge the general inference size would leave them. No
    filtering happens here: see read_slots for the geometry."""
    strip, y_offset = crop_strip(frame, hud)
    strip_h, strip_w = strip.shape[:2]
    scale = strip_scale(strip_w, hud.infer_width)
    small = cv2.resize(strip, (hud.infer_width, max(1, int(strip_h * scale))))
    results = model(small, conf=conf, verbose=False, device=device, imgsz=hud.infer_width)
    icons: list[IconBox] = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = (float(c) for c in box.xyxy[0].tolist())
        icons.append(IconBox(
            name=model.names[int(box.cls)],
            confidence=round(box.conf.item(), 3),
            x1=int(x1 / scale), y1=int(y1 / scale) + y_offset,
            x2=int(x2 / scale), y2=int(y2 / scale) + y_offset))
    return icons


# ── Geometry: pure functions over the boxes ───────────────────────────────────────────

def on_stock_row(icons: list[IconBox], hud: StockHud, frame_height: int) -> list[IconBox]:
    """The icons whose centers sit in the game's declared stock-row band. Rejects a stream's
    own character badges and any look-alike drawn elsewhere on the frame, on position alone
    (the detector cannot tell an identical sprite apart by looks)."""
    return [icon for icon in icons if hud.row_contains(icon.cy / frame_height)]


def keep_majority_row(icons: list[IconBox],
                      split_icon_heights: float = ROW_SPLIT_ICON_HEIGHTS) -> list[IconBox]:
    """The icons of the one level row that holds the most of them. Icons are grouped into
    rows by their vertical centers (a jump larger than `split_icon_heights` median icon
    heights starts a new row) and the largest row is kept; a tie goes to the lowest row on
    the frame, because the game feed reaches the bottom edge on every stream measured while a
    stream's badges sit above the in-game row. A lone stray detection on a fighter or a
    stage element thereby never joins the slots."""
    if not icons:
        return []
    split = split_icon_heights * median(icon.height for icon in icons)
    by_height = sorted(icons, key=lambda icon: icon.cy)
    rows: list[list[IconBox]] = [[by_height[0]]]
    for icon in by_height[1:]:
        if icon.cy - rows[-1][-1].cy > split:
            rows.append([icon])
        else:
            rows[-1].append(icon)
    # max() keeps the first of equal-length rows, so scanning from the lowest row up makes
    # the lowest one win a tie.
    return max(reversed(rows), key=len)


def cluster_slots(icons: list[IconBox], gap_icon_widths: float) -> list[list[IconBox]]:
    """Group the row's icons into HUD slots, left to right. Adjacent icons of one player's
    row nearly touch; the next player's slot starts after a gap wider than `gap_icon_widths`
    median icon widths (see games.StockHud.slot_gap_icon_widths)."""
    if not icons:
        return []
    gap = gap_icon_widths * median(icon.width for icon in icons)
    by_x = sorted(icons, key=lambda icon: icon.cx)
    slots: list[list[IconBox]] = [[by_x[0]]]
    for icon in by_x[1:]:
        if icon.x1 - slots[-1][-1].x2 > gap:
            slots.append([icon])
        else:
            slots[-1].append(icon)
    return slots


def read_slot(group: list[IconBox], max_icons: int) -> StockSlot:
    """One slot from its icons: the character is the weighted vote over the icons' classes
    (the icons of one slot all show the same character, so a disagreeing box is a misread),
    the confidence is the best box of that character, the bbox spans the whole row, and the
    stock count is the number of icons, capped at what the game can show."""
    name = vote_weighted([(icon.name, icon.confidence) for icon in group])
    assert name is not None, "a slot always has at least one icon"
    return {
        "name": name,
        "confidence": max(icon.confidence for icon in group if icon.name == name),
        "bbox": [min(icon.x1 for icon in group), min(icon.y1 for icon in group),
                 max(icon.x2 for icon in group), max(icon.y2 for icon in group)],
        "stocks": min(len(group), max_icons),
    }


def read_slots(icons: list[IconBox], hud: StockHud, frame_height: int) -> list[StockSlot]:
    """The frame's HUD slots from its raw icon detections: row gate, majority row, slot
    clustering, then one StockSlot per cluster, left to right."""
    row = keep_majority_row(on_stock_row(icons, hud, frame_height))
    return [read_slot(group, hud.max_icons_per_slot)
            for group in cluster_slots(row, hud.slot_gap_icon_widths)]


def slots_to_side_characters(slots: list[StockSlot]) -> SideCharacters | None:
    """The per-side character read a 1v1's two slots give (left = player 1, right = player 2,
    port order), in the shape the portrait detector produces so the match vote is shared.
    Any other slot count gives None: one slot is port-ambiguous (it could be either player),
    three or four is not a 1v1. A per-run memory of where each slot sat, which would resolve
    the lone-slot case, is a possible follow-up."""
    if len(slots) != SLOTS_IN_A_1V1:
        return None

    def entry(slot: StockSlot) -> CharacterDetection:
        return {"name": slot["name"], "confidence": slot["confidence"], "bbox": slot["bbox"]}

    return {"left": entry(slots[0]), "right": entry(slots[1])}


def detect_stock_icons(model: Any, frame: np.ndarray, device: str, hud: StockHud,
                       conf: float = STOCK_ICON_CONF_DEFAULT) -> StockRead:
    """Read the frame's stock-icon slots and the character read they give (see the module
    docstring). An empty strip gives StockRead([], None)."""
    icons = detect_icon_boxes(model, frame, device, hud, conf)
    slots = read_slots(icons, hud, frame.shape[0])
    return StockRead(slots=slots, characters=slots_to_side_characters(slots))
