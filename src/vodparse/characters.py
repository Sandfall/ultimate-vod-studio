"""Character-portrait detection.

Runs the character YOLO model on a full frame and reads the two in-match HUD
portrait tiles. The left tile is always player 1 and the right tile player 2
(fixed by controller port), so detections are keyed by screen side at detection
time: the per-match vote over these sides can never be perturbed by the
tournament overlay swapping its tag layout. The model is passed in so callers
control loading/device, same as `timer.detect_timer`."""

import re
from typing import Any

import cv2
import numpy as np

from vodparse.schema import CharacterDetection, SideCharacters

# Default top edge of the bottom HUD strip, as a fraction of frame height. In-match damage tiles
# cluster at 0.86-0.91; 0.72 sits in the empty gap above them and below the mid-screen gameplay
# band (on-stage fighters / floating glyphs at ~0.60-0.66). Exported so the training-crop harvester
# filters at the exact line production accepts, instead of keeping its own copy.
HUD_STRIP_TOP = 0.72

# A trailing parenthesized qualifier in a class name marks a variant form of a fighter:
# the model tells variants apart (separate classes, matching the distinct HUD tiles), but
# the player's *character* is the base name.
_VARIANT_QUALIFIER = re.compile(r" \([^)]*\)$")


def _display_name(class_name: str) -> str:
    """The character name reported to the pipeline: the model's class name minus any
    trailing parenthesized variant qualifier. Collapsing variants also keeps the
    per-match vote whole when the on-screen variant form changes mid-match. Purely
    textual fixups (punctuation, short forms) are normalized into the weight metadata
    itself when a release is packaged, so no alias table lives here."""
    return _VARIANT_QUALIFIER.sub("", class_name)


def detect_characters(model, frame: np.ndarray, device: str,
                      infer_size: int = 640, conf: float = 0.5,
                      hud_strip_top: float = HUD_STRIP_TOP,
                      raw_name: bool = False) -> SideCharacters | None:
    """Detect the left/right HUD portrait tiles in a frame.

    Returns {"left": entry | None, "right": entry | None} where each entry is
    {"name", "confidence", "bbox"} (bbox in source-frame pixels), or None when
    nothing was detected. `raw_name=True` reports the model's raw class name instead
    of the pipeline display name (for the training-crop harvester, whose folders are
    keyed by raw class): the geometry is identical, so both share this one path.

    Only boxes whose center sits in the bottom HUD strip count: below
    `hud_strip_top` as a fraction of frame height (default 0.72). The in-match
    damage HUD always sits in the bottom ~10-14% of the frame (measured real tile
    centers cluster at 0.86-0.91); look-alike portraits on victory / character-
    select screens sit higher, and mid-screen distractions: an on-stage fighter,
    a floating glyph: sit around 0.60-0.66. The 0.72 cut clears
    all of those while staying ~0.14 below the lowest real tile. Without it, when a
    real tile is briefly occluded a higher-confidence spurious box elsewhere in the
    lower half hijacks that side.

    The cut is a fraction of the WHOLE frame, so a full-width bottom bar
    (webcam/sponsor) that pushes the game render up moves the HUD to *smaller*
    fractions: a tile at 0.86 survives only up to a ~16% bar (0.86*(1-0.16)=0.72),
    higher tiles to ~21%; past that a real tile silently drops to None. Fine for
    full-bleed 16:9 (the clean self-captured training data); for heavily-overlaid streams,
    raise `hud_strip_top` or anchor it to the detected game-render region.

    Each side keeps its highest-confidence in-strip box, so a weak spurious
    detection next to a tile loses to the tile itself."""
    h, w = frame.shape[:2]
    scale = infer_size / max(h, w)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    results = model(small, conf=conf, verbose=False, device=device)
    if not results[0].boxes:
        return None
    sh, sw = small.shape[:2]
    strip_y = sh * hud_strip_top                            # top edge of the bottom HUD strip
    sides: dict[str, Any] = {"left": None, "right": None}   # YOLO box per side
    for box in results[0].boxes:
        cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
        if cy < strip_y:
            continue
        side = "left" if cx < sw / 2 else "right"
        if sides[side] is None or box.conf.item() > sides[side].conf.item():
            sides[side] = box
    if sides["left"] is None and sides["right"] is None:
        return None

    def entry(box) -> CharacterDetection | None:
        if box is None:
            return None
        x1, y1, x2, y2 = (int(c / scale) for c in box.xyxy[0].tolist())
        cls_name = model.names[int(box.cls)]
        return {"name": cls_name if raw_name else _display_name(cls_name),
                "confidence": round(box.conf.item(), 3),
                "bbox": [x1, y1, x2, y2]}

    return {"left": entry(sides["left"]), "right": entry(sides["right"])}
