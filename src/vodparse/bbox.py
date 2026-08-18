"""Bounding-box geometry helpers used across OCR detection stages."""

from collections.abc import Sequence

import numpy as np

from vodparse.schema import TextBox


def crop_frame(frame: np.ndarray, bbox: Sequence[float] | Sequence[int]) -> np.ndarray | None:
    """Crop a frame at an [x1, y1, x2, y2] pixel box, clamped to the frame bounds; None if the
    clamped box is degenerate. The clamp matters: numpy slicing wraps negative coordinates, so an
    out-of-frame box (e.g. a detector box nudged past an edge) would otherwise silently crop the
    wrong region instead of the visible part."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def bbox_center_y(text: TextBox) -> float:
    return sum(pt[1] for pt in text["bbox"]) / len(text["bbox"])


def bbox_center_x(text: TextBox) -> float:
    return sum(pt[0] for pt in text["bbox"]) / len(text["bbox"])


def bbox_min_x(text: TextBox) -> float:
    return min(pt[0] for pt in text["bbox"])


def bbox_max_x(text: TextBox) -> float:
    return max(pt[0] for pt in text["bbox"])


def bbox_height(text: TextBox) -> float:
    ys = [pt[1] for pt in text["bbox"]]
    return max(ys) - min(ys)


def drop_sublabels(candidates: list[TextBox], height_ratio: float = 0.8,
                    x_overlap_frac: float = 0.1) -> list[TextBox]:
    """Remove sub-labels that sit directly beneath a gamertag: seed numbers, twitter
    handles, and the like (e.g. `Seed 2` / `handle1` under `Squad PlayerTen`). A box is
    dropped when another candidate is clearly taller (by more than 1/height_ratio), sits
    above it, and overlaps its horizontal extent by at least x_overlap_frac of its width.
    The comparison is relative; a tag is only ever shadowed by something taller directly
    over it; so a large sponsor logo can't sweep out real tags, and standalone short text
    (nothing taller above it) is always kept."""
    kept = []
    for t in candidates:
        ty, th = bbox_center_y(t), bbox_height(t)
        t0, t1 = bbox_min_x(t), bbox_max_x(t)
        width  = max(t1 - t0, 1)
        shadowed = any(
            bbox_height(u) > th / height_ratio
            and bbox_center_y(u) < ty
            and min(t1, bbox_max_x(u)) - max(t0, bbox_min_x(u)) > x_overlap_frac * width
            for u in candidates if u is not t
        )
        if not shadowed:
            kept.append(t)
    return kept


def longest_text(boxes: list[TextBox]) -> TextBox:
    """The likely gamertag among a row/cluster of text boxes: the longest text wins
    (sponsors are short abbreviations). Ties break toward the larger center x: nearer
    screen center for a left-side cluster, nearer the right edge for a right-side one.
    The asymmetry is kept as-is for stability; exact length ties are rare in practice."""
    return max(boxes, key=lambda t: (len(t["text"].strip()), bbox_center_x(t)))


def best_mirrored_pair(candidates: list[TextBox], frame_width: int,
                        y_tolerance: int, sym_tolerance: int) -> list[TextBox] | None:
    """From a set of text boxes, return the [left, right] pair on opposite sides of
    center that sit at similar height and best mirror each other in distance from
    their nearest edge. Pairs within sym_tolerance are treated as equally mirrored
    and the one whose shorter box is tallest wins (real labels are both rendered
    tall). Returns None if no opposite-side, similar-height pair exists."""
    mid_x  = frame_width / 2
    scored = []  # (symmetry_bucket, -height, [left, right])

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            if abs(bbox_center_y(a) - bbox_center_y(b)) > y_tolerance:
                continue
            if (bbox_center_x(a) < mid_x) == (bbox_center_x(b) < mid_x):
                continue  # need one on each side of center
            left  = a if bbox_center_x(a) < mid_x else b
            right = b if bbox_center_x(b) >= mid_x else a
            error  = abs(bbox_min_x(left) - (frame_width - bbox_max_x(right)))
            height = min(bbox_height(left), bbox_height(right))
            scored.append((round(error / sym_tolerance), -height, [left, right]))

    if not scored:
        return None
    scored.sort(key=lambda c: (c[0], c[1]))
    return scored[0][2]
