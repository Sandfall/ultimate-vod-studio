"""Filters that discard non-overlay text detections before tag voting.

Static-overlay filter: keeps only screen-locked text (real HUD/gamertag
elements) by clustering detections by position over all OCR frames and
dropping anything that doesn't recur at the same spot.

Banner filter: strips persistent broadcast graphics (stream watermarks,
event titles) that appear at the same position with the same text across
many different matches.
"""

from math import ceil

from vodparse.bbox import bbox_center_x, bbox_center_y
from vodparse.schema import Event, Match, TextBox
from vodparse.text_utils import is_overlay_text


# Static-overlay filter constants
STABLE_POS_TOL  = 35    # px: detections whose centers fall within this share a "slot"
STABLE_MIN_FRAC = 0.5   # a slot must recur in >= this fraction of the match's OCR frames
STABLE_MIN_OCR  = 3     # below this many OCR frames there's too little evidence to filter

# Banner filter constants
BANNER_POS_TOL     = 40   # px: same text within this position across matches is one slot
BANNER_MIN_FRAC    = 0.9  # a slot in >= this fraction of all matches is a persistent graphic
BANNER_MIN_MATCHES = 8    # ... but never flag below this many matches (small VODs can't tell)


def filter_static_overlays(frames: list[Event], pos_tol: float = STABLE_POS_TOL,
                            min_frac: float = STABLE_MIN_FRAC,
                            min_ocr: int = STABLE_MIN_OCR) -> list[Event]:
    """Drop text detections that drift across the match (in-game scenery), keeping only
    position-locked overlay text. Detections are clustered by center (x, y) over every
    OCR frame: the string itself is ignored: and a slot is trusted only if it recurs in
    at least `min_frac` of those frames. Returns new frame dicts with filtered `texts`;
    the originals are left untouched. Matches with fewer than `min_ocr` OCR frames are
    returned unchanged (too little evidence to tell a static overlay from a one-off read)."""
    ocr_frames = [f for f in frames if f.get("texts")]
    if len(ocr_frames) < min_ocr:
        return frames

    clusters: list[dict] = []   # each: {cx, cy, frames: set, members: [(frame_i, det_i)]}
    for fi, f in enumerate(ocr_frames):
        for di, t in enumerate(f["texts"]):
            cx, cy = bbox_center_x(t), bbox_center_y(t)
            for c in clusters:
                if abs(cx - c["cx"]) <= pos_tol and abs(cy - c["cy"]) <= pos_tol:
                    c["members"].append((fi, di))
                    c["frames"].add(fi)
                    n = len(c["members"])
                    c["cx"] += (cx - c["cx"]) / n   # running mean keeps the slot centered
                    c["cy"] += (cy - c["cy"]) / n
                    break
            else:
                clusters.append({"cx": cx, "cy": cy, "frames": {fi}, "members": [(fi, di)]})

    need = max(2, ceil(min_frac * len(ocr_frames)))
    keep = {member for c in clusters if len(c["frames"]) >= need for member in c["members"]}

    filtered, fi = [], 0
    for f in frames:
        if not f.get("texts"):
            filtered.append(f)
            continue
        filtered.append({**f, "texts": [t for di, t in enumerate(f["texts"])
                                        if (fi, di) in keep]})
        fi += 1
    return filtered


def find_persistent_graphics(matches: list[Match], pos_tol: float = BANNER_POS_TOL,
                              min_frac: float = BANNER_MIN_FRAC,
                              min_matches: int = BANNER_MIN_MATCHES) -> list[tuple]:
    """Find persistent broadcast graphics: the same text at the same position across many
    different matches. Returns [(text, cx, cy), ...]. Gamertag slots are excluded for free
    (their text changes per match); overlay text (is_overlay_text) is skipped outright."""
    need = max(min_matches, ceil(min_frac * len(matches)))
    clusters: list[dict] = []   # {text, cx, cy, n, matches: set}
    for m in matches:
        mn = m["match_number"]
        for f in m["frames"]:
            for t in f.get("texts", []):
                if is_overlay_text(t):
                    continue
                txt, cx, cy = t["text"], bbox_center_x(t), bbox_center_y(t)
                for c in clusters:
                    if c["text"] == txt and abs(cx - c["cx"]) <= pos_tol and abs(cy - c["cy"]) <= pos_tol:
                        c["matches"].add(mn)
                        c["n"] += 1
                        c["cx"] += (cx - c["cx"]) / c["n"]
                        c["cy"] += (cy - c["cy"]) / c["n"]
                        break
                else:
                    clusters.append({"text": txt, "cx": cx, "cy": cy, "n": 1, "matches": {mn}})
    return [(c["text"], c["cx"], c["cy"]) for c in clusters if len(c["matches"]) >= need]


def strip_banners(frames: list[Event], banners: list,
                   pos_tol: float = BANNER_POS_TOL) -> list[Event]:
    """Drop detections matching a known persistent-graphic slot (same text + position)."""
    if not banners:
        return frames

    def is_banner(t: TextBox) -> bool:
        cx, cy = bbox_center_x(t), bbox_center_y(t)
        return any(t["text"] == bt and abs(cx - bx) <= pos_tol and abs(cy - by) <= pos_tol
                   for bt, bx, by in banners)

    out = []
    for f in frames:
        if not f.get("texts"):
            out.append(f)
            continue
        out.append({**f, "texts": [t for t in f["texts"] if not is_banner(t)]})
    return out
