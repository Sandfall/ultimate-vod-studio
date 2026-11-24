"""Reading a custom overlay layout's boxes off a frame: the shared crop + read primitives.

Both consumers of a ``vodparse.overlay_layout.CustomOverlayLayout`` read its boxes
through these functions — the editor's readings (via ``vodparse.cli.overlay_read``)
and the analysis pipeline (``vodparse.cli.pipeline``) — so what the editor previews is,
by construction, exactly what analysis reads. Keep them in sync by keeping them here.

A box read is the engine's full text pass (detection + recognition) over the padded
crop: detection finds the tight word boxes the recognizer reads accurately (feeding the
recognizer the loose drawn box directly was tried and misread badly). The pipeline pays
that read on every sample (a per-box ROI cache was tried and dropped, see
``vodparse.ocr_roi``); ``detect_region_texts`` is the read, ``layout_read_from_texts``
folds it into a LayoutRead for its region: a player box is folded into gamertag + sponsor
(``tags.read_player_row``), every other box into its joined words. ``read_region`` does
both, merging docTR's per-word boxes into phrases in between.

Module-level imports stay light: the editor's GUI process imports the helper module
that imports this one, and must never pay for cv2/torch by doing so; the crop helper is
imported inside the function.
"""

from __future__ import annotations

from typing import Any

from vodparse.overlay_layout import REGION_PLAYER1, REGION_PLAYER2
from vodparse.overlay_regions import REGION_SPONSOR1, REGION_SPONSOR2
from vodparse.schema import BBox, LayoutRead, TextBox

# The regions whose box holds a whole overlay row rather than one field: a player's box reads
# its gamertag AND sponsor out of the row (tags.read_player_row) instead of joining the words.
PLAYER_REGIONS = frozenset({REGION_PLAYER1, REGION_PLAYER2})

# The sponsor rows the side-bar pre-pass locks above a player's tag: read as a sponsor
# (text_utils.clean_sponsor over the row's words), never as a tag.
SPONSOR_REGIONS = frozenset({REGION_SPONSOR1, REGION_SPONSOR2})

# Padding added around a drawn box before cropping it for a read: a couple of pixels of
# background helps OCR with glyphs drawn right up against the box edge, without pulling
# in neighboring overlay text.
CROP_PAD_PX = 4

# An axis-aligned (x0, y0, x1, y1) pixel box in full-frame coordinates.
Rect = tuple[int, int, int, int]


def pad_box(bbox: BBox) -> BBox:
    """``bbox`` grown by CROP_PAD_PX on every side. Not clamped: every reader downstream
    clamps to the frame itself (``bbox.crop_frame``), so a box drawn at the frame edge
    still reads its visible part."""
    return [bbox[0] - CROP_PAD_PX, bbox[1] - CROP_PAD_PX,
            bbox[2] + CROP_PAD_PX, bbox[3] + CROP_PAD_PX]


def region_rect(bbox: BBox, frame_width: int, frame_height: int) -> Rect | None:
    """The exact full-frame pixel rect a read of ``bbox`` crops: padded, then clamped to the
    frame the way ``bbox.crop_frame`` clamps. None when nothing of it is on the frame. The
    pipeline's per-box ROI cache is keyed on this rect so it caches precisely the region
    the reads cover."""
    x0, y0, x1, y1 = pad_box(bbox)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(frame_width, x1), min(frame_height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def detect_region_texts(engine: Any, frame: Any, bbox: BBox, canvas_size: int) -> list[TextBox]:
    """The full text pass (detection + recognition) over ``bbox``'s padded crop, with
    ``engine`` (an ``ocr_engine.OverlayOcr``; typed loosely so importing this function
    stays light). Every read comes back as a TextBox in FULL-FRAME coordinates (the crop
    origin added back), so the ROI cache can key it against the region it lives in. No
    confidence threshold: the caller decides what a box's reads are worth."""
    height, width = frame.shape[:2]
    rect = region_rect(bbox, width, height)
    if rect is None:
        return []
    x_off, y_off, x1, y1 = rect
    crop = frame[y_off:y1, x_off:x1]
    texts: list[TextBox] = []
    for quad, text, conf in engine.read_band(crop, beamsearch=False, canvas_size=canvas_size):
        if not text.strip():
            continue
        texts.append({"text": text.strip(), "confidence": round(float(conf), 3),
                      "bbox": [[int(p[0]) + x_off, int(p[1]) + y_off] for p in quad]})
    return texts


def layout_read_from_texts(texts: list[TextBox], region: str | None = None) -> LayoutRead:
    """One box's reads folded into its LayoutRead. `region` is the box's region key
    (``overlay_layout.REGION_KEYS``): a player box (PLAYER_REGIONS) holds a whole overlay
    row ("Rob jr  blargh257  SHE/HER/HERS  0"), so it is read through tags.read_player_row,
    which drops the score/pronouns/handle words and splits the sponsor off: `text` is then
    the gamertag alone and `sponsor` the sponsor (None without one), empty when nothing
    tag-like survived. A sponsor row (SPONSOR_REGIONS) joins the words that can be a sponsor
    (no scores, pronouns or handles) and cleans them as one (text_utils.clean_sponsor). Any
    other region (or None, the editor's timer-less default) joins its words left to right (a
    round label like 'Winners Semi-Final' comes back as several words). Confidence = the best
    single read's; empty when nothing was read."""
    if not texts:
        return LayoutRead(text="", confidence=0.0)
    confidence = max(read["confidence"] for read in texts)
    if region in PLAYER_REGIONS:
        # Imported here, not at module level: the editor's GUI process imports this module
        # and must stay light (see the module docstring); tags pulls in the whole vote stack.
        from vodparse.tags import read_player_row
        entry = read_player_row(texts)
        if entry is None:
            return LayoutRead(text="", confidence=0.0, sponsor=None)
        return LayoutRead(text=entry["text"], confidence=confidence, sponsor=entry["sponsor"])
    ordered = sorted(texts, key=lambda read: min(point[0] for point in read["bbox"]))
    if region in SPONSOR_REGIONS:
        from vodparse.text_utils import (clean_sponsor, is_ignored_tag_text,
                                         is_misread_pronoun_set, is_numeric_text)
        words = [read["text"] for read in ordered
                 if not is_numeric_text(read) and not is_ignored_tag_text(read["text"])
                 and not is_misread_pronoun_set(read["text"])]
        return LayoutRead(text=clean_sponsor(" ".join(words)) or "", confidence=confidence)
    return LayoutRead(text=" ".join(read["text"] for read in ordered).strip(),
                      confidence=confidence)


def read_region(engine: Any, frame: Any, bbox: BBox, canvas_size: int,
                region: str | None = None) -> LayoutRead:
    """Read one box with the full text pass and fold it for its `region` (see
    layout_read_from_texts): what the editor's test read shows, and what the pipeline reads
    on every sample. Both callers pass the region key, so a player box reads the same
    gamertag + sponsor in the editor as in analysis.

    docTR reads one box per WORD, while the folding (tags.read_player_row above all) was
    shaped for EasyOCR's phrase boxes: fed word boxes, a multi-word gamertag like
    'Free PlayerNine' reads as sponsor 'Free' + tag 'PlayerNine'. So a docTR engine's words
    are merged back into phrases first (word_merge.merge_adjacent_words), the same merge a
    docTR run's overlay bands get at extraction time. The merge keeps a sponsor plate apart
    from its tag by the visible gap between them (word_merge.MERGE_GAP_FRAC): a tag drawn
    with an unusually wide space between its words still splits there.

    Then, whatever the engine, a player or sponsor box's phrases lose the words that can
    never be part of a name at their edges (a pronoun set, an @handle, a [W]/[L] marker,
    a misread pill: name_filters.strip_never_name_edges with player_row), where a pill
    drawn beside the tag sits once it is read into the tag's phrase ("HE/HIM PlayerOne")."""
    # Imported here, not at module level: word_merge pulls in numpy (via vodparse.bbox) and
    # the editor's GUI process imports this module (see the module docstring).
    from vodparse.name_filters import strip_never_name_edges
    from vodparse.ocr_engine import DoctrOverlayEngine
    from vodparse.word_merge import merge_adjacent_words

    texts = detect_region_texts(engine, frame, bbox, canvas_size)
    if isinstance(engine, DoctrOverlayEngine):
        texts = merge_adjacent_words(texts)
    if region in PLAYER_REGIONS or region in SPONSOR_REGIONS:
        texts = strip_never_name_edges(texts, player_row=True)
    return layout_read_from_texts(texts, region)
