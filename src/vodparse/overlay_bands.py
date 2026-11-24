"""
Overlay OCR bands
-----------------
The two horizontal strips of a frame the pipeline runs OCR over: a TOP band (tournament
tags, round, best-of) and a BOTTOM band (in-game nameplates, the at-bottom tournament
overlay). Every text consumer in ``vodparse.tags`` reads a box only if its center falls in
one of these bands (see ``TAG_TOP_FRAC`` / ``OVERLAY_BOTTOM_FRAC`` in ``vodparse.tags``), so
this module is the single place that turns those fractions into pixel boxes. It exists so
``cli.pipeline.detect_text`` (the crop that produces the reads), ``OcrRoiCache.for_bands``
(the per-VOD region cache that must cache the exact same pixels), and the GUI's layout
preview (which draws the bands for the user) all draw from one definition and can never
drift apart.

This module is Qt-free and torch-free on purpose, so the GUI preview can import it without
pulling in the pipeline's heavy dependencies (torch, ultralytics, easyocr).
"""

from __future__ import annotations

from dataclasses import dataclass

from vodparse.games import INGAME_BAND_NONE, Game, ingame_band_top_for
from vodparse.overlay_layout import CustomOverlayLayout, ingame_band_mode_for
from vodparse.schema import BBox
from vodparse.tags import OVERLAY_BOTTOM_FRAC, TAG_TOP_FRAC

# ── Band names ───────────────────────────────────────────────────────────────────
# The keys used throughout the pipeline and OcrRoiCache to name the two bands. Kept as
# constants (rather than inlined string literals) so a typo can't silently create a third,
# unrecognized band name.
BAND_TOP = "top"
BAND_BOTTOM = "bottom"


def band_boxes(frame_width: int, frame_height: int, *, top: bool = True, bottom: bool = True,
               bottom_frac: float = OVERLAY_BOTTOM_FRAC) -> dict[str, BBox]:
    """The pixel boxes of the requested bands, mirroring ``cli.pipeline.detect_text``'s crop
    exactly (same ``int()`` rounding): the definition every band consumer (the pipeline's own
    crop, ``OcrRoiCache.for_bands``, the GUI's layout preview) draws from, so they can never
    read different pixels for the "same" band.

    The bands are asymmetric: the TOP band is only the top third of the frame
    (``TAG_TOP_FRAC``), tall enough to fit the round/best-of second row under the gamertag
    row; the BOTTOM band runs from `bottom_frac` (``OVERLAY_BOTTOM_FRAC`` by default, a
    shrunk capture's in-game nametag band widens it below that, see
    ``tags.INGAME_TAG_BAND_FRAC_DEFAULT``) down to the bottom edge.

    `top` / `bottom` select which bands to return (both by default); a caller that has no use
    for a band (see ``band_ocr_plan``) simply omits it from the result."""
    boxes: dict[str, BBox] = {}
    if top:
        top_h = int(frame_height * TAG_TOP_FRAC)
        boxes[BAND_TOP] = [0, 0, frame_width, top_h]
    if bottom:
        bottom_y = int(frame_height * bottom_frac)
        boxes[BAND_BOTTOM] = [0, bottom_y, frame_width, frame_height]
    return boxes


def bottom_band_frac(game: Game, ultimate_widening: float | None) -> float:
    """The top edge of the BOTTOM band, as a fraction of frame height, for `game` under the
    run's `ultimate_widening` setting (``--ingame-tag-band-frac`` / the Settings knob; see
    ``games.ingame_band_top_for``). Clamped to ``OVERLAY_BOTTOM_FRAC`` so the returned
    fraction never sits below the fixed crop the tournament reader and row learner still use
    (mirrors the ``min()`` ``cli.pipeline.detect_text`` and ``OcrRoiCache.for_bands`` both
    apply to their own `bottom_frac`/`ingame_tag_frac` argument)."""
    return min(ingame_band_top_for(game, ultimate_widening), OVERLAY_BOTTOM_FRAC)


# ── Which overlay bands the run OCRs ────────────────────────────────────────────
# detect_text crops two bands out of each sampled frame (see its comment). Each band is a
# separate OCR pass and a separate cost, and a run whose custom overlay layout already
# covers one band's readers should not pay for it: the plan below says which bands still
# have a consumer, and travels from run() into every OCR pass of the sample.


@dataclass(frozen=True)
class BandOcrPlan:
    """Which of detect_text's two bands this run still OCRs (see band_ocr_plan)."""
    top: bool
    bottom: bool

    @property
    def reads_anything(self) -> bool:
        """Whether the band pass runs at all; when False a sample reads only the clock and
        the layout's own boxes."""
        return self.top or self.bottom


BOTH_BANDS = BandOcrPlan(top=True, bottom=True)


def band_ocr_plan(game: Game, layout: CustomOverlayLayout | None) -> BandOcrPlan:
    """Which overlay bands the per-sample OCR pass still has a consumer for, given the game
    and the custom overlay layout (if any) the run reads with.

    The band reads feed four things at extract time, each gated on one band:

    * the built-in tournament-overlay heuristics (TOP band; replaced by a custom layout's
      gamertag boxes when it draws them, still the tag reading when it does not),
    * the built-in round / best-of finders (TOP band; they still run over the band texts
      when the layout has no box for that field),
    * the in-game nameplate read (BOTTOM band; off when the run reads that band as nothing,
      ``INGAME_BAND_NONE``, see ``overlay_layout.ingame_band_mode_for``): with no layout this
      is the game's own default, so a game whose nameplate band defaults to "none" (Melee)
      reads no bottom band even on the built-in "Auto" run, same as it would under a layout
      that chose "none" explicitly,
    * ...unless the layout draws a nameplate box, in which case every side is voted from
      the boxes and the band is never searched, not even for a side left undrawn (see
      ``tags.attach_layout_ingame``): one drawn box retires the BOTTOM band.

    The clock read and the layout's own box reads are unaffected either way.

    Side effect of a dropped BOTTOM band: the built-in at-bottom tournament-overlay reading
    (``tags.find_tournament_tags(at_bottom=True)``, gated on texts below ``OVERLAY_BOTTOM_FRAC``)
    and smart-tags' ``--overlay-at-bottom`` sweep variants have no band text to read, so they
    find nothing on such a game."""
    return band_ocr_plan_for(ingame_band_mode_for(game, layout), layout)


def band_ocr_plan_for(ingame_band: str, layout: CustomOverlayLayout | None) -> BandOcrPlan:
    """``band_ocr_plan`` with the band mode already resolved: for the preview helper, which
    is told the mode a layout reads the band as (``cli.overlay_read.read_auto``'s
    ``background``) rather than the layout itself."""
    band_is_read = (ingame_band != INGAME_BAND_NONE
                    and not (layout is not None and layout.draws_ingame_boxes))
    top_is_read = (layout is None or not layout.draws_player_boxes
                   or layout.round_label is None or layout.best_of is None)
    return BandOcrPlan(top=top_is_read, bottom=band_is_read)
