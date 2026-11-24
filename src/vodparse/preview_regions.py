"""The region keys, labels and colours the overlay previews share, and which regions a
read of one frame answers for.

Three places draw boxes over a frame for the overlay's regions: the layout editor's
draw pane (``gui.region_pane``), the Analyze page's layout preview
(``gui.layout_preview``) and the run's own live preview (``vodparse.sample_preview``).
This module holds what they have in common, as plain strings and RGB tuples: Qt-free, so
the pipeline side and the tests can read them without a QApplication.

* the keys drawn only on a preview, beyond a layout's own regions: the built-in
  reading's bands, the clock search's boxes, the side-bar search's sponsor rows;
* a display label and a colour per key;
* ``preview_read_keys``: the regions a read of one frame answers for under a game and
  layout, which is what the layout preview's readout lists;
* ``clock_read_key``: which box the clock is read at, so the clock's box and its
  readout swatch share a colour.

The layout preview draws a drawn layout's boxes the run would read dotted until a read
fills them, and a background find at the box the read reported; ``preview_drawn_keys`` and
``preview_background_keys`` below say which keys are which, ``preview_label`` names them,
and under Auto every box comes from a read (``preview_read_keys``).
"""

from __future__ import annotations

from vodparse.cli.clock_locate import ClockLocateResult
from vodparse.games import INGAME_BAND_CHARACTER, INGAME_BAND_NONE, INGAME_BAND_TAGS, Game
from vodparse.overlay_bands import band_ocr_plan
from vodparse.overlay_layout import (REGION_BEST_OF, REGION_INGAME_P1, REGION_INGAME_P2,
                                     REGION_KEYS, REGION_PLAYER1, REGION_PLAYER2, REGION_ROUND,
                                     REGION_TIMER, CustomOverlayLayout)
from vodparse.overlay_regions import REGION_SPONSOR1, REGION_SPONSOR2

# ── Region keys drawn only on the preview ─────────────────────────────────────────────
# Beyond overlay_layout.REGION_KEYS (the drawable layout regions): the built-in reading's
# bands and the clock search's boxes. Never stored in a layout; the preview draws them
# from the run's own rules.

PREVIEW_BAND_TOP = "band_top"
PREVIEW_BAND_BOTTOM = "band_bottom"
PREVIEW_CLOCK_CALIBRATED = "clock_calibrated"
PREVIEW_CLOCK_DETECTED = "clock_detected"
"""The clock box the detector fired on first, drawn when the search found the clock but
could not lock a steady spot (see ClockLocateResult)."""

# ── Labels and palette ────────────────────────────────────────────────────────────────
# Display labels and box colours keyed by region. The layout regions' entries are the
# editor's (the same colour marks a region's row swatch, its drag rubber band and its
# drawn box there); the preview-only keys get colours that stay apart from those and
# from typical overlay art. The bands are muted so the layout's own boxes stand out on
# top of them.

REGION_LABELS: dict[str, str] = {
    REGION_PLAYER1: "Player 1 tag",
    REGION_PLAYER2: "Player 2 tag",
    REGION_ROUND: "Round (optional)",
    REGION_BEST_OF: "Best of (optional)",
    REGION_TIMER: "Timer",
    REGION_INGAME_P1: "Ingame P1 (optional)",
    REGION_INGAME_P2: "Ingame P2 (optional)",
    REGION_SPONSOR1: "Player 1 sponsor",
    REGION_SPONSOR2: "Player 2 sponsor",
    PREVIEW_BAND_TOP: "Top overlay (auto read)",
    PREVIEW_BAND_BOTTOM: "Bottom of screen (auto read)",
    PREVIEW_CLOCK_CALIBRATED: "Clock (found by the detector)",
    PREVIEW_CLOCK_DETECTED: "Clock (seen here inconsistently)",
}

# The preview's own wording for the same keys: the editor's labels say which of its rows
# may be left undrawn ("Round (optional)"), which means nothing on a frame being read;
# the readout under the player and the captions on its boxes use these instead.
PREVIEW_LABELS: dict[str, str] = {
    **REGION_LABELS,
    REGION_ROUND: "Round",
    REGION_BEST_OF: "Best of",
    REGION_INGAME_P1: "Ingame P1",
    REGION_INGAME_P2: "Ingame P2",
}

# What the in-game band's two rows are called per band mode: the same words the layout
# editor's rows use (gui.overlay_editor.INGAME_ROW_LABELS, minus its "(optional)" and
# "(bottom not read)" notes), so a preset's "Character name P1" box and the readout's row
# for it read the same. Under mode "none" the rows are never read; the neutral wording
# is the fallback for a context the main window has not set yet.
INGAME_MODE_LABELS: dict[str, str] = {
    INGAME_BAND_TAGS: "Ingame tag",
    INGAME_BAND_CHARACTER: "Character name",
    INGAME_BAND_NONE: "Ingame",
}


def preview_label(key: str, ingame_band: str) -> str:
    """The readout's name for ``key`` under the in-game band mode ``ingame_band``
    (``overlay_layout.ingame_band_mode_for``): the mode's wording for the two in-game
    rows ("In-game tag P1", "Character name P1"), ``PREVIEW_LABELS`` for every other key."""
    if key == REGION_INGAME_P1:
        return f"{INGAME_MODE_LABELS[ingame_band]} P1"
    if key == REGION_INGAME_P2:
        return f"{INGAME_MODE_LABELS[ingame_band]} P2"
    return PREVIEW_LABELS[key]


REGION_RGB: dict[str, tuple[int, int, int]] = {
    REGION_PLAYER1: (231, 76, 60),     # red
    REGION_PLAYER2: (52, 152, 219),    # blue
    REGION_ROUND: (241, 196, 15),      # amber
    REGION_BEST_OF: (46, 204, 113),    # green
    REGION_TIMER: (155, 89, 182),      # purple
    REGION_INGAME_P1: (230, 126, 34),  # orange
    REGION_INGAME_P2: (26, 188, 156),  # teal
    REGION_SPONSOR1: (231, 76, 60),    # a locked sponsor row: its player's red
    REGION_SPONSOR2: (52, 152, 219),   # a locked sponsor row: its player's blue
    PREVIEW_BAND_TOP: (127, 140, 141),          # grey
    PREVIEW_BAND_BOTTOM: (127, 140, 141),       # grey
    PREVIEW_CLOCK_CALIBRATED: (236, 240, 241),  # near-white
    PREVIEW_CLOCK_DETECTED: (189, 195, 199),    # lighter grey
}


# ── What a read answers for ───────────────────────────────────────────────────────────

def preview_read_keys(game: Game, layout: CustomOverlayLayout | None) -> list[str]:
    """The regions a read of one frame answers for, in ``REGION_KEYS`` order: under a
    custom layout the drawn boxes a run reads (every box but the in-game ones when the
    layout's band mode is "none", which the run skips too, see
    ``overlay_regions.layout_overlay_regions``) plus the fields the run still reads in
    the background for that layout (``preview_background_keys``); under Auto the top
    band's readers (both players, the round, the best-of) and the bottom band's (the
    in-game names) for the bands the game's Auto run reads
    (``overlay_bands.band_ocr_plan``), plus the clock when the game has a detector to
    find it with. Mirrors what ``cli.overlay_read`` returns, so the preview's readout can
    lay out its rows before any read lands and keep them put while the user scrubs."""
    if layout is not None:
        wanted = preview_drawn_keys(layout) | set(preview_background_keys(game, layout))
    else:
        plan = band_ocr_plan(game, None)
        wanted = set()
        if plan.top:
            wanted |= {REGION_PLAYER1, REGION_PLAYER2, REGION_ROUND, REGION_BEST_OF}
        if plan.bottom:
            wanted |= {REGION_INGAME_P1, REGION_INGAME_P2}
        if game.has_timer_model:
            wanted.add(REGION_TIMER)
    return [key for key in REGION_KEYS if key in wanted]


def preview_drawn_keys(layout: CustomOverlayLayout) -> set[str]:
    """The drawn boxes of ``layout`` a run reads: all of them, except the in-game boxes
    when the layout's band mode is "none" (the run drops those,
    ``overlay_regions.layout_overlay_regions``; the preview reads what the run reads)."""
    drawn = set(layout.regions())
    if not layout.reads_ingame_boxes:
        drawn -= {REGION_INGAME_P1, REGION_INGAME_P2}
    return drawn


def preview_background_keys(game: Game, layout: CustomOverlayLayout | None) -> list[str]:
    """The regions a run under ``layout`` reads with the built-in readers because the
    layout draws no box for them, in ``REGION_KEYS`` order. The run decides what to OCR
    per band (``overlay_bands.band_ocr_plan``: one drawn in-game box retires the bottom
    band, only all four of the players, round and best-of retire the top one) but uses
    the result per field (``tags.attach_overlay_regions``: a drawn box overrides; the
    band's finder answers otherwise): the two tags when neither player box is drawn, the
    round and best-of when their boxes are missing, the in-game names while the bottom
    band is still read, and the clock at its calibrated box when no timer box is drawn
    and the game has a detector. Empty under Auto, where everything is read that way
    and ``preview_read_keys`` lists it plainly."""
    if layout is None:
        return []
    plan = band_ocr_plan(game, layout)
    wanted: set[str] = set()
    if plan.top and not layout.draws_player_boxes:
        wanted |= {REGION_PLAYER1, REGION_PLAYER2}
    if plan.top and layout.round_label is None:
        wanted.add(REGION_ROUND)
    if plan.top and layout.best_of is None:
        wanted.add(REGION_BEST_OF)
    if plan.bottom:
        wanted |= {REGION_INGAME_P1, REGION_INGAME_P2}
    if layout.timer is None and game.has_timer_model:
        wanted.add(REGION_TIMER)
    return [key for key in REGION_KEYS if key in wanted]


def clock_read_key(layout: CustomOverlayLayout | None, clock: ClockLocateResult | None) -> str:
    """The key of the box the clock is read at, so the readout can mark its clock row in
    that box's colour: the layout's own timer box when it draws one, else the clock
    search's calibrated box, else the box the detector fired on first. ``REGION_TIMER``
    when nothing has found the clock yet (the row shows a dash then anyway)."""
    if layout is not None and layout.timer is not None:
        return REGION_TIMER
    if clock is not None and clock.error is None:
        if clock.bbox is not None:
            return PREVIEW_CLOCK_CALIBRATED
        if clock.first_hit_bbox is not None:
            return PREVIEW_CLOCK_DETECTED
    return REGION_TIMER

