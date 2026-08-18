"""Player and tournament tag detection: find gamertags in OCR frames and vote
across a match's frames to produce a stable label for each player."""

import argparse
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from statistics import median
from typing import cast

from vodparse.bbox import (bbox_max_x, bbox_min_x, bbox_center_y, bbox_center_x, bbox_height,
                              best_mirrored_pair, drop_sublabels, longest_text)
from vodparse.grouping import IMPOSTOR_MIN_IDENTITIES, confirmed_identity_count, similar
from vodparse.schema import Event, Match, TagEntry, TextBox
from vodparse.overlay_filter import (filter_static_overlays, find_persistent_graphics,
                                        strip_banners)
from vodparse.text_utils import (SEPARATOR_CHARS, clean_sponsor, alnum_fold, is_best_of_label,
                                    is_ignored_tag_text, is_numeric_text, is_progress_label,
                                    is_round_text, normalize_ocr, parse_best_of, round_priority,
                                    split_tag_sponsor)
from vodparse.voting import (vote, vote_weighted, vote_optional, is_unanimous, collect_positioned,
                                collect_positioned_entries)


# All overlay text the pipeline reads sits in the outer bands of the frame; the middle is
# HUD/stage/scenery noise. The bands are asymmetric (verified by visualizing real VOD overlays):
# the TOP carries a multi-row tournament overlay: gamertags in the score bar (~0.05–0.15h) with
# the round/best-of label on a second line that can dip to ~0.26h: so it needs a third. The
# BOTTOM carries the in-game name tags and (rarely) an at-bottom tournament overlay; both its
# OCR crop and every bottom-band reader gate on the single OVERLAY_BOTTOM_FRAC below.
# pipeline.detect_text imports the crop fractions, so the crop and the gates can never drift
# apart. Candidates for a future user-tunable setting.
TAG_TOP_FRAC = 1 / 3   # tournament tags / round / best-of: center_y <= h * TAG_TOP_FRAC

# Everything the bottom band must actually read sits in the bottom 8%: the game renders its
# in-game nametags at a fixed HUD position (~0.94h at 1080p, ~0.98h on one reference layout
# measured across the reference VODs via the row learner), and the at-bottom tournament overlay's gamertag
# row (~0.94h) with its metadata line (~0.98h) sits right on top of them. The in-game HUD text
# both readers must exclude sits higher (damage-counter misreads ~0.86-0.89h, spell menus /
# resource gauges ~0.78-0.84h), so gating: and OCR-cropping (pipeline.detect_text imports
# this): at the bottom 8% drops that junk wholesale and shrinks the CRAFT area. Shared by the
# at-bottom tournament reader and the row learner (find_player_names' in-game gate historically
# sat looser at 3/4: retired once the crop guaranteed no reads could exist between the lines).
OVERLAY_BOTTOM_FRAC = 0.92

# The in-game nametag gate is the one bottom band that must sometimes reach *higher* than
# OVERLAY_BOTTOM_FRAC, so it is a separate, tunable fraction (default: the same 0.92). The
# reason is the source of the two bands: the at-bottom tournament overlay is composited over
# the capture by streaming software at full frame, so it never moves: but the in-game HUD
# nametags are rendered by the game *inside* the capture, so a recording made with the Switch's
# "Adjust Screen Size" slider turned down shrinks the whole game image toward center and lifts
# the nametags above the 0.92 line (measured ~0.919h on such a VOD: a hair above the cut),
# where the crop never OCRs them. Lowering this fraction (via --ingame-tag-band-frac) widens
# both the crop (pipeline.detect_text takes min(this, OVERLAY_BOTTOM_FRAC)) and this gate to
# recover them; the tournament reader and row learner keep the fixed 0.92 because their text
# never shrinks. Floor ~0.86: below that the band reaches the damage-counter / spell-menu rows
# and the full-frame overlay handle plates (~0.78h), which can out-vote the real nametags.
INGAME_TAG_BAND_FRAC_DEFAULT = OVERLAY_BOTTOM_FRAC

# The lowest the in-game band may be widened to. Below this the band reaches the damage-counter
# row (~0.86-0.89h) and the full-frame overlay handle plates (~0.78h): non-nametag text that can
# out-vote the real pair: so the CLI/GUI clamp here. It is above the spell-menu / resource-gauge
# rows (~0.78-0.84h) on purpose: 0.86 is the boundary between "numeric percents (still filtered)"
# and "arbitrary character HUD text (not reliably filtered)".
INGAME_TAG_BAND_FRAC_FLOOR = 0.86


def parse_ingame_tag_band_frac(value: str) -> float:
    """argparse type for --ingame-tag-band-frac (shared by the pipeline and extract CLIs):
    a fraction in [INGAME_TAG_BAND_FRAC_FLOOR, INGAME_TAG_BAND_FRAC_DEFAULT]. Rejects values
    below the floor (they reach non-nametag HUD text) with a clear message rather than a silent
    misread. The upper bound is the default 0.92, not 1.0: the flag only ever *widens* the band
    (the crop clamps to min(this, OVERLAY_BOTTOM_FRAC)), so a value above 0.92 can't widen the
    crop and would only narrow this gate below the default: never useful, so it's rejected to
    match the GUI's 8-14% spinbox range."""
    frac = float(value)
    if not INGAME_TAG_BAND_FRAC_FLOOR <= frac <= INGAME_TAG_BAND_FRAC_DEFAULT:
        raise argparse.ArgumentTypeError(
            f"must be in [{INGAME_TAG_BAND_FRAC_FLOOR}, {INGAME_TAG_BAND_FRAC_DEFAULT}]; got {frac}")
    return frac


# ── Ingame tag detection ───────────────────────────────────────────────────────

def find_player_names(texts: list[TextBox], y_tolerance: int = 15, sym_tolerance: int = 20,
                      frame_height: int = 1080, frame_width: int = 1920,
                      ingame_tag_frac: float = INGAME_TAG_BAND_FRAC_DEFAULT) -> list[TextBox] | None:
    """Find the two ingame name tags in the bottom nameplate strip (at or below
    `ingame_tag_frac` * height: the same strip the OCR crop covers): one on the left, one
    on the right, at similar height and roughly mirror-symmetric in distance from their
    nearest edge (taller pair wins ties). Returned left → right so positions stay stable for
    voting. Damage percentages and other numeric texts are excluded, as is anything inside
    the center dead-zone (the same `_CENTER_FRAC` band the tournament path uses) so a
    dead-center graphic can't be read as a left/right name pair. `ingame_tag_frac` defaults
    to the fixed 0.92 band; a shrunk capture lowers it to reach the lifted nametags (see
    INGAME_TAG_BAND_FRAC_DEFAULT)."""
    mid, dead = frame_width / 2, _CENTER_FRAC * frame_width
    lower = [t for t in texts if bbox_center_y(t) >= frame_height * ingame_tag_frac
             and abs(bbox_center_x(t) - mid) >= dead
             and not is_numeric_text(t) and not is_ignored_tag_text(t["text"])]
    if len(lower) < 2:
        return None
    return best_mirrored_pair(lower, frame_width, y_tolerance, sym_tolerance)


def attach_ingame_tags(match: Match, frame_width: int = 1920, frame_height: int = 1080,
                       ingame_tag_frac: float = INGAME_TAG_BAND_FRAC_DEFAULT) -> Match:
    """Vote on each ingame-tag position across every OCR frame in the match."""
    finder  = partial(find_player_names, frame_width=frame_width, frame_height=frame_height,
                      ingame_tag_frac=ingame_tag_frac)
    columns = collect_positioned(match["frames"], finder)

    if not columns:
        match["ingame_tags"]           = None
        match["ingame_tags_confirmed"] = False
        return match

    ingame_tags = [vote(col) for col in columns]
    confirmed   = all(is_unanimous(col, tag) for col, tag in zip(columns, ingame_tags))
    if not confirmed:
        print(f"  Note: ingame tags varied across frames for match {match['match_number']} "
              f"→ voted {ingame_tags}")

    match["ingame_tags"]           = ingame_tags
    match["ingame_tags_confirmed"] = confirmed
    return match


# ── Tournament overlay detection ──────────────────────────────────────────────

# The overlay strip proper: where tournament graphics render their gamertag row: lives in
# this top fraction of the frame. The lone-tag salvage trusts only this band: deeper into the
# TAG_TOP_FRAC third sit character/stage-select HUD labels (see _lone_tag_salvage).
STRIP_TOP_FRAC = 0.12

# Both players' gamertags are rendered at the same size; a candidate more than this many times
# taller than the other side is a one-sided graphic, not a tag (see _sided_tag_pair).
_HEIGHT_RATIO = 2.0

# A sponsor sits on the same line as its gamertag. A box whose center is more than this fraction
# of the tag's height off that line is a different row (corner seed/round/best-of label like
# "TOP 24"), not a sponsor, and is dropped (see _sided_tag_pair).
_SAME_LINE_FRAC = 0.25

# Gamertags and their sponsors occupy a band beside center. A box within this fraction of
# frame-center is a centered label/logo ("BEST OF 5", "ABC", a giant "KO"); within this fraction
# of either frame edge it is a corner promo banner ("SPONSOR SLOGAN HERE"), the far-side timer, or a
# top-corner event/pool title ("TOP 16 QUALIFIER"). None is a gamertag or a player's sponsor. The
# edge gate is wider than the center one so the corner title block: which creeps further inward
# than a promo banner: stays out of the candidate pool. Applied via _x_gated across every
# tournament reading path: the sided pair, the mirror fallback, and the lone-tag salvage.
_CENTER_FRAC = 0.04
_EDGE_FRAC   = 0.12

# OCR sometimes detects the same nameplate twice on one frame: a full read plus a partial one
# nested inside its extent ("ORG SP PLAYERONE" x[540,698] plus "SP PLAYERONE" x[584,698], sharing
# the right edge). The partial is never a genuine neighbour: real [sponsor][tag] boxes are
# horizontally disjoint ("AB" x[540,572] then "PLAYERTWO" x[584,698]): but it fools the
# gamertag pick ("a box to the right of the longest" looks exactly like a sponsor+tag layout,
# electing the partial as the tag and the full read as its own sponsor). A row box with at
# least this fraction of its width inside another row box is such a duplicate and is dropped
# (see _drop_contained_reads; the fuller read wins). Adjacent legit boxes only graze each
# other, so their mutual overlap stays near zero.
_DUP_CONTAINED_FRAC = 0.65

# A duplicate read can also sit *shifted* rather than nested: the OCR ROI cache replays a
# spurious calibration box that juts past the plate's edge ("ree64" x[1423,1529] over
# "PlayerThree64" x[1337,1483]: only 57% of the ghost's width overlaps the full read, under the
# containment bar). For those, halve the overlap requirement but demand corroboration: the
# partial's alphanumeric fold must be a substring of the fuller read's ("ree64" ⊂
# "playerthree64"). A genuine neighbour can't trip this: sponsors aren't substrings of their
# tags, and adjacent boxes only graze each other anyway.
_DUP_ECHO_FRAC = 0.50

# Stacked overlay only: the gamertag sits on the line immediately below the metadata row, so its
# row must begin within this many y-tolerances of the metadata anchor. A first-below row further
# down means the gamertag wasn't read this frame and the next thing under the metadata is lower HUD
# (a "GAME 3" bracket counter, the stage-select screen): that side yields no tag rather than
# adopting the HUD text (see _sided_tag_pair). Tuned to clear the ~2.3x line gap while excluding the
# ~4x+ gap down to the HUD rows.
_STACK_TAG_GAP = 3.0

# OCR sometimes glues a player's run of sponsor abbreviations into one text box ("ABC DEF GH1")
# that out-lengths their short real gamertag in a neighbouring box. Such a run is at least this
# many whitespace tokens, each no longer than the abbreviation cap; a box that qualifies is read
# as sponsors, not the gamertag (see _is_merged_sponsor_run / _pick_gamertag_box). Kept case-
# independent (short abbreviations, not capitalization) so it holds for all-caps overlays and
# all-caps tags alike. Two tokens is too few: it would swallow a real two-word tag ("TAG TWO").
_SPONSOR_RUN_MIN_TOKENS = 3
_SPONSOR_ABBREV_MAX_LEN = 4

# When a merged sponsor-run is peeled away, the neighbouring box is taken as the gamertag only if
# it is at least this many characters: never promote a lone OCR splinter (see _pick_gamertag_box).
_MIN_GAMERTAG_LEN = 2

# Over an at-bottom overlay (rendered on top of the in-game HUD) the gamertag is reliably the last
# box on its line, so even a single long sponsor to its left is peeled: but only when the
# right-hand box is at least this many characters, so a short OCR splinter can't steal the slot
# from a real handle ("SponsorOne PlayerSix" -> PlayerSix; "PlayerTwo two" keeps PlayerTwo). See
# _pick_gamertag_box. Larger than _MIN_GAMERTAG_LEN because this rule fires on any longer left box,
# not just a recognised sponsor-run, so it needs a stronger "is a real handle" bar.
_MIN_PEELED_TAG_LEN = 4

# An at-bottom overlay renders THREE text tiers per player over the in-game HUD: HUD text
# (damage %, a spell menu, a resource gauge) above, the GAMERTAG row in the middle, and a
# smaller metadata line (pronoun / handle / location) below. No fixed edge anchor wins both
# directions: a metadata line that survives content filtering shadows the bottommost anchor
# ("he/him" OCR-read as "HEZHIM"), surviving HUD text would shadow a topmost one ("MP",
# "LIMIT"): so the reader learns the gamertag row's screen y VOD-wide instead: it is the row
# with a box on BOTH sides of center in the same frame (mirror-symmetric) in the most frames,
# a signature no impostor tier matches (measured on the at-bottom reference VODs: real tag rows
# are mirror-symmetric in 0.48-0.77 of OCR frames, the best impostor: a metadata row: in
# 0.20). See learn_tag_row (the VOD-wide pass) and _banded_tag_pair (the per-frame pick).

# The overlay tiers render at a fixed screen y; one tier's per-frame jitter is <=5px while the
# smallest inter-tier gap measured is ~25px (gamertag row -> metadata row). 10px clusters one
# tier tightly without ever bridging two: deliberately tighter than the static-overlay
# filter's STABLE_POS_TOL (35px), which would merge the gamertag and metadata tiers.
_ROW_CLUSTER_TOL = 10.0

# A frame's boxes join the learned tag row only within this of its y. Measured slack: real tag
# boxes fall <=7px from the learned center; the nearest wrong tier is >=24px away.
_ROW_PICK_TOL = 15.0

# The winning cluster must be mirror-symmetric in at least this fraction of the OCR frames,
# else the learner abstains and the at-bottom reader keeps its bottom-edge anchor (measured:
# real tag rows score 0.48-0.77; the best impostor scores 0.20).
_ROW_MIN_MIRROR_FRAC = 0.30


def _entry_with_sponsor(box: TextBox) -> TagEntry:
    """Wrap a chosen tag box so it carries the cleaned gamertag in "text" and the sponsor
    prefix (or None) in "sponsor"."""
    tag, sponsor = split_tag_sponsor(box["text"])
    return {**box, "text": tag, "sponsor": sponsor}


def _pick_tag_rows(cand: list[TextBox], mid: float, y_tolerance: int,
                   stacked: bool, meta_top: float, at_bottom: bool = False) -> list[list[TextBox]] | None:
    """The tag text row on each side of `mid` ([left, right]), or None if a side has none.
    Normally each side's row is its topmost. With `stacked`, it's the topmost row *below* the
    metadata anchor `meta_top`: but only within `_STACK_TAG_GAP` of it (the gamertag sits one
    line down). A side with nothing in that band contributes no tag for this frame (None)
    rather than falling back to the metadata row or reaching down to lower HUD (a bracket
    counter / the stage-select screen): so frames that miss the gamertag don't let those win
    the vote.

    With `at_bottom` (and not stacked) the overlay is rendered over the in-game HUD, so the
    gamertag row hugs the bottom edge: the pronoun line beneath it is content-filtered away and
    the in-game HUD (a spell menu, a resource gauge) sits *above* it. Anchor on the bottommost row
    instead of the topmost so that HUD can't shadow the tag."""
    rows: list[list[TextBox]] = []
    for is_left in (True, False):
        side = [t for t in cand if (bbox_center_x(t) < mid) == is_left]
        if stacked:
            side = [t for t in side if bbox_center_y(t) - meta_top > y_tolerance]
            if not side:
                return None  # nothing below the metadata line on this side: no gamertag this frame
        if not side:
            return None
        if at_bottom and not stacked:
            bottom = max(bbox_center_y(t) for t in side)
            rows.append([t for t in side if bottom - bbox_center_y(t) <= y_tolerance])
            continue
        top = min(bbox_center_y(t) for t in side)
        if stacked and top - meta_top > _STACK_TAG_GAP * y_tolerance:
            return None  # first row below the metadata is too far down: the gamertag wasn't read,
                         # so this is lower HUD (bracket counter / stage-select), not a tag
        rows.append([t for t in side if bbox_center_y(t) - top <= y_tolerance])
    return rows


def _drop_contained_reads(row: list[TextBox]) -> list[TextBox]:
    """Remove duplicate detections of the same plate from a tag row (see _is_partial_reread
    for what counts as one). Without this, the gamertag pick reads the partial as the tag and
    the full read as its own sponsor. One of a duplicate pair always survives: the fuller
    read represents the plate."""
    return [t for t in row if not any(u is not t and _is_partial_reread(t, u) for u in row)]


def _is_partial_reread(t: TextBox, u: TextBox) -> bool:
    """True when row box `t` is a duplicate partial read of the plate `u` reads more fully,
    in either of two shapes: NESTED: at least _DUP_CONTAINED_FRAC of `t`'s width lies inside
    `u`'s x-extent ("DEF GHI" inside "ABC DEF GHI"); or ECHO: a shifted box that juts
    past the plate's edge, needing only _DUP_ECHO_FRAC overlap but with `t`'s text folding to
    a proper substring of `u`'s ("ree64" over "PlayerThree64", replayed by the OCR ROI cache; see
    the constants' comments). Re-reads of one plate render at the same size, so `u` must be
    height-comparable (within _HEIGHT_RATIO): a giant one-sided graphic overlapping the row
    ("EVENT BANNER") is _repick_oversized's case, not a duplicate, and must not swallow the
    real tag here. The box with the longer text wins; equal-length twins keep the leftmost,
    so a duplicate pair can never eliminate both of its members."""
    if bbox_height(u) > _HEIGHT_RATIO * bbox_height(t):
        return False
    t_len, u_len = len(t["text"].strip()), len(u["text"].strip())
    t0, t1 = bbox_min_x(t), bbox_max_x(t)
    if u_len < t_len or (u_len == t_len and bbox_min_x(u) >= t0):
        return False  # `u` is no fuller a read than `t`: never drop the better box
    overlap = min(t1, bbox_max_x(u)) - max(t0, bbox_min_x(u))
    width = max(t1 - t0, 1)
    if overlap >= _DUP_CONTAINED_FRAC * width:
        return True   # nested
    t_fold, u_fold = alnum_fold(t["text"]), alnum_fold(u["text"])
    return (overlap >= _DUP_ECHO_FRAC * width
            and 0 < len(t_fold) < len(u_fold) and t_fold in u_fold)  # echo


def _repick_oversized(rows: list[list[TextBox]], tagboxes: list[TextBox]) -> list[TextBox]:
    """Fix a one-sided giant pick. The two gamertags are rendered the same size, so a pick far
    taller than the other side is a one-sided graphic (e.g. a giant "KO" elimination banner
    that out-lengths a missed tag). Re-pick that side as the longest box within _HEIGHT_RATIO
    of the shorter (tag-like) side. Returns the (possibly re-picked) tagboxes."""
    ref = min(bbox_height(t) for t in tagboxes)
    for i, (row, tagbox) in enumerate(zip(rows, tagboxes)):
        if bbox_height(tagbox) > ref * _HEIGHT_RATIO:
            fits = [t for t in row if bbox_height(t) <= ref * _HEIGHT_RATIO]
            if fits:
                tagboxes[i] = longest_text(fits)
    return tagboxes


def _split_sponsor_from_row(row: list[TextBox], tagbox: TextBox) -> TagEntry:
    """Build one side's TagEntry: the tag box's cleaned gamertag plus its sponsor. The sponsor
    combines the row's other qualifying boxes with any prefix inside the tag box: a sponsor
    precedes its gamertag ([sponsor][tag], so it's left of it), shares its line, and is no
    taller. (Dead-center logos and edge banners were already band-filtered away.)"""
    tag, prefix = split_tag_sponsor(tagbox["text"])
    tx   = bbox_center_x(tagbox)
    ty   = bbox_center_y(tagbox)
    line = _SAME_LINE_FRAC * bbox_height(tagbox)
    cap  = _HEIGHT_RATIO * bbox_height(tagbox)
    extra = [t["text"].strip() for t in row
             if t is not tagbox
             and bbox_center_x(t) < tx
             and abs(bbox_center_y(t) - ty) <= line
             and bbox_height(t) <= cap]
    sponsor = clean_sponsor(" | ".join([s for s in extra + ([prefix] if prefix else []) if s]))
    return {**tagbox, "text": tag, "sponsor": sponsor}


def _is_merged_sponsor_run(text: str) -> bool:
    """True when `text` is a run of sponsor abbreviations OCR glued into one box ("ABC DEF GH1"),
    not a gamertag: at least _SPONSOR_RUN_MIN_TOKENS whitespace tokens, each no longer than
    _SPONSOR_ABBREV_MAX_LEN, and carrying no in-box separator. Case-independent, so it holds for
    all-caps overlays and all-caps tags alike. A two-word tag ("TAG TWO") has too few tokens; a real
    multi-word handle ("Free PlayerNine") has a long token; a "SPONSOR | Tag" box carries a separator
    (its tag is split off by split_tag_sponsor instead): none qualify."""
    if any(char in SEPARATOR_CHARS for char in text):
        return False
    tokens = text.split()
    return (len(tokens) >= _SPONSOR_RUN_MIN_TOKENS
            and all(len(token) <= _SPONSOR_ABBREV_MAX_LEN for token in tokens))


def _pick_gamertag_box(row: list[TextBox], bottom_overlay: bool = False) -> TextBox:
    """The gamertag box on one side of the overlay. Normally the longest text wins (sponsors are
    short abbreviations). Two cases override that toward the rightmost box, because the overlay
    places the gamertag last:

    * a merged sponsor-run (see _is_merged_sponsor_run) glued into the longest box: on any overlay
      - recovers a short tag it out-lengths ("ABC DEF GH1 PlayerTen" -> PlayerTen);
    * over a `bottom_overlay` (at-bottom, non-stacked), even a *single* long sponsor to the left is
      peeled ("SponsorOne PlayerSix" -> PlayerSix), since the gamertag is reliably last there. This is
      gated to the at-bottom reader (and off the recall-sensitive top overlays) and guarded so the
      right-hand box must carry no less than _MIN_PEELED_TAG_LEN characters and the longest box no
      in-box separator: a short splinter or an already-delimited "SPONSOR | Tag" box won't trip it.

    Single-box sides are unaffected (the longest box is the rightmost)."""
    longest   = longest_text(row)
    rightmost = max(row, key=bbox_center_x)
    if longest is rightmost:
        return longest
    right_len = len(rightmost["text"].strip())
    if _is_merged_sponsor_run(longest["text"].strip()) and right_len >= _MIN_GAMERTAG_LEN:
        return rightmost
    if (bottom_overlay and right_len >= _MIN_PEELED_TAG_LEN
            and not any(char in SEPARATOR_CHARS for char in longest["text"])):
        return rightmost
    return longest


def _banded_tag_pair(cand: list[TextBox], mid: float,
                     tag_row_y: float) -> list[TagEntry | None] | None:
    """The at-bottom sided pick against a learned tag row: each side's row is its boxes within
    _ROW_PICK_TOL of `tag_row_y`, instead of an edge-anchored row (see the learned-row constants
    block). Returns a fixed-position ``[left, right]`` pair. A side with no row this frame is
    ``None``: the frame's other side keeps voting while the missing side's columns see nothing,
    where the edge anchor would instead vote whatever junk sits nearest the edge, and dropping
    the whole frame would starve the good side's sponsor of its vote_optional majority. The
    frame abstains outright (returns None) only when both sides are empty."""
    sides: list[list[TextBox]] = [
        _drop_contained_reads([t for t in cand if (bbox_center_x(t) < mid) == is_left
                               and abs(bbox_center_y(t) - tag_row_y) <= _ROW_PICK_TOL])
        for is_left in (True, False)]
    rows = [row for row in sides if row]
    if not rows:
        return None
    tagboxes = [_pick_gamertag_box(row, bottom_overlay=True) for row in rows]
    if len(rows) == 2:   # the oversized re-pick compares the two sides' heights: needs both
        tagboxes = _repick_oversized(rows, tagboxes)
    picked = [_split_sponsor_from_row(row, tagbox) for row, tagbox in zip(rows, tagboxes)]
    return [picked.pop(0) if row else None for row in sides]


def _x_gated(texts: list[TextBox], frame_width: int) -> list[TextBox]:
    """Boxes whose center-x sits in the gamertag band, dropping overlay furniture by position
    (see the _CENTER_FRAC / _EDGE_FRAC constants block): the dead-center zone: a box within
    _CENTER_FRAC of frame-center is a centered label/logo ("BEST OF 5", "ABC", a giant "KO"),
    never a tag: and everything outside the interior _EDGE_FRAC..(1 - _EDGE_FRAC) span
    corner promo banners, the far-side timer, a top-corner event title. Shared by every
    tournament reading path so the gates stay in lockstep."""
    mid = frame_width / 2
    dead = _CENTER_FRAC * frame_width
    lo, hi = _EDGE_FRAC * frame_width, (1 - _EDGE_FRAC) * frame_width
    return [t for t in texts
            if lo <= bbox_center_x(t) <= hi and abs(bbox_center_x(t) - mid) >= dead]


def _sided_tag_pair(upper: list[TextBox], frame_width: int, y_tolerance: int,
                    stacked: bool = False, at_bottom: bool = False,
                    tag_row_y: float | None = None) -> list[TagEntry | None] | None:
    """Symmetric overlay: a `[sponsor?][gamertag]` cluster on each side of center. On each side
    take a text row and read the gamertag from it (see _pick_gamertag_box: usually the longest
    box, since sponsors are short abbreviations); the remaining boxes and any sponsor prefix inside
    the tag box become the sponsor. Returns [left, right] entries each carrying the gamertag in
    "text" and the sponsor (or None) in "sponsor", or None if a side has no candidate (not a
    two-sided layout: the caller falls back).

    Which row is the tag row depends on the layout (see _pick_tag_rows): the topmost by default,
    the row *below* a metadata line with `stacked`, and the *bottommost* with `at_bottom` (the
    overlay sits over the in-game HUD, so the tag hugs the bottom edge and HUD sits above it)
    unless `tag_row_y` carries a VOD-wide learned row, in which case the at-bottom pick anchors
    on that row instead and may return per-side ``None`` slots (see _banded_tag_pair)."""
    mid = frame_width / 2
    bottom_overlay = at_bottom and not stacked
    # Tags and sponsors sit in a band beside center: drop dead-center labels/logos ("BEST OF 5",
    # "ABC", "KO") and edge-jammed corner banners ("SPONSOR SLOGAN HERE") / the far-side timer, so
    # neither can out-length the real tag.
    cand = _x_gated(upper, frame_width)
    if bottom_overlay and tag_row_y is not None:
        return _banded_tag_pair(cand, mid, tag_row_y)
    # The metadata line is the topmost row of the overlay block; anchor on it *globally* (across
    # both sides), not per side. A side whose metadata is entirely content-filtered (a pronoun or
    # seed line, e.g. "he/him" + "Seed 64") would otherwise have its gamertag row as its own top
    # row and skip it: the other side's surviving metadata keeps the anchor on the real top line.
    meta_top = min((bbox_center_y(t) for t in cand), default=0.0) if stacked else 0.0
    rows = _pick_tag_rows(cand, mid, y_tolerance, stacked, meta_top, at_bottom=at_bottom)
    if rows is None:
        return None
    rows = [_drop_contained_reads(row) for row in rows]   # duplicate re-reads of one plate
    tagboxes = _repick_oversized(rows, [_pick_gamertag_box(row, bottom_overlay) for row in rows])
    out: list[TagEntry | None] = [_split_sponsor_from_row(row, tagbox)
                                  for row, tagbox in zip(rows, tagboxes)]
    out.sort(key=lambda e: bbox_center_x(cast("TagEntry", e)))   # no Nones on this path
    return out


# A salvaged lone tag must be at least this many characters: a 1-char lone box is
# indistinguishable from OCR debris (a stray "り" or bracket fragment), and real gamertags
# this short are vanishingly rare: with no mirrored partner to corroborate, abstain instead.
_MIN_LONE_TAG_CHARS = 2


def _lone_tag_salvage(band: list[TextBox], frame_width: int,
                      frame_height: int) -> list[TagEntry | None] | None:
    """Last-resort read of a single gamertag from a top-band frame where every pairing path
    (sided, mirror) failed because only ONE player's plate survived OCR. Returns a
    fixed-slot pair: ``[entry, None]`` when the tag is left of center, ``[None, entry]`` when
    it is right of it: so the cross-frame column vote (positional_columns → per-column vote,
    Nones dropped by attach_tournament_overlay) elects the lone tag in its true position while
    the unread side abstains. Two-sided frames already emit ``[left, right]`` sorted by x, so
    these fixed slots line up with them positionally.

    `band` is the caller's already content-filtered, drop_sublabels'd top band (the salvage
    reuses it verbatim: no filters are re-run). The gates here are the strictest in the module,
    because a lone read has no mirrored partner to corroborate it:

      * only the overlay *strip* (the top STRIP_TOP_FRAC of the frame) is trusted: tournament
        overlays render their tag strip at the very top, while the deeper TAG_TOP_FRAC third
        also contains character/stage-select HUD labels ("Normal", stage names) that only the
        paired paths' mirror symmetry can safely see past;
      * the full `_x_gated` gates (center dead-zone + corner-banner exclusion);
      * all surviving candidates must sit strictly on one side of center: mixed sides are a
        two-sided layout the pair paths already had their chance at, so they abstain (None);
      * the picked tag must be at least _MIN_LONE_TAG_CHARS characters.

    The gamertag is the longest surviving box (sponsors are short abbreviations); the boxes on
    its rendered line become its sponsor via _split_sponsor_from_row, so an adjacent sponsor
    plate still attaches and an in-box "SPONSOR | Tag" still splits. The caller gates this on
    top-band overlay evidence (a round / best-of label on the frame), so it never fabricates a
    tag from stray in-game text on a frame without a tournament overlay."""
    strip = [t for t in band if bbox_center_y(t) <= frame_height * STRIP_TOP_FRAC]
    cand = _x_gated(strip, frame_width)
    if not cand:
        return None
    mid = frame_width / 2
    all_left  = all(bbox_center_x(t) < mid for t in cand)
    all_right = all(bbox_center_x(t) >= mid for t in cand)
    if not (all_left or all_right):
        return None   # both sides populated: a two-sided layout the pair paths already declined
    tagbox = longest_text(cand)
    # _split_sponsor_from_row applies its own same-line/height/left-of gates, so the whole
    # candidate set is passed as the row. The length check runs on the *split* tag: a lone
    # "ORG | K" box is long enough as a box but yields a 1-char tag, which is still debris.
    entry = _split_sponsor_from_row(cand, tagbox)
    if len(entry["text"].strip()) < _MIN_LONE_TAG_CHARS:
        return None
    return [entry, None] if bbox_center_x(tagbox) < mid else [None, entry]


def find_tournament_tags(texts: list[TextBox], y_tolerance: int = 20, sym_tolerance: int = 20,
                         frame_height: int = 1080, frame_width: int = 1920,
                         at_bottom: bool = False, stacked: bool = False,
                         tag_row_y: float | None = None) -> list[TagEntry | None] | None:
    """Find the two gamertags in the tournament-overlay band. Each returned entry carries the
    gamertag in "text" and the sponsor (or None) in "sponsor". The symmetric overlay layout
    a `[sponsor?][gamertag]` cluster on each side of center: is read per-side so the longest
    box on each side is the gamertag and the short sponsor boxes/prefixes are split off, with a
    mirror-pair fallback for remnants the per-side reader misses. The round label, the best-of
    label (including OCR's "BEST 0F"), score counts, and the timer are excluded.

    The overlay normally lives in the upper third; `at_bottom` switches the band gate to the
    bottom OVERLAY_BOTTOM_FRAC strip for the rare tournament that renders its overlay over the
    in-game HUD (only the y-gate moves: the layout/sponsor logic is band-agnostic).

    When both pairing paths fail on a non-stacked top-band frame because only ONE player's
    plate survived OCR, a lone-tag salvage fires as a last resort (see _lone_tag_salvage). It is
    gated on top-band overlay evidence: a round or best-of label the band filters out: so it
    can't fabricate a tag from stray in-game text on a frame with no tournament overlay, and it
    returns a fixed-slot pair (``[entry, None]`` or ``[None, entry]`` by the tag's side of
    center) whose unread side is ``None`` for the column vote to drop. It is deliberately limited
    to the non-stacked top band: stacked overlays read below the metadata via the sided path
    alone, and the at-bottom band carries no round/best-of evidence to gate on.

    `stacked` handles the overlay that prints a metadata line (location / Twitter handle / seed)
    *above* each gamertag: the symmetric path reads the row below the metadata, the inverted
    `drop_sublabels` pass is skipped (it would shadow-drop the gamertag under a tall metadata
    line), and game/pool HUD counters are filtered (they sit below the tag and become reachable
    once we look past the top row).

    `at_bottom` *inverts* stacking: an overlay rendered over the in-game HUD never prints metadata
    above the gamertag: any extra line (pronoun / handle / seed) sits *below* it. So `stacked` is
    forced off when `at_bottom` is set, and the at-bottom reader anchors on the bottommost row (see
    _sided_tag_pair / _pick_tag_rows); passing both flags reads the same as `at_bottom` alone.

    `tag_row_y` (at-bottom only) is the VOD-wide learned gamertag row (see learn_tag_row). When
    given, the at-bottom pick anchors on that row instead of the bottom edge and returns a
    fixed-position ``[left, right]`` pair whose unread side is ``None`` (callers vote per column
    after dropping the Nones). The mirrored-pair fallback is skipped on this path: the metadata
    line is itself mirror-symmetric per frame, so the fallback would re-admit exactly the
    impostor the learned row exists to exclude."""
    stacked = stacked and not at_bottom   # at-bottom overlays are inverted: metadata sits below the tag
    if at_bottom:
        in_band = lambda t: bbox_center_y(t) >= frame_height * OVERLAY_BOTTOM_FRAC
    else:
        in_band = lambda t: bbox_center_y(t) <= frame_height * TAG_TOP_FRAC
    # One pass sorts each in-band box into round/best-of labels (filtered out, but their
    # presence is the overlay evidence the lone-tag salvage gates on: a tournament overlay is
    # provably rendered this frame, so a lone surviving gamertag is a real read rather than
    # stray in-game text) and the tag-candidate band. Round/best-of always live in the top
    # band, so evidence is meaningful only for the (not at_bottom) case; the at-bottom
    # learned-row path has its own None-slot mechanism.
    band: list[TextBox] = []
    saw_overlay_label = False
    for t in texts:
        if not in_band(t):
            continue
        if is_round_text(t) or parse_best_of(t["text"]) or is_best_of_label(t["text"]):
            saw_overlay_label = True
            continue
        if (is_numeric_text(t) or is_ignored_tag_text(t["text"])
                or (stacked and is_progress_label(t["text"]))):
            continue
        band.append(t)
    overlay_evidence = saw_overlay_label and not at_bottom
    if not stacked:
        # Drop seed numbers / twitter handles stacked under a gamertag before pairing, so a
        # mirrored sub-label row can't beat the real tags on symmetry. Skipped when `stacked`:
        # there the metadata sits *above* the tag, so this pass would shadow-drop the gamertag.
        band = drop_sublabels(band)
    # The lone-tag salvage reads a single-sided top-band frame with overlay evidence (see
    # _lone_tag_salvage). Only the non-stacked top band qualifies: stacked overlays read the row
    # below the metadata via the sided path alone, and the at-bottom band never carries the
    # round/best-of evidence this gate needs.
    salvage_eligible = overlay_evidence and not stacked
    if len(band) < 2 and not (at_bottom and tag_row_y is not None) and not salvage_eligible:
        # The learned-row path is exempt: under it a lone surviving box can still be a valid
        # one-sided read (the other side becomes a None slot rather than failing the frame).
        # Salvage is likewise exempt: a 1-box band is exactly the lone-tag case it recovers.
        return None
    sided = _sided_tag_pair(band, frame_width, y_tolerance, stacked=stacked, at_bottom=at_bottom,
                            tag_row_y=tag_row_y)
    if sided is not None or stacked or (at_bottom and tag_row_y is not None):
        return sided
    # Remnants the per-side reader missed: fall back to mirror pairing. Same gates as the sided
    # path (both edge and center), so a dead-center label pair ("BEST"/"OF.") or an edge banner
    # can't be elected as a mirror pair. (Skipped when `stacked`: the fallback reads whichever
    # row mirrors best, so it would pick the metadata line above the gamertags.)
    pair = best_mirrored_pair(_x_gated(band, frame_width),
                              frame_width, y_tolerance, sym_tolerance)
    if pair:
        return [_entry_with_sponsor(t) for t in pair]
    # Every pairing path failed. If the top band still shows a tournament overlay (round/best-of
    # evidence) and one gamertag survived OCR on a single side, salvage it into its fixed slot so
    # the column vote can still elect it (see _lone_tag_salvage).
    if salvage_eligible:
        return _lone_tag_salvage(band, frame_width, frame_height)
    return None


def find_round(texts: list[TextBox], frame_height: int = 1080) -> TextBox | None:
    """Find the round label anywhere in the upper third, by keyword. Its position
    varies (centered, or under/above either gamertag), so we match on content rather
    than position. When several texts match, the one whose keyword ranks earliest in
    ROUND_KEYWORDS wins, breaking ties by confidence."""
    matches = [t for t in texts if bbox_center_y(t) <= frame_height * TAG_TOP_FRAC and is_round_text(t)]
    if not matches:
        return None
    best = min(matches, key=lambda t: (round_priority(t["text"]), -t.get("confidence", 0)))
    return {**best, "text": normalize_ocr(best["text"])}


def _vote_round(labels: list[str]) -> str | None:
    """Resolve a match's round from its per-frame round readings. A persistent bracket
    graphic ("BRACKET", "SIDE BRACKET") reads on far more frames than the brief on-screen
    round label, so plain frequency voting elects it; instead keep only the readings whose
    keyword ranks highest in ROUND_KEYWORDS (lowest index), then vote within that top tier.

    "bracket" is the weakest keyword: it only marks the bracket graphic, never an actual
    round (it stays in ROUND_KEYWORDS so that graphic is still recognized as overlay text).
    A reading whose only round keyword is "bracket" is dropped, so a match with no real
    round label resolves to None rather than "BRACKET"."""
    bracket = round_priority("bracket")
    ranked = [label for label in labels if round_priority(label) < bracket]
    if not ranked:
        return None
    best = min(round_priority(label) for label in ranked)
    return vote([label for label in ranked if round_priority(label) == best])


def find_best_of(texts: list[TextBox], frame_height: int = 1080) -> TextBox | None:
    """Find a 'best of N' / 'BoN' label in the upper third (usually under the right
    gamertag, but position varies). Returns a normalized 'BoN' entry or None."""
    candidates: list[tuple[TextBox, str]] = []
    for t in texts:
        if bbox_center_y(t) <= frame_height * TAG_TOP_FRAC and (bo := parse_best_of(t["text"])):
            candidates.append((t, bo))
    if not candidates:
        return None
    t, bo = max(candidates, key=lambda c: c[0].get("confidence", 0))
    return {**t, "text": bo}


def _sponsor_tokens(sponsor: str | None) -> set[str]:
    """The case-folded component abbreviations of a voted sponsor ("SP | TM" → {"sp", "tm"})."""
    return {part.strip().casefold() for part in sponsor.split("|")} if sponsor else set()


def _tag_candidates(entries: list[TagEntry], sponsor: str | None) -> list[str]:
    """Gamertag readings for one player's column, minus any token already voted as that player's
    sponsor. A confidently-voted sponsor abbreviation (e.g. "SP" in "SP | TM") is never a
    legitimate gamertag, so dropping it stops a stable sponsor from winning the tag slot on the
    frames where the real tag drops out of OCR (the reading then abstains rather than voting the
    sponsor). Comparison is case-folded; sponsor components are the "|"-separated parts of the
    voted sponsor string."""
    blocked = _sponsor_tokens(sponsor)
    return [e["text"] for e in entries
            if e["text"] and e["text"].strip().casefold() not in blocked]


def _column_center_x(entries: list[TagEntry]) -> float:
    """The median center-x of a column's boxes: its horizontal position on screen, used to line
    a filtered column up with its unfiltered counterpart."""
    xs = [bbox_center_x(e) for e in entries if e.get("bbox")]
    return median(xs) if xs else 0.0


def _rescue_filtered_sponsor(tag: str | None, sponsor: str | None, column: list[TagEntry],
                             raw_columns: list[list[TagEntry]]) -> tuple[str | None, str | None]:
    """Recover a gamertag the static-overlay filter pruned. In a sparse match that filter can keep
    a player's position-locked sponsor while dropping their flickering gamertag, so the filtered
    vote elects the sponsor as the tag. When the elected tag also reads as a sponsor in the
    *unfiltered* reads at this position, re-derive the tag from those reads (minus the sponsor),
    restoring the real gamertag (reference VOD A m37: "ORG" → "PlayerNine"). Keying on sponsor structure
    keeps it from reviving in-game scenery, which carries no sponsor. No-op when `raw_columns` is
    empty (the caller didn't run the static filter, so there is nothing to recover)."""
    if not (tag and raw_columns):
        return tag, sponsor
    raw = min(raw_columns, key=lambda c: abs(_column_center_x(c) - _column_center_x(column)))
    raw_sponsor = vote_optional([e.get("sponsor") for e in raw])
    if tag.strip().casefold() not in _sponsor_tokens(raw_sponsor):
        return tag, sponsor
    rescued = vote(_tag_candidates(raw, raw_sponsor))
    return (rescued, raw_sponsor) if rescued else (tag, sponsor)


def _reunite_split_tag(tag: str | None, sponsor: str | None,
                       entries: list[TagEntry]) -> tuple[str | None, str | None]:
    """Undo a multi-word gamertag that OCR split into two boxes. When a nameplate like
    "The Player" is read as separate "The" and "Player" boxes on some frames, the geometric
    splitter reads the wider box as the tag and demotes the first word to a sponsor. If the
    merged reading "<sponsor> <tag>" actually appears as a tag reading elsewhere in this column,
    that first word was really part of the tag: return the merged tag and clear the sponsor.
    Only fires on evidence OCR itself produced (it never fabricates a merge), so it is naturally
    limited to single-word mis-splits."""
    if not (tag and sponsor):
        return tag, sponsor
    merged = f"{sponsor} {tag}"
    if any(e["text"] and e["text"].strip().casefold() == merged.casefold() for e in entries):
        return merged, None
    return tag, sponsor


def attach_tournament_overlay(match: Match, frame_width: int = 1920, frame_height: int = 1080,
                              at_bottom: bool = False, stacked: bool = False,
                              raw_frames: list[Event] | None = None,
                              tag_row_y: float | None = None) -> Match:
    """Vote on each gamertag position, its sponsor, the round, and the best-of across every
    OCR frame. `at_bottom` reads the overlay from the bottom strip instead of the upper third
    (round/best-of are unaffected: they stay top-band). `stacked` reads each gamertag from the
    row below its location/handle metadata line. `raw_frames` are the pre-static-filter frames:
    when given, the sponsor rescue (see _rescue_filtered_sponsor) uses them to recover a gamertag
    the filter pruned in favour of a position-locked sponsor. `tag_row_y` is the VOD-wide learned
    at-bottom gamertag row (see learn_tag_row); on that path the finder emits per-side ``None``
    slots for unread sides, which are dropped per column so they don't vote."""
    finder  = partial(find_tournament_tags, frame_width=frame_width, frame_height=frame_height,
                      at_bottom=at_bottom, stacked=stacked, tag_row_y=tag_row_y)
    # The overlay finder yields TagEntry dicts (tag + sponsor); collect returns them generically.
    # The learned-row path also yields None for a side unread this frame: filter those out so
    # a missing side abstains from its own column instead of voting an absence.
    columns = [[e for e in col if e is not None]
               for col in cast("list[list[TagEntry | None]]",
                               collect_positioned_entries(match["frames"], finder))]
    raw_columns = ([[e for e in col if e is not None]
                    for col in cast("list[list[TagEntry | None]]",
                                    collect_positioned_entries(raw_frames, finder))]
                   if raw_frames is not None else [])
    if columns:
        tags: list[str | None] = []
        sponsors: list[str | None] = []
        for col in columns:
            # Vote the sponsor first so the tag vote can exclude this player's own sponsor tokens.
            sponsor = vote_optional([e.get("sponsor") for e in col])
            tag     = vote(_tag_candidates(col, sponsor))     # a sponsor token can't be the tag
            tag, sponsor = _reunite_split_tag(tag, sponsor, col)  # re-merge a split multi-word tag
            tag, sponsor = _rescue_filtered_sponsor(tag, sponsor, col, raw_columns)  # unprune tag
            tags.append(tag)
            sponsors.append(sponsor)
        match["tournament_tags"]     = tags
        match["tournament_sponsors"] = sponsors
    else:
        match["tournament_tags"]     = None
        match["tournament_sponsors"] = None

    round_texts: list[str] = []
    best_of_texts: list[str] = []
    for frame in match["frames"]:
        texts = frame.get("texts", [])
        if (rnd := find_round(texts, frame_height)) is not None:
            round_texts.append(rnd["text"])
        if (best_of := find_best_of(texts, frame_height)) is not None:
            best_of_texts.append(best_of["text"])
    match["round"] = _vote_round(round_texts)
    match["best_of"] = vote(best_of_texts)

    return match


# ── Character voting ───────────────────────────────────────────────────────────

def attach_characters(match: Match) -> Match:
    """Vote each HUD side's character across every frame in the match. The sides are
    fixed by detection geometry (left tile = player 1, right tile = player 2), so the
    columns stay consistent even when the tournament overlay swaps its tag layout.
    Frames without a character detection (go/game events, pre-character pipeline JSONs)
    contribute nothing."""
    columns: dict[str, list[tuple[str, float]]] = {"left": [], "right": []}
    for frame in match["frames"]:
        characters = frame.get("characters")
        if not characters:
            continue
        left, right = characters.get("left"), characters.get("right")
        if left:
            columns["left"].append((left["name"], left.get("confidence", 0.0)))
        if right:
            columns["right"].append((right["name"], right.get("confidence", 0.0)))
    if columns["left"] or columns["right"]:
        match["characters"] = [vote_weighted(columns["left"]), vote_weighted(columns["right"])]
    else:
        match["characters"] = None
    return match


# ── At-bottom tag-row learning ─────────────────────────────────────────────────

@dataclass
class RowCluster:
    """One horizontal text row of the lower band, accumulated VOD-wide: a running-mean
    y-cluster, the 1-D analogue of overlay_filter.filter_static_overlays's slot loop."""
    cy: float
    boxes: int = 0
    heights: list[float] = field(default_factory=list)
    texts: Counter[str] = field(default_factory=Counter)
    left_frames: set[int] = field(default_factory=set)
    right_frames: set[int] = field(default_factory=set)

    def add(self, box: TextBox, frame_index: int, is_left: bool) -> None:
        self.boxes += 1
        self.cy += (bbox_center_y(box) - self.cy) / self.boxes   # running mean stays centered
        self.heights.append(bbox_height(box))
        self.texts[box["text"]] += 1
        (self.left_frames if is_left else self.right_frames).add(frame_index)

    @property
    def frames(self) -> set[int]:
        return self.left_frames | self.right_frames

    @property
    def mirror_frames(self) -> set[int]:
        """Frames where this row had a box on BOTH sides of center: the gamertag signature."""
        return self.left_frames & self.right_frames

    def median_height(self) -> float:
        return median(self.heights)


@dataclass(frozen=True)
class LearnedTagRow:
    """The learned at-bottom gamertag row: its y plus the evidence that elected it."""
    y: float
    support: float          # fraction of OCR frames with any box in the row
    mirror_frac: float      # fraction of OCR frames with boxes on both sides of center
    median_height: float
    n_ocr_frames: int


def _row_candidates(texts: list[TextBox], frame_width: int, frame_height: int) -> list[TextBox]:
    """One frame's lower-band tag candidates, gated exactly like the at-bottom reading path:
    the OVERLAY_BOTTOM_FRAC band gate and content filters of find_tournament_tags, its
    drop_sublabels pass, and the _EDGE_FRAC/_CENTER_FRAC x-gates of _sided_tag_pair: so the
    learner sees the same boxes the per-frame reader chooses among. Lockstep with the reader is
    enforced by sharing _x_gated."""
    band = [t for t in texts if bbox_center_y(t) >= frame_height * OVERLAY_BOTTOM_FRAC
            and not is_round_text(t) and not is_numeric_text(t)
            and not parse_best_of(t["text"]) and not is_best_of_label(t["text"])
            and not is_ignored_tag_text(t["text"])]
    band = drop_sublabels(band)
    return _x_gated(band, frame_width)


def cluster_tag_rows(frames: list[Event], frame_width: int,
                     frame_height: int) -> tuple[list[RowCluster], int]:
    """Cluster every OCR frame's lower-band candidates by center-y. Returns the clusters
    ranked by mirror-frame count (then total frames) and the OCR-frame count the fractions
    are measured against."""
    ocr_frames = [f for f in frames if f.get("texts")]
    clusters: list[RowCluster] = []
    for frame_index, frame in enumerate(ocr_frames):
        for box in _row_candidates(frame.get("texts", []), frame_width, frame_height):
            cy, is_left = bbox_center_y(box), bbox_center_x(box) < frame_width / 2
            for cluster in clusters:
                if abs(cy - cluster.cy) <= _ROW_CLUSTER_TOL:
                    cluster.add(box, frame_index, is_left)
                    break
            else:
                fresh = RowCluster(cy=cy)
                fresh.add(box, frame_index, is_left)
                clusters.append(fresh)
    clusters.sort(key=lambda c: (len(c.mirror_frames), len(c.frames)), reverse=True)
    return clusters, len(ocr_frames)


def learn_tag_row(matches: list[Match], frame_width: int,
                  frame_height: int) -> LearnedTagRow | None:
    """Learn the at-bottom overlay's gamertag row across all of a VOD's matches: the row
    cluster mirror-symmetric in the most OCR frames (see the learned-row constants block for
    why that signal alone suffices: no font-height prior, no tier ordering). Returns None
    when no cluster is mirror-symmetric in _ROW_MIN_MIRROR_FRAC of the frames: the caller
    then keeps the bottom-edge anchor, so an unlearnable VOD reads exactly as before."""
    frames = [f for m in matches for f in m["frames"]]
    clusters, n_ocr_frames = cluster_tag_rows(frames, frame_width, frame_height)
    if not clusters or n_ocr_frames == 0:
        return None
    best = clusters[0]
    mirror_frac = len(best.mirror_frames) / n_ocr_frames
    if mirror_frac < _ROW_MIN_MIRROR_FRAC:
        return None
    return LearnedTagRow(y=best.cy, support=len(best.frames) / n_ocr_frames,
                         mirror_frac=mirror_frac, median_height=best.median_height(),
                         n_ocr_frames=n_ocr_frames)


# ── Top-band structure measurement ─────────────────────────────────────────────

@dataclass(frozen=True)
class TopBandStructure:
    """VOD-wide evidence that a tournament overlay exists in the TOP band: on how many
    OCR frames the top-band reader found a full, content-valid gamertag pair. Consumed by
    the smart-tags sweep to veto an --overlay-at-bottom winner (a top overlay too messy
    to *vote* still shows pair structure on most frames: see
    smart_tags.demote_at_bottom_on_top_structure)."""
    pair_frames: int        # OCR frames where the top reader returned BOTH sides
    n_ocr_frames: int
    matches_with_pair: int  # matches contributing at least one pair frame

    @property
    def pair_frame_frac(self) -> float:
        """Fraction of OCR frames showing a full top-band pair (0.0 when no OCR frames)."""
        return self.pair_frames / self.n_ocr_frames if self.n_ocr_frames else 0.0


def measure_top_band_structure(matches: list[Match], frame_width: int,
                               frame_height: int) -> TopBandStructure | None:
    """Measure how often the TOP band shows a full gamertag pair across a VOD's matches.

    Runs the real top-band reader (find_tournament_tags, at_bottom=False, no stacked
    handling, no learned row) on every OCR frame's raw texts and counts frames where BOTH
    sides read: i.e. a mirror-symmetric, content-valid name pair survived the reader's
    own junk gates (round/best-of/pronoun/handle filters, x-band gates, sublabel drop).
    This is deliberately a yes/no "does a top overlay exist" probe, not a row *learner*:
    no row election, no mirror clustering: the per-frame reader already encodes what a
    valid pair looks like. Structural presence is robust where voting is not: a messy
    overlay's text wanders too much to elect a stable string, but the pair is still
    there frame after frame. Returns None when the matches contain no OCR frames.

    Flag-independent like learn_tag_row (reads the raw frame texts), so the smart-tags
    sweep computes it once per VOD.

    Known blind spot: the raw texts are read WITHOUT persistent-graphics/banner filtering
    (deliberately, to stay flag-independent like learn_tag_row). A persistent, mirror-
    symmetric top-band banner that survives the reader's junk gates would therefore inflate
    the pair count here even though a real filter_banners run would strip it: potentially
    demoting a correct at-bottom overlay. This is latent, not observed: the
    TOP_STRUCTURE_PAIR_FRAME_FRAC gate is calibrated against this same unfiltered function
    (reference max 8% on genuine bottom-overlay/nameplate VODs), so the sample is
    self-consistent. If a banner-heavy VOD ever trips a false demotion, the fix is to filter
    banners here and re-calibrate the threshold against the filtered counts."""
    n_ocr_frames = pair_frames = matches_with_pair = 0
    for match in matches:
        match_has_pair = False
        for frame in match["frames"]:
            texts = frame.get("texts")
            if not texts:
                continue
            n_ocr_frames += 1
            found = find_tournament_tags(texts, frame_width=frame_width,
                                         frame_height=frame_height)
            # A full pair is a returned list with no None slot. find_tournament_tags returns
            # None or a two-element list, and a lone-tag salvage fills one slot with None
            # ([entry, None] / [None, entry]): those single-sided reads must not count.
            if found and None not in found:
                pair_frames += 1
                match_has_pair = True
        if match_has_pair:
            matches_with_pair += 1
    if n_ocr_frames == 0:
        return None
    return TopBandStructure(pair_frames=pair_frames, n_ocr_frames=n_ocr_frames,
                            matches_with_pair=matches_with_pair)


# ── Match enrichment ───────────────────────────────────────────────────────────

def build_vote_views(frames: list[Event], banners: list[tuple[str, float, float]] | None,
                     filter_overlays: bool) -> tuple[list[Event], list[Event]]:
    """The two frame views tag voting runs over, as ``(pre_static, voted)``:
    `pre_static` has the persistent stream banners stripped (when given), and `voted`
    additionally applies the static-overlay filter when `filter_overlays` is set.
    The vote itself sees `voted`; `pre_static` feeds the sponsor rescue (see
    _rescue_filtered_sponsor). The input frames are never mutated: both views are
    fresh lists (or the input itself when a step is off). Public so the smart-tags
    sweep (vodparse.smart_tags) can measure exactly the reads the vote saw."""
    pre_static = strip_banners(frames, banners) if banners else frames
    voted = filter_static_overlays(pre_static) if filter_overlays else pre_static
    return pre_static, voted


# ── Constant overlay-tag strip ─────────────────────────────────────────────────

# A voted tournament tag that reads as the SAME text in nearly every match of a VOD is a
# stream graphic, not a gamertag: a top-corner event title ("EXAMPLE WEEKLY #7",
# "EXAMPLE TOURNEY #23 - SPECIAL EDITION") that slipped past the x-gates and got elected match after
# match. Measured across 72 VODs: true banners vote in 97-100% of the matches that
# resolved their slot and never appear in the other slot (a graphic can't change sides),
# while the most dominant REAL players top out near 55% or swap slots with their seat
# (e.g. one VOD's most-played entrant voted 16x in slot 1 and 5x in slot 2). 0.8 splits
# the two populations with a wide margin on both sides.
CONSTANT_TAG_MIN_FRAC = 0.8

# ... and never on thin evidence: below this many matches a short VOD can't tell a
# constant graphic from a short winners-run (same floor as overlay_filter's
# BANNER_MIN_MATCHES).
CONSTANT_TAG_MIN_MATCHES = 8


def strip_constant_overlay_tags(matches: list[Match]) -> None:
    """Null out voted tournament tags that are a VOD-constant stream graphic (see the
    CONSTANT_TAG_MIN_FRAC block). A tag position is stripped only when every gate agrees
    it cannot be a player:

      * its (fuzzy-clustered) text won the vote in >= CONSTANT_TAG_MIN_FRAC of the
        matches that resolved the slot, over at least CONSTANT_TAG_MIN_MATCHES matches;
      * it only ever appears in that one slot (players change seats across a night;
        graphics don't);
      * the VOD's confirmed in-game nameplates span IMPOSTOR_MIN_IDENTITIES+ distinct
        pairs (in a long friendlies session the same two tags legitimately vote in
        every match);
      * no in-game nameplate anywhere in the VOD reads similar to it.

    Stripped slots become None (their sponsor slot too, staying index-aligned); a match
    left with no resolved position gets tournament_tags/tournament_sponsors = None so
    the in-game fallback engages downstream. Mutates `matches` in place; runs once per
    enrichment (enrich_matches), so every consumer: grouping, labels, titles, CSV,
    the smart-tags scorers: sees the cleaned votes."""
    if confirmed_identity_count(matches) < IMPOSTOR_MIN_IDENTITIES:
        return
    ingame_tags = [t for m in matches for t in (m.get("ingame_tags") or []) if t]
    slot_votes: dict[int, list[str]] = {}
    for m in matches:
        for slot, tag in enumerate(m.get("tournament_tags") or []):
            if tag:
                slot_votes.setdefault(slot, []).append(tag)

    for slot, votes in slot_votes.items():
        # Greedy fuzzy clustering (first match wins), same shape as confirmed_identity_count.
        clusters: list[list[str]] = []
        for tag in votes:
            for cluster in clusters:
                if similar(tag, cluster[0]):
                    cluster.append(tag)
                    break
            else:
                clusters.append([tag])
        top = max(clusters, key=len)
        if len(top) < max(CONSTANT_TAG_MIN_MATCHES, CONSTANT_TAG_MIN_FRAC * len(votes)):
            continue
        representative = top[0]
        other_slot_hits = any(similar(tag, representative)
                              for other, tags in slot_votes.items() if other != slot
                              for tag in tags)
        if other_slot_hits:
            continue
        if any(similar(representative, tag) for tag in ingame_tags):
            continue
        stripped = 0
        for m in matches:
            tags = m.get("tournament_tags")
            voted = tags[slot] if tags and slot < len(tags) else None
            if not voted or not similar(voted, representative):
                continue
            assert tags is not None   # narrowed by `voted` above; for the type checker
            tags[slot] = None
            sponsors = m.get("tournament_sponsors")
            if sponsors and slot < len(sponsors):
                sponsors[slot] = None
            if not any(tags):
                m["tournament_tags"] = None
                m["tournament_sponsors"] = None
            stripped += 1
        print(f"  Constant overlay text stripped from the tag votes ({stripped} matches): "
              f"{representative!r} is a stream graphic, not a gamertag")


def enrich_match(match: Match, filter_overlays: bool = True,
                 banners: list[tuple[str, float, float]] | None = None,
                 frame_width: int = 1920, frame_height: int = 1080,
                 overlay_at_bottom: bool = False, overlay_stacked: bool = False,
                 tag_row_y: float | None = None,
                 ingame_tag_frac: float = INGAME_TAG_BAND_FRAC_DEFAULT) -> Match:
    """Vote the ingame tags and tournament overlay for a match. When `filter_overlays`
    is set, the vote sees only screen-locked overlay text (see filter_static_overlays)
    so moving in-game scenery can't be mistaken for a gamertag. `banners` (from
    find_persistent_graphics) are stripped first so a VOD-wide stream graphic can't win the
    vote. `frame_width`/`frame_height` are the source-video resolution the position thresholds
    are measured against (the bboxes are in source-video pixels). `overlay_at_bottom` reads the
    tournament overlay from the bottom strip for the rare tournament that renders it there;
    `tag_row_y` is that mode's VOD-wide learned gamertag row (from learn_tag_row: computed
    once per VOD, e.g. by enrich_matches, since it spans all matches). `overlay_stacked` reads
    each gamertag from the row below its location/handle metadata line. `ingame_tag_frac` widens
    the in-game nametag band for shrunk captures (see INGAME_TAG_BAND_FRAC_DEFAULT); it does not
    touch the tournament reader. The match's stored `frames` keep their full text: only the
    voting uses the view."""
    raw_frames, frames = build_vote_views(match["frames"], banners, filter_overlays)
    view: Match = {**match, "frames": frames}
    attach_ingame_tags(view, frame_width, frame_height, ingame_tag_frac=ingame_tag_frac)
    # Pass the pre-static-filter frames so the overlay vote can recover a gamertag the filter
    # pruned in favour of a position-locked sponsor (see _rescue_filtered_sponsor).
    attach_tournament_overlay(view, frame_width, frame_height, at_bottom=overlay_at_bottom,
                              stacked=overlay_stacked,
                              raw_frames=raw_frames if filter_overlays else None,
                              tag_row_y=tag_row_y)
    match["ingame_tags"]           = view["ingame_tags"]
    match["ingame_tags_confirmed"] = view["ingame_tags_confirmed"]
    match["tournament_tags"]       = view["tournament_tags"]
    match["tournament_sponsors"]   = view["tournament_sponsors"]
    match["round"]                 = view["round"]
    match["best_of"]               = view["best_of"]
    # Characters ride on YOLO detections, not OCR text, so the overlay/banner text
    # filters above don't apply: vote on the raw frames.
    attach_characters(match)
    return match


def enrich_matches(raw: list[Match], *, filter_overlays: bool, filter_banners: bool,
                   frame_width: int, frame_height: int, overlay_at_bottom: bool = False,
                   overlay_stacked: bool = False, banners: list[tuple[str, float, float]] | None = None,
                   ingame_tag_frac: float = INGAME_TAG_BAND_FRAC_DEFAULT) -> list[Match]:
    """Vote each raw match's frames into stable tags/round/best-of, optionally stripping
    persistent stream graphics from the vote first. `overlay_at_bottom` reads the tournament
    overlay's gamertags/sponsors from the bottom of the frame instead of the top, anchored on
    the VOD-wide learned gamertag row when one is learnable (see learn_tag_row); round and
    best-of stay top-gated, so a fully-bottom overlay exports those two empty (known limit).
    `overlay_stacked` reads each gamertag from the line below its location/handle metadata.
    `banners` short-circuits the persistent-graphics scan for a caller that already ran it
    over the same matches (the smart-tags sweep enriches the same raw matches 16 times);
    it is only consulted when `filter_banners` is set, and when left None the scan runs
    here as before. `ingame_tag_frac` widens the in-game nametag band for shrunk captures
    (see INGAME_TAG_BAND_FRAC_DEFAULT)."""
    if not filter_banners:
        banners = None
    elif banners is None:
        banners = find_persistent_graphics(raw)
    if banners:
        print(f"  Persistent graphics stripped from voting: {sorted({b[0] for b in banners})}")
    tag_row_y: float | None = None
    if overlay_at_bottom:
        # Learn the gamertag row once across all matches; every match's vote then anchors on
        # it (see the learned-row constants block). Unlearnable VODs keep the edge anchor.
        row = learn_tag_row(raw, frame_width, frame_height)
        if row is not None:
            tag_row_y = row.y
            print(f"  Learned at-bottom tag row: y={row.y:.0f}px "
                  f"(mirror-symmetric in {row.mirror_frac:.0%} of {row.n_ocr_frames} OCR frames)")
        else:
            print("  No reliable at-bottom tag row learned; using the bottom-edge anchor")
    enriched = [enrich_match(m, filter_overlays=filter_overlays, banners=banners,
                             frame_width=frame_width, frame_height=frame_height,
                             overlay_at_bottom=overlay_at_bottom, overlay_stacked=overlay_stacked,
                             tag_row_y=tag_row_y, ingame_tag_frac=ingame_tag_frac)
                for m in raw]
    strip_constant_overlay_tags(enriched)
    return enriched
