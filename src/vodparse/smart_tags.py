"""Smart tag reading: sweep the four overlay-reading flags and keep the best result.

The four extract_matches flags (--overlay-stacked, --no-overlay-filter,
--overlay-at-bottom, --filter-banners) only change the cheap post-processing of an
already-computed pipeline JSON, so the right combination can be *searched* instead of
guessed per VOD: enrich the same raw matches under every combination, score each run,
and rank lexicographically: the ordered scorers below, lower = better, ties falling to
the next scorer and a dead tie to combo order (the all-defaults combo is deliberately
first, so it wins when nothing separates the runs):

  1. impostor_games: games living in a set whose games' confirmed in-game nameplates
                        span 3+ distinct player pairs (grouping.is_impostor_set): the
                        signature of a constant stream graphic elected as a gamertag
                        gluing different matches into one giant set.
  2. single_games: singleton sets from group_into_sets (a game whose players match
                        no neighbour); fewer means the tags grouped consecutive games.
  3. ingame_fallbacks: matches whose identity would fall back to the in-game nameplates
                        because no tournament-overlay tag was voted.
  4. half_reads: matches whose voted overlay tags resolved one position but not the
                        other (["TITLE", None]): a full pair is what a real overlay
                        yields; persistent half reads are usually a graphic, not a tag.
  5. tag_y_variance: vertical scatter of the voted overlay tags' reads across the run
                        (a real overlay is screen-locked; misreads wander).

Four post-ranking corrections guard the scorers' blind spots: all facets of the same
confusion, the tournament overlay and the in-game nameplates masquerading as each
other. A bottom-of-frame overlay reads as in-game nameplates, so a non-at-bottom combo
can group beautifully through the in-game fallback without voting a single overlay tag;
when the scorer winner grouped HEAVY_FALLBACK_SET_FRACTION of its sets that way, the
best-ranked --overlay-at-bottom run is promoted to winner instead
(promote_at_bottom_on_heavy_fallback). Conversely, on a VOD whose overlay is NOT at the
bottom, the at-bottom read returns the nameplate row itself: screen-locked and cleanly
grouping, so the scorers cannot tell it from a real overlay; when a non-at-bottom run
read the overlay on nearly every set and grouped almost as well, it is preferred over
the at-bottom winner (demote_at_bottom_on_light_fallback). That demotion is blind to a
top overlay too messy to *vote*: the at-bottom nameplate read then out-scores every
top run outright, so a third correction checks the physical evidence instead. When the
top band shows a content-valid gamertag pair on enough OCR frames, the overlay is at
the top no matter how badly it votes, and an at-bottom winner is demoted
(demote_at_bottom_on_top_structure). The same masquerade also skews WHICH non-at-bottom
combo wins: single_games is the primary scorer, so a combo that reads the overlay poorly
but whose many nameplate-fallback matches happen to group cleanly out-ranks the combo
that reads the real overlay everywhere. The demotions therefore promote the non-at-bottom
run with the fewest fallbacks that still grouped within the singleton margin (not merely
the top-ranked one), and a final correction applies the same preference when such a
poor reader won the scorers outright (prefer_overlay_reader_among_non_at_bottom). All
four corrections share one hard limit: none may promote a run with MORE games caught in
impostor sets than the run it replaces (_promotion_worsens_impostors): the corrections
reason from indirect signals, and an impostor set is direct evidence the candidate
grouped by something that isn't a player.

The scorer list is the tuning surface: add/remove/reorder DEFAULT_SCORERS entries (a
future neighboring-set tag-similarity criterion is one more entry), and
run_sweep(combos=...) restricts the sweep to a subset of combinations for slower
machines. Each run's per-combo wall time is kept so accumulated reports answer "which
flags ever win, and what does the sweep cost?".

Reproduce a sweep from a cached analysis:
    python -m vodparse.cli.extract_matches --input json_output/<stem>/pipeline.json --csv-only --smart-tags
The ranking is printed and persisted to json_output/<stem>/smart_tags_report.json.
"""

import contextlib
import io
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from itertools import product
from pathlib import Path
from statistics import fmean, median, pvariance
from typing import cast

from vodparse.bbox import bbox_center_y
from vodparse.games import DEFAULT_MAX_GAME_COUNT
from vodparse.grouping import group_into_sets, is_impostor_set, overlay_fell_back_to_ingame
from vodparse.overlay_filter import find_persistent_graphics
from vodparse.schema import Match, SetGroup, TagEntry
from vodparse.tags import (INGAME_TAG_BAND_FRAC_DEFAULT, TopBandStructure, build_vote_views,
                           enrich_matches, find_tournament_tags, learn_tag_row,
                           measure_top_band_structure)
from vodparse.voting import collect_positioned_entries

# The ranking file written next to matches.json (see extract_matches --smart-tags).
SMART_TAGS_REPORT_NAME = "smart_tags_report.json"

# tag_y_variance for a run in which no tag position collected enough reads: infinitely
# bad, so a run that voted no tags can never win the tiebreak by being vacuously stable.
NO_TAG_VARIANCE = float("inf")

# A tag position needs this many per-match samples before its variance is meaningful
# a single sample has variance 0 by definition, which would reward sparse runs.
MIN_VARIANCE_SAMPLES = 2

# When the scorer winner grouped at least this fraction of its sets from the in-game
# nameplate fallback, the tournament overlay went essentially unread: and a bottom-of-
# frame overlay is the usual culprit: its text masquerades as in-game nameplates, so the
# fallback grouping scores well without the overlay ever being read as an overlay. The
# best-ranked --overlay-at-bottom run is promoted instead
# (see promote_at_bottom_on_heavy_fallback).
HEAVY_FALLBACK_SET_FRACTION = 0.5

# The opposite blind spot: on a VOD whose overlay is NOT at the bottom, an at-bottom run
# reads the in-game nameplate row itself: screen-locked and cleanly grouping, so it can
# out-score the run that read the real overlay. The demotion fires only when a
# non-at-bottom run fell back on fewer than this fraction of its sets, i.e. read the
# overlay nearly everywhere. Measured: a real overlay reads on every set (reference VOD C 0%),
# while reference VOD D's sporadic junk text sat at 29% and reference VOD E's genuine
# bottom overlay at 64%: both must not qualify. Must stay below
# HEAVY_FALLBACK_SET_FRACTION: that gap is what keeps the two corrections from ever
# firing against each other (see demote_at_bottom_on_light_fallback).
LIGHT_FALLBACK_SET_FRACTION = 0.10

# ...and only when that run grouped nearly as well: at most this fraction of its sets more
# singletons than the at-bottom winner. Measured: reference VOD C's real overlay lost by 1 of its
# 22 sets (5%, a single OCR misread: must pass), while reference VOD F's out-of-sync side-bracket
# overlay lost by 15 of 29 (52%: must fail). Generous by design: a real overlay beats
# nameplates on identity (entrant names, sponsors), so a few misread-induced singletons
# are an acceptable price.
OVERLAY_SINGLETON_MARGIN = 0.25

# The third blind spot: a top overlay too messy to VOTE never trips the light-fallback
# demotion above (every non-at-bottom run fell back heavily), yet its gamertag pair is
# still physically present frame after frame: measure_top_band_structure counts the OCR
# frames where the top reader found a full content-valid pair. An at-bottom winner is
# demoted when that fraction reaches this threshold. Measured across the reference VODs: real top
# overlays show a pair on 25%+ of OCR frames even when voting collapses (reference VOD G 64%,
# reference VOD H 25%; clean overlays 80-100%), while genuine bottom-overlay and
# nameplate-identity VODs top out at 8% (reference VOD I cached read): the threshold sits in
# that gap (see demote_at_bottom_on_top_structure).
TOP_STRUCTURE_PAIR_FRAME_FRAC = 0.15

# ...and only on real evidence: at least this many pair frames, so a single lucky frame
# on a short VOD can never veto the at-bottom read (reference VOD H's genuine 2
# pair frames of 8 must still qualify).
MIN_TOP_STRUCTURE_PAIR_FRAMES = 2


# ── Flag combinations ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FlagCombo:
    """One combination of the four extract_matches tag-reading flags."""
    overlay_stacked: bool = False
    no_overlay_filter: bool = False
    overlay_at_bottom: bool = False
    filter_banners: bool = False

    # Field name → CLI flag, in the field order above (also the cli_flags/label order).
    _FLAG_BY_FIELD = {
        "overlay_stacked":   "--overlay-stacked",
        "no_overlay_filter": "--no-overlay-filter",
        "overlay_at_bottom": "--overlay-at-bottom",
        "filter_banners":    "--filter-banners",
    }

    @classmethod
    def all_combos(cls) -> list["FlagCombo"]:
        """All 16 combinations. The all-defaults combo comes first: rank_runs' stable
        sort keeps input order on ties, so full ties resolve to the conservative pick."""
        return [cls(*values) for values in product((False, True), repeat=4)]

    def cli_flags(self) -> list[str]:
        """The extract_matches flags this combo sets (empty for the all-defaults combo)."""
        return [flag for name, flag in self._FLAG_BY_FIELD.items() if getattr(self, name)]

    def label(self) -> str:
        """Human-readable name for tables/reports: the set flags, or "(defaults)"."""
        return " ".join(self.cli_flags()) or "(defaults)"


# ── Sweep runs and scorers ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class SweepContext:
    """Combo-independent inputs shared by every run (and available to scorers)."""
    frame_width: int
    frame_height: int
    banners: list[tuple]   # find_persistent_graphics(raw): flag-independent, computed once
    # learn_tag_row(raw): also flag-independent (it reads the raw frame texts); consulted
    # only by at-bottom combos, so scorers rebuild exactly the reads those votes saw. None
    # when no at-bottom combo is swept or the VOD has no reliable mirror-symmetric row.
    tag_row_y: float | None = None


@dataclass
class SweepRun:
    """One combo's post-processing result plus its scores.

    `notes` is the stdout the enrichment/grouping printed for this combo (per-match
    voting notes etc.); extract_matches replays the winner's verbatim so a smart run's
    output matches a plain run with the winning flags. `elapsed_sec` is deliberate
    developer telemetry: accumulated reports show what each combo costs, informing a
    future decision to sweep fewer combos on slower machines. `promotion` is set only
    when a post-ranking correction (promote_at_bottom_on_heavy_fallback,
    demote_at_bottom_on_light_fallback, demote_at_bottom_on_top_structure, or
    prefer_overlay_reader_among_non_at_bottom) moved this run to the front: the
    human-readable reason, echoed on the console and persisted in the report."""
    combo: FlagCombo
    matches: list[Match]
    sets: list[SetGroup]
    scores: dict[str, float] = field(default_factory=dict)   # scorer name → value, scorer order
    notes: str = ""
    elapsed_sec: float = 0.0
    promotion: str | None = None


@dataclass(frozen=True)
class Scorer:
    """One ranking criterion: `score` maps a run to a comparable metric (lower = better).
    A dataclass rather than a bare callable so tables and reports can carry the name and
    description alongside the value."""
    name: str
    description: str
    score: Callable[[SweepRun, SweepContext], float]


def score_impostor_games(run: SweepRun, ctx: SweepContext) -> float:
    """Criterion 1: how many games live in an impostor set: one whose games' confirmed
    in-game nameplates span 3+ distinct player pairs (grouping.is_impostor_set). A real
    set is one pair of players, however long; a set that keeps changing pairs underneath
    a constant grouping identity was glued together by a stream graphic elected as a
    gamertag. Counted in games (not sets) so partial corruption still compares: a combo
    collapsing 31 games into one impostor set ranks below one collapsing 10."""
    return float(sum(len(s["matches"]) for s in run.sets if is_impostor_set(s["matches"])))


def score_singleton_sets(run: SweepRun, ctx: SweepContext) -> float:
    """Criterion 2: how many sets are a single game. A game sharing players with no
    neighbour usually means its tags were misread (a genuine warm-up game also lands
    here, which is why this is minimized, never required to be zero)."""
    return float(sum(1 for s in run.sets if len(s["matches"]) == 1))


def score_ingame_fallback_matches(run: SweepRun, ctx: SweepContext) -> float:
    """Criterion 3: how many matches would take their identity from the in-game
    nameplates because no tournament-overlay tag was voted. Counted per match and
    ungated by --group-by (unlike the Review screen's marking): it measures raw overlay
    readability, which feeds titles and CSV columns under every grouping mode."""
    return float(sum(1 for m in run.matches
                     if overlay_fell_back_to_ingame(m.get("tournament_tags"),
                                                    m.get("ingame_tags"))))


def _voted_tag_y_samples(m: Match, combo: FlagCombo, ctx: SweepContext) -> list[float | None]:
    """Per tag position: this match's vertical sample for its voted tournament tag: the
    median bbox_center_y (normalized by frame height, so it is resolution-independent) of
    the vote-view reads whose text equals the voted tag (casefolded, matching vote()'s
    tally): or None when the position has no voted tag or no matching read. The views
    are rebuilt exactly as enrich_match built them (build_vote_views), so the samples
    measure the same reads the vote saw. A tag elected by the sponsor rescue
    (_rescue_filtered_sponsor) can have no matching read in the voted view; that position
    simply contributes no sample."""
    tags = m.get("tournament_tags") or []
    if not any(tags):
        return []
    banners = ctx.banners if combo.filter_banners else None
    _, voted_frames = build_vote_views(m["frames"], banners,
                                       filter_overlays=not combo.no_overlay_filter)
    finder = partial(find_tournament_tags, frame_width=ctx.frame_width,
                     frame_height=ctx.frame_height,
                     at_bottom=combo.overlay_at_bottom, stacked=combo.overlay_stacked,
                     tag_row_y=ctx.tag_row_y if combo.overlay_at_bottom else None)
    # The learned-row path yields None for a side unread in a frame; those never voted.
    columns = [[e for e in col if e is not None]
               for col in cast("list[list[TagEntry | None]]",
                               collect_positioned_entries(voted_frames, finder))]
    samples: list[float | None] = []
    for i, tag in enumerate(tags):
        if not tag or i >= len(columns):
            samples.append(None)
            continue
        key = tag.strip().casefold()
        ys = [bbox_center_y(e) / ctx.frame_height for e in columns[i]
              if (e.get("text") or "").strip().casefold() == key]
        samples.append(median(ys) if ys else None)
    return samples


def score_half_read_matches(run: SweepRun, ctx: SweepContext) -> float:
    """Criterion 4: how many matches voted a half-read overlay pair: one position
    resolved, the other None. A real overlay yields both gamertags; a position that
    persistently votes alone is usually a graphic the lone-tag salvage elected (a
    top-corner event title), which reads identically every game and so never trips
    the earlier scorers. Ranked below ingame_fallbacks: half a read is still better
    than no overlay read at all."""
    return float(sum(1 for m in run.matches
                     if _is_half_read(m.get("tournament_tags"))))


def _is_half_read(tags: list[str | None] | None) -> bool:
    """Whether a voted overlay read resolved at least one position but not all of them."""
    if not tags:
        return False
    return any(tags) and not all(tags)


def score_tag_y_variance(run: SweepRun, ctx: SweepContext) -> float:
    """Criterion 5 (tiebreak): how much the voted overlay tags' vertical position wanders
    across the run. A real overlay is screen-locked, so its reads sit at one height;
    scenery or HUD text mistaken for a tag scatters. Per tag position, the population
    variance of the per-match samples (see _voted_tag_y_samples); positions with fewer
    than MIN_VARIANCE_SAMPLES samples are skipped, and a run with no scorable position
    returns NO_TAG_VARIANCE so it can never win by having voted nothing. Positions are
    aggregated by mean."""
    per_position: dict[int, list[float]] = {}
    for m in run.matches:
        for i, y in enumerate(_voted_tag_y_samples(m, run.combo, ctx)):
            if y is not None:
                per_position.setdefault(i, []).append(y)
    variances = [pvariance(ys) for ys in per_position.values()
                 if len(ys) >= MIN_VARIANCE_SAMPLES]
    return fmean(variances) if variances else NO_TAG_VARIANCE


# The ranking, in order: criterion 1 decides, ties fall to 2, and so on. This tuple is
# the tuning surface: reorder/add/remove entries to change the rules (a future
# "neighboring-set tag similarity" criterion is one more entry here).
DEFAULT_SCORERS: tuple[Scorer, ...] = (
    Scorer("impostor_games",
           "games in sets whose confirmed ingame nameplates span 3+ player pairs",
           score_impostor_games),
    Scorer("single_games",
           "sets containing a single game (players matching no neighbour)",
           score_singleton_sets),
    Scorer("ingame_fallbacks",
           "matches with no voted overlay tag, falling back to ingame nameplates",
           score_ingame_fallback_matches),
    Scorer("half_reads",
           "matches whose voted overlay tags resolved one position but not the other",
           score_half_read_matches),
    Scorer("tag_y_variance",
           "vertical scatter of the voted overlay tags' reads (normalized; lower = steadier)",
           score_tag_y_variance),
)


# ── The sweep ──────────────────────────────────────────────────────────────────

def rank_runs(runs: list[SweepRun], scorers: Sequence[Scorer]) -> list[SweepRun]:
    """Runs ordered best-first: lexicographic over the scorers' values in scorer order
    (lower = better). sorted() is stable, so a full tie keeps the input order: and
    all_combos puts the all-defaults combo first, so it wins dead ties."""
    return sorted(runs, key=lambda run: tuple(run.scores[s.name] for s in scorers))


def _fallback_set_fraction(run: SweepRun) -> float:
    """Fraction of the run's sets whose identity came from the in-game nameplates only
    (no tournament-overlay tag voted anywhere in the set)."""
    if not run.sets:
        return 0.0
    fallen = sum(1 for s in run.sets
                 if overlay_fell_back_to_ingame(s.get("tournament_tags"),
                                                s.get("ingame_tags")))
    return fallen / len(run.sets)


def _singleton_count(run: SweepRun) -> int:
    """How many of the run's sets contain a single game. Recomputed from the sets rather
    than read from run.scores, so the post-ranking corrections work whatever scorers ran."""
    return sum(1 for s in run.sets if len(s["matches"]) == 1)


def _ingame_fallback_count(run: SweepRun) -> int:
    """How many of the run's matches fell back to the in-game nameplates (no overlay tag
    voted): the per-match count score_ingame_fallback_matches computes, recomputed here
    so the post-ranking corrections work whatever scorers ran (cf. _singleton_count)."""
    return sum(1 for m in run.matches
               if overlay_fell_back_to_ingame(m.get("tournament_tags"),
                                              m.get("ingame_tags")))


def _impostor_game_count(run: SweepRun) -> int:
    """How many of the run's games live in impostor sets (score_impostor_games' count),
    recomputed from the sets so the post-ranking corrections work whatever scorers ran
    (cf. _singleton_count)."""
    return sum(len(s["matches"]) for s in run.sets if is_impostor_set(s["matches"]))


def _promotion_worsens_impostors(winner: SweepRun, candidate: SweepRun) -> bool:
    """Whether promoting `candidate` over `winner` would increase the games caught in
    impostor sets. Every post-ranking correction checks this before promoting: the
    corrections reason from indirect signals (fallback fractions, top-band structure),
    but an impostor set is direct evidence the candidate's grouping identity is not a
    player (grouping.is_impostor_set), and no indirect signal may override it. Reference
    VOD J is the motivating case: its top-corner event title inflates the top-band pair count
    (the known measure_top_band_structure blind spot), so the top-structure demotion
    would otherwise reinstall the run whose lone banner "gamertag" glued all 31 games
    into one set."""
    return _impostor_game_count(candidate) > _impostor_game_count(winner)


def _best_overlay_reader_non_at_bottom(runs: list[SweepRun]) -> SweepRun | None:
    """The non-at-bottom run that read the tournament overlay the MOST (fewest in-game
    fallbacks), among those grouping within OVERLAY_SINGLETON_MARGIN of the best-grouping
    non-at-bottom run: the replacement candidate the two demotion corrections and
    prefer_overlay_reader_among_non_at_bottom share. Rank order alone is not enough:
    single_games is the primary scorer, so a combo that reads the overlay POORLY but
    whose nameplate-fallback matches group cleanly out-ranks the combo that reads the
    real overlay everywhere. `runs` must be rank-ordered: min() keeps the first of tied
    elements, so a fallback tie resolves to the highest-ranked run (the all-defaults
    combo when nothing separates them). None when every non-at-bottom run grouped into
    zero sets, or every swept combo was at-bottom."""
    candidates = [r for r in runs if not r.combo.overlay_at_bottom and r.sets]
    if not candidates:
        return None
    fewest_singletons = min(_singleton_count(r) for r in candidates)
    eligible = [r for r in candidates
                if _singleton_count(r) - fewest_singletons
                <= OVERLAY_SINGLETON_MARGIN * len(r.sets)]
    return min(eligible, key=_ingame_fallback_count)


def _promote_to_front(runs: list[SweepRun], candidate: SweepRun, reason: str) -> list[SweepRun]:
    """Move `candidate` to the front with `reason` recorded as its promotion (relative
    order otherwise unchanged): the shared tail of every post-ranking correction. Any
    promotion a prior correction left on another run is cleared, so the invariant
    "promotion is set only on the front run" survives correction chains (a top-structure
    demotion can reverse a heavy-fallback promotion; the reversed run must not keep its
    stale reason in the report or the GUI's runner-up line)."""
    for run in runs:
        run.promotion = None
    candidate.promotion = reason
    return [candidate, *(r for r in runs if r is not candidate)]


def promote_at_bottom_on_heavy_fallback(runs: list[SweepRun]) -> list[SweepRun]:
    """Post-ranking correction: when the scorer winner leaned on the in-game fallback for
    HEAVY_FALLBACK_SET_FRACTION of its sets, move the best-ranked --overlay-at-bottom run
    to the front (relative order otherwise unchanged).

    Rationale: a bottom-of-frame overlay is read as in-game nameplates, so a non-at-bottom
    combo can group its sets well *through the fallback* without voting a single overlay
    tag: out-scoring the at-bottom combo that reads the overlay properly. Heavy fallback
    in the winner is that signature. The promotion is skipped when no at-bottom run voted
    any overlay tag at all: then the overlay genuinely is not at the bottom, and the
    fallback grouping is the best available. The promoted run's `promotion` field records
    the reason for the console, the report, and the GUI summary."""
    winner = runs[0]
    if winner.combo.overlay_at_bottom:
        return runs
    fraction = _fallback_set_fraction(winner)
    if fraction < HEAVY_FALLBACK_SET_FRACTION:
        return runs
    candidate = next(
        (r for r in runs
         if r.combo.overlay_at_bottom
         and any(any(m.get("tournament_tags") or []) for m in r.matches)),
        None)
    if candidate is None or _promotion_worsens_impostors(winner, candidate):
        return runs
    return _promote_to_front(
        runs, candidate,
        f"{fraction:.0%} of the scorer winner's ({winner.combo.label()}) sets grouped via "
        "the ingame fallback: promoted the best --overlay-at-bottom run instead")


def demote_at_bottom_on_light_fallback(runs: list[SweepRun]) -> list[SweepRun]:
    """Post-ranking correction, the mirror of promote_at_bottom_on_heavy_fallback: when an
    --overlay-at-bottom winner is out-scoring a run that demonstrably read a real overlay,
    move the fullest-reading non-at-bottom run (_best_overlay_reader_non_at_bottom) to
    the front (relative order otherwise unchanged).

    Rationale: on a VOD whose overlay is NOT at the bottom, the at-bottom read returns the
    in-game nameplate row itself: screen-locked and cleanly grouping, so the scorers
    cannot tell it from a real overlay (the nameplate reads count as voted overlay tags).
    The non-at-bottom run is preferred when three things hold: it read the overlay on
    nearly every set (fallback under LIGHT_FALLBACK_SET_FRACTION: junk text reads far
    patchier), it voted at least two distinct set identities (an "overlay" that never
    changes identifies nobody), and it grouped within OVERLAY_SINGLETON_MARGIN of the
    winner's singleton count (an out-of-sync overlay groups far worse, and then the
    nameplates genuinely are the better identity source). The two corrections can never
    fight: this one needs a candidate under LIGHT_FALLBACK_SET_FRACTION, the other a
    winner over HEAVY_FALLBACK_SET_FRACTION, so a run demoted by either can never qualify
    the other. The promoted run's `promotion` field records the reason for the console,
    the report, and the GUI summary."""
    winner = runs[0]
    if not winner.combo.overlay_at_bottom:
        return runs
    candidate = _best_overlay_reader_non_at_bottom(runs)
    if candidate is None or not candidate.sets:
        return runs
    fraction = _fallback_set_fraction(candidate)
    if fraction >= LIGHT_FALLBACK_SET_FRACTION:
        return runs
    identities = {tuple(s.get("tournament_tags") or []) for s in candidate.sets
                  if any(s.get("tournament_tags") or [])}
    if len(identities) < 2:
        return runs
    excess = _singleton_count(candidate) - _singleton_count(winner)
    if excess > OVERLAY_SINGLETON_MARGIN * len(candidate.sets):
        return runs
    if _promotion_worsens_impostors(winner, candidate):
        return runs
    return _promote_to_front(
        runs, candidate,
        f"{candidate.combo.label()} read the tournament overlay on {1 - fraction:.0%} of "
        f"its sets and grouped within {max(excess, 0)} singleton set(s) of the "
        f"{winner.combo.label()} winner, whose tags mirror the ingame nameplates: "
        "promoted the overlay run instead")


def top_structure_clears_gates(top_structure: TopBandStructure | None) -> bool:
    """Whether the top band clears BOTH physical-evidence gates of
    demote_at_bottom_on_top_structure: a content-valid gamertag pair on at least
    MIN_TOP_STRUCTURE_PAIR_FRAMES OCR frames AND on at least TOP_STRUCTURE_PAIR_FRAME_FRAC
    of them (False for a None structure). This is the single home for that test:
    every caller shares it, so a future gate change (a third gate, a tweaked
    comparison) reaches them all at once."""
    return (top_structure is not None
            and top_structure.pair_frames >= MIN_TOP_STRUCTURE_PAIR_FRAMES
            and top_structure.pair_frame_frac >= TOP_STRUCTURE_PAIR_FRAME_FRAC)


def demote_at_bottom_on_top_structure(runs: list[SweepRun],
                                      top_structure: TopBandStructure | None) -> list[SweepRun]:
    """Post-ranking correction: when an --overlay-at-bottom winner contradicts physical
    top-band evidence: the top reader found a full gamertag pair on
    TOP_STRUCTURE_PAIR_FRAME_FRAC of the VOD's OCR frames: move the fullest-reading
    non-at-bottom run (_best_overlay_reader_non_at_bottom) to the front (relative order
    otherwise unchanged).

    Rationale: demote_at_bottom_on_light_fallback catches the nameplate masquerade only
    when some non-at-bottom run VOTED the top overlay nearly everywhere. A top overlay
    messy enough to defeat voting slips past it: the at-bottom nameplate read then wins
    with perfect scores while every top run drowns in fallback. But voting instability
    does not erase the overlay's structure: a content-valid pair still shows frame after
    frame (measure_top_band_structure), while genuine bottom-overlay and nameplate-identity
    VODs have next to none (the reader's own junk gates see to that). So this correction
    trusts the frames over the vote. It runs after the two fallback corrections and only
    ever moves an at-bottom run OUT of first place, so a light-fallback demotion leaves it
    nothing to do; a genuine bottom
    overlay promoted on heavy fallback sits far below the threshold (reference maximum 8%
    against the 15% gate), and should a heavy-fallback promotion ever be reversed here (a
    messy top overlay whose fallback grouping briefly out-scored the nameplate read), the
    reversed run's stale promotion reason is cleared by _promote_to_front. The promoted
    run's `promotion` field records the reason for the console, the report, and the GUI
    summary."""
    winner = runs[0]
    # top_structure is None kept as its own check so mypy narrows it non-None below (the
    # promotion message reads its fields); the gate predicate handles the numeric floors.
    if not winner.combo.overlay_at_bottom or top_structure is None:
        return runs
    if not top_structure_clears_gates(top_structure):
        return runs
    candidate = _best_overlay_reader_non_at_bottom(runs)
    # Refuse a zero-set candidate: promoting it would make an empty grouping the winner
    # the same degenerate case demote_at_bottom_on_light_fallback guards against. The
    # helper already requires sets; the explicit check keeps mypy narrowing for the
    # message below and survives a future helper change.
    if candidate is None or not candidate.sets:
        return runs
    if _promotion_worsens_impostors(winner, candidate):
        return runs
    return _promote_to_front(
        runs, candidate,
        f"the top band shows a valid gamertag pair in {top_structure.pair_frame_frac:.0%} "
        f"of {top_structure.n_ocr_frames} OCR frames: the overlay is at the top, and the "
        f"{winner.combo.label()} winner's tags mirror the ingame nameplates: promoted "
        f"the best non-at-bottom run ({candidate.combo.label()}) instead")


def prefer_overlay_reader_among_non_at_bottom(runs: list[SweepRun]) -> list[SweepRun]:
    """Post-ranking correction for the scorer-won blind spot: when the scorer winner is a
    non-at-bottom combo that read the overlay POORLY (its nameplate-fallback matches merely
    grouped cleanly), replace it with the non-at-bottom combo that read the real overlay
    the MOST, provided that reader grouped within OVERLAY_SINGLETON_MARGIN of the
    best-grouping non-at-bottom run (_best_overlay_reader_non_at_bottom). Only ever swaps
    one non-at-bottom run for another, so it cannot fight the earlier corrections: a
    heavy-fallback promotion leaves an at-bottom winner, which the first guard refuses to
    touch, and the two demotions trigger only on at-bottom winners: the opposite of this
    correction's trigger. Running last, it engages only on a surviving non-at-bottom poor
    reader: the overlay was never read as at-bottom, the poor reader simply won the
    scorers outright, and no earlier correction had reason to intervene."""
    winner = runs[0]
    if winner.combo.overlay_at_bottom:
        return runs
    candidate = _best_overlay_reader_non_at_bottom(runs)
    if candidate is None or candidate is winner:
        return runs
    if _ingame_fallback_count(candidate) >= _ingame_fallback_count(winner):
        return runs
    if _promotion_worsens_impostors(winner, candidate):
        return runs
    return _promote_to_front(
        runs, candidate,
        f"{winner.combo.label()} won the scorers but read the overlay on only part of the "
        f"VOD ({_ingame_fallback_count(winner)} ingame fallbacks); "
        f"{candidate.combo.label()} read it more fully "
        f"({_ingame_fallback_count(candidate)}) and grouped within the overlay margin: "
        "preferred the fuller overlay read")


def run_sweep(raw_matches: list[Match], *, group_by: str, set_characters: str,
              frame_width: int, frame_height: int,
              scorers: Sequence[Scorer] = DEFAULT_SCORERS,
              combos: Sequence[FlagCombo] | None = None,
              top_structure: TopBandStructure | None = None,
              ingame_tag_frac: float = INGAME_TAG_BAND_FRAC_DEFAULT,
              max_game_count: int = DEFAULT_MAX_GAME_COUNT) -> list[SweepRun]:
    """Enrich `raw_matches` under every flag combination, score each run, and return
    them ranked best-first (rank_runs, then the four post-ranking corrections: see
    promote_at_bottom_on_heavy_fallback, demote_at_bottom_on_light_fallback,
    demote_at_bottom_on_top_structure, and prefer_overlay_reader_among_non_at_bottom).
    `raw_matches` are never mutated: each combo
    enriches shallow per-match copies (enrich_match only assigns top-level keys and
    treats frames as read-only), and the winner's matches are exactly what a plain
    extract run with those flags would produce: same functions, same inputs, including
    the persistent-graphics scan, which is flag-independent and therefore run once.

    `top_structure` lets a caller that already ran measure_top_band_structure over the
    same matches pass the result through instead
    of having it measured a second time here; left None it is measured as usual (the
    same precomputed-input pattern as enrich_matches' `banners`).

    `group_by`/`set_characters` drive the set grouping criterion 1 counts, so it scores
    the grouping the Review screen will actually show; `max_game_count` is the run's game's
    highest game count, for that grouping's count fold. If the sweep ever gets slow, the
    isolated fix is caching build_vote_views per (filter_banners, no_overlay_filter)
    4 view variants serve all 16 combos: inside this loop and _voted_tag_y_samples."""
    if combos is None:
        combos = FlagCombo.all_combos()
    if not combos:
        raise ValueError("combos must not be empty")
    banners = (find_persistent_graphics(raw_matches)
               if any(c.filter_banners for c in combos) else [])
    sweeps_at_bottom = any(c.overlay_at_bottom for c in combos)
    row = learn_tag_row(raw_matches, frame_width, frame_height) if sweeps_at_bottom else None
    if top_structure is None and sweeps_at_bottom:
        top_structure = measure_top_band_structure(raw_matches, frame_width, frame_height)
    ctx = SweepContext(frame_width=frame_width, frame_height=frame_height, banners=banners,
                       tag_row_y=row.y if row is not None else None)

    runs: list[SweepRun] = []
    for combo in combos:
        started = time.perf_counter()
        copies = [m.copy() for m in raw_matches]
        # Enrichment and grouping print per-match voting notes; capture them (rather than
        # plumb a quiet= flag through half a dozen functions) so 16 runs don't print 16×
        # the usual output: extract_matches replays the winner's notes verbatim.
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            matches = enrich_matches(copies, filter_overlays=not combo.no_overlay_filter,
                                     filter_banners=combo.filter_banners,
                                     frame_width=frame_width, frame_height=frame_height,
                                     overlay_at_bottom=combo.overlay_at_bottom,
                                     overlay_stacked=combo.overlay_stacked,
                                     banners=banners if combo.filter_banners else None,
                                     ingame_tag_frac=ingame_tag_frac)
            sets = group_into_sets(matches, group_by, set_characters=set_characters,
                                   overlay_at_bottom=combo.overlay_at_bottom,
                                   max_game_count=max_game_count)
        run = SweepRun(combo=combo, matches=matches, sets=sets, notes=buffer.getvalue())
        for scorer in scorers:
            run.scores[scorer.name] = scorer.score(run, ctx)
        run.elapsed_sec = time.perf_counter() - started
        runs.append(run)
    ranked = promote_at_bottom_on_heavy_fallback(rank_runs(runs, scorers))
    ranked = demote_at_bottom_on_light_fallback(ranked)
    ranked = demote_at_bottom_on_top_structure(ranked, top_structure)
    return prefer_overlay_reader_among_non_at_bottom(ranked)


# ── Reporting ──────────────────────────────────────────────────────────────────

def _format_score(value: float) -> str:
    """A score for table display: counts as integers, variances with enough places to
    compare, and the NO_TAG_VARIANCE sentinel as "inf"."""
    if math.isinf(value):
        return "inf"
    if value == int(value):
        return str(int(value))
    return f"{value:.6f}"


def format_ranking_table(runs: Sequence[SweepRun], scorers: Sequence[Scorer]) -> str:
    """The ranked sweep as an aligned text table (best first): rank, the combo's flags,
    one column per scorer, and the combo's wall time."""
    headers = ["rank", "flags", *(s.name for s in scorers), "sec"]
    rows = [[str(i + 1), run.combo.label(),
             *(_format_score(run.scores[s.name]) for s in scorers),
             f"{run.elapsed_sec:.2f}"]
            for i, run in enumerate(runs)]
    widths = [max(len(header), *(len(row[col]) for row in rows))
              for col, header in enumerate(headers)]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths))
             for row in [headers, *rows]]
    return "\n".join(lines)


def _run_entry(rank: int, run: SweepRun) -> dict:
    """One run as a JSON-serializable report entry. Non-finite scores (the
    NO_TAG_VARIANCE sentinel) become None: json.dump would emit them as the
    non-standard token Infinity otherwise. `promotion` is null except on a run one of
    the post-ranking corrections moved to the front (see the `promotion` field's
    docstring on SweepRun)."""
    return {
        "rank": rank,
        "label": run.combo.label(),
        "flags": asdict(run.combo),
        "cli_flags": run.combo.cli_flags(),
        "scores": {name: (value if math.isfinite(value) else None)
                   for name, value in run.scores.items()},
        "elapsed_sec": round(run.elapsed_sec, 3),
        "promotion": run.promotion,
    }


def build_report(runs: Sequence[SweepRun], scorers: Sequence[Scorer]) -> dict:
    """The persisted sweep report (SMART_TAGS_REPORT_NAME): the scorer rules, the winner
    and runner-up (each with the exact CLI flags to reproduce them manually), every
    run's scores, and the per-combo/total wall time. Accumulated across VODs, this is
    the data for tuning the rules: which flags ever win, and what the sweep costs."""
    entries = [_run_entry(i + 1, run) for i, run in enumerate(runs)]
    return {
        "scorers": [{"name": s.name, "description": s.description} for s in scorers],
        "winner": entries[0],
        "runner_up": entries[1] if len(entries) > 1 else None,
        "runs": entries,
        "total_elapsed_sec": round(sum(r.elapsed_sec for r in runs), 3),
    }


def report_entry_text(title: str, entry: dict) -> str:
    """One report entry ("Picked"/"Runner-up") as text for a GUI summary popup: the flag
    label plus each scorer's value ("-" for the no-tags variance sentinel, stored as null),
    plus the heavy-fallback promotion reason on a winner the correction moved to the front."""
    scores = ", ".join(f"{name} {value:g}" if isinstance(value, (int, float)) else f"{name} -"
                       for name, value in entry["scores"].items())
    text = f"{title}: {entry['label']}\n{scores}"
    if entry.get("promotion"):
        text += f"\n{entry['promotion']}"
    return text


def read_report_summary(report_path: Path) -> str | None:
    """The winner and runner-up of the sweep report at `report_path`, as the text of a GUI
    summary popup (see report_entry_text). None when the report is missing or malformed: the
    popup is informational, so its callers skip it rather than fail."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        lines = [report_entry_text("Picked", report["winner"])]
        if report.get("runner_up"):
            lines.append(report_entry_text("Runner-up", report["runner_up"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return "\n\n".join(lines)
