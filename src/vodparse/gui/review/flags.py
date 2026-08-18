"""Review-flag logic: should this match be flagged for a human look, and why; no Qt, no I/O.

Three guards, each answering with a short human-readable reason: the over-long and too-short
length checks (a replay swept in / a missed split, or a warm-up / false positive / over-tight
split) and the clock-consensus bad-start checks (the stored start disagrees with what the
match's own timer readings imply). ``flag_reason`` runs them in priority order under the
user's ``FlagSettings`` toggles. Everything is pure functions over ``vodparse.schema.Match``
(never ReviewClip).

The QSettings toggle keys/defaults and the amber/yellow row tints live here next to the logic
they parameterize: vodparse.gui.dialogs' SettingsDialog and tools/vod_grabber.py write the keys;
screen.py reads them and paints the tints. Split out of model.py (the
clip/bounds domain logic).
"""

import statistics
from dataclasses import dataclass

from vodparse.gui.review.model import hms
from vodparse.schema import Match

# Flag matches longer than a full game for human review: usually a replay swept into the timeline
# or a missed split. A real unpaused game maxes at 7:00 (420s); the default leaves headroom for a
# paused game. The Review screen only flags these (never auto-excludes); both the toggle and the
# threshold are user settings (keys below), shared by vodparse.gui.dialogs (the SettingsDialog) and screen.py.
SETTING_FLAG_LONG_MATCHES = "review/flag_long_matches"  # bool, default True
SETTING_MAX_MATCH_SEC = "review/max_match_sec"  # seconds, default below
MAX_MATCH_SEC_DEFAULT = 480.0  # 8:00

# The mirror guard: flag matches *shorter* than a full game, a warm-up "hand-warmer", a false
# positive, or a split clipped too tight. Same flag-only treatment and same shared settings story.
SETTING_FLAG_SHORT_MATCHES = "review/flag_short_matches"  # bool, default True
SETTING_MIN_MATCH_SEC = "review/min_match_sec"  # seconds, default below
MIN_MATCH_SEC_DEFAULT = 60.0  # 1:00

# A third guard: flag matches whose clock readings don't back the stored start. Every clean clock read
# back-dates to a start (the clock falls 1:1 with wall time: start = read_ts - (FULL - clock)), so on a
# real game they all agree. Comparing the consensus (largest agreeing cluster of reads) to the stored
# start catches three failures, split across two independent toggles because their precision differs:
#   * SETTING_FLAG_WEIRD_TIMER: the severe, high-precision pair (amber): no two readings agree (a
#     replay/pause/misreads, not a clean countdown), or the game began more than START_CONSENSUS_TOL_DEFAULT
#     *later* than the marker (junk/an aborted start in front of the real game: the clip starts too early).
#   * SETTING_FLAG_LONG_LEADIN: the softer, noisier case (yellow): the start is back-dated further before
#     the first reading than the scan could have skipped (a replay or odd segment on its own, but also,
#     sometimes, a real game whose opening just wasn't sampled), tunable via SETTING_LONG_LEADIN_INTERVALS.
# Cadence-independent: it reads the spacing of the actual readings, so a deep-but-clean first read never
# trips it. Flag-only (never excludes/re-splits). When a match is both, the severe pair takes precedence.
SETTING_FLAG_WEIRD_TIMER = "review/flag_weird_timer"  # bool, default True (key kept for back-compat)
SETTING_FLAG_LONG_LEADIN = "review/flag_long_leadin"  # bool, default True
SETTING_LONG_LEADIN_INTERVALS = "review/long_leadin_intervals"  # float (scan steps), default below
START_CONSENSUS_TOL_DEFAULT = 5.0  # seconds the clock reads may disagree (OCR jitter <1.5s, 7:00 hold ~3s)
LONG_LEADIN_INTERVALS_DEFAULT = 2.0  # flag a start back-dated more than this many scan steps before the first read

# The 7:00 competitive clock. A local mirror of vodparse.match_extraction.FULL_GAME_SECONDS, kept here
# so this Qt-free domain layer needn't import the extraction module (and its heavier deps).
FULL_GAME_SECONDS = 420.0

# Tints the Review screen paints on a flagged row (kept here so the literals are shared by screen.py,
# which picks one per theme and builds the QColors, and the tests that assert them). Each tint has a
# light and a *_DARK variant: the deep base values read on a pale background, the *_DARK siblings are
# brightened for a dark one (same family as vodparse.gui.common.warning_color()); screen._style_clip_row
# chooses via vodparse.gui.common.is_dark_theme(). The base (light) values are also what the flag tests assert,
# since the offscreen test palette is light.
# REVIEW_FLAG_COLOR is amber: the more severe flags (over-long / too-short / a start that's likely wrong).
# REVIEW_LEADIN_COLOR is a softer yellow for a long lead-in: a real-but-unobserved opening that's worth a
# glance, not as off as a mis-sized match. Same red channel as the amber, more green, so it reads as a
# yellower sibling of the same warning family.
REVIEW_FLAG_COLOR = "#c86e00"
REVIEW_LEADIN_COLOR = "#c8a200"
REVIEW_FLAG_COLOR_DARK = "#ffb454"     # brighter amber for a dark background
REVIEW_LEADIN_COLOR_DARK = "#e6c34a"   # brighter yellow sibling


def mmss(seconds: float) -> str:
    """Seconds → MM:SS (no hours): for short spans where an hours field would only ever read 00:."""
    s = int(round(seconds))
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"


def long_match_reason(match: Match, threshold_sec: float) -> str | None:
    """If `match` runs longer than `threshold_sec`, a short reason to flag it for review (it may be
    a replay swept in or a missed split); otherwise None. Length is taken live from the start/end
    markers so it tracks boundary edits rather than relying on the stored duration_sec."""
    length = match["end"]["timestamp_sec"] - match["start"]["timestamp_sec"]
    if length <= threshold_sec:
        return None
    return (f"This game runs {mmss(length)}, over the set {mmss(threshold_sec)} limit for a full game. "
            f"Possibly a replay or a missed split. Check the boundaries.")


def short_match_reason(match: Match, threshold_sec: float) -> str | None:
    """If `match` runs shorter than `threshold_sec`, a short reason to flag it for review (it may be
    a warm-up game, a false positive, or a split clipped too tight); otherwise None. Length is taken
    live from the start/end markers, so a boundary edit that lengthens the match clears the flag."""
    length = match["end"]["timestamp_sec"] - match["start"]["timestamp_sec"]
    if length >= threshold_sec:
        return None
    return (f"This game runs only {mmss(length)}, which is under the set {mmss(threshold_sec)} minimum for a full game. "
            f"Possibly a handwarmer or a false positive. Check the boundaries.")


def first_timer_seconds(match: Match) -> float | None:
    """The game clock (seconds remaining) at the match's first frame that carried a reading, or None
    if no frame had one. This is the *as-detected* start clock, taken from the frames, not the
    editable start marker, so it measures how late detection first locked onto the game, regardless
    of any later boundary edit."""
    for frame in match.get("frames", ()):
        timer = frame.get("timer")
        if timer is not None:
            return timer["total_seconds"]
    return None


def _clean_reads(match: Match) -> list[tuple[float, float]]:
    """(timestamp, clock_seconds) for every frame carrying a clean timer reading, in order. The implied
    start of each is timestamp - (FULL_GAME_SECONDS - clock); the clock falls 1:1 with wall time, so on
    a real game every read implies the same start (within OCR jitter)."""
    out = []
    for frame in match.get("frames", ()):
        timer = frame.get("timer")
        if timer is not None and timer.get("total_seconds") is not None:
            out.append((frame["timestamp_sec"], timer["total_seconds"]))
    return out


def _consensus_cluster(implied: list[float], tolerance_sec: float) -> list[int]:
    """Indices of the largest cluster of `implied` starts that agree to within `tolerance_sec`. Greedy
    1D clustering (reads per match are few, so O(n^2) is fine) so a lone outlier read (an aborted
    attempt's clock, a misread, a replay) is left out of the agreed-on start rather than dragging it."""
    best: list[int] = []
    for value in implied:
        members = [i for i, u in enumerate(implied) if abs(u - value) <= tolerance_sec]
        if len(members) > len(best):
            best = members
    return best


def _read_cadence(reads: list[tuple[float, float]]) -> float:
    """Estimate the scan cadence from the spacing of a match's readings: the smallest gap between
    consecutive reads. A skipped sample only ever widens a gap, so the minimum is the sampling step."""
    gaps = [b[0] - a[0] for a, b in zip(reads, reads[1:]) if b[0] > a[0]]
    return min(gaps) if gaps else 0.0


def bad_start(match: Match, tolerance_sec: float,
              lead_in_intervals: float = LONG_LEADIN_INTERVALS_DEFAULT,
              suggest_trim: bool = True) -> tuple[str, str] | None:
    """If a match's clock readings don't back its stored start, `(kind, reason)` to flag it for review;
    otherwise None. `kind` is one of 'no_consensus', 'start_shift', 'long_leadin' so the caller can pick
    a severity/colour: 'long_leadin' is the soft case (a real-but-unobserved opening, worth a glance),
    the other two mean the start is likely wrong.

    `suggest_trim` controls only the start_shift wording: when True (the clip's first game, whose start
    is the editable trim point) it suggests where to trim; when False (a later game in a set, where the
    clip is exported continuously and there's no mid-set trim point) it points at a missed split instead.

    Every clean clock read back-dates to a start (see _clean_reads), so on a real game they agree. We
    compare the stored start marker to the consensus (largest agreeing cluster of reads) and catch three
    failures (cadence-independent: it reads the actual reading spacing, so a deep-but-clean first read
    never trips it; needs >=2 readings, else None):

      * no two readings agree on a start: a replay, a pause, or misreads, not a clean countdown;
      * the game began more than `tolerance_sec` *later* than the marker: junk (an aborted start /
        replay) in front of the real game, so the clip starts too early (suggests the trim point);
      * the start is back-dated further before the first reading than the scan could have skipped (over
        `lead_in_intervals` sampling steps), a replay or odd segment standing on its own, whose clock
        back-dates over footage that was never a game.

    Compares against the editable start marker, so trimming the start past the junk clears the first two."""
    reads = _clean_reads(match)
    if len(reads) < 2:
        return None
    implied = [ts - (FULL_GAME_SECONDS - clock) for ts, clock in reads]
    cluster = _consensus_cluster(implied, tolerance_sec)

    if len(cluster) < 2:
        return ("no_consensus",
                "The app seemed to miss the start of this game. Check this clip.")

    consensus = statistics.median(implied[i] for i in cluster)
    shift = consensus - match["start"]["timestamp_sec"]
    if shift > tolerance_sec:
        lede = f"This game's clock readings are off by about {mmss(shift)}. Likely a handwarmer, pause, or replay."
        advice = (f" Suggested start trim: {hms(consensus)}." if suggest_trim
                  else " Check this clip.")
        return ("start_shift", lede + advice)

    cadence = _read_cadence(reads)
    lead_in = min(reads[i][0] for i in cluster) - consensus
    if cadence > 0 and lead_in > lead_in_intervals * cadence:
        # Deliberately no magnitude: the back-dated gap is measured from the synthesized 7:00 start, so
        # it folds in the unseen pre-clock stretch (clock animating in / a late cut to the game) and
        # reads as "off by a minute" when the visible game only looks seconds late. Just flag the kind.
        return ("long_leadin",
                "The program missed the game's opening, so the start time may be off.")
    return None


def bad_start_reason(match: Match, tolerance_sec: float,
                     lead_in_intervals: float = LONG_LEADIN_INTERVALS_DEFAULT) -> str | None:
    """The reason string from `bad_start` (see there), or None: for callers/tests that only need the
    human-readable note, not the kind."""
    result = bad_start(match, tolerance_sec, lead_in_intervals)
    return result[1] if result else None


@dataclass(frozen=True)
class FlagSettings:
    """The Review screen's flag toggles and thresholds, read once per render from QSettings
    (see ReviewScreen._load_flag_settings) and passed to flag_reason."""
    flag_long_enabled: bool = True
    max_match_sec: float = MAX_MATCH_SEC_DEFAULT
    flag_short_enabled: bool = True
    min_match_sec: float = MIN_MATCH_SEC_DEFAULT
    flag_weird_timer_enabled: bool = True
    flag_long_leadin_enabled: bool = True
    long_leadin_intervals: float = LONG_LEADIN_INTERVALS_DEFAULT


def flag_reason(match: Match | None, is_first: bool,
                settings: FlagSettings) -> tuple[str, str] | None:
    """(reason, severity) for the first review-flag guard that fires on `match`: over-long,
    then too-short (mutually exclusive by length), then the bad-start checks: each only when
    its toggle is on; None when nothing fires. Severity is 'flag' (amber) except a long
    lead-in, which is the softer 'leadin' (yellow). The two bad-start kinds have separate
    toggles: a long lead-in is gated on the long-lead-in flag, the severe pair on the
    bad-start flag. bad_start resolves a both-match to the severe kind, so with the bad-start
    flag off such a match stays hidden even if the long-lead-in flag is on. is_first (the
    clip's first game) gates the start-shift trim suggestion: trimming only makes sense at
    the clip start, never mid-set."""
    if not match:
        return None
    if settings.flag_long_enabled:
        reason = long_match_reason(match, settings.max_match_sec)
        if reason:
            return reason, "flag"
    if settings.flag_short_enabled:
        reason = short_match_reason(match, settings.min_match_sec)
        if reason:
            return reason, "flag"
    if settings.flag_weird_timer_enabled or settings.flag_long_leadin_enabled:
        bs = bad_start(match, START_CONSENSUS_TOL_DEFAULT, settings.long_leadin_intervals,
                       suggest_trim=is_first)
        if bs:
            kind, reason = bs
            if kind == "long_leadin":
                if settings.flag_long_leadin_enabled:
                    return reason, "leadin"
            elif settings.flag_weird_timer_enabled:
                return reason, "flag"
    return None
