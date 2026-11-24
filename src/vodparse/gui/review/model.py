"""Pure clip/match domain logic for the Review screen: no Qt, no I/O.

`screen.py` is the Qt widget that orchestrates editing; this module holds the
computations it leans on so they can be read and unit-tested in isolation: turning a match
list into the clip rows the screen edits, seeding/clamping export boundaries, the marker
math behind Split match, the undo snapshot copy, and the read-only "what would propagate"
target computations.

A *clip* is the unit the screen exports (see screen.py for the full story). In memory it
is a `ReviewClip` (below): the exported ``schema.Clip`` keys plus the editing-only extras.

The "should this match be flagged for review, and why" logic (the length/bad-start guards,
their QSettings keys, and the flag row tints) lives in flags.py; the Tools-menu
tag-text fix-ups (Fix capitalization, Split sponsor from tag) live in tag_fixes.py.
"""

from collections.abc import Callable, Iterator, Sequence
from typing import Literal, NotRequired, TypedDict, cast

from vodparse.grouping import (merge_set_characters, overlay_reads_full_pairs, similar,
                               voted_sponsors, voted_tags, group_into_sets, grouping_tags)
from vodparse.schema import Clip, Match, Marker, ReviewedData
from vodparse.text_utils import title_round


# ── Constants & settings ──────────────────────────────────────────────────────

# 1v1 overlay: two player slots. Every per-player tag/sponsor/character list is indexed 0/1.
NUM_PLAYERS = 2

# Match extract_matches' argparse defaults; used only to seed editable boundaries.
START_BUFFER = 15.0
END_BUFFER = 20.0

# A split match is rejected if either resulting half would be shorter than this (seconds).
MIN_SPLIT_MATCH_SEC = 20.0

# Marks (tints) each set/match whose identity tags fell back to the in-game nameplates because the
# tournament-overlay tags couldn't be read: auto grouping only; a forced mode never falls back. Purely
# a review-screen cue derived at display time, so it's a marker rather than a flag, but it rides the same
# settings story: one shared key read by vodparse.gui.app's SettingsDialog and screen.py, default on.
SETTING_MARK_INGAME_FALLBACK = "review/mark_ingame_fallback"  # bool, default True

# How far into a game the playhead lands when the user clicks a game row inside a set. Production
# often cuts to the game a second or two after the detected start, so the first frame tends to show
# the transition or the previous scene; a small offset lands on the game itself. Game rows only: a
# set row still opens on its first frame. User setting shared by SettingsDialog and screen.py;
# the default of 0 keeps the long-standing "land on the start" behaviour until the user tunes it.
SETTING_MATCH_SEEK_OFFSET_SEC = "review/match_seek_offset_sec"  # seconds, default below
MATCH_SEEK_OFFSET_SEC_DEFAULT = 0.0
MATCH_SEEK_OFFSET_SEC_MAX = 30.0   # the Settings spinbox ceiling; past this it's no longer a "lag"

# The in-game-fallback marker tint (SETTING_MARK_INGAME_FALLBACK): a teal that reads as informational,
# deliberately outside the amber/yellow warning family above so a fallback never looks like a flag. Same
# light + *_DARK convention: the base reads on a pale background, the *_DARK sibling on a dark one.
REVIEW_FALLBACK_COLOR = "#2f7d8a"
REVIEW_FALLBACK_COLOR_DARK = "#5fc6d6"   # brighter teal for a dark background

# The missing-tag marker tint: a deep red, deliberately outside the amber/yellow flag family and the
# teal fallback marker: this one means "action required" (a set that can't be identified until the
# user fills in a tag), not merely "worth a glance." Same light + *_DARK convention as the tints above.
REVIEW_MISSING_TAG_COLOR = "#c0392b"        # deep red, legible on a pale background
REVIEW_MISSING_TAG_COLOR_DARK = "#ff6b6b"   # brighter red for a dark background

# The single-game-set marker: unlike every tint above (which recolor the *text*), this one paints the
# whole set row's *background*: a muted red band flagging a set that holds just one game (a merge
# candidate or stray warm-up). Kept as a background so it never collides with the deep-red missing-tag
# *foreground*, and so it can coexist with an amber flag on the same row. Muted enough that the default
# row text stays legible on top; same light + *_DARK convention (pale wash for a light background, a
# dark maroon for a dark one).
REVIEW_SINGLE_GAME_BG_COLOR = "#f6d5d2"       # pale red wash under dark default text
REVIEW_SINGLE_GAME_BG_COLOR_DARK = "#5a2320"  # dark maroon under light default text

# One source of truth for the fallback note, reused by the row tooltip and the tag-field tooltips.
INGAME_FALLBACK_TOOLTIP = (
    "The program couldn't read from the tournament overlay. Using ingame tags instead.")

# Tooltip for the single-game-set background band (see is_single_game_set / REVIEW_SINGLE_GAME_BG_COLOR).
SINGLE_GAME_SET_TOOLTIP = (
    "Sets don't usually have one game. Merge this into an adjacent set or exclude it.")

# Last-used values of the Split-sponsor and Fix-capitalization options dialogs (Tools ▸ Split
# sponsor from tag… / Fix capitalization across all games…), so each popup reopens the way the
# user left it. NOT under the reserved "general/" group (see the QSettings gotcha in
# gui/settings.py); these are options, and each dialog's Apply doubles as the confirm.
SETTING_SPLIT_SPONSOR_MIN_LEN = "review/split_sponsor_min_len"      # int; sponsor token min length
SETTING_SPLIT_SPONSOR_MAX_LEN = "review/split_sponsor_max_len"      # int; sponsor token max length
SETTING_SPLIT_SPONSOR_MAX_WORDS = "review/split_sponsor_max_words"  # int; max leading words to peel
SETTING_SPLIT_SPONSOR_ALL_CAPS = "review/split_sponsor_all_caps"    # bool; only peel fully-caps words
SETTING_FIX_CAPS_MODE = "review/fix_caps_mode"                      # str: "title" | "sentence" | "lower"

# Final Review dialog column toggles ("Review and export"): the sponsor and round columns
# are noise for most passes, tags and characters are what get eyeballed, so they start
# hidden; each toggle persists like the options above so the table reopens the way the
# user left it.
SETTING_FINAL_REVIEW_SHOW_SPONSORS = "review/final_review_show_sponsors"  # bool; sponsor columns visible
SETTING_FINAL_REVIEW_SHOW_ROUND = "review/final_review_show_round"        # bool; round column visible


# ── Clip / meta types ─────────────────────────────────────────────────────────

class ReviewClip(Clip):
    """The Review screen's in-memory clip shape: every key of the exported
    ``vodparse.schema.Clip`` plus editing-only extras that never reach reviewed.json
    (``build_reviewed_data`` projects a clip back down to the ``Clip`` keys before export):

    - ``detected_tags`` / ``detected_sponsors``: the as-detected originals, untouched by
      field edits; Propagate-tags matches other sets by these (they hold the OCR reading).
    - ``_matches``: the member ``Match`` dicts (a set's games, or the one match in match mode).
    - ``exclude_from_export``: the review-only "don't export this set" flag, set lazily by
      the session's exclude ops (most clips never carry it) and stripped by
      ``build_reviewed_data``'s projection before export, hence ``NotRequired``.
    - ``start_marked`` / ``end_marked``: the review-only "this bound was hand-marked, don't
      re-derive it" flags, set lazily by ``set_bound`` when the user stamps a Mark start/end
      (most clips never carry them). Unlike the tag/character edits, a marked ``start_sec`` /
      ``end_sec`` can't be told from a derived one by diffing against a from-matches rebuild
      (the live value carries seam/clamp neighbor adjustments the isolated rebuild lacks), so
      the mark is tracked by an explicit flag instead, mirroring ``exclude_from_export``,
      which likewise has no derived baseline. A restructure re-derives every bound, then
      re-stamps the marked ones per this flag (see the bound-routing helpers below).

    One ``Clip`` key goes the other way: ``first_game_sec`` is never stored on a
    ``ReviewClip``. ``build_reviewed_data`` derives it at export, from the first game the
    clip includes, so it can't go stale when a split or merge changes the set's games.

    This is the shape the whole review pipeline is typed against: the constructors
    (make_clip, clip_from_draft, copy_clips) set every non-NotRequired key, and every
    transform/reader takes and returns it. The per-player list fields keep ``Clip``'s
    ``list[str | None]`` item type, a slot whose tag was never read is None, and the
    producers that rewrite them (``pair`` and friends below) are typed to match.
    """
    detected_tags: list[str | None]
    detected_sponsors: list[str | None]
    _matches: list[Match]
    exclude_from_export: NotRequired[bool]
    start_marked: NotRequired[bool]
    end_marked: NotRequired[bool]


class ReviewMeta(TypedDict, total=False):
    """The session's ``meta`` dict: the ``ReviewedData`` identity/config keys minus
    ``clips`` (``video_path``/``stem``/``mode``/``tournament_name``/``group_by``).
    ``total=False`` because the screen holds an empty ``{}`` meta (or just
    ``{"stem": ...}`` after a failed draft resume) before a VOD is loaded."""
    video_path: str
    stem: str
    mode: str
    tournament_name: str
    group_by: str


# ── Small pure helpers ────────────────────────────────────────────────────────

def hms(seconds: float) -> str:
    """Seconds → HH:MM:SS."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def match_seek_target(match_start: float, match_end: float, offset_sec: float) -> float:
    """Where the playhead lands when a game row is clicked: ``offset_sec`` into the game (see
    SETTING_MATCH_SEEK_OFFSET_SEC). A game shorter than the offset would otherwise land on or past
    its own end, i.e. on the next game, so such a game opens on its start instead."""
    target = match_start + offset_sec
    if target >= match_end:
        return match_start
    return target


def seed_bounds(start_marker: Marker, end_marker: Marker) -> tuple[float, float]:
    """Apply the export buffers to raw markers: a lead-in before the start and a tail after the
    end. Contiguous split seams have their buffer collapsed afterwards by
    apply_contextual_buffers, and the tail is capped against the next clip by clamp_clip_ends."""
    return (max(0.0, start_marker["timestamp_sec"] - START_BUFFER),
            end_marker["timestamp_sec"] + END_BUFFER)


def pair(a: str, b: str) -> list[str | None]:
    """Two text values → a tag list, or [] when both are blank. Values are always ``str``;
    the return type matches the voted-tag slots (``list[str | None]``) the result is
    assigned into."""
    vals: list[str | None] = [a.strip(), b.strip()]
    return vals if any(vals) else []


def set_player_slot(values: list[str | None] | None, j: int, value: str) -> list[str | None]:
    """A copy of `values` (or [] when None) with slot `j` set to `value`, padded with ""
    up to that slot if the list is short. Used by Propagate tags to stamp one player's
    slot without disturbing the other player's."""
    vals: list[str | None] = list(values or [])
    vals += [""] * (j + 1 - len(vals))
    vals[j] = value
    return vals


def nth_or_blank(lst: Sequence[str | None] | None, i: int) -> str:
    """Element i of lst as a string, or '' when the list is None/short or the element
    itself is None. These lists are list[str | None], so a present None is coerced to ''
    too; callers get a real str and can index/strip/setText it without guarding."""
    lst = lst or []
    return (lst[i] if i < len(lst) else "") or ""


def marker_fps(*markers: Marker) -> float:
    """Frames-per-second estimated from the first marker carrying both a frame number and a
    nonzero timestamp (frame_number / timestamp_sec: the same arithmetic the extractor's
    _fps_of uses). 0.0 when none does, which only happens for a match resumed from a draft
    saved before markers carried frame numbers."""
    for m in markers:
        frame_number, ts = m.get("frame_number"), m.get("timestamp_sec")
        if frame_number and ts:
            return frame_number / ts
    return 0.0


def synth_marker(sec: float, fps: float) -> Marker:
    """A Review-screen-created start/end marker, in the same shape as the extractor's
    synthesized markers ({timestamp_sec, frame_number, synthesized}) so every marker in the
    system carries a frame number. With no usable fps (see marker_fps) the frame number
    degrades to 0."""
    return {"timestamp_sec": round(sec, 3),
            "frame_number": int(round(sec * fps)),
            "synthesized": True}


def display_tags(clip: ReviewClip) -> tuple[str, str]:
    """The set's player tags for the row label. Taken straight from grouping_tags, which is
    already the mode-resolved identity: under "auto" it folds overlay+in-game (the fallback
    lives in grouping_tags()), while a forced mode carries only its chosen source. Falling back
    to the raw tournament_tags/ingame_tags columns here would re-introduce the cross-source mix
    that the forced modes exist to avoid, so a blank chosen source shows "?", not the other
    source's tags."""
    tags = clip.get("grouping_tags") or []
    return nth_or_blank(tags, 0) or "?", nth_or_blank(tags, 1) or "?"


def missing_tag_players(clip: ReviewClip) -> list[int]:
    """Which players (1-based) still need a real tag: the `grouping_tags` slot is blank or
    whitespace-only. The literal tag "None" is deliberately NOT treated as missing: real players
    use it, and the hallucinated "None" OCR read this once guarded against appears nowhere in the
    analyzed VODs today (if it ever returns, filter it in the pipeline's tag vote, not here).
    Reads the same `grouping_tags` source the row label and Tag fields use, so this never
    disagrees with what's on screen. `[]` when both tags look real."""
    tags = clip.get("grouping_tags") or []
    return [i + 1 for i in range(NUM_PLAYERS) if not nth_or_blank(tags, i).strip()]


def missing_tag_message(players: list[int]) -> str:
    """The status-line/tooltip prompt for `missing_tag_players`' result, or "" when both tags are
    real: the empty string doubles as the presence test for callers."""
    if not players:
        return ""
    if players == [1, 2]:
        return "Give players 1 and 2 a tag"
    return f"Give player {players[0]} a tag"


def export_status_text(done: int, total: int, unit: str) -> str:
    """The caption shown above the export progress bar while the export subprocess runs,
    e.g. "Exporting games… 3 / 12". `unit` is the already-pluralized clip unit
    ("games"/"sets") for the review mode."""
    return f"Exporting {unit}… {done} / {total}"


def overlay_proven_for_clips(clips: list[ReviewClip]) -> bool:
    """The review screen's trust_half_overlay: whether any underlying match voted a full
    overlay pair (grouping.overlay_reads_full_pairs), taken from the clips' member matches
    (in match mode a clip is its own single match)."""
    matches = [m for c in clips for m in (c.get("_matches") or [cast("Match", c)])]
    return overlay_reads_full_pairs(matches)


def used_ingame_fallback(clip: ReviewClip, mode: str, group_by: str, *,
                         trust_half_overlay: bool = True) -> bool:
    """True when a clip's identity tags came from the in-game nameplates because the
    tournament-overlay tags were unreadable, i.e. the "auto" fallback in grouping_tags()
    fired. Mirrors grouping_tags()'s auto rule so the cyan marking never disagrees with
    where the label actually came from.

    Match mode always resolves identity with "auto" (build_clips), so it can always fall back;
    set mode only under group_by == "auto". A forced mode ("overlay"/"ingame") never falls back,
    so it never marks. The overlay is "unreadable" when no overlay tag was voted at all, and
    ALSO when only a half read survives on a VOD whose overlay never voted a full pair
    (`trust_half_overlay` False, from overlay_proven_for_clips): grouping_tags() hands such a
    clip's label to the nameplates, so the marking must follow."""
    if mode != "match" and group_by != "auto":
        return False
    if not clip.get("ingame_tags"):
        return False   # nothing to fall back to
    tags = clip.get("tournament_tags") or []
    if not any(tags):
        return True    # no overlay read at all
    is_full_pair = len(tags) == 2 and all(tags)
    return not is_full_pair and not trust_half_overlay


def is_single_game_set(clip: ReviewClip, mode: str) -> bool:
    """True for a set that holds exactly one game: a merge candidate or a stray warm-up worth a
    second look. A display-time cue recomputed from `_matches`, so merging two sets (their games
    combine) clears it for free on the next tree rebuild; nothing is stored. Always False in match
    mode, where a clip is one standalone match rather than a set of games."""
    return mode != "match" and len(clip.get("_matches", [])) == 1


# ── Clip construction ─────────────────────────────────────────────────────────

def make_clip(number: int, member_matches: list[Match], start: float, end: float, *,
              t_tags: list[str | None] | None, t_sponsors: list[str | None] | None,
              i_tags: list[str | None] | None, rnd: str | None, best_of: str | None,
              characters: list[str | None] | None = None,
              g_tags: list[str | None] | None = None) -> ReviewClip:
    """Build one `ReviewClip` from its member matches and the already-resolved
    tag/character/round values."""
    raw_chars = list(characters or [])
    chars = [(c or "") for c in raw_chars] + [""] * (NUM_PLAYERS - len(raw_chars))
    return {
        "clip_number": number,
        "match_numbers": [m["match_number"] for m in member_matches],
        "start_sec": start,
        "end_sec": end,
        "grouping_tags": list(g_tags or []),
        # As-built copies, untouched by field edits: Propagate tags matches other sets
        # by what was originally detected, since they still hold the OCR reading.
        "detected_tags": list(g_tags or []),
        "detected_sponsors": list(t_sponsors or []),
        "tournament_tags": list(t_tags or []),
        "tournament_sponsors": list(t_sponsors or []),
        "ingame_tags": list(i_tags or []),
        "characters": chars,
        "round": title_round(rnd or ""),
        "best_of": best_of or "",
        "_matches": member_matches,
    }


# ── Export projection ─────────────────────────────────────────────────────────

def included_index_span(clip: ReviewClip) -> tuple[int, int]:
    """Inclusive ``[0, n-1]`` index range of a set's member matches, i.e. the whole set. (Per-game
    first/last *trimming* was retired in favour of split-isolate, so every game of a non-excluded
    set is included; this stays a single helper so the bounds/CSV pipeline below is unchanged.)

    A clip always has at least one member match (make_clip builds it that way, and every restructure
    indexes matches[0]/matches[-1] upstream), so an empty set is a contract violation, not a case to
    handle here: assert it rather than silently return (0, -1), which would make included_span index
    an empty list and raise downstream."""
    n = len(clip["_matches"])
    assert n >= 1, "included_index_span requires a non-empty set"
    return 0, n - 1


def included_span(clip: ReviewClip) -> tuple[Match, Match]:
    """The first and last member matches of a set: ``(_matches[0], _matches[-1])``."""
    lo, hi = included_index_span(clip)
    ms = clip["_matches"]
    return ms[lo], ms[hi]


def included_match_numbers(clip: ReviewClip) -> list[int]:
    """The member match numbers that export: every game of the set (a non-excluded set exports all
    of its games; a whole excluded set is dropped upstream in build_reviewed_data)."""
    lo, hi = included_index_span(clip)
    return [m["match_number"] for m in clip["_matches"][lo:hi + 1]]


def build_reviewed_data(clips: list[ReviewClip], meta: ReviewMeta) -> ReviewedData:
    """Project the edited clips into the reviewed.json payload the exporter / chapters builder
    consume (vodparse.gui.app casts it to ReviewedData): `meta` carried through, with a `clips` list of the
    surviving sets, each reduced to the explicit key list below: exactly schema.Clip's fields,
    stripping the ReviewClip editing extras (detected_*, _matches, exclude_from_export).
    Pure: builds fresh dicts and never mutates `clips` or `meta` (the `{**meta, ...}` and the
    dict literal are shallow copies, which vodparse.gui.app's consumer relies on to mutate its result safely)."""
    def projected(clip: ReviewClip) -> Clip:
        # match_numbers lists every game of the set; a whole excluded set is dropped below (an
        # excluded single game lives in its own set, which is dropped that same way).
        return {
            "clip_number": clip["clip_number"],
            "match_numbers": included_match_numbers(clip),
            "start_sec": clip["start_sec"],
            "end_sec": clip["end_sec"],
            "grouping_tags": clip["grouping_tags"],
            "tournament_tags": clip["tournament_tags"],
            "tournament_sponsors": clip["tournament_sponsors"],
            "ingame_tags": clip["ingame_tags"],
            "characters": clip["characters"],
            "round": clip["round"],
            "best_of": clip["best_of"],
            "first_game_sec": included_span(clip)[0]["start"]["timestamp_sec"],
        }

    # Sets flagged "exclude from export" are dropped here, so they never reach reviewed.json
    # (no clip cut, no CSV row). Remaining sets keep their clip_number (numbering may gap).
    return cast(ReviewedData, {**meta, "clips": [projected(clip) for clip in clips
                                                  if not clip.get("exclude_from_export", False)]})


# ── Export-bounds pipeline ────────────────────────────────────────────────────

def seed_set_bounds(clip: ReviewClip) -> None:
    """(Re)seed one clip's export bounds from its first/last match, applying the buffers."""
    first, last = included_span(clip)
    clip["start_sec"], clip["end_sec"] = seed_bounds(first["start"], last["end"])


def apply_contextual_buffers(clips: list[ReviewClip]) -> None:
    """Collapse the buffer across a contiguous seam: where one clip's last match ends exactly
    where the next clip's first match starts (a split cut whose halves stayed back-to-back).
    Such seams get no tail/lead-in buffer so the same footage isn't shown twice. A split half
    moved away from its sibling no longer shares the timestamp, so its buffer is restored. The
    seam is each clip's first/last member match (every game of a non-excluded set is included)."""
    for i in range(len(clips) - 1):
        seam_end = included_span(clips[i])[1]["end"]["timestamp_sec"]
        seam_start = included_span(clips[i + 1])[0]["start"]["timestamp_sec"]
        if seam_end == seam_start:
            clips[i]["end_sec"] = seam_end
            clips[i + 1]["start_sec"] = seam_start


def clamp_clip_ends(clips: list[ReviewClip]) -> None:
    """Cap each clip's buffered end at the next clip's start so consecutive exports never
    overlap, but never below the clip's own last *included* detected end (so a clip can't lose
    real content). The final clip keeps its full tail."""
    for i in range(len(clips) - 1):
        raw_end = included_span(clips[i])[1]["end"]["timestamp_sec"]
        clips[i]["end_sec"] = max(raw_end, min(clips[i]["end_sec"], clips[i + 1]["start_sec"]))


def reseed_set_bounds(clips: list[ReviewClip]) -> None:
    """Re-seed every clip's bounds from its span, then re-collapse seams and clamp ends. Run after a
    restructure (merge/split/move/isolate) rebuilds clips: which sets are adjacent changes, so the
    contiguous-seam buffer collapse and end clamping have to be recomputed across the new clip set."""
    for clip in clips:
        seed_set_bounds(clip)
    apply_contextual_buffers(clips)
    clamp_clip_ends(clips)


# ── Building clip lists ───────────────────────────────────────────────────────

def build_clips(matches: list[Match], mode: str, group_by: str,
                set_characters: str) -> list[ReviewClip]:
    """Turn a just-extracted match list into clip rows. In match mode one match = one clip;
    otherwise matches are grouped into sets (one set = one clip, its games kept as members)."""
    clips: list[ReviewClip] = []
    if mode == "match":
        # Half overlay reads only name a clip on a VOD whose overlay provably reads as
        # full pairs somewhere (see grouping.grouping_tags).
        trust_half_overlay = overlay_reads_full_pairs(matches)
        for i, m in enumerate(matches, start=1):
            start, end = seed_bounds(m["start"], m["end"])
            # Single matches never pass through group_into_sets; derive their identity
            # tags with the auto preference (overlay first, in-game fallback).
            clips.append(make_clip(
                i, [m], start, end,
                t_tags=m.get("tournament_tags"), t_sponsors=m.get("tournament_sponsors"),
                i_tags=m.get("ingame_tags"),
                rnd=m.get("round"), best_of=m.get("best_of"),
                characters=m.get("characters"),
                g_tags=grouping_tags("auto", m.get("tournament_tags"), m.get("ingame_tags"),
                                     trust_half_overlay=trust_half_overlay)))
    else:
        for s in group_into_sets(matches, group_by, set_characters=set_characters):
            first, last = s["matches"][0], s["matches"][-1]
            start, end = seed_bounds(first["start"], last["end"])
            clips.append(make_clip(
                s["set_number"], s["matches"], start, end,
                t_tags=s.get("tournament_tags") or first.get("tournament_tags"),
                t_sponsors=s.get("tournament_sponsors") or first.get("tournament_sponsors"),
                i_tags=s.get("ingame_tags"),
                rnd=first.get("round"), best_of=first.get("best_of"),
                characters=s.get("characters"),
                g_tags=s.get("grouping_tags") or grouping_tags(
                    group_by, first.get("tournament_tags"), first.get("ingame_tags"))))
    apply_contextual_buffers(clips)
    clamp_clip_ends(clips)
    return clips


def build_clips_from_segments(segments: list[list[Match]], group_by: str,
                              set_characters: str) -> list[ReviewClip]:
    """Rebuild clips from an edited grouping (one match-group per clip), renumbered 1..N,
    re-voting each set's tags from its members and re-seeding bounds. The merge/split/move
    operations restructure the segment list and call this to turn it back into clips."""
    clips: list[ReviewClip] = []
    trust_half_overlay = overlay_reads_full_pairs(m for seg in segments for m in seg)
    for i, matches in enumerate(segments, start=1):
        first, last = matches[0], matches[-1]
        start, end = seed_bounds(first["start"], last["end"])
        t_tags = voted_tags(matches, "tournament_tags")
        i_tags = voted_tags(matches, "ingame_tags")
        clips.append(make_clip(
            i, matches, start, end,
            t_tags=t_tags,
            t_sponsors=voted_sponsors(matches),
            i_tags=i_tags,
            rnd=first.get("round"), best_of=first.get("best_of"),
            characters=merge_set_characters(matches, set_characters),
            g_tags=grouping_tags(group_by, t_tags, i_tags,
                                 trust_half_overlay=trust_half_overlay)))
    apply_contextual_buffers(clips)
    clamp_clip_ends(clips)
    return clips


# ── Manual-edit reapplication ─────────────────────────────────────────────────

# These Literal types let the reapply/manual-edit loops below index TypedDict clips with a
# variable field name.
PlayerField = Literal["grouping_tags", "tournament_sponsors", "characters", "ingame_tags"]
SetField = Literal["round", "best_of", "exclude_from_export"]

# Clip fields the detail panel (plus Swap characters and Exclude) let the user edit directly.
# build_clips_from_segments re-derives every one of them from the matches, so a restructure
# would silently revert typed corrections; reapply_manual_edits carries the edited ones back.
# The per-player fields hold one value per slot (player 0/1); their edits are tracked per slot
# so a merge can union player-1's edit from one set with player-2's from the other.
_PLAYER_FIELDS: tuple[PlayerField, ...] = (
    "grouping_tags", "tournament_sponsors", "characters", "ingame_tags")
_SET_FIELDS: tuple[SetField, ...] = ("round", "best_of", "exclude_from_export")

# The _SET_FIELDS members that diff against a derived baseline; exclude_from_export has
# none and is handled separately below the loop in _manual_edits.
_SET_TEXT_FIELDS: tuple[Literal["round", "best_of"], ...] = ("round", "best_of")


class _ManualEdits(TypedDict, total=False):
    """The per-clip diff ``_manual_edits`` computes and ``reapply_manual_edits`` replays after
    a restructure: per-player fields map slot index (0/1) to the edited value; set-wide fields
    map directly to their edited value."""
    grouping_tags: dict[int, str]
    tournament_sponsors: dict[int, str]
    characters: dict[int, str]
    ingame_tags: dict[int, str]
    round: str
    best_of: str
    exclude_from_export: bool


def _manual_edits(clip: ReviewClip, group_by: str, set_characters: str) -> _ManualEdits:
    """What the user changed on `clip`, measured against what rebuilding it from its own matches
    would re-derive. Per-player fields map to ``{slot: value}`` for the slots that differ (so a
    blanked tag counts, and each player is carried independently); set-wide fields map to their
    value. ``exclude_from_export`` has no derived baseline (build_clips_from_segments never sets
    it), so flagging a set out of the export counts as an edit on its own."""
    base = build_clips_from_segments([clip["_matches"]], group_by, set_characters)[0]
    edits: _ManualEdits = {}
    for field in _PLAYER_FIELDS:
        # Compare normalized: a slot the voting keeps as None reads as '' once any field edit
        # rewrites the list via _pair, and that '' vs None gap is not a manual blank: treating
        # it as one would let an edited set's phantom blanks strip a sponsor/tag the rebuild
        # correctly re-voted from another set on a merge.
        slots = {j: nth_or_blank(clip.get(field), j) for j in range(NUM_PLAYERS)
                 if nth_or_blank(clip.get(field), j) != nth_or_blank(base.get(field), j)}
        if slots:
            edits[field] = slots
    for set_field in _SET_TEXT_FIELDS:
        if clip.get(set_field) != base.get(set_field):
            edits[set_field] = clip.get(set_field)
    # exclude_from_export has no derived baseline, the rebuild never sets it, so a truthy value is
    # itself the edit to carry.
    if clip.get("exclude_from_export"):
        edits["exclude_from_export"] = clip["exclude_from_export"]
    return edits


def reapply_manual_edits(fresh_clips: list[ReviewClip], prior_clips: list[ReviewClip],
                         inherit: list[list[int]], group_by: str,
                         set_characters: str) -> None:
    """Carry manual field edits from the pre-restructure clips onto the freshly rebuilt ones,
    in place, so merge/split/move never discard typed tags/characters/rounds/sponsors or a
    set's exclude flag. build_clips_from_segments re-derives every field from the matches for
    *every* clip, including the sets the operation never touched, which is why an edit far
    from the edited set was being lost too.

    ``inherit[k]`` lists the prior-clip indices fresh clip k continues, primary first: a split's
    two halves both list their parent set; a merge lists both source sets; a move's untouched
    (and source/dest) sets each list themselves. Per-player edits apply slot by slot (primary
    first per slot, so a merge unions both sets' edited players); set-wide edits take the first
    listed prior that changed them. Anything no listed prior edited keeps the rebuild's fresh
    re-vote (so merging two unedited sets still re-votes their combined tags as before)."""
    needed = {p for priors in inherit for p in priors}
    edits = {p: _manual_edits(prior_clips[p], group_by, set_characters) for p in needed}
    for fresh, priors in zip(fresh_clips, inherit):
        for field in _PLAYER_FIELDS:
            slots: dict[int, str] = {}
            for p in priors:
                for j, value in edits[p].get(field, {}).items():
                    slots.setdefault(j, value)   # primary-first precedence, per slot
            if slots:
                result = list(fresh.get(field) or [])
                result += [""] * (max(slots) + 1 - len(result))
                for j, value in slots.items():
                    result[j] = value
                fresh[field] = result
        for set_field in _SET_FIELDS:
            for p in priors:
                if set_field in edits[p]:
                    fresh[set_field] = edits[p][set_field]
                    break


# ── Manual bound-mark reapplication ───────────────────────────────────────────

# A hand-marked start/end (see set_bound + ReviewClip.start_marked/end_marked) is one value
# per set, tied to the set's edge: the start belongs to the first game, the end to the last.
# When a restructure divides a set (split, exclude-isolate) or combines sets (merge), the mark
# must follow the resulting set that still owns that edge. `start_src`/`end_src` express that
# routing per fresh clip: the prior-clip index whose start (resp. end) mark it inherits, or
# None to re-derive that edge from the matches. They parallel `inherit` (used for field edits)
# but carry a single index per edge instead of a primary-first list.
BoundMark = tuple[int, Literal["start_sec", "end_sec"], float]


def default_bound_src(inherit: list[list[int]]) -> tuple[list[int | None], list[int | None]]:
    """The default `start_src`/`end_src` for a restructure, derived from its `inherit` map: a
    fresh clip's start edge continues its primary prior (``inherit[k][0]``) and its end edge the
    last listed prior (``inherit[k][-1]``). This is already correct for identity ops and merge
    (the merged slot lists both sources, so start←first source, end←last source); split-type ops
    override the couple of slots where a divided set's head keeps only the start and its tail only
    the end."""
    start_src: list[int | None] = [priors[0] for priors in inherit]
    end_src: list[int | None] = [priors[-1] for priors in inherit]
    return start_src, end_src


def collect_bound_marks(fresh_clips: list[ReviewClip], prior_clips: list[ReviewClip],
                        start_src: list[int | None],
                        end_src: list[int | None]) -> list[BoundMark]:
    """The marked bounds to re-stamp after a restructure, read off the *prior* clips (so this must
    run before the model swap). For each fresh clip, if its start edge continues a prior whose
    start was hand-marked, carry that prior's ``start_sec`` (and likewise the end). Fresh clips
    whose edge re-derives (``None`` src, or a prior that wasn't marked) contribute nothing, so the
    rebuild's freshly seeded bound stands."""
    marks: list[BoundMark] = []
    for k in range(len(fresh_clips)):
        src = start_src[k]
        if src is not None and prior_clips[src].get("start_marked"):
            marks.append((k, "start_sec", prior_clips[src]["start_sec"]))
        src = end_src[k]
        if src is not None and prior_clips[src].get("end_marked"):
            marks.append((k, "end_sec", prior_clips[src]["end_sec"]))
    return marks


def apply_bound_marks(clips: list[ReviewClip], marks: list[BoundMark]) -> None:
    """Re-stamp collected bound marks onto the rebuilt clips, in place, restoring both the value
    and its ``start_marked``/``end_marked`` flag. Runs *after* reseed_set_bounds so a hand-marked
    edge wins over the re-derived one (matching set_bound, which likewise sets the bound verbatim
    without seam/clamp adjustment)."""
    for k, key, value in marks:
        clips[k][key] = value
        clips[k]["start_marked" if key == "start_sec" else "end_marked"] = True


# ── Match splitting & gap-fill ────────────────────────────────────────────────

def split_match_at(match: Match, p: float, next_number: Callable[[], int]) -> tuple[Match, Match]:
    """Two new match dicts cut at second `p`: the head keeps the parent's number/start, the
    tail gets a fresh number (from `next_number`) and the parent's end. Both are shallow copies
    : the parent dict is left untouched so the undo snapshot (which shares match refs) can
    restore it. The cut markers are full synthesized markers (frame number derived from the
    parent's frame rate), and head.end == tail.start, a contiguous seam apply_contextual_buffers
    won't buffer while the halves stay adjacent."""
    frames = match.get("frames") or []
    start_ts = match["start"]["timestamp_sec"]
    end_ts = match["end"]["timestamp_sec"]
    fps = marker_fps(match["start"], match["end"])
    head: Match = {**match,
                   "end": synth_marker(p, fps),
                   "frames": [f for f in frames if f.get("timestamp_sec", 0) <= p],
                   "duration_sec": p - start_ts}
    tail: Match = {**match,
                   "match_number": next_number(),
                   "start": synth_marker(p, fps),
                   "frames": [f for f in frames if f.get("timestamp_sec", 0) > p],
                   "duration_sec": end_ts - p}
    return head, tail


def gap_fill_pair(template: Match, p: float, far: float,
                  next_number: Callable[[], int]) -> tuple[Match, Match]:
    """The (grown, filler) matches a dead-space split produces: `template` grown so its
    boundary meets the playhead `p`, and a new match (a frame-less copy of the template
    under a fresh number) spanning from `p` out to `far`. The direction follows where
    `far` sits: after a preceding match the filler runs forward (template's *end* grows to
    `p`, filler [p, far]); in the head buffer it mirrors (template's *start* grows back to
    `p`, filler [far, p])."""
    fps = marker_fps(template["start"], template["end"])
    grown: Match
    if far > p:
        grown = {**template, "end": synth_marker(p, fps),
                 "duration_sec": p - template["start"]["timestamp_sec"]}
        lo, hi = p, far
    else:
        grown = {**template, "start": synth_marker(p, fps),
                 "duration_sec": template["end"]["timestamp_sec"] - p}
        lo, hi = far, p
    filler: Match = {**template, "match_number": next_number(),
                     "start": synth_marker(lo, fps), "end": synth_marker(hi, fps),
                     "frames": [], "duration_sec": hi - lo}
    return grown, filler


# ── Undo snapshot copy ────────────────────────────────────────────────────────

def copy_clips(clips: list[ReviewClip]) -> list[ReviewClip]:
    """A safe copy of a clip list for undo/restore: clip dicts are copied with fresh
    tag/list fields, but `_matches` keeps the original match object references (matches
    are never mutated, so there's no need to deep-copy their frames)."""
    return [{**clip,
             "grouping_tags": list(clip["grouping_tags"]),
             "detected_tags": list(clip["detected_tags"]),
             "detected_sponsors": list(clip["detected_sponsors"]),
             "tournament_tags": list(clip["tournament_tags"]),
             "tournament_sponsors": list(clip.get("tournament_sponsors") or []),
             "ingame_tags": list(clip["ingame_tags"]),
             "characters": list(clip["characters"]),
             "match_numbers": list(clip["match_numbers"]),
             "_matches": list(clip["_matches"])}
            for clip in clips]


# ── Propagation ───────────────────────────────────────────────────────────────

def matched_player_slots(clips: list[ReviewClip], current: int) -> Iterator[tuple[int, int, int]]:
    """Yield (player position i in the current clip, other clip index, position j) for
    every slot in another set whose detected tag reads as the same player as the current
    clip's player i (per similar: OCR-noise-tolerant). Matching runs on the detected
    originals because other sets still hold the uncorrected OCR reading. The caller
    guarantees `current` indexes a clip."""
    clip = clips[current]
    for i in range(NUM_PLAYERS):
        original = nth_or_blank(clip["detected_tags"], i)
        if not original:
            continue  # nothing was detected: no way to find this player elsewhere
        for idx, other in enumerate(clips):
            if idx == current:
                continue
            for j in range(NUM_PLAYERS):
                other_original = nth_or_blank(other["detected_tags"], j)
                if other_original and similar(other_original, original):
                    yield i, idx, j


def propagation_targets(clips: list[ReviewClip], current: int,
                        player: int | None = None) -> list[tuple[int, int, str, str]]:
    """(clip index, player position, new tag, new sponsor) for every player slot in another
    set that the current set's tag/sponsor applies to. A *corrected* player (their tag or
    sponsor differs from what was originally detected here) propagates every change to a
    matched slot. An *un-corrected* player still propagates one thing: a pure capitalization
    mismatch in the tag (same letters, different case), so a player the OCR happened to read
    with inconsistent casing across sets can be normalized without a throwaway edit. The
    current set's capitalization wins, but the slot's own sponsor is left untouched. A
    substantive (different-letters, including OCR-confusable) disagreement between un-edited
    sets is left alone; it needs an explicit edit to confirm the right spelling first.
    `player` (0/1) restricts to one side. Drives both the Propagate-tags action's enablement
    and the action itself."""
    clip = clips[current]
    targets = []
    for i, idx, j in matched_player_slots(clips, current):
        if player is not None and i != player:
            continue
        tag = nth_or_blank(clip["grouping_tags"], i)
        sponsor = nth_or_blank(clip.get("tournament_sponsors"), i)
        edited = (tag != nth_or_blank(clip["detected_tags"], i)
                  or sponsor != nth_or_blank(clip.get("detected_sponsors"), i))
        other = clips[idx]
        other_tag = nth_or_blank(other["grouping_tags"], j)
        other_sponsor = nth_or_blank(other.get("tournament_sponsors"), j)
        if other_tag == tag and other_sponsor == sponsor:
            continue  # slot already matches: nothing to write
        if edited:
            targets.append((idx, j, tag, sponsor))  # spread the correction
        elif other_tag != tag and other_tag.lower() == tag.lower():
            # Un-corrected: normalize a tag whose only difference is case (.lower(), not
            # _canonical, so confusable slips stay gated). Leave the slot's sponsor as-is.
            targets.append((idx, j, tag, other_sponsor))
    return targets


def character_targets(clips: list[ReviewClip], current: int, overwrite: bool,
                      player: int | None = None) -> list[tuple[int, int, str]]:
    """(clip index, player position, character) for every matched player slot in another
    set that propagating the current set's characters would change: blank slots only by
    default (the custom-m_char workflow: detection reads default m_chars but not custom portraits),
    every differing slot when `overwrite`. `player` (0/1) restricts to one side. Unlike tags, the source
    needs no edit: a detected character is a valid source to fill blanks elsewhere."""
    clip = clips[current]
    targets = []
    for i, idx, j in matched_player_slots(clips, current):
        if player is not None and i != player:
            continue
        char = nth_or_blank(clip["characters"], i).strip()
        if not char:
            continue  # no source character for this player
        other_char = nth_or_blank(clips[idx]["characters"], j).strip()
        should_propagate = (other_char != char) if overwrite else (not other_char)
        if should_propagate:
            targets.append((idx, j, char))
    return targets
