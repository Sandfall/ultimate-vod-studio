"""Extract individual matches from a pipeline event stream and split merged
matches at timer resets."""

from vodparse.schema import Event, Marker, Match
from vodparse.text_utils import format_timestamp

FULL_GAME_SECONDS        = 420   # the 7:00 match clock
RESET_JUMP_SEC           = 20    # an upward jump larger than this looks like a new game's clock
TIMER_DROP_TOLERANCE_SEC = 5     # timer may exceed elapsed wall time by this much (OCR jitter)
END_LEAD_SEC      = 15    # a game ends this many seconds before the next game starts
# Fallback OCR sample cadence (seconds) when a caller doesn't pass the run's real interval.
# The live app never uses it: vodparse.cli.extract_matches.main resolves the actual cadence with
# resolve_sample_interval (the pipeline records it as `seek_every`/`ocr_interval_sec` in the JSON
# config) and threads it explicitly through build_matches into every function below. The GUI
# always runs seek mode with --seek-every set, so that value is always present. This default only
# bites direct/one-off callers (e.g. tests) that omit the argument. Note it is stale as a
# *representative* cadence: the live default is 30 (the GUI/pipeline --ocr-interval default, and
# resolve_sample_interval's own fallback), not this 15: kept only so bare calls still get a sane,
# if legacy, value rather than 0.
DEFAULT_SAMPLE_INTERVAL_SEC = 15

# A clock-less gap this many sample intervals long (or longer) is treated as a between-game
# intermission rather than a single dropped OCR read: the threshold scales with the run's OCR
# cadence so it means the same thing at 15s, 30s, or 60s sampling.
INTERMISSION_MIN_SAMPLES = 3

# Make the end-anchor apply the splitter's own "is this a new game's clock?" test: when a game's
# trailing clock reading looks like the start of a *next* game (a jump/intermission rise the splitter
# refused to cut on: e.g. a single foreign reading after a long break), don't anchor the game's end
# on it. Flip to False to restore the pre-fix behavior (end = the literal last clock reading) while
# iterating on the consecutive-non-reads logic in _timer_reset_indices.
GAP_AWARE_END_ANCHOR = True

# Debug dump of stripped (implausible) timer readings, written by _clean_timer_frames.
# Appended to within a run (the cleaner runs once per match); vodparse.cli.extract_matches clears
# the stale file at the start of each run so repeated runs don't accumulate duplicates.
IMPLAUSIBLE_TIMERS_NAME = "implausible_timers.json"


def is_ocr_sample(event: Event) -> bool:
    """Whether a periodic-OCR event carries anything the splitter must see, so it is kept as
    one of a match's frames: band texts, a clock, the ocr_gated marker (--gate-ocr-on-timer
    emptied its texts), or a custom overlay layout's box reads. A layout run writes
    `layout_reads` on every un-gated sample even when its band plan OCR'd no band at all
    (overlay_bands.band_ocr_plan retires a band the layout's boxes cover), so a clock-less
    sample of such a run is exactly the intermission frame _looks_like_new_game and
    _first_game_start rely on being present; dropping it would silently re-merge games.
    A stock-icon read (`stock_slots`, written on every sample of a run with that detector)
    counts too, an empty list included: it records that the stock detector looked at the
    sample, and an empty one (no stock slots seen) is exactly an intermission frame. A layout
    that draws only a timer box writes no `layout_reads`, so such a sample would otherwise
    vanish. A sample with none of these saw nothing, and is left out as it always was."""
    return bool(event.get("texts") or event.get("timer") or event.get("ocr_gated")
                or event.get("layout_reads") is not None
                or event.get("stock_slots") is not None)


def extract_matches(events: list[Event]) -> list[Match]:
    """Group pipeline events into matches using the detected splash screens: a match opens at
    a "go" and closes at the next "game", keeping every OCR frame in between for later tag
    voting. This is the normal path: extract_matches_timer_only handles seek-mode runs that
    have no go/game detections. A "go" with no following "game" is reported and discarded."""
    matches: list[Match] = []
    pending_start = None        # the open match's "go" event, or None when between matches
    between: list[Event] = []   # periodic-OCR frames seen since the current "go"

    for event in events:
        label = event.get("label")

        if label == "go":
            # Open (or re-open) a match. A second "go" before any "game" replaces the start.
            pending_start = event
            between = []

        elif label == "game":
            # Close the open match here. A stray "game" with no open match is ignored.
            if pending_start is None:
                continue
            duration = round(event["timestamp_sec"] - pending_start["timestamp_sec"], 3)
            matches.append({
                "match_number": len(matches) + 1,
                "duration_sec": duration,
                "start": pending_start,
                "end": event,
                "frames": [pending_start, *between, event],  # all OCR frames, go → game
            })
            pending_start = None
            between = []

        else:  # periodic OCR (label is None): keep if it falls inside an open match
            # Keep timer-bearing frames too (even with no texts) so a mid-match timer
            # reset isn't dropped before split_on_timer_reset can see it, and the other
            # clock-less samples the splitter still needs (see is_ocr_sample).
            if pending_start is not None and is_ocr_sample(event):
                between.append(event)

    if pending_start is not None:
        print(f"Warning: game started at {pending_start['timestamp_sec']:.2f}s "
              f"(frame {pending_start['frame_number']}) but no \"game\" event followed: discarded")

    return matches


def extract_matches_timer_only(events: list[Event], sample_interval: float = DEFAULT_SAMPLE_INTERVAL_SEC,
                               debug: bool = False, stem: str = "", stage: str = "", *,
                               full_game_seconds: float = FULL_GAME_SECONDS,
                               fighter_check: bool = True) -> list[Match]:
    """Build the list of matches using only the timer. This is for runs where the pipeline
    sampled frames on a fixed interval (--seek-every) and never detected the "go" / "game"
    screens that normally mark where a match begins and ends. Each time the clock jumps back
    up, a new game has started, so frames are grouped between those jumps.
    `full_game_seconds` is the full match clock (Ultimate's 7:00 by default; pass a game's own
    clock length for anything else), used to bound valid clock readings and back-calculate
    game starts. `fighter_check` is whether an intermission cut may demand a detected fighter
    (see _timer_reset_indices): False for a run without a character detector.
    Pass debug=True to print the frame/segment/game DataFrames from _build_games; stem/stage
    name its debug CSVs."""
    # Keep every frame that saw something (see is_ocr_sample: text, a clock, or the markers
    # of a clock-less sample the intermission splitter and _first_game_start rely on).
    all_frames = _clean_timer_frames([e for e in events
                                      if is_ocr_sample(e) and e.get("label") not in ("go", "game")],
                                     stem=stem, full_game_seconds=full_game_seconds)
    if not all_frames:
        return []

    cuts = _timer_reset_indices(all_frames, sample_interval, full_game_seconds=full_game_seconds,
                                fighter_check=fighter_check)
    first_start = _first_game_start(all_frames, full_game_seconds=full_game_seconds)
    if not cuts:
        return [{
            "duration_sec": round(all_frames[-1]["timestamp_sec"] - first_start["timestamp_sec"], 3),
            "start": first_start,
            "end": all_frames[-1],
            "frames": _match_voting_frames(all_frames, first_start["timestamp_sec"],
                                           full_game_seconds=full_game_seconds),
        }]

    # No real go/game boundaries: the first game's start is back-calculated from its clock (so a
    # clock-less intro isn't glued on; see _first_game_start) and the last game's end is synthesized
    # (last_end=None). Trim trailing clock-less frames so post-game menus don't pollute tag voting.
    return _build_games(all_frames, cuts, first_start, None,
                        sample_interval, trim_voting=True, debug=debug, stem=stem, stage=stage,
                        full_game_seconds=full_game_seconds, fighter_check=fighter_check)


def valid_secs(seconds: float | None, full_game_seconds: float = FULL_GAME_SECONDS) -> float | None:
    """A timer reading only makes sense between 0 and the full match clock length
    (`full_game_seconds`: Ultimate's 7:00 by default, or a game's own clock length).
    Anything outside that range is an OCR mistake, so return None as if there were no reading."""
    return seconds if (seconds is not None and 0 <= seconds <= full_game_seconds) else None


def timer_sec(frame: Event, full_game_seconds: float = FULL_GAME_SECONDS) -> float | None:
    """The frame's timer reading in seconds remaining, or None if it has no valid timer."""
    timer = frame.get("timer")
    return valid_secs(timer.get("total_seconds"), full_game_seconds) if timer else None


def _timer_readings(frames: list[Event],
                    full_game_seconds: float = FULL_GAME_SECONDS) -> list[tuple[int, float, float]]:
    """(index, seconds-remaining, timestamp) for every non-go/game frame with a valid clock.
    go/game frames are skipped because their full-screen graphic can obscure the clock."""
    return [(index, seconds, frames[index]["timestamp_sec"])
            for index, frame in enumerate(frames)
            if frame.get("label") not in ("go", "game")
            and (seconds := timer_sec(frame, full_game_seconds)) is not None]


def _looks_like_new_game(prev: tuple[int, float, float], cur: tuple[int, float, float],
                         frames: list[Event], sample_interval: float,
                         full_game_seconds: float = FULL_GAME_SECONDS) -> tuple[bool, bool]:
    """The splitter's two new-game signals between consecutive clock readings (each an
    (index, seconds, timestamp) triple from _timer_readings over `frames`):

    - jumped_up: the clock rose by more than RESET_JUMP_SEC: a fresh game's clock.
    - intermission: a smaller rise across a clock-less gap of at least
      INTERMISSION_MIN_SAMPLES sample intervals. A fresh game's clock reads meaningfully
      HIGHER than the old game's even after a long break; a real clock only counts down,
      so a clear rise across a multi-sample clock-less intermission is a new game the jump
      test missed (its clock had already counted down to within RESET_JUMP_SEC of the old
      reading before its first sample). The rise must clear OCR jitter
      (TIMER_DROP_TOLERANCE_SEC): a stalled/paused clock reads about the same afterward,
      and a sub-jitter "rise" would split one paused game into slivers.

    Returned as separate flags because the caller's follow-up validation differs by signal
    (see _timer_reset_indices)."""
    prev_index, prev_seconds, prev_ts = prev
    cur_index, cur_seconds, cur_ts = cur
    jumped_up = cur_seconds - prev_seconds > RESET_JUMP_SEC
    intermission = (cur_seconds - prev_seconds > TIMER_DROP_TOLERANCE_SEC
                    and _clockless_gap_between(prev, cur, frames, sample_interval,
                                               full_game_seconds))
    return jumped_up, intermission


def _clockless_gap_between(prev: tuple[int, float, float], cur: tuple[int, float, float],
                           frames: list[Event], sample_interval: float,
                           full_game_seconds: float = FULL_GAME_SECONDS) -> bool:
    """Whether a clock-less intermission separates two consecutive clock readings (each an
    (index, seconds, timestamp) triple from _timer_readings over `frames`): at least
    INTERMISSION_MIN_SAMPLES sample intervals apart, with a sampled frame between them that
    showed no clock. More than a single dropped OCR read, at any OCR cadence."""
    prev_index, _, prev_ts = prev
    cur_index, _, cur_ts = cur
    return (cur_ts - prev_ts >= INTERMISSION_MIN_SAMPLES * sample_interval
            and any(frames[b].get("label") not in ("go", "game")
                    and timer_sec(frames[b], full_game_seconds) is None
                    for b in range(prev_index + 1, cur_index)))


def _is_stray_reading_after_gap(prev: tuple[int, float, float], cur: tuple[int, float, float],
                                frames: list[Event], sample_interval: float,
                                full_game_seconds: float = FULL_GAME_SECONDS) -> bool:
    """Whether `cur`, a game's LAST clock reading, is a stray read rather than the game
    resuming: it comes after a clock-less intermission, it is not a rise (those are
    _looks_like_new_game's), and its frame shows no fighter. A lower reading after a long
    break was always taken for the same game un-pausing, which let one misread of
    on-screen text stretch a game across the whole break (a "coming up next" card read
    as 2:53, fourteen minutes after the game's real last reading of 4:34, on a Melee
    stream whose calibrated clock box is read without the detector confirming a clock is
    there). A game that really resumes shows fighters, or goes on to give further
    readings, in which case `cur` is not its last and this is never asked of it.

    The clock-less gap itself must show no fighter PAIR either: a stretch of gameplay where
    the clock merely failed to read (occluded, misparsed) is not an intermission, and a
    last reading after it on a frame where the fighter detector happened to miss would
    otherwise be dropped, cutting the game short at the reading before the gap. A pair,
    not one side: the detector fires on one side alone on a system menu or a results card
    (a lone "Hero" on a Switch home screen stretched a game 120 s onto it when one side
    was enough), where gameplay shows both.

    Only meaningful for a run that loaded a character detector: the caller skips it
    otherwise (fighter_check, as in _timer_reset_indices), since without one no frame ever
    carries a fighter."""
    prev_index, prev_seconds, _ = prev
    cur_index, cur_seconds, _ = cur
    return (cur_seconds - prev_seconds <= TIMER_DROP_TOLERANCE_SEC
            and _clockless_gap_between(prev, cur, frames, sample_interval, full_game_seconds)
            and not any(_has_fighter_pair(frames[b]) for b in range(prev_index + 1, cur_index))
            and not _has_characters(frames[cur_index]))


def _has_fighter_pair(frame: Event) -> bool:
    """True if the character detector found a fighter on BOTH HUD sides: gameplay, as
    against a menu or a results card where it fires on one side alone."""
    chars = frame.get("characters")
    return bool(chars and chars.get("left") and chars.get("right"))


def _has_characters(frame: Event) -> bool:
    """True if the character detector found a fighter on either HUD side: i.e. this frame is real
    in-game footage, not a menu/results/versus screen. Used to confirm a clock-less-gap split lands
    on an actual game and not an OCR misread of on-screen text (e.g. a name on a
    character-select screen read as "4:30")."""
    chars = frame.get("characters")
    return bool(chars and (chars.get("left") or chars.get("right")))


def _flag_implausible(frames: list[Event],
                      timer_readings: list[tuple[int, float, float]]) -> tuple[set[int], list[Event]]:
    """The cleaner's plausibility state machine: walk the clock readings and flag any that fell
    faster than wall time allows. Returns (indices of frames whose reading must be stripped,
    the flagged frames plus the good one just before each, for the debug dump).

    We keep the last two trusted readings, each an (index, seconds, timestamp) tuple. `prev` is
    the current anchor every new reading is checked against; `before_prev` is the one before it.
    Keeping two lets us tell a single high-misread *anchor* (the clock skips over it and resumes
    from before_prev → re-anchor, don't strip) from a genuine run of bad readings (nothing
    resumes the earlier line → strip them). Tuple fields are accessed positionally below:
    [0] = frame index, [1] = seconds remaining, [2] = timestamp."""
    bad_indices: set[int] = set()
    bad_pairs:   list[Event] = []
    dumped:      set[int] = set()   # frame indices already in bad_pairs, so a run of bad
                                    # readings doesn't re-append the one good frame before it
    before_prev = None
    prev = None
    for index, seconds, timestamp in timer_readings:
        if prev is not None:
            prev_index, prev_seconds, prev_timestamp = prev
            # Compare how far the clock fell against how much real time actually passed.
            elapsed = timestamp - prev_timestamp
            drop    = prev_seconds - seconds
            if drop > 0 and drop > elapsed + TIMER_DROP_TOLERANCE_SEC:
                # The clock fell faster than wall time allows since the anchor. Usually the *current*
                # reading is the misread, so strip it and keep checking against the last good anchor.
                # But if this reading instead resumes the countdown from the reading *before* the anchor,
                # then the anchor itself was a high misread: a small upward OCR blip the cleaner keeps,
                # since it only strips fast *drops*, never small rises. Stripping against such an anchor
                # cascades: every real reading after it looks too-fast and gets stripped, wiping a whole
                # game. So when the line skips over the anchor, re-anchor onto this reading instead.
                #
                # "Resumes before the anchor" = the current reading is a plausible countdown from
                # before_prev (skipping the anchor): the clock fell from before_prev's value
                # (before_prev[1]) to this one by an amount that is neither a rise beyond OCR jitter
                # (>= -TOLERANCE) nor faster than the wall time elapsed since before_prev allows
                # (<= that elapsed + TOLERANCE). It's the same plausibility test used for a normal
                # drop above, just measured from before_prev instead of the (suspect) anchor.
                resumes_before_anchor = (
                    before_prev is not None
                    and -TIMER_DROP_TOLERANCE_SEC <= before_prev[1] - seconds
                        <= (timestamp - before_prev[2]) + TIMER_DROP_TOLERANCE_SEC)
                if resumes_before_anchor:
                    assert before_prev is not None  # implied by resumes_before_anchor; narrows for mypy
                    print(f"    Re-anchored at {format_timestamp(timestamp)}: kept reading "
                          f"{format_timestamp(prev_seconds)} was a high misread; clock continues "
                          f"{format_timestamp(before_prev[1])} → {format_timestamp(seconds)} across it")
                    # Drop the distrusted anchor instead of carrying it forward: it's the high
                    # misread we just rejected, so it must not become the baseline a later re-anchor
                    # is tested against. (A high blip always sits above the real line, so it can't
                    # actually validate a later flagged reading: but propagating a rejected value as
                    # a trust baseline is misleading and a future change could make it bite.) The next
                    # trusted reading repopulates before_prev normally.
                    before_prev = None
                    prev = (index, seconds, timestamp)
                    continue
                # A genuine misread: strip it. Note we don't update `prev`, so the next reading is
                # checked against the last good one rather than this bad one.
                print(f"    Implausible timer at {format_timestamp(timestamp)}: "
                      f"{format_timestamp(prev_seconds)} → {format_timestamp(seconds)} "
                      f"({drop:.0f}s drop in {elapsed:.1f}s wall time): stripping")
                bad_indices.add(index)
                for j in (prev_index, index):
                    if j not in dumped:
                        dumped.add(j)
                        bad_pairs.append(frames[j])
                continue
            # Clock barely moved over a long stretch: likely a pause. Just log it; the
            # reading itself is still valid, so it stays.
            if 0 < drop < elapsed * 0.3 and elapsed >= 20:
                print(f"    Possible pause at {format_timestamp(timestamp)}: "
                      f"timer dropped {drop:.0f}s in {elapsed:.1f}s wall time")
        before_prev = prev
        prev = (index, seconds, timestamp)
    return bad_indices, bad_pairs


def _dump_implausible(bad_pairs: list[Event], stem: str) -> None:
    """Save the stripped readings (and the good one just before each) so they can be eyeballed
    later. Append rather than overwrite: the cleaner runs once per match
    within an extraction run, and each call must add to the same per-run file."""
    import json
    from .jsonio import write_json_atomic
    from .paths import debug_dir
    out_path = debug_dir(stem) / IMPLAUSIBLE_TIMERS_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    write_json_atomic(out_path, existing + bad_pairs)
    print(f"    Implausible timer frames (before+after) → {out_path}")


def _strip_timers(frames: list[Event], bad_indices: set[int]) -> list[Event]:
    """A new frames list with the flagged readings removed: flagged frames get a shallow copy
    without their timer, the rest are reused as-is."""
    cleaned: list[Event] = []
    for index, frame in enumerate(frames):
        if index in bad_indices:
            frame = frame.copy()
            frame.pop("timer", None)
        cleaned.append(frame)
    return cleaned


def _fps_of(frame: Marker) -> float:
    """The frame's effective frame rate (frame_number / timestamp), or 0.0 if its timestamp is 0."""
    timestamp = frame.get("timestamp_sec")
    return frame["frame_number"] / timestamp if timestamp else 0.0


def _synthesized_event(timestamp_sec: float, reference_frame: Marker) -> Marker:
    """Build a stand-in start/end marker {timestamp_sec, frame_number} for a time we
    calculated rather than detected. The frame number is worked out from the reference
    frame's frame rate so seeks and printouts still line up. synthesized=True tells the clip
    exporter not to add its usual end padding: a calculated end already sits END_LEAD_SEC
    before the next game starts, so padding would run into the next match."""
    timestamp = max(0.0, timestamp_sec)
    fps = _fps_of(reference_frame)
    if fps:
        frame_number = int(round(timestamp * fps))
    else:
        # No usable frame rate (reference frame had timestamp 0): reuse its frame number.
        frame_number = reference_frame["frame_number"]

    return {
        "timestamp_sec": round(timestamp, 3),
        "frame_number": frame_number,
        "synthesized": True,
    }


def _clean_timer_frames(frames: list[Event], stem: str = "",
                        full_game_seconds: float = FULL_GAME_SECONDS) -> list[Event]:
    """Remove timer readings that fell faster than real time allows: the clock can't count
    down more seconds than have actually passed, so these are OCR mistakes. Returns a new
    list; frames with a bad reading get a copy with the timer removed, the rest are reused
    as-is. The bad readings (and the good one just before each) are also written to a debug
    JSON file so you can look at those frames later.

    One reading is treated specially: a small *high* misread (e.g. 5:44 read as 5:51) isn't a fast
    drop, so it slips through and becomes the trusted anchor: then every real reading after it looks
    too-fast and would cascade-strip the whole game. When a flagged reading instead resumes the
    countdown from the reading *before* the anchor, the anchor was the misread, so we re-anchor onto
    the real run rather than stripping it (the lone blip is left in place; it's harmless to splitting)."""
    bad_indices, bad_pairs = _flag_implausible(frames, _timer_readings(frames, full_game_seconds))
    if bad_pairs:
        _dump_implausible(bad_pairs, stem)
    return _strip_timers(frames, bad_indices)


def _timer_reset_indices(frames: list[Event], sample_interval: float = DEFAULT_SAMPLE_INTERVAL_SEC,
                         full_game_seconds: float = FULL_GAME_SECONDS, *,
                         fighter_check: bool = True) -> list[int]:
    """Return the positions of frames where a new game starts. Two signals open a new game:

    1. The clock jumps back up by more than RESET_JUMP_SEC between adjacent readings. Checking
       against the reset value (not the old pre-reset value) is what lets a small reset survive a
       coarse sample interval: the new clock can count back down to within RESET_JUMP_SEC of the
       old value before the next sample without being mistaken for a misread.
    2. A long clock-less *intermission* (a venue/sponsor card or overlay with no game clock) spans
       at least INTERMISSION_MIN_SAMPLES sample intervals, and the clock reads HIGHER on the far
       side of it. A real game's clock only counts down, so a higher reading after a multi-minute
       clock-less break is a new game's fresh clock: one the jump test missed because, sampled only
       after the break, it had already counted back down to within RESET_JUMP_SEC of the old
       reading. The gap threshold is relative to `sample_interval`, so it means the same thing at
       15s / 30s / 60s OCR cadences. (A clock that merely stalled: the same game paused: reads the
       same or lower afterward, so it is left as one game rather than split into slivers.)

    A bare jump (no clock-less gap) is only trusted if the next reading is a plausible countdown
    continuation of the new clock: it sits at (or just above, for OCR jitter) the new value and has
    fallen no faster than wall time allows. A real reset (5:59 -> 6:40 -> 6:10) keeps ticking down
    from the new clock and splits; a one-off high misread (2:00 -> 6:50 -> 1:30) snaps back to the old
    game, an impossible drop for the elapsed time, and is ignored.

    An *intermission* needs no validating successor: the multi-sample clock-less gap already excludes
    a one-off misread (which has no such gap): so it splits even when the next reading belongs to a
    different game (a single-sample game like a hand-warmer the validation could never confirm). It
    must instead land on real gameplay: the new reading's frame has a detected fighter. That rejects a
    rise onto a menu/versus screen where the timer OCR misread on-screen text (e.g. a character-select name -> 4:30).
    That fighter evidence only exists for a run that loaded a character detector (Ultimate's
    portrait detector, or Melee's stock-icon detector, which fills `characters` only when it
    sees exactly two stock slots, so singles only): pass ``fighter_check=False`` for a run
    without one (Rivals 2, Other, a run whose declared weight was missing), whose frames never carry
    a fighter, or no intermission cut could ever fire for it and two games separated by a
    clock-less break would merge whenever the new game's first read arrived too late for the
    jump test (seen on a Rivals 2 Auto run, 2026-09-05). Such a run keeps the other guards (the
    clock-less gap, the rise, the back-calculation below); the residual risk is a misread of
    on-screen text that happens to parse as a plausible clock right after a break.
    With no successor to bound it, an intermission cut is also rejected when the new game's
    back-calculated start (clock implies it began that long ago) would precede the prior reading: an
    impossible overlap from a small rise over a short gap; the new game's elapsed time must fit the gap.
    "go"/"game" frames are skipped because their full-screen graphic can sometimes obscure the
    clock. Pass in frames already run through _clean_timer_frames."""
    readings = _timer_readings(frames, full_game_seconds)
    cuts: list[int] = []
    prev: tuple[int, float, float] | None = None   # last trusted reading: (index, seconds, timestamp)
    for position, (index, seconds, timestamp) in enumerate(readings):
        if prev is not None:
            _, _, prev_timestamp = prev
            # The two new-game signals (see _looks_like_new_game): a clear clock jump, or a
            # smaller rise across a multi-sample clock-less intermission.
            jumped_up, intermission = _looks_like_new_game(
                prev, (index, seconds, timestamp), frames, sample_interval, full_game_seconds)
            if jumped_up or intermission:
                cut = False
                # Strict path: the next reading must continue the new clock: dropped a plausible
                # amount (never more than wall time + jitter) and didn't rise above the reset (jitter
                # aside). Required for a bare jump (a possible one-off misread); a validated
                # intermission passes here too.
                if position + 1 < len(readings):
                    _, next_seconds, next_timestamp = readings[position + 1]
                    elapsed = next_timestamp - timestamp
                    drop    = seconds - next_seconds
                    if -TIMER_DROP_TOLERANCE_SEC <= drop <= elapsed + TIMER_DROP_TOLERANCE_SEC:
                        cut = True
                # Intermission path: a clock-less gap needs no validating successor, but the new
                # reading must be on real gameplay (a fighter is detected), so a rise onto a menu or
                # versus screen (the timer OCR misreading on-screen text) is not taken for a new game.
                # It must also not back-calculate a start *before* the prior game's clock was last
                # seen: with no successor to bound it, a small rise over the minimum gap extrapolates
                # the new game's start (timestamp - elapsed) backwards into the prior game, producing
                # two overlapping games. A new game's elapsed time has to fit within the clock-less gap.
                back_calc_start = timestamp - (full_game_seconds - seconds)
                on_gameplay = _has_characters(frames[index]) if fighter_check else True
                if (not cut and intermission and on_gameplay
                        and back_calc_start >= prev_timestamp - TIMER_DROP_TOLERANCE_SEC):
                    cut = True
                if cut:
                    cuts.append(index)
                    prev = (index, seconds, timestamp)
                continue
        prev = (index, seconds, timestamp)
    return cuts


def _back_calculated_start(seg_frames: list[Event],
                           full_game_seconds: float = FULL_GAME_SECONDS) -> Marker:
    """Work out when a game actually started, for a segment created by a timer reset. The
    first frame in the segment is usually mid-game, so use its clock: on Ultimate's 7:00 clock,
    if 6:00 is left, the game began (7:00 - 6:00) = 1 minute earlier; `full_game_seconds` swaps
    in a different clock length for other games. This assumes the game was not paused before
    that first reading; pause-aware refinement lives in pause_check.py as an opt-in postprocess."""
    first_frame = seg_frames[0]
    for frame in seg_frames:
        remaining = timer_sec(frame, full_game_seconds)
        if remaining is not None:
            # First clock reading in the segment: the game began (full_game_seconds - remaining) earlier.
            start = frame["timestamp_sec"] - (full_game_seconds - remaining)
            return _synthesized_event(start, first_frame)
    # No clock reading anywhere in the segment: fall back to the first frame's time.
    return _synthesized_event(first_frame["timestamp_sec"], first_frame)


def _first_game_start(frames: list[Event],
                      full_game_seconds: float = FULL_GAME_SECONDS) -> Marker:
    """The first seek-mode game's start. Like every later game, back-calculate it from the clock so a
    clock-less intro before the first reading (stream cards, player intros, the bracket) isn't glued
    onto match 1: but never earlier than the first sampled frame. A clock that implies the game began
    before the footage (e.g. the stream joined mid-game) is clamped to the footage start rather than
    extrapolated into unsampled pre-game; that also leaves a VOD whose first frame already shows a clock
    starting exactly where it does today."""
    back_calc = _back_calculated_start(frames, full_game_seconds)
    return back_calc if back_calc["timestamp_sec"] >= frames[0]["timestamp_sec"] else frames[0]


def _last_timer_frame(seg: list[Event],
                      full_game_seconds: float = FULL_GAME_SECONDS) -> Event:
    """Return the last frame in the segment that still shows a clock. "go"/"game" frames are
    skipped because their graphic can produce unreliable timer readings. Falls back to the
    final frame if none has a reading."""
    for frame in reversed(seg):
        if frame.get("label") not in ("go", "game") and timer_sec(frame, full_game_seconds) is not None:
            return frame
    return seg[-1]


def _end_anchor_frame(seg: list[Event], sample_interval: float,
                      full_game_seconds: float = FULL_GAME_SECONDS, *,
                      fighter_check: bool = True) -> Event:
    """Like _last_timer_frame, but it won't anchor a game's end on a *trailing* reading that looks
    like the start of the NEXT game, nor on a stray reading long after the game stopped showing
    a clock (_is_stray_reading_after_gap; only with `fighter_check`, i.e. for a run that loaded
    a character detector). When a split fails to fire (e.g. a single foreign clock sampled
    after a long intermission/character-select), that orphan reading would otherwise extend the clip
    far past where this game actually ended. So walk back over the clock readings and drop any whose
    rise over its predecessor is the splitter's own new-game signal: a jump over RESET_JUMP_SEC, or
    a smaller rise across a multi-sample clock-less intermission: anchoring instead on the end of the
    game's own contiguous clock run. A real last reading only *falls* relative to its predecessor,
    with no clock-less intermission before it (or with fighters on its frame), so it is never dropped. With GAP_AWARE_END_ANCHOR off this is exactly _last_timer_frame.
    (Applies _timer_reset_indices' own jumped_up/intermission test via _looks_like_new_game,
    so the two can't drift apart.)
    """
    return _end_anchor(seg, sample_interval, full_game_seconds, fighter_check=fighter_check)[0]


def _end_anchor(seg: list[Event], sample_interval: float,
                full_game_seconds: float = FULL_GAME_SECONDS, *,
                fighter_check: bool = True) -> tuple[Event, bool]:
    """_end_anchor_frame's walk, also saying whether it backed over a stray reading
    (_is_stray_reading_after_gap). A game whose end was anchored before a stray has an
    intermission after it inside its own frames, which must not vote on its tags
    (_build_games); an end backed over a next-game reading changes nothing about the vote,
    which already stops at the last clock reading as it always has."""
    if not GAP_AWARE_END_ANCHOR:
        return _last_timer_frame(seg, full_game_seconds), False
    readings = _timer_readings(seg, full_game_seconds)
    if not readings:
        return seg[-1], False
    # The stray rule reads fighter evidence, so it applies only where the run's frames
    # carry any: a run whose detector never fills a fighter pair (Melee doubles, read
    # singles-only) would otherwise call every reading after a clock dropout a stray and
    # walk the end back to the game's first minute.
    stray_rule = fighter_check and any(_has_fighter_pair(frame) for frame in seg)
    k = len(readings) - 1
    skipped_stray = False
    while k > 0:
        jumped_up, intermission = _looks_like_new_game(readings[k - 1], readings[k],
                                                       seg, sample_interval, full_game_seconds)
        stray = stray_rule and _is_stray_reading_after_gap(
            readings[k - 1], readings[k], seg, sample_interval, full_game_seconds)
        if not (jumped_up or intermission or stray):
            break
        # cur is a next-game clock the splitter wouldn't trust, or a stray read after the
        # game was over: back up to its predecessor. (A stray is a fall within
        # TIMER_DROP_TOLERANCE_SEC after a gap; a jump or an intermission needs a rise, so
        # the three never coincide.)
        if stray:
            skipped_stray = True
        k -= 1
    return seg[readings[k][0]], skipped_stray


def _synthesized_end(seg: list[Event], next_start_ts: float, sample_interval: float,
                     full_game_seconds: float = FULL_GAME_SECONDS, *,
                     fighter_check: bool = True) -> Marker:
    """Pick an end time for a game that isn't the last one: one sample interval after its
    final clock reading, but no later than END_LEAD_SEC before the next game starts
    (whichever comes first), and never before its own final clock reading."""
    last_frame = _end_anchor_frame(seg, sample_interval, full_game_seconds,
                                   fighter_check=fighter_check)
    return _synthesized_end_from(last_frame, next_start_ts, sample_interval)


def _synthesized_end_from(last_frame: Event, next_start_ts: float,
                          sample_interval: float) -> Marker:
    """_synthesized_end for an end anchor already found (see it for the rule)."""
    interval_end = last_frame["timestamp_sec"] + sample_interval
    lead_end = next_start_ts - END_LEAD_SEC
    # Floor at the last clock reading: a rapid rematch can put next_start within END_LEAD_SEC
    # of last_frame, making lead_end land before it and yield a negative-duration (ss>to) clip.
    end_ts = max(last_frame["timestamp_sec"], min(interval_end, lead_end))
    return _synthesized_event(end_ts, last_frame)


def _match_voting_frames(frames: list[Event], start_ts: float,
                         full_game_seconds: float = FULL_GAME_SECONDS, *,
                         end_ts: float | None = None) -> list[Event]:
    """The frames that should sway a game's tag vote: from its (clock-derived) start through its last
    clock reading, and never past `end_ts`, the end the game was given. Dropping frames *before* the
    start keeps a pre-game intro / VS card out of the vote (the in-game scoreboard overlay carries
    the same player/round tags during play); dropping the trailing clock-less frames keeps post-game
    char-select / results out; dropping the frames past the end keeps an intermission's cards out
    when the end was anchored before a stray reading (_end_anchor_frame). If nothing is left or the
    game shows no clock at all, keep what there is rather than emptying the vote."""
    kept = [f for f in frames
            if f["timestamp_sec"] >= start_ts and (end_ts is None or f["timestamp_sec"] <= end_ts)]
    if not kept:
        return frames
    last = max((i for i, f in enumerate(kept) if timer_sec(f, full_game_seconds) is not None),
              default=len(kept) - 1)
    return kept[: last + 1]


def _build_games(frames: list[Event], cuts: list[int], first_start: Marker,
                 last_end: Marker | None, sample_interval: float, trim_voting: bool,
                 debug: bool = False, stem: str = "", stage: str = "",
                 full_game_seconds: float = FULL_GAME_SECONDS, *,
                 fighter_check: bool = True) -> list[Match]:
    """Cut `frames` at `cuts` into one match per game and fill in each game's start and end.

    Shared by extract_matches_timer_only and split_on_timer_reset; expects `frames` already
    cleaned and at least one cut.
      first_start  start of the first game (the real `go`, or the first sampled frame).
      last_end     end of the last game (the real `game`); pass None to calculate it from the
                   last clock reading plus one sample interval.
      trim_voting  for each game, restrict the tag-voting frames to its own span: from its start
                   (so a pre-game intro / VS card before the first game doesn't sway the vote) through
                   its last clock reading (so trailing char-select / results don't either). See
                   _match_voting_frames.
      debug        print the per-frame, per-segment, and per-game DataFrames before returning.
      stem, stage  passed to dump_dataframes: stem picks the per-vod debug folder, stage names
                   the {name}_{stage}.csv debug CSVs.
      full_game_seconds  the full match clock (Ultimate's 7:00 by default), used to bound valid
                   readings and back-calculate starts.
      fighter_check  whether the run loaded a character detector, as in _timer_reset_indices:
                   lets a game's end skip a stray clock reading (_end_anchor_frame).
    Every game after the first starts from its clock, and ends at the earlier of: last clock
    reading + sample interval, or END_LEAD_SEC before the next game starts."""
    # Turn the cut points into [0, cut1, cut2, ..., len] and slice one frame run per game.
    # Each cut is the index of a new game's first frame, so it opens the next slice.
    bounds = [0, *cuts, len(frames)]
    segments = [frames[a:b] for a, b in zip(bounds, bounds[1:])]

    # The first game keeps the given start (real `go`, or the first sampled frame); every
    # later game has no detected boundary, so back-calculate its start from the clock.
    starts = [first_start] + [_back_calculated_start(seg, full_game_seconds) for seg in segments[1:]]

    games: list[Match] = []
    for idx, (seg, start) in enumerate(zip(segments, starts)):
        anchor, skipped_stray = _end_anchor(seg, sample_interval, full_game_seconds,
                                            fighter_check=fighter_check)
        if idx < len(segments) - 1:
            # Internal game: end just before the next game's start (see _synthesized_end).
            end = _synthesized_end_from(anchor, starts[idx + 1]["timestamp_sec"], sample_interval)
        elif last_end is not None:
            # Last game, caller supplied a real end (the `game` detection): use it as-is.
            end = last_end
        else:
            # Last game with no detected end: extend one sample interval past the final clock.
            end = _synthesized_event(anchor["timestamp_sec"] + sample_interval, anchor)

        if trim_voting:
            # Restrict voting frames to this game's own span (drop a pre-game intro before the
            # start and trailing char-select/results) so neither sways the player-tag vote.
            # When the end was anchored before a stray reading, the intermission between the
            # two is inside this segment too: its cards stop at the end as well. (Only then:
            # frames up to a refused next-game reading have always voted, and taking them
            # away moved the tags of a dozen stored runs, some for the worse.)
            seg = _match_voting_frames(seg, start["timestamp_sec"], full_game_seconds,
                                       end_ts=end["timestamp_sec"] if skipped_stray else None)

        games.append({
            "duration_sec": round(end["timestamp_sec"] - start["timestamp_sec"], 3),
            "start": start,
            "end": end,
            "frames": seg,
        })

    if debug:
        from . import debug as _debug   # lazy: keeps pandas off the hot path
        segments_df = _debug.segments_dataframe(frames, cuts)
        games_df = _debug.games_dataframe(frames, cuts, games)
        for path in _debug.dump_dataframes(segments_df, games_df, games,
                                            stem=stem, stage=stage):
            print(f"  debug CSV → {path}")

    return games


def split_on_timer_reset(match: Match, sample_interval: float = DEFAULT_SAMPLE_INTERVAL_SEC,
                         debug: bool = False, stem: str = "", stage: str = "", *,
                         full_game_seconds: float = FULL_GAME_SECONDS,
                         fighter_check: bool = True) -> list[Match]:
    """Split one match that actually contains several games into a separate match per game,
    cutting wherever the clock resets. The first game keeps the real `go` start and the last
    keeps the real `game` end; the starts in between are worked out from the clock, and each
    of those games ends at the earlier of last clock reading + sample interval or END_LEAD_SEC
    before the next game. Returns [match] unchanged if the clock never resets.
    `full_game_seconds` is the full match clock (Ultimate's 7:00 by default; pass a game's own
    clock length for anything else), used to bound valid clock readings and back-calculate
    game starts; `fighter_check` as in extract_matches_timer_only.
    Pass debug=True to print the frame/segment/game DataFrames from _build_games; stem/stage
    name its debug CSVs."""
    frames = _clean_timer_frames(match["frames"], stem=stem, full_game_seconds=full_game_seconds)
    cuts = _timer_reset_indices(frames, sample_interval, full_game_seconds=full_game_seconds,
                                fighter_check=fighter_check)
    if not cuts:
        return [match]

    print(f"  Timer reset in game {match.get('match_number', '?')} "
          f"→ split into {len(cuts) + 1} games")

    return _build_games(frames, cuts, match["start"], match["end"],
                        sample_interval, trim_voting=False, debug=debug, stem=stem, stage=stage,
                        full_game_seconds=full_game_seconds, fighter_check=fighter_check)
