"""Interactive debug helpers for match extraction.

Kept out of match_extraction.py so the hot path stays free of pandas. The builders lay out the
working variables of _build_games (frames, the per-game segment slices, and the finished games)
as DataFrames; _build_games imports this module lazily, only when called with debug=True."""

import csv
from pathlib import Path

from .match_extraction import timer_sec
from .overlay_layout import (REGION_INGAME_P1, REGION_INGAME_P2, REGION_PLAYER1, REGION_PLAYER2,
                             CustomOverlayLayout)
from .schema import Event, Match
from .text_utils import format_timestamp
from .games import INGAME_BAND_NONE, INGAME_BAND_TAGS
from .tags import (INGAME_TAG_BAND_FRAC_DEFAULT, find_best_of, find_player_names, find_round,
                   find_tournament_tags, nameplates_from_boxes)


def _segment_bounds(frames: list[Event], cuts: list[int]) -> list[int]:
    """The [0, cut1, ..., len(frames)] segment boundaries _build_games slices on, so seg_id and
    the per-segment frame runs here match its `segments`."""
    return [0, *cuts, len(frames)]


def games_dataframe(frames: list[Event], cuts: list[int], games: list[Match]):
    """Lay out the three _build_games working variables as one row-per-frame DataFrame. `game`
    mirrors the segment slicing (one id per cut run), `cut` flags the first frame of each new
    game, and the per-game start/end/duration are broadcast onto every frame of that game so a
    single `df.groupby("game")` reproduces `games`."""
    import pandas as pd

    bounds = _segment_bounds(frames, cuts)
    seg_id = {}   # frame index → game (segment) number
    for game, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        for i in range(lo, hi):
            seg_id[i] = game
    cut_set = set(cuts)

    rows = []
    timerless_before = 0   # consecutive clock-less frames immediately preceding this row
    for i, f in enumerate(frames):
        g = games[seg_id[i]]
        timer = timer_sec(f)
        rows.append({
            "ts": f["timestamp_sec"],
            "timer": timer,
            "timer_fmt": None if timer is None else format_timestamp(timer),
            "timerless_before": timerless_before,
            "i": i,
            "game": seg_id[i],
            "cut": i in cut_set,
            "frame": f.get("frame_number"),
            "label": f.get("label"),
            "game_start": g["start"]["timestamp_sec"],
            "game_end": g["end"]["timestamp_sec"],
            "game_dur": g["duration_sec"],
        })
        # A clock-less frame extends the run; a reading ends it. go/game frames are neither a
        # reading nor a blank (mirroring _readings / the sweep's n_blank) so they leave it as-is.
        if f.get("label") in ("go", "game"):
            pass
        elif timer is None:
            timerless_before += 1
        else:
            timerless_before = 0
    return pd.DataFrame(rows)


def segments_dataframe(frames: list[Event], cuts: list[int]):
    """One row per segment (the frame-run _build_games slices for each game). Shows each
    segment's frame index range, timestamp span, and timer range: the timer should fall
    high→low within a segment and jump back up at the next, so a `timer_last` that is much
    lower than the following row's `timer_first` confirms a clean cut."""
    import pandas as pd

    # Same bounds and slicing _build_games uses to build `segments`.
    bounds = _segment_bounds(frames, cuts)
    segments = [frames[a:b] for a, b in zip(bounds, bounds[1:])]

    rows = []
    for g, seg in enumerate(segments):
        # Deliberately NOT match_extraction._timer_readings: this debug view counts every clock
        # reading in the segment, including any on go/game frames that the production splitter skips.
        timers = [t for f in seg if (t := timer_sec(f)) is not None]
        rows.append({
            "ts_first": seg[0]["timestamp_sec"] if seg else None,
            "ts_last": seg[-1]["timestamp_sec"] if seg else None,
            "timer_first": timers[0] if timers else None,
            "timer_last": timers[-1] if timers else None,
            "game": g,
            "n_frames": len(seg),
            "i_start": bounds[g],
            "i_end": bounds[g + 1] - 1,
            "n_timer": len(timers),
        })
    return pd.DataFrame(rows)


def dump_dataframes(segments_df, games_df, games: list[Match],
                    out_dir: str | None = None, stem: str = "", stage: str = "") -> list[Path]:
    """Write the segments and per-frame DataFrames to CSV and return the paths. Defaults to the
    per-vod debug folder (json_output/<stem>/debug/); pass out_dir to override. When stage is
    provided, uses {name}_{stage}.csv naming like dump_frames_csv. Otherwise defaults to
    {name}_t{tag}s.csv with a timestamp tag for collision avoidance."""
    from .paths import debug_dir

    out = Path(out_dir) if out_dir is not None else debug_dir(stem)
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    for name, df in (("segments", segments_df), ("frames", games_df)):
        if stage:
            path = out / f"{name}_{stage}.csv"
        else:
            tag = int(games[0]["start"]["timestamp_sec"]) if games else 0
            path = out / f"{name}_t{tag}s.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    return paths


REJECTED_TIMERS_NAME = "rejected_timers.csv"

_REJECTED_CSV_FIELDS = [
    "stage", "rejection_reason", "timestamp_sec",
    "raw_ocr_string", "parsed_value",
    "prev_accepted_value", "prev_accepted_ts", "bbox",
]


def write_rejected_timers_csv(rejections: list[dict], path) -> None:
    """Dump the implausible/garbage timer reads dropped during a detection run (see
    timer.TimerRecovery and timer.read_timer_at_bbox) to a CSV for debugging: the readings that
    are otherwise nulled out and vanish. One row per dropped *reading* (not per frame: a single
    dropped frame can yield both an ocr_parse_detector row and one recovery-stage row, so dedupe on
    timestamp_sec if counting frames). `stage` says which gate rejected it (ocr_parse_detector /
    ocr_parse_recovery / recovery_accept) and `rejection_reason` why; `raw_ocr_string` carries what
    the OCR actually read. bbox is flattened to "x1,y1,x2,y2"."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_REJECTED_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rejections:
            row = dict(r)
            bbox = row.get("bbox")
            if isinstance(bbox, list | tuple):
                row["bbox"] = ",".join(str(int(v)) for v in bbox)
            writer.writerow(row)


def flush_rejected_timers(rejections: list[dict], stem: str):
    """Write one detection run's rejected reads to debug/rejected_timers.csv, clearing any stale file
    from a prior run *first*. Without the clear, a clean re-run that drops nothing would leave the
    previous run's log on disk, where the diagnostics bundle's debug/*.csv glob would ship it as if it
    were current. Returns the path written, or None when there were no rejections (file left absent)."""
    from .paths import debug_dir

    path = debug_dir(stem) / REJECTED_TIMERS_NAME
    path.unlink(missing_ok=True)               # never let a prior run's log masquerade as this run's
    if not rejections:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    write_rejected_timers_csv(rejections, path)
    return path


_FRAMES_CSV_FIELDS = [
    "timestamp_fmt", "timer_formatted",
    "set_number", "match_number",
    "match_tournament_tag_left", "match_tournament_tag_right",
    "match_ingame_tag_left", "match_ingame_tag_right",
    "match_round", "match_best_of",
    "match_character_left", "match_character_right",
    "match_end_sec", "match_end_fmt", "match_end_synthesized",
    "frame_number", "timestamp_sec",
    "label",
    "timer_seconds", "timer_confidence",
    "frame_tournament_tag_left", "frame_tournament_tag_right",
    "frame_ingame_tag_left", "frame_ingame_tag_right",
    "frame_round", "frame_best_of",
    "frame_character_left", "frame_character_right",
    "frame_stocks_left", "frame_stocks_right",
    "all_texts",
]


def _frame_tournament_tags(frame: Event, frame_width: int, frame_height: int) -> tuple[str, str]:
    """One frame's (left, right) tournament-tag reads for the frames CSV, routed as the
    match-level vote was: from the frame's player-box reads when the run read the overlay
    from both players' boxes (a drawn layout's gamertag boxes, or the rows the side-bar
    pre-pass locked: event["layout_reads"], see tags.attach_overlay_regions), else from the
    position heuristics over the frame's texts (tags.find_tournament_tags). A flagless heuristic read
    can still emit a fixed-slot pair with a None side (the lone-tag salvage): the slot
    positions are kept, so a salvaged right-only read lands in the right column."""
    reads = frame.get("layout_reads") or {}
    # BOTH player boxes, as the vote requires (OverlayRegions.reads_player_boxes): a layout
    # with one gamertag box drawn votes the tags from the heuristics, so the columns do too.
    if REGION_PLAYER1 in reads and REGION_PLAYER2 in reads:
        return (reads[REGION_PLAYER1]["text"], reads[REGION_PLAYER2]["text"])
    pair = find_tournament_tags(frame.get("texts", []), frame_width=frame_width,
                                frame_height=frame_height) or [None, None]
    return (pair[0]["text"] if pair[0] else "", pair[1]["text"] if pair[1] else "")


def _frame_nameplates(frame: Event, layout: CustomOverlayLayout | None, ingame_band: str,
                      frame_width: int, frame_height: int,
                      ingame_tag_frac: float) -> tuple[str, str]:
    """The (left, right) in-game nameplate columns for one frame, routed exactly as the
    match-level vote routes them (tags.enrich_match): from the layout's drawn nameplate boxes
    when the run votes those (tags.nameplates_from_boxes; the band is then never searched, and
    on such a run the pipeline never even OCR'd it, see overlay_bands.band_ocr_plan); nothing
    when the band is read as nothing (INGAME_BAND_NONE leaves it unread, drawn boxes or not);
    else by re-running the band search over the frame's texts (find_player_names)."""
    if nameplates_from_boxes(layout, ingame_band):
        assert layout is not None   # nameplates_from_boxes implies a layout
        reads = frame.get("layout_reads") or {}
        # Per side of the layout's own boxes, mirroring attach_layout_ingame's loop: an
        # undrawn side is simply not read (the pipeline never writes a read for it).
        left = reads.get(REGION_INGAME_P1) if layout.ingame_p1 is not None else None
        right = reads.get(REGION_INGAME_P2) if layout.ingame_p2 is not None else None
        return (left["text"] if left else "", right["text"] if right else "")
    if ingame_band == INGAME_BAND_NONE:
        return ("", "")
    found = find_player_names(frame.get("texts", []), frame_width=frame_width,
                              frame_height=frame_height, ingame_tag_frac=ingame_tag_frac) or []
    return (found[0]["text"] if found else "",
            found[1]["text"] if len(found) > 1 else "")


def write_frames_csv(matches: list[Match], path: str,
                     set_lookup: dict[int, int] | None = None,
                     frame_width: int = 1920, frame_height: int = 1080,
                     ingame_tag_frac: float = INGAME_TAG_BAND_FRAC_DEFAULT,
                     layout: CustomOverlayLayout | None = None,
                     ingame_band: str = INGAME_BAND_TAGS) -> None:
    """Write one row per frame per match for pandas inspection.
    Per-frame columns: raw detections and timer readings.
    Match-level columns: voted tags/round/best-of and end timestamp.

    `ingame_tag_frac` must be the run's resolved in-game nametag band (see
    tags.INGAME_TAG_BAND_FRAC_DEFAULT): the per-frame ingame-tag columns re-run
    find_player_names, so gating them at the default 0.92 while a widened run voted the
    match-level columns at a lower band would make the two disagree for exactly the shrunk
    captures the band exists to serve. `layout` (the custom overlay layout the run read
    with) and `ingame_band` (what the game's nameplate band holds this run, one of
    games.INGAME_BAND_MODES, as resolved for the vote) are threaded for the same reason:
    the per-frame nameplate columns are routed exactly as the vote was, from the layout's
    drawn boxes, from the band search, or from nothing (see _frame_nameplates)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FRAMES_CSV_FIELDS)
        writer.writeheader()
        for match in matches:
            m_ttags = match.get("tournament_tags") or []
            m_itags = match.get("ingame_tags") or []
            m_chars = match.get("characters") or []
            m_end   = match["end"]["timestamp_sec"]
            for frame in match["frames"]:
                texts   = frame.get("texts", [])
                timer   = frame.get("timer")
                chars   = frame.get("characters")
                char_l  = chars.get("left") if chars else None
                char_r  = chars.get("right") if chars else None
                # Stocks left per player, from a stock-icon read (stock_icons.detect_stock_icons):
                # only when exactly two HUD slots were seen, the same rule that fills the
                # character read; blank on portrait-detector runs, which record no slots.
                slots   = frame.get("stock_slots") or []
                stocks_l, stocks_r = ((slots[0]["stocks"], slots[1]["stocks"])
                                      if len(slots) == 2 else ("", ""))
                t_tag_l, t_tag_r = _frame_tournament_tags(frame, frame_width, frame_height)
                i_tag_l, i_tag_r = _frame_nameplates(frame, layout, ingame_band,
                                                     frame_width, frame_height, ingame_tag_frac)
                rnd     = find_round(texts, frame_height)
                best_of = find_best_of(texts, frame_height)
                ts      = frame.get("timestamp_sec", 0)
                writer.writerow({
                    "set_number":                 set_lookup.get(match["match_number"], "") if set_lookup else "",
                    "match_number":               match["match_number"],
                    "match_tournament_tag_left":  m_ttags[0] if m_ttags else "",
                    "match_tournament_tag_right": m_ttags[1] if len(m_ttags) > 1 else "",
                    "match_ingame_tag_left":      m_itags[0] if m_itags else "",
                    "match_ingame_tag_right":     m_itags[1] if len(m_itags) > 1 else "",
                    "match_round":                match.get("round") or "",
                    "match_best_of":              match.get("best_of") or "",
                    "match_character_left":       (m_chars[0] or "") if m_chars else "",
                    "match_character_right":      (m_chars[1] or "") if len(m_chars) > 1 else "",
                    "match_end_sec":              m_end,
                    "match_end_fmt":              format_timestamp(m_end),
                    "match_end_synthesized":      match["end"].get("synthesized", False),
                    "frame_number":               frame.get("frame_number", ""),
                    "timestamp_sec":              ts,
                    "timestamp_fmt":              format_timestamp(ts),
                    "label":                      frame.get("label", ""),
                    "timer_formatted":            timer["formatted"] if timer else "",
                    "timer_seconds":              timer["total_seconds"] if timer else "",
                    "timer_confidence":           timer.get("confidence", "") if timer else "",
                    "frame_tournament_tag_left":  t_tag_l,
                    "frame_tournament_tag_right": t_tag_r,
                    "frame_ingame_tag_left":      i_tag_l,
                    "frame_ingame_tag_right":     i_tag_r,
                    "frame_round":                rnd["text"] if rnd else "",
                    "frame_best_of":              best_of["text"] if best_of else "",
                    "frame_character_left":       char_l["name"] if char_l else "",
                    "frame_character_right":      char_r["name"] if char_r else "",
                    "frame_stocks_left":          stocks_l,
                    "frame_stocks_right":         stocks_r,
                    "all_texts":                  "|".join(
                                                      f"{t['text']}({t.get('confidence', 0):.2f})"
                                                      for t in texts
                                                  ),
                })
