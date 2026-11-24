"""
Extract Matches from Pipeline JSON
-----------------------------------
Reads a vodparse.cli.pipeline output JSON and groups events into matches.
A match begins at a "go" detection and ends at the next "game" detection.
Multiple consecutive "go" detections before a "game" overwrite the start.

Modes:
  tournament_set: group consecutive matches with the same players into one clip (default)
  match: export one clip per match

Clips are exported to video_out/ by default. The source video comes from --video,
or falls back to the video_path recorded in the pipeline JSON config. Pass --csv-only
to skip clip export and only write the matches JSON/CSV.

Usage:
    python -m vodparse.cli.extract_matches --input json_output/my_vod/pipeline.json
    python -m vodparse.cli.extract_matches --input json_output/my_vod/pipeline.json --csv-only
    python -m vodparse.cli.extract_matches --input json_output/my_vod/pipeline.json --video other/vods/my_vod.mp4 --mode tournament_set
    python -m vodparse.cli.extract_matches --input json_output/my_vod/pipeline.json --csv-only --smart-tags
"""

import argparse
import contextlib
import io
from functools import partial
from pathlib import Path

import cv2

from vodparse.cli.stage_io import prepare_stage_streams
from vodparse.debug import write_frames_csv
from vodparse.export import (export_basename, export_match_clips, export_set_clips,
                             merge_vod_rows, other_vod_filenames, print_kept_vods,
                             read_export_csv_or_exit, write_csv_or_exit)
from vodparse.extract_flags import EXTRACT_FLAGS_NAME, build_sidecar
from vodparse.games import (DEFAULT_MAX_GAME_COUNT, INGAME_BAND_CHARACTER, INGAME_BAND_NONE,
                            INGAME_BAND_TAGS, Game, default_game, full_game_seconds_for,
                            game_by_id, load_roster)
from vodparse.grouping import (confirmed_identity_count, fold_game_counts, group_into_sets,
                               is_impostor_set, refine_set_tags)
from vodparse.jsonio import read_json, write_json_atomic
from vodparse.schema import Event, InferenceSettings, Match, PipelineConfig, SetGroup
from vodparse.match_extraction import (FULL_GAME_SECONDS, IMPLAUSIBLE_TIMERS_NAME,
                                          extract_matches, extract_matches_timer_only,
                                          split_on_timer_reset)
from vodparse.overlay_regions import OverlayRegions, layout_overlay_regions
from vodparse.overlay_layout import (CustomOverlayLayout, ingame_band_mode_for,
                                     layout_from_dict)
from vodparse.paths import debug_dir, output_dir, video_out_dir
from vodparse.smart_tags import (DEFAULT_SCORERS, SMART_TAGS_REPORT_NAME, FlagCombo,
                                    build_report, format_ranking_table, run_sweep)
from vodparse.tags import (INGAME_TAG_BAND_FRAC_DEFAULT, enrich_matches, find_tournament_tags,
                           learn_tag_row, parse_ingame_tag_band_frac)
from vodparse.titles import add_title_args
from vodparse.name_filters import strip_never_name_edges_in_events
from vodparse.word_merge import merge_doctr_reads

# What the "Game:" console line adds for each way the in-game nameplate band can be read
# (overlay_layout.ingame_band_mode_for), so a run's output says where its tags and
# characters came from. Ultimate's default (player tags) is the historic behaviour and
# needs no note.
INGAME_BAND_NOTES: dict[str, str] = {
    INGAME_BAND_TAGS: "",
    INGAME_BAND_NONE: " (ingame band unread; overlay tags only)",
    INGAME_BAND_CHARACTER: " (ingame band read as character names)",
}

# What the same line adds when the run's characters came from a stock-icon detector
# (games.GameWeights.stock_icons, recorded as inference_settings.stock_model_path).
STOCK_ICONS_NOTE = " (characters read from stock icons)"


def characters_note(config: PipelineConfig | None) -> str:
    """The "Game:" line's note on where the run's characters came from, beyond the band
    note: the stock-icon note when the run recorded a stock detector, else nothing."""
    if config is None:
        return ""
    settings = config.get("inference_settings") or {}   # a config may predate the key
    return STOCK_ICONS_NOTE if settings.get("stock_model_path") else ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        required=True,                          help="Path to pipeline JSON")
    parser.add_argument("--video",        default=None,                           help="Source video to slice clips from (default: video_path from the pipeline JSON config)")
    parser.add_argument("--csv-only",     action="store_true",                    help="Skip clip export; only write the matches JSON/CSV")
    parser.add_argument("--output",       default=None,                           help="Output path (default: json_output/<stem>/matches.json)")
    parser.add_argument("--mode",         default="tournament_set",               help="tournament_set or match (default: tournament_set)",
                        choices=["match", "tournament_set"])
    parser.add_argument("--group-by",     default="auto",                         help="Which tags identify players when grouping games into sets: auto (prefer whichever source read both players, overlay before ingame, then lone tags), overlay (tournament-overlay gamertags only), or ingame (ingame nameplates only). Only applies to tournament_set mode",
                        choices=["auto", "overlay", "ingame"])
    parser.add_argument("--set-characters", default="join",                       help="A set's per-player character when their games disagree: join every distinct character ('Char1 / Char2') or keep the majority. Only applies to tournament_set mode",
                        choices=["join", "majority"])
    add_title_args(parser, limit_default=100, limit_default_help="100",
                   filenames_default_help="on for tournament_set mode, off for match mode")
    parser.add_argument("--sponsors",     action=argparse.BooleanOptionalAction, default=True,
                        help="Prefix each player's sponsor to their tag in clip titles (\"SPONSOR | TAG\"). Sponsors stay in the CSV either way")
    parser.add_argument("--start-buffer", type=float, default=15.0,               help="Seconds before game/set start (default: 15)")
    parser.add_argument("--end-buffer",   type=float, default=20.0,               help="Seconds after game/set end (default: 20)")
    parser.add_argument("--folder",       default=None,                           help="Output subfolder name under video_out/ (default: <vod>_<games|sets>)")
    parser.add_argument("--out-root",     type=Path, default=None,                help="Directory that takes the place of video_out/ for exported clips (default: video_out/ under the current directory); --folder names a subfolder inside it")
    parser.add_argument("--tournament-name", default="",                          help="Tournament name prefixed to each clip title")
    parser.add_argument("--no-overlay-filter", action="store_true",                help="Disable the static-overlay filter (keeps moving ingame text in the vote)")
    parser.add_argument("--filter-banners", action="store_true",                     help="Strip persistent stream graphics (a banner at the same spot across many games) from tag voting")
    parser.add_argument("--overlay-at-bottom", action="store_true",                   help="Read the tournament overlay's gamertags + sponsors from the bottom of the frame instead of the top, for tournaments that render them over the ingame HUD. Round and best-of are still read from the top")
    parser.add_argument("--overlay-stacked", action="store_true",                     help="Read each gamertag from the line below its metadata, for overlays that stack a location / Twitter handle / seed above the gamertag (otherwise the metadata line is mistaken for the tag)")
    parser.add_argument("--smart-tags",     action="store_true",                     help="Try every combination of the four tag-reading flags (--overlay-stacked, --no-overlay-filter, --overlay-at-bottom, --filter-banners) over the same analysis and keep the best-scoring result, writing the ranking to json_output/<stem>/smart_tags_report.json. Cannot be combined with setting those flags manually")
    parser.add_argument("--ingame-tag-band-frac", type=parse_ingame_tag_band_frac, default=None,
                        help="Override the ingame nametag OCR-band top edge (fraction of height) "
                             "for reading the nameplates. Normally read back from the pipeline run "
                             "(so the gate matches the crop that was OCR'd); set it here only to "
                             "re-gate an existing analysis. Widening it beyond what the pipeline "
                             "cropped can't recover text that was never OCR'd; re-run the pipeline "
                             "with its own --ingame-tag-band-frac for that")
    parser.add_argument("--debug",          action="store_true",                     help="Print and save (json_output/<stem>/debug/) the frame/segment/game DataFrames from timer grouping")
    return parser


def load_pipeline(path: str | Path) -> tuple[PipelineConfig | None, list[Event]]:
    """Read a pipeline JSON and return its (config, events). config is None for
    payloads without a "config" key. For a docTR run, events are re-merged from
    docTR's word-level boxes into EasyOCR-style phrase boxes (see word_merge). Every
    run's phrases then lose the never-name words at their edges: a pronoun pill read
    inside a tag's phrase ("HE/HIM PlayerOne" -> "PlayerOne", see name_filters)."""
    data = read_json(path)
    if not isinstance(data, dict) or "events" not in data:
        raise SystemExit(f"{path}: not a pipeline JSON (missing 'events'); "
                         "pass the vodparse.cli.pipeline output file.")
    config: PipelineConfig | None = data.get("config")
    # docTR emits one OCR box per WORD, but the tag/filter/sponsor logic downstream is shaped
    # for EasyOCR's phrase-level boxes. Normalize here; the one seam every extraction path
    # (CLI replay, GUI re-extract, the tag sweeps, the row learner) loads events through; so
    # stored docTR runs replay correctly without re-OCR. pipeline.json on disk keeps the raw
    # word-level reads as provenance.
    events = merge_doctr_reads(data["events"], config)
    # A pill drawn flush against a tag is now inside the tag's phrase, whichever engine read
    # it: strip it from the phrase's edge (the one place for this, see name_filters).
    events = strip_never_name_edges_in_events(events)
    return config, events


def resolve_video(video_arg: str | None, config: PipelineConfig | None,
                  input_json_dir: Path) -> str | None:
    """The source video for clip export: --video if given, else the
    video_path recorded in the pipeline JSON config, else None.

    A stored relative path is relative to the cwd where pipeline ran, which is usually
    the --input JSON's own directory. If it doesn't resolve from the current cwd, fall
    back to resolving it against input_json_dir so extract_matches works regardless of
    where it's launched from. Note the precedence: a path that exists relative to the
    cwd is used as-is, even if one also exists under input_json_dir; run from the
    same directory as the JSON to be sure."""
    raw = video_arg or (config.get("video_path") if config else None)
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute() and not path.exists():
        candidate = input_json_dir / path
        if candidate.exists():
            return str(candidate)
    return raw


def resolve_sample_interval(config: PipelineConfig | None) -> float:
    """The OCR/seek cadence (seconds) used to back-calculate match starts. Seek-mode
    runs sample on a seek_every clock; periodic runs use ocr_interval_sec."""
    # Single-arg .get (no {} default) so mypy doesn't check an empty dict against the total
    # InferenceSettings TypedDict; `or {}` keeps the original defensiveness for a config that
    # predates inference_settings.
    infer: InferenceSettings | dict[str, float] = (
        config.get("inference_settings") or {}) if config else {}
    if config and config.get("mode") == "seek":
        return infer.get("seek_every", 30.0)
    return infer.get("ocr_interval_sec", 30.0)


def build_matches(events: list[Event], config: PipelineConfig | None, sample_interval: float, *,
                  debug: bool, stem: str = "",
                  full_game_seconds: float = FULL_GAME_SECONDS) -> list[Match]:
    """Group events into numbered raw matches: timer-only grouping for seek-mode runs,
    standard go/game grouping otherwise, then a timer-reset split pass. `full_game_seconds`
    is the match clock's length at game start (Ultimate's 7:00; a custom overlay layout's own
    value for other games), which the clock math validates readings against and back-dates
    game starts from. The split pass may demand a detected fighter before cutting across a
    clock-less break only when the recorded run loaded a character detector (see
    match_extraction._timer_reset_indices, has_character_detector); a run without one has no
    such evidence."""
    fighter_check = has_character_detector(config)
    if config and config.get("mode") == "seek":
        print("Pipeline was in seek mode → using timer-only grouping")
        raw = extract_matches_timer_only(events, sample_interval=sample_interval, debug=debug,
                                         stem=stem, stage="timer_only",
                                         full_game_seconds=full_game_seconds,
                                         fighter_check=fighter_check)
    else:
        raw = extract_matches(events)
    # Each match may split into several games at timer resets; flatten in order.
    split: list[Match] = []
    for i, m in enumerate(raw):
        split.extend(split_on_timer_reset(m, sample_interval=sample_interval, debug=debug,
                                          stem=stem, stage=f"split{i}",
                                          full_game_seconds=full_game_seconds,
                                          fighter_check=fighter_check))
    raw = split

    for i, m in enumerate(raw):
        m["match_number"] = i + 1
    return raw


def dump_frames_csv(matches: list[Match], csv_dir: Path, stage: str,
                    set_lookup: dict[int, int] | None = None,
                    frame_width: int = 1920, frame_height: int = 1080,
                    ingame_tag_frac: float = INGAME_TAG_BAND_FRAC_DEFAULT,
                    layout: CustomOverlayLayout | None = None,
                    ingame_band: str = INGAME_BAND_TAGS) -> None:
    """Write the per-frame inspection CSV for one pipeline stage and log its path.
    `ingame_tag_frac` is the run's resolved band so the per-frame ingame-tag columns match the
    crop this run OCR'd (see resolve_ingame_tag_band_frac / write_frames_csv); `layout` and
    `ingame_band` are the run's custom overlay layout and band mode, so those columns are
    routed (boxes / band search / nothing) exactly as the vote was."""
    path = csv_dir / f"frames_{stage}.csv"
    write_frames_csv(matches, str(path), set_lookup=set_lookup,
                     frame_width=frame_width, frame_height=frame_height,
                     ingame_tag_frac=ingame_tag_frac, layout=layout, ingame_band=ingame_band)
    print(f"  Frames CSV ({stage}) → {path}")


def resolve_resolution(config: PipelineConfig | None, video: str | None) -> tuple[int, int]:
    """Source-video (width, height) the tag-position thresholds are measured against. Prefer the
    pipeline config, else read it off the video, else fall back to 1080p. A wrong resolution makes
    the lower-/upper-third and center-x cutoffs miss tags entirely (e.g. ingame tags on a 720p vod)."""
    if config and config.get("frame_width") and config.get("frame_height"):
        return int(config["frame_width"]), int(config["frame_height"])
    if video and Path(video).exists():
        cap = cv2.VideoCapture(video)
        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        if w and h:
            return w, h
    print("  Warning: could not determine video resolution; assuming 1920x1080")
    return 1920, 1080


def resolve_ingame_tag_band_frac(config: PipelineConfig | None, override: float | None) -> float:
    """The in-game nametag OCR-band top edge (fraction of height) this extract should gate on.
    An explicit --ingame-tag-band-frac wins; otherwise use the value the pipeline recorded (so
    the gate matches the crop that run actually OCR'd); otherwise the fixed default. Reading it
    back from the config is what keeps the two stages from drifting; a widened crop is useless
    if the gate here still rejects the boxes it produced (see tags.INGAME_TAG_BAND_FRAC_DEFAULT)."""
    if override is not None:
        return override
    # Single-arg .get + `or {}` mirrors resolve_sample_interval: tolerate a config predating
    # this key (older pipeline JSONs) by falling back to the default.
    infer: InferenceSettings | dict[str, float] = (
        config.get("inference_settings") or {}) if config else {}
    return infer.get("ingame_tag_band_frac", INGAME_TAG_BAND_FRAC_DEFAULT)


def resolve_overlay_regions(config: PipelineConfig | None, layout: CustomOverlayLayout | None,
                            frame_width: int, frame_height: int) -> OverlayRegions | None:
    """The overlay text regions the pipeline run read per sample (see
    vodparse.overlay_regions), or None for the built-in reading. Read back from the
    pipeline JSON's ``overlay_regions`` record when the run wrote one (a layout's boxes or
    the side-bar lock's rows); a run older than that record but with a layout derives the
    regions from the layout it recorded, so its votes are unchanged. Like the layout, never
    taken from the GUI's current selection: only the run that OCR'd the boxes knows them."""
    infer: InferenceSettings | dict[str, object] = (
        config.get("inference_settings") or {}) if config else {}
    record = infer.get("overlay_regions")
    if record is not None:
        regions = OverlayRegions.from_config(record)
        if regions is not None:
            return regions
        print("  Warning: ignoring the pipeline's overlay_regions record (unreadable); "
              "deriving the regions from the layout, if any")
    if layout is not None:
        return layout_overlay_regions(layout, frame_width, frame_height)
    return None


def resolve_overlay_layout(config: PipelineConfig | None) -> CustomOverlayLayout | None:
    """The custom overlay layout the pipeline run read with, or None for the built-in
    reading. Read back from the pipeline JSON (never from the GUI's current selection),
    like the nametag band fraction: the layout's boxes are only meaningful for the run
    that OCR'd them, and a re-read must vote the reads that run actually made. A record
    this build can't parse (a future shape) is treated as no layout, with a note."""
    infer: InferenceSettings | dict[str, object] = (
        config.get("inference_settings") or {}) if config else {}
    record = infer.get("overlay_layout")
    if not isinstance(record, dict):
        return None
    try:
        return layout_from_dict(record)
    except (ValueError, KeyError, TypeError) as err:
        print(f"  Warning: ignoring the pipeline's overlay_layout record ({err}); "
              "reading the overlay the built-in way")
        return None


def has_character_detector(config: PipelineConfig | None) -> bool:
    """Whether the run's frames can carry a detected fighter at all: the evidence the
    splitter's intermission cut may ask for (match_extraction._timer_reset_indices). Read
    from the detectors the pipeline recorded as LOADED, since a declared weight that was
    missing degrades to no character read: the portrait detector (Ultimate,
    ``character_model_path``) or the stock-icon detector (Melee, ``stock_model_path``).

    The stock-icon read fills a frame's ``characters`` only when it sees exactly two stock
    slots (vodparse.stock_icons), so a doubles or free-for-all VOD would never show a
    fighter and every intermission cut would be refused: the check is right for singles
    only, for now.

    A config recording neither key (a run older than the records) falls back to the recorded
    game's declared character weight, the rule extract used before. An older Ultimate run
    splits as it did (it recorded its declared character path whether or not it loaded).
    A Melee run analyzed with the stock-icon detector already recorded its stock path, so
    re-extracting it now asks for a fighter where it used to skip the check."""
    infer: InferenceSettings | dict[str, object] = (
        config.get("inference_settings") or {}) if config else {}
    if "character_model_path" not in infer and "stock_model_path" not in infer:
        game = resolve_game(config)
        return game.weights is not None and game.weights.character is not None
    return bool(infer.get("character_model_path") or infer.get("stock_model_path"))


def resolve_game(config: PipelineConfig | None) -> Game:
    """The game the pipeline run was analyzed as. Read back from the pipeline JSON (never from
    the GUI's current selection), like the overlay layout and the nametag band: the events were
    produced under that game's profile, and re-reading them under another one would describe a
    run that never happened. Conservative on anything unexpected: a config predating the key
    (every such run was Ultimate) or naming a game this build doesn't have resolves to the
    default game, the second with a note."""
    infer: InferenceSettings | dict[str, object] = (
        config.get("inference_settings") or {}) if config else {}
    default = default_game()
    recorded = infer.get("game")
    if recorded is None:
        return default
    game = game_by_id(recorded) if isinstance(recorded, str) else None
    if game is None:
        print(f"  Warning: this analysis names a game this build doesn't have ({recorded!r}); "
              f"reading it as {default.display_name}")
        return default
    return game


def bottom_band_was_read(config: PipelineConfig | None) -> bool:
    """Whether the pipeline run OCR'd the bottom overlay band, read back from the pipeline
    JSON like the game and the layout. The band plan (overlay_bands.band_ocr_plan) retires
    that band when the run reads its in-game band as nothing or reads drawn nameplate boxes
    instead, and then the at-bottom tournament-overlay reading (--overlay-at-bottom, and the
    smart-tags combinations that set it) has no band text to read. A config predating the
    key (every such run read both bands) counts as read."""
    infer: InferenceSettings | dict[str, object] = (
        config.get("inference_settings") or {}) if config else {}
    recorded = infer.get("band_ocr_bottom")
    return recorded if isinstance(recorded, bool) else True


def run_notes(layout: CustomOverlayLayout | None, overlay_regions: OverlayRegions | None,
              game: Game, full_game_seconds: float) -> list[str]:
    """The console lines that say how this run reads its overlay and its clock, in print
    order. A layout names itself and carries its own clock, so it needs no separate clock
    note. Without one, locked regions (the side-bar pre-pass, vodparse.side_overlay) are
    named, and a clock that isn't the default 7:00 (Ultimate's) is called out, since it came from
    the game's own profile (e.g. Melee's 8:00) and 7:00 is otherwise the silent assumption."""
    lines: list[str] = []
    if layout is not None:
        boxes_note = " and its nameplate boxes" if layout.reads_ingame_boxes else ""
        lines.append(f"  Custom overlay layout: {layout.name!r} (voting its box reads{boxes_note}; "
                     f"game clock {layout.clock_seconds // 60}:{layout.clock_seconds % 60:02d})")
        return lines
    if overlay_regions is not None:
        # Regions without a layout: the side-bar pre-pass locked them.
        lines.append(f"  Overlay regions ({overlay_regions.source}): voting the reads of "
                     f"{', '.join(overlay_regions.boxes)}")
    if full_game_seconds != FULL_GAME_SECONDS:
        clock_secs = int(full_game_seconds)
        lines.append(f"  Game clock: {clock_secs // 60}:{clock_secs % 60:02d} "
                     f"({game.display_name}'s full clock, not the default 7:00)")
    return lines


def smart_tags_combos(bottom_band_read: bool) -> list[FlagCombo]:
    """The flag combinations the smart-tags sweep tries: all of them, less the at-bottom
    ones when the run never OCR'd the bottom band (see bottom_band_was_read), since those
    would vote on nothing and only add runs to rank."""
    combos = FlagCombo.all_combos()
    if bottom_band_read:
        return combos
    return [combo for combo in combos if not combo.overlay_at_bottom]


def smart_enrich(raw: list[Match], args: argparse.Namespace, stem: str,
                 frame_width: int, frame_height: int,
                 ingame_tag_frac: float, *,
                 bottom_band_read: bool,
                 max_game_count: int = DEFAULT_MAX_GAME_COUNT) -> tuple[list[Match], FlagCombo]:
    """Run the smart-tags sweep (see vodparse.smart_tags) and adopt its winner: replay the
    winner's captured enrichment notes verbatim; so a smart run prints exactly what a plain
    run with the winning flags would; then print the full ranking and persist the report
    next to matches.json. Returns the winner's matches and its flag combo (which drives the
    set-tag refinement in export_clips). Set grouping for the ranking uses the run's
    --group-by; match mode scores with "auto", matching how the Review screen resolves
    identity there. `bottom_band_read` is bottom_band_was_read's answer for the run: without
    the bottom band the at-bottom combinations are left out (smart_tags_combos).
    `max_game_count` is the run's game's highest game count, for the ranking's grouping."""
    group_by = args.group_by if args.mode == "tournament_set" else "auto"
    runs = run_sweep(raw, group_by=group_by, set_characters=args.set_characters,
                     frame_width=frame_width, frame_height=frame_height,
                     combos=smart_tags_combos(bottom_band_read),
                     ingame_tag_frac=ingame_tag_frac, max_game_count=max_game_count)
    winner = runs[0]
    print(winner.notes, end="")
    print("Smart tag reading: ranked flag combinations (best first):")
    print(format_ranking_table(runs, DEFAULT_SCORERS))
    print(f"Smart tags picked: {winner.combo.label()}")
    if winner.promotion:
        print(f"  {winner.promotion}")
    report_path = output_dir(stem) / SMART_TAGS_REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_path, build_report(runs, DEFAULT_SCORERS))
    print(f"  Ranking report → {report_path}")
    return winner.matches, winner.combo


def warn_impostor_sets(sets: list[SetGroup]) -> None:
    """Warn when a grouped set spans implausibly many confirmed player pairs (see
    grouping.is_impostor_set): the signature of a stream graphic elected as a gamertag
    gluing different matches into one set. --smart-tags avoids such a result on its own
    (its first scorer minimizes exactly this), so the warning mostly reaches manual-flag
    runs, where the fix is a different overlay flag or grouping source."""
    for s in sets:
        if is_impostor_set(s["matches"]):
            print(f"  Warning: set {s['set_number']} spans "
                  f"{confirmed_identity_count(s['matches'])} confirmed player pairs across "
                  f"{len(s['matches'])} games. Its grouping identity {s.get('grouping_tags')} "
                  "looks like a stream graphic, not a player: try --overlay-at-bottom, "
                  "--group-by ingame, or --smart-tags")


def print_summary(matches: list[Match], output_path: str) -> None:
    print(f"{len(matches)} game(s) extracted → {output_path}")
    for m in matches:
        print(f"  Game {m['match_number']}: {m['start']['timestamp_sec']:.2f}s → "
              f"{m['end']['timestamp_sec']:.2f}s  ({m['duration_sec']:.1f}s)"
              f"  ingame_tags={m['ingame_tags']}")


def export_clips(matches: list[Match], args: argparse.Namespace, video: str, stem: str, csv_dir: Path,
                 frame_width: int = 1920, frame_height: int = 1080, *,
                 overlay_at_bottom: bool, overlay_stacked: bool,
                 ingame_tag_frac: float = INGAME_TAG_BAND_FRAC_DEFAULT,
                 layout: CustomOverlayLayout | None = None,
                 ingame_band: str = INGAME_BAND_TAGS,
                 max_game_count: int = DEFAULT_MAX_GAME_COUNT) -> None:
    """Slice video clips for each match (or tournament set) and write the clip metadata CSV
    (shared with other VODs of the same tournament; only this VOD's rows are replaced).
    `overlay_at_bottom`/`overlay_stacked` are the *resolved* overlay-layout flags driving the
    set-tag refinement: the winner's on a --smart-tags run, the CLI flags otherwise; so they
    are passed explicitly rather than read off `args`. `ingame_tag_frac` is the run's resolved
    in-game nametag band, threaded (with the run's `layout` and `ingame_band`) to the "sets"
    frames CSV so its per-frame columns match what this run OCR'd and voted (see
    dump_frames_csv). `max_game_count` is the run's game's highest game count, for the set
    grouping's count fold (games.Game.max_game_count)."""
    base_name   = export_basename(args.tournament_name, stem, args.mode != "match")
    folder_name = args.folder or base_name
    clip_dir    = video_out_dir(folder_name, args.out_root)
    clip_dir.mkdir(parents=True, exist_ok=True)
    csv_path    = str(clip_dir / f"{base_name}.csv")
    # Shared with any other VOD exported under this tournament name: read it before cutting,
    # then replace only this VOD's rows (see vodparse.export.merge_vod_rows).
    existing_rows = read_export_csv_or_exit(Path(csv_path))
    taken_names   = other_vod_filenames(existing_rows, stem)

    if args.mode == "match":
        title_filenames = False if args.title_filenames is None else args.title_filenames
        csv_rows = export_match_clips(matches, video, stem, clip_dir,
                                      args.start_buffer, args.end_buffer,
                                      tournament_name=args.tournament_name,
                                      title_limit=args.title_limit,
                                      title_filenames=title_filenames,
                                      use_sponsors=args.sponsors,
                                      taken_names=taken_names)
    else:
        sets = group_into_sets(matches, args.group_by, set_characters=args.set_characters,
                               overlay_at_bottom=overlay_at_bottom,
                               max_game_count=max_game_count)
        warn_impostor_sets(sets)
        # Recover a gamertag whose sponsor box out-lengths it in every game (see refine_set_tags);
        # the finder reads the overlay with this VOD's resolution and layout flags: including
        # the learned at-bottom tag row, so the refinement re-reads the same row the vote used
        # (learn_tag_row is deterministic, so this recomputes the enrichment's value).
        row = learn_tag_row(matches, frame_width, frame_height) if overlay_at_bottom else None
        finder = partial(find_tournament_tags, frame_width=frame_width, frame_height=frame_height,
                         at_bottom=overlay_at_bottom, stacked=overlay_stacked,
                         tag_row_y=row.y if row is not None else None)
        refine_set_tags(sets, finder)
        set_lookup = {m["match_number"]: s["set_number"] for s in sets for m in s["matches"]}
        dump_frames_csv(matches, csv_dir, "sets", set_lookup=set_lookup,
                        frame_width=frame_width, frame_height=frame_height,
                        ingame_tag_frac=ingame_tag_frac, layout=layout, ingame_band=ingame_band)
        title_filenames = True if args.title_filenames is None else args.title_filenames
        csv_rows = export_set_clips(sets, video, stem, clip_dir,
                                    args.start_buffer, args.end_buffer,
                                    tournament_name=args.tournament_name,
                                    title_limit=args.title_limit,
                                    title_filenames=title_filenames,
                                    use_sponsors=args.sponsors,
                                    taken_names=taken_names)

    write_csv_or_exit(merge_vod_rows(existing_rows, csv_rows, stem), csv_path)
    print(f"  CSV:   {csv_path}")
    print_kept_vods(existing_rows, stem)


def main() -> None:
    prepare_stage_streams()

    parser = build_parser()
    args = parser.parse_args()
    manual = [flag for flag, on in (("--overlay-stacked",   args.overlay_stacked),
                                    ("--no-overlay-filter", args.no_overlay_filter),
                                    ("--overlay-at-bottom", args.overlay_at_bottom),
                                    ("--filter-banners",    args.filter_banners)) if on]
    if args.smart_tags and manual:
        # Reject rather than silently override: the manual flag says the user knows this
        # VOD's layout, and the sweep might contradict it. (Pinning one flag while
        # sweeping the rest is possible future work.)
        parser.error(f"--smart-tags chooses the tag-reading flags automatically; "
                     f"drop {', '.join(manual)}")

    config, events = load_pipeline(args.input)
    sample_interval = resolve_sample_interval(config)
    video = resolve_video(args.video, config, Path(args.input).resolve().parent)
    frame_width, frame_height = resolve_resolution(config, video)
    ingame_tag_frac = resolve_ingame_tag_band_frac(config, args.ingame_tag_band_frac)
    layout = resolve_overlay_layout(config)
    overlay_regions = resolve_overlay_regions(config, layout, frame_width, frame_height)
    game = resolve_game(config)
    # What the game's own in-game nameplate band holds this run (the layout's choice, else the
    # game's default; see overlay_layout.ingame_band_mode_for): Ultimate's band carries the
    # players' in-game tags, Rivals 2's the characters' names, and a "none" band is left
    # unread, the stream overlay carrying the tags alone.
    ingame_band = ingame_band_mode_for(game, layout)
    bottom_band_read = bottom_band_was_read(config)
    smart_tags = args.smart_tags
    full_game_seconds = float(full_game_seconds_for(
        game, layout.clock_seconds if layout is not None else None))
    print(f"  Game: {game.display_name}{INGAME_BAND_NOTES[ingame_band]}{characters_note(config)}")
    # The character read snaps each voted name to the game's roster (games.Game.roster, a
    # bundled JSON list); without one the raw reads ship as voted.
    roster: tuple[str, ...] = ()
    if ingame_band == INGAME_BAND_CHARACTER:
        roster = load_roster(game)
        if roster:
            print(f"  Roster: {len(roster)} character names ({game.roster}); voted names snap to it")
        else:
            print(f"  Roster: none for {game.display_name}; character names stay as read")
    for line in run_notes(layout, overlay_regions, game, full_game_seconds):
        print(line)
    if args.overlay_at_bottom and not bottom_band_read:
        # Not an error: the flag still runs, it just has nothing to read. Only a pipeline run
        # that OCR'd the bottom band gives it text to read.
        print("  Warning: --overlay-at-bottom reads tags from the bottom band, but this "
              "analysis didn't read that band (its ingame band is set to nothing, or read "
              "from drawn nameplate boxes), so it will find no tags")
    if overlay_regions is not None and smart_tags:
        # The sweep varies the tag-reading flags, which only steer the position
        # heuristics; box reads don't use them, so every combination would tie.
        print("  --smart-tags has no effect when the overlay is read from boxes; ignoring it")
        smart_tags = False

    stem = Path(args.input).parent.name
    # The timer cleaner appends to this debug file once per match within the run; clear the
    # previous run's copy so re-extracting the same vod doesn't accumulate duplicate entries.
    (debug_dir(stem) / IMPLAUSIBLE_TIMERS_NAME).unlink(missing_ok=True)
    raw = build_matches(events, config, sample_interval, debug=args.debug, stem=stem,
                        full_game_seconds=full_game_seconds)

    # Pause-aware start refinement would go here as a postprocess (opt-in, requires --video):
    # for games whose start was back-calculated from the clock, a pause before the first
    # visible reading pushes the real start earlier. See vodparse.pause_check (TimerProbe +
    # refine_start_for_pause). Not wired in yet; pauses before the first reading are rare.

    csv_dir = debug_dir(stem)
    csv_dir.mkdir(parents=True, exist_ok=True)
    dump_frames_csv(raw, csv_dir, "raw", frame_width=frame_width, frame_height=frame_height,
                    ingame_tag_frac=ingame_tag_frac, layout=layout, ingame_band=ingame_band)

    # Both arms resolve to a FlagCombo: the smart winner's, or the CLI flags as given.
    # so the sidecar and the set-tag refinement can read one object regardless of mode.
    if smart_tags and ingame_band != INGAME_BAND_TAGS:
        # The sweep scores its combinations partly on the in-game tag read, and always reads
        # that band as player tags: for a run whose band holds something else it would vote
        # on character names or on whatever the game draws there. Today such a run has a
        # layout (which turned the sweep off above), but the rule is enforced here on its
        # own so a future game with a clock detector and a different band can't slip through.
        print(f"  --smart-tags needs the ingame band read as player tags (this run reads "
              f"it as {ingame_band!r}); ignoring it")
        smart_tags = False
    if smart_tags:
        matches, combo = smart_enrich(raw, args, stem, frame_width, frame_height, ingame_tag_frac,
                                      bottom_band_read=bottom_band_read,
                                      max_game_count=game.max_game_count)
    else:
        combo = FlagCombo(overlay_stacked=args.overlay_stacked,
                          no_overlay_filter=args.no_overlay_filter,
                          overlay_at_bottom=args.overlay_at_bottom,
                          filter_banners=args.filter_banners)
        matches = enrich_matches(raw, filter_overlays=not combo.no_overlay_filter,
                                 filter_banners=combo.filter_banners,
                                 frame_width=frame_width, frame_height=frame_height,
                                 overlay_at_bottom=combo.overlay_at_bottom,
                                 overlay_stacked=combo.overlay_stacked,
                                 ingame_tag_frac=ingame_tag_frac, layout=layout,
                                 ingame_band=ingame_band, roster=roster,
                                 overlay_regions=overlay_regions)
    # A game count read into a tag ("PlayerOne 0", "1 PlayerTwo") is constant within one game,
    # so no frame vote can see it; across the games of the run it can be told from a name
    # (grouping.fold_game_counts). Folded here so matches.json carries clean per-game tags;
    # the set grouping applies the same fold on its own, which changes nothing more (the
    # fold runs to a fixed point), so a matches.json from before this fold groups alike.
    # Only digits up to the game's highest game count fold (games.Game.max_game_count).
    matches = fold_game_counts(matches, game.max_game_count)
    dump_frames_csv(matches, csv_dir, "enriched", frame_width=frame_width,
                    frame_height=frame_height, ingame_tag_frac=ingame_tag_frac, layout=layout,
                    ingame_band=ingame_band)

    output_path = args.output or str(output_dir(stem) / "matches.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, matches)

    # Record the resolved tag-reading flags next to matches.json so the Review screen's
    # "Re-read tags" editor knows what flags produced the current review. Always in
    # output_dir(stem), even when --output redirects matches.json (same as the smart report).
    flags_path = output_dir(stem) / EXTRACT_FLAGS_NAME
    flags_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(flags_path, build_sidecar(
        smart_tags=smart_tags, combo=combo, mode=args.mode,
        group_by=args.group_by, set_characters=args.set_characters, sponsors=args.sponsors))
    print(f"  Extract flags → {flags_path}")

    print_summary(matches, output_path)

    if args.csv_only:
        if args.mode == "tournament_set":
            # Group solely to check the impostor signature (the clip-export path checks
            # the sets it already builds). The grouping chatter is swallowed: no sets are
            # exported here, only the impostor warning is relevant, and this is the path
            # the GUI runs, so the warning must reach its stage log.
            with contextlib.redirect_stdout(io.StringIO()):
                sets = group_into_sets(matches, args.group_by,
                                       set_characters=args.set_characters,
                                       overlay_at_bottom=combo.overlay_at_bottom,
                                       max_game_count=game.max_game_count)
            warn_impostor_sets(sets)
        return
    if not video:
        print("  Warning: no video to export clips from (none given and no video_path in the pipeline config); use --csv-only to silence this")
        return
    if not Path(video).exists():
        print(f"  Warning: video not found at {video}; skipping clip export (use --csv-only to silence this)")
        return
    export_clips(matches, args, video, stem, csv_dir, frame_width, frame_height,
                 overlay_at_bottom=combo.overlay_at_bottom, overlay_stacked=combo.overlay_stacked,
                 ingame_tag_frac=ingame_tag_frac, layout=layout, ingame_band=ingame_band,
                 max_game_count=game.max_game_count)


if __name__ == "__main__":
    main()
