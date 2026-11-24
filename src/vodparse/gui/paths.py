"""Filesystem layout shared by the desktop GUI modules (app.py, vodparse.gui.review.screen).

Where one source video's GUI artifacts live, anchored at the writable workspace root
(the repo root in a source checkout, the per-user app-data folder when frozen; see
vodparse.runtime_paths.workspace_dir):

    <root>/json_output/<stem>/pipeline.json       (pipeline_json_for)
    <root>/json_output/<stem>/matches.json
    <root>/json_output/<stem>/review_draft.json   (draft_path / review_draft_for)

Exported clips live elsewhere: under the user-visible **export root** (see
vodparse.gui.export_root). Resolving that root needs Qt, and this module is Qt-free,
so the helpers touching clip paths (export_csv_for, last_export_info, stale_export_dir,
resolve_export_csv) take the resolved root as an explicit parameter:

    <export_root>/<export_folder>/<basename>.csv  (export_csv_for)

Alongside the layout, this module hosts the small readers/checks over those artifacts
(existing_analysis / existing_analysis_summary / analysis_settings_differences,
video_metadata_mismatches, last_export_info / resolve_export_csv, other_vods_to_confirm,
vod_clip_filenames / outdated_clip_files):
what a saved analysis or export recorded, and whether the video on disk still
matches it.

Qt-free; safe to import from anywhere.
"""

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from vodparse.cli.clock_locate import ClockLocateResult
from vodparse.export import (CSV_READ_ERRORS, export_basename, is_vod_row,
                             other_vod_clip_counts, read_export_csv)
from vodparse.games import GAME_DEFAULT, default_game, full_game_seconds_for, game_by_id
from vodparse.gui.settings import engine_display_name, lang_display_name
from vodparse.match_extraction import FULL_GAME_SECONDS
from vodparse.ocr_engine import OCR_ENGINE_DEFAULT
from vodparse.ocr_langs import OCR_LANG_DEFAULT
from vodparse.overlay_layout import AUTO_LAYOUT_NAME
from vodparse.paths import strip_converted_suffix, vod_stem
from vodparse.runtime_paths import bundle_dir, is_frozen, workspace_dir
from vodparse.text_utils import strip_featuring_suffix

PROJECT_ROOT = workspace_dir()

_DRAFT_NAME = "review_draft.json"


def vod_picker_start_dir() -> Path:
    """Where the "choose a video" / "choose a VOD folder" dialogs open when the user has
    no default VOD folder configured yet. A checkout keeps opening at the repo root, as
    before; a frozen app opens at the home folder instead, because there PROJECT_ROOT is
    the semi-hidden per-user app-data directory (see vodparse.runtime_paths.workspace_dir)
    holding only the app's own json_output/, never where a user keeps their VODs."""
    return Path.home() if is_frozen() else PROJECT_ROOT


def output_dir_for(video_path: str) -> Path:
    """The per-vod folder holding this video's pipeline/matches/reviewed JSON. Named via
    vod_stem so it matches the folder the CLI stages actually create."""
    return PROJECT_ROOT / "json_output" / vod_stem(video_path)


def pipeline_json_for(video_path: str) -> Path:
    """The deterministic pipeline-JSON path vodparse.cli.pipeline writes for a given video."""
    return output_dir_for(video_path) / "pipeline.json"


class ExistingAnalysis(NamedTuple):
    """The text-reading settings a saved pipeline.json recorded. ``ocr_interval_sec`` is None
    when the JSON records no cadence at all, but the engine, language and game always resolve:
    a run predating any of those settings could only have been EasyOCR reading English on
    Ultimate."""
    ocr_interval_sec: float | None
    ocr_engine: str   # vodparse.ocr_engine ids ("easyocr" | "doctr")
    ocr_lang: str     # vodparse.ocr_langs codes ("en" | "ja" | …)
    overlay_layout: str  # a saved preset NAME, or "" when the run used the built-in heuristics
    game: str         # vodparse.games ids ("ultimate" | "melee" | …)


def game_display_name(game_id: str) -> str:
    """The Game dropdown's label for a game id, for naming a chosen or recorded game back to
    the user. An unknown id (a pipeline.json from a build with a game this one lacks) renders
    as the raw id rather than being mislabelled, exactly like engine_display_name."""
    game = game_by_id(game_id)
    return game.display_name if game is not None else game_id


def existing_analysis(video_path: str) -> ExistingAnalysis | None:
    """What the saved pipeline.json was analyzed with, or None when there is nothing readable
    to report; surfaced on the Analyze page so 'Open existing analysis' shows what it reuses."""
    try:
        infer = json.loads(
            pipeline_json_for(video_path).read_text(encoding="utf-8"),
        )["config"]["inference_settings"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    raw_overlay_layout = infer.get("overlay_layout")
    return ExistingAnalysis(
        # Seek-mode runs sample on the seek_every clock, periodic runs on ocr_interval_sec;
        # go/game mode records seek_every as 0, so `or` falls through to the periodic cadence.
        # Mirrors extract_matches.resolve_sample_interval, which is the cadence a reuse run
        # actually re-extracts at.
        ocr_interval_sec=infer.get("seek_every") or infer.get("ocr_interval_sec"),
        # Both keys are absent from a pipeline.json written before their setting existed. Those
        # runs were EasyOCR reading English, so the defaults describe them accurately (the same
        # reading vodparse.word_merge takes of a missing ocr_engine).
        ocr_engine=infer.get("ocr_engine", OCR_ENGINE_DEFAULT),
        ocr_lang=infer.get("ocr_lang", OCR_LANG_DEFAULT),
        # The "overlay_layout" key is present only when the pipeline ran with a custom layout
        # (see cli/pipeline.py's run()); absent means either a run predating layouts or one
        # that used the built-in heuristics, and both read the same as "": no layout. The
        # isinstance guard is belt-and-braces against a hand-edited or malformed value that
        # isn't the layout_to_dict shape, so a corrupt field degrades to "" rather than raising.
        overlay_layout=(raw_overlay_layout.get("name", "")
                        if isinstance(raw_overlay_layout, dict) else ""),
        # Absent from every pipeline.json written before the Game dropdown existed; all of
        # those runs were Ultimate, which is exactly what the default says.
        game=infer.get("game", GAME_DEFAULT),
    )


def existing_analysis_summary(video_path: str) -> str:
    """The one-line note shown above the run buttons: what a saved analysis would be reused with.
    Always names the OCR cadence and the font style; names the overlay language only when it
    isn't the English default, the overlay layout only when the analysis used a custom one, and
    the game only when it isn't the default, mirroring the Analyze page's own language note
    (which likewise stays silent on English, so a named language always means a non-default
    one)."""
    saved = existing_analysis(video_path)
    if saved is None:
        return "Existing analysis (details unavailable)"
    parts = [
        f"Interval: {saved.ocr_interval_sec:g}s" if saved.ocr_interval_sec is not None
        else "unknown interval",
        f"font style: {engine_display_name(saved.ocr_engine)}",
    ]
    if saved.ocr_lang != OCR_LANG_DEFAULT:
        parts.append(f"language: {lang_display_name(saved.ocr_lang)}")
    if saved.overlay_layout:
        parts.append(f"overlay layout: {saved.overlay_layout}")
    if saved.game != GAME_DEFAULT:
        parts.append(f"game: {game_display_name(saved.game)}")
    return f"Existing analysis: {', '.join(parts)}"


def recorded_clock_seconds(video_path: str) -> float:
    """The full match clock the saved analysis was run with: mirrors the layout -> game ->
    default resolution order extract_matches.resolve_game/full_game_seconds_for use, read back
    from the same raw JSON keys (defensively: a malformed record degrades to the default rather
    than raising). The Review screen's bad-start flag back-dates game starts from clock
    readings, so it must use the same clock length the extraction did."""
    try:
        infer = json.loads(
            pipeline_json_for(video_path).read_text(encoding="utf-8"),
        )["config"]["inference_settings"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return float(FULL_GAME_SECONDS)
    recorded_game_id = infer.get("game")
    game = (game_by_id(recorded_game_id) if isinstance(recorded_game_id, str) else None
            ) or default_game()
    layout_clock_seconds: int | None = None
    record = infer.get("overlay_layout")
    if isinstance(record, dict):
        clock = record.get("clock_seconds")
        if isinstance(clock, (int, float)) and not isinstance(clock, bool) and clock > 0:
            layout_clock_seconds = int(clock)
    return float(full_game_seconds_for(game, layout_clock_seconds))


def recorded_clock(video_path: str, game_id: str) -> ClockLocateResult | None:
    """The saved analysis's calibrated clock as the Preview pane's clock-search result, so a
    VOD analyzed before is not searched again: the pipeline's ``config["clock_calibration"]``
    record (its bbox is what the run read the clock at) plus the timestamp of the first
    event that carried a clock reading, which is where the preview lands. None when there
    is no readable analysis, when it was run for a different game (a different detector,
    so a different answer), or when it calibrated no clock (a drawn-box run, or a refused
    calibration: the search should run afresh). Defensive like recorded_clock_seconds: a
    malformed record reads as absent rather than raising."""
    try:
        data = json.loads(pipeline_json_for(video_path).read_text(encoding="utf-8"))
        config = data["config"]
        recorded_game = config["inference_settings"].get("game", GAME_DEFAULT)
        record = config.get("clock_calibration")
        events = data.get("events", [])
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if recorded_game != game_id or not isinstance(record, dict):
        return None
    # The record is ClockCalibration.as_config(); reading it back through the result model's
    # own defensive parser keeps one definition of "a usable bbox".
    parsed = ClockLocateResult.from_json({"bbox": record.get("bbox"), "hits": record.get("hits"),
                                          "samples": record.get("samples")})
    if parsed.bbox is None:
        return None
    first_hit_sec: float | None = None
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict) or event.get("timer") is None:
                continue
            timestamp = event.get("timestamp_sec")
            if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
                first_hit_sec = float(timestamp)
            break
    # A run's timer event is a parsed read, so the first one is a landing frame too.
    return ClockLocateResult(first_hit_sec=first_hit_sec, first_hit_bbox=parsed.bbox,
                             bbox=parsed.bbox, calibration=record,
                             hits=parsed.hits, samples=parsed.samples,
                             first_read_sec=first_hit_sec)


def analysis_settings_differences(saved: ExistingAnalysis, *, ocr_engine: str, ocr_lang: str,
                                  overlay_layout: str, game: str) -> list[str]:
    """How the text-reading settings a saved analysis was run with differ from the ones selected
    now: human-readable descriptions, empty when they agree. Mirrors video_metadata_mismatches,
    which reports a changed video the same way, so both warnings read alike.

    The OCR cadence is deliberately left out: it's on the note above the button already, and a
    differing cadence is the ordinary case (analyze at 30s, reuse without re-picking it), so
    including it would fire the warning nearly every time and train the user to click through."""
    differences = []
    if saved.ocr_engine != ocr_engine:
        differences.append(
            f"Font style: analyzed with {engine_display_name(saved.ocr_engine)}. "
            f"Currently set to {engine_display_name(ocr_engine)}")
    if saved.ocr_lang != ocr_lang:
        differences.append(
            f"Language: analyzed with {lang_display_name(saved.ocr_lang)}. "
            f"Currently set to {lang_display_name(ocr_lang)}")
    if saved.overlay_layout != overlay_layout:
        differences.append(
            f"Overlay layout: analyzed with {saved.overlay_layout or AUTO_LAYOUT_NAME}. "
            f"Currently set to {overlay_layout or AUTO_LAYOUT_NAME}")
    if saved.game != game:
        differences.append(
            f"Game: analyzed as {game_display_name(saved.game)}. "
            f"Currently set to {game_display_name(game)}")
    return differences


def video_metadata_mismatches(video_path: str) -> list[str]:
    """Compare the video file's properties against the metadata the pipeline recorded in its
    pipeline.json. Returns a list of human-readable mismatch descriptions (empty == the video
    looks unchanged). Conservative on uncertainty: an unreadable JSON, a config without the
    recorded fields, or any error returns [] so verification never blocks the user."""
    try:
        config = json.loads(
            pipeline_json_for(video_path).read_text(encoding="utf-8"),
        ).get("config")
    except (OSError, json.JSONDecodeError, AttributeError):
        return []
    if not config:
        return []

    import cv2  # local: keeps GUI startup fast (only the subprocesses need cv2 otherwise)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ["could not open the video file to verify it"]
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return _compare_video_metadata(config, total_frames=total_frames, fps=fps,
                                   width=width, height=height)


# How far a video's frame count may be from the analyzed one and still count as the same
# video. OpenCV's count for an MKV is an estimate, 1 to 2 frames off the count of the MP4
# converted from it, so an exact comparison warned about every analyzed MKV once it was
# converted. A trimmed or swapped video differs by far more than this.
FRAME_COUNT_TOLERANCE = 5


def _compare_video_metadata(config: Mapping[str, Any], *, total_frames: int, fps: float,
                            width: int, height: int) -> list[str]:
    """The pure comparison behind video_metadata_mismatches: the recorded pipeline.json config
    versus the probed file properties. Split out from the cv2 probe so the comparison rules
    (which fields participate, FRAME_COUNT_TOLERANCE, the 2-decimal fps tolerance,
    None-skips) are unit-testable without a video file."""
    issues: list[str] = []
    if config.get("total_frames") is not None and \
            abs(config["total_frames"] - total_frames) > FRAME_COUNT_TOLERANCE:
        issues.append(f"frame count: analyzed {config['total_frames']}, file now {total_frames}")
    if config.get("fps") is not None and round(config["fps"], 2) != round(fps, 2):
        issues.append(f"frame rate: analyzed {config['fps']:.2f}, file now {fps:.2f}")
    if config.get("frame_width") is not None and config.get("frame_height") is not None and \
            (config["frame_width"], config["frame_height"]) != (width, height):
        issues.append(f"resolution: analyzed {config['frame_width']}×{config['frame_height']}, "
                      f"file now {width}×{height}")
    return issues


def export_csv_for(tournament_name: str, stem: str, is_set: bool, export_root: Path) -> Path:
    """The metadata CSV an export with these settings writes:
    <export_root>/<basename>/<basename>.csv (see export_basename)."""
    base = export_basename(tournament_name, stem, is_set)
    return export_root / base / f"{base}.csv"


class PreviousExport(NamedTuple):
    """What reviewed.json records about this vod's last completed clip export."""
    csv: Path    # <export_root>/<export_folder>/<export_csv>
    mode: str    # "match" | "tournament_set"


def last_export_info(stem: str, export_root: Path) -> PreviousExport | None:
    """This vod's last completed clip export, read from reviewed.json: the metadata CSV's
    path and the mode the export ran with. The folder and CSV name prefer what
    vodparse.cli.export_reviewed wrote back on success; the folder record survives a
    --folder override, and the CSV record survives a later tournament rename (deriving
    the CSV name from the current tournament_name would point inside the old folder at a
    file that was never written there). Derivation from tournament name/mode remains only
    as a fallback for reviewed.json files that predate the write-back. The records are
    bare names reconstructed against the CURRENT export root: after the user moves the
    export-dir setting, an old export simply stops resolving (its CSV won't exist there),
    which the callers already treat as "no previous export". None when no export is
    recorded or the file is unreadable/malformed."""
    try:
        data = json.loads(
            (PROJECT_ROOT / "json_output" / stem / "reviewed.json").read_text(encoding="utf-8"))
        mode = data["mode"]
        base = export_basename(data.get("tournament_name", ""), stem, mode != "match")
        folder = data.get("export_folder") or base
        csv_name = data.get("export_csv") or f"{base}.csv"
        return PreviousExport(export_root / folder / csv_name, mode)
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def stale_export_dir(previous: PreviousExport | None, new_export_dir: Path,
                     new_mode: str, export_root: Path, *, stem: str) -> Path | None:
    """The previous export's folder when a fresh export is about to land somewhere else:
    what the GUI offers to delete once the new export succeeds (a tournament rename moves
    the export folder, silently duplicating gigabytes of clips otherwise).

    Only a same-mode previous export qualifies: a match↔set mode switch re-cuts the same
    vod into different clips, so the previous export is an alternate cut the user may
    want to keep, not a duplicate; no offer.

    Deletion targets are vetted hard: this app otherwise never deletes user data, and the
    export_folder record feeding `previous` is plain JSON on disk (hand-editable, or a
    --folder override), so a weird value must never aim the offer at unrelated files.
    The folder qualifies only if it:

    - differs from `new_export_dir`, compared case-blind: on a case-insensitive
      filesystem (macOS APFS) a case-only tournament rename keeps the same physical
      folder, and offering it would delete the export that just landed there (a
      conservative false negative on case-sensitive filesystems is acceptable);
    - still contains the recorded metadata CSV; proof it's a completed export of ours;
    - holds no other VOD's clips: one tournament's VODs share its export folder (see
      vodparse.export.merge_vod_rows), so renaming this VOD (`stem`) out of the tournament
      leaves the folder still in use. The recorded CSV must list no other VOD's rows (rows
      from before the "vod" column are nobody's); one that can't be read doesn't qualify;
    - is not itself a symlink;
    - fully resolved, sits DIRECTLY under `export_root` (the same root `previous` was
      resolved against); which also rules out the root itself, ".."/absolute-path
      escapes, and nested folders.

    Anything else returns None and the offer simply isn't made.

    `previous` is last_export_info's result, looked up before the export overwrites
    reviewed.json (that write destroys the export_folder record pointing at it)."""
    if previous is None or previous.mode != new_mode or not previous.csv.is_file():
        return None
    try:
        if other_vod_clip_counts(read_export_csv(previous.csv), stem):
            return None
    except CSV_READ_ERRORS:
        return None
    folder = previous.csv.parent
    # Both paths are built under the same export root, so a full-string casefold
    # comparison is safe (no aliasing concerns beyond the folder name itself).
    if str(folder).casefold() == str(new_export_dir).casefold() or folder.is_symlink():
        return None
    if folder.resolve().parent != export_root.resolve():
        return None
    return folder


def other_vods_to_confirm(csv_path: Path, stem: str) -> dict[str, int]:
    """The other VODs, with their clip counts, that this VOD's export would join in the
    shared tournament CSV at `csv_path` for the first time: what the GUI's "add to an
    existing tournament export?" prompt lists (see vodparse.export.merge_vod_rows for why
    one tournament's VODs share a CSV).

    Empty when there is nothing to confirm: no CSV yet, a CSV holding only this VOD's
    clips or rows from before the "vod" column (which the export replaces, as the build
    that wrote them would have), or one this VOD already joined (it has rows there, so the
    user agreed once; re-exports never ask again). Asking once per VOD catches a reused
    tournament name (a weekly whose number wasn't bumped) before its clips land in the
    wrong export. Raises one of vodparse.export.CSV_READ_ERRORS when the CSV exists but
    can't be read."""
    rows = read_export_csv(csv_path)
    if any(is_vod_row(row, stem) for row in rows):
        return {}
    return other_vod_clip_counts(rows, stem)


def vod_clip_filenames(csv_path: Path, stem: str) -> frozenset[str]:
    """The clip filenames VOD `stem` has in the export CSV at `csv_path`, read at export
    kickoff, before the export replaces those rows: what outdated_clip_files compares
    against once the export is done. Rows from before the "vod" column are nobody's, so
    their files are never offered for removal. Empty when there's no CSV or it can't be
    read, so the worst case is no cleanup offer, never a wrong one."""
    try:
        rows = read_export_csv(csv_path)
    except CSV_READ_ERRORS:
        return frozenset()
    return frozenset(row["filename"] for row in rows
                     if row.get("filename") and is_vod_row(row, stem))


def outdated_clip_files(export_dir: Path, csv_path: Path, old_filenames: frozenset[str]
                        ) -> list[Path]:
    """The clip files a re-export left behind in `export_dir`: named in this VOD's rows
    before the export (`old_filenames`, from vod_clip_filenames) but in no row of the CSV
    at `csv_path` now. A re-export that merged, split, retitled, or excluded clips writes
    new files and replaces the VOD's rows, but never deletes the old files itself: these
    are what the GUI offers to remove.

    Deletion targets, so vetted hard like stale_export_dir's. Each candidate must:
    - be a bare filename (a hand-edited CSV must not aim this at "../x" or a subfolder);
    - not match a name the CSV still lists, compared case-blind: on a case-insensitive
      filesystem (macOS APFS) a case-only retitle keeps the same physical file;
    - be a regular file, not a symlink, directly in `export_dir`;
    - not be the same physical file as one the CSV lists (APFS also folds differently
      composed Unicode names onto one file).
    Fails safe: an unreadable CSV, or a file that can't be stat'd, yields no candidate."""
    try:
        listed = [row["filename"] for row in read_export_csv(csv_path) if row.get("filename")]
    except CSV_READ_ERRORS:
        return []
    listed_casefolded = {name.casefold() for name in listed}
    listed_files: set[tuple[int, int]] = set()   # (device, inode) of each listed clip on disk
    for name in listed:
        try:
            info = (export_dir / name).stat()
        except OSError:
            continue  # not on disk: can't alias a candidate
        listed_files.add((info.st_dev, info.st_ino))
    outdated: list[Path] = []
    for name in sorted(old_filenames):
        if Path(name).name != name or name == "..":
            continue  # not a bare filename
        if name.casefold() in listed_casefolded:
            continue  # still listed (a case-only retitle included)
        candidate = export_dir / name
        try:
            if candidate.is_symlink() or not candidate.is_file():
                continue
            info = candidate.stat()
        except OSError:
            continue
        if (info.st_dev, info.st_ino) not in listed_files:
            outdated.append(candidate)
    return outdated


def same_physical_dir(a: Path, b: Path) -> bool:
    """Whether two paths point at the same directory on disk; the stat-level identity
    that the casefold check in stale_export_dir can't establish (it also catches APFS
    Unicode-normalization collisions, where differently-composed names open one folder).
    Used to vet the delete-previous-export offer, so it fails SAFE: when either path
    can't be stat'd (vanished, permissions), the answer is True; treat them as the same
    folder, make no offer, delete nothing."""
    try:
        return a.samefile(b)
    except OSError:
        return True


def resolve_export_csv(tournament_name: str, stem: str, is_set: bool,
                       export_root: Path) -> Path:
    """The CSV the Review screen's export-output buttons (Upload Assistant / Open export
    folder) should target. Prefers the CSV the current settings would export; when that
    doesn't exist; typically the tournament was renamed after exporting; falls back to
    the last completed export's CSV, so the buttons keep opening the clips that are
    actually on disk until a fresh export under the new name takes over. When neither
    exists, returns the current-name path (whose absence disables the buttons)."""
    csv_path = export_csv_for(tournament_name, stem, is_set, export_root)
    if csv_path.exists():
        return csv_path
    last = last_export_info(stem, export_root)
    if last is not None and last.csv.exists():
        return last.csv
    return csv_path


def draft_path(stem: str) -> Path:
    """The auto-saved edit draft for one vod (next to its matches.json). Single source of truth so
    the GUI can detect/clear it (the merged Resume/Open button, extract overwrite) using the same
    path the Review screen saves to."""
    return PROJECT_ROOT / "json_output" / stem / _DRAFT_NAME


def review_draft_for(video_path: str) -> Path:
    """The auto-saved Review-edit draft for a given video (see draft_path)."""
    return draft_path(vod_stem(video_path))


def app_icon_path() -> Path:
    """The bundled app icon (window/taskbar icon in app.py, About-dialog logo in dialogs.py).
    Both call sites guard the result with .exists(): a source checkout has no bundled
    assets/ folder, so the icon is only present in a frozen build."""
    return bundle_dir() / "assets" / "app_icon.png"


def third_party_licenses_path() -> Path:
    """The bundled third-party license texts, opened by the About dialog's "Licenses" button.

    One document holding every dependency's own license text, written by
    tools/licensing/license_report.py during the build. GPL-3.0 (PyQt6) and LGPL (Qt 6,
    python-bidi, OpenCV's video backend) require their text to accompany the binary, so a
    frozen build always has it. This project's OWN license is not in here: LICENSE and
    NOTICE sit at the bundle root, beside the licenses/ folder. The caller guards with
    .is_file(), since in a source checkout the document exists only once
    ``license_report.py bundle`` has been run."""
    return bundle_dir() / "licenses" / "THIRD-PARTY-LICENSES.txt"


def default_tournament_name(video_path: str) -> str:
    """A tournament name seeded from the vod's filename: underscores/dashes become spaces
    ("vod_142" → "vod 142"). Casing is left alone; the field is there to be edited. A trailing
    "featuring ..." / "ft ..." / "feat ..." player list (common in downloaded VOD titles) is
    dropped along with everything after it, and so is the ".converted" an MP4's conversion
    adds to its name (see vodparse.paths.strip_converted_suffix): it names the same VOD."""
    stem = strip_converted_suffix(Path(video_path).stem)
    name = re.sub(r"[_\-]+", " ", stem).strip()
    return strip_featuring_suffix(name)
