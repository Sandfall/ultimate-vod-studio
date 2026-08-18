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
video_metadata_mismatches, last_export_info / resolve_export_csv): what a saved analysis
or export recorded, and whether the video on disk still
matches it.

Qt-free; safe to import from anywhere.
"""

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from vodparse.export import export_basename
from vodparse.gui.settings import engine_display_name, lang_display_name
from vodparse.ocr_engine import OCR_ENGINE_DEFAULT
from vodparse.ocr_langs import OCR_LANG_DEFAULT
from vodparse.paths import vod_stem
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
    when the JSON records no cadence at all, but the engine and language always resolve: a run
    predating either setting could only have been EasyOCR reading English."""
    ocr_interval_sec: float | None
    ocr_engine: str   # vodparse.ocr_engine ids ("easyocr" | "doctr")
    ocr_lang: str     # vodparse.ocr_langs codes ("en" | "ja" | …)


def existing_analysis(video_path: str) -> ExistingAnalysis | None:
    """What the saved pipeline.json was analyzed with, or None when there is nothing readable
    to report; surfaced on the Analyze page so 'Open existing analysis' shows what it reuses."""
    try:
        infer = json.loads(
            pipeline_json_for(video_path).read_text(encoding="utf-8"),
        )["config"]["inference_settings"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
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
    )


def existing_analysis_summary(video_path: str) -> str:
    """The one-line note shown above the run buttons: what a saved analysis would be reused with.
    Always names the OCR cadence and the font style; names the overlay language only when it
    isn't the English default, mirroring the Analyze page's own language note (which likewise
    stays silent on English, so a named language always means a non-default one)."""
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
    return f"Existing analysis: {', '.join(parts)}"


def analysis_settings_differences(saved: ExistingAnalysis, *, ocr_engine: str,
                                  ocr_lang: str) -> list[str]:
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


def _compare_video_metadata(config: Mapping[str, Any], *, total_frames: int, fps: float,
                            width: int, height: int) -> list[str]:
    """The pure comparison behind video_metadata_mismatches: the recorded pipeline.json config
    versus the probed file properties. Split out from the cv2 probe so the comparison rules
    (which fields participate, the 2-decimal fps tolerance, None-skips) are unit-testable
    without a video file."""
    issues: list[str] = []
    if config.get("total_frames") is not None and config["total_frames"] != total_frames:
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
                     new_mode: str, export_root: Path) -> Path | None:
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
    - is not itself a symlink;
    - fully resolved, sits DIRECTLY under `export_root` (the same root `previous` was
      resolved against); which also rules out the root itself, ".."/absolute-path
      escapes, and nested folders.

    Anything else returns None and the offer simply isn't made.

    `previous` is last_export_info's result, looked up before the export overwrites
    reviewed.json (that write destroys the export_folder record pointing at it)."""
    if previous is None or previous.mode != new_mode or not previous.csv.is_file():
        return None
    folder = previous.csv.parent
    # Both paths are built under the same export root, so a full-string casefold
    # comparison is safe (no aliasing concerns beyond the folder name itself).
    if str(folder).casefold() == str(new_export_dir).casefold() or folder.is_symlink():
        return None
    if folder.resolve().parent != export_root.resolve():
        return None
    return folder


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
    dropped along with everything after it."""
    name = re.sub(r"[_\-]+", " ", Path(video_path).stem).strip()
    return strip_featuring_suffix(name)
