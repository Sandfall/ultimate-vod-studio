"""CSV metadata and video clip export for matches and tournament sets."""

import csv
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


from vodparse.ffmpeg_runtime import clip_match
from vodparse.grouping import grouping_tags as _grouping_tags
from vodparse.grouping import overlay_reads_full_pairs as _overlay_reads_full_pairs
from vodparse.schema import Clip, Match, ReviewedData, SetGroup
from vodparse.text_utils import format_timestamp
from vodparse.titles import build_title, fit_title, report_titles


# ── Export-progress protocol (writer + reader, kept side by side) ──────────────────
# export_reviewed streams one line per saved clip to stdout; the GUI counts those lines as
# they arrive to advance its export progress bar (gui.app._read_output → set_export_progress).
# The prefix is the contract between the two processes, so the print sites below and
# count_saved_lines both spell it via this constant: the seam that drifted in finding S2.
SAVED_LINE_PREFIX = "  Saved:"


def count_saved_lines(output: str) -> int:
    """How many clips ``export_reviewed`` has reported saved in ``output`` so far.

    ``output`` is the whole accumulated subprocess stdout, not a single read: counting
    newline-anchored ``SAVED_LINE_PREFIX`` occurrences over the full buffer keeps a read that
    splits a "  Saved:" line across two chunks from under- or double-counting. Anchoring on the
    leading newline also skips the prefix should it ever appear mid-line in other output; the
    writer prints a header line before any clip, so every real saved line is newline-preceded.
    """
    return output.count("\n" + SAVED_LINE_PREFIX)


CSV_FIELDS = [
    # Player 1
    "player_1_sponsor", "player_1_tag", "player_1_character",
    # Player 2
    "player_2_sponsor", "player_2_tag", "player_2_character",
    # Core match metadata
    "filename", "title", "round", "best_of", "match_count", "start_time", "end_time",
    # Everything else
    "player_1_ingame_tag", "player_2_ingame_tag",
    "player_1_tournament_tag", "player_2_tournament_tag",
    "matches",
    # Export-wide (the same value on every row): the user-entered tournament name, so
    # downstream tools (e.g. the upload assistant's {tournament} placeholder) can rebuild
    # titles from parts without re-asking for it.
    "tournament",
    # The VOD (its stem) the clip was cut from. One tournament can span several VODs that
    # share this CSV (see merge_vod_rows), and start_time/end_time are positions inside it.
    "vod",
    # Where the clip's first game starts inside the clip file (M:SS from the file's start;
    # empty when unknown, as for clips exported before the column existed). The upload
    # assistant takes its thumbnail a few seconds after it, past the lead-in.
    "first_game_offset",
]


# Characters that are illegal or troublesome in filenames across macOS/Windows/Linux.
# "/" matters most here: the set-characters join mode produces titles like "(Char1 / Char2)".
_FILENAME_BAD_CHARS = re.compile(r'[\\/:*?"<>|]')

# Windows reserved device names: CON, PRN, AUX, NUL, COM1-9, LPT1-9. Reserved if the part
# BEFORE THE FIRST DOT matches (case-insensitive, trailing spaces ignored), so "NUL" is
# unwritable as NUL.mp4, and "nul.backup" is unwritable as nul.backup.mp4.
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


# ext4 and APFS cap each path *component* at 255 UTF-8 bytes (not characters). A CJK title
# runs ~3 bytes/char, so a character-count cap silently produces 500+-byte names the
# filesystem rejects; ffmpeg then can't create the clip and the whole export aborts. Every
# name cap below is therefore measured on the encoded byte length. Callers that append an
# extension or suffix (".mp4", "_sets") reserve room for it against this budget themselves.
_MAX_NAME_BYTES = 255


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """`text` truncated to at most `max_bytes` UTF-8 bytes without splitting a multi-byte
    character: encode, slice at the byte boundary, then decode dropping any partial trailing
    character (``errors="ignore"`` discards the incomplete tail cleanly)."""
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def _safe_name(text: str) -> str:
    """`text` made safe for a file/folder name: illegal characters become "-", whitespace is
    collapsed, and the result is truncated to the 255-UTF-8-byte ext4/APFS component limit
    without splitting a multi-byte character. Windows reserved device names (NUL, COM1, etc.)
    are prefixed with "_" to defuse them. Returns "" when nothing usable remains: i.e. the
    text held no alphanumeric character in *any* script (so a fully non-Latin title like
    "格闘ゲーム大会" is kept, not discarded): letting callers apply their own fallback."""
    name = _FILENAME_BAD_CHARS.sub("-", text)
    name = re.sub(r"\s+", " ", name).strip().strip(".")
    if not any(ch.isalnum() for ch in name):
        return ""
    name = _truncate_utf8(name, _MAX_NAME_BYTES).rstrip()
    # Windows reserves these device names (case-insensitive, judged on the part before the
    # first dot), so "NUL" or "nul.backup" would be unwritable as filenames. A leading
    # underscore defuses them; a prefix (not suffix) because the pre-dot stem is what counts.
    if name.split(".", 1)[0].rstrip().upper() in _WINDOWS_RESERVED:
        name = "_" + name
    return name


# The folder-name half of the MAX_PATH story (see _MAX_CLIP_PATH below): the export folder
# is named after the tournament/stem and sits inside the path of every clip, so an unbounded
# folder name eats the budget the filenames need. 80 keeps the folder recognizable while
# leaving ~100 chars for clip names under the default workspace.
_EXPORT_BASENAME_CAP = 80


def export_basename(tournament_name: str, stem: str, is_set: bool) -> str:
    """The base name shared by an export's video_out/ folder and its CSV: the (sanitized)
    user-entered tournament name when there is one, else the vod stem, plus the mode
    suffix: "Example Tournament #142_sets". Capped so the folder leaves clip filenames
    room under the Windows path limit (see _MAX_CLIP_PATH)."""
    base = _safe_name(tournament_name or "") or stem
    # Leave byte room for the "_games"/"_sets" suffix so the folder component stays under the
    # 255-byte filesystem limit, then apply the char cap that keeps the folder recognizable.
    base = _truncate_utf8(base, _MAX_NAME_BYTES - len("_games"))[:_EXPORT_BASENAME_CAP].rstrip()
    return f"{base}_{'sets' if is_set else 'games'}"


# Windows refuses paths over 260 chars (MAX_PATH) unless a machine-wide opt-in is enabled:
# yet ffmpeg can still WRITE longer ones (it uses the \\?\ form internally), stranding clips
# that Explorer lists but nothing else can open: this app's upload assistant, a browser's
# upload picker. Cap the whole destination path with headroom, on every OS, so an export
# stays portable when it lands on a Windows machine.
_MAX_CLIP_PATH = 240

# Even under an absurdly deep clip_dir, keep enough filename to stay meaningful and unique;
# the path may then exceed _MAX_CLIP_PATH, but that's the folder's doing, not the clip's.
_MIN_NAME_BUDGET = 20


def _name_budget(clip_dir: Path) -> int:
    """Characters left for a clip's stem under `clip_dir` before the path outgrows
    _MAX_CLIP_PATH: the cap minus the dir, the path separator, and ".mp4". Measured on
    the absolute form: the export stage receives `clip_dir` relative to its working
    directory (``video_out/<export>``), but Windows' limit applies to the full path
    the filesystem sees: a relative measure under-counts by the workspace's depth."""
    return _MAX_CLIP_PATH - len(str(clip_dir.absolute())) - 1 - len(".mp4")


def warn_if_clip_paths_cramped(clip_dir: Path) -> None:
    """Print a warning when `clip_dir` is so deep that even truncated clip names may push
    paths past what Windows can open. Called once per export, not per clip."""
    if _name_budget(clip_dir) < _MIN_NAME_BUDGET:
        print(f"  WARNING: the export folder's full path is {len(str(clip_dir.absolute()))} characters long, "
              f"so clip paths may exceed the ~260-character limit Windows can open. "
              f"Export somewhere shallower to be safe.")


def _fit_stem(stem: str, clip_dir: Path, reserve: int = 0) -> str:
    """`stem` truncated so ``<stem>.mp4``: plus `reserve` characters held back for a
    " (2)"-style de-duplication suffix: satisfies *both* filename limits: the 255-UTF-8-byte
    ext4/APFS component cap and the _MAX_CLIP_PATH Windows whole-path cap. The Windows budget
    is applied first (in characters), then the byte budget, so a Latin name is bounded by the
    path length and a CJK name by its encoded byte length, whichever binds first.

    The Windows budget counts Python characters; MAX_PATH actually counts UTF-16 code units,
    where an astral-plane character (rare in a matchup title) is two; so an all-emoji name
    could sit a few units over MAX_PATH. That limit is soft (ffmpeg writes past it via the
    \\?\\ form; see _MAX_CLIP_PATH), while the byte cap is the hard one and is measured
    exactly. The de-dup suffix is ASCII, so `reserve` is the same count in characters, bytes,
    and UTF-16 units."""
    win_budget = max(_name_budget(clip_dir), _MIN_NAME_BUDGET) - reserve
    byte_budget = _MAX_NAME_BYTES - len(".mp4") - reserve
    return _truncate_utf8(stem[:win_budget], byte_budget).rstrip()


def _title_filename(title: str, fallback: str, used: set[str], clip_dir: Path,
                    tournament_name: str = "") -> str:
    """A clip filename derived from its title: the tournament-name prefix dropped (the
    export folder is already named after the tournament, so repeating it in every clip
    only eats the Windows path budget), sanitized via _safe_name, truncated so the
    full path under `clip_dir` stays Windows-openable (see _MAX_CLIP_PATH), with a
    " (2)"-style suffix when the name is already in `used` (an earlier clip of this
    export, or another VOD's clip in the same folder; see other_vod_filenames) so clips
    can't overwrite each other. `used` holds lower-cased names without the extension.
    `fallback` (the numbered default, no extension) covers an empty-after-sanitizing
    title."""
    # build_title composes "{tournament} - {round} - {matchup}" and fit_title never drops
    # the tournament, so the prefix is exact; removeprefix is a no-op when it isn't there.
    if tournament_name:
        title = title.removeprefix(f"{tournament_name} - ")
    stem = _safe_name(title) or fallback
    name, n = _fit_stem(stem, clip_dir), 2
    while name.lower() in used:
        suffix = f" ({n})"
        name = _fit_stem(stem, clip_dir, reserve=len(suffix)) + suffix
        n += 1
    used.add(name.lower())
    return f"{name}.mp4"


@dataclass
class ClipMeta:
    """One clip's export metadata, as _make_csv_row consumes it. Positional tag/sponsor/
    character lists carry None for a slot no game resolved."""
    filename: str
    ingame_tags: list | None
    tournament_tags: list | None
    round_str: str
    start_sec: float
    end_sec: float
    best_of: str = ""
    match_count: int = 1
    matches: str = ""
    characters: list | None = None
    tournament_sponsors: list | None = None
    grouping_tags: list | None = None
    # Seconds from the clip file's start to its first game's start (the CSV's
    # first_game_offset); None when the exporter doesn't know it.
    first_game_offset_sec: float | None = None


def _make_csv_row(meta: ClipMeta, *, vod: str = "", tournament_name: str = "",
                  use_sponsors: bool = True, title_limit: int = 100,
                  title_records: list[dict] | None = None) -> dict:
    """Build one clip's CSV row (keys = CSV_FIELDS) from its ClipMeta plus the export-wide
    knobs; `vod` is the stem of the VOD being exported. When `title_records` is given,
    append this row's {row, action, original} title-fit record to it for report_titles."""
    tags   = meta.ingame_tags or ["unknown", "unknown"]
    p1     = tags[0]
    p2     = tags[1] if len(tags) > 1 else ""
    t_tags = meta.tournament_tags or []
    spons  = meta.tournament_sponsors or []
    chars  = meta.characters or []
    c1     = chars[0] if len(chars) > 0 else ""
    c2     = chars[1] if len(chars) > 1 else ""
    s1     = spons[0] if len(spons) > 0 and spons[0] else ""
    s2     = spons[1] if len(spons) > 1 and spons[1] else ""

    title_tags = meta.grouping_tags or meta.tournament_tags or tags
    t1 = title_tags[0]
    t2 = title_tags[1] if len(title_tags) > 1 else ""
    # Sponsors prefix the title tags ("PRO | ABC") only when the include-sponsors
    # preference is on; the CSV sponsor columns are filled either way.
    ts1, ts2 = (s1, s2) if use_sponsors else ("", "")

    title, action = fit_title(t1, t2, c1=c1, c2=c2, tournament_name=tournament_name,
                              round_str=meta.round_str, s1=ts1, s2=ts2, title_limit=title_limit)

    row = {
        "filename":                meta.filename,
        "player_1_tag":            t1,
        "player_2_tag":            t2,
        "player_1_ingame_tag":     p1,
        "player_2_ingame_tag":     p2,
        "player_1_tournament_tag": t_tags[0] if len(t_tags) > 0 else "",
        "player_2_tournament_tag": t_tags[1] if len(t_tags) > 1 else "",
        "player_1_sponsor":        s1,
        "player_2_sponsor":        s2,
        "player_1_character":      c1,
        "player_2_character":      c2,
        "round":                   meta.round_str,
        "best_of":                 meta.best_of,
        "match_count":             meta.match_count,
        "matches":                 meta.matches,
        "start_time":              format_timestamp(meta.start_sec),
        "end_time":                format_timestamp(meta.end_sec),
        "title":                   title,
        "tournament":              tournament_name,
        "vod":                     vod,
        "first_game_offset":       ("" if meta.first_game_offset_sec is None
                                    else format_timestamp(meta.first_game_offset_sec)),
    }
    if title_records is not None:
        # The full unshortened title, for the report; only meaningful when fit_title acted.
        original = build_title(t1, t2, c1=c1, c2=c2, tournament_name=tournament_name,
                               round_str=meta.round_str, s1=ts1, s2=ts2) if action else None
        title_records.append({"row": row, "action": action, "original": original})
    return row


def build_chapters(data: ReviewedData, use_sponsors: bool) -> str:
    """Chapter timestamps for the full VOD, ready to paste into a video description.

    One `{timestamp} {title}` line per clip (in display order, exclusions already dropped by
    build_reviewed_data), prefixed with a `0:00 Intro` line: video platforms only render chapters
    when the first timestamp is 0:00. Titles reuse the same tag/sponsor/character selection as the
    CSV
    titles (see _make_csv_row) but omit the tournament-name prefix, which would repeat on every
    line, and skip the 100-char fit since chapter lines have no length cap. Each line, including
    the last, ends in a newline."""
    lines = ["0:00 Intro"]
    for clip in data["clips"]:
        tags       = clip.get("ingame_tags") or ["unknown", "unknown"]
        title_tags = clip.get("grouping_tags") or clip.get("tournament_tags") or tags
        spons      = clip.get("tournament_sponsors") or []
        chars      = clip.get("characters") or []
        t1 = title_tags[0] or ""
        t2 = (title_tags[1] or "") if len(title_tags) > 1 else ""
        c1 = chars[0] if len(chars) > 0 else ""
        c2 = chars[1] if len(chars) > 1 else ""
        s1 = spons[0] if len(spons) > 0 and spons[0] else ""
        s2 = spons[1] if len(spons) > 1 and spons[1] else ""
        ts1, ts2 = (s1, s2) if use_sponsors else ("", "")
        title = build_title(t1, t2, c1=c1, c2=c2, tournament_name="",
                            round_str=clip.get("round") or "", s1=ts1, s2=ts2)
        lines.append(f"{format_timestamp(clip['start_sec'])} {title}")
    return "".join(line + "\n" for line in lines)


def retitled_for_tournament(row: dict[str, str], new_name: str, title_limit: int) -> str:
    """The `title` column of a CSV `row` (as _make_csv_row wrote it) with its tournament
    renamed to `new_name`, for the Upload Assistant's Change tournament name.

    Export built the title with fit_title from the row's own columns, but two of its inputs
    aren't recorded in the CSV: whether sponsors were included, and the title limit then in
    effect. So the title is rebuilt with the old name both with and without sponsors, at
    `title_limit` (the export setting now); a match shows how export built it, and the same
    build with the new name replaces it, shortened as export would. When neither matches (the
    limit has changed since, or the title was edited by hand), a leading "<old name> - " is
    swapped for the new name instead, and a title without one is left as it is."""
    old_name = row.get("tournament", "")
    title = row.get("title", "")
    tags = (row.get("player_1_tag", ""), row.get("player_2_tag", ""))
    characters = {"c1": row.get("player_1_character", ""), "c2": row.get("player_2_character", "")}
    round_str = row.get("round", "")
    with_sponsors = (row.get("player_1_sponsor", ""), row.get("player_2_sponsor", ""))
    for s1, s2 in (with_sponsors, ("", "")):
        rebuilt, _action = fit_title(*tags, **characters, tournament_name=old_name,
                                     round_str=round_str, s1=s1, s2=s2, title_limit=title_limit)
        if rebuilt == title:
            renamed, _action = fit_title(*tags, **characters, tournament_name=new_name,
                                         round_str=round_str, s1=s1, s2=s2,
                                         title_limit=title_limit)
            return renamed
    old_prefix = f"{old_name} - "
    if old_name and title.startswith(old_prefix):
        rest = title.removeprefix(old_prefix)
        return " - ".join(part for part in (new_name, rest) if part)   # as build_title joins
    return title


def write_csv(rows: list[dict], path: str | Path,
              fieldnames: Sequence[str] = CSV_FIELDS) -> None:
    """Write the export rows to `path` as a CSV with the `fieldnames` header (the fixed
    CSV_FIELDS unless the caller preserves an existing file's columns, as the Upload
    Assistant does), in one atomic step: a sibling temp file replaced over the target, so
    a crash mid-write never leaves a truncated CSV (it may hold other VODs' rows; see
    merge_vod_rows). Columns missing from a row are written empty: a row read back from a
    CSV that predates a column. Keys outside `fieldnames` are dropped."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        # Plain UTF-8, no BOM: the primary reader of this CSV is our own upload assistant,
        # not Excel, so round-trip fidelity with the readers wins over Excel's BOM convenience.
        with tmp.open("w", newline="", encoding="utf-8") as f:
            # extrasaction="ignore" tolerates the None key DictReader adds to a read-back
            # row that had more values than the header.
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_csv_or_exit(rows: list[dict], path: str | Path) -> None:
    """write_csv for the export stages: a CSV that can't be replaced (on Windows, one open
    in Excel) stops the stage with a clear message instead of a traceback. The clips are
    cut by then, so the message says so: a second export re-cuts them in place."""
    try:
        write_csv(rows, path)
    except OSError as err:
        raise SystemExit(f"error: couldn't write {path} ({err}). The clips were cut; close "
                         "the CSV in Excel or the Upload Assistant and export again.") from err


# ── One tournament, several VODs: the shared CSV ───────────────────────────────────
# A tournament whose stream restarted arrives as several VOD files. export_basename
# names the export folder and its CSV after the tournament, not the VOD, so every VOD of
# the tournament exports into the same folder and the same CSV. Each row records its
# source VOD in the "vod" column, and an export replaces only its own VOD's rows: the
# Upload Assistant gets the whole tournament in one file, and re-exporting a VOD after
# fixing its clips (merged, split, retitled, excluded) swaps out exactly that VOD's block.
#
# Rows from before the "vod" column existed carry no VOD. They are from a build that
# overwrote the whole CSV on every export, and the next export into that folder replaces
# them the same way (the clip files stay on disk; only the rows go). Guessing whose they
# were is not worth it: nothing may ever delete a file on the strength of a guess.

class ExportCsvFormatError(ValueError):
    """An existing export CSV parses, but not as one of ours: its header has no "filename"
    column. A spreadsheet app re-saved it with another delimiter (every row then reads as
    one giant key), or the header was edited. Such rows can't be matched to clips, and
    writing them back would turn them into blank rows, so the file is refused instead."""


# What reading an existing export CSV can raise: the file vanished or is unreadable
# (OSError), it was re-saved in a non-UTF-8 encoding (UnicodeDecodeError), it is
# malformed (csv.Error), or its header is not an export CSV's (ExportCsvFormatError).
CSV_READ_ERRORS = (OSError, UnicodeDecodeError, csv.Error, ExportCsvFormatError)

# The one column every export CSV has had since the first build: the clip's file name.
# Its absence from a header means the file is not (any more) an export CSV.
CSV_REQUIRED_FIELD = "filename"


def read_export_csv(path: Path) -> list[dict[str, str]]:
    """The rows of the export CSV at `path`, or [] when there is none yet or the file is
    empty. Read as UTF-8, tolerating the BOM a spreadsheet app may add when the user
    re-saves the file. Raises one of CSV_READ_ERRORS when the file exists but can't be
    read, including a header without the "filename" column (ExportCsvFormatError)."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames  # None for an empty file: nothing to refuse, nothing to keep
        if header is not None and CSV_REQUIRED_FIELD not in header:
            raise ExportCsvFormatError(
                f"its header has no '{CSV_REQUIRED_FIELD}' column; it may have been re-saved "
                "with a different delimiter")
        return list(reader)


def read_export_csv_or_exit(path: Path) -> list[dict[str, str]]:
    """read_export_csv for the export stages: an unreadable CSV stops the export with a
    clear message instead of a traceback. The stages read it before cutting any clip, so
    the user loses no time, and the file (which may hold other VODs' rows) is never
    overwritten with this VOD's rows alone."""
    try:
        return read_export_csv(path)
    except CSV_READ_ERRORS as err:
        raise SystemExit(f"error: can't read the existing {path} ({err}). It may hold other "
                         "VODs' clips, so it was left untouched and nothing was exported. "
                         "Move it out of the folder (or re-save it as a comma-separated "
                         "UTF-8 CSV) and export again.") from err


def is_vod_row(row: Mapping[str, str], stem: str) -> bool:
    """Whether CSV `row` belongs to the VOD named `stem`: its "vod" value is that stem."""
    return (row.get("vod") or "") == stem


def is_unlabeled_row(row: Mapping[str, str]) -> bool:
    """Whether CSV `row` predates the "vod" column (no VOD recorded): a row an export into
    this folder replaces, as the build that wrote it would have (see the banner above)."""
    return not (row.get("vod") or "")


def is_other_vod_row(row: Mapping[str, str], stem: str) -> bool:
    """Whether CSV `row` belongs to a VOD other than `stem`: labeled, and not with that
    stem. Unlabeled rows are nobody's (see is_unlabeled_row)."""
    return not is_unlabeled_row(row) and not is_vod_row(row, stem)


def merge_vod_rows(existing_rows: list[dict[str, str]], new_rows: list[dict], stem: str
                   ) -> list[dict]:
    """The shared CSV's rows after the VOD named `stem` exports `new_rows`: every existing
    row of that VOD is dropped, and `new_rows` take the place of the first one, so a
    re-exported VOD keeps its position in the file. A VOD exporting here for the first
    time is appended at the end. Other VODs' rows are kept as they are, in order; rows
    from before the "vod" column are dropped (see is_unlabeled_row)."""
    merged: list[dict] = []
    new_rows_placed = False
    for row in existing_rows:
        if is_other_vod_row(row, stem):
            merged.append(row)
        elif is_vod_row(row, stem) and not new_rows_placed:
            merged.extend(new_rows)
            new_rows_placed = True
    if not new_rows_placed:
        merged.extend(new_rows)
    return merged


def other_vod_clip_counts(rows: list[dict[str, str]], stem: str) -> dict[str, int]:
    """How many clips each VOD other than `stem` has in the shared CSV `rows`, keyed by
    VOD stem in the order the VODs first appear. Rows from before the "vod" column are
    not counted: an export replaces them."""
    counts: dict[str, int] = {}
    for row in rows:
        if is_other_vod_row(row, stem):
            counts[row["vod"]] = counts.get(row["vod"], 0) + 1
    return counts


def other_vod_filenames(rows: list[dict[str, str]], stem: str) -> frozenset[str]:
    """The clip filenames that VODs other than `stem` already use in the shared CSV
    `rows`, lower-cased and without the extension: the form _title_filename's
    de-duplication set holds. Seeding that set with them keeps this VOD's title-based
    names off the other VODs' clips: a set split across a stream restart has the same
    title in both VODs, and the second export would otherwise overwrite the first's clip.
    This VOD's own old names are left out, so a re-export keeps its filenames; so are the
    names of rows from before the "vod" column, which the export replaces."""
    return frozenset(Path(row["filename"]).stem.lower() for row in rows
                     if row.get("filename") and is_other_vod_row(row, stem))


def print_kept_vods(existing_rows: list[dict[str, str]], stem: str) -> None:
    """Log which other VODs' rows an export kept in the shared CSV, so a multi-VOD
    tournament's export log says the file covers more than this VOD; and how many rows
    from before the "vod" column it replaced, so their absence is no surprise."""
    counts = other_vod_clip_counts(existing_rows, stem)
    for other_vod, count in counts.items():
        print(f"         kept {count} clip(s) from {other_vod}")
    replaced = sum(1 for row in existing_rows if is_unlabeled_row(row))
    if replaced:
        print(f"         replaced {replaced} row(s) from an export before the vod column "
              "(their clip files stay in the folder)")

def export_match_clips(matches: list[Match], video_path: str, stem: str,
                       clip_dir: Path, start_buffer: float, end_buffer: float,
                       tournament_name: str = "", title_limit: int = 100,
                       title_filenames: bool = False,
                       use_sponsors: bool = True,
                       taken_names: frozenset[str] = frozenset()) -> list[dict]:
    """Cut one clip per match (buffers applied to each match's markers) and return the CSV rows;
    also writes the per-clip title-length report into clip_dir. `taken_names` are filenames
    other VODs already use in clip_dir (see other_vod_filenames). See export_reviewed for the
    review-screen path that exports from human-corrected absolute boundaries."""
    print(f"\nSaving game clips to {clip_dir}/")
    warn_if_clip_paths_cramped(clip_dir)
    rows = []
    title_records: list[dict] = []
    used_names: set[str] = set(taken_names)
    # Half overlay reads only name a clip on a VOD whose overlay provably reads as full
    # pairs somewhere (see grouping.grouping_tags).
    trust_half_overlay = _overlay_reads_full_pairs(matches)
    for m in matches:
        default_name = f"{stem}_game_{m['match_number']:02d}"
        start = max(0.0, m["start"]["timestamp_sec"] - start_buffer)
        meta = ClipMeta(f"{default_name}.mp4", m.get("ingame_tags"),
                        m.get("tournament_tags"), m.get("round") or "",
                        m["start"]["timestamp_sec"], m["end"]["timestamp_sec"],
                        best_of=m.get("best_of") or "",
                        characters=m.get("characters"),
                        tournament_sponsors=m.get("tournament_sponsors"),
                        grouping_tags=_grouping_tags("auto", m.get("tournament_tags"),
                                                     m.get("ingame_tags"),
                                                     trust_half_overlay=trust_half_overlay),
                        first_game_offset_sec=m["start"]["timestamp_sec"] - start)
        row = _make_csv_row(meta, vod=stem, tournament_name=tournament_name,
                            use_sponsors=use_sponsors, title_limit=title_limit,
                            title_records=title_records)
        if title_filenames:
            row["filename"] = _title_filename(row["title"], default_name, used_names, clip_dir,
                                              tournament_name=tournament_name)
        clip_filename = row["filename"]
        buf   = 0.0 if m["end"].get("synthesized") else end_buffer
        end   = m["end"]["timestamp_sec"] + buf
        clip_match(video_path, start, end, str(clip_dir / clip_filename))
        print(f"{SAVED_LINE_PREFIX} {clip_filename}  frames {m['start']['frame_number']}→{m['end']['frame_number']}")
        rows.append(row)
    report_titles(title_records, title_limit, clip_dir)
    return rows


def export_set_clips(sets: list[SetGroup], video_path: str, stem: str,
                     clip_dir: Path, start_buffer: float, end_buffer: float,
                     tournament_name: str = "", title_limit: int = 100,
                     title_filenames: bool = True,
                     use_sponsors: bool = True,
                     taken_names: frozenset[str] = frozenset()) -> list[dict]:
    """Cut one clip per tournament set (first match's start to last match's end, buffers applied)
    and return the CSV rows; also writes the per-clip title-length report into clip_dir.
    `taken_names` are filenames other VODs already use in clip_dir (see other_vod_filenames)."""
    print(f"\n{len(sets)} tournament set(s) found. Saving to {clip_dir}/")
    warn_if_clip_paths_cramped(clip_dir)
    rows = []
    title_records: list[dict] = []
    used_names: set[str] = set(taken_names)
    for s in sets:
        first_match = s["matches"][0]
        match_nums  = [m["match_number"] for m in s["matches"]]
        start_frame = first_match["start"]["frame_number"]
        end_frame   = s["matches"][-1]["end"]["frame_number"]
        if not s["all_confirmed"]:
            print(f"  Warning: set {s['set_number']} has unconfirmed ingame tags")
        if s.get("tournament_tags"):
            label = f"{s['tournament_tags']} (tournament)"
        else:
            label = f"{s['ingame_tags']} (ingame)"
        print(f"  Set {s['set_number']}: {label}  games={match_nums}  "
              f"{s['start_timestamp']:.2f}s → {s['end_timestamp']:.2f}s  "
              f"frames {start_frame}→{end_frame}")
        default_name = f"{stem}_set_{s['set_number']:02d}"
        start = max(0.0, s["start_timestamp"] - start_buffer)
        meta = ClipMeta(f"{default_name}.mp4", s.get("ingame_tags"),
                        s.get("tournament_tags"), first_match.get("round") or "",
                        s["start_timestamp"], s["end_timestamp"],
                        best_of=first_match.get("best_of") or "",
                        match_count=len(s["matches"]),
                        matches=",".join(str(n) for n in match_nums),
                        characters=s.get("characters"),
                        tournament_sponsors=s.get("tournament_sponsors")
                                            or first_match.get("tournament_sponsors"),
                        grouping_tags=s.get("grouping_tags"),
                        first_game_offset_sec=s["start_timestamp"] - start)
        row = _make_csv_row(meta, vod=stem, tournament_name=tournament_name,
                            use_sponsors=use_sponsors, title_limit=title_limit,
                            title_records=title_records)
        if title_filenames:
            row["filename"] = _title_filename(row["title"], default_name, used_names, clip_dir,
                                              tournament_name=tournament_name)
        clip_filename = row["filename"]
        buf   = 0.0 if s["matches"][-1]["end"].get("synthesized") else end_buffer
        end   = s["end_timestamp"] + buf
        clip_match(video_path, start, end, str(clip_dir / clip_filename))
        print(f"{SAVED_LINE_PREFIX} {clip_filename}")
        rows.append(row)
    report_titles(title_records, title_limit, clip_dir)
    return rows


def clip_unit(data: ReviewedData) -> str:
    """The word for one exported clip in `data`'s mode: "set" in tournament-set mode,
    "game" in match mode. The one home for the mode→unit mapping: export log lines,
    single-clip headers, default filenames, and the GUI's dialog captions all use it
    (the Review screen's capitalized twin is ReviewSession.unit_label)."""
    return "set" if data["mode"] != "match" else "game"


def export_reviewed(data: ReviewedData, clip_dir: Path,
                    title_limit: int | None = None,
                    title_filenames: bool | None = None,
                    use_sponsors: bool | None = None) -> list[dict]:
    """Cut clips + write the CSV from a reviewed JSON (see vodparse.cli.export_reviewed / the GUI).

    Unlike export_match_clips/export_set_clips, the per-clip start/end seconds are used verbatim
    , no start/end buffer math, because the reviewer already set the exact boundaries by eye.
    `data` carries video_path, stem, mode ("match" | "tournament_set"), tournament_name, and a
    `clips` list of dicts with clip_number, match_numbers, start_sec, end_sec, grouping_tags,
    tournament_tags, ingame_tags, characters, round, best_of. The title character limit, whether
    clip files are named after their titles, and whether sponsors prefix the title tags come from
    the arguments when given, else the GUI-injected "title_char_limit"/"title_filenames"/
    "use_sponsors" in `data`, else the defaults (100; titles for sets, numbered names for
    matches; sponsors on).

    The CSV may already hold other VODs of the same tournament: only this VOD's rows are
    replaced (see merge_vod_rows). Returns this VOD's rows."""
    video = data["video_path"]
    stem  = data["stem"]
    unit  = clip_unit(data)
    title_limit, title_filenames, use_sponsors = _reviewed_title_settings(
        data, title_limit, title_filenames, use_sponsors)

    csv_path = _reviewed_csv_path(data, clip_dir)
    existing_rows = read_export_csv_or_exit(csv_path)   # before any clip is cut

    print(f"\nSaving {len(data['clips'])} {unit} clip(s) to {clip_dir}/")
    warn_if_clip_paths_cramped(clip_dir)
    rows = []
    title_records: list[dict] = []
    taken_names = other_vod_filenames(existing_rows, stem)
    for clip, row in _reviewed_rows(data, clip_dir, title_limit=title_limit,
                                    title_filenames=title_filenames,
                                    use_sponsors=use_sponsors, title_records=title_records,
                                    taken_names=taken_names):
        clip_filename = row["filename"]
        start, end = clip["start_sec"], clip["end_sec"]
        clip_match(video, start, end, str(clip_dir / clip_filename))
        print(f"{SAVED_LINE_PREFIX} {clip_filename}  {start:.2f}s → {end:.2f}s")
        rows.append(row)
    report_titles(title_records, title_limit, clip_dir)

    write_csv_or_exit(merge_vod_rows(existing_rows, rows, stem), csv_path)
    print(f"  CSV:   {csv_path}")
    print_kept_vods(existing_rows, stem)
    return rows


def _reviewed_csv_path(data: ReviewedData, clip_dir: Path) -> Path:
    """The metadata CSV a reviewed export writes into `clip_dir`: named after
    export_basename (not the folder, which a --folder override may rename)."""
    is_set = data["mode"] != "match"
    return clip_dir / f"{export_basename(data.get('tournament_name', ''), data['stem'], is_set)}.csv"


def _reviewed_title_settings(data: ReviewedData,
                             title_limit: int | None = None,
                             title_filenames: bool | None = None,
                             use_sponsors: bool | None = None) -> tuple[int, bool, bool]:
    """The export's effective (title_limit, title_filenames, use_sponsors): each from the
    argument when given, else the GUI-injected value in `data`, else the default (100;
    titles for sets, numbered names for matches; sponsors on)."""
    is_set = data["mode"] != "match"
    if title_limit is None:
        title_limit = data.get("title_char_limit") or 100
    if title_filenames is None:
        title_filenames = data.get("title_filenames")
        if title_filenames is None:
            title_filenames = is_set
    if use_sponsors is None:
        use_sponsors = data.get("use_sponsors")
        if use_sponsors is None:
            use_sponsors = True
    return title_limit, title_filenames, use_sponsors


def _first_game_offset(clip: Clip) -> float | None:
    """Seconds from a reviewed clip's start to its first game's start: the lead-in the
    reviewer kept before the game. None for a reviewed.json written before first_game_sec
    existed; 0 when the reviewer cut the clip to start inside the game."""
    first_game_sec = clip.get("first_game_sec")
    if first_game_sec is None:
        return None
    return max(0.0, first_game_sec - clip["start_sec"])


def _reviewed_rows(data: ReviewedData, clip_dir: Path, *,
                   title_limit: int, title_filenames: bool, use_sponsors: bool,
                   title_records: list[dict],
                   taken_names: frozenset[str]) -> list[tuple[Clip, dict]]:
    """Each projection clip paired with its CSV row, final filename included, in export
    order. This is the naming walk shared by export_reviewed and planned_clip_filename:
    a title-based filename dedups (" (2)"-style) against `taken_names` (other VODs' clips
    in clip_dir; see other_vod_filenames) and the *earlier* clips' names, and fits
    clip_dir's path budget, so the pairs must always be built in order, even when only
    one clip is wanted. Appends to `title_records` (report_titles' input); cuts nothing
    and touches no files."""
    stem            = data["stem"]
    is_set          = data["mode"] != "match"
    tournament_name = data.get("tournament_name", "")
    unit            = clip_unit(data)
    pairs: list[tuple[Clip, dict]] = []
    used_names: set[str] = set(taken_names)
    for clip in data["clips"]:
        default_name = f"{stem}_{unit}_{clip['clip_number']:02d}"
        match_nums = clip.get("match_numbers") or []
        meta = ClipMeta(
            f"{default_name}.mp4", clip.get("ingame_tags"), clip.get("tournament_tags"),
            clip.get("round") or "", clip["start_sec"], clip["end_sec"],
            best_of=clip.get("best_of") or "",
            match_count=len(match_nums) if is_set else 1,
            matches=",".join(str(n) for n in match_nums) if is_set else "",
            characters=clip.get("characters"),
            tournament_sponsors=clip.get("tournament_sponsors"),
            grouping_tags=clip.get("grouping_tags"),
            first_game_offset_sec=_first_game_offset(clip))
        row = _make_csv_row(meta, vod=stem, tournament_name=tournament_name,
                            use_sponsors=use_sponsors, title_limit=title_limit,
                            title_records=title_records)
        if title_filenames:
            row["filename"] = _title_filename(row["title"], default_name, used_names, clip_dir,
                                              tournament_name=tournament_name)
        pairs.append((clip, row))
    return pairs


def planned_clip_filename(data: ReviewedData, clip_number: int, clip_dir: Path) -> str:
    """The filename a full export into `clip_dir` would give clip `clip_number`: the
    save-dialog default for the Review tree's single-clip export, so a one-off file is
    named exactly like its regular-export counterpart. Replays the whole naming walk
    (duplicate-title suffixes depend on the earlier clips and on other VODs' clips listed
    in clip_dir's CSV) without cutting anything. Rows from before the CSV's "vod" column
    are left out (the export replaces them), and an unreadable CSV counts as no CSV: this is only the save
    dialog's suggestion, which the user can still edit. Raises ValueError for a
    clip_number not in the projection (numbering may gap: excluded sets keep their
    number but are dropped)."""
    title_limit, title_filenames, use_sponsors = _reviewed_title_settings(data)
    csv_path = _reviewed_csv_path(data, clip_dir)
    try:
        taken_names = other_vod_filenames(read_export_csv(csv_path), data["stem"])
    except CSV_READ_ERRORS:
        taken_names = frozenset()
    for clip, row in _reviewed_rows(data, clip_dir, title_limit=title_limit,
                                    title_filenames=title_filenames,
                                    use_sponsors=use_sponsors, title_records=[],
                                    taken_names=taken_names):
        if clip["clip_number"] == clip_number:
            return row["filename"]
    raise ValueError(f"no clip numbered {clip_number} in the reviewed data "
                     f"(is it excluded from export?)")


def export_single_clip(data: ReviewedData, clip_number: int, out_path: str) -> None:
    """Cut exactly one reviewed clip to a caller-chosen file (the Review tree's right-click
    "Export to a file…" action; see vodparse.cli.export_reviewed --only-clip/--out-file).

    Like export_reviewed, the clip's start/end seconds are used verbatim: the reviewer set
    the boundaries by eye; so this respects every current edit the projection carries.
    Unlike export_reviewed, nothing but the one video file is produced: no CSV, no title
    report, and the caller names the output, so the title/filename machinery is bypassed.
    `clip_number` identifies the clip inside data["clips"] (numbering may gap: excluded
    sets keep their number but are dropped from the projection)."""
    clip = next((c for c in data["clips"] if c["clip_number"] == clip_number), None)
    if clip is None:
        raise SystemExit(f"error: no clip numbered {clip_number} in the reviewed data "
                         f"(it may be excluded from export)")
    unit = clip_unit(data)
    # The header keeps the "  Saved:" line newline-preceded, which count_saved_lines
    # (the GUI's progress counter) relies on.
    print(f"\nSaving {unit} {clip_number} to {out_path}")
    # The caller chose this exact path, so unlike the full export (see _fit_stem) there is
    # no filename to truncate on their behalf; warn about the Windows budget, cut anyway.
    path_len = len(str(Path(out_path).absolute()))
    if path_len > _MAX_CLIP_PATH:
        print(f"  WARNING: the chosen path is {path_len} characters long, close to or "
              f"beyond the ~260-character limit; past it Windows lists the file but "
              f"other apps may not open it. Pick a shallower destination to be safe.")
    start, end = clip["start_sec"], clip["end_sec"]
    clip_match(data["video_path"], start, end, out_path)
    print(f"{SAVED_LINE_PREFIX} {Path(out_path).name}  {start:.2f}s → {end:.2f}s")
