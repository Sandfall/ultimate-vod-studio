"""CSV metadata and video clip export for matches and tournament sets."""

import csv
import re
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
    "スマブラ大会" is kept, not discarded): letting callers apply their own fallback."""
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
    " (2)"-style suffix for a duplicate title within the same export so clips can't
    overwrite each other. `fallback` (the numbered default, no extension) covers an
    empty-after-sanitizing title."""
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


def _make_csv_row(meta: ClipMeta, *, tournament_name: str = "",
                  use_sponsors: bool = True, title_limit: int = 100,
                  title_records: list[dict] | None = None) -> dict:
    """Build one clip's CSV row (keys = CSV_FIELDS) from its ClipMeta plus the export-wide
    knobs. When `title_records` is given, append this row's {row, action, original} title-fit
    record to it for report_titles."""
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


def write_csv(rows: list[dict], path: str) -> None:
    """Write the export rows to `path` as a CSV with the fixed CSV_FIELDS header."""
    # Plain UTF-8, no BOM: the primary reader of this CSV is our own upload assistant, not
    # Excel, so round-trip fidelity with the readers below wins over Excel's BOM convenience.
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def export_match_clips(matches: list[Match], video_path: str, stem: str,
                       clip_dir: Path, start_buffer: float, end_buffer: float,
                       tournament_name: str = "", title_limit: int = 100,
                       title_filenames: bool = False,
                       use_sponsors: bool = True) -> list[dict]:
    """Cut one clip per match (buffers applied to each match's markers) and return the CSV rows;
    also writes the per-clip title-length report into clip_dir. See export_reviewed for the
    review-screen path that exports from human-corrected absolute boundaries."""
    print(f"\nSaving game clips to {clip_dir}/")
    warn_if_clip_paths_cramped(clip_dir)
    rows = []
    title_records: list[dict] = []
    used_names: set[str] = set()
    # Half overlay reads only name a clip on a VOD whose overlay provably reads as full
    # pairs somewhere (see grouping.grouping_tags).
    trust_half_overlay = _overlay_reads_full_pairs(matches)
    for m in matches:
        default_name = f"{stem}_game_{m['match_number']:02d}"
        meta = ClipMeta(f"{default_name}.mp4", m.get("ingame_tags"),
                        m.get("tournament_tags"), m.get("round") or "",
                        m["start"]["timestamp_sec"], m["end"]["timestamp_sec"],
                        best_of=m.get("best_of") or "",
                        characters=m.get("characters"),
                        tournament_sponsors=m.get("tournament_sponsors"),
                        grouping_tags=_grouping_tags("auto", m.get("tournament_tags"),
                                                     m.get("ingame_tags"),
                                                     trust_half_overlay=trust_half_overlay))
        row = _make_csv_row(meta, tournament_name=tournament_name, use_sponsors=use_sponsors,
                            title_limit=title_limit, title_records=title_records)
        if title_filenames:
            row["filename"] = _title_filename(row["title"], default_name, used_names, clip_dir,
                                              tournament_name=tournament_name)
        clip_filename = row["filename"]
        start = max(0.0, m["start"]["timestamp_sec"] - start_buffer)
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
                     use_sponsors: bool = True) -> list[dict]:
    """Cut one clip per tournament set (first match's start to last match's end, buffers applied)
    and return the CSV rows; also writes the per-clip title-length report into clip_dir."""
    print(f"\n{len(sets)} tournament set(s) found. Saving to {clip_dir}/")
    warn_if_clip_paths_cramped(clip_dir)
    rows = []
    title_records: list[dict] = []
    used_names: set[str] = set()
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
        meta = ClipMeta(f"{default_name}.mp4", s.get("ingame_tags"),
                        s.get("tournament_tags"), first_match.get("round") or "",
                        s["start_timestamp"], s["end_timestamp"],
                        best_of=first_match.get("best_of") or "",
                        match_count=len(s["matches"]),
                        matches=",".join(str(n) for n in match_nums),
                        characters=s.get("characters"),
                        tournament_sponsors=s.get("tournament_sponsors")
                                            or first_match.get("tournament_sponsors"),
                        grouping_tags=s.get("grouping_tags"))
        row = _make_csv_row(meta, tournament_name=tournament_name, use_sponsors=use_sponsors,
                            title_limit=title_limit, title_records=title_records)
        if title_filenames:
            row["filename"] = _title_filename(row["title"], default_name, used_names, clip_dir,
                                              tournament_name=tournament_name)
        clip_filename = row["filename"]
        start = max(0.0, s["start_timestamp"] - start_buffer)
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
    matches; sponsors on)."""
    video           = data["video_path"]
    stem            = data["stem"]
    is_set          = data["mode"] != "match"
    tournament_name = data.get("tournament_name", "")
    unit            = clip_unit(data)
    title_limit, title_filenames, use_sponsors = _reviewed_title_settings(
        data, title_limit, title_filenames, use_sponsors)

    print(f"\nSaving {len(data['clips'])} {unit} clip(s) to {clip_dir}/")
    warn_if_clip_paths_cramped(clip_dir)
    rows = []
    title_records: list[dict] = []
    for clip, row in _reviewed_rows(data, clip_dir, title_limit=title_limit,
                                    title_filenames=title_filenames,
                                    use_sponsors=use_sponsors, title_records=title_records):
        clip_filename = row["filename"]
        start, end = clip["start_sec"], clip["end_sec"]
        clip_match(video, start, end, str(clip_dir / clip_filename))
        print(f"{SAVED_LINE_PREFIX} {clip_filename}  {start:.2f}s → {end:.2f}s")
        rows.append(row)
    report_titles(title_records, title_limit, clip_dir)

    csv_path = str(clip_dir / f"{export_basename(tournament_name, stem, is_set)}.csv")
    write_csv(rows, csv_path)
    print(f"  CSV:   {csv_path}")
    return rows


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


def _reviewed_rows(data: ReviewedData, clip_dir: Path, *,
                   title_limit: int, title_filenames: bool, use_sponsors: bool,
                   title_records: list[dict]) -> list[tuple[Clip, dict]]:
    """Each projection clip paired with its CSV row, final filename included, in export
    order. This is the naming walk shared by export_reviewed and planned_clip_filename:
    a title-based filename dedups against the *earlier* clips' names (" (2)"-style) and
    fits clip_dir's path budget, so the pairs must always be built in order, even when
    only one clip is wanted. Appends to `title_records` (report_titles' input); cuts
    nothing and touches no files."""
    stem            = data["stem"]
    is_set          = data["mode"] != "match"
    tournament_name = data.get("tournament_name", "")
    unit            = clip_unit(data)
    pairs: list[tuple[Clip, dict]] = []
    used_names: set[str] = set()
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
            grouping_tags=clip.get("grouping_tags"))
        row = _make_csv_row(meta, tournament_name=tournament_name, use_sponsors=use_sponsors,
                            title_limit=title_limit, title_records=title_records)
        if title_filenames:
            row["filename"] = _title_filename(row["title"], default_name, used_names, clip_dir,
                                              tournament_name=tournament_name)
        pairs.append((clip, row))
    return pairs


def planned_clip_filename(data: ReviewedData, clip_number: int, clip_dir: Path) -> str:
    """The filename a full export into `clip_dir` would give clip `clip_number`: the
    save-dialog default for the Review tree's single-clip export, so a one-off file is
    named exactly like its regular-export counterpart. Replays the whole naming walk
    (duplicate-title suffixes depend on the earlier clips) without cutting anything.
    Raises ValueError for a clip_number not in the projection (numbering may gap:
    excluded sets keep their number but are dropped)."""
    title_limit, title_filenames, use_sponsors = _reviewed_title_settings(data)
    for clip, row in _reviewed_rows(data, clip_dir, title_limit=title_limit,
                                    title_filenames=title_filenames,
                                    use_sponsors=use_sponsors, title_records=[]):
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
