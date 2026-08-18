"""Clip-title composition: assembly, length-fitting, and the title-length report.

Pure string/tables logic plus the title_report.txt writer; no ffmpeg or cv2. The
positional tag/sponsor/character inputs may carry None for an unresolved slot;
build_title coerces them so a title never renders the literal 'None'.
"""

import argparse
from pathlib import Path


def add_title_args(parser: argparse.ArgumentParser, *, limit_default: int | None,
                   limit_default_help: str, filenames_default_help: str) -> None:
    """Add the --title-limit/--title-filenames options shared by vodparse.cli.extract_matches and
    vodparse.cli.export_reviewed. The two scripts resolve an omitted flag differently (fixed defaults
    vs deferring to the values the GUI stored in the reviewed JSON), so each passes its own
    default and '(default: ...)' help text."""
    parser.add_argument("--title-limit", type=int, default=limit_default,
                        help="Max clip-title length in characters (YouTube's limit is 100). Over-long "
                             "titles are trimmed to fit (round abbreviated, then sponsors and round "
                             f"dropped) and flagged with a warning (default: {limit_default_help})")
    parser.add_argument("--title-filenames", action=argparse.BooleanOptionalAction, default=None,
                        help="Name clip files after their titles instead of "
                             f"<stem>_<set|game>_NN.mp4 (default: {filenames_default_help})")


def player_composite(tag: str, character: str = "", sponsor: str = "") -> str:
    """One player's start.gg-style display string — "SPONSOR | TAG (CHARACTER)" — with each
    separator present only when its part is (no dangling " | " or empty "()"). The building
    block of build_title's matchup, also shown per-player by the Final Review dialog's
    compact columns."""
    # Positional tag/sponsor/character lists carry None for a position no game resolved;
    # coerce so a missing field renders as an empty slot, never the literal string "None".
    tag, character, sponsor = tag or "", character or "", sponsor or ""
    name = f"{sponsor} | {tag}" if sponsor else tag
    return f"{name} ({character})" if character else name


def build_title(p1: str, p2: str, c1: str = "", c2: str = "",
                tournament_name: str = "", round_str: str = "",
                s1: str = "", s2: str = "") -> str:
    """Compose a clip's display title from its tags, characters, sponsors, tournament name,
    and round (e.g. "Tourney - WF - PRO | ABC (Char1) vs XYZ (Char2)")."""
    tournament_name, round_str = tournament_name or "", round_str or ""
    matchup = f"{player_composite(p1, c1, s1)} vs {player_composite(p2, c2, s2)}"
    # " - "-joined so the assistant's DEFAULT_TITLE_TEMPLATE reproduces this byte-for-byte;
    # an absent tournament or round drops out without leaving a stray separator.
    return " - ".join(part for part in (tournament_name, round_str, matchup) if part)


# Round-word abbreviations used when a title exceeds the limit. Mapped tokens and the
# digits after them fuse into one run ("Losers Round 3" -> "LR3"); other words keep their
# spacing ("Top 8 Losers Semis" -> "Top 8 LSF"). Already-short acronyms (WSF, WR1, GF)
# contain no mappable words and pass through unchanged.
_ROUND_ABBREVS = {
    "winners": "W", "losers": "L", "grand": "G", "grands": "G",
    "final": "F", "finals": "F",
    "semifinal": "SF", "semifinals": "SF", "semis": "SF",
    "quarterfinal": "QF", "quarterfinals": "QF", "quarter": "QF", "quarters": "QF",
    "round": "R",
}


def abbreviate_round(round_str: str) -> str:
    """Abbreviate round words per _ROUND_ABBREVS ("Losers Round 3" -> "LR3"), fusing mapped
    tokens and the digits after them into one run while leaving other words spaced. Already-short
    acronyms (WSF, GF) pass through unchanged."""
    out: list[str] = []
    run = False  # out[-1] is an abbreviation run; the next mapped token/digit glues onto it
    for tok in round_str.split():
        abbrev = _ROUND_ABBREVS.get(tok.lower())
        if abbrev is not None or (tok.isdigit() and run):
            piece = abbrev if abbrev is not None else tok
            if run:
                out[-1] += piece
            else:
                out.append(piece)
            run = True
        else:
            out.append(tok)
            run = False
    return " ".join(out)


def fit_title(p1: str, p2: str, c1: str = "", c2: str = "",
              tournament_name: str = "", round_str: str = "",
              s1: str = "", s2: str = "",
              title_limit: int = 100) -> tuple[str, str | None]:
    """build_title, shortened with the least intervention that fits `title_limit`
    (YouTube caps titles at 100 chars): abbreviate the round, then drop the sponsor
    prefixes, then drop the round entirely. Each step keeps the previous reductions
    the tournament name, tags, and characters always stay. Returns (title, action),
    action one of None | "abbreviated" | "sponsors_dropped" | "round_dropped" |
    "too_long". A title that can't fit even bare is returned in full ("too_long")
    shortening it further wouldn't reach the limit anyway."""
    title = build_title(p1, p2, c1, c2, tournament_name, round_str, s1, s2)
    if len(title) <= title_limit:
        return title, None
    short = abbreviate_round(round_str)   # == round_str when nothing abbreviates
    if short != round_str:
        abbreviated = build_title(p1, p2, c1, c2, tournament_name, short, s1, s2)
        if len(abbreviated) <= title_limit:
            return abbreviated, "abbreviated"
    if s1 or s2:
        # Both sponsors go together: an asymmetric "PRO | ABC vs PlayerThree" reads badly.
        unsponsored = build_title(p1, p2, c1, c2, tournament_name, short)
        if len(unsponsored) <= title_limit:
            return unsponsored, "sponsors_dropped"
    if round_str:
        dropped = build_title(p1, p2, c1, c2, tournament_name, "")
        if len(dropped) <= title_limit:
            return dropped, "round_dropped"
    return title, "too_long"


_TITLE_REPORT_NAME = "title_report.txt"

_TITLE_ACTION_LABELS = {
    "abbreviated":      "ROUND ABBREVIATED",
    "sponsors_dropped": "SPONSORS REMOVED",
    "round_dropped":    "ROUND REMOVED",
    "too_long":         "COULD NOT SHORTEN: edit the title by hand before uploading",
}


def report_titles(records: list[dict], title_limit: int, clip_dir: Path) -> None:
    """Print the per-category title-length warnings and write <clip_dir>/title_report.txt
    listing each affected clip with its original and final title. A clean export removes a
    stale report left by an earlier run into the same folder. `records` come from
    _make_csv_row: {"row", "action", "original"} per clip, in export order."""
    abbreviated = sum(1 for r in records if r["action"] == "abbreviated")
    unsponsored = sum(1 for r in records if r["action"] == "sponsors_dropped")
    dropped     = sum(1 for r in records if r["action"] == "round_dropped")
    too_long    = sum(1 for r in records if r["action"] == "too_long")
    if abbreviated:
        print(f"  Warning: {abbreviated} title(s) exceeded {title_limit} characters, so the "
              f"round was abbreviated to fit (e.g. \"Winners Finals\" → \"WF\")")
    if unsponsored:
        print(f"  Warning: {unsponsored} title(s) exceeded {title_limit} characters even with "
              f"the round abbreviated, so sponsor prefixes were removed from the title")
    if dropped:
        print(f"  Warning: {dropped} title(s) exceeded {title_limit} characters even without "
              f"sponsors, so the round was removed from the title")
    if too_long:
        print(f"  Warning: {too_long} title(s) exceed {title_limit} characters and could "
              f"not be shortened. Edit them by hand before uploading.")

    report_path = clip_dir / _TITLE_REPORT_NAME
    affected = [r for r in records if r["action"]]
    if not affected:
        report_path.unlink(missing_ok=True)
        return
    lines = [f"Titles over the {title_limit}-character limit "
             f"({len(affected)} of {len(records)} clip(s) affected)", ""]
    for r in affected:
        lines.append(f"[{_TITLE_ACTION_LABELS[r['action']]}]  {r['row']['filename']}")
        if r["action"] == "too_long":
            lines.append(f"  title    ({len(r['row']['title'])}): {r['row']['title']}")
        else:
            lines.append(f"  original ({len(r['original'])}): {r['original']}")
            lines.append(f"  final    ({len(r['row']['title'])}): {r['row']['title']}")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Title report: {report_path}")
