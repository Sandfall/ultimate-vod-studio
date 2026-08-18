"""Per-VOD sidecar recording the tag-reading flags that produced the current matches.json.

Every extract run writes ``json_output/<stem>/extract_flags.json`` capturing the
*resolved* tag-reading flags of the **last** run (the smart winner's flags on a
``--smart-tags`` run, or the manual CLI flags otherwise), plus the run context (mode,
group-by, set-characters, sponsors). Nothing else recorded which flags produced a given
matches.json: ``smart_tags_report.json`` covers only smart runs and isn't invalidated by
a later manual run, so the GUI's "Re-read tags" editor needs a single source of truth for
"what flags is the current review actually showing?". That is this file.

It **complements, never replaces** ``smart_tags_report.json``. A stale report left behind
after a manual re-read is harmless: the smart-tags summary popup is gated on the run's
kickoff argv (not the report), and this sidecar, not the report, is what seeds the
re-read dialog. No timestamp is stored (the file's mtime answers "when") and no scores
(the report's job); this file answers only "which flags".

The sidecar always lands in ``output_dir(stem)`` even when ``--output`` redirects
matches.json elsewhere: the same deliberate behavior as the smart report, since both are
per-VOD provenance keyed to the stem, and the GUI only ever reads the default location.

This module is intentionally Qt-free and cv2-free (unlike vodparse.cli.extract_matches, which
imports cv2): vodparse.gui.app reads the sidecar at startup and deliberately avoids pulling cv2 in.
It is not smart-specific either, so it doesn't belong in smart_tags.py.
"""

from dataclasses import asdict, fields
from pathlib import Path
from typing import TypedDict

from vodparse.jsonio import read_json
from vodparse.smart_tags import FlagCombo

# The sidecar filename written next to matches.json (see extract_matches).
EXTRACT_FLAGS_NAME = "extract_flags.json"
# Schema version, bumped when the shape changes so a future reader can migrate/reject.
EXTRACT_FLAGS_VERSION = 1

# The four tag-reading flag keys, taken straight from FlagCombo so the two never drift.
FLAG_KEYS = frozenset(f.name for f in fields(FlagCombo))


class ExtractFlagsSidecar(TypedDict):
    """The extract_flags.json payload (see the module docstring).

    ``flags`` mirrors ``asdict(FlagCombo)``: its keys match the four GUI extract-option
    keys exactly (see EXTRACT_OPT_DEFAULTS), so the re-read dialog can seed straight from
    it. ``cli_flags`` is the human-readable reproduce line (the flags actually set)."""
    version: int
    smart_tags: bool
    flags: dict[str, bool]
    cli_flags: list[str]
    mode: str
    group_by: str
    set_characters: str
    sponsors: bool


def build_sidecar(*, smart_tags: bool, combo: FlagCombo, mode: str, group_by: str,
                  set_characters: str, sponsors: bool) -> ExtractFlagsSidecar:
    """Assemble the sidecar for a finished run. ``combo`` is the resolved flag combination
    (the smart winner's on a ``--smart-tags`` run, the CLI flags otherwise); ``smart_tags``
    records whether the sweep chose it, which drives the dialog's Smart/Manual default."""
    return ExtractFlagsSidecar(
        version=EXTRACT_FLAGS_VERSION,
        smart_tags=smart_tags,
        flags=asdict(combo),
        cli_flags=combo.cli_flags(),
        mode=mode,
        group_by=group_by,
        set_characters=set_characters,
        sponsors=sponsors,
    )


def read_sidecar(path: Path) -> ExtractFlagsSidecar | None:
    """Read the sidecar at ``path``, or ``None`` if it's missing, unreadable, or the wrong
    shape (a pre-sidecar VOD, a truncated file, a future schema). Deliberately lenient: the
    caller falls back to today's global options when this returns ``None``, so a bad file
    self-heals on the next re-read rather than raising."""
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if set(data) != set(ExtractFlagsSidecar.__annotations__):
        return None
    if data.get("version") != EXTRACT_FLAGS_VERSION:
        return None
    flags = data.get("flags")
    if not isinstance(flags, dict) or set(flags) != FLAG_KEYS:
        return None
    return data  # type: ignore[return-value]
