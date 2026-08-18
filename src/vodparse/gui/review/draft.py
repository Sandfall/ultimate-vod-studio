"""Draft (de)serialization and file I/O for the Review screen: no Qt.

Edits are auto-saved to a per-vod draft JSON so a session can be resumed (the Resume edits
button) or survive quitting mid-review. This module is the pure half of that: turning the
in-memory clips/matches into the slim on-disk shape and back, plus the path-taking read/
write/clear helpers. `screen.py` keeps the Qt-side orchestration (the debounce timer,
when to save, status messages) and the project-root-relative `draft_path`; it hands an
explicit `Path` to the I/O helpers here.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from vodparse.gui.review.model import ReviewClip, ReviewMeta
from vodparse.jsonio import write_json_atomic
from vodparse.schema import Marker, Match


class DraftMarker(TypedDict):
    """One start/end boundary as written to the draft: `match_to_draft`'s inner `marker()`
    helper slims a full `Marker` down to just these fields, dropping the event payload
    (texts/bbox/label). `timestamp_sec` is always written; `frame_number` is absent on
    drafts saved before markers carried frame numbers; it's
    read whole (`_marker_from_draft`) rather than field-by-field, so nothing here is
    individually required on read, but `timestamp_sec` is the one field every marker the
    pipeline produces always has, so it's kept required in this type."""
    timestamp_sec: float
    frame_number: NotRequired[int]
    synthesized: NotRequired[bool]


class DraftMatch(TypedDict):
    """One match/game as written to the draft by `match_to_draft`. `match_from_draft` getitems
    `match_number`/`start`/`end` with no fallback, so those three are required here; every
    voted field is read with `.get` (missing means "never voted"), so those stay optional."""
    match_number: int
    start: DraftMarker
    end: DraftMarker
    tournament_tags: NotRequired[list[str | None] | None]
    tournament_sponsors: NotRequired[list[str | None] | None]
    ingame_tags: NotRequired[list[str | None] | None]
    round: NotRequired[str | None]
    best_of: NotRequired[str | None]
    duration_sec: NotRequired[float]


class DraftClip(TypedDict):
    """One clip as written to the draft by `clip_to_draft`. `clip_from_draft` getitems
    `clip_number`/`start_sec`/`end_sec` with no fallback, so those three are required here;
    every other field is read with `.get` and (for the tag lists) a legacy fallback chain,
    so an older draft missing a key still loads."""
    clip_number: int
    start_sec: float
    end_sec: float
    grouping_tags: NotRequired[list[str | None]]
    detected_tags: NotRequired[list[str | None]]
    detected_sponsors: NotRequired[list[str | None]]
    tournament_tags: NotRequired[list[str | None]]
    tournament_sponsors: NotRequired[list[str | None]]
    ingame_tags: NotRequired[list[str | None]]
    characters: NotRequired[list[str]]
    round: NotRequired[str]
    best_of: NotRequired[str]
    exclude_from_export: NotRequired[bool]
    start_marked: NotRequired[bool]
    end_marked: NotRequired[bool]
    matches: NotRequired[list[DraftMatch]]


class DraftPayload(TypedDict, total=False):
    """The full on-disk review_draft.json shape written by `draft_payload()`. Every key is
    optional: `ReviewSession.from_draft` reads `meta`/`clips`/`synth_counter` via `.get`
    throughout, tolerating a partial or legacy file (and `saved_at` is written but never
    read back)."""
    meta: ReviewMeta
    synth_counter: int
    saved_at: str
    clips: list[DraftClip]


def draft_payload(meta: ReviewMeta, synth_counter: int,
                  clips: list[ReviewClip]) -> DraftPayload:
    """The full draft JSON structure, stamped with the current save time."""
    return {
        "meta": meta,
        "synth_counter": synth_counter,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "clips": [clip_to_draft(c) for c in clips],
    }


def clip_to_draft(clip: ReviewClip) -> DraftClip:
    return {
        "clip_number": clip["clip_number"],
        "start_sec": clip["start_sec"],
        "end_sec": clip["end_sec"],
        "grouping_tags": list(clip["grouping_tags"]),
        "detected_tags": list(clip["detected_tags"]),
        "detected_sponsors": list(clip["detected_sponsors"]),
        "tournament_tags": list(clip["tournament_tags"]),
        "tournament_sponsors": list(clip.get("tournament_sponsors") or []),
        "ingame_tags": list(clip["ingame_tags"]),
        "characters": list(clip["characters"]),
        "round": clip["round"],
        "best_of": clip["best_of"],
        "exclude_from_export": clip.get("exclude_from_export", False),
        "start_marked": clip.get("start_marked", False),
        "end_marked": clip.get("end_marked", False),
        "matches": [match_to_draft(m) for m in clip["_matches"]],
    }


def match_to_draft(m: Match) -> DraftMatch:
    def marker(side: Marker) -> DraftMarker:
        """Slim a marker for the draft: drop the event payload (texts/bbox/label) but
        keep the marker fields, so a resumed draft's markers still carry frame numbers."""
        out: DraftMarker = {"timestamp_sec": side["timestamp_sec"]}
        if "frame_number" in side:
            out["frame_number"] = side["frame_number"]
        if side.get("synthesized"):
            out["synthesized"] = True
        return out
    return {
        "match_number": m["match_number"],
        "start": marker(m["start"]),
        "end": marker(m["end"]),
        "tournament_tags": m.get("tournament_tags"),
        "tournament_sponsors": m.get("tournament_sponsors"),
        "ingame_tags": m.get("ingame_tags"),
        "round": m.get("round"),
        "best_of": m.get("best_of"),
        "duration_sec": m.get("duration_sec"),
    }


def clip_from_draft(c: DraftClip) -> ReviewClip:
    matches = [match_from_draft(m) for m in c.get("matches", [])]
    return {
        "clip_number": c["clip_number"],
        "match_numbers": [m["match_number"] for m in matches],
        "start_sec": c["start_sec"],
        "end_sec": c["end_sec"],
        # Drafts written before grouping_tags existed fall back to the old display chain.
        "grouping_tags": list(c.get("grouping_tags") or c.get("tournament_tags")
                              or c.get("ingame_tags") or []),
        # Pre-propagation drafts: treat the saved values as the detected originals, so
        # Propagate tags simply requires a fresh edit after resuming.
        "detected_tags": list(c.get("detected_tags") or c.get("grouping_tags")
                              or c.get("tournament_tags") or c.get("ingame_tags") or []),
        "detected_sponsors": list(c.get("detected_sponsors")
                                  or c.get("tournament_sponsors") or []),
        "tournament_tags": list(c.get("tournament_tags") or []),
        "tournament_sponsors": list(c.get("tournament_sponsors") or []),
        "ingame_tags": list(c.get("ingame_tags") or []),
        "characters": list(c.get("characters") or ["", ""]),
        "round": c.get("round", ""),
        "best_of": c.get("best_of", ""),
        "exclude_from_export": c.get("exclude_from_export", False),
        "start_marked": c.get("start_marked", False),
        "end_marked": c.get("end_marked", False),
        "_matches": matches,
    }


def _marker_from_draft(m: DraftMarker) -> Marker:
    """A fresh runtime Marker copied from a draft marker. Legacy tolerance: drafts saved
    before markers carried frame numbers lack `frame_number`, which `Marker` declares
    required, hence the cast rather than a default of 0, so re-saving such a draft never
    invents a frame number (marker_fps degrades to 0.0 on these, exactly as before)."""
    return cast(Marker, dict(m))


def match_from_draft(m: DraftMatch) -> Match:
    return {
        "match_number": m["match_number"],
        "start": _marker_from_draft(m["start"]),
        "end": _marker_from_draft(m["end"]),
        "tournament_tags": m.get("tournament_tags"),
        "tournament_sponsors": m.get("tournament_sponsors"),
        "ingame_tags": m.get("ingame_tags"),
        "round": m.get("round"),
        "best_of": m.get("best_of"),
        "duration_sec": m.get("duration_sec", 0),
        "frames": [],   # stripped from the draft; no edit path consumes them
    }


def read_draft(path: Path) -> DraftPayload | None:
    """The draft at `path`, or None if it's missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_draft_file(path: Path, payload: DraftPayload) -> None:
    """Atomically write `payload` to `path`, creating the per-vod folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload)


def clear_draft_file(path: Path) -> None:
    """Remove the draft at `path` if it exists."""
    path.unlink(missing_ok=True)
