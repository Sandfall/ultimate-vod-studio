"""Qt-free editing model for the Review screen: the clip list, the undo/redo stacks, and
every edit operation, with the pure clip/match transforms delegated to model.py and the
draft (de)serialization to draft.py.

screen.ReviewScreen owns the widgets: it resolves tree/player events into calls here
and applies each returned :class:`OpResult` (status text, tree repopulate or in-place
relabel/restyle, selection restore). Confirms, dialogs, and the playhead are the view's
problem: ops take them as arguments or are gated by queries here before the view asks the
user.

This module never imports PyQt6; that is its core invariant. It sees no tree and no player:
the acting selection comes in as a :class:`Selection`, playheads as plain floats, and modal
confirmations happen in the view before the op is called.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple

from vodparse.gui.review.model import (
    MIN_SPLIT_MATCH_SEC,
    BoundMark,
    ReviewClip,
    ReviewMeta,
    apply_bound_marks,
    build_clips,
    build_clips_from_segments,
    build_reviewed_data,
    character_targets,
    collect_bound_marks,
    copy_clips,
    default_bound_src,
    gap_fill_pair,
    hms,
    nth_or_blank,
    pair,
    propagation_targets,
    reapply_manual_edits,
    reseed_set_bounds,
    set_player_slot,
    split_match_at,
)
from vodparse.gui.review.tag_fixes import (
    CAP_MODE_DEFAULT,
    CapMode,
    SPONSOR_MAX_LEN_DEFAULT,
    SPONSOR_MAX_WORDS_DEFAULT,
    SPONSOR_MIN_LEN_DEFAULT,
    capitalization_fixes,
    sponsor_splits,
)
from vodparse.gui.review.draft import DraftPayload, clip_from_draft, draft_payload
from vodparse.schema import Match, ReviewedData

# The clip keys a field edit may touch. The per-source tournament/in-game tags stay as
# detected: only the identity tags, sponsors, and characters are editable (the sources
# feed the CSV posterity columns).
EDITABLE_CLIP_FIELDS = ("grouping_tags", "tournament_sponsors", "characters",
                        "round", "best_of")


class UndoEntry(NamedTuple):
    """One undo/redo snapshot: the clips plus the selection to return to (the set/clip index and,
    in set mode, the highlighted child match's number) and a short label naming the op, so undo/redo
    can re-select the exact row the user was on and report "Undid <op>"."""
    clips: list[ReviewClip]
    clip_index: int | None
    match_number: int | None   # the selected child match, or None for a set row / match mode
    label: str                 # short op name, e.g. "split", "exclude", "edit"


class Selection(NamedTuple):
    """The acting tree selection at op time, resolved by the view: the current clip index
    and, in set mode, the highlighted child match's number. Recorded in each undo entry so
    undo/redo can re-select the exact row. The session never tracks the live selection."""
    clip_index: int | None
    match_number: int | None


@dataclass(frozen=True)
class OpResult:
    """What the view applies after a successful op: the status line, how much of the tree to
    refresh, and where the selection should land. Produced only by ReviewSession ops and
    consumed by ReviewScreen._apply_result."""
    status: str                                  # status-line text ("" = leave it alone)
    rebuilt: bool = False                        # True → repopulate the whole tree
    changed_indices: tuple[int, ...] = ()        # rows to relabel + restyle in place
    select: int | None = None                    # top-level row to select (already clamped)
    focus_match: int | None = None               # match number to re-select; wins over select
    extra_select_matches: tuple[int, ...] = ()   # Move-block multi-select restore only


@dataclass(frozen=True)
class OpRejected:
    """An op that refused to run: `message` goes to the status line, or to a modal warning
    when `warn` is set (the too-short-match cases)."""
    message: str
    warn: bool = False


class ReviewSession:
    """The Review screen's editing model. Owns the clips, the meta describing the load, and the
    undo/redo machinery; every mutation goes through an op method that snapshots first and
    returns an :class:`OpResult` for the view to apply.

    Public attributes (`clips`, `meta`, `undo_stack`, `redo_stack`, `resumed`) may be read
    freely by the view and tests, but are mutated only via the ops here.

    Invariants:
    - Undo snapshots and the baseline are always taken through model.copy_clips:
      clip dicts are never shared between the live list and a snapshot. Match dicts ARE
      shared by design: they are never mutated in place (split_match_at / gap_fill_pair
      build fresh dicts).
    - _restore adopts entry.clips directly (the entry was popped and is discarded; the
      reciprocal entry snapshots the live state first), so no extra copy is needed.
    - _synth_counter is monotonic and never rolled back by undo, so a re-split can't
      collide with a restored earlier split.
    """

    def __init__(self, meta: ReviewMeta, clips: list[ReviewClip], synth_counter: int,
                 set_characters: str, resumed: bool) -> None:
        """Adopt an already-built editing state. The GUI normally goes through from_matches
        (a fresh extract) or from_draft (Resume edits); the screen also holds an empty
        session (meta={}, clips=[]) before any load so is_dirty/queries stay callable."""
        self.clips = clips
        self.meta = meta
        self.undo_stack: list[UndoEntry] = []
        self.redo_stack: list[UndoEntry] = []
        self.resumed = resumed
        # Field edits coalesce per (clip, field): the first keystroke of a burst snapshots the
        # pre-edit state, further keystrokes in the same field ride that entry (see edit_fields).
        # Any other undoable op or a selection change ends the burst.
        self._field_burst: tuple[int | None, str] | None = None
        # A resumed draft starts edited relative to the original extract even with an empty
        # undo stack, so Reset stays offered and the draft is never auto-deleted as "clean".
        self._baseline: list[ReviewClip] = copy_clips(clips)
        # Meta edits (today just the tournament rename) live outside the undo stack, so they
        # need their own dirty marker or a rename-only session would never write a draft:
        # and _write_draft would even delete an existing one as "clean".
        self.meta_edited = False
        self._synth_counter = synth_counter   # source of unique match numbers for Split match halves
        self._set_characters = set_characters   # how a set's per-player characters collapse (join|majority)

    @classmethod
    def from_matches(cls, matches: list[Match], mode: str, group_by: str, video_path: str,
                     stem: str, tournament_name: str = "",
                     set_characters: str = "join") -> "ReviewSession":
        """A fresh review from a just-extracted match list (the model half of the screen's
        load_model)."""
        meta: ReviewMeta = {"video_path": video_path, "stem": stem, "mode": mode,
                            "tournament_name": tournament_name, "group_by": group_by}
        clips = build_clips(matches, mode, group_by, set_characters)
        synth_counter = max((m["match_number"] for m in matches), default=0)
        return cls(meta, clips, synth_counter, set_characters, resumed=False)

    @classmethod
    def from_draft(cls, draft: DraftPayload, video_path: str, stem: str,
                   set_characters: str = "join") -> "ReviewSession":
        """The editing state a saved draft describes (the model half of the screen's
        load_from_draft). The draft carries its own mode and tournament name; `video_path`
        is the current on-disk location (used in case the file moved). The resumed state is
        treated as edited and becomes the new Reset baseline."""
        meta: ReviewMeta = draft.get("meta") or {}
        clips = [clip_from_draft(c) for c in draft.get("clips", [])]
        synth_counter = draft.get("synth_counter") or max(
            (m["match_number"] for c in clips for m in c["_matches"]), default=0)
        full_meta: ReviewMeta = {"video_path": video_path, "stem": stem,
                                 "mode": meta.get("mode", "tournament_set"),
                                 "tournament_name": meta.get("tournament_name", ""),
                                 "group_by": meta.get("group_by", "auto")}
        return cls(full_meta, clips, synth_counter, set_characters, resumed=True)

    # ── Queries ─────────────────────────────────────────────────────────────────

    def unit_label(self) -> str:
        """"Set" or "Game": how the mode names a top-level row, used in every status line."""
        return "Set" if self.meta.get("mode") != "match" else "Game"

    def segments(self) -> list[list[Match]]:
        """The current grouping as an ordered list of match-groups (one per clip)."""
        return [list(clip["_matches"]) for clip in self.clips]

    def next_included_set(self, after: int) -> int | None:
        """Index of the first set after `after` still in the export, or None if every later set
        is excluded (or there are none). Drives the advance-on-exclude convenience."""
        for idx in range(after + 1, len(self.clips)):
            if not self.clips[idx].get("exclude_from_export", False):
                return idx
        return None

    def match_number_at(self, clip_idx: int, local: int) -> int:
        """The match number of game `local` within clip `clip_idx` (for confirm messages)."""
        return self.clips[clip_idx]["_matches"][local]["match_number"]

    # ── Field edits ─────────────────────────────────────────────────────────────

    def edit_fields(self, clip_idx: int, *, tags: tuple[str, str], sponsors: tuple[str, str],
                    characters: tuple[str, str], round_text: str, best_of: str,
                    burst_key: str | None, selection: Selection) -> OpResult | None:
        """Write the editable fields (EDITABLE_CLIP_FIELDS) of clip `clip_idx` from the detail
        panel's current text. The clip dict still holds the pre-keystroke values on entry, so
        pushing before the sync below snapshots exactly the state Undo should restore. One
        entry per typing burst: further keystrokes in the same (clip, field) ride it.

        A discrete (``burst_key is None``) commit whose normalized values already match the
        clip's current ones is a no-op: it returns None without pushing an undo entry (mirrors
        set_bound's no-op guard), so re-committing a whole row when only one cell was touched,
        as the Final Review dialog does, can't pile up empty undo entries. Burst edits skip
        that guard: they already coalesce into a single entry, and the ``_field_burst`` tracking
        needs every keystroke."""
        clip = self.clips[clip_idx]
        new_tags = pair(tags[0], tags[1])
        new_sponsors = pair(sponsors[0], sponsors[1])
        new_characters = [characters[0].strip(), characters[1].strip()]
        new_round = round_text.strip()
        new_best_of = best_of.strip()
        if (burst_key is None
                and new_tags == pair(nth_or_blank(clip["grouping_tags"], 0),
                                     nth_or_blank(clip["grouping_tags"], 1))
                and new_sponsors == pair(nth_or_blank(clip["tournament_sponsors"], 0),
                                         nth_or_blank(clip["tournament_sponsors"], 1))
                and new_characters == [nth_or_blank(clip["characters"], 0).strip(),
                                       nth_or_blank(clip["characters"], 1).strip()]
                and new_round == clip["round"].strip()
                and new_best_of == clip["best_of"].strip()):
            return None
        if burst_key is None or (clip_idx, burst_key) != self._field_burst:
            self._push_undo("edit", selection)
        self._field_burst = (clip_idx, burst_key) if burst_key else None
        clip["grouping_tags"] = new_tags
        clip["tournament_sponsors"] = new_sponsors
        clip["characters"] = new_characters
        clip["round"] = new_round
        clip["best_of"] = new_best_of
        return OpResult(status="", changed_indices=(clip_idx,))

    def end_field_burst(self) -> None:
        """End the current field-typing burst so the next field edit starts a fresh undo entry.
        Called when focus leaves a field (Enter/Escape/click-away/window blur) and on selection
        changes; harmless when no burst is open."""
        self._field_burst = None

    @staticmethod
    def _swap_clip_characters(clip: ReviewClip) -> None:
        """Swap one clip's port-anchored data (characters and in-game tags) between the two
        players: the pure per-clip mutation, with no undo bookkeeping. The tag/sponsor fields
        stay put: they follow the stream overlay, which is exactly what's backwards when a swap
        is needed. Callers own the undo step (see swap_characters / swap_characters_bulk)."""
        c1, c2 = nth_or_blank(clip["characters"], 0), nth_or_blank(clip["characters"], 1)
        clip["characters"] = [c2.strip(), c1.strip()]
        # The in-game tags are not a form field, so they swap directly on the clip to keep
        # the CSV posterity columns aligned with the characters.
        i1, i2 = nth_or_blank(clip["ingame_tags"], 0), nth_or_blank(clip["ingame_tags"], 1)
        clip["ingame_tags"] = pair(i2, i1)

    def swap_characters(self, clip_idx: int, selection: Selection) -> OpResult:
        """Swap the two players' characters (and in-game tags) on one clip as one undoable
        step, labelled "edit" so the status reads "Undid edit" exactly as a field edit does."""
        self._push_undo("edit", selection)
        self._swap_clip_characters(self.clips[clip_idx])
        return OpResult(status="", changed_indices=(clip_idx,))

    def swap_characters_bulk(
        self, clip_indices: Sequence[int], selection: Selection,
    ) -> OpResult | None:
        """Swap characters for several clips at once as a SINGLE undoable step: the Final
        Review dialog swaps every selected row in one action, so one Undo must reverse the whole
        action, not one row per press. Duplicate indices collapse (keeping first-seen order) so a
        clip is never swapped twice into a no-op; an empty selection returns None so it can't push
        an empty undo entry. Every swapped clip is reported in changed_indices so both views
        refresh."""
        unique = list(dict.fromkeys(clip_indices))
        if not unique:
            return None
        self._push_undo("edit", selection)
        for idx in unique:
            self._swap_clip_characters(self.clips[idx])
        return OpResult(status="", changed_indices=tuple(unique))

    # ── Bounds ──────────────────────────────────────────────────────────────────

    def set_bound(self, clip_idx: int, key: Literal["start_sec", "end_sec"], sec: float,
                  label: str, selection: Selection) -> OpResult | None:
        """Stamp a clip's start/end (`key` picks which bound; `label` names it in the
        status). Each stamp is one undoable step (snapshot the old bound first); a no-op
        restamp to the same value returns None so it can't pile up empty undo entries."""
        clip = self.clips[clip_idx]
        if clip[key] == sec:
            return None
        self._push_undo("trim", selection)
        clip[key] = sec
        # Flag the bound as hand-marked so a later restructure re-stamps it instead of
        # re-deriving it from the matches (see model.collect_bound_marks / apply_bound_marks).
        clip["start_marked" if key == "start_sec" else "end_marked"] = True
        return OpResult(
            status=f"Marked {label} of {self.unit_label()} {clip['clip_number']} → {hms(sec)}")

    # ── Undo / redo / reset ─────────────────────────────────────────────────────

    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    def _push_undo(self, label: str, selection: Selection) -> None:
        """Record the clips plus the selection to return to and a short `label` naming the op.
        Every new edit invalidates the redo trail and ends any field-typing burst (a follow-up
        keystroke in the same field then snapshots afresh; edit_fields re-arms its burst after
        calling this).

        Private: every op that starts an undoable step lives inside ReviewSession, so nothing
        outside this module ever needs to call it."""
        self.undo_stack.append(
            UndoEntry(self._snapshot(), selection.clip_index, selection.match_number, label))
        self.redo_stack.clear()
        self._field_burst = None

    def undo(self, current: Selection) -> OpResult | None:
        """Step back one entry, parking the live state (and `current`, the live selection) on
        the redo stack under the same label so the opposite stack names the same op. None when
        there is nothing to undo."""
        if not self.undo_stack:
            return None
        entry = self.undo_stack.pop()
        self.redo_stack.append(self._current_entry(entry.label, current))
        return self._restore(entry, f"Undid {entry.label}" if entry.label else "Undid last change")

    def redo(self, current: Selection) -> OpResult | None:
        """Step forward again. The live state goes straight onto the undo stack, not through
        push_undo; that would wipe the redo trail this is walking. None when there is nothing
        to redo."""
        if not self.redo_stack:
            return None
        entry = self.redo_stack.pop()
        self.undo_stack.append(self._current_entry(entry.label, current))
        return self._restore(entry, f"Redid {entry.label}" if entry.label else "Redid last change")

    def reset_all(self) -> OpResult:
        """Wipe both stacks and restore the state the session was loaded with. (The screen
        confirms first and flushes the draft after: a clean screen removes its saved draft.)"""
        return self._reset_to_baseline("Reset all edits")

    def reset_to_resumed(self) -> OpResult:
        """Fall back to the state the draft was resumed with. `resumed` stays set and the draft
        stays on disk: the restored state IS the draft's content, which still differs from the
        original extract and must survive quitting the app."""
        return self._reset_to_baseline("Reset to the resumed draft")

    def reset_to_extract(self, matches: list[Match]) -> OpResult:
        """Rebuild from the original matches.json, exactly as from_matches would have: the
        resumed draft's edits are discarded. (The screen removes the draft after, since the
        session now matches the extract again.)"""
        self.resumed = False
        self.clips = build_clips(matches, self.meta["mode"], self.meta["group_by"],
                                 self._set_characters)
        self._synth_counter = max((m["match_number"] for m in matches), default=0)
        self._baseline = self._snapshot()
        return self._reset_to_baseline("Reset to the original extract")

    def _reset_to_baseline(self, status: str) -> OpResult:
        """Wipe both stacks and restore the baseline snapshot."""
        self.undo_stack.clear()
        self.redo_stack.clear()
        return self._restore(UndoEntry(copy_clips(self._baseline), 0, None, ""), status)

    def _current_entry(self, label: str, selection: Selection) -> UndoEntry:
        """The live state as an undo entry, tagged with `label` (reused from the entry being
        undone/redone so the opposite stack names the same op)."""
        return UndoEntry(self._snapshot(), selection.clip_index, selection.match_number, label)

    def _snapshot(self) -> list[ReviewClip]:
        return copy_clips(self.clips)

    def _restore(self, entry: UndoEntry, status: str) -> OpResult:
        """Adopt an entry's clips and describe the selection to return to. Prefer the exact
        child match row the user was on (focus_match: the view's re-select also seeks the
        playhead back to that game); fall back to the set/clip row, clamped in case the
        restored list is shorter."""
        self.clips = entry.clips
        self._field_burst = None
        if not self.clips:
            return OpResult(status=status, rebuilt=True)
        select = (0 if entry.clip_index is None
                  else max(0, min(entry.clip_index, len(self.clips) - 1)))
        return OpResult(status=status, rebuilt=True,
                        select=select, focus_match=entry.match_number)

    # ── Restructure ─────────────────────────────────────────────────────────────
    # Merge, split, and move act on the segment list (see segments()) and hand the edited
    # grouping to _rebuild_from_segments below, which turns it back into clips. Each op pushes
    # its own undo entry first (see push_undo), builds an `inherit` map so the restructure
    # preserves manual edits (typed tags/characters/rounds/sponsors, exclude flags), then
    # returns the rebuild's OpResult.

    def _identity_inherit(self) -> list[list[int]]:
        """Inherit map (see model.reapply_manual_edits) for a restructure that keeps
        every set's identity: each new clip continues the old clip at the same index. Ops
        that split, merge, or drop sets adjust this before handing it on."""
        return [[i] for i in range(len(self.clips))]

    def merge(self, left: int, *, focus_match: int | None,
             selection: Selection) -> OpResult | None:
        """Merge clip `left` with the clip after it into one set. None when `left` isn't a
        valid merge point (no clip follows it): the screen's edge guards normally keep this
        from being called at all, but a stray call is a no-op rather than an IndexError."""
        segs = self.segments()
        if not 0 <= left < len(segs) - 1:
            return None
        self._push_undo("merge", selection)
        inherit = self._identity_inherit()
        merged = segs[left] + segs[left + 1]
        segs[left:left + 2] = [merged]
        inherit[left:left + 2] = [[left, left + 1]]   # merged set keeps both sources' edits
        return self._rebuild_from_segments(
            segs, select=left, focus_match=focus_match, inherit=inherit,
            status=f"Merged Set {left + 1} + Set {left + 2} → Set {left + 1}")

    def split_set(self, clip_idx: int, local: int, selection: Selection) -> OpResult | None:
        """Split the clip at `clip_idx` into two sets at match `local`.

        On a later match (the original behavior): split *before* it, so the selected match
        and those after it form the new set. On the first match: split *after* it, so the
        games after it form the new set while the first match stays in, and selected on,
        the original set. Either way the highlighted match follows across the split. None
        when there's nothing to split off (the first-match path on a lone-match set)."""
        segs = self.segments()
        matches = segs[clip_idx]
        # First match splits after itself (keeping selection on the original set); a lone
        # match has nothing to peel off. Later matches split before themselves, as before.
        if local <= 0:
            if len(matches) < 2:
                return None
            cut, select, where = 1, clip_idx, "after"
        else:
            cut, select, where = local, clip_idx + 1, "before"
        match_no = matches[local]["match_number"]
        self._push_undo("split", selection)
        inherit = self._identity_inherit()
        segs[clip_idx:clip_idx + 1] = [matches[:cut], matches[cut:]]
        # both halves inherit the original set's edits
        inherit[clip_idx:clip_idx + 1] = [[clip_idx], [clip_idx]]
        # A hand-marked start belongs to the original first game (now in the head) and a marked
        # end to the last game (now in the tail): the head keeps only the start, the tail only
        # the end; the seam edge each gains re-derives.
        start_src, end_src = default_bound_src(inherit)
        end_src[clip_idx] = None            # head's new end is a seam → re-derive
        start_src[clip_idx + 1] = None      # tail's new start is a seam → re-derive
        return self._rebuild_from_segments(
            segs, select=select, focus_match=match_no, inherit=inherit,
            start_src=start_src, end_src=end_src,
            status=f"Split Set {clip_idx + 1} {where} game {match_no}")

    def split_match(self, clip_idx: int, playhead_sec: float,
                    selection: Selection) -> OpResult | OpRejected:
        """Split at the playhead, acting on wherever the playhead actually is within clip
        `clip_idx` (not necessarily the selected match). Inside a match this cuts it into a
        head and tail; in dead space (a gap between matches or the head/tail buffer) it inserts
        a new match filling that gap (see _fill_gap). Either new match can then be moved/merged
        like any detected match.

        Scoped to the clip: the playhead is windowed to [start_sec, end_sec], so the red
        ±margin (which belongs to neighbouring sets) is rejected rather than guessed at. Also
        rejected, with warn=True, when a cut would leave a match shorter than
        MIN_SPLIT_MATCH_SEC."""
        clip = self.clips[clip_idx]
        p = playhead_sec
        if not (clip["start_sec"] <= p <= clip["end_sec"]):
            return OpRejected(f"Scrub inside Set {clip_idx + 1} to split there.")
        # Cut the match the playhead is strictly inside, if any; otherwise it's dead space.
        for local, match in enumerate(clip["_matches"]):
            start_ts = match["start"]["timestamp_sec"]
            end_ts = match["end"]["timestamp_sec"]
            if start_ts < p < end_ts:
                if p - start_ts < MIN_SPLIT_MATCH_SEC or end_ts - p < MIN_SPLIT_MATCH_SEC:
                    return OpRejected(self._match_too_short_message("both halves are"), warn=True)
                self._push_undo("split game", selection)
                inherit = self._identity_inherit()   # cutting a match keeps its set's edits
                head, tail = split_match_at(match, p, self._new_match_number)
                segs = self.segments()
                segs[clip_idx][local:local + 1] = [head, tail]
                return self._rebuild_from_segments(
                    segs, select=clip_idx, focus_match=tail["match_number"], inherit=inherit,
                    status=f"Split match {match['match_number']} at {hms(p)} "
                           f"→ added match {tail['match_number']}")
        return self._fill_gap(clip_idx, p, selection)

    def _fill_gap(self, clip_idx: int, p: float, selection: Selection) -> OpResult | OpRejected:
        """The dead-space branch of split_match: split the gap the playhead sits in within clip
        `clip_idx` into two abutting matches meeting at `p`. With a preceding match (a mid-set
        gap or the tail) that match is grown forward to `p` and a new match runs from `p` to the
        next match's start, or the clip's end bound when nothing follows. In the head buffer
        (no preceding match) it mirrors: the first match is grown back to `p` and the new match
        fills from the clip's start bound up to `p`. Used mainly after Mark start/end widens a
        clip, to turn the new dead space into real matches. Rejected (no warn) when the playhead
        sits inside the preceding match's own span (e.g. exactly on a match start, or a
        contiguous seam) rather than open dead space; rejected with warn=True when the gap being
        filled is itself shorter than MIN_SPLIT_MATCH_SEC."""
        clip = self.clips[clip_idx]
        matches = clip["_matches"]
        nxt_idx = next((j for j, m in enumerate(matches)
                        if m["start"]["timestamp_sec"] > p), len(matches))
        # Reject a playhead sitting inside the preceding match's span (e.g. exactly on a match
        # start, or on a contiguous seam); there's no gap to fill there.
        if nxt_idx > 0 and p < matches[nxt_idx - 1]["end"]["timestamp_sec"]:
            return OpRejected(
                f"Nothing to split at {hms(p)}. Scrub into a game or into a gap between them.")
        segs = self.segments()
        if nxt_idx > 0:
            # Gap or tail: grow the previous match up to the playhead; the new match fills from
            # the playhead to the next match's start, or to the clip's end bound when this is
            # the tail.
            far = (matches[nxt_idx]["start"]["timestamp_sec"] if nxt_idx < len(matches)
                   else clip["end_sec"])
            if far - p < MIN_SPLIT_MATCH_SEC:
                return OpRejected(self._match_too_short_message("the new game is"), warn=True)
            grown, new = gap_fill_pair(matches[nxt_idx - 1], p, far, self._new_match_number)
            self._push_undo("fill gap", selection)
            segs[clip_idx][nxt_idx - 1] = grown
            segs[clip_idx][nxt_idx:nxt_idx] = [new]
            status = (f"Grew match {grown['match_number']} to {hms(p)}, "
                      f"added match {new['match_number']} ({hms(p)}–{hms(far)})")
        else:
            # Head buffer: grow the first match back to the playhead; the new match fills from
            # the clip's start bound up to the playhead.
            near = clip["start_sec"]
            if p - near < MIN_SPLIT_MATCH_SEC:
                return OpRejected(self._match_too_short_message("the new game is"), warn=True)
            grown, new = gap_fill_pair(matches[0], p, near, self._new_match_number)
            self._push_undo("fill gap", selection)
            segs[clip_idx][0] = grown
            segs[clip_idx][0:0] = [new]
            status = (f"Grew match {grown['match_number']} back to {hms(p)}, "
                      f"added match {new['match_number']} ({hms(near)}–{hms(p)})")
        return self._rebuild_from_segments(
            segs, select=clip_idx, focus_match=new["match_number"],
            inherit=self._identity_inherit(),   # filling a gap keeps the set's edits
            status=status)

    @staticmethod
    def _match_too_short_message(subject: str) -> str:
        """The "Game too short" warning body for a split/gap-fill that would leave a game
        under MIN_SPLIT_MATCH_SEC. `subject` names what must be long enough, e.g. "both halves
        are" or "the new game is"."""
        secs = int(MIN_SPLIT_MATCH_SEC)
        return (f"Splitting here would leave a game under {secs} seconds. "
                f"Move the playhead so {subject} at least {secs} seconds.")

    def move_block(self, clip_idx: int, local_indices: list[int], dest: int,
                   selection: Selection) -> OpResult:
        """Move the contiguous match block at `local_indices` of clip `clip_idx` into clip
        `dest` (the previous or next set). `extra_select_matches` on the result carries the
        rest of a multi-match block's numbers, so the view can extend cursor-follow to the
        whole block (the anchor, the block's first match, is already `focus_match`)."""
        self._push_undo("move", selection)
        segs = self.segments()
        # Each surviving set keeps its own edits; moving matches between sets doesn't carry the
        # source set's set-level tag/character edits into the destination (block holds matches,
        # not set fields). Identity here; the empty-set drop below trims it in lockstep with segs.
        inherit = self._identity_inherit()
        block = [segs[clip_idx][i] for i in local_indices]  # contiguous, in selection order
        keep = set(local_indices)
        # Which source edges the move disturbs (captured before the removal below): a hand-marked
        # start rides on the source's first game, a marked end on its last. `appended` also fixes
        # which of the destination's edges the block becomes.
        appended = dest < clip_idx
        first_moved = 0 in keep
        last_moved = (len(segs[clip_idx]) - 1) in keep
        segs[clip_idx] = [m for i, m in enumerate(segs[clip_idx]) if i not in keep]
        if appended:
            segs[dest] = segs[dest] + block       # append to the previous set's end
        else:
            segs[dest] = block + segs[dest]       # prepend to the next set's front
        # Moving a whole set empties its source: the destination absorbed every match, so it
        # also inherits the emptied set's edits (after its own), the same as a merge. A partial
        # move leaves the source intact, so its edits stay with it and never leak into the
        # destination.
        source_emptied = not segs[clip_idx]
        if source_emptied:
            inherit[dest].append(clip_idx)
        # A set fully emptied by the move disappears; drop its inherit slot alongside it so the
        # map stays aligned with the surviving segments.
        surviving = [(s, h) for s, h in zip(segs, inherit) if s]
        segs = [s for s, _ in surviving]
        inherit = [h for _, h in surviving]
        # Locate the destination by the block's identity after the (possibly index-shifting)
        # empty-set removal, so the status and selection name the set it actually landed in.
        anchor = block[0]
        final = next(i for i, s in enumerate(segs) if any(m is anchor for m in s))
        # Route hand-marked bounds by set edge (default keeps every untouched set's marks). Only
        # the two sets the move touches deviate: a partial move re-derives an edge whose game left
        # it (drop only, never carried to the destination); a whole-set move is a merge and re-routes
        # the emptied source's edges into the surviving set in final order.
        start_src, end_src = default_bound_src(inherit)
        if source_emptied:
            # Whole set moved → merge in the final order: [source][dest] when prepended (start from
            # the source, end from the destination); [dest][source] when appended already matches
            # the default (start from the destination, end from the source).
            if not appended:
                start_src[final] = clip_idx
                end_src[final] = dest
        else:
            # Partial move: the source (still at clip_idx, nothing was dropped) re-derives an edge
            # whose game it lost; the destination re-derives the edge the block became.
            if first_moved:
                start_src[clip_idx] = None
            if last_moved:
                end_src[clip_idx] = None
            if appended:
                end_src[final] = None
            else:
                start_src[final] = None
        nums = [m["match_number"] for m in block]
        what = (f"game {nums[0]}" if len(nums) == 1
                else f"games {nums[0]}–{nums[-1]}")
        result = self._rebuild_from_segments(
            segs, select=final, focus_match=anchor["match_number"], inherit=inherit,
            start_src=start_src, end_src=end_src,
            status=f"Moved {what} → Set {final + 1}")
        return replace(result, extra_select_matches=tuple(nums[1:]))

    # ── Exclude ──────────────────────────────────────────────────────────────────
    # Flag a whole set (or match-clip) out of the export, or isolate a single game of a set into
    # its own excluded one-game set. build_reviewed_data drops flagged clips at export time; these
    # ops only ever set/clear the flag (isolate also restructures to carve the game out first).

    def toggle_exclude(self, clip_idx: int, selection: Selection) -> OpResult:
        """Flag/unflag clip `clip_idx` so it's left out of the export. One undoable step; the
        view restyles the row from `changed_indices` and the export filter (build_reviewed_data)
        drops it at write time. Excluding also advances the selection to the next still-included
        set (`next_included_set`), so a run of junk can be skipped without re-selecting between
        each click; re-including leaves the selection where it is."""
        self._push_undo("exclude", selection)
        clip = self.clips[clip_idx]
        clip["exclude_from_export"] = not clip.get("exclude_from_export", False)
        excluded = clip["exclude_from_export"]
        status = (f"{self.unit_label()} {clip['clip_number']} "
                  f"{'excluded from' if excluded else 'included in'} export")
        select = self.next_included_set(clip_idx) if excluded else None
        return OpResult(status=status, changed_indices=(clip_idx,), select=select)

    def isolate_creates_single_game_clip(self, clip_idx: int, local: int) -> bool:
        """True when isolating game `local` would leave a single-game clip on a surviving side: a
        lone game exported as its own set, not a full set. The head (matches[:local], length
        `local`) and tail (matches[local+1:], length n-1-local) each become their own set; an edge
        game has only one surviving side. An empty side makes no clip, so only a side of length
        exactly 1 counts."""
        n = len(self.clips[clip_idx]["_matches"])
        return local == 1 or (n - 1 - local) == 1

    def isolate_and_exclude_game(self, clip_idx: int, local: int, selection: Selection) -> OpResult:
        """Exclude a single game, first, middle, or last, by splitting it into its own one-game
        set flagged out of the export, with the games before/after it forming their own sets. A
        middle game splits three ways [before][this][after]; an EDGE game has no head (local == 0)
        or tail (local == n-1) piece, so those empty pieces are dropped (build_clips_from_segments
        can't take an empty segment). One undoable step; see split_set for the same restructure
        template. The view asks isolate_creates_single_game_clip first and confirms with the user
        before calling this, so by the time this runs the exclusion is a given.

        Every mutation (the splice, the rebuild, and the isolated piece's exclude flag) happens
        here before returning, so the view applies a single OpResult and its one repopulate
        already shows the new set struck-through (no separate post-rebuild restyle needed)."""
        self._push_undo("exclude game", selection)
        segs = self.segments()
        matches = segs[clip_idx]
        match_no = matches[local]["match_number"]
        head, mid, tail = matches[:local], [matches[local]], matches[local + 1:]
        pieces = [p for p in (head, mid, tail) if p]   # an edge game has no head/tail piece
        mid_idx = clip_idx + (1 if head else 0)         # where the isolated game lands after renumber
        inherit = self._identity_inherit()
        segs[clip_idx:clip_idx + 1] = pieces
        inherit[clip_idx:clip_idx + 1] = [[clip_idx]] * len(pieces)   # every piece continues the set
        # A hand-marked start rides on the original first game (only in the head piece, absent when
        # excluding the first game) and a marked end on the last game (only in the tail piece); the
        # excluded middle and every new seam edge re-derive.
        start_src, end_src = default_bound_src(inherit)
        for j in range(len(pieces)):
            if not (head and j == 0):
                start_src[clip_idx + j] = None
            if not (tail and j == len(pieces) - 1):
                end_src[clip_idx + j] = None
        result = self._rebuild_from_segments(
            segs, select=mid_idx, focus_match=None, inherit=inherit,
            start_src=start_src, end_src=end_src,
            status=f"Excluded game {match_no} from export")
        self.clips[mid_idx]["exclude_from_export"] = True
        return result

    # ── Restructure funnel ──────────────────────────────────────────────────────

    def _rebuild_from_segments(self, segments: list[list[Match]], *, select: int,
                               status: str, focus_match: int | None = None,
                               inherit: list[list[int]] | None = None,
                               start_src: list[int | None] | None = None,
                               end_src: list[int | None] | None = None) -> OpResult:
        """Replace the clips with ones built from `segments` (renumbered 1..N), re-voting
        each set's tags from its members and re-seeding bounds. The result asks the view to
        repopulate the tree and re-select the match `focus_match` (so the cursor follows it
        across the restructure) or, failing that, the set at index `select` (clamped here).
        The caller must have pushed an undo snapshot first.

        `inherit` (see model.reapply_manual_edits) maps each new clip to the prior
        clip(s) whose manual edits it continues, so the restructure preserves typed tags,
        characters, rounds, sponsors, and exclude flags. Without it every field is re-derived
        from the matches, dropping those edits.

        `start_src`/`end_src` (see model.collect_bound_marks) route hand-marked start/end bounds
        the same way per set edge; when omitted they default to model.default_bound_src(inherit),
        which is correct for every op except the ones that divide a set (split/exclude) or move a
        set edge, which pass their own. The marks are collected from the prior clips before the
        swap and re-stamped after the bound reseed, so a marked edge survives the re-derive."""
        group_by = self.meta.get("group_by", "auto")
        fresh = build_clips_from_segments(segments, group_by, self._set_characters)
        marks: list[BoundMark] = []
        if inherit is not None:
            # Runs against the still-current self.clips (the pre-restructure state) before the
            # swap below, reading each prior clip's edits to overlay onto the rebuilt clips.
            reapply_manual_edits(fresh, self.clips, inherit, group_by, self._set_characters)
            if start_src is None or end_src is None:
                start_src, end_src = default_bound_src(inherit)
            marks = collect_bound_marks(fresh, self.clips, start_src, end_src)
        self.clips = fresh
        # build_clips_from_segments seeded each clip's bounds, but a restructure changes which
        # sets are adjacent: re-seed so contiguous-seam buffers and end clamps are recomputed.
        reseed_set_bounds(self.clips)
        # Re-stamp hand-marked bounds after the reseed so they win over the re-derived values.
        # Known edge case (accepted, revisit if it bites): because this runs after
        # clamp_clip_ends, a mark that crosses a neighbour's boundary can leave two clips
        # overlapping (or gapped): clamp capped the neighbour against the *derived* bound, which
        # the mark then overrides. This mirrors set_bound's own verbatim behaviour (a direct mark
        # doesn't clamp either), so it's a pre-existing sharp edge, not new. If overlaps become a
        # real problem, clamp/seam after stamping (or make clamp_clip_ends mark-aware) instead.
        apply_bound_marks(self.clips, marks)
        return OpResult(status=status, rebuilt=True,
                        select=max(0, min(select, len(self.clips) - 1)),
                        focus_match=focus_match)

    # ── Split-match numbering ───────────────────────────────────────────────────

    def _new_match_number(self) -> int:
        """A fresh, session-unique match number for a Split match half. Monotonic across undo
        (never rolled back) so a re-split can't collide with a restored earlier split."""
        self._synth_counter += 1
        return self._synth_counter

    # ── Propagation / batch ──────────────────────────────────────────────────────
    # Spread one set's tag/sponsor/character corrections to every other set with the same
    # player (propagate_tags/propagate_characters), or rewrite a tag-shaped field across every
    # clip in one VOD-wide sweep (fix_capitalization/split_sponsors_from_tags). The view queries
    # candidacy first (to grey out a menu entry or skip an empty confirm dialog) and only calls
    # the op once it knows there's something to do.

    def propagation_targets(self, current: int | None,
                            player: int | None = None) -> list[tuple[int, int, str, str]]:
        """Targets for Propagate-tags, scoped to `current` (see model.propagation_targets).
        Returns [] when nothing is selected: `current` is None, or stale (pointing past the end
        of a list a rebuild just shrank)."""
        if current is None or current >= len(self.clips):
            return []
        return propagation_targets(self.clips, current, player)

    def character_targets(self, current: int | None, overwrite: bool,
                          player: int | None = None) -> list[tuple[int, int, str]]:
        """Targets for the character-propagation ops, scoped to `current` (see
        model.character_targets). Returns [] when nothing is selected."""
        if current is None or current >= len(self.clips):
            return []
        return character_targets(self.clips, current, overwrite, player)

    def propagate_tags(self, current: int, player: int | None,
                       selection: Selection) -> OpResult | None:
        """Apply `current`'s tag/sponsor corrections (one player's, via `player`, or both) to
        every other set with the same player, as one undoable step. None when there's nothing to
        propagate: the view's menu entries grey out individually when they have no targets, so
        a stray call is a no-op rather than an empty undo entry."""
        targets = self.propagation_targets(current, player)
        if not targets:
            return None
        self._push_undo("propagate tags", selection)
        for idx, j, tag, sponsor in targets:
            other = self.clips[idx]
            other["grouping_tags"] = set_player_slot(other.get("grouping_tags"), j, tag)
            other["tournament_sponsors"] = set_player_slot(
                other.get("tournament_sponsors"), j, sponsor)
        changed = tuple(sorted({idx for idx, *_ in targets}))
        players = ", ".join(sorted({tag for _, _, tag, _ in targets if tag}))
        return OpResult(status=f"Propagated {players} to {len(changed)} other set(s)",
                        changed_indices=changed)

    def propagate_characters(self, current: int, overwrite: bool, player: int | None,
                             selection: Selection) -> OpResult | None:
        """Apply `current`'s characters (one player's, via `player`, or both) to every other set
        with the same player: only blank slots unless `overwrite`. None when there's nothing to
        propagate, for the same reason as propagate_tags."""
        targets = self.character_targets(current, overwrite, player)
        if not targets:
            return None
        self._push_undo("overwrite characters" if overwrite else "fill characters", selection)
        for idx, j, char in targets:
            chars = list(self.clips[idx]["characters"])
            chars += [""] * (j + 1 - len(chars))
            chars[j] = char
            self.clips[idx]["characters"] = chars
        changed = tuple(sorted({idx for idx, *_ in targets}))
        names = ", ".join(sorted({char for _, _, char in targets}))
        verb = "Overwrote" if overwrite else "Filled"
        return OpResult(status=f"{verb} {names} in {len(changed)} other set(s)",
                        changed_indices=changed)

    def capitalization_changes(
        self, *, mode: CapMode = CAP_MODE_DEFAULT, excluded_tags: frozenset[str] = frozenset(),
        overrides: Mapping[str, str] = {},
    ) -> list[tuple[int, list[str | None]]]:
        """(clip index, re-cased grouping_tags) for every clip a Fix-capitalization pass would
        change (see tag_fixes.capitalization_fixes, which `mode`, `excluded_tags` and the
        dialog's hand-edit `overrides` tune). Feeds the options dialog's preview count and the
        view's empty-run guard."""
        return capitalization_fixes(self.clips, mode=mode, excluded_tags=excluded_tags,
                                    overrides=overrides)

    def sponsor_split_changes(
        self, *, min_len: int = SPONSOR_MIN_LEN_DEFAULT, max_len: int = SPONSOR_MAX_LEN_DEFAULT,
        max_words: int = SPONSOR_MAX_WORDS_DEFAULT, all_caps: bool = False,
        excluded_tags: frozenset[str] = frozenset(),
        overrides: Mapping[str, tuple[str, str]] = {},
    ) -> list[tuple[int, list[str | None], list[str | None]]]:
        """(clip index, new grouping_tags, new tournament_sponsors) for every clip a Split-sponsor
        pass would change (see tag_fixes.sponsor_splits, which the keyword params tune;
        `excluded_tags` is its per-run "don't split this tag" set and `overrides` its per-run
        hand-edited (sponsor, tag) results). Feeds the view's empty-run guard."""
        return sponsor_splits(self.clips, min_len=min_len, max_len=max_len,
                              max_words=max_words, all_caps=all_caps, excluded_tags=excluded_tags,
                              overrides=overrides)

    def fix_capitalization(
        self, selection: Selection, *, mode: CapMode = CAP_MODE_DEFAULT,
        excluded_tags: frozenset[str] = frozenset(), overrides: Mapping[str, str] = {},
    ) -> OpResult:
        """Re-case every single-case player tag across all clips under `mode` (see
        tag_fixes.apply_capitalization; `excluded_tags` opts specific tag texts out and
        `overrides` carries the dialog's hand-edited results) as one undoable step. The view
        calls capitalization_changes() first, with the same params, and only calls this once
        it's non-empty, so this assumes there's something to fix rather than guarding against
        an empty run itself."""
        changes = self.capitalization_changes(mode=mode, excluded_tags=excluded_tags,
                                              overrides=overrides)
        self._push_undo("capitalization", selection)
        for idx, new_tags in changes:
            self.clips[idx]["grouping_tags"] = new_tags
        changed = tuple(idx for idx, _ in changes)
        unit = self.unit_label().lower()
        return OpResult(status=f"Fixed capitalization in {len(changed)} {unit}(s)",
                        changed_indices=changed)

    def split_sponsors_from_tags(
        self, selection: Selection, *, min_len: int = SPONSOR_MIN_LEN_DEFAULT,
        max_len: int = SPONSOR_MAX_LEN_DEFAULT, max_words: int = SPONSOR_MAX_WORDS_DEFAULT,
        all_caps: bool = False, excluded_tags: frozenset[str] = frozenset(),
        overrides: Mapping[str, tuple[str, str]] = {},
    ) -> OpResult:
        """Move a colour-merged sponsor off the front of each player tag into the sponsor
        field, across every clip, as one undoable step (see tag_fixes.split_leading_sponsor via
        sponsor_split_changes; the keyword params tune what counts as sponsor-shaped,
        `excluded_tags` opts specific tag texts out, and `overrides` carries the dialog's
        hand-edited (sponsor, tag) results). The view calls sponsor_split_changes() first, with
        the same params, and only calls this once it's non-empty."""
        changes = self.sponsor_split_changes(min_len=min_len, max_len=max_len,
                                             max_words=max_words, all_caps=all_caps,
                                             excluded_tags=excluded_tags, overrides=overrides)
        self._push_undo("split sponsor", selection)
        for idx, new_tags, new_sponsors in changes:
            self.clips[idx]["grouping_tags"] = new_tags
            self.clips[idx]["tournament_sponsors"] = new_sponsors
        changed = tuple(idx for idx, *_ in changes)
        unit = self.unit_label().lower()
        return OpResult(status=f"Split sponsor from {len(changed)} {unit}(s)",
                        changed_indices=changed)

    def set_tournament_name(self, name: str) -> None:
        """Change the tournament name mid-review (Tools → Change tournament name…). Not an
        undoable clip edit; it marks the session dirty via `meta_edited` instead, so a
        rename-only session still writes a draft. Deliberately survives Reset: Reset restores
        the *clips* to their baseline, while the name is VOD-level metadata the Main-page
        field (kept in sync by the caller) would immediately contradict if it reverted."""
        self.meta["tournament_name"] = name
        self.meta_edited = True

    # ── Draft / export projections ──────────────────────────────────────────────

    def is_dirty(self) -> bool:
        """Any review edits since load. Every clip edit pushes to the undo stack, so an empty
        stack means the clips match the baseline, except after resuming a draft, whose edits
        predate this session (`resumed` keeps the draft alive and Reset offered), and after a
        meta edit such as the tournament rename (`meta_edited`), which lives outside the
        undo stack."""
        return bool(self.undo_stack) or self.resumed or self.meta_edited

    def draft_payload(self) -> DraftPayload:
        """The auto-save draft payload for the current state (see draft.py)."""
        return draft_payload(self.meta, self._synth_counter, self.clips)

    def reviewed_data(self) -> ReviewedData:
        """The reviewed.json projection of the current edits: excluded sets dropped, each
        surviving set reduced to schema.Clip's fields (see model.build_reviewed_data)."""
        return build_reviewed_data(self.clips, self.meta)
