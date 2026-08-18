"""The Review screen's Tools-menu whole-VOD tag-text fix-ups: no Qt, no I/O.

Two passes: Fix capitalization (``title_case_tag`` / ``capitalization_fixes``) and Split
sponsor from tag (``split_leading_sponsor`` / ``sponsor_splits``, with
``sponsor_split_preview`` rendering the pending splits for the options dialog). Each pass
computes the per-clip changes it *would* make (clip index, new values) rows and never
mutates a clip; session.py's ops apply the rows, and the empty list doubles as the "nothing
to do" signal the view uses to word its confirm. Split out of model.py (the
clip/bounds domain logic) as a text-normalization concern that only touches
``grouping_tags``/``tournament_sponsors``.
"""

from collections.abc import Mapping
from typing import Literal, NamedTuple

from vodparse.gui.review.model import ReviewClip, nth_or_blank


def title_case_tag(tag: str) -> str:
    """Capitalize each word of a tag ("my awesome tag" -> "My Awesome Tag"). Per-word
    str.capitalize() (not str.title()) keeps an apostrophe clean ("don't", not "Don'T")."""
    return " ".join(w.capitalize() for w in tag.split())


# The three Fix-capitalization modes the options dialog offers, and its default. Single source
# of truth: the dialog seeds its radio group from CAP_MODE_DEFAULT and the pure functions below
# default to it, mirroring the SPONSOR_*_DEFAULT block.
CapMode = Literal["title", "sentence", "lower"]
CAP_MODE_DEFAULT: CapMode = "title"


def apply_capitalization(tag: str, mode: CapMode) -> str:
    """Re-case a single tag under one of the three modes: "title" title-cases each word
    ("org sponsor" -> "Org Sponsor"), "sentence" capitalizes only the first word and lowercases
    the rest ("org sponsor" -> "Org sponsor"), "lower" lowercases everything. An empty tag is
    returned unchanged (sentence mode has no first word to capitalize)."""
    if mode == "lower":
        return tag.lower()
    if mode == "sentence":
        parts = tag.split()
        return " ".join([parts[0].capitalize(), *(w.lower() for w in parts[1:])]) if parts else tag
    return title_case_tag(tag)  # "title"


# Common short words that lead a two-word handle but aren't a sponsor: skip them so "Mr Example" /
# "The Player" aren't split. Matched case-insensitively; extend as new false positives turn up.
_TAG_PREFIX_STOPWORDS = frozenset({
    "mr", "ms", "dr", "the", "an", "of", "to", "in", "on", "at", "my", "by", "is", "it",
    "big", "lil", "top", "new", "old", "one", "two", "for", "you", "and",
})

# Defaults for a sponsor-shaped token: length bounds and how many leading words a sponsor may
# span. Single source of truth: the Split-sponsor options dialog pre-fills from these, and the
# pure functions below default to them.
SPONSOR_MIN_LEN_DEFAULT = 2
SPONSOR_MAX_LEN_DEFAULT = 3
SPONSOR_MAX_WORDS_DEFAULT = 1


def split_leading_sponsor(
    tag: str,
    *,
    min_len: int = SPONSOR_MIN_LEN_DEFAULT,
    max_len: int = SPONSOR_MAX_LEN_DEFAULT,
    max_words: int = SPONSOR_MAX_WORDS_DEFAULT,
    all_caps: bool = False,
) -> tuple[str, str | None]:
    """Split a colour-merged sponsor off the front of a gamertag: some overlays tell the
    sponsor from the tag only by colour, which OCR can't see, so "ORG PlayerOne" arrives as one tag.

    A token qualifies as sponsor-shaped when its length is within `min_len`..`max_len`, it
    contains a letter, it isn't a common word, and (with `all_caps`) it is fully uppercase.
    The stopword check is case-insensitive; the token's own capitalization is kept.

    Peel up to `max_words` leading qualifying words (a multi-org handle like "ORG SP PlayerFour"
    yields sponsor "ORG SP" at max_words=2), stopping at the first word that doesn't qualify:
    and always leaving the final word as the tag, so a sponsor-shaped gamertag is never
    stripped down to nothing.

    Returns (rest-of-tag, sponsor) when something was peeled; (tag, None) otherwise."""
    parts = tag.split()
    if len(parts) < 2:
        return (tag, None)

    def qualifies(token: str) -> bool:
        return (min_len <= len(token) <= max_len
                and any(c.isalpha() for c in token)
                and token.lower() not in _TAG_PREFIX_STOPWORDS
                and (not all_caps or token.isupper()))

    n = 0
    while n < max_words and n < len(parts) - 1 and qualifies(parts[n]):
        n += 1
    if n == 0:
        return (tag, None)
    return (" ".join(parts[n:]), " ".join(parts[:n]))


def capitalization_fixes(
    clips: list[ReviewClip],
    *,
    mode: CapMode = CAP_MODE_DEFAULT,
    excluded_tags: frozenset[str] = frozenset(),
    overrides: Mapping[str, str] = {},
) -> list[tuple[int, list[str | None]]]:
    """(clip index, re-cased grouping_tags) for every clip a Fix-capitalization pass would
    change; empty when the pass would be a no-op. `mode` picks the transform (see
    apply_capitalization) and `excluded_tags` is the options dialog's per-run "leave this tag
    alone" set, keyed on the slot's exact tag text.

    `overrides` is the dialog's per-run hand-edits, mapping a slot's pre-sweep tag text to the
    exact final text the user typed over the computed re-case; an overridden slot gets that text
    verbatim instead of `apply_capitalization`. Overrides share the exclusion keying (and its
    granularity: every slot with that tag text takes the same override) and only reach slots the
    sweep would touch anyway, so a key naming a mixed-case or None slot sits dormant. An
    exclusion beats an override, and an override equal to the original text simply drops the
    slot from the change list.

    Only **single-case** slots are touched: a slot re-cases iff its tag isn't None, is entirely
    one case (str.isupper() or str.islower()), and isn't excluded. Already-mixed-case tags like
    `xPLAYERTHREE`/`PlayerOne` pass through unchanged; that intentional styling is exactly what a blind
    per-word re-case would clobber. A tag with no cased characters (`"123"`) is neither upper nor
    lower, so it also passes through. A None slot (a player whose tag was never read or edited)
    passes through too, so the pass can't register a phantom change on a half-tagged set.

    Because the scope skips mixed-case tags, "lower" mode can't reach a tag a prior "title" run
    already produced (`Playerone` is mixed-case now): re-casing after an Apply isn't symmetric, and
    Undo is the way back."""
    def recased(t: str | None) -> str | None:
        if t is not None and (t.isupper() or t.islower()) and t not in excluded_tags:
            return overrides[t] if t in overrides else apply_capitalization(t, mode)
        return t

    changes = []
    for idx, clip in enumerate(clips):
        tags = list(clip.get("grouping_tags") or [])
        fixed = [recased(t) for t in tags]
        if fixed != tags:
            changes.append((idx, fixed))
    return changes


def sponsor_splits(
    clips: list[ReviewClip],
    *,
    min_len: int = SPONSOR_MIN_LEN_DEFAULT,
    max_len: int = SPONSOR_MAX_LEN_DEFAULT,
    max_words: int = SPONSOR_MAX_WORDS_DEFAULT,
    all_caps: bool = False,
    excluded_tags: frozenset[str] = frozenset(),
    overrides: Mapping[str, tuple[str, str]] = {},
) -> list[tuple[int, list[str | None], list[str | None]]]:
    """(clip index, new grouping_tags, new tournament_sponsors) for every clip where a leading
    sponsor can move off a tag into the sponsor slot. The keyword params tune what counts as
    sponsor-shaped (see split_leading_sponsor).

    A slot that already carries a sponsor is append-only: its words count against `max_words`,
    and any leftover budget keeps peeling off the tag onto the end of the sponsor (so a
    max_words=1 run leaving sponsor "ORG" / tag "SP PlayerFour" is finished by a max_words=2
    re-run). The existing sponsor text itself is never rewritten or removed.

    `excluded_tags` is the options dialog's per-run "don't split this tag" set, keyed on the
    slot's current tag text: any slot whose tag text is an exact member is left untouched, even
    when a sibling slot in the same clip still splits.

    `overrides` is the dialog's per-run hand-edits, mapping a slot's pre-sweep tag text to the
    exact (sponsor, tag) pair the user typed over the computed split. An overridden slot gets
    both texts verbatim, and the sponsor text replaces the whole slot, pre-existing sponsor
    included: the dialog's preview displays the final slot value, so what the user edited is
    what lands. Overrides share the exclusion keying (and its granularity: every splitting slot
    with that tag text takes the same pair) and apply only where the sweep would split anyway,
    so a key naming a non-splitting slot sits dormant; an exclusion beats an override, and an
    override that reproduces the slot's original state (its tag unchanged and its sponsor still
    what it was) leaves that slot untouched — nothing is mutated or padded — so a clip changed by
    nothing else drops out of the list."""
    def split(clip: ReviewClip) -> tuple[list[str | None], list[str | None]]:
        tags = list(clip.get("grouping_tags") or [])
        sponsors = list(clip.get("tournament_sponsors") or [])
        for i in range(len(tags)):
            key = tags[i] or ""   # the slot's pre-sweep tag text: exclusion + override key
            if key in excluded_tags:
                continue  # the dialog opted this exact tag text out of the sweep
            existing = nth_or_blank(sponsors, i).strip()
            remaining_words = max_words - len(existing.split())
            if remaining_words <= 0:
                continue  # the sponsor already spans the word budget; never rewrite it
            rest, peeled = split_leading_sponsor(
                key, min_len=min_len, max_len=max_len, max_words=remaining_words,
                all_caps=all_caps)
            if not peeled:
                continue
            if key in overrides:
                override_sponsor, override_tag = overrides[key]
                # An override typed back to the slot's original state (tag unchanged, sponsor
                # still exactly what it was) is a no-op: skip the slot outright so nothing is
                # mutated or padded, rather than writing new_sponsor "" onto a short/absent list
                # and registering a phantom change.
                if override_tag == key and override_sponsor == existing:
                    continue
                new_tag, new_sponsor = override_tag, override_sponsor
            else:
                new_tag, new_sponsor = rest, (f"{existing} {peeled}" if existing else peeled)
            tags[i] = new_tag
            sponsors += [""] * (i + 1 - len(sponsors))   # pad up to this slot before writing it
            sponsors[i] = new_sponsor
        return tags, sponsors

    changed = []
    for idx, clip in enumerate(clips):
        new_tags, new_sponsors = split(clip)
        if (new_tags != list(clip.get("grouping_tags") or [])
                or new_sponsors != list(clip.get("tournament_sponsors") or [])):
            changed.append((idx, new_tags, new_sponsors))
    return changed


class SponsorSplitPreviewRow(NamedTuple):
    """One row of the options dialog's live preview: the sponsor and tag a slot would end
    up with after the sweep, how many tag slots land on exactly that result across the VOD
    (named ``slots`` rather than "count": tuple.count is already taken), ``tag_before``: the
    slot's pre-sweep tag text, the exact key the dialog hands back in the sweep's
    ``excluded_tags`` set when the user opts this row out, and ``sponsor_before``: the slot's
    pre-sweep sponsor (stripped), so the dialog can recognise a row hand-edited back to its
    original ``(sponsor_before, tag_before)`` state as a no-op."""
    sponsor: str
    tag: str
    slots: int
    tag_before: str
    sponsor_before: str


def sponsor_split_preview(
    clips: list[ReviewClip],
    *,
    min_len: int = SPONSOR_MIN_LEN_DEFAULT,
    max_len: int = SPONSOR_MAX_LEN_DEFAULT,
    max_words: int = SPONSOR_MAX_WORDS_DEFAULT,
    all_caps: bool = False,
) -> list[SponsorSplitPreviewRow]:
    """The distinct sponsor/tag results a sponsor_splits run with these params would leave
    behind, in first-seen order: the options dialog's live preview. Built on sponsor_splits
    itself (same params), so the preview can never disagree with the sweep. The same player
    recurs across a VOD's sets, so identical results collapse into one row with a slot count;
    the total affected slots is the sum of the rows' ``slots``.

    Rows dedupe on (pre-sweep sponsor stripped, pre-sweep tag text): the state that fully
    determines the result, rather than on the result pair, so two slots with distinct sources
    but an identical after-sweep result never merge into one row. The preview stays
    exclusion-blind (no ``excluded_tags`` param): every row is shown and the dialog styles the
    opted-out ones itself, keyed on each row's ``tag_before``."""
    counts: dict[tuple[str, str], int] = {}
    rows: dict[tuple[str, str], SponsorSplitPreviewRow] = {}
    changes = sponsor_splits(clips, min_len=min_len, max_len=max_len,
                             max_words=max_words, all_caps=all_caps)
    for idx, new_tags, new_sponsors in changes:
        old_tags = list(clips[idx].get("grouping_tags") or [])
        old_sponsors = list(clips[idx].get("tournament_sponsors") or [])
        for i in range(len(new_tags)):
            # nth_or_blank both sides: a None slot and a "" slot are the same "no tag" for
            # display purposes, so an untouched None slot can't register as a phantom row.
            if (nth_or_blank(new_tags, i) == nth_or_blank(old_tags, i)
                    and nth_or_blank(new_sponsors, i) == nth_or_blank(old_sponsors, i)):
                continue
            tag_before = nth_or_blank(old_tags, i)
            sponsor_before = nth_or_blank(old_sponsors, i).strip()
            key = (sponsor_before, tag_before)
            counts[key] = counts.get(key, 0) + 1
            rows.setdefault(key, SponsorSplitPreviewRow(
                nth_or_blank(new_sponsors, i).strip(), nth_or_blank(new_tags, i), 0,
                tag_before, sponsor_before))
    return [row._replace(slots=counts[key]) for key, row in rows.items()]


class CapPreviewRow(NamedTuple):
    """One row of the Fix-capitalization dialog's live preview: the tag text ``before`` the
    sweep and ``after`` it, plus ``slots``: how many tag slots across the VOD start from exactly
    that ``before`` text (named ``slots`` rather than "count": tuple.count is already taken).
    ``before`` doubles as the exclusion key the dialog hands back in the sweep's ``excluded_tags``
    set when the user opts this row out."""
    before: str
    after: str
    slots: int


def capitalization_preview(
    clips: list[ReviewClip],
    *,
    mode: CapMode = CAP_MODE_DEFAULT,
) -> list[CapPreviewRow]:
    """The distinct before/after tag results a capitalization_fixes run in this `mode` would
    leave behind, in first-seen order: the options dialog's live preview. Built on
    capitalization_fixes itself (same mode), so the preview can never disagree with the sweep.
    The same player recurs across a VOD's sets, so identical before texts collapse into one row
    with a slot count; the total affected slots is the sum of the rows' ``slots``.

    Rows dedupe on ``before`` (the pre-sweep tag text): at a fixed mode it fully determines the
    result. The preview stays exclusion-blind (no ``excluded_tags`` param): every row is shown and
    the dialog styles the opted-out ones itself, keyed on each row's ``before``."""
    counts: dict[str, int] = {}
    rows: dict[str, CapPreviewRow] = {}
    changes = capitalization_fixes(clips, mode=mode)
    for idx, new_tags in changes:
        old_tags = list(clips[idx].get("grouping_tags") or [])
        for i in range(len(new_tags)):
            before = nth_or_blank(old_tags, i)
            after = nth_or_blank(new_tags, i)
            if before == after:   # an untouched slot (None/"" or a skipped mixed-case tag)
                continue
            counts[before] = counts.get(before, 0) + 1
            rows.setdefault(before, CapPreviewRow(before, after, 0))
    return [row._replace(slots=counts[key]) for key, row in rows.items()]
