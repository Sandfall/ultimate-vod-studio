"""Group consecutive matches with the same players into tournament sets."""

import re
from collections import Counter
from collections.abc import Callable, Iterable
from difflib import SequenceMatcher
from itertools import permutations
from typing import Literal, cast

from vodparse.schema import Match, SetGroup, TagEntry, TextBox
from vodparse.text_utils import alnum_fold
from vodparse.voting import positional_columns, vote

# Visually-confusable characters folded to a single representative so OCR slips don't
# read as different players ("PlayerOne"/"Player0ne", "a|b"/"a/b"). Lowercase keys only: _canonical
# lowercases first. Extend cautiously: folding a common letter raises the risk of merging
# two genuinely different tags.
_CONFUSABLES = str.maketrans({
    "o": "0",
    "i": "1", "l": "1", "|": "1", "/": "1", "\\": "1",
    "z": "2",
    "s": "5",
    "b": "8",
})

# Two tags count as the same player when their canonical forms are this similar or closer
# (difflib ratio, 0..1). 0.85 absorbs roughly one dropped/added character in a typical tag
# while still separating distinct names. Lower to merge more aggressively.
_SIMILARITY_THRESHOLD = 0.85


def _canonical(tag: str) -> str:
    """Lowercase and fold OCR-confusable characters so visually-equivalent readings collapse."""
    return tag.strip().lower().translate(_CONFUSABLES)


def similar(a: str, b: str) -> bool:
    """Whether two tags are the same up to minor OCR noise: a confusable substitution
    (handled by canonicalization) or a dropped/added character (handled by the ratio)."""
    ca, cb = _canonical(a), _canonical(b)
    if ca == cb:
        return True
    return SequenceMatcher(None, ca, cb).ratio() >= _SIMILARITY_THRESHOLD


def _keys_match(a: tuple[str, ...] | None, b: tuple[str, ...] | None) -> bool:
    """Fuzzy equality between two player-tag keys: True when a one-to-one pairing exists in
    which every tag matches its partner under similar. Both sides must line up, so a single
    shared player is not enough to merge two sets. Tag counts (usually two) are tiny, so an
    exhaustive pairing search is cheap."""
    if a is None or b is None or len(a) != len(b):
        return False
    tags_a = list(a)
    for paired_b in permutations(b):
        if all(similar(x, y) for x, y in zip(tags_a, paired_b)):
            return True
    return False


def _tag_key(tags: list[str | None] | None) -> tuple[str, ...] | None:
    """A match's player identity as a sorted lowercase tuple. A tuple (not a set) so a
    mirror match, both players showing the same tag, "X vs X", keeps both slots and can
    still group with its neighbours; a set would collapse it to one tag and the length
    check in _keys_match would then refuse every pairing."""
    present = [t for t in tags if t] if tags else []   # skip unresolved (None) positions
    return tuple(sorted(t.lower() for t in present)) if present else None


def _is_full_pair(key: tuple[str, ...] | None) -> bool:
    """Whether a player-tag key resolved both positions. _tag_key drops unresolved (None)
    slots, so a half-read like ["TITLE", None] yields a 1-tuple: only a read that named
    both players (a mirror match "X vs X" included) is a full pair."""
    return key is not None and len(key) == 2


def _is_same_set(group_by: str,
                 t_key: tuple[str, ...] | None, i_key: tuple[str, ...] | None,
                 cur_t_key: tuple[str, ...] | None, cur_i_key: tuple[str, ...] | None,
                 i_confirmed: bool, cur_i_confirmed: bool) -> bool:
    """Decide whether a match belongs to the currently open set. `group_by` selects the
    identity source: "overlay" forces the tournament-overlay gamertags, "ingame" forces the
    in-game nameplates, and "auto" prefers sources by completeness: a full overlay pair,
    then a full in-game pair, then a lone overlay tag, then a lone in-game tag.

    Completeness outranks source under "auto" because a lone overlay "tag" repeated on
    every game is how a stream graphic (a top-corner event title elected by the lone-tag
    salvage) glues different matches into one giant set (see is_impostor_set): when the
    overlay only half-read but both matches carry complete nameplate pairs, the pairs are
    the stronger identity. The pair rank is asymmetric about confirmation, though
    (`i_confirmed` / `cur_i_confirmed` carry each side's ingame_tags_confirmed):

    - MATCHING full pairs merge regardless of confirmation: two games misreading into
      the *same* full pair is no accident, and on VODs whose overlay reads are junk lone
      tags that differ every game, this merge is the only thing holding real sets together.
    - DIFFERING full pairs split only when both are confirmed reads. An unconfirmed pair
      varied across the game's frames, and one OCR slip inside it ("m" vs "rn") is enough
      to make two games of one real set look like different opponents: rather than let
      that overrule an overlay tag both games agree on, an unconfirmed disagreement falls
      through to the lone-tag ranks below and lets them decide.

    Lone keys still group when nothing fuller exists, so a VOD with only one readable
    tag per game keeps grouping at all. (The forced "ingame" mode stays ungated: there
    the user explicitly chose the nameplates, unconfirmed reads and all.)"""
    if group_by == "overlay":
        return _keys_match(t_key, cur_t_key)
    if group_by == "ingame":
        return _keys_match(i_key, cur_i_key)
    # auto
    if _is_full_pair(t_key) and _is_full_pair(cur_t_key):
        return _keys_match(t_key, cur_t_key)
    if _is_full_pair(i_key) and _is_full_pair(cur_i_key):
        if _keys_match(i_key, cur_i_key):
            return True     # matching full pairs merge, confirmed or not
        if i_confirmed and cur_i_confirmed:
            return False    # two vouched-for pairs really do name different opponents
        # An unconfirmed disagreement may be one OCR slip: let the lone-tag ranks decide.
    if t_key is not None and cur_t_key is not None:
        return _keys_match(t_key, cur_t_key)
    if i_key is not None and cur_i_key is not None:
        return _keys_match(i_key, cur_i_key)
    return False


# A set is one pair of players, so its games' in-game nameplates should all read the same
# pair. When the *confirmed* nameplate pairs across a set's games span at least this many
# distinct identities, the set is an impostor: consecutive different matches glued together
# by a grouping identity that isn't a player (typically a constant stream graphic, e.g. a
# top-corner event title elected as a lone overlay "gamertag" on every game). Two identities
# stay below the bar on purpose: one changed nameplate can be a mid-set OCR misread.
IMPOSTOR_MIN_IDENTITIES = 3


def confirmed_identity_count(matches: list[Match]) -> int:
    """How many distinct player-pair identities the matches' confirmed in-game nameplates
    span. Only matches whose ingame_tags_confirmed is set and whose read resolved BOTH
    positions count: unconfirmed or partial reads are OCR noise ("UI"/"UL") that would
    inflate the tally, and a genuine set keeps one identity among its confirmed pairs.
    Pairs are compared with the same fuzzy permutation-tolerant matching as set grouping
    (_keys_match), so an OCR slip doesn't read as a new pair of players."""
    identities: list[tuple[str, ...]] = []
    for match in matches:
        if not match.get("ingame_tags_confirmed"):
            continue
        key = _tag_key(match.get("ingame_tags"))
        if key is None or len(key) != 2:
            continue
        if not any(_keys_match(key, known) for known in identities):
            identities.append(key)
    return len(identities)


def is_impostor_set(matches: list[Match]) -> bool:
    """Whether a set's games span implausibly many confirmed player pairs (see
    IMPOSTOR_MIN_IDENTITIES): the signature of everything collapsing into one giant set
    because a stream graphic was mistaken for a gamertag. Shared by the smart-tags
    sweep's impostor scorer and the extraction-time warning."""
    return confirmed_identity_count(matches) >= IMPOSTOR_MIN_IDENTITIES


def overlay_reads_full_pairs(matches: Iterable[Match]) -> bool:
    """Whether any match voted a fully-resolved tournament-overlay pair: the evidence
    that the overlay is genuinely being read on this VOD. Half reads are only trustworthy
    where this holds (see grouping_tags); on a VOD with no full pair anywhere, the
    "overlay" reads are scenery/HUD noise (or the overlay is really at the bottom) and
    must not name sets."""
    return any(_is_full_pair(_tag_key(m.get("tournament_tags"))) for m in matches)


def grouping_tags(group_by: str, tournament_tags: list[str | None] | None,
                  ingame_tags: list[str | None] | None, *,
                  trust_half_overlay: bool = True) -> list[str | None] | None:
    """The tags that identify a set under `group_by`, applied to set- or match-level tags.
    None when the chosen source is absent. Under "auto" any overlay read wins, half reads
    included: a half overlay read usually carries one genuine gamertag ("?" vs "PlayerTwo"),
    which beats a complete-but-noisier nameplate pair for naming the set: PROVIDED the
    caller vouches for the overlay with `trust_half_overlay` (overlay_reads_full_pairs
    across the whole VOD). Where no match ever voted a full overlay pair, the half reads
    are noise, and the in-game tags label the set instead. VOD-constant graphics never
    reach the vote at all (tags.strip_constant_overlay_tags). Set MEMBERSHIP has the
    opposite preference (_is_same_set: full pairs first), since a lone repeated tag glues
    unrelated matches together while a label cannot."""
    if group_by == "overlay":
        return tournament_tags
    if group_by == "ingame":
        return ingame_tags
    if tournament_tags and len(tournament_tags) == 2 and all(tournament_tags):
        return tournament_tags
    if trust_half_overlay:
        return tournament_tags or ingame_tags
    return ingame_tags or tournament_tags


def overlay_fell_back_to_ingame(tournament_tags: list[str | None] | None,
                                ingame_tags: list[str | None] | None) -> bool:
    """Whether identity tags would come from the in-game nameplates only because the
    tournament-overlay tags are absent: the "auto" fallback in grouping_tags. The overlay
    is "absent" only when *no* overlay tag was voted: a partial read like ["playerone", None]
    is truthy, so grouping_tags keeps it and this stays False. Shared by the Review
    screen's fallback marking (vodparse.gui.review.model.used_ingame_fallback, which adds the
    mode/group_by gating) and the smart-tags sweep's fallback scorer, which counts raw
    overlay readability and deliberately applies no gate."""
    return not (tournament_tags or []) and bool(ingame_tags)


def _has_grouping_tag(group_by: str, t_key, i_key) -> bool:
    """Whether a match carries any tag usable under the active grouping mode."""
    if group_by == "overlay":
        return t_key is not None
    if group_by == "ingame":
        return i_key is not None
    return t_key is not None or i_key is not None


def voted_tags(matches: list[Match],
                key: Literal["tournament_tags", "ingame_tags"]) -> list[str | None]:
    """A set's tags from its member matches: the most frequent value per tag position
    across the matches' detected `key` ("tournament_tags"/"ingame_tags"). Voting across
    the games means one match with a backwards stream overlay (typically game 1, fixed
    once the streamer notices) can't dictate the set's tag order.

    Positions are KEPT (None for a slot no game resolved) so the result stays index-aligned
    with voted_sponsors and the schema's positional [p1, p2] contract: packing the Nones
    out would shift the surviving tag onto the other player's sponsor. When no position wins
    anything the list is empty, which the caller maps to None."""
    columns = positional_columns([m.get(key) or [] for m in matches])
    voted: list[str | None] = [vote(col) for col in columns]
    return voted if any(voted) else []


def voted_sponsors(matches: list[Match]) -> list:
    """Like voted_tags but for the per-position sponsors: keeps a value (possibly None) for
    every tag position so the list stays aligned with tournament_tags. Empty positions vote to
    None rather than being dropped, which would shift the remaining sponsors out of place."""
    columns = positional_columns([m.get("tournament_sponsors") or [] for m in matches])
    return [vote(col) for col in columns]


def merge_set_characters(matches: list[Match], mode: str) -> list[str | None] | None:
    """A set's [player 1, player 2] characters from its games' voted characters. A player
    may switch characters between games, so the per-side values can disagree: "join" keeps
    every distinct character in first-played order ("Char1 / Char2"), "majority" keeps the
    one played in the most games. Characters never affect set identity; only the tags do."""
    sides: list[str | None] = []
    for i in (0, 1):
        values = []
        for m in matches:
            chars = m.get("characters")
            if chars and len(chars) > i and (c := chars[i]):
                values.append(c)
        if mode == "majority":
            sides.append(vote(values))
        else:
            unique = list(dict.fromkeys(values))
            sides.append(" / ".join(unique) if unique else None)
    return sides if any(sides) else None


# A gamertag rarely ends in a space then a lone one- or two-digit number; that trailing token
# is the game/stock score OCR occasionally reads into the tag box ("PlayerOne 2"). Captures the base.
_SCORE_SUFFIX_RE = re.compile(r"^(?P<base>.+?)\s+\d{1,2}$")


def _score_suffix_base(tag: str) -> str | None:
    """The tag with a trailing " <score>" removed ("PlayerOne 2" -> "PlayerOne"), or None if it carries
    no such suffix."""
    match = _SCORE_SUFFIX_RE.match(tag.strip())
    base = match.group("base").strip() if match else None
    return base or None


def _score_variant_map(tag_lists: Iterable[list[str | None] | None]) -> dict[str, str]:
    """Map each score-suffixed tag reading to its base gamertag: but only when the base gamertag
    is *also read bare* somewhere in the VOD, so a stray score misread can't rename a genuine tag.
    "PlayerOne 2" collapses to "PlayerOne" because "PlayerOne" is read on its own elsewhere; a lone "PlayerTwo 2"
    with no bare "PlayerTwo" is left untouched (it may be a real tag), as is a best-of label misread
    into the tag slot ("FIRST TO 2" / "FIRST TO 3": there is no bare "FIRST TO"). Consolidating
    keeps an accidental score from fragmenting one player's set into "PlayerOne 0" / "PlayerOne" / "PlayerOne 2"
    pieces."""
    seen: Counter[str] = Counter()
    for tags in tag_lists:
        for tag in (tags or []):
            if tag:
                seen[tag] += 1
    canonical: dict[str, str] = {}
    for tag in seen:
        base = _score_suffix_base(tag)
        if base and base in seen:      # only when the bare base gamertag is read elsewhere
            canonical[tag] = base
    return canonical


def _stuttered_base(tag: str) -> str | None:
    """The tag with a duplicated leading fragment removed ("MR . MR. PLAYERONE" -> "MR. PLAYERONE"),
    or None when no leading token-run repeats. OCR sometimes detects a nameplate twice and
    EasyOCR merges the partial read in front of the full one, so the artifact's signature is
    the first token(s) reappearing, in order, at the start of the remainder. Compared
    alphanumerics-only so punctuation/spacing variance can't hide the repeat."""
    tokens = tag.split()
    # Largest duplicated head wins, so trailing punctuation of the fragment stays with it
    # ("MR . MR. PLAYERONE" splits after "MR ." -> "MR. PLAYERONE", not after "MR" -> ". MR. PLAYERONE").
    for head_end in reversed(range(1, len(tokens))):
        head, rest = alnum_fold(" ".join(tokens[:head_end])), alnum_fold(" ".join(tokens[head_end:]))
        if len(head) >= 2 and len(rest) >= len(head) and rest.startswith(head):
            return " ".join(tokens[head_end:])
    return None


def _stutter_variant_map(tag_lists: Iterable[list[str | None] | None]) -> dict[str, str]:
    """Map each stuttered tag reading ("MR . MR. PLAYERONE") to the clean reading it duplicates:
    but only when a matching clean form is *also read* somewhere in the VOD ("MR . PLAYERONE"),
    so a player genuinely named with a repeated word ("TAG TAG PLAYERTWO") is never renamed:
    their reads are consistently doubled and no clean form exists to corroborate a fold.
    Same count-then-fold-only-if-corroborated contract as _score_variant_map; the clean
    match is fuzzy (similar()) because the merged and standalone reads rarely agree on
    spacing, and the most-read clean form wins."""
    seen: Counter[str] = Counter()
    for tags in tag_lists:
        for tag in (tags or []):
            if tag:
                seen[tag] += 1
    canonical: dict[str, str] = {}
    for tag in seen:
        base = _stuttered_base(tag)
        if base is None:
            continue
        matches = [s for s in seen if s != tag and similar(s, base)]
        if matches:
            canonical[tag] = max(matches, key=lambda s: seen[s])
    return canonical


def group_into_sets(matches: list[Match], group_by: str = "auto",
                    set_characters: str = "join", overlay_at_bottom: bool = False) -> list[SetGroup]:
    """Group consecutive matches into tournament sets. `group_by` chooses the grouping
    identity: "overlay" uses the tournament-overlay (upper) gamertags, "ingame" uses the
    in-game (lower) nameplates, and "auto" (default) prefers sources by completeness: a
    full overlay pair, then a full in-game pair, then a lone tag from either (see
    _is_same_set for why a lone overlay tag must not outrank complete nameplate pairs —
    and why only *confirmed* pair disagreements get to split a set).
    Under a forced mode, a match lacking the chosen tag can't join a set and is exported on
    its own. `set_characters` ("join" | "majority") picks how each player's characters
    across the set's games collapse into the set-level value (see merge_set_characters).

    `overlay_at_bottom` disables the in-game fallback: an at-bottom overlay is read from the same
    lower band as the in-game nameplates, so those tags are a redundant, noisier copy: "auto"
    resolves to "overlay"-only rather than ever labelling/grouping a set by its in-game read.

    Score misreads and stuttered double-reads are consolidated first (see _score_variant_map /
    _stutter_variant_map): a tag differing from another only by a trailing " <number>" collapses
    to the shared base, and a duplicated-prefix read ("MR . MR. PLAYERONE") collapses to the clean
    read it corroborates elsewhere, so neither artifact can fragment one player's set or
    mislabel it."""
    if overlay_at_bottom and group_by == "auto":
        group_by = "overlay"

    canonical = _score_variant_map(m.get("tournament_tags") for m in matches)
    canonical.update(_stutter_variant_map(m.get("tournament_tags") for m in matches))

    def norm(tags: list[str | None] | None) -> list[str | None] | None:
        return [canonical.get(t, t) if t else t for t in tags] if tags else tags

    sets: list[SetGroup] = []
    current_set: SetGroup | None = None
    current_t_key = None   # tournament-tag key of the open set
    current_i_key = None   # ingame-tag key of the open set
    current_i_confirmed = False   # whether current_i_key came from a confirmed read

    for match in matches:
        t_key = _tag_key(norm(match.get("tournament_tags")))
        i_key = _tag_key(match.get("ingame_tags"))
        i_confirmed = bool(match.get("ingame_tags_confirmed"))

        if current_set is not None and _is_same_set(group_by, t_key, i_key,
                                                    current_t_key, current_i_key,
                                                    i_confirmed, current_i_confirmed):
            current_set["matches"].append(match)
            current_set["end_timestamp"] = match["end"]["timestamp_sec"]
            if not match.get("ingame_tags_confirmed"):
                current_set["all_confirmed"] = False
            if current_i_key is None and i_key is not None:
                current_i_confirmed = i_confirmed   # the adopted key's own confirmation
            current_t_key = current_t_key or t_key
            current_i_key = current_i_key or i_key
        else:
            if current_set is not None:
                sets.append(current_set)
            if not _has_grouping_tag(group_by, t_key, i_key):
                source = {"overlay": "overlay", "ingame": "in-game"}.get(group_by, "detected")
                print(f"  Warning: match {match['match_number']} has no {source} tags: "
                      "exported as its own set")
            current_set = {
                "set_number":      len(sets) + 1,
                "matches":         [match],
                "start_timestamp": match["start"]["timestamp_sec"],
                "end_timestamp":   match["end"]["timestamp_sec"],
                "all_confirmed":   match.get("ingame_tags_confirmed", False),
            }
            current_t_key = t_key
            current_i_key = i_key
            current_i_confirmed = i_confirmed

    if current_set is not None:
        sets.append(current_set)

    # Tags are voted per position across the set's games (not copied from game 1, whose
    # overlay is the one most often backwards). Set membership above is order-insensitive
    # (_keys_match tries permutations), so a swapped game still joins its set: the vote
    # just decides which order the set reports.
    trust_half_overlay = overlay_reads_full_pairs(matches)
    for s in sets:
        # Vote the set's tag from the score-consolidated readings so "PlayerOne 2" and "PlayerOne" count
        # as the same tag; sponsors stay index-aligned (consolidation never changes tag positions).
        voting_view = cast("list[Match]",
                           [{**m, "tournament_tags": norm(m.get("tournament_tags"))} for m in s["matches"]])
        s["tournament_tags"]     = voted_tags(voting_view, "tournament_tags") or None
        s["tournament_sponsors"] = (voted_sponsors(s["matches"])
                                    if s["tournament_tags"] else None)
        s["ingame_tags"]         = voted_tags(s["matches"], "ingame_tags") or None
        s["grouping_tags"]       = grouping_tags(group_by, s["tournament_tags"], s["ingame_tags"],
                                                 trust_half_overlay=trust_half_overlay)
        s["characters"]          = merge_set_characters(s["matches"], set_characters)

    return sets


def _sponsor_parts(sponsor: str | None) -> set[str]:
    """The case-folded component abbreviations of a sponsor string ("SP | TM" → {"sp", "tm"})."""
    return {p.strip().casefold() for p in sponsor.split("|")} if sponsor else set()


def refine_set_tags(sets: list[SetGroup],
                    finder: Callable[[list[TextBox]], list[TagEntry | None] | None],
                    min_games: int = 2) -> list[SetGroup]:
    """Recover a gamertag whose sponsor box out-lengths it in every game, so each game votes the
    sponsor as the tag and the set inherits it (reference VOD B: a set reads "ORG": the org tag
    : instead of the player "PlayerTwo"). A short tag like "PlayerTwo" is only read correctly on the
    rare frames where OCR merges the "ORG | PlayerTwo" plate into one box; per game that is too rare to
    win, but pooled across the set it is recoverable.

    For each set and tag position, the overlay reads are pooled across all the set's games (via
    `finder`, which runs find_tournament_tags with the VOD's resolution and overlay flags). A token
    seen as a sponsor in >= `min_games` games is a confirmed sponsor. Only a position whose set tag
    IS one of those confirmed sponsors is touched: it is re-voted from the pooled reads with the
    confirmed sponsors removed, and the winner must itself recur across >= `min_games` games so a
    one-off scenery read can't win. Every other set tag is left exactly as voted_tags produced it.
    Mutates and returns `sets`."""
    for s in sets:
        tags = s.get("tournament_tags")
        games = s["matches"]
        if not tags or len(games) < min_games:
            continue
        positions = range(len(tags))
        sponsors = list(s.get("tournament_sponsors") or [None] * len(tags))

        # Pool per position across the set's games: how many games each sponsor token / tag reading
        # appears in (game counts gate the >= min_games agreement), plus total frame counts and the
        # original-cased sponsor strings (to report the recovered tag's real sponsor).
        sponsor_games:   list[Counter[str]] = [Counter() for _ in positions]  # token -> games as sponsor
        tag_games:       list[Counter[str]] = [Counter() for _ in positions]  # tag (casefold) -> games
        tag_frames:      list[Counter[str]] = [Counter() for _ in positions]  # tag text -> total frames
        sponsor_strings: list[Counter[str]] = [Counter() for _ in positions]  # original sponsor string
        for game in games:
            game_sponsors: list[set[str]] = [set() for _ in positions]
            game_tags:     list[set[str]] = [set() for _ in positions]
            for frame in game["frames"]:
                for i, entry in enumerate(finder(frame.get("texts", [])) or []):
                    if i >= len(tags):
                        break
                    if entry is None:   # learned-row path: this side wasn't read in this frame
                        continue
                    sponsor = entry.get("sponsor")
                    if sponsor:
                        game_sponsors[i] |= _sponsor_parts(sponsor)
                        sponsor_strings[i][sponsor.strip()] += 1
                    text = (entry.get("text") or "").strip()
                    if text:
                        tag_frames[i][text] += 1
                        game_tags[i].add(text.casefold())
            for i in positions:
                for token in game_sponsors[i]:
                    sponsor_games[i][token] += 1
                for key in game_tags[i]:
                    tag_games[i][key] += 1

        for i in positions:
            confirmed = {tok for tok, n in sponsor_games[i].items() if n >= min_games}
            current = tags[i]
            if current is None or current.strip().casefold() not in confirmed:
                continue  # only override a set tag that is itself a confirmed cross-game sponsor
            candidates = {text: n for text, n in tag_frames[i].items()
                          if text.strip().casefold() not in confirmed
                          and tag_games[i][text.casefold()] >= min_games}
            if not candidates:
                continue
            tags[i] = max(candidates, key=lambda t: candidates[t])
            sponsors[i] = max((t for t in sponsor_strings[i] if _sponsor_parts(t) <= confirmed),
                              key=lambda t: sponsor_strings[i][t], default=sponsors[i])
        s["tournament_tags"]     = tags
        s["tournament_sponsors"] = sponsors
    return sets
