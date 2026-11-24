"""Text-content classifiers: numeric, round labels, best-of, and overlay detection."""

import re
import unicodedata
from difflib import SequenceMatcher
from string import punctuation

from vodparse.schema import TextBox


def format_timestamp(seconds: float) -> str:
    """Seconds → H:MM:SS (or M:SS when under an hour). Used for CSV timestamps, chapter
    markers, and debug printouts."""
    seconds = int(round(seconds))
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _is_ascii_digits(text: str) -> bool:
    """True for a non-empty run of the ASCII digits 0-9 only. ``str.isdigit`` alone also
    accepts superscripts and circled digits ("²", "①"), which ``int()`` rejects, and
    ``str.isdecimal`` accepts other scripts' digits ("٣"); format_timestamp writes
    neither."""
    return text.isascii() and text.isdecimal()


def parse_timestamp(text: str) -> float | None:
    """The inverse of format_timestamp: "H:MM:SS" or "M:SS" → seconds. None for anything
    else, including an empty CSV cell and a field holding any digit but ASCII 0-9."""
    parts = text.strip().split(":")
    if len(parts) not in (2, 3) or not all(_is_ascii_digits(part) for part in parts):
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return float(seconds)


_FEATURING_RE = re.compile(r"\s+(?:featuring|feat|ft)\.?\b", re.IGNORECASE)


def strip_featuring_suffix(name: str) -> str:
    """Drop a trailing "featuring ..." / "ft ..." / "feat ..." player list (and everything after
    it) from a VOD-derived tournament name. The marker must be a whole word, so "Drafting" and
    "Leftover" are left intact. Returns the name unchanged when no marker is present."""
    return _FEATURING_RE.split(name, maxsplit=1)[0].strip()


def is_numeric_text(text: TextBox) -> bool:
    """Purely numeric content: set counts, the match timer (6:59.76), damage
    percentages (126.5%), etc. Never a gamertag or name. Allows digits, decimal
    points, colons, commas, percent signs and spaces: two numbers the word merge
    glued side by side ("2 1.6%", a stock count beside a damage read) are still only
    numbers. Stray punctuation at either end is ignored, so OCR debris like
    "5:00.31-" or "(2)" still reads as numeric."""
    s = text["text"].strip().strip(punctuation)
    return bool(s) and any(c.isdigit() for c in s) and all(c.isdigit() or c in ".:,% " for c in s)


def is_punctuation_only(text: TextBox) -> bool:
    """No letter or digit at all: OCR debris such as "-", ":" or "- -" (docTR reads a sliver of
    stage art or a plate's edge this way). Never a gamertag or a sponsor. Left unfiltered, two
    "-" slivers clipped at the frame's top edge become the topmost row on their side and are
    read as that side's tag in place of the real plate below them."""
    return not any(c.isalnum() for c in text["text"])


# Ordered by priority: when several upper-third texts each match a round keyword, the one whose
# keyword appears earliest here wins (see find_round). Lower index = higher priority. The
# bracket-phase labels (qualifier / pool / "TOP N") sit just before "bracket": they rank below
# every real round, so a true round always wins when both are on screen, but above "bracket", so
# they survive the _vote_round drop and can stand in as the match phase when no real round shows.
# "top" only counts inside a "TOP <n>" pattern (see _TOP_PHASE_RE / _round_tokens), never as a
# bare word, so a gamertag merely containing "top" is not treated as a round.
ROUND_KEYWORDS = (
    "losers", "winners", "round", "grand", "semis", "quarterfinals", "semifinals",
    "grands", "quarter", "quarters", "finals", "wsf", "wqf", "lsf", "lqs", "wqs",
    "lqf", "final", "wr", "lf", "wf", "gf", "lr",
    "qualifier", "qualifiers", "pool", "pools", "top",   # bracket phases: below rounds, above bracket
    "bracket",
)
_ROUND_PRIORITY = {kw: i for i, kw in enumerate(ROUND_KEYWORDS)}

# A "Top N" bracket phase ("TOP 8", "TOP 16", "TOP 24", "TOP24"). Matched as a pattern so the
# "top" keyword fires only here: not on any text that merely contains the bare word "top".
_TOP_PHASE_RE = re.compile(r"\btop\s*\d+\b", re.IGNORECASE)

def alnum_fold(text: str) -> str:
    """The text reduced to case-folded alphanumerics: a spacing/punctuation/case-blind key for
    deciding whether two OCR reads come from the same rendered string ("MR ." and "MR." both
    reduce to "mr"; "ree64" is a substring of "PlayerThree64"'s fold).

    For ASCII input the output is bit-identical to the old regex fold (re.sub(r"[^a-z0-9]", "",
    text.lower())): a hard invariant, covered by a test. Going through NFKD + casefold + isalnum
    instead of the ASCII-only regex keeps non-Latin scripts (kana, kanji) alive: they used to fold
    to "" and lose their identity in voting/dedup, but now survive as themselves. NFKD also folds
    full-width Latin ("ＴＥＡＭ" → "team") and half-width katakana onto their standard forms so
    stylized OCR reads collapse together.

    Combining marks are dropped by the isalnum() filter: NFKD decomposes an accented or dakuten
    character into a base plus combining mark(s), and only the base is alphanumeric. This is
    intentional: it reproduces the old fold's accent-blindness for Latin ("Pokémon" folds like
    "Pokemon") and, for Japanese, merges dakuten OCR misreads ("ガ" and "カ" fold the same). Fine
    for a voting/dedup key, which is never shown to the user."""
    folded = unicodedata.normalize("NFKD", text).casefold()
    return "".join(c for c in folded if c.isalnum())


_OCR_DIGIT_FIXES = str.maketrans("IiZzOo", "112200")
# Short all-caps round tokens that are real words, not misread acronyms: exempt from the
# digit-fix so the phase labels "TOP"/"POOL" don't get corrupted into "T0P"/"P00L".
_NORMALIZE_KEEP = frozenset({"TOP", "POOL"})


def _is_phase_label(tok: str) -> bool:
    """True for a bracket-phase word/label the digit-fix must leave alone: the bare phase words
    (TOP, POOL) and a glued "TOP<n>" label. Some OCR engines read "TOP 8" as a single "TOP8"
    box, which is short, all-caps and absent from _NORMALIZE_KEEP; without this guard the fix
    would rewrite its 'O' to '0' ("TOP8" → "T0P8"), destroying the round keyword (see
    _round_tokens / _TOP_PHASE_RE). Compared on the token stripped of surrounding punctuation,
    so a parenthesized graphic read as "(TOP 8)" is protected the same way: mirroring the
    digit-fix itself, whose isupper() test also ignores punctuation ("WRI)" → "WR1)")."""
    core = tok.strip(punctuation)
    return core in _NORMALIZE_KEEP or bool(_TOP_PHASE_RE.fullmatch(core))


def normalize_ocr(text: str) -> str:
    """Fix digit/letter swaps in stylized fonts (WRI→WR1, WRZ→WR2). Applied only to
    short all-caps abbreviations so word-based rounds (Winners, Losers) keep their
    letters intact, with a carve-out for real phase words (see _is_phase_label)."""
    return " ".join(
        tok.translate(_OCR_DIGIT_FIXES)
        if len(tok) <= 4 and tok.isupper() and not _is_phase_label(tok) else tok
        for tok in text.split()
    )


def _round_tokens(text: str) -> set[str]:
    """The ROUND_KEYWORDS present in `text`. "top" is special: it counts only when it heads a
    "TOP <n>" phase label (TOP 8 / 16 / 24), never as a bare word, so a gamertag that merely
    contains "top" is not swept up as a round. Surrounding punctuation on a token is ignored, so
    an OCR read like "ROUND." still counts as the keyword (word-level OCR engines attach
    punctuation to tokens)."""
    tokens = {tok.strip(punctuation) for tok in text.lower().split()} & _ROUND_PRIORITY.keys()
    tokens.discard("top")
    if _TOP_PHASE_RE.search(text):
        tokens.add("top")
    return tokens


def is_round_text(text: TextBox) -> bool:
    """True when the box contains any round keyword (see _round_tokens)."""
    return bool(_round_tokens(text["text"]))


# The rank round_priority assigns to text containing no round keyword at all: one past the
# weakest real keyword ("bracket"), so such text loses every round comparison but never crashes
# the ranking. Reachable when a transform between is_round_text and ranking (normalize_ocr,
# a future rewrite) strips the only keyword from a matched read.
_NO_ROUND_PRIORITY = len(ROUND_KEYWORDS)


def round_priority(text: str) -> int:
    """The keyword priority of a round label: the lowest ROUND_KEYWORDS index among the
    keywords it contains, or _NO_ROUND_PRIORITY when it contains none. Lower wins when
    picking which label is the round (see find_round and _vote_round)."""
    return min((_ROUND_PRIORITY[w] for w in _round_tokens(text)), default=_NO_ROUND_PRIORITY)


def title_round(label: str) -> str:
    """Normal title capitalization for a detected round label heading into review
    ("WINNERS FINALS" → "Winners Finals"), while leaving short all-caps acronyms
    (WSF, WR1, GF) uppercase the way normalize_ocr emits them."""
    return " ".join(
        tok if len(tok) <= 4 and tok.isupper() else tok.title()
        for tok in label.split()
    )


# "best of 3", "best of 5", "bo3", "Bo5", etc.
_BEST_OF_RE = re.compile(r"\bb(?:est)?\s*o\s*f?\s*([0-9])", re.IGNORECASE)
# The bare "best of" label, even without a trailing number and with OCR's "OF"→"0F" slip
# ("BEST 0F"). Requires the full word "best" so real tags ("BofTag") aren't swept up.
_BEST_OF_LABEL_RE = re.compile(r"\bbest\s*[o0]\s*f\b", re.IGNORECASE)
# "first to 2", "FIRST TO 3": the same label as "best of 2n-1" (first to 2 wins is a Bo3,
# first to 3 a Bo5), which some overlays print instead. Requires the digit, so a tag that
# merely starts with "First to" is left alone.
_FIRST_TO_RE = re.compile(r"\bfirst\s*to\s*([0-9])", re.IGNORECASE)


def parse_best_of(text: str) -> str | None:
    """The best-of count as a "Bo<n>" label ("best of 3"/"bo3" → "Bo3", "first to 3" → "Bo5"),
    or None when absent."""
    m = _BEST_OF_RE.search(text)
    if m:
        return f"Bo{m.group(1)}"
    first_to = _FIRST_TO_RE.search(text)
    if first_to:
        wins_needed = int(first_to.group(1))
        return f"Bo{2 * wins_needed - 1}"
    return None


def is_best_of_label(text: str) -> bool:
    """True for a 'best of' overlay label even when it lacks the trailing number or OCR mangled
    it to 'BEST 0F', and for its 'first to n' form. Broader than parse_best_of (which needs a
    digit after 'best of') so the label can be kept out of the gamertag candidate pool."""
    return _BEST_OF_LABEL_RE.search(text) is not None or _FIRST_TO_RE.search(text) is not None


# Sub-labels that ride beside a gamertag but are never the tag: a pronoun set (he/him, she/her,
# she/her/hers, they/them/theirs; OCR mangles the "/" into a space, punctuation, or a slash
# look-alike letter, so the separator rule below is broader than a literal "/"), a Twitter handle
# (@handle), or a seed label (Seed 2). Filtered by content, complementing the positional
# drop_sublabels pass.
#
# Includes the possessive/reflexive third form for the sets a stream overlay prints as three
# words (she/her/hers, they/them/theirs, he/him/his, and the neopronoun equivalents xe/xem/xyrs,
# ze/zir/zirs, ze/hir/hirs); "it/its" is left as a pair only ("itself" is rare enough not to chase,
# per the plan). "any/all" is the pill for a player who takes any pronouns; it is printed in
# the same place and read the same way ("ANY/ALL", "ANYJALL", "ANYIALL" on a 2026 Rivals 2
# stream). The sets are kept in the order a pill prints them: a pair is a pronoun set only
# when it is two consecutive words of one set ("he/him", "him/his", "any/all") or two
# nominative forms ("she/they", "he/they"), never two pronoun words in any order: "Them All",
# "It All" and "ALL | Her" are names or parts of names ("Catch Them All"), not pills.
_PRONOUN_SETS: tuple[tuple[str, ...], ...] = (
    ("he", "him", "his"), ("she", "her", "hers"), ("they", "them", "theirs"), ("it", "its"),
    ("xe", "xem", "xyr", "xyrs"), ("ze", "zir", "zirs"), ("ze", "hir", "hirs"), ("fae", "faer"),
    ("ey", "em", "eir"), ("per", "pers"), ("ve", "ver", "vis"), ("any", "all"),
)
_PRONOUNS = frozenset(word for pronoun_set in _PRONOUN_SETS for word in pronoun_set)
# The nominative form of each set (its first word): any two of these make a pill ("she/they").
# "any" is left out: it is a pill only beside "all".
_NOMINATIVE_PRONOUNS = frozenset(pronoun_set[0] for pronoun_set in _PRONOUN_SETS) - {"any"}
# Every (left, right) pair of consecutive words within one set.
_PRONOUN_PAIRS = frozenset(
    (pronoun_set[index], pronoun_set[index + 1])
    for pronoun_set in _PRONOUN_SETS for index in range(len(pronoun_set) - 1))
# The common pronouns (he/him, she/her, they/them) and the any/all pill: the subset of _PRONOUNS
# whose pairs accept the slightly wider curated separator-letter set (_COMMON_PRONOUN_SEP_LETTERS,
# vs the tighter _PRONOUN_SEP_LETTERS for neopronoun/"it" pairs). See the separator-letters
# comment below for which letters qualify and why the two sets differ.
_COMMON_PRONOUNS = frozenset({"he", "him", "she", "her", "they", "them", "any", "all"})
# Two or three pronoun words joined by a real "/" or "|" (a stream overlay's three-part sets,
# "she/her/hers", need a real separator between every word; the mangled-separator fallback below
# only ever reassembles a two-part pair). The third slot is optional so the same pattern matches
# both shapes.
# What OCR may read a pronoun pill's rounded end as, at the start or end of the set.
_PILL_EDGE_BRACKETS = "([{)]}"
_PRONOUN_SET_RE = re.compile(
    r"^([a-z]+)\s*[/|]\s*([a-z]+)(?:\s*[/|]\s*([a-z]+))?$", re.IGNORECASE
)
_TWITTER_HANDLE_RE = re.compile(r"^@\w+$")
# A seed label: "seed" and what follows it (judged by _is_seed_label).
_SEED_RE           = re.compile(r"^seeds?(?P<rest>.*)$", re.IGNORECASE)
# Letters OCR reads a seed number's digits as, counted on a 2026 Melee stream's caption
# (printed in capitals with no space, "SEED5", under the name inside the locked box):
# "SEEDB" for 8 (19 reads), "SEEDI" for 1 (14), "SEEDII" for 11 (10), "SEED I" (4).
_SEED_DIGIT_LOOKALIKES = frozenset("IlOoB")
# A seed number has at most this many digits ("SEED20"), so at most this many look-alikes
# glued to "seed" are taken for one.
_SEED_NUMBER_MAX_CHARS = 2
# OCR reads the "/" in a pronoun pair as a slash look-alike, so "he/him" arrives as "helhim" or
# "HEZHIM". A separator qualifies if it is any non-letter (a real "/", "|", ".", "-", space, or a
# digit) or one of a curated set of slash-look-alike *letters*: straight/diagonal strokes such as
# "l"/"i" (vertical) and "z"/"j" (diagonal). The letter set is curated rather than "any letter" so a
# real word that happens to be two pronouns joined by one letter isn't eaten. Two sets, because the
# safe letters differ by pair:
#   * neopronoun / "it" pairs stay tight ("liz"): "m"/"s"/"r" would fold "Hermit" (her+m+it),
#     "Itsit" (it+s+it) or "Verver" (ve+r+ver) into a bogus "pronoun pair";
#   * common pairs (he/him/she/her/they/them) also allow "j", but curvy letters remain excluded so a
#     real gamertag like "Hesher" (he+s+her) survives: "s" is not a stroke OCR mistakes a "/" for.
_PRONOUN_SEP_LETTERS        = frozenset("liz")
_COMMON_PRONOUN_SEP_LETTERS = frozenset("lizj")


def _is_pronoun_separator(char: str, sep_letters: frozenset[str]) -> bool:
    """Whether a single character may join a mis-OCR'd pronoun pair whose "/" OCR mangled: any
    non-letter, or one of the curated slash-look-alike letters in `sep_letters`."""
    return not char.isalpha() or char in sep_letters


def _is_known_pronoun_pair(left: str, right: str) -> bool:
    """Whether two lower-cased words are a pronoun pair as a pill prints one: consecutive
    words of one set ("he"/"him", "any"/"all") or two nominative forms ("she"/"they"). Two
    pronoun words in any other order ("them"/"all", "it"/"all") are not."""
    if (left, right) in _PRONOUN_PAIRS:
        return True
    return left in _NOMINATIVE_PRONOUNS and right in _NOMINATIVE_PRONOUNS and left != right


def _is_known_pronoun_triple(words: list[str]) -> bool:
    """Whether three lower-cased words are a pronoun set as a pill prints one: the first
    three words of one set in order ("she"/"her"/"hers"); a known pair followed by a
    nominative form or by the next word of the pair's set ("she"/"her"/"they",
    "she"/"they"/"them"); or a nominative form followed by a known pair
    ("they"/"she"/"her")."""
    first, second, third = words
    for pronoun_set in _PRONOUN_SETS:
        if tuple(words) == pronoun_set[:3]:
            return True
    if _is_known_pronoun_pair(first, second):
        return (third in _NOMINATIVE_PRONOUNS or (second, third) in _PRONOUN_PAIRS
                or (first, third) in _PRONOUN_PAIRS)
    return first in _NOMINATIVE_PRONOUNS and _is_known_pronoun_pair(second, third)


def _is_pronoun_set(text: str) -> bool:
    """True for a pronoun set: a two-part pair like 'he/him' or 'she/they' (including when OCR
    mangled the '/'), or a three-part set like 'she/her/hers' or 'they/them/theirs'. The words
    must make a set the way a pill prints one (_is_known_pronoun_pair, _is_known_pronoun_triple),
    so a real tag with a slash ('Red/Blue'), a name ('Helmet') or a phrase of pronoun words in
    another order ('Them All', 'It All') is kept.

    The three-part form requires a real '/' or '|' between every word (see _PRONOUN_SET_RE): the
    mangled-separator fallback below only ever reassembles a two-part pair, and a third slot
    widens the mangled-separator search space enough to start risking real tags, so it isn't
    extended there. Two-part pairs go through the mangled separator too, via a curated
    slash-look-alike separator (_is_pronoun_separator); the two pair kinds differ only in which
    look-alike letters are allowed:

      * two COMMON pronouns (he/him/she/her/they/them): the slightly wider _COMMON_PRONOUN_SEP_LETTERS
        - OCR renders the '/' as a stroke-like letter ("he/him" -> "HEZHIM"), but curvy letters stay
        excluded so a real gamertag like "Hesher" (he+s+her) isn't eaten;
      * a pair involving a neopronoun or "it"/"its": the tighter _PRONOUN_SEP_LETTERS, so ordinary
        words like "Hermit" (her+m+it) aren't mistaken for a pronoun pair.

    A bracket OCR read at the pill's edge ("(He/Him", a 2026 stacked-panel Ultimate stream) is
    ignored: the pill's rounded end, not part of any set.
    """
    s = text.strip().strip(_PILL_EDGE_BRACKETS).strip()
    m = _PRONOUN_SET_RE.match(s)
    if m is not None:
        words = [word.lower() for word in m.groups() if word is not None]
        if len(words) == 2:
            return _is_known_pronoun_pair(words[0], words[1])
        return _is_known_pronoun_triple(words)
    # No '/' or '|' for the regex to latch onto: the separator is a space, punctuation, or a letter
    # OCR mistook the slash for. Two-part only (see the docstring). Split on each single char with
    # a known pronoun pair on both sides.
    low = s.lower()
    for i in range(1, len(low) - 1):
        left, right = low[:i], low[i + 1:]
        if not _is_known_pronoun_pair(left, right):
            continue
        both_common = left in _COMMON_PRONOUNS and right in _COMMON_PRONOUNS
        sep_letters = _COMMON_PRONOUN_SEP_LETTERS if both_common else _PRONOUN_SEP_LETTERS
        if _is_pronoun_separator(low[i], sep_letters):
            return True
    return False



# ── Misread pronoun pills (player rows only) ────────────────────────────────────────────
# A scoreboard panel prints the pronoun set in a small pill beside the tag, and at that size
# OCR rarely returns it letter-perfect: "SHE/HER/HERS" comes back as "She/HeR/HEAS",
# "SRE{HER(HEAS" or "ShE MERMHEAS". None of those is made of known pronoun words, so
# _is_pronoun_set keeps them, and as the last box on the row they win the gamertag pick.
# The three-part sets a pill prints, as letters only (separators are what OCR mangles most):
_PRONOUN_PILL_SETS = ("sheherhers", "hehimhis", "theythemtheirs")
# How alike a read's letters must be to one of those sets (difflib ratio, 0..1). Set from
# the pill reads recorded on a stacked-panel Melee capture:
# all but one of its distinct misreads score 0.60 or more, and the nearest real text on
# that stream or a per-side one scores 0.43.
_MISREAD_PRONOUN_MIN_SIMILARITY = 0.6
# ...and how far its letter count may be from that set's. A misread swaps letters or adds
# one for a mangled "/" (the recorded reads are 0 to 2 letters longer); a short text that
# merely sits inside a set ("Her Hero" inside she/her/hers) is not one.
_MISREAD_PRONOUN_MAX_LENGTH_DIFF = 2
# A pill cut off by the edge of the box it is read in ("SHE/HER/H") has too few letters for
# the comparison above. It still counts once this many "/"-separated words are present: two
# whole pronouns and the start of a third.
_CUT_OFF_PRONOUN_MIN_WORDS = 3
# The two-part pairs a pill prints, and their own, stricter bar. A pair has half the letters
# of a three-part set, so the looser bar above would reach real tags ("He-Man" scores 0.60
# against he/him, "The Hero" 0.77 against she/her). 0.8 allows one wrong letter in a pair:
# "SHF/HER", recorded on a per-side Melee capture, scores 0.83.
_PRONOUN_PILL_PAIRS = ("sheher", "hehim", "theythem")
_MISREAD_PRONOUN_PAIR_MIN_SIMILARITY = 0.8
_MISREAD_PRONOUN_PAIR_MAX_LENGTH_DIFF = 1


def _letters_only(text: str) -> str:
    """`text` lower-cased with everything that is not a letter removed."""
    return "".join(char for char in text.lower() if char.isalpha())


def pronoun_pill_similarity(text: str) -> float:
    """How alike `text`'s letters are to the nearest of _PRONOUN_PILL_SETS (difflib ratio,
    0..1; 1.0 = the same letters in the same order). Everything that is not a letter is
    ignored, so "SHE/HER/HERS" and "She Her{HERS" both score 1.0."""
    letters = _letters_only(text)
    return max(SequenceMatcher(None, letters, pill_set).ratio() for pill_set in _PRONOUN_PILL_SETS)


def _is_cut_off_pronoun_set(text: str) -> bool:
    """True for a pronoun set whose last word was cut off: "SHE/HER/H", what a pill reads
    as when the edge of the box being read runs through it. At least two whole pronoun
    words joined by a real "/" or "|", then the beginning of another pronoun."""
    words = [word.strip().lower() for word in re.split(r"[/|]", text)]
    if len(words) < _CUT_OFF_PRONOUN_MIN_WORDS:
        return False
    whole_words, fragment = words[:-1], words[-1]
    return (all(word in _PRONOUNS for word in whole_words)
            and fragment != ""
            and any(pronoun.startswith(fragment) for pronoun in _PRONOUNS))


def _letters_match_a_pill(letters: str, pills: tuple[str, ...], min_similarity: float,
                          max_length_diff: int) -> bool:
    """Whether `letters` is within `max_length_diff` letters of one of `pills` in count
    and at least `min_similarity` alike (difflib ratio)."""
    for pill in pills:
        if abs(len(letters) - len(pill)) > max_length_diff:
            continue
        if SequenceMatcher(None, letters, pill).ratio() >= min_similarity:
            return True
    return False


def is_misread_pronoun_set(text: str) -> bool:
    """True for a pronoun pill that OCR did not return whole. Two shapes:

      * misread ("She/HeR/HEAS" for "SHE/HER/HERS", "SHF/HER" for "SHE/HER"): the text
        carries at least one non-letter (the "/" or whatever OCR turned it into, a space
        included), and its letters alone are close in both content and count
        to one of _PRONOUN_PILL_SETS (_MISREAD_PRONOUN_MIN_SIMILARITY,
        _MISREAD_PRONOUN_MAX_LENGTH_DIFF) or of the shorter _PRONOUN_PILL_PAIRS
        (_MISREAD_PRONOUN_PAIR_MIN_SIMILARITY, _MISREAD_PRONOUN_PAIR_MAX_LENGTH_DIFF);
      * cut off ("SHE/HER/H"): see _is_cut_off_pronoun_set.

    Looser than _is_pronoun_set on purpose, so it is NOT part of is_ignored_tag_text: it
    is for the words inside one player's overlay row (a drawn player box, a locked
    side-bar row), where a pronoun pill is expected beside the tag. The band readers,
    which see every text on the overlay, keep the exact test. The non-letter requirement
    keeps a one-word gamertag out of reach whatever its letters are."""
    s = text.strip()
    if all(char.isalpha() for char in s):
        return False
    if _is_cut_off_pronoun_set(s):
        return True
    letters = _letters_only(s)
    return (_letters_match_a_pill(letters, _PRONOUN_PILL_SETS, _MISREAD_PRONOUN_MIN_SIMILARITY,
                                  _MISREAD_PRONOUN_MAX_LENGTH_DIFF)
            or _letters_match_a_pill(letters, _PRONOUN_PILL_PAIRS,
                                     _MISREAD_PRONOUN_PAIR_MIN_SIMILARITY,
                                     _MISREAD_PRONOUN_PAIR_MAX_LENGTH_DIFF))


# TOs mark the grand-finals bracket sides on the overlay ("PlayerTwo [L]", "[W] PlayerOne",
# "PlayerOne (L)": both bracket styles are in use), and after a bracket reset some mark the
# player from losers "(W/L)" ("(WL)" on a 2026 Rivals 2 stream). The marker isn't part of
# the gamertag and the side is apparent from the VOD, so a tag loses it wherever it rides
# along (split_tag_sponsor). {} covers OCR-confused brackets; only W, L or a W/L pair
# qualifies, so a bracketed clan tag ("[PlayerThirteen]") is untouched. OCR also damages
# the round-bracket form, as counted over every stored run (2026-10-25): the "(" read as a
# "C" ("CL)", "CW)") or the ")" lost ("(W"). Those count only as a word of their own, so
# "(Lucky)" keeps its "(L". Left alone: a bare "L" after a name (names end that way too)
# and a bare "CW" (a sponsor could). A round label keeps its marker ("TOP 16 QUALIFIER
# (W)" names the bracket side), which is why the marker is not a never-name word.
_WL_MARKER_INNER = r"(?:[WL]|W\s*/?\s*L|L\s*/?\s*W)"
_WL_MARKER_RE = re.compile(
    rf"[\[{{(]\s*{_WL_MARKER_INNER}\s*[\]}})]"                    # a whole marker, any bracket
    r"|(?<!\S)(?:[\[{(][WL]|C[WL][\]})])(?!\S)",                # OCR-damaged, as a word of its own
    re.IGNORECASE)


def is_grand_finals_marker(word: str) -> bool:
    """True for a single word that is a [W]/[L] grand-finals marker, in any bracket ("[L]",
    "(W)", "(W/L)") or OCR-damaged ("CL)", "(W"): see _WL_MARKER_RE."""
    return _WL_MARKER_RE.fullmatch(word.strip()) is not None


def _strip_wl_markers(text: str) -> str:
    """Remove [W]/[L] (or (W)/(L)) grand-finals markers, collapsing the whitespace they
    leave behind."""
    s = _WL_MARKER_RE.sub(" ", text)
    return re.sub(r"\s+", " ", s).strip()


# The stage-select screen header ("Current Stage", sometimes "Current Stage _" with a trailing
# glyph). It rides in the overlay band during stage select but never names a player.
_STAGE_LABEL_RE = re.compile(r"\bcurrent\s*stage\b", re.IGNORECASE)


def is_ignored_tag_text(text: str) -> bool:
    """True for text that should never be treated as a gamertag: pronoun sets (two- or three-part,
    e.g. "he/him" or "she/her/hers"), Twitter handles (@handle), seed labels (Seed 2), standalone
    [W]/[L] grand-finals markers (which would otherwise be read as a sponsor box beside the tag),
    and the stage-select header ("Current Stage")."""
    s = text.strip()
    return (_is_pronoun_set(s)
            or _TWITTER_HANDLE_RE.match(s) is not None
            or _is_seed_label(s)
            or _WL_MARKER_RE.fullmatch(s) is not None
            or _STAGE_LABEL_RE.search(s) is not None)


def _is_seed_label(text: str) -> bool:
    """True for a seed label: "seed" and an optional number, glued or apart and through any
    punctuation ("Seed 2", "Seed(4", "SEED", "seed %"), since overlays print it with or
    without the number and OCR mangles the spacing. A number OCR misread as look-alike
    letters counts too when it is set apart from "seed" ("SEED I"), holds a real digit
    ("SEEDI 2"), or is a glued one or two in an all-capitals label, as captions are printed
    ("SEEDI", "SEEDB"). So a name that merely starts with "seed" and look-alikes is kept
    ("Seedio", "Seedo"), and any other letter after "seed" is a real tag ("SeedTag",
    "Seed X9")."""
    match = _SEED_RE.match(text)
    if match is None:
        return False
    rest = match["rest"]
    number = re.sub(r"[\W_]", "", rest)   # the number's characters, punctuation and spaces out
    if number == "" or number.isdigit():
        return True
    if any(not char.isdigit() and char not in _SEED_DIGIT_LOOKALIKES for char in number):
        return False
    set_apart = not rest[0].isalnum()
    has_digit = any(char.isdigit() for char in number)
    glued_caption_number = text.isupper() and len(number) <= _SEED_NUMBER_MAX_CHARS
    return set_apart or has_digit or glued_caption_number


def is_never_name_word(word: str) -> bool:
    """True for a single word that is never part of a gamertag or sponsor, nor of any other
    overlay label, whatever stands next to it: a pronoun set ("he/him", "she/her/hers") or a
    Twitter handle (@handle). The subset of is_ignored_tag_text that holds for one word on
    its own: a seed label ("Seed 2") or the stage header ("Current Stage") is only
    recognisable as a phrase, and a bare "Seed" or "Current" can be part of a name ("Seeds
    of Chaos"). What strip_never_name_edge_words tests a phrase's edge words with.

    A [W]/[L] grand-finals marker is not one: it is part of a round label ("TOP 16 QUALIFIER
    (W)"). A tag loses it in split_tag_sponsor, and at a phrase's edge in
    strip_never_name_edge_words, whose callers keep round labels away."""
    s = word.strip()
    return _is_pronoun_set(s) or _TWITTER_HANDLE_RE.match(s) is not None


# A pronoun pill drawn flush against a gamertag is read as part of the tag's phrase ("HE/HIM
# PlayerOne", "PlayerTwo HEJHIM" on a 2026 Rivals 2 stream): EasyOCR returns it so, and
# docTR's words are merged into the same phrase. The pill can also arrive split by a stray
# OCR space ("HE /HIM", "SHE / HER"), so up to this many words at a phrase's edge are tried
# together as one never-name word: three covers "SHE / HER".
_NEVER_NAME_EDGE_MAX_WORDS = 3


def strip_never_name_edge_words(text: str, *, player_row: bool = False) -> str:
    """`text` without the never-name words (is_never_name_word: a pronoun set, an @handle)
    or [W]/[L] grand-finals markers at its start or end, where a pill drawn beside a gamertag
    sits once the two are read as one phrase: "HE/HIM PlayerOne" -> "PlayerOne",
    "PlayerTwo SHE/HER [L]" -> "PlayerTwo". Only the edges are tested, with the exact test,
    so a pronoun word inside a name ("Them All", "His Majesty") is never touched.

    With `player_row` (the text was read inside one player's overlay row, where a pill is
    expected beside the tag) a single edge word that is a misread pill goes too
    (is_misread_pronoun_set: "PlayerOne She/HeR/HEAS" -> "PlayerOne"). Single words only: the
    fuzzy test over a run of words could take a short real word along ("Jo SHF/HER").

    A text that is nothing but such words is returned unchanged: the whole-box filters
    (is_ignored_tag_text) drop it. So is a text with nothing to strip, whitespace and all.
    Callers keep round labels away from it: a marker is part of one
    (name_filters.strip_never_name_edges)."""
    words = text.split()
    kept = list(words)
    while (count := _leading_never_name_words(kept, player_row)) and count < len(kept):
        kept = kept[count:]
    while (count := _trailing_never_name_words(kept, player_row)) and count < len(kept):
        kept = kept[:-count]
    if len(kept) == len(words):
        return text
    return " ".join(kept)


def _leading_never_name_words(words: list[str], player_row: bool) -> int:
    """How many of `words`' first words form one never-name word (the longest such run, up
    to _NEVER_NAME_EDGE_MAX_WORDS), or 0."""
    for count in range(min(_NEVER_NAME_EDGE_MAX_WORDS, len(words)), 0, -1):
        if _is_edge_strippable(" ".join(words[:count]), fuzzy=player_row and count == 1):
            return count
    return 0


def _trailing_never_name_words(words: list[str], player_row: bool) -> int:
    """How many of `words`' last words form one never-name word (the longest such run, up
    to _NEVER_NAME_EDGE_MAX_WORDS), or 0."""
    for count in range(min(_NEVER_NAME_EDGE_MAX_WORDS, len(words)), 0, -1):
        if _is_edge_strippable(" ".join(words[-count:]), fuzzy=player_row and count == 1):
            return count
    return 0


def _is_edge_strippable(word: str, *, fuzzy: bool) -> bool:
    """What strip_never_name_edge_words takes off a phrase's edge: a never-name word or a
    [W]/[L] grand-finals marker, and with `fuzzy` a misread pronoun pill."""
    return (is_never_name_word(word) or is_grand_finals_marker(word)
            or (fuzzy and is_misread_pronoun_set(word)))


# A bracket-progress HUD counter ("GAME 3", "POOL 4"). Leading non-word chars and a missing
# space cover OCR variants ("(GAME 4", "GAME4"). Only digits/spaces may trail the keyword, so
# real text ("gameplay") is safe.
_PROGRESS_LABEL_RE = re.compile(r"^[^\w]*(game|pool)\s*\d*$", re.IGNORECASE)


def is_progress_label(text: str) -> bool:
    """True for a bracket-progress counter like 'GAME 3'/'POOL 4' (and OCR variants '(GAME 4',
    'GAME4'). These never name a player. Filtered only when reading a stacked overlay, where the
    gamertag sits below the metadata line and these lower HUD counters would otherwise become
    reachable as tag candidates."""
    return _PROGRESS_LABEL_RE.match(text.strip()) is not None


# A bracket/brace flanked by whitespace is a mis-OCR'd "|" sponsor separator ("FAM [ PlayerEleven").
# A lone "I", "l" or "-" between two words is NOT folded here: docTR does read the thin pipe
# that way ("LOV I PlayerOne", "LOV - PlayerOne"), but so does any engine read a name that
# holds one ("Who I Am", "Tea - Time"), and a single box cannot tell the two apart. That fold
# lives in word_merge.fold_pipe_lookalikes, on the run's own evidence.
_STRIP_SEP_RE      = re.compile(r"\s+[\[\]{}]\s+")
# A score sits at the very start ("0]PlayerFour") or end ("PlayerNine[2", "PlayerEight 2") of the box:
# one or two digits with a bracket or space between them and the name. The boundary is
# required: digits glued to the letters are the name's own ("PlayerOne08", "PlayerTwo64").
_STRIP_LEAD_SCORE  = re.compile(r"^[\s\[\]{}]*\d{1,2}[\s\[\]{}]+")
_STRIP_TRAIL_SCORE = re.compile(r"[\s\[\]{}]+\d{1,2}[\s\[\]{}]*$")
# Leftover bracket/brace/pipe junk hugging either edge once scores are gone.
_STRIP_EDGE_JUNK   = re.compile(r"^[\s\[\]{}|]+|[\s\[\]{}|]+$")


def _normalize_sponsor_seps(text: str) -> str:
    """Fold a whitespace-flanked bracket/brace OCR produces for a "|" sponsor separator back to
    a literal "|" ("FAM [ PlayerEleven" → "FAM|PlayerEleven"), so both split paths see one form."""
    return _STRIP_SEP_RE.sub("|", text)


def _clean_tag_text(text: str) -> str:
    """Strip the sponsor prefix and score digits a one-sided overlay packs into a gamertag box,
    so equivalent readings collapse to one vote ("FAM | PlayerEleven", "FAM [ PlayerEleven",
    "PlayerEleven [0" → "PlayerEleven"). The gamertag is the last "|"-separated segment (sponsors come
    first: "SP | TM | PlayerTwelve" → "PlayerTwelve"). Only applied to strip-layout tags, never the
    symmetric path, so real bracketed clan tags ("[PlayerThirteen]") are left untouched.
    Returns the cleaned text, or the original stripped if cleaning would empty it."""
    s = _normalize_sponsor_seps(text)
    s = s.rsplit("|", 1)[-1]
    s = _STRIP_LEAD_SCORE.sub("", s)
    s = _STRIP_TRAIL_SCORE.sub("", s)
    s = _STRIP_EDGE_JUNK.sub("", s)
    return s or text.strip()


# The separator/bracket characters OCR uses as: or mistakes for: a sponsor delimiter or set symbol.
SEPARATOR_CHARS = "|/\\[]{}"
# A separator right after an emoticon eye (optionally through a nose) is a mouth: ":]", ":/", ":|",
# ";-]": not a delimiter, so it is kept.
_EMOTICON_EYES = frozenset(":;=")
_EMOTICON_NOSE = "-^"


def _edge_is_emoticon(inward: str) -> bool:
    """`inward` is a tag read outward from the char next to an edge separator. True when it begins
    with an emoticon eye (after an optional nose), marking that separator as an emoticon mouth
    (":]", ";-]", ":/", ":|") to keep rather than trim as a delimiter."""
    return inward.lstrip(_EMOTICON_NOSE)[:1] in _EMOTICON_EYES


def _strip_singular_edge_sep(value: str) -> str:
    """Strip a lone separator (| / \\ [ ] { }) sitting at the start or end of a tag/sponsor
    ("ORG |" → "ORG", "[LTag" → "LTag"). Only fires when the value has exactly one separator
    character, so a matched pair: a clan tag "[PlayerOne]", a "[W]"/"[L]" grand-finals marker: and an
    internal delimiter ("SP | TM") are left untouched, as is a separator that is really an emoticon
    mouth ("sometag :]"). Returns the value trimmed of whitespace."""
    s = value.strip()
    if sum(c in SEPARATOR_CHARS for c in s) != 1:
        return s
    if s[-1] in SEPARATOR_CHARS and not _edge_is_emoticon(s[:-1][::-1]):
        return s[:-1].strip() or s
    if s[0] in SEPARATOR_CHARS and not _edge_is_emoticon(s[1:]):
        return s[1:].strip() or s
    return s


def clean_sponsor(sponsor: str | None) -> str | None:
    """Drop OCR score/separator noise from a sponsor. Each "|"-separated part has a lone edge
    separator trimmed (see _strip_singular_edge_sep), then is dropped when it is empty, a single
    character, or has no letter at all: a lone or slash-joined game count the OCR grabbed off the
    set symbol ("0 |", "2/3") or a stray single letter. Surviving parts are rejoined ("ORG |" →
    "ORG", "0 | SP" → "SP", "SP | TM" → "SP | TM"), or None when nothing real is left."""
    if not sponsor:
        return None
    parts = []
    for raw in sponsor.split("|"):
        part = _strip_singular_edge_sep(raw)
        if len(part) <= 1 or not any(c.isalpha() for c in part):
            continue
        parts.append(part)
    return " | ".join(parts) or None


def split_tag_sponsor(text: str) -> tuple[str, str | None]:
    """Separate a sponsor-prefixed gamertag box into (gamertag, sponsor). Splitting happens
    only on an explicit sponsor separator ("|", or the whitespace-flanked bracket OCR produces
    for one), so a lone tag keeps its trailing digits and clan brackets ("Player12" →
    ("Player12", None), "[PlayerThirteen]" → ("[PlayerThirteen]", None)). With a separator the gamertag is the
    last segment (score-cleaned) and everything before it is the sponsor ("FAM | PlayerEleven" →
    ("PlayerEleven", "FAM"), "SP | TM | PlayerTwelve" → ("PlayerTwelve", "SP | TM")). [W]/[L] grand-finals
    markers are stripped first, wherever they sit ("PlayerTwo [L]", "[W] ORG | PlayerThree")."""
    text = _strip_wl_markers(text) or text
    s = _normalize_sponsor_seps(text)
    if "|" not in s:
        return (_strip_singular_edge_sep(text), None)
    parts = [p.strip() for p in s.split("|") if p.strip()]
    if len(parts) < 2:
        # A separator with nothing on one side ("ORG |"): the tag box carried a dangling sponsor
        # pipe. Strip it so the reading matches the clean one on other frames.
        return (_strip_singular_edge_sep(text), None)
    tag = _clean_tag_text(parts[-1])
    sponsor = clean_sponsor(" | ".join(parts[:-1]))
    return (_strip_singular_edge_sep(tag) or text.strip(), sponsor)


def is_overlay_text(t: TextBox) -> bool:
    """True for texts that legitimately repeat across matches (round label, best-of, and
    numeric HUD like scores/percent/timer): exempt from banner stripping."""
    return is_numeric_text(t) or is_round_text(t) or parse_best_of(t["text"]) is not None
