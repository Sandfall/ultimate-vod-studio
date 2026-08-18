"""Text-content classifiers: numeric, round labels, best-of, and overlay detection."""

import re
import unicodedata
from string import punctuation

from vodparse.schema import TextBox


def format_timestamp(seconds: float) -> str:
    """Seconds → H:MM:SS (or M:SS when under an hour). Used for CSV timestamps, chapter
    markers, and debug printouts."""
    seconds = int(round(seconds))
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


_FEATURING_RE = re.compile(r"\s+(?:featuring|feat|ft)\.?\b", re.IGNORECASE)


def strip_featuring_suffix(name: str) -> str:
    """Drop a trailing "featuring ..." / "ft ..." / "feat ..." player list (and everything after
    it) from a VOD-derived tournament name. The marker must be a whole word, so "Drafting" and
    "Leftover" are left intact. Returns the name unchanged when no marker is present."""
    return _FEATURING_RE.split(name, maxsplit=1)[0].strip()


def is_numeric_text(text: TextBox) -> bool:
    """Purely numeric content: set counts, the match timer (6:59.76), damage
    percentages (126.5%), etc. Never a gamertag or name. Allows digits, decimal
    points, colons, commas and percent signs. Stray punctuation at either end is
    ignored, so OCR debris like "5:00.31-" or "(2)" still reads as numeric."""
    s = text["text"].strip().strip(punctuation)
    return bool(s) and any(c.isdigit() for c in s) and all(c.isdigit() or c in ".:,%" for c in s)


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


def parse_best_of(text: str) -> str | None:
    """The best-of count as a "Bo<n>" label ("best of 3"/"bo3" → "Bo3"), or None when absent."""
    m = _BEST_OF_RE.search(text)
    return f"Bo{m.group(1)}" if m else None


def is_best_of_label(text: str) -> bool:
    """True for a 'best of' overlay label even when it lacks the trailing number or OCR mangled
    it to 'BEST 0F'. Broader than parse_best_of (which needs a digit) so the label can be kept
    out of the gamertag candidate pool."""
    return _BEST_OF_LABEL_RE.search(text) is not None


# Sub-labels that ride beside a gamertag but are never the tag: a pronoun pair (he/him,
# she/they; OCR mangles the "/" into a space, punctuation, or a slash look-alike letter, so the
# separator rule below is broader than a literal "/"), a Twitter handle (@handle), or a seed
# label (Seed 2). Filtered by content, complementing the positional drop_sublabels pass.
_PRONOUNS = frozenset({
    "he", "him", "she", "her", "they", "them", "it", "its",
    "xe", "xem", "xyr", "ze", "zir", "hir", "fae", "faer",
    "ey", "em", "eir", "per", "pers", "ve", "ver", "vis",
})
# The common pronouns (he/him, she/her, they/them): the subset of _PRONOUNS whose pairs accept
# the slightly wider curated separator-letter set (_COMMON_PRONOUN_SEP_LETTERS, vs the tighter
# _PRONOUN_SEP_LETTERS for neopronoun/"it" pairs). See the separator-letters comment below for
# which letters qualify and why the two sets differ.
_COMMON_PRONOUNS = frozenset({"he", "him", "she", "her", "they", "them"})
_PRONOUN_PAIR_RE   = re.compile(r"^([a-z]+)\s*[/|]\s*([a-z]+)$", re.IGNORECASE)
_TWITTER_HANDLE_RE = re.compile(r"^@\w+$")
# The number is optional and may be glued through punctuation ("Seed 2", "Seed(4", bare "SEED",
# "seed %"): overlays render the label with or without the number, and OCR mangles the spacing.
# Word characters after "seed" still exempt real tags ("SeedTag", "Seed X9").
_SEED_RE           = re.compile(r"^seeds?[\W_]*\d*[\W_]*$", re.IGNORECASE)
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


def _is_pronoun_pair(text: str) -> bool:
    """True for a pronoun pair like 'he/him' or 'she/they', including when OCR mangled the '/'. Both
    sides must be known pronoun words, so a real tag with a slash ('Red/Blue') or a name ('Helmet')
    is kept. Both pairs use a curated slash-look-alike separator (_is_pronoun_separator); the two
    differ only in which look-alike letters are allowed:

      * two COMMON pronouns (he/him/she/her/they/them): the slightly wider _COMMON_PRONOUN_SEP_LETTERS
        - OCR renders the '/' as a stroke-like letter ("he/him" -> "HEZHIM"), but curvy letters stay
        excluded so a real gamertag like "Hesher" (he+s+her) isn't eaten;
      * a pair involving a neopronoun or "it"/"its": the tighter _PRONOUN_SEP_LETTERS, so ordinary
        words like "Hermit" (her+m+it) aren't mistaken for a pronoun pair.
    """
    s = text.strip()
    m = _PRONOUN_PAIR_RE.match(s)
    if m is not None:
        return m.group(1).lower() in _PRONOUNS and m.group(2).lower() in _PRONOUNS
    # No '/' or '|' for the regex to latch onto: the separator is a space, punctuation, or a letter
    # OCR mistook the slash for. Split on each single char with a known pronoun on both sides.
    low = s.lower()
    for i in range(1, len(low) - 1):
        left, right = low[:i], low[i + 1:]
        if left not in _PRONOUNS or right not in _PRONOUNS:
            continue
        both_common = left in _COMMON_PRONOUNS and right in _COMMON_PRONOUNS
        sep_letters = _COMMON_PRONOUN_SEP_LETTERS if both_common else _PRONOUN_SEP_LETTERS
        if _is_pronoun_separator(low[i], sep_letters):
            return True
    return False


# TOs mark the grand-finals bracket sides on the overlay ("PlayerTwo [L]", "[W] PlayerOne"). The
# marker isn't part of the gamertag and the side is apparent from the VOD, so it's dropped
# wherever it rides along. {} covers OCR-confused brackets; only a lone W/L qualifies, so
# bracketed clan tags ("[PlayerThirteen]", "[WL]") are untouched.
_WL_MARKER_RE = re.compile(r"[\[{]\s*[WL]\s*[\]}]", re.IGNORECASE)


def _strip_wl_markers(text: str) -> str:
    """Remove [W]/[L] grand-finals markers, collapsing the whitespace they leave behind."""
    s = _WL_MARKER_RE.sub(" ", text)
    return re.sub(r"\s+", " ", s).strip()


# The stage-select screen header ("Current Stage", sometimes "Current Stage _" with a trailing
# glyph). It rides in the overlay band during stage select but never names a player.
_STAGE_LABEL_RE = re.compile(r"\bcurrent\s*stage\b", re.IGNORECASE)


def is_ignored_tag_text(text: str) -> bool:
    """True for text that should never be treated as a gamertag: pronoun pairs, Twitter handles
    (@handle), seed labels (Seed 2), standalone [W]/[L] grand-finals markers (which would
    otherwise be read as a sponsor box beside the tag), and the stage-select header ("Current
    Stage")."""
    s = text.strip()
    return (_is_pronoun_pair(s)
            or _TWITTER_HANDLE_RE.match(s) is not None
            or _SEED_RE.match(s) is not None
            or _WL_MARKER_RE.fullmatch(s) is not None
            or _STAGE_LABEL_RE.search(s) is not None)


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
_STRIP_SEP_RE      = re.compile(r"\s+[\[\]{}]\s+")
# A score sits at the very start ("0]PlayerFour") or end ("PlayerNine[2", "PlayerEight 2") of the box:
# optional brackets/spaces around one or two digits.
_STRIP_LEAD_SCORE  = re.compile(r"^[\s\[\]{}]*\d{1,2}[\s\[\]{}]*")
_STRIP_TRAIL_SCORE = re.compile(r"[\s\[\]{}]*\d{1,2}[\s\[\]{}]*$")
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
