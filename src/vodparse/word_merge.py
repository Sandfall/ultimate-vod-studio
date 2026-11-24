"""Re-merge docTR's per-word overlay reads into phrase boxes at extraction time.

The docTR overlay engine (``vodparse.ocr_engine.doctr_result_to_readtext``) flattens
docTR's pages → blocks → lines → words tree and emits **one box per word**. The
downstream tag/filter logic (``vodparse.tags``, ``vodparse.text_utils``) was shaped for
EasyOCR's *phrase*-level boxes, so word-at-a-time reads defeat it:

  * ``"WINNERS ROUND 2"`` arriving as three word boxes slips past the round-keyword
    filter (which matches a single box's text);
  * ``"BEST OF 5"`` as ``"BEST"`` + ``"OF."`` + ``"5"`` slips past the best-of filter;
  * multi-word gamertags/sponsors get truncated because ``bbox.longest_text`` keeps only
    the single longest word.

The fix is to re-merge adjacent same-line docTR words back into phrase boxes. This
happens at **extraction time** (``extract_matches.load_pipeline``), not inside the
``doctr_result_to_readtext`` adapter, for two reasons: stored ``pipeline.json`` runs then
replay correctly without re-OCR (the source video lives on the user's computer and may
no longer be present when the JSON is re-read), and ``pipeline.json`` keeps the raw
word-level reads as provenance.

Merge threshold (measured on the only docTR reference VOD, 1920x1080):
word boxes *within* one rendered phrase touch or overlap: ``WINNERS`` x[892, 963] /
``ROUND.`` x[961, 1018] / ``2`` x[1016, 1031] all sit at y[99, 120] with inter-word gaps
<= 0 px. The visible gap between a sponsor plate and its gamertag measures ~21 px beside
a 22 px-tall tag (~0.95x its height). A threshold at 0.6x the taller neighbour's height
splits these cleanly: intra-phrase gaps (<= 0) merge, the plate gap (21 > 13.2) does not.

The merge deliberately does **not** blind-join whole docTR lines. The sponsor logic
downstream (``tags._split_sponsor_from_row``, ``bbox.longest_text``) depends on
``[sponsor] [tag]`` staying two boxes across that visible plate gap; ``text_utils.
split_tag_sponsor`` only re-splits a *single* box on an explicit "|" or a
whitespace-flanked bracket separator, so a wrongly-joined ``"and PLAYERONE"`` box would
never come back apart. Keeping the plate gap unmerged preserves that separation.

A pronoun pill drawn flush against the gamertag (a 2026 Rivals 2 stream: ``HE/HIM``
x[667, 733] 22 px tall touching ``PlayerOne`` x[733, 855] 34 px tall, gap 0) is merged into
the tag's phrase ("HE/HIM PlayerOne"), just as EasyOCR reads it. It comes off the phrase's
edge afterwards, the same way for every reader (``vodparse.name_filters``).

Qt-free; imports nothing heavy (the ``OCR_ENGINE_DOCTR`` gate constant lives in
``vodparse.ocr_engine``, which imports torch/easyocr/doctr lazily, so this module stays
cheap to import).
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from difflib import SequenceMatcher

from vodparse.bbox import bbox_center_y, bbox_height, bbox_max_x, bbox_min_x
from vodparse.ocr_engine import OCR_ENGINE_DOCTR
from vodparse.schema import Event, InferenceSettings, PipelineConfig, TextBox

# Merge two same-line neighbours when the gap between them is under this fraction of the
# taller neighbour's height. Measured on the docTR reference VOD (1920x1080): words within one
# rendered phrase touch or overlap (gaps <= 0 px), while a sponsor plate sits ~0.95x its
# gamertag's height away (~21 px beside a 22 px tag). 0.6x (13.2 px there) separates them.
MERGE_GAP_FRAC = 0.6

# Two boxes are on the same rendered line when their center-y delta is within this fraction
# of the SHORTER neighbour's height. The shorter-neighbour rule is what stops a tall box from
# swallowing a short one it merely brushes: the 81 px timer box (``5:00.31``) sits with its
# center ~20 px off a 22 px tag's center line, and 20 > 0.5 * 22 = 11 keeps them on separate
# lines even where their x-spans overlap.
#
# The test is made between the two boxes being glued, the run's rightmost word and the
# candidate beside it, never against a whole line (see merge_adjacent_words). Measured on
# a 2026 Rivals 2 stream (1920x1080): the 73 px clock (center-y 90.5) sits 29 px off the
# center of the 33 px game-count square beside it, against a 16.5 px limit, so it stays
# apart; measured against the 85 px event logo at the left edge, which an earlier
# line-clustering rule anchored on, it was 32 px off against a 36.5 px limit, and the clock
# then merged across both players ("HE/HIM PlayerOne 0 6:42, 1 PlayerTwo HE/HIM").
MERGE_SAME_LINE_FRAC = 0.5

# docTR sometimes reads part of a word a second time, as a box nested inside the full word's
# box: "IYNAR" x[1316, 1395] inside "SHINYNAR" x[1260, 1396] on a 2026 Arcadian overlay. The
# merge would glue it on as a word of its own ("SJSU SHINYNAR IYNAR"). A box with at least this
# fraction of its width inside a longer same-line word is such a re-read and is dropped (see
# drop_nested_rereads). Words of one phrase only graze each other: "ROUND." x[961, 1018] and
# "2" x[1016, 1031] share 2 of the 15 px. The same bar as tags._DUP_CONTAINED_FRAC, which
# drops whole duplicate phrase boxes later on.
NESTED_REREAD_FRAC = 0.65


def merge_adjacent_words(texts: list[TextBox]) -> list[TextBox]:
    """Merge adjacent same-line word boxes into phrase boxes (see the module docstring).

    One left-to-right walk. Boxes are visited sorted by min-x; each one joins the first
    open run whose rightmost box it belongs beside, which means the two boxes sit on the
    same rendered line (:func:`_same_line`) and the gap between them is under
    ``MERGE_GAP_FRAC`` of the taller of the two (:func:`_is_adjacent`; a negative or
    overlapping gap always qualifies). Otherwise it opens a run of its own. Each run then
    collapses to one box via :func:`_merge_run`.

    Both tests are made between the two boxes being glued, never against a line as a
    whole. Clustering boxes into lines first was tried three ways and each failed on a
    real overlay: anchoring a line on its first member let a tall event logo admit the
    clock, which then merged across both players; requiring every member fragmented a
    line whose rows are staggered, so "BEST" and "OF 3" never met; the nearest member
    chained stacked rows (a nameplate, the handle under it, the seed under that) into
    one. The pairwise rule is what the phrase itself looks like: a word beside a word.

    Output is ordered left-to-right by where each phrase starts (the order the runs were
    opened in). The finders work from the boxes' positions, and the one place a tie could
    have fallen back on list order (:func:`vodparse.bbox.best_mirrored_pair`) settles it
    by text and position instead, so this order is not a contract. The input boxes are
    never mutated; empty input yields an empty list.
    """
    runs: list[list[TextBox]] = []
    for box in sorted(texts, key=bbox_min_x):
        for run in runs:
            if _same_line(run[-1], box) and _is_adjacent(run[-1], box):
                run.append(box)
                break
        else:
            runs.append([box])
    return [_merge_run(run) for run in runs]


def _same_line(a: TextBox, b: TextBox) -> bool:
    """Whether `a` and `b` sit on the same rendered line: their center-y delta is within
    ``MERGE_SAME_LINE_FRAC`` of the shorter box's height (see that constant's comment)."""
    return abs(bbox_center_y(a) - bbox_center_y(b)) <= (
        MERGE_SAME_LINE_FRAC * min(bbox_height(a), bbox_height(b)))


def _is_adjacent(prev: TextBox, nxt: TextBox) -> bool:
    """Whether `nxt` is close enough to the left-neighbour `prev` to join its run: the
    horizontal gap between them is under ``MERGE_GAP_FRAC`` of the taller box's height. A
    negative gap (the boxes touch or overlap) is always under the threshold, so it merges."""
    gap = bbox_min_x(nxt) - bbox_max_x(prev)
    return gap < MERGE_GAP_FRAC * max(bbox_height(prev), bbox_height(nxt))


def _merge_run(run: list[TextBox]) -> TextBox:
    """Collapse a run of same-line word boxes (already in left-to-right order) into one box.

    A single-box run is returned unchanged (the original object). For a multi-box run:

      * ``text`` is the member texts joined by single spaces, left to right;
      * ``confidence`` is the ``min`` over members: a phrase is only as trustworthy as its
        weakest word, and taking the min (rather than mean/max) keeps ``tags.find_best_of``'s
        max-by-confidence pick and ``tags.find_round``'s confidence tie-break from being
        inflated by one strong word in an otherwise shaky phrase. Members are already rounded
        the way ``pipeline.json`` stores them, and a min of rounded values needs no re-rounding;
      * ``bbox`` is the axis-aligned union quad of all member corner points, kept in integer
        coordinates to match ``schema.QuadBBox`` (``list[list[int]]``).
    """
    if len(run) == 1:
        return run[0]

    text = " ".join(box["text"] for box in run)
    confidence = min(box["confidence"] for box in run)

    xs = [point[0] for box in run for point in box["bbox"]]
    ys = [point[1] for box in run for point in box["bbox"]]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

    return {"text": text, "confidence": confidence, "bbox": bbox}


def drop_nested_rereads(texts: list[TextBox]) -> list[TextBox]:
    """The word boxes minus those docTR read twice: a box with at least
    ``NESTED_REREAD_FRAC`` of its width inside a same-line box whose text is longer (see that
    constant's comment). The longer read always survives, so of two nested reads one is
    kept. A pipe or pipe look-alike ("I", "l", "-") is never dropped: a separator drawn flush
    against a sponsor overlaps it by most of its own narrow width ("I" x[1305, 1331] against
    "OUTI" x[1215, 1323]) and is :func:`fold_pipe_lookalikes`'s to judge. Order is kept and
    nothing is mutated."""
    return [box for box in texts
            if not any(other is not box and _is_nested_reread(box, other) for other in texts)]


def _is_nested_reread(box: TextBox, fuller: TextBox) -> bool:
    """Whether `box` is a second, partial read of the word `fuller` reads in full."""
    if box["text"].strip() in PIPE_LOOKALIKE_WORDS | {PIPE_WORD}:
        return False
    if len(fuller["text"].strip()) <= len(box["text"].strip()) or not _same_line(box, fuller):
        return False
    overlap = min(bbox_max_x(box), bbox_max_x(fuller)) - max(bbox_min_x(box), bbox_min_x(fuller))
    width = max(bbox_max_x(box) - bbox_min_x(box), 1)
    return overlap >= NESTED_REREAD_FRAC * width


# ── Pipe look-alikes ──────────────────────────────────────────────────────────────────
# docTR reads the thin "|" between a sponsor and its tag as the nearest letter or a dash,
# as a word box of its own ("Lov" "I" "Infamous" on one frame, "Lov" "-" "Infamous" on the
# next); on a 720p stream it never returns "|" at all. A lone "I", "l" or "-" is also a real
# word ("Who I Am", "Tea - Time"), and one frame cannot tell the two apart. The run can: a
# pipe wobbles between a stroke and a dash from frame to frame while a letter reads the
# same every time, so a look-alike folds to "|" only between a left and a right word that
# were also seen with the other kind of look-alike, or with a real "|", somewhere in the
# run. "I" and "l" are one kind (the same glyph in a sans-serif face, so a name's own "I"
# reads as either); "-" is the other. Keyed on both neighbours, so a common left word
# beside two different names ("THE I Guy", "THE - Rock") is no evidence for either; the
# right word matches loosely, because OCR misreads a long tag a letter or two differently
# from frame to frame ("BRARCC08", "BRARCE08", "BRARee08" are one tag).
PIPE_LOOKALIKE_WORDS = frozenset({"I", "l", "-"})
PIPE_WORD = "|"
# The reading each look-alike counts as: the two strokes are one.
PIPE_LOOKALIKE_KINDS = {"I": "stroke", "l": "stroke", "-": "dash", PIPE_WORD: PIPE_WORD}
# How many different kinds of reading between one pair of words make it a pipe.
PIPE_EVIDENCE_MIN_READINGS = 2
# How alike two right words must be (difflib ratio, 0..1) to count as one word: the OCR
# variants of a tag score well above this, two different names ("Guy", "Rock") far below.
PIPE_RIGHT_WORD_SIMILARITY = 0.6

PipeEvidence = Mapping[str, frozenset[str]]
"""For each lower-cased left word, the lower-cased right words it was read with a pipe
between (loosely matched at fold time, see ``_same_word``)."""
NO_PIPE_EVIDENCE: PipeEvidence = {}
"""What a single frame's read has: nothing folds."""


def _lookalike_triples(texts: list[TextBox]) -> list[tuple[TextBox, TextBox, TextBox]]:
    """Every (left, middle, right) where ``middle`` is a lone look-alike or "|" box with a
    same-line, adjacent word on each side (the merge's own tests, so exactly the boxes the
    merge would glue around it are its neighbours); the nearest such word on each side."""
    triples: list[tuple[TextBox, TextBox, TextBox]] = []
    for middle in texts:
        if middle["text"].strip() not in PIPE_LOOKALIKE_WORDS | {PIPE_WORD}:
            continue
        lefts = [box for box in texts
                 if box is not middle and bbox_min_x(box) < bbox_min_x(middle)
                 and _same_line(box, middle) and _is_adjacent(box, middle)]
        rights = [box for box in texts
                  if box is not middle and bbox_min_x(box) > bbox_min_x(middle)
                  and _same_line(middle, box) and _is_adjacent(middle, box)]
        if lefts and rights:
            triples.append((max(lefts, key=bbox_max_x), middle, min(rights, key=bbox_min_x)))
    return triples


def _word(box: TextBox) -> str:
    """A box's text as the evidence keys it: stripped and lower-cased."""
    return box["text"].strip().lower()


def _same_word(a: str, b: str) -> bool:
    """Whether two right words are one word read twice (PIPE_RIGHT_WORD_SIMILARITY)."""
    return a == b or SequenceMatcher(None, a, b).ratio() >= PIPE_RIGHT_WORD_SIMILARITY


def pipe_evidence_in(events: Iterable[Event]) -> PipeEvidence:
    """The word pairs a run read a pipe between (see the section comment): across every
    event's word boxes, each left word with the right words beside which it was read with
    a real "|", or with at least PIPE_EVIDENCE_MIN_READINGS different kinds of look-alike
    (PIPE_LOOKALIKE_KINDS) across readings whose right words are the same word."""
    readings: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)   # left -> (kind, right)
    for event in events:
        for left, middle, right in _lookalike_triples(event.get("texts") or []):
            readings[_word(left)].append((PIPE_LOOKALIKE_KINDS[middle["text"].strip()], _word(right)))
    evidence: dict[str, frozenset[str]] = {}
    for left_word, seen in readings.items():
        evidenced: set[str] = set()
        for _kind, right_word in seen:
            kinds_here = {other_kind for other_kind, other_right in seen
                          if _same_word(right_word, other_right)}
            if PIPE_WORD in kinds_here or len(kinds_here) >= PIPE_EVIDENCE_MIN_READINGS:
                evidenced.add(right_word)
        if evidenced:
            evidence[left_word] = frozenset(evidenced)
    return evidence


def fold_pipe_lookalikes(texts: list[TextBox], evidence: PipeEvidence) -> list[TextBox]:
    """``texts`` with each lone look-alike box that sits between a word pair in
    ``evidence`` rewritten as a "|" box, so the merged phrase splits on it like a pipe
    read whole (``text_utils.split_tag_sponsor``). Boxes are copied, never mutated; with
    no evidence (a single frame's read) nothing changes."""
    if not evidence:
        return texts
    pipes: set[int] = set()
    for left, middle, right in _lookalike_triples(texts):
        if middle["text"].strip() not in PIPE_LOOKALIKE_WORDS:
            continue
        if any(_same_word(_word(right), known) for known in evidence.get(_word(left), ())):
            pipes.add(id(middle))
    return [{**box, "text": PIPE_WORD} if id(box) in pipes else box for box in texts]


def merge_overlay_words(texts: list[TextBox],
                        pipe_evidence: PipeEvidence = NO_PIPE_EVIDENCE) -> list[TextBox]:
    """The band-path merge: :func:`drop_nested_rereads`, :func:`fold_pipe_lookalikes` (with
    the ``pipe_evidence`` the caller has: a run's, or none for a single frame), then
    :func:`merge_adjacent_words`. Every reader of a whole docTR band goes through this one
    entry point (:func:`merge_doctr_reads`, :func:`merge_band_words`, the layout preview's
    Auto read in ``cli.overlay_read``, the spotcheck tool). ``merge_adjacent_words`` itself
    stays a pure merge for the drawn-box reads (``layout_reads.read_region``)."""
    words = drop_nested_rereads(texts)
    return merge_adjacent_words(fold_pipe_lookalikes(words, pipe_evidence))


# ── Run-level gate ────────────────────────────────────────────────────────────────────

def merge_doctr_reads(events: list[Event], config: PipelineConfig | None) -> list[Event]:
    """Re-merge docTR word boxes into phrase boxes across every OCR'd event in a run.

    Gated on the recorded OCR engine: only docTR runs emit word-level boxes, so an EasyOCR
    (or engine-unknown) run is returned unchanged: the same list object. The engine is read
    defensively because ``config`` may be None and ``inference_settings`` may be absent in
    ``pipeline.json`` files that predate it (mirroring ``extract_matches.resolve_sample_interval``).

    A run whose overlay engine is EasyOCR but whose bottom nameplate band was read with docTR
    (``inference_settings.ingame_band_ocr_engine``, a game's own choice: see
    ``vodparse.games.IngameHud.ocr_engine``) merges only the boxes in that band, i.e. those
    whose center sits at or below ``ingame_tag_band_frac`` of the recorded frame height; the
    top band's EasyOCR phrases are left exactly as read.

    For a docTR run, a **new** list is returned. Each event carrying a non-empty ``texts`` list
    gets a shallow-copied dict with ``texts`` replaced by its merged phrases; events without
    texts, with empty texts, or that are ``ocr_gated`` markers (their texts are ``[]``) pass
    through untouched. Inputs are never mutated.
    """
    # Single-arg .get (no {} default) so mypy doesn't check an empty dict against the total
    # InferenceSettings TypedDict; `or {}` keeps the defensiveness for a config predating it.
    inference: InferenceSettings | dict[str, str] = (
        config.get("inference_settings") or {}) if config else {}
    whole_run = inference.get("ocr_engine") == OCR_ENGINE_DOCTR
    band_only = (not whole_run
                 and inference.get("ingame_band_ocr_engine") == OCR_ENGINE_DOCTR)
    if not whole_run and not band_only:
        return events
    band_top_px: float | None = None
    if band_only:
        band_frac = inference.get("ingame_tag_band_frac")
        frame_height = config.get("frame_height") if config else None
        if not isinstance(band_frac, (int, float)) or not isinstance(frame_height, int):
            # A record without the band geometry cannot say which boxes are docTR's: leave
            # the run as read rather than merge EasyOCR phrases.
            return events
        band_top_px = float(band_frac) * frame_height

    pipe_evidence = pipe_evidence_in(events)
    merged_events: list[Event] = []
    for event in events:
        texts = event.get("texts")
        if texts:
            merged_texts = (merge_overlay_words(texts, pipe_evidence) if band_top_px is None
                            else merge_band_words(texts, band_top_px, pipe_evidence))
            merged_event: Event = {**event, "texts": merged_texts}
            merged_events.append(merged_event)
        else:
            merged_events.append(event)
    return merged_events


def merge_band_words(texts: list[TextBox], band_top_px: float,
                     pipe_evidence: PipeEvidence = NO_PIPE_EVIDENCE) -> list[TextBox]:
    """Merge only the boxes whose center sits at or below `band_top_px` (the bottom band a
    docTR reader produced word boxes for); the rest keep their order and shape. Shared
    with the layout preview's Auto read (``cli.overlay_read.read_auto``), which merges a
    single frame's reads under the same gate this module applies to a stored run (and
    with no ``pipe_evidence``, which only a run has)."""
    above = [text for text in texts if bbox_center_y(text) < band_top_px]
    band = [text for text in texts if bbox_center_y(text) >= band_top_px]
    return above + merge_overlay_words(band, pipe_evidence)
