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

Qt-free; imports nothing heavy (the ``OCR_ENGINE_DOCTR`` gate constant lives in
``vodparse.ocr_engine``, which imports torch/easyocr/doctr lazily, so this module stays
cheap to import).
"""

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
MERGE_SAME_LINE_FRAC = 0.5


def merge_adjacent_words(texts: list[TextBox]) -> list[TextBox]:
    """Merge adjacent same-line word boxes into phrase boxes (see the module docstring).

    Two-pass, deliberately straightforward:

    1. **Group into lines.** Iterate boxes sorted top-to-bottom by center-y; a box joins the
       current line when it is on the same rendered line as that line's *first* member
       (:func:`_same_line`), otherwise it opens a new line. This greedy clustering against the
       first member (rather than a running average) is enough for overlay text, where every box
       in a phrase is rendered at one baseline.
    2. **Merge runs within a line.** Sort each line left-to-right by min-x and walk it, extending
       the current run while the next box's gap to the run's last box is under
       ``MERGE_GAP_FRAC`` of the taller of the two (negative/overlapping gaps always merge);
       otherwise the run closes and a new one opens. Each run collapses to one box via
       :func:`_merge_run`.

    Output is ordered line-by-line, left-to-right. The input boxes are never mutated; empty
    input yields an empty list.
    """
    if not texts:
        return []

    lines: list[list[TextBox]] = []
    for box in sorted(texts, key=bbox_center_y):
        if lines and _same_line(lines[-1][0], box):
            lines[-1].append(box)
        else:
            lines.append([box])

    merged: list[TextBox] = []
    for line in lines:
        run: list[TextBox] = []
        for box in sorted(line, key=bbox_min_x):
            if run and _is_adjacent(run[-1], box):
                run.append(box)
            else:
                if run:
                    merged.append(_merge_run(run))
                run = [box]
        if run:
            merged.append(_merge_run(run))
    return merged


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


def merge_doctr_reads(events: list[Event], config: PipelineConfig | None) -> list[Event]:
    """Re-merge docTR word boxes into phrase boxes across every OCR'd event in a run.

    Gated on the recorded OCR engine: only docTR runs emit word-level boxes, so an EasyOCR
    (or engine-unknown) run is returned unchanged: the same list object. The engine is read
    defensively because ``config`` may be None and ``inference_settings`` may be absent in
    ``pipeline.json`` files that predate it (mirroring ``extract_matches.resolve_sample_interval``).

    For a docTR run, a **new** list is returned. Each event carrying a non-empty ``texts`` list
    gets a shallow-copied dict with ``texts`` replaced by its merged phrases; events without
    texts, with empty texts, or that are ``ocr_gated`` markers (their texts are ``[]``) pass
    through untouched. Inputs are never mutated.
    """
    # Single-arg .get (no {} default) so mypy doesn't check an empty dict against the total
    # InferenceSettings TypedDict; `or {}` keeps the defensiveness for a config predating it.
    inference: InferenceSettings | dict[str, str] = (
        config.get("inference_settings") or {}) if config else {}
    if inference.get("ocr_engine") != OCR_ENGINE_DOCTR:
        return events

    merged_events: list[Event] = []
    for event in events:
        texts = event.get("texts")
        if texts:
            merged_event: Event = {**event, "texts": merge_adjacent_words(texts)}
            merged_events.append(merged_event)
        else:
            merged_events.append(event)
    return merged_events
