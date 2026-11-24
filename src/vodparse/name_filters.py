"""The one place a reader's output loses the words that are never part of a name.

A pronoun pill, an @handle or a [W]/[L] grand-finals marker drawn beside a gamertag is read
into the tag's phrase ("HE/HIM PlayerOne", "PlayerTwo [L]"). Every reader ends up there:
EasyOCR returns the phrase so, and docTR's per-word boxes are merged into it first
(``vodparse.word_merge``). So the words come off the phrase's edges here, after the merge,
the same way for every reader, at each of the three places phrases form:

  * a stored or fresh run, as it is loaded (``extract_matches.load_pipeline``);
  * the Analyze page preview's Auto read of one frame (``cli.overlay_read``);
  * a drawn layout box, as it is read (``layout_reads.read_region``: the run's reads and
    the layout editor's).

What a whole phrase is (a pill or handle on its own, a seed label, the stage header) is
decided where the phrases are used, by the finders (``text_utils.is_ignored_tag_text`` in
``vodparse.tags``, ``vodparse.side_overlay``, ``vodparse.grouping``): a dropped phrase still
tells some of them where the tag row is, so it can't be removed here.

Qt-free and light: imports only the schema types and text_utils.
"""

from vodparse.schema import Event, TextBox
from vodparse.text_utils import is_round_text, strip_never_name_edge_words


def strip_never_name_edges(texts: list[TextBox], *, player_row: bool = False) -> list[TextBox]:
    """Each phrase box with the never-name words at its start or end stripped from its text
    (``text_utils.strip_never_name_edge_words``; with `player_row`, a box read inside one
    player's overlay row, a misread pill too). The bbox is left as read, pill included: the
    text no longer says where the pill's pixels ended, and the edges the downstream geometry
    compares (``bbox.best_mirrored_pair`` measures each side's outer edge) stay as read.
    Order is kept and nothing is mutated; a box with nothing to strip is passed through as
    the same object.

    A round label is passed through whole: it is never a tag (the tag finders skip round
    text), and a grand-finals marker at its end is part of it ("TOP 16 QUALIFIER (W)" names
    the bracket side), where on a tag it is noise."""
    stripped: list[TextBox] = []
    for box in texts:
        if is_round_text(box):
            stripped.append(box)
            continue
        text = strip_never_name_edge_words(box["text"], player_row=player_row)
        stripped.append(box if text == box["text"] else {**box, "text": text})
    return stripped


def strip_never_name_edges_in_events(events: list[Event]) -> list[Event]:
    """:func:`strip_never_name_edges` over every OCR'd event of a run, whatever its engine
    (a docTR run's words are merged into phrases first: ``word_merge.merge_doctr_reads``).
    Returns a new list; an event whose texts changed is a shallow copy, the rest pass
    through. Inputs are never mutated."""
    stripped_events: list[Event] = []
    for event in events:
        texts = event.get("texts")
        if texts:
            stripped_event: Event = {**event, "texts": strip_never_name_edges(texts)}
            stripped_events.append(stripped_event)
        else:
            stripped_events.append(event)
    return stripped_events
