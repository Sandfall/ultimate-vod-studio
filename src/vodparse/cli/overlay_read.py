"""OCR a captured frame the way analysis will: the editor's test-read backend and the
Analyze page's layout preview backend.

The overlay-layout editor (``vodparse.gui.overlay_editor``) and the layout preview
(``vodparse.gui.layout_preview``) shell out to this stage, through the shared client
``vodparse.gui.overlay_read_client``, so the GUI process never imports torch — running
inference on an in-process Qt thread is exactly what the GUI's process-based
architecture exists to avoid (it fatally crashed the interpreter on macOS: torch + PyQt
threads + Python 3.13). Like ``vodparse.cli.doctr_support``, the heavy imports happen
only in here.

A **long-lived line protocol**, so the multi-second torch import + OCR model load is
paid once per editor session rather than once per click. One JSON object per stdin
line::

    {"token": 3, "engine": "easyocr", "lang": "en", "frame_path": "/abs/frame.png",
     "regions": {"player1": [x1, y1, x2, y2], ...}}

``regions`` maps region keys (vodparse.overlay_layout.REGION_KEYS) to full-frame pixel
boxes. An optional ``ingame_engine`` names the engine the game pins for its in-game
nameplate band (``games.ingame_band_engine_for``, e.g. "doctr" for Rivals 2; absent or
null when the game pins none or the band is not read)::

    {"token": 5, "engine": "easyocr", "lang": "en", "frame_path": "/abs/frame.png",
     "regions": {"ingame_p1": [x1, y1, x2, y2], ...}, "ingame_engine": "doctr"}

When it is set, differs from ``engine`` and can be built in this install, the in-game
boxes (``overlay_layout.INGAME_REGION_KEYS``) are read with it, in English, as a run reads
them; a missing docTR falls back to ``engine``, as the run does. The request names the
engine, never the game: the game would also bring its stricter clock rules to the timer
box, which a drawn box is deliberately read without (see the ``timer`` note below).

A request carrying ``auto`` instead of ``regions`` reads the frame the way a run
with no custom overlay layout does (the built-in "Auto" reading)::

    {"token": 4, "engine": "easyocr", "lang": "en", "frame_path": "/abs/frame.png",
     "auto": {"game": "ultimate", "ingame_tag_frac": null, "clock_bbox": [x1, y1, x2, y2]}}

That runs the pipeline's band OCR (``cli.pipeline.detect_text`` over the bands
``overlay_bands.band_ocr_plan`` reads for the game), re-merges docTR's per-word boxes
into phrases the way a run does at extraction time (``word_merge``: the finders were
shaped for phrase boxes, and "Winners Round 2" as three words reads as just "Winners"),
then its single-frame tag finders (``tags.find_tournament_tags`` / ``find_player_names``
/ ``find_round`` / ``find_best_of``), plus the clock at ``clock_bbox`` (the preview's
calibrated box) when one is given. Its readings are keyed by the same region keys, and each one that was
found carries ``bbox``, where on the frame it was read; a region the run reads but found
nothing for is present with empty text, so the preview can say so.

A game whose Auto reading locks side bars first (Melee) reads its player names from rows
only a search over many frames can find (``vodparse.side_overlay``). The request then
carries those rows as ``"locked"`` inside ``auto``, once the preview's own search
(``cli.side_overlay_locate``) has found them::

    "auto": {"game": "melee", ..., "locked": {"player1": [x1, y1, x2, y2], ...}}

Each locked player row is read as a run reads it (like a drawn player box, on the drawn-box
engine) and answers for that player in place of the top band's tag finder; a locked sponsor
row (``overlay_regions.REGION_SPONSOR1``/``2``) supplies the sponsor when the player row
split none off itself. Without ``locked`` such a game is read by its top band alone.

When the video already has an analysis, ``auto`` may also name its ``pipeline.json`` as
``"pipe_evidence_run"``. A run folds a pipe docTR read as "I" or "-" between a sponsor and its
tag back into "|" on evidence gathered across all its frames (``word_merge.fold_pipe_lookalikes``);
one frame has none, so the preview borrows the stored run's, and its readout then splits the
sponsor off as the analysis did. Without it nothing folds, as before::

    "auto": {"game": "ultimate", ..., "pipe_evidence_run": "json_output/<stem>/pipeline.json"}

A ``regions`` request may also carry ``background``: the same shape as ``auto`` plus
``keys``, the regions to answer from the bands and the clock box. A run under a custom
layout keeps reading, with the built-in readers, every field the layout draws no box
for (``preview_regions.preview_background_keys`` is the rule), and the preview shows
those alongside the drawn boxes::

    {"token": 5, ..., "regions": {"player1": [...], "player2": [...]},
     "background": {"game": "ultimate", "ingame_tag_frac": null, "ingame_band": "tags",
                    "clock_bbox": [...],
                    "keys": ["round", "best_of", "timer", "ingame_p1", "ingame_p2"]}}

Only the bands those keys need are OCR'd (the top band for the tags, round and best-of;
the bottom band for the in-game names), only those finders run, and the clock only when
``timer`` is among the keys. ``ingame_band`` is what the layout reads the nameplate band
as (``overlay_layout.ingame_band_mode_for``), so a layout that reads a band the game's
own Auto reading does not (Other with "tags") gets its in-game names answered; absent,
the game's default applies. The drawn boxes and the background never share a key.

Each request gets exactly one response line on stdout, marked with
``RESULT_LINE_PREFIX`` so it can't be confused with model-loading chatter (the
``export.SAVED_LINE_PREFIX`` seam-sharing idiom)::

    OVERLAY_READ_RESULT {"token": 3, "engine": "easyocr", "readings": {"player1": {"text": "...", "confidence": 0.93}, ...}}
    OVERLAY_READ_RESULT {"token": 3, "error": "..."}

``engine`` names the engine the frame was read with: always the one the request asked
for. Nothing is ever read with another engine in its place. The helper never downloads
docTR's text models itself (only the Analyze page's own offer may, when a run starts), so
a request naming docTR before they are downloaded (or with python-doctr not installed) is
answered with an ``error`` saying why (``ocr_engine.engine_unusable_reason``). A game whose
in-game names are read with docTR whatever the Font style (Rivals 2) gets them left unread
in that case, and the response carries a ``note`` saying so (``INGAME_NAMES_UNREAD_NOTE``).

Engines are cached per (engine, lang), errors are answered per-request (the process
keeps serving), and EOF on stdin exits cleanly. Try it by hand from a checkout::

    echo '{"token": 1, "engine": "easyocr", "lang": "en", "frame_path": "frame.png",
           "regions": {"player1": [100, 900, 400, 970]}}' \\
        | PYTHONPATH=src python -m vodparse.cli.overlay_read

The ``timer`` region is special-cased: it goes through the pipeline's own clock path
(``timer.read_timer_at_bbox`` on a dedicated always-English EasyOCR reader) instead of
a generic text read, so its result says whether a *clock parses* there — the question
that decides whether clock-based splitting will work on that video.

Module-level imports stay light (json/sys/argparse + the Qt-free overlay_layout): the
GUI imports this module for the protocol constants, and must never pay for torch by
doing so.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# The per-box crop + read primitives live in vodparse.layout_reads, shared with the
# analysis pipeline so a test read here previews exactly what analysis will read.
from vodparse.layout_reads import pad_box, read_region
from vodparse.ocr_engine import FONT_STYLE_STYLIZED, OCR_CANVAS_DEFAULT
from vodparse.overlay_layout import (INGAME_REGION_KEYS, REGION_BEST_OF, REGION_INGAME_P1,
                                     REGION_INGAME_P2, REGION_PLAYER1, REGION_PLAYER2,
                                     REGION_ROUND, REGION_TIMER)
from vodparse.paths import debug_dir
from vodparse.schema import TagEntry, TextBox

# Debug aid for tuning a timer box: every test read of the timer region writes the exact
# padded crop the clock reader receives (timer_crop.png), each preprocessing variant
# it OCRs (crop_<variant>.jpg, see ocr_timer.preprocess_variants) and a per-variant
# selection trace (timer.READ_TRACE_NAME: what the recognizer returned for each variant
# and which one won) into this directory, overwriting on each read, so what the
# recognizer actually saw can be opened and inspected. Set to None (or comment the
# assignment out) to turn the dump off.
TIMER_READ_DEBUG_DIR: Path | None = debug_dir("_overlay_editor")

# The stdout marker for protocol responses; everything else on stdout is model-loading
# chatter the GUI ignores. Shared with vodparse.gui.overlay_editor so the seam can't drift.
RESULT_LINE_PREFIX = "OVERLAY_READ_RESULT "

# What a response says when the game reads its in-game names with an engine that can't
# read here (Rivals 2's docTR, without its models): those names are left unread rather
# than read with the request's engine in its place (see _pin_unusable_reason).
# An f-string for the style's name; "{reason}" stays a placeholder for str.format.
INGAME_NAMES_UNREAD_NOTE = (f"Ingame names not read: this game needs the {FONT_STYLE_STYLIZED} "
                            "text model to read them. {reason}")

# The engine-cache key for the timer's dedicated reader (see _timer_reader). Lives in
# the same dict as the overlay engines; the fake "engine" name can never collide with a
# real (engine, lang) pair because "timer" is not an engine id.
_TIMER_READER_KEY = ("timer", "en")

# The engine-cache key prefix for a band engine (the overlay engine as the pipeline's
# band pass uses it, see _band_engine), kept apart from the drawn-box engine cached
# under the bare (engine, lang) pair (see _box_engine): under docTR the two differ,
# though the box engine is derived from the band engine and shares its models.
_BAND_ENGINE_PREFIX = "band:"


def read_timer_region(reader: Any, frame: Any, bbox: list[int]) -> dict[str, Any]:
    """Read the match clock inside one box via the pipeline's own timer path
    (``timer.read_timer_at_bbox``: preprocess variants + digit-allowlist recognizer),
    so the test result answers the question that matters for clock-based splitting:
    *does a clock parse here?* ``text`` is the parsed clock ("6:57"), empty when
    nothing parses; ``confidence`` is 1.0/0.0 accordingly (the timer path returns a
    parse, not a per-word OCR score).

    On a parse failure the response also carries ``raw``: what OCR actually read
    (each preprocessing variant's distinct string, joined with "|" by the timer's own
    rejection recorder), empty when nothing legible was read at all — so the editor
    can show *why* a clock didn't parse (a hallucinated digit, a dropped colon) instead
    of a bare failure. The parse uses the drawn-box flags the pipeline's own fixed-box read
    uses (``timer.DRAWN_BOX_DOT_AS_COLON`` / ``DRAWN_BOX_UNSEPARATED_CENTISECONDS``: a dot
    stands for the colon, the centiseconds may be unseparated, whatever the game), so the
    test read previews exactly what analysis will read: a Melee clock OCRs as "7:44 63"
    and parses only with the second flag on."""
    from vodparse.timer import (DRAWN_BOX_DOT_AS_COLON, DRAWN_BOX_UNSEPARATED_CENTISECONDS,
                                read_timer_at_bbox)

    padded = pad_box(bbox)
    if TIMER_READ_DEBUG_DIR is not None:
        _dump_timer_crop(frame, padded, TIMER_READ_DEBUG_DIR)
    # confidence= is the *detector's* score in the pipeline (stored, not used to
    # read); there is no detector here, the user drew the box.
    rejections: list[dict[str, Any]] = []
    reading = read_timer_at_bbox(reader, frame, padded, confidence=1.0,
                                 rejections=rejections, debug_dir=TIMER_READ_DEBUG_DIR,
                                 dot_as_colon=DRAWN_BOX_DOT_AS_COLON,
                                 unseparated_centiseconds=DRAWN_BOX_UNSEPARATED_CENTISECONDS)
    return _timer_payload(reading, rejections)


def _dump_timer_crop(frame: Any, padded_bbox: list[int], directory: Path) -> None:
    """Write the padded timer crop (exactly what read_timer_at_bbox goes on to preprocess)
    as timer_crop.png in ``directory``; the variants land beside it via preprocess_variants.
    Best-effort: a dump failure must never break the read."""
    import cv2

    from vodparse.bbox import crop_frame

    crop = crop_frame(frame, padded_bbox)
    if crop is None:
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / "timer_crop.png"), crop)
    except (OSError, cv2.error):
        pass


def _timer_payload(reading: Any, rejections: list[dict[str, Any]]) -> dict[str, Any]:
    """The response entry for a clock read: the parsed clock (empty when nothing parsed,
    with ``raw`` saying what OCR saw) and a 1.0/0.0 confidence, see read_timer_region."""
    if reading is None:
        raw = str(rejections[0].get("raw_ocr_string", "")) if rejections else ""
        return {"text": "", "confidence": 0.0, "raw": raw}
    return {"text": reading["formatted"], "confidence": 1.0}


def _text_reading(text: TextBox | None) -> dict[str, Any]:
    """The response entry for a band read's find: its text, confidence and where it was
    read; an empty entry when the finder found nothing."""
    from vodparse.bbox import quad_to_bbox

    if text is None:
        return {"text": "", "confidence": 0.0}
    return {"text": text["text"], "confidence": float(text["confidence"]),
            "bbox": quad_to_bbox(text)}


def _tag_reading(entry: TagEntry | None) -> dict[str, Any]:
    """As _text_reading, for a gamertag the tournament-tag finder chose (its sponsor
    split off, as the pipeline stores it)."""
    reading = _text_reading(entry)
    if entry is not None:
        reading["sponsor"] = entry["sponsor"]
    return reading


# Which background keys need which band read (read_auto with ``keys``): the top band
# feeds the tag finder and the round and best-of labels, the bottom band the in-game
# names. The two players always come together (one finder answers for both).
_TOP_BAND_KEYS = frozenset({REGION_PLAYER1, REGION_PLAYER2, REGION_ROUND, REGION_BEST_OF})
_BOTTOM_BAND_KEYS = frozenset({REGION_INGAME_P1, REGION_INGAME_P2})


# The pipe evidence of the stored run a preview names (see run_pipe_evidence), for the one
# pipeline.json last asked about, keyed with its modification time: computed once however
# many stills of the video are read, and again after a re-analysis rewrites the file.
_run_pipe_evidence_cache: dict[tuple[str, float], Mapping[str, frozenset[str]]] = {}


def run_pipe_evidence(pipeline_path: str | None) -> Mapping[str, frozenset[str]]:
    """The pipe evidence (``word_merge.pipe_evidence_in``) of the stored run at
    ``pipeline_path``: the word pairs the run saw a pipe between, read off its raw OCR
    events. None, a missing file or an unreadable one has no evidence (nothing folds)."""
    from vodparse.jsonio import read_json
    from vodparse.word_merge import NO_PIPE_EVIDENCE, pipe_evidence_in

    if not pipeline_path:
        return NO_PIPE_EVIDENCE
    path = Path(pipeline_path)
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return NO_PIPE_EVIDENCE
    if key not in _run_pipe_evidence_cache:
        try:
            data = read_json(path)
            events = data.get("events") if isinstance(data, dict) else None
            evidence = pipe_evidence_in(events) if isinstance(events, list) else NO_PIPE_EVIDENCE
        except (OSError, ValueError):
            evidence = NO_PIPE_EVIDENCE
        _run_pipe_evidence_cache.clear()   # one video at a time: keep only the latest run's
        _run_pipe_evidence_cache[key] = evidence
    return _run_pipe_evidence_cache[key]


def read_auto(frame: Any, auto: dict[str, Any], engine_name: str, lang: str,
              engines: dict[tuple[str, str], Any]) -> tuple[dict[str, Any], str | None]:
    """Read ``frame`` the way an Auto run reads a sample (see the module docstring):
    the band OCR with the game's own bottom-band engine when it pins one, then the
    single-frame finders over the band texts, then the clock at the given box. Returns
    the readings keyed by region, and a note for the response when the game's pinned
    bottom-band engine can't read here, so the in-game names were left unread
    (INGAME_NAMES_UNREAD_NOTE), else None.

    With ``keys`` in ``auto`` (a layout request's ``background``), only those regions
    are answered: the bands they need are read, their finders run, and the clock is read
    only when ``timer`` is among them. Without ``keys`` the game's whole Auto plan is
    read, as a run with no layout reads it."""
    from vodparse.cli.pipeline import detect_text, texts_to_payload
    from vodparse.games import (default_game, full_game_seconds_for, game_by_id,
                                ingame_band_engine_for, ingame_band_top_for)
    from vodparse.ocr_engine import OCR_ENGINE_DOCTR
    from vodparse.ocr_langs import OCR_LANG_DEFAULT
    from vodparse.overlay_bands import BandOcrPlan, band_ocr_plan_for
    from vodparse.overlay_layout import ingame_band_mode_for
    from vodparse.tags import find_best_of, find_player_names, find_round, find_tournament_tags
    from vodparse.timer import read_calibrated_clock
    from vodparse.name_filters import strip_never_name_edges
    from vodparse.word_merge import merge_band_words, merge_overlay_words

    game = game_by_id(str(auto.get("game", ""))) or default_game()
    # The nameplate band's top edge, resolved as the pipeline resolves it: the Settings
    # widening is a calibration of Ultimate's HUD, so every other game reads at its own band.
    widening = auto.get("ingame_tag_frac")
    ultimate_widening = float(widening) if isinstance(widening, (int, float)) else None
    ingame_tag_frac = ingame_band_top_for(game, ultimate_widening)
    # What the nameplate band is read as: a layout's own choice when the request says
    # (the background of a layout read), else the game's default (the built-in Auto
    # reading). An unknown or unoffered mode falls back to that default too: for a game
    # that offers only "none" (Melee) that is "none", as the run reads such a layout.
    band_mode = auto.get("ingame_band")
    if not isinstance(band_mode, str) or band_mode not in game.ingame_band_choices:
        band_mode = ingame_band_mode_for(game, None)
    plan = band_ocr_plan_for(band_mode, None)
    wanted = auto.get("keys")
    if isinstance(wanted, list):
        wanted = {str(key) for key in wanted}
        plan = BandOcrPlan(top=plan.top and bool(wanted & _TOP_BAND_KEYS),
                           bottom=plan.bottom and bool(wanted & _BOTTOM_BAND_KEYS))
    else:
        wanted = _TOP_BAND_KEYS | _BOTTOM_BAND_KEYS | {REGION_TIMER}
    overlay = _band_engine(engines, engine_name, lang)
    # The game's pinned nameplate-band engine for that mode, as _load_detectors loads it
    # (English, since the band holds the game's own text). A pin that can't read here is
    # never replaced by the request's engine: the band is skipped and the note says why.
    bottom: Any = None
    note: str | None = None
    pinned = ingame_band_engine_for(game, band_mode)
    if plan.bottom:
        pin_reason = _pin_unusable_reason(pinned, engine_name)
        if pin_reason is not None:
            plan = BandOcrPlan(top=plan.top, bottom=False)
            note = INGAME_NAMES_UNREAD_NOTE.format(reason=pin_reason)
        elif pinned is not None and pinned != engine_name:
            bottom = _band_engine(engines, pinned, OCR_LANG_DEFAULT)

    frame_height, frame_width = frame.shape[:2]
    texts = texts_to_payload(detect_text(overlay, frame, 0.0, 0, ingame_tag_frac=ingame_tag_frac,
                                         bottom_ocr=bottom, bands=plan))
    # docTR reads one box per word; a run re-merges them into phrases before its finders
    # see them (word_merge.merge_doctr_reads), under this same gate: every box when the
    # overlay engine is docTR, only the bottom band's when the game's pinned band reader is
    # docTR beside an EasyOCR overlay engine. Then, as the run does for every engine
    # (extract_matches.load_pipeline), a pill read inside a tag's phrase is stripped from the
    # phrase's edge (name_filters.strip_never_name_edges). The run's pipe look-alike
    # fold needs its evidence across frames; one frame has none, so the stored run's is used
    # when the request names one (see run_pipe_evidence), and otherwise a pipe read as "I"
    # stays as read here (word_merge.fold_pipe_lookalikes).
    pipe_evidence = run_pipe_evidence(auto.get("pipe_evidence_run"))
    if engine_name == OCR_ENGINE_DOCTR:
        texts = merge_overlay_words(texts, pipe_evidence)
    elif bottom is not None and pinned == OCR_ENGINE_DOCTR:
        texts = merge_band_words(texts, ingame_tag_frac * frame_height, pipe_evidence)
    texts = strip_never_name_edges(texts)
    readings: dict[str, Any] = {}
    if plan.top and REGION_PLAYER1 in wanted:
        entries = find_tournament_tags(texts, frame_height=frame_height, frame_width=frame_width)
        readings[REGION_PLAYER1] = _tag_reading(entries[0] if entries else None)
        readings[REGION_PLAYER2] = _tag_reading(entries[1] if entries else None)
    if plan.top and REGION_ROUND in wanted:
        readings[REGION_ROUND] = _text_reading(find_round(texts, frame_height=frame_height))
    if plan.top and REGION_BEST_OF in wanted:
        readings[REGION_BEST_OF] = _text_reading(find_best_of(texts, frame_height=frame_height))
    if plan.bottom:
        names = find_player_names(texts, frame_width=frame_width, frame_height=frame_height,
                                  ingame_tag_frac=ingame_tag_frac)
        readings[REGION_INGAME_P1] = _text_reading(names[0] if names else None)
        readings[REGION_INGAME_P2] = _text_reading(names[1] if names else None)
    readings.update(_locked_readings(frame, auto.get("locked"), engine_name, lang, engines))
    clock_bbox = auto.get("clock_bbox")
    if (REGION_TIMER in wanted and isinstance(clock_bbox, list) and len(clock_bbox) == 4
            and game.clock_hud is not None):
        box = [int(v) for v in clock_bbox]
        rejections: list[dict[str, Any]] = []
        reading = read_calibrated_clock(
            _timer_reader(engines), frame, box, clock_hud=game.clock_hud,
            max_clock_seconds=float(full_game_seconds_for(game, None)), rejections=rejections)
        readings[REGION_TIMER] = {**_timer_payload(reading, rejections), "bbox": box}
    return readings, note


def _locked_readings(frame: Any, locked: object, engine_name: str, lang: str,
                     engines: dict[tuple[str, str], Any]) -> dict[str, Any]:
    """The player readings from the side-bar search's locked rows (see the module
    docstring): each locked player row read like a drawn player box, with ``bbox`` the row
    itself, and the sponsor from that player's locked sponsor row when the player row's own
    read split none off. Empty when the request carries no usable ``locked`` map, so the
    band's readings stand."""
    from vodparse.overlay_regions import SOURCE_SIDE_LOCK, OverlayRegions

    regions = OverlayRegions.from_config({"source": SOURCE_SIDE_LOCK, "boxes": locked})
    if regions is None:
        return {}
    engine = _box_engine(engines, engine_name, lang)
    readings: dict[str, Any] = {}
    for player_key in (REGION_PLAYER1, REGION_PLAYER2):
        box = regions.boxes.get(player_key)
        if box is None:
            continue
        reading: dict[str, Any] = dict(read_region(engine, frame, box, OCR_CANVAS_DEFAULT,
                                                   region=player_key))
        sponsor_key = regions.sponsor_key(player_key)
        if sponsor_key is not None and not reading.get("sponsor"):
            sponsor = read_region(engine, frame, regions.boxes[sponsor_key], OCR_CANVAS_DEFAULT,
                                  region=sponsor_key)
            reading["sponsor"] = sponsor["text"] or None
        reading["bbox"] = list(box)
        readings[player_key] = reading
    return readings


def _pin_unusable_reason(pinned: object, read_engine: str) -> str | None:
    """Why a game's pinned in-game nameplate engine (``pinned``: Rivals 2's docTR) can't
    read here, or None when it can or doesn't apply (no pin, or the pin is the engine the
    request reads with anyway). The in-game names are then left unread: never read with
    ``read_engine`` in the pin's place."""
    from vodparse.ocr_engine import engine_unusable_reason

    if not isinstance(pinned, str) or pinned == read_engine:
        return None
    return engine_unusable_reason(pinned)


def _band_engine(engines: dict[tuple[str, str], Any], engine_name: str, lang: str) -> Any:
    """The overlay engine for reading whole bands (what the pipeline's band pass uses),
    built once per (engine, language) and cached under its own key (see
    _BAND_ENGINE_PREFIX). An English EasyOCR overlay engine's reader doubles as the timer
    reader (the pipeline reuses it the same way), saving a duplicate model load;
    non-English readers must not: they misread the clock's colon."""
    from vodparse.device import resolve_device
    from vodparse.ocr_engine import make_overlay_engine

    key = (_BAND_ENGINE_PREFIX + engine_name, lang)
    engine = engines.get(key)
    if engine is None:
        engine, reader = make_overlay_engine(engine_name, lang, resolve_device())
        engines[key] = engine
        if reader is not None and lang == "en" and _TIMER_READER_KEY not in engines:
            engines[_TIMER_READER_KEY] = reader
    return engine


def _box_engine(engines: dict[tuple[str, str], Any], engine_name: str, lang: str) -> Any:
    """The engine for reading drawn boxes, derived from the cached band engine the way the
    pipeline derives it (``ocr_engine.make_layout_box_engine``: EasyOCR reads boxes with
    the band engine itself, docTR with a sibling at a small detector canvas that shares
    the band engine's models), so reading both boxes and bands never loads the models
    twice. Cached under the bare (engine, language) pair."""
    from vodparse.ocr_engine import make_layout_box_engine

    key = (engine_name, lang)
    engine = engines.get(key)
    if engine is None:
        engine = make_layout_box_engine(engine_name, _band_engine(engines, engine_name, lang))
        engines[key] = engine
    return engine


def _ingame_box_engine(engines: dict[tuple[str, str], Any], pinned: object,
                       read_engine: str, *, fallback: Any) -> Any:
    """The engine a drawn in-game nameplate box is read with, as a run picks it
    (``cli.pipeline._load_detectors``): the drawn-box engine of ``pinned`` (the request's
    ``ingame_engine``, the game's pinned nameplate engine), in English since the band holds
    the game's own text, when it is set and differs from ``read_engine``; otherwise (no
    pin, or the pin is ``read_engine``) ``fallback``, the request's own
    drawn-box engine. The caller has already left the boxes unread when the pin can't
    read here (see _pin_unusable_reason)."""
    from vodparse.ocr_langs import OCR_LANG_DEFAULT

    if not isinstance(pinned, str) or pinned == read_engine:
        return fallback
    return _box_engine(engines, pinned, OCR_LANG_DEFAULT)


def _timer_reader(engines: dict[tuple[str, str], Any]) -> Any:
    """The always-English EasyOCR reader used only for the timer box, built once and
    cached in ``engines`` under its own key. Mirrors the pipeline's dedicated
    ``timer_ocr`` reader: the clock is read by EasyOCR's recognizer regardless of the
    overlay engine choice (docTR included), and always in English — sharing a
    non-English overlay reader misreads the colon (see _Detectors in cli.pipeline)."""
    reader = engines.get(_TIMER_READER_KEY)
    if reader is None:
        from vodparse.ocr_langs import make_reader

        reader = make_reader("en")
        engines[_TIMER_READER_KEY] = reader
    return reader


def handle_request(request: dict[str, Any], engines: dict[tuple[str, str], Any]) -> dict[str, Any]:
    """Serve one request dict (see the module docstring for its shape: drawn ``regions``,
    or the ``auto`` reading), building and caching the OCR engine on first use, and
    return the response dict. Every failure — an engine that can't read here (docTR not
    installed or its models not downloaded: checked up front, so nothing builds, and so
    downloads, an engine the guard refuses), an unreadable frame, an unusable
    engine/language pair (SystemExit from make_overlay_engine), an OCR error — comes
    back as an ``error`` response rather than raising, so the serve loop outlives it."""
    from vodparse.ocr_engine import engine_unusable_reason

    token = request.get("token", -1)
    try:
        import cv2

        engine_name = str(request["engine"])
        unusable = engine_unusable_reason(engine_name)
        if unusable is not None:
            return {"token": token, "error": unusable}
        lang = str(request["lang"])
        response: dict[str, Any] = {"token": token, "engine": engine_name}
        notes: list[str] = []   # why some regions were left unread (INGAME_NAMES_UNREAD_NOTE)
        if "auto" in request:
            frame = cv2.imread(str(request["frame_path"]))
            if frame is None:
                return {"token": token, "error": f"Couldn't read the frame image: {request['frame_path']}"}
            response["readings"], note = read_auto(frame, request["auto"], engine_name, lang,
                                                   engines)
            if note is not None:
                response["note"] = note
            return response
        # The boxes are read on the pipeline's drawn-box engine (under docTR a sibling
        # with a small detector canvas), so the preview matches analysis exactly.
        engine = _box_engine(engines, engine_name, lang)
        frame = cv2.imread(str(request["frame_path"]))
        if frame is None:
            return {"token": token, "error": f"Couldn't read the frame image: {request['frame_path']}"}
        readings: dict[str, Any] = {}   # LayoutRead for text boxes, the timer's own shape for the clock
        background = request.get("background")
        if isinstance(background, dict):
            # The fields a run under this layout reads from the bands and the clock box
            # (see the module docstring); their keys never coincide with the drawn ones.
            background_readings, note = read_auto(frame, background, engine_name, lang, engines)
            readings.update(background_readings)
            if note is not None:
                notes.append(note)
        pinned = request.get("ingame_engine")
        pin_reason = _pin_unusable_reason(pinned, engine_name)
        for region, bbox in request["regions"].items():
            box = [int(v) for v in bbox]
            if region == REGION_TIMER:
                readings[region] = read_timer_region(_timer_reader(engines), frame, box)
                continue
            region_engine = engine
            if region in INGAME_REGION_KEYS:
                if pin_reason is not None:
                    continue   # left unread, never read with the request's engine instead
                region_engine = _ingame_box_engine(engines, pinned, engine_name,
                                                   fallback=engine)
            readings[region] = read_region(region_engine, frame, box, OCR_CANVAS_DEFAULT,
                                           region=region)
        if pin_reason is not None and any(key in request["regions"] for key in INGAME_REGION_KEYS):
            notes.append(INGAME_NAMES_UNREAD_NOTE.format(reason=pin_reason))
        if notes:
            response["note"] = " ".join(dict.fromkeys(notes))   # each distinct note once
        response["readings"] = readings
        return response
    except SystemExit as exc:   # make_overlay_engine's actionable "not usable" message
        return {"token": token, "error": str(exc)}
    except Exception as exc:   # noqa: BLE001 - any per-request failure keeps the loop serving
        return {"token": token, "error": f"Read failed: {exc}"}


def serve(stdin: Any = None, stdout: Any = None) -> int:
    """The serve loop: one response line per request line, until EOF. ``stdin``/
    ``stdout`` default to the process streams; tests pass StringIO."""
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    engines: dict[tuple[str, str], Any] = {}
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            response: dict[str, Any] = {"token": -1, "error": f"Malformed request line: {line[:200]}"}
        else:
            response = handle_request(request, engines)
        stdout.write(RESULT_LINE_PREFIX + json.dumps(response) + "\n")
        stdout.flush()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Serve overlay-layout test reads over a stdin/stdout line protocol "
                    "(see the module docstring); used by the GUI's layout editor.")


def main() -> int:
    build_arg_parser().parse_args()
    return serve()


if __name__ == "__main__":
    sys.exit(main())
