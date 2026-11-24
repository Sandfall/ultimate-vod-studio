"""Which overlay text regions a run reads per sample, and where they came from.

The pipeline reads a fixed set of text boxes on every OCR sample and stores the reads on
the event under ``layout_reads`` (``vodparse.schema.Event``); at extract time the votes
over those reads replace the position heuristics for the fields the boxes cover
(``tags.attach_overlay_regions``). Two things can decide where those boxes are:

* a **custom overlay layout** the user drew (``vodparse.overlay_layout``): the gamertag,
  round, best-of and nameplate boxes of a saved preset;
* the **side-bar lock** (``vodparse.side_overlay``): the pre-pass that finds where a 4:3
  game's stream puts the player names in its side bars and locks a row per player, plus a
  sponsor row above the tag when the stream stacks one there.

Both produce an ``OverlayRegions``: the region key → pixel box map the pipeline reads and
the extract step votes, tagged with its source. The vote never needs the layout object or
the lock itself, only "which regions exist"; keeping that as one value means a lock votes
through exactly the drawn-layout path, and a run's JSON records what was read either way
(``InferenceSettings.overlay_regions``). The drawn-layout record (``overlay_layout``) stays
as it is beside it: the GUI reads that one for the preset's name and clock length.

Qt-free and import-light: the pipeline, the extract step and the tests share it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vodparse.overlay_layout import (INGAME_REGION_KEYS, REGION_KEYS, REGION_PLAYER1,
                                     REGION_PLAYER2, REGION_TIMER, CustomOverlayLayout)
from vodparse.schema import BBox

# ── Region keys beyond the layout's ───────────────────────────────────────────────────

REGION_SPONSOR1 = "sponsor1"
REGION_SPONSOR2 = "sponsor2"
"""A sponsor row the side-bar lock reads on its own: the stream prints the sponsor on the
line ABOVE the gamertag (a per-side column), so it is a separate box, unlike the inline
"ORG | Player" prefix a gamertag box carries. Not a drawable layout region: the editor
iterates ``overlay_layout.REGION_KEYS`` and these are deliberately not in it."""

SPONSOR_REGION_FOR_PLAYER = {REGION_PLAYER1: REGION_SPONSOR1, REGION_PLAYER2: REGION_SPONSOR2}
"""Which sponsor region belongs to which player region."""

TEXT_REGION_KEYS = tuple(key for key in REGION_KEYS if key != REGION_TIMER) + (
    REGION_SPONSOR1, REGION_SPONSOR2)
"""Every region an ``OverlayRegions`` may hold: the layout's text regions (the timer is read
by the clock path, never as text) plus the sponsor rows."""

# ── Sources ───────────────────────────────────────────────────────────────────────────

SOURCE_LAYOUT = "layout"
SOURCE_SIDE_LOCK = "side_lock"
SOURCES = (SOURCE_LAYOUT, SOURCE_SIDE_LOCK)


@dataclass(frozen=True)
class OverlayRegions:
    """The overlay text regions one run reads per sample, in source-frame pixels, and
    where they came from (``source``, one of ``SOURCES``)."""

    source: str
    boxes: dict[str, BBox]
    """Region key (``TEXT_REGION_KEYS``) → ``[x1, y1, x2, y2]`` pixel box."""

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"overlay regions source must be one of {SOURCES}; got {self.source!r}")
        for key, box in self.boxes.items():
            if key not in TEXT_REGION_KEYS:
                raise ValueError(f"unknown overlay region {key!r}; expected one of {TEXT_REGION_KEYS}")
            if len(box) != 4 or box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError(f"overlay region {key!r} has a degenerate box {box!r}")
        if not self.boxes:
            raise ValueError("overlay regions must hold at least one box")

    @property
    def reads_player_boxes(self) -> bool:
        """True when both gamertag regions are read: the run then votes the tournament tags
        and sponsors from those reads instead of the built-in position heuristics, and the
        top band's tag reading has no consumer (``overlay_bands.band_ocr_plan``)."""
        return REGION_PLAYER1 in self.boxes and REGION_PLAYER2 in self.boxes

    def sponsor_key(self, player_key: str) -> str | None:
        """The sponsor region read alongside ``player_key``'s box, or None when that player
        has no separate sponsor row (the sponsor, if any, is then the inline prefix the
        player box's own read split off)."""
        key = SPONSOR_REGION_FOR_PLAYER[player_key]
        return key if key in self.boxes else None

    def as_config(self) -> dict[str, Any]:
        """The JSON-ready record written under the run's inference settings."""
        return {"source": self.source,
                "boxes": {key: list(box) for key, box in self.boxes.items()}}

    @classmethod
    def from_config(cls, record: object) -> OverlayRegions | None:
        """The regions a run recorded (``as_config``), or None when the record is absent
        or not in a shape this build understands (a future shape, or a hand-edited file):
        conservative like ``overlay_layout.layout_from_dict``'s callers, so an unreadable
        record reads the overlay the built-in way rather than failing the extract."""
        if not isinstance(record, dict):
            return None
        source = record.get("source")
        raw_boxes = record.get("boxes")
        if not isinstance(source, str) or not isinstance(raw_boxes, dict):
            return None
        boxes: dict[str, BBox] = {}
        for key, value in raw_boxes.items():
            if not (isinstance(key, str) and isinstance(value, list) and len(value) == 4
                    and all(isinstance(v, int) and not isinstance(v, bool) for v in value)):
                return None
            boxes[key] = list(value)
        try:
            return cls(source=source, boxes=boxes)
        except ValueError:
            return None


@dataclass(frozen=True)
class SideOverlayRecord:
    """What a side-bar search concluded, as its JSON record carries it
    (``side_overlay.SideOverlayScan.as_config``): the form the search's outcome travels in
    once it leaves the process that ran it. The layout preview's search hands one to the
    main window, and the main window hands it on to the run so the run need not search
    again (``vodparse.prepass_handover``).

    ``status`` is one of ``side_overlay.SIDE_OVERLAY_STATUSES``; ``boxes`` are the locked
    regions (None unless the status is "locked"); ``note`` is the search's one-line reason;
    ``config`` is the whole record, which the run writes under its own config unchanged."""

    status: str
    boxes: dict[str, BBox] | None
    note: str
    config: dict[str, Any]

    @classmethod
    def from_config(cls, record: object) -> SideOverlayRecord | None:
        """The record a search printed or a run saved, or None when it is absent or not in
        a shape this build understands: the caller then searches for itself."""
        if not isinstance(record, dict):
            return None
        status = record.get("status")
        note = record.get("note")
        if not isinstance(status, str):
            return None
        raw_boxes = record.get("boxes")
        boxes: dict[str, BBox] | None = None
        if raw_boxes is not None:
            regions = OverlayRegions.from_config({"source": SOURCE_SIDE_LOCK, "boxes": raw_boxes})
            if regions is None:
                return None
            boxes = regions.boxes
        return cls(status=status, boxes=boxes, note=note if isinstance(note, str) else "",
                   config=dict(record))

    def overlay_regions(self) -> OverlayRegions | None:
        """The locked rows as a run's overlay regions, or None when nothing was locked."""
        if self.boxes is None:
            return None
        return OverlayRegions(source=SOURCE_SIDE_LOCK, boxes=dict(self.boxes))

    def feed_edges(self) -> tuple[int, int] | None:
        """The left and right edges of the game feed the search placed its bars from
        (``record["feed"]``), or None when the record does not say: a run takes handed-over
        rows only when it places the bars at the same edges."""
        feed = self.config.get("feed")
        if not isinstance(feed, dict):
            return None
        x0, x1 = feed.get("x0"), feed.get("x1")
        if not isinstance(x0, int) or not isinstance(x1, int):
            return None
        return x0, x1


def layout_overlay_regions(layout: CustomOverlayLayout, frame_width: int,
                           frame_height: int) -> OverlayRegions | None:
    """The text regions a custom overlay layout draws, resolved to this video's pixels: every
    drawn region but the timer (the clock path reads that box, ``cli.pipeline._sample_timer``)
    and, when the layout's band is read as nothing, the nameplate boxes
    (``CustomOverlayLayout.reads_ingame_boxes``; the vote would ignore their reads). None for
    a layout that pins only the clock: it reads no text region, so the overlay is read the
    built-in way. This is the one place a layout's normalized rects become the per-sample
    read boxes, so the pipeline and the extract step agree on them by construction."""
    boxes: dict[str, BBox] = {}
    for key, rect in layout.regions().items():
        if key == REGION_TIMER:
            continue
        if key in INGAME_REGION_KEYS and not layout.reads_ingame_boxes:
            continue
        boxes[key] = rect.to_pixels(frame_width, frame_height)
    if not boxes:
        return None
    return OverlayRegions(source=SOURCE_LAYOUT, boxes=boxes)
