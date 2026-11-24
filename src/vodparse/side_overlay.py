"""Locating the player names in a 4:3 game's side bars before the main pass (the "side-bar lock").

A 4:3 game (Melee) is dropped onto a 16:9 stream, so every such stream has bars beside the
game feed, and streams put the player names there in shapes the built-in overlay reading
never looks at (it OCRs the top and bottom bands only, ``cli.pipeline.detect_text``):

* **per-side columns**: one player per bar, the tag as a header near the top of the column
  with a seed / Twitter handle / score under it, sometimes the sponsor on the line ABOVE
  the tag;
* **a stacked panel**: both players in ONE bar, one row each ("sponsor  tag  pronouns
  flags  score"), the top row being player 1;
* **the top band** after all: a major's overlay prints the names over the feed, which the
  built-in reader (``tags.find_tournament_tags``) already handles.

So: before the main pass, seek a few dozen frames from the later part of the VOD (the
overlay is constant across a VOD, but the first minutes are where a stream's setup
fumbles happen), OCR both bars and the top band on each, cluster the bars' text into rows,
throw out what cannot be a name (scores, pronoun sets, handles, seeds, round labels; text
that never changes across the frames is stream furniture, names change between sets), and
decide the shape. A tag row is then LOCKED: a tight band around the row's y, across the
whole bar, and the main pass reads that box on every sample exactly like a drawn layout box
(``vodparse.overlay_regions``, source ``SOURCE_SIDE_LOCK``). No confident lock means
today's reading, and the run says so (``SideOverlayScan.status``), so the Review screen can
flag the tags.

Where the feed sits comes from the calibrated clock box (``vodparse.clock_calibration``):
the clock is part of the game picture, so its centre gives where the feed sits, and the
frame's height gives the feed's width (``games.FeedPlacement``: the clock's own size is
never used); without a calibrated clock the feed is assumed centred at full height, 4:3 on
purpose (the user's call: a clock that was not steady leaves the geometry unknowable anyway,
and the measured ``MELEE_FEED_WIDTH_AT_1080P`` applies only once a clock places it).

Cost: the pre-pass is ``SIDE_OVERLAY_SAMPLES`` seeks with two or three full OCR passes each
(the bars and the top band on the band engine's full detector canvas), a few percent of a
run; the main pass then reads two to four small locked boxes per sample.

Qt-free and video-library-light: the pure parts (feed span, row clustering and
classification, the shape decision, the lock geometry) are unit-tested on hand-made text
boxes; ``locate_side_overlay`` is the seeking wrapper, on a ``frame_reader.VideoFrameSource``
sampled through ``frame_reader.sample_timestamps``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from vodparse.bbox import bbox_center_y, bbox_height, bbox_min_x, quad_to_bbox
from vodparse.frame_reader import VideoFrameSource, sample_timestamps
from vodparse.games import FeedPlacement
from vodparse.grouping import similar
from vodparse.overlay_layout import REGION_PLAYER1, REGION_PLAYER2
from vodparse.overlay_regions import (SOURCE_SIDE_LOCK, SPONSOR_REGION_FOR_PLAYER,
                                      OverlayRegions)
from vodparse.schema import BBox, TextBox
from vodparse.tags import TAG_TOP_FRAC, top_band_pair_found
from vodparse.text_utils import (is_best_of_label, is_ignored_tag_text, is_numeric_text,
                                 is_progress_label, is_round_text, parse_best_of)

# ── Tunables ────────────────────────────────────────────────────────────────────────

SIDE_OVERLAY_SAMPLES = 30
"""Frames the pre-pass looks at."""

SIDE_OVERLAY_SPAN = (0.40, 0.95)
"""The stretch of the VOD the samples are spread over, as fractions of its duration: the
later part, where bracket play (more distinct players per bar) has replaced the pre-show
and the stream's setup fumbles are over. The end is kept clear of the outro card."""

MIN_FRAMES_READ = 12
"""In-game frames the pre-pass must have read before it trusts anything it clustered."""

SIDE_OVERLAY_ROUNDS = 3
"""Sampling rounds at most (see sample_rounds). A VOD with long breaks between games can land
most of the first round's samples off-game: a seven-hour Melee pools stream read 10 in-game
frames of 30 and gave up, though the 10 already held the name row on both bars. Each further
round samples the span again between the earlier samples, and the pre-pass stops as soon as
MIN_FRAMES_READ frames were in-game. An off-game sample costs a seek and the in-game check,
no OCR, so a VOD whose first round suffices pays nothing more."""

MIN_BAR_WIDTH_FRAC = 0.08
"""A bar narrower than this fraction of the frame is not a bar (a feed pushed to one edge
leaves a sliver on that side): it is not read."""

ROW_CLUSTER_TOL_FRAC = 10 / 1080
"""Text boxes whose centre-y lies within this fraction of the frame height of a row's running
centre join that row: 10 px at 1080p, scaled with the frame so a 720p or 4K stream clusters
the same rows. Overlay rows sit well apart (a tag row and the seed under it differ by a row
height)."""

ROW_MIN_SUPPORT = 0.5
"""A tag row must have shown text on at least this fraction of the in-game frames read: a
name is on screen for the whole game."""

ROW_MIN_DISTINCT_TEXTS = 2
"""A tag row must have read at least this many different texts across the frames (fuzzy:
``grouping.similar`` folds OCR jitter on one name): names change between sets, stream
furniture ("BROUGHT TO YOU BY OUR PARTNERS", a sponsor logo) never does. Two, not more:
a monthly's capture holds three players, so a bar shows at most two. A single-set VOD
cannot be told apart this way and takes the not-found path by design."""

ROW_MIN_HEIGHT_FRAC = 0.012
"""A row's median text height, as a fraction of the frame height, below which it is fine
print (a partners strip, a caption), never a name: about 13 px at 1080p."""

SPONSOR_MAX_GAP_HEIGHTS = 1.5
"""A sponsor row sits directly above its tag: its centre no further above the tag's centre
than this many tag heights."""

SPONSOR_MIN_SUPPORT = 0.3
"""A sponsor row's support floor, lower than a tag's: not every player has a sponsor."""

SPONSOR_MAX_HEIGHT_RATIO = 0.9
"""A sponsor row's text is smaller than the tag's: at most this ratio of its height."""

STACK_MAX_GAP_HEIGHTS = 2.5
"""Two stacked players' rows are adjacent: their centres no further apart than this many
row heights."""

STACK_HEIGHT_RATIO = 1.5
"""Two stacked players' rows are drawn at the same size: their median heights within this
ratio of each other."""

PER_SIDE_MAX_Y_DIFF_FRAC = 0.03
"""Per-side columns are symmetric: the two tag rows' centres within this fraction of the
frame height of each other. Two bars with a varying row each at unrelated heights (a chat
panel on one side, a name on the other) are not that shape."""

TOP_BAND_PAIR_MIN_FRAC = 0.5
"""When the top band shows a full gamertag pair (``tags.top_band_pair_found``) on at least
this fraction of the in-game frames, the names live there and the bars lock nothing."""

LOCK_Y_PAD_HEIGHTS = 0.4
"""The locked box extends this many row heights above and below the row's text."""

LOCK_PER_SIDE_Y_PAD_HEIGHTS = 1.0
"""...and this many for a per-side column's tag row. Such a panel can print a sponsor line
above the tag and push the tag down to make room, so the tag has two positions about a
text height apart; the row is centred on the commoner one, and the band must hold both
(``tags.read_player_row`` tells the sponsor line from the tag). A stacked panel keeps the
tight band: its two rows are adjacent, and a taller one would reach into the other
player's row."""


# ── Names for the outcomes ──────────────────────────────────────────────────────────

SIDE_OVERLAY_LOCKED = "locked"
SIDE_OVERLAY_TOP_BAND = "top_band"
SIDE_OVERLAY_NOT_FOUND = "not_found"
SIDE_OVERLAY_SKIPPED = "skipped"
SIDE_OVERLAY_STATUSES = (SIDE_OVERLAY_LOCKED, SIDE_OVERLAY_TOP_BAND, SIDE_OVERLAY_NOT_FOUND,
                         SIDE_OVERLAY_SKIPPED)
"""What the pre-pass concluded: it locked tag rows; the top band carries the names; nothing
confident was found; or it did not run (the pipeline's word for a run it never started)."""

SHAPE_PER_SIDE = "per_side"
SHAPE_STACKED = "stacked"

FEED_FROM_CLOCK = "calibrated_clock"
FEED_ASSUMED = "assumed"

BAR_LEFT = "left"
BAR_RIGHT = "right"

ROLE_TAG = "tag"
ROLE_SPONSOR = "sponsor"
REJECT_SUPPORT = "rejected:support"
REJECT_HEIGHT = "rejected:height"
REJECT_CONSTANT = "rejected:constant"
REJECT_UNUSED = "rejected:unused"
"""A row's verdict: locked as a tag or a sponsor row, or why it was not: too few frames
showed it, its text is fine print, its text never changed (furniture), or it qualified but
the shape decision had no place for it."""

# ── Types ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FeedSpan:
    """Where the 4:3 game feed sits across the frame: its left and right edges in source
    pixels, and how they were found (``FEED_FROM_CLOCK`` / ``FEED_ASSUMED``)."""
    x0: int
    x1: int
    source: str


@dataclass(frozen=True)
class Bar:
    """One side bar: which side, and its full-height pixel box."""
    side: str
    bbox: BBox

    @property
    def width(self) -> int:
        """The bar's width in pixels (the lock's x-slack is a fraction of it)."""
        return self.bbox[2] - self.bbox[0]


@dataclass
class BarRow:
    """One horizontal text row of a bar, accumulated across the sampled frames: a running
    centre-y, the frames it showed on, the row's joined text per frame (its boxes left to
    right) and its text heights."""
    bar: str
    cy: float
    boxes: int = 0
    frames: set[int] = field(default_factory=set)
    row_texts: Counter[str] = field(default_factory=Counter)
    heights: list[float] = field(default_factory=list)

    def add_box(self, box: TextBox) -> None:
        """Fold one text box into the row: nudges the running centre, records its height."""
        self.boxes += 1
        self.cy += (bbox_center_y(box) - self.cy) / self.boxes
        self.heights.append(bbox_height(box))

    def add_frame_text(self, frame_index: int, text: str) -> None:
        """Record what the row read on one frame (its boxes joined left to right)."""
        self.frames.add(frame_index)
        self.row_texts[text] += 1

    def support(self, frames_read: int) -> float:
        """The fraction of the in-game frames read on which the row showed text."""
        return len(self.frames) / frames_read if frames_read else 0.0

    def median_height(self) -> float:
        """Median height of the row's text boxes across every frame, in pixels; 0.0 before
        any box."""
        return median(self.heights) if self.heights else 0.0

    def distinct_texts(self) -> int:
        """How many different texts the row read, folding OCR jitter (greedy first-fit
        clusters under ``grouping.similar``)."""
        representatives: list[str] = []
        for text in self.row_texts:
            if not any(similar(text, seen) for seen in representatives):
                representatives.append(text)
        return len(representatives)

    def top_texts(self, count: int = 3) -> list[str]:
        """The row's most-read texts, for the console and the probe's table."""
        return [text for text, _n in self.row_texts.most_common(count)]


@dataclass(frozen=True)
class RowVerdict:
    """One row's fate: locked as a tag or sponsor, or the reason it was not (see the
    ``ROLE_*`` / ``REJECT_*`` names)."""
    row: BarRow
    role: str

    def as_config(self, frames_read: int) -> dict[str, Any]:
        """The row summarised for the run's config and the probe's table."""
        return {"bar": self.row.bar, "cy": round(self.row.cy, 1),
                "height": round(self.row.median_height(), 1),
                "support": round(self.row.support(frames_read), 3),
                "distinct_texts": self.row.distinct_texts(),
                "texts": self.row.top_texts(), "role": self.role}


@dataclass(frozen=True)
class SideOverlayLock:
    """What the pre-pass locked: the shape it recognised and the region boxes (region key →
    source-frame pixel box) the main pass reads on every sample."""
    shape: str
    boxes: dict[str, BBox]

    def overlay_regions(self) -> OverlayRegions:
        """The locked boxes as the run's overlay regions (``SOURCE_SIDE_LOCK``), read by the
        main pass like drawn boxes."""
        return OverlayRegions(source=SOURCE_SIDE_LOCK, boxes=dict(self.boxes))


@dataclass(frozen=True)
class SideOverlayScan:
    """Everything the pre-pass concluded, for the pipeline, the run's config and the probe."""
    status: str
    feed: FeedSpan
    bars: list[Bar]
    verdicts: list[RowVerdict]
    top_pair_frames: int
    frames_read: int
    samples: int
    lock: SideOverlayLock | None
    note: str
    """One line saying why the status is what it is, for the console banner."""

    def as_config(self) -> dict[str, Any]:
        """The JSON-ready record written under the run's config."""
        return {
            "status": self.status,
            "shape": self.lock.shape if self.lock is not None else None,
            "feed": {"x0": self.feed.x0, "x1": self.feed.x1, "source": self.feed.source},
            "bars": {bar.side: list(bar.bbox) for bar in self.bars},
            "top_pair_frames": self.top_pair_frames,
            "frames_read": self.frames_read,
            "samples": self.samples,
            "boxes": ({key: list(box) for key, box in self.lock.boxes.items()}
                      if self.lock is not None else None),
            "rows": [verdict.as_config(self.frames_read) for verdict in self.verdicts],
            "note": self.note,
        }

    def overlay_regions(self) -> OverlayRegions | None:
        """The locked rows as the run's overlay regions, or None when nothing was locked."""
        return self.lock.overlay_regions() if self.lock is not None else None


# ── The feed ────────────────────────────────────────────────────────────────────────

def feed_span_from_clock(clock_bbox: BBox, frame_width: int, frame_height: int,
                         placement: FeedPlacement) -> FeedSpan:
    """The feed's span from the calibrated clock box: the feed runs the frame's full
    height and is ``placement.width_per_frame_height`` frame heights wide, and the clock
    box's centre sits ``placement.clock_center_frac`` of the way across it. Only the box's
    position is used, never its size. Clamped to the frame, so a
    feed pushed against one edge yields a bar on the other side only."""
    clock_center = (clock_bbox[0] + clock_bbox[2]) / 2
    width = frame_height * placement.width_per_frame_height
    left = clock_center - placement.clock_center_frac * width
    x0 = max(0, round(left))
    x1 = min(frame_width, round(left + width))
    return FeedSpan(x0=x0, x1=x1, source=FEED_FROM_CLOCK)


def assumed_feed_span(frame_width: int, frame_height: int) -> FeedSpan:
    """Without a calibrated clock: a full-height 4:3 feed centred in the frame (the outer
    eighth of a 16:9 frame on each side is bar)."""
    width = round(frame_height * 4 / 3)
    x0 = max(0, round((frame_width - width) / 2))
    return FeedSpan(x0=x0, x1=min(frame_width, x0 + width), source=FEED_ASSUMED)


def feed_span_for(clock_bbox: BBox | None, frame_width: int, frame_height: int,
                  placement: FeedPlacement) -> FeedSpan:
    """The feed's span for a search: from the calibrated clock box when there is one
    (``feed_span_from_clock``), otherwise assumed centred (``assumed_feed_span``). The one
    place that choice is made, so the run's pre-pass and the layout preview's search
    (``cli.side_overlay_locate``) look in the same bars."""
    if clock_bbox is not None:
        return feed_span_from_clock(clock_bbox, frame_width, frame_height, placement)
    return assumed_feed_span(frame_width, frame_height)


def side_bars(feed: FeedSpan, frame_width: int, frame_height: int) -> list[Bar]:
    """The bars either side of the feed, dropping a sliver narrower than
    ``MIN_BAR_WIDTH_FRAC`` of the frame."""
    min_width = MIN_BAR_WIDTH_FRAC * frame_width
    bars: list[Bar] = []
    if feed.x0 >= min_width:
        bars.append(Bar(BAR_LEFT, [0, 0, feed.x0, frame_height]))
    if frame_width - feed.x1 >= min_width:
        bars.append(Bar(BAR_RIGHT, [feed.x1, 0, frame_width, frame_height]))
    return bars


# ── Rows ────────────────────────────────────────────────────────────────────────────

def bar_candidates(texts: Sequence[TextBox]) -> list[TextBox]:
    """The boxes of one bar read that can be part of a name row: everything but the content
    that never names a player (scores and other numerics, pronoun sets, @handles, seed
    labels, W/L markers, round and best-of labels, bracket-progress counters)."""
    return [t for t in texts
            if t["text"].strip()
            and not is_numeric_text(t)
            and not is_ignored_tag_text(t["text"])
            and not is_round_text(t)
            and not is_best_of_label(t["text"])
            and parse_best_of(t["text"]) is None
            and not is_progress_label(t["text"])]


def _row_index_near(rows: Sequence[BarRow], cy: float, tol: float) -> int | None:
    """The index of the first row whose running centre lies within ``tol`` pixels of ``cy``,
    or None when no row is that close."""
    for index, row in enumerate(rows):
        if abs(cy - row.cy) <= tol:
            return index
    return None


def cluster_bar_rows(frames: Sequence[Sequence[TextBox]], bar: Bar,
                     frame_height: int) -> list[BarRow]:
    """Cluster the candidate boxes of one bar across the sampled frames into rows by
    centre-y (greedy first-fit within ``ROW_CLUSTER_TOL_FRAC`` of ``frame_height``).
    ``frames`` holds one list of candidate boxes per in-game frame read (``bar_candidates``),
    in frame order. Each row also keeps what it read per frame: its boxes on that frame
    joined left to right."""
    tol = ROW_CLUSTER_TOL_FRAC * frame_height
    rows: list[BarRow] = []
    for frame_index, texts in enumerate(frames):
        boxes_by_row: dict[int, list[TextBox]] = {}
        for box in sorted(texts, key=bbox_center_y):
            cy = bbox_center_y(box)
            index = _row_index_near(rows, cy, tol)
            if index is None:
                rows.append(BarRow(bar=bar.side, cy=cy))
                index = len(rows) - 1
            rows[index].add_box(box)
            boxes_by_row.setdefault(index, []).append(box)
        for index, row_boxes in boxes_by_row.items():
            ordered = sorted(row_boxes, key=bbox_min_x)
            rows[index].add_frame_text(frame_index, " ".join(b["text"].strip() for b in ordered))
    return sorted(rows, key=lambda row: row.cy)


def classify_rows(rows: Sequence[BarRow], frames_read: int,
                  frame_height: int) -> list[RowVerdict]:
    """Every row's first verdict: rejected for fine print, for never changing its text or for
    showing on too few frames (below even a sponsor row's floor), else left as a candidate
    (``REJECT_UNUSED`` until the shape decision gives it a role)."""
    verdicts: list[RowVerdict] = []
    for row in rows:
        if row.median_height() < ROW_MIN_HEIGHT_FRAC * frame_height:
            role = REJECT_HEIGHT
        elif row.distinct_texts() < ROW_MIN_DISTINCT_TEXTS:
            role = REJECT_CONSTANT
        elif row.support(frames_read) < SPONSOR_MIN_SUPPORT:
            role = REJECT_SUPPORT
        else:
            role = REJECT_UNUSED
        verdicts.append(RowVerdict(row=row, role=role))
    return verdicts


def _candidates(verdicts: Sequence[RowVerdict]) -> list[BarRow]:
    return [v.row for v in verdicts if v.role == REJECT_UNUSED]


def pick_bar_tag_row(candidates: Sequence[BarRow], frames_read: int) -> BarRow | None:
    """The tag row of one bar among its candidates: those with a tag's support, the tallest
    first (a column's header is bigger than the handle strip under it), the higher one on a
    tie (the header sits above its sub-labels). None when no row has a tag's support."""
    eligible = [row for row in candidates if row.support(frames_read) >= ROW_MIN_SUPPORT]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (round(row.median_height()), -row.cy))


def sponsor_row_above(tag: BarRow, candidates: Sequence[BarRow]) -> BarRow | None:
    """The sponsor row directly above ``tag``, if the stream stacks one there: the nearest
    candidate above it within ``SPONSOR_MAX_GAP_HEIGHTS`` tag heights, drawn smaller than
    the tag. (A candidate already passed the distinct-text bar, so a constant label above
    the tag never qualifies.) None when there is none."""
    tag_height = tag.median_height()
    above = [row for row in candidates
             if row is not tag
             and row.cy < tag.cy
             and tag.cy - row.cy <= SPONSOR_MAX_GAP_HEIGHTS * tag_height + tag_height / 2
             and row.median_height() <= SPONSOR_MAX_HEIGHT_RATIO * tag_height]
    if not above:
        return None
    return max(above, key=lambda row: row.cy)


def stacked_pair(candidates: Sequence[BarRow], frames_read: int) -> tuple[BarRow, BarRow] | None:
    """Two players stacked in one bar: the adjacent pair of candidate rows (both with a tag's
    support) whose centres sit within ``STACK_MAX_GAP_HEIGHTS`` row heights and whose text
    heights match, taking the pair with the most combined support. Returned top first: the
    top row is player 1. None when no such pair exists."""
    eligible = sorted((row for row in candidates if row.support(frames_read) >= ROW_MIN_SUPPORT),
                      key=lambda row: row.cy)
    best: tuple[BarRow, BarRow] | None = None
    best_support = 0.0
    for top, bottom in zip(eligible, eligible[1:]):
        heights = (top.median_height(), bottom.median_height())
        if min(heights) <= 0 or max(heights) / min(heights) > STACK_HEIGHT_RATIO:
            continue
        if bottom.cy - top.cy > STACK_MAX_GAP_HEIGHTS * max(heights):
            continue
        support = top.support(frames_read) + bottom.support(frames_read)
        if support > best_support:
            best, best_support = (top, bottom), support
    return best


def lock_box(row: BarRow, bar: Bar, frame_height: int,
             y_pad_heights: float = LOCK_Y_PAD_HEIGHTS) -> BBox:
    """The box the main pass reads for ``row``: a band around the row's y
    (``y_pad_heights`` row heights of padding above and below its text), across the whole
    width of its bar.

    The whole bar, not the x-extent the pre-pass happened to see: the names shown on the
    sampled frames say nothing about how long a later player's name is, and a box sized
    from them cut longer names off (and came out a different width in each bar). What
    else sits on the row (flags, an icon, a score) is dropped when the row is read
    (``tags.read_player_row``)."""
    height = row.median_height()
    y0 = max(0, round(row.cy - height / 2 - y_pad_heights * height))
    y1 = min(frame_height, round(row.cy + height / 2 + y_pad_heights * height))
    return [bar.bbox[0], y0, bar.bbox[2], y1]


def share_y_band(first: BBox, second: BBox) -> tuple[BBox, BBox]:
    """Two boxes given one y band, the union of theirs, each keeping its own x. Per-side
    columns are drawn symmetrically, so the two players' rows are one band; the rows'
    measured heights differ a little with the names on show."""
    y0, y1 = min(first[1], second[1]), max(first[3], second[3])
    return [first[0], y0, first[2], y1], [second[0], y0, second[2], y1]


# ── The decision ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Shape:
    """A recognised shape: its name and the tag (and sponsor) rows per player region."""
    shape: str
    tags: dict[str, tuple[BarRow, Bar]]
    sponsors: dict[str, tuple[BarRow, Bar]]


def _bar_by_side(bars: Sequence[Bar], side: str) -> Bar | None:
    return next((bar for bar in bars if bar.side == side), None)


def bar_tag_rows(verdicts_by_bar: dict[str, list[RowVerdict]],
                 frames_read: int) -> dict[str, BarRow | None]:
    """Each bar's tag row (``pick_bar_tag_row`` over its candidates), keyed by side; None for
    a bar with no row of a tag's support."""
    return {side: pick_bar_tag_row(_candidates(verdicts), frames_read)
            for side, verdicts in verdicts_by_bar.items()}


def decide_shape(verdicts_by_bar: dict[str, list[RowVerdict]],
                 tags: dict[str, BarRow | None], bars: Sequence[Bar],
                 frames_read: int, frame_height: int) -> _Shape | None:
    """Which shape the bars' candidate rows make, if any: per-side columns (a tag row in
    BOTH bars at about the same height, each with the sponsor row above it when there is
    one) beat a stacked panel (two adjacent rows in one bar). None otherwise, including two
    bars whose tag rows sit at unrelated heights (``PER_SIDE_MAX_Y_DIFF_FRAC``): that is not
    a shape the pre-pass knows, so it locks nothing rather than guess. ``tags`` is each
    bar's tag row (``bar_tag_rows``), picked once by the caller, which also words the
    not-found note from them."""
    candidates = {side: _candidates(verdicts) for side, verdicts in verdicts_by_bar.items()}
    left, right = tags.get(BAR_LEFT), tags.get(BAR_RIGHT)
    left_bar, right_bar = _bar_by_side(bars, BAR_LEFT), _bar_by_side(bars, BAR_RIGHT)

    if (left is not None and right is not None and left_bar is not None and right_bar is not None
            and abs(left.cy - right.cy) <= PER_SIDE_MAX_Y_DIFF_FRAC * frame_height):
        tag_rows = {REGION_PLAYER1: (left, left_bar), REGION_PLAYER2: (right, right_bar)}
        sponsors: dict[str, tuple[BarRow, Bar]] = {}
        for key, (tag, bar) in tag_rows.items():
            sponsor = sponsor_row_above(tag, candidates[bar.side])
            if sponsor is not None:
                sponsors[SPONSOR_REGION_FOR_PLAYER[key]] = (sponsor, bar)
        return _Shape(SHAPE_PER_SIDE, tag_rows, sponsors)
    for bar in bars:
        pair = stacked_pair(candidates.get(bar.side, []), frames_read)
        if pair is not None:
            top, bottom = pair
            return _Shape(SHAPE_STACKED, {REGION_PLAYER1: (top, bar), REGION_PLAYER2: (bottom, bar)},
                          {})
    return None


def scan_from_reads(bar_reads: dict[str, list[list[TextBox]]], top_pair_frames: int,
                    feed: FeedSpan, bars: Sequence[Bar], frame_size: tuple[int, int], *,
                    frames_read: int, samples: int) -> SideOverlayScan:
    """The pre-pass's conclusion from what it read: ``bar_reads`` maps each bar's side to its
    candidate boxes per in-game frame (``bar_candidates``, one list per frame, the same
    frame order for every bar), ``top_pair_frames`` counts the in-game frames whose top
    band showed a full gamertag pair, ``frames_read`` the in-game frames and ``samples``
    every seek. The ``calibrate_from_boxes`` analogue: pure, so the decision is testable
    on hand-made boxes and the probe can re-run it from cached reads."""
    frame_width, frame_height = frame_size
    verdicts_by_bar = {
        bar.side: classify_rows(cluster_bar_rows(bar_reads.get(bar.side, []), bar, frame_height),
                                frames_read, frame_height)
        for bar in bars}
    verdicts = [v for side in verdicts_by_bar for v in verdicts_by_bar[side]]

    def scan(status: str, lock: SideOverlayLock | None, note: str) -> SideOverlayScan:
        return SideOverlayScan(status=status, feed=feed, bars=list(bars), verdicts=verdicts,
                               top_pair_frames=top_pair_frames, frames_read=frames_read,
                               samples=samples, lock=lock, note=note)

    if frames_read < MIN_FRAMES_READ:
        return scan(SIDE_OVERLAY_NOT_FOUND, None,
                    f"only {frames_read} of {samples} sampled frames were ingame "
                    f"(need {MIN_FRAMES_READ})")
    if top_pair_frames >= TOP_BAND_PAIR_MIN_FRAC * frames_read:
        return scan(SIDE_OVERLAY_TOP_BAND, None,
                    f"the top band shows a name pair on {top_pair_frames} of {frames_read} "
                    "ingame frames")
    if not bars:
        return scan(SIDE_OVERLAY_NOT_FOUND, None, "the feed leaves no side bar to read")
    tags = bar_tag_rows(verdicts_by_bar, frames_read)
    shape = decide_shape(verdicts_by_bar, tags, bars, frames_read, frame_height)
    if shape is None:
        left, right = tags.get(BAR_LEFT), tags.get(BAR_RIGHT)
        if left is not None and right is not None:
            # Both bars had a tag row, too far apart in height to be per-side columns.
            return scan(SIDE_OVERLAY_NOT_FOUND, None,
                        f"tag rows at y {left.cy:.0f} vs {right.cy:.0f}, not aligned")
        return scan(SIDE_OVERLAY_NOT_FOUND, None,
                    "no row in the bars reads like a player tag on enough frames")

    boxes: dict[str, BBox] = {}
    roles: dict[int, str] = {}
    tag_y_pad = (LOCK_PER_SIDE_Y_PAD_HEIGHTS if shape.shape == SHAPE_PER_SIDE
                 else LOCK_Y_PAD_HEIGHTS)
    for key, (row, bar) in shape.tags.items():
        boxes[key] = lock_box(row, bar, frame_height, tag_y_pad)
        roles[id(row)] = ROLE_TAG
    for key, (row, bar) in shape.sponsors.items():
        boxes[key] = lock_box(row, bar, frame_height)
        roles[id(row)] = ROLE_SPONSOR
    if shape.shape == SHAPE_PER_SIDE:
        boxes[REGION_PLAYER1], boxes[REGION_PLAYER2] = share_y_band(boxes[REGION_PLAYER1],
                                                                    boxes[REGION_PLAYER2])
    verdicts[:] = [RowVerdict(row=v.row, role=roles.get(id(v.row), v.role)) for v in verdicts]
    return scan(SIDE_OVERLAY_LOCKED, SideOverlayLock(shape=shape.shape, boxes=boxes),
                f"locked {shape.shape} rows {', '.join(boxes)} from {frames_read} ingame frames")


# ── The seeking wrapper ─────────────────────────────────────────────────────────────

IngameCheck = Callable[[np.ndarray], bool]
"""Whether a sampled frame is in-game (the clock detector fires on it): only those frames
count towards row support, so break screens and intermissions cannot starve a row."""

SideOverlayFrameCallback = Callable[[np.ndarray, list[BBox], float], None]
"""What ``locate_side_overlay`` calls per sampled frame: the frame, the boxes to draw on it
(the bars, then that frame's candidate boxes) and the timestamp. The pipeline uses it to
show the frames it is looking at in the GUI's preview."""


def sample_rounds(duration_sec: float, samples: int = SIDE_OVERLAY_SAMPLES,
                  span: tuple[float, float] = SIDE_OVERLAY_SPAN,
                  rounds: int = SIDE_OVERLAY_ROUNDS) -> list[list[float]]:
    """The pre-pass's timestamps, round by round. The first round is
    ``frame_reader.sample_timestamps``' even spread over ``span``; round k is that grid
    shifted later by k / ``rounds`` of its step, so the rounds interleave and never repeat a
    frame (a shifted timestamp past the span's end is dropped). A video too short for two
    samples gets the one round."""
    first = sample_timestamps(duration_sec, samples, span)
    if len(first) < 2:
        return [first]
    step = first[1] - first[0]
    end = duration_sec * span[1]
    all_rounds = [first]
    for k in range(1, rounds):
        shifted = [round(t + step * k / rounds, 3) for t in first]
        all_rounds.append([t for t in shifted if t <= end])
    return all_rounds


def locate_side_overlay(video_path: str | Path, band_engine: Any, canvas_size: int,
                        feed: FeedSpan, *, is_ingame: IngameCheck,
                        samples: int = SIDE_OVERLAY_SAMPLES,
                        on_frame: SideOverlayFrameCallback | None = None
                        ) -> SideOverlayScan | None:
    """Seek ``samples`` frames over the VOD's later part, OCR the side bars and the top band
    on each in-game frame with ``band_engine`` (the overlay engine at its full detector
    canvas: never the drawn-box engine, whose small canvas would miss a bar's text), and
    conclude (``scan_from_reads``). When fewer than MIN_FRAMES_READ of them were in-game,
    sample again between them, round by round (``sample_rounds``).

    ``is_ingame`` (the game's clock detector firing on the frame) is required: only a frame
    it passes is read or counted, so a stream's between-games layout never supplies a row.
    With no clock detector the search does not run at all (the callers refuse), rather than
    counting every sampled frame. The bars and the top band are placed on the decoded
    frames' size. Opens its own capture and releases it; the main pass opens the video
    afresh afterwards. None when the video cannot be opened, has no duration, or decodes
    no frame. ``on_frame`` sees every sampled frame, for a live preview."""
    from vodparse.layout_reads import detect_region_texts

    try:
        source = VideoFrameSource(str(video_path))
    except ValueError:
        return None
    with source:
        duration = source.duration_sec()
        if duration <= 0:
            return None
        frame_size = source.frame_size()
        if frame_size is None:
            return None
        frame_width, frame_height = frame_size
        bars = side_bars(feed, frame_width, frame_height)
        top_band: BBox = [0, 0, frame_width, int(frame_height * TAG_TOP_FRAC)]
        bar_reads: dict[str, list[list[TextBox]]] = {bar.side: [] for bar in bars}
        top_pair_frames = frames_read = sampled = 0
        for timestamps in sample_rounds(duration, samples):
            if frames_read >= MIN_FRAMES_READ:
                break   # enough in-game frames: no further round
            sampled += len(timestamps)
            for ts, frame in source.frames_at(timestamps):
                drawn: list[BBox] = [list(bar.bbox) for bar in bars]
                if is_ingame(frame):
                    frames_read += 1
                    for bar in bars:
                        candidates = bar_candidates(
                            detect_region_texts(band_engine, frame, bar.bbox, canvas_size))
                        bar_reads[bar.side].append(candidates)
                        drawn.extend(quad_to_bbox(box) for box in candidates)
                    top_texts = detect_region_texts(band_engine, frame, top_band, canvas_size)
                    if top_band_pair_found(top_texts, frame_width, frame_height):
                        top_pair_frames += 1
                if on_frame is not None:
                    on_frame(frame, drawn, ts)
    return scan_from_reads(bar_reads, top_pair_frames, feed, bars, frame_size,
                           frames_read=frames_read, samples=sampled)
