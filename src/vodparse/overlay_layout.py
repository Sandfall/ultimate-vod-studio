"""Custom overlay layouts: named presets of drawn boxes over a VOD frame.

Not every tournament stream uses the small set of overlay shapes the built-in
heuristics (tags.py, timer.py, ...) already know how to read. A custom overlay layout
lets a user teach the app a new one by hand: draw a box around player 1's tag, player
2's tag, and (optionally) the round label, the best-of label, and the match clock,
once, on a representative frame, and save it under a name. A later commit adds the GUI editor that draws these
boxes and the pipeline code that reads text out of them; this module only defines the
data model and the on-disk store.

Box coordinates are stored as **fractions of the frame's width/height**, not raw
pixels: the same convention tags.py uses for its overlay bands (see TAG_TOP_FRAC /
OVERLAY_BOTTOM_FRAC). A fraction is resolution-independent, so one saved preset covers
both a 720p and a 1080p capture of the same stream; a pixel box would need a separate
preset per resolution. ``NormRect.to_pixels``/``from_pixels`` convert to/from the
``BBox`` pixel convention (vodparse.schema.BBox: ``[x1, y1, x2, y2]`` ints) at the
point a frame's actual size is known.

Every preset belongs to one game (``CustomOverlayLayout.game``, a ``vodparse.games`` id):
the main screen's chooser and the editor only show the presets of the selected game. A
game without a clock detector of its own (Melee, Rivals 2, "Other") cannot use the
built-in "Auto" reading, so a preset for it must include a timer box: the model enforces
that rule itself, so the store, the pipeline and the editor all share one check.

A preset also says what the game's own in-game nameplate band holds
(``CustomOverlayLayout.ingame_band``, one of ``vodparse.games.INGAME_BAND_MODES``): the
players' in-game tags, the characters' names, or nothing worth reading. The game supplies
the default (``Game.ingame_band_default``); the preset stores the choice, so a stream whose
capture differs from the norm can override it. *Where* that band sits stays a property of
the game (``Game.ingame_hud``), never of the preset, with one refinement: a preset may draw
the two nameplate boxes themselves (``ingame_p1`` / ``ingame_p2``, optional). A drawn box is
read per sample like the overlay boxes and voted for its side, in place of searching the band
for a mirrored pair; an undrawn side is simply not read. ``ingame_band_mode_for`` resolves
the mode for a run.

Presets are stored together in a single versioned JSON sidecar,
``overlay_layouts.json``, at the root of the app's writable workspace (see
``vodparse.runtime_paths.workspace_dir``) — shared across all VODs, since a layout is a
property of the streaming setup/tournament, not of any one recording. The store is
read conservatively, mirroring ``vodparse.extract_flags``: any corruption (missing
file, bad JSON, wrong version, a malformed entry) degrades to "no presets" rather than
raising, so a hand-edited or half-written store can never crash the GUI on startup.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vodparse.games import INGAME_BAND_MODES, INGAME_BAND_NONE, Game, game_by_id
from vodparse.jsonio import read_json, write_json_atomic
from vodparse.match_extraction import FULL_GAME_SECONDS
from vodparse.runtime_paths import workspace_dir
from vodparse.schema import BBox

# Reserved preset name meaning "use the built-in heuristics instead of a custom
# layout" (the first item of the GUI's Overlay layout list); never a name a saved
# preset may take. Compared case-insensitively so "auto" and "AUTO" are also rejected.
AUTO_LAYOUT_NAME = "Auto"

# The layout store's filename, and its schema version (bumped when the on-disk shape
# changes, so a reader can reject an older file). Version 2 added the required "game"
# field and made "timer" / "clock_seconds" required; version 3 added the required
# "ingame_band" field; version 4 added the required (nullable) "ingame_p1" / "ingame_p2"
# boxes. Older stores load as "no presets" by design (no migration: the store predates any
# release; a hand edit adds the missing keys).
LAYOUT_STORE_NAME = "overlay_layouts.json"
LAYOUT_STORE_VERSION = 4

# The rule for games without a clock detector, worded for the user: the model raises it,
# the editor's Save shows it. Formatted with the game's display name.
TIMER_BOX_REQUIRED_MESSAGE = ("{game} has no clock detector, so a layout for it needs a "
                              "defined timer box.")

# The two gamertag boxes are a pair: the editor and the model refuse a preset with only one
# of them, and a preset with no box at all (nothing for a run to read).
PLAYER_BOXES_PAIR_MESSAGE = ("Draw both player tag boxes. (Or leave them blank to keep auto "
                             "detection.)")
NO_BOXES_MESSAGE = "Draw at least one box before saving."

# Region keys shared by the GUI editor and pipeline consumer, so both name these
# boxes the same way. Order matches drawing/tab order: the two tag boxes first (drawn
# together or not at all), then the optional labels. The timer box lets the match clock be read at a user-drawn
# spot instead of relying on the game's YOLO clock detector, which only knows its own
# game's clock graphics, so a drawn timer box is what makes splitting possible for a
# game without one ("Other").
REGION_PLAYER1 = "player1"
REGION_PLAYER2 = "player2"
REGION_ROUND = "round"
REGION_BEST_OF = "best_of"
REGION_TIMER = "timer"
# The game's own nameplate boxes (what the in-game band holds for each side: Ultimate's
# player tags, Rivals 2's character names). Optional: when drawn, that side is read at the
# box instead of searched for in the band (see the module docstring).
REGION_INGAME_P1 = "ingame_p1"
REGION_INGAME_P2 = "ingame_p2"
REGION_KEYS = (REGION_PLAYER1, REGION_PLAYER2, REGION_ROUND, REGION_BEST_OF,
               REGION_TIMER, REGION_INGAME_P1, REGION_INGAME_P2)
INGAME_REGION_KEYS = (REGION_INGAME_P1, REGION_INGAME_P2)

# A layout's full match clock at game start, in seconds. Ultimate's 7:00 is the default (the
# very constant the extraction math assumes, so a default layout changes nothing); other
# games run 6:00 or 8:00, and the extract stage uses a layout's value in place of the
# constant wherever it reads or back-dates from the clock. The cap is what the editor's
# m:ss control can express.
DEFAULT_CLOCK_SECONDS = FULL_GAME_SECONDS
CLOCK_SECONDS_MAX = 59 * 60 + 59


@dataclass(frozen=True)
class NormRect:
    """A box normalized to fractions of frame width/height: ``0 <= x1 < x2 <= 1`` and
    ``0 <= y1 < y2 <= 1``. See the module docstring for why fractions rather than
    pixels."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if not (0 <= self.x1 < self.x2 <= 1):
            raise ValueError(
                f"NormRect x-range must satisfy 0 <= x1 < x2 <= 1; got x1={self.x1}, x2={self.x2}")
        if not (0 <= self.y1 < self.y2 <= 1):
            raise ValueError(
                f"NormRect y-range must satisfy 0 <= y1 < y2 <= 1; got y1={self.y1}, y2={self.y2}")

    def to_pixels(self, frame_width: int, frame_height: int) -> BBox:
        """Convert to a pixel ``[x1, y1, x2, y2]`` box for a frame of the given size,
        rounding each coordinate to the nearest pixel."""
        return [
            round(self.x1 * frame_width),
            round(self.y1 * frame_height),
            round(self.x2 * frame_width),
            round(self.y2 * frame_height),
        ]

    @classmethod
    def from_pixels(cls, bbox: BBox, frame_width: int, frame_height: int) -> "NormRect":
        """Build a ``NormRect`` from a pixel ``[x1, y1, x2, y2]`` box and the frame size
        it was drawn on. Coordinates are clamped into ``[0, 1]`` first (a box drawn to
        the very edge of the frame, or nudged a pixel past it by the editor, should
        still normalize rather than raise); raises ``ValueError`` only if the clamped
        result is still degenerate (zero or negative width/height)."""
        x1 = min(max(bbox[0] / frame_width, 0.0), 1.0)
        y1 = min(max(bbox[1] / frame_height, 0.0), 1.0)
        x2 = min(max(bbox[2] / frame_width, 0.0), 1.0)
        y2 = min(max(bbox[3] / frame_height, 0.0), 1.0)
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)


@dataclass(frozen=True)
class CustomOverlayLayout:
    """A named preset of overlay boxes drawn on a representative VOD frame, for one game:
    the two required gamertag boxes, plus the round/best-of labels and the match clock
    when the stream's overlay shows them — and the length of that clock at game start
    (``clock_seconds``, see DEFAULT_CLOCK_SECONDS). The timer box is required when the
    preset's game has no clock detector (see ``vodparse.games.Game.has_timer_model``).
    ``ingame_band`` says what the game's own nameplate band holds for this stream, and
    ``ingame_p1`` / ``ingame_p2`` may pin where each side's nameplate sits (see the module
    docstring)."""

    name: str
    game: str
    """The ``vodparse.games`` id of the game this preset is for. Must name a registry
    entry; the chooser and the editor filter presets by it."""
    ingame_band: str
    """What the in-game nameplate band is read as: one of ``vodparse.games.INGAME_BAND_MODES``
    (``INGAME_BAND_NONE`` / ``_TAGS`` / ``_CHARACTER``). The editor preselects the game's
    ``ingame_band_default``; a saved preset always carries an explicit value."""
    player1: NormRect | None = None
    """The stream's gamertag boxes. Drawn together or not at all: a preset without them
    keeps the built-in tournament-overlay reading for the tags (the same way an undrawn
    round or best-of box keeps its built-in finder), so a preset can pin only the clock."""
    player2: NormRect | None = None
    round_label: NormRect | None = None
    best_of: NormRect | None = None
    timer: NormRect | None = None
    ingame_p1: NormRect | None = None
    ingame_p2: NormRect | None = None
    clock_seconds: int = DEFAULT_CLOCK_SECONDS

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("CustomOverlayLayout name must not be empty")
        if self.name.casefold() == AUTO_LAYOUT_NAME.casefold():
            raise ValueError(
                f"{self.name!r} is reserved for the built-in heuristics; choose another name")
        if not (1 <= self.clock_seconds <= CLOCK_SECONDS_MAX):
            raise ValueError(f"clock_seconds must be within 1..{CLOCK_SECONDS_MAX}; "
                             f"got {self.clock_seconds}")
        if self.ingame_band not in INGAME_BAND_MODES:
            raise ValueError(f"layout {self.name!r}: ingame_band must be one of "
                             f"{INGAME_BAND_MODES}; got {self.ingame_band!r}")
        game = game_by_id(self.game)
        if game is None:
            raise ValueError(f"layout {self.name!r} names an unknown game id {self.game!r}")
        if not game.has_timer_model and self.timer is None:
            # Without a timer detector the drawn box is the run's only clock source, and
            # without a clock nothing splits the VOD into games.
            raise ValueError(TIMER_BOX_REQUIRED_MESSAGE.format(game=game.display_name))
        if (self.player1 is None) != (self.player2 is None):
            # One gamertag box says nothing usable about the other side; the built-in
            # reading works per pair, so the choice is both boxes or the built-in pair.
            raise ValueError(PLAYER_BOXES_PAIR_MESSAGE)
        if not self.regions():
            raise ValueError(NO_BOXES_MESSAGE)

    def regions(self) -> dict[str, NormRect]:
        """The rects that are set, keyed by region, in ``REGION_KEYS`` order; every
        region is included only when drawn."""
        regions: dict[str, NormRect] = {}
        if self.player1 is not None:
            regions[REGION_PLAYER1] = self.player1
        if self.player2 is not None:
            regions[REGION_PLAYER2] = self.player2
        if self.round_label is not None:
            regions[REGION_ROUND] = self.round_label
        if self.best_of is not None:
            regions[REGION_BEST_OF] = self.best_of
        if self.timer is not None:
            regions[REGION_TIMER] = self.timer
        if self.ingame_p1 is not None:
            regions[REGION_INGAME_P1] = self.ingame_p1
        if self.ingame_p2 is not None:
            regions[REGION_INGAME_P2] = self.ingame_p2
        return regions

    @property
    def draws_player_boxes(self) -> bool:
        """True when the gamertag boxes are drawn: the run then votes the tournament tags
        and sponsors from those boxes (``tags.attach_overlay_regions``) instead of the
        built-in position heuristics, and the top band's tag reading has no consumer
        (``overlay_bands.band_ocr_plan``)."""
        return self.player1 is not None and self.player2 is not None

    @property
    def draws_ingame_boxes(self) -> bool:
        """True when at least one nameplate box is drawn: the run then reads the drawn
        side(s) at their boxes and never searches the band for a mirrored pair, so the
        in-game nameplate band is left un-OCR'd entirely (overlay_bands.band_ocr_plan)."""
        return self.ingame_p1 is not None or self.ingame_p2 is not None

    @property
    def reads_ingame_boxes(self) -> bool:
        """True when the drawn nameplate boxes are actually read: at least one is drawn and
        the band is read as something (``ingame_band`` isn't ``INGAME_BAND_NONE``). The editor
        keeps drawn boxes when the band is switched to none, so a layout can carry boxes
        nothing reads; the pipeline then skips them rather than OCR two unused boxes per
        sample (``overlay_regions.layout_overlay_regions``), and the vote ignores them
        (``tags.nameplates_from_boxes``). The band mode is the one a run resolves
        (``ingame_band_mode_for``): a mode the game does not offer reads as none."""
        game = game_by_id(self.game)
        band = ingame_band_mode_for(game, self) if game is not None else self.ingame_band
        return band != INGAME_BAND_NONE and self.draws_ingame_boxes


def _rect_to_list(rect: NormRect) -> list[float]:
    """A NormRect as a JSON-serializable ``[x1, y1, x2, y2]`` list of floats."""
    return [rect.x1, rect.y1, rect.x2, rect.y2]


def _rect_from_list(value: object) -> NormRect:
    """Inverse of ``_rect_to_list``; raises ``ValueError`` (via NormRect's own
    validation, or directly) if ``value`` isn't a 4-number list."""
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"expected a 4-number [x1, y1, x2, y2] list; got {value!r}")
    x1, y1, x2, y2 = (float(v) for v in value)
    return NormRect(x1=x1, y1=y1, x2=x2, y2=y2)


def layout_to_dict(layout: CustomOverlayLayout) -> dict[str, Any]:
    """Serialize one layout to its JSON shape (see ``load_layouts`` for the shape the
    store wraps this in). Optional regions serialize to ``None`` when unset."""
    return {
        "name": layout.name,
        "game": layout.game,
        "ingame_band": layout.ingame_band,
        "player1": _rect_to_list(layout.player1) if layout.player1 is not None else None,
        "player2": _rect_to_list(layout.player2) if layout.player2 is not None else None,
        "round": _rect_to_list(layout.round_label) if layout.round_label is not None else None,
        "best_of": _rect_to_list(layout.best_of) if layout.best_of is not None else None,
        "timer": _rect_to_list(layout.timer) if layout.timer is not None else None,
        "ingame_p1": _rect_to_list(layout.ingame_p1) if layout.ingame_p1 is not None else None,
        "ingame_p2": _rect_to_list(layout.ingame_p2) if layout.ingame_p2 is not None else None,
        "clock_seconds": layout.clock_seconds,
    }


def layout_from_dict(data: dict[str, Any]) -> CustomOverlayLayout:
    """Inverse of ``layout_to_dict``. Raises ``ValueError`` on any malformed field:
    missing keys, wrong types, an invalid rect, a rejected name (empty or the reserved
    ``AUTO_LAYOUT_NAME``), an unknown game, an unknown ``ingame_band`` value, or a missing
    timer box for a game that requires one. Every key is required: a version-4 store
    always writes all of them (an older, partial entry never reaches here, since
    ``load_layouts`` rejects the whole store on its version)."""
    required_keys = {"name", "game", "ingame_band", "player1", "player2", "round", "best_of",
                     "timer", "ingame_p1", "ingame_p2", "clock_seconds"}
    if missing := required_keys - data.keys():
        raise ValueError(f"layout entry is missing key(s): {sorted(missing)}")
    name = data["name"]
    if not isinstance(name, str):
        raise ValueError(f"layout name must be a string; got {name!r}")
    game = data["game"]
    if not isinstance(game, str):
        raise ValueError(f"layout game must be a string id; got {game!r}")
    ingame_band = data["ingame_band"]
    if not isinstance(ingame_band, str):
        raise ValueError(f"layout ingame_band must be a string; got {ingame_band!r}")
    player1_value = data["player1"]
    player2_value = data["player2"]
    round_value = data["round"]
    best_of_value = data["best_of"]
    timer_value = data["timer"]
    ingame_p1_value = data["ingame_p1"]
    ingame_p2_value = data["ingame_p2"]
    clock_value = data["clock_seconds"]
    if isinstance(clock_value, bool) or not isinstance(clock_value, (int, float)):
        raise ValueError(f"clock_seconds must be a number of seconds; got {clock_value!r}")
    return CustomOverlayLayout(
        name=name,
        game=game,
        ingame_band=ingame_band,
        player1=_rect_from_list(player1_value) if player1_value is not None else None,
        player2=_rect_from_list(player2_value) if player2_value is not None else None,
        round_label=_rect_from_list(round_value) if round_value is not None else None,
        best_of=_rect_from_list(best_of_value) if best_of_value is not None else None,
        timer=_rect_from_list(timer_value) if timer_value is not None else None,
        ingame_p1=_rect_from_list(ingame_p1_value) if ingame_p1_value is not None else None,
        ingame_p2=_rect_from_list(ingame_p2_value) if ingame_p2_value is not None else None,
        clock_seconds=int(clock_value),
    )


def ingame_band_mode_for(game: Game, layout: CustomOverlayLayout | None) -> str:
    """What a run reads the in-game nameplate band as (one of ``INGAME_BAND_MODES``): the
    layout's own choice when the run has a custom overlay layout, else the game's default.
    The no-layout case is the built-in "Auto" reading (offered by the games with a clock
    detector), so Ultimate keeps reading its in-game tags exactly as before layouts carried
    the field. A choice the game does not offer (``Game.ingame_band_choices``: a Melee
    layout saved with "tags" by an older build) reads as ``INGAME_BAND_NONE``, the one mode
    every game offers."""
    if layout is not None and layout.ingame_band in game.ingame_band_choices:
        return layout.ingame_band
    if layout is not None:
        return INGAME_BAND_NONE
    return game.ingame_band_default


def layouts_path() -> Path:
    """Where the layout store lives: ``LAYOUT_STORE_NAME`` at the root of the writable
    workspace (see ``vodparse.runtime_paths.workspace_dir``), shared across all VODs."""
    return workspace_dir() / LAYOUT_STORE_NAME


def load_layouts(path: Path) -> list[CustomOverlayLayout]:
    """Read the layout store at ``path``.

    Deliberately conservative, like ``extract_flags.read_sidecar``: a missing file, a
    file that isn't valid JSON, a JSON value that isn't the expected top-level shape, a
    version mismatch, or even one malformed entry among many all return ``[]`` rather
    than raising. Dropping the *whole* store on a single bad entry (rather than
    skipping just that entry) keeps the contract simple: callers never have to reason
    about a partially-loaded store, and a corrupt file degrades to "no presets" instead
    of crashing the GUI on startup.
    """
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    if data.get("version") != LAYOUT_STORE_VERSION:
        return []
    raw_layouts = data.get("layouts")
    if not isinstance(raw_layouts, list):
        return []
    try:
        return [layout_from_dict(entry) for entry in raw_layouts]
    except (ValueError, KeyError, TypeError):
        return []


def save_layouts(path: Path, layouts: Sequence[CustomOverlayLayout]) -> None:
    """Write ``layouts`` to ``path`` as the versioned store, atomically (see
    ``vodparse.jsonio.write_json_atomic``).

    Raises ``ValueError`` if two layouts share a name case-insensitively: the store is
    looked up by name (``layout_by_name``), so duplicates would make one preset
    permanently unreachable.
    """
    seen_names: set[str] = set()
    for layout in layouts:
        folded = layout.name.casefold()
        if folded in seen_names:
            raise ValueError(f"duplicate layout name (case-insensitive): {layout.name!r}")
        seen_names.add(folded)
    data = {
        "version": LAYOUT_STORE_VERSION,
        "layouts": [layout_to_dict(layout) for layout in layouts],
    }
    write_json_atomic(path, data)


def layouts_for_game(layouts: Sequence[CustomOverlayLayout],
                     game_id: str) -> list[CustomOverlayLayout]:
    """The presets in ``layouts`` saved for ``game_id``, in store order: what the main
    screen's chooser and the editor's preset combo list for the selected game."""
    return [layout for layout in layouts if layout.game == game_id]


def layout_by_name(layouts: Sequence[CustomOverlayLayout], name: str) -> CustomOverlayLayout | None:
    """The layout in ``layouts`` whose name matches ``name`` case-insensitively, or
    ``None`` if there isn't one."""
    folded = name.casefold()
    for layout in layouts:
        if layout.name.casefold() == folded:
            return layout
    return None


def with_layout_saved(layouts: Sequence[CustomOverlayLayout],
                      layout: CustomOverlayLayout) -> list[CustomOverlayLayout]:
    """``layouts`` with ``layout`` saved into it: a preset of the same name (matched
    case-insensitively, as names are unique across the store) is replaced where it stands,
    and a new name goes at the end. Keeping the place matters because the store's order is
    the order the layout lists show: saving an edit used to move the preset to the bottom."""
    if layout_by_name(layouts, layout.name) is None:
        return [*layouts, layout]
    folded = layout.name.casefold()
    return [layout if existing.name.casefold() == folded else existing for existing in layouts]
