"""
Supported games
---------------
The registry behind the main screen's "Game:" dropdown, and the per-game **profile** the
pipeline reads: which detection weights ship for the game, how long its match clock runs
at game start, and where its in-game HUD draws the player nameplates. The ``id`` is what
QSettings stores (``SETTING_GAME``) and what ``--game`` carries into the pipeline JSON; the
``display_name`` is the dropdown label.

What is *stream* and what is *game* (the split this module is built on): the tournament
overlay (gamertags, round, best-of in the outer bands) is composited by the stream and is
the same shape for any game, so its heuristics stay in ``vodparse.tags`` and a custom
overlay layout (``vodparse.overlay_layout``) can replace them for any game. The things that
belong to the game itself live here: the detectors trained on its graphics, its clock, and
its in-game HUD geometry.

The in-game HUD band has two independent properties. Its **position** (where the game draws
its nameplates: ``IngameHud``) is a fact about the game and lives here. Its **meaning** (what
the text in that band is: the players' in-game tags, as in Ultimate; the characters' names,
as in Rivals 2; or nothing worth reading) is chosen per custom overlay layout
(``vodparse.overlay_layout.CustomOverlayLayout.ingame_band``), with the game supplying the
default (``Game.ingame_band_default``, one of the ``INGAME_BAND_*`` constants).

A game without a clock detector ("Other" by design) cannot split a VOD on its own: the
built-in "Auto" overlay reading is not offered for it, and a run needs a custom overlay
layout with a timer box, whose clock length drives splitting (``Game.has_timer_model``).
Where each game draws its clock is declared up front (``Game.clock_hud``), so the harvest
and evaluation tools can use it before the weights exist. Characters come from whichever
read the game supports: a portrait-tile detector (Ultimate), the nameplate band read as
names (Rivals 2), or a stock-icon detector with its row geometry (Melee, ``StockHud``).

Qt-free and torch-free on purpose (it imports only ``vodparse.runtime_paths``), so the GUI,
the pipeline, the build spec and the weight downloader can all import it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from vodparse.ocr_engine import OCR_ENGINE_DOCTR
from vodparse.runtime_paths import bundle_dir


@dataclass(frozen=True)
class GameWeights:
    """The detection weights a game ships, as paths relative to the bundle root (see
    ``weight_path``). Flat names under ``weights/``, because GitHub release assets are flat
    and the downloader copies them by name: Ultimate's three keep their historic names, and
    a later game's files carry a tag for the game (``weights/rivals2_timer.pt``; Melee's tag
    is the suffix ``_dx``, as in ``weights/timer_dx.pt``)."""

    timer: str | None
    """The clock localizer, or None while the game has no clock detector of its own. It is
    what makes a game splittable without a drawn timer box (``Game.has_timer_model``); a game
    may ship other detectors before this one exists."""
    character: str | None = None
    """The character-portrait detector (the game's roster lives in the weight's own class
    metadata, never in source), or None when the game has no portrait detector."""
    m_char: str | None = None
    """An optional fine-grained classifier cascaded after the character detector."""
    hud_strip_top: float | None = None
    """Top edge of the game's bottom HUD strip, as a fraction of frame height: the
    character detector only accepts portrait tiles below this line. Set together with
    ``character`` (it is that detector's gate) and None without it."""
    timer_infer_size: int | None = None
    """The long-side size (pixels) the frame is scaled to for the clock detector, or None
    to use the run's general ``--size`` (320 for every GUI run). A localizer trained at
    640 on a clock ~100 px wide at 1080p sees a 17 px smudge at 320 and loses it on any
    stream that is not the one it trained on; declaring its training size here keeps the
    detector, the eval tools and the pipeline all looking at the same pixels."""
    stock_icons: str | None = None
    """The stock-icon detector (``vodparse.stock_icons``): names the characters of a game
    that shows no readable character text and no portrait tiles, from the small per-character
    icons its HUD draws for each remaining stock (Melee). A game has either this or
    ``character``, never both: they are two ways of filling the same character read. The
    roster lives in the weight's own class metadata, never in source."""
    stock_hud: StockHud | None = None
    """Where that game draws its stock-icon row and how the reader crops it (see
    ``StockHud``). Set together with ``stock_icons`` (it is that detector's gate) and None
    without it."""

    def __post_init__(self) -> None:
        if self.timer is None and self.character is None and self.stock_icons is None:
            raise ValueError("GameWeights declares no weight at all; a game with no detectors "
                             "uses Game.weights=None instead")
        if self.character is not None and self.stock_icons is not None:
            raise ValueError("GameWeights.character and GameWeights.stock_icons are two ways "
                             "of reading the same characters; a game declares one of them")
        if (self.stock_icons is None) != (self.stock_hud is None):
            raise ValueError("GameWeights.stock_hud is the stock-icon detector's gate: set "
                             "both stock_icons and stock_hud, or neither")
        if self.timer_infer_size is not None and (self.timer is None or self.timer_infer_size <= 0):
            raise ValueError("GameWeights.timer_infer_size is the clock detector's inference "
                             "size: a positive number, set only with timer")
        if self.m_char is not None and self.character is None:
            raise ValueError("GameWeights.m_char cascades after the character detector, so it "
                             "needs GameWeights.character")
        if (self.character is None) != (self.hud_strip_top is None):
            raise ValueError("GameWeights.hud_strip_top is the character detector's gate: set "
                             "both character and hud_strip_top, or neither")


# The long side, in pixels, every frame is scaled to before a YOLO detector runs: the CLI
# --size default and what every GUI run uses (the GUI passes no --size). A game may pin a
# different size for its clock detector alone (GameWeights.timer_infer_size); the eval tools
# fall back to this value so they look at the same pixels as the pipeline.
DEFAULT_INFER_SIZE = 320


def timer_infer_size_for(game: Game) -> int:
    """The inference size the clock detector runs at for `game`: its own declared
    ``GameWeights.timer_infer_size`` when it has one, else the general ``DEFAULT_INFER_SIZE``
    every other detector uses. A game with no weights at all (``Game.weights is None``) has
    no clock detector either, so it also falls back to ``DEFAULT_INFER_SIZE``; callers that
    reach this path only do so when a detector is actually loaded."""
    if game.weights is not None and game.weights.timer_infer_size is not None:
        return game.weights.timer_infer_size
    return DEFAULT_INFER_SIZE


@dataclass(frozen=True)
class IngameHud:
    """Where the game draws its own in-game player nameplates (the band
    ``vodparse.tags.find_player_names`` reads, left and right by screen side, even under a
    custom overlay layout). A vertical span in fractions of frame height, measured on a
    full-frame capture, so the record can say *where* the band is and not only how tall:
    Ultimate and Rivals 2 draw the names in a strip along the bottom edge
    (``name_band_bottom == 1.0``); a game that draws them in the top corners (Street
    Fighter 6) would declare a span near 0.0 instead. Only bottom-anchored spans are read
    today (see ``ingame_band_top_for``)."""

    name_band_top: float
    """Top edge of the nameplate band: nameplates sit at or below this line."""
    name_band_bottom: float
    """Bottom edge of the band; 1.0 for a strip that runs to the bottom of the frame."""
    name_band_floor: float
    """The lowest ``name_band_top`` may be widened to (the Settings /
    ``--ingame-tag-band-frac`` knob, for captures shrunk toward center); below it other HUD
    text enters the band. Equal to ``name_band_top`` when the game allows no widening."""
    ocr_engine: str | None = None
    """The overlay OCR engine that reads this band (a ``vodparse.ocr_engine`` id), or None
    to read it with the run's own engine (the user's Font style). A game's HUD font is a
    fact about the game, so it may pin the engine that reads it correctly: Rivals 2's
    condensed names misread systematically under EasyOCR and cleanly under docTR
    (measured 2026-09-04), so its band is always read with docTR while the stream overlay
    keeps the user's choice. The pipeline loads a second reader when the two differ."""

    def __post_init__(self) -> None:
        if not (0.0 <= self.name_band_top < self.name_band_bottom <= 1.0):
            raise ValueError("IngameHud needs 0 <= name_band_top < name_band_bottom <= 1; "
                             f"got {self.name_band_top} .. {self.name_band_bottom}")
        if not (0.0 <= self.name_band_floor <= self.name_band_top):
            raise ValueError("IngameHud.name_band_floor must lie at or above 0 and at or "
                             f"below name_band_top; got {self.name_band_floor}")

    @property
    def is_bottom_anchored(self) -> bool:
        """True when the band runs to the bottom edge of the frame: the only shape the
        bottom-band OCR crop and ``find_player_names`` read today."""
        return self.name_band_bottom == 1.0


@dataclass(frozen=True)
class FeedPlacement:
    """How to place a game feed that is narrower than the stream (a game not drawn 16:9,
    with the stream's side bars either side of it). The feed fills the frame from top to
    bottom, so the frame's height gives its width; the clock is part of the game picture,
    so the detector's clock box gives where the feed sits, wherever the stream put it.
    The clock box's SIZE is deliberately not used: it wobbles by a few pixels from stream
    to stream, and every pixel of it moved the feed's edges by many."""

    width_per_frame_height: float
    """The feed's width in multiples of the frame's height (the feed runs the frame's full
    height)."""
    clock_center_frac: float
    """Where the clock box's centre sits across the feed, 0 at the feed's left edge and 1
    at its right. Not always 0.5: a clock with small digits trailing its main ones is
    boxed off-centre."""

    def __post_init__(self) -> None:
        if self.width_per_frame_height <= 0:
            raise ValueError("FeedPlacement.width_per_frame_height must be positive; "
                             f"got {self.width_per_frame_height}")
        if not (0.0 < self.clock_center_frac < 1.0):
            raise ValueError("FeedPlacement.clock_center_frac must lie in (0, 1); "
                             f"got {self.clock_center_frac}")


@dataclass(frozen=True)
class ClockHud:
    """Where the game draws its match clock, and how its digits come back from the
    recognizer: the facts the clock detector path (``vodparse.timer.detect_timer``) needs
    per game. The span is where the detector's box *center* must fall, in fractions of the
    frame; a detection outside it is discarded, because post-game results screens and stream
    scoreboards carry clock-like digits elsewhere on the frame. The span is generous on
    purpose: a stream may place the game feed off-center (two Melee streams put the clock
    center at x 0.33 and x 0.48), so it covers every placement seen, not one spot."""

    center_x_min: float
    """Left bound of the box-center span, fraction of frame width."""
    center_x_max: float
    """Right bound of the box-center span, fraction of frame width."""
    center_y_max: float
    """Lowest the box center may sit, fraction of frame height (the clock is always in the
    top part of the frame)."""
    dot_as_colon: bool
    """True when the recognizer returns this clock's colon as "." often enough that "7.28"
    must be read as 7:28 (``vodparse.ocr_timer.parse_time(dot_as_colon=)``). False for
    Ultimate, whose "." is the centiseconds separator ("6:15.83")."""
    unseparated_centiseconds: bool = False
    """True when the game draws its centiseconds as smaller digits after a space ("07:46 84"),
    which the recognizer joins to "07:4684": ``parse_time`` may then take exactly two trailing
    digits as centiseconds. False for Ultimate, whose centiseconds follow a "."."""
    feed_placement: FeedPlacement | None = None
    """How the game feed is placed from the clock box, or None when the game fills the
    frame (16:9, no side bars). A narrower game dropped onto a 16:9 stream carries its
    clock with it, so the clock box says where the feed is, and everything outside that
    span belongs to the stream's side bars, not the game (see ``Game.has_side_bars``)."""

    def __post_init__(self) -> None:
        if not (0.0 <= self.center_x_min < self.center_x_max <= 1.0):
            raise ValueError("ClockHud needs 0 <= center_x_min < center_x_max <= 1; "
                             f"got {self.center_x_min} .. {self.center_x_max}")
        if not (0.0 < self.center_y_max <= 1.0):
            raise ValueError(f"ClockHud.center_y_max must lie in (0, 1]; got {self.center_y_max}")

    def contains(self, center_x_frac: float, center_y_frac: float) -> bool:
        """True when a detection whose center sits at these frame fractions is a candidate
        clock: inside the horizontal span and above ``center_y_max``."""
        return (self.center_x_min <= center_x_frac <= self.center_x_max
                and center_y_frac <= self.center_y_max)


@dataclass(frozen=True)
class StockHud:
    """Where a game draws its **stock icons** (Melee: one small character-head sprite per
    remaining stock, in a row above each player's damage %), and how the stock-icon reader
    (``vodparse.stock_icons``) crops the frame for them. Everything here is a fraction of the
    frame, measured on full-frame 1080p captures whose game feed reached the bottom edge.

    Only the VERTICAL geometry is declared. The icons' horizontal position is not a fact
    about the game: the HUD spreads four slots across the game feed and a player's icons sit
    in the slot of their controller PORT (1v1s are seen on ports 1+4, 1+2, 3+4), and streams
    place the feed itself differently (centred between two player cams, pushed to the left
    two thirds). The reader therefore clusters the icons it finds into slots by their gaps
    and orders the slots left to right, never by a stored x."""

    strip_top: float
    """Top edge of the bottom strip the reader crops and runs the detector on, as a fraction
    of frame height. Sits in the empty band above the icon row so a stream that lifts the
    feed a little still keeps the row inside the crop."""
    row_center_min: float
    """Highest an icon's center may sit, fraction of frame height. With ``row_center_max``
    this is the row gate: a stream's own character badges (one stream draws one next to each
    player's flags at 0.72 h, just above the in-game row) and look-alikes higher on the frame
    are rejected on position, the way ``ClockHud`` rejects results-screen digits."""
    row_center_max: float
    """Lowest an icon's center may sit, fraction of frame height (the damage % digits start
    just below the row)."""
    infer_width: int
    """The width, in pixels, the cropped strip is resized to before the detector runs. The
    icons are ~45 px on a 1080p frame: at the general 320 or 640 inference size they shrink
    to a smudge, so the strip alone is run at this width (~30 px icons at 1280)."""
    slot_gap_icon_widths: float
    """Two icons whose horizontal gap exceeds this many median icon widths belong to
    different slots. Adjacent icons of one player's row are ~0.2 widths apart; the nearest
    two slots are ~3 widths apart, so the threshold sits between them."""
    max_icons_per_slot: int
    """The most stocks a slot can show (4 in a tournament Melee game): a cluster with more
    icons than this is a detection glitch and is capped, not trusted."""

    def __post_init__(self) -> None:
        if not (0.0 < self.strip_top <= self.row_center_min < self.row_center_max <= 1.0):
            raise ValueError("StockHud needs 0 < strip_top <= row_center_min < row_center_max "
                             f"<= 1; got {self.strip_top}, {self.row_center_min}, "
                             f"{self.row_center_max}")
        if self.infer_width <= 0:
            raise ValueError(f"StockHud.infer_width must be positive; got {self.infer_width}")
        if self.slot_gap_icon_widths <= 0:
            raise ValueError("StockHud.slot_gap_icon_widths must be positive; got "
                             f"{self.slot_gap_icon_widths}")
        if self.max_icons_per_slot < 1:
            raise ValueError("StockHud.max_icons_per_slot must be at least 1; got "
                             f"{self.max_icons_per_slot}")

    def row_contains(self, center_y_frac: float) -> bool:
        """True when an icon whose center sits at this fraction of frame height is on the
        stock row: inside the declared vertical band."""
        return self.row_center_min <= center_y_frac <= self.row_center_max


# ── What the in-game band means ───────────────────────────────────────────────────────
# The three things the text in a game's nameplate band can be read as. A custom overlay
# layout stores one of these (``CustomOverlayLayout.ingame_band``); ``Game.ingame_band_default``
# is what the editor preselects for a new layout of that game and what the pipeline uses under
# the built-in "Auto" reading. The strings are what the layout store writes.
INGAME_BAND_NONE = "none"
"""Read nothing from the band: the game draws no nameplates there, or none worth reading
(Melee's tags float over the fighters). Grouping uses the stream overlay's tags alone."""
INGAME_BAND_TAGS = "tags"
"""The band holds the players' own in-game tags (Ultimate): voted into ``ingame_tags``, the
second identity the set grouping can use."""
INGAME_BAND_CHARACTER = "character"
"""The band holds the characters' names (Rivals 2): voted into ``characters`` in
place of a portrait detector; ``ingame_tags`` stays unread."""
INGAME_BAND_MODES: tuple[str, ...] = (INGAME_BAND_NONE, INGAME_BAND_TAGS, INGAME_BAND_CHARACTER)


# The highest game count a set shows beside a gamertag: a best-of-5 is won at 3. A digit at
# the end of a tag above it is the name's own ("Agent 9"), never a count (see
# grouping._game_count_map). Confirmed for Ultimate, Melee and Rivals 2 (user, 2026-10-25);
# also the value for any other game until it is known, settable per game (Game.max_game_count).
DEFAULT_MAX_GAME_COUNT = 3


@dataclass(frozen=True)
class Game:
    """One entry of the Game dropdown, with the profile the pipeline reads for it."""

    id: str
    """Stable identifier: stored in settings and the pipeline JSON, never shown."""
    display_name: str
    """The dropdown label."""
    weights: GameWeights | None
    """The game's detection weights, or None when it has no models (nothing is loaded)."""
    full_game_seconds: int | None
    """The match clock at game start, in seconds, or None when the game has no fixed
    clock of its own ("Other"): then only a layout's ``clock_seconds`` can say."""
    ingame_hud: IngameHud | None
    """The in-game nameplate geometry, or None while it is unmeasured. An unmeasured game
    whose band is read anyway (a layout asks for it) falls back to Ultimate's band (see
    ``ingame_hud_for``); Melee's tags float over the fighters, so it may stay unmeasured."""
    ingame_band_default: str
    """What the nameplate band holds for this game by default: one of ``INGAME_BAND_MODES``.
    A custom overlay layout may override it; the built-in "Auto" reading uses it as is."""
    roster: str | None = None
    """The game's character roster, as a bundle-relative path to a JSON list of names
    (see ``load_roster``), or None when the game has none to ship. Used only when the
    nameplate band is read as character names: each voted name snaps to the roster's
    spelling (``vodparse.tags.snap_to_roster``). Kept as data, not source: a roster is a
    fact about the game that changes with its patches."""
    clock_hud: ClockHud | None = None
    """Where the game draws its clock and how its digits read (see ``ClockHud``), or None
    while unmeasured. Required as soon as the game declares a clock detector: the detector
    path cannot pick a box or parse its digits without it."""
    ingame_band_choices: tuple[str, ...] = INGAME_BAND_MODES
    """The band modes a custom overlay layout may read this game's nameplate band as. Every
    mode for most games; only ``INGAME_BAND_NONE`` for a game with no band to read (Melee:
    its tags float over the fighters), so the editor offers no other choice there and a
    layout that carries one anyway is read as "none" (``overlay_layout.ingame_band_mode_for``),
    which also saves the run the bottom-band OCR pass."""
    max_game_count: int = DEFAULT_MAX_GAME_COUNT
    """The highest game count this game's sets show beside a gamertag (3 for a best-of-5).
    Extraction strips a one-digit count read into a tag ("PlayerOne 2") only up to this
    value (grouping.fold_game_counts); a higher end digit is kept as part of the name. Set
    it for a game whose sets run longer. The Review screen regroups matches.json, whose tags
    extraction already folded with this value, under the default; a game with a lower cap
    would need it passed there too."""

    def __post_init__(self) -> None:
        if self.ingame_band_default not in INGAME_BAND_MODES:
            raise ValueError(f"game {self.id!r}: ingame_band_default must be one of "
                             f"{INGAME_BAND_MODES}; got {self.ingame_band_default!r}")
        if not set(self.ingame_band_choices) <= set(INGAME_BAND_MODES):
            raise ValueError(f"game {self.id!r}: ingame_band_choices must be drawn from "
                             f"{INGAME_BAND_MODES}; got {self.ingame_band_choices!r}")
        if INGAME_BAND_NONE not in self.ingame_band_choices:
            raise ValueError(f"game {self.id!r}: ingame_band_choices must offer "
                             f"{INGAME_BAND_NONE!r}, the one mode every game can read")
        if self.ingame_band_default not in self.ingame_band_choices:
            raise ValueError(f"game {self.id!r}: ingame_band_default "
                             f"{self.ingame_band_default!r} is not among its choices "
                             f"{self.ingame_band_choices!r}")
        if self.max_game_count < 1:
            raise ValueError(f"game {self.id!r}: max_game_count must be at least 1; "
                             f"got {self.max_game_count}")
        if self.has_timer_model and self.clock_hud is None:
            raise ValueError(f"game {self.id!r} declares a clock detector but no clock_hud; "
                             "the detector path needs to know where the clock is drawn")

    @property
    def has_timer_model(self) -> bool:
        """True when the game ships a clock detector, which is what makes the built-in
        "Auto" overlay reading usable and a drawn timer box optional. Declared by the
        registry, never a check on disk, so the GUI and tests are deterministic; a
        declared weight that is missing from the bundle is refused by the pipeline unless a
        layout timer box stands in for it."""
        return self.weights is not None and self.weights.timer is not None

    @property
    def has_side_bars(self) -> bool:
        """True when this game is drawn 4:3 inside a 16:9 stream, so the stream carries side
        bars beside the game feed (Melee). Side bars are only locatable through the clock
        today (``ClockHud.feed_placement``), so a 4:3 game with no clock detector
        cannot declare them; that is acceptable because Melee has both."""
        return self.clock_hud is not None and self.clock_hud.feed_placement is not None


# ── Ultimate's profile ────────────────────────────────────────────────────────────────
# The numbers here are the one home for Ultimate's HUD geometry; vodparse.characters and
# vodparse.tags re-export them under their historic names for their many callers.

# In-match damage tiles cluster at 0.86-0.91 of frame height; 0.72 sits in the empty gap
# above them and below the mid-screen gameplay band (on-stage fighters / floating glyphs at
# ~0.60-0.66). The training-crop harvester filters at this exact line too.
ULTIMATE_HUD_STRIP_TOP = 0.72

# Ultimate renders its in-game nametags at a fixed HUD position (~0.94h at 1080p), so the
# band starts at 0.92: the same line as the stream's bottom overlay band
# (vodparse.tags.OVERLAY_BOTTOM_FRAC, a coincidence tests pin). A capture made with the
# Switch's "Adjust Screen Size" slider turned down shrinks the game image toward center and
# lifts the nametags above that line, so the band may be widened down to 0.86; below that
# it reaches the damage-counter row and overlay handle plates, which can out-vote the real
# pair. The full reasoning sits with the tunable in vodparse.tags.
ULTIMATE_INGAME_HUD = IngameHud(name_band_top=0.92, name_band_bottom=1.0, name_band_floor=0.86)

# ── Rivals 2's profile ────────────────────────────────────────────────────────────────
# Rivals 2 prints the characters' names (not the players' tags) in a strip along the bottom
# edge. Measured 2026-09-03 on a 1080p VOD at five timestamps: the name centers sit at
# 0.937h; the pipeline's box tops hugged Ultimate's 0.92 crop line (glyphs clipped,
# systematic misreads), so the band starts at 0.90. The damage counters sit at
# 0.874-0.879h and some of their misreads are not numeric, so the band may be widened no
# further than 0.88 before they enter it.
RIVALS2_INGAME_HUD = IngameHud(name_band_top=0.90, name_band_bottom=1.0, name_band_floor=0.88,
                               ocr_engine=OCR_ENGINE_DOCTR)
# The roster the band's character-name reads snap to (user-supplied 2026-09-04).
RIVALS2_ROSTER = "assets/rosters/rivals2.json"

# Ultimate draws its clock in the top-right corner. Only a detection whose center sits in the
# top-right quadrant counts: post-game results screens ("Out at 1:44") share the digit pattern
# lower on the frame and would otherwise be read as a live clock. Its "." separates the
# centiseconds ("6:15.83"), so a dot is never a colon here.
ULTIMATE_CLOCK_HUD = ClockHud(center_x_min=0.5, center_x_max=1.0, center_y_max=0.5,
                              dot_as_colon=False)

# Melee draws its clock top-center of the game feed as "07:46 84" (centiseconds smaller, after
# a space). Streams place the feed differently: measured 2026-09-05 on two 1080p tournament
# VODs, the clock center sat at x 0.48 (feed between two player cams) and x 0.33 (feed pushed
# to the left two-thirds), both near y 0.14, so the span covers both. The recognizer returned
# the colon as "." in 30% of a full-VOD run's reads, so a dot is read as a colon.
#
# Where the feed sits. Melee's picture always runs the frame's full height, and streams
# draw it at one shape: 1310 px wide on a 1080p frame (not 4:3, which would be 1440). So the
# frame's height gives the feed's width, and the clock box's centre gives where it sits.
# Measured on the clock detector's calibrated boxes of five 1080p streams (2026-10), against
# the feed edges seen on a frame:
#
#   clock box (x0, y0, x1, y1)     feed seen        feed width   clock centre at   edges derived
#   797, 108, 1163, 185            304 .. 1614      1310         0.516             305 .. 1615
#   494, 107,  856, 184              0 .. 1310      1310         0.515               0 .. 1310
#   797, 108, 1162, 186            305 .. 1615      1310         0.515             304 .. 1614
#   796, 109, 1162, 189            306 .. 1615      1309         0.514             304 .. 1614
#   785, 107, 1172, 187            277 .. 1638      1361         0.515             303 .. 1613
#
# The first four agree to within 2 px. The fifth stream stretches the picture 4% wider, so
# its derived edges sit about 25 px inside the real ones; it prints its names in the top
# band, so nothing is read from its bars. The clock box's centre is right of the feed's
# centre because the small centiseconds trail the main digits.
#
# Two earlier rules scaled the feed from the clock box's size instead and were dropped. Its
# WIDTH (3.52 to 3.62 feed widths across these streams, padded to 3.7) put the right edge 35
# to 56 px off. Its HEIGHT (17 feed widths) fit the first, second and fifth rows, but the
# detector boxes the same clock 77 px tall on one stream and 80 on another (rows one and
# four), and each pixel moved the feed's width by 17: on the fourth stream both bars came
# out 25 px too narrow and clipped the names drawn against the picture.
MELEE_FEED_WIDTH_AT_1080P = 1310
MELEE_FEED_PLACEMENT = FeedPlacement(width_per_frame_height=MELEE_FEED_WIDTH_AT_1080P / 1080,
                                     clock_center_frac=0.5155)
MELEE_CLOCK_HUD = ClockHud(center_x_min=0.20, center_x_max=0.75, center_y_max=0.30,
                           dot_as_colon=True, unseparated_centiseconds=True,
                           feed_placement=MELEE_FEED_PLACEMENT)

# Rivals 2 draws its clock top-center at a fixed 1080p spot ("6:16 36", centiseconds smaller,
# after a space); measured 2026-09-05 on three streams at center x 0.506, y 0.083 (a full-frame
# game feed puts it there every time; a zoomed or shifted feed is background by design). The span
# is kept tight around that spot on purpose: a wider one (x 0.30-0.70, y < 0.30, the first cut)
# let a stream's scoreboard digits at the span's edges, the badge under the clock and damage
# percentages outscore the clock on a synthetic-trained detector (29 of 47 misses on a second
# stream). The recognizer returned the colon as "." in 79% of a full-VOD run's reads, so a dot is
# read as a colon.
RIVALS2_CLOCK_HUD = ClockHud(center_x_min=0.40, center_x_max=0.60, center_y_max=0.13,
                             dot_as_colon=True, unseparated_centiseconds=True)

ULTIMATE_WEIGHTS = GameWeights(
    timer="weights/timer.pt",
    character="weights/character.pt",
    m_char="weights/m_char.pt",
    hud_strip_top=ULTIMATE_HUD_STRIP_TOP,
)

# Rivals 2 ships a clock detector only (trained 2026-09-05 on real frames from two stream
# captures used with permission, auto-labeled from a hand-drawn clock box).
# Its characters keep coming from the nameplate band (RIVALS2_INGAME_HUD). The clock detector
# runs at its training size, 640, not the pipeline's general 320: the model was trained at
# 640 (Ultralytics' scale augmentation shows it clocks no smaller than ~65 px wide), and at
# 320 a 1080p frame's ~130 px clock shrinks to ~22 px. On the stream it trained on that still
# works, on another stream it does not: probed on 60 frames of a second stream the clean
# checkpoint fired on 9 at 320 (2 above the 0.3 threshold) vs 58 at 640 (50 above).
RIVALS2_TIMER_INFER_SIZE = 640
RIVALS2_WEIGHTS = GameWeights(timer="weights/rivals2_timer.pt",
                              timer_infer_size=RIVALS2_TIMER_INFER_SIZE)

# Melee's stock-icon row. Measured 2026-09-06 on two permitted 1080p stream captures (one
# with the feed centred between two player cams, one with the feed pushed to the left two
# thirds) and consistent with a third stream read as reference: the icons are 35-45 px
# squares whose centers sit at 0.755-0.81 h on every stream (the feed reaches the bottom edge
# each time), the damage % digits at 0.85-0.93 h, and nothing HUD-like above 0.70 h. The
# first stream also draws a character badge in each side column at 0.72 h, which the row
# gate excludes. The strip is run at 1280 wide (icons ~30 px); the general 320 would shrink
# them to ~8 px.
MELEE_STOCK_HUD = StockHud(strip_top=0.70, row_center_min=0.74, row_center_max=0.83,
                           infer_width=1280, slot_gap_icon_widths=1.5, max_icons_per_slot=4)

# Melee ships a clock detector (trained 2026-09-05 on real frames from two stream captures
# used with permission, auto-labeled from a hand-drawn clock box per stream).
# The two streams place the game feed differently, so its training set holds the clock at
# two spots, and its honest eval is the placement it never trained on (fire 0.999,
# false-clock 0.0 at the general 320: the ~270 px clock is still ~45 px there, so no size
# pin). Its characters come from the stock-icon detector (MELEE_STOCK_HUD): Melee shows no
# character names and no portrait tiles, only the icons of the stocks each player has left.
# The files are named for what they find, with Melee's ``_dx`` tag (see GameWeights):
# ``character_dx.pt`` is the stock-icon detector, Melee's way of reading characters.
MELEE_WEIGHTS = GameWeights(timer="weights/timer_dx.pt",
                            stock_icons="weights/character_dx.pt",
                            stock_hud=MELEE_STOCK_HUD)

# ── The registry ──────────────────────────────────────────────────────────────────────

GAME_ULTIMATE = "ultimate"
GAME_MELEE = "melee"
GAME_RIVALS2 = "rivals2"
GAME_OTHER = "other"

# Dropdown order. Ultimate first: it is the default and the only game with a full set of
# detectors. "Other" last: the catch-all for any game the app has no profile for (a custom
# overlay layout with a timer box is all it needs). Melee and Rivals 2 each ship a clock
# detector (MELEE_WEIGHTS, RIVALS2_WEIGHTS). Rivals 2 also draws the characters' names in
# its bottom nameplate band (measured: RIVALS2_INGAME_HUD), so that band defaults to a
# character read. Melee's in-game tags float over the fighters, so its band stays unmeasured;
# its characters come from the stock icons instead (MELEE_STOCK_HUD).
GAMES: tuple[Game, ...] = (
    Game(GAME_ULTIMATE, "Ultimate", weights=ULTIMATE_WEIGHTS,
         full_game_seconds=7 * 60, ingame_hud=ULTIMATE_INGAME_HUD,
         ingame_band_default=INGAME_BAND_TAGS, clock_hud=ULTIMATE_CLOCK_HUD),
    Game(GAME_MELEE, "Melee", weights=MELEE_WEIGHTS,
         full_game_seconds=8 * 60, ingame_hud=None, ingame_band_default=INGAME_BAND_NONE,
         clock_hud=MELEE_CLOCK_HUD, ingame_band_choices=(INGAME_BAND_NONE,)),
    Game(GAME_RIVALS2, "Rivals 2", weights=RIVALS2_WEIGHTS,
         full_game_seconds=8 * 60, ingame_hud=RIVALS2_INGAME_HUD,
         ingame_band_default=INGAME_BAND_CHARACTER, roster=RIVALS2_ROSTER,
         clock_hud=RIVALS2_CLOCK_HUD),
    Game(GAME_OTHER, "Other", weights=None, full_game_seconds=None, ingame_hud=None,
         ingame_band_default=INGAME_BAND_NONE),
)
GAME_DEFAULT = GAME_ULTIMATE


def game_by_id(game_id: str) -> Game | None:
    """The registry entry for ``game_id``, or None when no game has that id."""
    return next((game for game in GAMES if game.id == game_id), None)


def default_game() -> Game:
    """The registry entry for ``GAME_DEFAULT`` (Ultimate): the game a run falls back to when
    nothing more specific is known. ``GAME_DEFAULT`` always names a registry entry, so this
    lookup always succeeds."""
    return next(game for game in GAMES if game.id == GAME_DEFAULT)


def full_game_seconds_for(game: Game, layout_clock_seconds: int | None) -> int:
    """The match clock length at game start, for the one place that needs to resolve it: a
    custom overlay layout's own clock wins when one is set (``layout_clock_seconds`` is not
    None), since the user drew that box for a specific clock length; otherwise the game's own
    ``full_game_seconds`` (Ultimate 420, Melee/Rivals 2 480); and only when neither is known
    (a game with no clock of its own and no layout, "Other" today) does the default game's
    clock apply. That last case is reachable only in theory: a game with no clock of its own
    must run with a layout (see ``Game.has_timer_model`` and the pipeline's guard rail), so a
    layout is always present by the time this branch would matter."""
    if layout_clock_seconds is not None:
        return layout_clock_seconds
    if game.full_game_seconds is not None:
        return game.full_game_seconds
    fallback = default_game().full_game_seconds
    assert fallback is not None, "the default game always has a full clock of its own"
    return fallback


def ingame_hud_for(game: Game) -> IngameHud:
    """The nameplate band a run reads for ``game``: its own measured record, or Ultimate's
    when the game's is unmeasured. The fallback exists for a layout that asks to read the
    band of an unmeasured game (Melee or Other with "In game tags"): Ultimate's bottom strip
    is the best guess there is, and it is exactly the band such runs read before games had
    profiles of their own."""
    if game.ingame_hud is not None:
        return game.ingame_hud
    return ULTIMATE_INGAME_HUD


def ingame_band_top_for(game: Game, ultimate_widening: float | None) -> float:
    """The top edge of the nameplate band the pipeline crops and ``find_player_names`` gates
    on, as a fraction of frame height. ``ultimate_widening`` is the ``--ingame-tag-band-frac``
    / Settings knob when the user set it: a calibration of Ultimate's HUD (its default and
    floor are Ultimate's numbers, for captures shrunk by the Switch's screen-size slider), so
    it applies only when the game is Ultimate. Every other game reads at its own measured
    band (or Ultimate's, unmeasured: see ``ingame_hud_for``). Only bottom-anchored bands can
    be read today: the crop and the gate both run from this line to the bottom of the frame,
    so a top-corner band (Street Fighter 6, one day) needs a second crop before it can be
    declared, and asking for one is an error rather than a silent misread."""
    hud = ingame_hud_for(game)
    if not hud.is_bottom_anchored:
        raise NotImplementedError(
            f"{game.display_name} declares a nameplate band that does not reach the bottom "
            "of the frame; only bottom-anchored bands are read today")
    if ultimate_widening is not None and game.id == GAME_ULTIMATE:
        return ultimate_widening
    return hud.name_band_top


def ingame_band_engine_for(game: Game, ingame_band: str) -> str | None:
    """The OCR engine the run's nameplate band must be read with, or None when the run's own
    overlay engine will do: the game's ``IngameHud.ocr_engine`` when the band is read at all
    (``ingame_band`` is not ``INGAME_BAND_NONE``, see ``overlay_layout.ingame_band_mode_for``),
    else None (nothing reads the band, so no engine is needed for it)."""
    if ingame_band == INGAME_BAND_NONE:
        return None
    return ingame_hud_for(game).ocr_engine


def resolve_saved_game(saved: str) -> str:
    """The game id the Game dropdown should preselect: the saved value, unless it is unknown
    (a settings file from a build with a game this one lacks); then the default."""
    return saved if game_by_id(saved) is not None else GAME_DEFAULT


def weight_path(relative: str) -> str:
    """A ``GameWeights`` entry resolved against the bundle root, so the weights load no
    matter the working directory: ``weights/`` at the repo root in a checkout (populate it
    from the GitHub Release), or ``sys._MEIPASS/weights/`` when frozen. A string, since
    ultralytics/torch accept path-likes either way."""
    return str(bundle_dir() / relative)


def load_roster(game: Game) -> tuple[str, ...]:
    """The character names in ``game``'s roster file, in file order, or an empty tuple when
    the game declares no roster or the file is absent from this build (a dev checkout
    without the asset still runs; extract says so). A file that is present but is not a
    JSON list of strings is a packaging error and raises ``ValueError``."""
    if game.roster is None:
        return ()
    path = bundle_dir() / game.roster
    if not path.exists():
        return ()
    names = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(names, list) or not all(isinstance(name, str) and name.strip()
                                              for name in names):
        raise ValueError(f"{path}: a roster must be a JSON list of non-empty names")
    return tuple(names)


def shipped_roster_files() -> tuple[str, ...]:
    """Every roster file the registry declares, bundle-relative, in registry order: what the
    PyInstaller spec bundles next to the weights."""
    return tuple(game.roster for game in GAMES if game.roster is not None)


def shipped_weight_files() -> tuple[str, ...]:
    """Every weight file the registry declares, bundle-relative, in registry order (timer,
    character, m_char, stock_icons per game, skipping the ones a game does not declare):
    what the PyInstaller spec bundles and what the weight downloader expects a release to
    carry."""
    files: list[str] = []
    for game in GAMES:
        if game.weights is None:
            continue
        for declared in (game.weights.timer, game.weights.character, game.weights.m_char,
                         game.weights.stock_icons):
            if declared is not None:
                files.append(declared)
    return tuple(files)
