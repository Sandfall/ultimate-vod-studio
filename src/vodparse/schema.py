"""The JSON data shapes shared across the pipeline, as TypedDicts.

Everything between stages is plain dicts serialized to JSON; these definitions give the
dicts names and checkable structure without changing anything at runtime (a TypedDict is
a regular dict: `json.dump`/`json.load` work untouched). This module is the
authority on the file layout and fields.

Conventions:
  * Timestamps are absolute source-video seconds.
  * YOLO boxes are `[x1, y1, x2, y2]` in source-frame pixels (`BBox`); OCR text boxes
    are four corner points `[[x, y], ...]` (`QuadBBox`).
  * Optional keys are declared with `typing.NotRequired`. (Some shapes still use the
    older private base-class + `total=False` pattern; new optional keys should prefer
    `NotRequired`.)

The Review screen's draft format (review_draft.json) is GUI-internal and lives with its
serializers in vodparse.gui.review.draft (`clip_to_draft` / `match_to_draft`).
"""

from typing import Any, NotRequired, TypedDict

BBox = list[int]            # [x1, y1, x2, y2]
QuadBBox = list[list[int]]  # [[x, y], [x, y], [x, y], [x, y]]


# ── OCR / detection results ────────────────────────────────────────────────────

class TextBox(TypedDict):
    """One OCR text detection within a frame."""
    text: str
    confidence: float
    bbox: QuadBBox


class TagEntry(TextBox):
    """A gamertag box chosen by find_tournament_tags: the cleaned tag in `text`, the
    sponsor prefix split off into `sponsor` (None when the player has none)."""
    sponsor: str | None


class ParsedTime(TypedDict):
    """A match-clock reading parsed from OCR text (ocr_timer.parse_time)."""
    raw: str
    minutes: int
    seconds: int
    milliseconds: int | None
    total_seconds: float
    formatted: str


class _TimerReadingBase(ParsedTime):
    """A located + parsed clock: the parse plus where the timer box sat and the
    detector's confidence (timer.detect_timer)."""
    bbox: BBox
    confidence: float


class TimerReading(_TimerReadingBase, total=False):
    """As above; `recovered` is set True only when the reading came from the dropped-read
    fallback (timer.detect_timer_with_recovery): the clock OCR'd at a remembered box because
    the detector missed it, rather than from a real detector box. `calibrated` is set True
    when it came from the run's calibrated clock box (vodparse.clock_calibration): the spot
    the detector agreed on across the VOD, read on every sample like a drawn layout box."""
    recovered: bool
    calibrated: bool


class CharacterDetection(TypedDict):
    """One HUD portrait-tile detection (characters.detect_characters)."""
    name: str
    confidence: float
    bbox: BBox


class StockSlot(TypedDict):
    """One HUD slot as read by stock_icons.detect_stock_icons: the character whose stock icons
    fill it, how many icons (= stocks left) it shows, and the box around the whole row of
    them in source-frame pixels. Slots are listed left to right, which is controller-port
    order, so the first two are player 1 and player 2 in a 1v1."""
    name: str
    confidence: float
    bbox: BBox
    stocks: int


class SideCharacters(TypedDict):
    """The two in-match HUD tiles keyed by screen side. Left is always player 1 and
    right player 2 (fixed by controller port); a side is None when its tile wasn't
    detected in this frame."""
    left: CharacterDetection | None
    right: CharacterDetection | None


# ── Markers and pipeline events ────────────────────────────────────────────────

class _MarkerBase(TypedDict):
    timestamp_sec: float
    frame_number: int


class Marker(_MarkerBase, total=False):
    """A match start/end boundary. Detected markers are full Events (Event inherits
    from Marker, so every Event is a Marker); calculated ones carry `synthesized: True`,
    which tells the clip exporter to add no end padding (a calculated end already sits
    END_LEAD_SEC before the next game). The Review screen's split/insert actions create
    synthesized markers too, deriving frame_number from the clip's frame rate."""
    synthesized: bool


class _EventBase(Marker):
    label: str | None   # "go" | "game" | None (None = periodic OCR frame)


class LayoutRead(TypedDict):
    """What one overlay text box read on one frame (vodparse.layout_reads.read_region): the
    text (empty when nothing was read) and the best single read's confidence. A player box
    (a drawn gamertag box, or a row the side-bar pre-pass locked) reads its row through
    tags.read_player_row, so its `text` is the gamertag alone and `sponsor` carries the
    sponsor split off the row (None when the player has none); other regions join their
    words as read and carry no `sponsor` key. Runs older than the key have none: their
    votes fall back to splitting `text`."""
    text: str
    confidence: float
    sponsor: NotRequired[str | None]


class Event(_EventBase, total=False):
    """One pipeline.json event: a go/game splash detection or a periodic-OCR frame.
    `confidence`/`bbox` appear only on splash detections; `texts`/`timer`/`characters`
    appear on OCR'd frames (always on periodic frames; on splash frames when OCR ran)."""
    confidence: float
    bbox: BBox
    texts: list[TextBox]
    timer: TimerReading | None
    characters: SideCharacters | None
    ocr_gated: bool      # True when the --gate-ocr-on-timer toggle skipped this frame's full-frame
                         # OCR (a timer-less "dead" frame); its texts are [] but it is still kept as
                         # a clock-less sample for intermission splitting / first-game-start anchoring
    layout_reads: dict[str, LayoutRead]   # a custom overlay layout's text boxes, read per frame and
                                          # keyed by region (overlay_layout.REGION_KEYS minus the
                                          # timer, which lands in `timer`); only on runs with a layout
    stock_slots: list[StockSlot]  # every HUD slot the stock-icon detector found, left to right
                                  # (stock_icons.detect_stock_icons); only on runs with that
                                  # detector, where an empty list means the strip showed no icons.
                                  # `characters` carries the first two when exactly two were seen


class _MatchBase(TypedDict):
    duration_sec: float
    start: Marker
    end: Marker
    frames: list[Event]   # the OCR frames inside the match window, used for tag voting


class Match(_MatchBase, total=False):
    """One game, as extracted (base keys) and then enriched (the rest). `match_number`
    is assigned by extract_matches.build_matches' renumber pass after timer-reset
    splitting; the tag/round/character fields are voted on by tags.enrich_match.
    All voted lists are positional ([left/player-1, right/player-2]) and an individual
    position can be None when nothing won its vote, which is also why sponsor entries
    align with tournament_tags positions rather than packing down."""
    match_number: int
    ingame_tags: list[str | None] | None
    ingame_tags_confirmed: bool
    tournament_tags: list[str | None] | None
    tournament_sponsors: list[str | None] | None
    round: str | None
    best_of: str | None
    characters: list[str | None] | None


# ── Tournament sets ────────────────────────────────────────────────────────────

class _SetBase(TypedDict):
    set_number: int
    matches: list[Match]
    start_timestamp: float
    end_timestamp: float
    all_confirmed: bool   # every member match's ingame_tags_confirmed


class SetGroup(_SetBase, total=False):
    """Consecutive matches with the same players, grouped by grouping.group_into_sets.
    The voted fields are optional only because the grouping loop builds sets
    incrementally and votes them in a final pass: every *returned* set has all of
    them. `grouping_tags` is whichever tag source defined the set's identity under the
    active --group-by mode; `characters` entries may join multiple games' characters
    ("Char1 / Char2") per --set-characters."""
    tournament_tags: list[str | None] | None
    tournament_sponsors: list[str | None] | None
    ingame_tags: list[str | None] | None
    grouping_tags: list[str | None] | None
    characters: list[str | None] | None


# ── Reviewed export (reviewed.json) ────────────────────────────────────────────

class Clip(TypedDict):
    """One human-reviewed clip: absolute boundaries (used verbatim: no buffer math)
    plus the corrected fields. `grouping_tags` are the editable identity tags that
    titles use; the tournament/ingame tag lists are the as-detected posterity columns
    for the CSV (positions may be None where no vote won, same as Match)."""
    clip_number: int
    match_numbers: list[int]
    start_sec: float
    end_sec: float
    grouping_tags: list[str | None]
    tournament_tags: list[str | None]
    tournament_sponsors: list[str | None]
    ingame_tags: list[str | None]
    characters: list[str]
    round: str
    best_of: str
    # The first game's start in the VOD, for the CSV's first_game_offset (the upload
    # assistant's thumbnail). Absent in reviewed.json files written before 1.0.0.
    first_game_sec: NotRequired[float]


class _ReviewedBase(TypedDict):
    video_path: str
    stem: str
    mode: str             # "match" | "tournament_set"
    tournament_name: str
    group_by: str         # "auto" | "overlay" | "ingame"
    clips: list[Clip]


class ReviewedData(_ReviewedBase, total=False):
    """reviewed.json: what the GUI Review screen hands to vodparse.cli.export_reviewed. The
    optional keys are export preferences the GUI injects from Settings; the exporter
    falls back to its defaults when they're absent (standalone/CLI-written files)."""
    title_char_limit: int
    title_filenames: bool
    use_sponsors: bool
    export_folder: str   # written back by export_reviewed: the video_out/ subfolder clips landed in
    export_csv: str      # written back by export_reviewed: the metadata CSV's filename in that folder


# ── Pipeline output (pipeline.json) ────────────────────────────────────────────

class InferenceSettings(TypedDict):
    logo_conf: float
    infer_size: int
    every_n: int
    text_conf: float
    use_ocr: bool
    ocr_interval_sec: float
    ocr_beamsearch: bool
    ocr_canvas: int           # CRAFT detection-input cap (px); 2560 = unscaled at 1080p
    gate_ocr_on_timer: bool   # skip full-frame OCR on timer-less frames
    ocr_roi_cache: bool       # recognizer-only reads of calibrated text regions (ocr_roi.py)
    ocr_lang: str             # "en" or a non-English code paired with English (ocr_langs.py)
    ocr_engine: str           # overlay-band engine, "easyocr" | "doctr" (ocr_engine.py)
    ingame_band_ocr_engine: NotRequired[str]   # the engine that read the BOTTOM band when the
                                               # game's HUD pins one (games.IngameHud.ocr_engine);
                                               # absent or equal to ocr_engine = the overlay engine
                                               # read it, or it was not read (band_ocr_bottom)
    ingame_tag_band_frac: float   # top edge of the bottom OCR band = the game's in-game nameplate
                                  # band (games.ingame_band_top_for); for Ultimate only, <0.92 means
                                  # the widening knob for shrunk captures. Read back at extract
    game: NotRequired[str]        # the game the run was analyzed as (a vodparse.games id);
                                  # absent = a run predating the setting, i.e. Ultimate. Read
                                  # back at extract for the game's in-game nameplate profile
    # The detector weights the run loaded, or None for a game with no detectors of its own.
    timer_model_path: str | None
    character_model_path: str | None
    stock_model_path: NotRequired[str | None]   # the stock-icon detector (games.GameWeights.
                                                # stock_icons); absent = a run predating it
    seek_every: float     # 0 in go/game mode
    # Which of the two overlay OCR bands the run read (overlay_bands.band_ocr_plan): False when
    # nothing consumed that band (a custom layout's own boxes covered its readers, or the in-game
    # band is read as nothing), so it was never OCR'd. Absent = an older run, which always read
    # both bands. Extract reads band_ocr_bottom back (extract_matches.bottom_band_was_read).
    band_ocr_top: NotRequired[bool]
    band_ocr_bottom: NotRequired[bool]
    overlay_regions: NotRequired[dict[str, Any]]   # the overlay text regions read per sample
                                                   # (overlay_regions.OverlayRegions.as_config):
                                                   # a layout's drawn boxes or the side-bar
                                                   # lock's rows; absent = no regions were
                                                   # read (built-in bands only), or a run
                                                   # older than the key, whose layout record
                                                   # below stands in for it at extract
    overlay_layout: NotRequired[dict[str, Any]]   # the custom overlay layout the run read with, as
                                                  # overlay_layout.layout_to_dict (name + boxes);
                                                  # absent = the built-in overlay reading. Read
                                                  # back at extract so both stages use one layout


class PipelineConfig(TypedDict):
    """The run parameters vodparse.cli.pipeline records alongside its events: extract_matches
    reads mode/seek cadence and the frame dimensions from here, and the GUI compares
    fps/frame counts against the video on disk to detect a swapped file."""
    mode: str             # "seek" | "go/game"
    vod_filename: str
    video_path: str
    fps: float
    total_frames: int
    duration_sec: float
    frame_width: int
    frame_height: int
    elapsed_sec: NotRequired[float]   # wall time of the whole analysis run, for timing one run
                                      # against another (GUI runs keep no console log); absent
                                      # = an older run
    clock_calibration: NotRequired[dict[str, Any] | None]
    """The calibrated clock box the run read on every sample
    (vodparse.clock_calibration.ClockCalibration.as_config), or None when the pre-pass found
    no agreeing spot (or did not run: a drawn timer box, no clock detector, --no-calibrate-
    clock); absent = an older run."""
    side_overlay: NotRequired[dict[str, Any] | None]
    """What the side-bar pre-pass concluded (vodparse.side_overlay.SideOverlayScan.as_config:
    its status, the feed span, every row it weighed and the boxes it locked), or None when it
    did not run (a game without side bars, a custom layout, --no-locate-overlay, no OCR);
    absent = an older run. The locked boxes themselves are the run's
    inference_settings.overlay_regions."""
    inference_settings: InferenceSettings


class PipelineData(TypedDict):
    """The whole pipeline.json payload."""
    config: PipelineConfig
    events: list[Event]


class PipelineProgress(TypedDict):
    """The transient progress sidecar vodparse.cli.pipeline rewrites throughout a run (and deletes
    nothing: the GUI owns cleanup). The GUI polls it to drive the run's progress bar:
    timestamp_sec is how far into the video the main loop has reached. `stage` names a
    phase before the main loop that the bar cannot express as a position: "calibrating"
    while the clock-box pre-pass runs (vodparse.clock_calibration); absent once the loop
    is under way, and on older pipelines. `clock_calibration` rides on every main-loop
    tick so the GUI can show the outcome for the whole run: "found" (the box is being read
    every sample), "not_found" (no agreeing spot: the detector reads each frame) or
    "skipped" (a drawn timer box, no clock detector, or --no-calibrate-clock)."""
    timestamp_sec: float
    duration_sec: float
    stage: NotRequired[str]
    clock_calibration: NotRequired[str]
    side_overlay: NotRequired[str]   # the side-bar pre-pass's outcome, one of
                                     # side_overlay.SIDE_OVERLAY_STATUSES ("locked", "top_band",
                                     # "not_found", "skipped"); rides on every main-loop tick like
                                     # clock_calibration; stage "locating_overlay" while it runs


# The values PipelineProgress carries: the pipeline writes them, the GUI's progress pane reads
# them, and both import them from here (the GUI must not import vodparse.cli.pipeline).
PROGRESS_STAGE_CALIBRATING = "calibrating"
"""PipelineProgress.stage while the clock-box pre-pass runs, before the main loop."""

PROGRESS_STAGE_LOCATING_OVERLAY = "locating_overlay"
"""PipelineProgress.stage while the side-bar pre-pass runs (vodparse.side_overlay), after the
clock-box pre-pass and before the main loop. Its outcome rides on every main-loop tick as
PipelineProgress.side_overlay (one of side_overlay.SIDE_OVERLAY_STATUSES)."""

CLOCK_CALIBRATION_FOUND = "found"
CLOCK_CALIBRATION_NOT_FOUND = "not_found"
CLOCK_CALIBRATION_SKIPPED = "skipped"
"""PipelineProgress.clock_calibration values, one per main-loop tick."""

# The pipeline's --ingame-band-engine choices: which engine reads a game's in-game nameplate
# band when the game pins one of its own (games.ingame_band_engine_for; Rivals 2 pins docTR).
# "pinned" loads the game's engine; "overlay" reads the band with --ocr-engine instead, which
# is what the GUI asks for when the user declines the pinned engine's model download. Here,
# not in vodparse.cli.pipeline, because the GUI builds the argv and must not import that.
INGAME_BAND_ENGINE_PINNED = "pinned"
INGAME_BAND_ENGINE_OVERLAY = "overlay"


# ── Diagnostics bundle (manifest.json inside diagnostics_*.zip) ─────────────────

class ManifestFile(TypedDict):
    """One bundled artifact, as recorded in the diagnostics manifest. `name` is the
    entry's path inside the zip (always relative, forward-slashed); the verifier
    recomputes `sha256`/`size` from the actual bytes and rejects any mismatch."""
    name: str
    size: int
    sha256: str


class DiagnosticsManifest(TypedDict):
    """manifest.json at the root of a diagnostics_*.zip. `format`/`format_version` let the
    verifier reject anything not claiming to be one of our bundles; `files` is the source
    of truth the verifier checks every other entry against (exact set, sizes, hashes).
    Everything here is inert text: no secrets, paths reduced to basenames before export."""
    format: str           # always "vodparse-diagnostics"
    format_version: int
    app_version: str
    python_version: str
    qt_version: str
    platform: str         # platform.platform(): carries no username
    exported_at: str      # ISO-8601 UTC
    stem: str
    settings: dict[str, str]
    files: list[ManifestFile]
