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

from typing import TypedDict

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
    the detector missed it, rather than from a real detector box."""
    recovered: bool


class CharacterDetection(TypedDict):
    """One HUD portrait-tile detection (characters.detect_characters)."""
    name: str
    confidence: float
    bbox: BBox


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
    ingame_tag_band_frac: float   # in-game nametag OCR-band top edge; <0.92 widens it for shrunk
                                  # captures (tags.INGAME_TAG_BAND_FRAC_DEFAULT); read back at extract
    timer_model_path: str
    character_model_path: str
    seek_every: float     # 0 in go/game mode


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
    inference_settings: InferenceSettings


class PipelineData(TypedDict):
    """The whole pipeline.json payload."""
    config: PipelineConfig
    events: list[Event]


class PipelineProgress(TypedDict):
    """The transient progress sidecar vodparse.cli.pipeline rewrites throughout a run (and deletes
    nothing: the GUI owns cleanup). The GUI polls it to drive the run's progress bar:
    timestamp_sec is how far into the video the main loop has reached."""
    timestamp_sec: float
    duration_sec: float


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
