"""Persisted app-settings keys and defaults for the desktop GUI (QSettings-backed).

Shared by app.py (MainWindow reads them) and dialogs.py (SettingsDialog and the
Advanced Analysis Options popup edit them). The Review screen's own flag toggles/thresholds
(and their row tints) live in vodparse.gui.review.flags, next to the logic they parameterize;
its fallback-marker and confirm-suppression keys stay in vodparse.gui.review.model. Qt-free.
"""

from collections.abc import Mapping
from typing import Any

from vodparse.ocr_engine import FONT_STYLE_NAMES

# OCR cadence choices for the Analyze VOD page (label → seconds). The GUI always runs the pipeline
# in seek mode, so the chosen value drives both the periodic OCR interval (--ocr-interval) and the
# seek step (--seek-every), keeping the two in sync; replacing the old hard-coded 30s.
OCR_INTERVAL_CHOICES = {
    "Every 5 seconds": 5,
    "Every 10 seconds": 10,
    "Every 15 seconds": 15,
    "Every 30 seconds": 30,
    "Every 60 seconds": 60,
}
OCR_INTERVAL_DEFAULT_SEC = 30  # matches vodparse.cli.pipeline's --ocr-interval default

# UI text-size presets for the Settings → General "Text size" dropdown (label → scale factor).
# The factor multiplies the platform default font at startup (see apply_font_scale in common.py),
# so the whole UI grows together. Applied once before any widget is built, so it takes effect on the
# next launch. 1.0 is the platform default; 1.75 is the largest we've verified stays un-clipped.
FONT_SCALE_CHOICES = {
    "Normal (100%)":  1.0,
    "Large (125%)":   1.25,
    "Larger (150%)":  1.5,
    "Largest (175%)": 1.75,
}
FONT_SCALE_DEFAULT = 1.0

# OCR overlay-language choices for the Settings → Analysis tab (label → language code). The codes
# ("en"/"ja") must exactly match the keys of vodparse.ocr_langs.OCR_LANG_LABELS, the source of truth
# for what vodparse.cli.pipeline's --ocr-lang accepts; keep both dicts in sync when adding a language. The
# labels here are intentionally GUI-specific ("English (default)") rather than reused verbatim.
OCR_LANG_DEFAULT_GUI = "en"
OCR_LANG_CHOICES = {"English (default)": "en", "Japanese (experimental)": "ja"}


def lang_display_name(lang_code: str) -> str:
    """The Analysis-tab label for a language code, for naming a chosen or recorded language
    back to the user. An unknown code (a settings file or pipeline.json from a build that has
    a language this one doesn't) renders as the raw code rather than vanishing."""
    return next((label for label, code in OCR_LANG_CHOICES.items() if code == lang_code),
                lang_code)

# Overlay text-engine choices for the Analyze VOD page (label → engine id). The ids
# ("easyocr"/"doctr") must exactly match the keys of vodparse.ocr_engine.OCR_ENGINE_LABELS, the
# source of truth for what vodparse.cli.pipeline's --ocr-engine accepts; keep both dicts in sync
# when adding an engine. docTR ships with the app; in an environment without it Stylized
# stays listed and selectable (marked "not installed"), and whatever needs it stops with
# ocr_engine.engine_unusable_reason rather than reading with Standard in its place.
OCR_ENGINE_DEFAULT_GUI = "easyocr"
OCR_ENGINE_CHOICES = {name: engine for engine, name in FONT_STYLE_NAMES.items()}


def resolve_saved_engine(saved: str) -> str:
    """The engine a Font style combo should preselect: the saved value, unless it is
    unknown (a settings file from a different version), then the default. A saved Stylized
    stays Stylized even when docTR can't run here: what needs it then stops and says why
    (ocr_engine.engine_unusable_reason), never reading with Standard instead."""
    if saved not in OCR_ENGINE_CHOICES.values():
        return OCR_ENGINE_DEFAULT_GUI
    return saved


def engine_display_name(engine_id: str) -> str:
    """The Font-style label for an engine id, for naming a chosen or recorded engine back to
    the user. An unknown id (a pipeline.json from a build with an engine this one lacks)
    renders as the raw id rather than being mislabelled as the default."""
    return next((label for label, eid in OCR_ENGINE_CHOICES.items() if eid == engine_id),
                engine_id)

# Persisted app settings (QSettings keys, grouped by the GUI page they affect).
SETTING_TITLE_CHAR_LIMIT       = "export/title_char_limit"         # default 255 (OS filename max)
# 255 is the per-filename character cap shared by macOS (APFS), Windows (NTFS) and Linux (ext4): the
# real ceiling once titles name the exported clip files. (YouTube caps titles at 100; lower this if
# the titles go straight to YouTube.)
TITLE_CHAR_LIMIT_DEFAULT       = 255
SETTING_TITLE_FILENAMES_SETS    = "export/title_filenames_sets"     # default True
SETTING_TITLE_FILENAMES_MATCHES = "export/title_filenames_matches"  # default False
# No include-sponsors key: sponsors always appear in exported clip titles and on the Review
# screen. The old "export/use_sponsors" (bool) is retired and left unread in existing stores.
# No chapters-delivery key: the Review screen's button always copies to the clipboard. The old
# "export/chapters_mode" ("copy" | "save" | "both") is retired and left unread in existing stores.
SETTING_SHOW_ETA         = "analysis/show_eta"          # default True
# Which game the VOD is of (a vodparse.games id; default GAME_DEFAULT). Drives the main
# screen's Game dropdown, which overlay-layout presets the chooser beside it offers, and
# --game on the pipeline (see vodparse.gui.commands.pipeline_argv).
SETTING_GAME             = "pipeline/game"
SETTING_OCR_INTERVAL_SEC = "pipeline/ocr_interval_sec"  # default OCR_INTERVAL_DEFAULT_SEC
SETTING_GATE_OCR_ON_TIMER = "pipeline/gate_ocr_on_timer"  # default GATE_OCR_ON_TIMER_DEFAULT
SETTING_OCR_ROI_CACHE     = "pipeline/ocr_roi_cache"      # default OCR_ROI_CACHE_DEFAULT
SETTING_OCR_LANG          = "pipeline/ocr_lang"           # default OCR_LANG_DEFAULT_GUI
SETTING_OCR_ENGINE        = "pipeline/ocr_engine"         # default OCR_ENGINE_DEFAULT_GUI
SETTING_INGAME_TAG_BAND_FRAC = "pipeline/ingame_tag_band_frac"  # default INGAME_TAG_BAND_FRAC_DEFAULT (tags.py)

# The two OCR speed-ups (skip OCR on frames without a readable clock; reuse the overlay's text
# boxes between reads) are on unless the user turns them off. One constant each, read by the
# main GUI's run and Settings pane and by the VOD Grabber's: the grabber was ported while they
# were still off by default and kept running without them after the main GUI turned them on.
GATE_OCR_ON_TIMER_DEFAULT = True
OCR_ROI_CACHE_DEFAULT = True


def band_pct_from_frac(frac: float) -> int:
    """The in-game nametag band's height as a whole-number percent of frame height, for a
    Settings spinbox: the stored top-edge fraction 0.92 (bottom 8%) shows as 8, 0.86 as 14.
    Shared by the main Settings dialog and the vod grabber's, so the mapping has one home."""
    return round((1.0 - frac) * 100)


def band_frac_from_pct(pct: int) -> float:
    """Inverse of band_pct_from_frac: the spinbox's height-percent back to the stored top-edge
    fraction the pipeline/CLI take (8 % -> 0.92). Rounded to 2 dp to match --ingame-tag-band-frac."""
    return round(1.0 - pct / 100, 2)

# The selected custom-overlay-layout preset NAME (see vodparse.overlay_layout). Empty string
# means "use the built-in overlay-reading heuristics" (vodparse.overlay_layout.AUTO_LAYOUT_NAME
# is the reserved display name for that choice, never a value stored here). The presets
# themselves live in their own JSON store (overlay_layouts.json), not in QSettings.
OVERLAY_LAYOUT_DEFAULT = ""


def overlay_layout_setting_key(game_id: str) -> str:
    """The QSettings key holding the chosen overlay-layout preset for one game (a
    vodparse.games id).

    One key per game, because a preset belongs to a single game (the chooser only lists that
    game's presets): switching the Game dropdown to another game and back restores each game's
    own last choice instead of dropping it. "Reset all preferences" clears the whole settings
    object, so these keys need no special handling there."""
    return f"pipeline/overlay_layout/{game_id}"


# "Don't show this again" suppression for the warnings that are safe to silence. Every such key
# lives under the one group so the Settings "Restore confirmation prompts" button can clear them
# with a single group removal — and any suppressible warning added under it later is covered for
# free. (A key outside the group would be silently missed by that button.)
SETTING_CONFIRM_GROUP = "confirm"
SETTING_CONFIRM_REUSE_SETTINGS = "confirm/reuse_settings_differ"  # bool; True = skip the settings-differ warning
SETTING_SHOW_DEBUG_MENU  = "ui/show_debug_menu"        # default False (Debug menu hidden until enabled)
# The muted one-liner under the main screen's "Font style:" combo, describing whichever
# style is selected. On by default; the Settings → Analysis toggle is for users who already
# know the difference and don't want the reminder taking up a line.
SETTING_SHOW_FONT_STYLE_NOTE = "ui/show_font_style_note"  # default True
# Whether the Analyze page's preview shows its "Tags read on this frame" box (the button
# beside the zoom buttons toggles it; see gui.layout_preview).
SETTING_PREVIEW_SHOW_DETAILS = "ui/preview_show_details"  # default PREVIEW_SHOW_DETAILS_DEFAULT
PREVIEW_SHOW_DETAILS_DEFAULT = False   # hidden until asked for: the boxes on the frame say most of it
# Whether the Analyze page's Preview box is the layout preview (gui.layout_preview): the
# chosen video scrubbed, the regions the run reads drawn on the frame, and the clock and
# player-name searches run as soon as a video is chosen. Off restores the first release's
# Preview: the video's first frame as a still, nothing detected or read until Analyze VOD
# runs. Settings → Analysis; app.py and preview.py read it.
SETTING_SHOW_LAYOUT_PREVIEW = "ui/show_layout_preview"  # default SHOW_LAYOUT_PREVIEW_DEFAULT
SHOW_LAYOUT_PREVIEW_DEFAULT = True
# Windows only: whether choosing an .mkv brings up the "convert to MP4" offer. Windows plays
# an MKV directly, so the offer is off by default there; a user whose MKVs don't show can
# turn it on (Settings → General). Other platforms always make the offer and never show or
# read this (see vodparse.ffmpeg_runtime.offers_conversion and mkv_plays_directly).
SETTING_ASK_CONVERT_MKV = "ui/ask_convert_mkv"  # default ASK_CONVERT_MKV_DEFAULT
ASK_CONVERT_MKV_DEFAULT = False
SETTING_SHOW_SMART_TAGS_POPUP = "ui/show_smart_tags_popup"  # default False (winner popup after a --smart-tags run; toggled in the Debug menu)
SETTING_FONT_SCALE       = "ui/font_scale"             # default 1.0 (whole-UI text size; see FONT_SCALE_CHOICES)
SETTING_DEFAULT_VOD_DIR  = "paths/default_vod_dir"    # default "" (Choose-video picker opens to last-used)
SETTING_EXPORT_DIR       = "paths/export_dir"         # default "" (frozen: platform Movies/Videos dir; checkout: video_out/ — see gui.export_root)
# Optional CUDA-PyTorch GPU acceleration (Windows). The manifest file (see vodparse.gpu_support)
# is the source of truth for whether it's installed/active; this only tracks the GUI's own state:
# whether to stop making the automatic first-run offer. It's set when the user explicitly opts out
# ("Don't ask again") or when no NVIDIA GPU is present (nothing to offer); a plain dismissal of the
# offer leaves it unset so the offer returns next launch. Never "general/…"; that INI section is
# reserved (see note below). The active/installed version is read live from the probe, not stored.
SETTING_GPU_SKIP_FIRST_RUN_OFFER = "gpu/skip_first_run_offer"  # default False (auto-offer still due)
# NOT "general/…": QSettings reserves the "General" INI section (case-insensitive), so a "general/"
# group is written to "[%General]" on save but never maps back on read; the value silently vanishes
# across restarts. Any non-reserved group ("paths") round-trips fine.
# Window geometry is saved per stacked screen (the analysis page and the Review page suit
# different sizes), so each remembers its own. Both are QByteArrays, absent until first save.
# Restoring is unconditional (there is no opt-out preference): the Settings dialog's
# "Reset window layouts" button clears both keys, which is the way back from a bad saved size.
SETTING_GEOMETRY_MAIN    = "window/geometry_main"      # the analysis page (stack index 0)
SETTING_GEOMETRY_REVIEW  = "window/geometry_review"    # the Review page (stack index 1)

# The upload assistant's two templates (see vodparse.gui.upload_assistant, which shares this
# store). Defined here rather than there so the Settings dialog's "Reset all preferences" can
# spare them without a circular import: they're the user's own writing, with their own reset
# button in the assistant's settings window, not preferences on any Settings tab.
SETTING_UPLOAD_TITLE_TEMPLATE = "upload/title_template"
SETTING_UPLOAD_DESC_TEMPLATE  = "upload/description_template"

# Default extract options flags (extract_matches flag name → default). Their last-used values are
# remembered in QSettings under "extract/opt_<key>" so the Advanced Analysis Options popup opens pre-filled.
# smart_tags is the Smart/Manual tag-reading choice: Smart runs extract_matches --smart-tags,
# which sweeps every combination of the four tag flags below and keeps the best result; they
# are inert (greyed out in the popup) while it's selected. Smart is the default: it needs no
# knowledge of the stream's overlay, so it's the right first run on an unfamiliar VOD.
EXTRACT_OPT_DEFAULTS = {
    "smart_tags": True,
    "filter_banners": False,
    "overlay_at_bottom": False,
    "overlay_stacked": False,
    "no_overlay_filter": False,
    "debug": True,
}
# --set-characters lives in the same Advanced Analysis Options popup but is a string (not a bool), so it's tracked
# separately from EXTRACT_OPT_DEFAULTS; remembered under "extract/opt_set_characters".
SET_CHARACTERS_DEFAULT = "join"
# "Export as" (the extract_matches --mode: whole sets or individual games) lives in the same
# popup and is likewise a string tracked separately; remembered under "extract/opt_export_mode".
# Sets mirrors extract_matches' own default.
EXPORT_MODE_DEFAULT = "tournament_set"

# The four tag-reading flag keys (a subset of EXTRACT_OPT_DEFAULTS above), and the full set of
# keys the Smart/Manual tag-reading control edits; those four plus the smart_tags radio. Order
# matches how _extract_flags historically emitted the CLI flags, so tag_reading_flags is stable.
TAG_FLAG_KEYS = ("filter_banners", "overlay_at_bottom", "overlay_stacked", "no_overlay_filter")
TAG_OPTION_KEYS = ("smart_tags", *TAG_FLAG_KEYS)

# extract_matches' Smart tag reading: it sweeps the four flags itself. The one spelling every
# GUI uses, both to send it and to recognise it in an argv (the summary popup keys on it).
SMART_TAGS_FLAG = "--smart-tags"

# key → extract_matches CLI flag, in TAG_FLAG_KEYS order.
_TAG_FLAG_CLI = {
    "filter_banners":    "--filter-banners",
    "overlay_at_bottom": "--overlay-at-bottom",
    "overlay_stacked":   "--overlay-stacked",
    "no_overlay_filter": "--no-overlay-filter",
}


def seed_tag_options(sidecar: Mapping[str, Any] | None, global_opts: dict) -> dict:
    """The initial Smart/Manual + four-toggle state for a tag-reading dialog: the five
    TAG_OPTION_KEYS.

    Seeded from the per-VOD extract_flags sidecar (see vodparse.extract_flags) when it
    exists; the radio from its ``smart_tags``, the toggles from its resolved ``flags``:
    so a smart winner pre-fills Manual with the flags it chose, ready to tweak just one.
    For a pre-sidecar VOD (``sidecar is None``) it falls back to the current global extract
    opts: the best available record, exactly what the Advanced Analysis Options popup would
    show, and it self-heals to the sidecar on the first re-read."""
    if sidecar is not None:
        return {"smart_tags": bool(sidecar["smart_tags"]),
                **{key: bool(sidecar["flags"][key]) for key in TAG_FLAG_KEYS}}
    return {key: bool(global_opts.get(key, EXTRACT_OPT_DEFAULTS[key]))
            for key in TAG_OPTION_KEYS}


def tag_reading_flags(opts: dict) -> list[str]:
    """The extract_matches tag-reading flags for a set of tag options: ``--smart-tags`` when
    Smart is chosen (it sweeps the four flags itself, so they must not be sent alongside:
    extract_matches rejects the combination), else the manual flags that are on. The global
    ``--debug`` preference is deliberately not handled here; the caller appends it, since it's
    a diagnostic, not a tag-reading choice."""
    if opts.get("smart_tags"):
        return [SMART_TAGS_FLAG]
    return [flag for key, flag in _TAG_FLAG_CLI.items() if opts.get(key)]


def review_mode_note(group_by: str, overlay_at_bottom: bool) -> str:
    """The parenthetical after the Review title when a non-default tag source is in effect:
    'in-game tags' when set-grouping is forced to the in-game nameplates, 'overlay at
    bottom' when the tournament overlay is read from the bottom of the frame, joined when
    both apply. Empty in the common (default) case."""
    notes = []
    if group_by == "ingame":
        notes.append("ingame tags")
    if overlay_at_bottom:
        notes.append("overlay at bottom")
    return ", ".join(notes)


# Extract toggles whose right value is VOD-specific and changes tag output, so a value left over
# from another VOD (they persist across VODs/sessions) silently corrupts the next one. Surfaced as
# a run-time warning whenever one differs from its EXTRACT_OPT_DEFAULTS value, in either direction:
# so flipping a default here doesn't break the check. debug and set_characters are preferences, not
# situational, so they're intentionally absent.
SITUATIONAL_EXTRACT_OPTS = {
    "overlay_at_bottom": "overlay-at-bottom",
    "overlay_stacked": "overlay-stacked",
    "no_overlay_filter": "no-overlay-filter",
}
