"""Output path layout.

Every artifact for one source video (pipeline JSON, matches/reviewed JSON, and the
debug CSVs) lives under a single per-vod folder so ``json_output/`` stays uncluttered,
mirroring how ``video_out/`` groups clips by source video.

    json_output/<stem>/pipeline.json
    json_output/<stem>/matches.json
    json_output/<stem>/reviewed.json
    json_output/<stem>/debug/...

The helpers are pure (no mkdir); callers create the directory when they write, as they
already do for every other output path.
"""

from pathlib import Path

JSON_OUTPUT_ROOT = Path("json_output")
VIDEO_OUT_ROOT   = Path("video_out")

# What the GUI's MP4 conversion adds to the stem of an .mp4 source, so the original is never
# replaced by its own conversion: "video.mp4" converts to "video.converted.mp4" (see
# vodparse.gui.convert). A converted file is the same VOD, so vod_stem drops it and the
# conversion keeps the original's analysis and Review draft.
CONVERTED_STEM_SUFFIX = ".converted"


def strip_converted_suffix(stem: str) -> str:
    """``stem`` without any trailing CONVERTED_STEM_SUFFIX: "video.converted" → "video".
    Every one is dropped, so a conversion of a conversion ("video.converted.converted")
    still maps to the original's stem."""
    while stem.endswith(CONVERTED_STEM_SUFFIX):
        stem = stem.removesuffix(CONVERTED_STEM_SUFFIX)
    return stem


def vod_stem(video_path: str | Path) -> str:
    """The filename stem used to name a video's per-vod artifact folders.

    The one sanctioned way to turn a source video path into a folder-name stem: every
    stem-derived path (here and in vodparse.gui.paths) must go through it so the CLI
    stages and the GUI always agree on the folder.

    A trailing ".converted" (CONVERTED_STEM_SUFFIX) is dropped, so the GUI's MP4
    conversion of "video.mp4", "video.converted.mp4", shares "video"'s folders: its
    saved analysis, its Review draft and its exports. A user's own file named
    "video.converted.mp4" shares them too; it is almost certainly the same video.

    Trailing spaces/dots are stripped because Windows removes them when *creating* a
    directory but not when *opening* paths through it: a recording named
    "Weekly 2026-07-15 .mp4" would mkdir "json_output/Weekly 2026-07-15" yet every write
    to "json_output/Weekly 2026-07-15 /pipeline.json" then fails with FileNotFoundError.
    Stripped on every platform so artifact folders stay portable across OSes. They are
    stripped after the converted suffix, so the conversion of "Weekly .mp4" lands in
    the original's "Weekly" folder too.
    """
    stem = strip_converted_suffix(Path(video_path).stem).rstrip(" .")
    return stem or "unnamed"


def output_dir(stem: str) -> Path:
    """The per-vod folder holding every JSON/CSV artifact for one source video."""
    return JSON_OUTPUT_ROOT / stem


def video_out_dir(folder: str, out_root: Path | None = None) -> Path:
    """The clip-output folder for one export (a <stem>_<matches|sets> basename or a
    --folder override). ``out_root`` is the directory that plays the role of
    ``video_out/`` (the export stages' --out-root flag; the GUI passes its resolved
    export root); by default the relative ``video_out/`` under the cwd, as before."""
    return (out_root if out_root is not None else VIDEO_OUT_ROOT) / folder


def debug_dir(stem: str) -> Path:
    """The per-vod debug subfolder for frame/segment CSVs and stripped-reading dumps."""
    return output_dir(stem) / "debug"
