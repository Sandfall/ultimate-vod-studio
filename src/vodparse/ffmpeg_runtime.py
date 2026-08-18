"""FFmpeg binary discovery and the stream-copy clip cut.

Importing this module wires the PATH so an ffmpeg binary is findable in every install
shape (frozen bundle, Homebrew dev setup, plain pip/uv install via imageio-ffmpeg).
"""

import functools
import os
import shutil
import subprocess
import sys
from pathlib import Path

import ffmpeg

# ffmpeg-python shells out to an `ffmpeg` binary found on PATH. Make sure one is found:
#  - Frozen app: a static ffmpeg is bundled next to the app's resources, so end users need
#    nothing installed. Put the bundle dir first so it always wins.
#  - Dev / GUI subprocess: the inherited PATH misses Homebrew's bin dirs, so add the standard
#    macOS install locations (a no-op on other platforms, which find ffmpeg normally).
from vodparse.runtime_paths import bundle_dir, is_frozen

_path_prefixes = []
if is_frozen():
    # The bundled static ffmpeg lives in bin/ (kept out of the bundle root so it can't shadow
    # the ffmpeg-python package directory: see UltimateVODStudio.spec).
    _path_prefixes.append(str(bundle_dir() / "bin"))
if sys.platform == "darwin":
    _path_prefixes += ["/opt/homebrew/bin", "/usr/local/bin"]
if _path_prefixes:
    os.environ["PATH"] = os.pathsep.join(_path_prefixes) + os.pathsep + os.environ.get("PATH", "")


# Windows-only Popen flag: spawn ffmpeg without flashing a console window. The packaged app
# is windowed (console=False), so a spawned console binary has no console to inherit and
# Windows pops up a new one per run: one flash per exported clip. 0 elsewhere (the constant
# only exists on Windows builds of Python).
NO_CONSOLE_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@functools.lru_cache(maxsize=1)
def ffmpeg_binary() -> str:
    """Best-effort path to an ffmpeg executable for ffmpeg-python's ``.run(cmd=...)``
    or a raw subprocess.

    Resolution order:
      1. An ``ffmpeg`` already on PATH: this covers a system/Homebrew install *and*
         the frozen bundle's ``bin/`` dir and the macOS locations wired in above.
      2. The static build shipped by the ``imageio-ffmpeg`` wheel, so plain ``pip``/
         ``uv`` source installs are self-contained with no separate install step.
      3. The literal ``"ffmpeg"`` as a last resort, so the caller's own run surfaces
         the standard "not found" error.
    """
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


@functools.lru_cache(maxsize=1)
def ensure_ffmpeg() -> str:
    """Return a verified-working ffmpeg path, or raise a clear, actionable error.

    Called once (cached) before clip export so a missing binary fails up-front with
    install guidance instead of a bare ``FileNotFoundError`` mid-export.
    """
    exe = ffmpeg_binary()
    try:
        subprocess.run([exe, "-version"], capture_output=True, check=True,
                       stdin=subprocess.DEVNULL, creationflags=NO_CONSOLE_WINDOW)
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            "FFmpeg is required to export clips, but no working ffmpeg binary was found.\n"
            "Fix it with any one of:\n"
            "  - macOS:   brew install ffmpeg\n"
            "  - Ubuntu:  sudo apt install ffmpeg\n"
            "  - Windows: winget install Gyan.FFmpeg\n"
            "  - Any OS:  pip install imageio-ffmpeg   (bundled static build)\n"
            "  - Or run the packaged Ultimate VOD Studio app, which ships ffmpeg."
        ) from e
    return exe


def clip_match(video_path: str, start_sec: float, end_sec: float, output_path: str) -> None:
    """Cut [start_sec, end_sec] from `video_path` to `output_path` via a stream-copy ffmpeg run."""
    cmd = ensure_ffmpeg()  # verified once (cached); raises a friendly error if absent
    stream = (
        ffmpeg
        .input(video_path, ss=start_sec, to=end_sec)
        .output(output_path, c="copy")
        .overwrite_output()
    )
    # ffmpeg-python's .run() offers no creationflags passthrough, so compile the argv and run
    # it ourselves; otherwise every clip cut flashes a console window on Windows. stdin is
    # detached (like remux_to_mp4_argv's -nostdin) so a long CLI export can't eat keystrokes.
    result = subprocess.run(stream.compile(cmd=cmd), capture_output=True,
                            stdin=subprocess.DEVNULL, creationflags=NO_CONSOLE_WINDOW)
    if result.returncode != 0:
        # A failed cut (e.g. disk full mid-write) can leave a truncated .mp4 that looks
        # finished in "Open folder"; remove it so only complete clips remain, then surface
        # the error so the export loop still stops.
        Path(output_path).unlink(missing_ok=True)
        stderr_tail = result.stderr.decode(errors="replace").strip().splitlines()[-15:]
        raise RuntimeError(f"ffmpeg failed cutting {output_path!r}:\n" + "\n".join(stderr_tail))


# Containers Qt's QMediaPlayer (macOS AVFoundation backend) can't render, so the GUI offers a one-off
# conversion to MP4 on selection. See MainWindow._offer_convert in vodparse.gui.app.
UNPLAYABLE_CONTAINER_EXTS = {".mkv", ".avi"}


def remux_to_mp4_argv(video_path: str, output_path: str, *, reencode: bool = False) -> list[str]:
    """The ffmpeg argv that rewrites `video_path` into a QMediaPlayer-friendly MP4 at `output_path`.

    Includes ``-progress pipe:1`` so a QProcess caller can drive a progress bar from stdout. Stream-
    copies by default (fast, disk-I/O bound, no quality loss). Pass ``reencode=True`` for the
    H.264/AAC fallback: used both when the source's codecs can't live in an MP4 (e.g. VP9/Opus/PCM)
    *and* when a copy leaves a codec the analyzer (OpenCV) can't decode (HEVC/10-bit, which macOS
    plays natively but OpenCV's build won't); ``-pix_fmt yuv420p`` forces 8-bit 4:2:0 so both OpenCV
    and AVFoundation read the result. ``avoid_negative_ts=make_zero`` pins the MP4 timeline to 0 so
    seek positions stay in the source's second-space (the base the detector, exporter, and trim
    player share). No ``+faststart``: it needs a second whole-file pass, and for a *local* file
    AVFoundation seeks fine without it.
    """
    common = [ensure_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-nostats",
              "-i", video_path]
    codec = (["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-preset", "veryfast"]
             if reencode else ["-map", "0", "-c", "copy"])
    return [*common, *codec, "-avoid_negative_ts", "make_zero", "-progress", "pipe:1",
            "-y", output_path]


def parse_progress_bytes(progress_text: str) -> int | None:
    """The latest ``total_size`` (bytes written so far) in a chunk of ffmpeg ``-progress`` output,
    or None if the chunk carries no such line. Divide by the source size for a stream-copy fraction
    (the copy's output is ~the same size as its input)."""
    latest: int | None = None
    for line in progress_text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key == "total_size":
            try:
                latest = int(value.strip())
            except ValueError:
                pass
    return latest
