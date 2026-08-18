"""Install (or probe) optional CUDA-PyTorch support: the headless backend.

The GUI shells out to this stage (``sys.executable -m vodparse.cli.gpu_support install``,
frozen: the app exe re-enters ``freeze_entry``) to download the pinned CUDA wheels, verify
them, unpack them into the per-user ``site/`` directory, and write the activation manifest.
``freeze_entry`` then calls ``gpu_support.activate_cuda_site()``, which installs a
``sys.meta_path`` finder that shadows the bundled CPU torch with this ``site/``'s CUDA torch
(a bare ``sys.path`` prepend can't; PyInstaller's own importer wins). See
:mod:`vodparse.gpu_support` for the pinned wheel table, the finder, and the paths.

Subcommands::

    python -m vodparse.cli.gpu_support status      # is a CUDA install present?
    python -m vodparse.cli.gpu_support install     # download + verify + unpack + activate
    python -m vodparse.cli.gpu_support uninstall    # remove the CUDA install
    python -m vodparse.cli.gpu_support probe        # print active compute setup as JSON

``install`` streams ``PROGRESS <percent>`` lines to stdout for the GUI's progress bar, plus a
``STAGE <name>`` line at each phase boundary (``downloading`` / ``unpacking`` / ``finishing``)
so the GUI can relabel the dialog; the download is byte-smooth, but unpacking and the final
swap give sparse feedback, so the label tells the user the bar isn't stuck. ``install`` is
safe to interrupt: everything lands in a scratch ``staging/`` dir and is swapped into
``site/`` only after every wheel verifies, and the manifest is written last; so a killed,
failed, or size/hash-mismatched run never half-activates a broken install.

Reproduce a manual install from a clean checkout (Windows, Python 3.13)::

    python -m vodparse.cli.gpu_support install
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from vodparse import gpu_support
from vodparse.jsonio import write_json_atomic

# The progress bar is split across the install's phases so it keeps moving in each of them.
# Downloading is byte-smooth and owns the bulk; unpacking (a slow, multi-gigabyte inflate) and
# the final swap each get their own visible band, so the bar advances during them instead of
# freezing at the download-end percentage: the "stuck near the end" the earlier single 0.97
# split produced (download 0→97%, then a silent extractall + swap to 100%).
_DOWNLOAD_FRACTION = 0.90  # downloading:                      0 → 90%
_UNPACK_FRACTION = 0.99    # unpacking:                       90 → 99%
                           # finishing (atomic swap + manifest): 99 → 100%
_CHUNK_BYTES = 1 << 20  # 1 MiB streaming reads
# Socket read timeout: a connection that silently stalls (dropped Wi-Fi, half-dead proxy) errors
# out into the normal failure path instead of hanging the install forever with a frozen bar.
_DOWNLOAD_TIMEOUT_SEC = 30
_USER_AGENT = "ultimate-vod-studio-gpu-installer"

ProgressCallback = Callable[[int], None]


# ── Progress protocol (the PROGRESS/STAGE lines the GUI parses) ──────────────────────
def _emit_progress(percent: int) -> None:
    """Print a ``PROGRESS`` line the GUI parses to drive its progress dialog."""
    print(f"PROGRESS {max(0, min(100, percent))}", flush=True)


def _emit_stage(name: str) -> None:
    """Print a ``STAGE`` line the GUI maps to the progress dialog's label, so the user sees which
    phase is running (downloading, unpacking, or finishing) rather than a bar that looks frozen
    once the (byte-smooth) download ends and the slower, feedback-poorer unpack/swap begin."""
    print(f"STAGE {name}", flush=True)


# ── Download / verify / unpack ────────────────────────────────────────────────────────
def _download_and_verify(wheel: gpu_support.Wheel, dest: Path,
                         on_bytes: ProgressCallback) -> None:
    """Stream one wheel to ``dest``, verifying byte count and SHA-256. Raises on mismatch.

    ``on_bytes`` is called with the running byte total for *this* wheel so the caller can
    render overall progress across all wheels."""
    url = wheel["url"]
    expected_size = wheel["size"]
    expected_sha = wheel["sha256"]
    dest.parent.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    written = 0
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SEC) as response, \
            open(dest, "wb") as out:  # noqa: S310 - pinned https/file URL
        while chunk := response.read(_CHUNK_BYTES):
            out.write(chunk)
            digest.update(chunk)
            written += len(chunk)
            on_bytes(written)

    if written != expected_size:
        raise OSError(f"size mismatch for {wheel['filename']}: got {written} bytes, "
                      f"expected {expected_size}")
    actual_sha = digest.hexdigest()
    if actual_sha != expected_sha:
        raise OSError(f"sha256 mismatch for {wheel['filename']}: got {actual_sha}, "
                      f"expected {expected_sha}")


def _unpack_wheels(wheel_paths: list[Path], dest: Path) -> None:
    """Extract every wheel into ``dest``, emitting ``PROGRESS`` across the unpack band
    (:data:`_DOWNLOAD_FRACTION`→:data:`_UNPACK_FRACTION`) by cumulative uncompressed bytes.

    Extracting member-by-member instead of one ``extractall`` per wheel is what lets the bar keep
    moving during the slow multi-gigabyte inflate; the phase that used to freeze it at the
    download-end percentage. Only whole-percent changes are emitted, so a wheel with thousands of
    small files can't flood the pipe with duplicate ``PROGRESS`` lines."""
    dest.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as archives:
        # Open every wheel, then flatten to (archive, member) pairs so one running byte total
        # spans all wheels: the bar advances continuously rather than restarting per wheel.
        opened = [archives.enter_context(zipfile.ZipFile(path)) for path in wheel_paths]
        members = [(zf, info) for zf in opened for info in zf.infolist()]
        total_bytes = sum(info.file_size for _, info in members) or 1  # avoid div-by-zero

        span = _UNPACK_FRACTION - _DOWNLOAD_FRACTION
        extracted = 0
        last_percent = -1
        for zf, info in members:
            zf.extract(info, dest)
            extracted += info.file_size
            percent = round((_DOWNLOAD_FRACTION + extracted / total_bytes * span) * 100)
            if percent != last_percent:  # emit only on a whole-percent change (see docstring)
                _emit_progress(percent)
                last_percent = percent


# ── Atomic swap + activation manifest ────────────────────────────────────────────────
def _atomic_swap(built_site: Path, site: Path) -> None:
    """Move a freshly-built site dir into place. If an install already exists it is moved
    aside first, then removed only after the new one is in place; so an interrupted swap
    leaves either the old install or none (which degrades to CPU), never a mangled mix."""
    site.parent.mkdir(parents=True, exist_ok=True)
    backup = site.with_name(site.name + ".old")
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)
    if site.exists():
        site.rename(backup)
    built_site.rename(site)  # atomic on the same filesystem
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


def _write_manifest() -> None:
    """Write the activation manifest last: its presence with ``complete: true`` is what
    ``gpu_support.active_site_dir`` (and thus ``freeze_entry``) keys off. Written atomically:
    a crash mid-write would otherwise leave a truncated file that ``read_manifest`` rejects;
    fail-closed, but it would silently waste the just-downloaded multi-gigabyte install."""
    manifest = {
        "complete": True,
        "cuda_tag": gpu_support.CUDA_TAG,
        "torch_version": gpu_support.TORCH_VERSION,
        "torchvision_version": gpu_support.TORCHVISION_VERSION,
        "wheels": [{"filename": w["filename"], "sha256": w["sha256"], "size": w["size"]}
                   for w in gpu_support.WHEELS],
        "installed_at": int(time.time()),
    }
    path = gpu_support.manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, manifest)


# ── Subcommands ───────────────────────────────────────────────────────────────────────
def install(*, force: bool = False) -> int:
    """Download, verify, unpack, and activate the pinned CUDA build. Returns a process
    exit code. ``force`` only relaxes the NVIDIA-GPU check (for maintainer testing on a
    GPU-less box); the platform and interpreter guards still apply; there is no CUDA wheel
    to fetch off Windows, and a wheel built for another CPython could never import."""
    if not gpu_support.is_supported_platform():
        print(f"error: CUDA builds are only available for Windows "
              f"({gpu_support.WHEEL_PLATFORM_TAG}); nothing to install on {sys.platform}.",
              file=sys.stderr)
        return 2
    running_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if running_tag != gpu_support.WHEEL_PYTHON_TAG:
        # The pinned wheels carry compiled extensions for exactly one CPython; unpacked under any
        # other interpreter they can never import (and once the site activates there is no CPU
        # fallback: the finder claims the whole torch namespace). Refuse before any download.
        print(f"error: the pinned CUDA wheels are {gpu_support.WHEEL_PYTHON_TAG} builds, but this "
              f"app is running Python {sys.version_info.major}.{sys.version_info.minor} "
              f"({running_tag}); their compiled extensions could never import here.\n"
              f"       Rebuild with a matching Python, or update the pins in vodparse.gpu_support.",
              file=sys.stderr)
        return 2
    if not force and not gpu_support.has_nvidia_gpu():
        print("error: no NVIDIA GPU detected (nvidia-smi not found or failed).\n"
              "       Pass --force to install anyway (e.g. for testing).", file=sys.stderr)
        return 2

    staging = gpu_support.gpu_staging_dir()
    shutil.rmtree(staging, ignore_errors=True)  # clear any leftover from an interrupted run
    wheels_dir = staging / "wheels"
    built_site = staging / "site"
    built_site.mkdir(parents=True, exist_ok=True)

    total_bytes = gpu_support.DOWNLOAD_BYTES
    completed_bytes = 0
    _emit_stage("downloading")
    _emit_progress(0)
    try:
        wheel_paths: list[Path] = []
        for wheel in gpu_support.WHEELS:
            base = completed_bytes
            dest = wheels_dir / wheel["filename"]
            print(f"Downloading {wheel['filename']} "
                  f"({wheel['size'] / 1e9:.2f} GB)...", flush=True)

            def on_bytes(written: int, _base: int = base) -> None:
                fraction = (_base + written) / total_bytes * _DOWNLOAD_FRACTION
                _emit_progress(round(fraction * 100))

            _download_and_verify(wheel, dest, on_bytes)
            completed_bytes += wheel["size"]
            wheel_paths.append(dest)

        _emit_stage("unpacking")
        print("Unpacking wheels...", flush=True)
        _emit_progress(round(_DOWNLOAD_FRACTION * 100))
        _unpack_wheels(wheel_paths, built_site)

        _emit_stage("finishing")
        print("Finishing install...", flush=True)
        _emit_progress(round(_UNPACK_FRACTION * 100))
        _atomic_swap(built_site, gpu_support.gpu_site_dir())
        _write_manifest()
    except BaseException as err:  # noqa: BLE001 - report, clean staging, never half-activate
        # site/ and the manifest are left untouched (manifest is written last), so the app
        # stays CPU-functional. Drop the scratch dir so a retry starts clean.
        shutil.rmtree(staging, ignore_errors=True)
        if isinstance(err, KeyboardInterrupt):
            print("\nCancelled.", file=sys.stderr)
            return 130
        print(f"error: GPU support install failed: {err}", file=sys.stderr)
        return 1

    shutil.rmtree(staging, ignore_errors=True)
    _emit_progress(100)
    print(f"Installed CUDA {gpu_support.CUDA_TAG} torch {gpu_support.TORCH_VERSION} "
          f"into {gpu_support.gpu_site_dir()}", flush=True)
    return 0


def uninstall() -> int:
    """Remove the CUDA install (site dir + manifest + any staging). Returns non-zero if the tree
    can't be fully removed; e.g. a torch DLL under ``site/`` is still loaded by a running analysis
    on Windows; rather than reporting success while the install lingers. ``ignore_errors`` lets it
    delete everything it can; the leftover ``root`` is the real success test."""
    root = gpu_support.gpu_support_root()
    if not root.exists():
        print("No GPU support install found.")
        return 0
    shutil.rmtree(root, ignore_errors=True)
    if root.exists():
        print(f"error: could not fully remove {root}; a file may be in use "
              f"(close any running analysis and try again).", file=sys.stderr)
        return 1
    print("Removed GPU support install.")
    return 0


def status() -> int:
    """Print whether a CUDA install is present and active."""
    site = gpu_support.active_site_dir()
    if site is None:
        print("GPU support: not installed (using the bundled CPU torch).")
        return 0
    manifest = gpu_support.read_manifest() or {}
    print(f"GPU support: installed: CUDA {manifest.get('cuda_tag')} "
          f"torch {manifest.get('torch_version')} at {site}")
    return 0


def probe() -> int:
    """Print the active compute setup as a single JSON line (consumed by the GUI status).

    Runs in this torch-importing process; which, frozen, has already had ``freeze_entry`` call
    ``gpu_support.activate_cuda_site`` (a meta-path finder that shadows the bundled CPU torch with
    any installed CUDA ``site/`` build); so the reported device is exactly what a pipeline run
    would use."""
    from vodparse.device import describe_compute

    print(json.dumps(describe_compute()), flush=True)
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="vodparse.cli.gpu_support",
        description="Install or probe optional CUDA-PyTorch GPU support.")
    sub = ap.add_subparsers(dest="command", required=True)
    p_install = sub.add_parser("install", help="download, verify, unpack, and activate the CUDA build")
    p_install.add_argument("--force", action="store_true",
                           help="install even if no NVIDIA GPU is detected (for testing)")
    sub.add_parser("uninstall", help="remove the CUDA install")
    sub.add_parser("status", help="report whether a CUDA install is present")
    sub.add_parser("probe", help="print the active compute setup as JSON")
    args = ap.parse_args(argv)

    if args.command == "install":
        return install(force=args.force)
    if args.command == "uninstall":
        return uninstall()
    if args.command == "status":
        return status()
    if args.command == "probe":
        return probe()
    return 2  # unreachable: subparser is required


if __name__ == "__main__":
    sys.exit(main())
