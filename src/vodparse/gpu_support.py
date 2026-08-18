"""Optional CUDA-PyTorch support: pinned wheel manifest, install paths, and probes.

The shipped Windows build installs a **CPU-only** torch (see build_windows.ps1) to keep
the download small: most users have no NVIDIA GPU. A user who *does* can opt in to GPU
acceleration, which downloads the matching CUDA wheels and unpacks them into a per-user
"site" directory (see vodparse.cli.gpu_support). At startup ``freeze_entry`` calls
:func:`activate_cuda_site`, which makes the CUDA torch **shadow** the bundled CPU torch in
every process: GUI and each pipeline subprocess alike.

This module is deliberately **stdlib-only and torch-free**: ``freeze_entry`` imports it
before torch is ever loaded, so it must not pull torch (or Qt) in. It holds four things:

* the pinned wheel table (:data:`WHEELS`): the exact CUDA build we download and verify,
* the install-path helpers (:func:`gpu_site_dir`, :func:`manifest_path`, …) shared by the
  installer, the activation seam, and the GUI so they all agree on one location,
* the activation seam (:func:`activate_cuda_site` + :class:`_CudaSiteFinder`) that redirects
  ``import torch`` to the downloaded CUDA build,
* two probes used by the GUI: :func:`active_site_dir` (is a valid CUDA install present?)
  and :func:`has_nvidia_gpu` (should we even offer the download?).

The **source of truth for activation is the manifest file**, not QSettings, precisely so
``freeze_entry`` can decide whether to activate without importing Qt.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from vodparse.runtime_paths import user_data_dir


class Wheel(TypedDict):
    """One pinned wheel to download and verify (a row of :data:`WHEELS`)."""
    package: str
    version: str
    filename: str
    url: str
    sha256: str
    size: int

# ── Pinned CUDA build ────────────────────────────────────────────────────────────────
# CUDA 12.4 pairs with torch 2.6.0 / torchvision 0.21.0 and targets the widest range of
# NVIDIA drivers and GPUs (GTX 10-series through RTX 40-series). Keep the CPU wheel pinned
# to the SAME versions in build_windows.ps1 so this GPU swap is an exact, ABI-matched
# replacement. Bump all of these together if you ever move to a newer CUDA/torch.
CUDA_TAG = "cu124"
TORCH_VERSION = "2.6.0"
TORCHVISION_VERSION = "0.21.0"

# The CUDA wheels only exist for these interpreter/platform tags. ``install`` refuses to
# run anywhere else (e.g. macOS, where no CUDA build is published) rather than fetching an
# incompatible wheel. cp313 matches the Windows build's Python (see build-windows.yml).
WHEEL_PYTHON_TAG = "cp313"
# CPython non-freethreaded wheels use the same value for the python-tag and ABI-tag slots of
# the wheel filename ("…-cp313-cp313-…"); naming the second slot keeps it from reading as a typo.
WHEEL_ABI_TAG = WHEEL_PYTHON_TAG
WHEEL_PLATFORM_TAG = "win_amd64"

_WHEEL_INDEX = f"https://download.pytorch.org/whl/{CUDA_TAG}"


def _wheel(package: str, version: str, sha256: str, size: int) -> Wheel:
    """One row of :data:`WHEELS`. ``filename`` keeps the literal ``+`` for the saved file;
    the download URL percent-encodes it as ``%2B``, matching PyTorch's index links."""
    filename = f"{package}-{version}+{CUDA_TAG}-{WHEEL_PYTHON_TAG}-{WHEEL_ABI_TAG}-{WHEEL_PLATFORM_TAG}.whl"
    url_name = filename.replace("+", "%2B")
    return Wheel(
        package=package,
        version=version,
        filename=filename,
        url=f"{_WHEEL_INDEX}/{url_name}",
        sha256=sha256,
        size=size,
    )


# The exact wheels to download, verify, and unpack. Sizes/hashes are from PyTorch's cu124
# index (reproduce: ``curl -s https://download.pytorch.org/whl/cu124/torch/`` → the
# ``#sha256=`` fragment; ``curl -sI`` the wheel URL → ``content-length``).
WHEELS: tuple[Wheel, ...] = (
    _wheel("torch", TORCH_VERSION,
           "519330eef09534acad8110b6f423d2fe58c1d8e9ada999ed077a637a0021f908", 2532308219),
    _wheel("torchvision", TORCHVISION_VERSION,
           "8fcf55321b206de70ff8e01c884fa42e57a60b1cb749341b96e0f22c8a7c9ec7", 6141095),
)

# Total download footprint, for the size warning the GUI shows before starting.
DOWNLOAD_BYTES = sum(w["size"] for w in WHEELS)


# ── Install paths ────────────────────────────────────────────────────────────────────
# Override the whole install root for tests (and the odd sandbox) via this env var.
GPU_DIR_ENV = "ULTIMATE_VOD_GPU_DIR"


def gpu_support_root() -> Path:
    """Root holding the GPU install (``site/``) and its ``manifest.json``."""
    override = os.environ.get(GPU_DIR_ENV)
    base = Path(override).expanduser() if override else user_data_dir()
    return base / "gpu_support"


def gpu_site_dir() -> Path:
    """The ``site-packages``-like directory the CUDA wheels unpack into. ``freeze_entry`` calls
    :func:`activate_cuda_site`, which installs a meta-path finder so its ``torch``/``torchvision``
    shadow the bundled CPU ones: a bare ``sys.path`` prepend can't win against PyInstaller's
    bundled importer, so the finder does the shadowing (see :class:`_CudaSiteFinder`)."""
    return gpu_support_root() / "site"


def gpu_staging_dir() -> Path:
    """Scratch directory the installer unpacks into before an atomic swap into ``site/``.
    A failed or cancelled install leaves this behind (cleaned on next run), never ``site/``."""
    return gpu_support_root() / "staging"


def manifest_path() -> Path:
    """The activation source of truth: written only after a fully verified, complete install."""
    return gpu_support_root() / "manifest.json"


def read_manifest() -> dict[str, object] | None:
    """Parsed manifest, or None if absent/unreadable/malformed (never raises)."""
    try:
        with open(manifest_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# The load-bearing packages the app imports out of the CUDA ``site/``. :func:`active_site_dir`
# refuses to activate unless **every** one is present: a site with ``torch/`` but no
# ``torchvision/`` (say one package was quarantined by antivirus) would otherwise load a CUDA
# ``torch`` while ``torchvision`` fell through to the bundled CPU build: a CPU-ABI extension
# against a CUDA torch, which crashes. Declining activation (staying all-CPU) beats half-shadowing.
#
# Granularity note: this check is package-*presence*, not package-*integrity*. Corruption INSIDE a
# required package (e.g. antivirus removing ``torch/_C*.pyd`` but leaving ``torch/__init__.py``)
# still passes, so the finder claims the torch namespace and ``import torch`` then raises with no
# CPU fallback. The install path can't create that state (sha-verified wheels → atomic swap →
# manifest last); it takes post-install mutation, and it's recoverable via Remove/Reinstall
# (``install_present``). Full-content verification of a multi-GB site at every startup isn't worth
# it: degradation is safe for every state the install path can create.
_REQUIRED_PACKAGES = frozenset({"torch", "torchvision"})


def _site_packages(site: Path) -> frozenset[str]:
    """The top-level importable packages present in ``site``: each a directory with an
    ``__init__.py``: ``torch``/``torchvision`` plus whatever the wheels drop alongside them
    (``functorch``, ``torchgen``). :class:`_CudaSiteFinder` shadows exactly this set, so *nothing*
    the CUDA site provides can leak back to the bundled copy. Empty if ``site`` doesn't exist."""
    if not site.is_dir():
        return frozenset()
    return frozenset(entry.name for entry in site.iterdir() if (entry / "__init__.py").is_file())


def active_site_dir() -> Path | None:
    """The site dir to activate, or None. Returns it only when a *complete* manifest matching
    every current pin (:data:`CUDA_TAG` and the torch/torchvision versions) exists AND every
    required package (:data:`_REQUIRED_PACKAGES`) is actually present in the site dir; so a
    partial, stale, or wrong-version install is ignored rather than half-activated (see
    :data:`_REQUIRED_PACKAGES`). The version comparison matters even within one CUDA tag:
    several torch releases publish ``cu124`` wheels, so after a future :data:`TORCH_VERSION`
    bump an old site would otherwise silently shadow the *newer* bundled torch. This is the one
    function ``freeze_entry`` calls, so it stays torch-free and cheap."""
    manifest = read_manifest()
    if manifest is None or manifest.get("complete") is not True:
        return None
    if manifest.get("cuda_tag") != CUDA_TAG:
        return None
    if manifest.get("torch_version") != TORCH_VERSION:
        return None
    if manifest.get("torchvision_version") != TORCHVISION_VERSION:
        return None
    site = gpu_site_dir()
    return site if _REQUIRED_PACKAGES <= _site_packages(site) else None


def install_present() -> bool:
    """True if *any* GPU-support install exists on disk: including a stale one the app can't
    activate: a wrong :data:`CUDA_TAG` left by an older app version, or a partial install missing a
    required package. Deliberately broader than :func:`active_site_dir`, which returns a path only
    for a *usable, current* install. The GUI keys its **Remove** button and the uninstaller off this
    (not ``active_site_dir``) so a multi-gigabyte stale install can always be cleared, not only an
    active one. Keys off the unpacked ``site/`` dir or the manifest; a scratch ``staging/`` leftover
    is not a user-facing install (the installer clears it on its next run) so it doesn't count."""
    return gpu_site_dir().is_dir() or manifest_path().is_file()


# ── Activation: shadow the bundled torch with the downloaded CUDA build ───────────────


class _CudaSiteFinder(importlib.abc.MetaPathFinder):
    """A meta-path finder that resolves the CUDA ``site/``'s packages from that dir instead of
    wherever they'd otherwise be found. It shadows every package the site provides
    (:func:`_site_packages`: ``torch``, ``torchvision``, and the incidental ones the wheels
    drop like ``functorch``/``torchgen``).

    Why a finder and not just ``sys.path``: in the PyInstaller-frozen app, ``import torch`` is
    served by PyInstaller's own meta-path importer (it bundles torch's pure modules into the
    archive). ``sys.meta_path`` is consulted in order, and the ``sys.path``-reading finder
    (``PathFinder``) sits **last**; so prepending ``site/`` to ``sys.path`` never wins against
    the bundled copy. Installing this finder at ``sys.meta_path[0]`` (see
    :func:`activate_cuda_site`) puts the CUDA build ahead of the bundled one. In a plain source
    checkout there's no such bundled importer and ``sys.path`` would already work, but the finder
    is harmless there too.

    It claims each shadowed package's **whole** namespace, not just the top-level module: if
    ``torch`` loaded from ``site/`` but ``torch.nn`` came from the bundle, the two copies would
    mix catastrophically."""

    def __init__(self, site: Path) -> None:
        self._search = [str(site)]
        self._names = _site_packages(site)

    def find_spec(self, fullname: str, path: Sequence[str] | None = None,
                  target: object = None) -> importlib.machinery.ModuleSpec | None:
        if fullname.split(".", 1)[0] not in self._names:
            return None  # not ours; let the rest of sys.meta_path (i.e. the bundle) handle it
        # Top-level import: search the site dir. Submodule import: `path` is the parent package's
        # __path__, which, because the parent already loaded from site/, points into site/, so
        # the normal path machinery resolves the submodule there.
        search = list(path) if path is not None else self._search
        return importlib.machinery.PathFinder.find_spec(fullname, search)


def activate_cuda_site() -> Path | None:
    """If a complete, version-matched CUDA install is present, install a :class:`_CudaSiteFinder`
    at the front of ``sys.meta_path`` so ``import torch`` resolves to it, and return the site dir.
    Return None (and change nothing) otherwise. Idempotent: a second call won't stack a finder.

    ``freeze_entry`` calls this once at startup, before anything imports torch. It is torch-free,
    so it stays ahead of the first torch import."""
    site = active_site_dir()
    if site is None:
        return None
    if not any(isinstance(f, _CudaSiteFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _CudaSiteFinder(site))
    return site


def is_supported_platform() -> bool:
    """True only where the pinned CUDA wheels can actually be installed (Windows x64,
    matching the frozen build's interpreter). CUDA wheels aren't published for macOS."""
    return sys.platform == "win32"


# ── GPU presence probe ───────────────────────────────────────────────────────────────
_NVIDIA_SMI_TIMEOUT_SEC = 5

# ``nvidia-smi`` is a console-subsystem binary, so on Windows a plain spawn flashes a console
# window (the probe runs at startup and whenever Settings opens). CREATE_NO_WINDOW suppresses it;
# the flag doesn't exist off Windows, so ``getattr`` yields 0 there: a no-op, keeping this
# stdlib-only module cross-platform without a ``sys.platform`` guard. (Same idiom as
# ffmpeg_runtime.NO_CONSOLE_WINDOW; duplicated rather than imported to keep gpu_support torch/Qt/
# dependency-free per this module's docstring.)
_NO_CONSOLE_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def has_nvidia_gpu() -> bool:
    """True if an NVIDIA GPU appears present, by invoking ``nvidia-smi`` (installed with
    every NVIDIA driver). Any failure, missing binary, non-zero exit, timeout, means "no",
    so we never push a multi-gigabyte download onto a machine that can't use it."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=_NVIDIA_SMI_TIMEOUT_SEC, check=False,
            creationflags=_NO_CONSOLE_WINDOW,  # no console-window flash on Windows
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0
