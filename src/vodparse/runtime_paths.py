"""Runtime path resolution that works both in a source checkout and inside a
frozen (PyInstaller) build.

Three distinct roots:

* :func:`bundle_dir`: **read-only** resources shipped with the app: the YOLO/m_char
  model weights and the GUI icon. Frozen, these live under ``sys._MEIPASS``; in a
  checkout they live at the repo root.
* :func:`workspace_dir`: the **writable** root for the app's internal data
  (``json_output/``, Review drafts, downloaded EasyOCR models). A frozen app's
  bundle is read-only (and on macOS, signed bundles must not be written to), so
  this goes to the per-user app-data directory (:func:`user_data_dir`). In a
  checkout it is the repo root, exactly as before, so nothing about the dev
  workflow changes.
* :func:`user_data_dir`: the platform-conventional app-data directory
  (``%LOCALAPPDATA%`` / ``~/Library/Application Support`` / ``$XDG_DATA_HOME``)
  plus :data:`APP_DIR_NAME`. Also the parent of the GPU-support install (see
  ``vodparse.gpu_support``).

Exported *clips* deliberately do NOT live under the workspace when frozen: they
are user-facing media, placed under a visible export root the GUI resolves (the
platform Movies/Videos folder by default; see ``vodparse.gui.export_root``).

Override the writable workspace with the ``ULTIMATE_VOD_WORKSPACE`` environment
variable.
"""

import os
import sys
from pathlib import Path

# src/vodparse/runtime_paths.py -> parents[2] is the repo root in a source checkout.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The folder name this app claims inside per-user platform directories.
APP_DIR_NAME = "Ultimate VOD Studio"

# Environment override for the writable workspace root (frozen builds only).
WORKSPACE_ENV = "ULTIMATE_VOD_WORKSPACE"


def is_frozen() -> bool:
    """True when running inside a PyInstaller (or similar) frozen build."""
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """Root of the read-only bundled resources (models, assets)."""
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if meipass else _REPO_ROOT


def user_data_dir() -> Path:
    """Per-user, writable, app-private data root (platform-appropriate, no repo
    pollution). Pure path computation: callers create the directories they need."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / APP_DIR_NAME


def workspace_dir() -> Path:
    """Writable root for the app's internal data. Created if missing when frozen.

    Checkout: the repo root, unchanged. Frozen: ``$ULTIMATE_VOD_WORKSPACE`` when
    set, else the platform app-data directory (see :func:`user_data_dir`)."""
    if not is_frozen():
        return _REPO_ROOT
    override = os.environ.get(WORKSPACE_ENV)
    root = Path(override).expanduser() if override else user_data_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


# Environment override for the GUI crash log: a path to write to, or an off switch (see
# CRASH_LOG_OFF_VALUES). The log itself is written by ``vodparse.gui.crash_log``; its
# *location* resolves here so Qt-free code (the diagnostics bundler) can find the file
# without importing that Qt-loading module.
CRASH_LOG_ENV = "VODPARSE_CRASH_LOG"

# Values of CRASH_LOG_ENV that mean "don't log at all" rather than "log here".
CRASH_LOG_OFF_VALUES = frozenset({"0", "off", "false", "no"})


def crash_log_path() -> Path:
    """Where the crash log is written: ``$VODPARSE_CRASH_LOG`` when that names a path, else
    ``crash.log`` in the writable workspace (the repo root in a checkout, the per-user
    app-data directory when frozen: see :func:`workspace_dir`)."""
    override = os.environ.get(CRASH_LOG_ENV, "").strip()
    if override and override.lower() not in CRASH_LOG_OFF_VALUES:
        return Path(override).expanduser()
    return workspace_dir() / "crash.log"
