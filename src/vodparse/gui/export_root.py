"""Where exported clips land: the GUI's export root.

The export root is the directory that plays the role of ``video_out/``: each export
creates its own subfolder inside it (see vodparse.paths.video_out_dir). Unlike the
app-internal workspace (vodparse.runtime_paths.workspace_dir), clips are user-facing
media, so the root is visible and configurable:

* the ``paths/export_dir`` setting, when set (frozen or checkout alike), else
* frozen: the platform Movies/Videos folder + "Ultimate VOD Studio", else
* checkout: ``<repo>/video_out``, the pre-split layout, so the dev workflow is unchanged.

The GUI passes the resolved root to the export-capable subprocesses as ``--out-root``
(they otherwise default to the relative ``video_out/`` under their cwd).

Split from gui.paths because resolving the root needs Qt (QSettings + QStandardPaths)
and gui.paths is deliberately Qt-free; the path *helpers* there instead take the
resolved root as a parameter.
"""

from pathlib import Path

from PyQt6.QtCore import QSettings, QStandardPaths

from vodparse.gui import paths as gui_paths
from vodparse.gui.settings import SETTING_EXPORT_DIR
from vodparse.runtime_paths import APP_DIR_NAME, is_frozen


def default_export_root() -> Path:
    """The export root used while ``paths/export_dir`` is unset. Reads
    gui.paths.PROJECT_ROOT through the module so tests that repoint it redirect this
    default too. Qt handles the platform Movies/Videos lookup (Windows Known-Folder
    redirection, localized XDG dirs); the ~/Videos fallback covers the rare platform
    where Qt reports no such location."""
    if not is_frozen():
        return gui_paths.PROJECT_ROOT / "video_out"
    movies = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.MoviesLocation)
    base = Path(movies) if movies else Path.home() / "Videos"
    return base / APP_DIR_NAME


def export_root(settings: QSettings | None = None) -> Path:
    """The live export root: the ``paths/export_dir`` setting when non-empty, else
    :func:`default_export_root`. Read at each use so a settings change takes effect
    immediately; an in-flight export captures the value once at kickoff instead (see
    MainWindow._on_review_export) so its folder, vetting, and argv all agree.

    Always absolute. The setting is free text (the Settings field accepts typing, not just
    Browse) and it reaches the export stages as ``--out-root``, which run with a different
    working directory than the GUI (MainWindow._start_process pins them to PROJECT_ROOT):
    a relative value would otherwise name two different folders, sending the clips
    somewhere the GUI's own export paths never look."""
    configured = ((settings or QSettings()).value(SETTING_EXPORT_DIR, "", type=str) or "").strip()
    if not configured:
        return default_export_root()
    return Path(configured).expanduser().absolute()


def export_root_reachable(root: Path) -> bool:
    """Whether an export can proceed into ``root``: it is a directory, or only its final
    component is missing while its parent exists (so the export stage's mkdir creates
    just the leaf — the frozen first-run case, where ``~/Movies`` exists but the app's
    folder inside it doesn't yet). A deeper-missing chain means the location itself is
    gone, typically an unplugged external drive whose mount point vanished: creating
    parents there would silently rebuild a phantom tree on the boot disk (worst on
    macOS, where a leftover folder at /Volumes/<name> then blocks the real drive from
    mounting under its own name). The export entry points check this and refuse with a
    message instead (see MainWindow._warn_export_root_unavailable)."""
    if root.exists():
        return root.is_dir()
    return root.parent.is_dir()
