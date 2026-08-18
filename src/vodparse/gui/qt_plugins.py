"""Qt plugin path setup shared by the PyQt6 apps (``app.py``, ``upload_assistant.py``).

``configure_qt_plugins()`` must run **before** any PyQt6 widget module is imported, so each
app calls it at the top of its file (right after importing this module). Kept in one place so
the conda-Qt5 shadowing fix can't drift between the two apps.
"""

import os
import sys
from pathlib import Path

import PyQt6


def configure_qt_plugins() -> None:
    """Point Qt at PyQt6's bundled plugins.

    In a conda env that also ships Qt5, Qt5's plugins live on
    the default search path and shadow PyQt6's, so the app can't find a compatible
    platform plugin and aborts on startup. Setting these before QApplication is
    created forces the correct ones. ``setdefault`` lets the user override.
    """
    platforms = Path(PyQt6.__file__).parent / "Qt6" / "plugins" / "platforms"
    if platforms.is_dir():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platforms))
        os.environ.setdefault("QT_PLUGIN_PATH", str(platforms.parent))

    # On macOS the default FFmpeg media backend renders black into QVideoWidget (audio still
    # plays); the native AVFoundation ("darwin") backend renders correctly. setdefault lets
    # the user override.
    if sys.platform == "darwin":
        os.environ.setdefault("QT_MEDIA_BACKEND", "darwin")
