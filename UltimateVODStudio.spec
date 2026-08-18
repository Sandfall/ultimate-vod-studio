# PyInstaller spec for the Ultimate VOD Studio desktop app.
#
# Build: pyinstaller UltimateVODStudio.spec (run on the TARGET OS: no cross-compiling)
#   Output: dist/UltimateVODStudio/           (onedir; on macOS also "dist/Ultimate VOD Studio.app")
#
# onedir, not onefile: a ~700MB–1GB onefile re-unpacks to a temp dir on every launch
# (10–30s startup). onedir launches fast and is what you wrap in a .dmg / installer.

import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH) # noqa: F821 - SPECPATH is injected by PyInstaller


def _app_version():
    """The version string from pyproject.toml, so the .app plist stays in sync with the package.

    Falls back to "0.0.0" if the field can't be read — matches macOS's own default so a
    missing version is at least obvious rather than crashing the build.
    """
    try:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
        if m:
            return m.group(1)
    except OSError:
        pass
    return "0.0.0"


APP_VERSION = _app_version()


def _exists(rel):
    """(src, dest_dir) tuple for a bundled data file, or None if it's absent."""
    p = ROOT / rel
    if not p.exists():
        print(f"[UltimateVODStudio.spec] WARNING: bundled file missing, skipping: {rel}")
        return None
    return (str(p), str(Path(rel).parent) or ".")


# --- Read-only resources shipped inside the bundle -------------------------------------
# Weights ship flat under weights/: the layout vodparse.__init__ resolves via bundle_dir().
# _exists() preserves each path's parent dir, so these land in <bundle>/weights/.
_data_specs = [
    # weights/logo.pt (go/game detector) is intentionally NOT bundled: it's no longer
    # distributed and the GUI never loads it (always seek mode).
    "weights/timer.pt",
    "weights/character.pt",
    "weights/m_char.pt",
    "assets/app_icon.png",
    # Licensing payload, all of which must ACCOMPANY the binary rather than merely exist
    # in the repo: GPL-3.0 (PyQt6, the bundled ffmpeg), LGPL (Qt 6, python-bidi, OpenCV's
    # video backend), and the permissive licenses' copyright notices.
    #
    #   LICENSE / NOTICE ............ this project's own terms, at the bundle root
    #   licenses/THIRD-PARTY-*.txt .. every dependency's text, written by
    #                                 tools/licensing/license_report.py before PyInstaller
    #                                 runs; Help ▸ About ▸ Licenses opens it
    #   licenses/FFMPEG-BUILD-*.txt . the configure line of the ffmpeg actually staged,
    #                                 recorded by the build scripts (the FFmpeg builds are
    #                                 not Python packages, so nothing else describes them)
    #
    # _exists() warns and skips when one is absent, so a dev freeze still runs; CI refuses
    # to publish an artifact missing them (.github/workflows/build-windows.yml).
    "LICENSE",
    "NOTICE",
    "licenses/THIRD-PARTY-LICENSES.txt",
    "licenses/FFMPEG-BUILD-CONFIG.txt",
    # Tracked, not generated: NOTICE names this file as where torchvision's BSD-3-Clause
    # notice is retained for the m_char classifier's fine-tuned weights, so it has to ship.
    "licenses/torchvision-LICENSE.txt",
]
datas = [d for d in (_exists(r) for r in _data_specs) if d]

binaries = []
# Everything freeze_entry dispatches to: the vodparse.cli.* stages and the vodparse.gui.*
# desktop app: rides along under collect_submodules("vodparse"); no bare entries needed.
hiddenimports = collect_submodules("vodparse") + [
    "pandas",  # imported lazily in vodparse.debug: name it so the freeze always includes it
]

# These load submodules / data files dynamically, so static analysis misses them.
for pkg in ("ultralytics", "easyocr", "doctr"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
# torch, torchvision, cv2 and PyQt6 are covered by PyInstaller's bundled hooks.

# Bundle a static ffmpeg binary so the app is self-contained (clip export shells out to it
# via ffmpeg-python; vodparse.ffmpeg_runtime puts the bundle dir on PATH). The build scripts stage it
# from the imageio-ffmpeg package: vendor/ffmpeg on macOS/Linux, vendor/ffmpeg.exe on Windows.
# If absent, the build still works but relies on a system ffmpeg at runtime.
_ffmpeg = ROOT / "vendor" / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
if _ffmpeg.exists():
    # Into a "bin/" subdir, NOT the bundle root: the root already holds the `ffmpeg` Python
    # package (ffmpeg-python), and a binary named `ffmpeg` there would shadow it and break
    # `import ffmpeg`. vodparse.ffmpeg_runtime adds this bin/ dir to PATH.
    binaries.append((str(_ffmpeg), "bin"))
else:
    print("[UltimateVODStudio.spec] NOTE: vendor/ffmpeg not found; app will rely on a system ffmpeg.")

# Bundle the EasyOCR weights so the app runs fully offline (no first-run download). build_mac.sh
# stages them to vendor/easyocr/model/; freeze_entry points EASYOCR_MODULE_PATH at this dir so
# the Reader loads them locally. If absent, the build works but downloads on first analysis.
_easyocr_models = ROOT / "vendor" / "easyocr" / "model"
if _easyocr_models.is_dir():
    for _w in _easyocr_models.glob("*.pth"):
        datas.append((str(_w), "easyocr_models/model"))
else:
    print("[UltimateVODStudio.spec] NOTE: vendor/easyocr/model not found; app will download EasyOCR weights on first run.")

# Likewise the docTR text-engine models (DBNet + PARSeq). The build scripts stage them to
# vendor/doctr/models/ via `vodparse.cli.doctr_support fetch`; freeze_entry points
# DOCTR_CACHE_DIR at the bundled copy: the "doctr_models/models" destination matches
# docTR's <cache>/models/<file> layout. If absent, the app offers a download on first use.
_doctr_models = ROOT / "vendor" / "doctr" / "models"
if _doctr_models.is_dir():
    for _w in _doctr_models.glob("*.pt"):
        datas.append((str(_w), "doctr_models/models"))
else:
    print("[UltimateVODStudio.spec] NOTE: vendor/doctr/models not found; app will download docTR models on first use.")

a = Analysis(
    ["freeze_entry.py"],
    pathex=[str(ROOT), str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Trim dev-only weight that would otherwise be pulled in transitively.
    excludes=["matplotlib.tests", "PyQt6.Qt5", "tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure)  # noqa: F821

# EXE icon: Windows PyInstaller requires a .ico (a .png errors out); macOS/Linux take the .png.
if sys.platform == "win32":
    _ico = ROOT / "assets" / "app_icon.ico"
    _exe_icon = str(_ico) if _ico.exists() else None
else:
    _exe_icon = str(ROOT / "assets" / "app_icon.png")

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    # The user-visible executable name: Windows users double-click this file directly, so it
    # carries the app's real name, spaces and all — as does the macOS .app bundle (BUNDLE
    # below). Only the plain COLLECT folder stays space-free for the terminal.
    name="Ultimate VOD Studio",
    console=False,  # GUI app: no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,  # set to "arm64" / "x86_64" / "universal2" if you need a specific slice
    icon=_exe_icon,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="UltimateVODStudio",
)

# macOS: wrap the onedir output in a .app bundle.
if sys.platform == "darwin":
    app = BUNDLE(  # noqa: F821
        coll,
        # The bundle filename is the app's Finder label, so it carries the real name too
        # (terminal users quote the path; Finder users never see the difference).
        name="Ultimate VOD Studio.app",
        icon=str(ROOT / "assets" / "app_icon.png"),
        bundle_identifier="app.ultimatevodstudio",
        info_plist={
            "NSHighResolutionCapable": True,
            # Dock / menu-bar display name.
            "CFBundleName": "Ultimate VOD Studio",
            "CFBundleDisplayName": "Ultimate VOD Studio",
            # Version shown in Finder → Get Info and the About panel; sourced from pyproject.toml
            # so there's a single source of truth. Without these, macOS defaults to 0.0.0.
            "CFBundleShortVersionString": APP_VERSION,
            "CFBundleVersion": APP_VERSION,
            # The app shells out to itself; let it spawn child processes freely.
            "LSBackgroundOnly": False,
        },
    )
