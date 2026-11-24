#!/usr/bin/env bash
# Build the macOS app bundle for Ultimate VOD Studio.
#
# Run this ON A MAC (PyInstaller cannot cross-compile). Produces:
# "dist/Ultimate VOD Studio.app": the double-clickable bundle
# dist/UltimateVODStudio/: the same payload as a plain folder
#
# Usage:  ./build_mac.sh [--allow-weightless]
set -euo pipefail

cd "$(dirname "$0")"

# --allow-weightless downgrades the OCR-weight verification gate (below the staging steps)
# from a hard failure to a warning, so the build continues even if EasyOCR/docTR weights
# failed to download. A command-line flag rather than an env var on purpose: an env var can
# be exported once in a shell profile and silently defeat the gate on every future build,
# where a flag has to be typed fresh for the one build it applies to.
ALLOW_WEIGHTLESS=0
for arg in "$@"; do
    case "$arg" in
        --allow-weightless)
            ALLOW_WEIGHTLESS=1
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: $0 [--allow-weightless]" >&2
            exit 1
            ;;
    esac
done

PY="${PYTHON:-python3}"
VENV=".buildenv"

# Remove any existing build venv first. `python -m venv` reuses an existing dir rather
# than recreating it, and a venv's scripts hardcode absolute-path shebangs: so a stale
# .buildenv left over from a moved/renamed checkout keeps pointing pip/python at the old
# path ("bad interpreter: .../bin/py: No such file or directory"). A clean slate avoids it.
echo "==> Creating clean build venv ($VENV)"
rm -rf "$VENV"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

# torch first, pinned, so the editable install below sees it satisfied instead of pulling
# the newest release. Pinned to the versions uv.lock resolves: an unpinned install drifts
# (it resolved torch 2.14.0 / torchvision 0.29.0 in 2026-09 against a locked 2.12.0 /
# 0.27.0), and that drift is what renamed torchvision's native libraries out from under
# PyInstaller's hooks. When bumping, bump uv.lock with it and re-run a frozen analysis.
# Default PyPI torch wheels on macOS already include MPS (Apple GPU) support: do NOT
# use the CPU-only index here, that would disable MPS. The code auto-selects MPS/CPU.
echo "==> Installing torch / torchvision (macOS wheels, pinned to uv.lock)"
pip install "torch==2.12.0" "torchvision==0.27.0"

# PyQt6 and its Qt runtime are pinned to the versions the app is developed and tested on.
# pyproject leaves PyQt6 unpinned (right for a pip install), so an unpinned build takes
# whatever is newest that day: in 2026-11 that was PyQt6 6.12.1 on Qt 6.12.0, a Qt that no
# test had run on. When bumping: run the tests and a frozen analysis on the new version
# first, and keep build_windows.ps1 and .github/workflows/ci.yml in step.
echo "==> Installing the package + GUI deps (PyQt6 pinned)"
pip install -e ".[gui]" "PyQt6==6.11.0" "PyQt6-Qt6==6.11.2" "PyQt6-sip==13.11.1"

# Pin the one dependency whose LICENSING changes with the version. opencv-python's macOS
# wheels vendor a Homebrew-built FFmpeg configured --enable-gpl --enable-version3, which
# makes the OpenCV video backend GPL-3.0 on macOS and LGPL-2.1 on Windows: NOTICE's
# "Bundled FFmpeg binaries" section states that per platform, quoting this wheel's configure
# prefixes. pyproject leaves it unpinned (right for a pip install), and what currently holds
# a release build to a 4.x wheel is python-doctr's transitive opencv-python<5.0.0 cap: a
# constraint in someone else's metadata, which a doctr bump can drop without notice. Pin it
# here instead, so NOTICE describes what ships because the build says so.
# When bumping: read the new wheel's configure lines out of licenses/FFMPEG-BUILD-CONFIG.txt
# after a build, and update NOTICE section 3 to match before shipping.
echo "==> Pinning opencv-python to the version NOTICE describes"
# easyocr requires opencv-python-headless, which installs into the same cv2/ folder as
# opencv-python: two wheels, possibly of different versions, overwriting each other's
# files, and a second OpenCV in the license documents that the app never loads. Keep
# exactly one. Removing the headless wheel deletes the cv2/ files the two share, so the
# pinned wheel is then force-reinstalled to put them back. --no-deps: its only
# dependency, numpy, is already installed.
pip uninstall -y opencv-python-headless
pip install --force-reinstall --no-deps "opencv-python==4.13.0.92"

echo "==> Installing PyInstaller"
pip install pyinstaller

# Stage a static, self-contained ffmpeg binary so end users need nothing installed.
# imageio-ffmpeg ships a portable, statically-linked ffmpeg for this platform/arch; we copy
# it to vendor/ffmpeg (named exactly "ffmpeg" so ffmpeg-python finds it on PATH). UltimateVODStudio.spec
# bundles it; vodparse.ffmpeg_runtime puts the bundle dir first on PATH at runtime.
echo "==> Staging a static ffmpeg binary (vendor/ffmpeg)"
pip install imageio-ffmpeg
mkdir -p vendor
python -c "import shutil, imageio_ffmpeg; shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), 'vendor/ffmpeg')"
chmod +x vendor/ffmpeg

# Collect every dependency's license text into the document the app ships. GPL-3.0 (PyQt6),
# LGPL (Qt 6, python-bidi, OpenCV's video backend) and the permissive licenses all require
# their text to accompany the binary. Run from THIS .buildenv so the collected set matches
# what actually gets frozen, rather than whatever a dev shell happens to have resolved.
echo "==> Collecting third-party license texts (licenses/THIRD-PARTY-LICENSES.txt)"
python tools/licensing/license_report.py bundle

# Record what the FFmpeg builds about to be frozen were configured with, and refuse to
# build past an --enable-nonfree one (which could not be redistributed at all). Package
# metadata says nothing about these binaries: the flags are the only evidence of whether a
# given copy is GPL-3.0, GPL-2.0-or-later or LGPL. Under `set -e` a nonfree build stops
# here rather than shipping.
echo "==> Recording the bundled FFmpeg builds (licenses/FFMPEG-BUILD-CONFIG.txt)"
python tools/licensing/license_report.py ffmpeg-config --also vendor/ffmpeg

# Stage the EasyOCR weights into the bundle so the app is fully offline: no first-run
# download. Pointing EASYOCR_MODULE_PATH at vendor/easyocr makes the Reader download exactly
# the two weights Reader(['en']) needs (craft_mlt_25k.pth + english_g2.pth, ~94MB) straight
# into vendor/easyocr/model/. UltimateVODStudio.spec bundles that dir; freeze_entry sets the same env var
# at runtime so the frozen Reader loads them locally instead of fetching.
echo "==> Staging EasyOCR weights (vendor/easyocr) for offline use (~94MB download, one time)"
mkdir -p vendor/easyocr
EASYOCR_MODULE_PATH="$PWD/vendor/easyocr" python -c "import easyocr; easyocr.Reader(['en'])" || \
  echo "    (EasyOCR staging failed, not fatal; the app will fetch on first run)"

# Stage the docTR text-engine models (DBNet + PARSeq) the same way: point docTR's cache at
# vendor/doctr and let the fetch stage download into vendor/doctr/models/. UltimateVODStudio.spec
# bundles that dir; freeze_entry points DOCTR_CACHE_DIR at the bundled copy at runtime. Safe to
# re-run: docTR skips models already in the cache.
echo "==> Staging docTR weights (vendor/doctr) for offline use (~190MB download, one time)"
mkdir -p vendor/doctr
DOCTR_CACHE_DIR="$PWD/vendor/doctr" python -m vodparse.cli.doctr_support fetch || \
  echo "    (docTR staging failed, not fatal; the app will offer the download on first use)"

# Verify the OCR weights actually landed before freezing. Both staging steps above are
# warn-and-continue (`|| echo ...`) so a transient network/TLS failure doesn't abort the
# whole build over a component with a first-run download fallback — but nothing downstream
# checks that the fallback ever gets exercised safely: UltimateVODStudio.spec silently skips
# a missing vendor/ directory, and while CI builds re-check the frozen bundle afterward
# (.github/workflows/build-mac-dev.yml), a local run of this script has no second line of
# defense. Without this gate, a TLS hiccup during staging
# ships a weight-less .app that looks identical to a good build until a user's first
# analysis fails offline — the same failure mode that shipped a weight-less Windows build
# in 2026-07 before build_windows.ps1 / build-windows.yml gated it in CI. The expected
# counts mirror that CI check: EasyOCR needs its CRAFT detector + English recognizer
# (>= 2 .pth), docTR needs its DBNet detector + PARSeq recognizer (>= 2 .pt).
echo "==> Verifying staged OCR weights"
# The directory checks are load-bearing under `set -euo pipefail`: a totally failed staging
# step can leave the model directory missing entirely, and `find` on a missing path exits
# non-zero, which pipefail would turn into a silent script abort — before the gate below
# ever gets to say what is wrong (or honor --allow-weightless).
EASYOCR_WEIGHT_COUNT=0
if [ -d vendor/easyocr/model ]; then
    EASYOCR_WEIGHT_COUNT=$(find vendor/easyocr/model -maxdepth 1 -name '*.pth' | wc -l | tr -d ' ')
fi
DOCTR_WEIGHT_COUNT=0
if [ -d vendor/doctr/models ]; then
    DOCTR_WEIGHT_COUNT=$(find vendor/doctr/models -maxdepth 1 -name '*.pt' | wc -l | tr -d ' ')
fi

if [ "$EASYOCR_WEIGHT_COUNT" -lt 2 ] || [ "$DOCTR_WEIGHT_COUNT" -lt 2 ]; then
    if [ "$ALLOW_WEIGHTLESS" -eq 1 ]; then
        echo "=================================================================="
        echo "==> WARNING: shipping a WEIGHT-LESS bundle (--allow-weightless)"
        echo "==>   EasyOCR weights: $EASYOCR_WEIGHT_COUNT/2 present (vendor/easyocr/model/*.pth)"
        echo "==>   docTR weights:   $DOCTR_WEIGHT_COUNT/2 present (vendor/doctr/models/*.pt)"
        echo "==>   The app will try to download whatever is missing on first use, and"
        echo "==>   that download fails outright if the end user is offline or their"
        echo "==>   certificate store can't verify the download host."
        echo "==>   DO NOT distribute this build to end users."
        echo "=================================================================="
    else
        echo "=================================================================="
        echo "==> ERROR: OCR weights are missing; refusing to freeze a weight-less bundle"
        echo "=================================================================="
        if [ "$EASYOCR_WEIGHT_COUNT" -lt 2 ]; then
            echo "    EasyOCR: found $EASYOCR_WEIGHT_COUNT/2 weights in vendor/easyocr/model/*.pth"
        fi
        if [ "$DOCTR_WEIGHT_COUNT" -lt 2 ]; then
            echo "    docTR:   found $DOCTR_WEIGHT_COUNT/2 weights in vendor/doctr/models/*.pt"
        fi
        echo
        echo "    Likely cause: a network or TLS failure during the staging steps above."
        echo "    Fix: restore network access and re-run ./build_mac.sh — staging is cached"
        echo "    in vendor/, so a re-run only downloads whatever is still missing."
        echo
        echo "    To ship anyway (NOT recommended): re-run with --allow-weightless."
        exit 1
    fi
fi

# Download the model weights (vodparse.games.shipped_weight_files) from the GitHub Release. weights/
# itself is gitignored, so a fresh checkout starts with nothing there: without this step
# UltimateVODStudio.spec silently skips every missing weight and ships an app that can't
# detect or read anything. Stdlib-only, so it runs fine with the build venv's python; `set -e`
# above already aborts the script if it exits nonzero. Set GITHUB_TOKEN to pull from a private
# release when testing before the repo is public. No --repo needed: download_weights'
# own default is vodparse.PUBLIC_REPO, so the slug lives in exactly one place here.
echo "==> Downloading model weights (weights/) from the GitHub Release"
python -m vodparse.cli.download_weights

echo "==> Freezing with PyInstaller"
pyinstaller --noconfirm --clean UltimateVODStudio.spec

echo
echo "==> Done. Bundle is at: \"dist/Ultimate VOD Studio.app\""
echo "    Launch from a terminal to see logs:  \"dist/Ultimate VOD Studio.app/Contents/MacOS/Ultimate VOD Studio\""
echo "    Workspace (json_output/, drafts) goes to: ~/Library/Application Support/Ultimate VOD Studio"
echo "    Exported clips default to: ~/Movies/Ultimate VOD Studio"
echo "    Override the workspace with:  ULTIMATE_VOD_WORKSPACE=/path open \"dist/Ultimate VOD Studio.app\""
