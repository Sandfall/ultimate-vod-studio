# Build the Windows executable for Ultimate VOD Studio.
#
# Run this ON WINDOWS (PyInstaller cannot cross-compile): a Windows PC/VM, or it's invoked
# by .github/workflows/build-windows.yml on a hosted runner. Produces:
# dist\UltimateVODStudio\: folder with "Ultimate VOD Studio.exe" + _internal\
# (no .app: that's macOS-only; wrap this folder in an Inno Setup / NSIS installer to ship)
#
# Usage:  .\build_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Force UTF-8 in every child python: the hosted-runner console is cp1252, and EasyOCR's
# download progress bar prints U+2588 blocks cp1252 can't encode: that crashed the staging
# step mid-download on CI and silently shipped an artifact without the OCR models.
$env:PYTHONUTF8 = "1"

$Venv = ".buildenv"
$Py = Join-Path $Venv "Scripts\python.exe"

# Remove any existing build venv first. `python -m venv` reuses an existing dir rather
# than recreating it, and a venv hardcodes its absolute home path: so a stale .buildenv
# left over from a moved/renamed checkout keeps resolving python/pip against the old path.
# A clean slate avoids it.
Write-Host "==> Creating clean build venv ($Venv)"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $Venv
python -m venv $Venv

# The in-app GPU installer downloads cp313 CUDA wheels (vodparse.gpu_support.WHEEL_PYTHON_TAG);
# an app built on any other Python would ship a GPU install whose extensions can never import.
# Fail the build fast instead of shipping that. Bump this together with the pins below/there.
$PyVersion = & $Py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($PyVersion -ne "3.13") {
    throw "Python 3.13 required (found $PyVersion) - the pinned CUDA wheels are cp313 builds."
}

& $Py -m pip install --upgrade pip wheel

Write-Host "==> Installing the package + GUI deps"
# CPU-only torch first (explicit index) so the editable install sees it satisfied and we never
# pull the multi-GB CUDA build: most end users have no NVIDIA GPU. Users who do can opt into GPU
# acceleration in-app, which downloads the matching CUDA wheels (see vodparse.cli.gpu_support).
# Pin to the SAME versions that in-app installer fetches (vodparse.gpu_support.TORCH_VERSION /
# TORCHVISION_VERSION) so the CUDA swap is an exact, ABI-matched replacement: keep them in step.
& $Py -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
# PyQt6 and its Qt runtime are pinned to the versions the app is developed and tested on.
# pyproject leaves PyQt6 unpinned (right for a pip install), so an unpinned build takes
# whatever is newest that day: in 2026-11 that was PyQt6 6.12.1 on Qt 6.12.0, a Qt that no
# test or Windows check had run on. When bumping: run the tests and check video playback
# on Windows with the new version first, and keep build_mac.sh and
# .github/workflows/ci.yml in step.
& $Py -m pip install -e ".[gui]" "PyQt6==6.11.0" "PyQt6-Qt6==6.11.2" "PyQt6-sip==13.11.1"

# Pin the one dependency whose LICENSING changes with the version. opencv-python vendors an
# FFmpeg build per platform (LGPL-2.1 here, GPL-3.0 in the macOS wheels), and NOTICE's
# "Bundled FFmpeg binaries" section states that per platform, quoting this wheel's configure
# prefixes. pyproject leaves it unpinned (right for a pip install), and what currently holds
# a release build to a 4.x wheel is python-doctr's transitive opencv-python<5.0.0 cap: a
# constraint in someone else's metadata, which a doctr bump can drop without notice. Pin it
# here instead, so NOTICE describes what ships because the build says so.
# When bumping: read the new wheel's configure lines out of licenses\FFMPEG-BUILD-CONFIG.txt
# after a build, and update NOTICE section 3 to match before shipping.
Write-Host "==> Pinning opencv-python to the version NOTICE describes"
# easyocr requires opencv-python-headless, which installs into the same cv2\ folder as
# opencv-python: two wheels, possibly of different versions, overwriting each other's
# files, and a second OpenCV in the license documents that the app never loads. Keep
# exactly one. Removing the headless wheel deletes the cv2\ files the two share, so the
# pinned wheel is then force-reinstalled to put them back. --no-deps: its only
# dependency, numpy, is already installed.
& $Py -m pip uninstall -y opencv-python-headless
& $Py -m pip install --force-reinstall --no-deps "opencv-python==4.13.0.92"
if ($LASTEXITCODE -ne 0) { throw "could not reinstall the pinned opencv-python" }

Write-Host "==> Installing PyInstaller + imageio-ffmpeg"
& $Py -m pip install pyinstaller imageio-ffmpeg

# Stage a static, self-contained ffmpeg.exe so end users need nothing installed. imageio-ffmpeg
# ships a portable ffmpeg for this platform; copy it to vendor\ffmpeg.exe (UltimateVODStudio.spec bundles it
# into bin\; vodparse.export puts that dir first on PATH at runtime).
Write-Host "==> Staging a static ffmpeg binary (vendor\ffmpeg.exe)"
New-Item -ItemType Directory -Force -Path "vendor" | Out-Null
& $Py -c "import shutil, imageio_ffmpeg; shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), 'vendor/ffmpeg.exe')"

# Collect every dependency's license text into the document the app ships. GPL-3.0 (PyQt6), LGPL
# (Qt 6, python-bidi, OpenCV's video backend) and the permissive licenses all require their text to
# accompany the binary. Run from THIS build env so the collected set matches what actually gets
# frozen, rather than whatever a dev shell happens to have resolved.
Write-Host "==> Collecting third-party license texts (licenses\THIRD-PARTY-LICENSES.txt)"
& $Py tools\licensing\license_report.py bundle
if ($LASTEXITCODE -ne 0) { throw "could not collect the third-party license texts" }

# Record what the FFmpeg builds about to be frozen were configured with, and refuse to build
# past an --enable-nonfree one (which could not be redistributed at all). Package metadata says
# nothing about these binaries: the flags are the only evidence of whether a given copy is
# GPL-3.0, GPL-2.0-or-later or LGPL. $ErrorActionPreference doesn't cover a native exit code, so
# check it explicitly.
Write-Host "==> Recording the bundled FFmpeg builds (licenses\FFMPEG-BUILD-CONFIG.txt)"
& $Py tools\licensing\license_report.py ffmpeg-config --also vendor\ffmpeg.exe
if ($LASTEXITCODE -ne 0) { throw "an FFmpeg build in this environment is not redistributable" }

# Stage the EasyOCR weights into the bundle so the app is fully offline: no first-run download.
# Pointing EASYOCR_MODULE_PATH at vendor\easyocr makes the Reader download exactly the two weights
# Reader(['en']) needs into vendor\easyocr\model\; UltimateVODStudio.spec bundles them and freeze_entry sets
# the same env var at runtime.
Write-Host "==> Staging EasyOCR weights (vendor\easyocr) for offline use (~94MB download, one time)"
New-Item -ItemType Directory -Force -Path "vendor\easyocr" | Out-Null
$env:EASYOCR_MODULE_PATH = (Join-Path (Get-Location) "vendor\easyocr")
try {
    # verbose=False suppresses the per-block progress bar (thousands of lines in a CI log).
    & $Py -c "import easyocr; easyocr.Reader(['en'], verbose=False)"
} catch {
    Write-Host "    (EasyOCR staging failed, not fatal; the app will fetch on first run)"
}
# A failing native command doesn't trip the catch above, so verify the result directly:
# Reader(['en']) needs the detector + recognizer, i.e. at least two .pth files.
$stagedOcr = @(Get-ChildItem "vendor\easyocr\model\*.pth" -ErrorAction SilentlyContinue)
if ($stagedOcr.Count -lt 2) {
    Write-Host "    WARNING: EasyOCR models not staged ($($stagedOcr.Count) .pth); the app will download on first run"
}
Remove-Item Env:EASYOCR_MODULE_PATH

# Stage the docTR text-engine models (DBNet + PARSeq) the same way: point docTR's cache at
# vendor\doctr and let the fetch stage download into vendor\doctr\models\. UltimateVODStudio.spec
# bundles that dir; freeze_entry points DOCTR_CACHE_DIR at the bundled copy at runtime. Safe to
# re-run: docTR skips models already in the cache. TQDM_DISABLE keeps docTR's per-block
# progress bars (thousands of lines) out of the CI log, like verbose=False does for EasyOCR.
Write-Host "==> Staging docTR weights (vendor\doctr) for offline use (~190MB download, one time)"
New-Item -ItemType Directory -Force -Path "vendor\doctr" | Out-Null
$env:DOCTR_CACHE_DIR = (Join-Path (Get-Location) "vendor\doctr")
$env:TQDM_DISABLE = "1"
try {
    & $Py -m vodparse.cli.doctr_support fetch
} catch {
    Write-Host "    (docTR staging failed, not fatal; the app will offer the download on first use)"
}
# A failing native command doesn't trip the catch above, so verify the result directly:
# the engine needs the detector + recognizer, i.e. at least two .pt files.
$stagedDoctr = @(Get-ChildItem "vendor\doctr\models\*.pt" -ErrorAction SilentlyContinue)
if ($stagedDoctr.Count -lt 2) {
    Write-Host "    WARNING: docTR models not staged ($($stagedDoctr.Count) .pt); the app will offer the download on first use"
}
Remove-Item Env:TQDM_DISABLE
Remove-Item Env:DOCTR_CACHE_DIR

# Download the model weights (vodparse.games.shipped_weight_files) from the GitHub Release. weights\
# itself is gitignored, so a fresh checkout (and every CI runner) starts with nothing there:
# without this step UltimateVODStudio.spec silently skips every missing weight and ships an
# app that can't detect or read anything. Stdlib-only, so it runs fine with the build venv's
# python. Set GITHUB_TOKEN to pull from a private release when testing before the repo is public.
# No --repo needed: download_weights' own default is vodparse.PUBLIC_REPO, so the slug
# lives in exactly one place here.
Write-Host "==> Downloading model weights (weights\) from the GitHub Release"
& $Py -m vodparse.cli.download_weights
if ($LASTEXITCODE -ne 0) {
    Write-Error "Weight download failed (vodparse.cli.download_weights exited $LASTEXITCODE): aborting build."
    exit 1
}

Write-Host "==> Freezing with PyInstaller"
& $Py -m PyInstaller --noconfirm --clean UltimateVODStudio.spec

Write-Host ""
Write-Host "==> Done. Build is at: dist\UltimateVODStudio\"
Write-Host "    Launch:   dist\UltimateVODStudio\'Ultimate VOD Studio.exe'"
Write-Host "    Workspace (json_output\, drafts) goes to: %LOCALAPPDATA%\Ultimate VOD Studio"
Write-Host "    Exported clips default to: %USERPROFILE%\Videos\Ultimate VOD Studio"
Write-Host "    Override the workspace with:  `$env:ULTIMATE_VOD_WORKSPACE = 'C:\path'"
