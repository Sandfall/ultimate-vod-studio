"""vodparse: shared runtime library for the tournament match-analysis pipeline.

This package holds the code that ships inside the eventual distributed
executable (currently the timer OCR helpers in `ocr_timer`). Dev-time tooling
that builds models lives outside the package, under `tools/`.

Naming convention: public (non-underscore) names are the package's internal
API: shared freely across vodparse modules and the app's scripts, but not a
stability promise to external consumers. `_underscore` names are private to
their defining module.
"""

from vodparse.games import ULTIMATE_WEIGHTS as _ULTIMATE_WEIGHTS
from vodparse.games import weight_path as _weight_path
from vodparse.runtime_paths import bundle_dir as _bundle_dir

# Frozen-safe source of truth for the app version — read directly by the GUI's About
# dialog (importlib.metadata is unreliable in the PyInstaller build). Keep this in sync
# with `version` in pyproject.toml.
__version__ = "1.0.0"

# Single source of truth for the public (OSS release) repo, as `owner/name`. The trained
# weights and the packaged app builds both ship as this repo's GitHub Releases, so it's read
# by the weight fetcher (vodparse.cli.download_weights), the update check (vodparse.update_check),
# and the About box's project link (vodparse.gui.dialogs). This is deliberately NOT the private
# `-dev` repo. Rename the OSS repo here and all three follow.
PUBLIC_REPO = "Sandfall/ultimate-vod-studio"

# Resolved against the bundle root so the weights load no matter the working directory:
# a `weights/` dir at the repo root in a checkout (populate it from the GitHub Release), or
# `sys._MEIPASS/weights/` when frozen. Flat names: the on-disk training-run layout
# (runs/.../detect_vN/weights/best.pt) is an internal artifact, not the shipped interface.
# Stored as strings since ultralytics/torch accept path-likes either way.
_BUNDLE = _bundle_dir()
LOGO_MODEL_PATH        = str(_BUNDLE / "weights/logo.pt")
# Ultimate's detection weights: declared once on its profile (vodparse.games), re-exported
# here under the names the tools and tests have always imported. Other games' weights are
# reached through their profile only.
# (Every profile field is optional because another game may ship only some of the three;
# Ultimate always declares all of them.)
assert _ULTIMATE_WEIGHTS.timer is not None
TIMER_MODEL_PATH       = _weight_path(_ULTIMATE_WEIGHTS.timer)
assert _ULTIMATE_WEIGHTS.character is not None
CHARACTER_MODEL_PATH   = _weight_path(_ULTIMATE_WEIGHTS.character)
# Fine-grained m_char classifier that cascades after the character detector. The pipeline
# no-ops when it's absent.
assert _ULTIMATE_WEIGHTS.m_char is not None
M_CHAR_CLASSIFIER_PATH = _weight_path(_ULTIMATE_WEIGHTS.m_char)
