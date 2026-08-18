"""Check GitHub Releases for a newer app version (used by the frozen desktop build).

This is the Qt-free core behind Help ▸ Check for Updates. It asks the public
release repo's GitHub Releases API for the latest published release, compares that
release's tag against the running :data:`vodparse.__version__`, and returns a
verdict (:class:`UpdateCheckResult`). It never downloads or installs anything and
never raises: a network/parse failure comes back as ``CHECK_FAILED`` so the GUI can
fail quietly. The GUI glue that runs this off the UI thread and shows the result
(a "Download" link to the release page) lives in ``vodparse.gui.updates``.

Stdlib only, mirroring ``vodparse.cli.download_weights``: the two talk to the same
GitHub Releases API on the same repo (weights *and* app builds ship as release
assets there), so keeping this dependency-free avoids pulling a version-parsing
library into the bundle for a single comparison. The actual GET and the
release-JSON shape check are shared with ``download_weights`` via
``vodparse.github_releases`` (see that module's docstring for the two callers'
differing auth posture — this one never sends a token).

Manual check from a checkout (normally the GUI drives this, frozen only)::

    python -c "from vodparse.update_check import check_for_update; print(check_for_update())"
"""

import http.client
import re
import urllib.error
from dataclasses import dataclass
from enum import Enum

from vodparse import PUBLIC_REPO, __version__, github_releases

# fetch_latest_release/check_for_update default to vodparse.PUBLIC_REPO, the package-level
# single source of truth for the public repo whose GitHub Releases carry the packaged app
# builds (installers ship as assets there, so "the latest release" is the latest app version).
# The app updates against that public repo, not this -dev repo.

_USER_AGENT = "ultimate-vod-studio-update-check"

# One quick GET on a user-initiated action; keep it short so a stalled network doesn't
# leave the "Checking…" dialog hanging. The GUI runs the call off the UI thread regardless.
_TIMEOUT_SEC = 10


class UpdateStatus(Enum):
    """The three outcomes of a check. The GUI branches on this to pick what to show."""

    UP_TO_DATE = "up_to_date"            # running the latest release (or a newer/dev build)
    UPDATE_AVAILABLE = "update_available"  # a newer release exists; release_url points at it
    CHECK_FAILED = "check_failed"        # couldn't reach GitHub or parse the answer; fail quietly


@dataclass(frozen=True)
class UpdateCheckResult:
    """Outcome of :func:`check_for_update`. ``latest_version``/``release_url`` are set when
    known (both on UPDATE_AVAILABLE, latest_version also on UP_TO_DATE); ``error`` carries a
    short human-readable reason only on CHECK_FAILED."""

    status: UpdateStatus
    current_version: str
    latest_version: str | None = None
    release_url: str | None = None
    error: str | None = None


def _parse_version(text: str) -> tuple[int, ...] | None:
    """Parse a ``MAJOR.MINOR.PATCH`` version into a comparable tuple of ints.

    Grammar: an optional leading ``v``/``V`` (release tags are ``vX.Y.Z``), then a
    dotted run of one or more integers, then an optional pre-release/build suffix
    starting with a non-digit, non-dot character (``"1.2.0-rc1" -> (1, 2, 0)``). Returns
    ``None`` when the text doesn't match that shape at all — including a dotted core with
    an empty component, like ``"1.2."`` or ``"1..2"`` — so callers treat an unrecognized
    version as "can't compare" rather than guessing a value. This matches the app's own
    simple versioning scheme (see ``pyproject.toml`` / ``__version__``) rather than
    implementing full PEP 440 ordering, which the release tags never use.
    """
    match = re.fullmatch(r"v?(\d+(?:\.\d+)*)(?:[^\d.].*)?", text.strip(), re.IGNORECASE)
    return tuple(int(part) for part in match.group(1).split(".")) if match else None


def is_newer(latest: str, current: str) -> bool | None:
    """Is release version ``latest`` strictly newer than ``current``?

    Returns ``None`` when either side can't be parsed, so the caller can treat an
    unrecognized version as "don't know" instead of falsely nagging or reassuring.
    Shorter versions are zero-padded, so ``0.9`` and ``0.9.0`` compare equal.
    """
    latest_tuple, current_tuple = _parse_version(latest), _parse_version(current)
    if latest_tuple is None or current_tuple is None:
        return None
    width = max(len(latest_tuple), len(current_tuple))
    latest_tuple += (0,) * (width - len(latest_tuple))
    current_tuple += (0,) * (width - len(current_tuple))
    return latest_tuple > current_tuple


def fetch_latest_release(repo: str = PUBLIC_REPO, *, timeout: int = _TIMEOUT_SEC) -> dict:
    """Return the JSON for the repo's latest published release.

    Uses the unauthenticated ``/releases/latest`` endpoint, which GitHub defines as the
    most recent non-draft, non-prerelease release: exactly the stable build an update
    prompt should point at. Raises ``urllib.error.URLError`` (network) or ``OSError`` /
    ``ValueError`` / ``http.client.HTTPException`` (bad response, e.g. a truncated body,
    or a body that parses as JSON but isn't a release object) on failure;
    :func:`check_for_update` turns those into a CHECK_FAILED result. Public repo, so no
    token: the once-per-click rate is nowhere near GitHub's unauthenticated limit.
    """
    return github_releases.fetch_release_json(repo, user_agent=_USER_AGENT, timeout=timeout)


def check_for_update(current_version: str = __version__,
                     repo: str = PUBLIC_REPO, *,
                     timeout: int = _TIMEOUT_SEC) -> UpdateCheckResult:
    """Compare the repo's latest release against ``current_version`` and return a verdict.

    Never raises: any network error, HTTP error, malformed JSON, or unparseable/absent
    tag comes back as ``CHECK_FAILED`` with a short reason, so the GUI can stay silent on
    an automatic check or show a gentle notice on a manual one. An unparseable *latest*
    tag is a failure (we can't compare), not a false "up to date".
    """
    try:
        release = fetch_latest_release(repo, timeout=timeout)
    except urllib.error.HTTPError as exc:
        return UpdateCheckResult(UpdateStatus.CHECK_FAILED, current_version,
                                 error=f"GitHub returned HTTP {exc.code}.")
    except urllib.error.URLError as exc:
        return UpdateCheckResult(UpdateStatus.CHECK_FAILED, current_version,
                                 error=f"Couldn't reach GitHub ({exc.reason}).")
    except (OSError, ValueError, http.client.HTTPException) as exc:
        # Socket timeout (OSError), bad/empty JSON (ValueError), or a truncated/garbled HTTP
        # body (HTTPException, e.g. IncompleteRead — urllib wraps errors raised while *opening*
        # the connection in URLError, but body-read errors escape unwrapped).
        return UpdateCheckResult(UpdateStatus.CHECK_FAILED, current_version,
                                 error=f"Couldn't read GitHub's response ({exc}).")

    latest_version = release.get("tag_name")
    if not isinstance(latest_version, str) or not latest_version:
        return UpdateCheckResult(UpdateStatus.CHECK_FAILED, current_version,
                                 error="The latest release has no version tag.")

    newer = is_newer(latest_version, current_version)
    if newer is None:
        return UpdateCheckResult(UpdateStatus.CHECK_FAILED, current_version,
                                 latest_version=latest_version,
                                 error=f"Couldn't compare versions ({latest_version!r}).")
    if not newer:
        return UpdateCheckResult(UpdateStatus.UP_TO_DATE, current_version,
                                 latest_version=latest_version)
    release_url = release.get("html_url")
    if not isinstance(release_url, str):
        release_url = None  # the GUI falls back to the releases-page URL
    return UpdateCheckResult(UpdateStatus.UPDATE_AVAILABLE, current_version,
                             latest_version=latest_version,
                             release_url=release_url)
