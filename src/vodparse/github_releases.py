"""Qt-free, stdlib-only core for talking to the GitHub Releases API.

Two callers share this: ``vodparse.update_check`` (the Help > Check for Updates
GUI feature) and ``vodparse.cli.download_weights`` (the weight-fetching CLI). Both
hit the same public repo's Releases API, so the request-building and release-JSON
fetch logic lives here once instead of twice.

The two callers deliberately differ on authentication, and that difference is
preserved, not "fixed": ``update_check`` never passes a token, so its check stays
one unauthenticated GET with a generic User-Agent — a documented privacy posture,
and the once-per-click rate is nowhere near GitHub's unauthenticated limit anyway.
``download_weights`` accepts ``--token``/``GITHUB_TOKEN`` because a first-run
weight fetch can burn through the unauthenticated rate limit or need to reach a
private repo. This module supports both: :func:`open_github_url` and
:func:`fetch_release_json` take an optional ``token`` that a caller either passes
or omits, and a required ``user_agent`` so each caller keeps identifying itself
by its own name in GitHub's logs.
"""

import json
import urllib.request
from http.client import HTTPResponse

from vodparse import net

# Base URL for the GitHub REST API. A module-level global (not a default parameter
# value) so tests can redirect it to a local socket via monkeypatch and have that
# redirect actually take effect on the next call (default parameter values are
# bound once, at function-definition time, and wouldn't see a later monkeypatch).
API = "https://api.github.com"


def open_github_url(
    url: str,
    *,
    user_agent: str,
    accept: str,
    token: str | None = None,
    timeout: float | None = None,
) -> HTTPResponse:
    """Open a GitHub API or asset URL and return the live HTTP response.

    Builds the two headers every caller needs (``User-Agent``, ``Accept``) and, if
    ``token`` is given, an ``Authorization: Bearer`` header — used for private-repo
    access and to dodge GitHub's unauthenticated rate limit, never sent by the
    unauthenticated update check (see module docstring). ``timeout`` is seconds, or
    ``None`` to use urllib's default (effectively no timeout) — the update check
    passes a short timeout so a stalled network doesn't hang its "Checking…"
    dialog; the weight-download path passes none, matching its long-standing
    behavior for large asset downloads.

    Returns the still-open response (caller is responsible for closing it, e.g. via
    a ``with`` block). Raises ``urllib.error.URLError`` (including its
    ``HTTPError`` subclass) on a network or HTTP-status failure.
    """
    request = urllib.request.Request(url, headers={
        "User-Agent": user_agent,
        "Accept": accept,
    })
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    # Verify against the system roots plus certifi's bundle (see vodparse.net), so the
    # update check and weight fetch don't depend on what the OS trust store has cached.
    context = net.make_ssl_context()
    if timeout is None:
        return urllib.request.urlopen(request, context=context)
    return urllib.request.urlopen(request, timeout=timeout, context=context)


def fetch_release_json(
    repo: str,
    *,
    user_agent: str,
    tag: str | None = None,
    token: str | None = None,
    timeout: float | None = None,
) -> dict:
    """Return the JSON for one release of ``repo``: the latest, or ``tag`` if given.

    Hits the unauthenticated-by-default ``/releases/latest`` endpoint (GitHub's
    "most recent non-draft, non-prerelease release") or, when ``tag`` is given,
    ``/releases/tags/{tag}`` for a specific one. See :func:`open_github_url` for
    ``token``/``timeout`` semantics.

    Raises ``urllib.error.URLError`` (network, or its ``HTTPError`` subclass for a
    bad HTTP status) or ``OSError`` / ``ValueError`` / ``http.client.HTTPException``
    (bad response: a truncated body, or a body that parses as JSON but isn't a
    release object — e.g. a list) on failure. Callers decide how to turn that into
    UX: ``update_check.check_for_update`` never raises past itself and folds every
    one of these into a quiet ``CHECK_FAILED`` result;
    ``download_weights.fetch_release`` prints a message and exits on the
    HTTP/network cases and lets the shape/parse ones propagate.
    """
    ref = f"tags/{tag}" if tag else "latest"
    url = f"{API}/repos/{repo}/releases/{ref}"
    with open_github_url(url, user_agent=user_agent, accept="application/vnd.github+json",
                          token=token, timeout=timeout) as response:
        body = json.load(response)
    if not isinstance(body, dict):
        # Both callers need this: a body that parses as JSON but isn't the expected
        # release object (a list, string, number, ...) must become a caught
        # exception here, not an AttributeError/KeyError later when the caller
        # treats it like a release mapping (see docstring for who catches what).
        raise ValueError("GitHub's response wasn't a release object.")
    return body
