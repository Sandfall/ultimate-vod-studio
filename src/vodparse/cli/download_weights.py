"""Download the trained model weights from the GitHub Release into weights/.

The weights are not stored in this repository; they ship as assets on a GitHub
Release. This pulls the .pt files attached to a release and drops them into
weights/, where the app loads them (see vodparse.__init__). Run it once after
cloning, before launching the app:

    python -m vodparse.cli.download_weights                  # latest release
    python -m vodparse.cli.download_weights --tag v0.9.0     # a specific release
    python -m vodparse.cli.download_weights --repo owner/name  # a fork / different repo

Stdlib only: no third-party dependencies, so it works before you've installed
anything. For a private repo or to dodge GitHub's unauthenticated API rate limit,
set GITHUB_TOKEN (or pass --token).

Re-running is safe: a weight already present with the expected size is skipped
unless you pass --force. Downloads are size-verified and written atomically, so an
interrupted run never leaves a half-written weight in place.

The GitHub Releases GET and the octet-stream asset download both go through the
shared, stdlib-only ``vodparse.github_releases`` core (also used by
``vodparse.update_check``); see that module's docstring for why this tool passes a
token and the update check never does.
"""

import argparse
import os
import shutil
import sys
import tempfile
import urllib.error
from pathlib import Path

from vodparse import PUBLIC_REPO, github_releases
from vodparse.games import shipped_weight_files

# The public release repo whose GitHub Releases carry the weights. Override with --repo.
# Aliased from the package-level single source of truth (see vodparse.PUBLIC_REPO). This module
# uses only stdlib, but importing vodparse requires the package on the path; run via
# `python -m vodparse.cli.download_weights` (documented above) with src/ on PYTHONPATH or an
# installed package — running by file path on a bare checkout will fail.
DEFAULT_REPO = PUBLIC_REPO
_USER_AGENT = "ultimate-vod-studio-weight-fetcher"

# The weights the app expects in weights/ (used only to warn if a release is missing one),
# as bare filenames since release assets are flat: derived from the game registry
# (vodparse.games) rather than listed here, so a future game's weights need no edit to this
# file. logo.pt (the go/game detector) is intentionally absent: it's no longer distributed.
# The GUI always runs seek mode and never loads it; go/game CLI mode is opt-in and supplies
# its own weight.
EXPECTED = tuple(Path(f).name for f in shipped_weight_files())


def fetch_release(repo: str, tag: str | None, token: str | None) -> dict:
    """Return the release JSON (latest, or a specific tag). Exits with a clear message on failure."""
    try:
        return github_releases.fetch_release_json(repo, user_agent=_USER_AGENT, tag=tag, token=token)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            sys.exit(f"error: no {'release ' + tag if tag else 'latest release'} found for {repo!r}.\n"
                     f"       Check --repo / --tag, or confirm a release with weight assets exists.")
        if e.code in (401, 403):
            sys.exit(f"error: GitHub API returned {e.code} for {repo!r}: rate-limited or private.\n"
                     f"       Set GITHUB_TOKEN (or pass --token).")
        raise
    except urllib.error.URLError as e:
        sys.exit(f"error: could not reach GitHub ({e.reason}). Check your network.")


def download(asset: dict, dest: Path, token: str | None) -> None:
    """Download one release asset to dest atomically, verifying the byte count."""
    # Private repos need the API asset URL + octet-stream Accept; public assets download
    # straight from browser_download_url with no auth. Pick based on whether we have a token.
    url = asset["url"] if token else asset["browser_download_url"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
    os.close(fd)
    try:
        with github_releases.open_github_url(url, user_agent=_USER_AGENT,
                                             accept="application/octet-stream",
                                             token=token) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        size = os.path.getsize(tmp)
        expected = asset.get("size")
        if expected and size != expected:
            raise OSError(f"size mismatch for {asset['name']}: got {size} bytes, expected {expected}")
        os.replace(tmp, dest)  # atomic on the same filesystem
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download model weights from the GitHub Release into weights/.")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"owner/name (default: {DEFAULT_REPO})")
    ap.add_argument("--tag", default=None, help="Release tag to fetch (default: the latest release)")
    ap.add_argument("--dest", default=Path("weights"), type=Path, help="Destination dir (default: weights/)")
    ap.add_argument("--force", action="store_true", help="Re-download even if a file is already present")
    ap.add_argument("--token", default=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"),
                    help="GitHub token for private repos or rate limits (or set GITHUB_TOKEN)")
    args = ap.parse_args(argv)

    release = fetch_release(args.repo, args.tag, args.token)
    assets = [a for a in release.get("assets", []) if a["name"].endswith(".pt")]
    if not assets:
        sys.exit(f"error: release {release.get('tag_name')!r} has no .pt weight assets attached.")

    print(f"Release {release.get('tag_name')} of {args.repo}: {len(assets)} weight file(s)")
    downloaded = 0
    for a in assets:
        dest = args.dest / a["name"]
        if dest.exists() and not args.force and dest.stat().st_size == a.get("size"):
            print(f"  skip   {dest}  (already present)")
            continue
        print(f"  fetch  {a['name']}  ({a.get('size', 0) / 1e6:.1f} MB) ...", flush=True)
        download(a, dest, args.token)
        downloaded += 1

    print(f"\nDone; {downloaded} downloaded into {args.dest}/ ({len(assets) - downloaded} already present).")

    missing = [n for n in EXPECTED if not (args.dest / n).exists()]
    if missing:
        print(f"warning: expected weights still missing from the release: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
