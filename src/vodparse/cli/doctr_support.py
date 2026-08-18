"""Download (or probe) the docTR overlay-OCR models: the headless backend.

The GUI shells out to this stage so it never imports doctr/torch itself: ``probe``
answers whether the docTR engine is usable, and ``fetch`` pre-downloads the pinned
detection + recognition models into docTR's cache so an analysis run never stalls on
a silent mid-run download. The build scripts reuse ``fetch`` (with ``DOCTR_CACHE_DIR``
pointed at ``vendor/doctr``) to stage the models the frozen app bundles. The pinned
architectures, the cache location, and the availability helpers live in
:mod:`vodparse.ocr_engine`.

Subcommands::

    python -m vodparse.cli.doctr_support probe   # one JSON line: {"installed": ..., "weights_cached": ...}
    python -m vodparse.cli.doctr_support fetch   # download the pinned models into the docTR cache

``fetch`` is safe to re-run: docTR skips models already in its cache. Reproduce a
manual model download from a clean checkout::

    python -m vodparse.cli.doctr_support fetch
"""

from __future__ import annotations

import argparse
import json
import sys

from vodparse.ocr_engine import (DOCTR_DET_ARCH, DOCTR_RECO_ARCH, DOCTR_WEIGHTS_MB_APPROX,
                                 doctr_available, doctr_cache_dir, doctr_weights_cached)


def probe() -> int:
    """Print the docTR engine's availability as a single JSON line (consumed by the GUI):
    whether the optional package is installed, and whether the pinned models are already
    in the cache. Metadata/filesystem checks only; nothing heavy is imported."""
    print(json.dumps({"installed": doctr_available(),
                      "weights_cached": doctr_weights_cached()}), flush=True)
    return 0


def fetch() -> int:
    """Download the pinned docTR models by constructing the predictor, which makes docTR
    fetch any missing weights into its cache. Returns a process exit code."""
    try:
        from doctr.models import ocr_predictor
    except ImportError:
        print("error: python-doctr is not installed in this environment. "
              "Install it with: pip install python-doctr", file=sys.stderr)
        return 2

    print(f"Fetching docTR models ({DOCTR_DET_ARCH} + {DOCTR_RECO_ARCH}, "
          f"~{DOCTR_WEIGHTS_MB_APPROX} MB on first download)...", flush=True)
    try:
        ocr_predictor(det_arch=DOCTR_DET_ARCH, reco_arch=DOCTR_RECO_ARCH, pretrained=True)
    except Exception as err:  # noqa: BLE001 - report the failure, exit non-zero for the GUI
        print(f"error: docTR model download failed: {err}", file=sys.stderr)
        return 1

    if not doctr_weights_cached():
        # The predictor built but the cache check can't see the files: most likely the
        # filename convention changed in a doctr upgrade. Fail loudly: the GUI's cached
        # check would otherwise re-offer this download forever.
        print(f"error: docTR models did not land in the expected cache "
              f"({doctr_cache_dir() / 'models'}); the docTR filename convention may have "
              f"changed.", file=sys.stderr)
        return 1
    print("docTR models ready.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="vodparse.cli.doctr_support",
        description="Download or probe the docTR overlay-OCR models.")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("probe", help="Print docTR availability (installed / weights cached) as JSON")
    sub.add_parser("fetch", help="Download the pinned docTR models into the cache")
    args = ap.parse_args(argv)

    if args.command == "probe":
        return probe()
    if args.command == "fetch":
        return fetch()
    return 2  # unreachable: subparser is required


if __name__ == "__main__":
    sys.exit(main())
