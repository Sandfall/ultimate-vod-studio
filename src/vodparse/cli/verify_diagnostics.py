"""Verify a diagnostics zip is genuine and safe before opening it (see vodparse.diagnostics).

Run this on any file someone sends as "diagnostics". PASS means the bundle holds only text
artifacts, every file's hash matches its manifest, and nothing can escape on extraction; FAIL
prints the specific reasons. Exits non-zero on FAIL so it can gate a script.

    python -m vodparse.cli.verify_diagnostics diagnostics_<stem>_<timestamp>.zip
"""

import argparse
import sys

from vodparse.diagnostics import verify_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a diagnostics bundle.")
    parser.add_argument("zip", help="Path to a diagnostics_*.zip")
    args = parser.parse_args()

    ok, report = verify_diagnostics(args.zip)
    for line in report:
        print(line)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
