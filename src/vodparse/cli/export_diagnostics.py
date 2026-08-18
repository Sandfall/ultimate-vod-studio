"""Bundle one vod's diagnostic outputs into a verifiable zip (see vodparse.diagnostics).

GUI-launched ("Export Diagnostics…") and standalone. The app's crash log (and its rotated
sibling), when present, ride along in the bundle. Run from the project root so json_output/
and video_out/ resolve, same as the other entry-point scripts:

    python -m vodparse.cli.export_diagnostics --stem <stem> --output diagnostics.zip
"""

import argparse
import json
from pathlib import Path

from vodparse.diagnostics import build_diagnostics, default_zip_name
from vodparse.paths import vod_stem
from vodparse.runtime_paths import crash_log_path


def main() -> None:
    """Bundle one vod's diagnostic outputs into a verifiable zip."""
    parser = argparse.ArgumentParser(description="Bundle a vod's diagnostics into a zip.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--stem", help="Vod stem (json_output/<stem>/)")
    source.add_argument("--input", help="Source video; its filename stem is used")
    parser.add_argument("--output", default=None,
                        help="Destination .zip (default: diagnostics_<stem>_<timestamp>.zip)")
    parser.add_argument("--settings-json", default=None,
                        help="JSON object of app settings to record in the manifest")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Directory the clips were exported under (the export stages' "
                             "--out-root; default: the sibling video_out/)")
    args = parser.parse_args()

    stem = args.stem or vod_stem(args.input)
    output = args.output or default_zip_name(stem)
    settings = json.loads(args.settings_json) if args.settings_json else None

    manifest = build_diagnostics(stem, output, settings=settings, export_root=args.out_root,
                                 crash_log=crash_log_path())
    print(f"Wrote {output} ({len(manifest['files'])} artifact(s)).")


if __name__ == "__main__":
    main()
