"""Cut match/set clips from a reviewed JSON (the GUI Review screen's output).

The reviewed JSON holds absolute, human-corrected clip boundaries plus edited tags/characters,
so this slices the clips and writes the metadata CSV with no buffer math. The CSV is shared by
every VOD exported under the same tournament name: only this VOD's rows are replaced. Run
standalone or via the GUI's "Export clips" button:

    python -m vodparse.cli.export_reviewed --input json_output/<stem>/reviewed.json

To cut a single clip to an exact file (the Review tree's right-click "Export to a file…"
action), name it and the destination; no CSV or export-record write-back happens then:

    python -m vodparse.cli.export_reviewed --input …/reviewed.json --only-clip 3 --out-file /path/set3.mp4
"""

import argparse
from pathlib import Path

from vodparse.cli.stage_io import prepare_stage_streams
from vodparse.export import export_basename, export_reviewed, export_single_clip
from vodparse.jsonio import read_json, write_json_atomic
from vodparse.paths import video_out_dir
from vodparse.titles import add_title_args


def main() -> None:
    prepare_stage_streams()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to reviewed JSON")
    parser.add_argument("--folder", default=None,
                        help="Output subfolder under video_out/ (default: <stem>_<games|sets>)")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Directory that takes the place of video_out/ for exported clips "
                             "(default: video_out/ under the current directory); --folder names "
                             "a subfolder inside it")
    parser.add_argument("--only-clip", type=int, default=None, metavar="N",
                        help="Cut just clip number N (requires --out-file)")
    parser.add_argument("--out-file", default=None, metavar="PATH",
                        help="Exact output file for --only-clip (requires --only-clip)")
    add_title_args(parser, limit_default=None,
                   limit_default_help="the value the GUI stored in the reviewed JSON, else 100",
                   filenames_default_help="the value the GUI stored in the reviewed JSON, "
                                          "else on for sets / off for matches")
    args = parser.parse_args()
    if (args.only_clip is None) != (args.out_file is None):
        parser.error("--only-clip and --out-file must be used together")
    if args.only_clip is not None:
        # These flags only shape a full export's folder and title/filename handling; silently
        # accepting them here would let a caller believe they took effect.
        clashing = []
        if args.folder is not None:
            clashing.append("--folder")
        if args.out_root is not None:
            clashing.append("--out-root")
        if args.title_limit is not None:
            clashing.append("--title-limit")
        if args.title_filenames is not None:
            # BooleanOptionalAction: False means the user typed --no-title-filenames.
            # name the spelling they actually used, not the positive form.
            clashing.append("--title-filenames" if args.title_filenames
                            else "--no-title-filenames")
        if clashing:
            parser.error(f"{', '.join(clashing)} cannot be combined with --only-clip: "
                         "single-clip mode produces exactly one file at --out-file, "
                         "so folder and title options don't apply")

    data = read_json(args.input)

    if args.only_clip is not None:
        # Single-clip mode: one video file at the caller-chosen path, nothing else; no CSV,
        # and no export_folder/export_csv write-back, so the last full export's records (and
        # the GUI buttons reading them) are left untouched.
        export_single_clip(data, args.only_clip, args.out_file)
        return

    is_set   = data["mode"] != "match"
    base     = export_basename(data.get("tournament_name", ""), data["stem"], is_set)
    folder   = args.folder or base
    clip_dir = video_out_dir(folder, args.out_root)
    clip_dir.mkdir(parents=True, exist_ok=True)

    # The CSV may be shared with other VODs of the same tournament (see
    # vodparse.export.merge_vod_rows): only this VOD's rows are replaced.
    export_reviewed(data, clip_dir, title_limit=args.title_limit,
                    title_filenames=args.title_filenames)

    # Record where clips actually landed (the --folder override defeats the default-basename
    # reconstruction) so the diagnostics bundler and the GUI's export-output buttons can find
    # the CSV under the right folder. The CSV is named after the basename, NOT the folder
    # (see vodparse.export), so both parts are recorded.
    data["export_folder"] = folder
    data["export_csv"] = f"{base}.csv"
    write_json_atomic(args.input, data)


if __name__ == "__main__":
    main()
