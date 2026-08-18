"""Diagnostics bundles: collect one vod's JSON/CSV outputs into a verifiable zip.

When a user hits a problem, this packages everything the app produced for one source video
(pipeline.json, matches.json, reviewed.json, the debug CSVs, the clips CSV) into a single
``diagnostics_<stem>_<timestamp>.zip`` they can send for troubleshooting. The GUI's crash log
rides along when the caller provides its path (see ``build_diagnostics``), so a crash report
and the diagnostics arrive in one bundle instead of two asks.

The bundle is deliberately boring so it's *safe to receive*: a plain (unencrypted) zip of
inert text files plus a ``manifest.json``. Plain so any antivirus can scan inside it
(encryption would defeat that); text-only + manifest so ``verify_diagnostics`` can prove a
file really is a diagnostics bundle and nothing else before anyone opens it. There's no
cryptographic signing: the bundle is built on the user's machine, so any embedded key is
forgeable; the real defense is structural validation (this module's verifier) + AV scan.

``build_diagnostics`` writes a bundle; ``verify_diagnostics`` checks one. Both are Qt-free
so they work from the CLI and the test suite; the GUI supplies its QSettings snapshot via
the vodparse.cli.export_diagnostics entry point. Paths resolve relative to the project root (the cwd
the entry points run from), the same convention paths.py uses for every other artifact; the
clips CSV alone may live under a separate export root (``export_root``: the GUI's
configurable clip destination) instead of the sibling ``video_out/``.
"""

import csv
import hashlib
import io
import json
import os
import platform
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from vodparse import paths
from vodparse.export import export_basename
from vodparse.match_extraction import IMPLAUSIBLE_TIMERS_NAME
from vodparse.schema import DiagnosticsManifest, ManifestFile

FORMAT = "vodparse-diagnostics"
FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"

# Diagnostics are text artifacts; anything else has no business in the bundle and the
# verifier rejects it, which is what stops a payload riding in under the "diagnostics" label.
ALLOWED_SUFFIXES = {".json", ".csv", ".txt"}
# Total uncompressed size cap (zip-bomb guard). These are text outputs for one vod: tens of
# MB at the very most; so anything past this is suspect. Enforced via capped reads, not the
# zip's self-reported sizes, so a lying header can't get past it.
MAX_TOTAL_UNCOMPRESSED = 50 * 1024 * 1024

# JSON keys whose values are absolute filesystem paths. Reduced to basenames before export so
# a shared bundle never leaks the sender's directory layout. Keyed by field name (not a blind
# whole-tree guess) so non-path strings are never mangled.
_PATH_KEYS = {"video_path", "timer_model_path", "character_model_path"}

# Arcnames for the GUI crash log and its rotated sibling. Stored as .txt so ALLOWED_SUFFIXES
# stays as tight as it is: renaming one artifact beats teaching the verifier (and so every
# future bundle) a whole new file type.
CRASH_LOG_ARCNAME = "crash_log.txt"
CRASH_LOG_ROTATED_ARCNAME = "crash_log.1.txt"

# Per-file cap on a crash log copied into a bundle. The live log only rotates when the GUI
# starts, so one long session (or the live log plus its rotated twin) can outgrow the whole
# bundle's MAX_TOTAL_UNCOMPRESSED budget and make the recipient's verify reject everything.
# Paired with vodparse.gui.crash_log._MAX_LOG_BYTES, the rotation cap: same size, so a log
# that rotated on schedule still ships in full and only a runaway one gets trimmed.
MAX_CRASH_LOG_BYTES = 2_000_000

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]


def _root(root: Path | None) -> Path:
    """The project root holding json_output/ and video_out/ (cwd unless given explicitly:
    the GUI passes its PROJECT_ROOT since its own cwd may differ)."""
    return root if root is not None else Path.cwd()


def available_stems(root: Path | None = None) -> list[str]:
    """Stems that have at least one diagnostic artifact on disk, most-recently-modified first.

    A vod qualifies when its json_output/<stem>/ folder holds a pipeline.json or matches.json,
    so the GUI picker only ever offers vods that actually have something to bundle."""
    base = _root(root) / paths.JSON_OUTPUT_ROOT
    if not base.is_dir():
        return []
    dirs = [d for d in base.iterdir()
            if d.is_dir() and ((d / "pipeline.json").exists() or (d / "matches.json").exists())]
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in dirs]


def _collect(stem: str, root: Path | None,
             export_root: Path | None) -> list[tuple[str, Path]]:
    """(arcname, source path) for every existing artifact of one vod. Missing files are
    skipped silently: a vod that was only analyzed (no review/export yet) still bundles.
    ``export_root`` is where the clips CSV lives when it isn't the sibling video_out/
    (see build_diagnostics)."""
    base = _root(root)
    out = base / paths.output_dir(stem)
    items: list[tuple[str, Path]] = []
    # reviewed_single.json is the projection a failed single-clip export read (the GUI
    # deletes it on success), so bundling it captures exactly what that subprocess saw.
    for name in ("pipeline.json", "matches.json", "reviewed.json", "review_draft.json",
                 "reviewed_single.json"):
        if (out / name).exists():
            items.append((name, out / name))
    dbg = base / paths.debug_dir(stem)
    if dbg.is_dir():
        for p in sorted(dbg.glob("*.csv")):
            items.append((f"debug/{p.name}", p))
        # The stripped-timer dump is the one debug artifact that isn't a CSV, so the glob above
        # misses it; add it explicitly so the implausible readings actually ship in the bundle.
        impl = dbg / IMPLAUSIBLE_TIMERS_NAME
        if impl.exists():
            items.append((f"debug/{impl.name}", impl))
    # The clips CSV + title report live under video_out/<folder>/, where the CSV is named after
    # export_basename but the folder may have been overridden at export time (export_reviewed
    # --folder). The exporter records the actual folder into reviewed.json as "export_folder";
    # fall back to the default basename for files written before that field existed.
    reviewed = out / "reviewed.json"
    if reviewed.exists():
        try:
            data = json.loads(reviewed.read_text(encoding="utf-8"))
            basename = export_basename(data.get("tournament_name", ""), stem,
                                       data.get("mode") != "match")
            clip_root = export_root if export_root is not None else base / paths.VIDEO_OUT_ROOT
            clip_dir = clip_root / (data.get("export_folder") or basename)
            found = False
            for fname in (f"{basename}.csv", "title_report.txt"):
                if (clip_dir / fname).exists():
                    items.append((fname, clip_dir / fname))
                    found = True
            if data.get("export_folder") and not found:
                print(f"  diagnostics: export ran (folder '{data['export_folder']}') but no clips "
                      f"CSV found under {clip_dir}: bundling without it")
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"  diagnostics: could not read {reviewed.name} ({e}): "
                  "bundling without the clips CSV / title report")
    return items


def _read_crash_log_tail(path: Path) -> bytes:
    """One crash log's contents, trimmed to the last ``MAX_CRASH_LOG_BYTES`` bytes if oversized.

    The tail is the valuable end (the most recent session is the one that crashed), so a
    truncated copy keeps the end and drops the beginning. The partial line the cut lands in
    goes too, so the copy never opens mid-traceback-frame — unless the window holds no
    newline at all (one runaway single-line message), in which case the raw tail ships
    whole rather than being thrown away. A one-line banner records that the trim happened
    and how big the original was."""
    with path.open("rb") as fh:
        size = fh.seek(0, os.SEEK_END)
        if size <= MAX_CRASH_LOG_BYTES:
            fh.seek(0)
            return fh.read()
        fh.seek(size - MAX_CRASH_LOG_BYTES)
        tail = fh.read()
    _, newline, after_cut = tail.partition(b"\n")
    if newline:
        tail = after_cut
    banner = (f"[diagnostics] crash log truncated: original was {size} bytes, "
              f"keeping the last {len(tail)}.\n")
    return banner.encode() + tail


def _crash_log_entries(crash_log: Path | None) -> list[tuple[str, bytes]]:
    """(arcname, contents) for the crash log and its rotated sibling, when they exist.

    The rotated name mirrors ``vodparse.gui.crash_log._rotate_if_large`` (``crash.log`` →
    ``crash.log.1``): a crash can be sitting in the rotated file while the live log holds only
    the fresh session, so both ship. Paths inside are *not* redacted, unlike the JSON
    artifacts: a crash log's absolute paths (tracebacks, Qt messages) are the diagnostic.

    Reading happens here, not in the caller, so both of this artifact's best-effort rules stay
    in one place: an oversized log is tail-capped (see ``_read_crash_log_tail``), and a log
    that disappears between the existence check and the read — the GUI or a second instance
    rotating it — is skipped rather than failing the whole bundle over one extra."""
    if crash_log is None:
        return []
    rotated = crash_log.with_suffix(crash_log.suffix + ".1")
    items: list[tuple[str, bytes]] = []
    for arcname, path in ((CRASH_LOG_ARCNAME, crash_log),
                          (CRASH_LOG_ROTATED_ARCNAME, rotated)):
        if not path.is_file():
            continue
        try:
            items.append((arcname, _read_crash_log_tail(path)))
        except OSError as e:
            print(f"  diagnostics: could not read {path.name} ({e}): bundling without it")
    return items


def _redact(obj: Json) -> Json:
    """A copy of a JSON value with absolute-path fields (see _PATH_KEYS) reduced to basenames."""
    if isinstance(obj, dict):
        return {k: (Path(v).name if k in _PATH_KEYS and isinstance(v, str) else _redact(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def _app_version() -> str:
    try:
        from importlib.metadata import version
        return version("vodparse")
    except Exception:
        try:
            text = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
                encoding="utf-8")
            m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
            if m:
                return m.group(1)
        except OSError:
            pass
        return "unknown"


def _qt_version() -> str:
    try:
        from PyQt6.QtCore import QT_VERSION_STR
        return QT_VERSION_STR
    except Exception:
        return "unavailable"


def build_diagnostics(stem: str, dest_zip: str | Path,
                      root: Path | None = None,
                      settings: dict[str, str] | None = None,
                      export_root: Path | None = None,
                      crash_log: Path | None = None) -> DiagnosticsManifest:
    """Bundle ``stem``'s artifacts into ``dest_zip`` (plain, unencrypted, text-only) and return
    the manifest written. ``settings`` is an optional app-preference snapshot the GUI passes
    through (absent in standalone CLI use). ``export_root`` is the directory that plays the
    role of ``video_out/`` for the clips CSV (the export stages' --out-root); by default the
    sibling ``video_out/`` under ``root``, the pre-split layout. ``crash_log`` is the app's
    crash log to include alongside the vod's artifacts (the CLI entry point passes
    ``runtime_paths.crash_log_path()``; the default None skips it, keeping bundles built
    against a bare ``root`` — e.g. in tests — independent of this machine's global state).
    The zip is written atomically (build to a .tmp sibling, then os.replace) like
    jsonio.write_json_atomic, so a crash never leaves a partial bundle. JSON copies are
    reindented and path-redacted; the pipeline's non-JSON artifacts are stored as-is, and the
    crash log is stored as-is up to ``MAX_CRASH_LOG_BYTES`` (an oversized one keeps its
    tail)."""
    dest_zip = Path(dest_zip)
    entries: list[tuple[str, bytes]] = []
    for arcname, src in _collect(stem, root, export_root):
        raw = src.read_bytes()
        if arcname.endswith(".json"):
            try:
                raw = json.dumps(_redact(json.loads(raw)), indent=2).encode()
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # a corrupt artifact is still worth sending; keep its bytes verbatim
        entries.append((arcname, raw))
    # The crash log arrives already read: it is capped and best-effort, unlike the artifacts
    # above, so its reading rules live in _crash_log_entries instead of in this loop.
    entries.extend(_crash_log_entries(crash_log))

    files: list[ManifestFile] = []
    for arcname, raw in entries:
        files.append({"name": arcname, "size": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest()})

    manifest: DiagnosticsManifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "app_version": _app_version(),
        "python_version": platform.python_version(),
        "qt_version": _qt_version(),
        "platform": platform.platform(),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stem": stem,
        "settings": dict(settings or {}),
        "files": files,
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode()

    tmp = dest_zip.with_name(dest_zip.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(MANIFEST_NAME, manifest_bytes)
            for arcname, raw in entries:
                zf.writestr(arcname, raw)
        os.replace(tmp, dest_zip)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return manifest


def default_zip_name(stem: str, when: datetime | None = None) -> str:
    """The suggested bundle filename, e.g. diagnostics_myvod_20260614-153000.zip."""
    when = when or datetime.now()
    return f"diagnostics_{stem}_{when:%Y%m%d-%H%M%S}.zip"


class _Unsafe(Exception):
    """Raised mid-verify to abort with a reason (kept internal; verify returns it as a report)."""


def _read_capped(zf: zipfile.ZipFile, name: str, budget: list[int]) -> bytes:
    """Decompress one entry while enforcing the shared size budget, so a zip bomb that lies
    about its uncompressed size still can't blow up memory: we read one byte past what's left
    and bail if it's bigger. ``budget`` is a one-item list used as a mutable counter."""
    with zf.open(name) as fh:
        data = fh.read(budget[0] + 1)
    if len(data) > budget[0]:
        raise _Unsafe(f"Bundle exceeds {MAX_TOTAL_UNCOMPRESSED}-byte uncompressed cap.")
    budget[0] -= len(data)
    return data


def verify_diagnostics(zip_path: str | Path) -> tuple[bool, list[str]]:
    """Check that ``zip_path`` is a genuine, safe diagnostics bundle and nothing else.

    Returns (ok, report). Never raises on a malformed or hostile archive: the whole point is
    that you can run it on a file from a stranger before opening anything inside. Fails unless:
    every entry is a text type with a safe relative name (no zip-slip), the bundle stays under
    the size cap (no zip-bomb), manifest.json declares our format, the set of entries exactly
    matches the manifest, every file's size + sha256 match, and each file parses as its type."""
    zip_path = Path(zip_path)
    if not zipfile.is_zipfile(zip_path):
        return False, ["Not a zip file."]
    report: list[str] = []
    budget = [MAX_TOTAL_UNCOMPRESSED]
    artifact_count = 0
    try:
        with zipfile.ZipFile(zip_path) as zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            for info in infos:
                name = info.filename
                parts = Path(name).parts
                if name.startswith(("/", "\\")) or "\\" in name or ":" in name or ".." in parts:
                    report.append(f"Unsafe entry name: {name!r}")
                elif name != MANIFEST_NAME and Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
                    report.append(f"Disallowed file type: {name!r}")
            if report:
                return False, report  # structural problems; don't decompress anything

            names = {i.filename for i in infos}
            if MANIFEST_NAME not in names:
                return False, ["Missing manifest.json."]
            try:
                manifest = json.loads(_read_capped(zf, MANIFEST_NAME, budget))
            except json.JSONDecodeError:
                return False, ["manifest.json is not valid JSON."]
            if (not isinstance(manifest, dict) or manifest.get("format") != FORMAT
                    or manifest.get("format_version") != FORMAT_VERSION):
                return False, ["Not a vodparse diagnostics bundle (bad format marker)."]

            listed = manifest.get("files")
            if not isinstance(listed, list) or any(
                    not isinstance(f, dict) or "name" not in f for f in listed):
                return False, ["manifest.json has a malformed files list."]
            by_name = {f["name"]: f for f in listed}
            artifact_count = len(by_name)

            present = names - {MANIFEST_NAME}
            if present != set(by_name):
                extra = sorted(present - set(by_name))
                missing = sorted(set(by_name) - present)
                if extra:
                    report.append(f"Entries not listed in manifest: {extra}")
                if missing:
                    report.append(f"Manifest lists entries not present: {missing}")
                return False, report

            for name, meta in by_name.items():
                data = _read_capped(zf, name, budget)
                if meta.get("sha256") != hashlib.sha256(data).hexdigest():
                    report.append(f"Hash mismatch: {name}")
                if meta.get("size") != len(data):
                    report.append(f"Size mismatch: {name}")
                report.extend(_check_content(name, data))
    except _Unsafe as e:
        return False, [str(e)]
    except zipfile.BadZipFile:
        return False, ["Corrupt zip file."]

    if report:
        return False, report
    return True, [f"PASS: verified diagnostics bundle, {artifact_count} artifact(s)."]


def _check_content(name: str, data: bytes) -> list[str]:
    """Confirm one entry parses as the type its extension claims, with a light shape check on
    the known JSON artifacts so a renamed payload can't masquerade as e.g. pipeline.json."""
    suffix = Path(name).suffix.lower()
    if suffix == ".json":
        try:
            obj = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return [f"Not valid JSON: {name}"]
        if name == "pipeline.json" and not (isinstance(obj, dict) and "config" in obj and "events" in obj):
            return [f"Unexpected shape: {name}"]
        if name == "matches.json" and not isinstance(obj, list):
            return [f"Unexpected shape: {name}"]
        if name == "reviewed.json" and not (isinstance(obj, dict) and "clips" in obj):
            return [f"Unexpected shape: {name}"]
    elif suffix == ".csv":
        try:
            rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
        except (UnicodeDecodeError, csv.Error):
            return [f"Not valid CSV: {name}"]
        if not rows or not rows[0]:
            return [f"CSV has no header row: {name}"]
    return []
