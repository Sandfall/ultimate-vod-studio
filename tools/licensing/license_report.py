#!/usr/bin/env python3
"""Collect the license texts of every shipped dependency into the document the app bundles.

Why this exists: the frozen app redistributes ~70 third-party packages, and the licenses
they carry are not optional paperwork. GPL-3.0 (PyQt6), LGPL-3.0 (Qt 6, python-bidi) and
LGPL-2.1 (OpenCV's FFmpeg backend) require their text to ACCOMPANY the binary; MIT, BSD
and Apache-2.0 require their copyright notices to travel with a binary redistribution too.
Before this tool the frozen app shipped whichever ``.dist-info`` directories PyInstaller
happened to drag along — which covered ten packages and none of the copyleft ones.

So the texts are gathered mechanically, from the wheels installed in the environment being
frozen:

  * ``bundle``  write ``licenses/THIRD-PARTY-LICENSES.txt``: every dependency's own
                LICENSE/COPYING/NOTICE files, concatenated under per-distribution headers.
                UltimateVODStudio.spec bundles that one file, and Help ▸ About ▸ Licenses
                opens it.
  * ``ffmpeg-config``
                write ``licenses/FFMPEG-BUILD-CONFIG.txt``: the configure lines of the
                FFmpeg executables and shared libraries that ship, read back out of the
                binaries. Exits non-zero if any was built --enable-nonfree, which would
                make the application non-redistributable.
  * ``report``  print ``name -> license`` for the resolved closure. An audit aid for when a
                dependency is added or bumped; nothing consumes its output.

One document rather than a directory tree: no license mandates a file layout (they require
the text to be provided, not to be provided in any particular shape), a single file is what
large redistributors ship, and it is far likelier to actually be read than 223 files nested
five directories deep. NOTICE names the copyleft components by hand and points here.

What this CANNOT know, and is therefore hand-written in NOTICE: which license each FFmpeg
build is under, and the terms of the bundled EasyOCR and docTR model weights. ``ffmpeg-
config`` records the flags, but reading GPL-3.0 or LGPL off them is a judgment call that
belongs in prose.

Reproduce from a clean checkout:

    pip install -e ".[gui]"                                 # the shipped runtime set
    python tools/licensing/license_report.py bundle         # write the shipped document
    python tools/licensing/license_report.py ffmpeg-config  # record the FFmpeg builds
    python tools/licensing/license_report.py report         # audit the license labels

Run it in the environment that is actually frozen (build_mac.sh / build_windows.ps1's
.buildenv), not a dev shell: a different resolve can pull a different set of transitive
dependencies, and the whole point is to describe what ships.
"""

import argparse
import mmap
import sys
import tomllib
from collections import deque
from importlib.metadata import Distribution, PackageNotFoundError, distribution
from pathlib import Path

from packaging.requirements import Requirement

# tools/licensing/license_report.py -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# The bundled document. Lives beside the tracked licenses/torchvision-LICENSE.txt; the
# generated file itself is gitignored, because it describes one machine's resolve.
DEFAULT_DOCUMENT = REPO_ROOT / "licenses" / "THIRD-PARTY-LICENSES.txt"

# The FFmpeg builds' recorded configure lines, written by the ``ffmpeg-config`` subcommand.
DEFAULT_FFMPEG_CONFIG = REPO_ROOT / "licenses" / "FFMPEG-BUILD-CONFIG.txt"

# The extra whose dependencies also ship: the desktop GUI is the product, so PyQt6 (and
# the Qt 6 wheel it pulls) are runtime, not optional. "dev" and "test" are deliberately
# excluded — they are never frozen into the app.
SHIPPED_EXTRAS: tuple[str, ...] = ("gui",)

# Words that mark a file inside a wheel as license text worth shipping. Matched
# case-insensitively at the start of a filename (LICENSE.txt, COPYING.LESSER) or at the end
# of one before its extension (numpy's vendored dragon4_LICENSE.txt).
LICENSE_FILE_STEMS: tuple[str, ...] = ("license", "licence", "copying", "notice")

# Suffixes that are code, never license text. A package can legitimately contain a module
# named licenses/ (packaging and its setuptools vendoring both do), and collecting its
# source would put Python files into a legal document.
CODE_SUFFIXES: frozenset[str] = frozenset(
    {".py", ".pyc", ".pyd", ".so", ".dll", ".dylib"})

# Substring marking a DIRECTORY whose contents are license texts whatever they are named.
# Some wheels lay them out that way: pypdfium2 ships LICENSES/<spdx>.txt (the REUSE
# convention) plus data/<platform>/BUILD_LICENSES/<component>.txt for everything statically
# linked into libpdfium, and h5py ships licenses/hdf5.txt and friends. Matching on filename
# alone collected NONE of pypdfium2's 19 texts and reported it as shipping no license at
# all — a package whose binary is in the app.
LICENSE_DIR_MARKER = "licen"

# Distributions whose declared license metadata is missing or misleading, corrected by
# hand. Only ``report`` reads this: the bundled document carries the actual license TEXT,
# so a wrong label here misleads a maintainer running an audit but cannot understate what
# is disclosed to a user.
LICENSE_OVERRIDES: dict[str, str] = {
    # ffmpeg-python's metadata carries no License field, only an Apache classifier.
    "ffmpeg-python": "Apache-2.0",
}

# Name prefixes for distributions that resolve on the DEV container but ship in no build.
#
# Linux `pip install torch` drags in the CUDA runtime wheels (several of them NVIDIA
# *proprietary*) plus triton. Neither frozen build has them: build_windows.ps1 installs
# torch from the CPU index and build_mac.sh installs the macOS wheels, so the CUDA stack is
# absent from both. It reaches an end user only if they opt into the in-app GPU download
# (vodparse.cli.gpu_support), which fetches from NVIDIA at the user's request — we never
# redistribute it, so it is disclosed as a hand-written NOTICE paragraph instead.
#
# Bundling their texts anyway would put NVIDIA's proprietary license into the third-party
# document of an app that does not, in fact, contain their software. Pass
# --include-unshipped to see them when auditing a dev environment.
UNSHIPPED_PREFIXES: tuple[str, ...] = ("nvidia-", "cuda-", "triton")

# Distributions that install in every build environment but that UltimateVODStudio.spec
# leaves out of the frozen app through its ``excludes=``, mapped to the import package the
# spec names. polars is an ultralytics dependency the app never calls, and imageio-ffmpeg
# is only the source the build scripts copy bin/ffmpeg from (that copy is recorded with
# ``ffmpeg-config --also``). Listing them would disclose software the app does not
# contain, and would record imageio-ffmpeg's own ffmpeg as shipping. A test checks that
# each import package here is still in the spec's excludes, so the two cannot drift.
EXCLUDED_FROM_BUNDLE: dict[str, str] = {
    "polars": "polars",
    "polars-runtime-32": "_polars_runtime_32",
    "imageio-ffmpeg": "imageio_ffmpeg",
}

# Declared dependencies the build scripts deliberately swap for another distribution that
# provides the same import package, keyed by normalized name. easyocr declares
# opencv-python-headless, while doctr and ultralytics declare opencv-python. Both install
# into cv2/, so build_mac.sh / build_windows.ps1 uninstall the headless one and keep the
# pinned opencv-python. The closure follows the replacement instead of reporting the
# removed name as missing, so the document lists the one OpenCV that actually ships.
REPLACED_DISTRIBUTIONS: dict[str, str] = {
    "opencv-python-headless": "opencv-python",
}

# Column width for the aligned "name .... license" report, matching NOTICE's style.
_NAME_COLUMN = 34

# Longest a distribution's free-text License field may be before it is treated as an
# embedded license BODY rather than a label. scipy puts its whole copyright header there.
_LABEL_MAX_LENGTH = 60

# Rule widths for the generated document's headers. Wide enough for the longest wheel-
# relative path in practice (torch's vendored texts nest five levels deep).
_BANNER_RULE = "=" * 78
_FILE_RULE = "-" * 78

# What a reader of the shipped document sees first. It has to answer, without any other
# context, "what is this file and what is it NOT" — someone opening it from Help ▸ About
# has no repo, no NOTICE open beside it, and no reason to know what a wheel is.
_DOCUMENT_PREAMBLE = f"""\
{_BANNER_RULE}
Ultimate VOD Studio - third-party license texts
{_BANNER_RULE}

This file collects the license text of every third-party Python package bundled with
Ultimate VOD Studio, reproduced verbatim from each package's own distribution. It is
generated by tools/licensing/license_report.py from the environment the application
was built in.

It does NOT cover the non-Python binaries that ship alongside them: the bundled FFmpeg
builds, and the EasyOCR and docTR model weights. Those are described in NOTICE, which
also names the components whose licenses constrain redistribution.

Ultimate VOD Studio's own license is AGPL-3.0; see the LICENSE file beside this one.
"""

# The companion document for the FFmpeg binaries, which no wheel metadata describes.
_FFMPEG_PREAMBLE = f"""\
{_BANNER_RULE}
Ultimate VOD Studio - bundled FFmpeg build configuration
{_BANNER_RULE}

Several FFmpeg builds ship with Ultimate VOD Studio, under different licenses depending
on how each was configured: the executable the program calls, plus the shared libraries
OpenCV and Qt use for decoding. Their licenses cannot be read from package metadata,
only from the flags they were built with, so those flags are recorded here verbatim as
they were read out of the binaries at build time.

A statically linked binary lists several configure lines: its own, and those of the
codec libraries linked into it. None of them may contain --enable-nonfree, which the
build refuses to proceed past. NOTICE explains what each build is and where to obtain
its corresponding source.
"""


# ── Dependency resolution ─────────────────────────────────────────────────────
def _read_roots(pyproject: Path) -> list[str]:
    """The declared runtime dependency names: ``[project].dependencies`` plus each extra
    in :data:`SHIPPED_EXTRAS`.

    Read from pyproject.toml rather than the installed vodparse distribution's metadata,
    so a stale editable install (whose recorded dependencies can lag the file) cannot
    quietly narrow the set this tool reports on.
    """
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    roots = list(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {})
    for extra in SHIPPED_EXTRAS:
        roots += list(optional.get(extra, []))
    return [Requirement(r).name for r in roots]


def _requirements_of(dist: Distribution) -> list[str]:
    """Names of ``dist``'s install-time requirements, skipping any whose environment
    marker excludes this platform and any that exist only under an extra.

    Extras are skipped because a dependency guarded by ``extra == "foo"`` is not installed
    unless that extra was requested; including them would list packages that never ship.
    """
    names: list[str] = []
    for raw in dist.requires or []:
        req = Requirement(raw)
        marker = req.marker
        if marker is not None:
            # An "extra" marker can't be evaluated without naming the extra, and we want
            # none of them; anything else is evaluated against this interpreter.
            if "extra" in str(marker):
                continue
            if not marker.evaluate():
                continue
        names.append(req.name)
    return names


def resolve_runtime_distributions(
        roots: list[str]) -> tuple[dict[str, Distribution], list[str]]:
    """Breadth-first closure of installed distributions reachable from ``roots``.

    Returns (name -> Distribution, names that could not be resolved). Uninstalled packages
    are collected rather than skipped silently, because generating the document from a
    partial environment produces a partial *legal disclosure* — the exact failure this tool
    exists to prevent. The caller decides whether that is fatal; see --allow-missing.
    A name in :data:`REPLACED_DISTRIBUTIONS` that is not installed resolves to its
    replacement instead, which is then reported missing if it is absent too.

    Note that this is the pip closure, which is a superset of what PyInstaller freezes:
    a declared dependency nothing imports resolves here but never reaches the bundle.
    Erring wide is deliberate — an extra license text costs a few kilobytes, a missing one
    is a compliance failure.
    """
    found: dict[str, Distribution] = {}
    missing: list[str] = []
    seen: set[str] = set()
    queue = deque(roots)
    while queue:
        name = queue.popleft()
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        seen.add(key)
        try:
            dist = distribution(name)
        except PackageNotFoundError:
            if key in REPLACED_DISTRIBUTIONS:
                queue.append(REPLACED_DISTRIBUTIONS[key])
            else:
                missing.append(name)
            continue
        found[dist.metadata["Name"]] = dist
        queue.extend(_requirements_of(dist))
    return found, missing


def drop_unshipped(dists: dict[str, Distribution]) -> dict[str, Distribution]:
    """``dists`` without the packages that reach no frozen app.

    Those are the dev-container-only packages (see :data:`UNSHIPPED_PREFIXES`) and the
    ones the spec excludes from the bundle (see :data:`EXCLUDED_FROM_BUNDLE`).
    """
    shipped: dict[str, Distribution] = {}
    for name, dist in dists.items():
        normalized = name.lower().replace("_", "-")
        if normalized.startswith(UNSHIPPED_PREFIXES):
            continue
        if normalized in EXCLUDED_FROM_BUNDLE:
            continue
        shipped[name] = dist
    return shipped


# ── License metadata ──────────────────────────────────────────────────────────
def license_of(dist: Distribution) -> str:
    """The best available license label for ``dist``.

    Preference order mirrors how authoritative each source is: the PEP 639
    ``License-Expression`` SPDX field, then the free-text ``License`` field, then the
    ``License ::`` trove classifiers. The free-text field is frequently a whole license
    *body* rather than a name (scipy embeds its copyright header there), so it is only
    used when it is short enough to be a label.
    """
    name = dist.metadata["Name"]
    if name in LICENSE_OVERRIDES:
        return LICENSE_OVERRIDES[name]

    expression = dist.metadata.get("License-Expression")
    if expression:
        return expression.strip()

    declared = (dist.metadata.get("License") or "").strip()
    if declared and "\n" not in declared and len(declared) <= _LABEL_MAX_LENGTH:
        return declared

    classifiers = [c.rsplit("::", 1)[-1].strip()
                   for c in dist.metadata.get_all("Classifier") or []
                   if c.startswith("License ::")]
    # "OSI Approved" is a category header, not a license, and appears alone on some dists.
    classifiers = [c for c in classifiers if c != "OSI Approved"]
    if classifiers:
        return "; ".join(classifiers)
    return "see the distribution's own license file"


def license_files(dist: Distribution) -> list[tuple[Path, Path]]:
    """``dist``'s bundled license texts as (relative path, absolute source) pairs.

    The relative path is kept because basenames collide *within a single distribution*:
    numpy ships 17 license files and torch 98, most of them named plain ``LICENSE`` for
    different vendored components. Flattening to the basename would silently drop all but
    one — losing exactly the third-party notices this tool exists to preserve.

    De-duplicated by destination path, since numpy ships several licenses twice (once
    beside the package, once under .dist-info) and a RECORD can name the same file twice.
    """
    seen: set[Path] = set()
    pairs: list[tuple[Path, Path]] = []
    for entry in dist.files or []:
        relative = _strip_dist_info(Path(str(entry)))
        if not is_license_file(relative):
            continue
        source = Path(str(dist.locate_file(entry))).resolve()
        if not source.is_file():
            continue
        if relative in seen:
            continue
        seen.add(relative)
        pairs.append((relative, source))
    return sorted(pairs)


def is_license_file(relative: Path) -> bool:
    """Whether a wheel-relative path is license text this should collect.

    Three rules, because wheels disagree about layout: a filename that starts with a
    license word (LICENSE.txt, COPYING.LESSER), one that ends with it before the extension
    (numpy's vendored dragon4_LICENSE.txt), or any file inside a directory whose name says
    it holds licenses (pypdfium2's LICENSES/ and BUILD_LICENSES/, h5py's licenses/).

    Anchoring the filename rule at either end rather than matching anywhere is what keeps
    OpenCV's haarcascade_license_plate_rus_16stages.xml — a detector for vehicle number
    plates — out of the license document.
    """
    if relative.suffix.lower() in CODE_SUFFIXES:
        return False
    name, stem = relative.name.lower(), relative.stem.lower()
    if any(name.startswith(word) or stem.endswith(word) for word in LICENSE_FILE_STEMS):
        return True
    return any(LICENSE_DIR_MARKER in part.lower() for part in relative.parts[:-1])


def _strip_dist_info(relative: Path) -> Path:
    """Drop the ``<name>-<version>.dist-info/licenses/`` wrapper from a wheel-relative path.

    Wheels expose license texts either beside the package or under a version-stamped
    ``.dist-info`` directory. Keeping that stamp would put a version number into every
    header in the generated document, churning it on each dependency bump. The remainder of
    the path is preserved, which is what keeps numpy's and torch's many same-named vendored
    licenses distinct.
    """
    parts = list(relative.parts)
    if parts and parts[0].endswith(".dist-info"):
        parts = parts[1:]
        if parts and parts[0] == "licenses":
            parts = parts[1:]
    return Path(*parts) if parts else relative


# ── The bundled FFmpeg builds ─────────────────────────────────────────────────
# Filename fragments identifying an FFmpeg executable or shared library among a wheel's
# files. Deliberately not just "av": libavif (the AVIF image codec, shipped by Pillow and
# OpenCV) is unrelated to FFmpeg and must not be reported as one.
FFMPEG_NAME_MARKERS: tuple[str, ...] = (
    "ffmpeg", "avcodec", "avformat", "avutil", "avdevice", "avfilter", "swscale",
    "swresample")

# The flag no shipped build may carry: --enable-nonfree produces a binary that cannot be
# redistributed at all, which would make the whole application undistributable.
NONFREE_FLAG = b"--enable-nonfree"


def ffmpeg_binaries(dists: dict[str, Distribution],
                    extra: list[Path]) -> list[tuple[str, Path]]:
    """The FFmpeg executables and shared libraries that ship, as (owner, path) pairs.

    Found by name among the installed distributions' own files, because that is where they
    come from: opencv-python vendors a video I/O backend, and PyQt6-Qt6 carries Qt's
    multimedia backend. ``extra`` covers the executable: the copy the build scripts stage
    into vendor/ from imageio-ffmpeg before PyInstaller runs, which belongs to no wheel
    once copied (imageio-ffmpeg itself is excluded, see :data:`EXCLUDED_FROM_BUNDLE`).
    """
    found: list[tuple[str, Path]] = []
    for name in sorted(dists, key=str.lower):
        for entry in dists[name].files or []:
            filename = Path(str(entry)).name.lower()
            if not any(marker in filename for marker in FFMPEG_NAME_MARKERS):
                continue
            path = Path(str(dists[name].locate_file(entry))).resolve()
            if path.is_file():
                found.append((name, path))
    found += [("staged by the build script", p.resolve()) for p in extra if p.is_file()]
    return found


def configure_strings(path: Path) -> list[str]:
    """Every build-configuration string embedded in ``path``, in the order found.

    An FFmpeg binary records the flags it was configured with as a plain NUL-terminated
    string, which is how a build's license can be read back out of a shipped artifact
    months later. A statically linked one contains several — its own, plus those of the
    codec libraries linked into it (a Windows ffmpeg.exe carries libvpx's as well) — so all
    of them are reported rather than a guess at which is "the" line. Empty when the file is
    not an FFmpeg build at all.
    """
    if path.stat().st_size == 0:
        # mmap refuses a zero-length file (ValueError). Nothing in the current dependency
        # set is an empty file with an FFmpeg-ish name, but a placeholder in some future
        # wheel must not crash the build's licensing step.
        return []
    found: list[bytes] = []
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0,
                                              access=mmap.ACCESS_READ) as data:
        at = data.find(b"--enable-")
        while at != -1:
            start = data.rfind(b"\x00", 0, at) + 1
            end = data.find(b"\x00", at)
            end = len(data) if end == -1 else end
            found.append(bytes(data[start:end]))
            at = data.find(b"--enable-", end)

    # De-duplicated after decoding, not before: the same line can appear twice with
    # different surrounding padding, which would survive a comparison of the raw bytes.
    # The executable stores its line behind the printf format it is reported with.
    lines: dict[str, None] = {}
    for raw in found:
        lines.setdefault(
            raw.decode("ascii", errors="replace").removeprefix("%sconfiguration: ").strip(),
            None)
    return list(lines)


def carries_nonfree(path: Path) -> bool:
    """Whether ``path`` was built --enable-nonfree, and so may not be redistributed.

    A plain whole-file search rather than a check against the parsed configure strings:
    this is the one question whose answer must not depend on a parsing heuristic being
    right, and a false positive fails a build loudly instead of shipping a bad artifact.
    """
    if path.stat().st_size == 0:
        return False  # mmap refuses a zero-length file; see configure_strings().
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0,
                                              access=mmap.ACCESS_READ) as data:
        return data.find(NONFREE_FLAG) != -1


def _read_text(path: Path) -> str:
    """``path`` decoded as text, tolerating a non-UTF-8 license file.

    Every license file in the current dependency set is valid UTF-8, but a future wheel
    could ship a Latin-1 copyright line with an author's name in it. Latin-1 decodes any
    byte sequence, so the fallback preserves the text rather than mangling it with
    replacement characters — this is legal text and it ships verbatim.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


# ── The generated document ────────────────────────────────────────────────────
def render_document(dists: dict[str, Distribution]) -> str:
    """The full text of ``licenses/THIRD-PARTY-LICENSES.txt``.

    Layout is a table of contents followed by one section per distribution, each file under
    its own wheel-relative path. That path is what makes torch's 98 vendored texts
    attributable to the component they cover instead of an undifferentiated wall.
    """
    names = sorted(dists, key=str.lower)
    sections: list[str] = []
    contents: list[str] = []

    for name in names:
        files = license_files(dists[name])
        # The version is here and not in NOTICE because it belongs to one build: GPL-3.0
        # section 6 obliges source corresponding to the binary conveyed, so anyone taking up
        # that offer needs to know which release of each component this app was built from.
        # Regenerated every build, so it cannot go stale the way a committed list would.
        version = dists[name].metadata.get("Version") or "unknown version"
        labelled = f"{name} {version}"
        dots = "." * max(3, _NAME_COLUMN - len(labelled))
        contents.append(f"  {labelled} {dots} {len(files)} file(s)"
                        if files else f"  {labelled} {dots} no license file in the wheel")

        body = [_BANNER_RULE, labelled, _BANNER_RULE, ""]
        if not files:
            # Legitimate: some wheels ship no license file at all. Say so rather than
            # leaving a silent gap a reader would have to interpret.
            body += [f"This distribution's wheel contains no license file. Declared "
                     f"license: {license_of(dists[name])}.", ""]
        for relative, source in files:
            body += [f"{relative.as_posix()}", _FILE_RULE, "", _read_text(source).rstrip(), ""]
        sections.append("\n".join(body))

    header = "\n".join([_DOCUMENT_PREAMBLE,
                        f"Contents ({len(names)} distributions):",
                        "",
                        *contents,
                        "",
                        ""])
    return header + "\n\n".join(sections) + "\n"


# ── Commands ──────────────────────────────────────────────────────────────────
def cmd_bundle(dists: dict[str, Distribution], args: argparse.Namespace) -> int:
    """Write the collected license texts to ``--out``."""
    document = render_document(dists)
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")

    without = [name for name in dists if not license_files(dists[name])]
    print(f"wrote {_display_path(out)}: {len(dists)} dependencies, "
          f"{len(document.encode('utf-8')) // 1024} KiB")
    if without:
        # Not fatal: a package can legitimately ship no license file in its wheel. The
        # document says so per package, and its declared license is still recorded there.
        print(f"note: no license file inside the wheel for: {', '.join(sorted(without))}",
              file=sys.stderr)
    return 0


def cmd_ffmpeg_config(dists: dict[str, Distribution], args: argparse.Namespace) -> int:
    """Record the shipped FFmpeg builds' configure lines, and refuse a nonfree one.

    The FFmpeg binaries are the one part of the bundle no wheel metadata describes: whether
    a given copy is GPL-3.0, GPL-2.0-or-later or LGPL is knowable only from the flags it was
    configured with. Writing them down at build time makes the license of what was actually
    redistributed answerable later, and turns an upstream wheel quietly changing its FFmpeg
    build into a visible diff rather than a silent relicensing.
    """
    binaries = ffmpeg_binaries(dists, args.also)
    lines = [_FFMPEG_PREAMBLE]
    nonfree: list[Path] = []

    for owner, path in binaries:
        configs = configure_strings(path)
        if not configs:
            continue  # Name matched but it is not an FFmpeg build (a stub or a wrapper).
        if carries_nonfree(path):
            nonfree.append(path)
        lines += [_FILE_RULE, f"{path.name}  ({owner})", _FILE_RULE, "", *configs, ""]

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {_display_path(out)}: {len(binaries)} FFmpeg binaries examined")

    if nonfree:
        listed = "\n    ".join(_display_path(p) for p in nonfree)
        print("error: these FFmpeg builds are configured --enable-nonfree, which makes "
              f"them non-redistributable:\n    {listed}\n"
              "The application must not be shipped with them. Check whether a dependency "
              "changed its bundled FFmpeg build.", file=sys.stderr)
        return 1
    return 0


def cmd_report(dists: dict[str, Distribution], args: argparse.Namespace) -> int:
    """Print ``name -> declared license`` for the resolved closure.

    An audit aid, not an input to anything: the shipped disclosure is the bundled document,
    so this exists for a human deciding whether a newly added dependency is acceptable.
    """
    for name in sorted(dists, key=str.lower):
        dots = "." * max(3, _NAME_COLUMN - len(name))
        print(f"  {name} {dots} {license_of(dists[name])}")
    return 0


def _display_path(path: Path) -> str:
    """``path`` relative to the repo root when it is inside it, else the path itself.

    --out may legitimately point anywhere (a CI scratch dir, a temp dir), so this must not
    be the bare relative_to() that raises ValueError on an outside path.
    """
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect the shipped dependencies' license texts into one document.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--pyproject", type=Path, default=REPO_ROOT / "pyproject.toml",
                        help="Project metadata naming the runtime dependency roots.")
    parser.add_argument("--include-unshipped", action="store_true",
                        help="Keep packages that resolve here but ship in no build (the "
                             "Linux-only CUDA/triton stack, and the packages the spec "
                             "excludes); for auditing only.")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Proceed even when a declared dependency is not installed. "
                             "Off by default: a partial environment yields a partial "
                             "legal disclosure.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    bundle = subparsers.add_parser(
        "bundle", help="Write licenses/THIRD-PARTY-LICENSES.txt (the file the app ships).")
    bundle.add_argument("--out", type=Path, default=DEFAULT_DOCUMENT,
                        help=f"Output file (default: {_display_path(DEFAULT_DOCUMENT)}).")
    bundle.set_defaults(func=cmd_bundle)

    ffmpeg = subparsers.add_parser(
        "ffmpeg-config",
        help="Write licenses/FFMPEG-BUILD-CONFIG.txt; fail on an --enable-nonfree build.")
    ffmpeg.add_argument("--out", type=Path, default=DEFAULT_FFMPEG_CONFIG,
                        help=f"Output file (default: {_display_path(DEFAULT_FFMPEG_CONFIG)}).")
    ffmpeg.add_argument("--also", type=Path, nargs="*", default=[],
                        help="Extra binaries to examine, for copies that belong to no "
                             "wheel (the build scripts pass the ffmpeg staged into vendor/).")
    ffmpeg.set_defaults(func=cmd_ffmpeg_config)

    report = subparsers.add_parser("report", help="Print name -> license, to audit.")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    dists, missing = resolve_runtime_distributions(_read_roots(args.pyproject))
    if not args.include_unshipped:
        dists = drop_unshipped(dists)
    if not dists:
        raise SystemExit("error: resolved no installed dependencies; is the environment "
                         'set up? (pip install -e ".[gui]")')
    if missing and not args.allow_missing:
        raise SystemExit(
            "error: these declared dependencies are not installed, so the document would "
            f"omit them:\n    {', '.join(sorted(missing))}\n"
            'Install the full shipped set (pip install -e ".[gui]") and re-run, or pass '
            "--allow-missing to accept an incomplete list.")
    result: int = args.func(dists, args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
