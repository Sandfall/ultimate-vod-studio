"""Single entry point for the frozen (PyInstaller) build: a multi-tool dispatcher.

In a source checkout the GUI launches its stages as separate processes:
``sys.executable -m vodparse.cli.pipeline --video ...`` (see vodparse.gui.app
``_start_process``). Frozen, ``sys.executable`` is the app binary, not a Python
interpreter, so that same call re-launches *this* program. We detect the leading
``-m <module>`` form and run that module's ``__main__`` block; with no recognized
module we fall through to the GUI.

This file is the only thing PyInstaller freezes as the program; it is never used in a
normal source checkout (you run ``python -m vodparse.gui.app`` directly there).
"""

import multiprocessing
import os
import runpy
import sys

from vodparse.runtime_paths import bundle_dir, is_frozen

# The stage modules the GUI (vodparse.gui.app) shells out to. Frozen, they ride along as normal package
# modules under vodparse.cli (see UltimateVODStudio.spec's collect_submodules("vodparse")):
# no separate data-file bundling needed, since runpy.run_module finds them like any import.
_STAGE_MODULES = frozenset({
    "vodparse.cli.pipeline",
    "vodparse.cli.extract_matches",
    "vodparse.cli.export_reviewed",
    "vodparse.cli.export_diagnostics",
    "vodparse.cli.gpu_support",
    "vodparse.cli.doctr_support",
    "vodparse.cli.overlay_read",
    "vodparse.cli.clock_locate",
    "vodparse.cli.side_overlay_locate",
})


def _run() -> None:
    """Dispatch to the requested stage module's __main__, else launch the GUI."""
    if len(sys.argv) >= 3 and sys.argv[1] == "-m" and sys.argv[2] in _STAGE_MODULES:
        module = sys.argv[2]
        # Re-shape argv to what the module's argparse expects (argv[0] = its own name).
        sys.argv = [module, *sys.argv[3:]]
        runpy.run_module(module, run_name="__main__")
        return

    # The stdlib respawns this binary as ``sys.executable [flags] -c <code>`` for its
    # multiprocessing helpers, the resource tracker in particular, whenever anything
    # (a library ThreadPool included) creates a multiprocessing semaphore. Without this
    # branch that helper falls through to the GUI: a phantom app instance that reopens
    # every time it is closed, because killing the tracker just makes the next semaphore
    # respawn it. CPython may insert interpreter flags (``-s``, ``-E``, ``-B`` …) before
    # ``-c`` via _args_from_interpreter_flags(), so scan for ``-c`` instead of assuming
    # it is argv[1]. Scanning is safe here: the ``-m`` stage dispatch above has already
    # returned, so no stage argv reaches this point. The slice excludes the last element
    # because ``-c`` needs a code string after it; everything past that is the helper's
    # own argv.
    if "-c" in sys.argv[1:-1]:
        c_index = sys.argv.index("-c", 1)
        code = sys.argv[c_index + 1]
        sys.argv = ["-c", *sys.argv[c_index + 2:]]
        exec(code)  # noqa: S102 - deliberate: same trust model as `python -c` itself
        return

    from vodparse.gui import app
    app.main()


def _force_utf8_streams() -> None:
    """Reconfigure stdout/stderr to UTF-8 so library log lines can't kill a stage.

    On Windows a piped subprocess gets the locale codec (cp1252) for its standard
    streams, and the pipeline/extract log lines use characters like "→" that cp1252
    can't encode: the resulting UnicodeEncodeError aborted the whole extract stage.
    The GUI decodes subprocess output as UTF-8 already (app._on_proc_output), so
    emitting UTF-8 matches the other end. In a windowed (console=False) build the
    streams can be None or PyInstaller stand-ins without reconfigure(), hence the guards.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _point_ocr_engines_at_bundle() -> None:
    """Point EasyOCR and docTR at their bundled weights, when those weights are present.

    Frozen only (``main`` calls this under ``is_frozen()``). Both engines default to a
    per-user cache and download on first use; the bundle already carries the weights, so
    redirecting them keeps a fresh install fully offline. ``setdefault`` throughout, so a
    user/env override still wins, and set here rather than in ``vodparse.cli.pipeline`` so
    the re-invoked stage subprocesses — which also enter through this file — inherit it.

    **Both redirects are conditional on the directory existing**, which is the whole point
    of this function. ``build_windows.ps1`` / ``build_mac.sh`` stage these weights by
    downloading them and only *warn* when that fails, so a bundle can legitimately reach a
    user with no ``easyocr_models`` / ``doctr_models`` directory inside it. Pointing an
    engine at a missing path inside a read-only bundle is the worst of both worlds: it can
    neither load nor download. Leaving the variable unset lets the engine fall back to its
    own writable cache, which at least works. (Windows gates the bundled weight counts in
    CI — ``.github/workflows/build-windows.yml`` — so a weightless bundle should never ship
    there. macOS has no CI build; ``build_mac.sh`` gates the same counts itself, but only
    when someone actually runs it by hand. Either way, this is the second line of defense,
    not the first.)
    """
    bundled_easyocr = bundle_dir() / "easyocr_models"
    if bundled_easyocr.is_dir():
        os.environ.setdefault("EASYOCR_MODULE_PATH", str(bundled_easyocr))

    # vodparse.ocr_engine reads DOCTR_CACHE_DIR too, so the GUI's weights-cached check sees
    # the bundle; falling back leaves the GUI's download-offer flow working.
    bundled_doctr = bundle_dir() / "doctr_models"
    if bundled_doctr.is_dir():
        os.environ.setdefault("DOCTR_CACHE_DIR", str(bundled_doctr))


def main() -> None:
    # Must run before anything else: if this process is a multiprocessing spawn child
    # (argv carries ``--multiprocessing-fork``), run the child's code and exit instead
    # of starting the app. PyInstaller's multiprocessing runtime hook extends this to
    # all platforms, not just Windows. A no-op for normal launches.
    multiprocessing.freeze_support()

    _force_utf8_streams()

    # Frozen-only runtime wiring: the bundled OCR weights, docTR's multiprocessing escape
    # hatch, and the opt-in CUDA torch shadow. All of it also applies to the re-invoked stage
    # subprocesses, which enter through this same file.
    if is_frozen():
        # docTR batches its inference through a multiprocessing.pool.ThreadPool
        # (doctr.utils.multithreading). The workers are threads, but the Pool machinery
        # still creates multiprocessing semaphores, and the first one spawns the stdlib
        # resource-tracker helper as ``sys.executable -c ...``, which, frozen, is this
        # binary re-launching itself (see the ``-c`` branch in _run, the backstop for
        # that). This env var is docTR's documented switch to plain sequential map(),
        # avoiding the helper process entirely. setdefault so a user override wins.
        os.environ.setdefault("DOCTR_MULTIPROCESSING_DISABLE", "TRUE")

        _point_ocr_engines_at_bundle()

        # If the user has opted into GPU acceleration, its downloaded CUDA torch lives in a
        # per-user "site" dir (see vodparse.cli.gpu_support). activate_cuda_site() makes it
        # SHADOW the bundled CPU torch: for this process and, since each pipeline stage
        # re-enters this file, for every subprocess too. It works by installing a meta_path
        # finder, NOT just a sys.path entry: PyInstaller's own importer serves the bundled
        # torch from sys.meta_path, which is consulted before the sys.path-reading finder, so
        # a bare sys.path prepend never wins in the frozen app. gpu_support is stdlib-only, so
        # this stays ahead of the first torch import; it's a no-op unless a complete,
        # version-matched install is present. The sys.path prepend rides along as a harmless
        # backstop for any non-torch data files under site/.
        from vodparse import gpu_support
        site = gpu_support.activate_cuda_site()
        if site is not None:
            sys.path.insert(0, str(site))

    # A Finder double-click throws away stderr, so any startup exception is invisible.
    # Persist the full traceback (plus a bit of environment) to a file we can read back
    # : via the bind mount during dev, or from the user's workspace in the field.
    try:
        _run()
    except SystemExit:
        # A deliberate exit, not a crash: argparse's --help/usage-error paths in the
        # dispatched stage modules end in parser.exit() (SystemExit is a BaseException,
        # so without this it would be logged as launch diagnostics below). Let the
        # process exit with the intended status.
        raise
    except BaseException:  # noqa: BLE001 - last-resort launch diagnostics
        import traceback
        from vodparse.runtime_paths import workspace_dir

        from pathlib import Path

        # Write to several spots so at least one is reachable: the user's workspace,
        # and right next to the executable (which lives under the project's dist/ and
        # so is visible through the dev bind mount without any copy-paste).
        targets = [
            workspace_dir() / "launch_error.log",
            Path(sys.executable).resolve().parent / "launch_error.log",
        ]
        for log in targets:
            try:
                with open(log, "w", encoding="utf-8") as fh:
                    fh.write("argv: %r\n" % (sys.argv,))
                    fh.write("executable: %r\n" % (sys.executable,))
                    fh.write("frozen: %r  _MEIPASS: %r\n"
                             % (getattr(sys, "frozen", False), getattr(sys, "_MEIPASS", None)))
                    fh.write("sys.path:\n  " + "\n  ".join(sys.path) + "\n\n")
                    traceback.print_exc(file=fh)
            except Exception:
                pass
        traceback.print_exc()  # also to stderr, for the terminal-run case
        raise


if __name__ == "__main__":
    main()
