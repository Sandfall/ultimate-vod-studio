"""Persist crashes and OS theme-flip activity to a file, for bugs the terminal can't show.

Motivating case: on Windows, flipping the OS light/dark setting while the app runs kills the
process (see the ``_ThemeWatcher`` docstring in ``vodparse.gui.common``). Neither of the two
ways the process can die leaves anything behind on its own:

* **A Python exception inside a slot or an event handler.** PyQt6 calls ``sys.excepthook``
  and then ``qFatal()``, which aborts: so the traceback exists, but it goes to stderr, and a
  windowed (``console=False``) frozen build has no stderr.
* **A native crash inside Qt.** No Python-level traceback at all, and nothing printed on the
  way down.

:func:`install` covers both. It hooks the excepthooks (so the traceback lands in a file as
well as on stderr) and enables :mod:`faulthandler` against that same file, which dumps the
Python stack of every thread on ``SIGSEGV``/``SIGABRT``/``SIGFPE``/``SIGBUS``/``SIGILL``.
The abort behind ``qFatal`` is a ``SIGABRT``, so faulthandler catches the PyQt path too, and
an access violation inside Qt shows up as a ``SIGSEGV`` dump naming the Python frame that was
running when Qt died. A Qt message handler is installed alongside them so Qt's own warnings
and the ``qFatal`` message text are interleaved into the same log.

:func:`trace` is the breadcrumb trail, for when a crash prints nothing at all: the log's *last
line* then says which step it died on. The theme code deliberately does **not** call it, and
nor do the widget handlers that react to a palette change: those breadcrumbs were investigation
scaffolding and came back out once the crash was isolated (see the handoff). The one caller left
is the Debug menu's theme-flip hook, which records a deliberate test action rather than tracing
a suspect.

**Always on**, in source checkouts and frozen builds alike, so a user who hits a crash in the
field produces the log without a special build. Two environment overrides:

* ``VODPARSE_CRASH_LOG=0`` (or ``off``/``false``/``no``) turns logging off entirely.
* ``VODPARSE_CRASH_LOG=<path>`` writes there instead of the default location
  (:func:`crash_log_path`).

Reading it back: ``Debug → Show crash log…`` in the main GUI, or open the path
:func:`crash_log_path` reports (it is printed to stderr in the session banner at startup).
"""

import ctypes
import faulthandler
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from types import TracebackType
from typing import IO, TextIO, override

from PyQt6 import sip
from PyQt6.QtCore import (QAbstractNativeEventFilter, QByteArray, QCoreApplication, QtMsgType,
                          QMessageLogContext, qInstallMessageHandler,
                          PYQT_VERSION_STR, qVersion)

# The log's location and off-switch (CRASH_LOG_ENV, crash_log_path) resolve in runtime_paths,
# which is Qt-free, so the diagnostics bundler can find the file without importing this module;
# they are re-imported here because this module is where GUI-side callers look for them.
from vodparse.runtime_paths import (CRASH_LOG_ENV, CRASH_LOG_OFF_VALUES, crash_log_path,
                                    is_frozen)

# Opts into the Windows native-message trace (install_native_event_trace), which is off by
# default because it can itself crash the process — see that function.
NATIVE_TRACE_ENV = "VODPARSE_NATIVE_TRACE"
_ON_VALUES = frozenset({"1", "on", "true", "yes"})

# Keep the log from growing without bound across many sessions: it is opened in append mode
# (a crash tells you more when the runs around it are still there), but rotated away once it
# passes this size. One flip-and-crash session is a few kilobytes.
_MAX_LOG_BYTES = 2_000_000

# The timestamp in every log line. The pid rides alongside it (see _write): two vodparse
# processes can share one log — a second GUI instance, the standalone upload assistant, the
# vod grabber — and interleaved lines are only readable when each one says who wrote it.
_LINE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# The open log file, or None when logging is off / no install() has run yet. Module-level
# because the excepthook, the Qt message handler and faulthandler all need the same handle,
# and it must stay open for the life of the process (faulthandler writes to its fd on the
# way down, after normal interpreter shutdown may already have started).
_log_file: IO[str] | None = None

# Guards the writes, since Qt messages and thread excepthooks can arrive off the main thread.
_write_lock = threading.Lock()

# The installed native-event filter, kept alive here because Qt does not own it.
_native_event_filter: "_WindowsThemeMessageFilter | None" = None

# Names for the Qt message severities, so a log line reads "qt.warning" not "QtMsgType.2".
_QT_MSG_NAMES = {
    QtMsgType.QtDebugMsg: "qt.debug",
    QtMsgType.QtInfoMsg: "qt.info",
    QtMsgType.QtWarningMsg: "qt.warning",
    QtMsgType.QtCriticalMsg: "qt.critical",
    QtMsgType.QtFatalMsg: "qt.FATAL",
}


def logging_enabled() -> bool:
    """False when ``$VODPARSE_CRASH_LOG`` is set to an off value (see the module docstring)."""
    return os.environ.get(CRASH_LOG_ENV, "").strip().lower() not in CRASH_LOG_OFF_VALUES


def install(app_label: str) -> None:
    """Start crash logging for this process, labelling the session `app_label` (the GUI's name).

    Call as early as possible in each entry point: before the QApplication is built, so a crash
    during startup is covered too. Idempotent, and never raises: an unwritable log directory
    degrades to "no logging" rather than taking down a launch. Safe (and useful) in a frozen
    build, where it is the only way a crash leaves any record at all.
    """
    global _log_file
    if _log_file is not None or not logging_enabled():
        return
    # Resolving the path is inside the try, not above it: in a frozen build it goes through
    # runtime_paths.workspace_dir(), which touches the filesystem and raises when the workspace
    # is unwritable or gone (an unplugged drive mid-session). That must degrade to "no logging"
    # like an unwritable log does, because an exception here would escape into a launch — or,
    # from a Qt slot, into PyQt6's qFatal() and abort the process.
    path: Path | None = None
    try:
        path = crash_log_path()
        _rotate_if_large(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Line buffered, and _write flushes as well: whatever killed the process must not take
        # the last (most interesting) lines down with it in a buffer.
        _log_file = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
    except OSError as exc:
        where = f" ({path})" if path is not None else ""
        _print_stderr(f"crash log unavailable{where}: {exc}")
        return

    # Native crashes and the abort() behind qFatal: dumps every thread's Python stack to the
    # log's file descriptor, bypassing Python-level buffering (so it survives an abort).
    faulthandler.enable(file=_log_file, all_threads=True)

    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook
    qInstallMessageHandler(_qt_message_handler)

    _write(f"=== session start: {app_label} ===")
    # qVersion() is the Qt runtime that is loaded. QT_VERSION_STR is the Qt that PyQt was
    # compiled against, which can be older (PyQt6 6.11.0 on the Qt 6.11.2 runtime says 6.11.0).
    _write(f"python {sys.version.split()[0]}  Qt {qVersion()}  PyQt {PYQT_VERSION_STR}")
    _write(f"platform {sys.platform}  frozen {is_frozen()}  pid {os.getpid()}")
    _write(f"argv {sys.argv!r}")
    _print_stderr(f"crash log: {path}")


def trace(message: str) -> None:
    """Record a breadcrumb (a no-op when logging is off).

    For steps that are suspected of crashing the process, where the value is in knowing which
    one ran last. Deliberately cheap and free of Qt calls, so it is safe to call from inside an
    event handler that is itself mid-teardown.
    """
    _write(message)


def install_native_event_trace(app: QCoreApplication, *, enabled: bool | None = None) -> None:
    """Log the Windows theme-change messages as they reach the event loop (Windows only).

    **Off unless asked for**, via ``VODPARSE_NATIVE_TRACE=1`` or an explicit ``enabled=True``.
    That default is not caution for its own sake: this filter is a Python callback invoked for
    *every* native Windows message, and on 2026-07-28 it was seen to crash the app on its own.
    A diagnostic that can take down the process it observes has no business running in anyone's
    normal launch, so it is opt-in and the app never sets the variable for you.

    It once looked like the filter was *required* for the theme-flip crash to reproduce. That was
    wrong, and instructively so: the "survived without it" runs behind that reading had flipped
    the theme by writing the registry without broadcasting ``WM_SETTINGCHANGE``, which Qt notices
    on its own watcher thread — a clean stack, and not the failing path. Once the flip broadcast
    the way the Settings app does, the crash reproduced with no filter installed at all.

    What it buys when you do ask: the filter runs ahead of Qt's own handling of the message, so

    * a message logged, then the log stops → the fault is in **Qt's** processing of it;
    * no message logged at all → it never reached this app's event loop.

    Only the four theme-related messages are logged: the filter sees *every* native message, and
    logging mouse moves would bury the one line that matters. No-op off Windows, and a no-op
    without a log file to write to.
    """
    global _native_event_filter
    if enabled is None:
        enabled = os.environ.get(NATIVE_TRACE_ENV, "").strip().lower() in _ON_VALUES
    if not enabled or sys.platform != "win32" or _native_event_filter is not None:
        return
    if _log_file is None:
        return
    _native_event_filter = _WindowsThemeMessageFilter()
    # Held in a module global as well: Qt does not take ownership of the filter, and a
    # garbage-collected one silently stops filtering (or crashes on the next message).
    app.installNativeEventFilter(_native_event_filter)
    trace("native-event trace installed (Windows theme messages)")


def log_exception(context: str, exc: BaseException) -> None:
    """Record `exc`'s traceback under a `context` label, for a caller that catches, logs and
    re-raises (the theme sync does this: see ``common.apply_system_theme``)."""
    _write(f"EXCEPTION in {context}")
    _write_traceback(type(exc), exc, exc.__traceback__)


# --- the Windows theme-message filter ----------------------------------------

# The four messages Windows sends an app when the appearance changes. WM_SETTINGCHANGE is the
# one that carries "ImmersiveColorSet" (the light/dark flip proper) and the one Qt turns into
# colorSchemeChanged; the other three ride along and are logged because a crash between them
# tells us which stage of Qt's response faulted.
_WM_SYSCOLORCHANGE = 0x0015
_WM_SETTINGCHANGE = 0x001A
_WM_THEMECHANGED = 0x031A
_WM_DWMCOLORIZATIONCOLORCHANGED = 0x0320

# The "no result" value a native-event filter returns alongside "not handled". Built once:
# the filter runs for every native message the app receives.
_NOT_HANDLED = sip.voidptr(0)


class _MSG(ctypes.Structure):
    """The Win32 ``MSG`` struct, as much of it as :func:`_windows_message_id` needs to read.

    Module level on purpose: defining a ``ctypes.Structure`` allocates a heap type, and
    ``ctypes.POINTER`` interns the pointer type it derives from it in a process-wide cache that
    is never pruned. Per call — and this is called for *every* native Windows message — that
    leaks a few kilobytes a message and churns type objects for no benefit. ``ctypes.Structure``
    subclasses fine on every platform, so no guard is needed here; the win32-only check stays
    where the pointer is actually dereferenced.
    """

    _fields_ = [("hwnd", ctypes.c_void_p),
                ("message", ctypes.c_uint),
                ("wParam", ctypes.c_size_t),
                ("lParam", ctypes.c_ssize_t),
                ("time", ctypes.c_uint),
                ("pt_x", ctypes.c_long),
                ("pt_y", ctypes.c_long)]


# Built once for the same reason (see _MSG): ctypes caches pointer types forever.
_MSG_POINTER = ctypes.POINTER(_MSG)


_THEME_MESSAGE_NAMES = {
    _WM_SYSCOLORCHANGE: "WM_SYSCOLORCHANGE",
    _WM_SETTINGCHANGE: "WM_SETTINGCHANGE",
    _WM_THEMECHANGED: "WM_THEMECHANGED",
    _WM_DWMCOLORIZATIONCOLORCHANGED: "WM_DWMCOLORIZATIONCOLORCHANGED",
}


class _WindowsThemeMessageFilter(QAbstractNativeEventFilter):
    """Logs theme-related Windows messages on their way in, then lets Qt handle them normally.

    Installed by :func:`install_native_event_trace`; see there for what the log proves. The
    filter never consumes a message (it always returns "not handled"), so it cannot change
    behavior — only observe it.
    """

    @override
    def nativeEventFilter(self, eventType: QByteArray | bytes | bytearray | memoryview,
                          message: sip.voidptr) -> tuple[bool, sip.voidptr]:
        message_id = _windows_message_id(message)
        if message_id in _THEME_MESSAGE_NAMES:
            trace(f"native: {_THEME_MESSAGE_NAMES[message_id]} arrived "
                  f"(Qt has not handled it yet)")
        # (not handled, no result): Qt goes on to process the message exactly as it would
        # have without this filter, which is what keeps the trace observation-only.
        return False, _NOT_HANDLED


def _windows_message_id(message: sip.voidptr) -> int:
    """The ``MSG.message`` field of a native event, or 0 when it can't be read.

    PyQt hands the ``MSG`` over as an opaque pointer, so it is read back through ctypes.
    Defensive to the point of paranoia: this runs for every single native message the app
    receives, and a mistake here would crash the very process we are trying to observe.
    """
    if sys.platform != "win32":
        return 0
    try:
        address = int(message)  # type: ignore[call-overload]  # sip.voidptr
        return int(ctypes.cast(address, _MSG_POINTER).contents.message)
    except Exception:  # noqa: BLE001 - an unreadable message is not worth a crash
        return 0


# --- internals ---------------------------------------------------------------


def _rotate_if_large(path: Path) -> None:
    """Move an oversized log aside to ``<name>.1`` so appending stays bounded."""
    try:
        if path.is_file() and path.stat().st_size > _MAX_LOG_BYTES:
            path.replace(path.with_suffix(path.suffix + ".1"))
    except OSError:
        pass  # a log we can't rotate is not a reason to skip logging


def _print_stderr(message: str) -> None:
    """Echo to stderr when there is one. A windowed frozen build has ``sys.stderr is None``
    (or a stand-in that fails on write), and this module must never be the thing that breaks a
    launch, so every terminal echo goes through here."""
    stream: TextIO | None = sys.stderr
    if stream is None:
        return
    try:
        print(message, file=stream)
    except (OSError, ValueError):
        pass


def _write(message: str) -> None:
    """Append one line stamped ``[<date> <time> pid <pid>]``, flushed.

    Silently does nothing when logging is off. The pid is part of the stamp because the log is
    per user, not per process, and processes share it (see :data:`_LINE_TIME_FORMAT`).
    """
    if _log_file is None:
        return
    stamp = f"{time.strftime(_LINE_TIME_FORMAT)} pid {os.getpid()}"
    with _write_lock:
        try:
            _log_file.write(f"[{stamp}] {message}\n")
            _log_file.flush()
        except (OSError, ValueError):
            pass  # a closed or full log must not raise inside an event handler


def _write_traceback(exc_type: type[BaseException], exc: BaseException,
                     tb: TracebackType | None) -> None:
    """Append a formatted traceback to the log (no timestamp prefix: it is multi-line).

    Formatted in full and written in *one* call rather than line by line: the log is opened in
    append mode, so a single moderate write lands whole, whereas one write per line lets a
    second vodparse process splice its own lines through the middle of this traceback.
    """
    if _log_file is None:
        return
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    with _write_lock:
        try:
            _log_file.write(text)
            _log_file.flush()
        except (OSError, ValueError):
            pass


def _excepthook(exc_type: type[BaseException], exc: BaseException,
                tb: TracebackType | None) -> None:
    """Log an unhandled exception, then hand it to the default hook (which prints to stderr).

    This is the hook PyQt6 calls for an exception that escapes a slot or a reimplemented
    virtual, just before it calls ``qFatal()`` and the process aborts: so it is the only
    chance to capture that traceback in a windowed build.
    """
    _write(f"UNHANDLED {exc_type.__name__}: {exc}")
    _write_traceback(exc_type, exc, tb)
    try:
        sys.__excepthook__(exc_type, exc, tb)
    except BaseException:  # noqa: BLE001 - no stderr (windowed build); the file log has it
        pass


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    """The same for an exception escaping a non-main thread's run()."""
    name = args.thread.name if args.thread is not None else "?"
    _write(f"UNHANDLED {args.exc_type.__name__} in thread {name}: {args.exc_value}")
    _write_traceback(args.exc_type, args.exc_value or args.exc_type(), args.exc_traceback)
    try:
        threading.__excepthook__(args)
    except BaseException:  # noqa: BLE001 - same as _excepthook: stderr may not exist
        pass


def _qt_message_handler(msg_type: QtMsgType, context: QMessageLogContext,
                        message: str | None) -> None:
    """Log Qt's own diagnostics into the same file, and keep printing them to stderr.

    Qt warnings around a crash are often the whole story ("QWidget: Must construct a
    QApplication…", a style complaining mid-teardown), and ``qFatal``'s message text arrives
    here as ``QtFatalMsg`` immediately before the abort.
    """
    label = _QT_MSG_NAMES.get(msg_type, "qt.?")
    where = f" ({context.file}:{context.line})" if context.file else ""
    _write(f"{label}: {message}{where}")
    _print_stderr(f"{label}: {message}{where}")  # Qt's own handler no longer prints it
