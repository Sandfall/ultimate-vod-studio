"""One-answer-per-key searches run in a torch sidecar process: the plumbing the Preview
pane's two searches share (``gui.clock_locate.ClockLocator``, ``gui.side_overlay_locate.
SideOverlayLocator``).

Each search asks a one-shot stage (``cli.clock_locate``, ``cli.side_overlay_locate``) one
question about the chosen video and remembers every answer for the session by a key that
names everything the answer depends on. ``SidecarSearch`` owns what is the same for both:

* the session cache and the request being waited on;
* the child process, launched with the conventions every GUI-spawned stage uses
  (workspace cwd, merged channels, the UTF-8 stage environment), read line by line;
* turning its progress lines into status text and its result line into the answer;
* reporting a child that died without answering.

A failure is delivered but never cached, whether the child died or the stage itself
printed the error (a video that would not open, an exception in the search, a missing
weight): the next request for the key after a cancel (the video chosen again, a run
ending) searches afresh, since a transient failure would otherwise stand for the whole
session. A repeat of the request being waited on is still a no-op, so a layout flip
after a failed search does not spawn a child that fails the same way.

A subclass supplies the stage's line parsers and wording through the constructor and
writes its own ``request``: build the key, call ``_begin``, and when that returns True
either ``_answer`` at once or ``_start_process`` with the stage's argv.

Torch runs only in the child: the GUI process is torch-free by design (see
``cli.overlay_read``'s docstring for the same stance). Import this only after
``qt_plugins.configure_qt_plugins()`` has run, as with every other GUI module.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Hashable
from dataclasses import dataclass

from PyQt6.QtCore import QObject, QProcess, pyqtSignal

from vodparse.gui.common import qobject_alive, stage_environment
from vodparse.gui.paths import PROJECT_ROOT


# What every search's status says from the moment its child process starts until the
# child reports its first sample. That wait is the process starting and loading its models
# (the clock detector, the text reader) and can run to many seconds, so the status names
# it: a search's own text with no count ("Finding the clock…") looked frozen.
STATUS_LOADING = "Loading models…"


@dataclass(frozen=True)
class SearchWording:
    """One search's status and failure text. ``progress`` shows once the child reports
    samples (formatted with ``sample`` and ``samples``; until then the status is
    ``STATUS_LOADING``); ``died``/``failed_to_start`` are the reasons given for a child
    that never answered."""

    progress: str
    died: str
    failed_to_start: str


class SidecarSearch(QObject):
    """A search answered once per key for the session (see the module docstring).

    ``parse_progress`` returns a stdout line's ``(sample, samples)`` or None;
    ``parse_result`` returns the answer a result line carries or None; ``failure`` builds
    the answer that reports a reason nothing ran; ``is_failure`` says whether an answer
    is such a report (never cached, see the module docstring). ``located`` delivers an
    answer for the request being waited on; ``searching`` carries the status text while
    a child runs.

    ``spawn=False`` builds the search with no child process, for offscreen tests: the
    argv it would have run is recorded in ``spawn_requests`` and the child's output is
    fed straight into ``_consume_text`` / ``_on_finished``."""

    searching = pyqtSignal(str)      # status text while a child process runs
    located = pyqtSignal(object)     # the answer (the subclass says what type)

    def __init__(self, parent: QObject | None, *, spawn: bool, wording: SearchWording,
                 parse_progress: Callable[[str], tuple[int, int] | None],
                 parse_result: Callable[[str], object | None],
                 failure: Callable[[str], object],
                 is_failure: Callable[[object], bool]) -> None:
        super().__init__(parent)
        self._spawn = spawn
        self._wording = wording
        self._parse_progress = parse_progress
        self._parse_result = parse_result
        self._failure = failure
        self._is_failure = is_failure
        self.spawn_requests: list[list[str]] = []   # the test seam (spawn=False)
        self._cache: dict[Hashable, object] = {}
        self._current: Hashable | None = None    # the request the caller is waiting on
        self._proc: QProcess | None = None
        self._running: Hashable | None = None    # the key the child process is answering
        self._buffer = ""                        # child stdout, may hold a partial line
        self._answered = False                   # the running child printed its result line

    # ── Requests ──────────────────────────────────────────────────────────────────

    def _begin(self, key: Hashable) -> bool:
        """Start waiting on ``key``. False when there is nothing more for the caller to
        do: a repeat of the request being waited on (its answer is on its way, or was
        delivered already; a search that died without answering forgets the request, so
        asking again retries), or a key the cache answers, delivered here at once. True
        when the caller must answer it or start a child for it; a child that was
        answering an earlier request has been killed by then."""
        if key == self._current:
            return False
        self._current = key
        self._stop_process()
        if key in self._cache:
            self.located.emit(self._cache[key])
            return False
        return True

    def cancel(self) -> bool:
        """Kill a search in flight and forget what was being waited on, so the next
        request for the same key searches again (from the cache when the answer landed
        before the cancel). For a pipeline run starting, the window closing, and a
        search that no longer applies. True when a request was being waited on (its
        progress text may be showing and nothing will come back to clear it), False
        when there was nothing to cancel."""
        was_waiting = self._current is not None
        self._stop_process()
        self._current = None
        return was_waiting

    def has_answer(self, key: Hashable) -> bool:
        """Whether the session cache holds an answer for ``key``."""
        return key in self._cache

    def cached_answer(self, key: Hashable) -> object:
        """The session cache's answer for ``key`` (check ``has_answer`` first: None is
        itself an answer for some searches)."""
        return self._cache[key]

    def _answer(self, key: Hashable, result: object) -> None:
        """Deliver ``result`` if it is what is being waited on, and remember it for ``key``
        unless it reports a failure (see the module docstring)."""
        if not self._is_failure(result):
            self._cache[key] = result
        if key == self._current:
            self.located.emit(result)

    # ── The child process ─────────────────────────────────────────────────────────

    def _start_process(self, key: Hashable, argv: list[str]) -> None:
        self._running = key
        self._buffer = ""
        self._answered = False
        self.searching.emit(STATUS_LOADING)
        if not self._spawn:
            self.spawn_requests.append(argv)
            return
        proc = QProcess(self)
        # Same launch conventions as every other stage process (see app._start_process).
        proc.setWorkingDirectory(str(PROJECT_ROOT))
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.setProcessEnvironment(stage_environment())
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        # finished never fires when the process fails to start; errorOccurred covers it.
        proc.errorOccurred.connect(self._on_error)
        self._proc = proc
        proc.start(sys.executable, argv)

    def _stop_process(self) -> None:
        """Kill the child (a no-op when none runs). Signals are blocked first so the
        deliberate kill doesn't surface as an unexpected exit."""
        proc, self._proc = self._proc, None
        self._running = None
        self._buffer = ""
        if proc is None:
            return
        proc.blockSignals(True)
        proc.kill()
        proc.waitForFinished(2000)
        proc.deleteLater()

    def _on_output(self) -> None:
        if self._proc is None or not qobject_alive(self._proc):
            return
        self._consume_text(
            self._proc.readAllStandardOutput().data().decode("utf-8", errors="replace"))

    def _consume_text(self, text: str) -> None:
        """Parse child output: progress lines update the status, the result line answers
        the running request; model-loading chatter and a trailing partial line (kept for
        the next read) are ignored."""
        self._buffer += text
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            progress = self._parse_progress(line)
            if progress is not None:
                sample, samples = progress
                self.searching.emit(self._wording.progress.format(sample=sample, samples=samples))
                continue
            result = self._parse_result(line)
            if result is not None and self._running is not None and not self._answered:
                self._answered = True
                self._answer(self._running, result)

    def _on_finished(self, _code: int, _status: QProcess.ExitStatus) -> None:
        """The child exited on its own. Normally its result line has already been
        consumed; one that died without answering is reported (and the request forgotten,
        so asking again retries) so the status line never sits on the searching text."""
        if self._running is None:
            return   # already stopped deliberately
        running, self._running = self._running, None
        if self._proc is not None:
            self._proc.deleteLater()
            self._proc = None
        if not self._answered and running == self._current:
            self._current = None   # nothing cached: the next request for the key retries
            self.located.emit(self._failure(self._wording.died))

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error != QProcess.ProcessError.FailedToStart or self._running is None:
            return
        running, self._running = self._running, None
        if self._proc is not None:
            self._proc.deleteLater()
            self._proc = None
        if running == self._current:
            self._current = None   # nothing cached: the next request for the key retries
            self.located.emit(self._failure(self._wording.failed_to_start))
