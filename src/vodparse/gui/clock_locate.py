"""The Preview pane's clock search: the client side of ``vodparse.cli.clock_locate``.

The Analyze page's layout preview wants, for the chosen video and the selected game, the
clock box the run will read at and the first in-game timestamp to land on. This object
answers one such request at a time and remembers every answer for the session, keyed by
(video path, game id): asking again for the same pair (the user flips a layout, which
changes nothing about the clock) is free and silent, and flipping games and back is
instant. Answers come from three places, tried in order:

1. the session cache;
2. the video's saved analysis (``gui.paths.recorded_clock``: a pipeline.json whose run
   calibrated a clock for this game already searched it). Switched off for now, see
   ``REUSE_SAVED_CLOCK``;
3. a ``python -m vodparse.cli.clock_locate`` child process, read line by line for its
   progress and result lines.

The cache and the child process are ``gui.sidecar_search.SidecarSearch``'s, shared with
the player-name search. A new request kills a search in flight; ``cancel`` kills it for a
pipeline run or the window closing.

Import this only after ``qt_plugins.configure_qt_plugins()`` has run, as with every
other GUI module.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject

from vodparse.cli.clock_locate import (ClockLocateResult, parse_progress_line,
                                       parse_result_line)
from vodparse.games import Game
from vodparse.gui.commands import clock_locate_argv
from vodparse.gui.paths import recorded_clock
from vodparse.gui.sidecar_search import SearchWording, SidecarSearch

# The status once the child reports samples; before that it is sidecar_search.STATUS_LOADING.
STATUS_SEARCHING = "Finding the clock…"
STATUS_SEARCHING_PROGRESS = STATUS_SEARCHING + " {sample}/{samples}"

# What a search that died without answering reports (a crash, a kill from outside, a
# missing interpreter). Like every failure, delivered but not cached (SidecarSearch).
PROCESS_DIED_ERROR = "The clock search stopped unexpectedly"
PROCESS_START_ERROR = "Couldn't start the clock search"

# Whether a saved analysis's clock box answers the search (source 2 in the module
# docstring). Off for now: the preview then always shows what a fresh search finds, so a
# box an older run got wrong is never carried forward.
REUSE_SAVED_CLOCK = False

SearchKey = tuple[str, str]
"""(video path, game id): what one answer is good for."""


def _is_failure(answer: object) -> bool:
    """Whether an answer reports that the search could not run (never cached). None, a
    game with no clock detector, is a real answer."""
    return isinstance(answer, ClockLocateResult) and answer.error is not None


class ClockLocator(SidecarSearch):
    """One clock-search answer per (video path, game id) for the session (see the module
    docstring for where answers come from). ``request`` answers now when it can and
    otherwise later through ``located``; ``searching`` carries the status text to show
    while a child process runs.

    ``located`` delivers a ``ClockLocateResult`` (which may carry ``error``), or None for
    a game with no clock detector: there is nothing to search for, and the preview draws
    the layout alone.

    ``spawn=False`` builds the client with no child process, for offscreen tests (see
    ``SidecarSearch``)."""

    def __init__(self, parent: QObject | None = None, *, spawn: bool = True) -> None:
        super().__init__(
            parent, spawn=spawn,
            wording=SearchWording(progress=STATUS_SEARCHING_PROGRESS,
                                  died=PROCESS_DIED_ERROR, failed_to_start=PROCESS_START_ERROR),
            parse_progress=parse_progress_line, parse_result=parse_result_line,
            failure=ClockLocateResult.failure, is_failure=_is_failure)

    def request(self, video_path: str, game: Game) -> None:
        """Ask for the clock of ``video_path`` under ``game``. A repeat of the request
        being waited on is a no-op, and a pair asked for before is answered from the
        cache at once (``SidecarSearch._begin``); a game with no clock detector is
        answered with None; a saved analysis answers while ``REUSE_SAVED_CLOCK`` is on;
        anything else starts a child process."""
        key: SearchKey = (video_path, game.id)
        if not self._begin(key):
            return
        if not game.has_timer_model:
            self._answer(key, None)
            return
        recorded = recorded_clock(video_path, game.id) if REUSE_SAVED_CLOCK else None
        if recorded is not None:
            self._answer(key, recorded)
            return
        self._start_process(key, clock_locate_argv(video_path, game.id))

    def answer_for(self, video_path: str, game: Game) -> tuple[bool, ClockLocateResult | None]:
        """Whether the search has answered for ``video_path`` under ``game`` this session,
        and that answer (None is an answer too: a game with no clock detector)."""
        key: SearchKey = (video_path, game.id)
        if not self.has_answer(key):
            return False, None
        answer = self.cached_answer(key)
        assert answer is None or isinstance(answer, ClockLocateResult)
        return True, answer
