"""The Preview pane's player-name search: the client side of
``vodparse.cli.side_overlay_locate``.

For a game whose player names sit in the bars beside the picture (Melee), under the
built-in "Auto" reading, the Analyze page's layout preview wants the rows an Auto run
would lock, to draw them and read them on the frame being scrubbed. This object answers
one such request at a time and remembers every answer for the session, keyed by
everything the answer depends on: the video, the game, the Font style and language the
bars are read with, and the clock box the bars are placed from. Answers come from the
session cache, else from a ``python -m vodparse.cli.side_overlay_locate`` child process
(``gui.sidecar_search.SidecarSearch`` holds the cache and the process).

The main window asks once the clock search has answered (the bars are placed from its
box) and hands the answer both to the preview and, when a run starts, to the run
(``vodparse.prepass_handover``), which then does not search again.

Import this only after ``qt_plugins.configure_qt_plugins()`` has run, as with every
other GUI module.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject

from vodparse.cli.side_overlay_locate import (SideOverlayLocateResult, parse_progress_line,
                                              parse_result_line)
from vodparse.games import Game
from vodparse.gui.commands import side_overlay_locate_argv
from vodparse.gui.sidecar_search import SearchWording, SidecarSearch
from vodparse.schema import BBox

# The status once the child reports samples; before that it is sidecar_search.STATUS_LOADING.
STATUS_SEARCHING = "Finding player tags…"
STATUS_SEARCHING_PROGRESS = STATUS_SEARCHING + " {sample}/{samples}"
PROCESS_DIED_ERROR = "The player tag search stopped unexpectedly"
PROCESS_START_ERROR = "Couldn't start the player tag search"

SearchKey = tuple[str, str, str, str, tuple[int, ...] | None]
"""(video path, game id, OCR engine, OCR language, clock box): what one answer is good for."""


def _is_failure(answer: object) -> bool:
    """Whether an answer reports that the search could not run (never cached)."""
    return isinstance(answer, SideOverlayLocateResult) and answer.error is not None


def search_key(video_path: str, game: Game, ocr_engine: str, ocr_lang: str,
               clock_bbox: BBox | None) -> SearchKey:
    """The cache key for one search (see ``SearchKey``)."""
    return (video_path, game.id, ocr_engine, ocr_lang,
            tuple(clock_bbox) if clock_bbox is not None else None)


class SideOverlayLocator(SidecarSearch):
    """One player-name-search answer per ``SearchKey`` for the session. ``request``
    answers now when the cache can and otherwise later through ``located``, which
    delivers a ``SideOverlayLocateResult`` (which may carry ``error``); ``searching``
    carries the status text to show while a child process runs.

    ``spawn=False`` builds the client with no child process, for offscreen tests (see
    ``SidecarSearch``)."""

    def __init__(self, parent: QObject | None = None, *, spawn: bool = True) -> None:
        super().__init__(
            parent, spawn=spawn,
            wording=SearchWording(progress=STATUS_SEARCHING_PROGRESS,
                                  died=PROCESS_DIED_ERROR, failed_to_start=PROCESS_START_ERROR),
            parse_progress=parse_progress_line, parse_result=parse_result_line,
            failure=SideOverlayLocateResult.failure, is_failure=_is_failure)

    def request(self, video_path: str, game: Game, ocr_engine: str, ocr_lang: str,
                clock_bbox: BBox | None) -> None:
        """Ask for the player-name rows of ``video_path`` under ``game``, read with
        ``ocr_engine`` in ``ocr_lang``, the bars placed from ``clock_bbox`` (None: the
        picture is assumed centred). The caller asks only for a game with side bars. A
        repeat of the request being waited on is a no-op, and a key asked for before is
        answered from the cache at once (``SidecarSearch._begin``); anything else starts
        a child process, killing one that was answering an earlier request."""
        key = search_key(video_path, game, ocr_engine, ocr_lang, clock_bbox)
        if not self._begin(key):
            return
        self._start_process(key, side_overlay_locate_argv(
            video_path, game.id, ocr_engine=ocr_engine, ocr_lang=ocr_lang, clock_bbox=clock_bbox))

    def answer_for(self, video_path: str, game: Game, ocr_engine: str, ocr_lang: str,
                   clock_bbox: BBox | None) -> SideOverlayLocateResult | None:
        """The answer the search gave for these inputs this session, or None when it has
        not answered for them."""
        key = search_key(video_path, game, ocr_engine, ocr_lang, clock_bbox)
        if not self.has_answer(key):
            return None
        answer = self.cached_answer(key)
        assert isinstance(answer, SideOverlayLocateResult)
        return answer
