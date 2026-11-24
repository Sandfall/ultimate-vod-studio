"""Hand a run what the layout preview already searched for, so it is not searched twice.

Before the main loop, an Auto run searches the VOD twice: the clock pre-pass
(``vodparse.clock_calibration``) and, for a game with side bars, the player-name search
(``vodparse.side_overlay``). The Analyze page's layout preview runs both of those same
searches as soon as a video is chosen (``cli.clock_locate``, ``cli.side_overlay_locate``:
the same samples and the same rules), so by the time the user presses Analyze the answers
usually exist already. The main window writes them to a small JSON file and names it on
the run's command line (``--prepass-handover``); the run uses each answer it is given in
place of the matching search and records it in its own config exactly as if it had
searched itself.

The file says what the answers were found *for*, and the run uses nothing from a file
that does not match it::

    {"video": "/abs/vod.mp4", "game": "melee",
     "clock_calibration": {...ClockCalibration.as_config()...} | null,
     "clock_searched": true | false,
     "clock_never_fired": true | false,
     "side_overlay": {"ocr_engine": "doctr", "ocr_lang": "en",
                      "record": {...SideOverlayScan.as_config()...}} | null}

* ``video`` and ``game`` must be the run's own: a clock box is a property of one video
  under one game's detector.
* ``clock_searched`` says the preview's clock search ran and calibrated nothing (the
  detector fired nowhere, or on no steady spot). The run's own pre-pass samples the same
  frames by the same rules and would conclude the same, so it skips the pre-pass and
  reads every sample with the detector, as it does after its own refusal.
* ``clock_never_fired`` narrows ``clock_searched``: the detector fired on none of the
  search's frames. The side-bar search reads only frames with a clock, so the run skips
  that pre-pass too, as it does when its own clock pre-pass sees no clock.
* ``side_overlay`` also names the Font style and language it was read with; the rows the
  search locks depend on what the OCR engine could read, so a run with another engine or
  language searches for itself. The record itself carries the feed span its bars were
  placed from, and the run (``cli.pipeline._handed_over_side_overlay``) takes the rows
  only when its own clock pre-pass places the bars at the same edges.

A missing, unreadable or mismatched file is never an error: the run searches as it always
has. Qt-free and import-light: the GUI writes the file, the pipeline reads it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PrepassHandover:
    """The searches' answers for one video under one game (see the module docstring).
    ``clock_calibration`` and ``side_overlay_record`` are the searches' own JSON records,
    None for a search with no answer to hand over; ``ocr_engine``/``ocr_lang`` are what the
    side-bar search read with, and only mean something beside its record."""

    video: str
    game: str
    clock_calibration: dict[str, Any] | None = None
    clock_searched: bool = False
    clock_never_fired: bool = False
    side_overlay_record: dict[str, Any] | None = None
    ocr_engine: str | None = None
    ocr_lang: str | None = None

    @property
    def is_empty(self) -> bool:
        """True when there is no answer to hand over at all: no clock box, no clock search
        that came back empty, no rows."""
        return (self.clock_calibration is None and not self.clock_searched
                and self.side_overlay_record is None)

    def to_json(self) -> dict[str, Any]:
        """The JSON object written to the handover file."""
        side_overlay: dict[str, Any] | None = None
        if self.side_overlay_record is not None:
            side_overlay = {"ocr_engine": self.ocr_engine, "ocr_lang": self.ocr_lang,
                            "record": self.side_overlay_record}
        return {"video": self.video, "game": self.game,
                "clock_calibration": self.clock_calibration,
                "clock_searched": self.clock_searched,
                "clock_never_fired": self.clock_never_fired, "side_overlay": side_overlay}

    def clock_calibration_for(self, video: str, game: str) -> dict[str, Any] | None:
        """The clock record, when it was found for this ``video`` under this ``game``."""
        if not self._is_for(video, game):
            return None
        return self.clock_calibration

    def clock_searched_for(self, video: str, game: str) -> bool:
        """Whether the preview's clock search already ran for this ``video`` under this
        ``game`` and calibrated nothing (see the module docstring)."""
        return self._is_for(video, game) and self.clock_searched

    def clock_never_fired_for(self, video: str, game: str) -> bool:
        """Whether that clock search found the detector firing on none of its frames, for
        this ``video`` under this ``game`` (see the module docstring)."""
        return self._is_for(video, game) and self.clock_never_fired

    def side_overlay_for(self, video: str, game: str, ocr_engine: str,
                         ocr_lang: str) -> dict[str, Any] | None:
        """The side-bar record, when it was found for this ``video`` under this ``game``
        and read with this Font style and language."""
        if not self._is_for(video, game):
            return None
        if self.ocr_engine != ocr_engine or self.ocr_lang != ocr_lang:
            return None
        return self.side_overlay_record

    def _is_for(self, video: str, game: str) -> bool:
        return self.game == game and _same_file(self.video, video)


def _same_file(first: str, second: str) -> bool:
    """Whether two paths name the same file, however each was spelled."""
    return Path(first).resolve() == Path(second).resolve()


def write_handover(path: Path, handover: PrepassHandover) -> bool:
    """Write ``handover`` to ``path``. False when the file could not be written: the caller
    then starts the run without it, which only costs the searches."""
    try:
        path.write_text(json.dumps(handover.to_json()), encoding="utf-8")
    except OSError:
        return False
    return True


def read_handover(path: Path) -> PrepassHandover | None:
    """The handover written to ``path``, or None when the file is missing, unreadable or
    not in the shape ``write_handover`` writes (the run then searches for itself)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    video, game = payload.get("video"), payload.get("game")
    if not isinstance(video, str) or not isinstance(game, str):
        return None
    clock = payload.get("clock_calibration")
    clock_searched = payload.get("clock_searched")
    clock_never_fired = payload.get("clock_never_fired")
    side = payload.get("side_overlay")
    record: dict[str, Any] | None = None
    ocr_engine: str | None = None
    ocr_lang: str | None = None
    if isinstance(side, dict) and isinstance(side.get("record"), dict):
        engine, lang = side.get("ocr_engine"), side.get("ocr_lang")
        if isinstance(engine, str) and isinstance(lang, str):
            record, ocr_engine, ocr_lang = side["record"], engine, lang
    return PrepassHandover(video=video, game=game,
                           clock_calibration=clock if isinstance(clock, dict) else None,
                           clock_searched=clock_searched is True,
                           clock_never_fired=clock_never_fired is True,
                           side_overlay_record=record, ocr_engine=ocr_engine, ocr_lang=ocr_lang)
