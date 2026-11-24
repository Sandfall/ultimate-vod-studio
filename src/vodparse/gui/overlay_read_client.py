"""The client side of the overlay read helper (``vodparse.cli.overlay_read``).

One warm child process serves OCR reads of a captured frame, one JSON request line per
read, so the torch import and OCR model load are paid once per session rather than once
per read. Two callers share it: the overlay-layout editor's readings (the
drawn boxes, ``request``) and the Analyze page's layout preview (the drawn boxes of the
selected preset, or under Auto the pipeline's own band read + tag vote, ``request_auto``).
Either way the answer is a ``RegionReading`` per region key, delivered through
``regions_read`` with the caller's token so a superseded request is dropped.

OCR runs in a separate *process* on purpose: the GUI process is torch-free by design
(see ``vodparse.cli.doctr_support``'s docstring for the same stance), and torch on an
in-process Qt thread fatally crashed the interpreter on macOS.

Import this only after ``qt_plugins.configure_qt_plugins()`` has run, as with every
other GUI module.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import QObject, QProcess, pyqtSignal

from vodparse.cli.overlay_read import RESULT_LINE_PREFIX
from vodparse.gui.common import stage_environment
from vodparse.gui.paths import PROJECT_ROOT
from vodparse.schema import BBox


@dataclass(frozen=True)
class RegionReading:
    """What a read found for one region: the joined text (empty when nothing was read)
    and the best single-word confidence among the reads. ``raw`` is set only on a
    timer-box parse failure: the string(s) OCR actually saw (variants joined with "|"),
    so the result line can say why no clock parsed. ``sponsor`` is the sponsor split off
    a player row (None when the player has none). ``bbox`` is where the text was read, in
    source-frame pixels, for a read that searched for it (the Auto band read); None when
    the read covered a box the caller drew, since that box is the location."""

    text: str
    confidence: float
    raw: str = ""
    sponsor: str | None = None
    bbox: BBox | None = None


# How a player reading shows its sponsor wherever the GUI prints one: the sponsor, a bar,
# then the tag, the way a stream plate lays them out. See tag_display_text.
SPONSOR_TAG_SEPARATOR = " | "


def tag_display_text(reading: RegionReading) -> str:
    """A player reading as the GUI prints it: ``Sponsor | Tag`` when the row carried a
    sponsor, the tag alone otherwise. Shared by the Analyze page's readout and the layout
    editor's test-read labels so the two screens never disagree."""
    if reading.sponsor:
        return f"{reading.sponsor}{SPONSOR_TAG_SEPARATOR}{reading.text}"
    return reading.text


def reading_from_payload(entry: dict[str, Any]) -> RegionReading:
    """One region's entry of a helper response as a RegionReading, defensively typed."""
    bbox = entry.get("bbox")
    sponsor = entry.get("sponsor")
    return RegionReading(
        text=str(entry.get("text", "")),
        confidence=float(entry.get("confidence", 0.0)),
        raw=str(entry.get("raw", "")),
        sponsor=sponsor if isinstance(sponsor, str) else None,
        bbox=[int(v) for v in bbox] if isinstance(bbox, list) and len(bbox) == 4 else None,
    )


class OverlayReadProcess(QObject):
    """A warm ``vodparse.cli.overlay_read`` QProcess answering one request line at a
    time (see the module docstring). The helper is spawned lazily on the first request
    and killed by ``stop()``; an unexpected exit is surfaced as ``failed`` and the next
    request simply respawns it."""

    regions_read = pyqtSignal(object, int)   # dict[str, RegionReading], request token
    failed       = pyqtSignal(str, int)      # status message, request token
    # A note the helper attached to an answered read (it left regions unread and says why:
    # see the helper's INGAME_NAMES_UNREAD_NOTE), emitted just before regions_read.
    read_noted   = pyqtSignal(str, int)      # note, request token

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._buffer = ""        # stdout accumulated across reads; may hold partial lines
        self._last_token = -1    # token of the most recent request, for spawn/exit errors

    def request(self, engine: str, lang: str, frame_path: str,
                rects_px: dict[str, list[int]], token: int, *,
                ingame_engine: str | None = None,
                background: dict[str, Any] | None = None) -> None:
        """Read the given boxes (region key -> full-frame pixel box) on the frame saved
        at ``frame_path``; the response arrives later as ``regions_read`` or ``failed``
        carrying ``token``. ``ingame_engine`` is the engine the game pins for its in-game
        nameplate band (``games.ingame_band_engine_for``), which the helper reads the
        in-game boxes with, as a run does; None reads them with ``engine``. It carries the
        engine and never the game (see the helper's docstring), except inside
        ``background``: the fields a run under this layout reads with the built-in
        readers because no box is drawn for them, ``{"game", "ingame_tag_frac",
        "ingame_band", "clock_bbox", "keys"}``: what ``request_auto`` sends, the band mode
        the layout reads the nameplate band as, and the region keys to
        answer (``preview_regions.preview_background_keys``). None reads the boxes only."""
        payload: dict[str, Any] = {"token": token, "engine": engine, "lang": lang,
                                   "frame_path": frame_path, "regions": rects_px,
                                   "ingame_engine": ingame_engine}
        if background is not None:
            payload["background"] = background
        self._send(payload)

    def request_auto(self, engine: str, lang: str, frame_path: str, *, game: str,
                     ingame_tag_frac: float | None, clock_bbox: BBox | None,
                     token: int, locked: dict[str, BBox] | None = None,
                     pipe_evidence_run: str | None = None) -> None:
        """Read the frame the way an Auto run (no custom overlay layout) reads it: the
        pipeline's band OCR and single-frame tag finders for ``game``, plus the clock at
        ``clock_bbox`` when the clock search found one (see the helper's docstring for
        the request shape). ``locked`` is the rows the side-bar search locked for the
        video (region key -> pixel box), which then answer for the players as they do in
        a run; None reads the players from the band. ``pipe_evidence_run`` is the video's
        stored pipeline.json when it has an analysis: its pipe evidence lets the read fold
        a pipe docTR misread as a letter, as the run did. Each answered region carries where
        it was read."""
        self._send({"token": token, "engine": engine, "lang": lang,
                    "frame_path": frame_path, "auto": {
                        "game": game, "ingame_tag_frac": ingame_tag_frac,
                        "clock_bbox": clock_bbox, "locked": locked,
                        "pipe_evidence_run": pipe_evidence_run}})

    def stop(self) -> None:
        """Kill the helper (a no-op when never spawned). Signals are blocked first so
        the deliberate kill doesn't surface as an 'exited unexpectedly' failure."""
        proc, self._proc = self._proc, None
        self._buffer = ""
        if proc is None:
            return
        proc.blockSignals(True)
        proc.kill()
        proc.waitForFinished(2000)
        proc.deleteLater()

    def _send(self, payload: dict[str, Any]) -> None:
        self._last_token = int(payload["token"])
        proc = self._ensure_started()
        proc.write((json.dumps(payload) + "\n").encode("utf-8"))

    def _ensure_started(self) -> QProcess:
        if self._proc is not None:
            return self._proc
        proc = QProcess(self)
        # Same launch conventions as every other stage process (see app._start_process):
        # workspace cwd, merged channels, the UTF-8 stage environment.
        proc.setWorkingDirectory(str(PROJECT_ROOT))
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.setProcessEnvironment(stage_environment())
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        # finished never fires when the process fails to start; errorOccurred covers it.
        proc.errorOccurred.connect(self._on_error)
        # Stored before start(): a FailedToStart can be delivered from inside start()
        # itself, and _on_error ignores it while _proc is still None.
        self._proc = proc
        proc.start(sys.executable, ["-m", "vodparse.cli.overlay_read"])
        return proc

    def _on_output(self) -> None:
        if self._proc is None:
            return
        self._consume_text(
            self._proc.readAllStandardOutput().data().decode("utf-8", errors="replace"))

    def _consume_text(self, text: str) -> None:
        """Parse helper output: only complete lines bearing RESULT_LINE_PREFIX are
        protocol responses; model-loading chatter and partial lines are kept out of the
        way (the buffer holds any trailing partial line for the next read)."""
        self._buffer += text
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            if not line.startswith(RESULT_LINE_PREFIX):
                continue
            try:
                payload = json.loads(line[len(RESULT_LINE_PREFIX):])
            except ValueError:
                continue   # a torn or corrupted line; the request will surface elsewhere
            token = int(payload.get("token", -1))
            if "error" in payload:
                self.failed.emit(str(payload["error"]), token)
                continue
            note = payload.get("note")
            if isinstance(note, str) and note:
                self.read_noted.emit(note, token)
            readings = {region: reading_from_payload(entry)
                        for region, entry in payload.get("readings", {}).items()}
            self.regions_read.emit(readings, token)

    def _on_finished(self, _code: int, _status: QProcess.ExitStatus) -> None:
        """The helper died on its own (a crash, an OOM kill): report it against the
        last request and forget the process so the next request respawns a fresh one."""
        if self._proc is None:
            return   # already stopped deliberately
        self._proc.deleteLater()
        self._proc = None
        self._buffer = ""
        self.failed.emit("The text reader stopped unexpectedly. Scrub to another frame to "
                         "try again.", self._last_token)

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error != QProcess.ProcessError.FailedToStart or self._proc is None:
            return
        self._proc.deleteLater()
        self._proc = None
        self.failed.emit("Couldn't start the text reader", self._last_token)
