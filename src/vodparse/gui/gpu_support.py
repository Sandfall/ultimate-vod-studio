"""GPU-acceleration setup for the desktop GUI (app.py): the download/probe front end.

:class:`GpuSupportManager` drives the optional CUDA-PyTorch install from the GUI: it offers
the (multi-gigabyte) download behind a size warning, runs the headless installer
(``vodparse.cli.gpu_support install``) on a ``QProcess`` behind a modal cancelable progress
dialog (parsing its ``PROGRESS`` lines), and reports success/failure. It also runs the
lightweight ``probe`` subcommand to learn what compute the pipeline will actually use, so
Settings/menus can show "Apple GPU (Metal/MPS)", "NVIDIA GPU (CUDA …)", or "CPU" on every
platform without the (torch-free) GUI ever importing torch itself.

The download applies only to the **frozen app**: ``activate_cuda_site()`` runs solely from
``freeze_entry``, so on a source checkout the downloaded site would never be used. There the
offer and the panel's install/remove buttons are replaced by a pip hint (install the CUDA
torch into the venv); the compute-status readout still works everywhere.

Mirrors the QProcess + QProgressDialog pattern of :mod:`vodparse.gui.convert`. Import this
only after ``qt_plugins.configure_qt_plugins()`` has run (every app entry module does so at
its top), since it imports PyQt6 widgets at load time.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import Qt, QEvent, QObject, QProcess, pyqtSignal
from PyQt6.QtGui import QHideEvent, QShowEvent
from PyQt6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                             QMessageBox, QProgressDialog, QPushButton, QVBoxLayout, QWidget)

from vodparse import gpu_support
from vodparse.gui.common import BusyDialog, qobject_alive, stage_environment
from vodparse.runtime_paths import is_frozen

# The compute-status snapshot as returned by device.describe_compute() and parsed back from the
# probe subprocess's JSON. Keeps the probe plumbing typed consistently with its source.
ComputeInfo = dict[str, object]

# Every deferred touch of a QProcess or dialog below is guarded by ``qobject_alive`` (see
# vodparse.gui.common): a fire-and-forget QProcess, or the widget awaiting its result, can be
# torn down before its ``finished`` slot runs — the Settings dialog closed via Escape while a
# probe is mid-flight. Each such QProcess is parented through the manager to the requesting
# panel, so a deleted ``proc`` also means the panel (and its callback target) is gone: one
# guard covers both.

# The stage argv the GUI shells out to (frozen: sys.executable is the app exe, which
# re-enters freeze_entry and dispatches this module: same seam the pipeline uses).
_INSTALL_MODULE = "vodparse.cli.gpu_support"

# The dialog label shown for each installer ``STAGE`` (see vodparse.cli.gpu_support). Downloading is
# byte-smooth, but unpacking (a slow multi-gigabyte inflate) and the final swap emit sparse progress,
# so a phase-specific label reassures the user the bar isn't hung when it lingers near the end. The
# "downloading" text is also the dialog's initial label, so this map is its single source of truth.
_STAGE_LABELS = {
    "downloading": "Downloading GPU support from PyTorch…",
    "unpacking": "Unpacking the download… this can take a minute.",
    "finishing": "Finishing up…",
}

# "More info" link in the install offer: PyTorch's own install page, which explains the CUDA
# build we download. QMessageBox renders the <a href> as rich text but does NOT open it on click
# by default (its label's openExternalLinks is off), so the offer builder turns that on for the
# message label: otherwise the link looks clickable but does nothing.
_PYTORCH_INFO_URL = "https://pytorch.org/get-started/locally/"

# The "no NVIDIA GPU" refusal, shown wherever the question is answered (the offer's own check
# and the panel's cached-probe path) so the two spots can't drift apart.
_NO_GPU_TITLE = "No NVIDIA GPU detected"
_NO_GPU_TEXT = "No NVIDIA GPU was found on this machine. GPU acceleration can't be used."

# The visible first-run check (see GpuSupportManager.check_then_offer). The label wording
# matches the GpuStatusPanel's "Checking compute…" family so the two readouts feel related.
_GPU_CHECK_TITLE = "Checking for GPU"
_GPU_CHECK_TEXT = "Checking for a compatible NVIDIA GPU…"
# The first-run conclusion for machines without an NVIDIA GPU. Distinct from _NO_GPU_TEXT:
# that one refuses an explicit install request, while this closes an unprompted check, so it
# reports what will happen instead of refusing.
_NO_GPU_RESULT_TEXT = "No compatible NVIDIA GPU was found: analysis will run (slowly) on the CPU."

# How long to wait for a killed probe to actually exit before giving up and letting teardown
# proceed. A killed child dies promptly; this only exists so the process is out of Running state
# before its QProcess is destroyed.
PROBE_KILL_WAIT_MS = 2000

# The last compute-status probe result this session. The probe spawns a torch-importing
# subprocess (a few seconds), so we run it once and share the answer across panels; it's
# cleared whenever the install set changes (see GpuSupportManager._mark_changed), since the
# device may then differ from the cached reading.
_probe_cache: ComputeInfo | None = None

# Bumped whenever the install set changes. A probe captures the value when it spawns and drops
# its result if the world moved on before it finished: otherwise a slow pre-change probe (the
# CUDA torch takes seconds to cold-load) lands *after* the fast post-change one and repopulates
# the just-invalidated cache, e.g. showing an uninstalled GPU as active for the whole session.
_probe_generation = 0

# The manager whose install is currently in flight, or None. PROCESS-wide, not per-manager:
# the app runs several managers (MainWindow._gpu plus one per GpuStatusPanel), and two
# concurrent installer subprocesses would rmtree each other's staging/. A dead owner (its
# widget torn down mid-install, killing the child QProcess) no longer blocks: see qobject_alive.
_install_owner: "GpuSupportManager | None" = None


def _install_in_flight() -> bool:
    """True while any manager in this process has an installer subprocess running."""
    return _install_owner is not None and qobject_alive(_install_owner)


def _invalidate_probe_cache() -> None:
    """Forget this session's cached compute-status probe and mark in-flight probes stale. Called
    whenever the install set changes (any install or uninstall) so the next probe: and any
    GpuStatusPanel shown afterwards: reflect the new device instead of a stale reading. It lives
    at module scope, not on a panel, because the cache is shared across every panel and some
    installs (Debug 'Force download', the first-run offer) run on a manager that owns no panel:
    see GpuSupportManager._mark_changed."""
    global _probe_cache, _probe_generation
    _probe_cache = None
    _probe_generation += 1


def _discard_staging() -> None:
    """Delete the installer's scratch ``staging/`` dir after an install ends without success. The
    installer process is dead by then, so nothing holds those files: and a cancelled download
    would otherwise strand multi-gigabyte partial wheels that ``install_present()`` deliberately
    ignores (so no Remove button ever reclaims them) until some future install run clears them.
    Best-effort: on Windows a just-killed child can briefly hold a handle; the next install's own
    staging reset is the backstop."""
    shutil.rmtree(gpu_support.gpu_staging_dir(), ignore_errors=True)


@dataclass
class _Install:
    """Transient state of an in-flight CUDA install (see GpuSupportManager._start_install)."""
    proc: QProcess
    dialog: QProgressDialog
    output: str = ""
    pending: str = ""  # bytes read but not yet ending in a newline: the tail of a split line
    cancelled: bool = False
    # Highest PROGRESS value shown so far. QProgressDialog.setValue on a shown modal dialog runs
    # processEvents, which can deliver the NEXT readyRead re-entrantly: when the suspended outer
    # loop then resumes, it would replay its remaining (older, lower) lines and drive the bar
    # backwards. Never setting a lower value makes the replay harmless. -1 so the initial 0 shows.
    highest_progress: int = -1


def _split_complete_lines(pending: str, chunk: str) -> tuple[list[str], str]:
    """Split ``pending + chunk`` into complete (newline-terminated) lines plus the leftover partial
    tail to carry into the next read. Buffering the tail is what lets a control line split across two
    QProcess reads: e.g. ``"PROG"`` then ``"RESS 42\\n"``: be reassembled rather than dropped (which
    would stall the progress bar or lose a stage change). Pure, so it's unit-testable without a live
    QProcess; :func:`_parse_progress` / :func:`_parse_stage` interpret the lines it returns."""
    *lines, leftover = (pending + chunk).split("\n")
    return lines, leftover


def _parse_progress(line: str) -> int | None:
    """The integer from a ``PROGRESS <n>`` line, or None for any other line. ``.strip()`` tolerates a
    trailing ``\\r`` from Windows CRLF streams; a non-integer payload (``PROGRESS x``) yields None."""
    if not line.startswith("PROGRESS "):
        return None
    try:
        return int(line[len("PROGRESS "):].strip())
    except ValueError:
        return None


def _parse_stage(line: str) -> str | None:
    """The name from a ``STAGE <name>`` line (used to relabel the dialog, see :data:`_STAGE_LABELS`),
    or None for any other line. ``.strip()`` tolerates a trailing ``\\r`` from Windows CRLF streams."""
    if not line.startswith("STAGE "):
        return None
    return line[len("STAGE "):].strip() or None


class GpuSupportManager(QObject):
    """See the module docstring. `parent` is the QObject parent (lifetime) and the parent of
    the offer/progress/result dialogs. At most one install runs at a time."""

    # Emitted after the install set changes (successful install or uninstall) so the caller
    # can refresh any status it shows.
    changed = pyqtSignal()
    # Emitted when the user opts out of the first-run offer ("Don't ask again"), so the caller
    # can stop making it. A plain dismissal of that offer does NOT emit this: it re-offers.
    declined = pyqtSignal()
    # Emitted when the visible first-run check (check_then_offer) concludes no NVIDIA GPU
    # exists, so the caller can stop running the check on later launches. A cancelled or
    # failed check emits nothing: the check simply runs again next launch.
    no_gpu_found = pyqtSignal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._parent_widget = parent
        self._install: _Install | None = None
        self._check_dialog: BusyDialog | None = None  # the "Checking for GPU…" dialog
        self._probes: list[QProcess] = []   # in-flight compute probes, so close can cancel them

    def _mark_changed(self) -> None:
        """Record that the install set changed: drop the shared compute-status cache, then notify
        listeners via ``changed``. Invalidating the cache belongs here: the one place installs and
        uninstalls complete: not only in the GpuStatusPanel slot that reacts to ``changed``: the
        Debug 'Force download' and first-run offers run on a manager with no panel attached, so
        without this their result would leave a stale reading behind and the next Settings panel
        would show it (its showEvent re-probes only when the cache is empty)."""
        _invalidate_probe_cache()
        self.changed.emit()

    # ── Offer / install ──────────────────────────────────────────────────────────────
    def offer_install(self, *, force: bool = False, assume_gpu: bool = False,
                      first_run: bool = False) -> None:
        """Confirm and start the CUDA download. ``force`` skips the NVIDIA-GPU check (the
        Debug-menu test hook on a GPU-less machine); the platform guard still applies.
        ``assume_gpu`` also skips that check, for a caller that already confirmed a GPU
        off-thread. ``first_run`` is the automatic first-launch offer: it also assumes the GPU
        (the probe just confirmed it) and adds a "Don't ask again" checkbox: checking it emits
        ``declined`` so the caller can stop offering, while a plain dismissal re-offers later."""
        if _install_in_flight():
            # An install is already running (possibly started by another manager: e.g. the
            # first-run probe returning after the user began a panel install). Offering now
            # would let a second installer rmtree the first's staging; the visible progress
            # dialog is the feedback, so decline silently.
            return
        if not gpu_support.is_supported_platform():
            QMessageBox.information(
                self._parent_widget, "GPU acceleration unavailable",
                "Downloadable GPU acceleration is only available on Windows. "
                "(On macOS the Apple GPU is already used automatically.)")
            return
        if not is_frozen():
            # Source checkout: the downloaded site would never be used: activate_cuda_site()
            # runs only from the frozen entry point: so refuse the multi-gigabyte download and
            # point at the venv route instead. Applies to every caller, including the Debug
            # force-download hook (maintainers test the real download in the frozen app).
            QMessageBox.information(
                self._parent_widget, "GPU acceleration (running from source)",
                "You're running from a source checkout, so the in-app GPU download doesn't "
                "apply here: install the CUDA build of PyTorch into this Python environment "
                "instead:\n\n"
                f"pip install torch=={gpu_support.TORCH_VERSION} "
                f"torchvision=={gpu_support.TORCHVISION_VERSION} "
                f"--index-url https://download.pytorch.org/whl/{gpu_support.CUDA_TAG}")
            return
        if not force and not assume_gpu and not first_run and not gpu_support.has_nvidia_gpu():
            QMessageBox.information(self._parent_widget, _NO_GPU_TITLE, _NO_GPU_TEXT)
            return

        already = gpu_support.active_site_dir() is not None
        download_gb = gpu_support.DOWNLOAD_BYTES / 1e9
        verb = "Reinstall" if already else "Install"
        box = QMessageBox(self._parent_widget)
        box.setWindowTitle(f"{verb} GPU acceleration?")
        box.setIcon(QMessageBox.Icon.Question)
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(
            f"Right now analysis runs on your CPU (this is very slow). This downloads graphics "
            f"card support so analysis runs on your NVIDIA GPU instead. This is about "
            f"{download_gb:.1f} GB to download and ~5 GB once installed. More info on the "
            f'download <a href="{_PYTORCH_INFO_URL}">here</a>.<br><br>'
            f"This is a one time setup. After, your GPU is used automatically.")
        # QMessageBox won't open the rich-text link on click by default, so enable it on the
        # message label(s). findChildren avoids depending on the internal "qt_msgbox_label" name;
        # it's harmless on the icon label, which has no links (see _PYTORCH_INFO_URL).
        for label in box.findChildren(QLabel):
            label.setOpenExternalLinks(True)
        go = box.addButton(f"{verb} ({download_gb:.1f} GB)", QMessageBox.ButtonRole.AcceptRole)
        # First-run: a "Don't ask again" checkbox lets the user explicitly opt out. Escape/Cancel
        # without it just dismisses (offer returns next launch); checking it emits `declined`.
        dont_ask = QCheckBox("Don't ask again") if first_run else None
        if dont_ask is not None:
            box.setCheckBox(dont_ask)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is go:
            self._start_install(force=force)
        elif dont_ask is not None and dont_ask.isChecked():
            self.declined.emit()

    def _start_install(self, *, force: bool) -> None:
        """Run the installer stage on a QProcess behind a modal, cancelable progress dialog."""
        global _install_owner
        if self._install is not None or _install_in_flight():
            return  # one at a time: process-wide, not just per-manager (shared staging/site)
        _install_owner = self
        argv = ["-m", _INSTALL_MODULE, "install"]
        if force:
            argv.append("--force")

        dialog = QProgressDialog(_STAGE_LABELS["downloading"], "Cancel", 0, 100,
                                 self._parent_widget)
        dialog.setWindowTitle("Installing GPU support")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.setProcessEnvironment(stage_environment())
        self._install = _Install(proc, dialog)
        proc.readyReadStandardOutput.connect(self._on_install_output)
        proc.finished.connect(self._on_install_finished)
        proc.errorOccurred.connect(self._on_install_error)
        dialog.canceled.connect(self._cancel_install)
        proc.start(sys.executable, argv)
        dialog.show()

    def _on_install_output(self) -> None:
        install = self._install
        if install is None:
            return
        chunk = install.proc.readAllStandardOutput().data().decode("utf-8", "ignore")
        install.output += chunk  # keep the full raw text for failure reporting
        # Interpret each complete line; _split_complete_lines buffers the partial tail across reads
        # so a control line split between two chunks isn't dropped (which would stall the bar or lose
        # a stage change). PROGRESS moves the bar; STAGE relabels the dialog (see _STAGE_LABELS).
        lines, install.pending = _split_complete_lines(install.pending, chunk)
        for line in lines:
            if self._install is not install:
                return  # finished arrived inside a nested setValue → the rest is stale
            value = _parse_progress(line)
            if value is not None:
                # Monotonic only: setValue's own processEvents can nest a newer read ahead of
                # this loop's remaining lines (see _Install.highest_progress): never go back.
                if value > install.highest_progress:
                    install.highest_progress = value
                    install.dialog.setValue(value)
                continue
            stage = _parse_stage(line)
            if stage is not None and stage in _STAGE_LABELS:
                install.dialog.setLabelText(_STAGE_LABELS[stage])

    def _cancel_install(self) -> None:
        install = self._install
        if install is not None:
            install.cancelled = True
            install.proc.kill()  # → finished fires; the installer leaves site/ untouched

    def _on_install_error(self, error: "QProcess.ProcessError") -> None:
        """QProcess-level failure. Only FailedToStart needs handling here: no ``finished`` ever
        follows it, which would leave the dialog stuck at 0% and the in-flight guard held for the
        session. Every other error is followed by ``finished`` (Crashed included: that's what
        cancel's kill() raises), so _on_install_finished owns those paths; acting on them here
        would double-report. Mirrors probe()'s dual-signal handling."""
        if error == QProcess.ProcessError.FailedToStart:
            self._on_install_finished(-1, QProcess.ExitStatus.CrashExit)

    def _on_install_finished(self, code: int, status: "QProcess.ExitStatus") -> None:
        global _install_owner
        install = self._install
        if install is None:
            return
        self._install = None
        if _install_owner is self:
            _install_owner = None
        if qobject_alive(install.dialog):
            # QProgressDialog emits canceled() from its closeEvent, so drop the connection
            # first: otherwise this close would re-enter _cancel_install (today that's a
            # no-op only because self._install was cleared above — don't lean on that
            # ordering). Same discipline BusyDialog owns for the indeterminate dialogs;
            # this one is determinate (a real 0-100 bar), so it stays hand-rolled.
            install.dialog.canceled.disconnect(self._cancel_install)
            install.dialog.close()
            install.dialog.deleteLater()
        if qobject_alive(install.proc):
            install.proc.deleteLater()

        if install.cancelled or not qobject_alive(self._parent_widget):
            # User backed out (or the window's gone): nothing installed, no one to tell. The
            # kill() bypassed the installer's own staging cleanup, so reclaim the partial
            # download here; site/ and the manifest were never touched (manifest written last).
            _discard_staging()
            return
        if code == 0 and status == QProcess.ExitStatus.NormalExit:
            self._mark_changed()
            QMessageBox.information(
                self._parent_widget, "GPU acceleration installed",
                "GPU acceleration installed.")
        else:
            _discard_staging()  # a failed installer usually cleans up; cover the paths it can't
            tail = "\n".join(install.output.strip().splitlines()[-8:])
            QMessageBox.warning(
                self._parent_widget, "GPU support install failed",
                "The GPU support download didn't complete. The app will keep using the CPU.\n\n"
                f"{tail or 'The installer exited with an error.'}")

    # ── Uninstall ────────────────────────────────────────────────────────────────────
    def uninstall(self) -> None:
        """Remove the CUDA install after confirmation (a quick, local rmtree). Proceeds for any
        install on disk: including a stale one the app can't activate (wrong CUDA tag / partial),
        which ``active_site_dir`` reports as absent: so its multi-gigabyte footprint is always
        clearable; the uninstaller removes the whole install root regardless of version."""
        if not gpu_support.install_present():
            return
        answer = QMessageBox.question(
            self._parent_widget, "Remove GPU acceleration?",
            "Remove the downloaded CUDA build? Analysis will run on the CPU until you reinstall it.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        proc = QProcess(self)
        proc.setProcessEnvironment(stage_environment())
        fired = False  # finished + errorOccurred can both arrive; run the completion once

        def _done(code: int) -> None:
            nonlocal fired
            if fired or not qobject_alive(proc):  # panel/dialog closed before the (fast) uninstall returned
                return
            fired = True
            proc.deleteLater()
            self._finish_uninstall(code)

        proc.finished.connect(lambda code, _status: _done(code))
        # FailedToStart never gets a finished signal: without this the click would silently do
        # nothing. Other errors do finish, and the once-guard keeps them single-delivery.
        proc.errorOccurred.connect(
            lambda err: _done(1) if err == QProcess.ProcessError.FailedToStart else None)
        proc.start(sys.executable, ["-m", _INSTALL_MODULE, "uninstall"])

    def _finish_uninstall(self, code: int) -> None:
        """The uninstaller exited: re-probe either way: the status line reflects what's really
        left on disk: and surface a non-zero exit honestly instead of implying success. Non-zero
        means the installer couldn't fully remove ``site/`` (a torch DLL is likely still loaded by
        a running analysis); _mark_changed leaves the status showing "installed", which is the
        truth, and the warning says why nothing happened."""
        self._mark_changed()
        if code != 0 and qobject_alive(self._parent_widget):
            QMessageBox.warning(
                self._parent_widget, "Couldn't remove GPU acceleration",
                "Some files are still in use: is an analysis running? "
                "Close it and try Remove again.")

    # ── Probe (compute status) ───────────────────────────────────────────────────────
    def probe(self, callback: Callable[[ComputeInfo | None], None]) -> QProcess:
        """Run ``gpu_support probe`` and hand the parsed compute-status dict to ``callback``
        (or None on failure). Non-modal; used to fill the Settings/menu status line. If the
        requesting panel is closed before the probe finishes, the result is dropped, not
        delivered onto a dead widget (see qobject_alive). A result from before an install/uninstall
        that completed mid-probe is dropped too (see _probe_generation): a slow pre-change
        probe would otherwise overwrite the fresh post-change reading and re-seed the cache.
        Returns the probe's QProcess: a caller may ``kill()`` it to cancel, which delivers a
        single ``callback(None)`` (the once-guard dedupes, and parsing the truncated output
        yields None). cancel_active_probes is stronger: a probe it cancelled delivers
        nothing at all, not even None (see its docstring for why)."""
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.setProcessEnvironment(stage_environment())
        self._probes.append(proc)  # so cancel_active_probes can reach it at close
        fired = False  # finished + errorOccurred can both arrive; deliver to callback once
        generation = _probe_generation  # the install-set state this probe describes

        def _finish(ok: bool) -> None:
            nonlocal fired
            if fired or not qobject_alive(proc):
                return
            fired = True
            if proc not in self._probes:
                # cancel_active_probes deregistered this probe before killing it: deliver
                # nothing, not even None. Its waitForFinished reaps the child synchronously
                # from inside a close/hide handler, and a child that happened to exit just
                # before the kill arrives with a *genuine* result — which would otherwise
                # walk into offer_install and open a modal on a widget being torn down.
                # (The canceller owns deleteLater for deregistered probes.)
                return
            self._probes.remove(proc)
            proc.deleteLater()
            if generation != _probe_generation:
                return  # superseded: the install set changed while this probe ran
            text = proc.readAllStandardOutput().data().decode("utf-8", "ignore") if ok else ""
            callback(_parse_probe_output(text) if ok else None)

        proc.finished.connect(lambda *_: _finish(True))
        proc.errorOccurred.connect(lambda *_: _finish(False))
        proc.start(sys.executable, ["-m", _INSTALL_MODULE, "probe"])
        return proc

    def cancel_active_probes(self) -> None:
        """Kill any in-flight compute probe (no-op when idle): e.g. so the window doesn't leave a
        torch-importing child running past close, and quitting doesn't block in ~QProcess while it
        kills one for us. A probe is read-only, so a kill needs no cleanup.

        Cancelled probes are deregistered *before* the kill, which _finish reads as "deliver
        nothing, not even None". That matters because waitForFinished emits finished
        synchronously from right here — and when the child happened to exit just before the
        kill, finished arrives with a genuine result, not a truncated one. This method runs
        from close/hide handlers, where delivering a result (worst case: one that opens the
        install-offer modal) would land on a widget mid-teardown.

        Waits briefly for each kill to land. Without that the QProcess can still be in Running
        state when its parent chain is destroyed moments later, which is the warning this exists
        to avoid."""
        cancelled = list(self._probes)
        self._probes.clear()  # before the kills: absence from the list is the cancel signal
        for proc in cancelled:
            if qobject_alive(proc) and proc.state() != QProcess.ProcessState.NotRunning:
                proc.kill()
                proc.waitForFinished(PROBE_KILL_WAIT_MS)
            if qobject_alive(proc):
                proc.deleteLater()  # _finish skips cleanup for deregistered probes

    # ── First-run check + offer ──────────────────────────────────────────────────────
    def check_then_offer(self) -> None:
        """The visible first-run flow: a modal "Checking for GPU…" dialog while the compute
        probe runs, resolving into the install offer (NVIDIA GPU found), a "runs on CPU"
        notice plus the ``no_gpu_found`` signal (none found), or nothing (cancelled or the
        probe failed: both retry next launch, so neither persists anything). Also the Debug
        menu's "Run first-run GPU offer" hook, which is why there is no platform/frozen guard
        here: off Windows or from a source checkout the probe still runs and offer_install
        then shows its own explanatory refusal, which is exactly what a maintainer previewing
        the flow wants to see."""
        if _install_in_flight():
            # Same reasoning as offer_install: the install's progress dialog is already
            # visible feedback, and offering again could start a second installer.
            return
        if self._check_dialog is not None and qobject_alive(self._check_dialog):
            return  # a check is already up (e.g. the Debug hook triggered twice)

        # Constructing the BusyDialog shows it, which is deliberately before the probe starts:
        # a synchronous FailedToStart runs on_result inside probe(), closing and
        # deleteLater()ing the dialog, and a show() after that would flash the dead dialog for
        # one more event-loop turn. (BusyDialog shows explicitly for the same reason the wait
        # is worth showing at all: Qt's own auto-show for an indeterminate dialog is ~4
        # seconds late.)
        dialog = BusyDialog(_GPU_CHECK_TEXT, _GPU_CHECK_TITLE, self._parent_widget)
        self._check_dialog = dialog

        def on_result(info: ComputeInfo | None) -> None:
            self._check_dialog = None
            # A programmatic close, which BusyDialog keeps from reading as a user cancel (that
            # trap would otherwise throw the just-arrived result away).
            dialog.close_quietly()
            if dialog.was_cancelled:
                return  # the user backed out: don't turn the leftover None into a verdict
            if info is None:
                return  # probe failed: leave everything unset so we retry next launch
            if info.get("has_nvidia_gpu"):
                self.offer_install(first_run=True)
            else:
                QMessageBox.information(self._parent_widget, _NO_GPU_TITLE, _NO_GPU_RESULT_TEXT)
                self.no_gpu_found.emit()

        # on_cancel is defined and connected before the probe starts, because QProcess.start
        # can deliver errorOccurred (→ on_result) synchronously on some FailedToStart paths,
        # i.e. before probe() returns. proc is forward-declared so on_cancel's closure resolves
        # even then; None only until probe() returns, which no cancel can precede (no events
        # are processed between the dialog appearing above and probe() returning).
        proc: QProcess | None = None

        def on_cancel() -> None:
            if proc is not None and qobject_alive(proc):
                proc.kill()  # the probe is read-only, so a kill needs no cleanup; it also
                # stops the torch cold-load from burning CPU after the user backed out

        dialog.cancelled.connect(on_cancel)
        proc = self.probe(on_result)


class GpuStatusPanel(QWidget):
    """A reusable compute-status panel: a live "Using: …" line (filled by a probe) plus, on
    Windows, Download/Reinstall + Remove buttons; off Windows, a "built in" hint. Shared
    by the Settings → General group and the File-menu GPU Acceleration dialog, so the readout
    is identical wherever it's shown."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._manager = GpuSupportManager(self)
        self._manager.changed.connect(self._refresh)
        self._probed = False  # probe runs on first show, not construction (see showEvent)
        self._watched_window: QWidget | None = None  # top-level watched for close (see showEvent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._status = QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._install_btn: QPushButton | None = None
        self._remove_btn: QPushButton | None = None
        if gpu_support.is_supported_platform() and is_frozen():
            row = QHBoxLayout()
            self._install_btn = QPushButton()
            self._install_btn.clicked.connect(self._offer_install)
            self._remove_btn = QPushButton("Remove…")
            self._remove_btn.clicked.connect(self._manager.uninstall)
            row.addWidget(self._install_btn)
            row.addWidget(self._remove_btn)
            row.addStretch(1)
            layout.addLayout(row)
        elif gpu_support.is_supported_platform():
            # Windows source checkout: the downloadable install never activates outside the
            # frozen app (offer_install refuses it for the same reason), so offer no buttons:
            # just the venv route. The status line above still reports the live device.
            hint = QLabel("Running from source; to use your NVIDIA GPU, install the CUDA "
                          "build of PyTorch into this Python environment.")
            hint.setWordWrap(True)
            hint.setEnabled(False)  # muted, matching the Settings hint rows
            layout.addWidget(hint)
        else:
            hint = QLabel("GPU acceleration is built in on macOS (Metal). Nothing to install.")
            hint.setWordWrap(True)
            hint.setEnabled(False)  # muted, matching the Settings hint rows
            layout.addWidget(hint)

        self._update_buttons()
        # Show a cached device label right away if we already probed this session; otherwise a
        # placeholder until the first show triggers a probe. Construction never spawns a subprocess.
        self._status.setText(f"Using: {_probe_cache['label']}" if _probe_cache
                             else "Checking compute…")

    def showEvent(self, event: QShowEvent | None) -> None:
        """Probe for the active device the first time the panel is actually shown (not at
        construction; so merely building the Settings dialog never spawns a subprocess).

        Also (re)arms the window watch: the panel wants to cancel its probes when its
        *window* closes, but its own hideEvent can't be trusted to see that — Qt only sends
        Hide on a visible→hidden transition, so a panel already hidden by a tab switch gets
        no hideEvent at all when the dialog later closes. Watching the top-level window
        itself (see eventFilter) catches the close whichever tab is current."""
        super().showEvent(event)
        window = self.window()
        if window is not self._watched_window:
            if self._watched_window is not None and qobject_alive(self._watched_window):
                self._watched_window.removeEventFilter(self)
            if window is not None:
                window.installEventFilter(self)
            self._watched_window = window
        if not self._probed:
            self._probed = True
            if _probe_cache is None:
                self._status.setText("Checking compute…")
                self._manager.probe(self._on_probe)

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        """Cancel in-flight probes when the watched top-level window hides or closes; the
        one signal that means "the dialog went away" regardless of which tab was current
        (pinned by test_dialog_close_from_another_tab_still_cancels)."""
        if (obj is self._watched_window and event is not None
                and event.type() in (QEvent.Type.Hide, QEvent.Type.Close)):
            self._cancel_and_rearm()
        return super().eventFilter(obj, event)

    def hideEvent(self, event: QHideEvent | None) -> None:
        """Cancel an in-flight probe when the panel's *window* goes away (its dialog closed),
        so a torch-importing child doesn't outlive the widget that wanted its answer.

        A plain hide is not evidence of that: switching Settings tabs hides the page (and
        this panel with it) while the dialog stays open, and cancelling there would silently
        discard both the status answer and, worse, a Download click waiting in
        _offer_after_probe. On every real close path (close/accept/reject) the window is
        already invisible by the time the child hide event arrives, so the window check
        cleanly separates "tab switched" from "dialog closed" (pinned by
        test_tab_switch_keeps_probe_and_pending_offer_alive). The eventFilter above covers
        the close this handler never sees: a dialog closed while the panel's tab was
        already hidden delivers no hideEvent here at all."""
        super().hideEvent(event)
        window = self.window()
        if window is None or not window.isVisible():
            self._cancel_and_rearm()

    def _cancel_and_rearm(self) -> None:
        """Kill this panel's in-flight probes and re-arm the show-time probe. ``_probed``
        is reset with the cancel: it exists to make the probe run once per panel, and a
        probe that was cancelled rather than answered has to be allowed to run again, or a
        panel shown a second time would sit on "Checking compute…" forever. Idempotent, so
        the window filter and hideEvent both firing on one close is harmless."""
        self._manager.cancel_active_probes()
        self._probed = False

    def _refresh(self) -> None:
        """React to an install/uninstall (the manager already dropped the shared probe cache in
        _mark_changed): update the buttons and re-probe now, since the device may have changed and
        the panel is visible; the user just clicked a button."""
        self._update_buttons()
        self._status.setText("Checking compute…")
        self._manager.probe(self._on_probe)

    def _offer_install(self) -> None:
        """The Download/Reinstall button → the install offer. The offer's own NVIDIA check runs
        nvidia-smi synchronously (up to 5 s); never on the GUI thread: answer from this session's
        cached probe when it already knows, and otherwise probe first (off-thread, like the
        first-run flow in app.py) and offer when the result lands."""
        if _probe_cache is not None:
            if _probe_cache.get("has_nvidia_gpu"):
                self._manager.offer_install(assume_gpu=True)
            else:
                QMessageBox.information(self, _NO_GPU_TITLE, _NO_GPU_TEXT)
            return
        self._status.setText("Checking compute…")
        self._manager.probe(self._offer_after_probe)

    def _offer_after_probe(self, info: ComputeInfo | None) -> None:
        """Cold-cache path of _offer_install: the probe returned; refresh the status line as
        usual, then make (or refuse) the offer off its verdict. A failed probe (None) makes no
        offer; _on_probe already shows the couldn't-determine line."""
        self._on_probe(info)
        if info is None:
            return
        if info.get("has_nvidia_gpu"):
            self._manager.offer_install(assume_gpu=True)
        else:
            QMessageBox.information(self, _NO_GPU_TITLE, _NO_GPU_TEXT)

    def _update_buttons(self) -> None:
        active = gpu_support.active_site_dir() is not None
        if self._install_btn is not None:
            # "Reinstall" only when the *current* build is active; a stale/partial install still
            # reads as "Download" (installing it replaces the stale one via the atomic swap).
            self._install_btn.setText(
                "Reinstall GPU acceleration…" if active else "Download GPU acceleration…")
        if self._remove_btn is not None:
            # Enable Remove whenever *any* install is on disk; including a stale one we can't
            # activate (wrong CUDA tag / partial), which active_site_dir() reports as absent; so
            # its multi-gigabyte footprint can always be cleared, not only an active build's.
            self._remove_btn.setEnabled(gpu_support.install_present())

    def _on_probe(self, info: ComputeInfo | None) -> None:
        global _probe_cache
        label = info.get("label") if info else None
        if info is not None and label:
            _probe_cache = info
            self._status.setText(f"Using: {label}")
        else:
            self._status.setText("Using: (could not determine the compute device)")


class GpuSupportDialog(QDialog):
    """Standalone GPU-acceleration window opened from the File menu: just the shared
    :class:`GpuStatusPanel` with a Close button."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("GPU Acceleration")
        layout = QVBoxLayout(self)
        layout.addWidget(GpuStatusPanel(self))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


def _parse_probe_output(text: str) -> ComputeInfo | None:
    """The last JSON object line from a probe run, or None. The probe prints its JSON last,
    after any library import chatter, so scan from the end for the first line that parses."""
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None
