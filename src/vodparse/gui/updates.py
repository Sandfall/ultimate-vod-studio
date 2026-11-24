"""Help ▸ Check for Updates for the desktop GUI (app.py).

:class:`UpdateChecker` runs ``vodparse.update_check.check_for_update`` off the UI thread
(a one-shot ``QThread`` doing a single GitHub Releases API GET) behind a modal, cancelable
"Checking for updates…" dialog, then shows the verdict: an "update available" box whose
Download button opens the release page in the browser, an "up to date" note, or a quiet
"couldn't check" notice. Nothing here touches the install; the user downloads and runs the
new build themselves.

Frozen only: on a source checkout there's nothing to update in place, so the action explains
the ``git pull`` route instead of phoning GitHub. Mirrors the ``BusyDialog`` (see
:mod:`vodparse.gui.common`) + background-work pattern of :mod:`vodparse.gui.gpu_support`
(there a ``QProcess`` for a torch cold-load; here a ``QThread``, since the work is one quick
network call).

Import only after ``qt_plugins.configure_qt_plugins()`` has run (every app entry module does so
at its top), since it imports PyQt6 widgets at load time.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QMessageBox, QWidget

from vodparse import PUBLIC_REPO, __version__
from vodparse.gui.common import BusyDialog
from vodparse.runtime_paths import is_frozen
from vodparse.update_check import UpdateCheckResult, UpdateStatus, check_for_update

# Where the Download button (and the failure notice) sends the user when the API response has
# no html_url, or the check couldn't reach the API at all: GitHub's own "latest release" page.
_RELEASES_PAGE = f"https://github.com/{PUBLIC_REPO}/releases/latest"

# How long shutdown() waits for an in-flight GET before giving up. The socket timeout
# (update_check._TIMEOUT_SEC) caps ordinary stalls, but DNS resolution and drip-fed
# responses escape it, so an unbounded wait could freeze the quit; this adds headroom
# over the socket timeout and then abandons the thread (see shutdown).
_SHUTDOWN_WAIT_MS = 15_000


class _CheckThread(QThread):
    """Runs the (blocking) network check once and hands the result back via ``completed``.

    ``check_for_update`` never raises, so ``run`` needs no try/except: every failure already
    arrives as a CHECK_FAILED result.
    """

    completed = pyqtSignal(object)  # UpdateCheckResult

    def __init__(self, current_version: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._current_version = current_version

    def run(self) -> None:
        self.completed.emit(check_for_update(self._current_version))


class UpdateChecker(QObject):
    """Drives the Help ▸ Check for Updates flow. One lives on the main window; ``check_interactive``
    is wired to the menu action. At most one check runs at a time."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._parent_widget = parent
        self._thread: _CheckThread | None = None
        self._dialog: BusyDialog | None = None
        self._cancelled = False

    def check_interactive(self) -> None:
        """The manual Help-menu flow: a modal "Checking…" dialog while the check runs, resolving
        into an update-available / up-to-date / couldn't-check message."""
        if not is_frozen():
            # Source checkout: there's no in-place install to replace, so skip the network call
            # and point at the git route (mirrors gpu_support's from-source refusal).
            QMessageBox.information(
                self._parent_widget, "Check for updates (running from source)",
                "You're running from a source checkout, so there's nothing to update here. "
                "Pull the latest code with `git pull` instead.\n\n"
                "This update check is for the packaged app.")
            return
        if self._thread is not None:
            if self._dialog is not None:
                return  # a check is already in flight; its dialog is the feedback
            # The previous check was cancelled while its GET was still resolving (the GET
            # can't be interrupted, so the thread lingers until it returns). Treat the click
            # as a fresh check that rides the in-flight request: reopen the busy dialog and
            # let the pending result answer it, instead of silently ignoring the click.
            self._cancelled = False
            self._show_busy_dialog()
            return

        self._cancelled = False
        self._show_busy_dialog()
        # Parented to self so the thread outlives this method; deleteLater on finished cleans it
        # up after run() returns (which is just after it emits ``completed``).
        thread = _CheckThread(__version__, self)
        self._thread = thread
        thread.completed.connect(self._on_result)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _show_busy_dialog(self) -> None:
        """Open the modal "Checking for updates…" dialog and remember it in ``_dialog``.
        :class:`BusyDialog` owns the mechanics: the indeterminate bar, the modality, and
        showing itself at once (Qt's own auto-show for a busy dialog wouldn't fire until its
        ~4-second force timer, far too late for a check that usually resolves in under a
        second)."""
        dialog = BusyDialog("Checking for updates…", "Check for Updates", self._parent_widget)
        dialog.cancelled.connect(self._on_cancel)
        self._dialog = dialog

    def dismiss(self) -> None:
        """Quietly drop an in-flight check's UI, as if the user had cancelled: close the busy
        dialog and suppress the verdict. The worker keeps running (the GET can't be
        interrupted); ``shutdown`` still waits it out. The main window calls this before its
        "Quit anyway?" prompt, whose nested event loop would otherwise deliver a resolving
        check's verdict dialog on top of the prompt. Safe to call with no check running."""
        if self._thread is None:
            return
        self._cancelled = True
        self._close_dialog_quietly()

    def shutdown(self) -> None:
        """Called on window teardown (``closeEvent`` and ``force_quit``): let an in-flight
        check finish first, because destroying a still-running ``QThread`` aborts the whole
        app. Normally instant, and the socket timeout (``update_check._TIMEOUT_SEC``) caps
        ordinary stalls — but DNS resolution and drip-fed responses escape that timeout, so
        the wait is bounded by ``_SHUTDOWN_WAIT_MS`` rather than hanging the quit forever.
        Safe to call with no check running."""
        self.dismiss()  # don't leave a frozen busy dialog on screen while we block below
        thread = self._thread
        self._thread = None
        if thread is None:
            return
        # The result would land on a dying window. This also suppresses an emission that is
        # already queued: PyQt6 drops posted deliveries to an undecorated Python slot on
        # disconnect (pinned by test_shutdown_after_result_emitted_stays_quiet). Decorating
        # _on_result with @pyqtSlot would revive the late-delivery race — don't.
        thread.completed.disconnect(self._on_result)
        # On timeout the thread is abandoned rather than freezing the quit; it stays parented
        # to this checker until process exit, by which point the GET has almost always resolved.
        thread.wait(_SHUTDOWN_WAIT_MS)

    def _on_cancel(self) -> None:
        """User backed out. The GET can't be interrupted mid-flight, so it still finishes into
        ``_on_result``, which sees this flag and shows nothing. The closed dialog is the
        feedback; releasing ``_dialog`` here (the BusyDialog has already retired its widget) is
        what lets ``check_interactive`` tell "cancelled but still resolving" (re-click reopens
        a dialog) from "check visibly in flight".

        The flag lives on the checker rather than being read off ``BusyDialog.was_cancelled``
        because one check can span two dialogs: the revival re-click builds a *fresh*
        BusyDialog over the same in-flight GET, and this flag has to go back to False with it.
        The state belongs to the check, not to the dialog."""
        self._cancelled = True
        self._dialog = None

    def _close_dialog_quietly(self) -> None:
        """Close the busy dialog programmatically, without it reading as a user cancel:
        ``BusyDialog.close_quietly`` owns that dance. Safe with no dialog open."""
        dialog = self._dialog
        self._dialog = None
        if dialog is not None:
            dialog.close_quietly()

    def _on_result(self, result: UpdateCheckResult) -> None:
        self._thread = None
        self._close_dialog_quietly()
        if self._cancelled:
            return  # user backed out
        self._present(result)

    def _present(self, result: UpdateCheckResult) -> None:
        if result.status is UpdateStatus.UPDATE_AVAILABLE:
            self._show_update_available(result)
        elif result.status is UpdateStatus.UP_TO_DATE:
            QMessageBox.information(
                self._parent_widget, "You're up to date",
                f"Ultimate VOD Studio {result.latest_version} is the latest version "
                f"(you're running {result.current_version}).")
        else:  # CHECK_FAILED
            QMessageBox.information(
                self._parent_widget, "Couldn't check for updates",
                f"{result.error}\n\nPlease try again later or check the releases page:\n"
                f"{_RELEASES_PAGE}")

    def _show_update_available(self, result: UpdateCheckResult) -> None:
        box = QMessageBox(self._parent_widget)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Update available")
        box.setText(f"Ultimate VOD Studio {result.latest_version} is available.")
        box.setInformativeText(
            f"You're running {result.current_version}. Open the download page to get the new "
            f"version? Your current install won't be changed.")
        download = box.addButton("Download…", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.setDefaultButton(download)
        box.exec()
        if box.clickedButton() is download:
            QDesktopServices.openUrl(QUrl(result.release_url or _RELEASES_PAGE))
