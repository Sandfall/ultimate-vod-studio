"""The Analyze page's preview + progress pane (app.py).

Shows the chosen video's first frame, swapped for the pipeline's live annotated frames
plus a progress bar and ETA while a run is active. The pipeline subprocess writes both
files; frames to ``--preview-file``, ``{"timestamp_sec", "duration_sec"}`` JSON to
``--progress-file``; and two QTimers poll them into the UI (no OpenCV popup window).

Import this only after ``qt_plugins.configure_qt_plugins()`` has run (every app entry
module does so at its top), since it imports PyQt6 widgets at load time.
"""

import json
import tempfile
import time
from pathlib import Path

from PyQt6.QtCore import Qt, QSettings, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QGroupBox, QLabel, QProgressBar, QVBoxLayout, QWidget

from vodparse.gui.settings import SETTING_SHOW_ETA

# How often the preview frame and the progress file are polled while a run is active.
PREVIEW_POLL_MS = 150
PROGRESS_POLL_MS = 250

# Shown once the bar tops out: the pipeline still tears down after its final 100% write,
# and the extract stage + the synchronous Review load follow before the screen switches;
# a frozen "100%" through that pause looks hung. Same wording as gpu_support's
# "finishing" stage label.
FINISHING_TEXT = "Finishing up…"

# Default file locations the pipeline subprocess is told to write to (see start_run's callers
# in app.py, which pass these as --preview-file/--progress-file). Overridable in the constructor
# so tests can point them at a tmp_path instead of the shared system tempdir.
DEFAULT_PREVIEW_FILE = Path(tempfile.gettempdir()) / "vodparse_pipeline_preview.jpg"
DEFAULT_PROGRESS_FILE = Path(tempfile.gettempdir()) / "vodparse_pipeline_progress.json"


def _unlink_quietly(path: Path) -> None:
    """Delete `path` if it exists, swallowing PermissionError too: on Windows the pipeline
    subprocess can hold the file open at this exact instant (a sharing-violation race, not a
    real failure), and missing_ok=True alone doesn't cover that case."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def format_remaining(seconds: float) -> str:
    """An ETA in minutes to one decimal, e.g. "1.5 mins", "1.0 min". Minutes (not seconds)
    so the estimate's own second-to-second wobble doesn't make the text twitch."""
    minutes = round(seconds / 60.0, 1)
    return f"{minutes:.1f} min" if minutes == 1.0 else f"{minutes:.1f} mins"


def progress_eta_seconds(anchor_time: float, anchor_frac: float,
                         now: float, frac: float) -> float | None:
    """Seconds remaining projected from the average rate since the first-progress anchor,
    or None when nothing has advanced. Throughput (video-seconds processed per wall-second)
    is roughly steady, so the average rate since the first real-progress sample projects the
    rest. Anchoring to that first sample rather than the run start keeps model-load time:
    which elapses before any progress is written; out of the rate."""
    elapsed, advanced = now - anchor_time, frac - anchor_frac
    if elapsed <= 0.0 or advanced <= 0.0:
        return None
    return elapsed / advanced * (1.0 - frac)


class PreviewProgressPane(QWidget):
    """The titled "Preview" box on the Analyze page: the chosen video's first frame, then the
    live detection frames (plus the progress bar) while the pipeline runs."""

    def __init__(self, settings: QSettings, *,
                 preview_file: Path = DEFAULT_PREVIEW_FILE,
                 progress_file: Path = DEFAULT_PROGRESS_FILE) -> None:
        super().__init__()
        self._settings = settings
        self._preview_file = preview_file
        self._progress_file = progress_file
        self._video_path: str | None = None

        # Embedded pipeline preview: the pipeline subprocess writes annotated frames to this file
        # and a timer pumps them into a label (no OpenCV popup).
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(PREVIEW_POLL_MS)
        self._preview_timer.timeout.connect(self._update_preview)

        # Pipeline progress bar: same file-polling scheme as the preview, shown for the whole run.
        self._progress_anchor: tuple[float, float] | None = None  # set per run by start_run
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(PROGRESS_POLL_MS)
        self._progress_timer.timeout.connect(self._update_progress)

        self._preview_label = QLabel("No video selected")
        self._preview_label.setMinimumSize(480, 270)
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # How far through the video the run has gotten. Visible only while the pipeline
        # runs; both fed by _update_progress polling the pipeline's progress file. The
        # label carries the percent + ETA text (too cramped on the bar itself); the bar
        # uses range 0–1000 so it moves smoothly on hours-long vods.
        self._progress_label = QLabel()
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.hide()
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.hide()

        # Wrapped in a titled box to match the Source/Analysis/Sets groupings on the left.
        preview_box = QGroupBox("Preview")
        preview_col = QVBoxLayout(preview_box)
        preview_col.addWidget(self._preview_label, stretch=1)
        preview_col.addWidget(self._progress_label)
        preview_col.addWidget(self._progress_bar)
        # ...then wrapped again in a plain widget so it's the *same kind* of layout item as the
        # left column. A QGroupBox added straight to a box layout overhangs ~15px into the title
        # space the layout reserves at the top; a plain-widget-wrapped one doesn't; so wrapping
        # both keeps their tops on the same line.
        preview_wrap = QVBoxLayout()
        preview_wrap.setContentsMargins(0, 0, 0, 0)
        preview_wrap.addWidget(preview_box)
        self.setLayout(preview_wrap)

    @property
    def preview_file(self) -> Path:
        """The file the pipeline subprocess is handed as --preview-file."""
        return self._preview_file

    @property
    def progress_file(self) -> Path:
        """The file the pipeline subprocess is handed as --progress-file."""
        return self._progress_file

    def _show_eta(self) -> bool:
        """Whether the analysis progress label appends an "~N mins left" estimate to the
        percent (default True)."""
        return self._settings.value(SETTING_SHOW_ETA, True, type=bool)

    def set_video(self, path: str | None) -> None:
        """Remember `path` as the current vod and show its first frame in the preview pane."""
        self._video_path = path
        self._show_first_frame()

    def start_run(self) -> None:
        """Begin polling both files for a fresh pipeline run: drop any stale frame/progress
        left over from a previous run, then start the timers."""
        _unlink_quietly(self._preview_file)   # drop any stale frame
        self._preview_timer.start()

        _unlink_quietly(self._progress_file)   # drop any stale progress
        self._progress_anchor = None   # (monotonic_time, frac) of the first real progress, for the ETA
        self._progress_bar.setValue(0)
        self._progress_label.setText("Starting…")
        self._progress_label.show()
        self._progress_bar.show()
        self._progress_timer.start()

    def stop_run(self) -> None:
        """End a pipeline run: stop polling and restore the pane to its idle look."""
        self._stop_polling()
        self.hide_progress()

    def show_finishing(self) -> None:
        """End the pipeline's polling like stop_run, but keep the bar pinned at 100% under
        FINISHING_TEXT instead of hiding: the extract stage and the synchronous Review load
        still run before the screen switches, and a vanished bar there looks hung.
        hide_progress() ends the hold."""
        self._stop_polling()
        # Pinned explicitly: the finished signal can beat the poll of the final 100% write,
        # and success (the only caller) means the run really did complete.
        self._progress_bar.setValue(self._progress_bar.maximum())
        self._progress_label.setText(FINISHING_TEXT)

    def hide_progress(self) -> None:
        """Hide the progress bar and label: ends a show_finishing hold (no-op when idle)."""
        self._progress_bar.hide()
        self._progress_label.hide()

    def _stop_polling(self) -> None:
        """Stop both file polls and drop their files: shared teardown for stop_run and
        show_finishing. Leaves the bar/label visibility untouched."""
        self._preview_timer.stop()
        _unlink_quietly(self._preview_file)
        self._show_first_frame()  # swap the last live frame back to the video's first frame

        self._update_progress()  # catch the pipeline's final 100% write
        self._progress_timer.stop()
        _unlink_quietly(self._progress_file)

    def _update_preview(self) -> None:
        pix = QPixmap(str(self._preview_file))
        if not pix.isNull():
            self._set_preview_pixmap(pix)

    def _update_progress(self) -> None:
        try:
            data = json.loads(self._progress_file.read_text(encoding="utf-8"))
            frac = data["timestamp_sec"] / data["duration_sec"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ZeroDivisionError):
            return  # not written yet (model loading takes a while); keep the last value
        frac = min(max(frac, 0.0), 1.0)
        self._progress_bar.setValue(round(frac * 1000))
        if frac >= 1.0:
            # The pipeline keeps tearing down (readers, pipeline.json) after its last
            # write; "100% (~0.0 mins left)" through that stretch looks hung.
            self._progress_label.setText(FINISHING_TEXT)
            return
        pct = round(frac * 100)
        eta = self._progress_eta(frac) if self._show_eta() else None
        self._progress_label.setText(f"{pct}% (~{eta} left)" if eta else f"{pct}%")

    def _progress_eta(self, frac: float) -> str | None:
        """Time remaining as an h/m/s string, or None until there's enough to estimate.
        Does the anchor bookkeeping (remembering the first real-progress sample) and
        delegates the actual rate math to the pure progress_eta_seconds."""
        now = time.monotonic()
        if self._progress_anchor is None:
            if frac > 0.0:
                self._progress_anchor = (now, frac)
            return None
        anchor_time, anchor_frac = self._progress_anchor
        seconds = progress_eta_seconds(anchor_time, anchor_frac, now, frac)
        return format_remaining(seconds) if seconds is not None else None

    def _set_preview_pixmap(self, pix: QPixmap) -> None:
        self._preview_label.setPixmap(pix.scaled(
            self._preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _show_first_frame(self) -> None:
        """Display the chosen video's first frame in the preview pane."""
        if not self._video_path:
            return
        import cv2  # local: keeps GUI startup fast (only the subprocesses need cv2 otherwise)
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            cap.release()
            self._preview_label.setText("Could not read video")
            return
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            self._preview_label.setText("Could not read video")
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        # frame.tobytes() (not frame.data) so the buffer is the bytes the QImage overloads
        # accept; the contiguous RGB frame matches the 3*w bytes-per-line.
        img = QImage(frame.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        self._set_preview_pixmap(QPixmap.fromImage(img))
