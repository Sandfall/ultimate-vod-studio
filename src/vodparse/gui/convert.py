"""Video → MP4 conversion offers for the desktop GUI (app.py).

:class:`Mp4Converter` serves two selection-time rescues. When the chosen VOD sits in a
container the Review player can't show (see ``UNPLAYABLE_CONTAINER_EXTS``),
``offer_convert`` offers to convert it to an MP4 saved next to the source (reusing one
that's already there): a stream copy first, with one H.264/AAC re-encode retry. When a
playable-container pick fails the analyzer's decode probe (``opencv_can_read``, e.g.
HEVC/10-bit), ``offer_reencode`` offers the re-encode directly. Both run ffmpeg on a
``QProcess`` behind a modal cancelable progress dialog and emit ``fileReady(path)``
with whichever file the user should adopt. ffmpeg writes a temporary
``.converting.mp4`` sibling that is renamed into place only on success, so cancel,
failure, or dismissing the offer emits nothing, removes only that temp file, and never
disturbs an MP4 that was already there.

Import this only after ``qt_plugins.configure_qt_plugins()`` has run (every app entry
module does so at its top), since it imports PyQt6 widgets at load time.
"""

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import Qt, QObject, QProcess, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QProgressDialog, QWidget

from vodparse.ffmpeg_runtime import parse_progress_bytes, remux_to_mp4_argv


def _file_bytes(path: str) -> int:
    """File size in bytes, or 0 if it can't be read: keeps size warnings/progress math crash-free."""
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _file_size_gb(path: str) -> float:
    """File size in GB (decimal, matching how drives are labelled) for the disk-space warning."""
    return _file_bytes(path) / 1e9


def opencv_can_read(path: str) -> bool:
    """True if OpenCV can actually decode a frame from `path`. The analysis pipeline reads frames
    via ``cv2.VideoCapture`` (see frame_reader.py), so a file that macOS/Qt play but OpenCV
    can't (HEVC/10-bit) is useless for analysis: this is the real 'is it usable' test. Also
    used by app.py as a choose-time pre-flight, so a corrupted pick fails with a dialog
    instead of minutes later inside Analyze VOD."""
    import cv2  # local import: keep GUI startup fast (matches _show_first_frame's pattern)
    cap = cv2.VideoCapture(path)
    try:
        ok, _ = cap.read()
        return bool(ok)
    finally:
        cap.release()


@dataclass
class _Conversion:
    """Transient state of an in-flight MKV/AVI → MP4 conversion (see Mp4Converter._start_conversion)."""
    proc: QProcess
    dialog: QProgressDialog
    source: str
    target: str        # the final .mp4 the caller adopts
    temp_target: str   # the .converting.mp4 ffmpeg writes; renamed over `target` on success
    source_bytes: int
    cancelled: bool = False
    tried_reencode: bool = False


class Mp4Converter(QObject):
    """See the module docstring. `parent` is both the QObject parent (lifetime) and the
    parent of the offer/progress/failure dialogs. At most one conversion is in flight at
    a time: the GUI's file picker is modal, so a second offer can't start mid-run."""

    # The file the caller should adopt: the converted MP4, an MP4 already sitting next to
    # the source, or the source itself when the user picks "Use as-is".
    fileReady = pyqtSignal(str)

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._parent_widget = parent
        self._conversion: _Conversion | None = None  # in-flight MKV/AVI → MP4 remux, if any

    def offer_convert(self, source: str) -> None:
        """A container the Review player can't show was chosen: offer to convert it to an MP4 saved
        next to it (reusing one that's already there). Every branch ends by either adopting a file
        via fileReady or aborting the selection: conversion continues in _start_conversion."""
        target = Path(source).with_suffix(".mp4")
        if target.exists():
            if not self._offer_existing_mp4(Path(source), target):
                return   # adopted the existing MP4, or the user cancelled
        else:
            size_gb = _file_size_gb(source)
            box = QMessageBox(self._parent_widget)
            box.setWindowTitle("Convert to MP4?")
            box.setIcon(QMessageBox.Icon.Question)
            box.setText(
                f"“{Path(source).name}” can’t be played in the Review screen.\n\n"
                f"Convert it to MP4? The new file is saved next to the original "
                f"(about {size_gb:.1f} GB extra).")
            convert = box.addButton("Convert to MP4", QMessageBox.ButtonRole.AcceptRole)
            use_as_is = box.addButton("Use as-is (no preview)", QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if clicked is use_as_is:
                # Adopting the file unconverted means the analyzer reads it directly, so it
                # must pass the same probe: otherwise Analyze VOD dies minutes later.
                if not opencv_can_read(source):
                    QMessageBox.warning(
                        self._parent_widget, "Can’t use this video as is",
                        f"“{Path(source).name}” can’t be decoded for analysis. Converting it "
                        f"to MP4 may still work.")
                    return
                self.fileReady.emit(source)   # analysis works; the Review preview will be black
                return
            if clicked is not convert:
                return   # Cancel or dialog dismissed
        self._start_conversion(source, str(target))

    def offer_reencode(self, source: str) -> None:
        """A video in a playable container was chosen but the analyzer can't decode it (see
        app.py's opencv_can_read pre-flight): offer to re-encode it to an MP4 saved next to it,
        which fixes the common case (HEVC/10-bit, which macOS plays but OpenCV's build won't)
        and fails honestly, with ffmpeg's own error, on real corruption. The target keeps
        offer_convert's `<stem>.mp4` naming except for a ``.mp4`` source, which gets
        `<stem>.converted.mp4` so the original is never replaced by its own re-encode. No size
        estimate in the offer: a re-encode's output size isn't predictable from the source's,
        unlike offer_convert's stream copy. Every branch ends by either adopting a file via
        fileReady or aborting the selection."""
        target = Path(source).with_suffix(".mp4")
        if target == Path(source):
            target = Path(source).with_name(Path(source).stem + ".converted.mp4")
        if target.exists():
            if not self._offer_existing_mp4(Path(source), target):
                return   # adopted the existing MP4, or the user cancelled
        else:
            box = QMessageBox(self._parent_widget)
            box.setWindowTitle("Video can’t be analyzed")
            box.setIcon(QMessageBox.Icon.Question)
            box.setText(
                f"“{Path(source).name}” can’t be decoded for analysis. The file may be "
                f"corrupted or incomplete, or use a codec the analyzer can’t read (often "
                f"HEVC/10-bit video).\n\n"
                f"Re-encoding it to an MP4 saved next to the original may fix it.")
            convert = box.addButton("Convert to MP4", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(convert)
            box.exec()
            if box.clickedButton() is not convert:
                return   # Cancel or dialog dismissed
        self._start_conversion(source, str(target), reencode=True)

    def _offer_existing_mp4(self, source: Path, target: Path) -> bool:
        """An MP4 already sits next to the chosen source: ask what to do with it. When it's
        usable and the user picks it, adopts it via fileReady. Returns True when the caller
        should convert again (replacing the file), False when the selection is settled
        (existing file adopted, or the offer declined)."""
        if not opencv_can_read(str(target)):
            # The MP4 is there but the analyzer can't decode it (HEVC/10-bit, or an unfinished
            # conversion), so the only useful offer is to redo it.
            box = QMessageBox(self._parent_widget)
            box.setWindowTitle("Existing MP4 can’t be used")
            box.setIcon(QMessageBox.Icon.Question)
            box.setText(
                f"You opened “{source.name}” but the MP4 already next to it can’t be decoded "
                f"for analysis (it may be HEVC/10-bit, or an unfinished conversion).\n\n"
                f"Convert again and replace it?")
            reconvert = box.addButton("Convert again", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(reconvert)
            box.exec()
            return box.clickedButton() is reconvert   # Cancel or dialog dismissed → False

        box = QMessageBox(self._parent_widget)
        box.setWindowTitle("MP4 already exists")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(
            f"You opened “{source.name}” but it looks like it’s been converted to an MP4 "
            f"already.\n\n"
            f"Use that MP4? Or convert again and replace it?")
        use_existing = box.addButton("Use existing MP4", QMessageBox.ButtonRole.AcceptRole)
        reconvert = box.addButton("Convert again", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(use_existing)
        box.exec()
        clicked = box.clickedButton()
        if clicked is use_existing:
            self.fileReady.emit(str(target))
            return False
        return clicked is reconvert   # Cancel or dialog dismissed → False

    def cancel_active(self) -> None:
        """Kill an in-flight conversion (no-op when idle): e.g. so the window doesn't leave an
        ffmpeg running past close. The kill routes through the normal finished handling, which
        removes the partial output."""
        if self._conversion is not None:
            self._cancel_conversion()

    def _start_conversion(self, source: str, target: str, *, reencode: bool = False) -> None:
        """Run ffmpeg (stream-copy, with a re-encode fallback) on a QProcess with a modal,
        cancelable progress dialog. ffmpeg writes a temporary sibling (`video.converting.mp4`:
        it must keep the .mp4 ending, since ffmpeg picks the muxer from the extension) that is
        renamed over `target` only on success, so cancel/failure removes just the temp file and
        leaves any pre-existing MP4 and the selection unchanged. ``reencode=True`` skips the
        stream-copy pass and starts straight in the H.264/AAC re-encode (offer_reencode's path:
        its source is already known to be OpenCV-unreadable, so a copy would just be a wasted
        whole-file pass before the automatic fallback re-encoded anyway), with the same busy-bar
        presentation _restart_as_reencode uses."""
        temp_target = str(Path(target).with_name(Path(target).stem + ".converting.mp4"))
        try:
            argv = remux_to_mp4_argv(source, temp_target, reencode=reencode)
        except RuntimeError as err:   # no working ffmpeg: ensure_ffmpeg's guidance
            QMessageBox.warning(self._parent_widget, "FFmpeg not found", str(err))
            return

        verb = "Re-encoding" if reencode else "Converting"
        dialog = QProgressDialog(f"{verb} {Path(source).name} to MP4…", "Cancel", 0, 100,
                                 self._parent_widget)
        dialog.setWindowTitle("Re-encoding to MP4" if reencode else "Converting to MP4")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        if reencode:
            dialog.setRange(0, 0)   # bytes-written isn't a fair ruler when re-encoding → show busy

        proc = QProcess(self)
        self._conversion = _Conversion(proc, dialog, source, target, temp_target,
                                       source_bytes=_file_bytes(source),
                                       tried_reencode=reencode)
        proc.readyReadStandardOutput.connect(self._on_conversion_progress)
        proc.finished.connect(self._on_conversion_finished)
        dialog.canceled.connect(self._cancel_conversion)
        proc.start(argv[0], argv[1:])
        dialog.show()

    def _on_conversion_progress(self) -> None:
        conv = self._conversion
        if conv is None or conv.tried_reencode:
            return   # re-encode shows a busy bar (see below); don't fight it with byte progress
        chunk = conv.proc.readAllStandardOutput().data().decode("utf-8", "ignore")
        written = parse_progress_bytes(chunk)
        if written is not None and conv.source_bytes > 0:
            conv.dialog.setValue(min(100, round(100 * written / conv.source_bytes)))

    def _cancel_conversion(self) -> None:
        conv = self._conversion
        if conv is not None:
            conv.cancelled = True
            conv.proc.kill()   # → finished fires with a non-clean exit; handled below

    def _on_conversion_finished(self, code: int, status: "QProcess.ExitStatus") -> None:
        conv = self._conversion
        if conv is None:
            return
        if code == 0 and status == QProcess.ExitStatus.NormalExit:
            # ffmpeg is happy: but a stream copy can leave a codec macOS plays yet OpenCV (the
            # analyzer) can't decode (HEVC/10-bit). The real success test is "OpenCV can read it";
            # if not, re-encode to baseline H.264 before adopting.
            if conv.tried_reencode or opencv_can_read(conv.temp_target):
                target = conv.target
                source_name = Path(conv.source).name
                try:
                    # Adopt atomically: only a fully-written temp ever lands on the .mp4 name.
                    Path(conv.temp_target).replace(target)
                except OSError as err:
                    self._end_conversion(delete_temp=True)
                    QMessageBox.warning(self._parent_widget, "Conversion failed",
                                        f"Couldn’t move the converted “{source_name}” into "
                                        f"place.\n\n{err}")
                    return
                self._end_conversion(delete_temp=False)   # nothing left at the temp name
                self.fileReady.emit(target)
            else:
                self._restart_as_reencode(conv)
            return
        if conv.cancelled:
            self._end_conversion(delete_temp=True)
            return
        # A stream copy also *fails* outright when the source's codecs can't live in an MP4: same
        # remedy: retry once by re-encoding to H.264/AAC before giving up.
        if not conv.tried_reencode:
            self._restart_as_reencode(conv)
            return
        error = conv.proc.readAllStandardError().data().decode("utf-8", "ignore").strip()
        source_name = Path(conv.source).name
        self._end_conversion(delete_temp=True)
        QMessageBox.warning(self._parent_widget, "Conversion failed",
                            f"Couldn’t convert “{source_name}” to MP4.\n\n"
                            f"{error[:800] or 'ffmpeg exited with an error.'}")

    def _restart_as_reencode(self, conv: _Conversion) -> None:
        """Relaunch the same QProcess as an H.264/AAC re-encode (the fallback when a stream copy
        can't be produced, or produces a file OpenCV can't decode)."""
        conv.tried_reencode = True
        conv.dialog.setLabelText(f"Re-encoding {Path(conv.source).name} to MP4…")
        conv.dialog.setRange(0, 0)   # bytes-written isn't a fair ruler when re-encoding → show busy
        argv = remux_to_mp4_argv(conv.source, conv.temp_target, reencode=True)
        conv.proc.start(argv[0], argv[1:])

    def _end_conversion(self, *, delete_temp: bool) -> None:
        """Close the dialog, optionally bin the temporary output, and clear the in-flight state."""
        conv = self._conversion
        if conv is None:
            return
        self._conversion = None   # clear first so a close()-triggered cancel signal is a no-op
        conv.dialog.close()
        if delete_temp:
            Path(conv.temp_target).unlink(missing_ok=True)
        conv.proc.deleteLater()
        conv.dialog.deleteLater()
