"""Video → MP4 conversion offers for the desktop GUI (app.py).

:class:`Mp4Converter` serves two selection-time rescues and one on request. When the chosen VOD sits in a
container the Review player can't show (AVI/WebM, and MKV except on Windows: see
``UNPLAYABLE_CONTAINER_EXTS``), or is an MKV on Windows with the offer turned on in Settings
(``vodparse.ffmpeg_runtime.offers_conversion``),
``offer_convert`` offers to convert it to an MP4 saved next to the source (reusing one
that's already there): a stream copy first, with one H.264/AAC re-encode retry. When the
pick fails the analyzer's decode probe (``opencv_can_read``, e.g. HEVC/10-bit, or AV1 on an
OpenCV build that can't decode it), ``offer_reencode`` offers the re-encode directly: a
stream copy would keep the same undecodable video, so it would be a wasted whole-file pass.
A third entry, ``convert_on_request``, serves File → "Convert to MP4…": the user asks
for the conversion themselves, for a video that loads but won't show, which the app has no
way to notice. It follows the same order (stream copy first, re-encode if that fails).
All three run ffmpeg on a ``QProcess`` behind a modal cancelable progress dialog and emit
``fileReady(path)`` with whichever file the user should adopt. ffmpeg writes a temporary
``.converting.mp4`` sibling that is renamed into place only on success, so cancel,
failure, or dismissing the offer emits nothing and removes only that temp file.

No conversion ever replaces a file. The MP4 is saved under the first free name next to
the source (see ``_mp4_slot_for``): an MP4 already there is offered for use when the
analyzer can read it, and left alone, with the new file saved beside it, when it can't.

Import this only after ``qt_plugins.configure_qt_plugins()`` has run (every app entry
module does so at its top), since it imports PyQt6 widgets at load time.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from PyQt6.QtCore import Qt, QObject, QProcess, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QProgressDialog, QWidget

from vodparse.ffmpeg_runtime import (
    AAC_AUDIO_CONTAINER_EXTS, UNPLAYABLE_CONTAINER_EXTS, parse_progress_bytes,
    remux_to_mp4_argv)
from vodparse.paths import CONVERTED_STEM_SUFFIX, strip_converted_suffix


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


def _mp4_target_for(source: Path) -> Path:
    """The first name tried for the MP4 made from ``source``: next to it as ``<stem>.mp4``,
    or as ``<stem>.converted.mp4`` (CONVERTED_STEM_SUFFIX) when ``source`` is itself an
    ``.mp4``, so an original is never replaced by its own conversion; vodparse.paths.vod_stem
    maps that name back to the original's artifact folders. The extension is matched whatever
    its case: on a disk that ignores case (the default on macOS and Windows) ``video.MP4``
    and ``video.mp4`` are the same file. When the name is taken, _mp4_slot_for moves on to
    _name_after."""
    if source.suffix.lower() == ".mp4":
        return _name_after(source)
    return source.with_suffix(".mp4")


def _name_after(taken: Path) -> Path:
    """The MP4 name tried when ``taken`` can't be used: one more ".converted" before the
    extension (``video.mp4`` → ``video.converted.mp4`` → ``video.converted.converted.mp4``).
    vodparse.paths.vod_stem drops every trailing ".converted", so all of these names share
    one set of artifact folders."""
    return taken.with_name(taken.stem + CONVERTED_STEM_SUFFIX + ".mp4")


def _first_free_name(wanted: Path) -> Path:
    """``wanted`` when no file has that name, otherwise the first free name after it (see
    _name_after)."""
    name = wanted
    while name.exists():
        name = _name_after(name)
    return name


class _Mp4Slot(NamedTuple):
    """Where the MP4 for a source belongs (see _mp4_slot_for)."""
    path: Path                # a free name, or an MP4 already there that the analyzer reads
    holds_mp4: bool           # True when ``path`` is that existing MP4
    unreadable: Path | None   # the first existing MP4 passed over, if any


def _mp4_slot_for(source: Path) -> _Mp4Slot:
    """Find where the MP4 for ``source`` belongs, without ever choosing a name whose file
    would have to be replaced. Starting at _mp4_target_for and moving on with _name_after,
    an existing MP4 the analyzer can't decode (see opencv_can_read) is passed over and left
    as it is. The search ends at the first name that is free (a conversion is saved there)
    or that holds a decodable MP4 (the offers adopt that one)."""
    path = _mp4_target_for(source)
    unreadable: Path | None = None
    while path.exists() and not opencv_can_read(str(path)):
        if unreadable is None:
            unreadable = path
        path = _name_after(path)
    return _Mp4Slot(path, holds_mp4=path.exists(), unreadable=unreadable)


def is_mp4_conversion_of(candidate: str, source: str) -> bool:
    """True when ``candidate`` is a name the MP4 made from ``source`` is saved under: the
    first one (_mp4_target_for) or any that follows it (_name_after). Every conversion
    offer adopts such a file for ``source``, whether converted now or already there."""
    first = _mp4_target_for(Path(source))
    candidate_path = Path(candidate)
    if candidate_path.parent != first.parent or candidate_path.suffix != first.suffix:
        return False
    if not candidate_path.stem.startswith(first.stem):
        return False
    extra = candidate_path.stem.removeprefix(first.stem)   # "", ".converted", ...
    return strip_converted_suffix(extra) == ""


@dataclass
class _Conversion:
    """Transient state of an in-flight MKV/AVI/WebM → MP4 conversion (see Mp4Converter._start_conversion)."""
    proc: QProcess
    dialog: QProgressDialog
    source: str
    target: str        # the final .mp4 the caller adopts
    temp_target: str   # the .converting.mp4 ffmpeg writes; renamed to `target` on success
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
        self._conversion: _Conversion | None = None  # in-flight → MP4 conversion, if any

    def offer_convert(self, source: str) -> None:
        """A container the Review player can't show was chosen (or an MKV on Windows, with
        the offer turned on in Settings): offer to convert it to an MP4 saved
        next to it (reusing one that's already there). The source is probed first: when the
        analyzer can't decode it (see opencv_can_read), a stream copy would keep that same video,
        so this hands over to offer_reencode, which goes straight to one re-encode. Every branch
        ends by either adopting a file via fileReady or aborting the selection: conversion
        continues in _start_conversion."""
        if not opencv_can_read(source):
            self.offer_reencode(source)
            return
        slot = _mp4_slot_for(Path(source))
        target = slot.path
        if slot.holds_mp4:
            self._offer_existing_mp4(Path(source), target)
            return
        if slot.unreadable is not None:
            if not self._confirm_past_unreadable_mp4(Path(source), slot.unreadable, target):
                return
        else:
            box = QMessageBox(self._parent_widget)
            box.setWindowTitle("Convert to MP4?")
            box.setIcon(QMessageBox.Icon.Question)
            if Path(source).suffix.lower() in UNPLAYABLE_CONTAINER_EXTS:
                size_gb = _file_size_gb(source)
                box.setText(
                    f"“{Path(source).name}” can’t be played in the Review screen.\n\n"
                    f"Convert it to MP4? The new file is saved next to the original "
                    f"(about {size_gb:.1f} GB extra).")
                use_as_is_label = "Use as-is (no preview)"
            else:
                # An MKV on Windows, where the player shows it as it is: the offer is here
                # only because the user turned it on (SETTING_ASK_CONVERT_MKV), so it
                # claims nothing about the Review screen or a missing preview.
                box.setText(f"Convert “{Path(source).name}” to .mp4?")
                use_as_is_label = "Use as-is"
            convert = box.addButton("Convert to MP4", QMessageBox.ButtonRole.AcceptRole)
            use_as_is = box.addButton(use_as_is_label, QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if clicked is use_as_is:
                # The probe above already vetted the source for the analyzer, so analysis
                # works; in an unplayable container the Review preview will be black.
                self.fileReady.emit(source)
                return
            if clicked is not convert:
                return   # Cancel or dialog dismissed
        self._start_conversion(source, str(target))

    def offer_reencode(self, source: str) -> None:
        """A video was chosen that the analyzer can't decode (see app.py's opencv_can_read
        pre-flight, and offer_convert's probe for unplayable containers): offer to re-encode it
        to an MP4 saved next to it, which fixes the common cases (HEVC/10-bit, which macOS plays
        but OpenCV's build won't, and AV1 on OpenCV builds without a software AV1 decoder) and
        fails honestly, with ffmpeg's own error, on real corruption. The target
        keeps offer_convert's `<stem>.mp4` naming except for a ``.mp4`` source, which gets
        `<stem>.converted.mp4` (see _mp4_target_for and _mp4_slot_for). No size
        estimate in the offer: a re-encode's output size isn't predictable from the source's,
        unlike offer_convert's stream copy. Every branch ends by either adopting a file via
        fileReady or aborting the selection."""
        slot = _mp4_slot_for(Path(source))
        target = slot.path
        if slot.holds_mp4:
            self._offer_existing_mp4(Path(source), target)
            return
        if slot.unreadable is not None:
            if not self._confirm_past_unreadable_mp4(Path(source), slot.unreadable, target):
                return
        else:
            box = QMessageBox(self._parent_widget)
            box.setWindowTitle("Video can’t be analyzed")
            box.setIcon(QMessageBox.Icon.Question)
            box.setText(
                f"“{Path(source).name}” can’t be decoded for analysis. The file may be "
                f"corrupted or incomplete, or use a codec the analyzer can’t read (often "
                f"HEVC/10-bit or AV1 video).\n\n"
                f"Re-encoding it to an MP4 saved next to the original may fix it.")
            convert = box.addButton("Convert to MP4", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(convert)
            box.exec()
            if box.clickedButton() is not convert:
                return   # Cancel or dialog dismissed
        self._start_conversion(source, str(target), reencode=True)

    def convert_on_request(self, source: str) -> None:
        """The user asked to convert the chosen video (File → "Convert to MP4…"), for a
        video that loads but won't show: confirm, then convert it to an MP4 saved next to it
        (reusing one that's already there, as the two offers do). Nothing is claimed about
        why: the app can't tell that a picture stays black.

        Fastest first: a stream copy, which _start_conversion follows with one re-encode
        when the copy can't be made or the analyzer can't decode its result. Two sources
        skip the copy, because it could change nothing for them: an ``.mp4`` (a copy is the
        same video in the same container) and a video the analyzer can't decode (a copy
        keeps that video). Does nothing while another conversion is running. Every branch
        ends by either adopting a file via fileReady or leaving the selection as it was."""
        if self._conversion is not None:
            return
        source_path = Path(source)
        slot = _mp4_slot_for(source_path)
        target = slot.path
        reencode = source_path.suffix.lower() == ".mp4" or not opencv_can_read(source)
        if slot.holds_mp4:
            self._offer_existing_mp4(source_path, target)
            return
        if slot.unreadable is not None:
            if not self._confirm_past_unreadable_mp4(source_path, slot.unreadable, target):
                return
        else:
            box = QMessageBox(self._parent_widget)
            box.setWindowTitle("Convert to MP4?")
            box.setIcon(QMessageBox.Icon.Question)
            if reencode:
                # No size estimate: a re-encode's output size isn't predictable (see
                # offer_reencode).
                box.setText(
                    f"Convert “{source_path.name}” to .mp4?\n\n"
                    f"This will re-encode the video, which can take a while. "
                    f"This won't overwrite your original VOD.")
            else:
                box.setText(
                    f"Convert “{source_path.name}” to .mp4?\n\n"
                    f"This won't overwrite your original file (it makes a new one "
                    f"that's about {_file_size_gb(source):.1f} GB).")
            convert = box.addButton("Convert to MP4", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(convert)
            box.exec()
            if box.clickedButton() is not convert:
                return   # Cancel or dialog dismissed
        self._start_conversion(source, str(target), reencode=reencode)

    def _offer_existing_mp4(self, source: Path, existing: Path) -> None:
        """An MP4 the analyzer can read already sits next to the chosen source: offer to use
        it. Adopts it via fileReady when the user accepts; otherwise the selection stays as it
        was. There is no offer to convert again: that would mean replacing ``existing``."""
        box = QMessageBox(self._parent_widget)
        box.setWindowTitle("MP4 already exists")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(
            f"You opened “{source.name}” but it looks like it’s been converted to an MP4 "
            f"already.\n\n"
            f"Use that MP4?")
        use_existing = box.addButton("Use existing MP4", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(use_existing)
        box.exec()
        if box.clickedButton() is use_existing:
            self.fileReady.emit(str(existing))

    def _confirm_past_unreadable_mp4(self, source: Path, unreadable: Path,
                                     target: Path) -> bool:
        """An MP4 already next to the chosen source can't be decoded by the analyzer
        (HEVC/10-bit, or an unfinished conversion), so it can't be offered for use. It is
        kept as it is; ask whether to convert ``source`` to ``target``, the free name beside
        it. Returns True when the user accepts."""
        box = QMessageBox(self._parent_widget)
        box.setWindowTitle("Existing MP4 can’t be used")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(
            f"You opened “{source.name}” but the MP4 already next to it can’t be decoded "
            f"for analysis (it may be HEVC/10-bit, or an unfinished conversion).\n\n"
            f"Convert “{source.name}” to “{target.name}”? “{unreadable.name}” is kept as "
            f"it is.")
        convert = box.addButton("Convert to MP4", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(convert)
        box.exec()
        return box.clickedButton() is convert   # Cancel or dialog dismissed → False

    def cancel_active(self) -> None:
        """Kill an in-flight conversion (no-op when idle): e.g. so the window doesn't leave an
        ffmpeg running past close. The kill routes through the normal finished handling, which
        removes the partial output."""
        if self._conversion is not None:
            self._cancel_conversion()

    def _start_conversion(self, source: str, target: str, *, reencode: bool = False) -> None:
        """Run ffmpeg (stream-copy, with a re-encode fallback; a WebM's audio is converted to AAC
        during the copy, see AAC_AUDIO_CONTAINER_EXTS) on a QProcess with a modal,
        cancelable progress dialog. ffmpeg writes a temporary sibling (`video.converting.mp4`:
        it must keep the .mp4 ending, since ffmpeg picks the muxer from the extension) that is
        renamed to `target` only on success, so cancel/failure removes just the temp file and
        leaves the selection unchanged. `target` is a free name (see _mp4_slot_for).
        ``reencode=True`` skips the
        stream-copy pass and starts straight in the H.264/AAC re-encode (offer_reencode's path:
        its source is already known to be OpenCV-unreadable, so a copy would just be a wasted
        whole-file pass before the automatic fallback re-encoded anyway), with the same busy-bar
        presentation _restart_as_reencode uses."""
        temp_target = str(Path(target).with_name(Path(target).stem + ".converting.mp4"))
        audio_to_aac = Path(source).suffix.lower() in AAC_AUDIO_CONTAINER_EXTS
        try:
            argv = remux_to_mp4_argv(source, temp_target, reencode=reencode,
                                     audio_to_aac=audio_to_aac)
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
        # finished never fires when ffmpeg fails to start; errorOccurred covers it.
        proc.errorOccurred.connect(self._on_conversion_error)
        dialog.canceled.connect(self._cancel_conversion)
        # Shown before start(): errorOccurred can fire synchronously inside start(), and its
        # handler closes the dialog, which a show() after start() would bring back.
        dialog.show()
        proc.start(argv[0], argv[1:])

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
                # The name was free when the conversion started. Should a file have taken
                # it since, the result goes to the next free name: nothing is replaced.
                target = str(_first_free_name(Path(conv.target)))
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
        self._fail_conversion(error)

    def _on_conversion_error(self, error: "QProcess.ProcessError") -> None:
        """QProcess-level failure. Only FailedToStart needs handling here (ffmpeg missing,
        or blocked from running): no ``finished`` ever follows it, which would leave the
        progress dialog up and ``_conversion`` set, so every later convert_on_request would
        return at its in-flight guard. Every other error is followed by ``finished``
        (Crashed included: that's what cancel's kill() raises), so _on_conversion_finished
        owns those paths; acting on them here would report the failure twice. No re-encode
        retry either: it would start the same ffmpeg."""
        conv = self._conversion
        if conv is None or error != QProcess.ProcessError.FailedToStart:
            return
        # A plain sentence first; Qt's own reason under it, for a bug report.
        self._fail_conversion(f"FFmpeg couldn’t be started.\n\n{conv.proc.errorString()}")

    def _fail_conversion(self, error: str) -> None:
        """End the in-flight conversion as failed: bin its temporary output and warn with
        ``error``, ffmpeg's own message (cut to its first 800 characters) or
        _on_conversion_error's text for an ffmpeg that couldn't be started."""
        conv = self._conversion
        if conv is None:
            return
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
