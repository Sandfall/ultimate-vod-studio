"""
Zoomable frame-view widget
---------------------------
A general-purpose zoomable image display widget: `ImagePane` (a `QGraphicsView` that
auto-fits until the user zooms, then follows Preview's trackpad conventions) and the
`ndarray_to_pixmap` conversion it's built on, plus `make_zoom_buttons`, the shared − / + /
Fit buttons wired to a pane, and `make_zoom_row`, those buttons as a row of their own.
Qt-only, no analysis logic: pure display plumbing over whatever BGR frame the caller
hands it.

Originally split out of a development tool for spot-checking overlay reads,
so its video mode's still pane could reuse `ImagePane` without importing the window
module; promoted here so shipped GUI code can reuse it too.

Import this only after ``qt_plugins.configure_qt_plugins()`` has run (every app entry
module does so at its top), since it imports PyQt6 widgets at load time.
"""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QEvent, QRectF, QSize, pyqtSignal
from PyQt6.QtGui import (QColor, QImage, QKeyEvent, QMouseEvent, QNativeGestureEvent,
                         QPainter, QPixmap, QResizeEvent, QTransform, QWheelEvent)
from PyQt6.QtWidgets import (QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
                             QHBoxLayout, QPushButton)

# ── Image-pane constants ────────────────────────────────────────────────────────

IMAGE_MIN_W       = 640    # the annotated-frame pane never shrinks below this
IMAGE_MIN_H       = 360

ZOOM_STEP         = 1.25   # one wheel notch / button press scales the view by this factor
ZOOM_MAX_SCALE    = 20.0   # zoom-in ceiling: far past per-pixel inspection needs
#                            (the zoom-out floor is not a constant: it is the fitted size)
FIT_SNAP_TOLERANCE = 1e-6  # a zoom-out landing within this fraction above the fit scale
#                            counts as reaching it: + then − multiplies the scale by 1.25
#                            and 1/1.25, which float rounding can leave a hair above fit
WHEEL_NOTCH_DEG   = 120.0  # Qt's angleDelta units for one standard mouse-wheel notch;
#                            trackpad scrolls arrive as many small fractions of this, so
#                            scaling the zoom exponent by delta/notch makes them smooth
ZOOM_BUTTON_WIDTH = 32     # the − / + buttons: square-ish, not full-row

DROP_HINT         = "Drop a screenshot here, or use Open Screenshot…"

# Around a shown frame the pane is black, as a QVideoWidget is around the live video: the
# Preview and the layout editor swap this pane in over their player's video, and with the
# palette's own background (a dark grey) the bars around a 16:9 frame changed colour at
# every swap (Mac, 2026-10-25). Before the first frame the pane keeps the palette's
# background, so its hint text stays legible in the light theme as in the dark.
FRAME_SURROUND_COLOR = QColor(0, 0, 0)


def fitted_rect(image: QSize, viewport: QSize) -> QRectF:
    """Where a fitted frame of size ``image`` sits in a ``viewport``, in viewport pixels:
    exactly where Qt draws a live video frame of that size in a video widget of that size,
    so a still swapped in over the video lands on it. Qt's rule, the same on its GPU path
    (``QVideoWindowPrivate::render``) and its software one (``QVideoFrame::paint``): the
    frame size is fitted with ``QSize.scaled``, which truncates to whole pixels, and the
    result is centred exactly, with no rounding, so on an odd margin the frame starts on
    a half pixel. Rounding that margin instead put the still a pixel off the video on
    every other pane height (measured on the Mac, 2026-10-25).
    Pure, so the placement is testable without a widget."""
    fitted = image.scaled(viewport, Qt.AspectRatioMode.KeepAspectRatio)
    return QRectF((viewport.width() - fitted.width()) / 2,
                  (viewport.height() - fitted.height()) / 2,
                  fitted.width(), fitted.height())


# ── Zoomable image pane ──────────────────────────────────────────────────────────

class ImagePane(QGraphicsView):
    """The annotated-frame viewer: auto-fits the image until the user zooms, then it's
    theirs, following Preview's trackpad conventions: two-finger scrolling pans, a
    pinch zooms smoothly (Qt delivers it as a ZoomNativeGesture), Ctrl/Cmd+scroll also zooms,
    click-dragging pans, and a double-click snaps back to fit-to-window. The − / +
    buttons zoom about the view center instead of the cursor. Zooming out stops at the
    fitted size: the image is never shown smaller than Fit. Fitted, the image fills
    the pane edge to edge (no frame border, no margin), exactly as a QVideoWidget of the
    same size fits the live video, so a pane swapped in over a player's video as its
    display page shows the frame at the same size and place. Drops are left unaccepted
    so they bubble up to the window's screenshot drag-and-drop handling.

    Left/right arrow keys are forwarded out as `navigateRequested(step)` instead of the
    QGraphicsView default (arrow-key panning): panning is already covered by dragging
    and two-finger scrolling, and the window uses the arrows to step the playlist."""

    navigateRequested = pyqtSignal(int)   # -1 = previous screenshot, +1 = next

    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item: QGraphicsPixmapItem | None = None
        self._fit_mode = True   # auto-fit until the user zooms; double-click restores it

        self.setMinimumSize(IMAGE_MIN_W, IMAGE_MIN_H)
        self.setFrameShape(QFrame.Shape.NoFrame)   # the viewport is the whole pane (see _fit)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setAcceptDrops(False)   # the window is the drop target, not the view
        # Filtered scaling, like the QVideoWidget this pane is swapped in over: the
        # default fast (nearest-neighbour) sampling drops rows and columns when a frame is
        # shrunk to fit, so overlay text looked jagged next to the smooth live video.
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self._hint = self._scene.addText(DROP_HINT)

    def set_hint_text(self, text: str) -> None:
        """Replace the placeholder text shown before any frame is set. Hosts whose empty
        state isn't a screenshot drop target (the overlay-layout editor) use this; a
        no-op once a frame has been shown, since set_frame removes the hint for good."""
        if self._hint is not None:
            self._hint.setPlainText(text)

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        """Show a BGR frame, replacing any previous one. Fit-to-window is only re-applied
        in fit mode, so re-annotating (gate toggle, re-interpretation) never yanks the
        view away from wherever the user zoomed."""
        pixmap = ndarray_to_pixmap(frame_bgr)
        if self._hint is not None:
            self._scene.removeItem(self._hint)
            self._hint = None
            self.setBackgroundBrush(FRAME_SURROUND_COLOR)   # the first frame: see the constant
        if self._pix_item is None:
            item = self._scene.addPixmap(pixmap)
            if item is None:   # Optional in the Qt stubs; never None in practice
                return
            item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            self._pix_item = item
        else:
            item = self._pix_item
            item.setPixmap(pixmap)
        self._scene.setSceneRect(item.boundingRect())
        if self._fit_mode:
            self._fit()

    def has_frame(self) -> bool:
        return self._pix_item is not None

    def frame_rect(self) -> QRectF:
        """The shown frame's rect in scene coordinates: the pixmap sits at the origin, so
        this is (0, 0, width, height) in source-frame pixels, and ``mapToScene`` of a
        viewport position is the frame pixel under it. Not ``sceneRect()``: while fitted
        the scene rect is the whole viewport (see _fit). Empty before any frame."""
        if self._pix_item is None:
            return QRectF()
        return self._pix_item.boundingRect()

    def zoom(self, factor: float) -> None:
        """Scale the view by `factor`, leaving fit mode. Bounded at both ends: a zoom in
        past ZOOM_MAX_SCALE is ignored, and a zoom out that would land at or below the
        fitted size goes back to fit mode instead (exactly what the Fit button does), so
        the image is never smaller than Fit and the last step out lands on it."""
        if self._pix_item is None:
            return
        target = self.transform().m11() * factor   # m11 is the zoom level (the fit's horizontal scale)
        if target > ZOOM_MAX_SCALE:
            return
        fit_scale = self._fit_scale()
        if fit_scale is None or target <= fit_scale * (1 + FIT_SNAP_TOLERANCE):
            self.reset_fit()
            return
        self._fit_mode = False
        self._show_scrollbars(True)   # zoomed in, the image can outgrow the pane
        # Back to the image's own rect and the view's own centring: the fit set a
        # viewport-sized rect aligned top-left (see _fit); zoomed, the scrollable area
        # should be the image, no more and no less, centred when smaller than the pane.
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSceneRect(self._pix_item.boundingRect())
        # One scale for both axes again (the fit's two differ by a hair, see _fit);
        # setTransform keeps the zoom anchored like scale() would.
        self.setTransform(QTransform.fromScale(target, target))

    def zoom_in(self) -> None:
        """One button-press zoom step, about the view center (the cursor is on the
        button, not the image, so the under-mouse anchor would be meaningless)."""
        self._zoom_about_center(ZOOM_STEP)

    def zoom_out(self) -> None:
        """One button-press zoom step out, stopping at the fitted size (see zoom)."""
        self._zoom_about_center(1 / ZOOM_STEP)

    def _zoom_about_center(self, factor: float) -> None:
        previous_anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.zoom(factor)
        self.setTransformationAnchor(previous_anchor)

    def reset_fit(self) -> None:
        self._fit_mode = True
        self._fit()

    def _fit_viewport_size(self) -> QSize:
        """The viewport's size with no scrollbars showing, which is what a fitted image
        has to fit. Not ``viewport().size()``: while zoomed, scrollbars that take up room
        (Windows, Linux) make the viewport smaller than the one the fit will get."""
        return self.maximumViewportSize()

    def _fit_scale(self) -> float | None:
        """The scale at which the whole image just fits the pane (aspect kept), for the
        pane's current size: the fitted zoom level, and the floor for zooming out. None
        when there is nothing to fit (no frame yet, or an empty one)."""
        if self._pix_item is None:
            return None
        image = self._pix_item.boundingRect()
        if image.isEmpty():
            return None
        available = self._fit_viewport_size()
        return min(available.width() / image.width(), available.height() / image.height())

    def _fit(self) -> None:
        """Scale the image to the largest size that fits the viewport (aspect kept) and
        center it. Done by hand rather than with QGraphicsView.fitInView, which always
        leaves a 2 px margin on every side: that margin made the still a little smaller
        than the live video it replaces, so the picture visibly shrank on each capture."""
        if self._pix_item is None:
            return
        # A fitted image never needs scrolling, but an exact fit can round a pixel past
        # the viewport, which would bring a scrollbar in and shrink the viewport under it.
        self._show_scrollbars(False)
        if self._fit_scale() is None:   # nothing to fit
            return
        viewport = self._fit_viewport_size()
        image = self._pix_item.boundingRect()
        # Placed where Qt places the live video this pane is swapped in over (fitted_rect).
        # Its size is truncated to whole pixels, so the horizontal and vertical scale
        # differ by a fraction of a percent: matching that, rather than one scale for
        # both, keeps all four edges on the video's. Its offset can be a half pixel, which
        # QGraphicsView's scrolling cannot express (it truncates the scene rect's offset
        # to a whole scroll position, which put the still a pixel off the video): so the
        # offset goes into the view transform's translation instead, and the scene rect
        # is the viewport mapped back through that transform, which leaves nothing to
        # scroll. The pixmap stays at the scene origin (frame pixels = scene units), and
        # the view is aligned top-left so it adds no centring of its own. zoom() undoes
        # all of it, going back to a uniform scale and the image's own rect.
        target = fitted_rect(image.size().toSize(), viewport)
        scale_x = target.width() / image.width()
        scale_y = target.height() / image.height()
        self.setTransform(QTransform(scale_x, 0, 0, scale_y, target.left(), target.top()))
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setSceneRect(QRectF(-target.left() / scale_x, -target.top() / scale_y,
                                 viewport.width() / scale_x, viewport.height() / scale_y))

    def _show_scrollbars(self, show: bool) -> None:
        """Scrollbars as needed while zoomed; none at all while fitted (see _fit)."""
        policy = (Qt.ScrollBarPolicy.ScrollBarAsNeeded if show
                  else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(policy)
        self.setVerticalScrollBarPolicy(policy)

    def wheelEvent(self, event: QWheelEvent | None) -> None:   # noqa: N802 (Qt override)
        """Preview's trackpad convention: two-finger scrolling pans, pinching zooms (see
        event()), and holding Cmd (Qt maps it to ControlModifier on macOS) while
        scrolling zooms. A plain mouse wheel keeps Qt's stock scrolling."""
        if event is None:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # One full mouse-wheel notch = one ZOOM_STEP; a trackpad's stream of tiny
            # deltas becomes correspondingly tiny exponents, i.e. smooth zooming.
            self.zoom(ZOOM_STEP ** (event.angleDelta().y() / WHEEL_NOTCH_DEG))
            event.accept()
            return
        pixel_delta = event.pixelDelta()
        if pixel_delta.isNull():
            super().wheelEvent(event)   # mouse wheel: stock vertical scrolling
            return
        # Trackpad scroll (macOS reports it per-pixel): pan both axes, Preview-style.
        horizontal = self.horizontalScrollBar()
        vertical = self.verticalScrollBar()
        if horizontal is not None:
            horizontal.setValue(horizontal.value() - pixel_delta.x())
        if vertical is not None:
            vertical.setValue(vertical.value() - pixel_delta.y())
        event.accept()

    def event(self, event: QEvent | None) -> bool:
        """Intercept macOS trackpad pinches (ZoomNativeGesture) for Preview-style smooth
        zoom; everything else takes the normal QGraphicsView path. `value()` is the
        pinch's relative spread change per event (~±0.0–0.2), so 1+value is the factor."""
        if (isinstance(event, QNativeGestureEvent)
                and event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture):
            self.zoom(1.0 + event.value())
            return True
        return super().event(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent | None) -> None:   # noqa: N802
        self.reset_fit()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:   # noqa: N802 (Qt override)
        if event is not None and event.key() == Qt.Key.Key_Left:
            self.navigateRequested.emit(-1)
        elif event is not None and event.key() == Qt.Key.Key_Right:
            self.navigateRequested.emit(1)
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event: QResizeEvent | None) -> None:   # noqa: N802 (Qt override)
        super().resizeEvent(event)
        if self._fit_mode:
            self._fit()


def make_zoom_buttons(pane: ImagePane) -> list[QPushButton]:
    """The − / + / Fit zoom buttons wired to `pane`, in that order, for a caller to place
    in a row it already has (the layout editor and the Analyze page's preview put them on
    their player's Play row, through ``TrimPlayer.add_transport_widget``). The − / +
    buttons zoom about the view center (the cursor is on the button, not the image), and
    − stops at the fitted size; Fit re-fits the image, which a double-click on the image
    also does."""
    zoom_out_button = QPushButton("−")
    zoom_out_button.setFixedWidth(ZOOM_BUTTON_WIDTH)
    zoom_out_button.setToolTip("Zoom out (or pinch, or Ctrl/Cmd+scroll on the image)")
    zoom_out_button.clicked.connect(pane.zoom_out)
    zoom_in_button = QPushButton("+")
    zoom_in_button.setFixedWidth(ZOOM_BUTTON_WIDTH)
    zoom_in_button.setToolTip("Zoom in (or pinch, or Ctrl/Cmd+scroll on the image)")
    zoom_in_button.clicked.connect(pane.zoom_in)
    fit_button = QPushButton("Fit")
    fit_button.setToolTip("Fit the image to the window (or double-click it)")
    fit_button.clicked.connect(pane.reset_fit)
    return [zoom_out_button, zoom_in_button, fit_button]


def make_zoom_row(pane: ImagePane) -> QHBoxLayout:
    """The zoom buttons (see make_zoom_buttons) as a row of their own, right-aligned behind
    a leading stretch. Shared by the spot-check window's image page and the video mode's
    still pane (spotcheck_video.VideoModePane) so both zoom identically.

    The layout owns only the zoom controls: a caller that also needs playlist-nav widgets
    (◀ ▶ and a position label, the image page's alone) inserts them ahead of the stretch
    with `insertWidget(0, …)`."""
    row = QHBoxLayout()
    row.addStretch(1)
    for button in make_zoom_buttons(pane):
        row.addWidget(button)
    return row


def ndarray_to_pixmap(frame_bgr: np.ndarray) -> QPixmap:
    """BGR ndarray → QPixmap, the exact recipe the GUI preview pane (vodparse.gui.preview) uses: tobytes() (not
    .data) so the buffer matches the QImage overloads, contiguous RGB rows = 3*w stride."""
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    image = QImage(frame.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image)
