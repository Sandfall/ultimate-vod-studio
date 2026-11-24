"""
Region-drawing frame pane
----------------------------
``RegionDrawPane``: an ``ImagePane`` (see ``vodparse.gui.image_pane``) that also shows a
frame with labelled, coloured region boxes drawn in source-pixel scene coordinates, and
supports rubber-band drawing of a new box when a region is armed. Built with
``adjustable=True``, it also lets the user select a drawn box and move, resize or nudge it
(the geometry is the Qt-free ``vodparse.gui.box_adjust``).

Shared by the custom-overlay-layout editor (``vodparse.gui.overlay_editor``, which draws
and edits the boxes of a named preset, and so builds its pane adjustable) and the main
screen's layout preview pane (which only shows boxes, read-only: its pane is not
adjustable, so a press there always pans).

Boxes are drawn through two entry points that share one private drawing primitive
(``_draw_boxes``):

* ``set_regions`` takes normalized rects (``vodparse.overlay_layout.NormRect``) keyed
  by the shared region keys (``REGION_PLAYER1`` and friends) — the shape the editor
  works in, since presets are stored as fractions of the frame. These are the boxes an
  adjustable pane lets the user select and adjust.
* ``set_boxes`` takes already-converted pixel boxes and their captions directly — for a
  caller (the layout preview) whose keys don't necessarily appear in ``REGION_COLORS``,
  who may not be drawing from a ``NormRect`` at all, and who may want bare boxes with no
  captions. Boxes shown this way are display-only, never selectable.

Import this only after ``qt_plugins.configure_qt_plugins()`` has run, the same
convention as every other GUI module here.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QKeyEvent, QMouseEvent, QPen, QTransform
from PyQt6.QtWidgets import QApplication, QGraphicsItem, QGraphicsRectItem, QGraphicsView

from vodparse import preview_regions
from vodparse.gui.box_adjust import Grab, PixelBox, adjust, hit_test
from vodparse.gui.image_pane import ImagePane
from vodparse.overlay_layout import NormRect
from vodparse.schema import BBox

# ── Region presentation ──────────────────────────────────────────────────────────

# Display labels and box colors, keyed by region: one definition for the editor's rows,
# its drawn boxes and the main-screen preview's, kept Qt-free in vodparse.preview_regions
# (the preview's own keys, the OCR bands and the detector's clock, live there too). The
# editor's per-mode nameplate-row labels and its "timer required" label are dialog-only
# concerns and stay in overlay_editor.py.
REGION_LABELS: dict[str, str] = preview_regions.REGION_LABELS
REGION_COLORS: dict[str, QColor] = {
    key: QColor(*rgb) for key, rgb in preview_regions.REGION_RGB.items()}

# The color a box drawn through set_boxes falls back to when its key isn't one of the
# named regions above (the main-screen layout preview can draw boxes for keys
# REGION_COLORS was never taught about).
DEFAULT_BOX_COLOR = QColor(149, 165, 166)   # neutral gray

# A completed drag narrower/shorter than this many frame pixels is treated as a slipped
# click, not a box: real overlay text regions are always far larger.
MIN_DRAW_PX = 4

# Box outline width (cosmetic: constant on screen at any zoom).
BOX_PEN_WIDTH = 2
# A dotted box (set_boxes' ``dotted`` keys: a drawn region that has read nothing) is a
# sparse row of dots, dash and gap in pen widths: 1 wide, 6 apart. Sparse so it reads as
# fainter than a solid box at a glance, and bare (no interior tint) for the same reason.
DOTTED_PEN_PATTERN = [1.0, 6.0]

# The tint of a drawn box's interior (alpha 0-255): enough to mark the area at a
# glance, light enough that the overlay text under it stays perfectly readable.
BOX_FILL_ALPHA = 40

# Label placement outside the box: the device-pixel gap between a box edge and its
# label, and the frame-pixel margin below which a box is "hugging the frame top", so
# its label flips below the box instead of landing outside the frame.
LABEL_GAP_PX = 2
LABEL_FLIP_AT_PX = 26
# A box whose centre lies past this fraction of the frame width hangs its label from its
# right edge, leftwards, so a label longer than the box (a tag read near the frame's right
# edge, "Player 2 tag: Jett The Jenga God") stays inside the frame instead of running off it.
LABEL_RIGHT_ALIGN_PAST = 0.5

# Adjusting a drawn box (see vodparse.gui.box_adjust). Screen-pixel sizes stay constant
# at any zoom: how close to a corner a press grabs it, the selected box's corner handles
# and outline. Arrow keys nudge the selected box by NUDGE_PX frame pixels, Shift+arrow by
# NUDGE_BIG_PX.
CORNER_GRAB_PX = 8
HANDLE_SIZE_PX = 7
SELECTED_PEN_WIDTH = 4
NUDGE_PX = 1
NUDGE_BIG_PX = 10

# The cursor over each part of a box, so a drag's effect shows before it starts.
GRAB_CURSORS = {
    Grab.MOVE: Qt.CursorShape.SizeAllCursor,
    Grab.TOP_LEFT: Qt.CursorShape.SizeFDiagCursor,
    Grab.BOTTOM_RIGHT: Qt.CursorShape.SizeFDiagCursor,
    Grab.TOP_RIGHT: Qt.CursorShape.SizeBDiagCursor,
    Grab.BOTTOM_LEFT: Qt.CursorShape.SizeBDiagCursor,
}

# Arrow key -> (dx, dy) direction for a nudge.
NUDGE_DIRECTIONS = {
    Qt.Key.Key_Left: (-1, 0),
    Qt.Key.Key_Right: (1, 0),
    Qt.Key.Key_Up: (0, -1),
    Qt.Key.Key_Down: (0, 1),
}


# ── Drawing pane ─────────────────────────────────────────────────────────────────

@dataclass
class _Adjustment:
    """A press on a drawn box, possibly becoming a drag: which box and which part of it,
    where the press landed, and the box as it was (restored if the drag is cancelled).
    ``current`` is the box as the drag has it so far, None until the cursor has moved past
    the drag threshold (a plain click only selects)."""
    key: str
    grab: Grab
    press_view: QPointF
    press_scene: QPointF
    original: PixelBox
    current: PixelBox | None = None


class RegionDrawPane(ImagePane):
    """An ``ImagePane`` that can also *draw* boxes and, when built ``adjustable``, *adjust*
    them.

    Drawing: arming a region swaps the pane's normal pan-drag for a crosshair click-drag
    that rubber-bands out that region's box. A finished drag is normalized to frame
    fractions and emitted as ``regionDrawn``; the pane then disarms itself, returning to
    pan-drag.

    Adjusting (``adjustable=True`` only, while nothing is armed): a press on a box shown
    through ``set_regions`` selects it (thicker outline, corner handles;
    ``regionSelected``). Dragging inside it moves it, dragging a corner resizes it (corners
    only), and arrow keys nudge it (Shift: further); Escape cancels a drag in progress,
    else deselects. The geometry is ``vodparse.gui.box_adjust``. Each finished adjustment
    is emitted as ``regionDrawn`` too, so the owner handles a moved box exactly like a
    redrawn one. A press on empty frame still pans. A pane that isn't adjustable (the
    Analyze page's read-only preview) always pans on an unarmed press.

    Scene coordinates equal source-frame pixels (the pixmap sits at the scene origin,
    see ``ImagePane.frame_rect``), so ``mapToScene`` of a mouse position *is* the frame
    pixel under the cursor — no separate coordinate mapping exists to drift.

    Boxes are shown via ``set_regions``/``set_boxes``: plain scene items over the pixmap,
    so the rubber band and an adjusting box can update every mouse move without
    re-rendering the frame."""

    regionDrawn = pyqtSignal(str, object)   # (region key, NormRect: drawn or adjusted)
    regionSelected = pyqtSignal(object)     # the selected region key, or None

    def __init__(self, *, adjustable: bool = False) -> None:
        super().__init__()
        self._adjustable = adjustable
        self._armed: str | None = None
        self._drag_origin: QPointF | None = None    # scene coords; None = no drag active
        self._rubber: QGraphicsRectItem | None = None
        self._region_items: list[QGraphicsItem] = []
        self._regions: dict[str, NormRect] = {}     # as last set (or adjusted here)
        self._texts: dict[str, str] = {}            # test-read results for the labels
        self._selected: str | None = None
        self._adjusting: _Adjustment | None = None
        viewport = self.viewport()
        if adjustable and viewport is not None:
            viewport.setMouseTracking(True)   # hover cursors over boxes and corners

    def armed_region(self) -> str | None:
        """The region key currently armed for drawing, or None when in pan mode."""
        return self._armed

    def selected_region(self) -> str | None:
        """The region key selected for adjusting, or None (always None unless the pane
        is adjustable)."""
        return self._selected

    def set_armed_region(self, key: str | None) -> None:
        """Arm drawing for one region (crosshair cursor, click-drag draws its box), or
        disarm with None (back to the base pane's pan-drag). Arming while a drag is in
        flight cancels the drag; arming also ends any box adjustment and selection, since
        the crosshair only ever draws."""
        self._armed = key
        self._cancel_drag()
        viewport = self.viewport()
        if key is not None:
            self._cancel_adjustment()
            self._select(None)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            if viewport is not None:
                viewport.setCursor(Qt.CursorShape.CrossCursor)
        else:
            # ScrollHandDrag restores the open-hand cursor on its own.
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def set_regions(self, regions: dict[str, NormRect],
                    texts: dict[str, str] | None = None) -> None:
        """Show ``regions`` as labeled colored boxes over the frame, replacing whatever
        was shown before. Call again after every ``set_frame``: the boxes are scene
        items laid out in that frame's pixels, so a new (possibly differently-sized)
        frame needs them re-placed. ``texts`` appends a test read's result to a box's
        label ('Player 1 tag: SomePlayer'), so what was read shows right on the box.
        A selected box stays selected while its region is still in ``regions``.

        Each region becomes a pixel box and a caption (REGION_LABELS[key], plus the text
        suffix), drawn by the same primitive as ``set_boxes``."""
        self._regions = dict(regions)
        self._texts = dict(texts) if texts else {}
        if self._selected is not None and self._selected not in self._regions:
            self._select(None)
        self._redraw()

    def set_boxes(self, boxes: dict[str, BBox], captions: dict[str, str],
                  dotted: Collection[str] = ()) -> None:
        """Show ``boxes`` (source-pixel ``[x1, y1, x2, y2]`` rects, keyed however the
        caller likes) as labeled colored boxes over the frame, replacing whatever was
        shown before. ``captions`` supplies each box's label text, keyed the same way; a
        key with no caption (or an empty one) gets a bare box, no label. A key in
        ``dotted`` is drawn as a sparse dotted outline with no interior tint (the
        main-screen preview's mark for a drawn region that has read nothing).

        Boxes shown this way are display-only: they replace any regions from
        ``set_regions`` and end a selection, so there is nothing to adjust. A key not
        found in ``REGION_COLORS`` (the main-screen layout preview may pass keys this
        module was never taught about) draws in ``DEFAULT_BOX_COLOR`` instead of raising."""
        self._regions = {}
        self._texts = {}
        self._adjusting = None
        self._select(None)
        self._draw_boxes(boxes, captions, dotted)

    def _redraw(self) -> None:
        """Re-place every region's box and label from ``set_regions`` (an adjusting box at
        its in-progress spot), with the selected box's thicker outline and corner handles."""
        if not self.has_frame():
            self._draw_boxes({}, {})
            return
        frame_w, frame_h = self._frame_size()
        adjusting = self._adjusting
        boxes: dict[str, Sequence[float]] = {}
        captions: dict[str, str] = {}
        for key, rect in self._regions.items():
            if adjusting is not None and adjusting.key == key and adjusting.current is not None:
                box_px = adjusting.current
                boxes[key] = (box_px.x1, box_px.y1, box_px.x2, box_px.y2)
            else:
                boxes[key] = rect.to_pixels(frame_w, frame_h)
            caption = REGION_LABELS[key]
            if self._texts.get(key):
                caption = f"{caption}: {self._texts[key]}"
            captions[key] = caption
        self._draw_boxes(boxes, captions)

    def _draw_boxes(self, boxes: Mapping[str, Sequence[float]],
                    captions: dict[str, str], dotted: Collection[str] = ()) -> None:
        """The drawing primitive behind ``set_regions`` and ``set_boxes``: replace the shown
        boxes with ``boxes`` (frame-pixel ``x1, y1, x2, y2``; fractional while a box is
        being dragged), each labeled with its caption (none: a bare box). The selected box
        gets a thicker outline and corner handles; a key in ``dotted`` gets the sparse
        dotted outline and no tint (see DOTTED_PEN_PATTERN)."""
        scene = self.scene()
        if scene is None:
            return
        for item in self._region_items:
            scene.removeItem(item)
        self._region_items = []
        for key, (x1, y1, x2, y2) in boxes.items():
            color = REGION_COLORS.get(key, DEFAULT_BOX_COLOR)
            fill = QColor(color)
            fill.setAlpha(BOX_FILL_ALPHA)   # tint the interior; the text stays readable
            pen = self._pen_for_color(color)
            if key == self._selected:
                pen.setWidth(SELECTED_PEN_WIDTH)
            if key in dotted:
                pen.setDashPattern(DOTTED_PEN_PATTERN)
                fill = QColor(Qt.GlobalColor.transparent)
            box = scene.addRect(QRectF(QPointF(x1, y1), QPointF(x2, y2)), pen, QBrush(fill))
            if box is None:   # Optional in the Qt stubs only
                continue
            self._region_items.append(box)
            caption = captions.get(key, "")
            if caption:
                self._region_items += self._add_label(caption, color, (x1, y1, x2, y2))
            if key == self._selected:
                self._region_items += self._add_corner_handles(color, (x1, y1, x2, y2))

    def _add_label(self, caption: str, color: QColor,
                   box: tuple[float, float, float, float]) -> list[QGraphicsItem]:
        """A box's caption, placed OUTSIDE the box so it never covers the very text being
        boxed: above the box normally, below it when the box hugs the frame top (above would
        land outside the frame). It starts at the box's left edge, or ends at its right edge
        for a box in the right part of the frame (see LABEL_RIGHT_ALIGN_PAST)."""
        x1, y1, x2, y2 = box
        scene = self.scene()
        label = scene.addSimpleText(caption) if scene is not None else None
        if label is None:   # Optional in the Qt stubs only
            return []
        label.setBrush(color)
        # With ItemIgnoresTransformations the item keeps a constant on-screen size and its
        # own units are device pixels, so the offset off the scene-coordinate anchor goes
        # in the item transform — a constant gap at any zoom.
        label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        right_aligned = (x1 + x2) / 2 > self.frame_rect().width() * LABEL_RIGHT_ALIGN_PAST
        anchor_x = x2 if right_aligned else x1
        offset_x = -label.boundingRect().width() if right_aligned else 0.0
        if y1 >= LABEL_FLIP_AT_PX:
            label.setPos(QPointF(anchor_x, y1))
            label.setTransform(QTransform.fromTranslate(
                offset_x, -label.boundingRect().height() - LABEL_GAP_PX))
        else:
            label.setPos(QPointF(anchor_x, y2))
            label.setTransform(QTransform.fromTranslate(offset_x, LABEL_GAP_PX))
        return [label]

    def _add_corner_handles(self, color: QColor,
                            corners: tuple[float, float, float, float]) -> list[QGraphicsItem]:
        """Small squares on the selected box's corners, a constant size on screen (same
        ItemIgnoresTransformations scheme as the labels): the grab points for resizing."""
        scene = self.scene()
        if scene is None:
            return []
        x1, y1, x2, y2 = corners
        half = HANDLE_SIZE_PX / 2
        outline = QPen(QColor("white"))
        outline.setCosmetic(True)
        handles: list[QGraphicsItem] = []
        for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
            handle = scene.addRect(QRectF(-half, -half, HANDLE_SIZE_PX, HANDLE_SIZE_PX),
                                   outline, QBrush(color))
            if handle is None:   # Optional in the Qt stubs only
                continue
            handle.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            handle.setPos(QPointF(x, y))
            handles.append(handle)
        return handles

    # ── Selection and adjusting ─────────────────────────────────────────────────

    def _select(self, key: str | None) -> None:
        """Select a box for adjusting (None deselects); redraws and tells the owner."""
        if key == self._selected:
            return
        self._selected = key
        self._redraw()
        self.regionSelected.emit(key)

    def _pixel_boxes(self) -> dict[str, PixelBox]:
        """The shown regions in frame pixels, for box_adjust's hit test and geometry."""
        frame_w, frame_h = self._frame_size()
        return {key: PixelBox(*(float(v) for v in rect.to_pixels(frame_w, frame_h)))
                for key, rect in self._regions.items()}

    def _hit(self, event: QMouseEvent) -> tuple[str, Grab] | None:
        """What part of which box is under the cursor. The corner grab zone is fixed in
        screen pixels, so it is converted to frame pixels at the current zoom."""
        zoom = self.transform().m11()   # uniform scaling: m11 is the zoom level
        pos = self.mapToScene(event.position().toPoint())
        return hit_test(self._pixel_boxes(), pos.x(), pos.y(),
                        corner_zone=CORNER_GRAB_PX / zoom)

    def _update_hover_cursor(self, event: QMouseEvent) -> None:
        """Show what a press here would do: a resize arrow over a corner, the move arrows
        over a box, the pan hand over empty frame. Only on an adjustable pane, and not
        while drawing or dragging."""
        viewport = self.viewport()
        if (not self._adjustable or viewport is None or self._armed is not None
                or not self.has_frame() or event.buttons() != Qt.MouseButton.NoButton):
            return
        hit = self._hit(event)
        viewport.setCursor(GRAB_CURSORS[hit[1]] if hit is not None
                           else Qt.CursorShape.OpenHandCursor)

    def _commit_box(self, key: str, box: PixelBox) -> None:
        """Store an adjusted box (rounded to whole frame pixels, like a drawn one) and
        emit it; the owner stores it too and hands it back through set_regions."""
        frame_w, frame_h = self._frame_size()
        pixels = [round(box.x1), round(box.y1), round(box.x2), round(box.y2)]
        norm = NormRect.from_pixels(pixels, frame_w, frame_h)
        self._regions[key] = norm
        self._redraw()
        self.regionDrawn.emit(key, norm)

    def _cancel_adjustment(self) -> None:
        """Drop a drag on a box in progress, putting the box back where it was."""
        if self._adjusting is None:
            return
        self._adjusting = None
        self._redraw()

    def _nudge(self, dx: int, dy: int) -> None:
        """Move the selected box by (dx, dy) frame pixels, stopping at the frame edges."""
        if self._selected is None:
            return
        box = self._pixel_boxes()[self._selected]
        frame_w, frame_h = self._frame_size()
        moved = adjust(box, Grab.MOVE, dx, dy, frame_w, frame_h, MIN_DRAW_PX)
        if moved != box:
            self._commit_box(self._selected, moved)

    # ── Mouse and keys ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent | None) -> None:   # noqa: N802 (Qt override)
        if (event is None or not self.has_frame()
                or event.button() != Qt.MouseButton.LeftButton):
            super().mousePressEvent(event)
            return
        if self._armed is None:
            if self._adjustable:
                self._press_to_adjust(event)
            else:
                super().mousePressEvent(event)   # read-only pane: always the pan-drag
            return
        self._drag_origin = self._clamped_scene_pos(event)
        scene = self.scene()
        if scene is not None:
            pen = self._region_pen(self._armed)
            pen.setStyle(Qt.PenStyle.DashLine)   # dashed while dragging, solid once set
            self._rubber = scene.addRect(QRectF(self._drag_origin, self._drag_origin), pen)
        event.accept()

    def _press_to_adjust(self, event: QMouseEvent) -> None:
        """A press with nothing armed: take hold of the box part under it (selecting that
        box), or deselect and pan on empty frame."""
        hit = self._hit(event)
        if hit is None:
            self._select(None)
            super().mousePressEvent(event)   # empty frame: the base pane's pan-drag
            return
        key, grab = hit
        self._select(key)
        self.setFocus()   # so the arrow keys nudge this box
        self._adjusting = _Adjustment(
            key=key, grab=grab, press_view=event.position(),
            press_scene=self.mapToScene(event.position().toPoint()),
            original=self._pixel_boxes()[key])
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:   # noqa: N802 (Qt override)
        if event is None:
            return
        adjusting = self._adjusting
        if adjusting is not None:
            travelled = (event.position() - adjusting.press_view).manhattanLength()
            if adjusting.current is None and travelled < QApplication.startDragDistance():
                event.accept()   # still a click, not a drag: a box never nudges by accident
                return
            pos = self.mapToScene(event.position().toPoint())
            frame_w, frame_h = self._frame_size()
            adjusting.current = adjust(
                adjusting.original, adjusting.grab, pos.x() - adjusting.press_scene.x(),
                pos.y() - adjusting.press_scene.y(), frame_w, frame_h, MIN_DRAW_PX)
            self._redraw()
            event.accept()
            return
        if self._drag_origin is None:
            super().mouseMoveEvent(event)
            self._update_hover_cursor(event)
            return
        if self._rubber is not None:
            current = self._clamped_scene_pos(event)
            self._rubber.setRect(QRectF(self._drag_origin, current).normalized())
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:   # noqa: N802 (Qt override)
        if event is None:
            return
        adjusting = self._adjusting
        if adjusting is not None:
            self._adjusting = None
            event.accept()
            if adjusting.current is not None and adjusting.current != adjusting.original:
                self._commit_box(adjusting.key, adjusting.current)
            else:
                self._redraw()   # a click, or a drag that ended where it began
            return
        if self._drag_origin is None:
            super().mouseReleaseEvent(event)
            self._update_hover_cursor(event)
            return
        origin = self._drag_origin
        end = self._clamped_scene_pos(event)
        key = self._armed
        self._cancel_drag()
        event.accept()
        drag = QRectF(origin, end).normalized()
        if key is None or drag.width() < MIN_DRAW_PX or drag.height() < MIN_DRAW_PX:
            return   # a click or a slipped drag, not a box; stay armed for a retry
        frame_w, frame_h = self._frame_size()
        box = [round(drag.left()), round(drag.top()), round(drag.right()), round(drag.bottom())]
        try:
            norm = NormRect.from_pixels(box, frame_w, frame_h)
        except ValueError:
            return   # rounding collapsed a hairline drag; treat like a slipped click
        self.set_armed_region(None)
        self.regionDrawn.emit(key, norm)

    def keyPressEvent(self, event: QKeyEvent | None) -> None:   # noqa: N802 (Qt override)
        """Arrow keys nudge the selected box (Shift: NUDGE_BIG_PX); Escape cancels a drag
        on a box, else deselects. Anything else, or Escape with nothing to undo, goes on to
        the base pane and then the owner (the editor's Escape cancels an armed Draw). On a
        pane that isn't adjustable nothing is ever selected, so every key goes on."""
        if event is None:
            return
        key = event.key()
        if key == Qt.Key.Key_Escape and self._adjusting is not None:
            self._cancel_adjustment()
            event.accept()
            return
        if key == Qt.Key.Key_Escape and self._selected is not None:
            self._select(None)
            event.accept()
            return
        direction = NUDGE_DIRECTIONS.get(Qt.Key(key))
        if direction is not None and self._selected is not None and self._adjusting is None:
            shift = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            step = NUDGE_BIG_PX if shift else NUDGE_PX
            self._nudge(direction[0] * step, direction[1] * step)
            event.accept()
            return
        super().keyPressEvent(event)

    def _cancel_drag(self) -> None:
        """Drop any in-flight drag: forget the origin and remove the rubber band."""
        self._drag_origin = None
        scene = self.scene()
        if self._rubber is not None and scene is not None:
            scene.removeItem(self._rubber)
        self._rubber = None

    def _frame_size(self) -> tuple[int, int]:
        """The captured frame's width and height in pixels (see ImagePane.frame_rect)."""
        frame_rect = self.frame_rect()
        return round(frame_rect.width()), round(frame_rect.height())

    def _clamped_scene_pos(self, event: QMouseEvent) -> QPointF:
        """The event's position in scene (frame-pixel) coordinates, clamped into the
        frame so a drag released outside the image still yields an in-frame box."""
        pos = self.mapToScene(event.position().toPoint())
        rect = self.frame_rect()
        return QPointF(min(max(pos.x(), rect.left()), rect.right()),
                       min(max(pos.y(), rect.top()), rect.bottom()))

    @staticmethod
    def _region_pen(key: str) -> QPen:
        """The pen for a named region's own color (used by the drag rubber band, which
        is always drawn for a real region key)."""
        return RegionDrawPane._pen_for_color(REGION_COLORS[key])

    @staticmethod
    def _pen_for_color(color: QColor) -> QPen:
        pen = QPen(color)
        pen.setWidth(BOX_PEN_WIDTH)
        pen.setCosmetic(True)   # constant on-screen width at any zoom
        return pen
