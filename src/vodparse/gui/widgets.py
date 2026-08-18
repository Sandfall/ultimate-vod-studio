"""Small custom widgets for the desktop GUI (app.py).

Import these only after ``qt_plugins.configure_qt_plugins()`` has run (every app entry
module does so at its top), since this module imports PyQt6 widgets at load time.
"""

import sys
from collections.abc import Sequence
from typing import override

from PyQt6.QtCore import QEvent, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPaintEvent, QPixmap
from PyQt6.QtWidgets import QApplication, QStackedWidget, QStyle, QToolButton, QWidget

GLYPH_PT_BUMP = 12   # points added to the base font so the header glyph reads as an icon, not text
HEADER_BTN_PAD = 2   # px of breathing room around the glyph on each side of its square button
WIN_SYMBOL_FONT = "Segoe UI Symbol"  # monochrome Windows symbol font; has ⚙ ⚒ ⟳ (avoids Segoe UI Emoji)


class FittingStack(QStackedWidget):
    """A QStackedWidget that reports the *current* page's size hint instead of the largest
    page's, so the shorter pipeline/extract screens don't reserve the (tall) review page's
    height; the window fits the page that's actually showing."""

    def __init__(self) -> None:
        super().__init__()
        self.currentChanged.connect(lambda _idx: self.updateGeometry())

    @override
    def sizeHint(self) -> QSize:
        page = self.currentWidget()
        return page.sizeHint() if page is not None else super().sizeHint()

    @override
    def minimumSizeHint(self) -> QSize:
        page = self.currentWidget()
        return page.minimumSizeHint() if page is not None else super().minimumSizeHint()


class GlyphButton(QToolButton):
    """A flat tool button that paints a centred glyph pixmap over the native button face. The
    style still draws the frame, hover highlight and (if a menu is set) the dropdown arrow; we
    just blit the glyph on top. Done this way because routing the glyph through setIcon lets
    QToolButton centre it with a half-pixel rounding error (visibly "1px too high/right" on
    Retina) and, when the icon is button-sized, the auto-raise frame insets and downscales it.
    The pixmap is centred at device-pixel precision by glyph_pixmap.

    The glyph is rendered in the palette's ``ButtonText`` color and **re-rendered on a live
    palette change** (``changeEvent``), so a runtime OS light/dark flip (which swaps the app
    palette via ``common.apply_system_theme``) doesn't leave a baked-in ink color (e.g. white
    glyphs invisible on the now-light button face) until the next restart."""

    def __init__(self, glyph: str, font: QFont, side: int,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._glyph = glyph
        self._glyph_font = font
        self._side = side
        self._glyph_pixmap = self._render()

    def _render(self) -> QPixmap:
        """Render the glyph in the button's *current* ButtonText color (see glyph_pixmap)."""
        return glyph_pixmap(self._glyph, self._glyph_font,
                            self.palette().buttonText().color(), self._side)

    @override
    def changeEvent(self, event: QEvent | None) -> None:
        super().changeEvent(event)
        # PaletteChange fires on a live theme flip; re-render so the ink tracks the new palette
        # (mirrors app.py's _title_note re-resolving its color on the same event).
        if event is not None and event.type() == QEvent.Type.PaletteChange:
            self._glyph_pixmap = self._render()
            self.update()

    @override
    def paintEvent(self, event: QPaintEvent | None) -> None:
        super().paintEvent(event)            # native face: frame, hover, menu arrow
        painter = QPainter(self)
        # Map the whole (high-res) pixmap onto the widget rect; rect-draw scales source→dest 1:1
        # at the screen's pixel ratio, independent of any dpr metadata.
        painter.drawPixmap(self.rect(), self._glyph_pixmap)
        painter.end()


def make_icon_button(glyph: str, tooltip: str, fallback: QStyle.StandardPixmap) -> QToolButton:
    """A flat header button showing a single glyph (e.g. ⚙/⚒). Prefers the glyph; on the rare
    font that lacks it, falls back to a native style icon so it never renders as a tofu box. The
    caller wires clicked/setMenu. BMP glyphs only; inFont takes a single QChar, so astral-plane
    emoji (🔧) can't be probed this way."""
    probe = QToolButton()
    glyph_font = probe.font()
    glyph_font.setPointSize(glyph_font.pointSize() + GLYPH_PT_BUMP)
    if sys.platform == "win32":
        # Windows' base UI font lacks ⚙/⚒, so Qt merges the *color-emoji* font (Segoe UI
        # Emoji) and the glyphs render as tacky multicolor emoji that ignore our pen color.
        # Pin the monochrome Segoe UI Symbol family and disable merging so we get flat,
        # palette-colored icons like macOS. macOS/Linux already merge to a monochrome symbol
        # font, so they're left as-is.
        glyph_font.setFamily(WIN_SYMBOL_FONT)
        glyph_font.setStyleStrategy(QFont.StyleStrategy.NoFontMerging)
    # Size the square to the glyph it holds (its line height + padding) rather than a fixed pixel
    # count, so the button grows with the app's Text-size preference instead of clipping the glyph.
    side = QFontMetrics(glyph_font).height() + 2 * HEADER_BTN_PAD
    btn: QToolButton
    if QFontMetrics(glyph_font).inFont(glyph):
        # Paint the glyph into a full button-sized square, centred at device-pixel precision
        # (see glyph_pixmap / GlyphButton). Routing it through setText or a tight-cropped
        # icon leaves the glyph a pixel off, because Qt rounds a half-pixel icon margin in
        # *logical* units (5.5px → 6); visible as "1px too high/right" on Retina. GlyphButton
        # keeps the glyph/font/side so it can re-render on a live palette (theme) change.
        btn = GlyphButton(glyph, glyph_font, side)
    else:
        btn = QToolButton()
        style = probe.style()
        if style is not None:
            btn.setIcon(style.standardIcon(fallback))
    btn.setAutoRaise(True)           # flat until hovered
    btn.setToolTip(tooltip)          # hover also works, for free
    btn.setFixedSize(side, side)     # square, stable size (tracks the glyph font)
    return btn


def pin_windows_symbol_font(widgets: Sequence[QWidget], glyphs: str) -> None:
    """On Windows, pin these widgets' font to the monochrome Segoe UI Symbol family so the
    given glyphs render as flat, palette-colored symbols instead of multicolor Segoe UI Emoji
    (same merging problem as make_icon_button; see the comment there). Unlike make_icon_button
    this keeps the widgets' ordinary text rendering path, so it suits buttons that mix a glyph
    with a label ("▶ Play"); Segoe UI Symbol carries its own Segoe-style Latin, so the label
    text still looks native. No-op off Windows, and when the font lacks any of the glyphs
    (all widgets are then left alone together, so none renders a tofu box and the group stays
    visually consistent). BMP glyphs only: inFont takes a single QChar (see make_icon_button)."""
    if sys.platform != "win32" or not widgets:
        return
    probe = QFont(WIN_SYMBOL_FONT)
    if not all(QFontMetrics(probe).inFont(glyph) for glyph in glyphs):
        return
    for widget in widgets:
        font = widget.font()   # keep each widget's inherited size/weight, swap only the family
        font.setFamily(WIN_SYMBOL_FONT)
        font.setStyleStrategy(QFont.StyleStrategy.NoFontMerging)
        widget.setFont(font)


def glyph_pixmap(glyph: str, font: QFont, color: QColor, size: int) -> QPixmap:
    """Draw a glyph, crop to its *actual* ink (non-transparent pixels), then centre that ink
    in a `size`×`size` square; all in device pixels (at the screen's ratio, so it stays crisp
    on Retina). An odd leftover pixel is biased down and right-margin-heavy, so the glyph can
    never read as "too high" or "too far right". Cropping to real ink rather than fontMetrics'
    tightBoundingRect avoids clipping antialiased edges by a pixel. See make_icon_button."""
    fm = QFontMetrics(font)
    pad = 4                                    # slack so antialiased edges can't be clipped
    screen = QApplication.primaryScreen()
    dpr = screen.devicePixelRatio() if screen is not None else 1.0
    canvas = QPixmap(round((fm.horizontalAdvance(glyph) + pad * 2) * dpr),
                     round((fm.height() + pad * 2) * dpr))
    canvas.setDevicePixelRatio(dpr)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    painter.setFont(font)
    painter.setPen(color)
    painter.drawText(pad, pad + fm.ascent(), glyph)
    painter.end()
    img = canvas.toImage()                     # device pixels
    img.setDevicePixelRatio(1.0)               # treat as raw pixels; else drawImage halves it
    xs = [x for x in range(img.width())
          for y in range(img.height()) if img.pixelColor(x, y).alpha() > 0]
    ys = [y for y in range(img.height())
          for x in range(img.width()) if img.pixelColor(x, y).alpha() > 0]
    side = round(size * dpr)
    square = QPixmap(side, side)
    square.fill(Qt.GlobalColor.transparent)
    if xs:
        x0, y0 = min(xs), min(ys)
        w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
        qp = QPainter(square)
        # floor the left margin (extra px lands on the right → never "too far right");
        # ceil the top margin (extra px lands on top → never "too high")
        qp.drawImage((side - w) // 2, (side - h + 1) // 2, img, x0, y0, w, h)
        qp.end()
    return square   # plain pixmap (no dpr); GlyphButton rect-draws it onto the widget
