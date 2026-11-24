"""Small custom widgets for the desktop GUI (app.py).

Import these only after ``qt_plugins.configure_qt_plugins()`` has run (every app entry
module does so at its top), since this module imports PyQt6 widgets at load time.
"""

import sys
from collections.abc import Sequence
from typing import override

from PyQt6.QtCore import QEvent, QSize, Qt, QTimer
from PyQt6.QtGui import (QColor, QFont, QFontMetrics, QHelpEvent, QPainter, QPaintEvent,
                         QPalette, QPixmap)
from PyQt6.QtWidgets import (QApplication, QComboBox, QStackedWidget, QStyle,
                             QStyleOptionComboBox, QStylePainter, QToolButton, QToolTip,
                             QWidget)

GLYPH_PT_BUMP = 12   # points added to the base font so the header glyph reads as an icon, not text
HEADER_BTN_PAD = 2   # px of breathing room around the glyph on each side of its square button
WIN_SYMBOL_FONT = "Segoe UI Symbol"  # monochrome Windows symbol font; has ⚙ ⚒ ⟳ (avoids Segoe UI Emoji)
COMBO_ELIDE_SAFETY_PX = 2  # px a shortened combo label is kept clear of where clipping would start
COMBO_LIST_ITEM_PAD = 6  # px allowed on each side of an item's text in a combo's drop-down list


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


class ElidingComboBox(QComboBox):
    """A QComboBox that ends its current item's text with "…" when the closed combo is too
    narrow to show it all. Qt's own combo simply clips the text there, mid-letter, which
    reads as a rendering fault; this is for a combo whose width is capped (see
    ``combo_width_for_chars``) while its items can be any length. The drop-down list is
    not touched: ``widen_popup_to_fit_items`` is what lets it show the items in full.

    Its minimum is its size hint, up to its maximum width (see ``minimumSizeHint``): a
    layout short of room shrinks something else, or the window grows, before this combo
    has to cut a name its cap would let it show whole.

    While its text is cut short, hovering the closed combo shows the whole text as a
    tooltip (see ``tooltip_text``), as an ElidingLabel does."""

    @override
    def minimumSizeHint(self) -> QSize:
        """As wide as the size hint, up to the combo's maximum width.

        QComboBox works its own minimum out from its items the first time it is asked and
        never again, and a layout gives a widget at least its minimum. So a combo that held
        a long item at that moment would stay that wide for good, even after the item is
        deleted or another list is loaded into it. The size hint does not have that
        problem: Qt works it out again whenever the items change (for a combo that adjusts
        to its contents), so a minimum taken from it follows the items in both directions.
        Capped at the maximum width because a name longer than the cap is cut short with
        "…" anyway, so there is no point asking for more room than the combo may take."""
        qt_minimum = super().minimumSizeHint()
        return QSize(min(self.sizeHint().width(), self.maximumWidth()), qt_minimum.height())

    @override
    def paintEvent(self, e: QPaintEvent | None) -> None:
        """Paint the combo as QComboBox does (its frame and arrow, then its label), with the
        label's text shortened to fit the space the style gives it."""
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        painter = QStylePainter(self)
        painter.setPen(self.palette().color(QPalette.ColorRole.Text))
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        option.currentText = self.shown_text()
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)

    @override
    def event(self, e: QEvent | None) -> bool:
        """Answer a tooltip request with ``tooltip_text`` (the whole text while it is cut
        short). Worked out when the tooltip is asked for, so it never lags behind a change
        of item or width. Every other event goes to QComboBox as usual."""
        if e is not None and e.type() == QEvent.Type.ToolTip and isinstance(e, QHelpEvent):
            text = self.tooltip_text()
            if text:
                QToolTip.showText(e.globalPos(), text, self)
            else:
                QToolTip.hideText()
                e.ignore()
            return True
        return super().event(e)

    def tooltip_text(self) -> str:
        """What hovering the closed combo shows: the current item's whole text while
        ``shown_text`` has cut it short, else the combo's own tooltip (empty unless one was
        set)."""
        text = self.currentText()
        if self.shown_text() != text:
            return text
        return self.toolTip()

    def shown_text(self) -> str:
        """The current item's text as the closed combo draws it at its present width: whole
        when it fits, else cut short and ended with "…".

        Whether it fits is judged from the combo's own width, not from the style's text
        area: each style insets the label by a different, unpublished amount (the Windows 11
        style most of all), but every style can show a text whole in a combo as wide as it
        would size one for that text. However many pixels the combo falls short of that
        width is how many the text has to lose.

        The text is measured the way QComboBox measures an item to size itself: by its
        ``boundingRect`` (the width of its ink), not its ``horizontalAdvance``, which also
        counts the space after the last letter and so usually comes out a pixel or two
        wider. With the advance, a combo exactly as wide as Qt sized it for its text would
        fall short by those pixels and cut a letter."""
        text = self.currentText()
        metrics = self.fontMetrics()
        text_width = metrics.boundingRect(text).width()
        shortfall = combo_width_for_text_width(self, text_width) - self.width()
        if shortfall <= 0:
            return text
        return metrics.elidedText(
            text, Qt.TextElideMode.ElideRight, text_width - shortfall - COMBO_ELIDE_SAFETY_PX)


class PopupPickComboBox(ElidingComboBox):
    """An ElidingComboBox that can tell a pick made in its popup list from every other way its
    current index moves: the arrow keys, the mouse wheel or type-ahead on the closed box,
    or a programmatic change. For a combo that carries an action item (the Overlay layout
    combo's "New layout…"): the action should run when the user picks it, never because a
    key press happened to land on it. Qt's own ``activated`` signal cannot tell the two
    apart; it fires for the keys and the wheel as well.

    How: on a pick Qt first closes the popup (``hidePopup``), then changes the index (unless
    the picked item is already current) and emits ``activated``, all inside one event, and
    nothing else closes the popup and does either within one event. So ``hidePopup`` raises
    a flag that lasts until control returns to the event loop, and the owner's
    ``currentIndexChanged`` or ``activated`` handler asks ``picked_from_popup()``; only
    ``activated`` hears a pick of the item already current. Dismissing the popup (Escape, a
    click elsewhere) changes no index and emits nothing, so the flag lapses unread."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._popup_just_closed = False

    def picked_from_popup(self) -> bool:
        """Whether the index change or ``activated`` being handled right now came from a
        pick in the popup."""
        return self._popup_just_closed

    @override
    def hidePopup(self) -> None:
        """Close the popup and, when one was up, remember that until the event loop runs
        again (a pick's index change follows before then). Raised only once the base call
        returns, in case a style closes the popup with a nested event loop, which would run
        the reset before the index changes."""
        view = self.view()
        popup_was_up = view is not None and view.isVisible()
        super().hidePopup()
        if popup_was_up:
            self._popup_just_closed = True
            QTimer.singleShot(0, self._forget_popup_close)

    def _forget_popup_close(self) -> None:
        """Back in the event loop: whatever moves the index from here on is not that pick."""
        self._popup_just_closed = False


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


def combo_width_for_text_width(combo: QComboBox, text_width: int) -> int:
    """The width `combo` asks for to show an item whose text is `text_width` px wide in
    full: the text plus the frame and drop-down arrow the current style draws around it.
    The same sum QComboBox does to size itself to its items."""
    option = QStyleOptionComboBox()
    option.initFrom(combo)
    option.editable = combo.isEditable()
    option.frame = combo.hasFrame()
    text_size = QSize(text_width, combo.fontMetrics().height())
    style = combo.style()
    assert style is not None  # a widget always has a style
    return style.sizeFromContents(
        QStyle.ContentsType.CT_ComboBox, option, text_size, combo).width()


def combo_width_for_chars(combo: QComboBox, chars: int) -> int:
    """The width `combo` asks for to show an item `chars` characters long in full (see
    ``combo_width_for_text_width``). Counted the way QComboBox counts a minimum contents
    length, in widths of the letter "x", so a real name of that many characters fits unless
    it is unusually wide. For capping a combo that grows with its contents:
    ``combo.setMaximumWidth(combo_width_for_chars(combo, 22))``."""
    return combo_width_for_text_width(combo, combo.fontMetrics().horizontalAdvance("x" * chars))


def widen_popup_to_fit_items(combo: QComboBox) -> None:
    """Let `combo`'s drop-down list be as wide as its longest item, even when the closed
    combo is narrower (a combo with a maximum width cuts a long current item short with
    "…", and the list is where the names can still be read in full). Call again whenever the
    items change. Some styles (macOS, Fusion) size the list to its items by themselves;
    the Windows styles make it exactly as wide as the combo, which this overrides."""
    view = combo.view()
    style = combo.style()
    assert view is not None and style is not None  # a combo always has both
    # Measured from the item texts rather than asked of the view (sizeHintForColumn): under
    # the styles that draw the list as a menu, the view answers with far less than a long
    # item needs.
    metrics = view.fontMetrics()
    longest_text_width = max(
        (metrics.horizontalAdvance(combo.itemText(index)) for index in range(combo.count())),
        default=0)
    scroll_bar_width = style.pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
    view.setMinimumWidth(longest_text_width + 2 * COMBO_LIST_ITEM_PAD
                         + 2 * view.frameWidth() + scroll_bar_width)


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
