"""The Review screen's custom-painted seek scrubber, built by vodparse.gui.player.trim_player.TrimPlayer.

Paints a flat track showing the current set's match coverage in green: with lighter
lead-in/tail buffers around it; neighbouring sets in a more-saturated muted red, thin amber
ticks at each match boundary, and a playhead. Clicking the bar jumps to that position;
dragging tracks the cursor (Qt's default slider only pages by a step). Paint and hit-testing
share one geometry (``_track``), so clicks land exactly where the bar is drawn.

``_bar_colors()`` picks the palette live, once per repaint, via ``vodparse.gui.common.is_dark_theme``,
so the bar re-tints itself on a light/dark theme switch.

Import this only after ``qt_plugins.configure_qt_plugins()`` has run (every app entry
module does so at its top), since this module imports PyQt6 widgets at load time.
"""

from dataclasses import dataclass

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent
from PyQt6.QtWidgets import QSlider

from vodparse.gui.common import is_dark_theme

# Scrubber colours: the current set vs. the surrounding ±margin context vs. the playhead. Each theme
# gets its own palette (the dark values are tuned for a dark track, the light ones for a pale track),
# picked live by _bar_colors() so the bar re-tints on a light/dark switch (see vodparse.gui.common.is_dark_theme).
@dataclass(frozen=True)
class _BarColors:
    """The six scrubber colours for one theme; paintEvent shows where each lands."""
    set_core: QColor     # green: the match coverage that's the heart of the clip
    set_buffer: QColor   # the same green a touch lighter: the lead-in/tail buffer
    outside: QColor      # context that isn't yours: the empty track under everything
    other_set: QColor    # a more-saturated muted red: a neighbouring set in view
    playhead: QColor     # the current position
    marker: QColor       # soft amber: where each (non-first) match starts


# Dark keeps the original hand-tuned values verbatim; light is tuned for a pale track/background.
_DARK_BARS = _BarColors(
    set_core=QColor(63, 185, 80),
    set_buffer=QColor(104, 202, 118),
    outside=QColor(110, 78, 82),
    other_set=QColor(130, 60, 64),
    playhead=QColor(235, 235, 235),
    marker=QColor(200, 172, 110),
)
_LIGHT_BARS = _BarColors(
    set_core=QColor(31, 136, 61),     # GitHub light-mode greens: legible on a pale track
    set_buffer=QColor(87, 171, 106),
    outside=QColor(222, 186, 188),    # light dusty rose: the empty "not yours" track, faintly red not grey
    other_set=QColor(176, 74, 76),    # muted brick red: a neighbouring set, clearly deeper than the track
    playhead=QColor(33, 33, 33),      # near-black, mirroring the dark theme's near-white
    marker=QColor(176, 122, 38),      # deeper amber-tan tick
)


def _bar_colors() -> _BarColors:
    """The scrubber palette for the current theme, read live so it follows a theme switch."""
    return _DARK_BARS if is_dark_theme() else _LIGHT_BARS

_TRACK_HEIGHT = 6
_TRACK_INSET = 7   # px on each end so the playhead isn't clipped at the extremes


class SeekSlider(QSlider):
    """Custom-painted horizontal scrubber: a flat track with the current set drawn green and the
    surrounding ±margin context dim red, plus a thin playhead. Clicking jumps to that position and
    dragging tracks the cursor (Qt's default only pages by a step). Paint and hit-testing share
    one geometry (_track) so clicks line up exactly with what's drawn; sliderMoved is emitted so
    the existing seek connection fires."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._clip_start_ms = 0
        self._clip_end_ms = 0
        self._core_regions: list[tuple[int, int]] = []
        self._other_regions: list[tuple[int, int]] = []
        self._markers: list[int] = []

    def set_clip_region(self, start_ms: int, end_ms: int) -> None:
        """The set's bounds (a sub-range of [minimum, maximum]) to paint green."""
        self._clip_start_ms = start_ms
        self._clip_end_ms = end_ms
        self.update()

    def set_core_regions(self, regions: list[tuple[int, int]]) -> None:
        """Match coverage inside the clip as (start_ms, end_ms) sub-ranges, painted in the main
        green; the rest of the clip; the lead-in/tail buffer (and mid-set gaps, in debug); stays
        a shade lighter. Pass [] to paint the whole clip in the main green."""
        self._core_regions = list(regions)
        self.update()

    def set_other_regions(self, regions: list[tuple[int, int]]) -> None:
        """Spans of *other* sets (absolute ms) to paint a deeper muted red, marking where
        neighbouring sets sit. Each is clipped to the current view range when painted."""
        self._other_regions = list(regions)
        self.update()

    def set_markers(self, markers_ms: list[int]) -> None:
        """Absolute ms positions to draw as match-boundary ticks (those within range are shown)."""
        self._markers = list(markers_ms)
        self.update()

    # ── Geometry (shared by paint + hit-testing) ────────────────────────────────
    def _track(self) -> QRect:
        y = (self.height() - _TRACK_HEIGHT) // 2
        return QRect(_TRACK_INSET, y, max(1, self.width() - 2 * _TRACK_INSET), _TRACK_HEIGHT)

    def _x_for(self, value: int) -> int:
        track = self._track()
        span = self.maximum() - self.minimum()
        frac = (value - self.minimum()) / span if span else 0.0
        return track.left() + round(frac * track.width())

    def _value_for(self, x: float) -> int:
        track = self._track()
        frac = (x - track.left()) / track.width() if track.width() else 0.0
        frac = min(1.0, max(0.0, frac))
        return self.minimum() + round(frac * (self.maximum() - self.minimum()))

    # ── Mouse: click-to-seek + drag ─────────────────────────────────────────────
    def mousePressEvent(self, ev: QMouseEvent | None) -> None:
        if ev is not None and ev.button() == Qt.MouseButton.LeftButton and self.maximum() > self.minimum():
            self.setSliderDown(True)
            self._seek_to(ev.position().x())
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent | None) -> None:
        if ev is not None and self.isSliderDown() and ev.buttons() & Qt.MouseButton.LeftButton:
            self._seek_to(ev.position().x())
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent | None) -> None:
        if ev is not None and ev.button() == Qt.MouseButton.LeftButton and self.isSliderDown():
            self.setSliderDown(False)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def _seek_to(self, x: float) -> None:
        val = self._value_for(x)
        self.setValue(val)
        self.sliderMoved.emit(val)

    # ── Custom paint ────────────────────────────────────────────────────────────
    def paintEvent(self, event: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        track = self._track()
        radius = track.height() / 2
        c = _bar_colors()

        painter.setBrush(c.outside)
        painter.drawRoundedRect(track, radius, radius)

        # Neighbouring sets within view: a deeper red over the dim track, so the edge of each
        # band marks where an adjacent set begins. The gaps between sets stay the dim red, and
        # the current set's green (drawn next) wins any overlap.
        painter.setBrush(c.other_set)
        for os, oe in self._other_regions:           # other sets' (start, end), like cs/ce below
            lo, hi = max(os, self.minimum()), min(oe, self.maximum())
            if hi > lo:
                ox1, ox2 = self._x_for(lo), self._x_for(hi)
                painter.drawRect(QRect(ox1, track.top(), max(1, ox2 - ox1), track.height()))

        if self._clip_end_ms > self._clip_start_ms:
            x1 = self._x_for(self._clip_start_ms)
            x2 = self._x_for(self._clip_end_ms)
            # The clip's buffer (lead-in/tail, plus mid-set gaps in debug) is a shade lighter; the
            # match coverage is the main green. When coverage is unknown, the whole clip is main.
            painter.setBrush(c.set_buffer if self._core_regions else c.set_core)
            painter.drawRect(QRect(x1, track.top(), max(1, x2 - x1), track.height()))
            painter.setBrush(c.set_core)
            for cs, ce in self._core_regions:
                lo, hi = max(cs, self._clip_start_ms), min(ce, self._clip_end_ms)
                if hi > lo:
                    kx1, kx2 = self._x_for(lo), self._x_for(hi)
                    painter.drawRect(QRect(kx1, track.top(), max(1, kx2 - kx1), track.height()))

        # Match markers: thin amber ticks at each individual-match boundary within view.
        painter.setBrush(c.marker)
        for m in self._markers:
            if self.minimum() <= m <= self.maximum():
                mx = self._x_for(m)
                painter.drawRect(QRect(mx - 1, track.top() - 4, 2, track.height() + 8))

        px = self._x_for(self.value())
        painter.setBrush(c.playhead)
        painter.drawRect(QRect(px - 1, track.top() - 3, 2, track.height() + 6))
