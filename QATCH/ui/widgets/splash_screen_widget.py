"""QATCH.ui.widgets.splash_screen_widget

Animated QATCH splash screen: the brand's mosaic-circle mark with a
shimmering brightness sweep across its tiles and a soft pulsing glow,
floating over a fully transparent background (no backdrop, no separate
loader) - ported from the "QATCH Splash.dc.html" design's shimmer motion /
transparent background / no-loader variant, with the design's own wordmark
added beneath the mark plus a smaller version/build caption (carried over
from the previous plain-text splash, restyled to match the wordmark).

Runs as its own subprocess (see app.py's "--splash" launch path), so its
entire public surface is just "construct it and show() it" - the main
process never calls any method on this instance directly, it only
terminates the subprocess once the real UI is ready.
"""

import sys
import types

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.core.constants import Constants
from QATCH.ui.styles.typography import make_qfont

# Tile geometry + base fill color, sampled directly from the QATCH
# mosaic-circle brand mark (the logo corner of QATCH/icons/qatch-splash.png)
# via the design file's own TILES table. Kept as literal data instead of an
# SVG/PNG asset so the whole splash paints itself with QPainter - no
# bundled image needed, and no image-loading failure mode in the
# subprocess.
_TILES = [
    # (x, y, base_color, row, col)
    (89, 7, "#72ccf2", 0, 1),
    (125, 7, "#a9def7", 0, 2),
    (162, 7, "#a8def8", 0, 3),
    (53, 43, "#72cbf3", 1, 0),
    (89, 43, "#a5def8", 1, 1),
    (125, 43, "#a4ddf8", 1, 2),
    (162, 43, "#a5ddf8", 1, 3),
    (198, 43, "#60c8f1", 1, 4),
    (53, 80, "#a7def8", 2, 0),
    (89, 80, "#a5ddf9", 2, 1),
    (125, 80, "#6bccf5", 2, 2),
    (162, 80, "#6acbf5", 2, 3),
    (198, 80, "#78cef5", 2, 4),
    (53, 116, "#26c3ee", 3, 0),
    (89, 116, "#57c7f5", 3, 1),
    (125, 116, "#a5ddf8", 3, 2),
    (162, 116, "#54c6f4", 3, 3),
    (198, 116, "#a5def8", 3, 4),
    (89, 152, "#26c2ee", 4, 1),
    (125, 152, "#a3ddf8", 4, 2),
    (198, 152, "#25c2ed", 4, 4),
    (125, 189, "#24c1ee", 5, 2),
]
_TILE_SIZE = 29.0
_TILE_RADIUS = 1.2
_VIEWBOX = 273.0  # native SVG width the coordinates above were sampled at
_CIRCLE_CX, _CIRCLE_CY, _CIRCLE_R = 139.5, 139.0, 119.0
_GRADIENT_TOP = QtGui.QColor("#17aeef")
_GRADIENT_BOTTOM = QtGui.QColor("#0698e0")
_GLOW_COLOR = QtGui.QColor(10, 164, 229)

_SHIMMER_PERIOD_S = 3.2  # matches the design file's default "speed" prop
_SHIMMER_DELAY_STEP_S = 0.13  # per (row + col) step, same as the design's shimmer branch
_GLOW_PERIOD_S = 4.0
_TICK_MS = 33  # ~30fps, same cadence used elsewhere in this app's hand-rolled animations

_LOGO_SIZE = 240.0  # rendered logo diameter, matches the design's own preview size
_GLOW_SIZE = 280.0  # matches the design's glow layer size
_MIN_WINDOW_SIZE = 320  # floor size (also the old window's fixed size) in case the text below is narrower than the glow
_TOP_CLEARANCE = (_MIN_WINDOW_SIZE - _GLOW_SIZE) / 2  # space above the glow's resting radius; unchanged from the old fixed-size window so the mark itself isn't affected by the text added below it

# Wordmark ("QATCH" + "Technologies"): geometry and color straight from the
# design file's wordmarkStyle/-Strong/-Light for the transparent/light
# background variant (the only variant this splash renders).
_WORDMARK_COLOR_HEX = "#0d3d55"
_WORDMARK_FONT_PX = 25
_WORDMARK_LETTER_SPACING_EM = 0.14
_WORDMARK_MARGIN_TOP = 26
_WORDMARK_LIGHT_OPACITY = 0.85

# Version/build caption: the two lines the previous plain-text splash
# showed ("Version: ..." / "Build Date: ..."), restyled smaller and
# uppercase/letter-spaced/translucent to read as a secondary line under
# the wordmark rather than competing with it.
_BUILD_FONT_PX = 11
_BUILD_LETTER_SPACING_EM = 0.10
_BUILD_MARGIN_TOP = 12
_BUILD_LINE_GAP = 3
_BUILD_OPACITY = 0.62

_SIDE_PADDING = 26  # horizontal clearance so wide text never touches the window edge
_BOTTOM_PADDING = 22


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _apply_css_filters(color: QtGui.QColor, brightness: float, saturate: float) -> QtGui.QColor:
    """Reproduces CSS `filter: brightness(b) saturate(s)` applied in that
    order, matching qc-shimmer's `brightness(1.5) saturate(1.25)` peak.

    brightness() is a per-channel linear scale; saturate() is the standard
    luminance-preserving saturation matrix - both per the CSS Filter
    Effects spec, not just an HSV value/saturation nudge, so the shimmer
    peak matches what a browser would actually render.
    """
    r, g, b, a = color.getRgb()
    r = min(255.0, r * brightness)
    g = min(255.0, g * brightness)
    b = min(255.0, b * brightness)

    s = saturate
    nr = (0.213 + 0.787 * s) * r + (0.715 - 0.715 * s) * g + (0.072 - 0.072 * s) * b
    ng = (0.213 - 0.213 * s) * r + (0.715 + 0.285 * s) * g + (0.072 - 0.072 * s) * b
    nb = (0.213 - 0.213 * s) * r + (0.715 - 0.715 * s) * g + (0.072 + 0.928 * s) * b

    return QtGui.QColor(
        max(0, min(255, round(nr))),
        max(0, min(255, round(ng))),
        max(0, min(255, round(nb))),
        a,
    )


class QatchSplashScreen(QtWidgets.QWidget):
    """Animated, transparent-backdrop QATCH splash: a shimmering
    mosaic-circle logo with a soft pulsing glow, the QATCH Technologies
    wordmark, and a small version/build caption - no separate loader chrome.
    """

    def __init__(self) -> None:
        # Qt.SplashScreen + WA_TranslucentBackground is a known bad
        # combination on Windows - it can fail to composite at all (the
        # window never becomes visible), since Qt.SplashScreen's own
        # native-window setup doesn't reliably enable the layered-window
        # style translucency needs. Frameless + StaysOnTop + Tool is the
        # combination already proven elsewhere in this app for translucent
        # top-level floating windows (see AccountPopup/AdvancedMainWidget),
        # so this uses that instead. Tool keeps it out of the taskbar/
        # alt-tab list, same as SplashScreen would.
        super().__init__(
            None,
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool,
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)

        self._build_fonts()
        self._layout = self._compute_layout()
        self.setFixedSize(round(self._layout.width), round(self._layout.height))
        self._center_on_screen()

        self._elapsed = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

        self.show()
        self.raise_()
        self.activateWindow()

    def _center_on_screen(self) -> None:
        screen = QtWidgets.QApplication.primaryScreen()
        geo = screen.geometry() if screen is not None else QtCore.QRect(0, 0, 1920, 1080)
        self.move(
            geo.x() + (geo.width() - self.width()) // 2,
            geo.y() + (geo.height() - self.height()) // 2,
        )

    def _build_fonts(self) -> None:
        self._wordmark_bold_font = make_qfont(pixel_size=_WORDMARK_FONT_PX, weight=QtGui.QFont.Bold)
        self._wordmark_bold_font.setLetterSpacing(
            QtGui.QFont.AbsoluteSpacing, _WORDMARK_FONT_PX * _WORDMARK_LETTER_SPACING_EM
        )

        # Same family/size as the bold run so the two sit on one baseline
        # with matching metrics - only the weight differs, per the design's
        # wordmarkStrong (700) / wordmarkLight (300) split.
        self._wordmark_light_font = QtGui.QFont(self._wordmark_bold_font)
        self._wordmark_light_font.setWeight(QtGui.QFont.Light)

        self._build_font = make_qfont(pixel_size=_BUILD_FONT_PX)
        self._build_font.setLetterSpacing(
            QtGui.QFont.AbsoluteSpacing, _BUILD_FONT_PX * _BUILD_LETTER_SPACING_EM
        )

    def _compute_layout(self) -> types.SimpleNamespace:
        """Lays out the wordmark + version/build caption beneath the logo.

        Computed from live QFontMetrics rather than hard-coded pixel widths
        so the window sizes itself correctly regardless of the actual build
        string's length (e.g. a "_nightly" suffix) or platform font metrics.
        The logo/glow itself keeps the exact center point it had in the old
        fixed 320x320 window - only the window grows downward (and wider if
        the text needs it) to make room for the text below.
        """
        fm_bold = QtGui.QFontMetricsF(self._wordmark_bold_font)
        fm_light = QtGui.QFontMetricsF(self._wordmark_light_font)
        fm_build = QtGui.QFontMetricsF(self._build_font)

        wordmark_bold_text = "QATCH "
        wordmark_light_text = "TECHNOLOGIES"
        bold_width = fm_bold.horizontalAdvance(wordmark_bold_text)
        light_width = fm_light.horizontalAdvance(wordmark_light_text)
        wordmark_width = bold_width + light_width
        wordmark_height = max(fm_bold.height(), fm_light.height())

        version = Constants.app_version
        if version[:1].lower() == "v":
            version = version[1:]
        build_line1 = f"VERSION {version.upper()}"
        build_line2 = f"BUILD {Constants.app_date}"
        build_line1_width = fm_build.horizontalAdvance(build_line1)
        build_line2_width = fm_build.horizontalAdvance(build_line2)
        build_line_height = fm_build.height()

        content_width = max(_GLOW_SIZE, wordmark_width, build_line1_width, build_line2_width) + _SIDE_PADDING * 2
        width = max(_MIN_WINDOW_SIZE, content_width)
        cx = width / 2.0

        logo_cy = _TOP_CLEARANCE + _GLOW_SIZE / 2.0
        logo_box_bottom = logo_cy + _LOGO_SIZE / 2.0

        wordmark_top = logo_box_bottom + _WORDMARK_MARGIN_TOP
        wordmark_baseline_y = wordmark_top + fm_bold.ascent()
        wordmark_bottom = wordmark_top + wordmark_height

        build_top = wordmark_bottom + _BUILD_MARGIN_TOP
        build_line1_baseline_y = build_top + fm_build.ascent()
        build_line2_baseline_y = build_line1_baseline_y + build_line_height + _BUILD_LINE_GAP
        build_bottom = build_line2_baseline_y + fm_build.descent()

        height = build_bottom + _BOTTOM_PADDING

        return types.SimpleNamespace(
            width=width,
            height=height,
            cx=cx,
            logo_cy=logo_cy,
            wordmark_bold_x=cx - wordmark_width / 2.0,
            wordmark_light_x=cx - wordmark_width / 2.0 + bold_width,
            wordmark_baseline_y=wordmark_baseline_y,
            build_line1=build_line1,
            build_line1_x=cx - build_line1_width / 2.0,
            build_line1_baseline_y=build_line1_baseline_y,
            build_line2=build_line2,
            build_line2_x=cx - build_line2_width / 2.0,
            build_line2_baseline_y=build_line2_baseline_y,
        )

    def _on_tick(self) -> None:
        self._elapsed += _TICK_MS / 1000.0
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # Not a QSplashScreen, so nothing auto-hides on click - ignore
        # defensively anyway so a stray click can't do anything unexpected.
        event.ignore()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # Triggered gracefully when the main app calls splash_process.terminate();
        # we want an instant, clean close, so just let it pass.
        event.accept()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        cx = self._layout.cx
        logo_cy = self._layout.logo_cy

        self._paint_glow(painter, cx, logo_cy)
        self._paint_logo(painter, cx, logo_cy)
        self._paint_text(painter)

        painter.end()

    def _paint_text(self, painter: QtGui.QPainter) -> None:
        """Wordmark ("QATCH" bold + "TECHNOLOGIES" light) plus the smaller
        version/build caption beneath it, per the design's wordmarkStyle/
        -Strong/-Light triple and the previous plain-text splash's build
        info respectively."""
        layout = self._layout
        painter.setPen(QtCore.Qt.NoPen)

        bold_color = QtGui.QColor(_WORDMARK_COLOR_HEX)
        painter.setFont(self._wordmark_bold_font)
        painter.setPen(bold_color)
        painter.drawText(
            QtCore.QPointF(layout.wordmark_bold_x, layout.wordmark_baseline_y), "QATCH "
        )

        light_color = QtGui.QColor(_WORDMARK_COLOR_HEX)
        light_color.setAlphaF(_WORDMARK_LIGHT_OPACITY)
        painter.setFont(self._wordmark_light_font)
        painter.setPen(light_color)
        painter.drawText(
            QtCore.QPointF(layout.wordmark_light_x, layout.wordmark_baseline_y), "TECHNOLOGIES"
        )

        build_color = QtGui.QColor(_WORDMARK_COLOR_HEX)
        build_color.setAlphaF(_BUILD_OPACITY)
        painter.setFont(self._build_font)
        painter.setPen(build_color)
        painter.drawText(
            QtCore.QPointF(layout.build_line1_x, layout.build_line1_baseline_y), layout.build_line1
        )
        painter.drawText(
            QtCore.QPointF(layout.build_line2_x, layout.build_line2_baseline_y), layout.build_line2
        )

    def _paint_glow(self, painter: QtGui.QPainter, cx: float, cy: float) -> None:
        """qc-glow: a soft pulsing radial glow behind the logo.

        The CSS also applies `filter: blur(26px)` on top of the gradient;
        rather than render-to-pixmap-then-blur for a plain QWidget
        paintEvent, the gradient's own smooth falloff already reads as a
        soft glow at this scale, so the blur step is skipped.
        """
        progress = (self._elapsed % _GLOW_PERIOD_S) / _GLOW_PERIOD_S
        if progress <= 0.5:
            t = progress / 0.5
            scale = _lerp(0.9, 1.06, t)
            layer_opacity = _lerp(0.45, 0.8, t)
        else:
            t = (progress - 0.5) / 0.5
            scale = _lerp(1.06, 0.9, t)
            layer_opacity = _lerp(0.8, 0.45, t)

        radius = (_GLOW_SIZE / 2.0) * scale
        gradient = QtGui.QRadialGradient(cx, cy, radius)
        center = QtGui.QColor(_GLOW_COLOR)
        center.setAlphaF(max(0.0, min(1.0, 0.35 * layer_opacity)))
        edge = QtGui.QColor(_GLOW_COLOR)
        edge.setAlphaF(0.0)
        gradient.setColorAt(0.0, center)
        gradient.setColorAt(0.7, edge)
        gradient.setColorAt(1.0, edge)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(gradient))
        painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

    def _paint_logo(self, painter: QtGui.QPainter, cx: float, cy: float) -> None:
        scale = _LOGO_SIZE / _VIEWBOX
        offset_x = cx - _CIRCLE_CX * scale
        offset_y = cy - _CIRCLE_CY * scale

        circle_center = QtCore.QPointF(
            _CIRCLE_CX * scale + offset_x, _CIRCLE_CY * scale + offset_y
        )
        circle_radius = _CIRCLE_R * scale

        # Base circle: vertical linear gradient, unclipped (it defines its
        # own circular shape already).
        base_gradient = QtGui.QLinearGradient(
            circle_center.x(), circle_center.y() - circle_radius,
            circle_center.x(), circle_center.y() + circle_radius,
        )
        base_gradient.setColorAt(0.0, _GRADIENT_TOP)
        base_gradient.setColorAt(1.0, _GRADIENT_BOTTOM)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(base_gradient))
        painter.drawEllipse(circle_center, circle_radius, circle_radius)

        # Tiles: clipped to the same circle so ones near the rim don't
        # poke outside the circular silhouette (mirrors the SVG's own
        # <g clip-path="url(#qc-clip)"> group).
        painter.save()
        clip_path = QtGui.QPainterPath()
        clip_path.addEllipse(circle_center, circle_radius, circle_radius)
        painter.setClipPath(clip_path)

        tile_size = _TILE_SIZE * scale
        tile_radius = _TILE_RADIUS * scale
        for x, y, base_hex, row, col in _TILES:
            fill = self._shimmer_color(base_hex, row, col)
            rect = QtCore.QRectF(
                x * scale + offset_x, y * scale + offset_y, tile_size, tile_size
            )
            painter.setBrush(QtGui.QBrush(fill))
            painter.drawRoundedRect(rect, tile_radius, tile_radius)

        painter.restore()

    def _shimmer_color(self, base_hex: str, row: int, col: int) -> QtGui.QColor:
        """qc-shimmer: a brightness/saturation pulse that sweeps across the
        tile grid, each tile's pulse offset by `(row + col) * 0.13`s so the
        peak travels diagonally rather than flashing all tiles at once."""
        delay = (row + col) * _SHIMMER_DELAY_STEP_S
        t_local = self._elapsed - delay
        if t_local < 0:
            # Hasn't started its first cycle yet - the 0% keyframe is just
            # the base color (opacity 1, brightness 1), so render as-is.
            return QtGui.QColor(base_hex)

        progress = (t_local % _SHIMMER_PERIOD_S) / _SHIMMER_PERIOD_S
        # Envelope from the qc-shimmer keyframes: flat at base until 38%,
        # ramps up to the peak at 50%, back down to base by 62%, flat to 100%.
        if progress < 0.38:
            peak = 0.0
        elif progress < 0.50:
            peak = (progress - 0.38) / (0.50 - 0.38)
        elif progress < 0.62:
            peak = 1.0 - (progress - 0.50) / (0.62 - 0.50)
        else:
            peak = 0.0

        if peak <= 0.0:
            return QtGui.QColor(base_hex)

        brightness = _lerp(1.0, 1.5, peak)
        saturate = _lerp(1.0, 1.25, peak)
        return _apply_css_filters(QtGui.QColor(base_hex), brightness, saturate)


def main():
    app = QtWidgets.QApplication(sys.argv)
    splash = QatchSplashScreen()  # noqa: F841 (keeps a strong ref alive for app.exec_())
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
