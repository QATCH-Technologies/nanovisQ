"""
QATCH.ui.widgets.splash_screen_widget.py

Animated QATCH splash screen.

Displays the brand's mosaic-circle mark with a shimmering brightness sweep
across its tiles and a soft pulsing glow. The splash screen floats over a fully
transparent background (no backdrop, no separate loader). It includes the QATCH
wordmark and a smaller version/build caption.

Runs as its own subprocess, so its entire public surface is constructing
and showing it; the main process terminates the subprocess once the UI is ready.

Author(s):
    Paul MacNichol

Date:
    2026-08-21
"""

import sys
import types

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.core.constants import Constants
from QATCH.ui.styles.typography import make_qfont

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
_VIEWBOX = 273.0
_CIRCLE_CX, _CIRCLE_CY, _CIRCLE_R = 139.5, 139.0, 119.0
_GRADIENT_TOP = QtGui.QColor("#17aeef")
_GRADIENT_BOTTOM = QtGui.QColor("#0698e0")
_GLOW_COLOR = QtGui.QColor(10, 164, 229)

_SHIMMER_PERIOD_S = 3.2
_SHIMMER_DELAY_STEP_S = 0.13
_GLOW_PERIOD_S = 4.0
_TICK_MS = 33  # ~30fps

_LOGO_SIZE = 240.0  # rendered logo diameter
_GLOW_SIZE = 280.0
_MIN_WINDOW_SIZE = 320  # floor size
_TOP_CLEARANCE = (_MIN_WINDOW_SIZE - _GLOW_SIZE) / 2  # space above the glow's resting radius

# Wordmark ("QATCH", "Technologies")
_WORDMARK_COLOR_HEX = "#0d3d55"
_WORDMARK_FONT_PX = 25
_WORDMARK_LETTER_SPACING_EM = 0.14
_WORDMARK_MARGIN_TOP = 26
_WORDMARK_LIGHT_OPACITY = 0.85

# Version/build caption
_BUILD_FONT_PX = 11
_BUILD_LETTER_SPACING_EM = 0.10
_BUILD_MARGIN_TOP = 12
_BUILD_LINE_GAP = 3
_BUILD_OPACITY = 0.62

_SIDE_PADDING = 26  # horizontal clearance
_BOTTOM_PADDING = 22


def _lerp(a: float, b: float, t: float) -> float:
    """Linearly interpolates between two values.

    Args:
        a (float): The starting value.
        b (float): The ending value.
        t (float): The interpolation factor (typically between 0.0 and 1.0).

    Returns:
        float: The interpolated value.
    """
    return a + (b - a) * t


def _apply_css_filters(
    color: QtGui.QColor,
    brightness: float,
    saturate: float,
) -> QtGui.QColor:
    """Reproduces CSS `filter: brightness(b) saturate(s)` applied in that order.

    Matches qc-shimmer's `brightness(1.5) saturate(1.25)` peak.
    brightness() is a per-channel linear scale; saturate() is the standard
    luminance-preserving saturation matrix per the CSS Filter Effects spec.

    Args:
        color (QtGui.QColor): The base color to manipulate.
        brightness (float): The brightness scale factor.
        saturate (float): The saturation scale factor.

    Returns:
        QtGui.QColor: The new color with the simulated CSS filters applied.
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
    """Animated, transparent-backdrop QATCH splash.

    Features a shimmering mosaic-circle logo with a soft pulsing glow, the QATCH
    Technologies wordmark, and a small version/build caption with no separate loader chrome.
    """

    def __init__(self) -> None:
        """Initializes the QatchSplashScreen instance.

        Uses FramelessWindowHint, WindowStaysOnTopHint, and Tool window flags to ensure
        reliable translucency on Windows while keeping the splash screen out of the taskbar.
        Starts a timer at 33ms intervals for ~30fps animations.
        """
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
        """Centers the splash screen window on the primary screen.

        If the primary screen cannot be retrieved, defaults to centering on a 1920x1080 boundary.
        """
        screen = QtWidgets.QApplication.primaryScreen()
        geo = screen.geometry() if screen is not None else QtCore.QRect(0, 0, 1920, 1080)
        self.move(
            geo.x() + (geo.width() - self.width()) // 2,
            geo.y() + (geo.height() - self.height()) // 2,
        )

    def _build_fonts(self) -> None:
        """Constructs and configures the QFont instances for the splash screen.

        Sets up the bold and light variants of the wordmark font to sit on the same baseline
        with matching metrics, and configures the smaller version/build font.
        """
        self._wordmark_bold_font = make_qfont(pixel_size=_WORDMARK_FONT_PX, weight=QtGui.QFont.Bold)
        self._wordmark_bold_font.setLetterSpacing(
            QtGui.QFont.AbsoluteSpacing, _WORDMARK_FONT_PX * _WORDMARK_LETTER_SPACING_EM
        )
        self._wordmark_light_font = QtGui.QFont(self._wordmark_bold_font)
        self._wordmark_light_font.setWeight(QtGui.QFont.Light)

        self._build_font = make_qfont(pixel_size=_BUILD_FONT_PX)
        self._build_font.setLetterSpacing(
            QtGui.QFont.AbsoluteSpacing, _BUILD_FONT_PX * _BUILD_LETTER_SPACING_EM
        )

    def _compute_layout(self) -> types.SimpleNamespace:
        """Lays out the wordmark + version/build caption beneath the logo.

        Computed from live QFontMetrics rather than hard-coded pixel widths so the window
        sizes itself correctly regardless of the actual build string's length or platform
        font metrics.

        Returns:
            types.SimpleNamespace: A namespace containing layout coordinates, dimensions,
                and calculated text positions.
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

        content_width = (
            max(_GLOW_SIZE, wordmark_width, build_line1_width, build_line2_width)
            + _SIDE_PADDING * 2
        )
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
        """Handles the animation timer tick.

        Increments the elapsed time and triggers a widget update to advance the animation frame.
        """
        self._elapsed += _TICK_MS / 1000.0
        self.update()

    def mousePressEvent(self, event) -> None:
        """Ignores mouse press events defensively.

        Ensures that stray clicks do not auto-hide the splash screen or cause unexpected behavior.

        Args:
            event (QtGui.QMouseEvent): The mouse press event payload.
        """
        event.ignore()

    def closeEvent(self, event) -> None:
        """Handles the window close event cleanly.

        Accepts the event to allow for an instant, clean close when the main app process
        calls `splash_process.terminate()`.

        Args:
            event (QtGui.QCloseEvent): The close event payload.
        """
        event.accept()

    def paintEvent(self, event) -> None:
        """Main rendering routine for the splash screen.

        Enables antialiasing and delegates rendering to specific paint methods for the
        glow, logo, and text.

        Args:
            event (QtGui.QPaintEvent): The paint event payload.
        """
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        cx = self._layout.cx
        logo_cy = self._layout.logo_cy

        self._paint_glow(painter, cx, logo_cy)
        self._paint_logo(painter, cx, logo_cy)
        self._paint_text(painter)

        painter.end()

    def _paint_text(self, painter: QtGui.QPainter) -> None:
        """Paints the wordmark and version/build caption.

        Draws the "QATCH" bold text, "TECHNOLOGIES" light text, and the restyled version/build
        caption beneath the logo.

        Args:
            painter (QtGui.QPainter): The active painter instance used for drawing.
        """
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
        """Paints qc-glow: a soft pulsing radial glow behind the logo.

        Simulates a blur using the gradient's own smooth falloff. Dynamically scales
        opacity and radius based on animation progress.

        Args:
            painter (QtGui.QPainter): The active painter instance used for drawing.
            cx (float): The center X coordinate for the radial gradient.
            cy (float): The center Y coordinate for the radial gradient.
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
        """Paints the QATCH mosaic-circle logo.

        Draws the base circular gradient background and clips the animated mosaic tiles
        so they do not poke outside the circular silhouette.

        Args:
            painter (QtGui.QPainter): The active painter instance used for drawing.
            cx (float): The scaled X offset for the logo's center.
            cy (float): The scaled Y offset for the logo's center.
        """
        scale = _LOGO_SIZE / _VIEWBOX
        offset_x = cx - _CIRCLE_CX * scale
        offset_y = cy - _CIRCLE_CY * scale

        circle_center = QtCore.QPointF(_CIRCLE_CX * scale + offset_x, _CIRCLE_CY * scale + offset_y)
        circle_radius = _CIRCLE_R * scale

        # Base circle
        base_gradient = QtGui.QLinearGradient(
            circle_center.x(),
            circle_center.y() - circle_radius,
            circle_center.x(),
            circle_center.y() + circle_radius,
        )
        base_gradient.setColorAt(0.0, _GRADIENT_TOP)
        base_gradient.setColorAt(1.0, _GRADIENT_BOTTOM)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(base_gradient))
        painter.drawEllipse(circle_center, circle_radius, circle_radius)

        # Tiles
        painter.save()
        clip_path = QtGui.QPainterPath()
        clip_path.addEllipse(circle_center, circle_radius, circle_radius)
        painter.setClipPath(clip_path)

        tile_size = _TILE_SIZE * scale
        tile_radius = _TILE_RADIUS * scale
        for x, y, base_hex, row, col in _TILES:
            fill = self._shimmer_color(base_hex, row, col)
            rect = QtCore.QRectF(x * scale + offset_x, y * scale + offset_y, tile_size, tile_size)
            painter.setBrush(QtGui.QBrush(fill))
            painter.drawRoundedRect(rect, tile_radius, tile_radius)

        painter.restore()

    def _shimmer_color(self, base_hex: str, row: int, col: int) -> QtGui.QColor:
        """Calculates the animated shimmer color for a specific tile.

        Applies a brightness/saturation pulse that sweeps across the tile grid, using an offset
        delay to create a diagonal travel effect rather than flashing all tiles at once.

        Args:
            base_hex (str): The starting hex color of the tile.
            row (int): The row index of the tile.
            col (int): The column index of the tile.

        Returns:
            QtGui.QColor: The calculated, filtered color for the current animation frame.
        """
        delay = (row + col) * _SHIMMER_DELAY_STEP_S
        t_local = self._elapsed - delay
        if t_local < 0:
            return QtGui.QColor(base_hex)

        progress = (t_local % _SHIMMER_PERIOD_S) / _SHIMMER_PERIOD_S
        # Envelope from the qc-shimmer keyframes
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
    """Entry point for testing the splash screen independently.

    Initializes the QApplication, constructs the QatchSplashScreen, and executes the event loop.
    """
    app = QtWidgets.QApplication(sys.argv)
    splash = QatchSplashScreen()  # noqa: F841
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
