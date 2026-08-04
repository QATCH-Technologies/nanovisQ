"""QATCH.ui.components.plot_status_banner.py

Provides :class:`PlotStatusBanner`: a compact pill-shaped status chip that floats
in the header strip of a :class:`pyqtgraph.PlotItem` - horizontally centred between
the left and right axis title labels, vertically centred in the header row.

Color-to-state mapping (backwards-compatible with legacy :meth:`pyqtgraph.LabelItem.setText`
color tuples used in `main_window.py`):

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-04
"""

import os
from typing import ClassVar

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.logger import Logger as Log
from QATCH.ui.styles.theme_manager import ThemeManager
from QATCH.ui.styles.typography import FONT_SANS_STACK

TAG = "[PlotStatusBanner]"


class PlotStatusBanner:
    """Pill-shaped, icon-driven status chip for a :class:`pyqtgraph.PlotItem` header.

    The chip is a :class:`QtWidgets.QGraphicsProxyWidget` wrapping a styled :class:`QtWidgets.QFrame`
    that floats over the :class:`pyqtgraph.PlotItem`'s title row - the narrow strip above the
    :class:`pyqtgraph.ViewBox` that shows the axis labels. It tracks plot resizes via
    :attr:`pyqtgraph.ViewBox.sigResized` and re-centers itself automatically.

    Args:
        plot_item (pyqtgraph.PlotItem): The plot item to anchor the banner to.
        icon_dir (str): Path to the QATCH icons folder (must contain `warning-circle.svg`,
            `checkmark-circle.svg`, and `info-circle.svg`).
        z_value (int, optional): Scene Z-order for the proxy widget (above labels).
            Defaults to `150`.

    Attributes:
        _FALLBACK_W (int): Fallback widget width used when layout bounding rect is uninitialized.
        _FALLBACK_H (int): Fallback widget height used when layout bounding rect is uninitialized.
        _COLOR_TO_STATE (dict[tuple, str]): Maps legacy RGB color tuples to theme state strings.
        _STATE_ICON (dict[str, str]): Maps state names to SVG icon filenames.
        _STATE_TOKEN (dict[str, str]): Maps state names to theme token names.
        _ICON_PIXMAP_CACHE (dict[tuple, QtGui.QPixmap]): Cache mapping `(icon_dir, state, mode)`
            tuples to pre-tinted pixmaps.
    """

    _FALLBACK_W = 270
    _FALLBACK_H = 22

    _COLOR_TO_STATE: ClassVar[dict[tuple, str]] = {
        (0, 200, 0): "success",
        (0, 0, 200): "warning",
        (200, 100, 0): "info",
    }

    _STATE_ICON: ClassVar[dict[str, str]] = {
        "warning": "warning-circle.svg",
        "success": "checkmark-circle.svg",
        "info": "info-circle.svg",
        "neutral": "info-circle.svg",
    }
    _STATE_TOKEN: ClassVar[dict[str, str]] = {
        "warning": "danger",
        "success": "success",
        "info": "warning",
        "neutral": "plot_text_muted",
    }

    _ICON_PIXMAP_CACHE: ClassVar[dict[tuple, "QtGui.QPixmap"]] = {}

    def __init__(self, plot_item, icon_dir: str, z_value: int = 150) -> None:
        """Initializes a new PlotStatusBanner instance.

        Creates the underlying pill container (:class:`QtWidgets.QFrame`), configures its layout,
        wraps it inside a :class:`QtWidgets.QGraphicsProxyWidget` attached to the target
        :class:`pyqtgraph.PlotItem`, and connects resize signals to handle automatic centering.

        Args:
            plot_item (pyqtgraph.PlotItem): The parent plot item to anchor the banner to.
            icon_dir (str): Path to the icons directory (must contain `warning-circle.svg`,
                `checkmark-circle.svg`, and `info-circle.svg`).
            z_value (int, optional): Scene Z-order layer for the graphics proxy widget.
                Defaults to `150`.
        """
        self._plot_item = plot_item
        self._icon_dir = icon_dir
        self._resize_cb = None
        self._current_state: str | None = None
        self._current_text: str | None = None
        self._current_mode: str | None = None
        self._pill = QtWidgets.QFrame()
        self._pill.setObjectName("PlotStatusBannerPill")

        lay = QtWidgets.QHBoxLayout(self._pill)
        lay.setContentsMargins(10, 3, 13, 3)
        lay.setSpacing(6)

        self._icon_lbl = QtWidgets.QLabel()
        self._icon_lbl.setFixedSize(13, 13)
        self._icon_lbl.setScaledContents(True)

        self._text_lbl = QtWidgets.QLabel()
        self._text_lbl.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self._text_lbl.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        lay.addWidget(self._icon_lbl)
        lay.addWidget(self._text_lbl)
        self._pill.adjustSize()
        self._proxy = QtWidgets.QGraphicsProxyWidget()
        self._proxy.setWidget(self._pill)
        self._proxy.setParentItem(plot_item.graphicsItem())
        self._proxy.setZValue(z_value)
        self._proxy.setVisible(False)

        self._resize_cb = self._make_reposition_cb(plot_item, self._proxy)
        plot_item.getViewBox().sigResized.connect(self._resize_cb)

    def set_state(self, state: str, text: str) -> None:
        """Show the banner with the specified state theme and text.

        Passing an empty or whitespace-only string hides the banner widget.
        Repeated calls with identical parameters and active theme mode skip
        redundant rendering updates.

        Args:
            state (str): Visual state name (e.g., `"warning"`, `"success"`,
                `"info"`, `"neutral"`).
            text (str): Status text message to display on the banner label.
        """
        if not text or not text.strip():
            self._proxy.setVisible(False)
            self._current_state = None
            self._current_text = None
            self._current_mode = None
            return

        mode = ThemeManager.instance().mode().value
        if (
            state == self._current_state
            and text == self._current_text
            and mode == self._current_mode
            and self._proxy.isVisible()
        ):
            return

        theme = self._state_theme(state, mode)
        restyle = state != self._current_state or mode != self._current_mode
        if restyle:
            cache_key = (self._icon_dir, state, mode)
            px = self._ICON_PIXMAP_CACHE.get(cache_key)
            if px is None:
                icon_path = os.path.join(self._icon_dir, theme["icon_svg"])
                if os.path.exists(icon_path):
                    px = _tinted_pixmap(icon_path, QtGui.QColor(theme["icon_hex"]), 13)
                    self._ICON_PIXMAP_CACHE[cache_key] = px
            if px is not None:
                self._icon_lbl.setPixmap(px)
                self._icon_lbl.setVisible(True)
            else:
                self._icon_lbl.setVisible(False)
        self._text_lbl.setText(text)
        if restyle:
            self._text_lbl.setStyleSheet(
                f"QLabel {{"
                f"  color: {theme['text_color']};"
                f"  font-family: {FONT_SANS_STACK};"
                f"  font-size: 9pt;"
                f"  font-weight: 500;"
                f"  background: transparent;"
                f"}}"
            )

            # Pill shell
            self._pill.setStyleSheet(
                f"QFrame#PlotStatusBannerPill {{"
                f"  background-color: {theme['bg']};"
                f"  border: 1.5px solid {theme['border']};"
                f"  border-radius: 10px;"
                f"}}"
            )

        self._pill.adjustSize()
        self._proxy.adjustSize()
        self._proxy.setVisible(True)
        if self._resize_cb:
            self._resize_cb()

        self._current_state = state
        self._current_text = text
        self._current_mode = mode

    @classmethod
    def _state_theme(cls, state: str, mode: str) -> dict:
        """Derives a state's icon, background, border, and text colors from active theme tokens.

        Allows the banner to adjust correctly in both light and dark theme modes rather
        than using a single fixed color palette.

        Args:
            state (str): One of `"warning"`, `"success"`, `"info"`, or `"neutral"`.
            mode (str): The active :class:`~QATCH.ui.styles.theme_manager.ThemeMode` value
                (`"light"` or `"dark"`).

        Returns:
            dict: Dictionary containing theme specs:
                * `"icon_svg"` (str): Filename of the SVG icon.
                * `"icon_hex"` (str): Hex color code for icon tinting.
                * `"bg"` (str): CSS `rgba()` background color.
                * `"border"` (str): CSS `rgba()` border color.
                * `"text_color"` (str): Hex color code for the text.
        """
        tok = ThemeManager.instance().tokens()
        icon_svg = cls._STATE_ICON.get(state, cls._STATE_ICON["neutral"])
        token_key = cls._STATE_TOKEN.get(state, cls._STATE_TOKEN["neutral"])
        r, g, b = tok[token_key][:3]
        dark = mode == "dark"

        icon_hex = f"#{r:02x}{g:02x}{b:02x}"
        bg = f"rgba({r}, {g}, {b}, {0.16 if dark else 0.10})"
        border = f"rgba({r}, {g}, {b}, {0.55 if dark else 0.65})"
        tr, tg, tb = _shade((r, g, b), 0.4 if dark else -0.35)
        text_color = f"#{tr:02x}{tg:02x}{tb:02x}"

        return {
            "icon_svg": icon_svg,
            "icon_hex": icon_hex,
            "bg": bg,
            "border": border,
            "text_color": text_color,
        }

    def hide(self) -> None:
        """Hide the banner graphics proxy widget without destroying it."""
        self._proxy.setVisible(False)

    def setText(self, text: str, color: tuple | None = None) -> None:
        """Backwards-compatible shim matching :meth:`pyqtgraph.LabelItem.setText`.

        Maps an RGB color tuple to one of the named visual states and delegates
        rendering to :meth:`set_state`.

        Args:
            text (str): Status string to render.
            color (tuple, optional): An `(R, G, B)` color tuple matching legacy label
                formatting. Defaults to `None`.
        """
        if not text or not text.strip():
            self.hide()
            return
        state = self._COLOR_TO_STATE.get(tuple(int(c) for c in color) if color else (), "neutral")
        self.set_state(state, text)

    def remove(self) -> None:
        """Detach the proxy widget from the scene and disconnect resize signals."""
        if self._resize_cb is not None:
            try:
                self._plot_item.getViewBox().sigResized.disconnect(self._resize_cb)
            except (RuntimeError, TypeError):
                pass
            self._resize_cb = None

        try:
            self._proxy.setParentItem(None)
            scene = self._proxy.scene()
            if scene is not None:
                scene.removeItem(self._proxy)
        except RuntimeError:
            pass

    def _make_reposition_cb(self, plot_item, proxy):
        """Construct a callback that centers the proxy widget within the plot header strip.

        Args:
            plot_item (pyqtgraph.PlotItem): Target plot container.
            proxy (QtWidgets.QGraphicsProxyWidget): Floating widget proxy to align.

        Returns:
            Callable[..., None]: Zero-argument repositioning callback function.
        """
        fw, fh = self._FALLBACK_W, self._FALLBACK_H

        def _reposition(*_args) -> None:
            try:
                vb = plot_item.getViewBox()
                vb_rect = vb.mapRectToItem(plot_item.graphicsItem(), vb.boundingRect())
                pw = proxy.boundingRect().width() or fw
                ph = proxy.boundingRect().height() or fh

                header_h = vb_rect.y()  # pixels above the ViewBox
                banner_x = vb_rect.x() + (vb_rect.width() - pw) / 2.0
                banner_y = max(1.0, (header_h - ph) / 2.0)
                proxy.setPos(banner_x, banner_y)

            except Exception as e:  # noqa: BLE001
                Log.w(TAG, f"Error repositioning banner: {e}")

        return _reposition


def _shade(rgb: tuple, amt: float) -> tuple:
    """Lightens or darkens an RGB color tuple.

    Args:
        rgb (tuple[int, int, int]): Original `(R, G, B)` color tuple.
        amt (float): Factor in range `[-1.0, 1.0]`. Positive values lighten toward white;
            negative values darken toward black; `0` returns `rgb` unchanged.

    Returns:
        tuple[int, int, int]: The shaded `(R, G, B)` color tuple.
    """
    r, g, b = rgb
    if amt >= 0:
        return (
            int(r + (255 - r) * amt),
            int(g + (255 - g) * amt),
            int(b + (255 - b) * amt),
        )
    return (int(r * (1 + amt)), int(g * (1 + amt)), int(b * (1 + amt)))


def _tinted_pixmap(svg_path: str, color: QtGui.QColor, size: int) -> QtGui.QPixmap:
    """Load an SVG icon and render it recolored to a specified color.

    Args:
        svg_path (str): Path to the target SVG asset.
        color (QtGui.QColor): Target color used to fill non-transparent icon regions.
        size (int): Width and height in pixels for the generated square pixmap.

    Returns:
        QtGui.QPixmap: The tinted pixmap image.
    """
    src = QtGui.QIcon(svg_path).pixmap(size, size)
    dst = QtGui.QPixmap(src.size())
    dst.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(dst)
    painter.drawPixmap(0, 0, src)
    painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceAtop)
    painter.fillRect(dst.rect(), color)
    painter.end()
    return dst
