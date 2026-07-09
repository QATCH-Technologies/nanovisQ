"""QATCH.ui.components.plot_status_banner

Provides PlotStatusBanner: a compact pill-shaped status chip that floats
in the header strip of a pyqtgraph PlotItem - horizontally centred between
the left and right axis title labels, vertically centred in the header row.

Color → state mapping (backwards-compatible with legacy pg.LabelItem.setText
color tuples used in main_window.py):
    (0, 200, 0)   → "success"  - Apply drop now
    (0, 0, 200)   → "warning"  - Sensor not dry / restart
    (200, 100, 0) → "info"     - Fill-state classifier message
    anything else → "neutral"
"""

import os

from PyQt5 import QtCore, QtGui, QtWidgets


class PlotStatusBanner:
    """Pill-shaped, icon-driven status chip for a pyqtgraph PlotItem header.

    The chip is a `QGraphicsProxyWidget` wrapping a styled `QFrame` that
    floats over the PlotItem's title row - the narrow strip above the
    ViewBox that shows the axis labels.  It tracks plot resizes via
    `ViewBox.sigResized` and re-centres itself automatically.

    Args:
        plot_item: The `pg.PlotItem` to anchor the banner to.
        icon_dir:  Path to the QATCH icons folder (must contain
                   `warning-circle.svg`, `checkmark-circle.svg`,
                   `info-circle.svg`).
        z_value:   Scene Z-order for the proxy (default 150, above labels).
    """

    # Fallback geometry when the widget hasn't laid out yet
    _FALLBACK_W = 270
    _FALLBACK_H = 22

    _COLOR_TO_STATE: dict[tuple, str] = {
        (0, 200, 0): "success",
        (0, 0, 200): "warning",
        (200, 100, 0): "info",
    }

    _THEMES: dict[str, dict] = {
        "warning": {
            "icon_svg": "warning-circle.svg",
            "icon_hex": "#dc2626",
            "bg": "rgba(239, 68, 68, 0.10)",
            "border": "rgba(220, 38, 38, 0.65)",
            "text_color": "#991b1b",
        },
        "success": {
            "icon_svg": "checkmark-circle.svg",
            "icon_hex": "#16a34a",
            "bg": "rgba(34, 197, 94, 0.10)",
            "border": "rgba(22, 163, 74, 0.65)",
            "text_color": "#14532d",
        },
        "info": {
            "icon_svg": "info-circle.svg",
            "icon_hex": "#d97706",
            "bg": "rgba(245, 158, 11, 0.10)",
            "border": "rgba(217, 119, 6, 0.65)",
            "text_color": "#92400e",
        },
        "neutral": {
            "icon_svg": "info-circle.svg",
            "icon_hex": "#64748b",
            "bg": "rgba(148, 163, 184, 0.10)",
            "border": "rgba(100, 116, 139, 0.65)",
            "text_color": "#334155",
        },
    }

    def __init__(self, plot_item, icon_dir: str, z_value: int = 150) -> None:
        self._plot_item = plot_item
        self._icon_dir = icon_dir
        self._resize_cb = None

        # ── Pill frame ────────────────────────────────────────────────
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
        # Prevent the label from triggering unwanted size expansion
        self._text_lbl.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        lay.addWidget(self._icon_lbl)
        lay.addWidget(self._text_lbl)

        # Size it once so boundingRect works before the first paint
        self._pill.adjustSize()

        # ── Proxy (free-floating, parented to plot's graphicsItem) ────
        self._proxy = QtWidgets.QGraphicsProxyWidget()
        self._proxy.setWidget(self._pill)
        self._proxy.setParentItem(plot_item.graphicsItem())
        self._proxy.setZValue(z_value)
        self._proxy.setVisible(False)

        # ── Track plot resizes to recentre the chip ───────────────────
        self._resize_cb = self._make_reposition_cb(plot_item, self._proxy)
        plot_item.getViewBox().sigResized.connect(self._resize_cb)

    # ── Public API ────────────────────────────────────────────────────

    def set_state(self, state: str, text: str) -> None:
        """Show the banner with *state* theme and *text*.

        Passing an empty or whitespace-only string hides the banner.
        """
        if not text or not text.strip():
            self._proxy.setVisible(False)
            return

        theme = self._THEMES.get(state, self._THEMES["neutral"])

        # Icon
        icon_path = os.path.join(self._icon_dir, theme["icon_svg"])
        if os.path.exists(icon_path):
            px = _tinted_pixmap(icon_path, QtGui.QColor(theme["icon_hex"]), 13)
            self._icon_lbl.setPixmap(px)
            self._icon_lbl.setVisible(True)
        else:
            self._icon_lbl.setVisible(False)

        # Text label
        self._text_lbl.setText(text)
        self._text_lbl.setStyleSheet(
            f"QLabel {{"
            f"  color: {theme['text_color']};"
            f"  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;"
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

    def hide(self) -> None:
        """Hide the banner without destroying it."""
        self._proxy.setVisible(False)

    def setText(self, text: str, color: tuple = None) -> None:
        """Backwards-compatible shim matching `pg.LabelItem.setText`.

        Maps the *color* tuple to one of the named visual states and
        delegates to :meth:`set_state`.
        """
        if not text or not text.strip():
            self.hide()
            return
        state = self._COLOR_TO_STATE.get(tuple(int(c) for c in color) if color else (), "neutral")
        self.set_state(state, text)

    def remove(self) -> None:
        """Detach the proxy from the scene and disconnect the resize signal."""
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

    # ── Helpers ───────────────────────────────────────────────────────

    def _make_reposition_cb(self, plot_item, proxy):
        """Return a zero-arg callback that centres the proxy in the header strip."""
        fw, fh = self._FALLBACK_W, self._FALLBACK_H

        def _reposition(*_args) -> None:
            try:
                vb = plot_item.getViewBox()
                # ViewBox rect in the PlotItem's own coordinate space.
                # vb_rect.y() is the height of the header strip above the ViewBox.
                vb_rect = vb.mapRectToItem(plot_item.graphicsItem(), vb.boundingRect())
                pw = proxy.boundingRect().width() or fw
                ph = proxy.boundingRect().height() or fh

                header_h = vb_rect.y()  # pixels above the ViewBox

                # Horizontal: centred over the ViewBox (between the two axis labels)
                banner_x = vb_rect.x() + (vb_rect.width() - pw) / 2.0

                # Vertical: centred within the header strip; clamp to ≥ 1 px from top
                banner_y = max(1.0, (header_h - ph) / 2.0)

                proxy.setPos(banner_x, banner_y)
            except Exception:
                pass

        return _reposition


# ── Module-level pixmap helper (used by PlotStatusBanner) ────────────────────


def _tinted_pixmap(svg_path: str, color: QtGui.QColor, size: int) -> QtGui.QPixmap:
    """Return a *size*×*size* pixmap loaded from *svg_path* recoloured to *color*."""
    src = QtGui.QIcon(svg_path).pixmap(size, size)
    dst = QtGui.QPixmap(src.size())
    dst.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(dst)
    painter.drawPixmap(0, 0, src)
    painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceAtop)
    painter.fillRect(dst.rect(), color)
    painter.end()
    return dst
