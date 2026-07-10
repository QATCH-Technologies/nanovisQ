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

from QATCH.ui.styles.theme_manager import ThemeManager


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

    # Which SVG and which theme token supplies each state's base hue. Colors
    # are derived from the active theme (see `_state_theme`) rather than a
    # fixed palette, so the banner reads correctly in both light and dark
    # mode instead of always rendering its original light-mode design.
    _STATE_ICON = {
        "warning": "warning-circle.svg",
        "success": "checkmark-circle.svg",
        "info": "info-circle.svg",
        "neutral": "info-circle.svg",
    }
    _STATE_TOKEN = {
        "warning": "danger",
        "success": "success",
        "info": "warning",
        "neutral": "plot_text_muted",
    }

    # Tinted status-icon pixmaps, keyed by (icon_dir, state). The themed
    # icons are a pure function of those two values, and this banner's
    # `setText`/`set_state` is called on every ~100ms plot tick for most of
    # a run's duration, so caching avoids re-reading and re-tinting the SVG
    # from disk on every tick.
    _ICON_PIXMAP_CACHE: dict[tuple, "QtGui.QPixmap"] = {}

    def __init__(self, plot_item, icon_dir: str, z_value: int = 150) -> None:
        self._plot_item = plot_item
        self._icon_dir = icon_dir
        self._resize_cb = None
        self._current_state: str | None = None
        self._current_text: str | None = None
        self._current_mode: str | None = None

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
            self._current_state = None
            self._current_text = None
            self._current_mode = None
            return

        mode = ThemeManager.instance().mode().value

        # This is invoked every plot tick (~10x/sec) for most of a run's
        # duration, almost always re-showing the same state/text as last
        # tick. Skip the icon reload, stylesheet re-parse, and relayout
        # entirely when nothing actually changed since the last call - a
        # light/dark switch counts as a change too, since the derived
        # colors depend on it.
        if (
            state == self._current_state
            and text == self._current_text
            and mode == self._current_mode
            and self._proxy.isVisible()
        ):
            return

        theme = self._state_theme(state, mode)
        restyle = state != self._current_state or mode != self._current_mode

        # Icon (cached per icon_dir+state+mode so the SVG is only read/tinted once)
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

        # Text label
        self._text_lbl.setText(text)
        if restyle:
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

        self._current_state = state
        self._current_text = text
        self._current_mode = mode

    @classmethod
    def _state_theme(cls, state: str, mode: str) -> dict:
        """Derives this state's icon/bg/border/text colors from the active
        theme's semantic tokens, so the banner reads correctly in both
        light and dark mode instead of a single fixed light-mode palette.

        Args:
            state: One of "warning", "success", "info", "neutral".
            mode: The active `ThemeMode` value ("light" or "dark").

        Returns:
            dict: icon_svg, icon_hex, bg, border, text_color.
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


# ── Module-level helpers (used by PlotStatusBanner) ──────────────────────────


def _shade(rgb: tuple, amt: float) -> tuple:
    """Lightens (amt > 0, toward white) or darkens (amt < 0, toward black)
    an (r, g, b) tuple. `amt` is roughly in [-1, 1]; 0 returns `rgb` as-is.
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
