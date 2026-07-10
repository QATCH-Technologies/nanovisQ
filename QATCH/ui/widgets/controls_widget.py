from typing import Optional
from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.styles.theme_manager import ThemeManager, ThemeMode


class ControlsWidget(QtWidgets.QWidget):
    """Container that provides the toolbar's card backdrop.

    Painted as a flat card (see QATCH.ui.components.flat_paint) using the
    same `surface`/`surface_border` tokens as the Mode sidebar and Log
    console, so the toolbar reads as part of the same panel family instead
    of the old frosted-glass look.

    Attributes:
        _RADIUS (float): The corner radius applied to the container's
            card geometry.
    """

    _RADIUS: float = 12.0

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initializes the ControlsWidget with transparency-aware attributes.

        Configures the widget to ignore system backgrounds, ensuring that
        the custom `paintEvent` (if implemented) or style attributes render
        correctly on top of other elements or backdrops.

        Args:
            parent (Optional[QtWidgets.QWidget]): The parent widget of this
                container. Defaults to None.
        """
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._bg_cache: Optional[QtGui.QPixmap] = None
        self._bg_cache_mode: Optional[ThemeMode] = None
        ThemeManager.instance().themeChanged.connect(lambda _: self.update())

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Paints the flat card surface (fill + border) behind the toolbar.

        Every live readout on the toolbar (TEC label, progress bar,
        Start/Stop button) sits on top of this translucent card and updates
        on the ~100ms plot tick during a run, forcing Qt to repaint this
        background each time. The fill/border only actually changes on
        resize or theme switch, so it's rendered once into an offscreen
        pixmap here and blitted on every other call - pixel-identical
        output, without re-running the antialiased draw calls on every tick.

        Args:
            event (QtGui.QPaintEvent): The paint event provided by the system.
        """
        mode = ThemeManager.instance().mode()
        size = self.size()
        if (
            self._bg_cache is None
            or self._bg_cache.size() != size
            or self._bg_cache_mode != mode
        ):
            self._bg_cache = self._render_background(size)
            self._bg_cache_mode = mode

        p = QtGui.QPainter(self)
        p.drawPixmap(0, 0, self._bg_cache)
        p.end()

    def _render_background(self, size: QtCore.QSize) -> QtGui.QPixmap:
        """Renders the flat card surface to an offscreen ARGB pixmap for
        `paintEvent` to cache and blit.

        Args:
            size (QtCore.QSize): Pixmap size to render at (the widget's
                current size).

        Returns:
            QtGui.QPixmap: The rendered background.
        """
        tok = ThemeManager.instance().tokens()
        pm = QtGui.QPixmap(size)
        pm.fill(QtCore.Qt.GlobalColor.transparent)

        p = QtGui.QPainter(pm)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        paint_flat_surface(
            self,
            radius=self._RADIUS,
            fill=QtGui.QColor(*tok["surface"]),
            border=QtGui.QColor(*tok["surface_border"]),
            painter=p,
        )
        p.end()
        return pm
