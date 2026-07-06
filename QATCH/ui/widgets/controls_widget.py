from typing import Optional
from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.styles.theme_manager import ThemeManager


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
        ThemeManager.instance().themeChanged.connect(lambda _: self.update())

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Paints the flat card surface (fill + border) behind the toolbar.

        Args:
            event (QtGui.QPaintEvent): The paint event provided by the system.
        """
        tok = ThemeManager.instance().tokens()
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        paint_flat_surface(
            self,
            radius=self._RADIUS,
            fill=QtGui.QColor(*tok["surface"]),
            border=QtGui.QColor(*tok["surface_border"]),
            painter=p,
        )
        p.end()
