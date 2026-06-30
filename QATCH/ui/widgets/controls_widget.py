from typing import Optional
from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.styles.theme_manager import ThemeManager


class ControlsWidget(QtWidgets.QWidget):
    """Container that provides the toolbar's gradient backdrop.

    Renders the same cool-blue gradient palette used by GlassCard in
    ui_login when no live backdrop is available, overlaid with the standard
    white-tint, shimmer, and dual-border glass language.

    Attributes:
        _RADIUS (float): The corner radius applied to the container's
            glass geometry.
    """

    _RADIUS: float = 10.0

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
        """Performs custom painting to render the frosted-glass toolbar background.

        The painting sequence applies a frosted-white base with a soft blue
        tint, a three-stop shimmer gradient, a bottom vignette for grounding,
        and double-stroke borders to provide a clean glass-like edge.

        Args:
            event (QtGui.QPaintEvent): The paint event provided by the system.
        """
        tok = ThemeManager.instance().tokens()
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        rect_f = QtCore.QRectF(self.rect())

        # Clip to rounded rectangle
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)

        # Base
        p.fillRect(self.rect(), QtGui.QColor(*tok["plot_glass_base"]))
        p.fillRect(self.rect(), QtGui.QColor(*tok["plot_glass_overlay"]))

        # Top shimmer
        shimmer = QtGui.QLinearGradient(0, 0, 0, 40)
        shimmer.setColorAt(0.0, QtGui.QColor(*tok["plot_glass_shimmer_top"]))
        shimmer.setColorAt(0.5, QtGui.QColor(*tok["plot_glass_shimmer_mid"]))
        shimmer.setColorAt(1.0, QtGui.QColor(*tok["plot_glass_shimmer_mid"][:3], 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Bottom vignette
        vg = QtGui.QLinearGradient(0, self.height() - 30, 0, self.height())
        vg.setColorAt(0.0, QtGui.QColor(*tok["plot_glass_vignette_end"][:3], 0))
        vg.setColorAt(1.0, QtGui.QColor(*tok["plot_glass_vignette_end"]))
        p.fillRect(self.rect(), QtGui.QBrush(vg))

        # Borders
        p.setClipping(False)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(*tok["plot_glass_rim"]), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)
        p.setPen(QtGui.QPen(QtGui.QColor(*tok["plot_glass_inset"]), 1.0))
        p.drawRoundedRect(
            rect_f.adjusted(1.5, 1.5, -1.5, -1.5),
            self._RADIUS - 1.5,
            self._RADIUS - 1.5,
        )

        p.end()
