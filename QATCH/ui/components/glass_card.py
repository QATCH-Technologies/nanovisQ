from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.styles.theme_manager import ThemeManager
from QATCH.ui.widgets.login_central_widget import LoginCentralWidget


class GlassCard(QtWidgets.QFrame):
    """A custom frame that renders a glassmorphism effect via backdrop sampling.

    This widget creates the illusion of translucent glass by sampling the blurred
    pixmap from a LoginCentralWidget. It maps its local coordinates to the
    backdrop's coordinate space to 'slice' the background perfectly.

    The rendering pipeline follows these steps:
        1. Create a rounded-rectangle clip path.
        2. Sample and translate the blurred backdrop slice.
        3. Apply a neutral 'frost' tint and a faint cool blue identifier tint.
        4. Render a top-down white shimmer gradient.
        5. Draw a multi-layered border (muted outer stroke + inner highlight rim).

    Attributes:
        _RADIUS (float): The corner radius for the rounded rectangle.
        _backdrop (LoginCentralWidget): Reference to the widget providing the
            blurred source image.
    """

    _RADIUS: float = 22.0

    def __init__(
        self,
        backdrop: LoginCentralWidget,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initializes the glass card and configures transparency attributes.

        Args:
            backdrop (LoginCentralWidget): The source widget for background blur.
            parent (QtWidgets.QWidget, optional): The parent widget.
        """
        super().__init__(parent)
        self._backdrop = backdrop
        self._border_frac: float = 0.0

        # Prevent the base class from painting opaque backgrounds
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)

    def set_border_frac(self, value: float) -> None:
        """Sets the emphasis-border progress (0.0 hidden -> 1.0 fully shown)."""
        self._border_frac = max(0.0, min(1.0, value))
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Executes the custom glassmorphism painting pipeline.

        This method manually handles all background and border rendering. It
        specifically avoids calling super().paintEvent() to prevent Qt Style
        Sheets from overwriting the translucent effects with opaque colors.
        """
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        rect_f = QtCore.QRectF(self.rect())
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)
        blurred = getattr(self._backdrop, "_blurred", None)
        sharp = getattr(self._backdrop, "_raw_snapshot", None)
        if (blurred is not None and not blurred.isNull()) or (
            sharp is not None and not sharp.isNull()
        ):
            origin = self.mapTo(self._backdrop, QtCore.QPoint(0, 0))
            p.save()
            p.translate(-origin.x(), -origin.y())
            # Mirror the backdrop's blur/dim progress so the card's sampled
            # slice matches the area around it exactly during transitions.
            blur_frac = getattr(self._backdrop, "_blur_frac", 0.0)
            dim_frac = getattr(self._backdrop, "_dim_frac", 0.0)
            LoginCentralWidget._compose_backdrop(
                p,
                self._backdrop.rect(),
                sharp,
                blurred,
                blur_frac,
                dim_frac,
                ThemeManager.instance().tokens(),
            )
            p.restore()
        else:
            tokens = ThemeManager.instance().tokens()
            grad = QtGui.QLinearGradient(0, 0, 0, self.height())
            grad.setColorAt(0.0, QtGui.QColor(*tokens["backdrop_fallback_start"]))
            grad.setColorAt(1.0, QtGui.QColor(*tokens["backdrop_fallback_end"]))
            p.fillRect(self.rect(), QtGui.QBrush(grad))

        # Primary white glass tint
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 70))
        p.fillRect(self.rect(), QtGui.QColor(210, 225, 240, 35))
        shimmer = QtGui.QLinearGradient(0, 0, 0, 80)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 60))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))
        p.setClipping(False)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(120, 160, 200, 135), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 175), 1.0))
        p.drawRoundedRect(
            rect_f.adjusted(1.5, 1.5, -1.5, -1.5),
            self._RADIUS - 1.5,
            self._RADIUS - 1.5,
        )

        # Emphasis border: a crisp 1px white edge that catches up shortly
        # after the card pops in, like glass catching the light.
        if self._border_frac > 0.0:
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, int(210 * self._border_frac)), 1.0))
            p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)

        p.end()
