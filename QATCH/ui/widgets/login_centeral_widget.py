import os
from typing import Optional
from PyQt5 import QtCore, QtGui, QtWidgets
from QATCH.common.architecture import Architecture


# ══════════════════════════════════════════════════════════════════════════════
class LoginCentralWidget(QtWidgets.QWidget):
    """Central widget that paints a blurred, lightly frosted backdrop.

    The backdrop is sourced either by:
      • calling :meth:`capture_backdrop` with the run-window (default), or
      • calling :meth:`set_background_pixmap` with a pre-loaded QPixmap
        (e.g. a branded splash image supplied by the caller).

    ``paintEvent`` draws the blurred pixmap wall-to-wall and then lays a very
    light neutral tint on top (~24 % opacity) so the effect reads as "frosted"
    without overpowering the glass card in front.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._blurred: Optional[QtGui.QPixmap] = None
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    # ── public ────────────────────────────────────────────────────────────────
    def capture_backdrop(self, run_window: QtWidgets.QMainWindow) -> None:
        """Grab *run_window*, blur it, and schedule a repaint.

        Grabbing the run window directly via ``.grab()`` means the login window
        never needs to be hidden/shown, so no spurious window events fire.
        """
        raw: QtGui.QPixmap = run_window.grab()
        if not self.size().isEmpty():
            raw = raw.scaled(
                self.size(),
                QtCore.Qt.KeepAspectRatioByExpanding,
                QtCore.Qt.SmoothTransformation,
            )
        self._blurred = self._apply_blur(raw, radius=22)
        self.update()

    def set_background_pixmap(self) -> None:
        pixmap = QtGui.QPixmap(
            os.path.join(Architecture.get_path(), "QATCH", "icons", "background.png")
        )
        if not self.size().isEmpty():
            pixmap = pixmap.scaled(
                self.size(),
                QtCore.Qt.KeepAspectRatioByExpanding,
                QtCore.Qt.SmoothTransformation,
            )
        self._blurred = self._apply_blur(pixmap, radius=22)
        self.update()
        # Children that paint based on _blurred won't be auto-invalidated because
        # they fully cover their own area. Tell them explicitly.
        for w in self.findChildren(QtWidgets.QWidget):
            w.update()

    # ── private ───────────────────────────────────────────────────────────────
    @staticmethod
    def _apply_blur(source: QtGui.QPixmap, radius: int = 22) -> QtGui.QPixmap:
        """Return a blurred copy of *source* using QGraphicsBlurEffect."""
        scene = QtWidgets.QGraphicsScene()
        item = QtWidgets.QGraphicsPixmapItem(source)
        blur = QtWidgets.QGraphicsBlurEffect()
        blur.setBlurRadius(radius)
        blur.setBlurHints(QtWidgets.QGraphicsBlurEffect.QualityHint)
        item.setGraphicsEffect(blur)
        scene.addItem(item)

        out = QtGui.QPixmap(source.size())
        out.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(out)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        scene.render(p, source=QtCore.QRectF(item.boundingRect()))
        p.end()
        return out

    # ── Qt events ─────────────────────────────────────────────────────────────
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        if self._blurred:
            p.drawPixmap(self.rect(), self._blurred, self._blurred.rect())
        else:
            # Gradient fallback shown for the instant before capture completes.
            grad = QtGui.QLinearGradient(0, 0, self.width(), self.height())
            grad.setColorAt(0.0, QtGui.QColor(0xD8, 0xE6, 0xF0))
            grad.setColorAt(1.0, QtGui.QColor(0xEE, 0xF4, 0xF8))
            p.fillRect(self.rect(), QtGui.QBrush(grad))

        # Very light frost tint (~24 % opacity).
        p.fillRect(self.rect(), QtGui.QColor(238, 243, 247, 62))
        p.end()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        # Re-render the backdrop at the new size so GlassCard's coordinate math stays valid.
        if self._blurred is not None:
            self.set_background_pixmap()
