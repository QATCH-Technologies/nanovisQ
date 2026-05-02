from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Optional
from QATCH.ui.widgets.login_centeral_widget import LoginCentralWidget


class GlassCard(QtWidgets.QFrame):
    """QFrame that renders true glassmorphism by sampling the frosted backdrop.

    In ``paintEvent`` it:
      1. Maps this widget's position onto the ``_LoginCentralWidget`` backdrop
         and blits the blurred pixmap slice that falls directly behind the card.
      2. Overlays a semi-transparent white glass tint.
      3. Adds a top-edge shimmer (lit-from-above glass highlight).
      4. Draws a two-layer border (muted outer + white inner highlight).

    All painting is done with clipping to the rounded rectangle so the glass
    effect is sharp-edged.  Child widgets paint on top normally; because the
    sign-in page uses 28 px horizontal margins the content stays well clear of
    the 22 px corner radii, so no clipping artefacts appear.
    """

    _RADIUS: float = 22.0

    def __init__(
        self,
        backdrop: LoginCentralWidget,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._backdrop = backdrop
        # Disable auto-fill so Qt does not pre-clear with the palette colour
        # before our paintEvent runs.
        self.setAutoFillBackground(False)
        # Belt-and-suspenders: also suppress the system/theme background so
        # no Qt style or compositor step can paint an opaque slab behind us.
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)

    # ── Qt events ─────────────────────────────────────────────────────────────
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        rf = QtCore.QRectF(self.rect())

        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rf, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)

        blurred = getattr(self._backdrop, "_blurred", None)
        if blurred is not None and not blurred.isNull():
            # Translate so the backdrop's origin aligns with where it paints on screen,
            # then issue the *same* drawPixmap call the centralwidget uses (full-rect stretch).
            # The clip path already restricts output to our card's rounded rect.
            origin = self.mapTo(self._backdrop, QtCore.QPoint(0, 0))
            p.save()
            p.translate(-origin.x(), -origin.y())
            p.drawPixmap(self._backdrop.rect(), blurred, blurred.rect())
            # Mirror the centralwidget's frost tint so the slice color-matches the surroundings.
            p.fillRect(self._backdrop.rect(), QtGui.QColor(238, 243, 247, 62))
            p.restore()
        else:
            grad = QtGui.QLinearGradient(0, 0, 0, self.height())
            grad.setColorAt(0.0, QtGui.QColor(0xD4, 0xE6, 0xF4))
            grad.setColorAt(1.0, QtGui.QColor(0xE6, 0xF2, 0xFA))
            p.fillRect(self.rect(), QtGui.QBrush(grad))

        # Glass tint + shimmer + borders unchanged below
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 70))

        # Optional: a very faint cool tint to give the card identity against pale backdrops.
        # Skip this if your backdrop already has color.
        p.fillRect(self.rect(), QtGui.QColor(210, 225, 240, 35))

        # Shimmer — same idea, but ease it back so it doesn't push us back to white.
        shimmer = QtGui.QLinearGradient(0, 0, 0, 80)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 60))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        p.setClipping(False)

        # 4a ── Outer muted blue-grey border.
        p.setPen(QtGui.QPen(QtGui.QColor(120, 160, 200, 135), 1.0))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRoundedRect(rf.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)

        # 4b ── Inner white highlight (gives the card a frosted-glass rim).
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 175), 1.0))
        p.drawRoundedRect(
            rf.adjusted(1.5, 1.5, -1.5, -1.5),
            self._RADIUS - 1.5,
            self._RADIUS - 1.5,
        )

        p.end()
        # Note: do NOT call super().paintEvent(event) — that would repaint
        # the QSS background-color on top of our glass effect.
