from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets


class AnimatedComboBox(QtWidgets.QComboBox):
    """A QComboBox that smoothly spins its drop-down arrow on open/close."""

    def __init__(self, icon_path: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setView(QtWidgets.QListView())
        dropdown_shadow = QtWidgets.QGraphicsDropShadowEffect(self.view())
        dropdown_shadow.setBlurRadius(25)
        dropdown_shadow.setColor(QtGui.QColor(15, 40, 70, 70))
        dropdown_shadow.setOffset(0, 8)
        self.view().setGraphicsEffect(dropdown_shadow)

        # --- Arrow Icon Setup ---
        self.arrow_lbl = QtWidgets.QLabel(self)
        self.arrow_lbl.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.arrow_lbl.setAlignment(QtCore.Qt.AlignCenter)

        self._base_pixmap = QtGui.QPixmap(icon_path).scaled(
            11, 11, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.arrow_lbl.setPixmap(self._base_pixmap)

        # --- Animation Setup ---
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(250)
        self._anim.setEasingCurve(QtCore.QEasingCurve.InOutQuad)
        self._anim.valueChanged.connect(self._on_spin_frame)
        self._current_angle = 0.0

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.arrow_lbl.setGeometry(self.width() - 32, 0, 32, self.height())

    def showPopup(self) -> None:
        popup_window = self.view().window()
        if popup_window:
            popup_window.setWindowFlags(
                QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint | QtCore.Qt.NoDropShadowWindowHint
            )
            popup_window.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        super().showPopup()
        self._spin_to(180.0)

    def hidePopup(self) -> None:
        super().hidePopup()
        self._spin_to(0.0)

    def _spin_to(self, target_angle: float) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._current_angle)
        self._anim.setEndValue(target_angle)
        self._anim.start()

    def _on_spin_frame(self, angle: float) -> None:
        self._current_angle = angle
        transform = QtGui.QTransform().rotate(angle)
        rotated = self._base_pixmap.transformed(transform, QtCore.Qt.SmoothTransformation)
        self.arrow_lbl.setPixmap(rotated)
