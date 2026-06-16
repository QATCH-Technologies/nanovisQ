import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as QtWidgets


class _GlassDeviceInfoInnerPanel(QtWidgets.QWidget):
    """Inner glass-morphism panel for the device info popup."""

    _RADIUS: float = 10.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        rect_f = QtCore.QRectF(self.rect())
        _R = self._RADIUS

        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, _R, _R)
        p.setClipPath(clip)

        # Frosted white base
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 235))
        p.fillRect(self.rect(), QtGui.QColor(228, 235, 241, 28))

        # Top shimmer
        shimmer = QtGui.QLinearGradient(0, 0, 0, 44)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 80))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Dual borders
        p.setClipping(False)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), _R, _R)
        p.setPen(QtGui.QPen(QtGui.QColor(200, 210, 220, 90), 1.0))
        p.drawRoundedRect(rect_f.adjusted(1.5, 1.5, -1.5, -1.5), _R - 1.5, _R - 1.5)

        p.end()


class DeviceInfoMainWidget(QtWidgets.QWidget):
    """Frosted-glass dropdown panel for the Device Info Configuration."""

    _SHADOW_MARGIN_L = 22
    _SHADOW_MARGIN_T = 18
    _SHADOW_MARGIN_R = 22
    _SHADOW_MARGIN_B = 26

    def __init__(self, parent=None) -> None:
        super().__init__(
            parent,
            QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint | QtCore.Qt.NoDropShadowWindowHint,
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

        self._main_window = None

        # Outer container with shadow margins
        self._panel = _GlassDeviceInfoInnerPanel(self)
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(
            self._SHADOW_MARGIN_L,
            self._SHADOW_MARGIN_T,
            self._SHADOW_MARGIN_R,
            self._SHADOW_MARGIN_B,
        )
        outer_layout.setSpacing(0)
        outer_layout.addWidget(self._panel)

        # Drop shadow
        shadow = QtWidgets.QGraphicsDropShadowEffect(self._panel)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 4)
        shadow.setColor(QtGui.QColor(0, 20, 40, 110))
        self._panel.setGraphicsEffect(shadow)

        # Inner layout for dynamic content
        self.content_layout = QtWidgets.QVBoxLayout(self._panel)
        self.content_layout.setContentsMargins(14, 14, 14, 14)

    def set_content_widget(self, widget: QtWidgets.QWidget):
        """Injects the existing device info container into the popup."""
        self.content_layout.addWidget(widget)
        widget.show()

    def show_anchored_to(self, anchor: QtWidgets.QWidget, main_window=None) -> None:
        """Shows the popup pinned to the anchor, clamped to the main window."""
        self._main_window = main_window
        self.adjustSize()

        popup_w, popup_h = self.width(), self.height()
        anchor_br = anchor.mapToGlobal(QtCore.QPoint(anchor.width(), anchor.height()))

        x = anchor_br.x() + self._SHADOW_MARGIN_R - popup_w
        y = anchor_br.y() + 2 - self._SHADOW_MARGIN_T

        # Clamp logic to ensure it doesn't render off-screen/off-app
        top_level = anchor.window() if anchor is not None else None
        bounds = (
            top_level.geometry()
            if top_level
            else (self._main_window.geometry() if self._main_window else QtCore.QRect())
        )

        if not bounds.isNull():
            visible = QtCore.QRect(
                x + self._SHADOW_MARGIN_L,
                y + self._SHADOW_MARGIN_T,
                popup_w - self._SHADOW_MARGIN_L - self._SHADOW_MARGIN_R,
                popup_h - self._SHADOW_MARGIN_T - self._SHADOW_MARGIN_B,
            )
            if visible.right() > bounds.right():
                x -= visible.right() - bounds.right()
            if visible.left() < bounds.left():
                x += bounds.left() - visible.left()
            if visible.bottom() > bounds.bottom():
                anchor_top = anchor.mapToGlobal(QtCore.QPoint(0, 0)).y()
                y_above = anchor_top - 2 - popup_h + self._SHADOW_MARGIN_B
                if (y_above + self._SHADOW_MARGIN_T) >= bounds.top():
                    y = y_above
                else:
                    y -= visible.bottom() - bounds.bottom()

        self.move(x, y)
        if self._main_window is not None:
            self._main_window.installEventFilter(self)
        self.show()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        # Auto-close if the main window moves or resizes
        if watched is self._main_window and event.type() in (
            QtCore.QEvent.Resize,
            QtCore.QEvent.Move,
            QtCore.QEvent.WindowStateChange,
        ):
            self.close()
        return super().eventFilter(watched, event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._main_window is not None:
            try:
                self._main_window.removeEventFilter(self)
            except Exception:
                pass
            self._main_window = None
        super().closeEvent(event)

# class DeviceInfoLayoutWidget(QtWidgets.QWidget):
#     def __init__(self, parent=None)
#         super()