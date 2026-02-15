from PyQt5 import QtCore, QtGui, QtWidgets


class RangeSlider(QtWidgets.QWidget):
    """A simple double-ended slider for selecting a range."""

    rangeChanged = QtCore.pyqtSignal(float, float)

    def __init__(self, min_val=0, max_val=100, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self._min = min_val
        self._max = max_val
        self._low = min_val
        self._high = max_val
        self._pressed_handle = None  # 'low', 'high', or None
        self._handle_radius = 8
        self._groove_height = 4

    def setRange(self, min_val, max_val):
        self._min = min_val
        self._max = max_val
        self.update()

    def setValues(self, low, high):
        self._low = max(self._min, min(low, self._high))
        self._high = min(self._max, max(high, self._low))
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Geometry
        w = self.width()
        h = self.height()
        cy = h / 2
        available_w = w - 2 * self._handle_radius

        def val_to_x(v):
            if self._max == self._min:
                return 0
            ratio = (v - self._min) / (self._max - self._min)
            return self._handle_radius + ratio * available_w

        x_low = val_to_x(self._low)
        x_high = val_to_x(self._high)

        # Draw Groove (Background)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#e0e0e0"))
        painter.drawRoundedRect(
            QtCore.QRectF(
                self._handle_radius,
                cy - self._groove_height / 2,
                available_w,
                self._groove_height,
            ),
            2,
            2,
        )

        # Draw Selected Range (Blue)
        painter.setBrush(QtGui.QColor("#0078D4"))
        rect_range = QtCore.QRectF(
            x_low, cy - self._groove_height / 2, x_high - x_low, self._groove_height
        )
        painter.drawRect(rect_range)

        # Draw Handles
        painter.setBrush(QtGui.QColor("#ffffff"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#0078D4"), 2))

        painter.drawEllipse(
            QtCore.QPointF(x_low, cy), self._handle_radius, self._handle_radius
        )
        painter.drawEllipse(
            QtCore.QPointF(x_high, cy), self._handle_radius, self._handle_radius
        )

    def mousePressEvent(self, event):
        w = self.width()
        available_w = w - 2 * self._handle_radius

        def val_to_x(v):
            if self._max == self._min:
                return 0
            return (
                self._handle_radius
                + ((v - self._min) / (self._max - self._min)) * available_w
            )

        # FIX: Use event.pos().x() for PyQt5 compatibility
        pos_x = event.pos().x()
        dist_low = abs(pos_x - val_to_x(self._low))
        dist_high = abs(pos_x - val_to_x(self._high))

        if dist_low < dist_high:
            self._pressed_handle = "low"
        else:
            self._pressed_handle = "high"

        self.mouseMoveEvent(event)

    def mouseMoveEvent(self, event):
        if not self._pressed_handle:
            return

        w = self.width()
        available_w = w - 2 * self._handle_radius

        # FIX: Use event.pos().x() for PyQt5 compatibility
        pos_x = max(self._handle_radius, min(event.pos().x(), w - self._handle_radius))

        ratio = (pos_x - self._handle_radius) / available_w
        val = self._min + ratio * (self._max - self._min)

        if self._pressed_handle == "low":
            self._low = min(val, self._high)
        else:
            self._high = max(val, self._low)

        self.update()
        self.rangeChanged.emit(self._low, self._high)

    def mouseReleaseEvent(self, event):
        self._pressed_handle = None
