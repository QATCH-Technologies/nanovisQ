from PyQt5 import QtCore, QtGui, QtWidgets


class RangeSlider(QtWidgets.QWidget):
    """A double-ended slider for selecting a range, with visual snapping to discrete steps."""

    rangeChanged = QtCore.pyqtSignal(float, float)

    def __init__(self, min_val=0, max_val=100, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self._min = min_val
        self._max = max_val
        self._low = min_val
        self._high = max_val
        self._step_values = None
        self._pressed_handle = None  # 'low', 'high', or None
        self._handle_radius = 8
        self._groove_height = 4

    def setRange(self, min_val, max_val):
        self._min = min_val
        self._max = max_val
        self.update()

    def setStepValues(self, steps):
        """Sets specific discrete values the slider should evenly space and snap to."""
        if steps and len(steps) > 1:
            self._step_values = sorted(steps)
            self._min = self._step_values[0]
            self._max = self._step_values[-1]
        else:
            self._step_values = None
        self.update()

    def setValues(self, low, high):
        self._low = max(self._min, min(low, self._high))
        self._high = min(self._max, max(high, self._low))
        self.update()

    def _val_to_x(self, v, available_w):
        """Maps a value to its X coordinate on the slider. Spaces steps evenly."""
        if self._step_values:
            # Find the closest step index to visually snap to an even notch
            try:
                idx = self._step_values.index(v)
            except ValueError:
                idx = min(
                    range(len(self._step_values)),
                    key=lambda i: abs(self._step_values[i] - v),
                )
            ratio = idx / (len(self._step_values) - 1)
        else:
            if self._max == self._min:
                return self._handle_radius
            ratio = (v - self._min) / (self._max - self._min)

        return self._handle_radius + ratio * available_w

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        cy = h / 2
        available_w = w - 2 * self._handle_radius

        x_low = self._val_to_x(self._low, available_w)
        x_high = self._val_to_x(self._high, available_w)

        # 1. Draw Groove (Background)
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

        # 2. Draw Ticks / Snap Indicators
        if self._step_values:
            painter.setPen(QtGui.QPen(QtGui.QColor("#9ca3af"), 2))
            for step in self._step_values:
                tx = self._val_to_x(step, available_w)
                painter.drawLine(QtCore.QPointF(tx, cy - 6), QtCore.QPointF(tx, cy + 6))

        # 3. Draw Selected Range (Blue)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#0078D4"))
        rect_range = QtCore.QRectF(
            x_low, cy - self._groove_height / 2, x_high - x_low, self._groove_height
        )
        painter.drawRect(rect_range)

        # 4. Draw Handles
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

        pos_x = event.pos().x()
        dist_low = abs(pos_x - self._val_to_x(self._low, available_w))
        dist_high = abs(pos_x - self._val_to_x(self._high, available_w))

        # Determine which handle is closer to the click
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

        pos_x = max(self._handle_radius, min(event.pos().x(), w - self._handle_radius))
        ratio = (pos_x - self._handle_radius) / available_w

        # If step values are provided, snap precisely to those index bounds
        if self._step_values:
            idx = int(round(ratio * (len(self._step_values) - 1)))
            idx = max(0, min(idx, len(self._step_values) - 1))
            val = self._step_values[idx]
        else:
            val = self._min + ratio * (self._max - self._min)

        # Apply constraints so low cannot cross high
        if self._pressed_handle == "low":
            if self._step_values:
                try:
                    high_idx = self._step_values.index(self._high)
                except ValueError:
                    high_idx = len(self._step_values) - 1
                idx = min(idx, high_idx)
                self._low = self._step_values[idx]
            else:
                self._low = min(val, self._high)
        else:
            if self._step_values:
                try:
                    low_idx = self._step_values.index(self._low)
                except ValueError:
                    low_idx = 0
                idx = max(idx, low_idx)
                self._high = self._step_values[idx]
            else:
                self._high = max(val, self._low)

        self.update()
        self.rangeChanged.emit(self._low, self._high)

    def mouseReleaseEvent(self, event):
        self._pressed_handle = None
