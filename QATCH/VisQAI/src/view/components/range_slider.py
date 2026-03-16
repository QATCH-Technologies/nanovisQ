"""
range_slider.py

Provides a double-ended range selection slider for PyQt5.

This module contains the RangeSlider class, which allows users to select a
minimum and maximum value simultaneously. It supports both continuous scales
and discrete step-based selection with visual snapping indicators.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.0
"""

from PyQt5 import QtCore, QtGui, QtWidgets


class RangeSlider(QtWidgets.QWidget):
    """A dual-handle slider widget for range selection.

    The slider consists of two handles (low and high) moving along a single
    groove. It supports 'snapping' to specific discrete values if step_values
    are provided, and ensures that the low handle cannot cross the high handle.

    Attributes:
        rangeChanged (QtCore.pyqtSignal): Signal emitted when the low or high
            values change. Emits (float, float).
        _min (float): The minimum possible value of the slider.
        _max (float): The maximum possible value of the slider.
        _low (float): The current value of the lower handle.
        _high (float): The current value of the upper handle.
        _step_values (list[float], optional): A sorted list of discrete values
            the slider handles will snap to.
        _pressed_handle (str, optional): Tracks which handle is currently
            being dragged ('low', 'high', or None).
    """

    rangeChanged = QtCore.pyqtSignal(float, float)

    def __init__(self, min_val=0, max_val=100, parent=None):
        """Initializes the RangeSlider with a default or custom range.

        Args:
            min_val (float): The lowest value available. Defaults to 0.
            max_val (float): The highest value available. Defaults to 100.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setFixedHeight(30)
        self._min = min_val
        self._max = max_val
        self._low = min_val
        self._high = max_val
        self._step_values = None
        self._pressed_handle = None
        self._handle_radius = 8
        self._groove_height = 4

    def setRange(self, min_val, max_val):
        """Updates the slider's absolute minimum and maximum bounds.

        Args:
            min_val (float): New minimum bound.
            max_val (float): New maximum bound.
        """
        self._min = min_val
        self._max = max_val
        self.update()

    def setStepValues(self, steps):
        """Sets specific discrete values the slider should snap to.

        If provided, the slider will divide the available width into equal
        segments based on the number of steps, rather than a linear scale.

        Args:
            steps (list[float]): A list of numbers representing the allowed
                snap points.
        """
        if steps and len(steps) > 1:
            self._step_values = sorted(steps)
            self._min = self._step_values[0]
            self._max = self._step_values[-1]
        else:
            self._step_values = None
        self.update()

    def setValues(self, low, high):
        """Sets the current positions of the low and high handles.

        Args:
            low (float): The value for the lower handle.
            high (float): The value for the upper handle.
        """
        self._low = max(self._min, min(low, self._high))
        self._high = min(self._max, max(high, self._low))
        self.update()

    def _val_to_x(self, v, available_w):
        """Maps a data value to a horizontal pixel coordinate.

        Args:
            v (float): The value to map.
            available_w (int): The width of the slider track in pixels.

        Returns:
            float: The X-coordinate relative to the widget.
        """
        if self._step_values:
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
        """Handles the visual rendering of the groove, ticks, and handles.

        Args:
            event (QPaintEvent): The paint event triggered by Qt.
        """
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        cy = h / 2
        available_w = w - 2 * self._handle_radius

        x_low = self._val_to_x(self._low, available_w)
        x_high = self._val_to_x(self._high, available_w)
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
        if self._step_values:
            painter.setPen(QtGui.QPen(QtGui.QColor("#9ca3af"), 2))
            for step in self._step_values:
                tx = self._val_to_x(step, available_w)
                painter.drawLine(QtCore.QPointF(tx, cy - 6), QtCore.QPointF(tx, cy + 6))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#0078D4"))
        rect_range = QtCore.QRectF(
            x_low, cy - self._groove_height / 2, x_high - x_low, self._groove_height
        )
        painter.drawRect(rect_range)

        painter.setBrush(QtGui.QColor("#ffffff"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#0078D4"), 2))

        painter.drawEllipse(
            QtCore.QPointF(x_low, cy), self._handle_radius, self._handle_radius
        )
        painter.drawEllipse(
            QtCore.QPointF(x_high, cy), self._handle_radius, self._handle_radius
        )

    def mousePressEvent(self, event):
        """Identifies which handle is closest to the click to begin dragging.

        Args:
            event (QMouseEvent): The mouse press event.
        """
        w = self.width()
        available_w = w - 2 * self._handle_radius

        pos_x = event.pos().x()
        dist_low = abs(pos_x - self._val_to_x(self._low, available_w))
        dist_high = abs(pos_x - self._val_to_x(self._high, available_w))

        if dist_low < dist_high:
            self._pressed_handle = "low"
        else:
            self._pressed_handle = "high"

        self.mouseMoveEvent(event)

    def mouseMoveEvent(self, event):
        """Updates handle values based on mouse position and snapping logic.

        Args:
            event (QMouseEvent): The mouse move event.
        """
        if not self._pressed_handle:
            return

        w = self.width()
        available_w = w - 2 * self._handle_radius

        pos_x = max(self._handle_radius, min(event.pos().x(), w - self._handle_radius))
        ratio = (pos_x - self._handle_radius) / available_w

        if self._step_values:
            idx = int(round(ratio * (len(self._step_values) - 1)))
            idx = max(0, min(idx, len(self._step_values) - 1))
            val = self._step_values[idx]
        else:
            val = self._min + ratio * (self._max - self._min)

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
        """Ends the dragging operation.

        Args:
            event (QMouseEvent): The mouse release event.
        """
        self._pressed_handle = None
