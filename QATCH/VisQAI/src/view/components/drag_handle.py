"""
drag_handle.py

Provides custom GUI components for interactive drag-and-drop operations.

This module contains the DragHandle class, a specialized PyQt5 widget designed
to provide a visual 'grip' for moving parent widgets within a layout. It
standardizes the mouse event flow (press, move, release) and expects the
parent's container to implement a specific API: `start_drag`, `update_drag`,
and `finish_drag`.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.0
"""

from PyQt5 import QtCore, QtGui, QtWidgets


class DragHandle(QtWidgets.QFrame):
    """A visual handle used to initiate drag-and-drop operations for parent widgets.

    This widget renders a 2x3 grid of dots (reminiscent of standard UI grabbers)
    and manages mouse events to communicate with a parent container's drag logic.

    Attributes:
        _dragging (bool): Internal state tracking if the handle is currently
            being held by the user.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(20)
        self.setFixedHeight(40)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.setStyleSheet("background: transparent;")
        self._dragging = False

    def paintEvent(self, event):
        """Renders the visual 'grabber' dots on the handle.

        Args:
            event (QPaintEvent): The paint event triggered by Qt.
        """
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setBrush(QtGui.QColor("#777777"))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        dot_size = 4
        spacing = 4
        start_x = (self.width() - (dot_size * 2 + spacing)) / 2
        start_y = 15
        for row in range(3):
            for col in range(2):
                x = start_x + col * (dot_size + spacing)
                y = start_y + row * (dot_size + spacing)
                painter.drawEllipse(int(x), int(y), dot_size, dot_size)

    def mousePressEvent(self, event):
        """Handles the initial mouse click to start a drag operation.

        Changes the cursor to a closed hand and attempts to call 'start_drag'
        on the parent's container.

        Args:
            event (QMouseEvent): The mouse press event.
        """
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = True
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            card = self.parent()
            container = card.parent()
            offset = self.mapTo(card, event.pos())
            if hasattr(container, "start_drag"):
                container.start_drag(card, event.globalPos(), offset)

    def mouseMoveEvent(self, event):
        """Updates the drag position as the mouse moves.

        Args:
            event (QMouseEvent): The mouse move event.
        """
        if self._dragging:
            card = self.parent()
            container = card.parent()
            if hasattr(container, "update_drag"):
                container.update_drag(event.globalPos())

    def mouseReleaseEvent(self, event):
        """Finalizes the drag operation and resets the cursor.

        Args:
            event (QMouseEvent): The mouse release event.
        """
        if self._dragging:
            self._dragging = False
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
            card = self.parent()
            container = card.parent()
            if hasattr(container, "finish_drag"):
                container.finish_drag()
