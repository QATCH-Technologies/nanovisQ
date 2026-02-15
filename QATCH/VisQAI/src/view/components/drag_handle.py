from PyQt5 import QtCore, QtGui, QtWidgets


class DragHandle(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(20)
        self.setFixedHeight(40)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.setStyleSheet("background: transparent;")
        self._dragging = False

    def paintEvent(self, event):
        # (Keep your existing paint logic here)
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
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = True
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)

            # Walk up hierarchy: Handle -> Card -> Container
            card = self.parent()
            container = card.parent()

            # Calculate where we clicked relative to the card's top-left
            # mapTo(card, ...) ensures we grab the card exactly where the mouse is
            offset = self.mapTo(card, event.pos())

            if hasattr(container, "start_drag"):
                container.start_drag(card, event.globalPos(), offset)

    def mouseMoveEvent(self, event):
        if self._dragging:
            card = self.parent()
            container = card.parent()
            if hasattr(container, "update_drag"):
                container.update_drag(event.globalPos())

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
            card = self.parent()
            container = card.parent()
            if hasattr(container, "finish_drag"):
                container.finish_drag()
