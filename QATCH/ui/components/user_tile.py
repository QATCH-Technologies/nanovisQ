from PyQt5 import QtCore, QtGui, QtWidgets

class DropPlaceholder(QtWidgets.QWidget):
    """A highly glassy, squarer highlight indicating where the profile will be dropped."""

    def __init__(self, width: int, height: int, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = QtCore.QRectF(self.rect()).adjusted(2.0, 2.0, -2.0, -2.0)
        radius = 12.0

        # 1. Glassy Gradient Fill
        grad = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
        grad.setColorAt(0.0, QtGui.QColor(255, 255, 255, 80))  # Brighter top-left
        grad.setColorAt(0.5, QtGui.QColor(220, 230, 240, 40))  # Highly transparent center
        grad.setColorAt(1.0, QtGui.QColor(180, 195, 210, 60))  # Slightly frosted bottom-right

        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QBrush(grad))
        p.drawRoundedRect(rect, radius, radius)

        # 2. Bright Inner Reflection (The "Glass Edge")
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 1.5))
        p.drawRoundedRect(rect, radius, radius)

        # 3. Subtle Outer Dark Stroke for depth
        p.setPen(QtGui.QPen(QtGui.QColor(140, 155, 170, 50), 1.0))
        p.drawRoundedRect(rect.adjusted(-1, -1, 1, 1), radius + 1, radius + 1)


class DraggableUserTile(QtWidgets.QWidget):
    """A custom widget representing a user tile that can be dragged."""

    clicked = QtCore.pyqtSignal(str, str)

    def __init__(
        self, name: str, initials: str, index: int, is_current: bool, avatar_size: int, parent=None
    ):
        super().__init__(parent)
        self.name = name
        self.initials = initials
        self.index = index
        self._avatar_size = avatar_size

        self._drag_start_pos = None
        self._global_drag_start = None
        self._is_dragging = False

        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setFixedSize(72, 88)

        self._build_ui(is_current)

        self.btn.installEventFilter(self)
        self.lbl.installEventFilter(self)

    def _build_ui(self, is_current: bool):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.btn = QtWidgets.QPushButton()
        self.btn.setObjectName("userTileBtn")
        self.btn.setFixedSize(self._avatar_size, self._avatar_size)
        self.btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn.setIcon(QtGui.QIcon(self._make_circular_pixmap(self.initials, self._avatar_size)))
        self.btn.setIconSize(QtCore.QSize(self._avatar_size, self._avatar_size))
        self.btn.setProperty("current", is_current)

        self.btn.clicked.connect(lambda: self.clicked.emit(self.name, self.initials))

        self.lbl = QtWidgets.QLabel(self._format_name_stacked(self.name))
        self.lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl.setObjectName("tileName")

        layout.addWidget(self.btn, 0, QtCore.Qt.AlignHCenter)
        layout.addWidget(self.lbl, 0, QtCore.Qt.AlignHCenter)

    def eventFilter(self, obj, event: QtCore.QEvent) -> bool:
        if obj in (self.btn, self.lbl):
            if (
                event.type() == QtCore.QEvent.MouseButtonPress
                and event.button() == QtCore.Qt.LeftButton
            ):
                self._drag_start_pos = event.pos()
                self._global_drag_start = event.globalPos()
                self._is_dragging = False

            elif event.type() == QtCore.QEvent.MouseMove and (
                event.buttons() & QtCore.Qt.LeftButton
            ):
                if self._drag_start_pos is not None:
                    # Determine if we've moved far enough to start a drag
                    if not self._is_dragging:
                        if (
                            event.globalPos() - self._global_drag_start
                        ).manhattanLength() >= QtWidgets.QApplication.startDragDistance():
                            self._is_dragging = True
                            self.btn.setDown(False)  # Visually un-press the button

                            # Safely find the window and trigger the custom drag
                            window = self.window()
                            if hasattr(window, "start_custom_drag"):
                                window.start_custom_drag(self)
                            return True

                    # If already dragging, physically move the widget
                    if self._is_dragging:
                        window = self.window()
                        if hasattr(window, "process_custom_drag"):
                            window.process_custom_drag(self, event.globalPos())
                        return True

            elif (
                event.type() == QtCore.QEvent.MouseButtonRelease
                and event.button() == QtCore.Qt.LeftButton
            ):
                if self._is_dragging:
                    self._is_dragging = False
                    self._drag_start_pos = None

                    window = self.window()
                    if hasattr(window, "end_custom_drag"):
                        window.end_custom_drag(self)
                    return True  # Prevent the click signal from firing

                self._drag_start_pos = None

        return super().eventFilter(obj, event)

    @staticmethod
    def _format_name_stacked(name: str) -> str:
        parts = name.split(" ", 1)
        if len(parts) == 2:
            return f"{parts[0]}\n{parts[1]}"
        return name

    @staticmethod
    def _make_circular_pixmap(initials: str, size: int) -> QtGui.QPixmap:
        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        hash_val = sum(ord(c) for c in initials)
        hues = [210, 200, 220, 190, 215]
        hue = hues[hash_val % len(hues)]
        base_color = QtGui.QColor.fromHsl(hue, 90, 190)

        rect = QtCore.QRectF(2.0, 2.0, size - 4.0, size - 4.0)
        p.setBrush(QtGui.QBrush(base_color))
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 1.5))
        p.drawEllipse(rect)

        p.setPen(QtGui.QColor(60, 60, 60, 200))
        font = p.font()
        font.setPixelSize(int(size * 0.42))
        font.setBold(True)
        p.setFont(font)
        p.drawText(rect, QtCore.Qt.AlignCenter, initials)
        p.end()
        return pm
