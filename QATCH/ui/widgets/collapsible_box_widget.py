from PyQt5 import QtCore, QtWidgets

from QATCH.ui.styles.theme_manager import ThemeManager, tok_css


class CollapsibleBox(QtWidgets.QWidget):
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)
        self._collapsed = True

        self.toggle_button = QtWidgets.QToolButton(
            text=title, checkable=True, checked=False
        )

        self.toggle_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonTextBesideIcon
        )
        self.toggle_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.toggle_button.setArrowType(QtCore.Qt.RightArrow)
        self.toggle_button.pressed.connect(self.toggle)

        self.toggle_animation = QtCore.QParallelAnimationGroup(self)

        self.content_area = QtWidgets.QScrollArea(
            maximumHeight=0, minimumHeight=0
        )
        self.content_area.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.content_area.setFrameShape(QtWidgets.QFrame.NoFrame)

        lay = QtWidgets.QVBoxLayout(self)
        # lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b"minimumHeight")
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b"maximumHeight")
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self.content_area, b"maximumHeight")
        )

        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Tints the toggle button (text/arrow/hover) from the active
        theme's flat_* tokens - this widget has no other chrome of its own."""
        tok = ThemeManager.instance().tokens()
        self.toggle_button.setStyleSheet(
            "QToolButton {"
            f"  color: {tok_css(tok['flat_text'])};"
            "  border: none;"
            "  background: transparent;"
            "  font-weight: 600;"
            "  font-size: 12px;"
            "}"
            "QToolButton:hover {"
            f"  color: {tok_css(tok['flat_accent'])};"
            "}"
        )

    def setCollapsed(self, checked):
        if self._collapsed is not checked:
            self._collapsed = checked
            self.repaint()

    def isCollapsed(self):
        return self._collapsed

    def toggle(self):
        self.setCollapsed(not self.isCollapsed())
        # calls self.repaint()

    def repaint(self):
        checked = self._collapsed
        self.toggle_button.setArrowType(
            QtCore.Qt.DownArrow if not checked else QtCore.Qt.RightArrow
        )
        self.toggle_animation.setDirection(
            QtCore.QAbstractAnimation.Forward
            if not checked
            else QtCore.QAbstractAnimation.Backward
        )

        self.toggle_animation.start()

    def setContentLayout(self, layout):
        lay = self.content_area.layout()
        del lay
        self.content_area.setLayout(layout)
        collapsed_height = (
            self.sizeHint().height() - self.content_area.maximumHeight()
        )
        content_height = layout.sizeHint().height()
        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(500)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(
            self.toggle_animation.animationCount() - 1
        )
        content_animation.setDuration(500)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)
