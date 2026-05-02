from PyQt5 import QtCore, QtWidgets


class FloatingMessageBadge(QtWidgets.QWidget):
    """A frameless, glassy floating badge for alerts and info."""

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        # Make it a frameless tool window that stays on top but doesn't steal focus
        self.setWindowFlags(
            QtCore.Qt.Tool | QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)

        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # The actual styled label
        self.label = QtWidgets.QLabel("")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.label)

        self.hide()

    def show_message(
        self, text: str, is_error: bool = False, parent_widget: QtWidgets.QWidget = None
    ) -> None:
        """Updates the text and style, positions it, and shows it."""
        self.label.setText(text)

        # Style based on whether it's an error (red) or info (amber)
        if is_error:
            self.label.setStyleSheet("""
                QLabel {
                    background-color: rgba(255, 230, 230, 240);
                    border: 1.5px solid rgba(230, 50, 50, 200);
                    border-radius: 14px;
                    color: rgba(200, 30, 30, 255);
                    font-size: 8.5pt; font-weight: 600; padding: 4px 16px;
                }
            """)
        else:
            self.label.setStyleSheet("""
                QLabel {
                    background-color: rgba(255, 245, 230, 240);
                    border: 1.5px solid rgba(230, 150, 50, 200);
                    border-radius: 14px;
                    color: rgba(220, 130, 40, 255);
                    font-size: 8.5pt; font-weight: 600; padding: 4px 16px;
                }
            """)

        self.adjustSize()

        # Center it slightly above the target widget (like the login card)
        if parent_widget:
            global_pos = parent_widget.mapToGlobal(QtCore.QPoint(0, 0))
            x = global_pos.x() + (parent_widget.width() - self.width()) // 2
            y = global_pos.y() - self.height() - 15  # 15px above the card
            self.move(x, y)

        self.show()

    def clear(self) -> None:
        self.hide()
