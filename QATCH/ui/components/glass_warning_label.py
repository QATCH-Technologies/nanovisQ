"""
glass_warning_label.py

This module defines the `GlassWarningLabel` widget, designed to replace traditional,
harshly colored (e.g., loud orange) warning labels with a calm, aesthetically pleasing
blue-gray glass strip. It is intended for use in dialogs, such as Advanced Settings,
where information needs to be conveyed clearly without unnecessarily alarming the user.

The widget is built to maintain drop-in API compatibility with standard `QLabel`
methods (like `setText` and `text`) to ensure seamless integration into existing
codebases without requiring extensive refactoring.

"""

import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as QtWidgets


class GlassWarningLabel(QtWidgets.QWidget):
    """Calm informational banner for the Advanced Settings dialog.

    Renders as a soft blue-gray glass strip with an optional leading icon and
    informational (not alarming) text. Replaces the old loud orange warning.
    Keeps a QLabel-like ``setText`` so existing call sites still work.

    Attributes:
        icon_lbl (QtWidgets.QLabel): The label widget responsible for displaying
            the optional leading icon.
        text_lbl (QtWidgets.QLabel): The label widget containing the main
            informational text.
    """

    _RADIUS: float = 6.0

    def __init__(self, text: str = "", icon_path: str = "", parent=None) -> None:
        """Initializes the GlassWarningLabel.

        Args:
            text (str, optional): The informational text to display in the banner.
                Defaults to "".
            icon_path (str, optional): The file path to the leading icon. If provided,
                the icon is loaded and displayed. Defaults to "".
            parent (QtWidgets.QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(10, 6, 10, 6)
        row.setSpacing(8)

        # Leading icon slot populated when an icon path is provided.
        self.icon_lbl = QtWidgets.QLabel(self)
        self.icon_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.icon_lbl.setStyleSheet("background: transparent; border: none;")
        self.icon_lbl.setFixedSize(16, 16)
        self.icon_lbl.setScaledContents(True)
        if icon_path:
            self.set_icon(icon_path)
        else:
            self.icon_lbl.hide()
        row.addWidget(self.icon_lbl, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

        self.text_lbl = QtWidgets.QLabel(text, self)
        self.text_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.text_lbl.setStyleSheet(
            "QLabel { color: rgba(45, 75, 105, 220); font-size: 11px; "
            "font-weight: normal; background: transparent; border: none; }"
        )
        self.text_lbl.setWordWrap(True)
        row.addWidget(self.text_lbl, 1, QtCore.Qt.AlignmentFlag.AlignVCenter)

    def set_icon(self, icon_path: str) -> None:
        """Sets the leading icon for the warning label.

        Attempts to load an image from the provided file path. If the image is
        loaded successfully (i.e., the resulting pixmap is not null), it updates
        the icon label's pixmap and ensures the label is visible.

        Args:
            icon_path (str): The file path to the icon image to be displayed.
        """
        pix = QtGui.QPixmap(icon_path)
        if not pix.isNull():
            self.icon_lbl.setPixmap(pix)
            self.icon_lbl.show()

    def setText(self, text: str) -> None:
        """Sets the informational text of the banner.

        This method mimics the standard `QLabel.setText` API to maintain
        drop-in compatibility with existing call sites that previously
        interacted with a standard label.

        Args:
            text (str): The new informational text to display in the banner.
        """
        self.text_lbl.setText(text)

    def text(self) -> str:
        """Returns the current informational text of the banner.

        This method mimics the standard `QLabel.text` API to maintain
        compatibility with existing code that expects standard label behavior.

        Returns:
            str: The text currently displayed in the banner.
        """
        return self.text_lbl.text()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Handles the custom painting of the widget's background.

        Renders the widget's visual styling, including a rounded shape,
        a soft blue-gray gradient background, a subtle white shimmer effect
        at the top to simulate glass, and a delicate hairline border.

        Args:
            event (QtGui.QPaintEvent): The paint event parameters provided
                by the Qt framework.
        """
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing)

        rect_f = QtCore.QRectF(self.rect())
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)
        grad = QtGui.QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, QtGui.QColor(120, 165, 210, 40))
        grad.setColorAt(1.0, QtGui.QColor(95, 140, 190, 30))
        p.fillRect(self.rect(), QtGui.QBrush(grad))

        # Top shimmer
        shimmer = QtGui.QLinearGradient(0, 0, 0, self.height() * 0.6)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 40))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Border
        p.setClipping(False)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(120, 160, 200, 110), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)

        p.end()
