from typing import Optional
from PyQt5 import QtCore, QtGui, QtWidgets


class BorderlessActionButton(QtWidgets.QPushButton):
    """A flat, borderless text button that lightens on hover.

    Used for the per-field Save / Reset / Default actions so they read as quiet
    inline links rather than heavy glass pills. No border or fill at rest; a
    subtle translucent gray wash appears on hover and deepens on press.

    Attributes:
        _text_color_map (dict): Mapping of tone identifiers to color strings.
    """

    def __init__(
        self, text: str = "", parent: Optional[QtWidgets.QWidget] = None, *, tone: str = "neutral"
    ) -> None:
        """Initializes the button with custom hover/press states and tone.

        Args:
            text (str): The button label.
            parent (Optional[QtWidgets.QWidget]): The parent widget.
            tone (str): Determines the text color. 'primary' for important
                actions (e.g., Save), 'neutral' for standard actions.
        """
        super().__init__(text, parent)

        # UI/UX configuration
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setFlat(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        # Resolve text color based on tone
        _text_color = {
            "neutral": "rgba(60, 78, 96, 230)",
            "primary": "rgba(20, 120, 180, 235)",
        }.get(tone, "rgba(60, 78, 96, 230)")

        self.setStyleSheet(f"""
            QPushButton {{
                border: none;
                background: transparent;
                color: {_text_color};
                font-size: 12px;
                font-weight: 600;
                padding: 4px 10px;
                border-radius: 7px;
            }}
            QPushButton:hover {{
                background: rgba(120, 140, 160, 45);
            }}
            QPushButton:pressed {{
                background: rgba(120, 140, 160, 80);
            }}
            QPushButton:disabled {{
                color: rgba(120, 135, 150, 120);
                background: transparent;
            }}
        """)
