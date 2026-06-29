from typing import Optional
from PyQt5 import QtWidgets


class SectionHeader(QtWidgets.QLabel):
    """Soft, muted section header mirroring the account dropdown's typography.

    Replaces the heavy blue HeaderLabel pills inside the advanced panel
    with quiet uppercase gray text, so the panel reads as clean grouped
    sections rather than a grid of competing colored bars.
    """

    def __init__(self, text: str = "", parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initializes the SectionHeader with muted, uppercase styling.

        The text is automatically converted to uppercase to maintain visual
        consistency across the UI. Styles are applied via a transparent
        stylesheet to ensure it blends seamlessly into the panel background.

        Args:
            text (str): The label text to display. Defaults to an empty string.
            parent (Optional[QtWidgets.QWidget]): The parent widget.
                Defaults to None.
        """
        super().__init__(text.upper(), parent)
        self.setStyleSheet(
            "QLabel { color: rgba(70, 90, 110, 200); font-size: 10px; "
            "font-weight: bold; letter-spacing: 1px; background: transparent; "
            "border: none; padding: 0px 1px; }"
        )
