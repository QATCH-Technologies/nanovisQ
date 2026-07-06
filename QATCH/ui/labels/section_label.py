from typing import Optional
from PyQt5 import QtWidgets

from QATCH.ui.styles.fonts import FONT_SANS_SEMIBOLD
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css


class SectionHeader(QtWidgets.QLabel):
    """Soft, muted section header matching the app's flat control system.

    Replaces the heavy blue HeaderLabel pills inside the advanced panel
    with quiet uppercase text, so the panel reads as clean grouped
    sections rather than a grid of competing colored bars. Colors come
    from the "flat_*" tokens (see QATCH.ui.styles.tokens) and update
    automatically on light/dark theme changes.
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
        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_theme()

    def _apply_theme(self) -> None:
        tok = ThemeManager.instance().tokens()
        self.setStyleSheet(
            f"QLabel {{ color: {tok_css(tok['flat_text_muted'])}; "
            f"font-family: '{FONT_SANS_SEMIBOLD}'; font-size: 10px; "
            "letter-spacing: 1px; background: transparent; "
            "border: none; padding: 0px 1px; }"
        )
