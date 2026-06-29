from typing import Optional
from PyQt5 import QtWidgets


class DeviceConfigLabel(QtWidgets.QLabel):
    """Title label for the device-config perspective that stays banner-compatible.

    This widget maps legacy "banner" strings to a more modern, friendly UI
    rendering. It maintains full API compatibility by intercepting `setText`
    and `text` methods, ensuring that existing string parsing logic (like
    `endswith` checks) in the parent application continues to function
    uninterrupted.

    Attributes:
        _PREFIX (str): The legacy prefix expected by external logic.
        _DISPLAY_BASE (str): The human-readable title shown in the UI.
    """

    _PREFIX: str = "Configuration Editor for Device"
    _DISPLAY_BASE: str = "Device Configuration"

    def __init__(self, text: str = "", parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initializes the label and applies the initial banner text."""
        super().__init__(parent)
        self._raw_text: str = ""
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802
        """Sets the raw banner text and updates the rendered display.

        Overrides the base method to store the raw input string for legacy
        API compatibility while rendering a formatted version to the UI.

        Args:
            text (str): The raw string expected by the legacy API.
        """
        self._raw_text = text if text is not None else ""
        super().setText(self._render(self._raw_text))

    def text(self) -> str:
        """Returns the raw banner string for legacy API compliance.

        Returns:
            str: The raw text string, preserving any device handles or
                metadata expected by external parsing logic.
        """
        return self._raw_text

    def _render(self, raw: str) -> str:
        """Maps a raw banner string to a friendly, styled visible title.

        Parses the device handle from the legacy prefix and formats the
        output using HTML/CSS spans to create a "chip" effect for the
        device handle.

        Args:
            raw (str): The raw banner string to parse.

        Returns:
            str: An HTML-formatted string suitable for display.
        """
        handle: str = ""
        if raw.startswith(self._PREFIX):
            handle = raw[len(self._PREFIX) :].strip()

        # Build the styled base title
        base = (
            f"<span style='color: rgba(28,40,52,235); font-size:14px; "
            f"font-weight:bold;'>{self._DISPLAY_BASE}</span>"
        )

        if handle:
            # Render handle as a stylized UI chip
            chip = (
                "<span style='"
                "background: rgba(10,163,230,38); "
                "color: rgba(12,110,160,255); "
                "border: 1px solid rgba(10,163,230,120); "
                "border-radius: 7px; "
                "padding: 1px 7px; "
                "font-size: 12px; font-weight: bold; "
                "letter-spacing: 0.5px;"
                f"'>&nbsp;{handle}&nbsp;</span>"
            )
            # Use an em-space (&#8195;) for clean horizontal spacing
            return f"{base}&#8195;{chip}"

        return base
