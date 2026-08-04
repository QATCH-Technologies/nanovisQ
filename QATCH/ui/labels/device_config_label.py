"""
QATCH.ui.labels.device_config_label.py

Device configuration banner label with legacy API compatibility.

This module provides :class:`DeviceConfigLabel`, a specialized
:class:`QtWidgets.QLabel` that presents a modern, themed device configuration
title while remaining fully compatible with the application's legacy banner
API.

The widget separates the displayed text from the underlying value exposed
through :meth:`text`. Legacy code continues to interact with the original
banner string (for example, checking prefixes or extracting device handles),
while users see a cleaner, HTML-formatted title with an optional themed
device "chip".

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-04
"""

from PyQt5 import QtWidgets

from QATCH.ui.styles.theme_manager import ThemeManager, tok_css


class DeviceConfigLabel(QtWidgets.QLabel):
    """A themed device configuration title with legacy banner compatibility.

    This widget acts as a drop-in replacement for a standard QLabel while
    preserving the application's historical banner string format. Calls to
    :meth:`setText` store the original string internally, allowing
    :meth:`text` to return the exact legacy value expected by existing code,
    while the visible label displays a cleaner, HTML-formatted title.

    When the banner contains a device handle, it is rendered as a themed
    "chip" beside the title using colors from the active application theme.

    Attributes:
        _PREFIX (str): Legacy banner prefix used to identify and extract the
            device handle.
        _DISPLAY_BASE (str): User-facing title displayed before the optional
            device handle.
    """

    _PREFIX: str = "Configuration Editor for Device"
    _DISPLAY_BASE: str = "Device Configuration"

    def __init__(self, text: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        """Initializes the themed configuration label.

        Stores the initial banner text, renders the visible title, and
        subscribes to theme change notifications so the embedded HTML colors
        remain synchronized with the active application theme.

        Args:
            text (str): Initial legacy banner string.
            parent (Optional[QtWidgets.QWidget]): Parent widget, if any.
        """
        super().__init__(parent)
        self._raw_text: str = ""
        self.setText(text)
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        """Updates the rendered HTML after a theme change.

        Rich-text colors are embedded directly into the generated HTML rather
        than supplied through Qt stylesheets. When the application theme
        changes, the current raw banner text is rendered again using the
        updated theme tokens.

        Args:
            _mode (str): Name of the newly activated theme. The value is not
                used directly because the active theme is retrieved from the
                ThemeManager singleton.
        """
        super().setText(self._render(self._raw_text))

    def setText(self, text: str) -> None:
        """Stores the raw banner text and updates the displayed title.

        Overrides :meth:`QtWidgets.QLabel.setText` to preserve the original
        banner string for legacy API compatibility while displaying a themed,
        HTML-formatted representation.

        Args:
            text (str): Legacy banner string, typically beginning with
                :attr:`_PREFIX` followed by an optional device handle.
        """
        self._raw_text = text if text is not None else ""
        super().setText(self._render(self._raw_text))

    def text(self) -> str:
        """Returns the original, unmodified banner string.

        This override preserves compatibility with existing application code
        that parses the banner text rather than the rendered HTML.

        Returns:
            str: The raw banner string supplied via :meth:`setText`.
        """
        return self._raw_text

    def _render(self, raw: str) -> str:
        """Converts a legacy banner string into themed rich text.

        The method extracts an optional device handle from the legacy banner
        prefix and generates an HTML representation.

        The generated HTML is recreated whenever the theme changes so that
        embedded colors always reflect the active theme.

        Args:
            raw (str): Raw legacy banner string.

        Returns:
            str: HTML-formatted rich text suitable for display by QLabel.
        """
        handle: str = ""
        if raw.startswith(self._PREFIX):
            handle = raw[len(self._PREFIX) :].strip()

        tok = ThemeManager.instance().tokens()
        base = (
            f"<span style='color: {tok_css(tok['flat_text'])}; font-size:14px; "
            f"font-weight:bold;'>{self._DISPLAY_BASE}</span>"
        )

        if handle:
            r, g, b, _ = tok["flat_accent"]
            chip = (
                "<span style='"
                f"background: {tok_css(tok['flat_accent_weak'])}; "
                f"color: {tok_css(tok['flat_accent'])}; "
                f"border: 1px solid rgba({r},{g},{b},120); "
                "border-radius: 7px; "
                "padding: 1px 7px; "
                "font-size: 12px; font-weight: bold; "
                "letter-spacing: 0.5px;"
                f"'>&nbsp;{handle}&nbsp;</span>"
            )
            return f"{base}&#8195;{chip}"

        return base
