"""
QATCH.ui.labels.section_label.py

Minimal themed section header widget.

This module provides :class:`SectionHeader`, a lightweight
:class:`QtWidgets.QLabel` used to visually separate groups of controls
without introducing heavy visual hierarchy. Unlike the application's
banner-style headers, section headers are rendered as muted uppercase text
that integrates naturally into flat control panels.

The widget automatically updates its appearance when the application theme
changes by applying colors from the active theme tokens.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-04
"""

from PyQt5 import QtWidgets

from QATCH.ui.styles.theme_manager import ThemeManager, tok_css
from QATCH.ui.styles.typography import FONT_SANS_STACK


class SectionHeader(QtWidgets.QLabel):
    """A muted uppercase label used to separate groups of controls.

    This widget provides a subtle alternative to the application's larger
    banner headers. It renders uppercase text using the application's
    semibold font and muted text color, creating visual grouping without
    competing for attention.

    Styling is derived from the active application theme and automatically
    refreshes whenever the theme changes.
    """

    def __init__(self, text: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        """Initializes the themed section header.

        The supplied text is converted to uppercase before being displayed
        to provide a consistent visual style throughout the application.
        Theme-aware styling is then applied and kept synchronized with future
        theme changes.

        Args:
            text (str): Section title to display.
            parent (QtWidgets.QWidget | None): Parent widget, if any.
        """
        super().__init__(text.upper(), parent)
        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        """Refreshes the widget styling after a theme change.

        The active theme colors are reapplied so the section header remains
        consistent with the rest of the user interface.

        Args:
            _mode (str): Name of the newly activated theme. The value is not
                used directly because the current theme is obtained from the
                ThemeManager singleton.
        """
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Applies the current theme styling to the label.

        Configures the widget using the application's muted text color,
        semibold font family, uppercase-friendly letter spacing, and a
        transparent background. The styling is regenerated whenever the
        application theme changes.

        The stylesheet intentionally avoids borders and decorative elements
        so the widget serves as a lightweight visual grouping cue rather than
        a prominent title.
        """
        tok = ThemeManager.instance().tokens()
        self.setStyleSheet(
            f"QLabel {{ color: {tok_css(tok['flat_text_muted'])}; "
            f"font-family: {FONT_SANS_STACK}; font-size: 10px; font-weight: 600; "
            "letter-spacing: 1px; background: transparent; "
            "border: none; padding: 0px 1px; }"
        )
