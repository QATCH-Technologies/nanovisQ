"""
QATCH.ui.components.qatch_warning_label.py

This module defines the `QATCHWarningLabel` widget: a calm, styled
informational banner used in place of harshly colored inline warning text.

Three severities are supported - "info" (default, calm blue-gray), "warning"
(amber) and "danger" (red) - all resolved from the app's `flat_*` tokens so
the banner tracks light/dark theme changes automatically instead of being
hardcoded to one fixed palette.

The widget keeps a QLabel-like `setText`/`text` API for drop-in use at
existing call sites.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-05
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.styles.theme_manager import ThemeManager

_SEVERITY_TOKENS = {
    "info": ("flat_accent", "flat_accent_weak"),
    "warning": ("flat_warning", "flat_warning_weak"),
    "danger": ("flat_error", "flat_error_weak"),
}


class QATCHWarningLabel(QtWidgets.QWidget):
    """Display a compact informational banner with severity-based styling.

    The widget provides a soft, style inline message with an optional
    leading icon. Its appearance is controlled by a severity level, which
    selects the corresponding color tokens from the application's theme.

    The class provides a QLabel-like interface through methods such as
    `setText` while retaining a composite widget structure for the icon,
    text, and themed background.

    Attributes:
        icon_lbl: QLabel used to display the optional leading icon. The label
            is hidden when no icon is configured.
        text_lbl: QLabel containing the informational message text.
    """

    _RADIUS: float = 6.0

    def __init__(
        self,
        text: str = "",
        icon_path: str = "",
        parent: QtWidgets.QWidget | None = None,
        *,
        severity: str = "info",
    ) -> None:
        """Initialize the warning label.

        Args:
            text: Informational text to display in the banner.
            icon_path: Path to the optional leading icon. If provided, the
                icon is loaded and displayed in the icon slot.
            parent: Optional parent widget.
            severity: Severity level used to determine the banner's theme
                colors. Supported values are `"info"`, `"warning"`, and
                `"danger"`. Invalid values default to `"info"`.
        """
        super().__init__(parent)
        self._severity = severity if severity in _SEVERITY_TOKENS else "info"
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
        self.text_lbl.setWordWrap(True)
        row.addWidget(self.text_lbl, 1, QtCore.Qt.AlignmentFlag.AlignVCenter)

        self._apply_text_style()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def set_severity(self, severity: str) -> None:
        """Update the banner's severity and corresponding color scheme.

        Invalid severity values are normalized to `"info"`. If the normalized
        severity differs from the current value, the banner's text styling is
        refreshed and the widget is scheduled for repainting.

        Args:
            severity: Severity level to apply. Supported values are defined by
                `_SEVERITY_TOKENS`. Invalid values default to `"info"`.

        Returns:
            None.
        """
        severity = severity if severity in _SEVERITY_TOKENS else "info"
        if severity != self._severity:
            self._severity = severity
            self._apply_text_style()
            self.update()

    def set_icon(self, icon_path: str) -> None:
        """Set the leading icon displayed by the warning label.

        The icon is loaded from the specified file path. If the image can be
        loaded successfully, it is assigned to the leading icon label and the
        label is made visible. Invalid or unreadable image paths are ignored.

        Args:
            icon_path: File path to the icon image to display.

        Returns:
            None.
        """
        pix = QtGui.QPixmap(icon_path)
        if not pix.isNull():
            self.icon_lbl.setPixmap(pix)
            self.icon_lbl.show()

    def setText(self, text: str) -> None:
        """Set the informational text displayed by the banner.

        Provides QLabel-compatible `setText` behavior by forwarding the
        supplied text to the internal text label.

        Args:
            text: Informational text to display in the banner.

        Returns:
            None.
        """
        self.text_lbl.setText(text)

    def text(self) -> str:
        """Return the current informational text displayed by the banner.

        Provides QLabel-compatible `text` behavior by returning the text from
        the internal text label.

        Returns:
            The informational text currently displayed in the banner.
        """
        return self.text_lbl.text()

    def _on_theme_changed(self, _mode: str) -> None:
        """Refresh the banner styling after a theme change.

        Reapplies the text and severity-dependent styling using the newly
        selected theme and schedules the widget for repainting.

        Args:
            _mode: Theme mode identifier supplied by the `themeChanged` signal.
                The value is not otherwise used by this handler.

        Returns:
            None.
        """
        self._apply_text_style()
        self.update()

    def _apply_text_style(self) -> None:
        """Apply the current theme and severity styling to the text label.

        Retrieves the active theme tokens and uses the color associated with the
        current severity level to style the banner's text. The text is rendered
        with a transparent background, no border, an 11-pixel font size, and
        normal font weight.

        Returns:
            None.
        """
        tok = ThemeManager.instance().tokens()
        text_key, _ = _SEVERITY_TOKENS[self._severity]
        r, g, b, a = tok[text_key]
        self.text_lbl.setStyleSheet(
            f"QLabel {{ color: rgba({r}, {g}, {b}, {a}); font-size: 11px; "
            f"font-weight: normal; background: transparent; border: none; }}"
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Render the severity-tinted background and border.

        Paints the banner using the currently active theme tokens and severity
        level. The background consists of a vertically graduated translucent
        severity tint with a subtle white top shimmer. A thin, semi-transparent
        border is then drawn around the rounded perimeter.

        The painting is clipped to the widget's rounded rectangle so that the
        background gradients remain contained within the banner's corners.

        Args:
            event: Qt paint event provided by Qt when the widget needs to be
                repainted. The event is not otherwise used by this implementation.

        Returns:
            None.
        """
        tok = ThemeManager.instance().tokens()
        text_key, weak_key = _SEVERITY_TOKENS[self._severity]
        weak = tok[weak_key]
        border_rgb = tok[text_key]

        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing)

        rect_f = QtCore.QRectF(self.rect())
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)
        grad = QtGui.QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, QtGui.QColor(weak[0], weak[1], weak[2], 90))
        grad.setColorAt(1.0, QtGui.QColor(weak[0], weak[1], weak[2], 60))
        p.fillRect(self.rect(), QtGui.QBrush(grad))

        # Top shimmer
        shimmer = QtGui.QLinearGradient(0, 0, 0, self.height() * 0.6)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 40))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Border
        p.setClipping(False)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(border_rgb[0], border_rgb[1], border_rgb[2], 110), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)

        p.end()
