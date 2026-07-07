"""
glass_warning_label.py

This module defines the `GlassWarningLabel` widget: a calm, glass-styled
informational banner used in place of harshly colored inline warning text.

Three severities are supported - "info" (default, calm blue-gray), "warning"
(amber) and "danger" (red) - all resolved from the app's `flat_*` tokens so
the banner tracks light/dark theme changes automatically instead of being
hardcoded to one fixed palette.

The widget keeps a QLabel-like `setText`/`text` API for drop-in use at
existing call sites.
"""

from __future__ import annotations

from typing import Optional

import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as QtWidgets

from QATCH.ui.styles.theme_manager import ThemeManager

# severity -> (text/border token, weak-background token)
_SEVERITY_TOKENS = {
    "info": ("flat_accent", "flat_accent_weak"),
    "warning": ("flat_warning", "flat_warning_weak"),
    "danger": ("flat_error", "flat_error_weak"),
}


class GlassWarningLabel(QtWidgets.QWidget):
    """Calm informational banner for inline messaging.

    Renders as a soft glass strip tinted per `severity`, with an optional
    leading icon. Keeps a QLabel-like `setText` so existing call sites still
    work.

    Attributes:
        icon_lbl (QtWidgets.QLabel): The label widget responsible for displaying
            the optional leading icon.
        text_lbl (QtWidgets.QLabel): The label widget containing the main
            informational text.
    """

    _RADIUS: float = 6.0

    def __init__(
        self,
        text: str = "",
        icon_path: str = "",
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        severity: str = "info",
    ) -> None:
        """Initializes the GlassWarningLabel.

        Args:
            text: The informational text to display in the banner.
            icon_path: The file path to the leading icon. If provided, the
                icon is loaded and displayed.
            parent: The parent widget.
            severity: One of "info" (default), "warning", or "danger" -
                selects which `flat_*` token pair colors the banner.
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
        """Switches the banner's color scheme at runtime."""
        severity = severity if severity in _SEVERITY_TOKENS else "info"
        if severity != self._severity:
            self._severity = severity
            self._apply_text_style()
            self.update()

    def set_icon(self, icon_path: str) -> None:
        """Sets the leading icon for the warning label.

        Args:
            icon_path: The file path to the icon image to be displayed.
        """
        pix = QtGui.QPixmap(icon_path)
        if not pix.isNull():
            self.icon_lbl.setPixmap(pix)
            self.icon_lbl.show()

    def setText(self, text: str) -> None:  # noqa: N802
        """Sets the informational text of the banner (QLabel-API parity)."""
        self.text_lbl.setText(text)

    def text(self) -> str:
        """Returns the current informational text (QLabel-API parity)."""
        return self.text_lbl.text()

    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_text_style()
        self.update()

    def _apply_text_style(self) -> None:
        tok = ThemeManager.instance().tokens()
        text_key, _ = _SEVERITY_TOKENS[self._severity]
        r, g, b, a = tok[text_key]
        self.text_lbl.setStyleSheet(
            "QLabel { color: rgba(%d, %d, %d, %d); font-size: 11px; "
            "font-weight: normal; background: transparent; border: none; }" % (r, g, b, a)
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Renders the glass background: a rounded shape tinted per
        `severity`, a subtle top shimmer, and a hairline border - all
        resolved fresh from the active theme's tokens.
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
