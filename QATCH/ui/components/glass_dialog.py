"""QATCH.ui.components.glass_dialog

Glassmorphic modal dialog that matches the app's frosted-glass aesthetic.
Replaces QMessageBox across all PopUp static methods — zero call-site changes.

Renders a semi-transparent dim overlay over the root window with a centred
frosted glass card containing title, body text, optional expandable details,
and GlassPushButton actions.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.ui.components.glass_push_button import GlassPushButton
from QATCH.ui.styles.theme_manager import ThemeManager

# (button_label, GlassPushButton_variant, return_value)
ButtonSpec = Tuple[str, str, int]

_ICONS_DIR = os.path.join(Architecture.get_path(), "QATCH", "icons")

# Accent colours used for the icon badge when no SVG is supplied
_BADGE_COLORS = {
    "information": QtGui.QColor(10, 163, 230),
    "question":    QtGui.QColor(10, 163, 230),
    "warning":     QtGui.QColor(255, 193, 7),
    "critical":    QtGui.QColor(220, 53, 69),
}

# SVG filenames — drop these into QATCH/icons/ to get tinted icons
_ICON_FILES = {
    "information": "info-circle.svg",
    "question":    "question-circle.svg",
    "warning":     "warning-circle.svg",
    "critical":    "critical-circle.svg",
}

_CARD_W = 440
_CARD_RADIUS = 18.0
_HEADER_H = 52


def _tinted_icon(path: str, color: QtGui.QColor, size: int = 22) -> QtGui.QPixmap:
    src = QtGui.QIcon(path).pixmap(size, size)
    dst = QtGui.QPixmap(src.size())
    dst.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(dst)
    p.drawPixmap(0, 0, src)
    p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
    p.fillRect(dst.rect(), color)
    p.end()
    return dst


class _GlassCard(QtWidgets.QFrame):
    """Frosted glass card: same paint pipeline as PlotContainer."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        tok = ThemeManager.instance().tokens()
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        rf = QtCore.QRectF(self.rect())
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rf, _CARD_RADIUS, _CARD_RADIUS)
        p.setClipPath(clip)

        p.fillRect(self.rect(), QtGui.QColor(*tok["plot_glass_base"]))
        p.fillRect(self.rect(), QtGui.QColor(*tok["plot_glass_overlay"]))

        sh = QtGui.QLinearGradient(0, 0, 0, 50)
        sh.setColorAt(0, QtGui.QColor(*tok["plot_glass_shimmer_top"]))
        sh.setColorAt(0.5, QtGui.QColor(*tok["plot_glass_shimmer_mid"]))
        sh.setColorAt(1, QtGui.QColor(0, 0, 0, 0))
        p.fillRect(self.rect(), QtGui.QBrush(sh))

        vg = QtGui.QLinearGradient(0, self.height() - 30, 0, self.height())
        vg.setColorAt(0, QtGui.QColor(0, 0, 0, 0))
        vg.setColorAt(1, QtGui.QColor(*tok["plot_glass_vignette_end"]))
        p.fillRect(self.rect(), QtGui.QBrush(vg))

        p.setClipping(False)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(*tok["plot_glass_rim"]), 1.0))
        p.drawRoundedRect(rf.adjusted(0.5, 0.5, -0.5, -0.5), _CARD_RADIUS, _CARD_RADIUS)
        p.setPen(QtGui.QPen(QtGui.QColor(*tok["plot_glass_inset"]), 1.0))
        p.drawRoundedRect(
            rf.adjusted(1.5, 1.5, -1.5, -1.5), _CARD_RADIUS - 1.5, _CARD_RADIUS - 1.5
        )

        # Header divider
        p.setPen(QtGui.QPen(QtGui.QColor(*tok["plot_glass_header_line"]), 1.0))
        p.drawLine(0, _HEADER_H, self.width(), _HEADER_H)
        p.end()


class GlassDialog(QtWidgets.QDialog):
    """Modal frosted-glass dialog.

    Renders a semi-transparent dim overlay sized to the root window with a
    centred glass card containing title, message, optional expandable details,
    and one or more GlassPushButton actions.

    Args:
        parent:     Ancestor widget — used to find the root window for sizing.
        title:      Bold header text shown above the divider.
        message:    Body text shown below the divider.
        details:    Optional expandable text (shown in a scroll area).
        buttons:    List of (label, variant, return_value) tuples. The last
                    entry is the primary / default action (focused on open).
        icon_type:  One of "information", "question", "warning", "critical".
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        title: str,
        message: str,
        details: str = "",
        buttons: Optional[List[ButtonSpec]] = None,
        icon_type: str = "information",
    ) -> None:
        root = self._find_root(parent)
        super().__init__(root)

        self._result_value: int = 0
        self._icon_type = icon_type

        if buttons is None:
            buttons = [("OK", "primary", 1)]

        self.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.FramelessWindowHint)  # type: ignore[arg-type]
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self._build_ui(title, message, details, buttons)

        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _find_root(widget: Optional[QtWidgets.QWidget]) -> Optional[QtWidgets.QWidget]:
        # Prefer the currently active visible top-level window (e.g. ModeWindow)
        # over the parent chain, which may lead to a non-displayed QMainWindow.
        active = QtWidgets.QApplication.activeWindow()
        if active and active.isVisible():
            return active
        if not isinstance(widget, QtWidgets.QWidget):
            return None
        w = widget
        while w.parent() and isinstance(w.parent(), QtWidgets.QWidget):
            w = w.parent()
        return w

    def result_value(self) -> int:
        return self._result_value

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(
        self,
        title: str,
        message: str,
        details: str,
        buttons: List[ButtonSpec],
    ) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addStretch()

        card_row = QtWidgets.QHBoxLayout()
        card_row.setContentsMargins(0, 0, 0, 0)
        card_row.addStretch()

        self._card = _GlassCard(self)
        self._card.setFixedWidth(_CARD_W)

        card_v = QtWidgets.QVBoxLayout(self._card)
        card_v.setContentsMargins(0, 0, 0, 0)
        card_v.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────
        header_w = QtWidgets.QWidget()
        header_w.setFixedHeight(_HEADER_H)
        header_w.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        header_layout = QtWidgets.QHBoxLayout(header_w)
        header_layout.setContentsMargins(16, 0, 16, 0)
        header_layout.setSpacing(10)

        self._icon_label = QtWidgets.QLabel()
        self._icon_label.setFixedSize(24, 24)
        self._icon_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._refresh_icon()
        header_layout.addWidget(self._icon_label)

        self._title_label = QtWidgets.QLabel(title)
        self._title_label.setObjectName("GlassDialogTitle")
        self._title_label.setWordWrap(True)
        self._apply_title_style()
        header_layout.addWidget(self._title_label, 1)

        card_v.addWidget(header_w)

        # ── Body ──────────────────────────────────────────────────────
        body_w = QtWidgets.QWidget()
        body_w.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        body_layout = QtWidgets.QVBoxLayout(body_w)
        body_layout.setContentsMargins(20, 16, 20, 0)
        body_layout.setSpacing(10)

        self._msg_label = QtWidgets.QLabel(message)
        self._msg_label.setObjectName("GlassDialogMessage")
        self._msg_label.setWordWrap(True)
        self._msg_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )
        self._apply_body_style()
        body_layout.addWidget(self._msg_label)

        if details:
            body_layout.addWidget(self._build_details(details))

        card_v.addWidget(body_w)

        # ── Buttons ───────────────────────────────────────────────────
        btn_row = QtWidgets.QWidget()
        btn_row.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        btn_layout = QtWidgets.QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(20, 16, 20, 20)
        btn_layout.setSpacing(8)
        btn_layout.addStretch()

        for i, (label, variant, val) in enumerate(buttons):
            btn = GlassPushButton(label, variant=variant)
            btn.setFixedHeight(34)
            btn.setMinimumWidth(90)
            btn.clicked.connect(lambda checked=False, r=val: self._on_button(r))
            btn_layout.addWidget(btn)
            if i == len(buttons) - 1:
                btn.setDefault(True)
                btn.setFocus()

        card_v.addWidget(btn_row)

        card_row.addWidget(self._card)
        card_row.addStretch()
        outer.addLayout(card_row)
        outer.addStretch()

    def _build_details(self, details: str) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        container.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        lay = QtWidgets.QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        toggle_btn = GlassPushButton("Show Details ▾", variant="neutral")
        toggle_btn.setFixedHeight(28)
        toggle_btn.set_border_visible(False)
        lay.addWidget(toggle_btn)

        text_area = QtWidgets.QPlainTextEdit(details)
        text_area.setReadOnly(True)
        text_area.setMaximumHeight(100)
        text_area.setStyleSheet(
            "QPlainTextEdit {"
            "  background: rgba(0,0,0,40);"
            "  border: 1px solid rgba(255,255,255,60);"
            "  border-radius: 6px;"
            "  color: rgba(60,60,60,220);"
            "  font-size: 11px;"
            "}"
        )
        text_area.hide()
        lay.addWidget(text_area)

        def _toggle() -> None:
            visible = text_area.isVisible()
            text_area.setVisible(not visible)
            toggle_btn.setText("Hide Details ▴" if not visible else "Show Details ▾")
            self._card.adjustSize()

        toggle_btn.clicked.connect(_toggle)
        return container

    # ── Styling ───────────────────────────────────────────────────────────────

    def _apply_title_style(self) -> None:
        tok = ThemeManager.instance().tokens()
        r, g, b, _ = tok["plot_text_bright"]
        self._title_label.setStyleSheet(
            f"QLabel {{ color: rgb({r},{g},{b}); font-size: 14px; font-weight: 700; }}"
        )

    def _apply_body_style(self) -> None:
        tok = ThemeManager.instance().tokens()
        r, g, b, _ = tok["plot_text_normal"]
        self._msg_label.setStyleSheet(
            f"QLabel {{ color: rgb({r},{g},{b}); font-size: 13px; }}"
        )

    def _refresh_icon(self) -> None:
        svg = os.path.join(_ICONS_DIR, _ICON_FILES.get(self._icon_type, "info-circle.svg"))
        color = _BADGE_COLORS.get(self._icon_type, QtGui.QColor(10, 163, 230))
        if os.path.isfile(svg):
            pm = _tinted_icon(svg, color, size=22)
        else:
            pm = QtGui.QPixmap(22, 22)
            pm.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(pm)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.setBrush(QtGui.QBrush(color))
            painter.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
            painter.drawEllipse(2, 2, 18, 18)
            painter.end()
        self._icon_label.setPixmap(pm)

    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_title_style()
        self._apply_body_style()
        self._refresh_icon()
        self._card.update()

    # ── Events ────────────────────────────────────────────────────────────────

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        root = self._find_root(None)  # re-resolve at show time in case active window changed
        if isinstance(root, QtWidgets.QWidget):
            self.setGeometry(root.geometry())
        else:
            self.setGeometry(QtWidgets.QDesktopWidget().availableGeometry())
        super().showEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 140))
        p.end()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self._on_button(0)
            return
        super().keyPressEvent(event)

    def _on_button(self, result: int) -> None:
        self._result_value = result
        self.accept()
