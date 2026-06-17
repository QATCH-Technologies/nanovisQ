import os

import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as QtWidgets

from QATCH.common.architecture import Architecture


class GlassWarningLabel(QtWidgets.QWidget):
    """Calm informational banner for the Advanced Settings dialog.

    Renders as a soft blue-gray glass strip with an optional leading icon and
    informational (not alarming) text. Replaces the old loud orange warning.
    Keeps a QLabel-like ``setText`` so existing call sites still work.
    """

    _RADIUS: float = 6.0

    def __init__(self, text: str = "", icon_path: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(10, 6, 10, 6)
        row.setSpacing(8)

        # Leading icon slot — populated when an icon path is provided.
        self.icon_lbl = QtWidgets.QLabel(self)
        self.icon_lbl.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.icon_lbl.setStyleSheet("background: transparent; border: none;")
        self.icon_lbl.setFixedSize(16, 16)
        self.icon_lbl.setScaledContents(True)
        if icon_path:
            self.set_icon(icon_path)
        else:
            self.icon_lbl.hide()  # TODO: provide warning/info SVG icon
        row.addWidget(self.icon_lbl, 0, QtCore.Qt.AlignVCenter)

        self.text_lbl = QtWidgets.QLabel(text, self)
        self.text_lbl.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.text_lbl.setStyleSheet(
            "QLabel { color: rgba(45, 75, 105, 220); font-size: 11px; "
            "font-weight: normal; background: transparent; border: none; }"
        )
        self.text_lbl.setWordWrap(True)
        row.addWidget(self.text_lbl, 1, QtCore.Qt.AlignVCenter)

    def set_icon(self, icon_path: str) -> None:
        pix = QtGui.QPixmap(icon_path)
        if not pix.isNull():
            self.icon_lbl.setPixmap(pix)
            self.icon_lbl.show()

    def setText(self, text: str) -> None:
        self.text_lbl.setText(text)

    def text(self) -> str:
        return self.text_lbl.text()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing)

        rect_f = QtCore.QRectF(self.rect())
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)

        # Soft, neutral blue-gray info glass.
        grad = QtGui.QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, QtGui.QColor(120, 165, 210, 40))
        grad.setColorAt(1.0, QtGui.QColor(95, 140, 190, 30))
        p.fillRect(self.rect(), QtGui.QBrush(grad))

        # Top shimmer
        shimmer = QtGui.QLinearGradient(0, 0, 0, self.height() * 0.6)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 40))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Hairline border
        p.setClipping(False)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(120, 160, 200, 110), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)

        p.end()


class _InfoIcon(QtWidgets.QLabel):
    """Info icon (SVG) that brightens on hover and shows a tooltip.

    Loads ``info.svg`` from the icons dir. The icon is rendered at reduced
    opacity at rest and full opacity on hover, giving a subtle hover effect
    without needing a second asset.
    """

    _D: int = 16  # display size

    def __init__(self, icon_path: str, tooltip: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(self._D, self._D)
        self.setCursor(QtCore.Qt.WhatsThisCursor)
        self.setAttribute(QtCore.Qt.WA_Hover, True)
        self.setStyleSheet("background: transparent; border: none;")
        self.setToolTip(tooltip)

        src = QtGui.QPixmap(icon_path)
        if not src.isNull():
            src = src.scaled(
                self._D,
                self._D,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        self._pix_rest = self._with_opacity(src, 0.55)
        self._pix_hover = self._with_opacity(src, 1.0)
        self.setPixmap(self._pix_rest)

    @staticmethod
    def _with_opacity(src: QtGui.QPixmap, opacity: float) -> QtGui.QPixmap:
        if src.isNull():
            return src
        out = QtGui.QPixmap(src.size())
        out.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(out)
        p.setOpacity(opacity)
        p.drawPixmap(0, 0, src)
        p.end()
        return out

    def enterEvent(self, event) -> None:
        if not self._pix_hover.isNull():
            self.setPixmap(self._pix_hover)

    def leaveEvent(self, event) -> None:
        if not self._pix_rest.isNull():
            self.setPixmap(self._pix_rest)


class _GlassAdvancedInnerPanel(QtWidgets.QWidget):
    """Inner glass-morphism panel for the advanced settings popup."""

    _RADIUS: float = 10.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        rect_f = QtCore.QRectF(self.rect())
        _R = self._RADIUS

        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, _R, _R)
        p.setClipPath(clip)

        # Frosted white base
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 235))
        p.fillRect(self.rect(), QtGui.QColor(228, 235, 241, 28))

        # Top shimmer
        shimmer = QtGui.QLinearGradient(0, 0, 0, 44)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 80))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Dual borders
        p.setClipping(False)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), _R, _R)
        p.setPen(QtGui.QPen(QtGui.QColor(200, 210, 220, 90), 1.0))
        p.drawRoundedRect(rect_f.adjusted(1.5, 1.5, -1.5, -1.5), _R - 1.5, _R - 1.5)

        p.end()


class AdvancedMainWidget(QtWidgets.QWidget):
    """Frosted-glass dropdown panel for the Advanced Settings.

    This widget owns the entire advanced-settings surface: the frosted popup
    shell, the orange warning banner, and the container that hosts the controls.
    Callers build the individual controls (combo boxes, buttons, etc.) in their
    own grid layout and hand that layout to :meth:`build_content`; this widget
    wraps it with the warning banner inside the owned ``content_container``.
    """

    _SHADOW_MARGIN_L = 22
    _SHADOW_MARGIN_T = 18
    _SHADOW_MARGIN_R = 22
    _SHADOW_MARGIN_B = 26

    _INFO_TEXT = (
        "These are advanced settings. Changes here affect device operation \u2014 "
        "adjust them only if you know what they do."
    )

    def __init__(self, parent=None) -> None:
        super().__init__(
            parent,
            QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint | QtCore.Qt.NoDropShadowWindowHint,
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

        self._main_window = None
        self.content_container = None  # built lazily by build_content()

        # Outer container with shadow margins
        self._panel = _GlassAdvancedInnerPanel(self)
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(
            self._SHADOW_MARGIN_L,
            self._SHADOW_MARGIN_T,
            self._SHADOW_MARGIN_R,
            self._SHADOW_MARGIN_B,
        )
        outer_layout.setSpacing(0)
        outer_layout.addWidget(self._panel)

        # Drop shadow
        shadow = QtWidgets.QGraphicsDropShadowEffect(self._panel)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 4)
        shadow.setColor(QtGui.QColor(0, 20, 40, 110))
        self._panel.setGraphicsEffect(shadow)

        # Inner layout for dynamic content
        self.content_layout = QtWidgets.QVBoxLayout(self._panel)
        self.content_layout.setContentsMargins(14, 14, 14, 14)

    # ------------------------------------------------------------------
    # Content assembly
    # ------------------------------------------------------------------
    @staticmethod
    def build_container(controls_layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        """Create the advanced container (title header + controls), hidden.

        The header is a title row in the top-left: a gear icon, the
        "Advanced Options" title, and an info icon that reveals the advanced
        usage message on hover. ``controls_layout`` holds the actual control
        widgets (created/wired by the caller); only the container, header, and
        wrapping layout are owned here.
        """
        icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")

        container = QtWidgets.QWidget()
        container.setWhatsThis(AdvancedMainWidget._INFO_TEXT)

        # ---- Title header (top-left): gear + title + info-on-hover ----
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(2, 0, 2, 0)
        header.setSpacing(8)

        gear = QtWidgets.QLabel()
        gear.setFixedSize(18, 18)
        gear.setScaledContents(True)
        gear.setStyleSheet("background: transparent; border: none;")
        _gear_pix = QtGui.QPixmap(os.path.join(icons_dir, "gear.svg"))
        if not _gear_pix.isNull():
            gear.setPixmap(_gear_pix)
        header.addWidget(gear, 0, QtCore.Qt.AlignVCenter)

        title = QtWidgets.QLabel("Advanced Options")
        title.setStyleSheet(
            "QLabel { color: rgba(28, 40, 52, 235); font-size: 14px; "
            "font-weight: bold; background: transparent; border: none; }"
        )
        header.addWidget(title, 0, QtCore.Qt.AlignVCenter)

        # Info icon (SVG) with a hover brighten effect; shows the advanced
        # usage message on hover. TODO: ensure info.svg exists in icons dir.
        info = _InfoIcon(
            os.path.join(icons_dir, "warning-circle.svg"),
            tooltip=AdvancedMainWidget._INFO_TEXT,
        )
        header.addWidget(info, 0, QtCore.Qt.AlignVCenter)

        header.addStretch()

        wrap = QtWidgets.QVBoxLayout(container)
        wrap.setSpacing(10)
        wrap.addLayout(header)
        wrap.addLayout(controls_layout)

        container.hide()  # shown when anchored
        return container

    def build_content(self, controls_layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        """Build and adopt a fresh advanced container around ``controls_layout``."""
        self.content_container = self.build_container(controls_layout)
        return self.content_container

    def set_content_widget(self, widget: QtWidgets.QWidget):
        """Injects an already-built container into the popup.

        Retained for backward compatibility / parity with the device-info
        popup. Prefer :meth:`build_content` for the advanced settings.
        """
        self.content_container = widget
        self.content_layout.addWidget(widget)
        widget.show()

    # ------------------------------------------------------------------
    # Popup lifecycle
    # ------------------------------------------------------------------
    @classmethod
    def toggle(cls, owner, anchor, controls_layout, main_window=None, attr="_advanced_popup"):
        """Open or close the advanced popup, owning the full lifecycle.

        ``owner`` stores the popup instance (typically the UIControls instance).
        The container is taken from ``owner._advanced_content_container`` if it
        was pre-built; otherwise it is built from ``controls_layout`` and cached
        there. If a popup is already visible it is closed (toggle) -> ``None``.
        """
        existing = getattr(owner, attr, None)
        if existing is not None and existing.isVisible():
            existing.close()
            return None

        popup = cls()

        cache_attr = "_advanced_content_container"
        container = getattr(owner, cache_attr, None)
        if container is None:
            container = popup.build_content(controls_layout)
            setattr(owner, cache_attr, container)
        else:
            popup.content_container = container

        popup.content_layout.addWidget(container)
        container.show()
        setattr(owner, attr, popup)
        popup.show_anchored_to(anchor, main_window=main_window)
        return popup

    def show_anchored_to(self, anchor: QtWidgets.QWidget, main_window=None) -> None:
        """Shows the popup pinned to the anchor, clamped to the main window."""
        self._main_window = main_window
        self.adjustSize()

        popup_w, popup_h = self.width(), self.height()
        anchor_br = anchor.mapToGlobal(QtCore.QPoint(anchor.width(), anchor.height()))

        x = anchor_br.x() + self._SHADOW_MARGIN_R - popup_w
        y = anchor_br.y() + 2 - self._SHADOW_MARGIN_T

        # Clamp logic to ensure it doesn't render off-screen/off-app
        top_level = anchor.window() if anchor is not None else None
        bounds = (
            top_level.geometry()
            if top_level
            else (self._main_window.geometry() if self._main_window else QtCore.QRect())
        )

        if not bounds.isNull():
            visible = QtCore.QRect(
                x + self._SHADOW_MARGIN_L,
                y + self._SHADOW_MARGIN_T,
                popup_w - self._SHADOW_MARGIN_L - self._SHADOW_MARGIN_R,
                popup_h - self._SHADOW_MARGIN_T - self._SHADOW_MARGIN_B,
            )
            if visible.right() > bounds.right():
                x -= visible.right() - bounds.right()
            if visible.left() < bounds.left():
                x += bounds.left() - visible.left()
            if visible.bottom() > bounds.bottom():
                anchor_top = anchor.mapToGlobal(QtCore.QPoint(0, 0)).y()
                y_above = anchor_top - 2 - popup_h + self._SHADOW_MARGIN_B
                if (y_above + self._SHADOW_MARGIN_T) >= bounds.top():
                    y = y_above
                else:
                    y -= visible.bottom() - bounds.bottom()

        self.move(x, y)
        if self._main_window is not None:
            self._main_window.installEventFilter(self)
        self.show()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        # Auto-close if the main window moves or resizes
        if watched is self._main_window and event.type() in (
            QtCore.QEvent.Resize,
            QtCore.QEvent.Move,
            QtCore.QEvent.WindowStateChange,
        ):
            self.close()
        return super().eventFilter(watched, event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._main_window is not None:
            try:
                self._main_window.removeEventFilter(self)
            except Exception:
                pass
            self._main_window = None
        super().closeEvent(event)
