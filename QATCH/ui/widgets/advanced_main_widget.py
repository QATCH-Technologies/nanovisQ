import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as QtWidgets


class GlassWarningLabel(QtWidgets.QLabel):
    """Orange glass warning banner for the Advanced Settings dialog."""

    _RADIUS: float = 4.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setStyleSheet(
            "QLabel { color: white; font-weight: bold; "
            "padding: 2px 6px; background: transparent; }"
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing)

        rect_f = QtCore.QRectF(self.rect())
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)

        # Warm orange glass gradient
        grad = QtGui.QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0.0, QtGui.QColor(210, 80, 0))
        grad.setColorAt(1.0, QtGui.QColor(255, 125, 20))
        p.fillRect(self.rect(), QtGui.QBrush(grad))

        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 40))

        shimmer = QtGui.QLinearGradient(0, 0, 0, self.height() * 0.65)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 50))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        p.setClipping(False)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(190, 80, 0, 140), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 120), 1.0))
        p.drawRoundedRect(
            rect_f.adjusted(1.5, 1.5, -1.5, -1.5),
            self._RADIUS - 1.5,
            self._RADIUS - 1.5,
        )

        p.end()
        super().paintEvent(event)


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

    _WARNING_TEXT = "These settings are for Advanced Users ONLY!"

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
        """Create the advanced container (warning banner + controls), hidden.

        This is a free-standing builder so the container can exist before any
        popup is shown. ``controls_layout`` holds the actual control widgets,
        which are created and wired by the caller; only the container, banner,
        and wrapping layout are owned here.
        """
        container = QtWidgets.QWidget()
        container.setWhatsThis(AdvancedMainWidget._WARNING_TEXT)

        warning = GlassWarningLabel(f"WARNING: {AdvancedMainWidget._WARNING_TEXT}")

        wrap = QtWidgets.QVBoxLayout(container)
        wrap.addWidget(warning)
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
