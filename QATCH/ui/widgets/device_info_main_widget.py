"""
QATCH.ui.widgets.device_info_main_widget.py

Popup widget for Device Info configuration.

Provides the visual container used to display Device Info configuration
content as an anchored, frameless popup. The module defines a rounded
frosted-glass inner panel with custom painting, translucent backgrounds,
dual borders, and a drop shadow.

The popup supports dynamically injected content and automatically clamps
its position to the associated application window. It also closes when
the main window moves, resizes, or changes window state so the popup does
not become detached from its anchor or parent window.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-21
"""

import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as QtWidgets


class _DeviceInfoInnerPanel(QtWidgets.QWidget):
    """Render the surface used inside the device info popup.

    Provides a rounded, translucent panel with a frosted white base,
    top-edge shimmer, and dual border treatment. The panel is intended to
    serve as the visual surface inside `DeviceInfoMainWidget` while the
    parent widget provides the surrounding shadow margins.
    """

    _RADIUS: float = 10.0

    def __init__(self, parent=None) -> None:
        """Initialize the frosted-glass device information panel.

        Disables automatic background filling and the system background so the
        panel can render its custom translucent surface entirely through
        `paintEvent`.

        Args:
            parent (QtWidgets.QWidget, optional): The parent widget for the
                panel. Defaults to `None`.
        """
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint  panel surface.

        Renders the panel using antialiased painting with a rounded clipping
        path. The surface consists of a translucent white base, a subtle cool
        tint, a top-edge shimmer, and dual light and gray borders.

        Args:
            event (QtGui.QPaintEvent): The Qt paint event requesting the panel
                to be redrawn.
        """
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
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), _R, _R)
        p.setPen(QtGui.QPen(QtGui.QColor(200, 210, 220, 90), 1.0))
        p.drawRoundedRect(rect_f.adjusted(1.5, 1.5, -1.5, -1.5), _R - 1.5, _R - 1.5)

        p.end()


class DeviceInfoMainWidget(QtWidgets.QWidget):
    """Display a device information popup.

    Provides a popup container for the device information configuration
    content. The popup uses a translucent outer surface, shadow margins,
    a rounded inner panel, and a drop shadow to create the application's
    dropdown appearance.

    Content is injected dynamically through :meth:`set_content_widget`.
    The popup can be positioned relative to an anchor widget and is
    automatically closed when its associated main window moves, resizes,
    or changes window state.
    """

    _SHADOW_MARGIN_L = 22
    _SHADOW_MARGIN_T = 18
    _SHADOW_MARGIN_R = 22
    _SHADOW_MARGIN_B = 26

    def __init__(self, parent=None) -> None:
        """Initialize the device information popup.

        Configures the widget as a frameless popup with a translucent
        background, creates the inner panel, applies its drop shadow,
        and initializes the layout used for dynamically supplied
        content.

        Args:
            parent (QtWidgets.QWidget, optional): The parent widget for the
                popup. Defaults to `None`.
        """
        super().__init__(
            parent,
            QtCore.Qt.WindowType.Popup
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.NoDropShadowWindowHint,
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

        self._main_window = None

        # Outer container with shadow margins
        self._panel = _DeviceInfoInnerPanel(self)
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

    def set_content_widget(self, widget: QtWidgets.QWidget):
        """Add an existing device information widget to the popup.

        Inserts the supplied widget into the popup's content layout and makes
        the widget visible. The widget is not recreated or otherwise modified
        beyond being added to the popup.

        Args:
            widget (QtWidgets.QWidget): The device information container to
                display inside the popup.
        """
        self.content_layout.addWidget(widget)
        widget.show()

    def show_anchored_to(self, anchor: QtWidgets.QWidget, main_window=None) -> None:
        """Show the popup anchored to a widget within the application window.

        Sizes the popup to its current content, positions it relative to the
        bottom-right corner of the anchor, and clamps the resulting visible
        panel to the bounds of the anchor's top-level window or the supplied
        main window. If there is insufficient space below the anchor, the
        popup is positioned above it when possible.

        The main window is monitored while the popup is visible so that the
        popup can close automatically if the window moves, resizes, or changes
        window state.

        Args:
            anchor (QtWidgets.QWidget): The widget to which the popup should be
                visually anchored.
            main_window (QtWidgets.QWidget, optional): The main application
                window whose geometry should be used for clamping and whose
                movement or resizing should close the popup. Defaults to
                `None`.
        """
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
        """Close the popup when its associated main window changes geometry.

        Monitors the configured main window for resize, move, and window-state
        change events. When one of these events occurs, the popup is closed to
        prevent it from becoming detached from the application window.

        Args:
            watched (QtCore.QObject): The object that generated the event.
            event (QtCore.QEvent): The Qt event being processed.

        Returns:
            bool: The result of the base class event-filter implementation.
        """
        if watched is self._main_window and event.type() in (
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.Move,
            QtCore.QEvent.Type.WindowStateChange,
        ):
            self.close()
        return super().eventFilter(watched, event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Clean up the main-window event filter when the popup closes.

        Removes the popup's event filter from the associated main window and
        clears the stored window reference before allowing the base class to
        complete the close operation.

        Args:
            event (QtGui.QCloseEvent): The Qt close event being processed.
        """
        if self._main_window is not None:
            try:
                self._main_window.removeEventFilter(self)
            except Exception:
                pass
            self._main_window = None
        super().closeEvent(event)
