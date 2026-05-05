"""
ui_login.py

Authentication and user session management interface for nanovisQ.

This module implements a modern, glass-morphism inspired login system. It
orchestrates the user's entry into the application by managing:
    - Secure credential input via custom animated GlassLineEdit widgets.
    - User authentication and role-based access control (RBAC) via UserProfiles.
    - Password recovery workflows and interface navigation via a sliding panel.
    - Persistent user preferences (e.g., 'Remember Me') utilizing QSettings.
    - Hardware status monitoring, specifically tracking and displaying the
      system Caps Lock state.

The module is designed to operate as a standalone modal or a primary entry
point, providing fluid transitions and high-fidelity visual feedback through
manual painting and property-based animations.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-05-05
"""

import os
from typing import Callable, Optional
from PyQt5 import QtCore, QtGui, QtWidgets
from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserProfiles, UserRoles
from QATCH.core.constants import Constants
from QATCH.ui.widgets.floating_message_badge_widget import FloatingMessageBadgeWidget

_CARD_W: int = 320
_INPUT_H: int = 34
_BTN_H: int = 32
_PAGE_H: int = 400

_P_SIGNIN = 0
_P_RECOVER = 1

import os
from typing import Optional
from PyQt5 import QtCore, QtGui, QtWidgets
from QATCH.common.architecture import Architecture


class LoginCentralWidget(QtWidgets.QWidget):
    """Central widget providing a blurred, frosted backdrop for the login card.

    This widget manages the 'frosted glass' background effect. It can either
    capture a live snapshot of the main application window or load a static
    branded asset. The backdrop is pre-blurred and cached as a QPixmap to
    ensure high-performance rendering during window resizes and animations.

    The visual stack consists of:
        1. A blurred source Pixmap (Wall-to-wall).
        2. A neutral light-gray overlay tint (~24% opacity) to provide 'frost'.
        3. The login card and UI elements (added as children).
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initializes the central widget and configures paint attributes."""
        super().__init__(parent)
        self._blurred: Optional[QtGui.QPixmap] = None

        # Optimization: Tell Qt this widget handles its own background painting
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    def capture_backdrop(self, run_window: QtWidgets.QMainWindow) -> None:
        """Captures a live snapshot of the provided window, blurs it, and repaints.

        This allows the login screen to feel integrated into the current app state
        by using the actual UI as the background.

        Args:
            run_window (QtWidgets.QMainWindow): The window to grab for the backdrop.
        """
        raw: QtGui.QPixmap = run_window.grab()

        if not self.size().isEmpty():
            raw = raw.scaled(
                self.size(),
                QtCore.Qt.KeepAspectRatioByExpanding,
                QtCore.Qt.SmoothTransformation,
            )

        self._blurred = self._apply_blur(raw, radius=22)
        self.update()

    def set_background_pixmap(self) -> None:
        """Loads the default branded background image, blurs it, and updates UI.

        This method scales the icon-sourced background to fill the current widget
        geometry while maintaining aspect ratio and applying the frost effect.
        """
        path = os.path.join(Architecture.get_path(), "QATCH", "icons", "background.png")
        pixmap = QtGui.QPixmap(path)

        if not self.size().isEmpty() and not pixmap.isNull():
            pixmap = pixmap.scaled(
                self.size(),
                QtCore.Qt.KeepAspectRatioByExpanding,
                QtCore.Qt.SmoothTransformation,
            )

        self._blurred = self._apply_blur(pixmap, radius=22)
        self.update()

        # Explicitly trigger children to repaint to ensure glass-card
        # transparency stays synchronized with the new background.
        for child in self.findChildren(QtWidgets.QWidget):
            child.update()

    @staticmethod
    def _apply_blur(source: QtGui.QPixmap, radius: int = 22) -> QtGui.QPixmap:
        """Applies a Gaussian-style blur to a QPixmap.

        Utilizes QGraphicsScene and QGraphicsBlurEffect to perform a
        high-quality blur of the source image.

        Args:
            source (QtGui.QPixmap): The raw pixmap to blur.
            radius (int): The intensity of the blur. Defaults to 22.

        Returns:
            QtGui.QPixmap: The processed, blurred pixmap.
        """
        scene = QtWidgets.QGraphicsScene()
        item = QtWidgets.QGraphicsPixmapItem(source)

        blur = QtWidgets.QGraphicsBlurEffect()
        blur.setBlurRadius(radius)
        blur.setBlurHints(QtWidgets.QGraphicsBlurEffect.QualityHint)
        item.setGraphicsEffect(blur)

        scene.addItem(item)

        out = QtGui.QPixmap(source.size())
        out.fill(QtCore.Qt.transparent)

        p = QtGui.QPainter(out)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        scene.render(p, source=QtCore.QRectF(item.boundingRect()))
        p.end()

        return out

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Renders the backdrop and the frosting overlay.

        Args:
            event (QtGui.QPaintEvent): The paint event triggered by Qt.
        """
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        if self._blurred:
            p.drawPixmap(self.rect(), self._blurred, self._blurred.rect())
        else:
            # Fallback gradient shown during initial load/capture
            grad = QtGui.QLinearGradient(0, 0, self.width(), self.height())
            grad.setColorAt(0.0, QtGui.QColor(0xD8, 0xE6, 0xF0))
            grad.setColorAt(1.0, QtGui.QColor(0xEE, 0xF4, 0xF8))
            p.fillRect(self.rect(), QtGui.QBrush(grad))
        p.fillRect(self.rect(), QtGui.QColor(238, 243, 247, 62))
        p.end()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Handles widget resizing by re-generating the blurred backdrop.

        Args:
            event (QtGui.QResizeEvent): The resize event triggered by Qt.
        """
        super().resizeEvent(event)
        if self._blurred is not None:
            # Regenerate background to match new dimensions
            self.set_background_pixmap()


class GlassCard(QtWidgets.QFrame):
    """A custom frame that renders a glassmorphism effect via backdrop sampling.

    This widget creates the illusion of translucent glass by sampling the blurred
    pixmap from a LoginCentralWidget. It maps its local coordinates to the
    backdrop's coordinate space to 'slice' the background perfectly.

    The rendering pipeline follows these steps:
        1. Create a rounded-rectangle clip path.
        2. Sample and translate the blurred backdrop slice.
        3. Apply a neutral 'frost' tint and a faint cool blue identifier tint.
        4. Render a top-down white shimmer gradient.
        5. Draw a multi-layered border (muted outer stroke + inner highlight rim).

    Attributes:
        _RADIUS (float): The corner radius for the rounded rectangle.
        _backdrop (LoginCentralWidget): Reference to the widget providing the
            blurred source image.
    """

    _RADIUS: float = 22.0

    def __init__(
        self,
        backdrop: LoginCentralWidget,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initializes the glass card and configures transparency attributes.

        Args:
            backdrop (LoginCentralWidget): The source widget for background blur.
            parent (QtWidgets.QWidget, optional): The parent widget.
        """
        super().__init__(parent)
        self._backdrop = backdrop

        # Prevent the base class from painting opaque backgrounds
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Executes the custom glassmorphism painting pipeline.

        This method manually handles all background and border rendering. It
        specifically avoids calling super().paintEvent() to prevent Qt Style
        Sheets from overwriting the translucent effects with opaque colors.
        """
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        rect_f = QtCore.QRectF(self.rect())
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)
        blurred = getattr(self._backdrop, "_blurred", None)
        if blurred is not None and not blurred.isNull():
            origin = self.mapTo(self._backdrop, QtCore.QPoint(0, 0))
            p.save()
            p.translate(-origin.x(), -origin.y())
            p.drawPixmap(self._backdrop.rect(), blurred, blurred.rect())
            p.fillRect(self._backdrop.rect(), QtGui.QColor(238, 243, 247, 62))
            p.restore()
        else:
            grad = QtGui.QLinearGradient(0, 0, 0, self.height())
            grad.setColorAt(0.0, QtGui.QColor(0xD4, 0xE6, 0xF4))
            grad.setColorAt(1.0, QtGui.QColor(0xE6, 0xF2, 0xFA))
            p.fillRect(self.rect(), QtGui.QBrush(grad))

        # Primary white glass tint
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 70))
        p.fillRect(self.rect(), QtGui.QColor(210, 225, 240, 35))
        shimmer = QtGui.QLinearGradient(0, 0, 0, 80)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 60))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))
        p.setClipping(False)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(120, 160, 200, 135), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 175), 1.0))
        p.drawRoundedRect(
            rect_f.adjusted(1.5, 1.5, -1.5, -1.5),
            self._RADIUS - 1.5,
            self._RADIUS - 1.5,
        )

        p.end()


class SlidingPanel(QtWidgets.QWidget):
    """A horizontal viewport that slides between multiple UI pages.

    This widget acts as a clipping container. It maintains a fixed width (the
    viewport) and hosts an inner container that can be significantly wider.
    By animating the horizontal position of the inner container, the panel
    reveals different pages (e.g., Sign In, Password Recovery) with a fluid
    sliding transition.

    Attributes:
        _pw (int): The fixed width of a single page/viewport.
        _pages (list[QtWidgets.QWidget]): Ordered list of widgets added as pages.
        _inner (QtWidgets.QWidget): The horizontal host for all page widgets.
        _anim (Optional[QtCore.QPropertyAnimation]): The current active transition.
    """

    def __init__(self, page_width: int, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initializes the slider with a fixed viewport width.

        Args:
            page_width (int): The width in pixels for the visible area.
            parent (QtWidgets.QWidget, optional): The parent widget.
        """
        super().__init__(parent)
        self._pw = page_width
        self._anim: Optional[QtCore.QPropertyAnimation] = None
        self._pages: List[QtWidgets.QWidget] = []
        self.setFixedWidth(page_width)
        self.setContentsMargins(0, 0, 0, 0)
        self.setAutoFillBackground(False)
        self._inner = QtWidgets.QWidget(self)
        self._inner.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self._inner.setAutoFillBackground(False)
        self._inner.move(0, 0)

    def add_page(self, widget: QtWidgets.QWidget) -> int:
        """Adds a widget as a new page in the slider.

        Args:
            widget (QtWidgets.QWidget): The widget to be used as a page.

        Returns:
            int: The index of the added page.
        """
        idx = len(self._pages)
        widget.setParent(self._inner)
        widget.setFixedWidth(self._pw)
        widget.move(idx * self._pw, 0)
        self._pages.append(widget)
        return idx

    def finalize(self, fallback_height: int = 300) -> None:
        """Synchronizes the geometry of the inner container and all child pages.

        This should be called once after all pages have been added and the
        main window layout has settled. It ensures the inner host is tall
        enough to prevent vertical clipping.

        Args:
            fallback_height (int, optional): Height to use if the current
                widget height is zero. Defaults to 300.
        """
        if not self._pages:
            return

        h = self.height() if self.height() > 0 else fallback_height

        # Ensure every page fills the vertical space
        for i, p in enumerate(self._pages):
            p.setFixedSize(self._pw, h)
            p.move(i * self._pw, 0)

        # Expand inner host to accommodate the total number of pages
        self._inner.setFixedSize(len(self._pages) * self._pw, h)

    def slide_to(self, page_idx: int, duration: int = 360) -> None:
        """Animates the panel to reveal the page at the specified index.

        Args:
            page_idx (int): The index of the page to show.
            duration (int, optional): Animation length in milliseconds.
                Defaults to 360.
        """
        if page_idx >= len(self._pages) or page_idx < 0:
            return

        target_x = -page_idx * self._pw

        # Stop existing animation to prevent conflicting positional updates
        if self._anim and self._anim.state() == QtCore.QPropertyAnimation.Running:
            self._anim.stop()

        self._anim = QtCore.QPropertyAnimation(self._inner, b"pos")
        self._anim.setStartValue(self._inner.pos())
        self._anim.setEndValue(QtCore.QPoint(target_x, 0))
        self._anim.setDuration(duration)
        self._anim.setEasingCurve(QtCore.QEasingCurve.InOutQuart)

        # DeleteWhenStopped helps with memory management in long-running apps
        self._anim.start(QtCore.QPropertyAnimation.DeleteWhenStopped)


class GlassLineEdit(QtWidgets.QLineEdit):
    """A custom QLineEdit with a translucent glass aesthetic and shimmer animations.

    This widget overrides the standard QPaintEvent to manually render a frosted
    glass background and a dynamic border. On focus, a 'shimmer' sweep effect
    animates across the border. Visual states for 'error' and 'focused' are
    handled through manual painting rather than QSS.

    Attributes:
        _shimmer_t (float): Animation progress normalized from 0.0 to 1.0.
        _focused (bool): Internal state tracking focus for rendering logic.
        _in_error (bool): Internal state tracking validation failure for rendering logic.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initializes the line edit with custom styles and animation timers."""
        super().__init__(parent)
        self._shimmer_t: float = 0.0
        self._focused: bool = False
        self._in_error: bool = False
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(12)
        self._timer.timeout.connect(self._tick)

        self.setFrame(False)
        self.setAutoFillBackground(False)
        self.setStyleSheet("""
            QLineEdit {
                background: transparent;
                border: none;
                padding: 0px 15px;
                color: rgba(38, 48, 58, 230);
                font-size: 10pt;
                selection-background-color: rgba(10, 163, 230, 60);
                selection-color: rgba(0, 0, 0, 255);
            }
            QLineEdit QToolButton { 
                background: transparent; 
                border: none; 
            }
            QLineEdit QToolButton:hover {
                background: rgba(255, 255, 255, 55);
                border-radius: 12px;
            }
        """)

    def set_error(self, on: bool) -> None:
        """Toggles the visual error state of the widget.

        Args:
            on (bool): If True, the widget paints with a red 'error' theme.
        """
        if on != self._in_error:
            self._in_error = on
            self.update()

    def _tick(self) -> None:
        """Increments the shimmer progress and triggers a repaint."""
        self._shimmer_t = min(1.0, self._shimmer_t + 0.022)
        self.update()
        if self._shimmer_t >= 1.0:
            self._timer.stop()

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        """Handles focus entry: resets error and triggers the shimmer animation."""
        super().focusInEvent(event)
        self._focused = True
        self._in_error = False
        self._shimmer_t = 0.0
        self._timer.start()
        self.update()

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        """Handles focus exit: stops the shimmer animation."""
        super().focusOutEvent(event)
        self._focused = False
        self._timer.stop()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Manually paints the glass background and animated border.

        This method executes before the standard QLineEdit paint event to
        draw the underlying 'glass' container. It then delegates text and
        cursor rendering back to the base class.
        """
        # Calculate geometry
        radius = self.height() / 2.0
        r = radius - 1.0
        rect = QtCore.QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        if self._in_error:
            fill = QtGui.QColor(255, 220, 220, 68)
        elif self._focused:
            fill = QtGui.QColor(255, 255, 255, 100)
        else:
            fill = QtGui.QColor(255, 255, 255, 58)

        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QBrush(fill))
        p.drawRoundedRect(rect, r, r)
        p.setBrush(QtCore.Qt.NoBrush)

        if self._in_error:
            p.setPen(QtGui.QPen(QtGui.QColor(210, 55, 55, 150), 1.0))

        elif self._focused:
            t = self._shimmer_t
            width = float(self.width())
            grad = QtGui.QLinearGradient(0.0, 0.0, width, 0.0)

            if t < 1.0:
                spread = 0.30
                accent_color = QtGui.QColor(185, 218, 248, 115)  # Soft blue
                peak_color = QtGui.QColor(255, 255, 255, 240)  # Bright white

                grad.setColorAt(0.0, accent_color)

                # Pre-peak
                pre = max(0.0, t - spread)
                if pre > 0.0:
                    grad.setColorAt(pre, accent_color)

                # Peak
                grad.setColorAt(max(0.0, t - spread * 0.12), peak_color)
                grad.setColorAt(min(1.0, t + spread * 0.12), peak_color)

                # Post-peak
                post = min(1.0, t + spread)
                if post < 1.0:
                    grad.setColorAt(post, accent_color)

                grad.setColorAt(1.0, accent_color)
            else:
                settled_color = QtGui.QColor(185, 218, 248, 130)
                grad.setColorAt(0.0, settled_color)
                grad.setColorAt(1.0, settled_color)

            p.setPen(QtGui.QPen(QtGui.QBrush(grad), 1.5))

        else:
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 105), 1.0))

        p.drawRoundedRect(rect, r, r)
        p.end()
        super().paintEvent(event)


class _UserInfoProxy:
    """Compatibility shim: routes legacy user_info label calls to the floating badge."""

    def __init__(self, badge: FloatingMessageBadgeWidget, card) -> None:
        self._badge = badge
        self._card = card
        self.suppress = False  # Add a suppression flag

    def clear(self) -> None:
        self._badge.clear()

    def setText(self, text: str) -> None:
        if self.suppress:
            return  # Ignore legacy text updates during auth success

        if text and text.strip():
            self._badge.show_message(text.strip(), is_error=False, parent_widget=self._card)
        else:
            self._badge.clear()

    def text(self) -> str:
        return ""

    def __getattr__(self, name):
        return lambda *a, **kw: None


class UILogin:
    """Manages the lifecycle and state of the nanovisQ login interface.

    This class is responsible for constructing the glass-morphism UI, handling
    user authentication flows, and managing transitions between the login card
    and the password recovery interface. It utilizes a sliding panel architecture
    to provide a fluid, single-window user experience.

    Attributes:
        parent (QtWidgets.QMainWindow): The main application window used for
            post-login role assignment and UI state transitions.
        caps_lock_on (bool): Internal tracker for the hardware Caps Lock state.
        password_shown (bool): Toggle state for password masking (echo mode).
        floating_badge (FloatingMessageBadge): A frameless overlay used to
            display non-blocking alerts, errors, and status notifications.
        user_info (_UserInfoProxy): A compatibility shim that routes legacy
            text updates to the modern floating badge system.
        remember_me_cb (QtWidgets.QCheckBox): Toggle for persisting the
            username across application sessions via QSettings.

    Primary Interactions:
        - Authentication: Validates input, calls the UserProfiles auth backend,
          and configures the application mode (Run vs. Analyze) upon success.
        - Navigation: Handles horizontal sliding transitions between the
          Sign In and Password Recovery pages.
        - Persistence: Reads and writes user preferences to the system registry
          using the QSettings framework.
        - Feedback: Provides real-time hardware status (Caps Lock) and
          interactive error feedback (shake animations and floating badges).
    """

    def setup_ui(
        self,
        MainWindow5: QtWidgets.QMainWindow,
        parent: QtWidgets.QMainWindow,
    ) -> None:
        """Initializes all UI elements for the simplified login window.

        Constructs the main window properties, central widget, and the sliding panel
        pages (Sign In and Password Recovery). Connects all necessary UI signals to
        their respective slots and sets up session management.

        Args:
            MainWindow5 (QtWidgets.QMainWindow): The primary window object to build the UI onto.
            parent (QtWidgets.QMainWindow): The parent application window, used for state transitions
                and role assignments upon successful authentication.
        """
        self.parent = parent
        self.caps_lock_on = False
        self._login_window: QtWidgets.QMainWindow = MainWindow5  # used by transition

        global _CARD_W, _INPUT_H, _PAGE_H, _BTN_H
        _CARD_W = 320
        _INPUT_H = 34
        _BTN_H = 32
        _PAGE_H = 400

        # Window
        MainWindow5.setObjectName("MainWindow5")
        MainWindow5.setMinimumSize(QtCore.QSize(800, 500))
        MainWindow5.resize(800, 500)
        MainWindow5.setTabShape(QtWidgets.QTabWidget.Rounded)

        # Central widget
        self.centralwidget = LoginCentralWidget(MainWindow5)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        MainWindow5.setCentralWidget(self.centralwidget)

        MainWindow5.setStyleSheet("""
            QWidget {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            }
            #centralwidget { background: transparent; border: none; }
            #slidingPanel  { background: transparent; border: none; }
            #loginCard     { background: transparent; border: none; }
        """)

        # Sign In
        signInPage = QtWidgets.QWidget()
        signInPage.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        si = QtWidgets.QVBoxLayout(signInPage)
        si.setContentsMargins(28, 26, 28, 22)
        si.setSpacing(10)
        si.setAlignment(QtCore.Qt.AlignTop)

        # Logo
        logoLabel = QtWidgets.QLabel()
        logoLabel.setAlignment(QtCore.Qt.AlignCenter)
        logoLabel.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        logo_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "qatch-icon.png")
        logo_pm = QtGui.QPixmap(logo_path)

        if not logo_pm.isNull():
            logo_pm = logo_pm.scaled(
                54, 54, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            logoLabel.setPixmap(logo_pm)
            logoLabel.setFixedSize(54, 54)

        si.addWidget(logoLabel, alignment=QtCore.Qt.AlignCenter)
        si.addSpacing(2)

        # Title
        siTitle = QtWidgets.QLabel("Sign In")
        siTitle.setObjectName("cardTitle")
        siTitle.setAlignment(QtCore.Qt.AlignCenter)
        siTitle.setStyleSheet("color: rgba(50, 55, 65, 220); font-size: 12pt; font-weight: 700;")
        si.addWidget(siTitle)
        si.addSpacing(4)

        self.user_username = GlassLineEdit()
        self.user_username.setObjectName("user_username")
        self.user_username.setFixedHeight(_INPUT_H)
        self.user_username.setPlaceholderText("Username")
        self.user_username.textChanged.connect(lambda _: self.user_username.set_error(False))
        si.addWidget(self.user_username)
        self.user_password = GlassLineEdit()
        self.user_password.setObjectName("user_password")
        self.user_password.setFixedHeight(_INPUT_H)
        self.user_password.setPlaceholderText("Password")
        self.user_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.user_password.textChanged.connect(lambda _: self.user_password.set_error(False))
        self.user_password.returnPressed.connect(self.action_sign_in)
        self.user_username.returnPressed.connect(self.user_password.setFocus)
        self.user_initials = self.user_username

        # Password field
        self.user_password = GlassLineEdit()
        self.user_password.setObjectName("user_password")
        self.user_password.setFixedHeight(_INPUT_H)
        self.user_password.setPlaceholderText("Password")
        self.user_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.user_password.textChanged.connect(lambda _: self.user_password.set_error(False))
        self.user_password.returnPressed.connect(self.action_sign_in)
        self.user_password.installEventFilter(MainWindow5)
        self.user_username.installEventFilter(MainWindow5)
        MainWindow5.installEventFilter(MainWindow5)

        # Eye action
        self.visibleIcon = QtGui.QIcon(
            os.path.join(Architecture.get_path(), "QATCH", "icons", "eye-on.svg")
        )
        self.hiddenIcon = QtGui.QIcon(
            os.path.join(Architecture.get_path(), "QATCH", "icons", "eye-off.svg")
        )
        self.password_shown = False
        self.togglepasswordAction = self.user_password.addAction(
            self.visibleIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.togglepasswordAction.triggered.connect(self.on_toggle_password_Action)
        si.addWidget(self.user_password)

        # Caps Lock indicator — always occupies its row; text is blank when off
        self.caps_indicator = QtWidgets.QLabel("")
        self.caps_indicator.setObjectName("capsIndicator")
        self.caps_indicator.setAlignment(QtCore.Qt.AlignCenter)
        self.caps_indicator.setFixedHeight(16)
        self.caps_indicator.setStyleSheet(
            "color: rgba(200, 130, 30, 235); font-size: 7.5pt; font-weight: 600;"
        )
        si.addWidget(self.caps_indicator, alignment=QtCore.Qt.AlignCenter)

        # Remember Me Toggle
        self.remember_me_cb = QtWidgets.QCheckBox("Remember me")
        self.remember_me_cb.setObjectName("rememberMe")
        self.remember_me_cb.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.remember_me_cb.setStyleSheet("""
            QCheckBox {
                color: rgba(60, 70, 80, 220);
                font-size: 9pt;
                font-weight: 500;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border-radius: 4px;
                border: 1px solid rgba(150, 160, 170, 180);
                background: rgba(255, 255, 255, 120);
            }
            QCheckBox::indicator:hover {
                border: 1px solid rgba(10, 163, 230, 150);
            }
            QCheckBox::indicator:checked {
                background: rgba(10, 163, 230, 210);
                border: 1px solid rgba(10, 150, 210, 255);
            }
        """)

        # Wrap in a horizontal layout to keep it left-aligned cleanly
        rm_layout = QtWidgets.QHBoxLayout()
        rm_layout.setContentsMargins(4, 0, 0, 0)
        rm_layout.addWidget(self.remember_me_cb)
        rm_layout.addStretch()
        si.addLayout(rm_layout)
        si.addSpacing(6)

        # Sign In button
        self.sign_in_btn = QtWidgets.QPushButton("Sign In")
        self.sign_in_btn.setObjectName("signInBtn")
        self.sign_in_btn.setFixedHeight(_BTN_H)
        self.sign_in_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.sign_in_btn.setStyleSheet(self._make_primary_btn_style())
        self.sign_in_btn.clicked.connect(self.action_sign_in)
        si.addWidget(self.sign_in_btn)

        si.addStretch()

        # Forgot Password link
        forgotPasswordLbl = QtWidgets.QLabel("Forgot Password?")
        forgotPasswordLbl.setObjectName("forgotPassword")
        forgotPasswordLbl.setAlignment(QtCore.Qt.AlignCenter)
        forgotPasswordLbl.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        forgotPasswordLbl.setStyleSheet("""
            QLabel {
                color: rgba(100, 110, 120, 180);
                font-size: 9pt; font-weight: 500;
            }
            QLabel:hover { color: rgba(60, 60, 60, 220); text-decoration: underline; }
        """)
        forgotPasswordLbl.mousePressEvent = lambda _e: self._slide_to(_P_RECOVER)
        si.addWidget(forgotPasswordLbl)

        # Forgot Password
        recoverPage = QtWidgets.QWidget()
        recoverPage.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        rec = QtWidgets.QVBoxLayout(recoverPage)
        rec.setContentsMargins(28, 18, 28, 18)
        rec.setSpacing(10)
        rec.setAlignment(QtCore.Qt.AlignTop)

        back_icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "left-arrow.svg")

        rec.addWidget(
            self._make_back_btn(
                "Back to Sign In",
                lambda: self._slide_to(_P_SIGNIN),
                align="left",
                icon_path=back_icon_path,
            ),
            alignment=QtCore.Qt.AlignLeft,
        )

        recTitle = QtWidgets.QLabel("Reset Password")
        recTitle.setObjectName("recoverTitle")
        recTitle.setAlignment(QtCore.Qt.AlignCenter)
        recTitle.setStyleSheet("color: rgba(50, 55, 65, 220); font-size: 12pt; font-weight: 700;")
        rec.addWidget(recTitle)

        recInfo = QtWidgets.QLabel("Contact your administrator to reset your password.")
        recInfo.setAlignment(QtCore.Qt.AlignCenter)
        recInfo.setWordWrap(True)
        recInfo.setStyleSheet("color: rgba(100, 110, 120, 220); font-size: 8.5pt;")
        rec.addWidget(recInfo)
        # TODO: Currently placeholder workflow to contact admin to reset
        self.recoverEmail = GlassLineEdit()
        self.recoverEmail.setObjectName("recoverEmail")
        self.recoverEmail.setPlaceholderText("Email Address")
        self.recoverEmail.setFixedHeight(_INPUT_H)
        self.recoverEmail.setVisible(False)
        rec.addWidget(self.recoverEmail)

        self.sendResetBtn = QtWidgets.QPushButton("Send Reset Link")
        self.sendResetBtn.setObjectName("sendResetBtn")
        self.sendResetBtn.setFixedHeight(_BTN_H)
        self.sendResetBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.sendResetBtn.setStyleSheet(self._make_primary_btn_style())
        self.sendResetBtn.clicked.connect(self._on_send_reset)
        self.sendResetBtn.setVisible(False)
        rec.addWidget(self.sendResetBtn)

        self.recoverStatus = QtWidgets.QLabel("")
        self.recoverStatus.setAlignment(QtCore.Qt.AlignCenter)
        self.recoverStatus.setWordWrap(True)
        self.recoverStatus.setFixedHeight(34)
        self.recoverStatus.setStyleSheet(
            "color: rgba(46, 139, 87, 220); font-size: 8.5pt; font-weight: 500;"
        )
        rec.addWidget(self.recoverStatus)
        rec.addStretch()

        # Sliding panel
        self._slider = SlidingPanel(_CARD_W)
        self._slider.setObjectName("slidingPanel")
        self._slider.setStyleSheet(
            "QWidget#slidingPanel { background: transparent; border: none; }"
        )

        # Page layout: [0=SignIn, 1=Recover]
        self._slider.add_page(signInPage)
        self._slider.add_page(recoverPage)
        self._slider.setFixedHeight(_PAGE_H)

        def _init_slider() -> None:
            self._slider.finalize(_PAGE_H)
            # Position instantly at Sign In (page 0) with no animation
            self._slider._inner.move(0, 0)

        QtCore.QTimer.singleShot(0, _init_slider)

        self.loginCard = GlassCard(self.centralwidget)
        self.loginCard.setObjectName("loginCard")
        self.loginCard.setAttribute(QtCore.Qt.WA_StyledBackground, False)
        self.loginCard.setContentsMargins(0, 0, 0, 0)

        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(44)
        shadow.setOffset(0, 10)
        shadow.setColor(QtGui.QColor(15, 40, 70, 90))
        self.loginCard.setGraphicsEffect(shadow)

        card_vbox = QtWidgets.QVBoxLayout(self.loginCard)
        card_vbox.setContentsMargins(0, 0, 0, 0)
        card_vbox.setSpacing(0)
        card_vbox.addWidget(self._slider)

        v_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        v_layout.setAlignment(QtCore.Qt.AlignCenter)
        v_layout.addStretch(2)
        v_layout.addSpacing(20)
        v_layout.addWidget(self.loginCard, alignment=QtCore.Qt.AlignCenter)
        v_layout.addStretch(3)

        # Floating badge for error / info messages
        self.floating_badge = FloatingMessageBadgeWidget(
            MainWindow5,
            os.path.join(Architecture.get_path(), "QATCH", "icons", "clear.svg"),
        )

        # user_info proxy — legacy code calls .user_info.clear() / .setText(); route to badge
        self.user_info = _UserInfoProxy(self.floating_badge, self.loginCard)

        self._sessionTimer = QtCore.QTimer()
        self._sessionTimer.setSingleShot(True)
        self._sessionTimer.timeout.connect(self.check_user_session)
        self._sessionTimer.setInterval(1000 * 60 * 60)

        # Initialize settings and load remembered user
        self._settings = QtCore.QSettings("QATCH", "nanovisQ")
        self._load_remembered_user()

        QtCore.QTimer.singleShot(0, lambda: self.centralwidget.set_background_pixmap())

    def _make_input_style(self, error: bool = False) -> str:
        """Generates the QSS stylesheet for the GlassLineEdit components.

        Provides a translucent, frosted glass aesthetic for text inputs. If the
        error flag is set, it shifts the color palette to a high-visibility red
        to indicate invalid or missing user input.

        Args:
            error (bool, optional): Whether to generate the error-state styling.
                Defaults to False.

        Returns:
            str: The formatted Qt Style Sheet (QSS) string.
        """
        r = _INPUT_H // 2

        if error:
            return f"""
                QLineEdit {{
                    background-color: rgba(255, 230, 230, 160);
                    border: 1.5px solid rgba(230, 50, 50, 200);
                    border-radius: {r}px;
                    padding: 0px 15px;
                    color: rgba(200, 30, 30, 255);
                    font-size: 10pt;
                    selection-background-color: rgba(230, 50, 50, 80);
                }}
                QLineEdit:focus {{
                    background-color: rgba(255, 245, 245, 255);
                    border: 2px solid rgba(255, 50, 50, 255);
                }}
                QLineEdit QToolButton {{ 
                    background: transparent; 
                    border: none; 
                }}
                QLineEdit QToolButton:hover {{ 
                    background: rgba(230, 50, 50, 20); 
                    border-radius: 12px; 
                }}
            """

        return f"""
            QLineEdit {{
                background-color: rgba(255, 255, 255, 72);
                border: 1px solid rgba(255, 255, 255, 130);
                border-radius: {r}px;
                padding: 0px 15px;
                color: rgba(40, 50, 60, 230);
                font-size: 10pt;
                selection-background-color: rgba(10, 163, 230, 80);
            }}
            QLineEdit:hover {{
                background-color: rgba(255, 255, 255, 110);
                border-color: rgba(255, 255, 255, 200);
            }}
            QLineEdit:focus {{
                background-color: rgba(255, 255, 255, 145);
                border: 1.5px solid rgba(10, 163, 230, 140);
            }}
            QLineEdit QToolButton {{ 
                background: transparent; 
                border: none; 
            }}
            QLineEdit QToolButton:hover {{ 
                background: rgba(255, 255, 255, 60); 
                border-radius: 12px; 
            }}
        """

    def _make_primary_btn_style(self) -> str:
        """Generates the QSS stylesheet for the primary action buttons.

        Creates a rich, semi-transparent blue gradient button with custom borders
        to simulate physical depth and lighting. Includes states for hover,
        pressed, and disabled interactions.

        Returns:
            str: The formatted Qt Style Sheet (QSS) string.
        """
        r = _BTN_H // 2
        return f"""
            QPushButton {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(45, 165, 250, 210),
                    stop:1 rgba(15, 125, 210, 190)
                );
                border-top:    1px solid rgba(255, 255, 255, 100);
                border-left:   1px solid rgba(255, 255, 255, 50);
                border-right:  1px solid rgba(0, 80, 150, 50);
                border-bottom: 1px solid rgba(0, 80, 150, 80);
                border-radius: {r}px;
                color: rgba(255, 255, 255, 255);
                font-size: 9.5pt;
                font-weight: 600;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(65, 185, 255, 240),
                    stop:1 rgba(25, 145, 230, 220)
                );
                border-top: 1px solid rgba(255, 255, 255, 140);
            }}
            QPushButton:pressed {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(15, 115, 200, 220),
                    stop:1 rgba(5, 95, 160, 200)
                );
                border-top: 1px solid rgba(255, 255, 255, 40);
                border-bottom: 1px solid rgba(255, 255, 255, 20);
            }}
            QPushButton:disabled {{
                background: rgba(150, 170, 190, 100);
                color: rgba(255, 255, 255, 150);
                border: 1px solid rgba(255, 255, 255, 40);
            }}
        """

    @staticmethod
    def _make_back_btn(
        text: str, callback: Callable, align: str = "left", icon_path: str = ""
    ) -> QtWidgets.QPushButton:
        """Creates a styled, transparent text button used for backward navigation.

        Args:
            text (str): The label to display on the button (e.g., "← Back to Sign In").
            callback (typing.Callable): The function or method to execute when the button is clicked.
            align (str, optional): The text alignment within the button's bounding box
                ("left", "right", or "center"). Defaults to "left".

        Returns:
            QtWidgets.QPushButton: The configured, styled navigation button instance.
        """
        btn = QtWidgets.QPushButton(text)
        btn.setObjectName("backBtn")
        btn.setFixedHeight(24)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # Add the icon if a path is provided
        if icon_path and os.path.exists(icon_path):
            btn.setIcon(QtGui.QIcon(icon_path))
            btn.setIconSize(QtCore.QSize(14, 14))

        btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: rgba(100, 110, 120, 200);
                border: none;
                font-size: 8.5pt;
                font-weight: 600;
                text-align: {align};
                padding-left: 2px;
            }}
            QPushButton:hover {{ 
                color: rgba(60, 60, 60, 220); 
                text-decoration: underline; 
            }}
        """)
        btn.clicked.connect(callback)
        return btn

    def _slide_to(self, page_idx: int) -> None:
        """Animates the sliding panel to a specified page and cleans up stale state.

        Transitions the UI to the requested layout index. If the user is returning to the
        primary Sign In page, this method automatically clears any text inputs, disables
        any active error states, or resets status messages left behind on the secondary
        pages (e.g., the Password Recovery page).

        Args:
            page_idx (int): The index of the target page to display (e.g., `_P_SIGNIN` or `_P_RECOVER`).
        """
        self._slider.slide_to(page_idx)

        if page_idx == _P_SIGNIN:
            # Reset the password recovery page to its default state
            self.recoverEmail.clear()
            self.recoverStatus.clear()
            self.sendResetBtn.setEnabled(True)

    def on_toggle_password_Action(self) -> None:
        """Toggles the visibility of the password text.

        Switches the password field between 'Password' mode (masked) and 'Normal'
        mode (plain text). This method also updates the trailing action icon
        to reflect the current visibility state.
        """
        # Determine new state
        self.password_shown = not self.password_shown

        # Select mode and icon based on new state
        mode = QtWidgets.QLineEdit.Normal if self.password_shown else QtWidgets.QLineEdit.Password
        icon = self.hiddenIcon if self.password_shown else self.visibleIcon

        # Apply changes
        self.user_password.setEchoMode(mode)
        self.togglepasswordAction.setIcon(icon)

    def update_caps_lock_state(self, caps_lock_on: bool) -> None:
        """Updates the UI indicator based on the system Caps Lock state.

        This method is typically invoked by the event filter. It updates the
        internal state tracking and modifies the visibility of the warning
        label to provide immediate feedback to the user.

        Args:
            caps_lock_on (bool): The current physical state of the Caps Lock key.
        """
        self.caps_lock_on = caps_lock_on
        self.caps_indicator.setText("Caps Lock is On" if caps_lock_on else "")

    def _on_send_reset(self) -> None:
        """Validates the recovery email and triggers the reset log/action.

        Checks for a non-empty email address. If valid, it provides positive
        feedback to the user and disables the submission button to prevent
        duplicate requests. If invalid, it displays a localized error message
        in red.
        """
        email = self.recoverEmail.text().strip()

        # Style constants for status messaging
        COLOR_ERR = "rgba(200, 30, 30, 230)"
        COLOR_SUC = "rgba(46, 139, 87, 220)"
        BASE_STYLE = "font-size: 8.5pt; font-weight: 500;"

        if not email:
            self.recoverStatus.setStyleSheet(f"color: {COLOR_ERR}; {BASE_STYLE}")
            self.recoverStatus.setText("Please enter your email address.")
            return

        # Log request and provide success feedback
        Log.i(f"Password reset requested for: {email}")

        self.recoverStatus.setStyleSheet(f"color: {COLOR_SUC}; {BASE_STYLE}")
        self.recoverStatus.setText(
            "If an account exists for that address,\na reset link has been sent."
        )
        self.sendResetBtn.setEnabled(False)

    def action_sign_in(self) -> None:
        """Executes the authentication flow and manages the transition to the main application.

        This method validates the input fields, performs a credential check against the
        UserProfiles database, and handles the post-login state. If successful, it:
          1. Synchronizes the 'Remember me' preference.
          2. Configures the parent window's user roles and UI labels.
          3. Determines the initial software mode (Run vs. Analyze) based on permissions.
          4. Starts the session inactivity timer.

        If authentication fails, it triggers error animations and displays a
        'Password invalid' message.
        """
        username = self.user_username.text().strip()

        if not username:
            self._shake_widget(self.user_username)
            self.user_username.set_error(True)
            self.floating_badge.show_message(
                "Please enter your username", is_error=True, parent_widget=self.loginCard
            )
            return

        if not self.user_password.text():
            self._shake_widget(self.user_password)
            self.user_password.set_error(True)
            self.floating_badge.show_message(
                "Password required", is_error=True, parent_widget=self.loginCard
            )
            return
        pwd = self.user_password.text()
        authenticated, filename, params = UserProfiles.auth(username, pwd, UserRoles.ANY)

        if authenticated:
            self.floating_badge.clear()
            self.user_info.suppress = True
            # Persistence
            self._save_remembered_user(username)

            Log.i(f"Welcome, {params[0]}! Role: {params[2].name}.")
            name, init, role = params[0], params[1], params[2].value
            self._sessionTimer.start()
        else:
            name, init, role = None, None, 0
        self._clear_credentials()

        if name is not None:
            controls = self.parent.ControlsWin
            controls.username.setText(f"User: {name}")
            controls.userrole = UserRoles(role)
            controls.signinout.setText("&Sign Out")
            controls.ui1.tool_User.setText(name)
            self.parent.AnalyzeProc.tool_User.setText(name)

            # Restrict password management for non-admins
            if controls.userrole != UserRoles.ADMIN:
                controls.manage.setText("&Change Password...")
            has_capture_perm = UserProfiles().check(controls.userrole, UserRoles.CAPTURE)
            if has_capture_perm:
                self.parent.MainWin.ui0._set_run_mode(None)
            else:
                self.parent.MainWin.ui0._set_analyze_mode(None)

            # Cleanup temporary download attributes
            if hasattr(self.parent, "url_download"):
                delattr(self.parent, "url_download")

            self.user_info.suppress = False
        else:
            self.user_info.suppress = False
            self.error_invalid()

    def _clear_credentials(self) -> None:
        """Securely clears sensitive input fields and resets visibility.

        Wipes the password field text and ensures the masking (echo mode)
        is restored if the user had previously toggled the password visibility 'on'.
        """
        self.user_password.clear()
        if self.password_shown:
            self.on_toggle_password_Action()

    def clear_form(self) -> None:
        """Performs a full reset of the login interface to its initial state.

        This is the primary cleanup method called during application startup,
        on Escape key presses, or upon user sign-out. It re-shows the hidden
        login window, clears all input fields, resets error states, hides
        the Caps Lock indicator, and slides the panel back to the Sign In page.
        """
        self._login_window.show()
        self.loginCard.show()

        # Clear inputs and error styling
        self.user_username.clear()
        self.user_username.set_error(False)
        self.user_password.set_error(False)
        self._clear_credentials()

        # Reset indicators and navigation
        self.update_caps_lock_state(False)
        self._slide_to(_P_SIGNIN)

    def error_invalid(self, message: str = "Invalid Credentials") -> None:
        """Triggers the UI sequence for a failed authentication attempt.

        Applies red error styling to the password field, executes a shake
        animation, clears the invalid input, and displays a floating
        error badge with the provided message.

        Args:
            message (str, optional): The error text to display in the floating badge.
                Defaults to "Invalid Credentials".
        """
        self.user_password.set_error(True)
        self._shake_widget(self.user_password)
        self.user_password.clear()
        self.user_password.setFocus()
        self.floating_badge.show_message(message, is_error=True, parent_widget=self.loginCard)

    def error_loggedout(self) -> None:
        """Displays a logout notification and resets the password field state."""
        self.user_password.clear()
        self.user_password.set_error(False)
        self.show_signout_message()

    def error_expired(self) -> None:
        """Displays a session expiration warning and resets the password field state."""
        self.user_password.clear()
        self.user_password.set_error(False)
        self.floating_badge.show_message(
            "Your session has expired", is_error=True, parent_widget=self.loginCard
        )

    def show_signout_message(self) -> None:
        """Triggers a floating badge notification confirming the user has signed out."""
        self.floating_badge.show_message(
            "You have been signed out.", is_error=True, parent_widget=self.loginCard
        )

    def check_user_session(self) -> None:
        valid, _ = UserProfiles().session_info()
        if not valid:
            if self.parent.ControlsWin.userrole != UserRoles.NONE:
                Log.w("User session has expired.")
                self.parent.ControlsWin.set_user_profile()
        else:
            Log.d("User session valid at hourly check.")
            self._sessionTimer.start()

    def _load_remembered_user(self) -> None:
        """Loads the remembered username from system settings on startup.

        Retrieves the 'Remember me' toggle state from QSettings. If previously enabled,
        it restores the saved username and automatically shifts keyboard focus to the
        password input field for a quicker sign-in experience.
        """
        remembered = self._settings.value("login/remember_me", False, type=bool)
        self.remember_me_cb.setChecked(remembered)

        if remembered:
            saved_user = self._settings.value("login/username", "", type=str)
            if saved_user:
                self.user_username.setText(saved_user)
                self.user_password.setFocus()

    def _save_remembered_user(self, username: str) -> None:
        """Saves or clears the remembered username based on the toggle state.

        Synchronizes the current state of the 'Remember me' checkbox with QSettings.
        If checked, the username is stored for future sessions. If unchecked,
        any previously stored username is wiped from the system's settings registry.

        Args:
            username (str): The authenticated username to store if the toggle is active.
        """
        is_remembered = self.remember_me_cb.isChecked()
        self._settings.setValue("login/remember_me", is_remembered)

        if is_remembered:
            self._settings.setValue("login/username", username)
        else:
            self._settings.remove("login/username")

    def retranslateUi(self, MainWindow5: QtWidgets.QMainWindow) -> None:
        """Sets the window titles and localizable strings for the interface.

        Configures the primary window title using application constants and
        loads the software icon from the localized asset path.

        Args:
            MainWindow5 (QtWidgets.QMainWindow): The window instance to update.
        """
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "qatch-icon.png")
        MainWindow5.setWindowIcon(QtGui.QIcon(icon_path))
        title_text = f"{Constants.app_title} {Constants.app_version} - Login"
        MainWindow5.setWindowTitle(_translate("MainWindow5", title_text))

    def _shake_widget(self, widget: QtWidgets.QWidget) -> None:
        """Executes a rapid horizontal jiggle animation to provide error feedback.

        Uses a QPropertyAnimation to manipulate the 'pos' property of the widget.
        The animation uses a series of decaying keyframes to simulate a physical
        shake that settles back at the original position.

        Args:
            widget (QtWidgets.QWidget): The target widget (e.g., LineEdit) to animate.
        """
        if not widget or not widget.isVisible():
            return

        self._shake_anim = QtCore.QPropertyAnimation(widget, b"pos")
        self._shake_anim.setDuration(380)

        base = widget.pos()
        self._shake_anim.setKeyValueAt(0.0, base)
        self._shake_anim.setKeyValueAt(0.1, base + QtCore.QPoint(-6, 0))
        self._shake_anim.setKeyValueAt(0.3, base + QtCore.QPoint(6, 0))
        self._shake_anim.setKeyValueAt(0.5, base + QtCore.QPoint(-4, 0))
        self._shake_anim.setKeyValueAt(0.7, base + QtCore.QPoint(4, 0))
        self._shake_anim.setKeyValueAt(0.9, base + QtCore.QPoint(-2, 0))
        self._shake_anim.setKeyValueAt(1.0, base)

        self._shake_anim.start(QtCore.QPropertyAnimation.DeleteWhenStopped)
