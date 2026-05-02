# ─────────────────────────────────────────────────────────────────────────────
# ui_login.py  —  QATCH nanovisQ Login Window
#
# Changelog (glass-morphism refresh):
#   • _LoginCentralWidget  – unchanged frosted-backdrop logic; added
#     set_background_pixmap() so callers can supply a custom splash image.
#   • _SlidingPanel        – new: two-page clip-and-animate container used for
#     the sign-in ↔ recover-password transition.
#   • UserSwitchDialog     – new: frameless profile-picker anchored near the
#     avatar button.  Emits user_selected(name, initials) or add_user_requested.
#   • UILogin              – restructured card to use _SlidingPanel; input
#     heights tightened; Sign-In button QATCH-blue; avatar + switch badge added;
#     Forgot-Password triggers the slide animation.
# ─────────────────────────────────────────────────────────────────────────────
import os
from typing import List, Optional, Tuple
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel
from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserProfiles, UserRoles
from QATCH.core.constants import Constants
from QATCH.ui.popUp import PopUp

# ── Card geometry constants ────────────────────────────────────────────────────
_CARD_W: int = 290  # loginCard fixed pixel width (drives sliding-panel clipping)
_AVATAR_D: int = 70  # user-avatar circle diameter
_BADGE_D: int = 22  # switch-user overlay badge diameter
_INPUT_H: int = 26  # uniform height for all QLineEdit fields
_BTN_H: int = 28  # height for action buttons
# Pre-computed page height so the card sizes correctly on first paint —
# before QTimer.singleShot(0) finalize() has had a chance to run.
# Formula: top_margin(22) + cardTitle(22) + sp(9) + avatarOuter(81) + sp(9)
#          + user_label(20) + sp(9) + password(26) + sp(9) + sign_in(28) + sp(9)
#          + user_info(16) + sp(9) + user_error(16) + sp(9) + forgotPwd(20)
#          + bottom_margin(18)  =  332  → use 342 for a small comfortable buffer.
_PAGE_H: int = 354  # +10 px spacer before forgotPassword + 2 px buffer


class _FloatingMessageBadge(QtWidgets.QWidget):
    """A frameless, glassy floating badge for alerts and info."""

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        # Make it a frameless tool window that stays on top but doesn't steal focus
        self.setWindowFlags(
            QtCore.Qt.Tool | QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)

        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # The actual styled label
        self.label = QtWidgets.QLabel("")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.label)

        self.hide()

    def show_message(
        self, text: str, is_error: bool = False, parent_widget: QtWidgets.QWidget = None
    ) -> None:
        """Updates the text and style, positions it, and shows it."""
        self.label.setText(text)

        # Style based on whether it's an error (red) or info (amber)
        if is_error:
            self.label.setStyleSheet("""
                QLabel {
                    background-color: rgba(255, 230, 230, 240);
                    border: 1.5px solid rgba(230, 50, 50, 200);
                    border-radius: 14px;
                    color: rgba(200, 30, 30, 255);
                    font-size: 8.5pt; font-weight: 600; padding: 4px 16px;
                }
            """)
        else:
            self.label.setStyleSheet("""
                QLabel {
                    background-color: rgba(255, 245, 230, 240);
                    border: 1.5px solid rgba(230, 150, 50, 200);
                    border-radius: 14px;
                    color: rgba(220, 130, 40, 255);
                    font-size: 8.5pt; font-weight: 600; padding: 4px 16px;
                }
            """)

        self.adjustSize()

        # Center it slightly above the target widget (like the login card)
        if parent_widget:
            global_pos = parent_widget.mapToGlobal(QtCore.QPoint(0, 0))
            x = global_pos.x() + (parent_widget.width() - self.width()) // 2
            y = global_pos.y() - self.height() - 15  # 15px above the card
            self.move(x, y)

        self.show()

    def clear(self) -> None:
        self.hide()


# ══════════════════════════════════════════════════════════════════════════════
class _LoginCentralWidget(QtWidgets.QWidget):
    """Central widget that paints a blurred, lightly frosted backdrop.

    The backdrop is sourced either by:
      • calling :meth:`capture_backdrop` with the run-window (default), or
      • calling :meth:`set_background_pixmap` with a pre-loaded QPixmap
        (e.g. a branded splash image supplied by the caller).

    ``paintEvent`` draws the blurred pixmap wall-to-wall and then lays a very
    light neutral tint on top (~24 % opacity) so the effect reads as "frosted"
    without overpowering the glass card in front.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._blurred: Optional[QtGui.QPixmap] = None
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    # ── public ────────────────────────────────────────────────────────────────
    def capture_backdrop(self, run_window: QtWidgets.QMainWindow) -> None:
        """Grab *run_window*, blur it, and schedule a repaint.

        Grabbing the run window directly via ``.grab()`` means the login window
        never needs to be hidden/shown, so no spurious window events fire.
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
        """Use a pre-supplied pixmap as the frosted backdrop.

        Call this instead of (or after) :meth:`capture_backdrop` when you want
        a fixed branded image rather than a live grab of the run window.

        Example::

            img = QtGui.QPixmap("assets/login_bg.jpg")
            self.centralwidget.set_background_pixmap(img)
        """
        pixmap = QtGui.QPixmap(
            os.path.join(Architecture.get_path(), "QATCH", "icons", "background.png")
        )
        if not self.size().isEmpty():
            pixmap = pixmap.scaled(
                self.size(),
                QtCore.Qt.KeepAspectRatioByExpanding,
                QtCore.Qt.SmoothTransformation,
            )
        self._blurred = self._apply_blur(pixmap, radius=22)
        self.update()

    # ── private ───────────────────────────────────────────────────────────────
    @staticmethod
    def _apply_blur(source: QtGui.QPixmap, radius: int = 22) -> QtGui.QPixmap:
        """Return a blurred copy of *source* using QGraphicsBlurEffect."""
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

    # ── Qt events ─────────────────────────────────────────────────────────────
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        if self._blurred:
            p.drawPixmap(self.rect(), self._blurred, self._blurred.rect())
        else:
            # Gradient fallback shown for the instant before capture completes.
            grad = QtGui.QLinearGradient(0, 0, self.width(), self.height())
            grad.setColorAt(0.0, QtGui.QColor(0xD8, 0xE6, 0xF0))
            grad.setColorAt(1.0, QtGui.QColor(0xEE, 0xF4, 0xF8))
            p.fillRect(self.rect(), QtGui.QBrush(grad))

        # Very light frost tint (~24 % opacity).
        p.fillRect(self.rect(), QtGui.QColor(238, 243, 247, 62))
        p.end()


# ══════════════════════════════════════════════════════════════════════════════
class _SlidingPanel(QtWidgets.QWidget):
    """Two-page sliding container for the sign-in ↔ recover-password transition.

    The widget's own width is fixed to *page_width* so Qt automatically clips
    any child content that extends beyond that boundary.  Both pages are placed
    side-by-side inside an inner container; animating that container's ``pos``
    property slides between them without any off-screen content leaking through.

    Typical usage::

        slider = _SlidingPanel(page_width=290)
        slider.add_page(sign_in_widget)
        slider.add_page(recover_widget)
        # After all pages have been added and their layouts are populated:
        QtCore.QTimer.singleShot(0, slider.finalize)
        # Later, to animate:
        slider.slide_to(1)   # → recover
        slider.slide_to(0)   # → sign-in
    """

    def __init__(self, page_width: int, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._pw = page_width
        self._anim: Optional[QtCore.QPropertyAnimation] = None

        # Fixed width causes Qt to clip children that extend past the right edge.
        self.setFixedWidth(page_width)
        self.setContentsMargins(0, 0, 0, 0)

        # Inner host — spans (n_pages × page_width) horizontally.
        self._inner = QtWidgets.QWidget(self)
        self._inner.move(0, 0)
        self._inner.setContentsMargins(0, 0, 0, 0)
        self._pages: List[QtWidgets.QWidget] = []

    # ── public ────────────────────────────────────────────────────────────────
    def add_page(self, widget: QtWidgets.QWidget) -> int:
        """Reparent *widget* into the inner container and return its page index."""
        idx = len(self._pages)
        widget.setParent(self._inner)
        widget.setFixedWidth(self._pw)
        widget.move(idx * self._pw, 0)
        self._pages.append(widget)
        return idx

    def finalize(self, fallback_height: int = 300) -> None:
        """Size pages and inner container to match the slider's fixed height.

        setFixedHeight must be called on the slider *before* this runs so the
        loginCard renders at the correct size on first paint.  finalize only
        sizes the inner host and each page to match that height so the slide
        animation works correctly.
        """
        if not self._pages:
            return
        h = self.height() if self.height() > 0 else fallback_height
        for i, p in enumerate(self._pages):
            p.setFixedSize(self._pw, h)
            p.move(i * self._pw, 0)
        self._inner.setFixedSize(len(self._pages) * self._pw, h)

    def slide_to(self, page_idx: int, duration: int = 360) -> None:
        """Animate the inner container to reveal *page_idx*."""
        target_x = -page_idx * self._pw
        anim = QtCore.QPropertyAnimation(self._inner, b"pos")
        anim.setStartValue(self._inner.pos())
        anim.setEndValue(QtCore.QPoint(target_x, 0))
        anim.setDuration(duration)
        anim.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        anim.start()
        self._anim = anim  # keep reference to prevent GC mid-animation


# ══════════════════════════════════════════════════════════════════════════════
class _GlassCard(QtWidgets.QFrame):
    """QFrame that renders true glassmorphism by sampling the frosted backdrop.

    In ``paintEvent`` it:
      1. Maps this widget's position onto the ``_LoginCentralWidget`` backdrop
         and blits the blurred pixmap slice that falls directly behind the card.
      2. Overlays a semi-transparent white glass tint.
      3. Adds a top-edge shimmer (lit-from-above glass highlight).
      4. Draws a two-layer border (muted outer + white inner highlight).

    All painting is done with clipping to the rounded rectangle so the glass
    effect is sharp-edged.  Child widgets paint on top normally; because the
    sign-in page uses 28 px horizontal margins the content stays well clear of
    the 22 px corner radii, so no clipping artefacts appear.
    """

    _RADIUS: float = 22.0

    def __init__(
        self,
        backdrop: _LoginCentralWidget,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._backdrop = backdrop
        # Disable auto-fill so Qt does not pre-clear with the palette colour
        # before our paintEvent runs.
        self.setAutoFillBackground(False)

    # ── Qt events ─────────────────────────────────────────────────────────────
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        rf = QtCore.QRectF(self.rect())

        # Clip all fill operations to the rounded rect shape.
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rf, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)

        # 1 ── Blurred backdrop slice behind the card (the core glass layer).
        #      mapTo() gives the card's origin in the backdrop's coordinate
        #      system, letting us blit exactly the right region of the pixmap.
        if self._backdrop._blurred:
            origin = self.mapTo(self._backdrop, QtCore.QPoint(0, 0))
            src = QtCore.QRect(origin, self.size())
            p.drawPixmap(self.rect(), self._backdrop._blurred, src)
        else:
            # Fallback gradient while the backdrop capture is still loading.
            grad = QtGui.QLinearGradient(0, 0, 0, self.height())
            grad.setColorAt(0.0, QtGui.QColor(0xD4, 0xE6, 0xF4))
            grad.setColorAt(1.0, QtGui.QColor(0xE6, 0xF2, 0xFA))
            p.fillRect(self.rect(), QtGui.QBrush(grad))

        # 2 ── Semi-transparent white tint  (makes the card clearly "glass").
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 170))

        # 3 ── Top-edge shimmer: white highlight fading to transparent.
        shimmer = QtGui.QLinearGradient(0, 0, 0, 60)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 105))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        p.setClipping(False)

        # 4a ── Outer muted blue-grey border.
        p.setPen(QtGui.QPen(QtGui.QColor(120, 160, 200, 135), 1.0))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRoundedRect(rf.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)

        # 4b ── Inner white highlight (gives the card a frosted-glass rim).
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 175), 1.0))
        p.drawRoundedRect(
            rf.adjusted(1.5, 1.5, -1.5, -1.5),
            self._RADIUS - 1.5,
            self._RADIUS - 1.5,
        )

        p.end()
        # Note: do NOT call super().paintEvent(event) — that would repaint
        # the QSS background-color on top of our glass effect.


# ══════════════════════════════════════════════════════════════════════════════


class DropPlaceholder(QtWidgets.QWidget):
    """A highly glassy, squarer highlight indicating where the profile will be dropped."""

    def __init__(self, width: int, height: int, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = QtCore.QRectF(self.rect()).adjusted(2.0, 2.0, -2.0, -2.0)
        radius = 12.0

        # 1. Glassy Gradient Fill
        grad = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
        grad.setColorAt(0.0, QtGui.QColor(255, 255, 255, 80))  # Brighter top-left
        grad.setColorAt(0.5, QtGui.QColor(220, 230, 240, 40))  # Highly transparent center
        grad.setColorAt(1.0, QtGui.QColor(180, 195, 210, 60))  # Slightly frosted bottom-right

        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QBrush(grad))
        p.drawRoundedRect(rect, radius, radius)

        # 2. Bright Inner Reflection (The "Glass Edge")
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 1.5))
        p.drawRoundedRect(rect, radius, radius)

        # 3. Subtle Outer Dark Stroke for depth
        p.setPen(QtGui.QPen(QtGui.QColor(140, 155, 170, 50), 1.0))
        p.drawRoundedRect(rect.adjusted(-1, -1, 1, 1), radius + 1, radius + 1)


class DraggableUserTile(QtWidgets.QWidget):
    """A custom widget representing a user tile that can be dragged."""

    clicked = QtCore.pyqtSignal(str, str)

    def __init__(
        self, name: str, initials: str, index: int, is_current: bool, avatar_size: int, parent=None
    ):
        super().__init__(parent)
        self.name = name
        self.initials = initials
        self.index = index
        self._avatar_size = avatar_size

        self._drag_start_pos = None
        self._global_drag_start = None
        self._is_dragging = False

        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setFixedSize(72, 88)

        self._build_ui(is_current)

        self.btn.installEventFilter(self)
        self.lbl.installEventFilter(self)

    def _build_ui(self, is_current: bool):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.btn = QtWidgets.QPushButton()
        self.btn.setObjectName("userTileBtn")
        self.btn.setFixedSize(self._avatar_size, self._avatar_size)
        self.btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn.setIcon(QtGui.QIcon(self._make_circular_pixmap(self.initials, self._avatar_size)))
        self.btn.setIconSize(QtCore.QSize(self._avatar_size, self._avatar_size))
        self.btn.setProperty("current", is_current)

        self.btn.clicked.connect(lambda: self.clicked.emit(self.name, self.initials))

        self.lbl = QtWidgets.QLabel(self._format_name_stacked(self.name))
        self.lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl.setObjectName("tileName")

        layout.addWidget(self.btn, 0, QtCore.Qt.AlignHCenter)
        layout.addWidget(self.lbl, 0, QtCore.Qt.AlignHCenter)

    def eventFilter(self, obj, event: QtCore.QEvent) -> bool:
        if obj in (self.btn, self.lbl):
            if (
                event.type() == QtCore.QEvent.MouseButtonPress
                and event.button() == QtCore.Qt.LeftButton
            ):
                self._drag_start_pos = event.pos()
                self._global_drag_start = event.globalPos()
                self._is_dragging = False

            elif event.type() == QtCore.QEvent.MouseMove and (
                event.buttons() & QtCore.Qt.LeftButton
            ):
                if self._drag_start_pos is not None:
                    # Determine if we've moved far enough to start a drag
                    if not self._is_dragging:
                        if (
                            event.globalPos() - self._global_drag_start
                        ).manhattanLength() >= QtWidgets.QApplication.startDragDistance():
                            self._is_dragging = True
                            self.btn.setDown(False)  # Visually un-press the button

                            # Safely find the window and trigger the custom drag
                            window = self.window()
                            if hasattr(window, "start_custom_drag"):
                                window.start_custom_drag(self)
                            return True

                    # If already dragging, physically move the widget
                    if self._is_dragging:
                        window = self.window()
                        if hasattr(window, "process_custom_drag"):
                            window.process_custom_drag(self, event.globalPos())
                        return True

            elif (
                event.type() == QtCore.QEvent.MouseButtonRelease
                and event.button() == QtCore.Qt.LeftButton
            ):
                if self._is_dragging:
                    self._is_dragging = False
                    self._drag_start_pos = None

                    window = self.window()
                    if hasattr(window, "end_custom_drag"):
                        window.end_custom_drag(self)
                    return True  # Prevent the click signal from firing

                self._drag_start_pos = None

        return super().eventFilter(obj, event)

    @staticmethod
    def _format_name_stacked(name: str) -> str:
        parts = name.split(" ", 1)
        if len(parts) == 2:
            return f"{parts[0]}\n{parts[1]}"
        return name

    @staticmethod
    def _make_circular_pixmap(initials: str, size: int) -> QtGui.QPixmap:
        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        hash_val = sum(ord(c) for c in initials)
        hues = [210, 200, 220, 190, 215]
        hue = hues[hash_val % len(hues)]
        base_color = QtGui.QColor.fromHsl(hue, 90, 190)

        rect = QtCore.QRectF(2.0, 2.0, size - 4.0, size - 4.0)
        p.setBrush(QtGui.QBrush(base_color))
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 1.5))
        p.drawEllipse(rect)

        p.setPen(QtGui.QColor(60, 60, 60, 200))
        font = p.font()
        font.setPixelSize(int(size * 0.42))
        font.setBold(True)
        p.setFont(font)
        p.drawText(rect, QtCore.Qt.AlignCenter, initials)
        p.end()
        return pm


class UserSwitchDialog(QtWidgets.QDialog):
    """Frameless, popup profile-picker anchored near the avatar button."""

    user_selected = QtCore.pyqtSignal(str, str)
    add_user_requested = QtCore.pyqtSignal()
    users_reordered = QtCore.pyqtSignal(list)

    _AVATAR_D: int = 52
    _COLS: int = 3

    _TILE_W = 72
    _TILE_H = 88
    _H_SPACE = 16
    _V_SPACE = 12

    def __init__(
        self,
        users: List[Tuple[str, str]],
        parent: Optional[QtWidgets.QWidget] = None,
        current_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            parent,
            QtCore.Qt.FramelessWindowHint | QtCore.Qt.Popup | QtCore.Qt.NoDropShadowWindowHint,
        )

        self._current_name = current_name
        self._users = list(users)

        self._tile_widgets = []
        self._live_tiles = []
        self.placeholder = None
        self._drag_offset = QtCore.QPoint()

        self._animations = []

        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self._apply_styles()
        self._build_ui()

    def _apply_styles(self) -> None:
        self.setStyleSheet(f"""
            QFrame#switchCard {{
                background: rgba(244, 247, 249, 230);
                border: 1px solid rgba(255, 255, 255, 220);
                border-radius: 12px;
            }}
            QLabel#switchTitle {{
                color: rgba(60, 60, 60, 220);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
                font-size: 11pt;
                font-weight: 600;
            }}
            QLabel#switchSubtitle {{
                color: rgba(60, 60, 60, 160);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
                font-size: 8.5pt;
            }}
            QFrame#switchSep {{
                border-top: 1px solid rgba(200, 210, 220, 150);
                max-height: 1px;
            }}
            QLabel#tileName {{
                color: rgba(60, 60, 60, 200);
                font-size: 8.5pt;
                font-weight: 500;
            }}
            
            /* --- Scroll Area & Scrollbar Styling --- */
            QScrollArea#gridScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollArea#gridScrollArea > QWidget > QWidget {{
                background: transparent;
            }}
            QScrollBar:vertical {{
                border: none;
                background: transparent;
                width: 8px;
                margin: 0px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(160, 175, 190, 150);
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: rgba(130, 150, 170, 200);
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
                border: none;
            }}
            
            QPushButton#userTileBtn,
            QPushButton#addTileBtn {{
                background: transparent;
                border-radius: {self._AVATAR_D // 2}px;
                border: 2px solid transparent;
            }}
            QPushButton#addTileBtn {{
                background-color: rgba(229, 229, 229, 150);
            }}
            QPushButton#addTileBtn:hover {{
                background-color: rgba(210, 215, 220, 180);
            }}
            QPushButton#userTileBtn:hover {{
                background-color: rgba(229, 229, 229, 120);
            }}
            QPushButton#userTileBtn[current="true"] {{
                border-color: rgba(10, 163, 230, 120);
                background: rgba(10, 163, 230, 25);
            }}
        """)

    def _build_ui(self) -> None:
        self.root_layout = QtWidgets.QVBoxLayout(self)
        self.root_layout.setContentsMargins(10, 10, 10, 10)

        self.card = QtWidgets.QFrame()
        self.card.setObjectName("switchCard")

        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QtGui.QColor(0, 0, 0, 30))
        shadow.setOffset(0, 4)
        self.card.setGraphicsEffect(shadow)

        self.root_layout.addWidget(self.card)

        cv = QtWidgets.QVBoxLayout(self.card)
        cv.setContentsMargins(16, 16, 16, 20)
        cv.setSpacing(8)

        title = QtWidgets.QLabel("Switch Users")
        title.setObjectName("switchTitle")
        title.setAlignment(QtCore.Qt.AlignCenter)
        cv.addWidget(title)

        sep = QtWidgets.QFrame()
        sep.setObjectName("switchSep")
        cv.addWidget(sep)

        # 1. Setup Scroll Area
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setObjectName("gridScrollArea")
        self.scroll_area.setWidgetResizable(False)  # We will manually size the inner widget
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # 2. Setup the Absolute Positioning Container
        self.grid_container = QtWidgets.QWidget()
        self.grid_container.setObjectName("gridContainer")
        self.scroll_area.setWidget(self.grid_container)

        cv.addWidget(self.scroll_area, 0, QtCore.Qt.AlignHCenter)

        self._build_add_button()
        self._init_tiles()

    def _build_add_button(self):
        self.add_container = QtWidgets.QWidget(self.grid_container)
        self.add_container.setFixedSize(self._TILE_W, self._TILE_H)

        alayout = QtWidgets.QVBoxLayout(self.add_container)
        alayout.setContentsMargins(0, 0, 0, 0)
        alayout.setSpacing(6)

        btn = QtWidgets.QPushButton()
        btn.setObjectName("addTileBtn")
        btn.setFixedSize(self._AVATAR_D, self._AVATAR_D)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        btn.setIcon(
            QtGui.QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "add-user.svg"))
        )
        btn.setIconSize(QtCore.QSize(self._AVATAR_D, self._AVATAR_D))
        btn.clicked.connect(self._on_add)

        lbl = QtWidgets.QLabel("Add\nUser")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setObjectName("tileName")

        alayout.addWidget(btn, 0, QtCore.Qt.AlignHCenter)
        alayout.addWidget(lbl, 0, QtCore.Qt.AlignHCenter)

    def _init_tiles(self):
        for widget in self._tile_widgets:
            widget.setParent(None)
            widget.deleteLater()
        self._tile_widgets.clear()

        for idx, (name, initials) in enumerate(self._users):
            tile = DraggableUserTile(
                name=name,
                initials=initials,
                index=idx,
                is_current=(name == self._current_name),
                avatar_size=self._AVATAR_D,
                parent=self.grid_container,
            )
            tile.clicked.connect(self._on_select)
            tile.show()
            self._tile_widgets.append(tile)

        self._live_tiles = list(self._tile_widgets)
        self._layout_tiles(animate=False)

    def _layout_tiles(self, animate=True, exclude=None):

        total_items = len(self._live_tiles) + 1
        cols = max(1, min(self._COLS, total_items))
        rows = (total_items + cols - 1) // cols

        # Calculate strict geometries
        container_w = cols * self._TILE_W + (cols - 1) * self._H_SPACE
        container_h = rows * self._TILE_H + (rows - 1) * self._V_SPACE
        self.grid_container.setFixedSize(container_w, container_h)

        # Cap the scroll area viewport to exactly 2 rows tall (188px)
        max_scroll_h = (2 * self._TILE_H) + self._V_SPACE
        scrollbar_allowance = 12 if container_h > max_scroll_h else 0

        self.scroll_area.setFixedWidth(container_w + scrollbar_allowance)
        self.scroll_area.setFixedHeight(min(container_h, max_scroll_h))

        self._animations.clear()

        # Flow live tiles
        for idx, widget in enumerate(self._live_tiles):
            if widget == exclude:
                continue

            row, col = divmod(idx, cols)
            target_x = col * (self._TILE_W + self._H_SPACE)
            target_y = row * (self._TILE_H + self._V_SPACE)

            self._slide_widget(widget, target_x, target_y, animate)

        # Flow the Add button to the very end
        add_idx = len(self._live_tiles)
        row, col = divmod(add_idx, cols)
        target_x = col * (self._TILE_W + self._H_SPACE)
        target_y = row * (self._TILE_H + self._V_SPACE)

        self._slide_widget(self.add_container, target_x, target_y, animate)
        self.adjustSize()

    def _slide_widget(self, widget, target_x, target_y, animate):
        target_pos = QtCore.QPoint(target_x, target_y)

        if not animate or widget.pos() == target_pos:
            widget.move(target_pos)
            return

        anim = QtCore.QPropertyAnimation(widget, b"pos", self)
        anim.setDuration(200)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim.setEndValue(target_pos)
        anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
        self._animations.append(anim)

    def _get_slot_center(self, idx: int) -> QtCore.QPoint:
        total_items = len(self._tile_widgets) + 1
        cols = max(1, min(self._COLS, total_items))
        row, col = divmod(idx, cols)

        x = col * (self._TILE_W + self._H_SPACE)
        y = row * (self._TILE_H + self._V_SPACE)

        return QtCore.QPoint(x + self._TILE_W // 2, y + self._TILE_H // 2)

    # --- CUSTOM GEOMETRIC DRAG SYSTEM ---

    def start_custom_drag(self, tile) -> None:
        self._drag_offset = tile.mapFromGlobal(QtGui.QCursor.pos())

        self.placeholder = DropPlaceholder(self._TILE_W, self._TILE_H, self.grid_container)
        self.placeholder.show()

        self.placeholder.lower()
        tile.raise_()

        idx = self._live_tiles.index(tile)
        self._live_tiles[idx] = self.placeholder

        self._layout_tiles(animate=True, exclude=tile)

    def process_custom_drag(self, tile, global_mouse_pos: QtCore.QPoint) -> None:
        # 1. Edge-Detection Auto-Scrolling
        vp_pos = self.scroll_area.viewport().mapFromGlobal(global_mouse_pos)
        vbar = self.scroll_area.verticalScrollBar()

        # If cursor is within 20px of the top/bottom viewport edges, scroll
        if vp_pos.y() < 20:
            vbar.setValue(vbar.value() - 8)
        elif vp_pos.y() > self.scroll_area.viewport().height() - 20:
            vbar.setValue(vbar.value() + 8)

        # 2. Update Tile Position
        local_pos = self.grid_container.mapFromGlobal(global_mouse_pos)

        new_x = local_pos.x() - self._drag_offset.x()
        new_y = local_pos.y() - self._drag_offset.y()

        # Clamp bounds strictly to the inner grid container
        max_x = self.grid_container.width() - tile.width()
        max_y = self.grid_container.height() - tile.height()

        new_x = max(0, min(new_x, max_x))
        new_y = max(0, min(new_y, max_y))

        tile.move(new_x, new_y)

        # 3. Collision Logic
        tile_center = tile.geometry().center()
        closest_idx = self._live_tiles.index(self.placeholder)
        min_dist = float("inf")

        for idx in range(len(self._live_tiles)):
            slot_center = self._get_slot_center(idx)
            dist = (tile_center - slot_center).manhattanLength()
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx

        add_slot_center = self._get_slot_center(len(self._live_tiles))
        dist_to_add = (tile_center - add_slot_center).manhattanLength()
        if dist_to_add < min_dist:
            closest_idx = len(self._live_tiles) - 1

        current_idx = self._live_tiles.index(self.placeholder)
        if closest_idx != current_idx:
            self._live_tiles.remove(self.placeholder)
            self._live_tiles.insert(closest_idx, self.placeholder)
            self._layout_tiles(animate=True, exclude=tile)

    def end_custom_drag(self, tile) -> None:
        final_idx = self._live_tiles.index(self.placeholder)
        source_idx = self._tile_widgets.index(tile)

        self._live_tiles[final_idx] = tile

        if self.placeholder:
            self.placeholder.deleteLater()
            self.placeholder = None

        if final_idx != source_idx:
            user = self._users.pop(source_idx)
            self._users.insert(final_idx, user)

            widget = self._tile_widgets.pop(source_idx)
            self._tile_widgets.insert(final_idx, widget)

            for i, w in enumerate(self._tile_widgets):
                w.index = i

            self.users_reordered.emit(self._users)

        self._layout_tiles(animate=True)

    def _on_select(self, name: str, initials: str) -> None:
        self.user_selected.emit(name, initials)
        self.accept()

    def _on_add(self) -> None:
        self.add_user_requested.emit()
        self.accept()


# ══════════════════════════════════════════════════════════════════════════════
class UILogin:
    # ── setup ─────────────────────────────────────────────────────────────────
    def setup_ui(
        self,
        MainWindow5: QtWidgets.QMainWindow,
        parent: QtWidgets.QMainWindow,
    ) -> None:
        """Initialise and arrange all UI elements for the login window."""
        self.parent = parent
        self.caps_lock_on = False

        global _AVATAR_D, _CARD_W, _INPUT_H, _PAGE_H, _BTN_H
        _AVATAR_D = 70
        _CARD_W = 320
        _INPUT_H = 34
        _PAGE_H = 300
        _BTN_H = 38

        # ── Window basics ──────────────────────────────────────────────────────
        MainWindow5.setObjectName("MainWindow5")
        MainWindow5.setMinimumSize(QtCore.QSize(1000, 500))
        MainWindow5.resize(500, 500)
        MainWindow5.setTabShape(QtWidgets.QTabWidget.Rounded)

        # ── Custom central widget ──────────────────────────────────────────────
        self.centralwidget = _LoginCentralWidget(MainWindow5)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        self.centralwidget.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        MainWindow5.setCentralWidget(self.centralwidget)

        # ── INJECT MASTER STYLESHEET ───────────────────────────────────────────
        MASTER_QSS = f"""
            QWidget {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }}
            
            #centralwidget {{ background: transparent; border: none; }}
            #slidingPanel {{ background: transparent; }}

            /* ─── NEW GLASSY CARD EFFECT ─── */
            #loginCard {{ 
                background-color: rgba(244, 247, 249, 220); 
                border: 1px solid rgba(255, 255, 255, 200); 
                border-radius: 16px; 
            }}

            #user_welcome {{ color: rgba(60, 60, 60, 230); font-size: 15pt; font-weight: 700; }}
        """
        MainWindow5.setStyleSheet(MASTER_QSS)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 0 — Sign-In
        # ══════════════════════════════════════════════════════════════════════
        signInPage = QtWidgets.QWidget()
        si = QtWidgets.QVBoxLayout(signInPage)
        si.setContentsMargins(28, 22, 28, 18)
        si.setSpacing(9)
        si.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)

        cardTitle = QtWidgets.QLabel("Sign In")
        cardTitle.setObjectName("cardTitle")
        cardTitle.setAlignment(QtCore.Qt.AlignCenter)
        cardTitle.setStyleSheet("color: rgba(60, 60, 60, 220); font-size: 11pt; font-weight: 700;")
        si.addWidget(cardTitle)

        # ── Avatar ─────────────────────────────────────────────────────────────
        avatarOuter = QtWidgets.QWidget()
        avatarOuter.setFixedSize(_AVATAR_D, _AVATAR_D)
        avatarOuter.setContentsMargins(0, 0, 0, 0)

        self.userAvatarBtn = QtWidgets.QPushButton("", avatarOuter)
        self.userAvatarBtn.setObjectName("userAvatarBtn")
        self.userAvatarBtn.setFixedSize(_AVATAR_D, _AVATAR_D)
        self.userAvatarBtn.move(0, 0)
        self.userAvatarBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.userAvatarBtn.setToolTip("Click to switch user account")

        self.userAvatarBtn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(229, 229, 229, 100);
                border-radius: {_AVATAR_D // 2}px; 
                border: 2px solid transparent;
            }}
            QPushButton:hover {{
                background-color: rgba(210, 215, 220, 150);
                border: 2px solid rgba(255, 255, 255, 150);
            }}
            QPushButton[hasIcon="true"] {{
                background-color: transparent;
                border: 2px solid rgba(255, 255, 255, 180);
            }}
            QPushButton[hasIcon="true"]:hover {{
                border-color: rgba(255, 255, 255, 240);
                background-color: rgba(255, 255, 255, 40);
            }}
        """)
        self.userAvatarBtn.clicked.connect(self._show_user_switch_dialog)
        si.addWidget(avatarOuter, alignment=QtCore.Qt.AlignCenter)

        # ── Selected user label ────────────────────────────────────────────────
        self.user_label = QtWidgets.QLabel("Select a User")
        self.user_label.setObjectName("user_label")
        self.user_label.setAlignment(QtCore.Qt.AlignCenter)
        self.user_label.setFixedHeight(18)
        self.user_label.setProperty("placeholder", True)
        self.user_label.setStyleSheet(
            "color: rgba(60, 60, 60, 200); font-size: 8.5pt; font-weight: 600;"
        )
        si.addWidget(self.user_label, alignment=QtCore.Qt.AlignCenter)

        si.addSpacing(4)

        self.user_initials = QtWidgets.QLineEdit()
        self.user_initials.setObjectName("user_initials")
        self.user_initials.setMaxLength(4)
        self.user_initials.setVisible(False)

        # ── Password + compact sign-in row ──────────────────────────────────────
        credentialsRow = QtWidgets.QWidget()
        credentialsRow.setObjectName("credentialsRow")
        credentialsRow.setFixedWidth(_CARD_W - 56)
        credentialsRow.setFixedHeight(_INPUT_H)

        credentials = QtWidgets.QHBoxLayout(credentialsRow)
        credentials.setContentsMargins(0, 0, 0, 0)
        credentials.setSpacing(8)

        # 1. DIRECTLY STYLED PASSWORD FIELD
        self.user_password = QtWidgets.QLineEdit()
        self.user_password.setObjectName("user_password")
        self.user_password.setFixedHeight(_INPUT_H)
        self.user_password.setPlaceholderText("Password")
        self.user_password.setEchoMode(QtWidgets.QLineEdit.Password)

        # ── NEW: Persistent Glass Border & Neutral Selection ──
        # ── STORE STYLES FOR STATE SWAPPING ──

        self._pw_style_normal = f"""
            QLineEdit {{
                background-color: rgba(250, 252, 255, 160);
                border: 1.5px solid rgba(180, 195, 210, 180);
                border-style: solid;
                border-radius: {_INPUT_H // 2}px; 
                padding: 0px 15px;
                color: rgba(40, 50, 60, 240);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
                font-size: 10pt;
                selection-background-color: rgba(10, 163, 230, 100);
            }}
            QLineEdit:hover {{ background-color: rgba(255, 255, 255, 200); border-color: rgba(10, 163, 230, 150); }}
            QLineEdit:focus {{ background-color: rgba(255, 255, 255, 255); border: 2px solid #0AA3E6; outline: none; }}
            QLineEdit QToolButton {{ background: transparent; border: none; margin: 0px; padding: 0px; }}
            QLineEdit QToolButton:hover {{ background: rgba(10, 163, 230, 20); border-radius: 12px; }}
        """

        self._pw_style_error = f"""
            QLineEdit {{
                background-color: rgba(255, 230, 230, 160); /* Light red frosted tint */
                border: 1.5px solid rgba(230, 50, 50, 200); /* Glassy red border */
                border-style: solid;
                border-radius: {_INPUT_H // 2}px; 
                padding: 0px 15px;
                color: rgba(200, 30, 30, 255);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
                font-size: 10pt;
                selection-background-color: rgba(230, 50, 50, 80);
            }}
            QLineEdit:focus {{ background-color: rgba(255, 245, 245, 255); border: 2px solid rgba(255, 50, 50, 255); outline: none; }}
            QLineEdit QToolButton {{ background: transparent; border: none; margin: 0px; padding: 0px; }}
            QLineEdit QToolButton:hover {{ background: rgba(230, 50, 50, 20); border-radius: 12px; }}
        """

        # Apply normal style to start
        self.user_password.setStyleSheet(self._pw_style_normal)

        # Revert back to normal automatically when the user starts typing a correction
        self.user_password.textChanged.connect(
            lambda: self.user_password.setStyleSheet(self._pw_style_normal)
        )

        self.user_password.installEventFilter(MainWindow5)
        self.user_password.returnPressed.connect(self.action_sign_in)
        credentials.addWidget(self.user_password, stretch=1)

        # 2. DIRECTLY STYLED SIGN-IN BUTTON
        self.sign_in = QtWidgets.QPushButton()
        self.sign_in.setObjectName("sign_in")
        self.sign_in.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(229, 229, 229, 150);
                border: 1.5px solid rgba(255, 255, 255, 200);
                border-style: solid;
                border-radius: {_INPUT_H // 2}px; 
            }}
            QPushButton:hover {{ background-color: rgba(210, 215, 220, 180); }}
            QPushButton:pressed {{ background-color: rgba(190, 200, 210, 200); }}
        """)

        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "right-arrow.svg")
        self.sign_in.setIcon(QtGui.QIcon(icon_path))
        self.sign_in.setIconSize(QtCore.QSize(24, 24))
        self.sign_in.setFixedSize(_INPUT_H, _INPUT_H)
        self.sign_in.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.sign_in.setToolTip("Sign in")
        self.sign_in.setAccessibleName("Sign In")
        self.sign_in.clicked.connect(self.action_sign_in)
        self.sign_in.installEventFilter(MainWindow5)

        credentials.addWidget(self.sign_in, alignment=QtCore.Qt.AlignVCenter)
        si.addWidget(credentialsRow, alignment=QtCore.Qt.AlignCenter)

        # ── CAPS LOCK INDICATOR ────────────────────────────────────────────────
        self.floating_badge = _FloatingMessageBadge(MainWindow5)

        # ── REMEMBER ME CHECKBOX ───────────────────────────────────────────────
        self.remember_me = QtWidgets.QCheckBox("Remember me")
        self.remember_me.setObjectName("rememberMe")
        self.remember_me.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # Inline glassy style for the checkbox
        self.remember_me.setStyleSheet("""
            QCheckBox {
                color: rgba(100, 110, 120, 200);
                font-size: 8.5pt;
                font-weight: 500;
                spacing: 8px; /* Space between box and text */
            }
            QCheckBox:hover { color: rgba(60, 60, 60, 220); }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border-radius: 4px;
                border: 1px solid rgba(180, 195, 210, 180);
                background-color: rgba(255, 255, 255, 140);
            }
            QCheckBox::indicator:hover {
                border-color: #0AA3E6;
                background-color: rgba(255, 255, 255, 220);
            }
            QCheckBox::indicator:checked {
                background-color: #0AA3E6;
                border: 1px solid #0AA3E6;
            }
        """)
        si.addWidget(self.remember_me, alignment=QtCore.Qt.AlignCenter)

        # ── ERROR LABEL ────────────────────────────────────────────────────────
        self.user_error = QtWidgets.QLabel("")
        self.user_error = QtWidgets.QLabel("")
        self.user_error.setObjectName("user_error")
        self.user_error.setFixedHeight(16)
        si.addWidget(self.user_error, alignment=QtCore.Qt.AlignCenter)

        si.addSpacing(10)
        self.forgotPassword = QtWidgets.QLabel("Forgot Password?")
        self.forgotPassword.setObjectName("forgotPassword")
        self.forgotPassword.setAlignment(QtCore.Qt.AlignCenter)
        self.forgotPassword.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.forgotPassword.setFixedHeight(20)
        self.forgotPassword.setStyleSheet("""
            QLabel { color: rgba(100, 110, 120, 180); font-size: 9pt; font-weight: 500; }
            QLabel:hover { color: rgba(60, 60, 60, 220); text-decoration: underline; }
        """)
        self.forgotPassword.mousePressEvent = lambda _e: self._slide_to_recover()
        si.addWidget(self.forgotPassword)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 1 — Recover Password
        # ══════════════════════════════════════════════════════════════════════
        recoverPage = QtWidgets.QWidget()
        rec = QtWidgets.QVBoxLayout(recoverPage)
        rec.setContentsMargins(28, 18, 28, 18)
        rec.setSpacing(9)
        rec.setAlignment(QtCore.Qt.AlignTop)

        backBtn = QtWidgets.QPushButton("← Back to Sign In")
        backBtn.setObjectName("backBtn")
        backBtn.setFixedHeight(24)
        backBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        backBtn.setStyleSheet("""
            QPushButton { background: transparent; color: rgba(100, 110, 120, 200); border: none; font-size: 8.5pt; font-weight: 600; text-align: left; }
            QPushButton:hover { color: rgba(60, 60, 60, 220); text-decoration: underline; }
        """)
        backBtn.clicked.connect(self._slide_to_signin)
        rec.addWidget(backBtn, alignment=QtCore.Qt.AlignLeft)

        recoverTitle = QtWidgets.QLabel("Reset Password")
        recoverTitle.setObjectName("recoverTitle")
        recoverTitle.setAlignment(QtCore.Qt.AlignCenter)
        recoverTitle.setStyleSheet(
            "color: rgba(60, 60, 60, 220); font-size: 11pt; font-weight: 700;"
        )
        rec.addWidget(recoverTitle)

        recoverInfo = QtWidgets.QLabel(
            "Enter the email address linked to your account\n"
            "and we'll send you a password reset link."
        )
        recoverInfo.setObjectName("recoverInfo")
        recoverInfo.setAlignment(QtCore.Qt.AlignCenter)
        recoverInfo.setWordWrap(True)
        recoverInfo.setStyleSheet("color: rgba(100, 110, 120, 220); font-size: 8.5pt;")
        rec.addWidget(recoverInfo)

        self.recoverEmail = QtWidgets.QLineEdit()
        self.recoverEmail.setObjectName("recoverEmail")
        self.recoverEmail.setPlaceholderText("Email Address")
        self.recoverEmail.setFixedHeight(_INPUT_H)

        self.recoverEmail.setStyleSheet(f"""
            QLineEdit {{
                background-color: rgba(255, 255, 255, 140);
                border: 1.5px solid rgba(255, 255, 255, 160);
                border-style: solid;
                border-radius: {_INPUT_H // 2}px; 
                padding: 0px 15px;
                color: rgba(60, 60, 60, 220);
                font-size: 10pt;
                selection-background-color: rgba(200, 210, 220, 150);
                selection-color: rgba(40, 40, 40, 255);
            }}
            QLineEdit:hover {{ background-color: rgba(255, 255, 255, 180); border-color: rgba(255, 255, 255, 220); }}
            QLineEdit:focus {{ background-color: rgba(255, 255, 255, 240); border: 1.5px solid rgba(255, 255, 255, 255); outline: none; }}
        """)
        rec.addWidget(self.recoverEmail)

        self.sendResetBtn = QtWidgets.QPushButton("Send Reset Link")
        self.sendResetBtn.setObjectName("sendResetBtn")
        self.sendResetBtn.setFixedHeight(_BTN_H)
        self.sendResetBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.sendResetBtn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(229, 229, 229, 150);
                border: 1.5px solid rgba(255, 255, 255, 200);
                border-style: solid;
                border-radius: {_BTN_H // 2}px; 
                color: rgba(60, 60, 60, 220);
                font-size: 10pt; font-weight: 600;
            }}
            QPushButton:hover {{ background-color: rgba(210, 215, 220, 180); }}
            QPushButton:pressed {{ background-color: rgba(190, 200, 210, 200); }}
        """)
        self.sendResetBtn.clicked.connect(self._on_send_reset)
        rec.addWidget(self.sendResetBtn)

        self.recoverStatus = QtWidgets.QLabel("")
        self.recoverStatus.setObjectName("recoverStatus")
        self.recoverStatus.setAlignment(QtCore.Qt.AlignCenter)
        self.recoverStatus.setWordWrap(True)
        self.recoverStatus.setFixedHeight(36)
        self.recoverStatus.setStyleSheet(
            "color: rgba(46, 139, 87, 220); font-size: 8.5pt; font-weight: 500;"
        )
        rec.addWidget(self.recoverStatus)
        rec.addStretch()

        # ══════════════════════════════════════════════════════════════════════
        # ══════════════════════════════════════════════════════════════════════
        # Sliding panel and Glass Card
        # ══════════════════════════════════════════════════════════════════════
        self._slider = _SlidingPanel(_CARD_W)
        self._slider.setObjectName("slidingPanel")

        # Keep slider completely transparent so the card shows through
        self._slider.setStyleSheet(
            "QWidget#slidingPanel { background: transparent; border: none; }"
        )

        self._slider.add_page(signInPage)
        self._slider.add_page(recoverPage)
        self._slider.setFixedHeight(_PAGE_H)
        QtCore.QTimer.singleShot(0, lambda: self._slider.finalize(_PAGE_H))

        # --- REVERTED GLASSCARD SETUP ---
        self.loginCard = _GlassCard(self.centralwidget)
        self.loginCard.setObjectName("loginCard")
        self.loginCard.setContentsMargins(0, 0, 0, 0)
        # REMOVED: setAttribute(WA_StyledBackground)
        # REMOVED: loginCard.setStyleSheet(...)

        # Keep your existing drop shadow
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

        self._errorTimer = QtCore.QTimer()
        self._errorTimer.setSingleShot(True)
        self._errorTimer.timeout.connect(self.user_error.clear)

        self._sessionTimer = QtCore.QTimer()
        self._sessionTimer.setSingleShot(True)
        self._sessionTimer.timeout.connect(self.check_user_session)
        self._sessionTimer.setInterval(1000 * 60 * 60)

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

        QtCore.QTimer.singleShot(0, lambda: self.centralwidget.set_background_pixmap())

    # ── Avatar / user-switch ──────────────────────────────────────────────────

    @staticmethod
    def _make_circular_pixmap(initials: str, size: int) -> QtGui.QPixmap:
        """2. Shared method to generate muted pastel/slate avatars"""
        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.transparent)

        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        hash_val = sum(ord(c) for c in initials)
        hues = [210, 200, 220, 190, 215]
        hue = hues[hash_val % len(hues)]
        base_color = QtGui.QColor.fromHsl(hue, 90, 190)

        rect = QtCore.QRectF(2.0, 2.0, size - 4.0, size - 4.0)
        p.setBrush(QtGui.QBrush(base_color))
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 1.5))
        p.drawEllipse(rect)

        p.setPen(QtGui.QColor(60, 60, 60, 200))
        font = p.font()
        font.setPixelSize(int(size * 0.42))
        font.setBold(True)
        p.setFont(font)
        p.drawText(rect, QtCore.Qt.AlignCenter, initials)

        p.end()
        return pm

    def _show_user_switch_dialog(self) -> None:
        try:
            _, raw = UserProfiles.get_all_user_info()
            users: List[Tuple[str, str]] = [
                (info[0], info[1]) for info in raw if info[0] is not None and info[1] is not None
            ]
        except Exception as exc:
            Log.w(f"Could not load user profiles for switcher: {exc}")
            users = []

        current_name = self._current_user_display_name_for_switcher()

        dlg = UserSwitchDialog(
            users,
            parent=self.centralwidget,
            current_name=current_name,
        )
        dlg.user_selected.connect(self._on_user_selected)
        dlg.add_user_requested.connect(self._on_add_user_requested)

        dlg.adjustSize()
        self._position_user_switch_dialog(dlg)

        dlg.exec_()

    def _current_user_display_name_for_switcher(self) -> Optional[str]:
        for attr_name in (
            "current_user_display_name",
            "_current_user_display_name",
            "current_user_name",
            "_current_user_name",
        ):
            value = getattr(self, attr_name, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        prop_value = self.userAvatarBtn.property("display_name")
        if isinstance(prop_value, str) and prop_value.strip():
            return prop_value.strip()

        return None

    def _position_user_switch_dialog(self, dlg: QtWidgets.QDialog) -> None:
        avatar = self.userAvatarBtn
        gap = 10
        screen_padding = 12

        avatar_top_left = avatar.mapToGlobal(QtCore.QPoint(0, 0))
        avatar_center_x = avatar_top_left.x() + avatar.width() // 2

        # 5. Dialog Centering Fix: Use sizeHint() because the geometry is rarely final here
        preferred_x = avatar_center_x - dlg.sizeHint().width() // 2
        preferred_y = avatar_top_left.y() + avatar.height() + gap

        screen = QtWidgets.QApplication.screenAt(QtCore.QPoint(avatar_center_x, preferred_y))
        if screen is None:
            window_handle = self.window().windowHandle()
            screen = window_handle.screen() if window_handle is not None else None
        if screen is None:
            screen = QtWidgets.QApplication.primaryScreen()

        available = screen.availableGeometry()

        if preferred_y + dlg.height() > available.bottom() - screen_padding:
            preferred_y = avatar_top_left.y() - dlg.sizeHint().height() - gap

        x = max(
            available.left() + screen_padding,
            min(
                preferred_x,
                available.right() - dlg.sizeHint().width() - screen_padding,
            ),
        )

        y = max(
            available.top() + screen_padding,
            min(
                preferred_y,
                available.bottom() - dlg.sizeHint().height() - screen_padding,
            ),
        )

        dlg.move(x, y)

    def _on_user_selected(self, display_name: str, initials: str) -> None:
        self.user_initials.setText(initials)

        self.user_label.setText(display_name)
        self.user_label.setProperty("placeholder", False)
        self.user_label.style().unpolish(self.user_label)
        self.user_label.style().polish(self.user_label)

        self.userAvatarBtn.setProperty("display_name", display_name)
        self.userAvatarBtn.setProperty("hasIcon", True)

        # 2. Re-generates utilizing the shared slate/pastel aesthetic
        avatar_px = self._make_circular_pixmap(initials, _AVATAR_D)
        self.userAvatarBtn.setText("")
        self.userAvatarBtn.setIcon(QtGui.QIcon(avatar_px))
        self.userAvatarBtn.setIconSize(QtCore.QSize(_AVATAR_D, _AVATAR_D))

        self.userAvatarBtn.style().unpolish(self.userAvatarBtn)
        self.userAvatarBtn.style().polish(self.userAvatarBtn)

        self.user_password.setFocus()

    def _on_add_user_requested(self) -> None:
        Log.i("Add user requested from the login screen.")
        UserProfiles.create_new_user(UserRoles.OPERATE)
        QtCore.QTimer.singleShot(150, self._show_user_switch_dialog)

    # ── Slide transitions ──────────────────────────────────────────────────────
    def _slide_to_recover(self) -> None:
        self._slider.slide_to(1)

    def _slide_to_signin(self) -> None:
        self._slider.slide_to(0)
        self.recoverEmail.clear()
        self.recoverStatus.clear()
        self.sendResetBtn.setEnabled(True)

    # ── Reset-link handler ────────────────────────────────────────────────────
    def _on_send_reset(self) -> None:
        email = self.recoverEmail.text().strip()
        if not email:
            self.recoverStatus.setText("Please enter your email address.")
            return
        Log.i(f"Password reset requested for: {email}")
        self.recoverStatus.setText(
            f"If an account exists for that address,\na reset link has been sent."
        )
        self.sendResetBtn.setEnabled(False)

    # ── Password toggle ────────────────────────────────────────────────────────
    def on_toggle_password_Action(self) -> None:
        if not self.password_shown:
            self.user_password.setEchoMode(QtWidgets.QLineEdit.Normal)
            self.password_shown = True
            self.togglepasswordAction.setIcon(self.hiddenIcon)
        else:
            self.user_password.setEchoMode(QtWidgets.QLineEdit.Password)
            self.password_shown = False
            self.togglepasswordAction.setIcon(self.visibleIcon)

    # ── Session ───────────────────────────────────────────────────────────────
    def check_user_session(self) -> None:
        valid, infos = UserProfiles().session_info()
        if not valid:
            if self.parent.ControlsWin.userrole == UserRoles.NONE:
                Log.d("Hourly session check: user already signed out, skipping prompt.")
            else:
                Log.w("User session has expired.")
                Log.i("Please sign in to continue.")
                self.parent.ControlsWin.set_user_profile()
        else:
            Log.d("User session is still valid at the hourly check.")
            self._sessionTimer.start()

    # ── Retranslate ───────────────────────────────────────────────────────────
    def retranslateUi(self, MainWindow5: QtWidgets.QMainWindow) -> None:
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/qatch-icon.png")
        MainWindow5.setWindowIcon(QtGui.QIcon(icon_path))
        MainWindow5.setWindowTitle(
            _translate(
                "MainWindow5",
                "{} {} - Login".format(Constants.app_title, Constants.app_version),
            )
        )

    # ── Error helpers ─────────────────────────────────────────────────────────
    def error_loggedout(self) -> None:
        """Handles the forced logout state by notifying the user via the floating badge."""
        self.kickErrorTimer()

        # 1. Clear any leftover password text
        self.user_password.clear()

        # 2. Trigger the message on the floating overlay
        if hasattr(self, "floating_badge"):
            self.floating_badge.show_message(
                "You have been signed out", is_error=True, parent_widget=self.loginCard
            )

            # Auto-hide the badge after 5 seconds
            QtCore.QTimer.singleShot(5000, self.floating_badge.clear)

        # 3. Optional: Reset the password field style to normal in case it was red
        if hasattr(self, "_pw_style_normal"):
            self.user_password.setStyleSheet(self._pw_style_normal)

    def error_invalid(self, message: str = "Invalid Credentials") -> None:
        """
        Triggers the visual error state.
        The current user remains selected, but the password field alerts the user.
        """
        # 1. Apply the glassy red stylesheet to the password box
        if hasattr(self, "_pw_style_error"):
            self.user_password.setStyleSheet(self._pw_style_error)

        # 2. Trigger the physical jiggle animation
        if hasattr(self, "_shake_widget"):
            self._shake_widget(self.user_password)

        # 3. Clear only the password, keep the user selected
        self.user_password.clear()
        self.user_password.setFocus()

        # 4. Fire the message to the floating secondary window
        if hasattr(self, "floating_badge"):
            self.floating_badge.show_message(message, is_error=True, parent_widget=self.loginCard)

            # Auto-hide error after 4 seconds
            QtCore.QTimer.singleShot(4000, self.floating_badge.clear)

    def show_signout_message(self) -> None:
        # Show red message above the login card
        self.floating_badge.show_message(
            "You have been signed out.", is_error=True, parent_widget=self.loginCard
        )

        # Optional: Auto-hide after a few seconds
        QtCore.QTimer.singleShot(3000, self.floating_badge.clear)

    def error_expired(self) -> None:
        """Handles the session expiration state by notifying the user via the floating badge."""
        # 1. Clear the password field for security
        self.user_password.clear()

        # 2. Fire the message to the floating secondary window
        if hasattr(self, "floating_badge"):
            self.floating_badge.show_message(
                "Your session has expired", is_error=True, parent_widget=self.loginCard
            )

            # Auto-hide the message after 5 seconds
            QtCore.QTimer.singleShot(5000, self.floating_badge.clear)

        # 3. Optional: Reset the password box style if it was left in an error state
        if hasattr(self, "_pw_style_normal"):
            self.user_password.setStyleSheet(self._pw_style_normal)

    def kickErrorTimer(self) -> None:
        if self._errorTimer.isActive():
            Log.d("Error Timer was restarted while running")
            self._errorTimer.stop()
        self._errorTimer.start(10000)

    # ── Input helpers ─────────────────────────────────────────────────────────
    def text_transform(self) -> None:
        text = self.user_initials.text()
        if text:
            self.user_initials.setText(text.upper())

    # ── Sign-In ───────────────────────────────────────────────────────────────
    def action_sign_in(self) -> None:
        # Check if a user is actually selected (initials are not empty)
        if not self.user_initials.text():
            self.floating_badge.show_message(
                "Please select a user account first", is_error=True, parent_widget=self.loginCard
            )
            return

        # Check if password was entered
        if not self.user_password.text():
            # Jiggle the empty box and show the floating error
            self._shake_widget(self.user_password)
            self.floating_badge.show_message(
                "Password required", is_error=True, parent_widget=self.loginCard
            )
            return

        initials = self.user_initials.text().upper()
        pwd = self.user_password.text()
        authenticated, filename, params = UserProfiles.auth(initials, pwd, UserRoles.ANY)

        if authenticated:
            Log.i(f"Welcome, {params[0]}! Your assigned role is {params[2].name}.")
            name, init, role = params[0], params[1], params[2].value
            self._sessionTimer.start()
        else:
            name, init, role = None, None, 0
        self.clear_form()

        if name is not None:
            self.parent.ControlsWin.username.setText(f"User: {name}")
            self.parent.ControlsWin.userrole = UserRoles(role)
            self.parent.ControlsWin.signinout.setText("&Sign Out")
            self.parent.ControlsWin.ui1.tool_User.setText(name)
            self.parent.AnalyzeProc.tool_User.setText(name)
            if self.parent.ControlsWin.userrole != UserRoles.ADMIN:
                self.parent.ControlsWin.manage.setText("&Change Password...")

            check_result = UserProfiles().check(self.parent.ControlsWin.userrole, UserRoles.CAPTURE)
            if check_result:
                self.parent.MainWin.ui0._set_run_mode(self.user_label)
            else:
                self.parent.MainWin.ui0._set_analyze_mode(self.user_label)

            if UserProfiles().check(self.parent.ControlsWin.userrole, UserRoles.ADMIN):
                enabled, error, expires = UserProfiles.checkDevMode()
                if enabled is not True and error is not False:
                    is_expired = expires != ""
                    from QATCH.common.userProfiles import (
                        UserConstants,
                        UserProfilesManager,
                    )

                    if PopUp.question(
                        self.parent,
                        "Developer Mode " + ("Expired" if is_expired else "Error"),
                        (
                            "<b>Developer Mode "
                            + ("has expired" if is_expired else "is invalid")
                            + " and is no longer active!</b><br/>"
                            + f"Renewal Period: Every {UserConstants.DEV_EXPIRE_LEN} days<br/><br/>"
                            + "Would you like to renew Developer Mode now?<br/><br/>"
                            + "<small>NOTE: This setting can be changed in the"
                            + ' "Manage Users" window.</small>'
                        ),
                    ):
                        temp_upm = UserProfilesManager(self.parent, name)
                        temp_upm.developerModeChk.setChecked(True)
                        Log.i("Developer Mode renewed!")
                    else:
                        Log.w("Developer Mode NOT renewed!")

            if hasattr(self.parent, "url_download"):
                delattr(self.parent, "url_download")
            QtCore.QTimer.singleShot(1, self.parent.start_download)
        else:
            self.error_invalid()

    # ── Clear / caps-lock ─────────────────────────────────────────────────────
    def clear_form(self) -> None:
        """Clears the form gracefully. Pressing once clears password, pressing again clears user."""
        if len(self.user_password.text()) > 0:
            # If there's a password typed, Escape just clears the password
            self.user_password.clear()
        else:
            # If the password box is already empty, Escape clears the selected user
            self.user_initials.clear()
            self.user_label.setText("Select a User")
            self.user_label.setProperty("placeholder", True)
            self.userAvatarBtn.setIcon(QtGui.QIcon())
            self.userAvatarBtn.setProperty("hasIcon", False)

            # Re-apply styling to remove the avatar border
            self.user_label.style().unpolish(self.user_label)
            self.user_label.style().polish(self.user_label)
            self.userAvatarBtn.style().unpolish(self.userAvatarBtn)
            self.userAvatarBtn.style().polish(self.userAvatarBtn)

        self.user_error.clear()

        if self.password_shown:
            self.on_toggle_password_Action()

    def update_caps_lock_state(self, caps_lock_on: bool) -> None:
        if caps_lock_on:
            # Show amber warning above the login card
            self.floating_badge.show_message(
                "Caps Lock is ON", is_error=False, parent_widget=self.loginCard
            )
        else:
            # Hide it
            self.floating_badge.clear()

    def load_saved_credentials(self) -> None:
        """Loads the saved password and user state on launch."""
        settings = QtCore.QSettings("QATCH", "nanovisQ")

        # Retrieve the boolean (defaults to False)
        remembered = settings.value("login/remember_me", False, type=bool)
        self.remember_me.setChecked(remembered)

        if remembered:
            # Load the saved data
            saved_password = settings.value("login/password", "")
            saved_user_name = settings.value("login/user_name", "")

            if saved_password:
                self.user_password.setText(saved_password)

            if saved_user_name:
                # Assuming you have a method to programmatically select a user
                # e.g., self.set_active_user(saved_user_name)
                pass

    def save_credentials_on_success(self, user_name: str, password: str) -> None:
        """Called upon successful login to save or clear stored credentials."""
        settings = QtCore.QSettings("QATCH", "nanovisQ")

        if self.remember_me.isChecked():
            settings.setValue("login/remember_me", True)
            settings.setValue("login/user_name", user_name)
            # IMPORTANT: In a production app, you should hash or encrypt this before saving.
            # QSettings saves in plain text (registry on Windows, .plist on Mac)
            settings.setValue("login/password", password)
        else:
            # Wipe out the saved data if they unchecked the box
            settings.setValue("login/remember_me", False)
            settings.remove("login/user_name")
            settings.remove("login/password")

    def _shake_widget(self, widget: QtWidgets.QWidget) -> None:
        """Applies a rapid left-right jiggle animation to indicate an error."""
        # Store animation as class attribute so it doesn't get garbage collected
        self._shake_anim = QtCore.QPropertyAnimation(widget, b"pos")
        self._shake_anim.setDuration(400)

        base_pos = widget.pos()

        # Keyframes for a smooth, decaying shake
        self._shake_anim.setKeyValueAt(0.0, base_pos)
        self._shake_anim.setKeyValueAt(0.1, base_pos + QtCore.QPoint(-6, 0))
        self._shake_anim.setKeyValueAt(0.3, base_pos + QtCore.QPoint(6, 0))
        self._shake_anim.setKeyValueAt(0.5, base_pos + QtCore.QPoint(-4, 0))
        self._shake_anim.setKeyValueAt(0.7, base_pos + QtCore.QPoint(4, 0))
        self._shake_anim.setKeyValueAt(0.9, base_pos + QtCore.QPoint(-2, 0))
        self._shake_anim.setKeyValueAt(1.0, base_pos)

        self._shake_anim.start(QtCore.QPropertyAnimation.DeleteWhenStopped)
