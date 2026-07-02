"""DataManagementWidget - glassmorphic overlay container.

Owns the overlay lifecycle (open/close, fade, fullscreen, click-outside
dismiss) and the shared `DataServices`. Links in the five mode submodules and
drives switching between them via a glass segmented control + QStackedWidget.

Replaces `export_widget.Ui_Export` in the controls UI. Compatibility shims
(`showNormal` / `open_mode`) preserve the existing call sites.
"""

import os

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.common.data_service import DataServices
from QATCH.common.logger import Logger as Log
from QATCH.ui.widgets.data_mode_advanced import AdvancedMode
from QATCH.ui.widgets.data_mode_export import ExportMode
from QATCH.ui.widgets.data_mode_history import HistoryMode
from QATCH.ui.widgets.data_mode_import import ImportMode
from QATCH.ui.widgets.data_mode_recover import RecoverMode

TAG = "[DataManagement]"

# Mode classes in display order. Adding a mode = add its class here.
MODE_CLASSES = [ImportMode, ExportMode, RecoverMode, AdvancedMode, HistoryMode]

# Map the old integer tab indices to mode keys for showNormal() compatibility.
_LEGACY_TAB_TO_KEY = {0: "import", 1: "export", 2: "recover", 3: "advanced", 4: "history"}

# Position of each mode in the sidebar, used to pick slide direction.
_MODE_ORDER = {cls.MODE_KEY: i for i, cls in enumerate(MODE_CLASSES)}


# ======================================================================
#  Segmented mode selector
# ======================================================================
class GlassSegmentedControl(QtWidgets.QFrame):
    modeChanged = QtCore.pyqtSignal(str)

    def __init__(self, modes, parent=None, orientation=QtCore.Qt.Vertical):
        # modes: list of (key, label) or (key, label, icon_path)
        super().__init__(parent)
        self._orientation = orientation
        self.setObjectName("segmentedControl")

        if orientation == QtCore.Qt.Vertical:
            self.setFixedWidth(132)
            radius = 16
        else:
            self.setFixedHeight(38)
            radius = 19

        self.setStyleSheet(f"""
            QFrame#segmentedControl {{
                background: transparent;
                border: none;
                border-radius: {radius}px;
            }}
        """)
        self._buttons = {}
        self._icons = {}  # key -> (inactive QIcon, active QIcon)
        self._active_key = None

        if orientation == QtCore.Qt.Vertical:
            lay = QtWidgets.QVBoxLayout(self)
        else:
            lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)

        # Every row is identical: an 18px stroke icon + label, left-aligned.
        # Rows with no icon asset fall back to text-only rather than mixing
        # icon/no-icon rows, which would break the "every row identical" look.
        icon_size = QtCore.QSize(18, 18)
        inactive_color = QtGui.QColor(80, 92, 108, 235)
        active_color = QtGui.QColor(0, 100, 150, 255)

        for mode in modes:
            if len(mode) == 3:
                key, label, icon_path = mode
            else:
                key, label = mode
                icon_path = None
            btn = QtWidgets.QToolButton()
            btn.setText(f" {label}")
            btn.setCheckable(True)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            if icon_path and os.path.exists(icon_path):
                icon_inactive = self._tinted_icon(icon_path, inactive_color, icon_size.width())
                icon_active = self._tinted_icon(icon_path, active_color, icon_size.width())
                btn.setIcon(icon_inactive)
                btn.setIconSize(icon_size)
                btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
                self._icons[key] = (icon_inactive, icon_active)
            else:
                btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            if orientation == QtCore.Qt.Vertical:
                btn.setFixedHeight(38)
                btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            else:
                btn.setFixedHeight(30)
                btn.setMinimumWidth(78)
            btn.setStyleSheet(self._segment_qss())
            btn.clicked.connect(lambda _=False, k=key: self.set_active(k))
            self._group.addButton(btn)
            lay.addWidget(btn)
            self._buttons[key] = btn

        if orientation == QtCore.Qt.Vertical:
            lay.addStretch()  # keep buttons pinned to the top of the sidebar

    @staticmethod
    def _tinted_icon(path, color, size=18):
        """A copy of the icon at *path* fully painted in *color* (SourceAtop,
        so transparent SVG areas stay transparent)."""
        src = QtGui.QIcon(path).pixmap(size, size)
        dst = QtGui.QPixmap(src.size())
        dst.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(dst)
        p.drawPixmap(0, 0, src)
        p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
        p.fillRect(dst.rect(), color)
        p.end()
        return QtGui.QIcon(dst)

    @staticmethod
    def _segment_qss():
        return """
            QToolButton {
                background: transparent;
                border: 2px solid transparent;
                border-radius: 11px;
                color: rgba(70, 80, 95, 215);
                font-size: 12px; font-weight: 600;
                padding: 0px 9px;
                text-align: left;
            }
            QToolButton:hover {
                background: rgba(255, 255, 255, 120);
            }
            QToolButton:checked {
                background: rgba(255, 255, 255, 240);
                border: 2px solid rgba(10, 163, 230, 110);
                color: rgba(0, 90, 135, 250);
                font-weight: 700;
            }
            QToolButton:checked:hover {
                background: rgba(255, 255, 255, 250);
            }
        """

    def set_active(self, key):
        # NOTE: the glow used to be a QGraphicsDropShadowEffect applied to the
        # checked QToolButton. Combining a graphics effect with a stylesheet
        # background/border on a QToolButton is a known Qt gotcha - the
        # button can render fully blank (icon, text, and background all
        # gone) depending on platform/paint timing. The QSS-only halo below
        # (a wider, low-alpha border standing in for "glow") gets the same
        # look with zero risk of that failure mode.
        if key not in self._buttons:
            return
        for k, btn in self._buttons.items():
            is_active = k == key
            btn.setChecked(is_active)
            icons = self._icons.get(k)
            if icons is not None:
                btn.setIcon(icons[1] if is_active else icons[0])
        if key != self._active_key:
            self._active_key = key
            self.modeChanged.emit(key)

    def active_key(self):
        return self._active_key


class _GlassPanel(QtWidgets.QFrame):
    """The overlay's main frosted panel - background/border/radius are
    painted directly instead of via QSS.

    The fullscreen toggle animates alpha/border-width/radius every frame.
    Driving that through `setStyleSheet()` (the original approach) means a
    full CSS reparse + repolish of this frame AND its entire child tree
    (header, sidebar, the active mode's whole widget tree) on every tick -
    cheap when this panel was simple, but the per-mode content has since
    grown a lot heavier (chips, option cards, scroll areas...), and that
    repolish cascade is what showed up as animation stutter. Painting these
    three properties manually and calling `update()` instead skips the
    style system entirely, so a tick is just "redraw one rounded rect."
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bg_alpha = 235
        self._border_width = 1.5
        self._radius = 12.0

    def set_appearance(self, alpha, border_width, radius):
        self._bg_alpha = alpha
        self._border_width = border_width
        self._radius = radius
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        bw = self._border_width
        rect = QtCore.QRectF(self.rect()).adjusted(bw / 2, bw / 2, -bw / 2, -bw / 2)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(255, 255, 255, int(self._bg_alpha)))
        if self._radius > 0:
            painter.drawRoundedRect(rect, self._radius, self._radius)
        else:
            painter.drawRect(rect)

        if bw > 0:
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 230), bw))
            painter.setBrush(QtCore.Qt.NoBrush)
            if self._radius > 0:
                painter.drawRoundedRect(rect, self._radius, self._radius)
            else:
                painter.drawRect(rect)
        painter.end()


# ======================================================================
#  Container
# ======================================================================
class DataManagementWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ICON_MAIN = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "import-export.svg"
        )
        self.ICON_EXPAND = os.path.join(Architecture.get_path(), "QATCH", "icons", "expand.svg")
        self.ICON_COLLAPSE = os.path.join(Architecture.get_path(), "QATCH", "icons", "collapse.svg")

        self.parent = parent

        # Shared machinery, injected into every mode.
        self.services = DataServices(self)

        # --- Overlay / glass setup (child-widget scrim; no WA_TranslucentBackground) ---
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        if parent is not None:
            parent.installEventFilter(self)
            top = parent.window()
            if top is not parent:
                top.installEventFilter(self)

        # Animation / geometry state
        self._scrim_alpha = 0
        self._panel_alpha = 235
        self._closing = False
        self._is_fullscreen = False
        self._default_margin_pct = 0.175
        self._revealed = False
        self._current_key = None  # active mode (container-tracked, not segmented)
        self._slide_group = None  # running slide transition, if any
        self._slide_clip = None  # proxy clip frame for the running slide, if any
        self._fade_anim = None  # single reusable open/close fade
        self._fs_anim = None  # fullscreen geometry animation
        self._close_in_progress = False
        self._close_fade_proxy = None  # static-pixmap stand-in animated during close

        self.base_layout = QtWidgets.QVBoxLayout(self)
        self.base_layout.setContentsMargins(0, 0, 0, 0)

        self.glass_frame = _GlassPanel(self)
        self.glass_frame.setObjectName("dmview")

        # Fade via a composited opacity effect (animating the property is a
        # paint-time multiply) instead of re-setting the stylesheet each frame,
        # which would reparse + repolish the entire glass subtree per frame.
        self._glass_opacity = QtWidgets.QGraphicsOpacityEffect(self.glass_frame)
        self._glass_opacity.setOpacity(1.0)
        self.glass_frame.setGraphicsEffect(self._glass_opacity)

        self.main_layout = QtWidgets.QVBoxLayout(self.glass_frame)
        self.main_layout.setContentsMargins(12, 8, 12, 12)
        self.main_layout.setSpacing(10)

        self._modes = {}  # key -> mode widget
        self._build_taskbar()
        self._build_stack()

        # Header across the top.
        self.main_layout.addLayout(self.top_section_layout)

        # Body: vertical sidebar on the left, content stack on the right.
        self.body_layout = QtWidgets.QHBoxLayout()
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(10)
        self.body_layout.addWidget(self.sidebar_frame, 0)
        self.body_layout.addWidget(self.content_stack, 1)
        self.main_layout.addLayout(self.body_layout, 1)

        self.base_layout.addWidget(self.glass_frame)
        self.setLayout(self.base_layout)

        # Start hidden + pre-fitted to avoid the tiny-dialog flash.
        self.hide()
        self._scrim_alpha = 0
        self.glass_frame.hide()
        self._refit_to_parent()

    # ------------------------------------------------------------------
    #  Build
    # ------------------------------------------------------------------
    @staticmethod
    def _tinted_icon(path, color, size=14):
        """Returns a copy of the icon at *path* fully painted in *color*.

        Uses SourceAtop composition so the tint respects the original alpha
        channel - transparent SVG areas stay transparent.
        """
        src = QtGui.QIcon(path).pixmap(size, size)
        dst = QtGui.QPixmap(src.size())
        dst.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(dst)
        p.drawPixmap(0, 0, src)
        p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
        p.fillRect(dst.rect(), color)
        p.end()
        return QtGui.QIcon(dst)

    def _build_taskbar(self):
        # Header row (title only) sits across the top.
        self.top_section_layout = QtWidgets.QVBoxLayout()
        self.top_section_layout.setSpacing(10)

        self.header_layout = QtWidgets.QHBoxLayout()
        self.header_layout.setContentsMargins(0, 0, 0, 0)
        self.window_icon_label = QtWidgets.QLabel()
        self.window_icon_label.setPixmap(QtGui.QIcon(self.ICON_MAIN).pixmap(16, 16))
        self.window_title_label = QtWidgets.QLabel("Data Management")
        self.window_title_label.setStyleSheet(
            "QLabel { color:#333; font-weight:bold; font-size:13px; background:transparent; }"
        )
        self.header_layout.addWidget(self.window_icon_label)
        self.header_layout.addWidget(self.window_title_label)
        self.header_layout.addStretch()
        self.top_section_layout.addLayout(self.header_layout)

        # Vertical sidebar (glass pill) holding the mode selector, on the left.
        self.sidebar_frame = QtWidgets.QFrame()
        self.sidebar_frame.setObjectName("sidebarFrame")
        self.sidebar_frame.setStyleSheet("""
            QFrame#sidebarFrame {
                background: rgba(248, 250, 253, 130);
                border: 1px solid rgba(255, 255, 255, 200);
                border-radius: 16px;
            }
        """)
        # NOTE: no QGraphicsDropShadowEffect here - the glass frame now carries a
        # QGraphicsOpacityEffect for fading, and Qt doesn't compose nested
        # widget graphics effects reliably. The border provides the separation.

        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar_frame)
        sidebar_layout.setContentsMargins(4, 6, 4, 6)
        sidebar_layout.setSpacing(0)

        # Per-mode icon placeholders. These point at expected SVG filenames in
        # the icons folder; drop the assets in to light them up. Until then the
        # buttons fall back to text-only (see GlassSegmentedControl.__init__).
        _icon_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")
        _mode_icons = {
            "import": os.path.join(_icon_dir, "import.svg"),
            "export": os.path.join(_icon_dir, "export.svg"),
            "recover": os.path.join(_icon_dir, "recover.svg"),
            "advanced": os.path.join(_icon_dir, "gear.svg"),
            "history": os.path.join(_icon_dir, "history.svg"),
        }
        modes = [
            (cls.MODE_KEY, cls.MODE_LABEL, _mode_icons.get(cls.MODE_KEY)) for cls in MODE_CLASSES
        ]
        self.segmented = GlassSegmentedControl(modes, orientation=QtCore.Qt.Vertical)
        self.segmented.modeChanged.connect(self._on_mode_changed)
        sidebar_layout.addWidget(self.segmented)

        # Window control buttons (positioned absolutely on refit).
        button_size = 28
        icon_size = QtCore.QSize(14, 14)

        # Fullscreen isn't destructive like Close, so its hover tints to a
        # lighter grey instead of the close button's red.
        self._FS_NORMAL = QtGui.QColor(110, 120, 130, 190)
        self._FS_HOVER = QtGui.QColor(175, 185, 196, 235)
        self._fs_normal_icon = self._tinted_icon(self.ICON_EXPAND, self._FS_NORMAL, size=14)
        self._fs_hover_icon = self._tinted_icon(self.ICON_EXPAND, self._FS_HOVER, size=14)

        self.btn_fullscreen = QtWidgets.QPushButton("", self)
        self.btn_fullscreen.setFixedSize(button_size, button_size)
        self.btn_fullscreen.setIcon(self._fs_normal_icon)
        self.btn_fullscreen.setIconSize(icon_size)
        self.btn_fullscreen.setToolTip("Toggle Fullscreen")
        self.btn_fullscreen.setStyleSheet("QPushButton { background: transparent; border: none; }")
        self.btn_fullscreen.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_fullscreen.installEventFilter(self)
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)

        self.btn_close = QtWidgets.QPushButton("x", self)
        self.btn_close.setFixedSize(button_size, button_size)
        self.btn_close.setToolTip("Close")
        self.btn_close.setStyleSheet("""
            QPushButton {
                background: transparent; border: none;
                color: rgba(110, 120, 130, 190);
                font-size: 18px; font-weight: bold; padding-bottom: 2px;
            }
            QPushButton:hover   { color: rgba(210, 55, 55, 230); }
            QPushButton:pressed { color: rgba(160, 30, 30, 255); }
        """)
        self.btn_close.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_close.clicked.connect(self.close)

    def _build_stack(self):
        self.content_stack = QtWidgets.QStackedWidget()
        self.content_stack.setObjectName("contentStack")
        self.content_stack.setStyleSheet("QStackedWidget#contentStack { background: transparent; }")
        # Instantiate each mode with the shared services and register it.
        for cls in MODE_CLASSES:
            mode = cls(self.services, parent=self.content_stack)
            self._modes[cls.MODE_KEY] = mode
            self.content_stack.addWidget(mode)

    def _on_mode_changed(self, key):
        prev_key = self._current_key
        mode = self._modes.get(key)
        if mode is None or key == prev_key:
            return

        prev_mode = self._modes.get(prev_key) if prev_key else None
        if prev_mode is not None:
            prev_mode.on_leave()

        # First switch (or not yet revealed): no slide, just set the page.
        if prev_mode is None or not self._revealed or not self.isVisible():
            self.content_stack.setCurrentWidget(mode)
            self._current_key = key
            # When the overlay isn't visible yet (e.g. open_mode set the tab
            # before the fade), DON'T populate here - heavy on_enter work would
            # block the open animation. The reveal calls on_enter after the fade
            # has started. Only populate inline if we're already visible.
            if self._revealed and self.isVisible():
                mode.on_enter()
            return

        # Direction: moving DOWN the sidebar slides the new page up from below;
        # moving UP slides it down from above.
        going_down = _MODE_ORDER.get(key, 0) > _MODE_ORDER.get(prev_key, 0)
        self._slide_to(prev_mode, mode, key, going_down)

    def _teardown_slide(self):
        """Stop any running slide and destroy its proxy clip immediately.

        QAbstractAnimation.stop() does NOT emit finished(), so the _finish
        callback never runs on an interrupted slide. Without this, the proxy
        clip (and its captured page pixmaps) stay parented to the glass frame
        and keep painting over the live stack for the rest of the session -
        the persistent "ghost of the previous mode" bug.
        """
        group = self._slide_group
        if group is not None:
            try:
                group.stop()
                group.deleteLater()
            except RuntimeError:
                pass
            self._slide_group = None
        clip = self._slide_clip
        if clip is not None:
            try:
                clip.hide()
                clip.setParent(None)
                clip.deleteLater()
            except RuntimeError:
                pass
            self._slide_clip = None
        # The live stack must be visible again once the proxies are gone.
        if self.content_stack is not None:
            self.content_stack.show()

    def _slide_to(self, old_mode, new_mode, key, going_down):
        """Cross-slide the outgoing/incoming pages vertically over the stack."""
        # Cancel any in-flight slide and fully tear down its proxy clip.
        self._teardown_slide()

        stack = self.content_stack
        geo = stack.geometry()  # in glass_frame coords
        size = geo.size()
        if size.width() <= 0 or size.height() <= 0:
            # Layout not settled - fall back to instant.
            stack.setCurrentWidget(new_mode)
            self._current_key = key
            new_mode.on_enter()
            return

        h = size.height()

        # Grab static pixmaps of both pages at the stack's current size.
        old_mode.resize(size)
        old_pix = old_mode.grab()
        # Make the incoming page current first so it's laid out, then grab it.
        stack.setCurrentWidget(new_mode)
        new_mode.resize(size)
        new_mode.on_enter()
        new_pix = new_mode.grab()

        # Clip wrapper over the stack region so the moving labels are masked.
        clip = QtWidgets.QFrame(self.glass_frame)
        clip.setObjectName("slideClip")
        clip.setStyleSheet("QFrame#slideClip { background: transparent; border: none; }")
        clip.setGeometry(geo)
        clip.show()
        clip.raise_()
        self._slide_clip = clip

        rest = QtCore.QPoint(0, 0)
        if going_down:
            old_end = QtCore.QPoint(0, -h)  # old exits upward
            new_start = QtCore.QPoint(0, h)  # new enters from below
        else:
            old_end = QtCore.QPoint(0, h)  # old exits downward
            new_start = QtCore.QPoint(0, -h)  # new enters from above

        old_lbl = QtWidgets.QLabel(clip)
        old_lbl.setPixmap(old_pix)
        old_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        old_lbl.setGeometry(QtCore.QRect(rest, size))
        old_lbl.show()

        new_lbl = QtWidgets.QLabel(clip)
        new_lbl.setPixmap(new_pix)
        new_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        new_lbl.setGeometry(QtCore.QRect(new_start, size))
        new_lbl.show()
        new_lbl.raise_()

        # Hide the live stack during the slide so only the proxies show.
        stack.hide()

        anim_old = QtCore.QPropertyAnimation(old_lbl, b"pos", self)
        anim_old.setDuration(260)
        anim_old.setEasingCurve(QtCore.QEasingCurve.OutQuint)
        anim_old.setStartValue(rest)
        anim_old.setEndValue(old_end)

        anim_new = QtCore.QPropertyAnimation(new_lbl, b"pos", self)
        anim_new.setDuration(260)
        anim_new.setEasingCurve(QtCore.QEasingCurve.OutQuint)
        anim_new.setStartValue(new_start)
        anim_new.setEndValue(rest)

        group = QtCore.QParallelAnimationGroup(self)
        group.addAnimation(anim_old)
        group.addAnimation(anim_new)

        def _finish():
            stack.show()
            stack.raise_()
            clip.hide()
            clip.setParent(None)
            clip.deleteLater()
            self._slide_clip = None
            self._slide_group = None
            self._current_key = key
            self.btn_close.raise_()
            self.btn_fullscreen.raise_()

        group.finished.connect(_finish)
        self._slide_group = group
        group.start()

    # ------------------------------------------------------------------
    #  Public entry points (call surface)
    # ------------------------------------------------------------------
    def open_mode(self, key):
        """Open the overlay on a specific mode key."""
        self.segmented.set_active(key)
        self.setVisible(True)

    def showNormal(self, tab_idx=0):
        """Backwards-compatible shim for the old Ui_Export call sites."""
        self.open_mode(_LEGACY_TAB_TO_KEY.get(tab_idx, "import"))

    # ------------------------------------------------------------------
    #  Geometry / fit
    # ------------------------------------------------------------------
    def _apply_shadow(self, widget, blur_radius=15, alpha=40, offset=(0, 4)):
        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(blur_radius)
        shadow.setColor(QtGui.QColor(0, 0, 0, alpha))
        shadow.setOffset(offset[0], offset[1])
        widget.setGraphicsEffect(shadow)

    def _refit_to_parent(self, frac=None):
        if self.parent is None:
            geo = QtWidgets.QApplication.primaryScreen().availableGeometry()
        else:
            geo = self.parent.rect()
        self.setGeometry(geo)

        # If a fullscreen toggle animation is actively running, let it handle the
        # margins. _apply_margin_frac (called below) repositions btn_close/
        # btn_fullscreen, and those buttons have an eventFilter installed on
        # them, so every animation frame's button move re-enters here via a
        # Move event - without this guard it would snap the margins to the
        # final _is_fullscreen state on every tick, defeating the animation.
        if self._fs_anim is not None and self._fs_anim.state() == QtCore.QAbstractAnimation.Running:
            return

        if frac is None:
            frac = 0.0 if self._is_fullscreen else self._default_margin_pct
        self._apply_margin_frac(frac)

    def _apply_margin_frac(self, frac):
        """Position the glass panel inset by `frac` of each dimension.

        frac == 0 -> fullscreen (flush, no radius/border); frac == self._default_margin_pct
        -> fully inset. Border, radius and background alpha interpolate
        continuously with frac (matching the live margin) so the fullscreen
        toggle reads as one smooth motion instead of snapping partway through.
        """
        w, h = self.width(), self.height()
        mx = int(w * frac)
        my = int(h * frac)
        self.base_layout.setContentsMargins(mx, my, mx, my)

        p = 0.0 if self._default_margin_pct <= 0 else min(1.0, frac / self._default_margin_pct)
        alpha = int(255 + (235 - 255) * p)
        border = 1.5 * p
        radius = 12.0 * p
        self.glass_frame.set_appearance(alpha, border, radius)

        self._update_buttons_position(mx, my)

    def _update_buttons_position(self, mx, my):
        w = self.width()
        btn_sz = 28
        close_x = w - mx - 14 - btn_sz
        close_y = my + 14
        self.btn_close.setGeometry(close_x, close_y, btn_sz, btn_sz)
        self.btn_close.raise_()
        self.btn_fullscreen.setGeometry(close_x - 6 - btn_sz, close_y, btn_sz, btn_sz)
        self.btn_fullscreen.raise_()

    def toggle_fullscreen(self):
        self._is_fullscreen = not self._is_fullscreen

        # Swap the icon to reflect the action the button now performs.
        _icon_path = self.ICON_COLLAPSE if self._is_fullscreen else self.ICON_EXPAND
        self._fs_normal_icon = self._tinted_icon(_icon_path, self._FS_NORMAL, size=14)
        self._fs_hover_icon = self._tinted_icon(_icon_path, self._FS_HOVER, size=14)
        if self.btn_fullscreen.underMouse():
            self.btn_fullscreen.setIcon(self._fs_hover_icon)
        else:
            self.btn_fullscreen.setIcon(self._fs_normal_icon)

        target = 0.0 if self._is_fullscreen else self._default_margin_pct
        start = self._default_margin_pct if self._is_fullscreen else 0.0

        # Tear down any prior fullscreen animation.
        if self._fs_anim is not None:
            try:
                self._fs_anim.stop()
                self._fs_anim.valueChanged.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._fs_anim.deleteLater()
            self._fs_anim = None

        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(240)
        anim.setEasingCurve(QtCore.QEasingCurve.InOutCubic)
        anim.setStartValue(float(start))
        anim.setEndValue(float(target))
        anim.valueChanged.connect(lambda f: self._apply_margin_frac(float(f)))

        def _fs_done():
            self._apply_margin_frac(target)
            done = self._fs_anim
            self._fs_anim = None
            if done is not None:
                done.deleteLater()

        anim.finished.connect(_fs_done)
        self._fs_anim = anim
        anim.start()

    # ------------------------------------------------------------------
    #  Show / hide with fade
    # ------------------------------------------------------------------
    def setVisible(self, visible):
        if visible and not self.isVisible():
            self.services.start()  # spin up the shared USB loop on open
            self._scrim_alpha = 0
            self._panel_alpha = 0
            self._set_glass_opacity(0.0)  # start transparent; fade brings it in
            self.glass_frame.hide()
            self._refit_to_parent()
            super().setVisible(True)
            self._refit_to_parent()
            if self.layout() is not None:
                self.layout().activate()

            def _reveal():
                self._refit_to_parent()
                if self.layout() is not None:
                    self.layout().activate()
                self._revealed = True
                self.glass_frame.show()
                self.glass_frame.raise_()
                self.btn_close.raise_()
                self.btn_fullscreen.raise_()
                self.update()
                self._animate_open()
                # Populate the active mode AFTER the fade has started, on a
                # later event-loop turn, so heavy on_enter work (e.g. parsing a
                # long import/export history) doesn't block the open animation.
                key = self.segmented.active_key()

                def _populate():
                    if key in self._modes:
                        self.content_stack.setCurrentWidget(self._modes[key])
                        self._current_key = key
                        self._modes[key].on_enter()

                QtCore.QTimer.singleShot(16, _populate)

            QtCore.QTimer.singleShot(0, _reveal)
            return
        super().setVisible(visible)

    def showEvent(self, event):
        self._refit_to_parent()
        self.raise_()
        super().showEvent(event)

    def closeEvent(self, event):
        if self._closing:
            event.accept()
            return
        anim = getattr(self, "_fade_anim", None)
        if anim is not None and anim.state() == QtCore.QAbstractAnimation.Running:
            # A close fade is already running; let it finish.
            if self._close_in_progress:
                event.ignore()
                return
        event.ignore()
        self._close_in_progress = True
        self._animate_close()

    def _set_panel_alpha(self, alpha):
        # Kept for compatibility (close/do_close reset state through this), but
        # the glass stylesheet alpha is now FIXED; visual fade is via the
        # opacity effect. We only restyle when border/radius actually change
        # (fullscreen toggle handles that separately).
        self._panel_alpha = int(alpha)

    def _set_glass_opacity(self, frac):
        frac = max(0.0, min(1.0, float(frac)))
        if self._glass_opacity is not None:
            self._glass_opacity.setOpacity(frac)

    # ------------------------------------------------------------------
    #  Animation driver - ONE reusable QVariantAnimation, never accumulate.
    # ------------------------------------------------------------------
    def _stop_anim(self):
        """Stop and fully tear down any running fade so it can't keep firing."""
        anim = getattr(self, "_fade_anim", None)
        if anim is not None:
            try:
                anim.stop()
                anim.valueChanged.disconnect()
                anim.finished.disconnect()
            except (TypeError, RuntimeError):
                pass
            anim.deleteLater()
            self._fade_anim = None

    def _run_fade(
        self,
        scrim_from,
        scrim_to,
        op_from,
        op_to,
        duration,
        easing,
        on_done=None,
        opacity_effect=None,
    ):
        """Animate the scrim alpha (cheap paintEvent) and an opacity effect
        (composited) from fixed endpoints. No per-frame stylesheet.

        `opacity_effect` defaults to the live glass_frame's own effect
        (the open fade). The close fade passes a lightweight effect on a
        static pixmap proxy instead - see `_animate_close`.
        """
        self._stop_anim()
        effect = opacity_effect if opacity_effect is not None else self._glass_opacity
        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(duration)
        anim.setEasingCurve(easing)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)

        def _set_opacity(frac):
            if effect is not None:
                effect.setOpacity(max(0.0, min(1.0, float(frac))))

        def _step(t):
            self._scrim_alpha = int(scrim_from + (scrim_to - scrim_from) * t)
            _set_opacity(op_from + (op_to - op_from) * t)
            self.update()  # repaint the scrim only

        def _settle():
            self._scrim_alpha = int(scrim_to)
            _set_opacity(op_to)
            self.update()
            done_anim = self._fade_anim
            self._fade_anim = None
            if done_anim is not None:
                done_anim.deleteLater()
            if on_done is not None:
                on_done()

        anim.valueChanged.connect(_step)
        anim.finished.connect(_settle)
        self._fade_anim = anim
        anim.start()

    def _animate_open(self):
        self._scrim_alpha = 0
        self._set_glass_opacity(0.0)
        self.update()
        self._run_fade(
            scrim_from=0,
            scrim_to=65,
            op_from=0.0,
            op_to=1.0,
            duration=200,
            easing=QtCore.QEasingCurve.OutQuad,
        )

    def _animate_close(self):
        # Snapshot the panel once and fade a flat pixmap proxy instead of
        # leaving the live QGraphicsOpacityEffect on glass_frame - that
        # effect forces Qt to re-render the ENTIRE live widget tree (header,
        # sidebar, and whatever the active mode's full content is) into an
        # offscreen buffer on every single animation frame. That per-frame
        # full-subtree recomposite is what showed up as close-animation
        # stutter once the per-mode content grew heavier. A static pixmap
        # has nothing left to re-render - fading it is just "blit one image
        # at a lower alpha," every frame, regardless of how complex the
        # live content underneath is.
        self._teardown_close_fade_proxy()
        cur_op = self._glass_opacity.opacity() if self._glass_opacity else 1.0
        geo = self.glass_frame.geometry()
        pix = self.glass_frame.grab()
        self.glass_frame.hide()

        proxy = QtWidgets.QLabel(self)
        proxy.setObjectName("closeFadeProxy")
        proxy.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        proxy.setPixmap(pix)
        proxy.setGeometry(geo)
        proxy.show()
        proxy.raise_()
        # Keep the window-control buttons (siblings of glass_frame, drawn
        # above it) on top of the new proxy too - raise_() puts the proxy
        # at the top of self's stacking order, which would otherwise cover
        # them for the duration of the close fade.
        self.btn_close.raise_()
        self.btn_fullscreen.raise_()
        self._close_fade_proxy = proxy

        proxy_opacity = QtWidgets.QGraphicsOpacityEffect(proxy)
        proxy_opacity.setOpacity(cur_op)
        proxy.setGraphicsEffect(proxy_opacity)

        self._run_fade(
            scrim_from=self._scrim_alpha,
            scrim_to=0,
            op_from=cur_op,
            op_to=0.0,
            duration=180,
            easing=QtCore.QEasingCurve.InQuad,
            on_done=self._do_close,
            opacity_effect=proxy_opacity,
        )

    def _teardown_close_fade_proxy(self):
        proxy = self._close_fade_proxy
        if proxy is not None:
            try:
                proxy.hide()
                proxy.setParent(None)
                proxy.deleteLater()
            except RuntimeError:
                pass
            self._close_fade_proxy = None

    def _do_close(self):
        self.services.stop()  # tear down the shared USB loop on close
        self._stop_anim()
        self._teardown_close_fade_proxy()
        self._teardown_slide()
        self._closing = True
        self.close()
        self._closing = False
        self._close_in_progress = False
        self._is_fullscreen = False
        # Restore the expand icon for the next open.
        self._fs_normal_icon = self._tinted_icon(self.ICON_EXPAND, self._FS_NORMAL, size=14)
        self._fs_hover_icon = self._tinted_icon(self.ICON_EXPAND, self._FS_HOVER, size=14)
        self.btn_fullscreen.setIcon(self._fs_normal_icon)
        self._panel_alpha = 235
        self._current_key = None
        self._set_glass_opacity(1.0)
        self.content_stack.show()
        self.glass_frame.show()
        self._refit_to_parent()

    # ------------------------------------------------------------------
    #  Events
    # ------------------------------------------------------------------
    def eventFilter(self, obj, event):
        if hasattr(self, "btn_fullscreen") and obj is self.btn_fullscreen:
            if event.type() == QtCore.QEvent.Enter:
                self.btn_fullscreen.setIcon(self._fs_hover_icon)
            elif event.type() == QtCore.QEvent.Leave:
                self.btn_fullscreen.setIcon(self._fs_normal_icon)
        if event.type() in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Move):
            if self.isVisible():
                self._refit_to_parent()
        return super().eventFilter(obj, event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, self._scrim_alpha))
        painter.end()

    def mousePressEvent(self, event):
        if not self.glass_frame.geometry().contains(event.pos()):
            self.close()
        else:
            super().mousePressEvent(event)
