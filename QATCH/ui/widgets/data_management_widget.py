"""DataManagementWidget - glassmorphic overlay container.

Owns the overlay lifecycle (open/close, fade, fullscreen, click-outside
dismiss) and the shared `DataServices`. Links in the five mode submodules and
drives switching between them via a vertical sidebar + QStackedWidget.

Replaces `export_widget.Ui_Export` in the controls UI. Compatibility shims
(`showNormal` / `open_mode`) preserve the existing call sites.
"""

import os

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.common.data_service import DataServices
from QATCH.ui.components import SegmentedControl
from QATCH.ui.components.overlay_shell import (
    FULLSCREEN_ANIM_EASING,
    OverlayLifecycleMixin,
    rebuild_fullscreen_icons,
    run_stack_slide,
    run_variant_animation,
    teardown_stack_slide,
)
from QATCH.ui.styles.theme_manager import ThemeManager, surface_panel_qss
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


class _GlassPanel(QtWidgets.QFrame):
    """The overlay's main frosted panel - background/border/radius are
    painted directly instead of via QSS.

    The fullscreen toggle animates alpha/border-width/radius every frame.
    Driving that through `setStyleSheet()` (the original approach) means a
    full CSS reparse + repolish of this frame AND its entire child tree
    (header, chip bar, the active mode's whole widget tree) on every tick -
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
        # WA_TranslucentBackground alone stops Qt from auto-erasing this
        # widget's backing store to an opaque palette color before every
        # paint - needed for correct LIVE rendering of the four corners
        # outside the rounded rect this paintEvent draws (so the dimmed
        # scrim behind shows through instead of solid squares). It does NOT
        # reliably survive grab()/render() for a plain child widget under a
        # non-translucent top-level window though (verified empirically:
        # grab() still comes back with opaque corner pixels) - the close-fade
        # snapshot in _animate_close needs the mask below for that case.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        ThemeManager.instance().themeChanged.connect(lambda _: self.update())

    def set_appearance(self, alpha, border_width, radius):
        self._bg_alpha = alpha
        self._border_width = border_width
        self._radius = radius
        self._update_mask()
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_mask()

    def _update_mask(self) -> None:
        """Clips the widget's paintable area to the rounded-rect shape, so
        the corners outside it are excluded from rendering entirely - both
        on-screen AND via grab()/render(), which respect an active mask even
        when they don't reliably respect WA_TranslucentBackground alone.
        This is what actually keeps grab()-based snapshots (the close-fade
        proxy) free of the opaque-corner artifact.
        """
        if self._radius <= 0:
            self.clearMask()
            return
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(self.rect()), self._radius, self._radius)
        self.setMask(QtGui.QRegion(path.toFillPolygon().toPolygon()))

    def paintEvent(self, event):
        tok = ThemeManager.instance().tokens()
        base = tok["plot_glass_base"]
        rim = tok["plot_glass_rim"]

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        bw = self._border_width
        rect = QtCore.QRectF(self.rect()).adjusted(bw / 2, bw / 2, -bw / 2, -bw / 2)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(base[0], base[1], base[2], int(self._bg_alpha)))
        if self._radius > 0:
            painter.drawRoundedRect(rect, self._radius, self._radius)
        else:
            painter.drawRect(rect)

        if bw > 0:
            painter.setPen(QtGui.QPen(QtGui.QColor(rim[0], rim[1], rim[2], 230), bw))
            painter.setBrush(QtCore.Qt.NoBrush)
            if self._radius > 0:
                painter.drawRoundedRect(rect, self._radius, self._radius)
            else:
                painter.drawRect(rect)
        painter.end()


# ======================================================================
#  Container
# ======================================================================
class DataManagementWidget(OverlayLifecycleMixin, QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ICON_MAIN = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "import-export.svg"
        )
        self.ICON_EXPAND = os.path.join(Architecture.get_path(), "QATCH", "icons", "expand.svg")
        self.ICON_COLLAPSE = os.path.join(Architecture.get_path(), "QATCH", "icons", "collapse.svg")

        # Shared machinery, injected into every mode.
        self.services = DataServices(self)

        # Animation / geometry state not covered by the shared overlay shell.
        self._current_key = None  # active mode (container-tracked, not nav)
        self._slide_group = None  # running slide transition, if any (see overlay_shell.run_stack_slide)
        self._slide_clip = None  # proxy clip frame for the running slide, if any
        self._slide_stack = None  # stack hidden by the running slide, if any
        self._close_fade_proxy = None  # static-pixmap stand-in animated during close

        # Overlay scaffolding (scrim/panel state, base_layout + opacity
        # effect + main_layout, parent-resize event filter) - shared with
        # UserProfilesManagerWidget/UserPreferencesWidget via
        # OverlayLifecycleMixin. `_GlassPanel` is custom-painted (see its
        # class docstring for why), so it's built here and handed in rather
        # than letting the shell create a plain QSS-styled QFrame.
        glass_frame = _GlassPanel(self)
        glass_frame.setObjectName("dmview")
        self._init_overlay_shell(
            parent,
            "dmview",
            panel_alpha=235,
            margin_pct=0.175,
            # Same outer margins/spacing as UserPreferencesWidget's overlay
            # shell, so the two panels read as one consistent chrome system.
            content_margins=(20, 14, 20, 20),
            content_spacing=14,
            glass_frame=glass_frame,
        )

        self._modes = {}  # key -> mode widget
        self._build_nav()
        self._build_stack()

        # Header across the top; below it, a vertical sidebar of modes next
        # to the content stack - same shell convention as
        # UserPreferencesWidget's section nav / content_stack split.
        self.main_layout.addLayout(self.header_layout)
        body_row = QtWidgets.QHBoxLayout()
        # No gap: the nav sits directly against the content well so the
        # highlighted mode and the content card read as adjacent, related
        # surfaces instead of two panels floating apart - same convention as
        # UserPreferencesWidget's nav/section-well split.
        body_row.setSpacing(0)
        body_row.addWidget(self.nav)
        body_row.addWidget(self.content_stack, 1)
        self.main_layout.addLayout(body_row, 1)

        # Start hidden + pre-fitted to avoid the tiny-dialog flash.
        self._finish_overlay_shell()

    # ------------------------------------------------------------------
    #  Build
    # ------------------------------------------------------------------
    def _build_nav(self):
        self.header_layout = self._build_overlay_header(
            self.ICON_MAIN, "Data Management", fullscreen=True
        )

        # Vertical sidebar of modes beside the content stack - same shell
        # convention as UserPreferencesWidget's section nav (fixed-width
        # column, top-aligned, one mode highlighted at a time).
        #
        # Per-mode icon placeholders. These point at expected SVG filenames in
        # the icons folder; drop the assets in to light them up. Until then the
        # buttons fall back to text-only (see SegmentedControl.__init__).
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
        self.nav = SegmentedControl(modes, orientation=QtCore.Qt.Vertical)
        self.nav.modeChanged.connect(self._on_mode_changed)

        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    # ------------------------------------------------------------------
    #  Theming
    # ------------------------------------------------------------------
    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_theme()

    def _apply_theme(self) -> None:
        self._refresh_header_theme()
        # Guarded: the first call happens from _build_nav(), before
        # _build_stack() has created content_stack.
        content_stack = getattr(self, "content_stack", None)
        if content_stack is not None:
            content_stack.setStyleSheet(surface_panel_qss("contentStack", radius=11))

    def _rebuild_fs_icons(self) -> None:
        """Rebuilds the fullscreen button's normal/hover icon pixmaps for the
        icon matching the current fullscreen state - shared with
        UserProfilesManagerWidget via `overlay_shell.rebuild_fullscreen_icons`."""
        rebuild_fullscreen_icons(self, self.ICON_EXPAND, self.ICON_COLLAPSE)

    def _build_stack(self):
        self.content_stack = QtWidgets.QStackedWidget()
        self.content_stack.setObjectName("contentStack")
        # QStackedWidget is a QFrame subclass, so the same bordered/rounded
        # "well" chrome UserPreferencesWidget uses for its section pages
        # applies here directly - each mode page itself paints transparent
        # (see DataModeWidget), so this background/border shows through as
        # the content card behind whichever mode is active. radius=11
        # matches SegmentedControl's vertical checked-row radius for visual
        # consistency with the nav beside it.
        self.content_stack.setStyleSheet(surface_panel_qss("contentStack", radius=11))
        # Let _slide_to truly hide() the stack during a mode transition
        # (cheapest possible "don't paint this") while it still reserves its
        # layout space. Without this, stack.hide()/show() at the start/end of
        # every slide invalidates main_layout (content_stack briefly collapses
        # to zero height), which forces a full repaint of glass_frame - header,
        # chip bar, everything - bracketing each slide. That's what showed up
        # as the whole window seeming to re-render during a mode switch.
        # Same technique as ExportMode._slide_step's step_stack.
        stack_policy = self.content_stack.sizePolicy()
        stack_policy.setRetainSizeWhenHidden(True)
        self.content_stack.setSizePolicy(stack_policy)
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

        # Direction: picking a mode BELOW the current one in the sidebar
        # slides the new page in from below (old exits up); picking one
        # ABOVE slides it in from above - matching the vertical sidebar's
        # top-to-bottom ordering.
        going_down = _MODE_ORDER.get(key, 0) > _MODE_ORDER.get(prev_key, 0)
        self._slide_to(prev_mode, mode, key, going_down)

    def _slide_to(self, old_mode, new_mode, key, going_down):
        """Cross-slide the outgoing/incoming pages vertically over the stack
        via the shared `overlay_shell.run_stack_slide` helper - matches the
        sidebar's top-to-bottom ordering (vs. UserPreferencesWidget's
        section nav, which drives the same helper for its own pages)."""

        def _enter_new():
            new_mode.on_enter()

        def _finished():
            self._current_key = key
            self.btn_close.raise_()
            self.btn_fullscreen.raise_()

        run_stack_slide(
            self,
            self.content_stack,
            old_mode,
            new_mode,
            axis="y",
            forward=going_down,
            before_show_new=_enter_new,
            on_finished=_finished,
        )

    # ------------------------------------------------------------------
    #  Public entry points (call surface)
    # ------------------------------------------------------------------
    def open_mode(self, key):
        """Open the overlay on a specific mode key."""
        self.nav.set_active(key)
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

    def _apply_panel_appearance(self, frac: float) -> None:
        """Interpolates the (custom-painted) glass panel's alpha/border/
        radius continuously with `frac`, matching the live margin so the
        fullscreen toggle reads as one smooth motion instead of snapping
        partway through. frac == 0 -> fullscreen (flush, no radius/border);
        frac == `_default_margin_pct` -> fully inset."""
        p = 0.0 if self._default_margin_pct <= 0 else min(1.0, frac / self._default_margin_pct)
        alpha = int(255 + (235 - 255) * p)
        border = 1.5 * p
        radius = 12.0 * p
        self.glass_frame.set_appearance(alpha, border, radius)

    def toggle_fullscreen(self):
        self._is_fullscreen = not self._is_fullscreen

        # Swap the icon to reflect the action the button now performs.
        self._rebuild_fs_icons()

        target = 0.0 if self._is_fullscreen else self._default_margin_pct
        start = self._default_margin_pct if self._is_fullscreen else 0.0

        def _step(t):
            frac = start + (target - start) * t
            self._apply_margin_frac(frac)

        run_variant_animation(
            self, "_fs_anim", duration=240, easing=FULLSCREEN_ANIM_EASING, on_step=_step,
        )

    # ------------------------------------------------------------------
    #  Show / hide with fade
    # ------------------------------------------------------------------
    def _on_before_reveal(self) -> None:
        self.services.start()  # spin up the shared USB loop on open

    def _on_after_reveal(self) -> None:
        # Populate the active mode AFTER the fade has started, on a later
        # event-loop turn, so heavy on_enter work (e.g. parsing a long
        # import/export history) doesn't block the open animation.
        key = self.nav.active_key()

        def _populate():
            if key in self._modes:
                self.content_stack.setCurrentWidget(self._modes[key])
                self._current_key = key
                self._modes[key].on_enter()

        QtCore.QTimer.singleShot(16, _populate)

    def resizeEvent(self, event):
        # The eventFilter below only catches resizes of `parent`/its window -
        # it does not fire when Qt resizes `self` directly (this widget IS
        # the top-level with no parent, or the parent resizes it via layout
        # rather than a bare geometry change the filter sees). Without this,
        # _apply_margin_frac's mx/my margins are computed once against
        # whatever self.width()/height() happened to be at the last refit and
        # then stay frozen - after a fullscreen-toggle animation followed by
        # a later resize, glass_frame (and everything inside it, including
        # the mode bar strip) can end up rendered at a stale, too-small size
        # instead of tracking the panel's actual current width.
        super().resizeEvent(event)
        if self.isVisible():
            self._refit_to_parent()

    # ------------------------------------------------------------------
    #  Overlay shell hooks (see QATCH.ui.components.overlay_shell for the
    #  shared init scaffolding, fade engine, reveal-on-open sequence, and
    #  scrim/mouse event handlers this widget inherits from
    #  OverlayLifecycleMixin)
    # ------------------------------------------------------------------
    def _animate_close(self):
        # Snapshot the panel once and fade a flat pixmap proxy instead of
        # leaving the live QGraphicsOpacityEffect on glass_frame - that
        # effect forces Qt to re-render the ENTIRE live widget tree (header,
        # chip bar, and whatever the active mode's full content is) into an
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
        # grab() bakes an opaque erase color into glass_frame's corners
        # (outside its rounded-rect paint) rather than leaving them
        # transparent - Qt's grab()/render() don't reliably respect a source
        # widget's WA_TranslucentBackground for a plain child widget under a
        # non-translucent top-level window. Masking the PROXY itself to the
        # same rounded shape sidesteps that entirely: whatever is baked into
        # the pixmap's corners, this on-screen label simply never displays
        # those pixels - this is what actually fixes the white/opaque
        # corners during the close fade.
        proxy_radius = self.glass_frame._radius
        proxy_path = QtGui.QPainterPath()
        proxy_path.addRoundedRect(QtCore.QRectF(proxy.rect()), proxy_radius, proxy_radius)
        proxy.setMask(QtGui.QRegion(proxy_path.toFillPolygon().toPolygon()))
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
        self._teardown_close_fade_proxy()
        teardown_stack_slide(self)
        self._is_fullscreen = False
        # Restore the expand icon for the next open.
        self._rebuild_fs_icons()
        self._panel_alpha = 235
        self._current_key = None
        self.content_stack.show()
        super()._do_close()
