"""
QATCH.ui.components.segmented_control.py

Segmented selection control for the QATCH flat control system.

This module provides `SegmentedControl`, a reusable mutually-exclusive
selection widget built from checkable `QToolButton` instances managed by a
`QButtonGroup`. The control supports both horizontal and vertical
orientations and can display either text-only segments or segments combining
text with icons.

Segment appearance is derived from the active application's `flat_*` theme
tokens through `ThemeManager`, allowing the control to remain visually
consistent across light and dark themes. Icon assets are retained by source
path so their tint can be regenerated whenever the active theme changes.

The control was extracted from `data_management_widget.py`, where the
original implementation was specific to the data-management sidebar and
used hardcoded light-theme colors. Centralizing the implementation here
allows the same interaction and styling model to be shared by other UI
components, including text-only filter chips used by `data_mode_history.py`.

Typical usage::

    # Icon + label segments in a vertical sidebar.
    nav = SegmentedControl(
        [
            ("import", "Import", icon_path),
            ("export", "Export", icon_path2),
        ],
        orientation=QtCore.Qt.Vertical,
    )
    nav.modeChanged.connect(handler)
    nav.set_active("import")

    # Text-only segments in a horizontal filter row.
    filt = SegmentedControl(
        [
            ("all", "All"),
            ("export", "Export"),
        ],
        orientation=QtCore.Qt.Horizontal,
    )

The control exposes the selected segment through its active-key state and
emits a change signal when the user selects a different segment.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-18
"""

from __future__ import annotations

import os

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.icon_utils import tinted_icon
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css


class SegmentedControl(QtWidgets.QFrame):
    """Theme-aware group of mutually exclusive selectable segments.

    Provides a horizontal or vertical collection of checkable
    :class:`~PyQt5.QtWidgets.QToolButton` instances managed by an exclusive
    :class:`~PyQt5.QtWidgets.QButtonGroup`. Each segment is identified by a
    unique key and may optionally display an icon alongside its label.

    The control supports the standard flat segmented appearance as well as
    the horizontal `"chips"` variant. Visual styling, icon tinting, and
    other theme-dependent properties are refreshed automatically when the
    application's active theme changes.

    Attributes:
        modeChanged (pyqtSignal): Emitted with the key of the segment whenever
            the active selection changes.
        _orientation: Qt orientation used to lay out the segments.
        _icon_size: Configured icon size in pixels.
        _filled: Whether filled container styling is enabled.
        _variant: Selected visual styling variant.
        _chip_shadow: Optional shadow effect applied to the active chip in
            the `"chips"` variant.
        _is_chips: Whether the control is using the horizontal chip variant.
        _radius: Corner radius used by the control's visual styling.
        _buttons (dict): Mapping of segment keys to their corresponding
            :class:`QToolButton` instances.
        _icons (dict): Mapping of segment keys to inactive and active
            :class:`QIcon` pairs.
        _icon_paths (dict): Mapping of segment keys to source icon paths,
            retained so icons can be re-tinted after a theme change.
        _active_key (Optional[str]): Key of the currently selected segment,
            or `None` when no segment is active.
        _group (QButtonGroup): Exclusive button group containing all segment
            buttons.

    Note:
        Icon paths are only registered when the referenced file exists.
        Missing or invalid icon paths cause the corresponding segment to fall
        back to text-only presentation.

        Vertical controls use a fixed sidebar width and keep their segments
        pinned to the top of the control. Horizontal controls use a fixed
        height and size their segments according to the selected visual
        variant.

        The control connects to :attr:`ThemeManager.themeChanged` during
        initialization so theme-dependent styling and icon colors remain
        synchronized with the application's active theme.
    """

    modeChanged = QtCore.pyqtSignal(str)

    def __init__(
        self,
        modes,
        parent: QtWidgets.QWidget | None = None,
        orientation=QtCore.Qt.Vertical,
        icon_size: int = 18,
        *,
        filled: bool = False,
        variant: str = "default",
    ) -> None:
        """Initialize the segmented control and create its selectable segments.

        Configures the control's orientation and visual variant, constructs
        the appropriate horizontal or vertical layout, creates an exclusive
        button group, and populates it from the supplied segment definitions.
        Each button is connected to :meth:`set_active` so selecting a segment
        updates the control's active key and emits :attr:`modeChanged`.

        Args:
            modes: Sequence of `(key, label)` or
                `(key, label, icon_path)` segment definitions.
            parent: Optional parent widget.
            orientation: Qt layout orientation. Supported values are
                `QtCore.Qt.Vertical` and `QtCore.Qt.Horizontal`.
            icon_size: Width and height of segment icons, in pixels.
            filled: Whether to use the filled styling variant.
            variant: Visual variant name. `"default"` uses the standard
                segmented-control appearance. `"chips"` enables the
                horizontal pill-shaped chip treatment.

        Returns:
            None.

        Note:
            The `"chips"` variant is only enabled for horizontal controls.
            Vertical controls always use the standard sidebar-style geometry.

            Icon-bearing segments use `ToolButtonTextBesideIcon` when their
            icon path exists. Otherwise, the button falls back to
            `ToolButtonTextOnly`.
        """
        super().__init__(parent)
        self._orientation = orientation
        self._icon_size = icon_size
        self._filled = filled
        self._variant = variant
        self._chip_shadow = None
        self.setObjectName("segmentedControl")

        self._is_chips = self._variant == "chips" and orientation == QtCore.Qt.Horizontal
        if orientation == QtCore.Qt.Vertical:
            self.setFixedWidth(132)
            self._radius = 16
        else:
            self.setFixedHeight(38)
            self._radius = 20 if self._is_chips else 19

        self._apply_container_qss()
        self._buttons: dict = {}
        self._icons: dict = {}  # key -> (inactive QIcon, active QIcon)
        self._icon_paths: dict = {}
        self._active_key: str | None = None

        lay = (
            QtWidgets.QVBoxLayout(self)
            if orientation == QtCore.Qt.Vertical
            else QtWidgets.QHBoxLayout(self)
        )
        if orientation == QtCore.Qt.Vertical:
            lay.setContentsMargins(6, 6, 6, 6)
            lay.setSpacing(4)
        else:
            lay.setContentsMargins(8, 4, 8, 4)
            lay.setSpacing(6)
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)

        for mode in modes:
            if len(mode) == 3:
                key, label, icon_path = mode
            else:
                key, label = mode
                icon_path = None
            btn = QtWidgets.QToolButton()
            btn.setText(f" {label}" if icon_path else label)
            btn.setCheckable(True)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            if icon_path and os.path.exists(icon_path):
                self._icon_paths[key] = icon_path
                btn.setIconSize(QtCore.QSize(icon_size, icon_size))
                btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            else:
                btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            if orientation == QtCore.Qt.Vertical:
                btn.setFixedHeight(38)
                btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            else:
                btn.setFixedHeight(30)
                btn.setMinimumWidth(0 if self._is_chips else 78)
            btn.clicked.connect(lambda _=False, k=key: self.set_active(k))
            self._group.addButton(btn)
            lay.addWidget(btn)
            self._buttons[key] = btn

        if orientation == QtCore.Qt.Vertical:
            lay.addStretch()  # keep buttons pinned to the top of the sidebar

        self._refresh_icons()
        self._apply_qss()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        """Refresh the control after an application theme change.

        Reapplies the container styling, regenerates theme-dependent icon tints,
        and reapplies the segment button styles so the control remains visually
        synchronized with the active application theme.

        Args:
            _mode: Theme identifier supplied by the `ThemeManager` theme-change
                signal. The value is not used directly because the current theme
                is obtained from `ThemeManager` by the individual refresh
                methods.

        Returns:
            None.
        """
        self._apply_container_qss()
        self._refresh_icons()
        self._apply_qss()

    def _apply_container_qss(self) -> None:
        """Apply the segmented-control container stylesheet.

        Updates the outer `QFrame` styling according to the control's
        `_filled` configuration and the currently active application theme.
        Unfilled controls remain transparent with no border, while filled
        controls use the theme's secondary flat surface and border tokens.

        Returns:
            None.
        """
        if not self._filled:
            self.setStyleSheet(
                f"QFrame#segmentedControl {{ background: transparent; border: none; "
                f"border-radius: {self._radius}px; }}"
            )
            return

        tok = ThemeManager.instance().tokens()
        self.setStyleSheet(
            f"QFrame#segmentedControl {{ background: {tok_css(tok['flat_surface2'])}; "
            f"border: 1px solid {tok_css(tok['flat_border'])}; "
            f"border-radius: {self._radius}px; }}"
        )

    def _refresh_icons(self) -> None:
        """Regenerate segment icons using the active theme colors.

        Re-tints each registered segment icon for both its inactive and active
        states using the current theme's muted text and accent colors. The
        appropriate icon variant is then applied to each button based on the
        currently selected segment.

        Returns:
            None.

        Note:
            Icon source paths are retained in `_icon_paths` so all icons can be
            regenerated when the application theme changes. The resulting
            `QIcon` pairs are cached in `_icons` as
            `(inactive_icon, active_icon)` for subsequent state updates.
        """
        tok = ThemeManager.instance().tokens()
        inactive_color = QtGui.QColor(*tok["flat_text_muted"])
        active_color = QtGui.QColor(*tok["flat_accent"])

        for key, path in self._icon_paths.items():
            icon_inactive = tinted_icon(path, inactive_color, self._icon_size)
            icon_active = tinted_icon(path, active_color, self._icon_size)
            self._icons[key] = (icon_inactive, icon_active)

            btn = self._buttons[key]
            btn.setIcon(icon_active if key == self._active_key else icon_inactive)

    def _apply_qss(self) -> None:
        """Apply the standard segmented-button stylesheet.

        Generates and applies the theme-aware QSS used by the control's
        `QToolButton` segments. Vertical controls use left-aligned labels to
        maintain a consistent sidebar presentation, while horizontal controls
        center their content within each segment.

        If the control is using the horizontal `"chips"` variant, styling is
        delegated to :meth:`_apply_chip_qss` instead.

        Returns:
            None.
        """
        if self._is_chips:
            self._apply_chip_qss()
            return
        tok = ThemeManager.instance().tokens()
        text_align = "left" if self._orientation == QtCore.Qt.Vertical else "center"
        qss = f"""
            QToolButton {{
                background: transparent;
                border: 1.5px solid transparent;
                border-radius: {self._radius - 5}px;
                color: {tok_css(tok["flat_text_muted"])};
                font-size: 12px; font-weight: 600;
                padding: 0px 9px;
                text-align: {text_align};
            }}
            QToolButton:hover {{
                background: {tok_css(tok["flat_surface2"])};
            }}
            QToolButton:checked {{
                background: {tok_css(tok["flat_accent_weak"])};
                border: 1.5px solid {tok_css(tok["flat_accent_ring"])};
                color: {tok_css(tok["flat_accent"])};
                font-weight: 700;
            }}
            QToolButton:checked:hover {{
                background: {tok_css(tok["flat_accent_weak"])};
            }}
        """
        for btn in self._buttons.values():
            btn.setStyleSheet(qss)

    def _apply_chip_qss(self) -> None:
        """Apply the pill-shaped stylesheet used by the chip variant.

        Styles each segment as a fully rounded, compact pill with theme-aware
        text, surface, border, and accent colors. Inactive chips remain
        transparent, while the selected chip uses the standard flat surface and
        border to create a raised appearance distinct from the default
        accent-wash selection style.

        The selected chip's soft elevation effect is implemented separately using
        a :class:`QGraphicsDropShadowEffect`, since QSS cannot provide the
        required drop shadow for a `QToolButton`. The shadow is synchronized
        with the active segment by :meth:`_sync_chip_shadow`.

        Returns:
            None.
        """
        tok = ThemeManager.instance().tokens()
        radius = 15
        qss = f"""
            QToolButton {{
                background: transparent;
                border: 1px solid transparent;
                border-radius: {radius}px;
                color: {tok_css(tok["flat_text_muted"])};
                font-size: 12px; font-weight: 600;
                padding: 0px 13px;
                text-align: center;
            }}
            QToolButton:hover {{
                color: {tok_css(tok["flat_text"])};
            }}
            QToolButton:checked {{
                background: {tok_css(tok["flat_surface"])};
                border: 1px solid {tok_css(tok["flat_border"])};
                color: {tok_css(tok["flat_accent"])};
                font-weight: 700;
            }}
        """
        for btn in self._buttons.values():
            btn.setStyleSheet(qss)
        self._sync_chip_shadow()

    def _sync_chip_shadow(self) -> None:
        """Synchronize the drop shadow with the currently active chip.

        Ensures that the `"chips"` variant has a single soft drop shadow
        attached only to the currently selected segment. Because a
        `QGraphicsEffect` can belong to only one widget at a time, the existing
        shadow is removed from all inactive buttons before being attached to the
        active button.

        When no segment is active, the shadow reference is cleared and no effect
        is installed. The method is a no-op for controls that are not using the
        horizontal chip variant.

        Returns:
            None.

        Note:
            The shadow is intentionally implemented with
            `QGraphicsDropShadowEffect` rather than QSS because Qt stylesheets
            do not provide a direct equivalent to the CSS `box-shadow` used by
            the chip design. Qt takes ownership of a graphics effect when it is
            assigned to a widget, so `_chip_shadow` is updated alongside the
            widget's effect to avoid retaining a stale reference.
        """
        if not self._is_chips:
            return
        active_btn = self._buttons.get(self._active_key)
        for btn in self._buttons.values():
            if btn is not active_btn and btn.graphicsEffect() is not None:
                btn.setGraphicsEffect(None)
        if active_btn is None:
            self._chip_shadow = None
            return
        if active_btn.graphicsEffect() is None:
            shadow = QtWidgets.QGraphicsDropShadowEffect(active_btn)
            shadow.setBlurRadius(8)
            shadow.setColor(QtGui.QColor(20, 40, 60, 60))
            shadow.setOffset(0, 1)
            active_btn.setGraphicsEffect(shadow)
            self._chip_shadow = shadow

    def set_active(self, key: str) -> None:
        """Set the active button and update its visual state.

        The specified button is marked as checked, while all other buttons are
        cleared. If alternate active/inactive icons have been registered for a
        button, the appropriate icon is also applied. The chip drop shadow is
        re-anchored after every call to ensure it follows the active button,
        including the initial selection.

        If the active key changes, a `modeChanged` signal is emitted with the
        new key.

        Args:
            key: Key identifying the button to activate. If the key is not
                registered in `_buttons`, the method returns without making
                any changes.

        Returns:
            None.
        """
        if key not in self._buttons:
            return
        for k, btn in self._buttons.items():
            is_active = k == key
            btn.setChecked(is_active)
            icons = self._icons.get(k)
            if icons is not None:
                btn.setIcon(icons[1] if is_active else icons[0])
        changed = key != self._active_key
        self._active_key = key
        self._sync_chip_shadow()
        if changed:
            self.modeChanged.emit(key)

    def active_key(self) -> str | None:
        """Return the key of the currently active button.

        Returns:
            The key associated with the currently active button, or `None` if
            no button has been activated.
        """
        return self._active_key
