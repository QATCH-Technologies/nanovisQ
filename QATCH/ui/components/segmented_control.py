"""
QATCH.ui.components.segmented_control

Exclusive-selection segmented control matching the app's flat control system
(see QATCH.ui.components.flat_paint) - a row (or column) of checkable
QToolButtons in a QButtonGroup, styled from the `flat_*` tokens so it stays
theme-correct automatically.

Promoted out of `data_management_widget.py` (where it lived as
`GlassSegmentedControl`, sidebar-only and hardcoded to light-mode colors) so
it can also back `data_mode_history.py`'s text-only filter chips instead of
that file hand-rolling a near-duplicate.

Usage
-----
    # Icon + label rows (vertical sidebar):
    nav = SegmentedControl(
        [("import", "Import", icon_path), ("export", "Export", icon_path2)],
        orientation=QtCore.Qt.Vertical,
    )
    nav.modeChanged.connect(handler)
    nav.set_active("import")

    # Text-only chips (horizontal filter row):
    filt = SegmentedControl([("all", "All"), ("export", "Export")],
                             orientation=QtCore.Qt.Horizontal)
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.icon_utils import tinted_icon
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css


class SegmentedControl(QtWidgets.QFrame):
    """A row/column of mutually-exclusive checkable segments.

    Attributes:
        _buttons (dict[str, QToolButton]): key -> button.
        _icon_paths (dict[str, str]): key -> source icon path (for re-tinting
            on theme change).
        _active_key (Optional[str]): Currently selected segment's key.
    """

    modeChanged = QtCore.pyqtSignal(str)

    def __init__(
        self,
        modes,
        parent: Optional[QtWidgets.QWidget] = None,
        orientation=QtCore.Qt.Vertical,
        icon_size: int = 18,
        *,
        filled: bool = False,
    ) -> None:
        # modes: list of (key, label) or (key, label, icon_path)
        super().__init__(parent)
        self._orientation = orientation
        self._icon_size = icon_size
        self._filled = filled
        self.setObjectName("segmentedControl")

        if orientation == QtCore.Qt.Vertical:
            self.setFixedWidth(132)
            self._radius = 16
        else:
            self.setFixedHeight(38)
            self._radius = 19

        self._apply_container_qss()
        self._buttons: dict = {}
        self._icons: dict = {}  # key -> (inactive QIcon, active QIcon)
        self._icon_paths: dict = {}
        self._active_key: Optional[str] = None

        lay = (
            QtWidgets.QVBoxLayout(self)
            if orientation == QtCore.Qt.Vertical
            else QtWidgets.QHBoxLayout(self)
        )
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
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
                btn.setMinimumWidth(78)
            btn.clicked.connect(lambda _=False, k=key: self.set_active(k))
            self._group.addButton(btn)
            lay.addWidget(btn)
            self._buttons[key] = btn

        if orientation == QtCore.Qt.Vertical:
            lay.addStretch()  # keep buttons pinned to the top of the sidebar

        self._refresh_icons()
        self._apply_qss()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    # ------------------------------------------------------------------
    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_container_qss()
        self._refresh_icons()
        self._apply_qss()

    def _apply_container_qss(self) -> None:
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
        tok = ThemeManager.instance().tokens()
        qss = f"""
            QToolButton {{
                background: transparent;
                border: 1.5px solid transparent;
                border-radius: {self._radius - 5}px;
                color: {tok_css(tok["flat_text_muted"])};
                font-size: 12px; font-weight: 600;
                padding: 0px 9px;
                text-align: left;
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

    # ------------------------------------------------------------------
    def set_active(self, key: str) -> None:
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

    def active_key(self) -> Optional[str]:
        return self._active_key
