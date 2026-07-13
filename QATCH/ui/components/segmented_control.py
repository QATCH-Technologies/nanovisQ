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
        variant: str = "default",
    ) -> None:
        # modes: list of (key, label) or (key, label, icon_path)
        super().__init__(parent)
        self._orientation = orientation
        self._icon_size = icon_size
        self._filled = filled
        # variant "chips" => pill-shaped segments with a raised (surface fill
        # + hairline border + soft shadow) selected state, per the Data
        # Management mode-bar redesign (design 1c). "default" keeps the
        # accent-weak wash the vertical sidebar and history filter chips use.
        self._variant = variant
        self._chip_shadow = None  # drop shadow attached to the active chip (chips variant)
        self.setObjectName("segmentedControl")

        self._is_chips = self._variant == "chips" and orientation == QtCore.Qt.Horizontal
        if orientation == QtCore.Qt.Vertical:
            self.setFixedWidth(132)
            self._radius = 16
        else:
            self.setFixedHeight(38)
            # Fully-rounded pill for chips (mock uses borderRadius:999 on a
            # ~30px-tall chip); the default horizontal look keeps its 19px.
            self._radius = 20 if self._is_chips else 19

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
        if orientation == QtCore.Qt.Vertical:
            lay.setContentsMargins(6, 6, 6, 6)
            lay.setSpacing(4)
        else:
            # Top/bottom margin sized so 30px buttons land exactly inside the
            # container's fixed 38px height (4 + 30 + 4) instead of overflowing
            # it, and a little more breathing room between chips than the
            # vertical rail needs.
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
                # Chips hug their content (mock pads 7x13 and lets each chip
                # size to its label); the default look keeps an even 78px min.
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
        if self._is_chips:
            self._apply_chip_qss()
            return
        tok = ThemeManager.instance().tokens()
        # Vertical rows read best left-aligned (they share one fixed-width
        # column); horizontal chips are sized to their own content plus a
        # minimum width, so left-aligning there biases the label toward the
        # icon instead of centering the pill's content.
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
        """Pill-chip styling for the Data Management mode bar (design 1c).

        Selected chip reads as a raised element - solid `flat_surface` fill,
        a hairline `flat_border` outline, and accent-tinted text/icon - lifted
        off the strip below rather than the flat accent-weak wash the default
        variant paints. The soft drop shadow the mock shows can't come from
        QSS on a QToolButton, so it's a QGraphicsDropShadowEffect moved onto
        whichever chip is active (see _sync_chip_shadow).
        """
        tok = ThemeManager.instance().tokens()
        radius = 15  # half the 30px chip height => fully-rounded pill
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
        """Move the soft drop shadow onto the active chip (chips variant only).

        A QGraphicsEffect belongs to exactly one widget at a time, so rather
        than juggling one effect per chip we keep a single shadow and re-parent
        it onto whichever button is currently checked, clearing it from the
        rest. Matches the mock's `boxShadow` on the selected chip.
        """
        if not self._is_chips:
            return
        active_btn = self._buttons.get(self._active_key)
        for btn in self._buttons.values():
            if btn is not active_btn and btn.graphicsEffect() is not None:
                # setGraphicsEffect(None) deletes the previously-set effect
                # (Qt ownership), so drop our reference alongside it.
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
        changed = key != self._active_key
        self._active_key = key
        # Re-anchor the pill's drop shadow onto the now-active chip. Done every
        # call (not just on change) so it also lands on the very first
        # selection, where _active_key was None going in.
        self._sync_chip_shadow()
        if changed:
            self.modeChanged.emit(key)

    def active_key(self) -> Optional[str]:
        return self._active_key
