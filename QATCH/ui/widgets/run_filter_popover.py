"""Compact "▾ filters" popover for AnalyzeUI's run search field.

Lays its controls out as a single inline row of chips (device, date range,
new/unanalyzed toggle, sort, clear) rather than the taller labelled-group
popover layout, per the "1c" filter direction from the task-bar redesign -
the "2a" layout it's anchored to only supplies the search field + position
stepper; this is deliberately the more compact 1c filter popover instead of
2a's own stacked one.

Mirrors QATCH.ui.widgets.account_popup.AccountPopup's anchored-popup
mechanics (Qt.Popup + translucent shadow-margin outer widget + clamp to the
main window), but this class only builds/lays out its controls - like
AnalyzeActionBar, it does not own filtering behavior. Every control change
is reported through a callback supplied at construction; the caller (
UIAnalyze) decides what to do with it and owns the actual run-list state.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.ui.components import AnimatedComboBox
from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.components.qatch_push_button import QATCHPushButton
from QATCH.ui.components.segmented_control import SegmentedControl
from QATCH.ui.styles.fonts import FONT_SANS, FONT_SANS_SEMIBOLD
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css

_ANY_DEVICE = "All devices"
_SORT_ITEMS = (
    ("Date (newest)", 1),
    ("Date (oldest)", 3),
    ("Name (A–Z)", 0),
    ("Name (Z–A)", 4),
)  # label -> sort_order


def _animated_combo(items: List[str]) -> AnimatedComboBox:
    """The same rounded/animated combo box used everywhere else in the app
    (cBox_Runs, cBox_Speed, cBox_Port, ...) - kept consistent here rather
    than a bespoke stock QComboBox, per the "filter dropdowns should be the
    animated dropdown menus" note on this popover's first pass.
    """
    combo = AnimatedComboBox(
        icon_path=os.path.join(Architecture.get_path(), "QATCH", "icons", "down-chevron.svg")
    )
    combo.addItems(items)
    combo.setFixedHeight(30)
    combo.setMinimumWidth(112)
    return combo


def _flat_date_edit() -> QtWidgets.QDateEdit:
    edit = QtWidgets.QDateEdit()
    edit.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    edit.setCalendarPopup(True)
    edit.setDisplayFormat("yyyy-MM-dd")
    edit.setMinimumDate(QtCore.QDate(2000, 1, 1))
    edit.setMaximumDate(QtCore.QDate(2100, 1, 1))
    edit.setSpecialValueText("Any")
    edit.setDate(edit.minimumDate())
    tok = ThemeManager.instance().tokens()
    edit.setStyleSheet(
        f"""
        QDateEdit {{
            background: {tok_css(tok["flat_surface2"])};
            border: 1px solid {tok_css(tok["flat_border"])};
            border-radius: 13px;
            padding: 4px 10px;
            color: {tok_css(tok["flat_text"])};
            font-family: '{FONT_SANS}';
            font-size: 11.5px;
        }}
        QDateEdit::drop-down {{ border: none; width: 16px; }}
        """
    )
    return edit


class _FilterInnerPanel(QtWidgets.QWidget):
    """Flat card background for the popover, matching AccountInnerPanel."""

    _RADIUS = 12.0

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        tok = ThemeManager.instance().tokens()
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        paint_flat_surface(
            self,
            radius=self._RADIUS,
            fill=QtGui.QColor(*tok["flat_surface"]),
            border=QtGui.QColor(*tok["flat_border"]),
            painter=p,
        )


class RunFilterPopover(QtWidgets.QWidget):
    """Anchored popover holding the run list's device/date/new/sort filters.

    All state (which device, date bounds, new-only, sort order) is owned by
    the caller and passed in at construction; this widget only reports
    changes back out through the `on_*` callbacks; it never mutates
    anything the caller didn't already have.
    """

    closed = QtCore.pyqtSignal()

    _SHADOW_MARGIN_L = 18
    _SHADOW_MARGIN_T = 14
    _SHADOW_MARGIN_R = 18
    _SHADOW_MARGIN_B = 22

    def __init__(
        self,
        devices: List[str],
        current_device: Optional[str],
        show_all: bool,
        date_from: Optional[str],
        date_to: Optional[str],
        new_only: bool,
        sort_order: int,
        on_device_changed: Callable[[Optional[str]], None],
        on_date_range_changed: Callable[[Optional[str], Optional[str]], None],
        on_new_only_changed: Callable[[bool], None],
        on_sort_changed: Callable[[int], None],
        on_clear: Callable[[], None],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(
            parent,
            QtCore.Qt.WindowType.Popup
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.NoDropShadowWindowHint,
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

        self._on_device_changed = on_device_changed
        self._on_date_range_changed = on_date_range_changed
        self._on_new_only_changed = on_new_only_changed
        self._on_sort_changed = on_sort_changed
        self._on_clear = on_clear
        self._main_window: Optional[QtWidgets.QWidget] = None
        self._suspend_callbacks = True  # guard while pre-selecting initial state below

        self._panel = _FilterInnerPanel(self)

        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(
            self._SHADOW_MARGIN_L, self._SHADOW_MARGIN_T, self._SHADOW_MARGIN_R, self._SHADOW_MARGIN_B
        )
        outer_layout.setSpacing(0)
        outer_layout.addWidget(self._panel)

        shadow = QtWidgets.QGraphicsDropShadowEffect(self._panel)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 3)
        shadow.setColor(QtGui.QColor(0, 20, 40, 100))
        self._panel.setGraphicsEffect(shadow)

        self._title = QtWidgets.QLabel("FILTER RUNS")
        self._field_labels: List[QtWidgets.QLabel] = []

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(14)

        self._device_combo = _animated_combo([_ANY_DEVICE, *devices])
        self._device_combo.setCurrentText(_ANY_DEVICE if show_all else (current_device or _ANY_DEVICE))
        self._device_combo.currentTextChanged.connect(self._device_selected)
        row.addLayout(self._field("Device", self._device_combo))

        self._from_edit = _flat_date_edit()
        if date_from:
            qd = QtCore.QDate.fromString(date_from, "yyyy-MM-dd")
            if qd.isValid():
                self._from_edit.setDate(qd)
        self._from_edit.dateChanged.connect(self._date_range_edited)
        row.addLayout(self._field("From", self._from_edit))

        self._to_edit = _flat_date_edit()
        if date_to:
            qd = QtCore.QDate.fromString(date_to, "yyyy-MM-dd")
            if qd.isValid():
                self._to_edit.setDate(qd)
        self._to_edit.dateChanged.connect(self._date_range_edited)
        row.addLayout(self._field("To", self._to_edit))

        self._new_toggle = SegmentedControl(
            [("all", "All"), ("new", "New / unanalyzed")],
            orientation=QtCore.Qt.Horizontal,
            variant="chips",
        )
        self._new_toggle.set_active("new" if new_only else "all")
        self._new_toggle.modeChanged.connect(lambda key: self._on_new_only_changed(key == "new"))
        row.addLayout(self._field("Show", self._new_toggle))

        self._sort_combo = _animated_combo([label for label, _order in _SORT_ITEMS])
        start_index = next(
            (i for i, (_label, order) in enumerate(_SORT_ITEMS) if order == sort_order), 0
        )
        self._sort_combo.setCurrentIndex(start_index)
        self._sort_combo.currentIndexChanged.connect(self._sort_selected)
        row.addLayout(self._field("Sort By", self._sort_combo))

        row.addStretch(1)

        self._clear_btn = QATCHPushButton("Clear filter", variant="ghost")
        self._clear_btn.setFixedHeight(26)
        self._clear_btn.clicked.connect(self._clear_clicked)
        # Bottom-aligned so it sits level with the controls, not their
        # labels, despite not having a label column of its own.
        row.addWidget(self._clear_btn, 0, QtCore.Qt.AlignmentFlag.AlignBottom)

        layout = QtWidgets.QVBoxLayout(self._panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        layout.addWidget(self._title)
        layout.addLayout(row)

        self._suspend_callbacks = False
        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    # -- theming ----------------------------------------------------------

    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_theme()

    def _apply_theme(self) -> None:
        tok = ThemeManager.instance().tokens()
        self._title.setStyleSheet(
            f"color: {tok_css(tok['flat_text_muted'])}; font-family: '{FONT_SANS_SEMIBOLD}'; "
            "font-size: 10px; letter-spacing: 1px; background: transparent; border: none;"
        )
        field_label_qss = (
            f"color: {tok_css(tok['flat_text_muted'])}; font-family: '{FONT_SANS_SEMIBOLD}'; "
            "font-size: 10px; background: transparent; border: none;"
        )
        for label in self._field_labels:
            label.setStyleSheet(field_label_qss)

    def _field(self, label_text: str, control: QtWidgets.QWidget) -> QtWidgets.QVBoxLayout:
        """A small muted caption stacked above `control` - e.g. "Device"
        over the device combo - so every filter is self-explanatory at a
        glance instead of relying on the control's own placeholder/value to
        convey what it filters by.
        """
        label = QtWidgets.QLabel(label_text.upper())
        self._field_labels.append(label)
        group = QtWidgets.QVBoxLayout()
        group.setSpacing(4)
        group.addWidget(label)
        group.addWidget(control)
        return group

    # -- change handlers ----------------------------------------------------

    def _device_selected(self, text: str) -> None:
        if self._suspend_callbacks:
            return
        self._on_device_changed(None if text == _ANY_DEVICE else text)

    def _date_range_edited(self, _value: QtCore.QDate) -> None:
        if self._suspend_callbacks:
            return
        from_val = (
            None
            if self._from_edit.date() == self._from_edit.minimumDate()
            else self._from_edit.date().toString("yyyy-MM-dd")
        )
        to_val = (
            None
            if self._to_edit.date() == self._to_edit.minimumDate()
            else self._to_edit.date().toString("yyyy-MM-dd")
        )
        self._on_date_range_changed(from_val, to_val)

    def _sort_selected(self, index: int) -> None:
        if self._suspend_callbacks:
            return
        self._on_sort_changed(_SORT_ITEMS[index][1])

    def _clear_clicked(self) -> None:
        self._suspend_callbacks = True
        self._device_combo.setCurrentText(_ANY_DEVICE)
        self._from_edit.setDate(self._from_edit.minimumDate())
        self._to_edit.setDate(self._to_edit.minimumDate())
        self._new_toggle.set_active("all")
        self._sort_combo.setCurrentIndex(0)
        self._suspend_callbacks = False
        self._on_clear()

    # -- public API -----------------------------------------------------------

    def show_anchored_to(
        self,
        anchor: QtWidgets.QWidget,
        main_window: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Show the popover pinned under `anchor`, clamped to `main_window`.

        Mirrors AccountPopup.show_anchored_to: the visible panel's left
        edge aligns with the anchor's left edge, 2px below it, clamped so
        it stays inside the main window's frame.
        """
        self._main_window = main_window
        self.adjustSize()

        size = self.sizeHint()
        popup_w, popup_h = size.width(), size.height()

        anchor_bl = anchor.mapToGlobal(QtCore.QPoint(0, anchor.height()))
        x = anchor_bl.x() - self._SHADOW_MARGIN_L
        y = anchor_bl.y() + 2 - self._SHADOW_MARGIN_T

        x, y = self._clamp_to_main_window(x, y, popup_w, popup_h, anchor)

        if self._main_window is not None:
            self._main_window.installEventFilter(self)

        self.setWindowOpacity(0.0)
        self.move(QtCore.QPoint(-9999, -9999))
        self.show()

        fade = QtCore.QVariantAnimation(self)
        fade.setDuration(160)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        fade.valueChanged.connect(lambda v: self.setWindowOpacity(float(v)))
        self._fade = fade

        def _start():
            self.move(QtCore.QPoint(x, y))
            fade.start()

        QtCore.QTimer.singleShot(0, _start)

    # -- positioning helpers --------------------------------------------------

    def _visible_rect_for(self, x: int, y: int, w: int, h: int) -> QtCore.QRect:
        return QtCore.QRect(
            x + self._SHADOW_MARGIN_L,
            y + self._SHADOW_MARGIN_T,
            w - self._SHADOW_MARGIN_L - self._SHADOW_MARGIN_R,
            h - self._SHADOW_MARGIN_T - self._SHADOW_MARGIN_B,
        )

    def _clamp_to_main_window(
        self, x: int, y: int, popup_w: int, popup_h: int, anchor: QtWidgets.QWidget
    ) -> tuple:
        top_level = anchor.window() if anchor is not None else None
        if top_level is not None:
            bounds = top_level.geometry()
        elif self._main_window is not None:
            bounds = self._main_window.geometry()
        else:
            screen = QtWidgets.QApplication.screenAt(anchor.mapToGlobal(QtCore.QPoint(0, 0)))
            bounds = screen.availableGeometry() if screen is not None else QtCore.QRect()

        if bounds.isNull():
            return x, y

        visible = self._visible_rect_for(x, y, popup_w, popup_h)

        if visible.right() > bounds.right():
            x -= visible.right() - bounds.right()
            visible = self._visible_rect_for(x, y, popup_w, popup_h)
        if visible.left() < bounds.left():
            x += bounds.left() - visible.left()
            visible = self._visible_rect_for(x, y, popup_w, popup_h)

        if visible.bottom() > bounds.bottom():
            anchor_top = anchor.mapToGlobal(QtCore.QPoint(0, 0)).y()
            y_above = anchor_top - 2 - popup_h + self._SHADOW_MARGIN_B
            visible_above = self._visible_rect_for(x, y_above, popup_w, popup_h)
            if visible_above.top() >= bounds.top():
                y = y_above
            else:
                y -= visible.bottom() - bounds.bottom()

        return x, y

    # -- event handling -------------------------------------------------------

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if watched is self._main_window and event.type() in (
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.Move,
            QtCore.QEvent.Type.WindowStateChange,
        ):
            self.close()
        return super().eventFilter(watched, event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        if self._main_window is not None:
            try:
                self._main_window.removeEventFilter(self)
            except Exception:
                pass
            self._main_window = None
        super().closeEvent(event)
        self.closed.emit()
