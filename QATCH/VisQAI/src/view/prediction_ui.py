import csv
import difflib  # Add to imports
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg

# --- MATPLOTLIB IMPORTS ---
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- PYQT IMPORTS ---
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

# Configure pyqtgraph for a white background theme
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
pg.setConfigOptions(antialias=True)
try:
    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database
    from src.models.formulation import Formulation
    from src.models.predictor import Predictor

    BACKEND_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    BACKEND_AVAILABLE = False

    class Formulation:
        pass

    class Predictor:
        pass

    class IngredientController:
        def get_all_ingredients(self):
            return []

    class Database:
        def __init__(self, parse_file_key=False):
            pass

    class Log:
        @staticmethod
        def i(tag, msg):
            print(f"INFO [{tag}]: {msg}")

        @staticmethod
        def e(tag, msg):
            print(f"ERROR [{tag}]: {msg}")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_DOWN = os.path.join(BASE_DIR, "icons", "down-chevron-svgrepo-com.svg").replace(
    "\\", "/"
)
ICON_UP = os.path.join(BASE_DIR, "icons", "up-chevron-svgrepo-com.svg").replace(
    "\\", "/"
)
ICON_BROWSE_MODEL = os.path.join(
    BASE_DIR, "icons", "machine-learning-01-svgrepo-com.svg"
).replace("\\", "/")
TAG = "[ViscosityUI]"

LIGHT_STYLE_SHEET = """
/* ---------------------------------------------------------------------------
   1. Container Styling
   --------------------------------------------------------------------------- */
QWidget#leftPanel {
    background-color: transparent; 
    border-right: 1px solid #e1e4e8;
}
QScrollArea {
    background-color: transparent;
    border: none;
}
QWidget#scrollContent {
    background-color: transparent;
}

/* ---------------------------------------------------------------------------
   2. Prediction Cards
   --------------------------------------------------------------------------- */
QFrame[class="card"] {
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    margin-top: 4px;
    margin-bottom: 8px;
    margin-left: 2px;
    margin-right: 4px;
}
QFrame[class="card"]:hover {
    border: 1px solid #00adee;
}
QFrame[class="card"][measured="true"] {
    background-color: #f6fffa;
    border: 1px solid #4caf50;
}

/* --- INSERT THIS BLOCK --- */
QFrame[class="card"][selected="true"] {
    background-color: #eff6fc; /* Light Blue Background */
    border: 2px solid #0078D4; /* Strong Blue Border */
}

/* ---------------------------------------------------------------------------
   3. Typography & Headers
   --------------------------------------------------------------------------- */
QFrame[class="card"] QLabel {
    color: #24292f;
}
QLabel[class="header-title"] {
    font-weight: 600;
    color: #00adee;
    font-size: 10pt;
}
QFrame[class="divider"] {
    color: #eaecef;
}

/* ---------------------------------------------------------------------------
   4. SLEEK INPUTS (Dropdowns & SpinBoxes)
   --------------------------------------------------------------------------- */
/* Common Input Styling */
QFrame[class="card"] QLineEdit, 
QFrame[class="card"] QComboBox, 
QFrame[class="card"] QDoubleSpinBox,
QFrame[class="card"] QSpinBox {
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 24px;
    color: #24292f;
    selection-background-color: #00adee;
}

/* Focus State */
QFrame[class="card"] QLineEdit:focus, 
QFrame[class="card"] QComboBox:focus, 
QFrame[class="card"] QDoubleSpinBox:focus,
QFrame[class="card"] QSpinBox:focus {
    border: 1px solid #00adee;
    background-color: #ffffff;
}

/* -- ComboBox Arrow Fix -- */
QFrame[class="card"] QComboBox {
    padding-right: 20px;
}
QFrame[class="card"] QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid #d1d5db;
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
    background-color: #e9ecef;
}
QFrame[class="card"] QComboBox::drop-down:hover {
    background-color: #d0e8ff;
}

/* Down Arrow - Using your SVG variable */
QFrame[class="card"] QComboBox::down-arrow {
    width: 12px;
    height: 12px;
    image: url("__ICON_DOWN__");
}

/* -- SpinBox Arrow Fix -- */
QFrame[class="card"] QDoubleSpinBox,
QFrame[class="card"] QSpinBox {
    padding-right: 24px;
}

/* Up Button */
QFrame[class="card"] QDoubleSpinBox::up-button, 
QFrame[class="card"] QSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid #d1d5db;
    border-bottom: 1px solid #d1d5db;
    border-top-right-radius: 3px;
    background-color: #e9ecef;
}

/* Down Button */
QFrame[class="card"] QDoubleSpinBox::down-button, 
QFrame[class="card"] QSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 22px;
    border-left: 1px solid #d1d5db;
    border-bottom-right-radius: 3px;
    background-color: #e9ecef;
}

/* Up Arrow - Using your SVG variable */
QFrame[class="card"] QDoubleSpinBox::up-arrow, 
/* Up Arrow */
QFrame[class="card"] QDoubleSpinBox::up-arrow, 
QFrame[class="card"] QSpinBox::up-arrow {
    width: 10px;
    height: 10px;
    image: url("__ICON_UP__");
}

/* Down Arrow */
QFrame[class="card"] QDoubleSpinBox::down-arrow, 
QFrame[class="card"] QSpinBox::down-arrow {
    width: 10px;
    height: 10px;
    image: url("__ICON_DOWN__");
}
/* Hover/Pressed Effects remain same... */
QFrame[class="card"] QDoubleSpinBox::up-button:hover,
QFrame[class="card"] QDoubleSpinBox::down-button:hover,
QFrame[class="card"] QSpinBox::up-button:hover,
QFrame[class="card"] QSpinBox::down-button:hover {
    background-color: #d0e8ff;
}

/* ---------------------------------------------------------------------------
   5. Buttons
   --------------------------------------------------------------------------- */
QFrame[class="card"] QPushButton {
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    padding: 5px 12px;
    font-weight: 500;
}
QFrame[class="card"] QPushButton:hover {
    background-color: #f6f8fa;
    border-color: #00adee;
    color: #00adee;
}
QFrame[class="card"] QPushButton[class="primary"] {
    background-color: #f1f8ff;
    color: #00adee;
    border: 1px solid #00adee;
}
QFrame[class="card"] QPushButton[class="primary"]:hover {
    background-color: #00adee;
    color: #ffffff;
}
QPushButton#btnBrowseModel {
    qproperty-icon: url("__BROWSE_MODEL__");
    qproperty-iconSize: 18px 18px;
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 4px;
}

QPushButton#btnBrowseModel:hover {
    background-color: #f6f8fa;
    border-color: #00adee;
}
QFrame[class="card"] QLineEdit[class="title-input"] {
    background-color: transparent;
    border: none;
    border-bottom: 2px solid #d1d5db; /* Standard underline */
    border-radius: 0px;
    padding: 2px 0px;
    font-size: 10pt;
    font-weight: 500;
    color: #24292f;
}

/* Change the underline color when the user clicks in to type */
QFrame[class="card"] QLineEdit[class="title-input"]:focus {
    border-bottom: 2px solid #00adee; /* Brand blue underline */
    background-color: transparent;
}
/* ---------------------------------------------------------------------------
    Temperature Slider
   --------------------------------------------------------------------------- */
QSlider::groove:horizontal {
    border: 1px solid #d1d5db;
    height: 6px;
    background: #f3f4f6;
    margin: 2px 0;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #00adee;
    width: 16px;
    height: 16px;
    margin: -6px 0; /* Centers handle vertically on the 6px groove */
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    background: #00adee;
    border: 2px solid #008fca;
}

QSlider::sub-page:horizontal {
    background: #00adee;
    border-radius: 3px;
}

QSlider::add-page:horizontal {
    background: #e5e7eb;
    border-radius: 3px;
}
"""
LIGHT_STYLE_SHEET = LIGHT_STYLE_SHEET.replace("__ICON_DOWN__", ICON_DOWN)
LIGHT_STYLE_SHEET = LIGHT_STYLE_SHEET.replace("__ICON_UP__", ICON_UP)
LIGHT_STYLE_SHEET = LIGHT_STYLE_SHEET.replace("__BROWSE_MODEL__", ICON_BROWSE_MODEL)


class PlaceholderWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)

        # Icon (Optional: Use a custom SVG if you have one)
        self.lbl_icon = QtWidgets.QLabel()
        self.lbl_icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Replace 'icons/empty_state.svg' with your actual file path
        icon_path = "QATCH/VisQAI/src/view/icons/info-circle-svgrepo-com.svg"

        pixmap = QtGui.QPixmap(icon_path)
        self.lbl_icon.setPixmap(
            pixmap.scaled(
                48,
                48,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )

        layout.addWidget(self.lbl_icon)

        # Message
        self.lbl_text = QtWidgets.QLabel(
            "No data yet.\nImport or add data to continue."
        )
        self.lbl_text.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_text.setStyleSheet("color: #888; font-size: 14px; font-weight: 500;")
        layout.addWidget(self.lbl_text)


class RangeSlider(QtWidgets.QWidget):
    """A simple double-ended slider for selecting a range."""

    rangeChanged = QtCore.Signal(float, float)

    def __init__(self, min_val=0, max_val=100, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self._min = min_val
        self._max = max_val
        self._low = min_val
        self._high = max_val
        self._pressed_handle = None  # 'low', 'high', or None
        self._handle_radius = 8
        self._groove_height = 4

    def setRange(self, min_val, max_val):
        self._min = min_val
        self._max = max_val
        self.update()

    def setValues(self, low, high):
        self._low = max(self._min, min(low, self._high))
        self._high = min(self._max, max(high, self._low))
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Geometry
        w = self.width()
        h = self.height()
        cy = h / 2
        available_w = w - 2 * self._handle_radius

        def val_to_x(v):
            if self._max == self._min:
                return 0
            ratio = (v - self._min) / (self._max - self._min)
            return self._handle_radius + ratio * available_w

        x_low = val_to_x(self._low)
        x_high = val_to_x(self._high)

        # Draw Groove (Background)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#e0e0e0"))
        painter.drawRoundedRect(
            QtCore.QRectF(
                self._handle_radius,
                cy - self._groove_height / 2,
                available_w,
                self._groove_height,
            ),
            2,
            2,
        )

        # Draw Selected Range (Blue)
        painter.setBrush(QtGui.QColor("#0078D4"))
        rect_range = QtCore.QRectF(
            x_low, cy - self._groove_height / 2, x_high - x_low, self._groove_height
        )
        painter.drawRect(rect_range)

        # Draw Handles
        painter.setBrush(QtGui.QColor("#ffffff"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#0078D4"), 2))

        painter.drawEllipse(
            QtCore.QPointF(x_low, cy), self._handle_radius, self._handle_radius
        )
        painter.drawEllipse(
            QtCore.QPointF(x_high, cy), self._handle_radius, self._handle_radius
        )

    def mousePressEvent(self, event):
        w = self.width()
        available_w = w - 2 * self._handle_radius

        def val_to_x(v):
            if self._max == self._min:
                return 0
            return (
                self._handle_radius
                + ((v - self._min) / (self._max - self._min)) * available_w
            )

        # FIX: Use event.pos().x() for PyQt5 compatibility
        pos_x = event.pos().x()
        dist_low = abs(pos_x - val_to_x(self._low))
        dist_high = abs(pos_x - val_to_x(self._high))

        if dist_low < dist_high:
            self._pressed_handle = "low"
        else:
            self._pressed_handle = "high"

        self.mouseMoveEvent(event)

    def mouseMoveEvent(self, event):
        if not self._pressed_handle:
            return

        w = self.width()
        available_w = w - 2 * self._handle_radius

        # FIX: Use event.pos().x() for PyQt5 compatibility
        pos_x = max(self._handle_radius, min(event.pos().x(), w - self._handle_radius))

        ratio = (pos_x - self._handle_radius) / available_w
        val = self._min + ratio * (self._max - self._min)

        if self._pressed_handle == "low":
            self._low = min(val, self._high)
        else:
            self._high = max(val, self._low)

        self.update()
        self.rangeChanged.emit(self._low, self._high)

    def mouseReleaseEvent(self, event):
        self._pressed_handle = None


class FilterMenuButton(QtWidgets.QPushButton):
    """Button that opens a checkbox menu for multi-selection."""

    selectionChanged = QtCore.Signal()

    def __init__(self, title, items, parent=None):
        super().__init__(title, parent)
        self.items_map = {}  # name -> Action
        self.menu = QtWidgets.QMenu(self)
        self.menu.setStyleSheet("QMenu { menu-scrollable: 1; }")

        # 'All' Action
        self.act_all = self.menu.addAction("Select All")
        self.act_all.triggered.connect(self.select_all)
        self.menu.addSeparator()

        for item in items:
            name = item.name if hasattr(item, "name") else str(item)
            act = self.menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(True)  # Default to all selected
            act.toggled.connect(self._on_item_toggled)
            self.items_map[name] = act

        self.setMenu(self.menu)
        self.update_text()

        # Style
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            """
            QPushButton {
                text-align: left;
                padding: 5px 10px;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                background-color: white;
            }
            QPushButton::menu-indicator {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                padding-right: 10px; 
            }
        """
        )

    def _on_item_toggled(self):
        self.update_text()
        self.selectionChanged.emit()

    def select_all(self):
        self.menu.blockSignals(True)
        for act in self.items_map.values():
            act.setChecked(True)
        self.menu.blockSignals(False)
        self._on_item_toggled()

    def update_text(self):
        selected = [name for name, act in self.items_map.items() if act.isChecked()]
        total = len(self.items_map)

        if len(selected) == 0:
            self.setText("None selected")
            self.setStyleSheet(
                self.styleSheet().replace(
                    "border: 1px solid #d1d5db", "border: 1px solid #e57373"
                )
            )  # Red border warning
        elif len(selected) == total:
            self.setText("All selected")
            self.setStyleSheet(
                self.styleSheet().replace(
                    "border: 1px solid #e57373", "border: 1px solid #d1d5db"
                )
            )
        else:
            self.setText(f"{len(selected)} selected")
            self.setStyleSheet(
                self.styleSheet().replace(
                    "border: 1px solid #e57373", "border: 1px solid #d1d5db"
                )
            )

    def get_selected_items(self):
        return [name for name, act in self.items_map.items() if act.isChecked()]


class PredictionFilterWidget(QtWidgets.QWidget):
    filter_changed = QtCore.Signal(dict)

    def __init__(self, ingredients_data, parent=None):
        super().__init__(parent)
        self.ingredients_data = ingredients_data

        # Overlay Styling
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            """
            PredictionFilterWidget {
                background-color: #ffffff;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
                border: 1px solid #d1d5db;
                border-top: none;
            }
            QLabel { font-weight: 600; color: #555; font-size: 11px; }
            QGroupBox { font-weight: bold; border: 1px solid #d1d5db; border-radius: 4px; margin-top: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; color: #0078D4; }
        """
        )

        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setYOffset(10)
        shadow.setColor(QtGui.QColor(0, 0, 0, 60))
        self.setGraphicsEffect(shadow)

        self.setVisible(False)
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # --- Row 1: State & Model ---
        row1_layout = QtWidgets.QHBoxLayout()
        row1_layout.setSpacing(15)

        # State
        grp_state = QtWidgets.QGroupBox("State")
        state_layout = QtWidgets.QHBoxLayout(grp_state)
        self.chk_measured = QtWidgets.QCheckBox("Measured")
        self.chk_measured.setChecked(True)
        self.chk_predicted = QtWidgets.QCheckBox("Predicted")
        self.chk_predicted.setChecked(True)
        state_layout.addWidget(self.chk_measured)
        state_layout.addWidget(self.chk_predicted)
        row1_layout.addWidget(grp_state)

        # Model
        grp_model = QtWidgets.QGroupBox("Model")
        model_layout = QtWidgets.QVBoxLayout(grp_model)
        self.txt_model = QtWidgets.QLineEdit()
        self.txt_model.setPlaceholderText("Name contains...")
        self.txt_model.setStyleSheet(
            "border: 1px solid #ccc; border-radius: 3px; padding: 4px;"
        )
        model_layout.addWidget(self.txt_model)
        row1_layout.addWidget(grp_model)

        layout.addLayout(row1_layout)

        # --- Row 2: Temperature (Double-Ended Slider) ---
        grp_temp = QtWidgets.QGroupBox("Temperature Range (°C)")
        temp_layout = QtWidgets.QHBoxLayout(grp_temp)
        temp_layout.setSpacing(10)

        self.spin_temp_min = QtWidgets.QDoubleSpinBox()
        self.spin_temp_min.setRange(0, 100)
        self.spin_temp_min.setValue(0)
        self.spin_temp_min.setFixedWidth(60)

        self.range_slider = RangeSlider(0, 100)

        self.spin_temp_max = QtWidgets.QDoubleSpinBox()
        self.spin_temp_max.setRange(0, 100)
        self.spin_temp_max.setValue(100)
        self.spin_temp_max.setFixedWidth(60)

        # Connect Sliders <-> Spins
        self.range_slider.rangeChanged.connect(self._on_slider_changed)
        self.spin_temp_min.valueChanged.connect(self._on_spin_changed)
        self.spin_temp_max.valueChanged.connect(self._on_spin_changed)

        temp_layout.addWidget(self.spin_temp_min)
        temp_layout.addWidget(self.range_slider)
        temp_layout.addWidget(self.spin_temp_max)

        layout.addWidget(grp_temp)

        # --- Row 3: Ingredients (Multi-Select) ---
        grp_comp = QtWidgets.QGroupBox("Composition Ingredients")
        self.comp_layout = QtWidgets.QGridLayout(grp_comp)
        self.comp_layout.setVerticalSpacing(10)
        self.comp_layout.setHorizontalSpacing(15)

        self.ing_buttons = {}
        row, col = 0, 0
        MAX_COLS = 3

        for ing_type, items in self.ingredients_data.items():
            container = QtWidgets.QWidget()
            c_layout = QtWidgets.QVBoxLayout(container)
            c_layout.setContentsMargins(0, 0, 0, 0)
            c_layout.setSpacing(4)

            lbl = QtWidgets.QLabel(f"{ing_type}")

            # Custom Multi-Select Button
            btn = FilterMenuButton("All selected", items)
            self.ing_buttons[ing_type] = btn

            c_layout.addWidget(lbl)
            c_layout.addWidget(btn)

            self.comp_layout.addWidget(container, row, col)

            col += 1
            if col >= MAX_COLS:
                col = 0
                row += 1

        layout.addWidget(grp_comp)

        # --- Footer ---
        footer_layout = QtWidgets.QHBoxLayout()

        self.btn_reset = QtWidgets.QPushButton("Reset Filters")
        self.btn_reset.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_reset.setStyleSheet("color: #666; border: none; font-weight: 500;")
        self.btn_reset.clicked.connect(self.reset_filters)

        self.btn_apply = QtWidgets.QPushButton("Apply Filters")
        self.btn_apply.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_apply.setStyleSheet(
            """
            QPushButton {
                background-color: #0078D4; 
                color: white; 
                font-weight: bold; 
                border-radius: 4px; 
                padding: 6px 20px;
            }
            QPushButton:hover { background-color: #106EBE; }
        """
        )
        self.btn_apply.clicked.connect(self.emit_filter)

        footer_layout.addWidget(self.btn_reset)
        footer_layout.addStretch()
        footer_layout.addWidget(self.btn_apply)

        layout.addLayout(footer_layout)

    def emit_filter(self):
        filters = {
            "show_measured": self.chk_measured.isChecked(),
            "show_predicted": self.chk_predicted.isChecked(),
            "model_text": self.txt_model.text().lower(),
            "temp_min": self.spin_temp_min.value(),
            "temp_max": self.spin_temp_max.value(),
            "ingredients": {},
        }

        for ing_type, btn in self.ing_buttons.items():
            selected_items = btn.get_selected_items()
            total_items = len(btn.items_map)

            # OPTIMIZATION: Only add to filters if NOT "All Selected"
            # This makes the "is_default" check in the main UI much easier
            if len(selected_items) < total_items:
                filters["ingredients"][ing_type] = selected_items

        self.filter_changed.emit(filters)

    def _on_slider_changed(self, low, high):
        self.spin_temp_min.blockSignals(True)
        self.spin_temp_max.blockSignals(True)
        self.spin_temp_min.setValue(low)
        self.spin_temp_max.setValue(high)
        self.spin_temp_min.blockSignals(False)
        self.spin_temp_max.blockSignals(False)

    def _on_spin_changed(self):
        low = self.spin_temp_min.value()
        high = self.spin_temp_max.value()
        if low > high:
            low = high  # clamp
            self.spin_temp_min.setValue(low)

        self.range_slider.setValues(low, high)

    def reset_filters(self):
        """Resets all UI elements to default and emits the update."""
        # Block signals to prevent intermediate updates if you have auto-triggering on
        self.blockSignals(True)

        # 1. Reset UI Components
        self.chk_measured.setChecked(True)
        self.chk_predicted.setChecked(True)
        self.txt_model.clear()
        self.spin_temp_min.setValue(0)
        self.spin_temp_max.setValue(100)
        self.range_slider.setValues(0, 100)

        # Reset custom multi-select buttons
        for btn in self.ing_buttons.values():
            btn.select_all()

        self.blockSignals(False)

        # 2. CRITICAL: Emit the filter signal so the parent UI knows we reset
        self.emit_filter()

    def apply_filters(self, filter_data):
        """
        1. Filters the cards.
        2. Updates the 'active' style of the top bar button.
        3. Closes the menu.
        """
        search_text = self.search_bar.text().lower()

        # --- 1. Filter Logic ---
        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()

            if widget and isinstance(widget, PredictionConfigCard):
                matches_complex = widget.matches_filter(filter_data)
                matches_search = True
                if search_text:
                    matches_search = search_text in widget.get_searchable_text()

                if matches_complex and matches_search:
                    widget.show()
                else:
                    widget.hide()

        self.update_placeholder_visibility()

        # --- 2. Check if Default (Reset) ---
        # A filter is 'default' if it matches the reset state (everything allowed)
        is_default = (
            filter_data["show_measured"]
            and filter_data["show_predicted"]
            and filter_data["model_text"] == ""
            and filter_data["temp_min"] == 0
            and filter_data["temp_max"] == 100
            and
            # Check if all ingredient filters are either empty or "All Selected"
            # (Note: Your filter widget might send specific lists even for 'All'.
            #  We need to check if the filter effectively restricts anything.)
            #  For now, let's assume if the ingredients dict is empty it's default.
            not any(filter_data["ingredients"].values())
        )

        # --- 3. Update Button Style ---
        # Set "active" property: True if filtering, False if default
        self.btn_filter.setProperty("active", not is_default)

        # Force style refresh so the color changes immediately
        self.btn_filter.style().unpolish(self.btn_filter)
        self.btn_filter.style().polish(self.btn_filter)

        # --- 4. Hide Menu ---
        # This closes the dropdown immediately after clicking Reset or Apply
        self.filter_widget.hide()


class ModelOptionsDialog(QtWidgets.QDialog):
    def __init__(self, current_params, is_measured=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Inference Options")
        self.resize(350, 250)
        self.setModal(True)

        self.setStyleSheet(
            """
            QDialog { background-color: #ffffff; }
            QLabel { color: #333; }
            QDoubleSpinBox, QSpinBox { 
                border: 1px solid #ccc; border-radius: 4px; padding: 4px; min-height: 20px; 
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbb; background: white; height: 6px; border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #00adee; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: white; border: 1px solid #888; width: 14px; margin: -5px 0; border-radius: 7px;
            }
            /* Style for the Reset Button */
            QPushButton#btnReset {
                color: #d32f2f; background: transparent; border: none; text-align: left;
            }
            QPushButton#btnReset:hover {
                text-decoration: underline; background: #ffebee; border-radius: 4px;
            }
        """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # -- Header --
        lbl_header = QtWidgets.QLabel("Hyperparameters")
        lbl_header.setStyleSheet("font-weight: bold; font-size: 11pt; color: #00adee;")
        layout.addWidget(lbl_header)

        # -- Form Layout --
        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(10)

        # Learning Rate
        self.spin_lr = QtWidgets.QDoubleSpinBox()
        self.spin_lr.setRange(0.0001, 1.0)
        self.spin_lr.setSingleStep(0.001)
        self.spin_lr.setDecimals(4)
        self.spin_lr.setValue(current_params.get("lr", 0.01))
        self.spin_lr.setReadOnly(is_measured)
        form_layout.addRow("Learning Rate:", self.spin_lr)

        # Steps
        self.spin_steps = QtWidgets.QSpinBox()
        self.spin_steps.setRange(1, 10000)
        self.spin_steps.setSingleStep(10)
        self.spin_steps.setValue(current_params.get("steps", 50))
        self.spin_steps.setReadOnly(is_measured)
        form_layout.addRow("Inference Steps:", self.spin_steps)

        layout.addLayout(form_layout)

        # -- CI Slider Section --
        ci_group = QtWidgets.QVBoxLayout()
        ci_group.setSpacing(5)

        ci_header_layout = QtWidgets.QHBoxLayout()
        ci_header_layout.addWidget(QtWidgets.QLabel("Confidence Interval:"))
        self.lbl_ci_val = QtWidgets.QLabel(f"{current_params.get('ci', 95)}%")
        self.lbl_ci_val.setStyleSheet("font-weight: bold; color: #555;")
        self.lbl_ci_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        ci_header_layout.addWidget(self.lbl_ci_val)

        ci_group.addLayout(ci_header_layout)

        self.slider_ci = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_ci.setRange(0, 100)
        self.slider_ci.setValue(current_params.get("ci", 95))
        self.slider_ci.setEnabled(not is_measured)
        self.slider_ci.valueChanged.connect(lambda v: self.lbl_ci_val.setText(f"{v}%"))
        ci_group.addWidget(self.slider_ci)

        layout.addLayout(ci_group)
        layout.addStretch()

        # -- Footer Buttons (Reset + OK/Cancel) --
        btn_layout = QtWidgets.QHBoxLayout()

        # Reset Button (Left aligned)
        self.btn_reset = QtWidgets.QPushButton("Reset to Defaults")
        self.btn_reset.setObjectName("btnReset")
        self.btn_reset.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_reset.clicked.connect(self.reset_defaults)
        # Disable reset if card is measured/read-only
        self.btn_reset.setEnabled(not is_measured)

        btn_layout.addWidget(self.btn_reset)
        btn_layout.addStretch()

        # Standard Dialog Buttons (Right aligned)
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        btn_layout.addWidget(btn_box)

        layout.addLayout(btn_layout)

    def reset_defaults(self):
        """Resets the UI elements to default values."""
        self.spin_lr.setValue(0.01)
        self.spin_steps.setValue(50)
        self.slider_ci.setValue(95)

    def get_settings(self):
        return {
            "lr": self.spin_lr.value(),
            "steps": self.spin_steps.value(),
            "ci": self.slider_ci.value(),
        }


# Replace your existing DragHandle class with this one
class DragHandle(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(20)
        self.setFixedHeight(40)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.setStyleSheet("background: transparent;")
        self._dragging = False

    def paintEvent(self, event):
        # (Keep your existing paint logic here)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setBrush(QtGui.QColor("#777777"))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        dot_size = 4
        spacing = 4
        start_x = (self.width() - (dot_size * 2 + spacing)) / 2
        start_y = 15
        for row in range(3):
            for col in range(2):
                x = start_x + col * (dot_size + spacing)
                y = start_y + row * (dot_size + spacing)
                painter.drawEllipse(int(x), int(y), dot_size, dot_size)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = True
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)

            # Walk up hierarchy: Handle -> Card -> Container
            card = self.parent()
            container = card.parent()

            # Calculate where we clicked relative to the card's top-left
            # mapTo(card, ...) ensures we grab the card exactly where the mouse is
            offset = self.mapTo(card, event.pos())

            if hasattr(container, "start_drag"):
                container.start_drag(card, event.globalPos(), offset)

    def mouseMoveEvent(self, event):
        if self._dragging:
            card = self.parent()
            container = card.parent()
            if hasattr(container, "update_drag"):
                container.update_drag(event.globalPos())

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
            card = self.parent()
            container = card.parent()
            if hasattr(container, "finish_drag"):
                container.finish_drag()


class ReorderableCardContainer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(10)

        # 1. Force items to stack at the top
        self.main_layout.setAlignment(QtCore.Qt.AlignTop)

        self.dragged_card = None
        self.placeholder = None
        self.drag_offset = QtCore.QPoint(0, 0)

    def start_drag(self, card, global_mouse_pos, offset):
        self.dragged_card = card
        self.drag_offset = offset

        # Create placeholder
        self.placeholder = QtWidgets.QWidget()
        self.placeholder.setFixedSize(card.size())
        self.placeholder.setStyleSheet(
            "background: rgba(0, 173, 238, 0.1); border: 1px dashed #00adee; border-radius: 6px;"
        )

        idx = self.main_layout.indexOf(card)
        self.main_layout.takeAt(idx)
        self.main_layout.insertWidget(idx, self.placeholder)

        card.setParent(self)
        card.raise_()
        self.update_drag(global_mouse_pos)
        card.show()

    def update_drag(self, global_mouse_pos):
        if not self.dragged_card:
            return

        # Move card
        local_pos = self.mapFromGlobal(global_mouse_pos)
        target_y = local_pos.y() - self.drag_offset.y()
        target_x = self.main_layout.contentsMargins().left()
        self.dragged_card.move(target_x, target_y)

        # 2. Logic: Use the TOP of the dragged card (the handle position)
        drag_focus_y = target_y

        placeholder_idx = self.main_layout.indexOf(self.placeholder)
        new_idx = placeholder_idx

        count = self.main_layout.count()

        for i in range(count):
            item = self.main_layout.itemAt(i)
            widget = item.widget()

            if widget is None or widget == self.dragged_card:
                continue

            w_geo = widget.geometry()
            w_center_y = w_geo.y() + w_geo.height() / 2

            # If our handle is above the center of the target card, insert before it
            if drag_focus_y < w_center_y:
                # If we are currently "after" this slot, we need to move "up"
                if i < placeholder_idx:
                    new_idx = i
                else:
                    new_idx = i
                break
            else:
                # If we are below the center, we belong after.
                # If loop finishes without breaking, new_idx will stay at 'count' (handled below)
                if i >= new_idx:
                    new_idx = i + 1

        # 3. Apply Move
        if new_idx != placeholder_idx:
            new_idx = max(0, min(new_idx, count))
            self.main_layout.takeAt(placeholder_idx)
            self.main_layout.insertWidget(new_idx, self.placeholder)

    def finish_drag(self):
        if not self.dragged_card:
            return

        final_idx = self.main_layout.indexOf(self.placeholder)
        self.main_layout.takeAt(final_idx)
        self.placeholder.deleteLater()
        self.placeholder = None

        self.main_layout.insertWidget(final_idx, self.dragged_card)
        self.dragged_card = None


class PredictionConfigCard(QtWidgets.QFrame):
    removed = QtCore.Signal(object)
    run_requested = QtCore.Signal(dict)
    save_requested = QtCore.Signal(dict)
    expanded = QtCore.Signal(object)
    selection_changed = QtCore.Signal(bool)

    def __init__(
        self,
        default_name,
        ingredients_data,
        ingredient_types,
        ingredient_units,
        parent=None,
    ):
        super().__init__(parent)
        self.setProperty("class", "card")

        # Shadow
        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QtGui.QColor(0, 0, 0, 30))
        self.setGraphicsEffect(shadow)

        self.animation = QtCore.QPropertyAnimation(self, b"maximumHeight")
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuad)

        self.ingredients_master = ingredients_data
        self.INGREDIENT_TYPES = ingredient_types
        self.INGREDIENT_UNITS = ingredient_units

        self.active_ingredients = {}
        # Store ML Params in memory since widgets are gone
        self.ml_params = {"lr": 0.01, "steps": 50, "ci": 95}
        self.use_in_icl = True
        self.last_results = None
        self.is_expanded = True
        self.is_measured = False
        self.notes_visible = False
        self.is_selectable = False
        self.is_selected = False
        self.debounce_timer = QtCore.QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(300)
        self.debounce_timer.timeout.connect(self.emit_run_request)

        self._init_ui(default_name)
        self._connect_auto_updates()

    def _init_ui(self, default_name):
        root_layout = QtWidgets.QHBoxLayout(self)
        root_layout.setContentsMargins(5, 5, 15, 5)
        root_layout.setSpacing(5)
        # 1. Left Drag Handle
        self.drag_handle = DragHandle()
        root_layout.addWidget(self.drag_handle, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        self.drag_handle.setStyleSheet("background: transparent; margin-top: 5px;")

        # 2. Central Container
        self.center_widget = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(self.center_widget)
        center_layout.setContentsMargins(0, 10, 0, 5)
        center_layout.setSpacing(10)
        root_layout.addWidget(self.center_widget, stretch=1)

        # --- Header Section ---
        self.header_frame = QtWidgets.QFrame()
        header_layout = QtWidgets.QHBoxLayout(self.header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        self.name_input = QtWidgets.QLineEdit(default_name)
        self.name_input.setPlaceholderText("Prediction Name")
        self.name_input.setProperty("class", "title-input")

        self.lbl_measured = QtWidgets.QLabel("✓ Measured Data")
        self.lbl_measured.setProperty("class", "badge-success")
        self.lbl_measured.setVisible(False)

        # Delete Button
        self.btn_delete = QtWidgets.QPushButton()
        self.btn_delete.setIcon(
            QtGui.QIcon("QATCH/VisQAI/src/view/icons/delete-2-svgrepo-com.svg")
        )
        self.btn_delete.setFixedSize(32, 32)
        self.btn_delete.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_delete.setToolTip("Delete Prediction")

        # Style: Circular with Red Hover
        self.btn_delete.setStyleSheet(
            """
            QPushButton {
                border-radius: 16px; /* Perfect circle */
                background-color: transparent;
                border: 1px solid transparent;
            }
            QPushButton:hover {
                background-color: #ffebee; /* Light Red */
                color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #ffcdd2;
            }
        """
        )
        self.btn_delete.clicked.connect(lambda: self.removed.emit(self))
        # Hamburger Menu
        self.btn_options = QtWidgets.QPushButton()
        self.btn_options.setIcon(
            QtGui.QIcon("QATCH/VisQAI/src/view/icons/options-svgrepo-com.svg")
        )
        self.btn_options.setFixedSize(32, 32)
        self.btn_options.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        # Style: Circular with Blue Hover, hide menu indicator
        self.btn_options.setStyleSheet(
            """
            QPushButton {
                border-radius: 16px;
                background-color: transparent;
                color: #555;
                font-size: 16px;
                font-weight: bold;
                border: 1px solid transparent;
                padding-bottom: 2px; /* Center text vertically */
            }
            QPushButton::menu-indicator { image: none; } /* Hide default triangle */
            QPushButton:hover {
                background-color: #e3f2fd; /* Light Blue */
                color: #0078D4;
            }
            QPushButton:pressed {
                background-color: #bbdefb;
            }
        """
        )

        self.options_menu = QtWidgets.QMenu(self)
        self.options_menu.addAction("Save model")
        self.options_menu.addAction("Save model as...")
        self.options_menu.addSeparator()
        self.options_menu.addAction("Export Formulation").triggered.connect(
            self.export_formulation
        )
        self.act_clear = self.options_menu.addAction("Clear Formulation")
        self.act_clear.triggered.connect(self.clear_formulation)
        self._update_clear_state()
        self.options_menu.addSeparator()
        self.act_use_icl = self.options_menu.addAction("Use in ICL")
        self.act_use_icl.setCheckable(True)
        self.act_use_icl.setEnabled(False)
        self.act_use_icl.toggled.connect(self.set_icl_usage)

        # -- Model Options Action --
        self.act_model_opts = self.options_menu.addAction("Model Options")
        self.act_model_opts.triggered.connect(self.open_model_options)

        self.btn_options.setMenu(self.options_menu)

        header_layout.addWidget(self.name_input, stretch=1)
        header_layout.addWidget(self.lbl_measured)
        header_layout.addWidget(self.btn_delete)
        header_layout.addWidget(self.btn_options)

        center_layout.addWidget(self.header_frame)

        # --- Content Body ---
        self.content_frame = QtWidgets.QFrame()
        self.content_frame.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_LayoutUsesWidgetRect
        )
        content_layout = QtWidgets.QVBoxLayout(self.content_frame)
        content_layout.setContentsMargins(5, 0, 5, 0)
        content_layout.setSpacing(15)

        # Model Selection
        model_layout = QtWidgets.QHBoxLayout()
        model_label = QtWidgets.QLabel("Model:")
        model_label.setStyleSheet("font-weight: bold; color: #555;")
        self.model_display = QtWidgets.QLineEdit("VisQAI-ICL_v1_nightly")
        self.model_display.setReadOnly(True)
        self.model_display.setPlaceholderText("No model selected")
        self.model_display.setProperty("class", "sleek")
        self.btn_select_model = QtWidgets.QPushButton()
        self.btn_select_model.setObjectName("btnBrowseModel")  # Matches the CSS ID
        self.btn_select_model.setFixedWidth(40)
        self.btn_select_model.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_select_model.setToolTip("Select Model")
        self.btn_select_model.clicked.connect(self.browse_model_file)

        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_display, stretch=1)
        model_layout.addWidget(self.btn_select_model)

        content_layout.addLayout(model_layout)
        # -------------------------------

        self._add_divider(content_layout)

        # Ingredients
        self._add_header_with_help(
            content_layout,
            "Formulation Composition",
            "Define the components in your formulation (i.e. Protein, Buffer, Surfactant, Stabilizer, Salt, Excipient).",
        )
        self.ing_container_layout = QtWidgets.QVBoxLayout()
        self.ing_container_layout.setSpacing(5)
        content_layout.addLayout(self.ing_container_layout)

        self.btn_add_ing = QtWidgets.QPushButton("+ Add Component")
        self.btn_add_ing.setProperty("class", "primary")
        self.btn_add_ing.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_add_ing.clicked.connect(self.show_add_menu)
        content_layout.addWidget(self.btn_add_ing)

        self._add_divider(content_layout)

        # Environment
        self._add_header_with_help(
            content_layout,
            "Environment",
            "Set environmental conditions (i.e. Temperature in C)",
        )
        temp_layout = QtWidgets.QHBoxLayout()
        temp_lbl = QtWidgets.QLabel("Temperature:")
        temp_lbl.setFixedWidth(100)
        temp_layout.addWidget(temp_lbl)

        self.slider_temp = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_temp.setRange(0, 100)
        self.slider_temp.setValue(25)
        self.slider_temp.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        self.spin_temp = QtWidgets.QDoubleSpinBox()
        self.spin_temp.setProperty("class", "sleek")
        self.spin_temp.setRange(0.0, 100.0)
        self.spin_temp.setValue(25.0)
        self.spin_temp.setSuffix(" °C")
        self.spin_temp.setFixedWidth(90)
        self.spin_temp.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.slider_temp.valueChanged.connect(
            lambda v: self.spin_temp.setValue(float(v))
        )
        self.spin_temp.valueChanged.connect(lambda v: self.slider_temp.setValue(int(v)))

        temp_layout.addWidget(self.slider_temp)
        temp_layout.addWidget(self.spin_temp)
        content_layout.addLayout(temp_layout)

        self._add_divider(content_layout)

        lbl_notes = QtWidgets.QLabel("Notes:")
        lbl_notes.setStyleSheet("font-weight: bold; color: #555;")
        content_layout.addWidget(lbl_notes)

        # 2. Add the text edit (Always visible now)
        self.notes_edit = QtWidgets.QTextEdit()
        self.notes_edit.setPlaceholderText("Enter notes about this run...")
        self.notes_edit.setMaximumHeight(80)  # Fixed height since it doesn't animate
        self.notes_edit.setStyleSheet(
            "border: 1px solid #d1d5db; background: #fcfcfc; border-radius: 4px;"
        )

        content_layout.addWidget(self.notes_edit)

        center_layout.addWidget(self.content_frame)

        # Footer
        self.footer_frame = QtWidgets.QFrame()
        footer_layout = QtWidgets.QHBoxLayout(self.footer_frame)
        footer_layout.setContentsMargins(0, 5, 0, 0)
        footer_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.btn_toggle = QtWidgets.QToolButton()
        self.btn_toggle.setArrowType(QtCore.Qt.ArrowType.UpArrow)
        self.btn_toggle.clicked.connect(self.toggle_content)
        self.btn_toggle.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_toggle.setStyleSheet("border: none; background: transparent;")

        footer_layout.addWidget(self.btn_toggle)
        center_layout.addWidget(self.footer_frame)

    def get_searchable_text(self):
        """Returns a single lowercase string containing all card data."""
        # 1. Basic Fields
        parts = [
            self.name_input.text(),
            self.model_display.text(),
            self.notes_edit.toPlainText(),
            "Measured" if self.is_measured else "Predicted",
        ]

        # 2. Ingredient Values
        for ing_type, (combo, spin) in self.active_ingredients.items():
            parts.append(ing_type)  # e.g. "Protein"
            parts.append(combo.currentText())  # e.g. "mAb-1"
            # Optional: Add concentration if you want to search by numbers
            parts.append(str(spin.value()))

        # Join and normalize
        return " ".join(parts).lower()

    def _update_clear_state(self):
        """Disables Clear action if data is imported OR if no ingredients exist."""
        # Safety check in case UI isn't fully init yet
        if not hasattr(self, "act_clear"):
            return

        # Condition 1: Imported Data -> Always Disabled
        if self.is_measured:
            self.act_clear.setEnabled(False)
            return

        # Condition 2: Empty Formulation -> Disabled
        has_ingredients = len(self.active_ingredients) > 0
        self.act_clear.setEnabled(has_ingredients)

    def open_model_options(self):
        dlg = ModelOptionsDialog(self.ml_params, self.is_measured, self)
        if dlg.exec_() == QtWidgets.QDialog.DialogCode.Accepted:
            # Update internal params
            new_settings = dlg.get_settings()
            self.ml_params.update(new_settings)
            # Trigger re-run
            self.trigger_update()

    def _connect_auto_updates(self):
        self.name_input.textChanged.connect(self.trigger_update)
        self.model_display.textChanged.connect(self.trigger_update)
        self.spin_temp.valueChanged.connect(self.trigger_update)

    def set_icl_usage(self, checked):
        self.use_in_icl = checked
        self.trigger_update()

    def _add_divider(self, layout):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        line.setLineWidth(1)
        line.setProperty("class", "divider")
        layout.addWidget(line)

    def _add_header_with_help(self, layout, title, help_text):
        container = QtWidgets.QWidget()
        h_layout = QtWidgets.QHBoxLayout(container)
        h_layout.setContentsMargins(0, 5, 0, 5)
        h_layout.setSpacing(8)
        lbl = QtWidgets.QLabel(title)
        lbl.setProperty("class", "header-title")
        btn_help = QtWidgets.QToolButton()
        btn_help.setText("?")
        btn_help.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn_help.setProperty("class", "help-btn")
        btn_help.clicked.connect(
            lambda: QtWidgets.QMessageBox.information(self, title, help_text)
        )
        h_layout.addWidget(lbl)
        h_layout.addWidget(btn_help)
        h_layout.addStretch()
        layout.addWidget(container)

    def set_measured(self, is_measured: bool):
        self.set_measured_state(is_measured)

    def set_measured_state(self, is_measured: bool):
        self.is_measured = is_measured
        self.lbl_measured.setVisible(is_measured)
        self.setProperty("measured", is_measured)
        self.style().unpolish(self)
        self.style().polish(self)

        lock_state = is_measured

        self.name_input.setReadOnly(lock_state)
        self.btn_add_ing.setVisible(not lock_state)

        for combo, spin in self.active_ingredients.values():
            combo.setEnabled(not lock_state)
            spin.setReadOnly(lock_state)
        self.slider_temp.setEnabled(not lock_state)
        self.spin_temp.setReadOnly(lock_state)
        self.btn_select_model.setEnabled(not lock_state)
        self.notes_edit.setReadOnly(lock_state)

    def toggle_content(self):
        if not self.is_expanded:
            self.expanded.emit(self)
            self.emit_run_request()

        self.is_expanded = not self.is_expanded
        arrow = (
            QtCore.Qt.ArrowType.UpArrow
            if self.is_expanded
            else QtCore.Qt.ArrowType.DownArrow
        )
        self.btn_toggle.setArrowType(arrow)

        if not hasattr(self, "_anim_accordion"):
            self._anim_accordion = QtCore.QPropertyAnimation(
                self.content_frame, b"maximumHeight"
            )
            self._anim_accordion.setDuration(200)
            self._anim_accordion.setEasingCurve(QtCore.QEasingCurve.InOutQuad)

        try:
            self._anim_accordion.finished.disconnect()
        except TypeError:
            pass

        if self.is_expanded:
            self.content_frame.setVisible(True)
            self.content_frame.setMaximumHeight(16777215)
            target_height = self.content_frame.sizeHint().height()
            self._anim_accordion.setStartValue(0)
            self._anim_accordion.setEndValue(target_height)
            self._anim_accordion.finished.connect(
                lambda: self.content_frame.setMaximumHeight(16777215)
            )
        else:
            self._anim_accordion.setStartValue(self.content_frame.height())
            self._anim_accordion.setEndValue(0)
            self._anim_accordion.finished.connect(
                lambda: self.content_frame.setVisible(False)
            )
        self._anim_accordion.start()

    def matches_filter(self, filters):
        """Checks if card configuration matches the filter criteria."""

        # 1. State Filter
        if self.is_measured and not filters["show_measured"]:
            return False
        if not self.is_measured and not filters["show_predicted"]:
            return False

        # 2. Model Name Filter
        if filters["model_text"]:
            txt = filters["model_text"]
            if (
                txt not in self.name_input.text().lower()
                and txt not in self.model_display.text().lower()
            ):
                return False

        # 3. Temperature Filter
        current_temp = self.spin_temp.value()
        if not (filters["temp_min"] <= current_temp <= filters["temp_max"]):
            return False

        # 4. Ingredient Composition Filter (Multi-Select Support)
        if filters["ingredients"]:
            for type_filter, allowed_names in filters["ingredients"].items():

                # A. If the filter has NO items selected for this type, it excludes everything
                if not allowed_names:
                    # Special case: If card has NO ingredient of this type, does it pass?
                    # Usually "Selected: None" means "Show nothing".
                    # But if the card doesn't use this ingredient type at all, it's irrelevant.
                    # Let's assume strict filtering:
                    # If user unchecked all "Buffers", show only cards that have one of the (0) selected buffers.
                    # Which is impossible, unless the card has NO buffer?
                    if type_filter in self.active_ingredients:
                        return False
                    continue

                # B. If card HAS this ingredient type, check if value is in allowed list
                if type_filter in self.active_ingredients:
                    combo, _ = self.active_ingredients[type_filter]
                    current_name = combo.currentText()
                    if current_name not in allowed_names:
                        return False

                # C. If card DOES NOT have this ingredient type
                # (e.g. Card has no Buffer, but Filter says "Histidine" is allowed)
                # Typically we don't filter out cards that lack the ingredient entirely
                # unless we are enforcing "Must have Buffer".
                # For now, we assume if the ingredient isn't present, the filter for that type is ignored.
                pass

        return True

    def collapse(self):
        if not self.is_expanded:
            return
        self.is_expanded = False
        self.btn_toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow)

        if not hasattr(self, "_anim_accordion"):
            self._anim_accordion = QtCore.QPropertyAnimation(
                self.content_frame, b"maximumHeight"
            )
            self._anim_accordion.setDuration(300)
            self._anim_accordion.setEasingCurve(QtCore.QEasingCurve.InOutQuad)

        try:
            self._anim_accordion.finished.disconnect()
        except TypeError:
            pass
        self._anim_accordion.setStartValue(self.content_frame.height())
        self._anim_accordion.setEndValue(0)
        self._anim_accordion.finished.connect(
            lambda: self.content_frame.setVisible(False)
        )
        self._anim_accordion.start()

    def set_results(self, data):
        """Stores the simulation/measured data for export."""
        self.last_results = data

    def browse_model_file(self):
        """Opens file dialog to select a model."""
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "VisQAI Models (*.visq)",
        )
        if fname:
            file_info = QtCore.QFileInfo(fname)
            display_name = file_info.fileName()

            # Update the display
            self.model_display.setText(display_name)

            # (Optional) Store full path if you need it later
            self.selected_model_path = fname

            # Trigger run
            self.trigger_update()

    def show_add_menu(self):
        menu = QtWidgets.QMenu(self.btn_add_ing)
        menu.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        present_types = list(self.active_ingredients.keys())
        available = [t for t in self.INGREDIENT_TYPES if t not in present_types]
        if not available:
            action = menu.addAction("All types added")
            action.setEnabled(False)
        else:
            for t in available:
                action = menu.addAction(t)
                action.triggered.connect(
                    lambda checked, type_name=t: self.add_ingredient_row(type_name)
                )
        pos = self.btn_add_ing.mapToGlobal(QtCore.QPoint(0, self.btn_add_ing.height()))
        menu.exec(pos)

    def add_ingredient_row(self, ing_type):
        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)  # Add slight spacing between elements

        # Label
        lbl = QtWidgets.QLabel(f"{ing_type}:")
        lbl.setFixedWidth(80)
        lbl.setStyleSheet("color: #555; font-weight: 500;")

        # ComboBox
        combo = QtWidgets.QComboBox()
        combo.setProperty("class", "sleek")
        items = self.ingredients_master.get(ing_type, [])
        combo.addItem("None", None)
        for item in items:
            combo.addItem(item.name, item)

        # SpinBox
        spin = QtWidgets.QDoubleSpinBox()
        combo.setProperty("class", "sleek")
        spin.setRange(0, 1000)
        spin.setSingleStep(1.0)
        spin.setSuffix(f" {self.INGREDIENT_UNITS.get(ing_type,'')}")
        spin.setFixedWidth(90)

        # Connect Signals
        combo.currentTextChanged.connect(self.trigger_update)
        spin.valueChanged.connect(self.trigger_update)

        # --- IMPROVED REMOVE BUTTON ---
        btn_rem = QtWidgets.QPushButton()
        btn_rem.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TrashIcon)
        )
        btn_rem.setFixedSize(24, 24)  # Small, square button
        btn_rem.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn_rem.setToolTip("Remove Component")

        # Style: Transparent gray normally, Red background on hover
        btn_rem.setStyleSheet(
            """
            QPushButton {
                border: none;
                background-color: transparent;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #ffebee; /* Light Red */
                border: 1px solid #ffcdd2;
            }
        """
        )

        btn_rem.clicked.connect(
            lambda: self.remove_ingredient_row(ing_type, row_widget)
        )
        # ------------------------------

        row_layout.addWidget(lbl)
        row_layout.addWidget(combo, stretch=1)
        row_layout.addWidget(spin)
        row_layout.addWidget(btn_rem)

        self.ing_container_layout.addWidget(row_widget)
        self.active_ingredients[ing_type] = (combo, spin)

        # Update Clear State since we added an item
        self._update_clear_state()

    def trigger_update(self):
        if self.is_expanded:
            self.debounce_timer.start()

    def emit_run_request(self):
        config = self.get_configuration()
        self.run_requested.emit(config)

    def remove_ingredient_row(self, ing_type, widget):
        if ing_type in self.active_ingredients:
            del self.active_ingredients[ing_type]
        self._update_clear_state()
        self.trigger_update()
        if widget:
            anim = QtCore.QPropertyAnimation(widget, b"maximumHeight", widget)
            anim.setDuration(100)
            anim.setEasingCurve(QtCore.QEasingCurve.InQuad)
            anim.setStartValue(widget.height())
            anim.setEndValue(0)
            anim.finished.connect(widget.deleteLater)
            anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def on_selected_toggled(self, checked):
        """Updates the card style when selected."""
        self.setProperty("selected", checked)
        # Refresh style to apply the border change
        self.style().unpolish(self)
        self.style().polish(self)

    def get_configuration(self):
        formulation = {}
        for t, (combo, spin) in self.active_ingredients.items():
            formulation[t] = {
                "component": combo.currentText(),
                "concentration": spin.value(),
            }

        config = {
            "name": self.name_input.text(),
            "model": self.model_display.text(),
            "temp": self.spin_temp.value(),
            "formulation": formulation,
            "measured": self.is_measured,
            "notes": self.notes_edit.toPlainText(),
            "use_in_icl": self.use_in_icl,
        }
        config.update(self.ml_params)
        return config

    def load_data(self, data):
        if "name" in data:
            self.name_input.setText(data["name"])
        if "model" in data:
            self.model_display.setText(data["model"])
        if "temp" in data:
            self.spin_temp.setValue(float(data["temp"]))

        # Update ML Params storage directly
        if "lr" in data:
            self.ml_params["lr"] = float(data["lr"])
        if "steps" in data:
            self.ml_params["steps"] = int(data["steps"])
        if "ci" in data:
            self.ml_params["ci"] = int(data["ci"])
        if "use_in_icl" in data:
            self.use_in_icl = data["use_in_icl"]
        if "measured" in data:
            self.set_measured(data["measured"])
        if "notes" in data:
            self.notes_edit.setText(data["notes"])
        if "formulation" in data:
            for ing_type, details in data["formulation"].items():
                self.add_ingredient_row(ing_type)
                if ing_type in self.active_ingredients:
                    combo, spin = self.active_ingredients[ing_type]
                    spin.setValue(float(details.get("concentration", 0.0)))
                    comp_name = details.get("component")
                    if comp_name:
                        idx = combo.findText(comp_name)
                        if idx >= 0:
                            combo.setCurrentIndex(idx)

    def set_selectable(self, active: bool):
        """
        Enables Selection Mode.
        Disables editing inputs but keeps the expand/collapse button active.
        """
        self.is_selectable = active

        # 1. Visual Cue: Change cursor for the whole card
        if active:
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            # Optional: Deselect when exiting mode
            if self.is_selected:
                self.toggle_selection()

        # 2. Selectively disable inputs so clicks fall through to the card,
        #    BUT keep the footer (expand button) enabled.
        self.header_frame.setEnabled(not active)
        self.content_frame.setEnabled(not active)

    def toggle_selection(self):
        """Toggles the selected state and updates the style."""
        self.is_selected = not self.is_selected
        self.setProperty("selected", self.is_selected)

        # Force style refresh
        self.style().unpolish(self)
        self.style().polish(self)
        self.selection_changed.emit(self.is_selected)

    def mousePressEvent(self, event):
        """Intercept clicks to toggle selection if in Selection Mode."""
        if self.is_selectable:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.toggle_selection()
                return  # Consume event

        # Otherwise, behave normally
        super().mousePressEvent(event)

    def clear_formulation(self):
        """Clears all ingredients and resets environment to default."""
        if self.is_measured:
            QtWidgets.QMessageBox.warning(
                self, "Action Denied", "Cannot clear imported data."
            )
            return
        for ing_type in list(self.active_ingredients.keys()):
            combo, _ = self.active_ingredients[ing_type]
            row_widget = combo.parentWidget()

            # Use existing remove logic
            self.remove_ingredient_row(ing_type, row_widget)
        self.slider_temp.setValue(25)
        self.spin_temp.setValue(25.0)

        self.trigger_update()

    def export_formulation(self):
        """UI Trigger: Validates data and opens a file dialog to save a single CSV."""
        # 1. Validation
        if not self.last_results:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No data available to export.\nPlease run a prediction or import data first.",
            )
            return

        # 2. Get Filename
        default_path = f"{self.name_input.text()}.csv"
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Formulation",
            default_path,
            "CSV Files (*.csv)",
        )

        if fname:
            try:
                self.save_to_csv(fname)
                QtWidgets.QMessageBox.information(
                    self, "Export Successful", f"Successfully exported to:\n{fname}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Export Error", f"Failed to export file:\n{str(e)}"
                )

    def save_to_csv(self, filepath):
        """The functional logic: Writes the current card data to a specific path."""
        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # --- HEADER / METADATA ---
            writer.writerow(["--- Viscosity Profile ---"])
            xs = self.last_results.get("x", [])

            if self.is_measured:
                writer.writerow(
                    [
                        "Shear Rate (1/s)",
                        "Measured Viscosity (cP)",
                        "Lower CI (cP)",
                        "Upper CI (cP)",
                    ]
                )

                # FIX: Explicitly check for None instead of using 'or'
                meas_y = self.last_results.get("measured_y")
                if meas_y is None:
                    meas_y = [0] * len(xs)

                lower = self.last_results.get("lower")
                if lower is None:
                    lower = [0] * len(xs)

                upper = self.last_results.get("upper")
                if upper is None:
                    upper = [0] * len(xs)

                for x, m, l, u in zip(xs, meas_y, lower, upper):
                    writer.writerow([f"{x:.4f}", f"{m:.4f}", f"{l:.4f}", f"{u:.4f}"])
            else:
                writer.writerow(
                    [
                        "Shear Rate (1/s)",
                        "Predicted Viscosity (cP)",
                        "Lower CI (cP)",
                        "Upper CI (cP)",
                    ]
                )

                pred_y = self.last_results.get("y")
                if pred_y is None:
                    pred_y = [0] * len(xs)

                lower = self.last_results.get("lower")
                if lower is None:
                    lower = [0] * len(xs)

                upper = self.last_results.get("upper")
                if upper is None:
                    upper = [0] * len(xs)

                for x, y, l, u in zip(xs, pred_y, lower, upper):
                    writer.writerow([f"{x:.4f}", f"{y:.4f}", f"{l:.4f}", f"{u:.4f}"])


class VisualizationPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_data = None
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Splitter
        viz_splitter = QtWidgets.QSplitter(Qt.Vertical)

        # --- Graph Area Container ---
        self.graph_container = QtWidgets.QWidget()
        graph_layout = QtWidgets.QVBoxLayout(self.graph_container)
        graph_layout.setContentsMargins(0, 0, 0, 0)

        # 1. Hover Label (Top)
        self.hover_label = QtWidgets.QLabel("Hover over graph")
        self.hover_label.setStyleSheet("font-weight: bold; color: #333; padding: 2px;")
        self.hover_label.setAlignment(Qt.AlignCenter)
        graph_layout.addWidget(self.hover_label)

        # 2. Stacked Plot Area (Plot + Overlay)
        self.plot_stack = QtWidgets.QWidget()
        self.stack_layout = QtWidgets.QGridLayout(self.plot_stack)
        self.stack_layout.setContentsMargins(0, 0, 0, 0)

        # Layer A: The Plot
        self.plot_widget = pg.PlotWidget(title="Viscosity Profile")
        self.plot_widget.setLabel("left", "Viscosity", units="cP")
        self.plot_widget.setLabel("bottom", "Shear Rate", units="1/s")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()

        # [Re-add plot event filters and crosshairs here from previous code...]
        self.plot_widget.installEventFilter(self)
        self.vLine = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("k", style=Qt.DashLine)
        )
        self.hLine = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen("k", style=Qt.DashLine)
        )
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)
        self.proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved
        )

        # [MODIFIED SECTION: Overlay & Progress Bar]
        self.overlay_widget = QtWidgets.QFrame()
        self.overlay_widget.setStyleSheet("background-color: rgba(255, 255, 255, 180);")
        self.overlay_widget.setVisible(False)

        overlay_layout = QtWidgets.QVBoxLayout(self.overlay_widget)
        overlay_layout.setAlignment(Qt.AlignCenter)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setFixedWidth(300)
        self.progress_bar.setFixedHeight(25)

        # [NEW] Enable Text
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setFormat("Running Inference... %p%")  # T
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #999;
                border-radius: 4px;
                background-color: #fff;
                color: #333; /* Text color */
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #0078D7;
                border-radius: 3px;
            }
        """
        )
        overlay_layout.addWidget(self.progress_bar)

        # Add both to the same grid cell (0,0) so they stack
        self.stack_layout.addWidget(self.plot_widget, 0, 0)
        self.stack_layout.addWidget(self.overlay_widget, 0, 0)

        graph_layout.addWidget(self.plot_stack)
        viz_splitter.addWidget(self.graph_container)

        # --- Table Area ---
        self.results_table = QtWidgets.QTableWidget()
        self.results_table.setColumnCount(4)

        viz_splitter.addWidget(self.results_table)
        viz_splitter.setStretchFactor(0, 3)
        viz_splitter.setStretchFactor(1, 1)

        layout.addWidget(viz_splitter)

        # [Re-add Option Button logic]
        self.btn_opts = QtWidgets.QPushButton("⚙ Graph Options", self.plot_widget)
        self.btn_opts.setCursor(Qt.ArrowCursor)
        self.btn_opts.clicked.connect(self.show_options_menu)
        self._init_controls()

    def _init_controls(self):
        """Initialize actions and widgets for the options menu."""
        # Toggles
        self.act_log_x = QtWidgets.QAction("Log Scale X", self, checkable=True)
        self.act_log_x.setChecked(True)
        self.act_log_x.toggled.connect(self.update_plot)

        self.act_log_y = QtWidgets.QAction("Log Scale Y", self, checkable=True)
        self.act_log_y.setChecked(False)
        self.act_log_y.toggled.connect(self.update_plot)

        self.act_ci = QtWidgets.QAction(
            "Show Confidence Interval", self, checkable=True
        )
        self.act_ci.setChecked(True)
        self.act_ci.toggled.connect(self.update_plot)

        self.act_cp = QtWidgets.QAction("Show CP Overlay", self, checkable=True)
        self.act_cp.setChecked(False)
        self.act_cp.toggled.connect(self.update_plot)

        self.act_measured = QtWidgets.QAction(
            "Show Measured Profile", self, checkable=True
        )
        self.act_measured.setChecked(True)
        self.act_measured.setEnabled(False)
        self.act_measured.toggled.connect(self.update_plot)

        # Range Selector (Min / Max SpinBoxes)
        # Using SpinBoxes allows variable ranges beyond just fixed combo items
        self.spin_min_shear = QtWidgets.QDoubleSpinBox()
        self.spin_min_shear.setRange(0, 15000000)
        self.spin_min_shear.setValue(100)
        self.spin_min_shear.setGroupSeparatorShown(True)
        self.spin_min_shear.setDecimals(0)
        self.spin_min_shear.setSingleStep(1000)
        self.spin_min_shear.valueChanged.connect(self.update_plot)

        self.spin_max_shear = QtWidgets.QDoubleSpinBox()
        self.spin_max_shear.setRange(0, 15000000)
        self.spin_max_shear.setValue(15000000)
        self.spin_max_shear.setGroupSeparatorShown(True)
        self.spin_max_shear.setDecimals(0)
        self.spin_max_shear.setSingleStep(10000)
        self.spin_max_shear.valueChanged.connect(self.update_plot)

        # Hypothesis Button
        self.btn_hypothesis = QtWidgets.QPushButton("Add Hypothesis")
        self.btn_hypothesis.clicked.connect(self.open_hypothesis_dialog)

    def set_plot_title(self, title_text):
        """Updates the title of the plot widget."""
        # Check if plot_widget exists to avoid errors during init
        if hasattr(self, "plot_widget"):
            self.plot_widget.setTitle(f"Viscosity Profile: {title_text}")

    def update_table(self):
        if not self.last_data:
            return

        x, y = self.last_data["x"], self.last_data["y"]
        l, u = self.last_data["lower"], self.last_data["upper"]

        # Safe retrieval of measured data
        meas = self.last_data.get("measured_y")
        if meas is None:
            meas = [None] * len(x)

        self.results_table.setRowCount(len(x))

        for i, (sr, visc, low, high, m_visc) in enumerate(zip(x, y, l, u, meas)):
            # Format Shear Rate
            sr_txt = f"{sr:.2e}" if sr >= 10000 else f"{sr:.1f}"
            self.results_table.setItem(i, 0, QtWidgets.QTableWidgetItem(sr_txt))

            # Format Predicted Viscosity
            self.results_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{visc:.2f}"))

            # Format Interval
            self.results_table.setItem(
                i, 2, QtWidgets.QTableWidgetItem(f"{low:.2f} - {high:.2f}")
            )

            # Format Measured
            if m_visc is not None:
                # Handle potential numpy scalars or strings
                try:
                    val = float(m_visc)
                    m_text = f"{val:.2f}"
                except (ValueError, TypeError):
                    m_text = "-"
            else:
                m_text = "-"

            self.results_table.setItem(i, 3, QtWidgets.QTableWidgetItem(m_text))

    def show_loading(self):
        self.overlay_widget.setVisible(True)
        self.progress_bar.setValue(0)

        self.anim_timer = QtCore.QTimer()
        self.anim_timer.timeout.connect(self._animate_step)
        self.anim_timer.start(10)

    def _animate_step(self):
        """Increments progress bar."""
        val = self.progress_bar.value()
        if val < 90:  # Cap at 90% until process actually finishes
            self.progress_bar.setValue(val + 1)

    def hide_loading(self):
        """Stops animation and hides overlay."""
        if hasattr(self, "anim_timer"):
            self.anim_timer.stop()
        self.overlay_widget.setVisible(False)

    def eventFilter(self, source, event):
        """Handle resize events to keep the overlay button in the top right."""
        if source == self.plot_widget and event.type() == QtCore.QEvent.Resize:
            self._reposition_overlay_button()
        return super().eventFilter(source, event)

    def _reposition_overlay_button(self):
        """Moves button to Top-Right with a defined margin to prevent overrun."""
        margin_right = 25  # Offset from the right edge
        margin_top = 10  # Offset from the top edge

        # Calculate x position
        x = self.plot_widget.width() - self.btn_opts.width() - margin_right
        y = margin_top

        # Ensure x doesn't go negative if the window is extremely small
        if x < 0:
            x = 0

        self.btn_opts.move(x, y)

    def show_options_menu(self):
        menu = QtWidgets.QMenu(self)

        menu.addAction(self.act_log_x)
        menu.addAction(self.act_log_y)
        menu.addSeparator()
        menu.addAction(self.act_ci)
        menu.addAction(self.act_cp)
        menu.addAction(self.act_measured)
        menu.addSeparator()

        # Range Selectors
        range_widget = QtWidgets.QWidget()
        range_layout = QtWidgets.QGridLayout(range_widget)
        range_layout.setContentsMargins(10, 2, 10, 2)

        range_layout.addWidget(QtWidgets.QLabel("Min Shear (1/s):"), 0, 0)
        range_layout.addWidget(self.spin_min_shear, 0, 1)

        range_layout.addWidget(QtWidgets.QLabel("Max Shear (1/s):"), 1, 0)
        range_layout.addWidget(self.spin_max_shear, 1, 1)

        range_action = QtWidgets.QWidgetAction(menu)
        range_action.setDefaultWidget(range_widget)
        menu.addAction(range_action)

        menu.addSeparator()

        # Hypothesis Button
        hyp_action = QtWidgets.QWidgetAction(menu)
        hyp_btn_widget = QtWidgets.QWidget()
        hyp_layout = QtWidgets.QVBoxLayout(hyp_btn_widget)
        hyp_layout.setContentsMargins(10, 5, 10, 5)
        hyp_layout.addWidget(self.btn_hypothesis)
        hyp_action.setDefaultWidget(hyp_btn_widget)
        menu.addAction(hyp_action)

        # Show menu below the button
        menu.exec_(self.btn_opts.mapToGlobal(QtCore.QPoint(0, self.btn_opts.height())))

    def open_hypothesis_dialog(self):
        QtWidgets.QMessageBox.information(
            self, "Add Hypothesis", "Hypothesis dialog placeholder."
        )

    def set_data(self, data):
        self.last_data = data
        has_measured = "measured_y" in data and data["measured_y"] is not None
        self.act_measured.setEnabled(has_measured)
        if not has_measured:
            self.act_measured.setChecked(False)
        self.update_plot()
        self.update_table()

    def update_plot(self):
        if not self.last_data:
            return

        self.plot_widget.clear()
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)

        x_full = np.array(self.last_data["x"])

        # Filter based on Range Selector
        min_shear = self.spin_min_shear.value()
        max_shear = self.spin_max_shear.value()

        mask = (x_full >= min_shear) & (x_full <= max_shear)

        x = x_full[mask]
        y = np.array(self.last_data["y"])[mask]
        lower = np.array(self.last_data["lower"])[mask]
        upper = np.array(self.last_data["upper"])[mask]

        if len(x) == 0:
            return

        # 1. Plot CI
        if self.act_ci.isChecked():
            fill = pg.FillBetweenItem(
                pg.PlotDataItem(x, lower),
                pg.PlotDataItem(x, upper),
                brush=pg.mkBrush(0, 0, 255, 50),
            )
            self.plot_widget.addItem(fill)

        # 2. Plot Measured
        measured_data = self.last_data.get("measured_y")
        if (
            self.act_measured.isChecked()
            and self.act_measured.isEnabled()
            and measured_data is not None
        ):
            meas_y = np.array(measured_data)[mask]
            self.plot_widget.plot(
                x,
                meas_y,
                pen=pg.mkPen("r", width=2, style=Qt.DashLine),
                symbol="x",
                symbolBrush="r",
                name="Measured",
            )

        # 3. Plot Predicted
        self.plot_widget.plot(
            x,
            y,
            pen=pg.mkPen("b", width=2),
            symbol="o",
            symbolSize=6,
            symbolBrush="b",
            name="Predicted",
        )

        # 4. Plot CP
        if self.act_cp.isChecked():
            cp_y = y * (1 + (np.random.rand(len(y)) - 0.5) * 0.05)
            self.plot_widget.plot(
                x,
                cp_y,
                pen=None,
                symbol="t1",
                symbolSize=10,
                symbolBrush="g",
                name="CP Measure",
            )

        # --- UPDATE AXIS SETTINGS ---
        log_x = self.act_log_x.isChecked()
        log_y = self.act_log_y.isChecked()

        # 1. Set Log Mode first
        self.plot_widget.setLogMode(x=log_x, y=log_y)

        # 2. Define X Limits (Fixed Requirement)
        limit_x_min = 100
        limit_x_max = 15000000

        if log_x:
            vb_x_min = np.log10(limit_x_min)
            vb_x_max = np.log10(limit_x_max)
        else:
            vb_x_min = limit_x_min
            vb_x_max = limit_x_max

        # 3. Define Y Limits (No Maximum)
        # [FIX] We replace 'None' with concrete large numbers to avoid TypeError
        if log_y:
            # Log Mode:
            # Min: -10 (corresponds to 1e-10, effectively 0)
            # Max: 300 (corresponds to 1e300, near max float limit)
            vb_y_min = -10.0
            vb_y_max = 300.0
        else:
            # Linear Mode:
            # Min: 0
            # Max: 1e300 (Huge float, effectively infinite)
            vb_y_min = 0.0
            vb_y_max = 1e300

        # 4. Apply Limits
        # Explicitly pass all 4 values as floats. No 'None' allowed.
        self.plot_widget.plotItem.vb.setLimits(
            xMin=vb_x_min, xMax=vb_x_max, yMin=vb_y_min, yMax=vb_y_max
        )

    def mouse_moved(self, evt):
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x_val = mousePoint.x()
            y_val = mousePoint.y()

            # Fix OverflowError when zooming out too far in Log mode
            if self.act_log_x.isChecked():
                try:
                    # Prevent overflow by clamping or checking exponent bounds
                    if x_val > 300:  # 10^300 is near limit
                        x_val = float("inf")
                    else:
                        x_val = 10**x_val
                except OverflowError:
                    x_val = float("inf")

            if self.act_log_y.isChecked():
                try:
                    if y_val > 300:
                        y_val = float("inf")
                    else:
                        y_val = 10**y_val
                except OverflowError:
                    y_val = float("inf")

            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())

            def fmt(v):
                if v == float("inf"):
                    return "Inf"
                if abs(v) >= 10000 or (abs(v) < 0.01 and v != 0):
                    return f"{v:.2e}"
                return f"{v:.2f}"

            self.hover_label.setText(
                f"Shear Rate: {fmt(x_val)} 1/s  |  Viscosity: {fmt(y_val)} cP"
            )


class PredictionUI(QtWidgets.QWidget):
    INGREDIENT_TYPES = [
        "Protein",
        "Buffer",
        "Surfactant",
        "Stabilizer",
        "Excipient",
        "Salt",
    ]
    INGREDIENT_UNITS = {
        "Protein": "mg/mL",
        "Buffer": "mM",
        "Surfactant": "%w",
        "Stabilizer": "M",
        "Excipient": "mM",
        "Salt": "mM",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ingredients_by_type = {}
        self.selection_mode_active = False  # Track selection state
        self._load_mock_data()
        self.init_ui()
        self.setStyleSheet(LIGHT_STYLE_SHEET)

    def _load_mock_data(self):
        mk_obj = lambda n, t: type("obj", (object,), {"name": n, "type": t})
        self.ingredients_by_type["Protein"] = [
            mk_obj("Ibalizumab", "Protein"),
            mk_obj("mAb-1", "Protein"),
        ]
        self.ingredients_by_type["Buffer"] = [
            mk_obj("Histidine", "Buffer"),
            mk_obj("Acetate", "Buffer"),
        ]
        self.ingredients_by_type["Salt"] = [
            mk_obj("NaCl", "Salt"),
            mk_obj("KCl", "Salt"),
        ]
        self.ingredients_by_type["Surfactant"] = [
            mk_obj("PS20", "Surfactant"),
            mk_obj("PS80", "Surfactant"),
        ]
        self.ingredients_by_type["Excipient"] = [
            mk_obj("Sucrose", "Excipient"),
            mk_obj("Arginine", "Excipient"),
        ]
        for t in self.INGREDIENT_TYPES:
            if t not in self.ingredients_by_type:
                self.ingredients_by_type[t] = []

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)

        # --- LEFT PANEL ---
        self.left_widget = QtWidgets.QWidget()
        self.left_widget.setObjectName("leftPanel")
        left_layout = QtWidgets.QVBoxLayout(self.left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)  # Remove spacing between toolbar and list

        # 1. New Top Toolbar
        top_bar = self._create_top_bar()
        left_layout.addWidget(top_bar)
        # Parent it to left_widget so it sits inside that container
        self.filter_widget = PredictionFilterWidget(
            self.ingredients_by_type, parent=self.left_widget
        )
        self.filter_widget.filter_changed.connect(self.apply_filters)
        self.filter_widget.hide()  # Start hidden

        # Connect the Top Bar Filter Button
        # self.btn_filter.toggled.connect(self.toggle_filter_menu_manual())

        # Install event filter to track resize
        self.left_widget.installEventFilter(self)
        # 2. Scroll Area
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        # Custom Container
        # Assuming ReorderableCardContainer is defined in your project
        self.cards_container = ReorderableCardContainer()
        self.cards_container.setObjectName("scrollContent")
        self.cards_layout = self.cards_container.main_layout

        self.scroll_area.setWidget(self.cards_container)
        self.cards_layout.setContentsMargins(15, 15, 15, 15)
        self.cards_layout.setSpacing(10)
        self.placeholder = PlaceholderWidget()
        self.cards_layout.addWidget(self.placeholder)
        self.placeholder.hide()  # Hidden initially if you add a default card
        left_layout.addWidget(self.scroll_area)
        self._create_fab()
        self.left_widget.installEventFilter(self)
        splitter.addWidget(self.left_widget)

        # --- RIGHT PANEL ---
        # Assuming VisualizationPanel is defined in your project
        self.viz_panel = VisualizationPanel()
        splitter.addWidget(self.viz_panel)

        splitter.setSizes([450, 700])
        main_layout.addWidget(splitter)

        self.current_task = None
        self.add_prediction_card()

    def _create_top_bar(self):
        container = QtWidgets.QWidget()
        container.setObjectName("topBar")
        container.setFixedHeight(50)
        container.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        container.setStyleSheet(
            """
            QWidget#topBar {
                background-color: #ffffff;
                border-bottom: 1px solid #e0e0e0;
            }
        """
        )

        # Optional: Add a subtle drop shadow for depth
        shadow = QtWidgets.QGraphicsDropShadowEffect(container)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QtGui.QColor(0, 0, 0, 25))
        container.setGraphicsEffect(shadow)
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(15, 5, 15, 5)  # Increased side margins
        layout.setSpacing(10)

        # Search Bar
        self.search_bar = QtWidgets.QLineEdit()
        self.search_bar.setPlaceholderText("Search...")
        self.search_bar.setClearButtonEnabled(True)
        self.search_bar.addAction(
            QtGui.QIcon("QATCH/VisQAI/src/view/icons/search-svgrepo-com.svg"),
            QtWidgets.QLineEdit.ActionPosition.LeadingPosition,
        )
        self.search_bar.setStyleSheet(
            """
            QLineEdit {
                border: 1px solid #D1D1D1;
                border-radius: 16px;
                padding: 0 12px;
                background: #FFFFFF;
                height: 32px;
            }
            QLineEdit:focus { border: 1px solid #0078D4; }
        """
        )
        self.search_bar.textChanged.connect(self.filter_cards)
        layout.addWidget(self.search_bar, stretch=1)

        # Shared Button Style
        circle_btn_style = """
            QToolButton { border: 1px solid transparent; border-radius: 16px; background: transparent; }
            QToolButton:hover { background-color: #E1E1E1; }
            QToolButton:pressed { background-color: #D0D0D0; }
            QToolButton:checked { background-color: #CCE4F7; border: 1px solid #005A9E; }
        """

        # Filter
        self.btn_filter = QtWidgets.QToolButton()
        self.btn_filter.setIcon(
            QtGui.QIcon("QATCH/VisQAI/src/view/icons/filter-svgrepo-com.svg")
        )
        self.btn_filter.setToolTip("Filter Options")
        self.btn_filter.setCheckable(True)
        self.btn_filter.setFixedSize(32, 32)

        # CHANGE 2: Update Stylesheet to use [active="true"] for the blue state
        self.btn_filter.setStyleSheet(
            """
            QToolButton { border: 1px solid transparent; border-radius: 16px; background: transparent; }
            QToolButton:hover { background-color: #E1E1E1; }
            QToolButton:pressed { background-color: #D0D0D0; }
            
            /* New Active State Style */
            QToolButton[active="true"] { 
                background-color: #CCE4F7; 
                border: 1px solid #005A9E; 
            }
        """
        )

        # CHANGE 3: Connect clicked instead of toggled
        self.btn_filter.clicked.connect(self.toggle_filter_menu_manual)

        layout.addWidget(self.btn_filter)

        # Select Mode
        self.btn_select_mode = QtWidgets.QToolButton()
        self.btn_select_mode.setIcon(
            QtGui.QIcon("QATCH/VisQAI/src/view/icons/select-svgrepo-com.svg")
        )
        self.btn_select_mode.setToolTip("Enter Selection Mode")
        self.btn_select_mode.setCheckable(True)
        self.btn_select_mode.setFixedSize(32, 32)
        self.btn_select_mode.setStyleSheet(circle_btn_style)
        self.btn_select_mode.toggled.connect(self.toggle_selection_mode)
        layout.addWidget(self.btn_select_mode)

        # Add Menu for "Select All"
        self.btn_select_all = QtWidgets.QToolButton()
        self.btn_select_all.setIcon(
            QtGui.QIcon("QATCH/VisQAI/src/view/icons/select-multiple-svgrepo-com.svg")
        )  # Or SP_DialogApplyButton
        self.btn_select_all.setToolTip("Select All")
        self.btn_select_all.setFixedSize(32, 32)
        self.btn_select_all.setStyleSheet(circle_btn_style)
        self.btn_select_all.clicked.connect(self.select_all_cards)

        # Start disabled (only active in selection mode)
        self.btn_select_all.setEnabled(False)
        layout.addWidget(self.btn_select_all)

        # Import
        self.btn_import = QtWidgets.QToolButton()

        self.btn_import.setIcon(
            QtGui.QIcon("QATCH/VisQAI/src/view/icons/import-content-svgrepo-com.svg")
        )
        self.btn_import.setToolTip("Import Data")
        self.btn_import.setFixedSize(32, 32)
        self.btn_import.setStyleSheet(circle_btn_style)
        self.btn_import.clicked.connect(self.import_data_file)
        layout.addWidget(self.btn_import)
        # Export
        self.btn_export_top = QtWidgets.QToolButton()
        self.btn_export_top.setIcon(
            QtGui.QIcon("QATCH/VisQAI/src/view/icons/export-content-svgrepo-com.svg")
        )  # Replace with your icon path
        self.btn_export_top.setToolTip("Export Selected (or Open Card)")
        self.btn_export_top.setFixedSize(32, 32)
        self.btn_export_top.setStyleSheet(circle_btn_style)
        self.btn_export_top.clicked.connect(self.export_analysis)
        layout.addWidget(self.btn_export_top)
        # Separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        line.setFixedHeight(20)
        layout.addWidget(line)

        # --- NEW: Run Button ---
        self.btn_run_top = QtWidgets.QToolButton()
        self.btn_run_top.setIcon(
            QtGui.QIcon("QATCH/VisQAI/src/view/icons/play-svgrepo-com.svg")
        )
        self.btn_run_top.setToolTip("Run Inference (Selected or Open Card)")
        self.btn_run_top.setFixedSize(32, 32)
        self.btn_run_top.setStyleSheet(circle_btn_style)
        self.btn_run_top.clicked.connect(self.run_analysis)
        layout.addWidget(self.btn_run_top)

        # --- NEW: Delete Button ---
        self.btn_delete_top = QtWidgets.QToolButton()
        self.btn_delete_top.setIcon(
            QtGui.QIcon("QATCH/VisQAI/src/view/icons/delete-2-svgrepo-com.svg")
        )
        self.btn_delete_top.setToolTip("Delete (Selected or Open Card)")
        self.btn_delete_top.setFixedSize(32, 32)
        self.btn_delete_top.setStyleSheet(
            """
            QToolButton { border: 1px solid transparent; border-radius: 16px; background: transparent; }
            QToolButton:hover { background-color: #ffebee; color: #d32f2f; }
            QToolButton:pressed { background-color: #ffcdd2; }
            """
        )
        self.btn_delete_top.clicked.connect(self.delete_analysis)
        layout.addWidget(self.btn_delete_top)

        return container

    def _on_card_selection_changed(self):
        """
        Called whenever a card is selected/deselected.
        Auto-exits selection mode if the last item is deselected.
        """
        # Only perform this check if we are currently IN selection mode
        if not self.selection_mode_active:
            return

        selected_count = 0
        for i in range(self.cards_layout.count()):
            widget = self.cards_layout.itemAt(i).widget()
            if isinstance(widget, PredictionConfigCard) and widget.is_selected:
                selected_count += 1

        # If user deselected the last item, turn off the button
        if selected_count == 0:
            self.btn_select_mode.setChecked(False)
            # setChecked(False) triggers toggle_selection_mode(False) automatically

    def _create_fab(self):
        """Creates the Floating Action Button for Adding Cards."""
        self.btn_add_fab = QtWidgets.QPushButton("+", self.left_widget)
        self.btn_add_fab.setToolTip("New Prediction")
        self.btn_add_fab.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add_fab.resize(50, 50)

        # Shadow Effect
        shadow = QtWidgets.QGraphicsDropShadowEffect(self.btn_add_fab)
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QtGui.QColor(0, 0, 0, 80))
        self.btn_add_fab.setGraphicsEffect(shadow)

        self.btn_add_fab.setStyleSheet(
            """
            QPushButton {
                background-color: #0078D4;
                color: white;
                border-radius: 25px; /* Perfect Circle (50/2) */
                font-size: 30px;
                font-weight: 300;
                padding-bottom: 4px;
                border: none;
            }
            QPushButton:hover { background-color: #106EBE; transform: scale(1.05); }
            QPushButton:pressed { background-color: #005A9E; }
            QPushButton:disabled { background-color: #ccc; }
        """
        )
        self.btn_add_fab.clicked.connect(lambda: self.add_prediction_card(None))
        self.btn_add_fab.show()

    def eventFilter(self, source, event):
        """
        Handle resize events for the left panel to resize the filter overlay
        and reposition the Floating Action Button (FAB).
        """
        if source == self.left_widget and event.type() == QtCore.QEvent.Type.Resize:
            # 1. Update Filter Menu Geometry
            self._update_filter_geometry()

            # 2. Update FAB Geometry (Bottom Right Corner)
            fab_size = self.btn_add_fab.size()
            margin = 20

            # X = Width - ButtonWidth - Margin
            x = self.left_widget.width() - fab_size.width() - margin

            # Y = Height - ButtonHeight - Margin
            y = self.left_widget.height() - fab_size.height() - margin

            self.btn_add_fab.move(x, y)
            self.btn_add_fab.raise_()

        return super().eventFilter(source, event)

    def update_placeholder_visibility(self):
        """
        Updates the placeholder state based on visible cards.
        """
        visible_count = 0
        total_cards = 0

        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()

            if isinstance(widget, PredictionConfigCard):
                total_cards += 1
                # Check actual visibility attribute
                if not widget.isHidden():
                    visible_count += 1

        # Logic to show/hide placeholder
        if visible_count > 0:
            self.placeholder.hide()
        else:
            self.placeholder.show()

            if total_cards == 0:
                self.placeholder.lbl_text.setText(
                    "No predictions yet.\nClick the + button to add one."
                )
            else:
                self.placeholder.lbl_text.setText(
                    "No results found.\nTry adjusting your filters or search."
                )

    def _is_filter_default(self, filters):
        """Checks if the provided filter dict matches the default state."""
        # 1. Check Booleans
        if not filters["show_measured"] or not filters["show_predicted"]:
            return False

        # 2. Check Text/Numbers
        if filters["model_text"] != "":
            return False
        if filters["temp_min"] != 0 or filters["temp_max"] != 100:
            return False

        # 3. Check Ingredients (Should be empty or all empty lists)
        # The widget sends: {'Protein': ['mAb1'], ...}. Default is {} or keys with None/Empty.
        if filters["ingredients"]:
            for selected_list in filters["ingredients"].values():
                if (
                    selected_list
                ):  # If any list has items, it's not default (assuming "Select All" logic passes empty or specific list)
                    # Wait, in our logic "All Selected" usually passes a full list.
                    # Let's check against the "Reset" behavior of the widget.
                    # The widget's emit_filter sends specific items.
                    # If we simply want to close when "Reset" is clicked, we can check
                    # if the user manually cleared everything.
                    # BUT: The simplest way is to rely on the Reset Button explicitly closing it?
                    # No, the user said "un-press if no filter is applied".
                    return False
        return True

    def apply_filters(self, filter_data):
        """Iterates over cards and toggles visibility based on match."""
        search_text = self.search_bar.text().lower()

        # 1. Apply Filtering Logic
        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()

            if widget and isinstance(widget, PredictionConfigCard):
                matches_complex = widget.matches_filter(filter_data)
                matches_search = True
                if search_text:
                    matches_search = search_text in widget.get_searchable_text()

                if matches_complex and matches_search:
                    widget.show()
                else:
                    widget.hide()

        self.update_placeholder_visibility()

        # 2. Check if filters are currently default (Inactive)
        self.update_placeholder_visibility()

        # 1. Check if filters are default (Inactive)
        is_default = (
            filter_data["show_measured"]
            and filter_data["show_predicted"]
            and filter_data["model_text"] == ""
            and filter_data["temp_min"] == 0
            and filter_data["temp_max"] == 100
            and not any(filter_data["ingredients"].values())
        )

        # 2. Update Button Style (Blue if filters are active)
        # We use a custom property so it persists regardless of clicks
        self.btn_filter.setProperty("active", not is_default)

        # Force style refresh
        self.btn_filter.style().unpolish(self.btn_filter)
        self.btn_filter.style().polish(self.btn_filter)

        # 3. Always close the menu when "Apply" is clicked
        self.filter_widget.hide()

    def toggle_filter_menu_manual(self):
        """Toggles filter menu visibility on click."""
        if self.filter_widget.isVisible():
            self.filter_widget.hide()
        else:
            self.filter_widget.show()
            self.filter_widget.raise_()
            self._update_filter_geometry()

    # def toggle_filter_menu(self, checked):
    #     """Shows/Hides the filter menu and ensures it stays on top."""
    #     self.filter_widget.setVisible(checked)
    #     if checked:
    #         self.filter_widget.raise_()
    #         self._update_filter_geometry()

    def _update_filter_geometry(self):
        """Positions the filter widget directly below the top bar."""
        if self.filter_widget.isVisible():
            # Geometry: x=0, y=50 (height of top bar), width=panel width
            self.filter_widget.setGeometry(
                0, 50, self.left_widget.width(), self.filter_widget.sizeHint().height()
            )

    def run_analysis(self):
        """Runs inference on Selected cards (if any) or the currently Open card."""
        target_cards = []

        if self.selection_mode_active:
            # Gather all selected cards
            for i in range(self.cards_layout.count()):
                widget = self.cards_layout.itemAt(i).widget()
                if isinstance(widget, PredictionConfigCard) and widget.is_selected:
                    target_cards.append(widget)
        else:
            # Find the expanded card
            for i in range(self.cards_layout.count()):
                widget = self.cards_layout.itemAt(i).widget()
                if isinstance(widget, PredictionConfigCard) and widget.is_expanded:
                    target_cards.append(widget)
                    break

        if not target_cards:
            QtWidgets.QMessageBox.information(
                self, "Run Inference", "No cards selected or open to run."
            )
            return

        # Trigger Run
        # Note: In a real app, you might want to queue these.
        # Here we trigger them, which calls run_prediction logic.
        for card in target_cards:
            card.emit_run_request()

    def delete_analysis(self):
        """Deletes Selected cards (if any) or the currently Open card."""
        target_cards = []

        if self.selection_mode_active:
            for i in range(self.cards_layout.count()):
                widget = self.cards_layout.itemAt(i).widget()
                if isinstance(widget, PredictionConfigCard) and widget.is_selected:
                    target_cards.append(widget)
        else:
            for i in range(self.cards_layout.count()):
                widget = self.cards_layout.itemAt(i).widget()
                if isinstance(widget, PredictionConfigCard) and widget.is_expanded:
                    target_cards.append(widget)
                    break

        if not target_cards:
            return

        # Confirm deletion
        count = len(target_cards)
        msg = f"Are you sure you want to delete {count} prediction(s)?"
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Delete",
            msg,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            for card in target_cards:
                self.remove_card(card)

            # --- NEW: Turn off selection mode if we just deleted selected items ---
            if self.selection_mode_active:
                self.btn_select_mode.setChecked(False)
            # ----------------------------------------------------------------------

            # If we deleted the expanded card in normal mode, expand the last one
            elif self.cards_layout.count() > 0:
                last_item = self.cards_layout.itemAt(self.cards_layout.count() - 1)
                if last_item and last_item.widget():
                    last_item.widget().toggle_content()

    def toggle_selection_mode(self, active):
        """Toggles selection mode and handles FAB state."""
        self.selection_mode_active = active

        # Disable Add FAB and Import when selecting
        self.btn_add_fab.setEnabled(not active)
        self.btn_import.setEnabled(not active)
        self.btn_select_all.setEnabled(active)
        # Dim the FAB visually if disabled
        opacity = 0.5 if active else 1.0
        effect = QtWidgets.QGraphicsOpacityEffect(self.btn_add_fab)
        effect.setOpacity(opacity)
        self.btn_add_fab.setGraphicsEffect(effect)

        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()
            if widget and isinstance(widget, PredictionConfigCard):
                if hasattr(widget, "set_selectable"):
                    widget.set_selectable(active)

    def select_all_cards(self):
        """Selects all visible cards. If all are already selected, deselects all."""
        visible_cards = []
        for i in range(self.cards_layout.count()):
            w = self.cards_layout.itemAt(i).widget()
            if isinstance(w, PredictionConfigCard) and not w.isHidden():
                visible_cards.append(w)

        if not visible_cards:
            return

        selected_visible = [w for w in visible_cards if w.is_selected]

        # If everything visible is already selected, deselect them all
        should_select = len(selected_visible) < len(visible_cards)

        for widget in visible_cards:
            if widget.is_selected != should_select:
                widget.toggle_selection()

    def filter_cards(self, text):
        """Filters cards based on search text and updates placeholder."""
        search_text = text.lower().strip()
        search_tokens = search_text.split()

        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()

            if widget and isinstance(widget, PredictionConfigCard):
                # 1. Get content
                card_content = widget.get_searchable_text()

                # 2. Check Search Match
                matches_search = True
                if search_tokens:
                    for token in search_tokens:
                        if token not in card_content:
                            matches_search = False
                            break

                # 3. Check Filter Widget Match (if active)
                # We need to respect the filter menu even when searching
                matches_filter = True
                if hasattr(self, "filter_widget") and self.filter_widget.isVisible():
                    # We can't easily access the raw filter dict here without storing it,
                    # but typically search narrows down the *current* view.
                    # If you want strict intersection, you'd need to store 'self.current_filters'
                    pass

                # Apply Visibility
                if matches_search:
                    widget.show()
                else:
                    widget.hide()

        # --- CRITICAL FIX: Update placeholder after loop ---
        self.update_placeholder_visibility()

    def add_prediction_card(self, data=None):
        if data and "name" in data:
            name = data["name"]
        else:
            # Optional: Fix naming count to ignore placeholder/other widgets
            # (Currently counts placeholder as 1, so first card is "Prediction 2")
            count = 0
            for i in range(self.cards_layout.count()):
                if isinstance(
                    self.cards_layout.itemAt(i).widget(), PredictionConfigCard
                ):
                    count += 1
            name = f"Prediction {count + 1}"

        card = PredictionConfigCard(
            default_name=name,
            ingredients_data=self.ingredients_by_type,
            ingredient_types=self.INGREDIENT_TYPES,
            ingredient_units=self.INGREDIENT_UNITS,
        )
        card.removed.connect(self.remove_card)
        card.run_requested.connect(self.run_prediction)
        card.expanded.connect(self.on_card_expanded)
        card.selection_changed.connect(self._on_card_selection_changed)
        # --- FIX 2: Simplified Insertion ---
        insert_idx = self.cards_layout.count()
        self.cards_layout.insertWidget(insert_idx, card)

        # --- CRITICAL FIX: Explicitly show card so visibility check passes ---
        card.show()
        # --------------------------------------------------------------------

        self.update_placeholder_visibility()

        if data:
            if hasattr(card, "load_data"):
                card.load_data(data)
            if data.get("measured", False):
                card.set_measured_state(True)
            card.emit_run_request()

        self.on_card_expanded(card)
        QtCore.QTimer.singleShot(100, lambda: self._scroll_to_card(card))

        # Update again just in case data loading changed visibility state
        self.update_placeholder_visibility()

    def _scroll_to_card(self, card_widget):
        """Helper to ensure the new card is visible in the scroll area."""
        self.scroll_area.ensureWidgetVisible(card_widget, 0, 0)

    def on_card_expanded(self, active_card):
        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, PredictionConfigCard) and widget is not active_card:
                widget.collapse()

    def remove_card(self, card_widget):
        card_widget.setDisabled(True)
        anim = QtCore.QPropertyAnimation(card_widget, b"maximumHeight", card_widget)
        anim.setDuration(200)
        anim.setStartValue(card_widget.height())
        anim.setEndValue(0)
        anim.setEasingCurve(QtCore.QEasingCurve.InBack)

        def cleanup():
            self.cards_layout.removeWidget(card_widget)
            card_widget.deleteLater()
            QtCore.QTimer.singleShot(10, self.update_placeholder_visibility)

        anim.finished.connect(cleanup)
        anim.start()

    def import_data_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Run Data",
            "",  # Default directory
            "CSV Files (*.csv);;All Files (*)",
        )
        if not fname:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

        try:

            imported_config = {
                "name": f"Imported: {QtCore.QFileInfo(fname).baseName()}",
                "measured": True,
                "use_in_icl": True,
                "notes": "Imported dataset.",
            }
            self.add_prediction_card(imported_config)
            self.run_prediction(imported_config)

        finally:

            QtWidgets.QApplication.restoreOverrideCursor()

    def get_target_cards(self):
        """
        Returns a list of cards to act upon.
        Priority:
        1. All cards marked 'is_selected' if selection_mode is active.
        2. The single 'is_expanded' card if not in selection mode.
        """
        targets = []

        if self.selection_mode_active:
            for i in range(self.cards_layout.count()):
                widget = self.cards_layout.itemAt(i).widget()
                if isinstance(widget, PredictionConfigCard) and widget.is_selected:
                    targets.append(widget)
        else:
            # Fallback to the currently open/expanded card
            for i in range(self.cards_layout.count()):
                widget = self.cards_layout.itemAt(i).widget()
                if isinstance(widget, PredictionConfigCard) and widget.is_expanded:
                    targets.append(widget)
                    break  # Only one can be expanded at a time

        if not targets:
            QtWidgets.QMessageBox.information(
                self,
                "No Selection",
                "Please select cards or expand one to perform this action.",
            )

        return targets

    def export_analysis(self):
        """Batch export logic for the top bar."""
        target_cards = self.get_target_cards()  # Helper to get selected or expanded

        if not target_cards:
            return

        if len(target_cards) == 1:
            target_cards[0].export_formulation()
        else:
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Export Directory"
            )
            if folder:
                success = 0
                for card in target_cards:
                    if not card.last_results:
                        continue

                    # Clean filename
                    base_name = card.name_input.text().replace(" ", "_")
                    file_path = os.path.join(folder, f"{base_name}.csv")

                    try:
                        card.save_to_csv(file_path)
                        success += 1
                    except Exception as e:
                        print(f"Error exporting {base_name}: {e}")

                QtWidgets.QMessageBox.information(
                    self, "Batch Export", f"Exported {success} files to {folder}"
                )

    def run_prediction(self, config=None):
        """
        Runs prediction using the robust PredictionThread subclass.
        """
        sender_card = self.sender()
        if isinstance(sender_card, PredictionConfigCard):
            self.running_card = sender_card
        else:
            self.running_card = None
        if self.current_task is not None and self.current_task.isRunning():
            print("Stopping previous task...")
            self.current_task.stop()

        # 2. Visual Feedback
        name = config.get("name", "Unknown Sample") if config else "Unknown Sample"
        self.viz_panel.set_plot_title(f"Calculating: {name}...")
        self.viz_panel.show_loading()

        # 3. Create & Start New Thread
        # We assign it to self.current_task to keep the Python object alive
        self.current_task = PredictionThread(config)

        # 4. Connect Signals
        self.current_task.data_ready.connect(self._on_prediction_finished)

        # Clean up reference when done (Optional, but good for memory)
        self.current_task.finished.connect(self._on_task_complete)

        # 5. Start
        self.current_task.start()

    def _on_prediction_finished(self, data_package):
        final_name = data_package.get("config_name", "Unknown")
        self.viz_panel.set_plot_title(final_name)
        self.viz_panel.set_data(data_package)
        self.viz_panel.hide_loading()
        if hasattr(self, "running_card") and self.running_card:
            self.running_card.set_results(data_package)

    def _on_task_complete(self):
        """Called when thread naturally finishes."""
        # We don't set self.current_task to None here immediately
        # to avoid race conditions, but it's safe to leave it bound.
        pass

    def closeEvent(self, event):
        """
        Guaranteed cleanup on close.
        """
        if self.current_task is not None and self.current_task.isRunning():
            print("Closing application: Stopping background thread...")
            self.current_task.stop()

        super().closeEvent(event)


class PredictionThread(QtCore.QThread):
    """
    A robust QThread subclass.
    Combines the thread and logic into one object to prevent Garbage Collection errors.
    """

    # Define the signal that sends data back to the main UI
    data_ready = QtCore.Signal(dict)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._is_running = True  # Logic flag

    def run(self):
        """
        The code that runs in the background.
        """
        import time

        import numpy as np

        # 1. Simulate Calculation (Safe to sleep here)
        time.sleep(0.8)

        # 2. Check if we were told to stop during the sleep
        if not self._is_running:
            return

        # 3. The Math Logic
        shear_rates = np.logspace(1, 6, 20)
        K = 50.0
        n = 0.6

        if self.config:
            temp_mod = float(self.config.get("temp", 25)) / 25.0
            K = K / max(0.1, temp_mod)

        viscosity = K * np.power(shear_rates, n - 1)

        measured_y = None
        if self.config and self.config.get("measured", False):
            noise = np.random.normal(0, 2, len(shear_rates))
            measured_y = viscosity + noise

        data_package = {
            "x": shear_rates,
            "y": viscosity,
            "upper": viscosity * 1.15,
            "lower": viscosity * 0.85,
            "measured_y": measured_y,
            "config_name": (
                self.config.get("name", "Unknown") if self.config else "Unknown"
            ),
        }

        # 4. Emit Data
        if self._is_running:
            self.data_ready.emit(data_package)

    def stop(self):
        """Call this to stop the thread gracefully."""
        self._is_running = False
        self.quit()
        self.wait(1000)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # app.setStyleSheet(LIGHT_STYLE_SHEET)
    win = PredictionUI()
    win.setWindowTitle("Viscosity AI - Hyperparameter Tuning")
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())
