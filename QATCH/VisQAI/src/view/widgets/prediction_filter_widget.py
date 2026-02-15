"""
PredictionFilterWidget - Advanced filtering panel for predictions

Provides comprehensive filtering options including state, model, temperature,
and ingredient composition. Styling handled by theme.qss.
"""

from components.filter_menu_button import FilterMenuButton
from components.range_slider import RangeSlider
from PyQt5 import QtCore, QtGui, QtWidgets


class PredictionFilterWidget(QtWidgets.QWidget):
    """
    A dropdown filter panel for refining prediction card visibility.

    Emits:
        filter_changed: Signal with dict containing all filter criteria

    Styling is handled by theme.qss via PredictionFilterWidget and
    specific element selectors.
    """

    filter_changed = QtCore.pyqtSignal(dict)

    def __init__(self, ingredients_data, parent=None):
        super().__init__(parent)
        self.ingredients_data = ingredients_data

        # Enable styling via QSS
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        # ✨ No inline stylesheet - handled by theme.qss

        # Add shadow effect for depth
        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setYOffset(10)
        shadow.setColor(QtGui.QColor(0, 0, 0, 60))
        self.setGraphicsEffect(shadow)

        self.setVisible(False)
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # --- Row 1: State & Model ---
        row1_layout = QtWidgets.QHBoxLayout()
        row1_layout.setSpacing(15)

        # State Group
        grp_state = QtWidgets.QGroupBox("State")
        state_layout = QtWidgets.QHBoxLayout(grp_state)
        self.chk_measured = QtWidgets.QCheckBox("Measured")
        self.chk_measured.setChecked(True)
        self.chk_predicted = QtWidgets.QCheckBox("Predicted")
        self.chk_predicted.setChecked(True)
        state_layout.addWidget(self.chk_measured)
        state_layout.addWidget(self.chk_predicted)
        row1_layout.addWidget(grp_state)

        # Model Group
        grp_model = QtWidgets.QGroupBox("Model")
        model_layout = QtWidgets.QVBoxLayout(grp_model)
        self.txt_model = QtWidgets.QLineEdit()
        self.txt_model.setObjectName("filterModelInput")  # ✨ For QSS styling
        self.txt_model.setPlaceholderText("Name contains...")
        # ✨ No inline stylesheet - handled by theme.qss
        model_layout.addWidget(self.txt_model)
        row1_layout.addWidget(grp_model)

        layout.addLayout(row1_layout)

        # --- Row 2: Temperature Range ---
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
        self.btn_reset.setObjectName("btnResetFilters")  # ✨ For QSS styling
        self.btn_reset.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        # ✨ No inline stylesheet - handled by theme.qss
        self.btn_reset.clicked.connect(self.reset_filters)

        self.btn_apply = QtWidgets.QPushButton("Apply Filters")
        self.btn_apply.setObjectName("btnApplyFilters")  # ✨ For QSS styling
        self.btn_apply.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        # ✨ No inline stylesheet - handled by theme.qss
        self.btn_apply.clicked.connect(self.emit_filter)

        footer_layout.addWidget(self.btn_reset)
        footer_layout.addStretch()
        footer_layout.addWidget(self.btn_apply)

        layout.addLayout(footer_layout)

    def emit_filter(self):
        """
        Gather all filter values and emit them to parent.
        """
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

            # Optimization: Only add to filters if NOT "All Selected"
            # This makes the "is_default" check in the main UI easier
            if len(selected_items) < total_items:
                filters["ingredients"][ing_type] = selected_items

        self.filter_changed.emit(filters)

    def _on_slider_changed(self, low, high):
        """Update spinboxes when slider changes"""
        self.spin_temp_min.blockSignals(True)
        self.spin_temp_max.blockSignals(True)
        self.spin_temp_min.setValue(low)
        self.spin_temp_max.setValue(high)
        self.spin_temp_min.blockSignals(False)
        self.spin_temp_max.blockSignals(False)

    def _on_spin_changed(self):
        """Update slider when spinboxes change"""
        low = self.spin_temp_min.value()
        high = self.spin_temp_max.value()

        # Clamp low to not exceed high
        if low > high:
            low = high
            self.spin_temp_min.setValue(low)

        self.range_slider.setValues(low, high)

    def reset_filters(self):
        """
        Reset all UI elements to default and emit the update.
        """
        # Block signals to prevent intermediate updates
        self.blockSignals(True)

        # Reset UI Components
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

        # Emit the filter signal so parent knows we reset
        self.emit_filter()

    def get_filter_state(self):
        """
        Get the current filter state.

        Returns:
            dict: Current filter values
        """
        return {
            "show_measured": self.chk_measured.isChecked(),
            "show_predicted": self.chk_predicted.isChecked(),
            "model_text": self.txt_model.text().lower(),
            "temp_min": self.spin_temp_min.value(),
            "temp_max": self.spin_temp_max.value(),
            "ingredients": {
                ing_type: btn.get_selected_items()
                for ing_type, btn in self.ing_buttons.items()
            },
        }

    def is_default_state(self):
        """
        Check if filters are in default (unfiltered) state.

        Returns:
            bool: True if all filters are at default values
        """
        state = self.get_filter_state()

        # Check basic filters
        if not (state["show_measured"] and state["show_predicted"]):
            return False
        if state["model_text"]:
            return False
        if state["temp_min"] != 0 or state["temp_max"] != 100:
            return False

        # Check if any ingredients are filtered
        for ing_type, btn in self.ing_buttons.items():
            total_items = len(btn.items_map)
            selected_items = btn.get_selected_items()
            if len(selected_items) < total_items:
                return False

        return True
