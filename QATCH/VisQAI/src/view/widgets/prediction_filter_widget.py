"""
prediction_filter_widget.py

Filtering panel for refining prediction card visibility in the
VisQAI prediction UI.

Provides a collapsible dropdown panel (``PredictionFilterWidget``) with four
filter dimensions: prediction state (measured vs. predicted), model name
substring, temperature range, and per-ingredient-type multi-selection.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.0
"""

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from components.filter_menu_button import FilterMenuButton
    from components.range_slider import RangeSlider
except (ImportError, ModuleNotFoundError):
    from QATCH.VisQAI.src.view.components.filter_menu_button import FilterMenuButton
    from QATCH.VisQAI.src.view.components.range_slider import RangeSlider


class PredictionFilterWidget(QtWidgets.QWidget):
    """A dropdown filter panel for refining prediction card visibility.

    Renders as a floating ``QWidget`` with a drop-shadow effect.  The panel
    starts hidden; callers toggle it with ``setVisible``.  All styling is
    handled by ``theme.qss`` via the ``PredictionFilterWidget`` type selector
    and the object-name selectors ``#filterModelInput``, ``#btnResetFilters``,
    and ``#btnApplyFilters``.

    A footer row provides "Reset Filters" (left) and "Apply Filters" (right)
    buttons.

    Attributes:
        ingredients_data (dict[str, list]): Mapping of ingredient category name
            to a list of ingredient items forwarded to each ``FilterMenuButton``.
        ing_buttons (dict[str, FilterMenuButton]): Mapping of ingredient
            category name to its corresponding multi-select button, used by
            ``emit_filter``, ``reset_filters``, ``get_filter_state``, and
            ``is_default_state``.
        chk_measured (QtWidgets.QCheckBox): Toggles inclusion of measured
            data points.
        chk_predicted (QtWidgets.QCheckBox): Toggles inclusion of predicted
            data points.
        txt_model (QtWidgets.QLineEdit): Free-text input for model name
            filtering.
        spin_temp_min (QtWidgets.QDoubleSpinBox): Lower bound of the
            temperature range (°C), synchronised with ``range_slider``.
        spin_temp_max (QtWidgets.QDoubleSpinBox): Upper bound of the
            temperature range (°C), synchronised with ``range_slider``.
        range_slider (RangeSlider): Dual-handle slider kept in sync with
            ``spin_temp_min`` and ``spin_temp_max``.
    """

    filter_changed = QtCore.pyqtSignal(dict)

    def __init__(self, ingredients_data, parent=None):
        """Initialise the filter panel, apply shadow styling, and build the UI.

        Attaches a ``QGraphicsDropShadowEffect`` for visual depth, sets the
        widget hidden by default, and delegates full UI construction to
        ``_init_ui``.

        Args:
            ingredients_data (dict[str, list]): Mapping of ingredient category
                name (e.g. ``"Protein"``, ``"Buffer"``) to a list of ingredient
                items.  Passed directly to each ``FilterMenuButton`` so users
                can multi-select specific ingredients per category.
            parent (QtWidgets.QWidget | None): Optional Qt parent widget.
                Defaults to ``None``.
        """
        super().__init__(parent)
        self.ingredients_data = ingredients_data
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setYOffset(10)
        shadow.setColor(QtGui.QColor(0, 0, 0, 60))
        self.setGraphicsEffect(shadow)
        self.setVisible(False)
        self._init_ui()

    def _init_ui(self):
        """Build and wire all child widgets in a top-to-bottom ``QVBoxLayout``.

        Constructs four filter groups and a footer in order:

        1. "State" group box containing
           ``chk_measured`` and ``chk_predicted``; "Model" group box containing
           ``txt_model``.
        2. "Temperature Range (°C)" group box containing
           ``spin_temp_min``, ``range_slider``, and ``spin_temp_max``.
           ``range_slider.rangeChanged`` is wired to ``_on_slider_changed``;
           both spin boxes' ``valueChanged`` are wired to ``_on_spin_changed``
           to maintain bidirectional synchronisation.
        3. "Composition Ingredients" group box containing a
           three-column ``QGridLayout`` of ``FilterMenuButton`` widgets, one
           per entry in ``ingredients_data``.
        4. ``btn_reset`` ("Reset Filters") on the left and
           ``btn_apply`` ("Apply Filters") on the right, connected to
           ``reset_filters`` and ``emit_filter`` respectively.
        """
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
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
        self.txt_model.setObjectName("filterModelInput")
        self.txt_model.setPlaceholderText("Name contains...")
        model_layout.addWidget(self.txt_model)
        row1_layout.addWidget(grp_model)

        layout.addLayout(row1_layout)

        # Temperature Range
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

        self.range_slider.rangeChanged.connect(self._on_slider_changed)
        self.spin_temp_min.valueChanged.connect(self._on_spin_changed)
        self.spin_temp_max.valueChanged.connect(self._on_spin_changed)

        temp_layout.addWidget(self.spin_temp_min)
        temp_layout.addWidget(self.range_slider)
        temp_layout.addWidget(self.spin_temp_max)

        layout.addWidget(grp_temp)

        # Ingredients
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

        # Footer
        footer_layout = QtWidgets.QHBoxLayout()

        self.btn_reset = QtWidgets.QPushButton("Reset Filters")
        self.btn_reset.setObjectName("btnResetFilters")
        self.btn_reset.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_reset.clicked.connect(self.reset_filters)

        self.btn_apply = QtWidgets.QPushButton("Apply Filters")
        self.btn_apply.setObjectName("btnApplyFilters")
        self.btn_apply.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_apply.clicked.connect(self.emit_filter)

        footer_layout.addWidget(self.btn_reset)
        footer_layout.addStretch()
        footer_layout.addWidget(self.btn_apply)

        layout.addLayout(footer_layout)

    def emit_filter(self):
        """Collect all current filter values and emit ``filter_changed``.

        Builds the filter dict by reading every control's current state.
        Ingredient entries are only included in the ``"ingredients"`` sub-dict
        when the user has deselected at least one item (i.e. the selection is
        not "all selected"), which simplifies the ``is_default`` check in
        consuming code.

        Emits:
            filter_changed (dict): A snapshot of all filter criteria with the
                schema described in the module docstring.
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
            if len(selected_items) < total_items:
                filters["ingredients"][ing_type] = selected_items

        self.filter_changed.emit(filters)

    def _on_slider_changed(self, low, high):
        """Synchronise the temperature spin boxes when the range slider moves.

        Blocks both spin boxes' signals during the update to prevent the
        reciprocal ``_on_spin_changed`` slot from firing and causing a
        feedback loop.

        Args:
            low (float): New lower handle value emitted by ``RangeSlider``.
            high (float): New upper handle value emitted by ``RangeSlider``.
        """
        self.spin_temp_min.blockSignals(True)
        self.spin_temp_max.blockSignals(True)
        self.spin_temp_min.setValue(low)
        self.spin_temp_max.setValue(high)
        self.spin_temp_min.blockSignals(False)
        self.spin_temp_max.blockSignals(False)

    def _on_spin_changed(self):
        """Synchronise the range slider when either temperature spin box changes.

        Reads the current values of ``spin_temp_min`` and ``spin_temp_max``,
        clamps ``low`` to not exceed ``high`` (correcting ``spin_temp_min`` in
        place if needed), then pushes both values to ``range_slider`` via
        ``setValues``.
        """
        low = self.spin_temp_min.value()
        high = self.spin_temp_max.value()

        if low > high:
            low = high
            self.spin_temp_min.setValue(low)

        self.range_slider.setValues(low, high)

    def reset_filters(self):
        """Reset every filter control to its default state and emit the update.

        Blocks the widget's own signals during the reset so that intermediate
        control changes do not trigger partial ``filter_changed`` emissions.
        After all controls are restored to their defaults
        (both state checks checked, model text cleared, temperature range
        0 - 100 °C, all ingredient buttons fully selected), signals are
        unblocked and ``emit_filter`` is called once to notify the parent of
        the clean state.
        """
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
        """Return a complete snapshot of the current filter control values.

        Unlike ``emit_filter``, this method does *not* apply the "omit if all
        selected" optimisation for ingredients — every ingredient type is
        always present in the returned dict, making it suitable for
        persistence or comparison use cases.

        Returns:
            dict: Current filter state with keys ``show_measured``,
                ``show_predicted``, ``model_text``, ``temp_min``,
                ``temp_max``, and ``ingredients`` (a dict mapping every
                ingredient category name to its full list of selected items).
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
        """Check whether all filter controls are at their default (unfiltered) values.

        Inspects the current state via ``get_filter_state`` and the
        ``ing_buttons`` map.  The filter is considered default when both state
        check boxes are checked, the model text is empty, the temperature range
        is exactly 0 - 100 °C, and every ingredient button has all of its
        items selected.

        Returns:
            bool: ``True`` if every filter is at its default value and no
                records would be excluded; ``False`` if any filter is active.
        """
        state = self.get_filter_state()
        if not (state["show_measured"] and state["show_predicted"]):
            return False
        if state["model_text"]:
            return False
        if state["temp_min"] != 0 or state["temp_max"] != 100:
            return False
        for _, btn in self.ing_buttons.items():
            total_items = len(btn.items_map)
            selected_items = btn.get_selected_items()
            if len(selected_items) < total_items:
                return False

        return True
