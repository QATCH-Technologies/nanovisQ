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


TAG = "[ViscosityUI]"


class PredictionConfigCard(QtWidgets.QFrame):
    removed = QtCore.Signal(object)
    run_requested = QtCore.Signal(dict)
    save_requested = QtCore.Signal(dict)
    expanded = QtCore.Signal(object)

    def __init__(
        self,
        default_name,
        ingredients_data,
        ingredient_types,
        ingredient_units,
        parent=None,
    ):
        super().__init__(parent)

        # Native frame styling
        self.setFrameStyle(QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Raised)
        self.setLineWidth(1)

        self.ingredients_master = ingredients_data
        self.INGREDIENT_TYPES = ingredient_types
        self.INGREDIENT_UNITS = ingredient_units

        self.active_ingredients = {}
        self.is_expanded = True
        self.is_measured = False
        self.notes_visible = False
        self.debounce_timer = QtCore.QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(300)
        self.debounce_timer.timeout.connect(self.emit_run_request)
        self._init_ui(default_name)
        self._connect_auto_updates()

    def _init_ui(self, default_name):
        layout = QtWidgets.QVBoxLayout(self)

        # --- 1. Header ---
        self.header_frame = QtWidgets.QFrame()
        header_layout = QtWidgets.QHBoxLayout(self.header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)

        # Toggle Arrow
        self.btn_toggle = QtWidgets.QToolButton()
        self.btn_toggle.setArrowType(Qt.ArrowType.DownArrow)
        self.btn_toggle.clicked.connect(self.toggle_content)

        # Name Input
        self.name_input = QtWidgets.QLineEdit(default_name)
        self.name_input.setPlaceholderText("Prediction Name")

        # Measured Badge
        self.lbl_measured = QtWidgets.QLabel("✓ Measured Data")
        self.lbl_measured.setStyleSheet(
            "color: #2E7D32; font-weight: bold; border: 1px solid #2E7D32; border-radius: 4px; padding: 2px 6px; background-color: #E8F5E9;"
        )
        self.lbl_measured.setVisible(False)

        # Delete Button
        self.btn_delete = QtWidgets.QPushButton("Delete")
        self.btn_delete.clicked.connect(lambda: self.removed.emit(self))

        header_layout.addWidget(self.btn_toggle)
        header_layout.addWidget(self.name_input, stretch=1)
        header_layout.addWidget(self.lbl_measured)
        header_layout.addWidget(self.btn_delete)
        layout.addWidget(self.header_frame)

        # --- 2. Content Body ---
        self.content_frame = QtWidgets.QFrame()
        content_layout = QtWidgets.QVBoxLayout(self.content_frame)

        # Model Selection
        model_layout = QtWidgets.QHBoxLayout()
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["VisQAI-ICL_v1_nightly", "VisQAI-ICL_v2_beta"])
        btn_browse = QtWidgets.QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_model_file)

        model_layout.addWidget(QtWidgets.QLabel("Model:"))
        model_layout.addWidget(self.model_combo, stretch=1)
        model_layout.addWidget(btn_browse)
        content_layout.addLayout(model_layout)

        self._add_divider(content_layout)

        # --- INGREDIENTS SECTION (With Help) ---
        self._add_header_with_help(
            content_layout,
            "Formulation Composition",
            "Define the chemical makeup of the sample.\n\n"
            "Select a component type (e.g., Buffer, Salt) and specific molecule.\n"
            "Units are automatically assigned based on component type.",
        )

        self.ing_container_layout = QtWidgets.QVBoxLayout()
        content_layout.addLayout(self.ing_container_layout)

        self.btn_add_ing = QtWidgets.QPushButton("Add Component")
        self.btn_add_ing.clicked.connect(self.show_add_menu)
        content_layout.addWidget(self.btn_add_ing)

        self._add_divider(content_layout)

        # --- ENVIRONMENT SECTION (With Help) ---
        self._add_header_with_help(
            content_layout,
            "Environment",
            "Set the physical conditions for the simulation.\n\n"
            "Temperature affects viscosity predictions significantly.",
        )

        temp_layout = QtWidgets.QHBoxLayout()
        temp_layout.addWidget(QtWidgets.QLabel("Temperature:"))

        self.slider_temp = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider_temp.setRange(0, 100)
        self.slider_temp.setValue(25)

        self.spin_temp = QtWidgets.QDoubleSpinBox()
        self.spin_temp.setRange(0.0, 100.0)
        self.spin_temp.setValue(25.0)
        self.spin_temp.setSuffix(" °C")
        self.spin_temp.setFixedWidth(80)

        self.slider_temp.valueChanged.connect(
            lambda v: self.spin_temp.setValue(float(v))
        )
        self.spin_temp.valueChanged.connect(lambda v: self.slider_temp.setValue(int(v)))

        temp_layout.addWidget(self.slider_temp)
        temp_layout.addWidget(self.spin_temp)
        content_layout.addLayout(temp_layout)

        self._add_divider(content_layout)

        # --- ML PARAMS SECTION (With Help) ---
        self._add_header_with_help(
            content_layout,
            "ML Hyperparameters",
            "Tune the inference engine parameters:\n\n"
            "• Learning Rate: Controls step size during optimization.\n"
            "• Steps: Number of inference iterations.\n"
            "• Confidence Interval: The statistical certainty range (95% is standard).",
        )

        params_grid = QtWidgets.QGridLayout()
        params_grid.setVerticalSpacing(10)

        params_grid.addWidget(QtWidgets.QLabel("Learning Rate:"), 0, 0)
        self.spin_lr = QtWidgets.QDoubleSpinBox()
        self.spin_lr.setRange(0.0001, 1.0)
        self.spin_lr.setSingleStep(0.001)
        self.spin_lr.setDecimals(4)
        self.spin_lr.setValue(0.01)
        params_grid.addWidget(self.spin_lr, 0, 1)

        params_grid.addWidget(QtWidgets.QLabel("Steps:"), 1, 0)
        self.spin_steps = QtWidgets.QSpinBox()
        self.spin_steps.setRange(1, 10000)
        self.spin_steps.setSingleStep(10)
        self.spin_steps.setValue(50)
        params_grid.addWidget(self.spin_steps, 1, 1)

        params_grid.addWidget(QtWidgets.QLabel("Confidence Interval:"), 2, 0)
        ci_container = QtWidgets.QWidget()
        ci_layout = QtWidgets.QHBoxLayout(ci_container)
        ci_layout.setContentsMargins(0, 0, 0, 0)
        self.slider_ci = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider_ci.setRange(0, 100)
        self.slider_ci.setValue(95)
        self.lbl_ci_val = QtWidgets.QLabel("95%")
        self.lbl_ci_val.setFixedWidth(40)
        self.slider_ci.valueChanged.connect(lambda v: self.lbl_ci_val.setText(f"{v}%"))
        ci_layout.addWidget(self.slider_ci)
        ci_layout.addWidget(self.lbl_ci_val)
        params_grid.addWidget(ci_container, 2, 1)

        content_layout.addLayout(params_grid)

        self._add_divider(content_layout)

        # --- NOTES & ACTIONS ---
        self.btn_toggle_notes = QtWidgets.QPushButton("Show Notes")
        self.btn_toggle_notes.setCheckable(True)
        self.btn_toggle_notes.setStyleSheet(
            "text-align: left; padding-left: 0px; border: none; color: #0078D7;"
        )
        self.btn_toggle_notes.clicked.connect(self.toggle_notes)

        self.notes_edit = QtWidgets.QTextEdit()
        self.notes_edit.setPlaceholderText("Enter notes about this run...")
        self.notes_edit.setFixedHeight(80)
        self.notes_edit.setVisible(False)

        content_layout.addWidget(self.btn_toggle_notes)
        content_layout.addWidget(self.notes_edit)

        self._add_divider(content_layout)

        action_layout = QtWidgets.QHBoxLayout()
        content_layout.addLayout(action_layout)

        layout.addWidget(self.content_frame)

    def _connect_auto_updates(self):
        # Text Inputs
        self.name_input.textChanged.connect(self.trigger_update)
        self.model_combo.currentTextChanged.connect(self.trigger_update)

        # Spinners & Sliders
        self.spin_temp.valueChanged.connect(self.trigger_update)
        self.spin_lr.valueChanged.connect(self.trigger_update)
        self.spin_steps.valueChanged.connect(self.trigger_update)
        self.slider_ci.valueChanged.connect(self.trigger_update)

    def set_measured_state(self, is_measured: bool):
        """
        Updates the UI to reflect measured state.

        Locked (Immutable): Formulation (Ingredients), Environment (Temp), Name
        Mutable (Editable): ML Hyperparams, Model, Notes
        """
        self.is_measured = is_measured
        self.lbl_measured.setVisible(is_measured)

        # 1. Visual Style
        if is_measured:
            self.setStyleSheet(
                """
                PredictionConfigCard {
                    background-color: #F4F7F6; 
                    border: 1px solid #2E7D32;
                }
            """
            )
        else:
            self.setStyleSheet("")

        # 2. Immutable Sections (Locked when measured)
        # We disable interactions or set ReadOnly for these specific fields
        lock_state = is_measured

        # Name (Identity)
        self.name_input.setReadOnly(lock_state)

        # Formulation (Ingredients)
        self.btn_add_ing.setEnabled(not lock_state)  # Cannot add new
        for combo, spin in self.active_ingredients.values():
            combo.setEnabled(not lock_state)  # Cannot change type
            spin.setReadOnly(lock_state)  # Cannot change concentration

        # Environment (Temperature)
        self.slider_temp.setEnabled(not lock_state)
        self.spin_temp.setReadOnly(lock_state)

        # 3. Mutable Sections (Always Editable)
        # We explicitly ensure these are enabled, even if is_measured is True
        self.model_combo.setEnabled(True)

        # ML Params
        self.spin_lr.setReadOnly(False)
        self.spin_steps.setReadOnly(False)
        self.slider_ci.setEnabled(True)

        # Notes
        self.btn_toggle_notes.setEnabled(True)
        self.notes_edit.setReadOnly(False)

    # --- NEW HELPER METHOD FOR HELP HEADERS ---
    def _add_header_with_help(self, layout, title, help_text):
        """Creates a bold header label with a small (?) info button."""
        container = QtWidgets.QWidget()
        h_layout = QtWidgets.QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(5)

        # Label
        lbl = QtWidgets.QLabel(title)
        font = lbl.font()
        font.setBold(True)
        lbl.setFont(font)

        # Help Button
        btn_help = QtWidgets.QToolButton()
        btn_help.setText("?")
        btn_help.setCursor(Qt.PointingHandCursor)
        # Style it to look like a small circular badge or simple link
        btn_help.setStyleSheet(
            "QToolButton { border: 1px solid #CCC; border-radius: 10px; font-weight: bold; color: #555; background-color: #EEE; min-width: 20px; min-height: 20px; } QToolButton:hover { background-color: #DDD; color: #000; }"
        )

        btn_help.clicked.connect(
            lambda: QtWidgets.QMessageBox.information(self, title, help_text)
        )

        h_layout.addWidget(lbl)
        h_layout.addWidget(btn_help)
        h_layout.addStretch()  # Push everything to the left

        layout.addWidget(container)

    def toggle_notes(self):
        self.notes_visible = not self.notes_visible
        self.notes_edit.setVisible(self.notes_visible)
        text = "Hide Notes" if self.notes_visible else "Show Notes"
        self.btn_toggle_notes.setText(text)

    def on_save_clicked(self):
        config = self.get_configuration()
        self.save_requested.emit(config)

    def set_measured(self, is_measured: bool):
        self.is_measured = is_measured
        self.lbl_measured.setVisible(is_measured)

    def on_run_clicked(self):
        config = self.get_configuration()
        self.run_requested.emit(config)

    def toggle_content(self):
        if not self.is_expanded:
            self.expanded.emit(self)
            self.emit_run_request()

        self.is_expanded = not self.is_expanded
        self.content_frame.setVisible(self.is_expanded)
        arrow = Qt.ArrowType.DownArrow if self.is_expanded else Qt.ArrowType.RightArrow
        self.btn_toggle.setArrowType(arrow)

    def collapse(self):
        if self.is_expanded:
            self.is_expanded = False
            self.content_frame.setVisible(False)
            self.btn_toggle.setArrowType(Qt.ArrowType.RightArrow)

    def browse_model_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Model", "", "Model Files (*.pt *.onnx)"
        )
        if fname:
            self.model_combo.addItem(fname)
            self.model_combo.setCurrentText(fname)

    def _add_divider(self, layout):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(line)

    def show_add_menu(self):
        menu = QtWidgets.QMenu(self.btn_add_ing)
        present = self.active_ingredients.keys()
        available = [t for t in self.INGREDIENT_TYPES if t not in present]
        if not available:
            menu.addAction("All types added").setEnabled(False)
        else:
            for t in available:
                action = menu.addAction(t)
                action.triggered.connect(
                    lambda checked, _t=t: self.add_ingredient_row(_t)
                )
        menu.exec(
            self.btn_add_ing.mapToGlobal(QtCore.QPoint(0, self.btn_add_ing.height()))
        )

    def add_ingredient_row(self, ing_type):
        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        lbl = QtWidgets.QLabel(f"{ing_type}:")
        lbl.setFixedWidth(80)

        combo = QtWidgets.QComboBox()
        items = self.ingredients_master.get(ing_type, [])
        combo.addItem("None", None)
        for item in items:
            combo.addItem(item.name, item)

        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0, 1000)
        spin.setSingleStep(1.0)
        spin.setSuffix(f" {self.INGREDIENT_UNITS.get(ing_type,'')}")
        spin.setFixedWidth(90)
        combo.currentTextChanged.connect(self.trigger_update)
        spin.valueChanged.connect(self.trigger_update)
        btn_rem = QtWidgets.QPushButton("Remove")
        btn_rem.clicked.connect(
            lambda: self.remove_ingredient_row(ing_type, row_widget)
        )

        row_layout.addWidget(lbl)
        row_layout.addWidget(combo, stretch=1)
        row_layout.addWidget(spin)
        row_layout.addWidget(btn_rem)

        self.ing_container_layout.addWidget(row_widget)
        self.active_ingredients[ing_type] = (combo, spin)

    def trigger_update(self):
        """Called whenever a field changes. Restarts the debounce timer."""
        # Only run if we are actually expanded (don't run background calculations for hidden cards)
        if self.is_expanded:
            self.debounce_timer.start()

    def emit_run_request(self):
        config = self.get_configuration()
        self.run_requested.emit(config)

    def remove_ingredient_row(self, ing_type, widget):
        if ing_type in self.active_ingredients:
            del self.active_ingredients[ing_type]
        widget.deleteLater()

    def get_configuration(self):
        formulation = {}
        for t, (combo, spin) in self.active_ingredients.items():
            formulation[t] = {
                "component": combo.currentText(),
                "concentration": spin.value(),
            }
        return {
            "name": self.name_input.text(),
            "model": self.model_combo.currentText(),
            "temp": self.spin_temp.value(),
            "lr": self.spin_lr.value(),
            "steps": self.spin_steps.value(),
            "ci": self.slider_ci.value(),
            "formulation": formulation,
            "measured": self.is_measured,
            "notes": self.notes_edit.toPlainText(),
        }

    def load_data(self, data):
        if "name" in data:
            self.name_input.setText(data["name"])
        if "model" in data:
            idx = self.model_combo.findText(data["model"])
            if idx == -1:
                self.model_combo.addItem(data["model"])
                self.model_combo.setCurrentText(data["model"])
            else:
                self.model_combo.setCurrentIndex(idx)
        if "temp" in data:
            self.spin_temp.setValue(float(data["temp"]))
        if "lr" in data:
            self.spin_lr.setValue(float(data["lr"]))
        if "steps" in data:
            self.spin_steps.setValue(int(data["steps"]))
        if "ci" in data:
            self.slider_ci.setValue(int(data["ci"]))
        if "measured" in data:
            self.set_measured(data["measured"])
        if "notes" in data:
            self.notes_edit.setText(data["notes"])
            if data["notes"].strip():
                self.toggle_notes()

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
        self._load_mock_data()
        self.init_ui()

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
        splitter = QtWidgets.QSplitter(Qt.Horizontal)

        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton("New Prediction")
        btn_add.clicked.connect(lambda: self.add_prediction_card(None))
        btn_import = QtWidgets.QPushButton("Import Data...")
        btn_import.clicked.connect(self.import_data_file)
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_import)
        left_layout.addLayout(btn_layout)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.cards_container = QtWidgets.QWidget()
        self.cards_layout = QtWidgets.QVBoxLayout(self.cards_container)
        self.cards_layout.addStretch()
        scroll.setWidget(self.cards_container)
        left_layout.addWidget(scroll)

        splitter.addWidget(left_widget)

        self.viz_panel = VisualizationPanel()
        splitter.addWidget(self.viz_panel)

        splitter.setSizes([450, 700])
        main_layout.addWidget(splitter)
        self.add_prediction_card()

    def add_prediction_card(self, data=None):
        # 1. Determine name
        if data and "name" in data:
            name = data["name"]
        else:
            name = f"Prediction {self.cards_layout.count()}"

        # 2. Create the card
        card = PredictionConfigCard(
            name, self.ingredients_by_type, self.INGREDIENT_TYPES, self.INGREDIENT_UNITS
        )

        # 3. [NEW] Check for 'measured' flag in data and update card
        if data and data.get("measured", False):
            card.set_measured_state(True)

        # ... existing signal connections (removed, run_requested, expanded, etc.) ...
        card.removed.connect(self.remove_card)
        card.run_requested.connect(self.run_prediction)
        card.expanded.connect(self.on_card_expanded)

        # 4. Add to layout
        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)

        # 5. Collapse others
        self.on_card_expanded(card)

        if data:
            card.load_data(data)
            # If it's a measured run, set state
            if data.get("measured"):
                card.set_measured_state(True)
            # Trigger initial run for this data
            card.emit_run_request()

    def on_card_expanded(self, active_card):
        """[NEW] Slot to collapse all cards except the active one."""
        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()

            # Check if it's a card and NOT the one currently expanding
            if isinstance(widget, PredictionConfigCard) and widget is not active_card:
                widget.collapse()

    def remove_card(self, card_widget):
        self.cards_layout.removeWidget(card_widget)
        card_widget.deleteLater()

    def handle_single_run(self, config):
        self.run_prediction(config)

    def handle_save(self, config):
        pass

    def run_all_predictions(self):
        self.run_prediction()

    def import_data_file(self):
        # Mock Import with Measured Data
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Run Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not fname:
            return

        # Simulating an imported config that HAS measured data
        imported_config = {
            "name": "Imported: Lab Run 23",
            "measured": True,
            # In a real app, you would parse 'fname' here to get actual data
        }

        # Add the card (this sets the UI state)
        self.add_prediction_card(imported_config)

        # Immediately run/display this imported data to show the feature
        self.run_prediction(imported_config)

    def run_prediction(self, config=None):
        """
        Runs the prediction.
        Note: In a real app, this should be done in a background thread
        (QThread) to keep the UI responsive.
        """
        if config and "name" in config:
            self.viz_panel.set_plot_title(config["name"])
        else:
            self.viz_panel.set_plot_title("Unknown Sample")

        # 1. Start Visuals
        self.viz_panel.show_loading()

        # 2. Schedule logic
        QtCore.QTimer.singleShot(100, lambda: self._execute_prediction_logic(config))

    def _execute_prediction_logic(self, config):
        """Actual logic execution (formerly inside run_prediction)."""

        # Simulate calculation time (now safe to sleep briefly for demo effect,
        # though in production you'd use a QThread)
        import time

        time.sleep(0.5)

        print("Auto-Running Prediction...")
        shear_rates = np.logspace(1, 6, 20)
        K = 50
        n = 0.6

        if config:
            temp_mod = float(config.get("temp", 25)) / 25.0
            K = K / temp_mod

        viscosity = K * np.power(shear_rates, n - 1)

        measured_y = None
        if config and config.get("measured", False):
            measured_y = viscosity + np.random.normal(0, 2, len(shear_rates))

        data_package = {
            "x": shear_rates,
            "y": viscosity,
            "upper": viscosity * 1.15,
            "lower": viscosity * 0.85,
            "measured_y": measured_y,
        }

        # 3. Finish
        self.viz_panel.set_data(data_package)
        self.viz_panel.hide_loading()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = PredictionUI()
    win.setWindowTitle("Viscosity AI - Hyperparameter Tuning")
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())
