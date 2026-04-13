"""
formulation_config_card.py

Widget module for the FormulationConfigCard UI component.

Provides a collapsible card widget used to configure biopharmaceutical
formulations for viscosity prediction. Each card manages ingredient
composition, environmental conditions (temperature), model selection,
and result export within the VisQAI application.
"""

import csv
import glob
import os
import shutil

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    TAG = ["FormulationConfigCard (HEADLESS)"]

    class Log:
        @staticmethod
        def d(TAG, msg=""):
            print("DEBUG:", TAG, msg)

        @staticmethod
        def i(TAG, msg=""):
            print("INFO:", TAG, msg)

        @staticmethod
        def w(TAG, msg=""):
            print("WARNING:", TAG, msg)

        @staticmethod
        def e(TAG, msg=""):
            print("ERROR:", TAG, msg)

    from architecture import Architecture
    from components.drag_handle import DragHandle
    from dialogs.generic_ingredient_dialog import GenericIngredientDialog
    from dialogs.model_options_dialog import ModelOptionsDialog
    from dialogs.protein_config_dialog import ProteinConfigDialog
    from src.models.formulation import Formulation, ViscosityProfile
    from src.models.ingredient import (
        Buffer,
        Excipient,
        Ingredient,
        Protein,
        Salt,
        Stabilizer,
        Surfactant,
    )
    from src.view.dialogs.model_selection_dialog import (
        ModelSelectionDialog,
    )
    from styles.style_loader import load_stylesheet

except ImportError:
    TAG = "[FormulationConfigCard]"
    from QATCH.common.architecture import Architecture
    from QATCH.VisQAI.src.models.formulation import Formulation, ViscosityProfile
    from QATCH.VisQAI.src.models.ingredient import (
        Buffer,
        Excipient,
        Ingredient,
        Protein,
        Salt,
        Stabilizer,
        Surfactant,
    )
    from QATCH.VisQAI.src.view.components.drag_handle import DragHandle
    from QATCH.VisQAI.src.view.dialogs.generic_ingredient_dialog import (
        GenericIngredientDialog,
    )
    from QATCH.VisQAI.src.view.dialogs.model_options_dialog import ModelOptionsDialog
    from QATCH.VisQAI.src.view.dialogs.model_selection_dialog import (
        ModelSelectionDialog,
    )
    from QATCH.VisQAI.src.view.dialogs.protein_config_dialog import ProteinConfigDialog
    from QATCH.VisQAI.src.view.styles.style_loader import load_stylesheet


class FormulationConfigCard(QtWidgets.QFrame):
    """A collapsible card widget representing a single formulation configuration.

    Displays controls for selecting a viscosity prediction model, composing a
    biopharmaceutical formulation (protein, buffer, excipients, etc.), setting
    temperature, and viewing or exporting prediction results. Cards can be
    marked as measured (imported data) or predicted, and support accordion-style
    expand/collapse animations.

    Signals:
        removed (object): Emitted when the user clicks the delete button. Carries
            the card instance.
        run_requested (dict): Emitted (via debounce) when the card configuration
            changes and a new prediction run should be triggered.
        save_requested (dict): Emitted when a save operation is requested.
        expanded (object): Emitted when the card is expanded. Carries the card
            instance.
        selection_changed (bool): Emitted when the card's selection state toggles.
        color_changed (str): Emitted when the user picks a new plot colour.
        visibility_toggled (object): Emitted when the hide-from-plot action is
            toggled. Carries the card instance.
    """

    removed = QtCore.pyqtSignal(object)
    run_requested = QtCore.pyqtSignal(dict)
    save_requested = QtCore.pyqtSignal(dict)
    expanded = QtCore.pyqtSignal(object)
    selection_changed = QtCore.pyqtSignal(bool)
    color_changed = QtCore.pyqtSignal(str)
    visibility_toggled = QtCore.pyqtSignal(object)
    _card_counter = 0

    def __init__(
        self,
        default_name,
        ingredients_data,
        ingredient_types,
        ingredient_units,
        parent=None,
    ):
        """Initialize a new FormulationConfigCard.

        Args:
            default_name (str): The initial display name shown in the card header.
            ingredients_data (dict): Mapping of ingredient type strings (e.g.
                ``"Protein"``) to lists of ingredient objects available for
                selection.
            ingredient_types (list[str]): Ordered list of ingredient type names
                that the card supports (e.g. ``["Protein", "Buffer", ...]``).
            ingredient_units (dict): Mapping of ingredient type strings to their
                concentration unit labels (e.g. ``{"Protein": "mg/mL"}``).
            parent (QtWidgets.QWidget, optional): Parent widget. Defaults to None.
        """
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
        self._buffer_ph_spin = None
        self.ml_params = {
            "lr": 0.01,
            "steps": 50,
            "ci": 95,
            "icl_filter": {"logic": "AND", "fields": ["Protein_type"]},
        }
        self.use_in_icl = True
        self.last_results = None
        self.is_expanded = True
        self.is_measured = False
        self.is_optimized = False
        self.notes_visible = False
        self.is_selectable = False
        self.is_selected = False
        self.plot_color = self._generate_auto_color()
        FormulationConfigCard._card_counter += 1

        # Internal Formulation Object
        self.formulation = Formulation()

        self.debounce_timer = QtCore.QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(300)
        self.debounce_timer.timeout.connect(self.emit_run_request)
        self._loading = False
        self.setStyleSheet(load_stylesheet())
        self._init_ui(default_name)
        self._connect_auto_updates()

        # Initialize internal formulation state
        self._update_internal_formulation()

    @property
    def controller(self):
        """Walk up the widget hierarchy to find the ingredient controller.

        Returns:
            The first ancestor widget that exposes an ``ing_ctrl`` attribute,
            or ``None`` if no such ancestor exists.
        """
        widget = self.parent()
        while widget is not None:
            if hasattr(widget, "ing_ctrl"):
                return widget.ing_ctrl
            widget = widget.parent()
        return None

    @property
    def formulation_controller(self):
        """Walk up the widget hierarchy to find the formulation controller.

        Returns:
            The first ancestor widget that exposes a ``form_ctrl`` attribute,
            or ``None`` if no such ancestor exists.
        """
        widget = self.parent()
        while widget is not None:
            if hasattr(widget, "form_ctrl"):
                return widget.form_ctrl
            widget = widget.parent()
        return None

    def _init_ui(self, default_name):
        """Build and lay out all child widgets for the card.

        Constructs the drag handle, header (name input, badges, action buttons),
        content body (model selector, ingredient rows, temperature controls,
        notes), and footer (expand/collapse toggle).

        Args:
            default_name (str): Initial text to pre-populate the name input field.
        """
        root_layout = QtWidgets.QHBoxLayout(self)
        root_layout.setContentsMargins(5, 5, 15, 5)
        root_layout.setSpacing(5)

        # Left Drag Handle
        self.drag_handle = DragHandle()
        root_layout.addWidget(self.drag_handle, 0, QtCore.Qt.AlignmentFlag.AlignTop)

        # Central Container
        self.center_widget = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(self.center_widget)
        center_layout.setContentsMargins(0, 10, 0, 5)
        center_layout.setSpacing(10)
        center_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        root_layout.addWidget(self.center_widget, stretch=1)

        # Header Section
        self.header_frame = QtWidgets.QFrame()
        header_layout = QtWidgets.QHBoxLayout(self.header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        self.color_swatch = QtWidgets.QFrame()
        self.color_swatch.setFixedSize(6, 24)
        self.color_swatch.setStyleSheet(f"background-color: {self.plot_color}; border-radius: 3px;")
        self.color_swatch.setToolTip("Plot Color")
        header_layout.addWidget(self.color_swatch)

        self.name_input = QtWidgets.QLineEdit(default_name)
        self.name_input.setPlaceholderText("Prediction Name")
        self.name_input.setProperty("class", "title-input")

        self.lbl_measured = QtWidgets.QLabel("\u2713 Measured Data")
        self.lbl_measured.setProperty("class", "badge-success")
        self.lbl_measured.setVisible(False)

        # Optimized badge
        self.lbl_optimized = QtWidgets.QLabel("\u26a1 Optimized")
        self.lbl_optimized.setProperty("class", "badge-optimized")
        self.lbl_optimized.setVisible(False)

        # Warning Indicator for Incomplete Data
        self.lbl_warning = QtWidgets.QLabel("\u26a0 Incomplete")
        self.lbl_warning.setStyleSheet(
            "color: #c62828; background-color: #ffcdd2; "
            "border-radius: 4px; padding: 4px 8px; font-size: 11px; font-weight: bold;"
        )
        self.lbl_warning.setVisible(False)
        self.lbl_warning.setToolTip(
            "Missing required ingredient information or valid concentrations. Inference is blocked."
        )

        # Delete Button
        self.btn_delete = QtWidgets.QPushButton()
        self.btn_delete.setObjectName("btnCardDelete")
        self.btn_delete.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "delete-2-svgrepo-com.svg",
                )
            )
        )
        self.btn_delete.setFixedSize(32, 32)
        self.btn_delete.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_delete.setToolTip("Delete Prediction")
        self.btn_delete.clicked.connect(lambda: self.removed.emit(self))

        # Hamburger Menu
        self.btn_options = QtWidgets.QPushButton()
        self.btn_options.setObjectName("btnCardOptions")
        self.btn_options.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "options-svgrepo-com.svg",
                )
            )
        )
        self.btn_options.setFixedSize(32, 32)
        self.btn_options.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        self.options_menu = QtWidgets.QMenu(self)
        self.options_menu.addAction("Save model")
        self.options_menu.addAction("Save model as...")
        self.options_menu.addSeparator()
        self.act_hide_series = self.options_menu.addAction("Hide from Plot")
        self.act_hide_series.setCheckable(True)
        self.act_hide_series.setChecked(False)
        self.act_hide_series.toggled.connect(lambda _: self.visibility_toggled.emit(self))
        self.act_pick_color = self.options_menu.addAction("Select Plot Color...")
        self.act_pick_color.triggered.connect(self.select_plot_color)
        self.options_menu.addSeparator()
        self.options_menu.addAction("Export Formulation").triggered.connect(self.export_formulation)
        self.act_clear = self.options_menu.addAction("Clear Formulation")
        self.act_clear.triggered.connect(self.clear_formulation)
        self._update_clear_state()
        self.options_menu.addSeparator()
        self.act_use_icl = self.options_menu.addAction("Use in ICL")
        self.act_use_icl.setCheckable(True)
        self.act_use_icl.setEnabled(False)
        self.act_use_icl.toggled.connect(self.set_icl_usage)

        # Model Options Action
        self.act_model_opts = self.options_menu.addAction("Model Options")
        self.act_model_opts.triggered.connect(self.open_model_options)

        self.btn_options.setMenu(self.options_menu)

        header_layout.addWidget(self.name_input, stretch=1)
        header_layout.addWidget(self.lbl_warning)
        header_layout.addWidget(self.lbl_measured)
        header_layout.addWidget(self.lbl_optimized)
        header_layout.addWidget(self.btn_delete)
        header_layout.addWidget(self.btn_options)

        center_layout.addWidget(self.header_frame)

        # Content Body
        self.content_frame = QtWidgets.QFrame()
        self.content_frame.setAttribute(QtCore.Qt.WidgetAttribute.WA_LayoutUsesWidgetRect)
        self.content_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum
        )
        content_layout = QtWidgets.QVBoxLayout(self.content_frame)
        content_layout.setContentsMargins(5, 0, 5, 0)
        content_layout.setSpacing(15)
        content_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.assets_path = os.path.join(Architecture.get_path(), "QATCH", "VisQAI", "assets")
        if not os.path.exists(self.assets_path):
            os.makedirs(self.assets_path, exist_ok=True)

        # Model Selection
        model_layout = QtWidgets.QHBoxLayout()
        model_label = QtWidgets.QLabel("Model:")
        model_label.setObjectName("cardSectionLabel")
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setProperty("class", "sleek")
        self.model_combo.setToolTip("Select a prediction model from assets")
        self._populate_model_list()
        self.btn_select_model = QtWidgets.QPushButton()
        self.btn_select_model.setObjectName("btnBrowseModel")
        self.btn_select_model.setFixedWidth(40)
        self.btn_select_model.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_select_model.setToolTip("Import New Model (.visq)")
        self.btn_select_model.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "file-plus-2-svgrepo-com.svg",
                )
            )
        )
        self.btn_select_model.clicked.connect(self.browse_model_file)

        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo, stretch=1)
        model_layout.addWidget(self.btn_select_model)

        content_layout.addLayout(model_layout)

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
        self.btn_add_ing.setFixedWidth(140)
        self.btn_add_ing.setMinimumHeight(32)
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

        self.slider_temp.valueChanged.connect(lambda v: self.spin_temp.setValue(float(v)))
        self.spin_temp.valueChanged.connect(lambda v: self.slider_temp.setValue(int(v)))

        temp_layout.addWidget(self.slider_temp)
        temp_layout.addWidget(self.spin_temp)
        content_layout.addLayout(temp_layout)

        self._add_divider(content_layout)

        lbl_notes = QtWidgets.QLabel("Notes:")
        lbl_notes.setObjectName("cardSectionLabel")
        content_layout.addWidget(lbl_notes)

        # Add the text edit
        self.notes_edit = QtWidgets.QTextEdit()
        self.notes_edit.setObjectName("notesEdit")
        self.notes_edit.setPlaceholderText("Enter notes about this run...")
        self.notes_edit.setMaximumHeight(80)

        content_layout.addWidget(self.notes_edit)

        center_layout.addWidget(self.content_frame)

        # Footer
        self.footer_frame = QtWidgets.QFrame()
        footer_layout = QtWidgets.QHBoxLayout(self.footer_frame)
        footer_layout.setContentsMargins(0, 5, 0, 0)
        footer_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.btn_toggle = QtWidgets.QToolButton()
        self.btn_toggle.setObjectName("btnCardToggle")
        self.btn_toggle.setArrowType(QtCore.Qt.ArrowType.UpArrow)
        self.btn_toggle.clicked.connect(self.toggle_content)
        self.btn_toggle.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        footer_layout.addWidget(self.btn_toggle)
        center_layout.addWidget(self.footer_frame)
        self.add_ingredient_row("Buffer", deletable=False)

    def _populate_model_list(self):
        self.model_combo.clear()
        pattern = os.path.join(self.assets_path, "*.visq")
        files = glob.glob(pattern)
        files.sort()

        if not files:
            self.model_combo.addItem("No models found")
            self.model_combo.setEnabled(False)
        else:
            self.model_combo.setEnabled(True)

            for f in files:
                filename = os.path.basename(f)
                self.model_combo.addItem(filename, f)
            index = self.model_combo.findText("VisQAI(base).visq")
            if index >= 0:
                self.model_combo.setCurrentIndex(index)

    def _generate_auto_color(self):
        golden_angle = 137.508
        hue = (FormulationConfigCard._card_counter * golden_angle) % 360
        color = QtGui.QColor.fromHsvF(hue / 360.0, 0.75, 0.95)
        return color.name()

    def select_plot_color(self):
        current = QtGui.QColor(self.plot_color)
        color = QtWidgets.QColorDialog.getColor(current, self, "Select Plot Color")

        if color.isValid():
            self.plot_color = color.name()
            self.color_swatch.setStyleSheet(
                f"background-color: {self.plot_color}; border-radius: 3px;"
            )
            self.trigger_update()
            self.color_changed.emit(self.plot_color)

    def get_searchable_text(self):
        parts = [
            self.name_input.text(),
            self.model_combo.currentText(),
            self.notes_edit.toPlainText(),
            "Measured" if self.is_measured else "Predicted",
        ]
        for ing_type, (combo, spin, _, _) in self.active_ingredients.items():
            parts.append(ing_type)
            parts.append(combo.currentText())
            parts.append(str(spin.value()))

        return " ".join(parts).lower()

    def _update_clear_state(self):
        if not hasattr(self, "act_clear"):
            return
        if self.is_measured:
            self.act_clear.setEnabled(False)
            return
        has_ingredients = len(self.active_ingredients) > 0
        self.act_clear.setEnabled(has_ingredients)

    def open_model_options(self):
        dlg = ModelOptionsDialog(self.ml_params, self.is_measured, self)
        if dlg.exec_() == QtWidgets.QDialog.DialogCode.Accepted:
            new_settings = dlg.get_settings()
            self.ml_params.update(new_settings)
            self.trigger_update()

    def _connect_auto_updates(self):
        self.name_input.textChanged.connect(self.trigger_update)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.spin_temp.valueChanged.connect(self.trigger_update)

    def _on_model_changed(self, text):
        """Handle model selection changes."""
        self.trigger_update()
        self._save_metadata_to_db()

    def set_icl_usage(self, checked, save_db=True):
        self.use_in_icl = checked

        # Update the UI Action check state (in case this was called programmatically)
        if hasattr(self, "act_use_icl") and self.act_use_icl.isChecked() != checked:
            self.act_use_icl.blockSignals(True)
            self.act_use_icl.setChecked(checked)
            self.act_use_icl.blockSignals(False)

        self.trigger_update()  # Updates internal object

        if save_db:
            self._save_metadata_to_db()

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
        btn_help.clicked.connect(lambda: QtWidgets.QMessageBox.information(self, title, help_text))
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

        # Ingredients Locking
        for ing_type, (combo, spin, _, btn_rem) in self.active_ingredients.items():
            combo.setEnabled(not lock_state)
            spin.setReadOnly(lock_state)
            if ing_type == "Buffer":
                btn_rem.setEnabled(False)
                if self._buffer_ph_spin is not None:
                    self._buffer_ph_spin.setReadOnly(lock_state)
            else:
                btn_rem.setEnabled(not lock_state)

        self.slider_temp.setEnabled(not lock_state)
        self.spin_temp.setReadOnly(lock_state)
        self.notes_edit.setReadOnly(lock_state)

        # Model Selection
        if hasattr(self, "model_combo"):
            self.model_combo.setEnabled(True)
        if hasattr(self, "btn_select_model"):
            self.btn_select_model.setEnabled(True)

        # ICL Option: Enabled ONLY for Measured data; Disabled for Predictions
        if hasattr(self, "act_use_icl"):
            self.act_use_icl.setEnabled(is_measured)
            if not is_measured:
                self.act_use_icl.setChecked(False)

    def set_optimized_state(self, is_optimized: bool):
        self.is_optimized = is_optimized

        # Badge
        self.lbl_optimized.setVisible(is_optimized)
        self.setProperty("optimized", is_optimized)
        self.style().unpolish(self)
        self.style().polish(self)

        # Ingredient rows
        if hasattr(self, "btn_add_ing"):
            self.btn_add_ing.setEnabled(not is_optimized)
            self.btn_add_ing.setVisible(not is_optimized)
        for ing_type, (combo, spin, _, btn_rem) in self.active_ingredients.items():
            combo.setEnabled(not is_optimized)
            spin.setReadOnly(is_optimized)
            if is_optimized:
                btn_rem.setEnabled(False)
                if ing_type == "Buffer" and self._buffer_ph_spin is not None:
                    self._buffer_ph_spin.setReadOnly(True)
            else:
                btn_rem.setEnabled(ing_type != "Buffer")
                if ing_type == "Buffer" and self._buffer_ph_spin is not None:
                    self._buffer_ph_spin.setReadOnly(False)

        # Temperature controls
        if hasattr(self, "slider_temp"):
            self.slider_temp.setEnabled(not is_optimized)
        if hasattr(self, "spin_temp"):
            self.spin_temp.setReadOnly(is_optimized)

        # Model selection
        if hasattr(self, "model_combo"):
            self.model_combo.setEnabled(not is_optimized)
        if hasattr(self, "btn_select_model"):
            self.btn_select_model.setEnabled(not is_optimized)

        # Name and notes kept writable
        self.name_input.setReadOnly(False)
        self.name_input.setEnabled(True)
        if hasattr(self, "notes_edit"):
            self.notes_edit.setReadOnly(False)
            self.notes_edit.setEnabled(True)

    def toggle_content(self):
        if not self.is_expanded:
            self.expanded.emit(self)
            self.emit_run_request()

        self.is_expanded = not self.is_expanded
        arrow = QtCore.Qt.ArrowType.UpArrow if self.is_expanded else QtCore.Qt.ArrowType.DownArrow
        self.btn_toggle.setArrowType(arrow)

        if not hasattr(self, "_anim_accordion"):
            self._anim_accordion = QtCore.QPropertyAnimation(self.content_frame, b"maximumHeight")
            self._anim_accordion.setDuration(250)
            self._anim_accordion.setEasingCurve(QtCore.QEasingCurve.OutCubic)

        try:
            self._anim_accordion.finished.disconnect()
        except TypeError:
            pass

        if self.is_expanded:
            self.content_frame.setVisible(True)
            self.content_frame.setMaximumHeight(0)
            QtWidgets.QApplication.processEvents()
            target_height = self.content_frame.sizeHint().height()

            self._anim_accordion.setStartValue(0)
            self._anim_accordion.setEndValue(target_height)
            self._anim_accordion.finished.connect(
                lambda: self.content_frame.setMaximumHeight(16777215)
            )
        else:
            start_height = self.content_frame.height()
            self._anim_accordion.setStartValue(start_height)
            self._anim_accordion.setEndValue(0)

            def on_collapse_complete():
                self.content_frame.setVisible(False)
                self.content_frame.setMaximumHeight(0)

            self._anim_accordion.finished.connect(on_collapse_complete)

        self._anim_accordion.start()

    def matches_filter(self, filters):
        # State Filter
        if self.is_measured and not filters["show_measured"]:
            return False
        if not self.is_measured and not filters["show_predicted"]:
            return False

        # Model Name Filter
        if filters["model_text"]:
            txt = filters["model_text"]
            if (
                txt not in self.name_input.text().lower()
                and txt not in self.model_display.text().lower()
            ):
                return False

        # Temperature Filter
        current_temp = self.spin_temp.value()
        if not (filters["temp_min"] <= current_temp <= filters["temp_max"]):
            return False

        # Ingredient Composition Filter
        if filters["ingredients"]:
            for type_filter, allowed_names in filters["ingredients"].items():
                if not allowed_names:
                    if type_filter in self.active_ingredients:
                        return False
                    continue

                if type_filter in self.active_ingredients:
                    combo, _, _, _ = self.active_ingredients[type_filter]
                    current_name = combo.currentText()
                    if current_name not in allowed_names:
                        return False
                pass

        return True

    def collapse(self):
        if not self.is_expanded:
            return
        self.is_expanded = False
        self.btn_toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow)

        if not hasattr(self, "_anim_accordion"):
            self._anim_accordion = QtCore.QPropertyAnimation(self.content_frame, b"maximumHeight")
            self._anim_accordion.setDuration(250)
            self._anim_accordion.setEasingCurve(QtCore.QEasingCurve.OutCubic)

        try:
            self._anim_accordion.finished.disconnect()
        except TypeError:
            pass

        current_height = self.content_frame.height()
        self.content_frame.setMaximumHeight(current_height)

        self._anim_accordion.setStartValue(current_height)
        self._anim_accordion.setEndValue(0)

        def on_collapse_complete():
            self.content_frame.setMaximumHeight(0)
            self.content_frame.setVisible(False)

        self._anim_accordion.finished.connect(on_collapse_complete)
        self._anim_accordion.start()

    def set_results(self, data):
        self.last_results = data
        self._update_internal_formulation()

    def browse_model_file(self):
        fname = None
        try:
            model_dialog = ModelSelectionDialog()
            model_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            model_dialog.setNameFilter("VisQAI Models (*.visq)")
            model_dialog.setViewMode(QtWidgets.QFileDialog.Detail)
            model_path = os.path.join(Architecture.get_path(), "QATCH", "VisQAI", "assets")
            if os.path.exists(model_path):
                model_dialog.setDirectory(model_path)
            if model_dialog.exec_():
                selected_files = model_dialog.selectedFiles()
                if selected_files:
                    fname = selected_files[0]
        except Exception:
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Import Model File",
                "",
                "VisQAI Models (*.visq)",
            )
        if fname:
            try:
                filename = os.path.basename(fname)
                dest_path = os.path.join(self.assets_path, filename)
                if os.path.abspath(fname) != os.path.abspath(dest_path):
                    shutil.copy2(fname, dest_path)

                # Refresh list and select the new item
                self._populate_model_list()
                index = self.model_combo.findText(filename)
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)

                self.trigger_update()

                QtWidgets.QMessageBox.information(
                    self, "Import Successful", f"Model '{filename}' imported to assets."
                )

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Import Failed", f"Could not import model:\n{str(e)}"
                )

    def show_add_menu(self):
        menu = QtWidgets.QMenu(self)
        menu.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        menu.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, False)
        menu.setObjectName("cardComponentMenu")

        present_types = list(self.active_ingredients.keys())
        available = [t for t in self.INGREDIENT_TYPES if t not in present_types]

        if not available:
            action = menu.addAction("✓ All components added")
            action.setEnabled(False)
        else:
            header = menu.addAction("Select Component Type")
            header.setEnabled(False)
            menu.addSeparator()

            for t in available:
                action = menu.addAction(t)
                action.triggered.connect(
                    lambda checked, type_name=t: self.add_ingredient_row(type_name)
                )

        button_global_pos = self.btn_add_ing.mapToGlobal(QtCore.QPoint(0, 0))
        pos = QtCore.QPoint(
            button_global_pos.x(),
            button_global_pos.y() + self.btn_add_ing.height() + 4,
        )

        menu.exec(pos)

    def add_ingredient_row(self, ing_type, deletable=True):
        if ing_type in self.active_ingredients:
            return

        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        lbl = QtWidgets.QLabel(f"{ing_type}:")
        lbl.setFixedWidth(80)
        lbl.setObjectName("ingredientLabel")

        combo = QtWidgets.QComboBox()
        combo.setProperty("class", "sleek")
        items = self.ingredients_master.get(ing_type, [])

        for item in items:
            combo.addItem(item.name, item)

        btn_configure = QtWidgets.QPushButton()
        btn_configure.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn_configure.setToolTip(f"Edit {ing_type}")
        btn_configure.setProperty("mode", "edit")
        btn_configure.setObjectName("btnIngredientConfigure")
        btn_configure.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "pen-svgrepo-com.svg",
                )
            )
        )

        def update_configure_button():
            if combo.currentIndex() >= 0:
                btn_configure.setToolTip(f"Edit {combo.currentText()}")
                btn_configure.setProperty("mode", "edit")
                btn_configure.setIcon(
                    QtGui.QIcon(
                        os.path.join(
                            Architecture.get_path(),
                            "QATCH",
                            "VisQAI",
                            "src",
                            "view",
                            "icons",
                            "pen-svgrepo-com.svg",
                        )
                    )
                )
            else:
                btn_configure.setToolTip(f"Add new {ing_type}")
                btn_configure.setProperty("mode", "add")
                btn_configure.setIcon(
                    QtGui.QIcon(
                        os.path.join(
                            Architecture.get_path(),
                            "QATCH",
                            "VisQAI",
                            "src",
                            "view",
                            "icons",
                            "add-plus-svgrepo-com.svg",
                        )
                    )
                )

        btn_configure.setFixedSize(32, 32)
        combo.currentIndexChanged.connect(update_configure_button)
        btn_configure.clicked.connect(
            lambda: self.open_ingredient_config_dialog(ing_type, combo, btn_configure)
        )

        spin = QtWidgets.QDoubleSpinBox()
        spin.setProperty("class", "sleek")
        spin.setRange(0, 1000)
        spin.setSingleStep(1.0)

        suffix_str = f" {self.INGREDIENT_UNITS.get(ing_type, '')}"
        spin.setSuffix(suffix_str)
        fm = spin.fontMetrics()
        widest_suffix = max(
            (f" {u}" for u in self.INGREDIENT_UNITS.values()),
            key=lambda s: fm.horizontalAdvance(f"999.9{s}"),
        )
        conc_width = fm.horizontalAdvance(f"999.9{widest_suffix}") + 50
        spin.setFixedWidth(conc_width)

        combo.currentTextChanged.connect(self.trigger_update)
        spin.valueChanged.connect(self.trigger_update)

        # pH spinbox — Buffer rows only, replaces the delete button slot
        if ing_type == "Buffer":
            spin_ph = QtWidgets.QDoubleSpinBox()
            spin_ph.setProperty("class", "sleek")
            spin_ph.setRange(-1.0, 14.0)
            spin_ph.setValue(-1.0)
            spin_ph.setSpecialValueText("---")
            spin_ph.setDecimals(2)
            spin_ph.setSuffix(" pH")
            spin_ph.setToolTip("Per-formulation buffer pH")
            fm_ph = spin_ph.fontMetrics()
            ph_width = fm_ph.horizontalAdvance("14.00 pH")
            spin_ph.setMinimumWidth(ph_width + 40)
            spin_ph.setMaximumWidth(ph_width + 60)
            spin_ph.valueChanged.connect(self.trigger_update)
            self._buffer_ph_spin = spin_ph

        btn_rem = QtWidgets.QPushButton()
        btn_rem.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "delete-2-svgrepo-com.svg",
                )
            )
        )
        btn_rem.setFixedSize(32, 32)
        btn_rem.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn_rem.setToolTip("Remove Component")
        btn_rem.setObjectName("btnIngredientRemove")

        if not deletable:
            btn_rem.setEnabled(False)
            btn_rem.setVisible(False)  # Buffer: hide entirely, don't just disable

        btn_rem.clicked.connect(lambda: self.remove_ingredient_row(ing_type, row_widget))

        # Layout: Label | Combo | Conc | [pH] | Edit | [Delete]
        row_layout.addWidget(lbl)
        row_layout.addWidget(combo, stretch=1)
        row_layout.addWidget(spin)
        if ing_type == "Buffer":
            row_layout.addWidget(self._buffer_ph_spin)
        row_layout.addWidget(btn_configure)
        if ing_type != "Buffer":
            row_layout.addWidget(btn_rem)

        self.ing_container_layout.addWidget(row_widget)
        self.active_ingredients[ing_type] = (combo, spin, btn_configure, btn_rem)

        # Ensure button state is correct immediately after population
        update_configure_button()
        self._update_clear_state()

        # Flag button red if the currently-selected ingredient needs completion
        self._refresh_configure_button(ing_type, combo, btn_configure)

        # Re-check whenever the user picks a different ingredient from the combo
        combo.currentIndexChanged.connect(
            lambda _idx, t=ing_type, c=combo, b=btn_configure: (
                self._refresh_configure_button(t, c, b)
            )
        )

    def _create_ingredient_instance(self, ing_type, name, **kwargs):
        if ing_type == "Protein":
            return Protein(name, **kwargs)
        elif ing_type == "Buffer":
            return Buffer(name, **kwargs)
        elif ing_type == "Stabilizer":
            return Stabilizer(name, **kwargs)
        elif ing_type == "Surfactant":
            return Surfactant(name, **kwargs)
        elif ing_type == "Salt":
            return Salt(name, **kwargs)
        elif ing_type == "Excipient":
            return Excipient(name, **kwargs)
        else:

            class SimpleIngredient:
                def __init__(self, name, **k):
                    self.name = name
                    self.__dict__.update(k)

            return SimpleIngredient(name, **kwargs)

    def _refresh_configure_button(self, ing_type, combo, btn_configure):
        ingredient = combo.currentData()
        needs_work = False

        if ing_type == "Protein" and ingredient is not None:
            needs_work = ProteinConfigDialog.protein_needs_completion(ingredient)

        if needs_work:
            btn_configure.setStyleSheet(
                "QPushButton#btnIngredientConfigure {"
                "  border: 1.5px solid #e53935;"
                "  border-radius: 4px;"
                "  background-color: #fff5f5;"
                "}"
                "QPushButton#btnIngredientConfigure:hover {"
                "  background-color: #ffcdd2;"
                "}"
            )
            btn_configure.setToolTip(
                f"\u26a0 {combo.currentText()} has required fields that are not set — click to complete"
            )
            # Add matching error style to the combo box
            combo.setStyleSheet(
                "border: 1.5px solid #e53935;" "border-radius: 4px;" "background-color: #fff5f5;"
            )
        else:
            btn_configure.setStyleSheet("")
            combo.setStyleSheet("")
            if combo.currentIndex() >= 0:
                btn_configure.setToolTip(f"Edit {combo.currentText()}")

    def open_ingredient_config_dialog(self, ing_type, combo, btn_configure=None):
        dialog = None
        current_ingredient = None
        is_edit_mode = False

        if btn_configure and btn_configure.property("mode") == "edit":
            is_edit_mode = True
            current_ingredient = combo.currentData()

        controller = self.controller
        if not controller:
            QtWidgets.QMessageBox.critical(self, "Error", "Database controller not found.")
            return

        if ing_type == "Protein":
            dialog = ProteinConfigDialog(
                ing_ctrl=controller,
                existing_protein=current_ingredient,
                parent=self,
            )
        elif ing_type == "Buffer":
            dialog = GenericIngredientDialog(
                ingredient_type="Buffer",
                ing_ctrl=controller,
                existing_ingredient=current_ingredient,
                parent=self,
            )
        else:
            dialog = GenericIngredientDialog(
                ingredient_type=ing_type,
                ing_ctrl=controller,
                existing_ingredient=current_ingredient,
                parent=self,
            )

        # Handle Dialog Result
        if dialog and dialog.exec_() == QtWidgets.QDialog.DialogCode.Accepted:
            result = dialog.get_data()

            if isinstance(result, dict):
                self._handle_legacy_dialog_result(ing_type, combo, result, is_edit_mode)

            elif isinstance(result, Ingredient):
                new_ingredient = result

                if not is_edit_mode:
                    if ing_type not in self.ingredients_master:
                        self.ingredients_master[ing_type] = []
                    existing_names = [ing.name for ing in self.ingredients_master[ing_type]]
                    if new_ingredient.name in existing_names:
                        index = combo.findText(new_ingredient.name)
                        if index >= 0:
                            combo.setCurrentIndex(index)
                    else:
                        self.ingredients_master[ing_type].append(new_ingredient)
                        combo.addItem(new_ingredient.name, new_ingredient)
                        combo.setCurrentIndex(combo.count() - 1)
                else:
                    current_index = combo.currentIndex()
                    combo.setItemText(current_index, new_ingredient.name)

                self.trigger_update()
            self.broadcast_ingredient_update()

    def _handle_legacy_dialog_result(self, ing_type, combo, data, is_edit_mode):
        if not data.get("name"):
            QtWidgets.QMessageBox.warning(
                self, "Invalid Input", "Please provide a name for the ingredient."
            )
            return

        if is_edit_mode:
            current_ingredient = combo.currentData()
            if current_ingredient:
                for key, value in data.items():
                    if key == "class":
                        setattr(current_ingredient, "protein_class", value)
                    setattr(current_ingredient, key, value)

                current_index = combo.currentIndex()
                combo.setItemText(current_index, data["name"])
                self.trigger_update()
        else:
            new_ingredient = self._create_ingredient_instance(ing_type, **data)

            if ing_type not in self.ingredients_master:
                self.ingredients_master[ing_type] = []

            existing_names = [ing.name for ing in self.ingredients_master[ing_type]]
            if data["name"] in existing_names:
                QtWidgets.QMessageBox.information(
                    self,
                    "Ingredient Exists",
                    f"An ingredient named '{data['name']}' already exists.",
                )
                index = combo.findText(data["name"])
                if index >= 0:
                    combo.setCurrentIndex(index)
                return

            self.ingredients_master[ing_type].append(new_ingredient)
            combo.addItem(new_ingredient.name, new_ingredient)
            combo.setCurrentIndex(combo.count() - 1)
            self.trigger_update()

    def trigger_update(self):
        self._update_internal_formulation()
        self._update_warning_indicator()
        if self.is_expanded:
            self.debounce_timer.start()

    def broadcast_ingredient_update(self):
        """Notify all sibling ``FormulationConfigCard`` widgets to refresh their UI.

        Walks up to the immediate parent widget and calls
        ``refresh_ingredient_ui`` on every ``FormulationConfigCard`` child
        found via ``findChildren``.  Used after a new ingredient is created
        or edited through a config dialog so that all cards sharing
        ``ingredients_master`` reflect the change without a full rebuild.
        """
        parent = self.parentWidget()
        if parent:
            for child in parent.findChildren(FormulationConfigCard):
                if hasattr(child, "refresh_ingredient_ui"):
                    child.refresh_ingredient_ui()

    def refresh_ingredient_ui(self):
        """Recheck all configure-button states and trigger a formulation update.

        Calls ``_refresh_configure_button`` for every entry in
        ``active_ingredients``, then calls ``trigger_update`` to ensure
        ``lbl_warning`` and the debounce timer are refreshed after an
        ingredient edit that originated from a sibling card.
        """
        for ing_type, (
            combo,
            _,
            btn_configure,
            _,
        ) in self.active_ingredients.items():
            self._refresh_configure_button(ing_type, combo, btn_configure)
        self.trigger_update()

    def _update_internal_formulation(self):
        old_id = self.formulation.id if self.formulation else None
        old_name = self.formulation.name if self.formulation else None
        old_signature = self.formulation.signature if self.formulation else None
        self.formulation = Formulation(id=old_id, name=old_name, signature=old_signature)
        self.formulation.icl = self.use_in_icl

        current_model = self.model_combo.currentText()
        if " (Missing)" in current_model:
            current_model = current_model.replace(" (Missing)", "")
        self.formulation.last_model = current_model

        # Setters mapping
        setters = {
            "Protein": self.formulation.set_protein,
            "Buffer": self.formulation.set_buffer,
            "Stabilizer": self.formulation.set_stabilizer,
            "Surfactant": self.formulation.set_surfactant,
            "Salt": self.formulation.set_salt,
            "Excipient": self.formulation.set_excipient,
        }

        # Update Ingredients
        for ing_type, (combo, spin, _, _) in self.active_ingredients.items():
            ingredient = combo.currentData()
            concentration = spin.value()
            units = self.INGREDIENT_UNITS.get(ing_type, "")

            if ingredient and ing_type in setters:
                try:
                    if ing_type == "Buffer" and self._buffer_ph_spin is not None:
                        ph_val = self._buffer_ph_spin.value()
                        setters[ing_type](
                            ingredient, concentration, units, pH=ph_val if ph_val >= 0.0 else None
                        )
                    else:
                        setters[ing_type](ingredient, concentration, units)
                except TypeError as e:
                    Log.w(TAG, f"Could not set {ing_type} in formulation: {e}")

        # Update Temperature
        temp = self.spin_temp.value()
        self.formulation.set_temperature(temp)

        # Update Viscosity Profile
        if self.last_results:
            try:
                if self.is_measured and "measured_y" in self.last_results:
                    y_vals = self.last_results["measured_y"]
                    is_meas = True
                elif "y" in self.last_results:
                    y_vals = self.last_results["y"]
                    is_meas = False
                else:
                    y_vals = []
                    is_meas = False

                x_vals = self.last_results.get("x", [])

                if len(x_vals) > 0 and len(y_vals) > 0 and len(x_vals) == len(y_vals):
                    clean_pairs = [
                        (x, y) for x, y in zip(x_vals, y_vals) if x is not None and y is not None
                    ]
                    if clean_pairs:
                        sx, sy = zip(*clean_pairs)
                        profile = ViscosityProfile(list(sx), list(sy))
                        profile.is_measured = is_meas
                        self.formulation.set_viscosity_profile(profile)
            except Exception as e:
                Log.w(TAG, f"Failed to update viscosity profile: {e}")

    def get_formulation_dataframe(self, encoded=True, training=True):
        self._update_internal_formulation()
        try:
            return self.formulation.to_dataframe(encoded=encoded, training=training)
        except Exception as e:
            Log.e(TAG, f"Error converting formulation to dataframe: {e}")
            return None

    def _update_warning_indicator(self):
        if not hasattr(self, "lbl_warning"):
            return

        is_ready = True
        msg = ""

        if "Buffer" not in self.active_ingredients:
            is_ready = False
            msg = "Missing Buffer"
        else:
            for ing_type, (combo, spin, _, _) in self.active_ingredients.items():
                ingredient = combo.currentData()
                if ingredient is not None:
                    if spin.value() <= 0.0:
                        is_ready = False
                        msg = "Check Conc."
                        break
                    if ing_type == "Protein" and ProteinConfigDialog.protein_needs_completion(
                        ingredient
                    ):
                        is_ready = False
                        msg = "Incomplete Data"
                        break
                    if (
                        ing_type == "Buffer"
                        and self._buffer_ph_spin is not None
                        and self._buffer_ph_spin.value() < 0.0
                    ):
                        is_ready = False
                        msg = "Set Buffer pH"
                        break

        if is_ready:
            self.lbl_warning.setVisible(False)
        else:
            self.lbl_warning.setText(f"\u26a0. {msg}")
            self.lbl_warning.setVisible(True)

    def is_valid(self):
        if self.formulation is None:
            return False
        if "Buffer" not in self.active_ingredients:
            return False
        if not self.isVisible():
            return False
        for ing_type, (
            combo,
            spin,
            _,
            _,
        ) in self.active_ingredients.items():
            ingredient = combo.currentData()
            if ingredient is not None:
                if spin.value() <= 0.0:
                    return False
                if ing_type == "Protein" and ProteinConfigDialog.protein_needs_completion(
                    ingredient
                ):
                    return False
                if (
                    ing_type == "Buffer"
                    and self._buffer_ph_spin is not None
                    and self._buffer_ph_spin.value() == 0.0
                ):
                    return False
        return True

    def emit_run_request(self):
        if not self.is_valid():
            return

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
        self.setProperty("selected", checked)
        self.style().unpolish(self)
        self.style().polish(self)

    def get_configuration(self):
        self._update_internal_formulation()

        formulation_data = {}
        for t, (combo, spin, _, _) in self.active_ingredients.items():
            entry = {
                "component": combo.currentText(),
                "concentration": spin.value(),
            }
            if t == "Buffer" and self._buffer_ph_spin is not None:
                entry["pH"] = self._buffer_ph_spin.value()
            formulation_data[t] = entry

        config = {
            "name": self.name_input.text(),
            "model": self.model_combo.currentText(),
            "temp": self.spin_temp.value(),
            "formulation": formulation_data,
            "measured": self.is_measured,
            "notes": self.notes_edit.toPlainText(),
            "use_in_icl": self.use_in_icl,
            "color": self.plot_color,
            "formulation_object": self.formulation,
        }
        config.update(self.ml_params)
        return config

    def load_data(self, data):
        self._loading = True
        try:
            self._load_data_impl(data)
        finally:
            self._loading = False
            self._update_internal_formulation()
            self._update_warning_indicator()

    def _load_data_impl(self, data):
        if "id" in data:
            self.formulation.id = data["id"]
        if "name" in data:
            self.name_input.setText(data["name"])
        if "color" in data:
            self.plot_color = data["color"]
            if hasattr(self, "color_swatch"):
                self.color_swatch.setStyleSheet(
                    f"background-color: {self.plot_color}; border-radius: 3px;"
                )
        target_model = data.get("last_model") or data.get("model")
        if target_model:
            index = self.model_combo.findText(target_model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
            else:
                self.model_combo.addItem(f"{target_model} (Missing)", None)
                self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
        else:
            if self.model_combo.count() > 0:
                self.model_combo.setCurrentIndex(0)

        ml_params = data.get("ml_params", {})
        # Load LR
        if "lr" in ml_params:
            self.ml_params["lr"] = float(ml_params["lr"])
        elif "lr" in data:
            self.ml_params["lr"] = float(data["lr"])

        # Load Steps
        if "steps" in ml_params:
            self.ml_params["steps"] = int(ml_params["steps"])
        elif "steps" in data:
            self.ml_params["steps"] = int(data["steps"])

        # Load CI
        if "ci" in ml_params:
            self.ml_params["ci"] = int(ml_params["ci"])
        elif "ci" in data:
            self.ml_params["ci"] = int(data["ci"])

        # Load ICL Filter
        if "icl_filter" in ml_params:
            self.ml_params["icl_filter"] = ml_params["icl_filter"]
        elif "icl_filter" in data:
            self.ml_params["icl_filter"] = data["icl_filter"]

        icl_state = data.get("icl", data.get("use_in_icl", True))
        self.use_in_icl = icl_state
        if hasattr(self, "act_use_icl"):
            self.act_use_icl.blockSignals(True)
            self.act_use_icl.setChecked(icl_state)
            self.act_use_icl.blockSignals(False)

        if "temp" in data or "temperature" in data:
            temp_val = data.get("temp") or data.get("temperature")
            self.spin_temp.setValue(float(temp_val))

        # Populate Notes
        if "notes" in data:
            self.notes_edit.setText(data["notes"])

        # Populate Ingredients
        ingredients_data = data.get("formulation") or data.get("ingredients")
        if ingredients_data:
            for ing_type, details in ingredients_data.items():
                is_mandatory = ing_type == "Buffer"
                self.add_ingredient_row(ing_type, deletable=not is_mandatory)

                if ing_type in self.active_ingredients:
                    combo, spin, _, _ = self.active_ingredients[ing_type]
                    spin.setValue(float(details.get("concentration", 0.0)))
                    if ing_type == "Buffer" and self._buffer_ph_spin is not None:
                        ph = details.get("pH") or details.get("ph")
                        self._buffer_ph_spin.setValue(float(ph) if ph is not None else -1.0)

                    spin.setValue(float(details.get("concentration", 0.0)))
                    comp_name = details.get("component") or details.get("name")

                    if comp_name:
                        idx = combo.findText(comp_name)
                        if idx >= 0:
                            combo.setCurrentIndex(idx)
                        else:
                            new_ingredient = self._create_ingredient_instance(ing_type, **details)
                            if ing_type not in self.ingredients_master:
                                self.ingredients_master[ing_type] = []
                            self.ingredients_master[ing_type].append(new_ingredient)
                            combo.addItem(new_ingredient.name, new_ingredient)
                            combo.setCurrentIndex(combo.count() - 1)
        if "measured" in data:
            self.set_measured(data["measured"])

        if "missing_fields" in data:
            self.mark_missing_fields(data["missing_fields"])

    def mark_missing_fields(self, missing_list):
        """Visually indicates missing information on the card."""
        if not missing_list:
            return

        for field in missing_list:
            if "Type" in field:
                ing_type = field.replace(" Type", "")
                if ing_type not in self.active_ingredients:
                    if ing_type in self.INGREDIENT_TYPES:
                        self.add_ingredient_row(ing_type)

                if ing_type in self.active_ingredients:
                    combo, spin, _, _ = self.active_ingredients[ing_type]
                    combo.setStyleSheet("border: 1px solid #e53935; background-color: #ffebee;")
                    combo.setToolTip(f"Missing {ing_type} in imported file. Please select.")

            elif field == "Buffer pH":
                if self._buffer_ph_spin is not None:
                    self._buffer_ph_spin.setStyleSheet(
                        "border: 1px solid #e53935; background-color: #ffebee;"
                    )
                    self._buffer_ph_spin.setToolTip(
                        "Buffer pH was missing in the imported file. Please set."
                    )

            elif field == "Viscosity Data":
                self.lbl_measured.setText("No Viscosity Data")
                self.lbl_measured.setProperty("class", "badge-warning")
                self.lbl_measured.style().unpolish(self.lbl_measured)
                self.lbl_measured.style().polish(self.lbl_measured)
                self.lbl_measured.setToolTip(
                    "Viscosity profile could not be loaded from the import source."
                )

    def set_selectable(self, active: bool):
        self.is_selectable = active
        if active:
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            if self.is_selected:
                self.toggle_selection()

        self.header_frame.setEnabled(not active)
        self.content_frame.setEnabled(not active)

    def toggle_selection(self):
        self.is_selected = not self.is_selected
        self.setProperty("selected", self.is_selected)
        self.style().unpolish(self)
        self.style().polish(self)
        self.selection_changed.emit(self.is_selected)

    def mousePressEvent(self, event):
        if self.is_selectable:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.toggle_selection()
                return

        super().mousePressEvent(event)

    def clear_formulation(self):
        if self.is_measured:
            QtWidgets.QMessageBox.warning(self, "Action Denied", "Cannot clear imported data.")
            return

        for ing_type in list(self.active_ingredients.keys()):
            if ing_type == "Buffer":
                combo, spin, _, _ = self.active_ingredients[ing_type]
                if combo.count() > 0:
                    combo.setCurrentIndex(0)
                spin.setValue(0.0)
                if self._buffer_ph_spin is not None:
                    self._buffer_ph_spin.setValue(-1.0)
            else:
                combo, _, _, _ = self.active_ingredients[ing_type]
                row_widget = combo.parentWidget()
                self.remove_ingredient_row(ing_type, row_widget)

        self.slider_temp.setValue(25)
        self.spin_temp.setValue(25.0)

        self.trigger_update()

    def expand_silent(self):
        if self.is_expanded:
            return

        self.expanded.emit(self)
        self.is_expanded = True
        self.btn_toggle.setArrowType(QtCore.Qt.ArrowType.UpArrow)

        if not hasattr(self, "_anim_accordion"):
            self._anim_accordion = QtCore.QPropertyAnimation(self.content_frame, b"maximumHeight")
            self._anim_accordion.setDuration(250)
            self._anim_accordion.setEasingCurve(QtCore.QEasingCurve.OutCubic)

        try:
            self._anim_accordion.finished.disconnect()
        except TypeError:
            pass

        self.content_frame.setVisible(True)
        self.content_frame.setMaximumHeight(0)
        QtWidgets.QApplication.processEvents()
        target_height = self.content_frame.sizeHint().height()

        self._anim_accordion.setStartValue(0)
        self._anim_accordion.setEndValue(target_height)
        self._anim_accordion.finished.connect(lambda: self.content_frame.setMaximumHeight(16777215))
        self._anim_accordion.start()

    def export_formulation(self):
        """Prompt the user for a file path and save the current results to CSV.

        Shows a warning dialog when ``last_results`` is ``None``.  Otherwise
        opens a ``QFileDialog.getSaveFileName`` dialog defaulting to
        ``"<card_name>.csv"``, then delegates to ``save_to_csv``.  Shows an
        informational confirmation or a critical error dialog depending on the
        outcome.
        """
        if not self.last_results:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No data available to export.\nPlease run a prediction or import data first.",
            )
            return

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
        """Write the current viscosity profile and confidence interval to a CSV file.

        Writes a ``"--- Viscosity Profile ---"`` header row, then a
        column-header row and one data row per shear-rate point.

        * **Measured cards** - columns: Shear Rate (1/s), Measured Viscosity
          (cP), Lower CI (cP), Upper CI (cP).  Missing ``measured_y``,
          ``lower``, or ``upper`` arrays are replaced with zero-filled lists
          of the same length as ``x``.
        * **Predicted cards** - columns: Shear Rate (1/s), Predicted Viscosity
          (cP), Lower CI (cP), Upper CI (cP).  Same zero-fill fallback for
          missing arrays.

        All numeric values are formatted to four decimal places.

        Args:
            filepath (str): Absolute path of the CSV file to write (created
                or overwritten).
        """
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

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

    def _save_metadata_to_db(self):
        if self._loading:
            return
        if not self.formulation or not self.formulation.id:
            return

        ctrl = self.formulation_controller
        if ctrl:
            try:
                current_model = self.model_combo.currentText().replace(" (Missing)", "")
                ctrl.update_formulation_metadata(
                    self.formulation.id,
                    icl=self.use_in_icl,
                    last_model=current_model or None,
                )
                Log.i(TAG, f"Saved metadata for formulation {self.formulation.id}")
            except Exception as e:
                Log.e(TAG, f"Failed to auto-save formulation metadata: {e}")
