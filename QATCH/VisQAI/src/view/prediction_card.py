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
