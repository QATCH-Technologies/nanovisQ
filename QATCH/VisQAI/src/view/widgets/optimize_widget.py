"""
optimize_widget.py

Overlay panel for configuring and launching formulation optimization.
Mirrors the layout conventions of GenerateSampleWidget and EvaluationWidget.

Signals
-------
optimize_requested(model_file: str, targets: list[dict], constraints_data: list[dict])
    Emitted when the user clicks "Optimize".  Each target dict has keys
    ``shear_rate`` (int) and ``viscosity`` (float).  Each constraint dict
    matches the schema produced by GenerateSampleWidget.
closed()
    Emitted when the close button is pressed.
resized()
    Emitted whenever the widget's preferred size changes so the dashboard
    can reposition the overlay.
"""

import glob
import os
import shutil

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from src.view.architecture import Architecture
    from src.view.checkable_combo_box import CheckableComboBox
    from src.view.model_selection_dialog import ModelSelectionDialog
except (ImportError, ModuleNotFoundError):
    from QATCH.common.architecture import Architecture
    from QATCH.VisQAI.src.view.checkable_combo_box import CheckableComboBox
    from QATCH.VisQAI.src.view.model_selection_dialog import ModelSelectionDialog


# ── Constants ─────────────────────────────────────────────────────────────────
SHEAR_RATE_OPTIONS = [100, 1_000, 10_000, 100_000, 15_000_000]
SHEAR_RATE_LABELS = [
    "100 s⁻¹",
    "1,000 s⁻¹",
    "10,000 s⁻¹",
    "100,000 s⁻¹",
    "15,000,000 s⁻¹",
]
MAX_TARGETS = 5


class _CompactCheckableComboBox(CheckableComboBox):
    """Wraps CheckableComboBox so it shows a summary instead of raw CSV."""

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        opt = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(opt)

        try:
            items = self.getItems()
            if not items:
                opt.currentText = "Select..."
            elif len(items) == 1:
                opt.currentText = items[0]
            else:
                opt.currentText = f"{len(items)} items selected"
        except Exception:
            pass

        self.style().drawComplexControl(
            QtWidgets.QStyle.CC_ComboBox, opt, painter, self
        )
        self.style().drawControl(QtWidgets.QStyle.CE_ComboBoxLabel, opt, painter, self)


class OptimizeWidget(QtWidgets.QFrame):
    """Overlay widget: viscosity targets + constraints → emits optimize_requested."""

    optimize_requested = QtCore.pyqtSignal(str, list, list)
    closed = QtCore.pyqtSignal()
    resized = QtCore.pyqtSignal()

    def __init__(self, ingredients_by_type, parent=None):
        super().__init__(parent)
        self.ingredients_by_type = ingredients_by_type

        self.assets_path = os.path.join(
            Architecture.get_path(), "QATCH", "VisQAI", "assets"
        )
        os.makedirs(self.assets_path, exist_ok=True)

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setProperty("class", "card")

        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setYOffset(10)
        shadow.setColor(QtGui.QColor(0, 0, 0, 40))
        self.setGraphicsEffect(shadow)

        self.setVisible(False)
        self.setMinimumWidth(620)

        self.target_rows: list = []
        self.constraint_rows: list = []
        self._init_ui()

    # ──────────────────────────────────────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────────────────────────────────────

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 20)
        layout.setSpacing(15)

        # ── Header ────────────────────────────────────────────────────────────
        header = QtWidgets.QHBoxLayout()
        lbl_title = QtWidgets.QLabel("Optimize Formulation")
        lbl_title.setObjectName("evalTitle")
        header.addWidget(lbl_title)
        header.addStretch()

        icon_path = os.path.join(
            Architecture.get_path(),
            "QATCH",
            "VisQAI",
            "src",
            "view",
            "icons",
            "close-circle-svgrepo-com.svg",
        )
        btn_close = QtWidgets.QToolButton()
        btn_close.setObjectName("btnEvalClose")
        btn_close.setIcon(QtGui.QIcon(icon_path))
        btn_close.setIconSize(QtCore.QSize(18, 18))
        btn_close.setFixedSize(24, 24)
        btn_close.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn_close.clicked.connect(self.close_widget)
        header.addWidget(btn_close)
        layout.addLayout(header)

        # ── Configuration ─────────────────────────────────────────────────────
        grp_cfg = QtWidgets.QGroupBox("Configuration")
        cfg_form = QtWidgets.QFormLayout(grp_cfg)
        cfg_form.setSpacing(12)

        # Model selector row
        model_row = QtWidgets.QHBoxLayout()
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setStyleSheet("background-color: #ffffff; height: 26px;")
        self.model_combo.setToolTip("Select a prediction model (.visq)")
        self._populate_model_list()

        _icon_import = os.path.join(
            Architecture.get_path(),
            "QATCH",
            "VisQAI",
            "src",
            "view",
            "icons",
            "file-plus-2-svgrepo-com.svg",
        )
        self.btn_select_model = QtWidgets.QPushButton()
        self.btn_select_model.setFixedSize(40, 26)
        self.btn_select_model.setIcon(QtGui.QIcon(_icon_import))
        self.btn_select_model.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_select_model.setToolTip("Import new model (.visq)")
        self.btn_select_model.clicked.connect(self.browse_model_file)

        model_row.addWidget(self.model_combo, stretch=1)
        model_row.addWidget(self.btn_select_model)
        cfg_form.addRow("Model:", model_row)

        # Max iterations
        self.spin_maxiter = QtWidgets.QSpinBox()
        self.spin_maxiter.setRange(10, 1000)
        self.spin_maxiter.setValue(100)
        self.spin_maxiter.setFixedWidth(100)
        self.spin_maxiter.setFixedHeight(26)
        self.spin_maxiter.setToolTip("Maximum iterations for differential evolution")
        cfg_form.addRow("Max Iterations:", self.spin_maxiter)
        layout.addWidget(grp_cfg)

        # ── Viscosity Targets ─────────────────────────────────────────────────
        grp_tgt = QtWidgets.QGroupBox(
            "Viscosity Targets  (1 – 5 shear-rate / target pairs)"
        )
        tgt_vbox = QtWidgets.QVBoxLayout(grp_tgt)
        tgt_vbox.setContentsMargins(15, 15, 15, 10)
        tgt_vbox.setSpacing(6)

        # Column header row
        hdr_row = QtWidgets.QHBoxLayout()
        hdr_row.setContentsMargins(0, 0, 0, 0)
        for text, width in [("Shear Rate", 148), ("Target Viscosity (cP)", 0)]:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet("color: #6b7280; font-size: 10px; font-weight: 600;")
            if width:
                lbl.setFixedWidth(width)
            hdr_row.addWidget(lbl)
        hdr_row.addStretch()
        tgt_vbox.addLayout(hdr_row)

        # Scrollable container for target rows
        self.targets_scroll = QtWidgets.QScrollArea()
        self.targets_scroll.setWidgetResizable(True)
        self.targets_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.targets_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.targets_scroll.setFixedHeight(30)  # grows as rows are added

        self.targets_container = QtWidgets.QWidget()
        self.targets_layout = QtWidgets.QVBoxLayout(self.targets_container)
        self.targets_layout.setContentsMargins(0, 0, 4, 0)
        self.targets_layout.setSpacing(6)

        self.lbl_no_targets = QtWidgets.QLabel(
            "No targets added.  Add at least one viscosity target."
        )
        self.lbl_no_targets.setStyleSheet("color: #6b7280; font-style: italic;")
        self.targets_layout.addWidget(self.lbl_no_targets)
        self.targets_layout.addStretch()

        self.targets_scroll.setWidget(self.targets_container)
        tgt_vbox.addWidget(self.targets_scroll)

        self.btn_add_target = QtWidgets.QPushButton("+ Add Target")
        self.btn_add_target.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_add_target.setFixedHeight(28)
        self.btn_add_target.setFixedWidth(120)
        self.btn_add_target.clicked.connect(self.add_target_row)
        tgt_vbox.addWidget(
            self.btn_add_target, alignment=QtCore.Qt.AlignmentFlag.AlignLeft
        )
        layout.addWidget(grp_tgt)

        # ── Constraints (Optional) ────────────────────────────────────────────
        grp_con = QtWidgets.QGroupBox("Constraints  (Optional)")
        con_vbox = QtWidgets.QVBoxLayout(grp_con)
        con_vbox.setContentsMargins(15, 15, 15, 15)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.constraints_container = QtWidgets.QWidget()
        self.constraints_layout = QtWidgets.QVBoxLayout(self.constraints_container)
        self.constraints_layout.setContentsMargins(0, 0, 5, 0)
        self.constraints_layout.setSpacing(8)

        self.lbl_none = QtWidgets.QLabel("No constraints added.")
        self.lbl_none.setStyleSheet("color: #6b7280; font-style: italic;")
        self.constraints_layout.addWidget(self.lbl_none)
        self.constraints_layout.addStretch()

        self.scroll_area.setWidget(self.constraints_container)
        con_vbox.addWidget(self.scroll_area)
        layout.addWidget(grp_con)

        # ── Footer ────────────────────────────────────────────────────────────
        layout.addSpacing(5)
        footer = QtWidgets.QHBoxLayout()

        self.btn_add_constraint = QtWidgets.QPushButton("+ Add Constraint")
        self.btn_add_constraint.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_add_constraint.setFixedHeight(34)
        self.btn_add_constraint.clicked.connect(self.add_constraint_row)

        self.btn_optimize = QtWidgets.QPushButton("Optimize")
        self.btn_optimize.setObjectName("btnApplyFilters")
        self.btn_optimize.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_optimize.setFixedHeight(34)
        self.btn_optimize.setFixedWidth(140)
        self.btn_optimize.clicked.connect(self.emit_optimize)

        footer.addWidget(self.btn_add_constraint)
        footer.addStretch()
        footer.addWidget(self.btn_optimize)
        layout.addLayout(footer)

        self._update_scroll_height()
        self._validate()
        # Seed one default target row so the widget is immediately usable
        self.add_target_row()

    # ──────────────────────────────────────────────────────────────────────────
    # Model helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _populate_model_list(self):
        self.model_combo.clear()
        files = sorted(glob.glob(os.path.join(self.assets_path, "*.visq")))
        if not files:
            self.model_combo.addItem("No models found")
            self.model_combo.setEnabled(False)
        else:
            self.model_combo.setEnabled(True)
            for f in files:
                self.model_combo.addItem(os.path.basename(f), f)
            idx = self.model_combo.findText("VisQAI(base).visq")
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)

    def browse_model_file(self):
        fname = None
        try:
            dlg = ModelSelectionDialog()
            dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            dlg.setNameFilter("VisQAI Models (*.visq)")
            dlg.setViewMode(QtWidgets.QFileDialog.Detail)
            mp = os.path.join(Architecture.get_path(), "QATCH/VisQAI/assets")
            if os.path.exists(mp):
                dlg.setDirectory(mp)
            if dlg.exec_():
                sel = dlg.selectedFiles()
                if sel:
                    fname = sel[0]
        except Exception:
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Import Model File", "", "VisQAI Models (*.visq)"
            )
        if fname:
            try:
                dest = os.path.join(self.assets_path, os.path.basename(fname))
                if os.path.abspath(fname) != os.path.abspath(dest):
                    shutil.copy2(fname, dest)
                self._populate_model_list()
                idx = self.model_combo.findText(os.path.basename(fname))
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Import Failed", f"Could not import model:\n{e}"
                )

    # ──────────────────────────────────────────────────────────────────────────
    # Target rows
    # ──────────────────────────────────────────────────────────────────────────

    def add_target_row(self):
        if len(self.target_rows) >= MAX_TARGETS:
            return

        self.lbl_no_targets.hide()

        combo_style = (
            "background-color: #ffffff; height: 26px; "
            "border: 1px solid #d1d5db; border-radius: 4px;"
        )

        row_w = QtWidgets.QWidget()
        row_l = QtWidgets.QHBoxLayout(row_w)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.setSpacing(8)

        cb_shear = QtWidgets.QComboBox()
        cb_shear.setStyleSheet(combo_style)
        cb_shear.setFixedWidth(148)
        cb_shear.addItems(SHEAR_RATE_LABELS)
        cb_shear.setCurrentIndex(2)  # default: 10 000 s⁻¹

        lbl_arrow = QtWidgets.QLabel("→")
        lbl_arrow.setStyleSheet("color: #9ca3af;")

        spin_visc = QtWidgets.QDoubleSpinBox()
        spin_visc.setStyleSheet(combo_style)
        spin_visc.setRange(0.01, 100_000.0)
        spin_visc.setDecimals(2)
        spin_visc.setValue(1.0)
        spin_visc.setSuffix(" cP")
        spin_visc.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        spin_visc.setFixedWidth(120)
        spin_visc.setToolTip("Target viscosity at the selected shear rate")

        _icon_del = os.path.join(
            Architecture.get_path(),
            "QATCH",
            "VisQAI",
            "src",
            "view",
            "icons",
            "delete-2-svgrepo-com.svg",
        )
        btn_del = QtWidgets.QToolButton()
        btn_del.setIcon(QtGui.QIcon(_icon_del))
        btn_del.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn_del.setStyleSheet("border: none;")

        row_l.addWidget(cb_shear)
        row_l.addWidget(lbl_arrow)
        row_l.addWidget(spin_visc)
        row_l.addStretch()
        row_l.addWidget(btn_del)

        # Insert before the trailing stretch in targets_layout
        insert_at = max(0, self.targets_layout.count() - 1)
        self.targets_layout.insertWidget(insert_at, row_w)

        row_data = {"widget": row_w, "shear_cb": cb_shear, "visc_spin": spin_visc}
        self.target_rows.append(row_data)

        btn_del.clicked.connect(lambda: self._remove_target_row(row_data))
        spin_visc.valueChanged.connect(lambda _: self._validate())

        self.btn_add_target.setEnabled(len(self.target_rows) < MAX_TARGETS)
        self._update_target_scroll_height()
        self._validate()

    def _remove_target_row(self, row_data):
        row_data["widget"].deleteLater()
        self.target_rows.remove(row_data)
        if not self.target_rows:
            self.lbl_no_targets.show()
        self.btn_add_target.setEnabled(True)
        self._update_target_scroll_height()
        self._validate()

    def _update_target_scroll_height(self):
        n = len(self.target_rows)
        if n == 0:
            self.targets_scroll.setFixedHeight(30)
        else:
            self.targets_scroll.setFixedHeight(min(n, 3) * 38 + 8)
        self.adjustSize()
        self.resized.emit()

    # ──────────────────────────────────────────────────────────────────────────
    # Constraint rows  (mirrors GenerateSampleWidget exactly)
    # ──────────────────────────────────────────────────────────────────────────

    def add_constraint_row(self):
        self.lbl_none.hide()

        combo_style = (
            "background-color: #ffffff; height: 26px; "
            "border: 1px solid #d1d5db; border-radius: 4px;"
        )
        _icon_del = os.path.join(
            Architecture.get_path(),
            "QATCH",
            "VisQAI",
            "src",
            "view",
            "icons",
            "delete-2-svgrepo-com.svg",
        )

        row_w = QtWidgets.QWidget()
        row_l = QtWidgets.QHBoxLayout(row_w)
        row_l.setContentsMargins(0, 0, 0, 0)

        cb_ingredient = QtWidgets.QComboBox()
        cb_ingredient.addItem("Ingredient...")
        cb_ingredient.model().item(0).setEnabled(False)
        cb_ingredient.addItems(
            ["Protein", "Buffer", "Surfactant", "Stabilizer", "Salt", "Excipient"]
        )
        cb_ingredient.setStyleSheet(combo_style)

        cb_attribute = QtWidgets.QComboBox()
        cb_attribute.addItem("Attribute...")
        cb_attribute.model().item(0).setEnabled(False)
        cb_attribute.setStyleSheet(combo_style)

        cb_condition = QtWidgets.QComboBox()
        cb_condition.addItem("Condition...")
        cb_condition.model().item(0).setEnabled(False)
        cb_condition.setStyleSheet(combo_style)

        val_stack = QtWidgets.QStackedWidget()
        cb_value = _CompactCheckableComboBox()
        cb_value.setStyleSheet(combo_style)
        spin_value = QtWidgets.QDoubleSpinBox()
        spin_value.setStyleSheet(combo_style)
        spin_value.setRange(0.0, 10_000.0)
        spin_value.setDecimals(3)
        spin_value.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        val_stack.addWidget(cb_value)  # index 0  — categorical
        val_stack.addWidget(spin_value)  # index 1  — numeric

        btn_del = QtWidgets.QToolButton()
        btn_del.setIcon(QtGui.QIcon(_icon_del))
        btn_del.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn_del.setStyleSheet("border: none;")

        row_l.addWidget(cb_ingredient)
        row_l.addWidget(cb_attribute)
        row_l.addWidget(cb_condition)
        row_l.addWidget(val_stack, stretch=1)
        row_l.addWidget(btn_del)

        self.constraints_layout.insertWidget(len(self.constraint_rows), row_w)

        row_data = {
            "widget": row_w,
            "ingredient": cb_ingredient,
            "attribute": cb_attribute,
            "condition": cb_condition,
            "value_stack": val_stack,
            "value_cb": cb_value,
            "value_spin": spin_value,
        }
        self.constraint_rows.append(row_data)

        btn_del.clicked.connect(lambda: self._remove_constraint_row(row_data))
        cb_ingredient.currentIndexChanged.connect(
            lambda: self._on_ingredient_changed(row_data)
        )
        cb_attribute.currentIndexChanged.connect(
            lambda: self._on_attribute_changed(row_data)
        )
        cb_condition.currentIndexChanged.connect(self._validate)
        cb_value.model().dataChanged.connect(self._validate)
        spin_value.valueChanged.connect(lambda _: self._validate())

        self._update_scroll_height()
        self._validate()

    def _remove_constraint_row(self, row_data):
        row_data["widget"].deleteLater()
        self.constraint_rows.remove(row_data)
        if not self.constraint_rows:
            self.lbl_none.show()
        self._update_scroll_height()
        self._validate()

    def _on_ingredient_changed(self, row_data):
        ing_type = row_data["ingredient"].currentText()
        cb_attr = row_data["attribute"]
        cb_attr.blockSignals(True)
        cb_attr.clear()
        cb_attr.addItem("Attribute...")
        cb_attr.model().item(0).setEnabled(False)
        if row_data["ingredient"].currentIndex() > 0:
            attrs = ["Type", "Concentration"]
            if ing_type == "Protein":
                attrs.append("Class")
            cb_attr.addItems(attrs)
        cb_attr.setCurrentIndex(0)
        cb_attr.blockSignals(False)
        self._on_attribute_changed(row_data)

    def _on_attribute_changed(self, row_data):
        attr_type = row_data["attribute"].currentText()
        cb_cond = row_data["condition"]
        val_stack = row_data["value_stack"]
        cb_cond.blockSignals(True)
        cb_cond.clear()
        cb_cond.addItem("Condition...")
        cb_cond.model().item(0).setEnabled(False)
        if row_data["attribute"].currentIndex() > 0:
            if attr_type == "Concentration":
                cb_cond.addItems([">", ">=", "=", "!=", "<=", "<"])
                val_stack.setCurrentIndex(1)
            elif attr_type in ("Type", "Class"):
                cb_cond.addItems(["is", "is not"])
                val_stack.setCurrentIndex(0)
        cb_cond.setCurrentIndex(0)
        cb_cond.blockSignals(False)
        self._populate_values(row_data)

    def _populate_values(self, row_data):
        ing_idx = row_data["ingredient"].currentIndex()
        attr_idx = row_data["attribute"].currentIndex()
        val_stack = row_data["value_stack"]
        val_cb = row_data["value_cb"]

        if ing_idx <= 0 or attr_idx <= 0:
            if val_stack.currentIndex() == 0:
                val_cb.clear()
            self._validate()
            return

        ing_type = row_data["ingredient"].currentText()
        attr_type = row_data["attribute"].currentText()

        if attr_type in ("Type", "Class"):
            val_cb.clear()
            if attr_type == "Class" and ing_type == "Protein":
                classes = set()
                for p in self.ingredients_by_type.get("Protein", []):
                    if hasattr(p, "class_type") and p.class_type:
                        c_val = str(
                            getattr(
                                p.class_type,
                                "value",
                                getattr(p.class_type, "name", str(p.class_type)),
                            )
                        )
                        if c_val != "-":
                            classes.add(c_val)
                items = sorted(classes)
            else:
                items = [obj.name for obj in self.ingredients_by_type.get(ing_type, [])]
                if ing_type not in ("Protein", "Buffer") and "None" not in items:
                    items.insert(0, "None")
            val_cb.addItems(items)

        self._validate()

    def _update_scroll_height(self):
        n = len(self.constraint_rows)
        self.scroll_area.setFixedHeight(30 if n == 0 else min(n, 3) * 36 + 10)
        self.adjustSize()
        self.resized.emit()

    # ──────────────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────────────

    def _validate(self):
        has_targets = len(self.target_rows) > 0
        has_model = self.model_combo.isEnabled()

        constraints_ok = True
        for row in self.constraint_rows:
            ing_ok = row["ingredient"].currentIndex() > 0
            attr_ok = row["attribute"].currentIndex() > 0
            cond_ok = row["condition"].currentIndex() > 0
            if row["value_stack"].currentIndex() == 0:
                val_ok = len(row["value_cb"].getItems()) > 0
            else:
                val_ok = True
            if not (ing_ok and attr_ok and cond_ok and val_ok):
                constraints_ok = False
                break

        self.btn_add_constraint.setEnabled(constraints_ok)
        self.btn_optimize.setEnabled(has_targets and has_model and constraints_ok)

    # ──────────────────────────────────────────────────────────────────────────
    # Emit / Close
    # ──────────────────────────────────────────────────────────────────────────

    def emit_optimize(self):
        targets = []
        for row in self.target_rows:
            idx = row["shear_cb"].currentIndex()
            shear = (
                SHEAR_RATE_OPTIONS[idx]
                if 0 <= idx < len(SHEAR_RATE_OPTIONS)
                else 10_000
            )
            targets.append({"shear_rate": shear, "viscosity": row["visc_spin"].value()})

        constraints_data = []
        for row in self.constraint_rows:
            val = (
                row["value_cb"].getItems()
                if row["value_stack"].currentIndex() == 0
                else row["value_spin"].value()
            )
            constraints_data.append(
                {
                    "ingredient": row["ingredient"].currentText(),
                    "attribute": row["attribute"].currentText(),
                    "condition": row["condition"].currentText(),
                    "values": val,
                }
            )

        self.optimize_requested.emit(
            self.model_combo.currentText(), targets, constraints_data
        )

    def close_widget(self):
        self.hide()
        self.closed.emit()
