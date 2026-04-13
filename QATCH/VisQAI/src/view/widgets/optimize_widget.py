"""
optimize_widget.py

Overlay panel for configuring and launching differential-evolution-based
formulation optimization within the VisQAI dashboard.

Provides a self-contained PyQt5 ``QFrame`` overlay (``OptimizeWidget``) that
lets users define one to five shear-rate / target-viscosity pairs and an
optional set of ingredient constraints, then fire them at a chosen ``.visq``
model file.  Layout conventions mirror ``GenerateSampleWidget`` and
``EvaluationWidget`` so the three overlays feel visually consistent.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.0
"""

import glob
import os
import shutil

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from src.view.architecture import Architecture
    from src.view.components.checkable_combo_box import CheckableComboBox
    from src.view.dialogs.model_selection_dialog import (
        ModelSelectionDialog,
    )
except (ImportError, ModuleNotFoundError):
    from QATCH.common.architecture import Architecture
    from QATCH.VisQAI.src.view.components.checkable_combo_box import CheckableComboBox
    from QATCH.VisQAI.src.view.dialogs.model_selection_dialog import (
        ModelSelectionDialog,
    )


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
    """A ``CheckableComboBox`` subclass that renders a concise selection summary.

    ``CheckableComboBox`` defaults to displaying a raw comma-separated string
    of every checked item in the combo label.  This subclass overrides
    ``paintEvent`` to show a friendlier summary: nothing when empty, the single
    item name when exactly one is checked, or ``"N selected"`` otherwise.
    """

    def paintEvent(self, event):
        """Paint the combo box with a human-readable selection summary.

        Overrides the default ``QComboBox`` paint path so the label area shows
        ``"Select..."`` when nothing is checked, the item's own text when
        exactly one is checked, or ``"N selected"`` (where *N* is the count)
        when multiple items are checked.

        Args:
            event (QtGui.QPaintEvent): The paint event delivered by Qt.
        """

        painter = QtGui.QPainter(self)
        opt = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(opt)

        try:
            from PyQt5.QtCore import Qt

            model = self.model()
            checked = []
            for i in range(model.rowCount()):
                item = model.item(i)
                if item is not None and item.checkState() == Qt.Checked:
                    checked.append(item.text())
            if not checked:
                opt.currentText = "Select..."
            elif len(checked) == 1:
                opt.currentText = checked[0]
            else:
                opt.currentText = f"{len(checked)} selected"
        except Exception:
            pass

        self.style().drawComplexControl(QtWidgets.QStyle.CC_ComboBox, opt, painter, self)
        self.style().drawControl(QtWidgets.QStyle.CE_ComboBoxLabel, opt, painter, self)


class OptimizeWidget(QtWidgets.QFrame):
    """Overlay widget for configuring and launching formulation optimization.

    Renders as a floating ``QFrame`` card over the dashboard.  Exposes:

    * A model selector (populated from ``*.visq`` files in the assets folder,
      with an import button to copy additional files in).
    * A max-iterations spin box fed directly to the differential-evolution
      solver.
    * A scrollable list of shear-rate / target-viscosity pairs.
    * An optional scrollable list of ingredient constraints.
    * An "Optimize" footer button that emits ``optimize_requested`` when all
      required fields are valid.

    Attributes:
        ingredients_by_type (dict[str, list]): Mapping of ingredient category
            name (e.g. ``"Protein"``, ``"Buffer"``) to a list of ingredient
            objects.  Used to populate constraint value combo boxes.
        assets_path (str): Absolute path to the VisQAI assets directory where
            ``.visq`` model files are stored and discovered.
        target_rows (list[dict]): Live list of target-row state dicts, each
            containing keys ``widget``, ``shear_cb``, and ``visc_spin``.
        constraint_rows (list[dict]): Live list of constraint-row state dicts,
            each containing keys ``widget``, ``ingredient``, ``attribute``,
            ``condition``, ``value_stack``, ``value_cb``, and ``value_spin``.
    """

    optimize_requested = QtCore.pyqtSignal(str, list, list)
    closed = QtCore.pyqtSignal()
    resized = QtCore.pyqtSignal()

    def __init__(self, ingredients_by_type, parent=None):
        """Initialise the overlay, build the UI, and add one default target row.

        Sets up the drop-shadow card appearance, creates the assets directory
        if it does not already exist, and delegates full UI construction to
        ``_init_ui``.  The widget starts hidden; callers must call ``show()``
        explicitly.

        Args:
            ingredients_by_type (dict[str, list]): Mapping of ingredient
                category name to a list of ingredient objects.  Forwarded to
                constraint-row helpers for populating value combo boxes.
            parent (QtWidgets.QWidget | None): Optional Qt parent widget.
                Defaults to ``None``.
        """
        super().__init__(parent)
        self.ingredients_by_type = ingredients_by_type

        self.assets_path = os.path.join(Architecture.get_path(), "QATCH", "VisQAI", "assets")
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

    def _init_ui(self):
        """Build and wire all child widgets in a top-to-bottom ``QVBoxLayout``."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 20)
        layout.setSpacing(15)

        # Header
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

        # Configuration
        grp_cfg = QtWidgets.QGroupBox("Configuration")
        cfg_form = QtWidgets.QFormLayout(grp_cfg)
        cfg_form.setSpacing(12)

        model_row = QtWidgets.QHBoxLayout()
        self.model_combo = QtWidgets.QComboBox()
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

        self.spin_maxiter = QtWidgets.QSpinBox()
        self.spin_maxiter.setRange(1, 1000)
        self.spin_maxiter.setValue(10)
        self.spin_maxiter.setFixedWidth(100)
        self.spin_maxiter.setFixedHeight(26)
        self.spin_maxiter.setToolTip("Maximum iterations for differential evolution")
        cfg_form.addRow("Max Iterations:", self.spin_maxiter)
        layout.addWidget(grp_cfg)

        # Viscosity Targets
        grp_tgt = QtWidgets.QGroupBox("Viscosity Targets  (1 - 5 shear-rate / target pairs)")
        tgt_vbox = QtWidgets.QVBoxLayout(grp_tgt)
        tgt_vbox.setContentsMargins(15, 15, 15, 10)
        tgt_vbox.setSpacing(6)

        # Column header row
        hdr_row = QtWidgets.QHBoxLayout()
        hdr_row.setContentsMargins(0, 0, 0, 0)
        for text, width in [("Shear Rate", 148), ("Target Viscosity (cP)", 0)]:
            lbl = QtWidgets.QLabel(text)
            lbl.setProperty("class", "row-label")
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
        self.targets_scroll.setFixedHeight(30)

        self.targets_container = QtWidgets.QWidget()
        self.targets_layout = QtWidgets.QVBoxLayout(self.targets_container)
        self.targets_layout.setContentsMargins(0, 0, 4, 0)
        self.targets_layout.setSpacing(6)

        self.lbl_no_targets = QtWidgets.QLabel(
            "No targets added.  Add at least one viscosity target."
        )
        self.lbl_no_targets.setObjectName("lblNone")
        self.targets_layout.addWidget(self.lbl_no_targets)
        self.targets_layout.addStretch()

        self.targets_scroll.setWidget(self.targets_container)
        tgt_vbox.addWidget(self.targets_scroll)
        layout.addWidget(grp_tgt)

        # Constraints
        grp_con = QtWidgets.QGroupBox("Constraints  (Optional)")
        con_vbox = QtWidgets.QVBoxLayout(grp_con)
        con_vbox.setContentsMargins(15, 15, 15, 15)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.constraints_container = QtWidgets.QWidget()
        self.constraints_layout = QtWidgets.QVBoxLayout(self.constraints_container)
        self.constraints_layout.setContentsMargins(0, 0, 5, 0)
        self.constraints_layout.setSpacing(8)

        self.lbl_none = QtWidgets.QLabel("No constraints added.")
        self.lbl_none.setObjectName("lblNone")
        self.constraints_layout.addWidget(self.lbl_none)
        self.constraints_layout.addStretch()

        self.scroll_area.setWidget(self.constraints_container)
        con_vbox.addWidget(self.scroll_area)
        layout.addWidget(grp_con)

        # Footer — Add Target | Add Constraint ——stretch—— Optimize
        layout.addSpacing(5)
        footer = QtWidgets.QHBoxLayout()
        footer.setSpacing(8)

        self.btn_add_target = QtWidgets.QPushButton("+ Add Target")
        self.btn_add_target.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_add_target.setFixedHeight(34)
        self.btn_add_target.clicked.connect(self.add_target_row)

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

        footer.addWidget(self.btn_add_target)
        footer.addWidget(self.btn_add_constraint)
        footer.addStretch()
        footer.addWidget(self.btn_optimize)
        layout.addLayout(footer)

        self._update_scroll_height()
        self._validate()
        self.add_target_row()

    def _populate_model_list(self):
        """Scan the assets directory and refresh the model combo box.

        Clears the current combo contents, then glob-searches ``assets_path``
        for ``*.visq`` files.  If none are found the combo is disabled and
        shows ``"No models found"``; otherwise each file is added by basename
        (with the full path stored as ``UserRole`` data) and
        ``"VisQAI(base).visq"`` is pre-selected when present.
        """
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
        """Open a file dialog so the user can import a new ``.visq`` model.

        Tries ``ModelSelectionDialog`` first; falls back to the plain
        ``QFileDialog.getOpenFileName`` if that dialog raises.  On a valid
        selection the chosen file is copied into ``assets_path`` (unless it is
        already there), the model combo is refreshed, and the newly imported
        file is made the active selection.  A critical ``QMessageBox`` is shown
        if the copy fails.
        """
        fname = None
        try:
            dlg = ModelSelectionDialog()
            dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            dlg.setNameFilter("VisQAI Models (*.visq)")
            dlg.setViewMode(QtWidgets.QFileDialog.Detail)
            mp = os.path.join(Architecture.get_path(), "QATCH", "VisQAI", "assets")
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

    def add_target_row(self):
        """Append a new shear-rate / target-viscosity row to the target list."""
        if len(self.target_rows) >= MAX_TARGETS:
            return

        self.lbl_no_targets.hide()

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
        row_l.setSpacing(8)

        cb_shear = QtWidgets.QComboBox()
        cb_shear.setFixedWidth(148)
        cb_shear.addItems(SHEAR_RATE_LABELS)
        cb_shear.setCurrentIndex(2)

        lbl_arrow = QtWidgets.QLabel("->")
        lbl_arrow.setObjectName("lblArrow")

        spin_visc = QtWidgets.QDoubleSpinBox()
        spin_visc.setRange(0.01, 100_000.0)
        spin_visc.setDecimals(2)
        spin_visc.setValue(1.0)
        spin_visc.setSuffix(" cP")
        spin_visc.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        spin_visc.setFixedWidth(120)
        spin_visc.setToolTip("Target viscosity at the selected shear rate")

        btn_del = QtWidgets.QToolButton()
        btn_del.setObjectName("btnConstraintDelete")
        btn_del.setIcon(QtGui.QIcon(_icon_del))
        btn_del.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        row_l.addWidget(cb_shear)
        row_l.addWidget(lbl_arrow)
        row_l.addWidget(spin_visc)
        row_l.addStretch()
        row_l.addWidget(btn_del)

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
        """Remove a target row from the list and update related UI state.

        Schedules the row widget for deletion with ``deleteLater``, removes
        the dict from ``target_rows``, re-shows ``lbl_no_targets`` when the
        list becomes empty, re-enables ``btn_add_target``, and refreshes the
        scroll height and validation state.

        Args:
            row_data (dict): The target-row state dict to remove, as previously
                appended by ``add_target_row``.
        """
        row_data["widget"].deleteLater()
        self.target_rows.remove(row_data)
        if not self.target_rows:
            self.lbl_no_targets.show()
        self.btn_add_target.setEnabled(True)
        self._update_target_scroll_height()
        self._validate()

    def _update_target_scroll_height(self):
        """Resize the targets scroll area to fit the current number of rows.

        Sets the fixed height of ``targets_scroll`` to 30 px when empty, or to
        ``min(n, 3) * 38 + 8`` px for *n* rows (capping visible rows at three
        before scrolling activates).  Calls ``adjustSize()`` and emits
        ``resized`` so the dashboard can reposition the overlay.
        """
        n = len(self.target_rows)
        if n == 0:
            self.targets_scroll.setFixedHeight(30)
        else:
            self.targets_scroll.setFixedHeight(min(n, 3) * 38 + 8)
        self.adjustSize()
        self.resized.emit()

    def add_constraint_row(self):
        """Append a new ingredient-constraint row to the constraint list."""
        self.lbl_none.hide()

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

        cb_attribute = QtWidgets.QComboBox()
        cb_attribute.addItem("Attribute...")
        cb_attribute.model().item(0).setEnabled(False)

        cb_condition = QtWidgets.QComboBox()
        cb_condition.addItem("Condition...")
        cb_condition.model().item(0).setEnabled(False)

        val_stack = QtWidgets.QStackedWidget()
        cb_value = _CompactCheckableComboBox()
        spin_value = QtWidgets.QDoubleSpinBox()
        spin_value.setRange(0.0, 10_000.0)
        spin_value.setDecimals(3)
        spin_value.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        val_stack.addWidget(cb_value)
        val_stack.addWidget(spin_value)

        btn_del = QtWidgets.QToolButton()
        btn_del.setObjectName("btnConstraintDelete")
        btn_del.setIcon(QtGui.QIcon(_icon_del))
        btn_del.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

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
        cb_ingredient.currentIndexChanged.connect(lambda: self._on_ingredient_changed(row_data))
        cb_attribute.currentIndexChanged.connect(lambda: self._on_attribute_changed(row_data))
        cb_condition.currentIndexChanged.connect(self._validate)
        cb_value.model().dataChanged.connect(self._validate)
        spin_value.valueChanged.connect(lambda _: self._validate())

        self._update_scroll_height()
        self._validate()

    def _remove_constraint_row(self, row_data):
        """Remove a constraint row from the list and update related UI state.

        Schedules the row widget for deletion with ``deleteLater``, removes
        the dict from ``constraint_rows``, re-shows ``lbl_none`` when the list
        becomes empty, and refreshes the scroll height and validation state.

        Args:
            row_data (dict): The constraint-row state dict to remove, as
                previously appended by ``add_constraint_row``.
        """
        row_data["widget"].deleteLater()
        self.constraint_rows.remove(row_data)
        if not self.constraint_rows:
            self.lbl_none.show()
        self._update_scroll_height()
        self._validate()

    def _on_ingredient_changed(self, row_data):
        """Repopulate the Attribute combo when the Ingredient selection changes.

        Clears and rebuilds ``cb_attribute`` based on the newly selected
        ingredient type.  Proteins additionally expose a ``"Class"`` attribute.
        Signals are blocked during the rebuild to prevent spurious cascades.
        Delegates to ``_on_attribute_changed`` at the end to keep the
        downstream Condition and Value widgets in sync.

        Args:
            row_data (dict): The constraint-row state dict containing the
                ``ingredient`` and ``attribute`` combo box references.
        """
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
            elif ing_type == "Buffer":
                attrs.append("pH")
            cb_attr.addItems(attrs)
        cb_attr.setCurrentIndex(0)
        cb_attr.blockSignals(False)
        self._on_attribute_changed(row_data)

    def _on_attribute_changed(self, row_data):
        """Repopulate the Condition combo and switch the Value widget stack.

        Updates ``cb_condition`` with comparison operators appropriate for the
        selected attribute:

        * ``"Concentration"`` -> numeric operators (``>``, ``>=``, …) and
          switches ``value_stack`` to index 1 (spin box).
        * ``"Type"`` or ``"Class"`` -> ``"is"`` / ``"is not"`` and switches
          ``value_stack`` to index 0 (checkable combo).

        Signals are blocked during the rebuild, then ``_populate_values`` is
        called to fill the value widget with the appropriate options.

        Args:
            row_data (dict): The constraint-row state dict containing the
                ``attribute``, ``condition``, and ``value_stack`` references.
        """
        attr_type = row_data["attribute"].currentText()
        cb_condition = row_data["condition"]
        val_stack = row_data["value_stack"]
        cb_condition.blockSignals(True)
        cb_condition.clear()
        cb_condition.addItem("Condition...")
        cb_condition.model().item(0).setEnabled(False)
        if row_data["attribute"].currentIndex() > 0:
            if attr_type == "Concentration":
                cb_condition.addItems([">", ">=", "=", "!=", "<=", "<"])
                val_stack.setCurrentIndex(1)
                row_data["value_spin"].setRange(0.0, 10000.0)
            elif attr_type == "pH":
                cb_condition.addItems([">", ">=", "=", "!=", "<=", "<"])
                val_stack.setCurrentIndex(1)
                row_data["value_spin"].setRange(4.0, 8.0)
            elif attr_type in [
                "Type",
                "Class",
            ]:
                cb_condition.addItems(["is", "is not"])
                val_stack.setCurrentIndex(0)
        cb_condition.setCurrentIndex(0)
        cb_condition.blockSignals(False)
        self._populate_values(row_data)

    def _populate_values(self, row_data):
        """Fill the Value combo box with options for the current attribute.

        Only operates when both an ingredient and an attribute have been
        chosen (indices > 0).  For ``"Class"`` attributes on Proteins,
        collects unique ``class_type`` values from the ingredient objects in
        ``ingredients_by_type``; for ``"Type"`` attributes, uses ingredient
        names; numeric attributes leave the spin box unchanged.  Calls
        ``_validate`` at the end.

        Args:
            row_data (dict): The constraint-row state dict containing the
                ``ingredient``, ``attribute``, ``value_stack``, and
                ``value_cb`` references.
        """
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
        """Resize the constraints scroll area to fit the current number of rows.

        Sets the fixed height of ``scroll_area`` to 30 px when empty, or to
        ``min(n, 3) * 36 + 10`` px for *n* rows (capping visible rows at three
        before scrolling activates).  Calls ``adjustSize()`` and emits
        ``resized`` so the dashboard can reposition the overlay.
        """
        n = len(self.constraint_rows)
        self.scroll_area.setFixedHeight(30 if n == 0 else min(n, 3) * 36 + 10)
        self.adjustSize()
        self.resized.emit()

    @staticmethod
    def _checked_items(combo_box) -> list:
        """Return the text of every item whose check state is ``Qt.Checked``.

        ``CheckableComboBox.getItems()`` returns all items in the model
        regardless of check state and is therefore only useful as a
        "has any items" test.  This helper inspects the underlying
        ``QStandardItemModel`` directly to find which items the user has
        actually ticked.

        Args:
            combo_box (CheckableComboBox): The checkable combo box to inspect.

        Returns:
            list[str]: Ordered list of display texts for all checked items.
                Empty list if none are checked.
        """
        from PyQt5.QtCore import Qt

        checked = []
        model = combo_box.model()
        for i in range(model.rowCount()):
            item = model.item(i)
            if item is not None and item.checkState() == Qt.Checked:
                checked.append(item.text())
        return checked

    def _validate(self):
        """Enable or disable footer buttons based on current form completeness.

        Enables ``btn_optimize`` only when all three conditions hold:

        1. At least one target row exists.
        2. The model combo is enabled (i.e. at least one ``.visq`` file was
           found or imported).
        3. Every constraint row has a valid ingredient, attribute, condition,
           and value selection (categorical rows must have at least one item
           checked).

        ``btn_add_constraint`` is also disabled whenever any existing
        constraint row is incomplete, preventing partially filled rows from
        stacking  up.
        """
        has_targets = len(self.target_rows) > 0
        has_model = self.model_combo.isEnabled()

        constraints_ok = True
        for row in self.constraint_rows:
            ing_ok = row["ingredient"].currentIndex() > 0
            attr_ok = row["attribute"].currentIndex() > 0
            cond_ok = row["condition"].currentIndex() > 0
            if row["value_stack"].currentIndex() == 0:
                val_ok = len(self._checked_items(row["value_cb"])) > 0
            else:
                val_ok = True
            if not (ing_ok and attr_ok and cond_ok and val_ok):
                constraints_ok = False
                break

        self.btn_add_constraint.setEnabled(constraints_ok)
        self.btn_optimize.setEnabled(has_targets and has_model and constraints_ok)

    def emit_optimize(self):
        """Collect form data and emit ``optimize_requested``.

        Iterates over ``target_rows`` to build a list of
        ``{"shear_rate": int, "viscosity": float}`` dicts, falling back to
        10 000 s 1/s if a combo index is somehow out of range.  Iterates over
        ``constraint_rows`` to build a list of
        ``{"ingredient": str, "attribute": str, "condition": str, "values": ...}``
        dicts, where ``values`` is a list of strings for categorical constraints
        or a single float for numeric ones.

        Emits:
            optimize_requested (str, list, list): The selected model file path
                (from the combo's display text), the targets list, and the
                constraints list.
        """
        targets = []
        for row in self.target_rows:
            idx = row["shear_cb"].currentIndex()
            shear = SHEAR_RATE_OPTIONS[idx] if 0 <= idx < len(SHEAR_RATE_OPTIONS) else 10_000
            targets.append({"shear_rate": shear, "viscosity": row["visc_spin"].value()})

        constraints_data = []
        for row in self.constraint_rows:
            val = (
                self._checked_items(row["value_cb"])
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

        self.optimize_requested.emit(self.model_combo.currentText(), targets, constraints_data)

    def close_widget(self):
        """Hide the overlay and notify the parent dashboard.

        Calls ``hide()`` to make the widget invisible, then emits ``closed``
        so the dashboard can reset any toggle button or pointer state without
        needing to poll widget visibility.

        Emits:
            closed (): Unconditionally after hiding the widget.
        """
        self.hide()
        self.closed.emit()
