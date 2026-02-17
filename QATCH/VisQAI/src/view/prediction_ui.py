import os
import sys

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

# --- 1. NEW IMPORTS ADDED HERE ---
try:
    from architecture import Architecture
    from src.controller.formulation_controller import FormulationController
    from src.controller.ingredient_controller import IngredientController

    # Database Integration Imports
    from src.db.db import Database
    from src.io.parser import Parser
    from styles.style_loader import load_stylesheet
    from widgets.evaluation_widget import EvaluationWidget
    from widgets.formulation_config_card_widget import (
        FormulationConfigCard,
    )
    from widgets.placeholder_widget import PlaceholderWidget
    from widgets.prediction_filter_widget import PredictionFilterWidget
    from widgets.reordable_container_widget import ReorderableCardContainer
    from widgets.visualization_panel import VisualizationPanel
    from workers.import_worker import ImportWorker
    from workers.prediction_worker import PredictionThread
except (ImportError, ModuleNotFoundError):
    from QATCH.common.architecture import Architecture
    from QATCH.common.logger import Logger as Log
    from QATCH.common.userProfiles import UserPreferences, UserProfiles
    from QATCH.core.constants import Constants
    from QATCH.VisQAI.src.controller.formulation_controller import FormulationController
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.io.parser import Parser
    from QATCH.VisQAI.src.view.styles.style_loader import load_stylesheet
    from QATCH.VisQAI.src.view.widgets.evaluation_widget import EvaluationWidget
    from QATCH.VisQAI.src.view.widgets.formulation_config_card_widget import (
        FormulationConfigCard,
    )
    from QATCH.VisQAI.src.view.widgets.placeholder_widget import PlaceholderWidget
    from QATCH.VisQAI.src.view.widgets.prediction_filter_widget import (
        PredictionFilterWidget,
    )
    from QATCH.VisQAI.src.view.widgets.reordable_container_widget import (
        ReorderableCardContainer,
    )
    from QATCH.VisQAI.src.view.widgets.visualization_panel import VisualizationPanel
    from QATCH.VisQAI.src.view.workers.import_worker import ImportWorker
    from QATCH.VisQAI.src.view.workers.prediction_worker import PredictionThread

TAG = "[VisQ.AI]"


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
        self.selection_mode_active = False
        self._pending_color = None

        self.db = Database(parse_file_key=True)
        self.ing_ctrl = IngredientController(self.db)
        self.form_ctrl = FormulationController(self.db)
        self._load_database_data()

        self.init_ui()
        self.setStyleSheet(load_stylesheet())
        self._batch_queue = []
        self._batch_results = []
        self._is_batch_collecting = False
        self._is_batch_running = False

    def _load_database_data(self):
        """Initializes DB connection and fetches ingredients for dropdowns."""

        # Fetch lists for each ingredient type
        self.ingredients_by_type["Protein"] = self.ing_ctrl.get_all_proteins()
        self.ingredients_by_type["Buffer"] = self.ing_ctrl.get_all_buffers()
        self.ingredients_by_type["Salt"] = self.ing_ctrl.get_all_salts()
        self.ingredients_by_type["Surfactant"] = self.ing_ctrl.get_all_surfactants()
        self.ingredients_by_type["Stabilizer"] = self.ing_ctrl.get_all_stabilizers()
        self.ingredients_by_type["Excipient"] = self.ing_ctrl.get_all_excipients()

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Create Top Bar (Buttons Only)
        # Note: We do NOT create eval_widget here anymore
        top_bar = self._create_unified_top_bar()
        main_layout.addWidget(top_bar)

        # 2. Splitter Configuration
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet("QSplitter::handle { background-color: #d1d5db; }")

        # --- Left Panel ---
        self.left_widget = QtWidgets.QWidget()
        self.left_widget.setObjectName("leftPanel")
        left_layout = QtWidgets.QVBoxLayout(self.left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # --- OVERLAY WIDGETS (Created NOW, parenting to left_widget) ---

        # 1. Filter Widget
        self.filter_widget = PredictionFilterWidget(
            self.ingredients_by_type, parent=self.left_widget
        )
        self.filter_widget.filter_changed.connect(self.apply_filters)
        self.filter_widget.hide()

        # 2. Evaluation Widget (Fixed: Created here, safely parenting to left_widget)
        self.eval_widget = EvaluationWidget(parent=self)
        self.eval_widget.run_requested.connect(self.run_evaluation_analysis)
        self.eval_widget.closed.connect(self.exit_evaluation_mode)
        self.eval_widget.hide()

        # --- Content ---
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        self.cards_container = ReorderableCardContainer()
        self.cards_container.setObjectName("scrollContent")
        self.cards_layout = self.cards_container.main_layout

        self.scroll_area.setWidget(self.cards_container)
        self.cards_layout.setContentsMargins(15, 15, 15, 15)
        self.cards_layout.setSpacing(10)

        self.placeholder = PlaceholderWidget()
        self.cards_layout.addWidget(self.placeholder)
        self.placeholder.hide()

        left_layout.addWidget(self.scroll_area)
        self._create_fab()

        # Install Event Filter for Resizing Overlays
        self.left_widget.installEventFilter(self)
        splitter.addWidget(self.left_widget)

        # --- Right Panel ---
        self.right_widget = QtWidgets.QWidget()
        self.right_widget.setObjectName("rightPanel")
        self.right_widget.setStyleSheet("background-color: #ffffff;")
        right_layout = QtWidgets.QVBoxLayout(self.right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.viz_panel = VisualizationPanel()
        right_layout.addWidget(self.viz_panel)

        splitter.addWidget(self.right_widget)
        splitter.setSizes([450, 700])
        main_layout.addWidget(splitter)

        self.current_task = None
        self.update_placeholder_visibility()

    def _create_unified_top_bar(self):
        """Creates the single unified top toolbar."""
        container = QtWidgets.QWidget()
        container.setObjectName("topBar")
        container.setFixedHeight(50)
        container.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)

        shadow = QtWidgets.QGraphicsDropShadowEffect(container)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QtGui.QColor(0, 0, 0, 25))
        container.setGraphicsEffect(shadow)

        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(15, 5, 15, 5)
        layout.setSpacing(10)

        # --- Left Side Controls ---
        self.search_bar = QtWidgets.QLineEdit()
        self.search_bar.setObjectName("searchBar")
        self.search_bar.setPlaceholderText("Search...")
        self.search_bar.setClearButtonEnabled(True)
        self.search_bar.addAction(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "search-svgrepo-com.svg",
                )
            ),
            QtWidgets.QLineEdit.ActionPosition.LeadingPosition,
        )
        self.search_bar.textChanged.connect(self.filter_cards)
        layout.addWidget(self.search_bar, stretch=1)

        # Filter Button
        self.btn_filter = QtWidgets.QToolButton()
        self.btn_filter.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "filter-svgrepo-com.svg",
                )
            )
        )
        self.btn_filter.setToolTip("Filter Options")
        self.btn_filter.setCheckable(True)
        self.btn_filter.setFixedSize(32, 32)  # <--- Added to match others
        self.btn_filter.clicked.connect(self.toggle_filter_menu_manual)
        layout.addWidget(self.btn_filter)

        # Select Mode Button
        self.btn_select_mode = QtWidgets.QToolButton()
        self.btn_select_mode.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "select-svgrepo-com.svg",
                )
            )
        )
        self.btn_select_mode.setToolTip("Enter Selection Mode")
        self.btn_select_mode.setCheckable(True)
        self.btn_select_mode.setFixedSize(32, 32)  # <--- Added to match others
        self.btn_select_mode.toggled.connect(self.toggle_selection_mode)
        layout.addWidget(self.btn_select_mode)

        self.btn_select_all = QtWidgets.QToolButton()
        self.btn_select_all.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "select-multiple-svgrepo-com.svg",
                )
            )
        )
        self.btn_select_all.setToolTip("Select All")
        self.btn_select_all.setFixedSize(32, 32)
        self.btn_select_all.clicked.connect(self.select_all_cards)
        self.btn_select_all.setEnabled(False)
        layout.addWidget(self.btn_select_all)

        self.btn_import = QtWidgets.QToolButton()
        self.btn_import.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "import-content-svgrepo-com.svg",
                )
            )
        )
        self.btn_import.setToolTip("Import Data")
        self.btn_import.setFixedSize(32, 32)
        self.btn_import.clicked.connect(self.import_run)
        layout.addWidget(self.btn_import)

        self.btn_export_top = QtWidgets.QToolButton()
        self.btn_export_top.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "export-content-svgrepo-com.svg",
                )
            )
        )
        self.btn_export_top.setToolTip("Export Selected")
        self.btn_export_top.setFixedSize(32, 32)
        self.btn_export_top.clicked.connect(self.export_analysis)
        layout.addWidget(self.btn_export_top)

        # --- NEW BUTTONS (Inline Styles Removed to use theme.qss) ---

        # Generate Sample
        self.btn_generate = QtWidgets.QToolButton()
        self.btn_generate.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "flask-sample-test-svgrepo-com.svg",
                )
            )
        )
        self.btn_generate.setToolTip("Generate Sample")
        self.btn_generate.setFixedSize(32, 32)
        # No setStyleSheet here; QToolButton in theme.qss handles radius & hover
        self.btn_generate.clicked.connect(self.handle_generate_sample)
        layout.addWidget(self.btn_generate)

        # Evaluate
        self.btn_evaluate = QtWidgets.QToolButton()
        self.btn_evaluate.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "scorecard-svgrepo-com.svg",
                )
            )
        )
        self.btn_evaluate.setToolTip("Evaluate")
        self.btn_evaluate.setFixedSize(32, 32)
        self.btn_evaluate.setCheckable(True)
        # No setStyleSheet here; theme.qss handles Checked state styling
        self.btn_evaluate.clicked.connect(self.handle_evaluate)
        layout.addWidget(self.btn_evaluate)

        # Hypothesis
        self.btn_hypothesis = QtWidgets.QToolButton()
        self.btn_hypothesis.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "dice-svgrepo-com.svg",
                )
            )
        )
        self.btn_hypothesis.setToolTip("Hypothesis")
        self.btn_hypothesis.setFixedSize(32, 32)
        # No setStyleSheet here
        self.btn_hypothesis.clicked.connect(self.handle_hypothesis)
        layout.addWidget(self.btn_hypothesis)

        # Optimize
        self.btn_optimize = QtWidgets.QToolButton()
        self.btn_optimize.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "optimize-svgrepo-com.svg",
                )
            )
        )
        self.btn_optimize.setToolTip("Optimize")
        self.btn_optimize.setFixedSize(32, 32)
        # No setStyleSheet here
        self.btn_optimize.clicked.connect(self.handle_optimize)
        layout.addWidget(self.btn_optimize)

        # --- Separator ---
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        line.setFixedHeight(20)
        layout.addWidget(line)

        # Run & Delete
        self.btn_run_top = QtWidgets.QToolButton()
        self.btn_run_top.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "play-svgrepo-com.svg",
                )
            )
        )
        self.btn_run_top.setToolTip("Run Inference (Selected or Open Card)")
        self.btn_run_top.setFixedSize(32, 32)
        self.btn_run_top.clicked.connect(self.run_analysis)
        layout.addWidget(self.btn_run_top)

        self.btn_delete_top = QtWidgets.QToolButton()
        self.btn_delete_top.setObjectName("btnDelete")
        self.btn_delete_top.setIcon(
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
        self.btn_delete_top.setToolTip("Delete (Selected or Open Card)")
        self.btn_delete_top.setFixedSize(32, 32)
        self.btn_delete_top.clicked.connect(self.delete_analysis)
        layout.addWidget(self.btn_delete_top)

        # --- Spacer ---
        layout.addStretch(1)

        # Options
        self.btn_right_options = QtWidgets.QToolButton()
        self.btn_right_options.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(),
                    "QATCH",
                    "VisQAI",
                    "src",
                    "view",
                    "icons",
                    "three-dots-svgrepo-com.svg",
                )
            )
        )
        self.btn_right_options.setToolTip("More Options")
        self.btn_right_options.setFixedSize(32, 32)
        layout.addWidget(self.btn_right_options)

        return container

    def _on_card_selection_changed(self):
        """
        Called whenever a card is selected/deselected.
        Auto-exits selection mode if the last item is deselected.
        """
        if not self.selection_mode_active:
            return

        selected_count = 0
        for i in range(self.cards_layout.count()):
            widget = self.cards_layout.itemAt(i).widget()
            if isinstance(widget, FormulationConfigCard) and widget.is_selected:
                selected_count += 1

        if selected_count == 0:
            self.btn_select_mode.setChecked(False)

    def handle_generate_sample(self):
        """Handler for the Generate Sample button."""
        # TODO: Replace with: dialog = GenerateSampleDialog(self); dialog.exec_()
        QtWidgets.QMessageBox.information(
            self,
            "Generate Sample",
            "Opening Sample Generation Wizard...\n\n(This feature is under construction)",
        )

    def handle_evaluate(self):
        """
        Toggles the Evaluation Widget and applies 'Measured Data Only' filter.
        """
        is_active = self.btn_evaluate.isChecked()

        if is_active:
            # 1. Show Evaluation Widget
            self.eval_widget.show()
            self.eval_widget.raise_()
            self._update_overlay_geometry()

            # 2. Hide Filter Widget if open
            if self.filter_widget.isVisible():
                self.filter_widget.hide()
                self.btn_filter.setChecked(False)

            # 3. Apply Filter: Show ONLY cards with measured data
            visible_count = 0
            for i in range(self.cards_layout.count()):
                item = self.cards_layout.itemAt(i)
                widget = item.widget()
                if isinstance(widget, FormulationConfigCard):
                    # Logic: Check if card has measured data marked
                    # Assuming card has a method/property or we check internal state
                    # Here we check if the 'measured' toggle is True in the card config
                    # OR if the card has loaded imported data.
                    has_data = widget.is_measured  # Using UI state as proxy

                    if has_data:
                        widget.show()
                        visible_count += 1
                    else:
                        widget.hide()

            self.viz_panel.set_plot_title(
                f"Evaluation Mode: {visible_count} Datasets Ready"
            )

        else:
            self.exit_evaluation_mode()

        self.update_placeholder_visibility()

    def handle_hypothesis(self):
        """Handler for the Hypothesis button."""
        # TODO: Replace with: dialog = HypothesisInputDialog(self); dialog.exec_()
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Hypothesis Testing", "Enter hypothesis name or ID:"
        )
        if ok and text:
            print(f"Hypothesis '{text}' initialized.")

    def handle_optimize(self):
        """Handler for the Optimize button."""
        target_cards = self.get_target_cards()
        if not target_cards:
            return

        # TODO: Replace with: dialog = OptimizerConfigDialog(target_cards[0], self); dialog.exec_()
        reply = QtWidgets.QMessageBox.question(
            self,
            "Optimize Formulation",
            "Run Bayesian Optimization on the selected formulation to minimize viscosity?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.viz_panel.set_plot_title("Optimization Initialized...")
            # Trigger your optimization worker here

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
        self.btn_add_fab.setObjectName("fabAdd")
        self.btn_add_fab.clicked.connect(lambda: self.add_prediction_card(None))
        self.btn_add_fab.show()

    def eventFilter(self, source, event):
        """
        Handle resize events for the left panel to resize the filter overlay
        and reposition the Floating Action Button (FAB).
        """
        if source == self.left_widget and event.type() == QtCore.QEvent.Type.Resize:
            self._update_filter_geometry()
            fab_size = self.btn_add_fab.size()
            margin = 20
            x = self.left_widget.width() - fab_size.width() - margin
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

            if isinstance(widget, FormulationConfigCard):
                total_cards += 1
                if not widget.isHidden():
                    visible_count += 1

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

    def check_evaluation_eligibility(self):
        """
        Checks if any card has measured data enabled.
        Enables/Disables the Evaluate button accordingly.
        """
        has_measured_data = False

        # Iterate over all cards in the layout
        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()

            if isinstance(widget, FormulationConfigCard):
                # FIX: Check the internal boolean property instead of a checkbox
                if widget.is_measured:
                    has_measured_data = True
                    break

        # Update Button State
        if hasattr(self, "btn_evaluate"):
            self.btn_evaluate.setEnabled(has_measured_data)

            # If we just disabled the button and it was currently active (checked),
            # we must force exit the evaluation mode to restore the UI.
            if not has_measured_data and self.btn_evaluate.isChecked():
                self.btn_evaluate.setChecked(False)
                self.exit_evaluation_mode()

    def _is_filter_default(self, filters):
        """Checks if the provided filter dict matches the default state."""
        if not filters["show_measured"] or not filters["show_predicted"]:
            return False
        if filters["model_text"] != "":
            return False
        if filters["temp_min"] != 0 or filters["temp_max"] != 100:
            return False
        if filters["ingredients"]:
            for selected_list in filters["ingredients"].values():
                if selected_list:
                    return False
        return True

    def exit_evaluation_mode(self):
        """Restores UI to normal state."""
        self.eval_widget.hide()
        self.btn_evaluate.setChecked(False)

        # Restore all cards visibility (or re-apply standard filters)
        # For simplicity, show all, or trigger re-filter
        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, FormulationConfigCard):
                widget.show()

        self.viz_panel.set_plot_title("")
        self.update_placeholder_visibility()

    def run_evaluation_analysis(self, config):
        """
        Calculates metrics for visible cards and updates the plot.
        """
        metric_name = config["metric"]
        shear_min = config["shear_min"]
        shear_max = config["shear_max"]

        results_to_plot = []
        scores = []

        # Iterate over visible cards (which we know are the Measured ones)
        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            card = item.widget()

            # Skip hidden or non-card widgets
            if not isinstance(card, FormulationConfigCard) or card.isHidden():
                continue

            # Access stored results in the card (populated by previous runs)
            if not hasattr(card, "last_results") or not card.last_results:
                continue

            data = card.last_results

            # Ensure we have both predicted and measured data
            # Data format assumption: {'shear_rate': [], 'viscosity': [], 'measured_viscosity': []}
            if "measured_viscosity" not in data or "viscosity" not in data:
                continue

            shear = np.array(data["shear_rate"])
            y_pred = np.array(data["viscosity"])
            y_true = np.array(data["measured_viscosity"])

            # Filter by Shear Range
            mask = (shear >= shear_min) & (shear <= shear_max)
            if not np.any(mask):
                continue

            y_p_filt = y_pred[mask]
            y_t_filt = y_true[mask]

            # Calculate Metric
            score = 0.0
            if "RMSE" in metric_name:
                score = np.sqrt(np.mean((y_p_filt - y_t_filt) ** 2))
            elif "MAE" in metric_name:
                score = np.mean(np.abs(y_p_filt - y_t_filt))
            elif "MAPE" in metric_name:
                # Avoid division by zero
                valid = y_t_filt != 0
                score = (
                    np.mean(
                        np.abs((y_t_filt[valid] - y_p_filt[valid]) / y_t_filt[valid])
                    )
                    * 100
                )
            elif "R²" in metric_name or "R-Squared" in metric_name:
                ss_res = np.sum((y_t_filt - y_p_filt) ** 2)
                ss_tot = np.sum((y_t_filt - np.mean(y_t_filt)) ** 2)
                score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

            scores.append(f"{card.get_config()['name']}: {score:.4f}")

            # Add to plot list (we re-send the data to VizPanel)
            # We modify the name to include the score for the legend
            data_copy = data.copy()
            data_copy["config_name"] = (
                f"{data['config_name']} ({metric_name}={score:.2f})"
            )
            results_to_plot.append(data_copy)

        if not results_to_plot:
            QtWidgets.QMessageBox.warning(
                self, "Evaluation Failed", "No valid data found in range."
            )
            return

        # Update Visualization
        self.viz_panel.set_data(results_to_plot)

        # Calculate Average Score for Title
        avg_score = 0
        if scores:
            # Simple parsing for display
            vals = [float(s.split(": ")[1]) for s in scores]
            avg_score = sum(vals) / len(vals)

        self.viz_panel.set_plot_title(
            f"Evaluation Results: Avg {metric_name} = {avg_score:.4f}"
        )

    def apply_filters(self, filter_data):
        """Iterates over cards and toggles visibility based on match."""
        search_text = self.search_bar.text().lower()

        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()

            if widget and isinstance(widget, FormulationConfigCard):
                matches_complex = widget.matches_filter(filter_data)
                matches_search = True
                if search_text:
                    matches_search = search_text in widget.get_searchable_text()

                if matches_complex and matches_search:
                    widget.show()
                else:
                    widget.hide()

        self.update_placeholder_visibility()

        # Check if filters are currently default
        self.update_placeholder_visibility()
        is_default = (
            filter_data["show_measured"]
            and filter_data["show_predicted"]
            and filter_data["model_text"] == ""
            and filter_data["temp_min"] == 0
            and filter_data["temp_max"] == 100
            and not any(filter_data["ingredients"].values())
        )

        self.btn_filter.setProperty("active", not is_default)
        self.btn_filter.style().unpolish(self.btn_filter)
        self.btn_filter.style().polish(self.btn_filter)
        self.filter_widget.hide()

    def toggle_filter_menu_manual(self):
        """Toggles filter menu visibility on click."""
        if self.filter_widget.isVisible():
            self.filter_widget.hide()
        else:
            self.filter_widget.show()
            self.filter_widget.raise_()
            self._update_filter_geometry()

    def _update_filter_geometry(self):
        """Positions the filter widget directly below the top bar (which is outside the panel now)."""
        if self.filter_widget.isVisible():
            # Coordinate 0,0 is now the top of the left_widget (content area)
            self.filter_widget.setGeometry(
                0, 0, self.left_widget.width(), self.filter_widget.sizeHint().height()
            )

    def run_analysis(self):
        """
        Modified to support Batch Analysis of selected cards.
        """
        target_cards = self.get_target_cards()
        if not target_cards:
            return

        # 1. Initialize Batch State
        self._batch_queue = []
        self._batch_results = []
        self._is_batch_collecting = True  # Start listening for run requests

        # 2. Trigger requests from cards (which call run_prediction via signals)
        # Because _is_batch_collecting is True, run_prediction will Queue them instead of running.
        for card in target_cards:
            card.emit_run_request()

        self._is_batch_collecting = False  # Stop listening

        # 3. Start processing the queue
        if self._batch_queue:
            self._is_batch_running = True
            self.viz_panel.set_plot_title("Initializing Batch Analysis...")
            self.viz_panel.show_loading()
            self._process_next_in_batch()
        else:
            QtWidgets.QMessageBox.warning(
                self, "Error", "Failed to collect configuration data."
            )

    def _process_next_in_batch(self):
        """
        Pops the next item from the batch queue and runs it.
        """
        if not self._batch_queue:
            # --- BATCH COMPLETE ---
            self._is_batch_running = False
            self.viz_panel.hide_loading()

            # Summarize
            count = len(self._batch_results)
            self.viz_panel.set_plot_title(f"Analysis Results ({count} Profiles)")

            # Send LIST of results to panel (triggers multi-plot)
            self.viz_panel.set_data(self._batch_results)
            return

        # Get next item
        card, config = self._batch_queue.pop(0)

        # Manually set the running card so results go back to the correct UI card
        self.running_card = card

        # Update Title for progress
        self.viz_panel.set_plot_title(f"Calculating: {config.get('name')}...")

        # Run specific card (batch collecting is False, so this will actually run)
        self.run_prediction(config)

    def delete_analysis(self):
        """Deletes Selected cards (if any) or the currently Open card and updates the plot."""
        target_cards = []

        # 1. Identify Target Cards
        if self.selection_mode_active:
            for i in range(self.cards_layout.count()):
                widget = self.cards_layout.itemAt(i).widget()
                if isinstance(widget, FormulationConfigCard) and widget.is_selected:
                    target_cards.append(widget)
        else:
            for i in range(self.cards_layout.count()):
                widget = self.cards_layout.itemAt(i).widget()
                if isinstance(widget, FormulationConfigCard) and widget.is_expanded:
                    target_cards.append(widget)
                    break

        if not target_cards:
            return

        # 2. Confirm Deletion
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
            # --- UPDATE PLOT: Remove data associated with deleted cards ---

            # Gather references to the data objects and names from the cards being deleted
            results_to_remove = []
            names_to_remove = []

            for card in target_cards:
                # If the card has a stored result object, we target that specific object
                if hasattr(card, "last_results") and card.last_results:
                    results_to_remove.append(card.last_results)
                # We also track the name as a fallback
                names_to_remove.append(card.name_input.text())

            # Filter the current data series in the Visualization Panel
            current_series = self.viz_panel.data_series
            new_series = []

            for data_pkg in current_series:
                # FIX: Use 'is' identity check to avoid numpy array comparison errors
                # Replaces: if data_pkg in results_to_remove:
                if any(data_pkg is res for res in results_to_remove):
                    continue

                # Exclude if the name matches (fallback if object identity fails)
                if data_pkg.get("config_name") in names_to_remove:
                    continue

                new_series.append(data_pkg)

            # Update the visualization with the filtered list
            self.viz_panel.set_data(new_series)

            # --- REMOVE WIDGETS ---
            for card in target_cards:
                self.remove_card(card)

            # Cleanup selection state
            if self.selection_mode_active:
                self.btn_select_mode.setChecked(False)
            elif self.cards_layout.count() > 0:
                pass

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
            if widget and isinstance(widget, FormulationConfigCard):
                if hasattr(widget, "set_selectable"):
                    widget.set_selectable(active)

    def select_all_cards(self):
        """Selects all visible cards. If all are already selected, deselects all."""
        visible_cards = []
        for i in range(self.cards_layout.count()):
            w = self.cards_layout.itemAt(i).widget()
            if isinstance(w, FormulationConfigCard) and not w.isHidden():
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

            if widget and isinstance(widget, FormulationConfigCard):
                # Get content
                card_content = widget.get_searchable_text()

                # Check Search Match
                matches_search = True
                if search_tokens:
                    for token in search_tokens:
                        if token not in card_content:
                            matches_search = False
                            break

                # Check Filter Widget Match (if active)
                matches_filter = True
                if hasattr(self, "filter_widget") and self.filter_widget.isVisible():
                    pass

                # Apply Visibility
                if matches_search:
                    widget.show()
                else:
                    widget.hide()
        self.update_placeholder_visibility()

    def add_prediction_card(self, data=None):
        if data and "name" in data:
            name = data["name"]
        else:
            count = 0
            for i in range(self.cards_layout.count()):
                if isinstance(
                    self.cards_layout.itemAt(i).widget(), FormulationConfigCard
                ):
                    count += 1
            name = f"Prediction {count + 1}"

        card = FormulationConfigCard(
            default_name=name,
            ingredients_data=self.ingredients_by_type,
            ingredient_types=self.INGREDIENT_TYPES,
            ingredient_units=self.INGREDIENT_UNITS,
            parent=self,
        )
        card.removed.connect(self.remove_card)
        card.run_requested.connect(self.run_prediction)
        card.expanded.connect(self.on_card_expanded)
        card.selection_changed.connect(self._on_card_selection_changed)
        if card.is_measured:
            self.check_evaluation_eligibility()
        insert_idx = self.cards_layout.count()
        self.cards_layout.insertWidget(insert_idx, card)
        card.show()
        self.update_placeholder_visibility()

        if data:
            if hasattr(card, "load_data"):
                # load_data handles ingredient population if data['ingredients'] is a dict
                card.load_data(data)

            if data.get("measured", False):
                card.set_measured_state(True)

            # Only trigger a run if it's a prediction (not measured data)
            if not data.get("measured", False):
                card.emit_run_request()

        self.on_card_expanded(card)
        QtCore.QTimer.singleShot(100, lambda: self._scroll_to_card(card))

        self.update_placeholder_visibility()
        self.check_evaluation_eligibility()

        return card  # <--- CRITICAL: Return the card instance

    def _scroll_to_card(self, card_widget):
        """Helper to ensure the new card is visible in the scroll area."""
        self.scroll_area.ensureWidgetVisible(card_widget, 0, 0)

    def on_card_expanded(self, active_card):
        """
        Handles card expansion: collapses others and updates the visualization
        with the active card's data (including measured profiles).
        """
        # 1. Collapse other cards
        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, FormulationConfigCard) and widget is not active_card:
                widget.collapse()

        # 2. Automatically plot data if available (Fixes missing plot on open)
        if hasattr(active_card, "last_results") and active_card.last_results:
            # Sync name in case it changed
            data = active_card.last_results
            data["config_name"] = active_card.name_input.text()

            # Send to Visualization Panel immediately
            self.viz_panel.set_plot_title(data["config_name"])
            self.viz_panel.set_data(data)

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
            self.check_evaluation_eligibility()

        anim.finished.connect(cleanup)
        anim.start()

    def import_run(self):
        """
        Initiates the background import process for one or multiple directories.
        """
        # 1. Retrieve Load Path from Preferences
        try:
            if (
                not hasattr(UserProfiles, "user_preferences")
                or UserProfiles.user_preferences is None
            ):
                UserProfiles.user_preferences = UserPreferences(
                    UserProfiles.get_session_file()
                )

            prefs = UserProfiles.user_preferences.get_preferences()
            path_from_prefs = prefs.get("load_data_path")

            if (
                path_from_prefs
                and isinstance(path_from_prefs, str)
                and path_from_prefs.strip()
            ):
                self.load_data_path = path_from_prefs
            else:
                self.load_data_path = Constants.working_logged_data_path

        except Exception as e:
            Log.e(TAG, f"Error reading load path from preferences: {e}")
            self.load_data_path = Constants.working_logged_data_path

        # 2. Open Directory Selection Dialog
        dialog = QtWidgets.QFileDialog(
            self, "Select Run Directory(s)", self.load_data_path
        )
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)

        # Try to enable multi-selection for directories if the OS supports it
        # (Note: On standard Windows native dialogs, this often restricts to single folder,
        # but using the QFileDialog non-native view can allow multiple)
        # dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        # dialog.setViewMode(QtWidgets.QFileDialog.List)

        if dialog.exec_():
            fnames = dialog.selectedFiles()
        else:
            return

        if not fnames:
            return

        # 3. Setup Progress Dialog
        self.progress_dialog = QtWidgets.QProgressDialog(
            "Scanning directories...", "Cancel", 0, 100, self
        )
        self.progress_dialog.setWindowTitle("Importing Data")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)

        # 4. Configure and Start Worker
        # Pass directory paths; worker will discover files inside
        self.worker = ImportWorker(fnames)

        # Connect Worker Signals
        self.worker.progress_changed.connect(self.progress_dialog.setValue)
        self.worker.status_changed.connect(self.progress_dialog.setLabelText)
        self.worker.import_finished.connect(self._on_import_finished)
        self.worker.import_error.connect(self._on_import_error)

        self.progress_dialog.canceled.connect(self.worker.stop)
        self.worker.start()

    def _on_import_finished(self, results):
        """
        Callback when the import worker finishes successfully.
        Creates cards for each imported formulation and populates ingredients.
        """
        if not results:
            return

        if not hasattr(self, "imported_runs"):
            self.imported_runs = []
        self.imported_runs.extend(results)

        count = 0
        for formulation in results:
            try:
                # [Ingredient Mapping Logic Omitted for Brevity - Same as before]
                ingredients_map = {}
                attr_map = {
                    "protein": "Protein",
                    "buffer": "Buffer",
                    "surfactant": "Surfactant",
                    "stabilizer": "Stabilizer",
                    "excipient": "Excipient",
                    "salt": "Salt",
                }
                for attr_name, type_name in attr_map.items():
                    if hasattr(formulation, attr_name):
                        component = getattr(formulation, attr_name)
                        if (
                            component
                            and hasattr(component, "ingredient")
                            and component.ingredient
                        ):
                            ing_obj = component.ingredient
                            name = getattr(ing_obj, "name", "None")
                            conc = getattr(component, "concentration", 0.0)
                            units = getattr(component, "units", "")
                            if name and name != "None":
                                ingredients_map[type_name] = {
                                    "name": name,
                                    "component": name,
                                    "concentration": float(conc) if conc else 0.0,
                                    "units": units,
                                }

                # 2. Extract Viscosity Data
                shear_rates = []
                viscosities = []
                temp = 25.0

                if (
                    hasattr(formulation, "viscosity_profile")
                    and formulation.viscosity_profile
                ):
                    if hasattr(formulation.viscosity_profile, "shear_rates"):
                        shear_rates = formulation.viscosity_profile.shear_rates
                    if hasattr(formulation.viscosity_profile, "viscosities"):
                        viscosities = formulation.viscosity_profile.viscosities

                if hasattr(formulation, "temperature"):
                    temp = float(formulation.temperature)

                # Extract Notes & Missing Fields
                notes = getattr(formulation, "notes", "")
                missing_fields = getattr(formulation, "missing_fields", [])

                card_data = {
                    "name": (
                        formulation.name
                        if formulation.name
                        else f"Imported Run {len(self.imported_runs)}"
                    ),
                    "ingredients": ingredients_map,
                    "measured": True,
                    "temperature": temp,
                    "notes": notes,
                    "missing_fields": missing_fields,
                }

                card = self.add_prediction_card(card_data)

                if card:
                    data_package = {
                        "config_name": card_data["name"],
                        "shear_rate": shear_rates,
                        "viscosity": viscosities,  # Line Plot (Predicted/Reference)
                        "measured_viscosity": viscosities,  # For Evaluation Metric Calc
                        "measured_y": viscosities,  # <--- CRITICAL: Key for VisualizationPanel Dash Line
                        "y": viscosities,  # Fallback for main plot line
                        "x": shear_rates,  # Key for X-axis
                        "temperature": temp,
                        "color": card.plot_color,
                        "measured": True,
                    }

                    card.set_results(data_package)
                    self.viz_panel.set_plot_title(f"Imported: {card_data['name']}")
                    self.viz_panel.set_data(data_package)
                    count += 1

            except Exception as e:
                Log.e(TAG, f"Error creating card for imported run: {e}")

        if count > 0:
            QtWidgets.QMessageBox.information(
                self,
                "Import Complete",
                f"Successfully created {count} formulation cards.",
            )
        else:
            QtWidgets.QMessageBox.warning(
                self, "Import Warning", "No valid formulations could be processed."
            )

    def _on_import_error(self, error_msg):
        """
        Callback when the import worker encounters an error.
        """
        # Ensure progress dialog is closed
        if hasattr(self, "progress_dialog") and self.progress_dialog.isVisible():
            self.progress_dialog.cancel()

        QtWidgets.QMessageBox.critical(
            self, "Import Failed", f"An error occurred during import:\n{error_msg}"
        )

    def get_target_cards(self):
        """
        Returns a list of cards to act upon.
        Priority:
        """
        targets = []

        if self.selection_mode_active:
            for i in range(self.cards_layout.count()):
                widget = self.cards_layout.itemAt(i).widget()
                if isinstance(widget, FormulationConfigCard) and widget.is_selected:
                    targets.append(widget)
        else:
            # Fallback to the currently open/expanded card
            for i in range(self.cards_layout.count()):
                widget = self.cards_layout.itemAt(i).widget()
                if isinstance(widget, FormulationConfigCard) and widget.is_expanded:
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
        target_cards = self.get_target_cards()

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
        Handles requests. Now supports Queuing.
        """
        sender_card = self.sender()

        # INTERCEPTION: If in batch collection mode, just queue and return
        if self._is_batch_collecting:
            if isinstance(sender_card, FormulationConfigCard) and config:
                # Store tuple of (card_reference, config_data)
                self._batch_queue.append((sender_card, config))
            return

        # Standard Execution Logic starts here
        if isinstance(sender_card, FormulationConfigCard):
            self.running_card = sender_card
        else:
            self.running_card = None

        # Determine Color
        self._pending_color = None
        if config and "color" in config and config["color"]:
            self._pending_color = config["color"]
        elif self.running_card:
            self._pending_color = self.running_card.plot_color

        if self.current_task is not None and self.current_task.isRunning():
            self.current_task.stop()

        # Update Visuals (Only if not in the middle of a batch loop controlled elsewhere)
        # If we are simply running one card (not batch mode), show specific loading
        name = config.get("name", "Unknown Sample") if config else "Unknown Sample"

        if not self._is_batch_running:
            self.viz_panel.set_plot_title(f"Calculating: {name}...")
            self.viz_panel.show_loading()

        # Create & Start Thread
        self.current_task = PredictionThread(config)
        self.current_task.data_ready.connect(self._on_prediction_finished)
        self.current_task.finished.connect(self._on_task_complete)
        self.current_task.start()

    def _on_prediction_finished(self, data_package):
        # Restore color
        if self._pending_color and "color" not in data_package:
            data_package["color"] = self._pending_color

        # Update the specific card that requested this
        if hasattr(self, "running_card") and self.running_card:
            self.running_card.set_results(data_package)

        # Branch logic: Batch vs Single
        if self._is_batch_running:
            # Accumulate and continue
            self._batch_results.append(data_package)
            self._process_next_in_batch()
        else:
            # Standard single run: Update Viz Panel immediately
            final_name = data_package.get("config_name", "Unknown")
            self.viz_panel.set_plot_title(final_name)
            self.viz_panel.set_data(data_package)
            self.viz_panel.hide_loading()

    def _on_task_complete(self):
        """Called when thread naturally finishes."""
        pass

    def closeEvent(self, event):
        """
        Guaranteed cleanup on close.
        """
        if self.current_task is not None and self.current_task.isRunning():
            print("Closing application: Stopping background thread...")
            self.current_task.stop()
        self.db.close()
        super().closeEvent(event)

    def _update_overlay_geometry(self):
        """Updates positions of overlay widgets."""

        # 1. Filter Widget (Relative to Left Panel)
        if hasattr(self, "filter_widget") and self.filter_widget.isVisible():
            self.filter_widget.setGeometry(
                0, 0, self.left_widget.width(), self.filter_widget.sizeHint().height()
            )

        # 2. Evaluation Widget (Relative to Toolbar Button)
        if hasattr(self, "eval_widget") and self.eval_widget.isVisible():
            # Calculate Global Position of the Button's bottom-left corner
            btn_geo = self.btn_evaluate.geometry()
            global_pos = self.btn_evaluate.mapToGlobal(
                QtCore.QPoint(0, btn_geo.height())
            )

            # Map back to PredictionUI (self) coordinates
            local_pos = self.mapFromGlobal(global_pos)

            # Set Width (e.g., 300px standard dropdown width)
            menu_width = 300

            # Ensure it doesn't go off the right side of the screen
            x = local_pos.x()
            if x + menu_width > self.width():
                x = self.width() - menu_width - 10

            self.eval_widget.setGeometry(
                x, local_pos.y(), menu_width, self.eval_widget.sizeHint().height()
            )

            # Ensure it sits on top of everything (splitter, right panel, etc.)
            self.eval_widget.raise_()

    def resizeEvent(self, event):
        self._update_overlay_geometry()
        super().resizeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet())

    win = PredictionUI()
    win.setWindowTitle("Viscosity AI - Hyperparameter Tuning")
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())
