import os
import sys

from architecture import Architecture
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from styles.style_loader import load_stylesheet
from widgets.formulation_config_card_widget import (
    FormulationConfigCard,
)
from widgets.placeholder_widget import PlaceholderWidget
from widgets.prediction_filter_widget import PredictionFilterWidget
from widgets.reordable_container_widget import ReorderableCardContainer
from widgets.visualization_panel import VisualizationPanel
from workers.prediction_worker import PredictionThread


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
        self._load_mock_data()
        self.init_ui()
        self.setStyleSheet(load_stylesheet())

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

        # Splitter Configuration
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)  # Thin line to avoid covering content
        splitter.setChildrenCollapsible(False)  # Prevent full collapse

        # Style the splitter handle to look like a clean border
        splitter.setStyleSheet(
            """
            QSplitter::handle {
                background-color: #d1d5db;
            }
        """
        )

        # --- Left Panel ---
        self.left_widget = QtWidgets.QWidget()
        self.left_widget.setObjectName("leftPanel")
        left_layout = QtWidgets.QVBoxLayout(self.left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # Left Toolbar
        top_bar = self._create_top_bar()
        left_layout.addWidget(top_bar)

        self.filter_widget = PredictionFilterWidget(
            self.ingredients_by_type, parent=self.left_widget
        )
        self.filter_widget.filter_changed.connect(self.apply_filters)
        self.filter_widget.hide()

        self.left_widget.installEventFilter(self)

        # Scroll Area
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
        self.left_widget.installEventFilter(self)
        splitter.addWidget(self.left_widget)

        # --- Right Panel ---
        self.right_widget = QtWidgets.QWidget()
        self.right_widget.setObjectName("rightPanel")

        # Ensure right panel has a white background so splitter doesn't look like a gap
        self.right_widget.setStyleSheet("background-color: #ffffff;")

        right_layout = QtWidgets.QVBoxLayout(self.right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Right Toolbar
        right_top_bar = self._create_right_top_bar()
        right_layout.addWidget(right_top_bar)

        # Visualization Panel
        self.viz_panel = VisualizationPanel()
        right_layout.addWidget(self.viz_panel)

        splitter.addWidget(self.right_widget)

        splitter.setSizes([450, 700])
        main_layout.addWidget(splitter)

        self.current_task = None
        self.add_prediction_card()

    def _create_top_bar(self):
        """Creates the top toolbar (search + action buttons)"""
        container = QtWidgets.QWidget()
        container.setObjectName("topBar")
        container.setFixedHeight(50)
        container.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)

        # Drop shadow effect
        shadow = QtWidgets.QGraphicsDropShadowEffect(container)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QtGui.QColor(0, 0, 0, 25))
        container.setGraphicsEffect(shadow)

        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(15, 5, 15, 5)
        layout.setSpacing(10)

        # Search Bar
        self.search_bar = QtWidgets.QLineEdit()
        self.search_bar.setObjectName("searchBar")
        self.search_bar.setPlaceholderText("Search...")
        self.search_bar.setClearButtonEnabled(True)
        self.search_bar.addAction(
            QtGui.QIcon(
                os.path.join(Architecture.get_path(), "icons/search-svgrepo-com.svg")
            ),
            QtWidgets.QLineEdit.ActionPosition.LeadingPosition,
        )
        self.search_bar.textChanged.connect(self.filter_cards)
        layout.addWidget(self.search_bar, stretch=1)

        # Filter Button
        self.btn_filter = QtWidgets.QToolButton()
        self.btn_filter.setIcon(
            QtGui.QIcon(
                os.path.join(Architecture.get_path(), "icons/filter-svgrepo-com.svg")
            )
        )
        self.btn_filter.setToolTip("Filter Options")
        self.btn_filter.setCheckable(True)
        self.btn_filter.setFixedSize(32, 32)
        self.btn_filter.clicked.connect(self.toggle_filter_menu_manual)
        layout.addWidget(self.btn_filter)

        # Select Mode Button
        self.btn_select_mode = QtWidgets.QToolButton()
        self.btn_select_mode.setIcon(
            QtGui.QIcon(
                os.path.join(Architecture.get_path(), "icons/select-svgrepo-com.svg")
            )
        )
        self.btn_select_mode.setToolTip("Enter Selection Mode")
        self.btn_select_mode.setCheckable(True)
        self.btn_select_mode.setFixedSize(32, 32)
        self.btn_select_mode.toggled.connect(self.toggle_selection_mode)
        layout.addWidget(self.btn_select_mode)

        # Select All Button
        self.btn_select_all = QtWidgets.QToolButton()
        self.btn_select_all.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(), "icons/select-multiple-svgrepo-com.svg"
                )
            )
        )
        self.btn_select_all.setToolTip("Select All")
        self.btn_select_all.setFixedSize(32, 32)
        self.btn_select_all.clicked.connect(self.select_all_cards)
        self.btn_select_all.setEnabled(False)
        layout.addWidget(self.btn_select_all)

        # Import Button
        self.btn_import = QtWidgets.QToolButton()
        self.btn_import.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(), "icons/import-content-svgrepo-com.svg"
                )
            )
        )
        self.btn_import.setToolTip("Import Data")
        self.btn_import.setFixedSize(32, 32)
        self.btn_import.clicked.connect(self.import_data_file)
        layout.addWidget(self.btn_import)

        # Export Button
        self.btn_export_top = QtWidgets.QToolButton()
        self.btn_export_top.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(), "icons/export-content-svgrepo-com.svg"
                )
            )
        )
        self.btn_export_top.setToolTip("Export Selected (or Open Card)")
        self.btn_export_top.setFixedSize(32, 32)
        self.btn_export_top.clicked.connect(self.export_analysis)
        layout.addWidget(self.btn_export_top)

        # Separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        line.setFixedHeight(20)
        layout.addWidget(line)

        # Run Button
        self.btn_run_top = QtWidgets.QToolButton()
        self.btn_run_top.setIcon(
            QtGui.QIcon(
                os.path.join(Architecture.get_path(), "icons/play-svgrepo-com.svg")
            )
        )
        self.btn_run_top.setToolTip("Run Inference (Selected or Open Card)")
        self.btn_run_top.setFixedSize(32, 32)
        self.btn_run_top.clicked.connect(self.run_analysis)
        layout.addWidget(self.btn_run_top)

        # Delete Button
        self.btn_delete_top = QtWidgets.QToolButton()
        self.btn_delete_top.setObjectName("btnDelete")
        self.btn_delete_top.setIcon(
            QtGui.QIcon(
                os.path.join(Architecture.get_path(), "icons/delete-2-svgrepo-com.svg")
            )
        )
        self.btn_delete_top.setToolTip("Delete (Selected or Open Card)")
        self.btn_delete_top.setFixedSize(32, 32)
        self.btn_delete_top.clicked.connect(self.delete_analysis)
        layout.addWidget(self.btn_delete_top)

        return container

    def _create_right_top_bar(self):
        """Creates the right-side top toolbar."""
        # Same ObjectName 'topBar' to inherit the exact same CSS
        container = QtWidgets.QWidget()
        container.setObjectName("topBar")
        container.setFixedHeight(50)
        container.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)

        # Same drop shadow effect
        shadow = QtWidgets.QGraphicsDropShadowEffect(container)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QtGui.QColor(0, 0, 0, 25))
        container.setGraphicsEffect(shadow)

        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(15, 5, 15, 5)
        layout.setSpacing(10)

        # Push items to the right
        layout.addStretch(1)

        # Single Icon Button on the right
        self.btn_right_options = QtWidgets.QToolButton()
        self.btn_right_options.setIcon(
            QtGui.QIcon(
                os.path.join(
                    Architecture.get_path(), "icons/three-dots-svgrepo-com.svg"
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
        """Positions the filter widget directly below the top bar."""
        if self.filter_widget.isVisible():
            self.filter_widget.setGeometry(
                0, 50, self.left_widget.width(), self.filter_widget.sizeHint().height()
            )

    def run_analysis(self):
        """Runs inference on Selected cards (if any) or the currently Open card."""
        target_cards = []

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
            QtWidgets.QMessageBox.information(
                self, "Run Inference", "No cards selected or open to run."
            )
            return

        # Trigger Run
        for card in target_cards:
            card.emit_run_request()

    def delete_analysis(self):
        """Deletes Selected cards (if any) or the currently Open card."""
        target_cards = []

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
            if self.selection_mode_active:
                self.btn_select_mode.setChecked(False)
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
        )
        card.removed.connect(self.remove_card)
        card.run_requested.connect(self.run_prediction)
        card.expanded.connect(self.on_card_expanded)
        card.selection_changed.connect(self._on_card_selection_changed)
        insert_idx = self.cards_layout.count()
        self.cards_layout.insertWidget(insert_idx, card)
        card.show()
        self.update_placeholder_visibility()

        if data:
            if hasattr(card, "load_data"):
                card.load_data(data)
            if data.get("measured", False):
                card.set_measured_state(True)
            card.emit_run_request()

        self.on_card_expanded(card)
        QtCore.QTimer.singleShot(100, lambda: self._scroll_to_card(card))

        self.update_placeholder_visibility()

    def _scroll_to_card(self, card_widget):
        """Helper to ensure the new card is visible in the scroll area."""
        self.scroll_area.ensureWidgetVisible(card_widget, 0, 0)

    def on_card_expanded(self, active_card):
        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, FormulationConfigCard) and widget is not active_card:
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
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not fname:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

        try:

            imported_config = {
                "name": "High Concentration Etanercept",
                "measured": True,
                "use_in_icl": True,
                "notes": "Some notes about formulation components here",
                "model": "VisQ.AI 1.3.1 (Base)",
                "temperature": 25.0,  # °C
                # Format: "ingredient_type": {"name": "ingredient_name", "concentration": value}
                "ingredients": {
                    "Protein": {
                        "name": "Etanercept",
                        "concentration": 300,  # mg/mL
                        "class": "IgG1",
                        "mw": 150,
                        "pi_mean": 7.9,
                        "pi_range": 0.8,
                    },
                    "Buffer": {
                        "name": "Histidine",
                        "concentration": 20.0,  # mM
                        "ph": 6.0,
                    },
                    "Surfactant": {
                        "name": "Tween-80",
                        "concentration": 0.05,  # %w
                    },
                    "Stabilizer": {"name": "Sucrose", "concentration": 0.2},  # M
                    "Excipient": {"name": "Lysin", "concentration": 50.0},  # mM
                    "Salt": {"name": "NaCl", "concentration": 150.0},  # mM
                },
                # ML Parameters (optional, for model options)
                "ml_params": {"lr": 0.01, "steps": 100, "ci": 95},
            }
            self.add_prediction_card(imported_config)
            self.run_prediction(imported_config)

        finally:

            QtWidgets.QApplication.restoreOverrideCursor()

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
        Runs prediction using the robust PredictionThread subclass.
        """
        sender_card = self.sender()
        if isinstance(sender_card, FormulationConfigCard):
            self.running_card = sender_card
        else:
            self.running_card = None
        if self.current_task is not None and self.current_task.isRunning():
            print("Stopping previous task...")
            self.current_task.stop()

        # Visual Feedback
        name = config.get("name", "Unknown Sample") if config else "Unknown Sample"
        self.viz_panel.set_plot_title(f"Calculating: {name}...")
        self.viz_panel.show_loading()

        # Create & Start New Thread
        self.current_task = PredictionThread(config)
        self.current_task.data_ready.connect(self._on_prediction_finished)
        self.current_task.finished.connect(self._on_task_complete)
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
        pass

    def closeEvent(self, event):
        """
        Guaranteed cleanup on close.
        """
        if self.current_task is not None and self.current_task.isRunning():
            print("Closing application: Stopping background thread...")
            self.current_task.stop()

        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet())

    win = PredictionUI()
    win.setWindowTitle("Viscosity AI - Hyperparameter Tuning")
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())
