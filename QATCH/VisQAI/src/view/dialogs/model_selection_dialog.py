"""
model_selection_dialog.py

Model selection dialog for browsing and managing VisQ-AI ``.visq`` model files.

This module provides :class:`ModelSelectionDialog`, a :class:`QDialog` subclass
that emulates the :class:`QFileDialog` interface while offering VisQ-AI-specific
features such as pinning, renaming, deleting, and visualising the training
lineage of models stored in a directory.

Author(s):
    Alexander J. Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.1
"""

import json
import os
from datetime import datetime

from PyQt5.QtCore import QDir, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    TAG = "[ModelSelectionDialog]"
    from QATCH.common.architecture import Architecture
    from QATCH.common.logger import Logger as Log
    from QATCH.VisQAI.src.managers.version_manager import VersionManager

except ImportError:
    TAG = "[ModelSelectionDialog (HEADLESS)]"

    class Log:
        @staticmethod
        def d(tag, msg=""):
            print("DEBUG:", tag, msg)

        @staticmethod
        def i(tag, msg=""):
            print("INFO:", tag, msg)

        @staticmethod
        def w(tag, msg=""):
            print("WARNING:", tag, msg)

        @staticmethod
        def e(tag, msg=""):
            print("ERROR:", tag, msg)

    from architecture import Architecture

_NOT_PROVIDED = object()  # module-level sentinel


class QFancyListWidget(QListWidget):
    """A QListWidget that clears selection when clicking on an empty area."""

    def mousePressEvent(self, event):
        """Handle mouse press events, clearing selection on empty area clicks.

        Args:
            event (QMouseEvent): The mouse press event.
        """
        item = self.itemAt(event.pos())
        if item is None:
            # Clicked on empty area
            self.clearSelection()  # Deselect current selection
            # Emit a custom signal or handle as needed
            self.itemClicked.emit(item)
        else:
            # Let normal behavior happen (selecting items, etc.)
            super().mousePressEvent(event)


class ModelSelectionDialog(QDialog):
    """Custom model selection dialog for browsing and managing VisQ-AI models.

    Displays pinned and recent models in separate lists with support for
    pinning, renaming, deleting, and viewing the training tree of models.

    Attributes:
        fileSelected (pyqtSignal): Signal emitted with the selected model file
            path when a model is accepted.
    """

    fileSelected: pyqtSignal = pyqtSignal(str)

    def __init__(self, models_directory=None, parent=None):
        """Initialize the dialog and populate model lists.

        Args:
            models_directory (str): Path to the directory containing ``.visq``
                model files. Defaults to the current working directory.
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.models_directory = (
            models_directory if models_directory is not None else os.getcwd()
        )
        self.pinned_models = []
        self.pinned_names = {}
        self.selected_model = None

        self.init_ui()
        self.populate_models()

    def init_ui(self):
        """Build and wire up all widgets in the dialog."""
        self.setWindowTitle("VisQ.AI Model Selection")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout()

        _icons_dir = os.path.join(
            Architecture.get_path(), "QATCH", "VisQAI", "src", "view", "icons"
        )

        # Pinned models section
        self.pinned_label = QWidget()
        _pinned_label_layout = QHBoxLayout(self.pinned_label)
        _pinned_label_layout.setContentsMargins(0, 0, 0, 0)
        _pin_icon = QLabel()
        _pin_icon.setPixmap(
            QIcon(os.path.join(_icons_dir, "pin-circle-svgrepo-com.svg")).pixmap(
                QSize(16, 16)
            )
        )
        _pinned_label_layout.addWidget(_pin_icon)
        _pinned_section_label = QLabel("PINNED MODELS")
        _pinned_section_label.setProperty("class", "section-header")
        _pinned_label_layout.addWidget(_pinned_section_label)
        _pinned_label_layout.addStretch()
        self.pinned_list = QFancyListWidget()
        self.pinned_list.setAlternatingRowColors(True)

        # Recent models section
        self.recent_label = QWidget()
        _recent_label_layout = QHBoxLayout(self.recent_label)
        _recent_label_layout.setContentsMargins(0, 0, 0, 0)
        _folder_icon = QLabel()
        _folder_icon.setPixmap(
            QIcon(
                os.path.join(_icons_dir, "folder-path-connect-svgrepo-com.svg")
            ).pixmap(QSize(16, 16))
        )
        _recent_label_layout.addWidget(_folder_icon)
        _recent_section_label = QLabel("RECENT MODELS")
        _recent_section_label.setProperty("class", "section-header")
        _recent_label_layout.addWidget(_recent_section_label)
        _recent_label_layout.addStretch()
        self.recent_list = QFancyListWidget()
        self.recent_list.setAlternatingRowColors(True)

        # Detail panel
        self.detail_panel = QLabel("Select a model to view details")
        self.detail_panel.setObjectName("detailPanel")
        self.detail_panel.setWordWrap(True)
        self.detail_panel.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.detail_panel)

        # Buttons — secondary actions (left cluster)
        self.view_details_btn = QPushButton("View Details")
        self.view_details_btn.setCursor(Qt.PointingHandCursor)
        self.rename_btn = QPushButton("Rename")
        self.rename_btn.setCursor(Qt.PointingHandCursor)
        self.pin_btn = QPushButton("Pin / Unpin")
        self.pin_btn.setCursor(Qt.PointingHandCursor)

        # Danger action
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setProperty("class", "danger")
        self.delete_btn.setCursor(Qt.PointingHandCursor)

        # Dismiss action
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setProperty("class", "ghost")
        self.cancel_btn.setCursor(Qt.PointingHandCursor)

        # Primary confirm action
        self.select_btn = QPushButton("Select Model")
        self.select_btn.setProperty("class", "primary")
        self.select_btn.setCursor(Qt.PointingHandCursor)
        self.select_btn.setDefault(True)

        # Add to layout
        layout.addWidget(self.pinned_label)
        layout.addWidget(self.pinned_list)
        layout.addWidget(self.recent_label)
        layout.addWidget(self.recent_list)
        layout.addWidget(self.scroll_area)

        # Single footer row: [secondary...] ——stretch—— [Delete] [Cancel] [Select]
        footer = QHBoxLayout()
        footer.setSpacing(6)
        footer.addWidget(self.view_details_btn)
        footer.addWidget(self.rename_btn)
        footer.addWidget(self.pin_btn)
        footer.addStretch()
        footer.addWidget(self.delete_btn)
        footer.addWidget(self.cancel_btn)
        footer.addWidget(self.select_btn)

        layout.addLayout(footer)

        self.setLayout(layout)

        # Connect signals
        self.pinned_list.itemClicked.connect(self.on_model_selected)
        self.recent_list.itemClicked.connect(self.on_model_selected)
        self.pinned_list.currentItemChanged.connect(self.on_model_selected)
        self.recent_list.currentItemChanged.connect(self.on_model_selected)
        self.view_details_btn.clicked.connect(self.show_detailed_view)
        self.rename_btn.clicked.connect(self.rename_model)
        self.pin_btn.clicked.connect(self.toggle_pin)
        self.delete_btn.clicked.connect(self.delete_model)
        self.pinned_list.itemDoubleClicked.connect(self.accept)
        self.recent_list.itemDoubleClicked.connect(self.accept)
        self.select_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def populate_models(self):
        """Scan the models directory and repopulate the pinned and recent lists."""
        self.all_models = self.scan_model_directory()

        # Sort by modification time (most recent first)
        self.all_models.sort(key=lambda x: x["created"], reverse=True)

        # Separate pinned and unpinned
        pinned = [m for m in self.all_models if m["filename"] in self.pinned_models]
        unpinned = [
            m for m in self.all_models if m["filename"] not in self.pinned_models
        ]

        # Populate lists
        self.pinned_list.clear()
        for model in pinned:
            item_text = self.get_item_text(model)
            self.pinned_list.addItem(item_text)

        self.recent_list.clear()
        for model in unpinned:
            item_text = self.get_item_text(model)
            self.recent_list.addItem(item_text)

    def get_item_text(self, model):
        """Generate the two-line display text shown for each model list item.

        The first line shows the model name and creation time; the second line
        shows the parent model, number of learned runs, and file size.

        Args:
            model (dict): Model info dictionary as returned by
                :meth:`scan_model_directory`.

        Returns:
            str: Formatted two-line display string for the list item.
        """

        age_days = (datetime.now() - datetime.fromtimestamp(model["created"])).days
        if age_days <= 7:
            # if this week, show relative time (e.g., "3 days ago")
            if age_days == 0:
                created_time = "Today"
            elif age_days == 1:
                created_time = "Yesterday"
            else:
                created_time = f"{age_days} days ago"
        else:
            # show absolute creation time
            created_time = datetime.fromtimestamp(model["created"]).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        parent_model = model["metadata"].get("parent_model", "N/A")
        if parent_model == "N/A":
            # Try alternative key, from legacy models
            parent_model = model["metadata"].get("base_model", "N/A")
        if parent_model != "N/A":
            parent_model = os.path.basename(parent_model)
            parent_model = self.pinned_names.get(parent_model, parent_model)

        learned_runs = model["metadata"].get("learned_runs", [])
        if learned_runs:
            num_runs = len(learned_runs)
        else:
            num_runs = 0

        size_value = model["size"]
        size_units = "bytes"
        if size_value > 1024:
            size_value /= 1024
            size_units = "KB"
        if size_value > 1024:
            size_value /= 1024
            size_units = "MB"
        if size_value > 1024:
            size_value /= 1024
            size_units = "GB"

        model_text: str = self.pinned_names.get(model["filename"], model["filename"])
        created_text: str = f"Created: {created_time}"
        parent_text: str = f"Parent Model: {parent_model}"
        runs_text: str = f"Runs: {num_runs}"
        size_text: str = f"Size: {size_value:.2f} {size_units}"

        return f"""
{model_text.ljust(42)} \t\t\t\t\t {created_text}
{parent_text.ljust(42)} \t\t {runs_text.ljust(10)} \t\t\t {size_text}
                """.strip()

    def scan_model_directory(self):
        """Scan the models directory for ``.visq`` model files.

        Reads ``index.json`` (if present) for metadata, then iterates over all
        ``.visq`` files and builds a list of model info dictionaries. Deleted
        models are skipped. The base model is automatically pinned and protected.

        Returns:
            list[dict]: Each dict has the keys ``filename``, ``filepath``,
                ``pinned_name``, ``created``, ``size``, ``metadata``, and
                ``sha``.
        """
        models = []
        index_data = {}

        for filename in os.listdir(self.models_directory):
            if filename == "index.json":
                index_path = os.path.join(self.models_directory, filename)
                try:
                    with open(index_path, "r") as f:
                        index_data = json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    Log.e(TAG, f"Failed to parse index.json: {e}")

            # if not filename.startswith('VisQAI'):
            #     continue
            if not filename.endswith(".visq"):
                continue

            filepath = os.path.join(self.models_directory, filename)

            try:
                # Parse metadata from model ZIP
                metadata, sha = self.parse_model_metadata(index_data, filepath)

                if metadata.get("deleted", False):
                    Log.d(TAG, "Skipping deleted model:", filename)
                    continue  # Skip deleted models

                if filename == "VisQAI-base.visq":
                    if not metadata.get("protected", None):
                        metadata["protected"] = True  # internal flag
                    if not metadata.get("pin", None):
                        metadata["pin"] = True  # Always pin base model
                        self.pinned_models.append(filename)
                    if not metadata.get("pinned_name", None):
                        metadata["pinned_name"] = "Base Model"
                        self.pinned_names[filename] = metadata["pinned_name"]

                models.append(
                    {
                        "filename": filename,
                        "filepath": filepath,
                        "pinned_name": self.pinned_names.get(
                            filename, metadata.get("name", filename)
                        ),
                        "created": os.path.getctime(filepath),
                        "size": os.path.getsize(filepath),
                        "metadata": metadata,
                        "sha": sha,
                    }
                )
            except Exception as e:
                Log.e(TAG, f"Error parsing model {filename}: {e}")
                continue

        return models

    def parse_model_metadata(self, index_data, model_path):
        """Look up metadata for a model file in the index.

        Matches the model filename against SHA-prefixed entries in
        ``index_data``. As a side effect, updates :attr:`pinned_names` and
        :attr:`pinned_models` based on the stored metadata flags.

        Args:
            index_data (dict): Parsed contents of ``index.json``, keyed by
                full SHA hash strings.
            model_path (str): Absolute path to the ``.visq`` model file.

        Returns:
            tuple[dict, str | None]: A ``(metadata, sha)`` pair. ``metadata``
                is the metadata dict from the index (empty dict if not found).
                ``sha`` is the full SHA key string, or ``None`` if not found.
        """
        # Search for metadata in index_data where filename contains the start of the sha hash
        base_name = os.path.basename(model_path)
        for entry in index_data.keys():
            if base_name.endswith(f"{entry[0:7]}.visq"):
                metadata = index_data[entry].get("metadata", {})

                pinned_name = metadata.get("pinned_name", None)
                if pinned_name:
                    self.pinned_names[base_name] = pinned_name

                is_pinned = metadata.get("pin", False)
                if is_pinned and base_name not in self.pinned_models:
                    self.pinned_models.append(base_name)
                elif not is_pinned and base_name in self.pinned_models:
                    self.pinned_models.remove(base_name)

                return metadata, entry

        return {}, None  # entry not found

    def on_model_selected(self, item=None, previous=_NOT_PROVIDED):
        """Update the detail panel and selected_model for the given item.

        Connected to both ``itemClicked`` and ``currentItemChanged`` on the
        pinned and recent lists. Clears the opposing list's selection so that
        only one item is active at a time.

        Args:
            item (QListWidgetItem, optional): The newly selected item, or
                ``None`` to clear the selection. Defaults to ``None``.
            previous: Unused; present to match the ``currentItemChanged``
                signature. Defaults to ``NameError`` as a sentinel that
                distinguishes an explicit ``None`` from a missing argument.
        """
        if not item:
            # Update detail panel to reflect no selection
            self.detail_panel.setText("Select a model to view details")
            self.selected_model = None
            return
        # if not previous:
        #     return

        if not self.pinned_list.isAncestorOf(item.listWidget()):
            self.pinned_list.clearSelection()
        if not self.recent_list.isAncestorOf(item.listWidget()):
            self.recent_list.clearSelection()

        selected_text = item.text()
        model_name_line = selected_text.splitlines()[0]
        model_name = model_name_line.split("\t")[0].strip()

        # Find model info
        for model in self.all_models:
            if (
                self.pinned_names.get(model["filename"], model["filename"])
                == model_name
            ):
                self.selected_model = model["filepath"]
                detail_text = f"Model: {self.pinned_names.get(model['filename'], model['filename'])}\n"
                detail_text += json.dumps(model["metadata"], indent=4)
                self.detail_panel.setText(detail_text)
                break

    def select_model_by_name(self, model_name):
        """Programmatically select a model by its display name and accept the dialog.

        If ``model_name`` is ``None`` or ``"Base Model"``, the base model is
        selected. Emits :attr:`fileSelected` and calls :meth:`accept`.

        Args:
            model_name (str | None): Display name (pinned name or filename) of
                the model to select.
        """
        if model_name is None or model_name == "Base Model":
            model_names = ["VisQAI-base", "Base Model"]
        else:
            model_names = [model_name]
        for model in self.all_models:
            if (
                self.pinned_names.get(model["filename"], model["filename"])
                in model_names
            ):
                self.selected_model = model["filepath"]
                self.fileSelected.emit(self.selected_model)
                self.accept()
                break

    def toggle_pin(self):
        """Toggle the pin status of the currently selected model.

        Persists the change via :class:`VersionManager` (``index.json``),
        then refreshes the lists and restores the selection. Shows a warning
        if no model is selected or if the model is protected.
        """
        if not self.selected_model:
            Log.w(TAG, "No model selected to pin/unpin")
            return

        # Implementation for pinning/unpinning within `index.json`
        mvc = VersionManager(self.models_directory, retention=255)

        # Get the sha of the selected model
        model_name = os.path.basename(self.selected_model)
        model_info = next(
            (m for m in self.all_models if m["filename"] == model_name), None
        )
        if not model_info:
            Log.e(TAG, f'Selected model "{model_name}" not found in model list')
            return
        sha = model_info["sha"]

        if model_info["metadata"].get("protected", False):
            QMessageBox.warning(
                self,
                "Protected Model",
                "This model is protected and cannot be unpinned.",
            )
            return

        try:
            if model_info["metadata"].get("pin", False):
                mvc.unpin(sha)
            else:
                mvc.pin(sha)
        except (ValueError, KeyError, PermissionError) as e:
            Log.e(TAG, f"Failed to toggle pin for {model_name}: {e}")
            QMessageBox.warning(self, "Operation Failed", str(e))
            return

        if self.pinned_list.currentItem():
            search_list = "recent"
        else:
            search_list = "pinned"

        self.populate_models()  # Refresh lists

        if search_list == "pinned":
            search_list = self.pinned_list
        else:
            search_list = self.recent_list

        # Find index of the selected model in the new list
        select_index = 0
        current_name = self.pinned_names.get(
            model_info["filename"], model_info["filename"]
        )
        for i in range(search_list.count()):
            item = search_list.item(i)
            if item.text().splitlines()[0].split("\t")[0].strip() == current_name:
                select_index = i
                break

        search_list.setCurrentRow(select_index)  # Restore selection
        # Refresh the detail panel
        self.on_model_selected(item=search_list.currentItem())

    def rename_model(self):
        """Prompt the user to rename the currently selected model.

        Opens an input dialog pre-filled with the current display name.
        Validates that the new name is non-duplicate and non-empty before
        persisting the change via :class:`VersionManager`. Shows appropriate
        warnings for protected models, duplicate names, or empty input.
        """
        if not self.selected_model:
            Log.w(TAG, "No model selected to pin/unpin")
            return

        # Implementation for renaming within `index.json`
        mvc = VersionManager(self.models_directory, retention=255)

        # Get the sha of the selected model
        model_name = os.path.basename(self.selected_model)
        model_info = next(
            (m for m in self.all_models if m["filename"] == model_name), None
        )
        if not model_info:
            Log.e(TAG, f'Selected model "{model_name}" not found in model list')
            return
        sha = model_info["sha"]

        if model_info["metadata"].get("protected", False):
            QMessageBox.warning(
                self,
                "Protected Model",
                "This model is protected and cannot be renamed.",
            )
            return

        current_name = self.pinned_names.get(
            model_info["filename"], model_info["filename"]
        )
        new_name, ok = QInputDialog.getText(
            self, "Rename Model", "Enter new name:", text=current_name
        )

        if new_name == current_name or not ok:
            return
        else:
            new_name = new_name.strip()

        if new_name == "":
            if (
                QMessageBox.question(
                    self,
                    "Empty Name",
                    "Would you like to rename this model to its filename?\nFilename: {}".format(
                        model_info["filename"]
                    ),
                    QMessageBox.Yes | QMessageBox.No,
                )
                == QMessageBox.Yes
            ):
                new_name = model_info["filename"]
            else:
                return
        else:
            if new_name.casefold() in [
                v.casefold() for v in self.pinned_names.values()
            ]:
                QMessageBox.warning(
                    self, "Duplicate Name", "A model with this name already exists."
                )
                return
            if new_name.casefold() in [
                m["filename"].casefold() for m in self.all_models
            ]:
                QMessageBox.warning(
                    self, "Duplicate Name", "A model with this filename already exists."
                )
                return

        index = mvc._load_index()
        if sha not in index:
            Log.e(TAG, f"SHA {sha} not found in index during rename")
            return
        index[sha]["metadata"]["pinned_name"] = new_name
        mvc._write_index(index)

        if self.pinned_list.currentItem():
            select_index = self.pinned_list.currentRow()
            select_list = self.pinned_list
        else:
            select_index = self.recent_list.currentRow()
            select_list = self.recent_list

        self.populate_models()  # Refresh lists
        select_list.setCurrentRow(select_index)  # Restore selection
        # Refresh the detail panel
        self.on_model_selected(item=select_list.currentItem())

    def delete_model(self):
        """Prompt the user to confirm deletion of the currently selected model.

        Marks the model as deleted via :class:`VersionManager` after
        confirmation. The operation cannot be undone. Shows a warning if no
        model is selected or if the model is protected.
        """
        if not self.selected_model:
            Log.w(TAG, "No model selected to pin/unpin")
            return

        # Implementation for deletion within `index.json`
        mvc = VersionManager(self.models_directory, retention=255)

        # Get the sha of the selected model
        model_name = os.path.basename(self.selected_model)
        model_info = next(
            (m for m in self.all_models if m["filename"] == model_name), None
        )
        if not model_info:
            Log.e(TAG, f'Selected model "{model_name}" not found in model list')
            return
        sha = model_info["sha"]

        if model_info["metadata"].get("protected", False):
            QMessageBox.warning(
                self,
                "Protected Model",
                "This model is protected and cannot be deleted.",
            )
            return

        confirm = QMessageBox.question(
            self,
            "Delete Model",
            f'Are you sure you want to delete "{model_name}"?\nWARNING: This operation cannot be undone!',
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm == QMessageBox.Yes:
            mvc.delete(sha)

        # NOTE: On item delete, no selection to restore; but lists and detail panel need to be refreshed
        self.populate_models()  # Refresh lists
        self.on_model_selected()  # Refresh detail panel

    def show_detailed_view(self):
        """Open a separate window showing the training tree for the selected model.

        Traverses parent/child relationships stored in model metadata to build
        a ``QTreeWidget`` rooted at the earliest known ancestor. The selected
        model is highlighted with an arrow marker. A "Switch to Selected"
        button accepts the dialog with that model; "Close" dismisses the window.
        """
        # Implementation for detailed view from `index.json`

        # Parse parent model and learned runs from metadata and display them
        item_list = self.pinned_list
        indices = item_list.selectedIndexes()
        if not len(indices) == 1:
            item_list = self.recent_list
            indices = item_list.selectedIndexes()
        item = item_list.currentItem()
        if not item:
            Log.w(TAG, "No model selected for detailed view")
            return

        selected_text = item.text()
        model_name_line = selected_text.splitlines()[0]
        model_name = model_name_line.split("\t")[0].strip()
        selected_model_name = model_name

        # Find model info
        training_tree = {}
        key_index = 0
        while True:
            model_found = False
            for model in self.all_models:
                if (
                    self.pinned_names.get(model["filename"], model["filename"])
                    == model_name
                ):
                    metadata = model["metadata"]
                    parent = metadata.get("parent", None)
                    children = metadata.get("children", [])
                    if not key_index in training_tree:
                        training_tree[key_index] = []
                    for child in children:
                        child_model = next(
                            (m for m in self.all_models if m["sha"] == child), None
                        )
                        if not child_model:
                            Log.w(
                                TAG,
                                f'Child model sha "{child}" not found in model list',
                            )
                            continue  # Skip this child
                        child_name = self.pinned_names.get(
                            child_model["filename"], child_model["filename"]
                        )
                        training_tree[key_index].append((model_name, child_name))
                    if parent:
                        key_index -= 1
                        if not key_index in training_tree:
                            training_tree[key_index] = []
                        if parent == "base":
                            parent_name = "Base Model"
                        else:
                            parent_model = next(
                                (m for m in self.all_models if m["sha"] == parent), None
                            )
                            if not parent_model:
                                Log.w(
                                    TAG,
                                    f'Parent model sha "{parent}" not found in model list',
                                )
                                parent_name = f"Unknown ({parent[:7]})"  # Fallback to truncated SHA
                            else:
                                parent_name = self.pinned_names.get(
                                    parent_model["filename"], parent_model["filename"]
                                )
                    elif parent := metadata.get("parent_model", None):
                        # Found parent model in legacy metadata, missing sha hash reference
                        parent_name = self.pinned_names.get(
                            os.path.basename(parent), os.path.basename(parent)
                        )
                    else:
                        parent = "base"  # stop searching
                        parent_name = "Unspecified Parent"
                    training_tree[key_index].append((parent_name, model_name))
                    model_name = parent_name if parent != "base" else None
                    model_found = True
                    break
            if not model_name:
                break
            if not model_found:
                Log.e(TAG, f'Parent model "{parent}" not found in model list')
                break

        training_tree = dict(sorted(training_tree.items()))

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name"])

        # detail_text = "Training Tree:\n"
        first_i = None
        for i, pair in training_tree.items():
            if first_i is None:
                # detail_text += "Base Model\n"
                first_i = i
                base_tree = QTreeWidgetItem([pair[0][0]])
                self.tree.addTopLevelItem(base_tree)
            for parent, child in pair:
                # detail_text += "    " * (i - first_i) + f"{child}\n"
                items = self.tree.findItems(
                    parent, Qt.MatchRecursive | Qt.MatchExactly, 0
                )
                if items:
                    if self.tree.findItems(
                        child, Qt.MatchRecursive | Qt.MatchExactly, 0
                    ):
                        continue  # already added
                    parent_item = items[0]
                    child_item = QTreeWidgetItem([child])
                    parent_item.addChild(child_item)
                else:
                    Log.w(TAG, f'Parent item "{parent}" not found in tree')
        if selected_model_name:
            items = self.tree.findItems(
                selected_model_name, Qt.MatchRecursive | Qt.MatchExactly, 0
            )
            if items:
                items[0].setText(0, f"{selected_model_name}\t⬅️")
                self.tree.setCurrentItem(items[0])
        else:
            Log.w(TAG, "Selected model not found in training tree")

        if hasattr(self, "details_win"):
            self.details_win.close()

        self.details_win = QWidget()
        self.details_win.setObjectName("modelDetailsWin")
        details_layout = QVBoxLayout()
        self.details_win.setLayout(details_layout)

        self.tree.expandAll()  # optional

        switch_btn = QPushButton("Switch to Selected")
        switch_btn.setDefault(True)
        switch_btn.setProperty("class", "primary")
        close_btn = QPushButton("Close")

        switch_btn.clicked.connect(
            lambda: self.select_model_by_name(
                self.tree.currentItem().text(0).split("\t")[0]
            )
        )
        switch_btn.clicked.connect(self.details_win.close)
        close_btn.clicked.connect(self.details_win.close)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(switch_btn)
        buttons_layout.addWidget(close_btn)

        details_layout.addWidget(self.tree)
        details_layout.addLayout(buttons_layout)

        self.details_win.setWindowTitle("Model Training Tree")
        self.details_win.show()

    """ Base class methods override to emulate QFileDialog behavior """

    def accept(self) -> None:
        """Emit fileSelected and close the dialog, or reject if no model is chosen.

        Overrides :meth:`QDialog.accept` to emit :attr:`fileSelected` with the
        selected model path before closing. If no model is selected the dialog
        is rejected instead.
        """
        if self.selected_model:
            self.fileSelected.emit(self.selected_model)
            super().accept()
        else:
            super().reject()

    def directory(self) -> QDir:
        """Return the current models directory as a QDir.

        Returns:
            QDir: The directory currently being scanned for models.
        """
        return QDir(self.models_directory)

    def setDirectory(self, directory: str) -> None:
        """Set the models directory and refresh the model lists.

        Args:
            directory (str): Absolute path to the new models directory.
        """
        self.models_directory = directory
        self.populate_models()

    def setOption(self, option, on=...) -> None:
        """No-op stub to satisfy the QFileDialog interface.

        Args:
            option: Ignored.
            on: Ignored.
        """
        pass

    def setFileMode(self, mode) -> None:
        """No-op stub to satisfy the QFileDialog interface.

        Args:
            mode: Ignored.
        """
        pass

    def setNameFilter(self, filter) -> None:
        """No-op stub to satisfy the QFileDialog interface.

        Args:
            filter: Ignored.
        """
        pass

    def setViewMode(self, mode) -> None:
        """No-op stub to satisfy the QFileDialog interface.

        Args:
            mode: Ignored.
        """
        pass

    def selectNameFilter(self, filter) -> None:
        """No-op stub to satisfy the QFileDialog interface.

        Args:
            filter: Ignored.
        """
        pass

    def selectedFiles(self) -> list[str]:
        """Return the selected model file path as a list.

        Returns:
            list[str]: A single-element list containing the selected model's
                absolute file path, or an empty list if no model is selected.
        """
        if self.selected_model:
            return [self.selected_model]
        return []

    def show(self) -> None:
        """Show the dialog and refresh the model lists.

        Overrides :meth:`QDialog.show` to repopulate the lists with any
        models added or removed since the dialog was last shown.
        """
        super().show()
        self.populate_models()
