from PyQt5.QtWidgets import QDialog, QFileDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QLabel, QScrollArea, QTreeWidget, QTreeWidgetItem, QWidget
from PyQt5.QtCore import Qt, QFileInfo, QDir, pyqtSignal
# import zipfile
import json
from datetime import datetime
import os

TAG = "[ModelSelectionDialog]"

try:
    from QATCH.common.logger import Logger as Log

except ImportError:

    class Log:
        @staticmethod
        def d(tag, msg=""): print("DEBUG:", tag, msg)
        @staticmethod
        def i(tag, msg=""): print("INFO:", tag, msg)
        @staticmethod
        def w(tag, msg=""): print("WARNING:", tag, msg)
        @staticmethod
        def e(tag, msg=""): print("ERROR:", tag, msg)

## TODO: Implement pinning, renaming, detailed view features using VersionManager's index.json

class QFancyListWidget(QListWidget):
    def mousePressEvent(self, event):
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
    """
    Custom model selection dialog for VisQ-AI models.
    """
    
    fileSelected: pyqtSignal = pyqtSignal(str)

    def __init__(self, models_directory=os.getcwd(), parent=None):
        super().__init__(parent)
        self.models_directory = models_directory
        self.pinned_models = []
        self.pinned_names = {}
        self.selected_model = None
        
        self.init_ui()
        self.populate_models()
    
    def init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("VisQ-AI Model Selection")
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout()
        
        # Pinned models section
        self.pinned_label = QLabel("📌 PINNED MODELS")
        self.pinned_list = QFancyListWidget()
        self.pinned_list.setAlternatingRowColors(True)
        self.pinned_list.setStyleSheet("""
            QListWidget::item {
                padding: 8px 5px;  /* vertical horizontal */
            }
        """)
        
        # Recent models section
        self.recent_label = QLabel("📁 RECENT MODELS")
        self.recent_list = QFancyListWidget()
        self.recent_list.setAlternatingRowColors(True)
        self.recent_list.setStyleSheet("""
            QListWidget::item {
                padding: 8px 5px;  /* vertical horizontal */
            }
        """)
        
        # Detail panel
        self.detail_panel = QLabel("Select a model to view details")
        self.detail_panel.setWordWrap(True)
        self.detail_panel.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.detail_panel.setStyleSheet("""
            QLabel {
                padding: 8px 5px;  /* vertical horizontal */
            }
        """)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.detail_panel)
        
        # Buttons
        self.view_details_btn = QPushButton("View Details")
        self.rename_btn = QPushButton("Rename")
        self.pin_btn = QPushButton("Pin/Unpin")
        self.delete_btn = QPushButton("Delete")
        self.select_btn = QPushButton("Select")
        self.cancel_btn = QPushButton("Cancel")

        # Set Select button as default
        self.select_btn.setDefault(True)
        
        # Add to layout
        layout.addWidget(self.pinned_label)
        layout.addWidget(self.pinned_list)
        layout.addWidget(self.recent_label)
        layout.addWidget(self.recent_list)
        layout.addWidget(self.scroll_area)

        # Add buttons
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.view_details_btn)
        hbox1.addWidget(self.rename_btn)
        hbox1.addWidget(self.pin_btn)
        hbox1.addWidget(self.delete_btn)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.select_btn)
        hbox2.addWidget(self.cancel_btn)
        
        layout.addLayout(hbox1)
        layout.addLayout(hbox2)

        self.setLayout(layout)
        
        # Connect signals
        self.pinned_list.itemClicked.connect(self.on_model_selected)
        self.recent_list.itemClicked.connect(self.on_model_selected)
        self.pinned_list.currentItemChanged.connect(self.on_model_selected)
        self.recent_list.currentItemChanged.connect(self.on_model_selected)
        self.pin_btn.clicked.connect(self.toggle_pin)
        self.rename_btn.clicked.connect(self.rename_model)
        self.view_details_btn.clicked.connect(self.show_detailed_view)
        self.pinned_list.itemDoubleClicked.connect(self.accept)
        self.recent_list.itemDoubleClicked.connect(self.accept)
        self.select_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    
    def populate_models(self):
        """Scan directory and populate model lists."""
        self.all_models = self.scan_model_directory()
        
        # Sort by modification time (most recent first)
        self.all_models.sort(key=lambda x: x['created'], reverse=True)
        
        # Separate pinned and unpinned
        pinned = [m for m in self.all_models if m['filename'] in self.pinned_models]
        unpinned = [m for m in self.all_models if m['filename'] not in self.pinned_models]

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
        """Generate display text for a model item."""

        age_days = (datetime.now() - datetime.fromtimestamp(model['created'])).days
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
            created_time = datetime.fromtimestamp(model['created']).strftime("%Y-%m-%d %H:%M:%S")
        

        parent_model = model['metadata'].get('parent_model', 'N/A')
        if parent_model == 'N/A':
            # Try alternative key, from legacy models
            parent_model = model['metadata'].get('base_model', 'N/A')
        if parent_model != 'N/A':
            parent_model = os.path.basename(parent_model)
            parent_model = self.pinned_names.get(parent_model, parent_model)

        learned_runs = model['metadata'].get('learned_runs', [])
        if learned_runs:
            num_runs = len(learned_runs)
        else:
            num_runs = 0

        size_value = model['size']
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
        
        model_text: str = self.pinned_names.get(model['filename'], model['filename'])
        created_text: str = f"Created: {created_time}"
        parent_text: str = f"Parent Model: {parent_model}"
        runs_text: str = f"Runs: {num_runs}"
        size_text: str = f"Size: {size_value:.2f} {size_units}"

        return f"""
{model_text.ljust(42)} \t\t\t\t\t {created_text}
{parent_text.ljust(42)} \t\t {runs_text.ljust(10)} \t\t\t {size_text}
                """.strip()
    
    def scan_model_directory(self):
        """
        Scan models directory for VisQ-AI model ZIP files.
        
        Returns:
            list: List of model info dictionaries
        """
        models = []
        index_data = {}
        
        for filename in os.listdir(self.models_directory):
            if filename == 'index.json':
                index_path = os.path.join(self.models_directory, filename)
                with open(index_path, 'r') as f:
                    index_data = json.load(f)

            if not filename.startswith('VisQAI'):
                continue
            if not filename.endswith('.zip'):
                continue
            
            filepath = os.path.join(self.models_directory, filename)
            
            try:
                # Parse metadata from model ZIP
                metadata, sha = self.parse_model_metadata(index_data, filepath)
                
                models.append({
                    'filename': filename,
                    'filepath': filepath,
                    'pinned_name': self.pinned_names.get(filename, metadata.get('name', filename)),
                    'created': os.path.getctime(filepath),
                    'size': os.path.getsize(filepath),
                    'metadata': metadata,
                    'sha': sha
                })
            except Exception as e:
                print(f"Error parsing model {filename}: {e}")
                continue
        
        return models
    
    def parse_model_metadata(self, index_data, model_path):
        """
        Parse metadata from model ZIP file.
        
        Args:
            model_path: Path to model ZIP file
            
        Returns:
            dict: Model metadata
        """
        # Search for metadata in index_data where filename contains the start of the sha hash
        base_name = os.path.basename(model_path)
        for entry in index_data.keys():
            if base_name.endswith(f"{entry[0:7]}.zip"):
                metadata = index_data[entry].get('metadata', {})

                pinned_name = metadata.get('pinned_name', None)
                if pinned_name:
                    self.pinned_names[base_name] = pinned_name

                is_pinned = metadata.get('pin', False)
                if is_pinned and base_name not in self.pinned_models:
                    self.pinned_models.append(base_name)
                elif not is_pinned and base_name in self.pinned_models:
                    self.pinned_models.remove(base_name)

                return metadata, entry

        return {}, None # entry not found
    
    def on_model_selected(self, item=None, previous=NameError):
        """Handle model selection from list."""
        if not item:
            # Update detail panel to reflect no selection
            self.detail_panel.setText("Select a model to view details")
            self.selected_model = None
            return
        if not previous:
            return
        
        if not self.pinned_list.isAncestorOf(item.listWidget()):
            self.pinned_list.clearSelection()
        if not self.recent_list.isAncestorOf(item.listWidget()):
            self.recent_list.clearSelection()

        selected_text = item.text()
        model_name_line = selected_text.splitlines()[0]
        model_name = model_name_line.split('\t')[0].strip()
        
        # Find model info
        for model in self.all_models:
            if self.pinned_names.get(model['filename'], model['filename']) == model_name:
                self.selected_model = model['filepath']
                detail_text = f"Model: {self.pinned_names.get(model['filename'], model['filename'])}\n"
                detail_text += json.dumps(model['metadata'], indent=4)
                self.detail_panel.setText(detail_text)
                break

    def toggle_pin(self):
        """Toggle pin status of selected model."""
        # Implementation for pinning/unpinning within `index.json`
        pass
    
    def rename_model(self):
        """Show rename dialog for selected model."""
        # Implementation for renaming within `index.json`
        pass
    
    def show_detailed_view(self):
        """Show detailed view panel with full model information."""
        # Implementation for detailed view from `index.json`

        # Parse parent model and learned runs from metadata and display them
        item_list = self.pinned_list
        indices = item_list.selectedIndexes()
        if not len(indices) == 1:
            item_list = self.recent_list
            indices = item_list.selectedIndexes()
        item = item_list.currentItem()
        if not item:
            Log.w("No model selected for detailed view")
            return
        
        selected_text = item.text()
        model_name_line = selected_text.splitlines()[0]
        model_name = model_name_line.split('\t')[0].strip()
        selected_model_name = model_name

        # Find model info
        training_tree = {}
        key_index = 0
        while True:
            model_found = False
            for model in self.all_models:
                if self.pinned_names.get(model['filename'], model['filename']) == model_name:
                    metadata = model['metadata']
                    parent = metadata.get('parent', None)
                    children = metadata.get('children', [])
                    if not key_index in training_tree:
                        training_tree[key_index] = []
                    for child in children:
                        child_model = next((m for m in self.all_models if m['sha'] == child), None)
                        if not child_model:
                            Log.w(TAG, f"Child model sha \"{child}\" not found in model list")
                        child_name = self.pinned_names.get(child_model['filename'], child_model['filename'])
                        training_tree[key_index].append((model_name, child_name))
                    if parent:
                        key_index -= 1
                        if not key_index in training_tree:
                            training_tree[key_index] = []
                        if parent == 'base':
                            parent_name = 'Base Model'
                        else:
                            parent_model = next((m for m in self.all_models if m['sha'] == parent), None)
                            if not parent_model:
                                Log.w(TAG, f"Parent model sha \"{parent}\" not found in model list")
                            parent_name = self.pinned_names.get(parent_model['filename'], parent_model['filename'])
                    elif parent := metadata.get('parent_model', None):
                        # Found parent model in legacy metadata, missing sha hash reference
                        parent_name = self.pinned_names.get(os.path.basename(parent), os.path.basename(parent))
                    else:
                        parent = 'base' # stop searching
                        parent_name = 'Unspecified Parent'
                    training_tree[key_index].append((parent_name, model_name))
                    model_name = parent_name if parent != 'base' else None
                    model_found = True
                    break
            if not model_name:
                break
            if not model_found:
                Log.e(TAG, f"Parent model \"{parent}\" not found in model list")
                break

        training_tree = dict(sorted(training_tree.items()))

        tree = QTreeWidget()
        tree.setHeaderLabels(["Name"])

        # detail_text = "Training Tree:\n"
        first_i = None
        for i, pair in training_tree.items():
            if first_i is None:
                # detail_text += "Base Model\n"
                first_i = i
                base_tree = QTreeWidgetItem([pair[0][0]])
                tree.addTopLevelItem(base_tree)
            for parent, child in pair:
                # detail_text += "    " * (i - first_i) + f"{child}\n"
                items = tree.findItems(parent, Qt.MatchRecursive | Qt.MatchExactly, 0)
                if items:
                    if tree.findItems(child, Qt.MatchRecursive | Qt.MatchExactly, 0):
                        continue  # already added
                    parent_item = items[0]
                    child_item = QTreeWidgetItem([child])
                    parent_item.addChild(child_item)
                else:
                    Log.w(TAG, f"Parent item \"{parent}\" not found in tree")
        if selected_model_name:
            items = tree.findItems(selected_model_name, Qt.MatchRecursive | Qt.MatchExactly, 0)
            if items:
                items[0].setText(0, f"{selected_model_name}\t⬅️")
                tree.setCurrentItem(items[0])
        else:
            Log.w(TAG, "Selected model not found in training tree")
        

        if hasattr(self, 'details_win'):
            self.details_win.close()
        
        self.details_win = QWidget()
        details_layout = QVBoxLayout()
        self.details_win.setLayout(details_layout)

        tree.expandAll()   # optional

        details_layout.addWidget(tree)
        self.details_win.setWindowTitle("Model Training Tree")
        self.details_win.show()


    """ Base class methods override to emulate QFileDialog behavior """

    def accept(self) -> None:
        """Handle dialog acceptance."""
        if self.selected_model:
            self.fileSelected.emit(self.selected_model)
            super().accept()
        else:
            super().reject()

    def directory(self) -> QDir:
        return QDir(self.models_directory)

    def setDirectory(self, directory: str) -> None:
        """Set the directory to scan for models."""
        self.models_directory = directory
        self.populate_models()

    def setOption(self, option, on = ...) -> None:
        pass

    def setFileMode(self, mode) -> None:
        pass

    def setNameFilter(self, filter) -> None:
        pass

    def setViewMode(self, mode) -> None:
        pass

    def selectNameFilter(self, filter) -> None:
        pass

    def selectedFiles(self) -> list[str]:
        """Return the selected model file path."""
        if self.selected_model:
            return [self.selected_model]
        return []

    def show(self) -> None:
        super().show()
        self.populate_models()

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    assets = r'C:\Users\Alexander J. Ross\Documents\QATCH Work\v2.6x branch\commit-branch\QATCH\VisQAI\assets'
    dialog = ModelSelectionDialog(models_directory=assets)
    dialog.fileSelected.connect(lambda f: print(f"File selected: {f}"))
    if dialog.exec_() == QDialog.Accepted:
        selected_model = dialog.selected_model
        print(f"Selected model: {selected_model}")
    sys.exit(0)