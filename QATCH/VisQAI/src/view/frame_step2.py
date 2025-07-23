try:
    from QATCH.ui.popUp import PopUp
    from QATCH.core.constants import Constants
    from QATCH.common.logger import Logger as Log
    from QATCH.common.architecture import Architecture
except:
    print("Running VisQAI as standalone app")

    class Log:
        def d(tag, msg=""): print("DEBUG:", tag, msg)
        def i(tag, msg=""): print("INFO:", tag, msg)
        def w(tag, msg=""): print("WARNING:", tag, msg)
        def e(tag, msg=""): print("ERROR:", tag, msg)

from PyQt5 import QtCore, QtGui, QtWidgets
import os
from typing import TYPE_CHECKING

try:
    from src.utils.constraints import Constraints
    from src.utils.icon_utils import IconUtils
    from src.utils.list_utils import ListUtils
    from src.view.constraints_ui import ConstraintsUI
    if TYPE_CHECKING:
        from src.view.frame_step1 import FrameStep1
        from src.view.main_window import VisQAIWindow

except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.utils.constraints import Constraints
    from QATCH.VisQAI.src.utils.icon_utils import IconUtils
    from QATCH.VisQAI.src.utils.list_utils import ListUtils
    from QATCH.VisQAI.src.view.constraints_ui import ConstraintsUI
    if TYPE_CHECKING:
        from QATCH.VisQAI.src.view.frame_step1 import FrameStep1
        from QATCH.VisQAI.src.view.main_window import VisQAIWindow

TAG = "[FrameStep2]"


class FrameStep2(QtWidgets.QDialog):
    def __init__(self, parent=None, step=2):
        super().__init__(parent)
        self.parent: VisQAIWindow = parent
        self.step = step

        self.model_path = None

        if step == 4:
            self.setWindowTitle("Learn")
        elif step == 6:
            self.setWindowTitle("Optimize")
        else:
            self.setWindowTitle(f"FrameStep{step}")

        if step == 6:  # Optimize
            self.constraints_ui = ConstraintsUI(self, self.step)
            self.constraints = None

        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        main_layout.addWidget(splitter)
        top_menu_widget = QtWidgets.QWidget()
        top_menu_layout = QtWidgets.QVBoxLayout(top_menu_widget)

        # Browse model layout
        self.model_dialog = QtWidgets.QFileDialog()
        self.model_dialog.setOption(
            QtWidgets.QFileDialog.DontUseNativeDialog, True)
        model_path = os.path.join(
            os.getcwd(), "QATCH/VisQAI/assets")
        self.model_dialog.setDirectory(model_path)
        self.model_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self.model_dialog.setNameFilter("VisQ.AI Models (VisQAI-*.zip)")
        self.model_dialog.selectNameFilter("VisQ.AI Models (VisQAI-*.zip)")

        self.select_model_group = QtWidgets.QGroupBox("Select Model")
        select_model_layout = QtWidgets.QHBoxLayout(self.select_model_group)
        self.select_model_btn = QtWidgets.QPushButton("Browse...")
        self.select_model_label = QtWidgets.QLineEdit()
        self.select_model_label.setPlaceholderText("No model selected")
        self.select_model_label.setReadOnly(True)
        select_model_layout.addWidget(self.select_model_btn)
        select_model_layout.addWidget(self.select_model_label)
        select_model_layout.addStretch()

        # Action summary layout
        group_title = "Action Summary"
        group_text = "The following changes will occur:"
        if step == 4:  # Learn
            group_title = "Learn Summary"
            group_text = "The following experiments will be learned:"
        if step == 6:  # Optimize
            group_title = "Optimize Summary"
            group_text = "The following features will be optimized:"
        self.summary_group = QtWidgets.QGroupBox(group_title)
        summary_layout = QtWidgets.QVBoxLayout(self.summary_group)
        summary_label = QtWidgets.QLabel(group_text)
        self.summary_text = QtWidgets.QPlainTextEdit()
        self.summary_text.setPlaceholderText("No changes")
        self.summary_text.setReadOnly(True)
        if step == 6:
            optimize_btn_layout = QtWidgets.QHBoxLayout()
            optimize_feat_btn = QtWidgets.QPushButton("Define...")
            optimize_feat_btn.clicked.connect(self.load_changes)
            optimize_btn_layout.addWidget(optimize_feat_btn)
            optimize_btn_layout.addStretch()
            summary_layout.addLayout(optimize_btn_layout)
        summary_layout.addWidget(summary_label)
        summary_layout.addWidget(self.summary_text)

        # Progress layout
        self.progress_group = QtWidgets.QGroupBox("Learning Progress")
        progress_layout = QtWidgets.QVBoxLayout(self.progress_group)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_buttons = QtWidgets.QWidget()
        self.progress_btn_layout = QtWidgets.QHBoxLayout(self.progress_buttons)
        self.progress_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_resume = QtWidgets.QPushButton("Resume")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.progress_btn_layout.addWidget(self.btn_pause)
        self.progress_btn_layout.addWidget(self.btn_resume)
        self.progress_btn_layout.addWidget(self.btn_cancel)
        self.progress_label = QtWidgets.QLabel()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        self.progress_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_buttons)

        # Top menu layout
        top_menu_layout.addWidget(self.select_model_group)
        top_menu_layout.addWidget(self.summary_group)
        top_menu_layout.addWidget(self.progress_group)
        splitter.addWidget(top_menu_widget)

        # Bottom split view layout
        figure = QtWidgets.QLabel("[Figure here]")
        figure.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        splitter.addWidget(figure)
        splitter.setSizes([1, 1000])

        # Signals
        self.select_model_btn.clicked.connect(self.model_dialog.show)
        self.model_dialog.fileSelected.connect(self.model_selected)
        self.btn_resume.clicked.connect(
            lambda: self.progress_bar.setValue(self.progress_bar.value()+1))
        self.btn_resume.clicked.connect(
            getattr(self, "learn" if step == 4 else "optimize")
        )
        self.btn_cancel.clicked.connect(
            lambda: self.progress_bar.setValue(0))
        self.btn_cancel.clicked.connect(
            lambda: self.model_selected(None))
        self.progress_bar.valueChanged.connect(
            lambda v: self.progress_label.setText(
                f"{v}% - " + ("Learning" if self.step == 4 else "Optimizing")))

        self.progress_bar.setValue(0)

    def on_tab_selected(self):

        # Reload all excipients from DB
        self.load_all_excipient_types()

        # Select a pre-selected model, if none selected here
        if not self.model_path:
            select_tab: FrameStep1 = self.parent.tab_widget.widget(0)
            suggest_tab: FrameStep1 = self.parent.tab_widget.widget(1)
            import_tab: FrameStep1 = self.parent.tab_widget.widget(2)
            learn_tab: FrameStep2 = self.parent.tab_widget.widget(3)
            predict_tab: FrameStep1 = self.parent.tab_widget.widget(4)
            optimize_tab: FrameStep2 = self.parent.tab_widget.widget(5)
            all_model_paths = [select_tab.model_path,
                               suggest_tab.model_path,
                               import_tab.model_path,
                               learn_tab.model_path,
                               predict_tab.model_path,
                               optimize_tab.model_path]
            found_model_path = next(
                (x for x in all_model_paths if x is not None), None)
            if found_model_path:
                self.model_selected(found_model_path)

        if self.step == 4:  # learn
            self.load_changes()

    def load_all_excipient_types(self):
        self.proteins: list[str] = []
        self.buffers: list[str] = []
        self.surfactants: list[str] = []
        self.stabilizers: list[str] = []
        self.salts: list[str] = []
        self.class_types: list[str] = []
        self.proteins_by_class: dict[str, str] = {}

        self.proteins, self.buffers, self.surfactants, \
            self.stabilizers, self.salts, \
            self.class_types, self.proteins_by_class = ListUtils.load_all_excipient_types(
                self.parent.ing_ctrl)

        Log.d("Proteins:", self.proteins)
        Log.d("Buffers:", self.buffers)
        Log.d("Surfactants:", self.surfactants)
        Log.d("Stabilizers:", self.stabilizers)
        Log.d("Salts:", self.salts)
        Log.d("Class Types:", self.class_types)
        Log.d("Proteins By Class:", self.proteins_by_class)

    def model_selected(self, path: str | None):
        self.model_path = path

        if path is None:
            self.select_model_label.clear()
            return

        self.select_model_label.setText(
            path.split('\\')[-1].split('/')[-1].split('.')[0])

    def load_changes(self):
        changes = []  # list of changes

        if self.step == 4:  # learn
            select_run_tab: FrameStep1 = self.parent.tab_widget.widget(0)
            experiments_tab: FrameStep1 = self.parent.tab_widget.widget(2)
            changes.append(select_run_tab.select_label.text())
            changes.extend(experiments_tab.all_files.keys())

        if self.step == 6:  # optimize
            self.constraints_ui.add_suggestion_dialog()
            self.constraints_ui.suggest_dialog.exec_()  # blocking

            if not self.constraints:
                Log.w(TAG, "No constraints set for optimization.")
                return

            bounds, encoding = self.constraints.build()
            for feature in encoding:
                if feature["type"] == "cat":
                    if feature["feature"] not in self.constraints._choices:
                        Log.d(
                            TAG, f"No choices set for categorical feature '{feature['feature']}'.")
                        continue
                    choices = self.constraints._choices.get(feature["feature"])
                    changes.append(
                        f"{feature['feature']}: {', '.join([ing.name for ing in choices])}")
                elif feature["type"] == "num":
                    if feature["feature"] not in self.constraints._ranges:
                        Log.d(
                            TAG, f"No range set for numeric feature '{feature['feature']}'.")
                        continue
                    low, high = bounds[encoding.index(feature)]
                    changes.append(f"{feature['feature']}: {low} - {high}")
                else:
                    Log.w(
                        TAG, f"Unknown feature type '{feature['type']}' for feature '{feature['feature']}'.")

        self.summary_text.setPlainText("\n".join(changes).strip())

    def set_constraints(self, constraints):
        """Set the constraints for the optimizer."""
        if self.step != 6:
            Log.w(TAG, "set_constraints called, but step is not 6 (Optimize).")
            return

        if not isinstance(constraints, Constraints):
            Log.e(TAG, "set_constraints expects a Constraints instance.")
            return

        self.constraints = constraints
        Log.i(TAG, "Constraints set for optimization.")

    def learn(self):
        raise NotImplementedError()

    def optimize(self):
        raise NotImplementedError()
