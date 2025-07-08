try:
    from QATCH.ui.popUp import PopUp
    from QATCH.core.constants import Constants
    from QATCH.common.logger import Logger as Log
except:
    print("Running VisQAI as standalone app")

    class Log:
        def d(tag, msg=""): print("DEBUG:", tag, msg)
        def i(tag, msg=""): print("INFO:", tag, msg)
        def w(tag, msg=""): print("WARNING:", tag, msg)
        def e(tag, msg=""): print("ERROR:", tag, msg)

from PyQt5 import QtCore, QtWidgets
import os
from typing import TYPE_CHECKING

try:
    if TYPE_CHECKING:
        from src.view.frame_step1 import FrameStep1
        from src.view.main_window import VisQAIWindow

except (ModuleNotFoundError, ImportError):
    if TYPE_CHECKING:
        from QATCH.VisQAI.src.view.frame_step1 import FrameStep1
        from QATCH.VisQAI.src.view.main_window import VisQAIWindow


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

        self.load_changes()

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
            pass

        self.summary_text.setPlainText("\n".join(changes).strip())

    def learn(self):
        raise NotImplementedError()

    def optimize(self):
        raise NotImplementedError()
