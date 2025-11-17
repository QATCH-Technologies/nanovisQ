try:
    from QATCH.ui.popUp import PopUp
    from QATCH.core.constants import Constants
    from QATCH.common.logger import Logger as Log
    from QATCH.common.architecture import Architecture
except (ModuleNotFoundError, ImportError):
    print("Running VisQAI as standalone app")

    class Log:
        def d(tag, msg=""): print("DEBUG:", tag, msg)
        def i(tag, msg=""): print("INFO:", tag, msg)
        def w(tag, msg=""): print("WARNING:", tag, msg)
        def e(tag, msg=""): print("ERROR:", tag, msg)

from PyQt5 import QtCore, QtGui, QtWidgets
import os
import numpy as np
import time
from typing import Optional
from typing import TYPE_CHECKING
from shutil import make_archive
from pathlib import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d

try:
    from src.models.formulation import Formulation, ViscosityProfile
    from src.models.predictor import Predictor
    from src.threads.executor import Executor, ExecutionRecord
    from src.utils.constraints import Constraints
    from src.utils.icon_utils import IconUtils
    from src.utils.list_utils import ListUtils
    from src.view.constraints_ui import ConstraintsUI
    from src.managers.version_manager import VersionManager
    from src.utils.progress_tracker import Lite_QProgressDialog
    from src.view.model_selection_dialog import ModelSelectionDialog
    if TYPE_CHECKING:
        from src.view.frame_step1 import FrameStep1
        from src.view.main_window import VisQAIWindow

except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.formulation import Formulation, ViscosityProfile
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.threads.executor import Executor, ExecutionRecord
    from QATCH.VisQAI.src.utils.constraints import Constraints
    from QATCH.VisQAI.src.utils.icon_utils import IconUtils
    from QATCH.VisQAI.src.utils.list_utils import ListUtils
    from QATCH.VisQAI.src.view.constraints_ui import ConstraintsUI
    from QATCH.VisQAI.src.managers.version_manager import VersionManager
    from QATCH.VisQAI.src.utils.progress_tracker import Lite_QProgressDialog
    from QATCH.VisQAI.src.view.model_selection_dialog import ModelSelectionDialog
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
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter)
        left_menu_widget = QtWidgets.QWidget()
        left_menu_layout = QtWidgets.QVBoxLayout(left_menu_widget)

        # Browse model layout
        self.model_dialog = ModelSelectionDialog()
        self.model_dialog.setOption(
            QtWidgets.QFileDialog.DontUseNativeDialog, True)
        model_path = os.path.join(Architecture.get_path(),
                                  "QATCH/VisQAI/assets")
        if os.path.exists(model_path):
            # working or bundled directory, if exists
            self.model_dialog.setDirectory(model_path)
        else:
            # fallback to their local logged data folder
            self.model_dialog.setDirectory(Constants.log_prefer_path)
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
        self.progress_label = QtWidgets.QLabel()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        self.progress_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setText("0% - Not Started")
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_buttons)

        # Left menu buttons
        self.btn_start = QtWidgets.QPushButton("Start ")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_next = QtWidgets.QPushButton("Next Step: ")
        if self.step == 4:
            self.btn_start.setText(self.btn_start.text() + "Learning")
            self.btn_next.setText(self.btn_next.text() + "Predict")
        else:  # 6: Optimize
            self.btn_start.setText(self.btn_start.text() + "Optimizing")
            self.btn_next.setText("Final Report")
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.btn_cancel)
        button_layout.addWidget(self.btn_next)

        # Top menu layout
        left_menu_layout.addWidget(self.select_model_group)
        left_menu_layout.addWidget(self.summary_group)
        left_menu_layout.addWidget(self.btn_start)
        left_menu_layout.addLayout(button_layout)

        splitter.addWidget(left_menu_widget)

        # Set fixed width for left widget
        left_menu_widget.setMinimumWidth(450)

        # Bottom split view layout
        self.run_figure = Figure()
        self.run_figure_valid = False
        self.run_canvas = FigureCanvas(self.run_figure)

        # Configure right widget and layout
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.addWidget(self.progress_group)
        right_layout.addWidget(self.run_canvas, 1)

        # Add to main splitter
        splitter.addWidget(right_widget)
        splitter.setSizes([10, 10000])

        # Signals
        self.select_model_btn.clicked.connect(self.model_dialog.show)
        global_handler = getattr(
            self.parent, 'set_global_model_path', None)
        self.model_dialog.fileSelected.connect(
            global_handler if callable(global_handler) else self.model_selected)
        self.btn_resume.clicked.connect(
            lambda: self.progress_bar.setValue(self.progress_bar.value()+1))
        self.btn_start.clicked.connect(
            getattr(self, "learn" if step == 4 else "optimize")
        )
        # self.btn_cancel.clicked.connect(
        #     lambda: self.model_selected(None))
        self.btn_cancel.clicked.connect(self.action_cancel)
        self.btn_next.clicked.connect(self.proceed_to_next_step)

        self.progress_bar.setValue(0)

    def on_tab_selected(self):

        # Reload all excipients from DB
        self.load_all_ingredients()

        # Select a pre-selected model, if none selected here
        if not self.model_path:
            select_tab: FrameStep1 = self.parent.tab_widget.widget(0)
            suggest_tab: FrameStep1 = self.parent.tab_widget.widget(1)
            import_tab: FrameStep1 = self.parent.tab_widget.widget(2)
            learn_tab: FrameStep2 = self.parent.tab_widget.widget(3)
            predict_tab: FrameStep1 = self.parent.tab_widget.widget(5)
            optimize_tab: FrameStep2 = self.parent.tab_widget.widget(6)
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
            self.action_cancel()
            self.load_changes()

    def action_cancel(self):
        if hasattr(self, "progressState"):
            self.progressState.cancel()
        if hasattr(self, "timer") and self.timer.isActive():
            self.progress_label.setText("% - Canceling... please wait...")
        else:
            self.progress_label.setText("0% - Not Started")
            self.progress_bar.setValue(0)
        self.run_figure.clear()
        self.run_figure_valid = False
        self.run_canvas.draw()

    def load_all_ingredients(self):
        self.proteins: list[str] = []
        self.buffers: list[str] = []
        self.surfactants: list[str] = []
        self.stabilizers: list[str] = []
        self.salts: list[str] = []
        self.excipients: list[str] = []
        self.class_types: list[str] = []
        self.proteins_by_class: dict[str, str] = {}

        self.proteins, self.buffers, self.surfactants, \
            self.stabilizers, self.salts, self.excipients, \
            self.class_types, self.proteins_by_class = ListUtils.load_all_ingredient_types(
                self.parent.ing_ctrl)

        Log.d("Proteins:", self.proteins)
        Log.d("Buffers:", self.buffers)
        Log.d("Surfactants:", self.surfactants)
        Log.d("Stabilizers:", self.stabilizers)
        Log.d("Salts:", self.salts)
        Log.d("Excipients:", self.excipients)
        Log.d("Class Types:", self.class_types)
        Log.d("Proteins By Class:", self.proteins_by_class)

    def model_selected(self, path: Optional[str] = None):
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
            changes.append(select_run_tab.select_label.text())
            changes.extend(self.parent.import_run_names)

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

    def proceed_to_next_step(self):
        if self.step == 4:
            # proceed to step 6: Learn -> Predict
            if self.parent is not None:
                i = self.parent.tab_widget.currentIndex()
                self.parent.tab_widget.setCurrentIndex(i+2)
        else:
            # show final report
            raise NotImplementedError(
                "Final report generation not implemented.")

    def _get_viscosity_list(self, formulation: Formulation) -> list:
        rate_list = []
        vp = formulation.viscosity_profile
        for rate in [100, 1000, 10000, 100000, 15000000]:
            rate_list.append(vp.get_viscosity(rate))
        return rate_list

    def update_ui_next_step(self, this_idx):
        value_0_to_100 = ((this_idx - self.progressState.minimum()) * 100 //
                          (self.progressState.maximum() - self.progressState.minimum()))
        self.progress_bar.setValue(value_0_to_100)

        if not self.predictor.save_path():

            lines = self.summary_text.toPlainText().splitlines()
            if len(lines) > this_idx + 1:
                run_label = lines[this_idx + 1]
            else:
                run_label = f"Import #{this_idx + 1}"
            format_str = "{}% - Learning run \"{}\"...".format(
                value_0_to_100, run_label)
            self.progress_label.setText(format_str)

            try:
                form_idx = self.parent.import_run_names.index(run_label)
            except ValueError:
                Log.e(
                    f"ERROR: Failed to find import formulation index for {run_label}")
                form_idx = this_idx

            vf = (0 <= form_idx < len(self.parent.import_formulations))
            if not vf:
                Log.e(
                    f"ERROR: form_idx {form_idx} out of range; using {this_idx}")
                form_idx = min(max(this_idx, 0), len(
                    self.parent.import_formulations) - 1)

            vp_obj = self.parent.import_formulations[form_idx].viscosity_profile if vf else None
            if vf and vp_obj is not None and getattr(vp_obj, "is_measured", False):
                # Get the viscosity profile or y target to update with.
                vp = self._get_viscosity_list(
                    self.parent.import_formulations[form_idx])
                self.plot_figure(vp)
        else:
            self.run_figure.clear()
            self.run_figure_valid = False
            self.run_canvas.draw()

    def learn(self):

        if self.model_path is None or not Path(self.model_path).exists():
            Log.e("No model selected. Please select a model to train first.")
            return

        self.predictor = Predictor(zip_path=self.model_path)
        select_df = self.parent.select_formulation.to_dataframe(
            encoded=False, training=True)

        # Progress bar has 1 interim step:
        # value -1 = "Learning run {select_df}...""
        # value 0 = "Learning run {import_df[0]}..."
        # [...]
        # value {total_steps-1} = "Learning run {import_df[-1]}..."
        # value {total_steps} = 100% (finished)
        total_steps = len(self.parent.import_formulations)
        self.learn_idx = -1

        self.progressState = Lite_QProgressDialog(  # QtWidgets.QProgressDialog(
            "Learning...", "Cancel", self.learn_idx, total_steps, self)
        # progressState is retained for cancel state and min/max tracking

        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.check_finished)
        self.timer.start()

        def learn_run_result(record: Optional[ExecutionRecord] = None):
            """NOTE: This method runs on a different thread than the main UI and cannot interact with PyQt5 objects without causing app instability."""

            if record and record.exception:
                # NOTE: Progress bar and timer will end on next call to `check_finished()`
                Log.e(
                    f"Error occurred while updating the model: {record.exception}")
                return

            if self.progressState.wasCanceled():
                Log.w("Learning canceled!")
                return

            # Update time tracking for learning speed
            now = time.time()
            self.run_learn_time = now - self.last_learn_start_time
            self.last_learn_start_time = now

            self.learn_idx += 1  # step to next run queued
            self.progressState.setValue(self.learn_idx)

            if not self.predictor.save_path():

                # process next queued run
                this_idx = self.learn_idx
                lines = self.summary_text.toPlainText().splitlines()
                if len(lines) > this_idx + 1:
                    run_label = lines[this_idx + 1]
                else:
                    run_label = f"Import #{this_idx + 1}"

                try:
                    form_idx = self.parent.import_run_names.index(run_label)
                except ValueError:
                    Log.e(
                        f"ERROR: Failed to find import formulation index for {run_label}")
                    form_idx = this_idx

                vf = (0 <= form_idx < len(self.parent.import_formulations))
                if not vf:
                    Log.e(
                        f"ERROR: form_idx {form_idx} out of range; using {this_idx}")
                    form_idx = min(max(this_idx, 0), len(
                        self.parent.import_formulations) - 1)

                is_final_run = (this_idx == self.progressState.maximum() - 1)
                queued_df = self.parent.import_formulations[form_idx].to_dataframe(
                    encoded=False, training=True)

                vp_obj = self.parent.import_formulations[form_idx].viscosity_profile if vf else None
                if vf and vp_obj is not None and getattr(vp_obj, "is_measured", False):

                    # Get the viscosity profile or y target to update with.
                    vp = self._get_viscosity_list(
                        self.parent.import_formulations[form_idx])

                    # Target needs to be form np.array([[Viscosity_100, ..., Viscosity_15000000]])
                    # Also I have this set so updating does not overwrite the existing model until
                    # we figure out how model storage works
                    self.executor.run(
                        self.predictor,
                        method_name="learn",
                        new_df=queued_df,
                        n_epochs=10,
                        save=True,
                        callback=learn_run_result)

                else:
                    # Schedule next step without recursion
                    QtCore.QTimer.singleShot(0, lambda: learn_run_result(None))
                    Log.w(
                        f"Not learning run #{self.learn_idx+1} because it has no measured Viscosity Profile. Run skipped...")

            else:

                # final queued run learned, pack and commit new saved model...
                try:
                    save_dir = self.predictor.save_path()
                    # Create the archive next to the chosen directory (not CWD)
                    zip_base = os.path.join(
                        self.model_dialog.directory().path(), "VisQAI-model")
                    saved_model = make_archive(
                        base_name=zip_base,
                        format="zip",
                        root_dir=save_dir,
                        base_dir="."
                    )
                    enc_ok = self.predictor.add_security_to_zip(saved_model)
                    if not enc_ok:
                        Log.w(
                            TAG, "Failed to add security to ZIP; committing unencrypted archive.")
                    sha = self.mvc.commit(
                        model_file=saved_model,
                        metadata={
                            "parent_model": os.path.basename(self.model_path),
                            "learned_runs": self.summary_text.toPlainText().splitlines()
                        }
                    )
                    os.remove(saved_model)  # delete temporary ZIP
                    restored_path = self.mvc.get(
                        sha, self.model_dialog.directory().path())
                    # Rename to VisQAI-<sha7>.zip
                    restored_path = Path(restored_path)
                    target_path = restored_path.with_name(
                        f"VisQAI-{sha[:7]}.zip")
                    if target_path.exists():
                        target_path.unlink()
                    restored_path.rename(target_path)
                    Log.i(f"Created new model: {target_path}")
                    self.new_model_path = str(target_path)

                except Exception as e:
                    # NOTE: Progress bar and timer will end on next call to `check_finished()`
                    Log.e(f"Error occurred while saving the model: {e}")
                    return

                finally:
                    # Cleanup temp files when finished
                    self.predictor.cleanup()

        self.executor = Executor()
        self.mvc = VersionManager(
            self.model_dialog.directory().path(), retention=255)

        # Track success/failure of save; used to decide whether to prompt loading
        self.new_model_path = None

        self.last_learn_start_time = time.time()
        self.run_learn_time = 20  # starting assumption

        # Start from 0%, do not increment learn_idx yet
        value_0_to_100 = ((self.learn_idx - self.progressState.minimum()) * 100 //
                          (self.progressState.maximum() - self.progressState.minimum()))
        self.progress_bar.setValue(value_0_to_100)

        lines = self.summary_text.toPlainText().splitlines()
        first_label = lines[0] if lines else "Initial Selection"
        format_str = "{}% - Learning run \"{}\"...".format(
            value_0_to_100, first_label)
        self.progress_label.setText(format_str)

        if self.parent.select_formulation.viscosity_profile.is_measured:

            # Get the viscosity profile or y target to update with.
            vp = self._get_viscosity_list(self.parent.select_formulation)
            self.plot_figure(vp)

            # Target needs to be form np.array([[Viscosity_100, ..., Viscosity_15000000]])
            # Also I have this set so updating does not overwrite the existing model until
            # we figure out how model storage works
            self.executor.run(
                self.predictor,
                method_name="learn",
                new_df=select_df,
                n_epochs=10,
                save=True,
                callback=learn_run_result)

        else:
            Log.w(
                f"Not learning run #{self.learn_idx+1} because it has no measured Viscosity Profile. Run skipped...")
            learn_run_result()

    def calc_limits(self, yall):
        # For log scale, ymin must be > 0
        EPS = 1e-6
        ymin, ymax = EPS, 1000
        lower_limit = np.amin(yall) / 1.5
        power = 1
        while power > -5:
            if lower_limit > 10**power:
                lower_limit = 10**power
                break
            power -= 1
        upper_limit = np.amax(yall) * 1.5
        power = 0
        while power < 5:
            if upper_limit < 10**power:
                upper_limit = 10**power
                break
            power += 1
        if lower_limit <= EPS or lower_limit >= upper_limit:
            Log.d(
                f"Limits were auto-calculated but are in an invalid range! Using ylim [{EPS}, 1000]."
            )
        elif np.isfinite(lower_limit) and np.isfinite(upper_limit):
            Log.d(
                f"Auto-calculated y-range limits for figure are: [{lower_limit}, {upper_limit}]"
            )
            ymin = max(lower_limit, EPS)
            ymax = upper_limit
        else:
            Log.d(
                f"Limits were auto-calculated but were not finite values! Using ylim [{EPS}, 1000]."
            )
        return ymin, ymax

    def plot_figure(self, vp: list):
        self.profile_shears = [100, 1000, 10000, 100000, 15000000]
        self.profile_viscos = vp

        # Helper functions for plotting
        def smooth_log_interpolate(x, y, num=200, expand_factor=0.05):
            xlog = np.log10(x)
            ylog = np.log10(y)
            f_interp = interp1d(xlog, ylog, kind='linear',
                                fill_value='extrapolate')
            xlog_min, xlog_max = xlog.min(), xlog.max()
            margin = (xlog_max - xlog_min) * expand_factor
            xs_log = np.linspace(xlog_min - margin, xlog_max + margin, num)
            xs = 10**xs_log
            ys = 10**f_interp(xs_log)
            return xs, ys

        self.run_figure.clear()
        self.run_figure_valid = False
        ax = self.run_figure.add_subplot(111)
        ax.set_xlabel("Shear rate (s⁻¹)", fontsize=10)
        ax.set_ylabel("Viscosity (cP)", fontsize=10)
        ax.grid(True, which="both", ls=":")
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))

        if len(self.profile_viscos) > 0:
            xs, ys = smooth_log_interpolate(
                self.profile_shears, self.profile_viscos)
            ax.set_xlim(xs.min(), xs.max())
            ax.set_ylim(self.calc_limits(yall=self.profile_viscos))
            ax.plot(self.profile_shears, self.profile_viscos,
                    lw=2.5, color="blue")
            ax.scatter(self.profile_shears, self.profile_viscos,
                       s=40, color="blue", zorder=5)

            # Calculate offset as percentage of y-range
            ylim = ax.get_ylim()
            y_range = max(ylim) - min(ylim)
            offset = y_range * 0.05  # 5% of the y-range
            # Add labels with offset above points
            for i in range(len(self.profile_viscos)):
                ax.text(self.profile_shears[i], self.profile_viscos[i] + offset,
                        f'{self.profile_viscos[i]:.02f}',
                        ha='center', va='bottom',
                        fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

            self.run_figure_valid = True

        else:
            ax.text(0.5, 0.5, "Invalid Results",
                    transform=ax.transAxes,
                    ha='center', va='center',
                    bbox=dict(facecolor='yellow', edgecolor='black'))
        ax.set_xscale("log")
        ax.set_yscale("log")
        self.run_canvas.draw()

    def check_finished(self):
        # at least 1 record expected, but may be more based on task count
        expect_record_count = max(1, self.executor.task_count())
        if self.executor.active_count() == 0 and len(self.executor.get_task_records()) == expect_record_count:
            self.timer.stop()

            self.progress_bar.setValue(100)
            self.progress_label.setText("100% - Finished")
            Log.i(TAG, "Learning finished successfully.")

            if not self.progressState.wasCanceled() and getattr(self, "new_model_path", None):
                QtCore.QTimer.singleShot(1000, self.done_learning)
            elif not self.progressState.wasCanceled():
                Log.w(TAG, "Learning finished but no new model was created.")

        else:
            if not self.progressState.queue().empty():
                self.update_ui_next_step(
                    self.progressState.queue().get_nowait())

            # Increment progress bar periodically to show that the thread isn't frozen solid (~20s per run)
            pct_per_run = 100 // (self.progressState.maximum() -
                                  self.progressState.minimum())
            # secs per run (tracked dynamically based on real timing)
            sec_per_run = self.run_learn_time
            # ms per increment (min 50ms)
            pct_period = max(
                50, int(1000 * sec_per_run // max(1, pct_per_run)))
            max_pct_at = (self.learn_idx + 2) * pct_per_run

            now_ms = int(time.time() * 1000)
            last_bump_ms = int(getattr(self, "last_pct_bump_time", 0))
            if now_ms - last_bump_ms >= pct_period:
                if self.progress_bar.value() < min(99, max(0, max_pct_at)):
                    self.progress_bar.setValue(self.progress_bar.value() + 1)
                    new_progress_text = self.progress_label.text()
                    new_progress_text = new_progress_text.split("%", 1)[1]
                    self.progress_label.setText(
                        f"{self.progress_bar.value()}%{new_progress_text}")
                self.last_pct_bump_time = now_ms

    def done_learning(self):
        # ask user if they want to proceed to using this model for predictions
        reply = QtWidgets.QMessageBox.question(
            self,  # parent
            "Load New Model?",  # title
            "<b>New model created!</b><br/><br/>" +
            "Would you like to load this new model to make predictions?",  # text
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,  # buttons
            QtWidgets.QMessageBox.Yes  # defaultButton
        )

        if reply == QtWidgets.QMessageBox.Yes:
            self.parent.set_global_model_path(self.new_model_path)
            self.proceed_to_next_step()  # proceed to prediction step or update UI
        elif reply == QtWidgets.QMessageBox.No:
            pass  # Do nothing, user declined
        elif reply == QtWidgets.QMessageBox.Cancel:
            pass  # User cancelled

    def optimize(self):
        raise NotImplementedError()
