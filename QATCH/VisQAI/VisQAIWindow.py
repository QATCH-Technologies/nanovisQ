from xml.dom import minidom
from numpy import loadtxt
from PyQt5 import QtCore, QtGui, QtWidgets
import os

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

TAG = "[VisQ.AI]"


class HorizontalTabBar(QtWidgets.QTabBar):

    def tabSizeHint(self, index):
        sz = super().tabSizeHint(index)
        return QtCore.QSize(sz.width() + 20, sz.height() + 40)

    def paintEvent(self, event):
        painter = QtWidgets.QStylePainter(self)
        opt = QtWidgets.QStyleOptionTab()
        for idx in range(self.count()):
            self.initStyleOption(opt, idx)
            opt.shape = QtWidgets.QTabBar.RoundedNorth    # draw as if tabs were on top
            # draw the tab “shell”
            painter.drawControl(QtWidgets.QStyle.CE_TabBarTab, opt)
            # draw the label
            painter.drawControl(QtWidgets.QStyle.CE_TabBarTabLabel, opt)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisQ.AI Mockup")
        self.setMinimumSize(900, 600)
        self.init_ui()

    def init_ui(self):
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setTabBar(HorizontalTabBar())
        self.tab_widget.setTabPosition(QtWidgets.QTabWidget.North)

        self.tab_widget.addTab(FrameStep1(self, 1),
                               "\u2460 Select Candidate")
        self.tab_widget.addTab(FrameStep1(self, 2),
                               "\u2461 Suggest Experiments")
        self.tab_widget.addTab(FrameStep1(self, 3),
                               "\u2462 Import Experiments")
        self.tab_widget.addTab(FrameStep2(self, 4),
                               "\u2463 Learn")
        self.tab_widget.addTab(FrameStep1(self, 5),
                               "\u2464 Predict")

        self.setCentralWidget(self.tab_widget)


class FrameStep1(QtWidgets.QDialog):
    def __init__(self, parent=None, step=1):
        super().__init__(parent)
        self.parent: MainWindow = parent
        self.step = step

        self.all_files = {}

        if step == 1:
            self.setWindowTitle("Select Candidate")
        elif step == 2:
            self.setWindowTitle("Suggest Experiments")
        elif step == 3:
            self.setWindowTitle("Select Experiments")
        else:
            self.setWindowTitle(f"FrameStep{step}")

        # Main layout
        main_layout = QtWidgets.QHBoxLayout(self)
        h_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(h_splitter)

        # Left panel: Candidate selection
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        form_layout = QtWidgets.QFormLayout()
        if step == 1:
            left_group = QtWidgets.QGroupBox("Candidate Run")
        elif step == 2:
            left_group = QtWidgets.QGroupBox("Suggested Runs")
        elif step == 3:
            left_group = QtWidgets.QGroupBox("Experiment Runs")
        elif step == 5:
            left_group = QtWidgets.QGroupBox("Predictions")
        left_group_layout = QtWidgets.QVBoxLayout(left_group)
        left_group_layout.addLayout(form_layout)
        left_layout.addWidget(left_group)

        # Browse candidate
        self.file_dialog = QtWidgets.QFileDialog()
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self.file_dialog.setDirectory("logged_data")
        self.file_dialog.setNameFilter("Captured Runs (capture.zip)")
        self.file_dialog.selectNameFilter("Captured Runs (capture.zip)")
        self.select_candidate = QtWidgets.QPushButton(
            "Add Run..." if step == 3 else "Browse...")
        self.select_label = QtWidgets.QLineEdit()
        self.select_label.setPlaceholderText("No file selected")
        self.select_label.setReadOnly(True)
        # candidate_select = QtWidgets.QHBoxLayout()
        # candidate_select.addWidget(self.select_candidate)
        # candidate_select.addWidget(self.select_label)

        if step == 1:
            form_layout.addRow(self.select_candidate, self.select_label)
        elif step == 2 or step == 3 or step == 5:
            self.list_view = QtWidgets.QListView()
            self.list_view.setEditTriggers(
                QtWidgets.QAbstractItemView.NoEditTriggers)
            self.model = QtGui.QStandardItemModel()
            string_list = []
            if step == 2 or step == 5:
                for i in range(4):
                    base_name = "Suggestion" if step == 2 else "Prediction"
                    string_list.append(f"{base_name} {i+1}")
            for string in string_list:
                self.model.appendRow(QtGui.QStandardItem(string))
            self.list_view.setModel(self.model)
            if step == 1 or step == 3:
                form_layout.addRow(self.select_candidate, self.list_view)
            elif step == 2:
                form_layout.addRow("Experiment:", self.list_view)
            elif step == 5:
                form_layout.addRow("Prediction:", self.list_view)
            if step == 3:
                self.list_view.clicked.connect(
                    lambda: self.file_selected(self.all_files[self.model.itemFromIndex(self.list_view.selectedIndexes()[0]).text()]))
            else:
                self.list_view.clicked.connect(
                    lambda: self.feature_table.setData(self.dummy_features[self.list_view.selectedIndexes()[0].row()]))
            self.btn_remove = QtWidgets.QPushButton("Remove Selected Run")
            self.btn_remove.clicked.connect(
                lambda: self.model.removeRow(self.list_view.selectedIndexes()[0].row()))
            form_layout.addRow("", self.btn_remove)

        self.run_notes = QtWidgets.QTextEdit()
        self.run_notes.setPlaceholderText("None")
        self.run_notes.setReadOnly(True)

        # Run information
        self.run_name = QtWidgets.QLabel()
        self.run_date_time = QtWidgets.QLabel()
        self.run_duration = QtWidgets.QLabel()
        self.run_batch = QtWidgets.QLabel()
        self.run_fill_type = QtWidgets.QLabel()

        # Audits
        self.run_captured = QtWidgets.QLabel()
        self.run_updated = QtWidgets.QLabel()
        self.run_analyzed = QtWidgets.QLabel()

        if step == 2 or step == 5:
            self.run_captured.setText("N/A")
            self.run_updated.setText("N/A")
            self.run_analyzed.setText("N/A")
        else:
            form_layout.addRow("Notes:", self.run_notes)
            form_layout.addRow("<b>Run Information</b>", None)
            form_layout.addRow("Name:", self.run_name)
            form_layout.addRow("Date / Time:", self.run_date_time)
            form_layout.addRow("Duration:", self.run_duration)
            form_layout.addRow("Batch Number:", self.run_batch)
            form_layout.addRow("Fill Type:", self.run_fill_type)
            form_layout.addRow("Captured:", self.run_captured)
            form_layout.addRow("Updated:", self.run_updated)
            form_layout.addRow("Analyzed:", self.run_analyzed)

        # Action buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        if step == 1:
            self.btn_next = QtWidgets.QPushButton(
                "Next Step: Suggest Experiments")
        elif step == 2:
            self.btn_next = QtWidgets.QPushButton(
                "Next Step: Import Experiments")
        elif step == 3:
            self.btn_next = QtWidgets.QPushButton(
                "Next Step: Learn + Predict")
        elif step == 5:
            self.btn_next = QtWidgets.QPushButton("Finish")
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_next)
        left_layout.addLayout(btn_layout)

        # Right panel: Initialize features
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        v_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_layout.addWidget(v_splitter)
        right_header = QtWidgets.QGroupBox("Initialize Features")
        right_group = QtWidgets.QVBoxLayout(right_header)
        v_splitter.addWidget(right_header)

        # Features table
        self.default_features = {"Feature": ["Protein Type", "Protein Concentration",
                                             "Buffer Type",
                                             "Surfactant Type", "Surfactant Concentration",
                                             "Stabilizer Type", "Stabilizer Concentration"],
                                 "Value": [["Sample A", "Sample B", "Sample C"], "",
                                           ["Acetate", "Histidine", "PBS"],
                                           ["TWEEN20", "TWEEN80"], "",
                                           ["Sucrose", "Trehalose"], ""],
                                 "Units": ["", "mg/mL",
                                           "",
                                           "", "%w",
                                           "", "M"]}
        self.default_rows, self.default_cols = \
            (len(list(self.default_features.values())[0]),
             len(list(self.default_features.keys())))

        self.feature_table = TableView(self.default_features,
                                       self.default_rows, self.default_cols)
        self.feature_table.clear()
        right_group.addWidget(self.feature_table)

        # TODO: Testing only, create dummy features
        from random import randint
        import copy
        self.dummy_features = []
        for i in range(4):
            dummy_feature = copy.deepcopy(self.default_features)
            value_tags = [0, range(5, 95),
                          range(3),
                          range(2), range(5, 95),
                          range(2), range(5, 95)]
            for x in range(len(dummy_feature["Value"])):
                try:
                    current_value = dummy_feature["Value"][x]
                    current_tag = value_tags[x]
                    if isinstance(current_value, list):
                        if isinstance(current_tag, int):
                            dummy_feature["Value"][x] = [
                                current_value[current_tag]]
                        else:
                            dummy_feature["Value"][x] = [current_value[randint(
                                current_tag[0], current_tag[-1])]]
                    else:
                        if isinstance(current_tag, range):
                            dummy_feature["Value"][x] = randint(
                                current_tag[0], current_tag[-1])
                except Exception as e:
                    print(e)
            self.dummy_features.append(dummy_feature)

        self.run_figure = Figure()
        self.run_canvas = FigureCanvas(self.run_figure)
        v_splitter.addWidget(self.run_canvas)

        # Build main layout
        h_splitter.addWidget(left_widget)
        h_splitter.addWidget(right_widget)
        h_splitter.setSizes([100, 300])
        v_splitter.setSizes([100, 100])

        # Signals
        self.btn_cancel.clicked.connect(
            lambda: self.file_selected(None))
        self.btn_next.clicked.connect(self.next_step)
        self.select_candidate.clicked.connect(self.file_dialog.show)
        self.file_dialog.fileSelected.connect(self.file_selected)

    def next_step(self):
        # Are we ready to proceed?
        # Yes, if and only if:
        #   1. All audits contain valid values
        #   2. All initial features are set
        if (len(self.run_captured.text()) and
            len(self.run_updated.text()) and
            len(self.run_analyzed.text()) and
                self.feature_table.allSet()):
            # ready to proceed
            if self.parent is not None:
                i = self.parent.tab_widget.currentIndex()
                self.parent.tab_widget.setCurrentIndex(i+1)
            else:
                self.run_notes.setText(
                    "ERROR: self.parent is None.\n" +
                    "Cannot proceed to next step!")
        else:  # not ready
            QtWidgets.QMessageBox.information(
                None, "Missing Information", "Please correct the highlighted fields first.", QtWidgets.QMessageBox.Ok)

    def file_selected(self, path: str | None):
        self.candidate_file_run = path
        self.candidate_file_xml = None
        self.candidate_file_analyze = None

        # clear all fields, before repopulating them
        self.select_label.clear()
        self.run_notes.clear()
        self.run_name.clear()
        self.run_date_time.clear()
        self.run_duration.clear()
        self.run_batch.clear()
        self.run_fill_type.clear()
        self.run_captured.clear()
        self.run_updated.clear()
        self.run_analyzed.clear()
        self.run_figure.clear()
        self.run_canvas.draw()
        self.feature_table.clear()

        if path is None:
            if self.step == 3:
                self.list_view.clearSelection()
            return

        self.select_label.setText(
            os.path.basename(os.path.dirname(self.candidate_file_run)))

        if self.step == 3:
            item = QtGui.QStandardItem(self.select_label.text())
            found = self.model.findItems(item.text())
            if len(found) == 0:
                self.model.appendRow(item)
                new_index = self.model.indexFromItem(item)
                self.list_view.setCurrentIndex(new_index)
                self.all_files[item.text()] = path

        folder = os.path.dirname(self.candidate_file_run)
        files: list[str] = os.listdir(folder)
        for f in files:
            if f.endswith(".xml"):
                self.candidate_file_xml = os.path.join(folder, f)
            if f.startswith("analyze") and f.endswith(".zip"):
                self.candidate_file_analyze = os.path.join(folder, f)
        if self.candidate_file_xml == None:
            self.run_notes.setTextBackgroundColor(Color.light_red)
            self.run_notes.setText("ERROR: Cannot find XML file for this run!")
            return
        if self.candidate_file_analyze == None:
            self.run_notes.setTextBackgroundColor(Color.light_yellow)
            self.run_notes.setText("This run has not been analyzed yet.\n" +
                                   "Please Analyze and try again!")
            return

        doc = minidom.parse(self.candidate_file_xml)

        xml_metrics = {}
        metrics = doc.getElementsByTagName(
            "metrics")[-1]  # most recent element
        for m in metrics.childNodes:
            if m.nodeType == m.TEXT_NODE:
                continue  # only process elements
            name = m.getAttribute("name")
            value = m.getAttribute("value")
            if m.hasAttribute("units"):
                value = f"{value} {m.getAttribute('units')}"
            xml_metrics[name] = value

        xml_audits = {}
        audits = doc.getElementsByTagName(
            "audits")[-1]  # most recent element
        for a in audits.childNodes:
            if a.nodeType == a.TEXT_NODE:
                continue  # only process elements
            key = a.getAttribute("action")
            captured_by = a.getAttribute("username")
            captured_at = a.getAttribute("recorded")
            value = (captured_by, captured_at)
            xml_audits[key] = value

        xml_params = {}
        params = doc.getElementsByTagName(
            "params")[-1]  # most recent element
        for p in params.childNodes:
            if p.nodeType == p.TEXT_NODE:
                continue  # only process elements
            name = p.getAttribute("name")
            value = p.getAttribute("value")
            if p.hasAttribute("found"):
                value = f"{value} ({'Valid' if eval(p.getAttribute('found')) else 'Unknown'})"
            xml_params[name] = value

        try:
            self.run_notes.setTextBackgroundColor(Color.white)
            self.run_notes.setPlainText(
                xml_params["notes"].replace("\\n", "\n"))
        except:
            self.run_notes.setPlainText(None)
        try:
            self.run_name.setText(xml_params["run_name"])
        except:
            self.run_name.setText(self.select_label.text())
        try:
            self.run_date_time.setText(xml_metrics["start"].replace("T", " "))
        except:
            self.run_date_time.setText("(Unknown)")
        try:
            self.run_duration.setText(xml_metrics["duration"])
        except:
            self.run_duration.setText("(Unknown)")
        try:
            self.run_batch.setText(xml_params["batch_number"])
        except:
            self.run_batch.setText("(Not Provided)")
        try:
            self.run_fill_type.setText(xml_params["fill_type"])
        except:
            self.run_fill_type.setText("3")
        try:
            audit: tuple[str, str] = xml_audits['CAPTURE']
            captured_by, captured_at = audit
            captured_at = captured_at.replace(
                "T", " ")[:captured_at.index(".")]
            self.run_captured.setText(f"{captured_by} at {captured_at}")
        except:
            self.run_captured.setText("(Not Performed)")
        try:
            audit: tuple[str, str] = xml_audits['PARAMS']
            captured_by, captured_at = audit
            captured_at = captured_at.replace(
                "T", " ")[:captured_at.index(".")]
            self.run_updated.setText(f"{captured_by} at {captured_at}")
        except:
            # if no PARAMS in records, then last updated is time of CAPTURE:
            self.run_updated.setText(self.run_captured.text())
        try:
            audit: tuple[str, str] = xml_audits['ANALYZE']
            captured_by, captured_at = audit
            captured_at = captured_at.replace(
                "T", " ")[:captured_at.index(".")]
            self.run_analyzed.setText(f"{captured_by} at {captured_at}")
        except:
            self.run_analyzed.setText("(Not Performed)")

        def is_number(s: str):
            try:
                float(s)
                return True
            except ValueError:
                return False

        run_features = self.default_features.copy()
        value_tags = ["protein_type", "protein_concentration",
                      "buffer_type",
                      "surfactant_type", "surfactant_concentration",
                      "stabilizer_type", "stabilizer_concentration"]
        for x, y in enumerate(value_tags):
            try:
                if y in xml_params.keys():
                    if not is_number(xml_params[y]):
                        run_features["Value"][x] = [xml_params[y]]
                    else:
                        run_features["Value"][x] = xml_params[y]
            except Exception as e:
                print(e)
        self.feature_table.setData(run_features)

        self.run_figure.clear()
        ax = self.run_figure.add_subplot(111)
        ax.set_xlabel("Shear rate")
        ax.set_ylabel("Viscosity")
        ax.legend()
        self.run_canvas.draw()


class FrameStep2(QtWidgets.QDialog):
    def __init__(self, parent=None, step=2):
        super().__init__(parent)
        self.setWindowTitle("Learn")

        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        main_layout.addWidget(splitter)

        progress_widget = QtWidgets.QWidget()
        progress_layout = QtWidgets.QVBoxLayout(progress_widget)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_buttons = QtWidgets.QWidget()
        self.progress_btn_layout = QtWidgets.QHBoxLayout(self.progress_buttons)
        self.progress_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_resume = QtWidgets.QPushButton("Resume")
        self.btn_resume.clicked.connect(
            lambda: self.progress_bar.setValue(self.progress_bar.value()+1))
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(
            lambda: self.progress_bar.setValue(0))
        self.progress_btn_layout.addWidget(self.btn_pause)
        self.progress_btn_layout.addWidget(self.btn_resume)
        self.progress_btn_layout.addWidget(self.btn_cancel)
        self.progress_label = QtWidgets.QLabel()
        self.progress_bar.valueChanged.connect(
            lambda v: self.progress_label.setText(f"{v}% - Learning"))
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(50)
        self.progress_bar.setTextVisible(False)
        self.progress_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_buttons)
        progress_layout.addWidget(self.progress_label)
        splitter.addWidget(progress_widget)

        figure = QtWidgets.QLabel("[Figure here]")
        figure.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        splitter.addWidget(figure)
        splitter.setSizes([1, 1000])


class Color:
    black = QtGui.QColor(0, 0, 0)
    light_red = QtGui.QColor(255, 127, 127)
    light_yellow = QtGui.QColor(255, 255, 127)
    white = QtGui.QColor(255, 255, 255)


class TableView(QtWidgets.QTableWidget):

    def __init__(self, data, *args):
        QtWidgets.QTableWidget.__init__(self, *args)
        self.itemChanged.connect(self._on_item_changed)
        self.setData(data)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    def setData(self, data: dict[str, str]) -> None:
        self.data = data
        self.clear()
        horHeaders = []
        for n, key in enumerate(self.data.keys()):
            horHeaders.append(key)
            for m, item in enumerate(self.data[key]):
                if isinstance(item, list):
                    newitem = QtWidgets.QComboBox()
                    newitem.addItems(item)
                    newitem.addItem("add new...")
                    if len(item) > 1:
                        self.data["Units"][m] = "\u2190"
                        newitem.currentIndexChanged.connect(
                            lambda idx, row=m: self._row_combo_set(row))
                    else:
                        self.data["Units"][m] = ""  # clear flag
                else:
                    newitem = QtWidgets.QTableWidgetItem(str(item))
                # disable 1st and last column items (non-editable)
                if n == 0 or n == 2:
                    if n == 0:  # bold 1st column items (headers)
                        font = newitem.font()
                        font.setBold(True)
                        newitem.setFont(font)
                    newitem.setFlags(newitem.flags() &
                                     ~QtCore.Qt.ItemFlag.ItemIsEditable)
                if isinstance(newitem, QtWidgets.QWidget):
                    self.setCellWidget(m, n, newitem)
                else:
                    self.setItem(m, n, newitem)
        self.setHorizontalHeaderLabels(horHeaders)
        header = self.horizontalHeader()
        header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)

    def allSet(self) -> bool:
        for n, key in enumerate(self.data.keys()):
            for m, _ in enumerate(self.data[key]):
                item = self.item(m, n)
                if item is None:
                    continue  # QComboBox will return a None item
                if item.background().color().name() in [Color.light_yellow.name(), Color.light_red.name()]:
                    return False
        return True

    def _row_combo_set(self, idx):
        item = self.item(idx, 2)
        if item is not None:
            self.blockSignals(True)  # prevent recursion
            item.setBackground(QtGui.QBrush(Color.white))
            self.item(idx, 2).setText("")
            self.blockSignals(False)

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem):
        row, col, text = item.row(), item.column(), item.text()
        print(f"Cell ({row}, {col}) changed to: {text}")

        if col == 2 and text == "\u2190":
            item.setBackground(QtGui.QBrush(Color.light_yellow))

        if not (item.flags() & QtCore.Qt.ItemFlag.ItemIsEditable):
            # print("skip, disabled")
            return

        def is_number(s: str):
            try:
                float(s)
                return True
            except ValueError:
                return False

        now_bg = item.background()
        now_fg = item.foreground()
        new_bg = QtGui.QBrush(now_bg.color())
        new_fg = QtGui.QBrush(now_fg.color())

        if len(text) == 0:
            new_bg.setColor(Color.light_yellow)
            new_fg.setColor(Color.black)
        elif not is_number(text):
            new_bg.setColor(Color.light_red)
            new_fg.setColor(Color.light_yellow)
        else:
            new_bg.setColor(Color.white)
            new_fg.setColor(Color.black)

        self.blockSignals(True)  # prevent recursion
        if new_bg.color().name() != now_bg.color().name():
            item.setBackground(new_bg)
        if new_fg.color().name() != now_fg.color().name():
            item.setForeground(new_fg)
        self.blockSignals(False)

        self.clearSelection()  # unselect on item change


if __name__ == '__main__':
    _app = QtWidgets.QApplication([])
    _win = MainWindow()
    _win.show()
    _app.exec()
    _app.exit()
