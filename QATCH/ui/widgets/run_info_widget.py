import os
from PyQt5 import QtCore, QtGui, QtWidgets
from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants
from QATCH.ui.popUp import PopUp
from QATCH.ui.widgets.query_run_info_widget import QueryRunInfoWidget

TAG = "[RunInfoWidget]"


class RunInfoWindow(QtWidgets.QWidget):
    finished = QtCore.pyqtSignal()

    @staticmethod
    def test():
        num_runs_to_test = 4

        self = QtCore.QObject()
        # self.indicate_done = lambda: None
        self.bThread = []
        self.bWorker = []
        self.RunInfoWindow = None
        for i in range(num_runs_to_test):
            ask4info = True
            subDir = f"dummy_dir_{i+1}"
            new_path = f"C:\\test\\dummy_path_{i+1}.csv"
            is_good = True

            if ask4info:
                ask4info = False
                # self.indicate_finalizing()
                self.bThread.append(QtCore.QThread())
                # None if self.parent == None else self.parent.ControlsWin.username.text()[6:]
                user_name = "Alexander J. Ross"
                # TODO: more secure to pass user_hash (filename)
                self.bWorker.append(
                    QueryRunInfoWidget(subDir, new_path, is_good, user_name, parent=self.parent)
                )
                self.bThread[-1].started.connect(self.bWorker[-1].show)
                self.bWorker[-1].finished.connect(self.bThread[-1].quit)
                # self.bWorker[-1].finished.connect(self.indicate_done) # add here
                # self.finished.disconnect(self.indicate_done) # remove here

        num_runs_saved = len(self.bThread)
        for i in range(num_runs_saved):
            # if '1' more fields shown in QueryRunInfo
            self.bWorker[i].setRuns(num_runs_saved, i)
        if num_runs_saved == 0:
            pass  # do nothing
        elif num_runs_saved == 1:
            self.bThread[-1].start()  # only 1 run to save
        else:
            self.RunInfoWindow = RunInfoWindow(
                self.bWorker, self.bThread
            )  # more than 1 run to save

        return (self.bThread, self.bWorker, self.RunInfoWindow)

    def __init__(self, bWorkers, bThreads):
        super(RunInfoWindow, self).__init__(None)
        self.bWorker = bWorkers
        self.bThread = bThreads
        self.num_runs_saved = len(self.bThread)
        self._portIDfromIndex = lambda pid: hex(pid)[2:].upper()

        run_name, run_path, recall_from, run_ruling, user_name = self.bWorker[0].getRunParams()
        # run name root, without port # at end
        self.run_name = run_name[0 : run_name.rindex("_")]
        self.run_path = run_path
        self.xml_path = run_path[0:-4] + ".xml"
        self.recall_xml = recall_from
        self.run_ruling = "good" if run_ruling else "bad"
        self.username = user_name
        self.post_run = self.recall_xml != self.xml_path
        self.unsaved_changes = self.post_run  # force save on post-run
        self.batch_found = False
        # self.batch_warned = False

        self.RunInfoLayout = QtWidgets.QGridLayout()
        self.setLayout(self.RunInfoLayout)
        self.DockingWidgets = []
        for i in range(self.num_runs_saved):
            if self.num_runs_saved == 4:
                row = 2  # int(i / 2)
                col = i  # int(i % 2)
            else:  # default, fallback grid layout
                row = int(i % 4) + 2
                col = int(i / 4)
            self.DockingWidgets.append(
                QtWidgets.QDockWidget(f"Enter Run Info (Port {self._portIDfromIndex(i+1)})", self)
            )
            self.DockingWidgets[-1].setFeatures(
                QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures
            )
            self.DockingWidgets[-1].setWidget(self.bWorker[i])
            self.RunInfoLayout.addWidget(self.DockingWidgets[-1], row, col)
            self.bThread[i].start()
            self.RunInfoLayout.setRowMinimumHeight(row, self.DockingWidgets[-1].height())
            self.RunInfoLayout.setColumnMinimumWidth(col, self.DockingWidgets[-1].width())

        self.q_runpath = QtWidgets.QVBoxLayout()  # location #
        self.q_runbar = QtWidgets.QHBoxLayout()
        self.l_runpath = QtWidgets.QLabel()
        self.l_runpath.setText("Saved Run To\t=")
        self.q_runbar.addWidget(self.l_runpath)
        self.q_runbar.addStretch()
        self.cpy_runpath = QtWidgets.QLabel("Copied!")
        self.cpy_runpath.setVisible(False)
        self.q_runbar.addWidget(self.cpy_runpath, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.cb_runpath = QtWidgets.QLabel("&#x1F4CB;")  # clipboard icon
        self.cb_runpath.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.cb_runpath.setToolTip("Copy path to clipboard")
        self.cb_runpath.setContentsMargins(0, 0, 10, 0)
        self.cb_runpath.mousePressEvent = self.copyText
        self.q_runbar.addWidget(self.cb_runpath, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.q_runpath.addLayout(self.q_runbar)
        self.t_runpath = QtWidgets.QPlainTextEdit()
        self.t_runpath.setStyleSheet("background-color: #eee;")
        # self.t_runpath.setFixedHeight(50)
        self.t_runpath.setReadOnly(True)
        self.q_runpath.addWidget(self.t_runpath)
        self.RunInfoLayout.addLayout(self.q_runpath, 0, 1, 2, 1)
        # self.RunInfoLayout.setRowStretch(0, 0)
        # self.RunInfoLayout.setRowStretch(1, 0)

        self.q_common = QtWidgets.QVBoxLayout()
        self.l_common = QtWidgets.QLabel("Enter Run Info (All Ports)")
        self.l_common.setStyleSheet(
            "background-color: #ddd; border: 1 solid #bbb; font-size: 15px; padding: 1px;"
        )
        self.q_common.addWidget(self.l_common)
        self.q_common.addSpacing(10)

        self.q_runname = QtWidgets.QHBoxLayout()  # runname #
        self.q_runname.setContentsMargins(10, 0, 10, 0)
        self.l_runname = QtWidgets.QLabel()
        self.l_runname.setText("Run Name\t=")
        self.q_runname.addWidget(self.l_runname)
        self.t_runname = QtWidgets.QLineEdit()
        self.t_runname.textChanged.connect(self.detect_change)
        self.t_runname.editingFinished.connect(self.update_hidden_child_fields)
        self.q_runname.addWidget(self.t_runname)
        self.h_runname = QtWidgets.QLabel()
        self.h_runname.setText("<u>?</u>")
        self.h_runname.setToolTip("<b>Hint:</b> This name applies to all ports captured this run.")
        self.q_runname.addWidget(self.h_runname)
        self.q_common.addLayout(self.q_runname)

        self.q_batch = QtWidgets.QHBoxLayout()  # batch #
        self.q_batch.setContentsMargins(10, 0, 10, 0)
        self.l_batch = QtWidgets.QLabel()
        self.l_batch.setText("Batch Number\t=")
        self.q_batch.addWidget(self.l_batch)
        self.t_batch = QtWidgets.QLineEdit()
        self.t_batch.textChanged.connect(self.detect_change)
        self.t_batch.textEdited.connect(self.prevent_duplicate_scans)
        self.t_batch.editingFinished.connect(self.update_hidden_child_fields)
        self.q_batch.addWidget(self.t_batch)
        self.h_batch = QtWidgets.QLabel()
        self.h_batch.setText("<u>?</u>")
        self.h_batch.setToolTip("<b>Hint:</b> Find this # on the crystal's packaging.")
        self.q_batch.addWidget(self.h_batch)
        self.q_common.addLayout(self.q_batch)

        self.blankIcon = QtGui.QIcon()
        self.foundIcon = QtGui.QIcon(
            os.path.join(Architecture.get_path(), "QATCH", "icons", "checkmark.png")
        )
        self.missingIcon = QtGui.QIcon(
            os.path.join(Architecture.get_path(), "QATCH", "icons", "warning.png")
        )
        self.t_batchAction = self.t_batch.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.t_batchAction.triggered.connect(self.find_batch_num)
        self.t_batch.textChanged.connect(self.find_batch_num)
        self.t_batch.editingFinished.connect(self.find_batch_num)

        self.q_common.addStretch()
        self.RunInfoLayout.addLayout(self.q_common, 0, 0, 2, 1)

        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setPlaceholderText("Notes")
        self.notes.setTabChangesFocus(True)
        self.notes.textChanged.connect(self.detect_change)
        self.RunInfoLayout.addWidget(self.notes, 0, 2, 2, col - 1)

        self.q_recall = QtWidgets.QCheckBox("Remember for next run")
        self.q_recall.setChecked(True)
        self.q_recall.setEnabled(self.unsaved_changes)
        self.q_recall.stateChanged.connect(self.detect_change)
        self.q_recall.stateChanged.connect(self.update_hidden_child_fields)
        self.RunInfoLayout.addWidget(
            self.q_recall, row + 1, 0, 1, col + 1, QtCore.Qt.AlignmentFlag.AlignCenter
        )

        self.btn = QtWidgets.QPushButton("Save")
        self.btn.pressed.connect(self.confirm)
        self.RunInfoLayout.addWidget(
            self.btn, row + 2, 0, 1, col + 1, QtCore.Qt.AlignmentFlag.AlignCenter
        )

        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/info.png")
        self.setWindowIcon(QtGui.QIcon(icon_path))  # .png
        self.setWindowTitle("Enter Run Info (Multiple Ports)")
        self.show()
        self.raise_()
        self.activateWindow()
        # set min sizes for docking widgets to prevent layout changes on 'save'
        for dw in self.DockingWidgets:
            dw.setMinimumSize(dw.width(), dw.height())
        # center window in desktop geometry
        # NOTE: This position applies only to multiplex run info window
        width = self.width()
        height = self.height()
        area = QtWidgets.QDesktopWidget().availableGeometry()
        left = int((area.width() - width) / 2)
        top = int((area.height() - height) / 2)
        self.move(left, top)

        self.btn.setFixedWidth(int(width / (col + 1)))

        if self.num_runs_saved <= 4:
            self.showMaximized()
            QtCore.QTimer.singleShot(250, self.resizeNormal)
        else:
            QtCore.QTimer.singleShot(250, self.showMaximized)

        # find common run path, from all run paths
        self.all_run_paths = []
        for i in range(self.num_runs_saved):
            _, run_path, _, _, _ = self.bWorker[i].getRunParams()
            full_run_path = os.path.join(os.getcwd(), run_path)
            self.all_run_paths.append(full_run_path)
        self.common_run_path = os.path.commonpath(self.all_run_paths)
        self.t_runpath.setPlainText(self.common_run_path)
        self.t_runname.setText(self.run_name)
        self.t_batch.setFocus()

        ###### scannow widget for batch number ######
        # note: this must be after self.setLayout() #
        self.l_scannow = QtWidgets.QWidget(self)  #
        self.l_scannow.setVisible(False)  #
        #############################################

        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter), self, activated=self.confirm)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self, activated=self.confirm)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self, activated=self.close)

        QtCore.QTimer.singleShot(500, self.showScanNow)
        QtCore.QTimer.singleShot(1000, self.flashScanNow)

    def showScanNow(self):
        self.l_scannow.resize(self.t_batch.size())
        self.l_scannow.move(self.t_batch.pos())
        self.l_scannow.setObjectName("scannow")
        self.l_scannow.setStyleSheet(
            "#scannow { background-color: #F5FE49; border: 1px solid #7A7A7A; }"
        )
        self.h_scannow = QtWidgets.QHBoxLayout()
        self.h_scannow.setContentsMargins(3, 0, 6, 0)
        self.t_scannow = QtWidgets.QLabel("Scan or enter now!")
        self.h_scannow.addWidget(self.t_scannow)
        self.h_scannow.addStretch()
        self.i_scannow = QtWidgets.QLabel()
        self.i_scannow.setPixmap(
            QtGui.QPixmap(
                os.path.join(Architecture.get_path(), "QATCH", "icons", "scan.png")
            ).scaledToHeight(self.l_scannow.height() - 2)
        )
        self.h_scannow.addWidget(self.i_scannow)
        self.l_scannow.setLayout(self.h_scannow)
        self.l_scannow.setVisible(True)
        self.t_batch.textEdited.connect(self.l_scannow.hide)

    def flashScanNow(self):
        if self.l_scannow.isVisible():
            if not self.t_batch.hasFocus():
                self.l_scannow.hide()
            elif self.t_scannow.styleSheet() == "":
                self.t_scannow.setStyleSheet("color: #F5FE49;")
                QtCore.QTimer.singleShot(250, self.flashScanNow)
            else:
                self.t_scannow.setStyleSheet("")
                QtCore.QTimer.singleShot(500, self.flashScanNow)

    def resizeNormal(self):
        self.showNormal()
        self.resize(100, 100)  # to minimums

    def copyText(self, obj):
        no_error = True
        try:
            cb = QtWidgets.QApplication.clipboard()
            cb.clear(mode=cb.Clipboard)
            cb.setText(self.t_runpath.toPlainText(), mode=cb.Clipboard)
        except Exception as e:
            Log.e("Clipboard Error:", e)
            no_error = False
        if no_error:
            self.cpy_runpath.show()
            QtCore.QTimer.singleShot(3000, self.cpy_runpath.hide)

    def prevent_duplicate_scans(self):
        current_text = self.t_batch.text()
        min_batch_num_len = 3  # characters, i.e. 'MMx'
        given_batch_num_len = len(current_text)
        split_at_idx = int(given_batch_num_len / 2)
        if split_at_idx > min_batch_num_len:
            if given_batch_num_len % 2 == 0:  # even only
                first_half = current_text[:split_at_idx]
                second_half = current_text[split_at_idx:]
                Log.d(f"prevent_duplicate_scans(): Checking '{first_half}|{second_half}'")
                if first_half == second_half:
                    Log.w(f"Duplicate scan ignored: {second_half}")
                    self.t_batch.setText(first_half)

    def find_batch_num(self):
        batch = self.t_batch.text().strip()
        found = False
        if len(batch) == 0:
            # hide icon to indicate no batch is provided
            self.t_batchAction.setIcon(self.blankIcon)
        elif Constants.get_batch_param(self.t_batch.text()):
            # set check mark to indicate batch is known/found
            self.t_batchAction.setIcon(self.foundIcon)
            found = True
        else:
            # set question mark to indicate batch is not known/found
            self.t_batchAction.setIcon(self.missingIcon)
        self.t_batch.setStyleSheet(
            "border: 1px solid black;" if not found else "background-color: #eee;"
        )
        # detect when AUDIT says 'found = false' but now it is found
        if self.batch_found != found:
            self.batch_found = found
            self.detect_change()

    def update_hidden_child_fields(self):
        run_name = self.t_runname.text()
        batch_num = self.t_batch.text()
        notes_txt = self.notes.toPlainText()
        do_recall = self.q_recall.isChecked()
        for i in range(self.num_runs_saved):
            self.bWorker[i].setHiddenFields(run_name, batch_num, notes_txt, do_recall)

    def detect_change(self):
        self.unsaved_changes = True

    def confirm(self, force=False):
        # save each child widget, which already *should* have updated hidden fields
        for i in range(self.num_runs_saved):
            if not self.bWorker[i].isVisible():
                Log.d(f"Skipping RUN_IDX = {i} (already saved)")
                self.mark_child_as_saved(i)
                continue  # skip to next child
            if i == 0:  # force propogation before saving in case edited field still has focus
                self.update_hidden_child_fields()
            Log.d(f"Saving RUN_IDX = {i}")
            if not self.bWorker[i].confirm(force):
                try:
                    self.bWorker[i].finished.disconnect(self.confirm)
                    Log.w(
                        "Save is waiting on additional user input (i.e. signature, missing fields, etc.)"
                    )
                except TypeError:
                    pass
                self.bWorker[i].finished.connect(self.confirm)
                Log.d(f"Save paused at RUN_IDX {i}/{self.num_runs_saved}")
                return False  # abort if not confirmed
            self.mark_child_as_saved(i)
        self.unsaved_changes = False
        self.close()
        return True

    def mark_child_as_saved(self, i):
        self.DockingWidgets[i].setWidget(QtWidgets.QLabel("Saved!"))
        self.DockingWidgets[i].widget().setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def closeEvent(self, event):
        # check for undetected changes in children widgets
        for i in range(self.num_runs_saved):
            self.unsaved_changes |= self.bWorker[i].unsaved_changes
        if self.unsaved_changes:
            res = PopUp.question(
                self,
                Constants.app_title,
                "You have unsaved changes!\n\nAre you sure you want to close this window?",
                False,
            )
            if res:
                if self.post_run:
                    try:
                        self.confirm(force=True)
                    except Exception as e:
                        Log.e(e)
                # pre-hide so callers connected to 'finished' signal can see the "hidden" widget
                self.setVisible(False)
                self.finished.emit()
            else:
                event.ignore()
        else:  # closing, with no changes
            # pre-hide so callers connected to 'finished' signal can see the "hidden" widget
            self.setVisible(False)
            self.finished.emit()
