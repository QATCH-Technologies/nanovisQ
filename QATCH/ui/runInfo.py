from QATCH.common.architecture import Architecture
from QATCH.common.userProfiles import UserProfiles
from QATCH.common.fileStorage import FileStorage, secure_open
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants
from QATCH.ui.popUp import PopUp
from QATCH.VisQAI.src.db.sqlite_db import SQLiteDB
from PyQt5 import QtCore, QtGui, QtWidgets
from xml.dom import minidom
import numpy as np
import os
import datetime as dt
import hashlib
# import re
import datetime

TAG = "[RunInfo]"


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
                self.bWorker.append(QueryRunInfo(
                    subDir, new_path, is_good, user_name, parent=self.parent))
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
                self.bWorker, self.bThread)  # more than 1 run to save

        return (self.bThread, self.bWorker, self.RunInfoWindow)

    def __init__(self, bWorkers, bThreads):
        super(RunInfoWindow, self).__init__(None)
        self.bWorker = bWorkers
        self.bThread = bThreads
        self.num_runs_saved = len(self.bThread)
        self._portIDfromIndex = lambda pid: hex(pid)[2:].upper()

        run_name, run_path, recall_from, run_ruling, user_name = self.bWorker[0].getRunParams(
        )
        # run name root, without port # at end
        self.run_name = run_name[0:run_name.rindex('_')]
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
            self.DockingWidgets.append(QtWidgets.QDockWidget(
                f"Enter Run Info (Port {self._portIDfromIndex(i+1)})", self))
            self.DockingWidgets[-1].setFeatures(
                QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
            self.DockingWidgets[-1].setWidget(self.bWorker[i])
            self.RunInfoLayout.addWidget(self.DockingWidgets[-1], row, col)
            self.bThread[i].start()
            self.RunInfoLayout.setRowMinimumHeight(
                row, self.DockingWidgets[-1].height())
            self.RunInfoLayout.setColumnMinimumWidth(
                col, self.DockingWidgets[-1].width())

        self.q_runpath = QtWidgets.QVBoxLayout()  # location #
        self.q_runbar = QtWidgets.QHBoxLayout()
        self.l_runpath = QtWidgets.QLabel()
        self.l_runpath.setText("Saved Run To\t=")
        self.q_runbar.addWidget(self.l_runpath)
        self.q_runbar.addStretch()
        self.cpy_runpath = QtWidgets.QLabel("Copied!")
        self.cpy_runpath.setVisible(False)
        self.q_runbar.addWidget(
            self.cpy_runpath, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.cb_runpath = QtWidgets.QLabel("&#x1F4CB;")  # clipboard icon
        self.cb_runpath.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.cb_runpath.setToolTip("Copy path to clipboard")
        self.cb_runpath.setContentsMargins(0, 0, 10, 0)
        self.cb_runpath.mousePressEvent = self.copyText
        self.q_runbar.addWidget(
            self.cb_runpath, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
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
            "background-color: #ddd; border: 1 solid #bbb; font-size: 15px; padding: 1px;")
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
        self.h_runname.setToolTip(
            "<b>Hint:</b> This name applies to all ports captured this run.")
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
        self.h_batch.setToolTip(
            "<b>Hint:</b> Find this # on the crystal's packaging.")
        self.q_batch.addWidget(self.h_batch)
        self.q_common.addLayout(self.q_batch)

        self.blankIcon = QtGui.QIcon()
        self.foundIcon = QtGui.QIcon(os.path.join(
            Architecture.get_path(), "QATCH", "icons", "checkmark.png"))
        self.missingIcon = QtGui.QIcon(os.path.join(
            Architecture.get_path(), "QATCH", "icons", "warning.png"))
        self.t_batchAction = self.t_batch.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition)
        self.t_batchAction.triggered.connect(self.find_batch_num)
        self.t_batch.textChanged.connect(self.find_batch_num)
        self.t_batch.editingFinished.connect(self.find_batch_num)

        self.q_common.addStretch()
        self.RunInfoLayout.addLayout(self.q_common, 0, 0, 2, 1)

        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setPlaceholderText("Notes")
        self.notes.setTabChangesFocus(True)
        self.notes.textChanged.connect(self.detect_change)
        self.RunInfoLayout.addWidget(self.notes, 0, 2, 2, col-1)

        self.q_recall = QtWidgets.QCheckBox("Remember for next run")
        self.q_recall.setChecked(True)
        self.q_recall.setEnabled(self.unsaved_changes)
        self.q_recall.stateChanged.connect(self.detect_change)
        self.q_recall.stateChanged.connect(self.update_hidden_child_fields)
        self.RunInfoLayout.addWidget(
            self.q_recall, row+1, 0, 1, col+1, QtCore.Qt.AlignmentFlag.AlignCenter)

        self.btn = QtWidgets.QPushButton("Save")
        self.btn.pressed.connect(self.confirm)
        self.RunInfoLayout.addWidget(
            self.btn, row+2, 0, 1, col+1, QtCore.Qt.AlignmentFlag.AlignCenter)

        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/info.png')
        self.setWindowIcon(QtGui.QIcon(icon_path))  # .png
        self.setWindowTitle("Enter Run Info (Multiple Ports)")
        self.show()
        self.raise_()
        self.activateWindow()
        # set min sizes for docking widgets to prevent layout changes on 'save'
        for dw in self.DockingWidgets:
            dw.setMinimumSize(dw.width(), dw.height())
        # center window in desktop geometry
        width = self.width()
        height = self.height()
        area = QtWidgets.QDesktopWidget().availableGeometry()
        left = int((area.width() - width) / 2)
        top = int((area.height() - height) / 2)
        self.move(left, top)

        self.btn.setFixedWidth(int(width / (col+1)))

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
        self.l_scannow = QtWidgets.QWidget(self)    #
        self.l_scannow.setVisible(False)            #
        #############################################

        QtWidgets.QShortcut(QtGui.QKeySequence(
            QtCore.Qt.Key_Enter), self, activated=self.confirm)
        QtWidgets.QShortcut(QtGui.QKeySequence(
            QtCore.Qt.Key_Return), self, activated=self.confirm)
        QtWidgets.QShortcut(QtGui.QKeySequence(
            QtCore.Qt.Key_Escape), self, activated=self.close)

        QtCore.QTimer.singleShot(500, self.showScanNow)
        QtCore.QTimer.singleShot(1000, self.flashScanNow)

    def showScanNow(self):
        self.l_scannow.resize(self.t_batch.size())
        self.l_scannow.move(self.t_batch.pos())
        self.l_scannow.setObjectName("scannow")
        self.l_scannow.setStyleSheet(
            "#scannow { background-color: #F5FE49; border: 1px solid #7A7A7A; }")
        self.h_scannow = QtWidgets.QHBoxLayout()
        self.h_scannow.setContentsMargins(3, 0, 6, 0)
        self.t_scannow = QtWidgets.QLabel("Scan or enter now!")
        self.h_scannow.addWidget(self.t_scannow)
        self.h_scannow.addStretch()
        self.i_scannow = QtWidgets.QLabel()
        self.i_scannow.setPixmap(QtGui.QPixmap(os.path.join(Architecture.get_path(), "QATCH", "icons", "scan.png"))
                                 .scaledToHeight(self.l_scannow.height()-2))
        self.h_scannow.addWidget(self.i_scannow)
        self.l_scannow.setLayout(self.h_scannow)
        self.l_scannow.setVisible(True)
        self.t_batch.textEdited.connect(self.l_scannow.hide)

    def flashScanNow(self):
        if self.l_scannow.isVisible():
            if not self.t_batch.hasFocus():
                self.l_scannow.hide()
            elif self.t_scannow.styleSheet() == '':
                self.t_scannow.setStyleSheet("color: #F5FE49;")
                QtCore.QTimer.singleShot(250, self.flashScanNow)
            else:
                self.t_scannow.setStyleSheet('')
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
                Log.d(
                    f"prevent_duplicate_scans(): Checking '{first_half}|{second_half}'")
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
            "border: 1px solid black;" if not found else "background-color: #eee;")
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
            self.bWorker[i].setHiddenFields(
                run_name, batch_num, notes_txt, do_recall)

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
                        "Save is waiting on additional user input (i.e. signature, missing fields, etc.)")
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
        self.DockingWidgets[i].widget().setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter)

    def closeEvent(self, event):
        # check for undetected changes in children widgets
        for i in range(self.num_runs_saved):
            self.unsaved_changes |= self.bWorker[i].unsaved_changes
        if self.unsaved_changes:
            res = PopUp.question(
                self, Constants.app_title, "You have unsaved changes!\n\nAre you sure you want to close this window?", False)
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


class QueryRunInfo(QtWidgets.QWidget):
    finished = QtCore.pyqtSignal()
    updated_run = QtCore.pyqtSignal(str, str, str, str)
    updated_xml_path = QtCore.pyqtSignal(str)

    def __init__(self, run_name, run_path, run_ruling, user_name="NONE", recall_from=Constants.query_info_recall_path, parent=None):
        super(QueryRunInfo, self).__init__(None)
        self.parent = parent

        self.run_name_changed = False
        self.run_name = run_name
        self.run_path = run_path
        self.xml_path = run_path[0:-4] + ".xml"
        self.recall_xml = recall_from
        self.run_ruling = "good" if run_ruling else "bad"
        self.username = user_name
        self.post_run = self.recall_xml != self.xml_path
        self.unsaved_changes = self.post_run  # force save on post-run
        self.batch_found = False
        self.batch_warned = False
        self.run_count = 0
        self.run_idx = 0
        self.run_port = ""  # if multirun, will be something like "_1"
        self.auto_st = 0
        self.auto_ca = 0
        self.auto_dn = 0
        self.auto_nc = 0

        self.excipient_db = SQLiteDB()

        self.q_runname = QtWidgets.QHBoxLayout()  # runname #
        # self.q_runname.setContentsMargins(10, 0, 10, 0)
        self.l_runname = QtWidgets.QLabel()
        self.l_runname.setText("Run Name\t=")
        self.q_runname.addWidget(self.l_runname)
        self.t_runname = QtWidgets.QLineEdit()
        self.t_runname.setText(self.run_name)
        self.q_runname.addWidget(self.t_runname)
        self.h_runname = QtWidgets.QLabel()
        self.h_runname.setText("<u>?</u>")
        self.h_runname.setToolTip(
            "<b>Hint:</b> This name applies to all ports captured this run.")
        self.q_runname.addWidget(self.h_runname)

        self.q_batch = QtWidgets.QHBoxLayout()  # batch #
        self.l_batch = QtWidgets.QLabel()
        self.l_batch.setText("Batch Number\t=")
        self.q_batch.addWidget(self.l_batch)
        self.t_batch = QtWidgets.QLineEdit()
        self.t_batch.textEdited.connect(self.prevent_duplicate_scans)
        self.q_batch.addWidget(self.t_batch)
        self.h_batch = QtWidgets.QLabel()
        self.h_batch.setText("<u>?</u>")
        self.h_batch.setToolTip(
            "<b>Hint:</b> Find this # on the crystal's packaging.")
        self.q_batch.addWidget(self.h_batch)

        self.blankIcon = QtGui.QIcon()
        self.foundIcon = QtGui.QIcon(os.path.join(
            Architecture.get_path(), "QATCH", "icons", "checkmark.png"))
        self.missingIcon = QtGui.QIcon(os.path.join(
            Architecture.get_path(), "QATCH", "icons", "warning.png"))
        self.t_batchAction = self.t_batch.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition)
        self.t_batchAction.triggered.connect(self.find_batch_num)
        self.t_batch.textChanged.connect(self.find_batch_num)
        self.t_batch.editingFinished.connect(self.find_batch_num)

        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setPlaceholderText("Notes")
        self.notes.setTabChangesFocus(True)
        self.notes.setFixedHeight(100)

        self.q1 = QtWidgets.QHBoxLayout()
        self.l1 = QtWidgets.QLabel()
        self.l1.setText("Is this a bioformulation?\t")
        self.q1.addWidget(self.l1)
        self.g1 = QtWidgets.QButtonGroup()
        self.b1 = QtWidgets.QCheckBox("Yes")
        self.b2 = QtWidgets.QCheckBox("No")
        self.g1.addButton(self.b1, 1)
        self.g1.addButton(self.b2, 2)
        self.q1.addWidget(self.b1)
        self.q1.addWidget(self.b2)
        self.g1.buttonClicked.connect(self.show_hide_gui)

        self.q2 = QtWidgets.QHBoxLayout()
        self.l2 = QtWidgets.QLabel()
        self.l2.setText("Type\t\t=")  # solvent type
        self.q2.addWidget(self.l2)
        self.t0 = QtWidgets.QLineEdit()

        self.fluids = []
        self.surface_tensions = []
        self.densities = []
        try:
            # auto complete options
            # prefer working resource path, if exists
            working_resource_path = os.path.join(
                os.getcwd(), "QATCH/resources/")
            # bundled_resource_path = os.path.join(Architecture.get_path(), "QATCH/resources/") # otherwise, use bundled resource path
            # if os.path.exists(working_resource_path) else bundled_resource_path
            resource_path = working_resource_path
            data = np.genfromtxt(os.path.join(
                resource_path, "lookup_by_solvent.csv"), dtype='str', delimiter='\t', skip_header=1)
            fluids_with_commas = data[:, 0]
            surface_tensions = data[:, 1]
            densities = data[:, 2]
            for idx, name in enumerate(fluids_with_commas):
                st = float(surface_tensions[idx].strip())
                dn = float(densities[idx].strip())
                for n in name.split(','):
                    if len(n.strip()) > 0:
                        self.fluids.append(n.strip())
                        self.surface_tensions.append(st * 1000)  # show as mN/m
                        self.densities.append(dn)
            # Sort all imported arrays by fluid, alphabetically
            self.fluids = np.array(self.fluids)
            self.surface_tensions = np.array(self.surface_tensions)
            self.densities = np.array(self.densities)
            idxs = np.argsort(self.fluids)
            self.fluids = self.fluids[idxs].tolist()
            self.surface_tensions = self.surface_tensions[idxs].tolist()
            self.densities = self.densities[idxs].tolist()
            Log.d("SUCCESS: Loaded solvents list @ 'lookup_by_solvent.csv'")
            completer = QtWidgets.QCompleter(self)
            completer_model = QtCore.QStringListModel(self.fluids, completer)
            completer.setModel(completer_model)
            completer.setModelSorting(
                QtWidgets.QCompleter.CaseInsensitivelySortedModel)
            completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
            completer.setFilterMode(QtCore.Qt.MatchContains)
            completer.setMaxVisibleItems(3)
            self.t0.setCompleter(completer)
            self.t0.setClearButtonEnabled(True)
        except Exception as e:
            Log.e("ERROR:", e)
            Log.w("WARNING: Failed to load solvents list @ 'lookup_by_solvent.csv'")
            Log.w("You will need to enter your solvent run parameters manually.")

        self.q2.addWidget(self.t0)
        self.h0 = QtWidgets.QLabel()
        self.h0.setText("<u>?</u>")
        self.h0.setToolTip(
            "<b>Hint:</b> If not listed, enter parameters manually.")
        self.q2.addWidget(self.h0)
        self.t0.textChanged.connect(self.lookup_completer)
        self.t0.editingFinished.connect(self.enforce_completer)

        # Solvent Groupbox
        self.groupSolvent = QtWidgets.QGroupBox("Solvent Information")
        self.groupSolvent.setCheckable(False)
        self.vbox0 = QtWidgets.QVBoxLayout()
        self.groupSolvent.setLayout(self.vbox0)
        self.vbox0.addLayout(self.q2)

        self.q3 = QtWidgets.QHBoxLayout()
        self.l3 = QtWidgets.QLabel()
        self.l3.setText("Surfactant\t=")
        self.q3.addWidget(self.l3)
        self.t3 = QtWidgets.QLineEdit()
        self.validSurfactant = QtGui.QDoubleValidator(0, 1, 5)
        self.validSurfactant.setNotation(
            QtGui.QDoubleValidator.StandardNotation)
        self.t3.setValidator(self.validSurfactant)
        self.q3.addWidget(self.t3)
        self.h3 = QtWidgets.QLabel()
        self.h3.setText("<u>%w</u>")
        self.h3.setToolTip("<b>Hint:</b> For 0.010%w enter \"0.010\".")
        self.q3.addWidget(self.h3)
        self.t3.textChanged.connect(self.calc_params)
        self.t3.editingFinished.connect(self.calc_params)

        self.q4 = QtWidgets.QHBoxLayout()
        self.l4 = QtWidgets.QLabel()
        self.l4.setText("Concentration\t=")
        self.q4.addWidget(self.l4)
        self.t4 = QtWidgets.QLineEdit()
        self.validConcentration = QtGui.QDoubleValidator(0, 1000, 3)
        self.validConcentration.setNotation(
            QtGui.QDoubleValidator.StandardNotation)
        self.t4.setValidator(self.validConcentration)
        self.q4.addWidget(self.t4)
        self.h4 = QtWidgets.QLabel()
        self.h4.setText("<u>mg/mL</u>")
        self.h4.setToolTip("<b>Hint:</b> For 100mg/mL enter \"100\".")
        self.q4.addWidget(self.h4)
        self.t4.textChanged.connect(self.calc_params)
        self.t4.editingFinished.connect(self.calc_params)

        # Protein Type
        self.q10 = QtWidgets.QHBoxLayout()
        self.l10 = QtWidgets.QLabel()
        self.l10.setText("Type\t\t=")
        self.q10.addWidget(self.l10)
        self.c10 = QtWidgets.QComboBox()
        self.q10.addWidget(self.c10, 1)
        self.h10 = QtWidgets.QLabel()
        self.h10.setText("<u>?</u>")
        self.h10.setToolTip(
            "<b>Hint:</b> If not listed, add a new entry to the list.")
        self.q10.addWidget(self.h10)
        self.c10.currentTextChanged.connect(self.new_protein_type)

        # Protein Concentration
        self.q12 = QtWidgets.QHBoxLayout()
        self.l12 = QtWidgets.QLabel()
        self.l12.setText("Concentration\t=")
        self.q12.addWidget(self.l12)
        self.t12 = QtWidgets.QLineEdit()
        self.validProteinConcentration = QtGui.QDoubleValidator(0, 1000, 3)
        self.validProteinConcentration.setNotation(
            QtGui.QDoubleValidator.StandardNotation)
        self.t12.setValidator(self.validProteinConcentration)
        self.q12.addWidget(self.t12)
        self.h12 = QtWidgets.QLabel()
        self.h12.setText("<u>mg/mL</u>")
        self.h12.setToolTip("<b>Hint:</b> For 100mg/mL enter \"100\".")
        self.q12.addWidget(self.h12)
        self.t12.textChanged.connect(self.calc_params)
        self.t12.editingFinished.connect(self.calc_params)

        # Protein Groupbox
        self.groupProtein = QtWidgets.QGroupBox("Protein Information")
        self.groupProtein.setCheckable(False)
        self.vbox2 = QtWidgets.QVBoxLayout()
        self.groupProtein.setLayout(self.vbox2)
        self.vbox2.addLayout(self.q10)
        self.vbox2.addLayout(self.q12)

        # Buffer Type
        self.q13 = QtWidgets.QHBoxLayout()
        self.l13 = QtWidgets.QLabel()
        self.l13.setText("Type\t\t=")
        self.q13.addWidget(self.l13)
        self.c13 = QtWidgets.QComboBox()
        self.q13.addWidget(self.c13, 1)
        self.h13 = QtWidgets.QLabel()
        self.h13.setText("<u>?</u>")
        self.h13.setToolTip(
            "<b>Hint:</b> If not listed, add a new entry to the list.")
        self.q13.addWidget(self.h13)
        self.c13.currentTextChanged.connect(self.new_buffer_type)

        # Buffer Concentration
        self.q14 = QtWidgets.QHBoxLayout()
        self.l14 = QtWidgets.QLabel()
        self.l14.setText("Concentration\t=")
        self.q14.addWidget(self.l14)
        self.t14 = QtWidgets.QLineEdit()
        self.validBufferConcentration = QtGui.QDoubleValidator(0, 1000, 1)
        self.validBufferConcentration.setNotation(
            QtGui.QDoubleValidator.StandardNotation)
        self.t14.setValidator(self.validBufferConcentration)
        self.q14.addWidget(self.t14)
        self.h14 = QtWidgets.QLabel()
        self.h14.setText("<u>mg/mL</u>")
        self.h14.setToolTip("<b>Hint:</b> For 10mg/mL enter \"10\".")
        self.q14.addWidget(self.h14)

        # Buffer Groupbox
        self.groupBuffer = QtWidgets.QGroupBox("Buffer Information")
        self.groupBuffer.setCheckable(False)
        self.vbox3 = QtWidgets.QVBoxLayout()
        self.groupBuffer.setLayout(self.vbox3)
        self.vbox3.addLayout(self.q13)
        self.vbox3.addLayout(self.q14)

        # Surfactant Type
        self.q9 = QtWidgets.QHBoxLayout()
        self.l9 = QtWidgets.QLabel()
        self.l9.setText("Type\t\t=")
        self.q9.addWidget(self.l9)
        self.c9 = QtWidgets.QComboBox()
        self.q9.addWidget(self.c9, 1)
        self.h9 = QtWidgets.QLabel()
        self.h9.setText("<u>?</u>")
        self.h9.setToolTip(
            "<b>Hint:</b> If not listed, add a new entry to the list.")
        self.q9.addWidget(self.h9)
        self.c9.currentTextChanged.connect(self.new_surfactant_type)

        # Surfactant Concentration
        self.q6 = QtWidgets.QHBoxLayout()
        self.l6 = QtWidgets.QLabel()
        self.l6.setText("Concentration\t=")
        self.q6.addWidget(self.l6)
        self.t6 = QtWidgets.QLineEdit()
        self.validSurfactantConcentration = QtGui.QDoubleValidator(0, 1, 5)
        self.validSurfactantConcentration.setNotation(
            QtGui.QDoubleValidator.StandardNotation)
        self.t6.setValidator(self.validSurfactantConcentration)
        self.q6.addWidget(self.t6)
        self.h6 = QtWidgets.QLabel()
        self.h6.setText("<u>%w</u>")
        self.h6.setToolTip("<b>Hint:</b> For 0.010%w enter \"0.010\".")
        self.q6.addWidget(self.h6)

        # Surfactant Groupbox
        self.groupSurfactant = QtWidgets.QGroupBox("Surfactant Information")
        self.groupSurfactant.setCheckable(False)
        self.vbox1 = QtWidgets.QVBoxLayout()
        self.groupSurfactant.setLayout(self.vbox1)
        self.vbox1.addLayout(self.q9)
        self.vbox1.addLayout(self.q6)

        # Stabilizer Type
        self.q11 = QtWidgets.QHBoxLayout()
        self.l11 = QtWidgets.QLabel()
        self.l11.setText("Type\t\t=")
        self.q11.addWidget(self.l11)
        self.c11 = QtWidgets.QComboBox()
        self.q11.addWidget(self.c11, 1)
        self.h11 = QtWidgets.QLabel()
        self.h11.setText("<u>?</u>")
        self.h11.setToolTip(
            "<b>Hint:</b> If not listed, add a new entry to the list.")
        self.q11.addWidget(self.h11)
        self.c11.currentTextChanged.connect(self.new_stabilizer_type)

        # Stabilizer Concentration
        self.q8 = QtWidgets.QHBoxLayout()
        self.l8 = QtWidgets.QLabel()
        self.l8.setText("Concentration\t=")
        self.q8.addWidget(self.l8)
        self.t8 = QtWidgets.QLineEdit()
        self.validStabilizerConcentration = QtGui.QDoubleValidator(0, 1, 3)
        self.validStabilizerConcentration.setNotation(
            QtGui.QDoubleValidator.StandardNotation)
        self.t8.setValidator(self.validStabilizerConcentration)
        self.q8.addWidget(self.t8)
        self.h8 = QtWidgets.QLabel()
        self.h8.setText("<u>M</u>")
        self.h8.setToolTip(
            "<b>Hint:</b> For a molar mass of 0.50 enter \"0.50\".")
        self.q8.addWidget(self.h8)
        self.t8.textChanged.connect(self.calc_params)
        self.t8.editingFinished.connect(self.calc_params)

        # Stabilizer Groupbox
        self.groupStabilizer = QtWidgets.QGroupBox("Stabilizer Information")
        self.groupStabilizer.setCheckable(False)
        self.vbox3 = QtWidgets.QVBoxLayout()
        self.groupStabilizer.setLayout(self.vbox3)
        self.vbox3.addLayout(self.q11)
        self.vbox3.addLayout(self.q8)

        # read from excipient DB
        self.load_all_excipient_types()
        self.populate_excipient_proteins()
        self.populate_excipient_buffers()
        self.populate_excipient_surfactants()
        self.populate_excipient_stabilizers()

        self.r1 = QtWidgets.QHBoxLayout()
        self.l6 = QtWidgets.QLabel()
        self.l6.setText("Surface Tension\t=")
        self.r1.addWidget(self.l6)
        self.t1 = QtWidgets.QLineEdit()
        self.validSurfaceTension = QtGui.QDoubleValidator(1, 1000, 3)
        self.validSurfaceTension.setNotation(
            QtGui.QDoubleValidator.StandardNotation)
        self.t1.setValidator(self.validSurfaceTension)
        self.r1.addWidget(self.t1)
        self.h1 = QtWidgets.QLabel()
        self.h1.setText("<u>mN/m</u>")
        self.h1.setToolTip(
            "<b>This field is auto-calculated.</b>\nYou can modify it to a custom value.")
        self.r1.addWidget(self.h1)

        self.r2 = QtWidgets.QHBoxLayout()
        self.l7 = QtWidgets.QLabel()
        self.l7.setText("Contact Angle\t=")
        self.r2.addWidget(self.l7)
        self.t2 = QtWidgets.QLineEdit()
        self.validContactAngle = QtGui.QDoubleValidator(10, 80, 1)
        self.validContactAngle.setNotation(
            QtGui.QDoubleValidator.StandardNotation)
        self.t2.setValidator(self.validContactAngle)
        self.r2.addWidget(self.t2)
        self.h2 = QtWidgets.QLabel()
        self.h2.setText("<u>deg</u>")
        self.h2.setToolTip(
            "<b>This field is auto-calculated.</b>\nYou can modify it to a custom value.")
        self.r2.addWidget(self.h2)

        self.r3 = QtWidgets.QHBoxLayout()
        self.l8 = QtWidgets.QLabel()
        self.l8.setText("Density\t\t=")
        self.r3.addWidget(self.l8)
        self.t5 = QtWidgets.QLineEdit()
        self.validDensity = QtGui.QDoubleValidator(0.001, 25, 3)
        self.validDensity.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.t5.setValidator(self.validDensity)
        self.r3.addWidget(self.t5)
        self.h5 = QtWidgets.QLabel()
        self.h5.setText("<u>g/cm<sup>3</sup></u>")
        self.h5.setToolTip(
            "<b>This field is auto-calculated.</b>\nYou can modify it to a custom value.")
        self.r3.addWidget(self.h5)

        # -------------- Number of Channels (Start) --------------
        self.l_channels = QtWidgets.QLabel("Fill Channels\t=")
        self.f_channels = QtWidgets.QFrame()
        f_channels_layout = QtWidgets.QHBoxLayout()
        f_channels_layout.setContentsMargins(0, 0, 0, 0)
        self.t_channels = QtWidgets.QSpinBox()
        self.t_channels.setRange(0, 3)            # enforce 0–3
        # arrows increment/decrement by 1
        self.t_channels.setSingleStep(1)
        # if you want units, you could add " channels"
        self.t_channels.setSuffix("")
        # NOTE: setting the value must be after XML recall
        f_channels_layout.addWidget(self.t_channels)
        self.f_channels.setLayout(f_channels_layout)
        self.l_channels_hint = QtWidgets.QLabel()
        self.l_channels_hint.setText("<u>?</u>")
        self.l_channels_hint.setToolTip(
            "<b>This field is auto-calculated.</b>\nYou can modify it to a custom value.")
        h_channels = QtWidgets.QHBoxLayout()
        h_channels.addWidget(self.l_channels)
        h_channels.addWidget(self.f_channels, 1)  # stretch
        h_channels.addWidget(self.l_channels_hint)
        # -------------- Number of Channels (End) --------------

        layout_v = QtWidgets.QVBoxLayout()
        self.l0 = QtWidgets.QLabel()
        self.l0.setText(f"<b><u>Run Info for \"{self.run_name}\":</b></u>")
        # layout_v.addWidget(self.l0)
        layout_v.addLayout(self.q_runname)
        layout_v.addLayout(self.q_batch)
        layout_v.addWidget(self.notes)
        layout_v.addLayout(self.q1)  # show "Is this a bioformulation?"
        # layout_v.addLayout(self.q2)  # hide Solvent
        # layout_v.addLayout(self.q3) # hide Surfactant
        # layout_v.addLayout(self.q4) # hide Concentration
        layout_v.addWidget(self.groupSolvent)
        layout_v.addWidget(self.groupProtein)
        layout_v.addWidget(self.groupBuffer)
        layout_v.addWidget(self.groupSurfactant)
        layout_v.addWidget(self.groupStabilizer)
        self.l_channels = QtWidgets.QLabel("Number of Channels:")

        self.l5 = QtWidgets.QLabel()
        self.l5.setText("<b><u>Estimated Parameters:</b></u>")
        # layout_v.addWidget(self.l5)
        # layout_v.addLayout(self.r1) # hide Surface Tension
        # layout_v.addLayout(self.r2) # hide Contact Angle
        layout_v.addLayout(self.r3)  # show Density
        # self.l_channels.setAlignment(QtCore.Qt.AlignCenter)
        layout_v.addLayout(h_channels)
        self.q_recall = QtWidgets.QCheckBox("Remember for next run")
        self.q_recall.setChecked(True)
        self.q_recall.setEnabled(self.unsaved_changes)
        layout_v.addWidget(
            self.q_recall, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        from QATCH.common.userProfiles import UserProfiles
        valid, infos = UserProfiles.session_info()
        if valid:
            Log.d(f"Found valid session: {infos}")
            self.username = infos[0]
            self.initials = infos[1]
        else:
            try:
                # all we can do is trust the user provided a valid username (session is now expired)
                infos = UserProfiles.get_user_info(
                    UserProfiles().find(self.username, None)[1])
                self.initials = infos[1]
            except:
                self.initials = None

        if UserProfiles().count() and self.username != None:
            layout_r4 = QtWidgets.QHBoxLayout()
            self.signed = QtWidgets.QLabel("Signature\t=")
            layout_r4.addWidget(self.signed)
            self.sign = QtWidgets.QLineEdit()
            self.sign.setMaxLength(4)
            self.sign.setReadOnly(False)
            self.sign.setPlaceholderText("Initials")
            self.sign.textEdited.connect(self.sign_edit)
            self.sign.textEdited.connect(self.text_transform)
            layout_r4.addWidget(self.sign)
            # layout_v.addLayout(layout_r4) # hide here, show in external widget
            self.parent.signed_at = "[NEVER]"
            self.parent.signature_required = True
            self.parent.signature_received = False
        else:
            self.sign = QtWidgets.QLineEdit()
            self.sign.setText("[NONE]")
            self.initials = self.sign.text()
            self.parent.signed_at = self.sign.text()
            self.parent.signature_required = False
            self.parent.signature_received = False

        # START CAPTURE SIGNATURE CODE:
        # This code also exists in popUp.py in class QueryRunInfo for "ANALYZE SIGNATURE CODE"
        # The following method also is duplicated in both files: 'self.switch_user_at_sign_time'
        # There is duplicated logic code within the submit button handler: 'self.confirm'
        # The method for handling keystroke shortcuts is also duplicated too: 'self.eventFilter'
        self.signForm = QtWidgets.QWidget()
        # | QtCore.Qt.WindowStaysOnTopHint)
        self.signForm.setWindowFlags(QtCore.Qt.Dialog)
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/sign.png')
        self.signForm.setWindowIcon(QtGui.QIcon(icon_path))  # .png
        self.signForm.setWindowTitle("Signature")
        layout_sign = QtWidgets.QVBoxLayout()
        layout_curr = QtWidgets.QHBoxLayout()
        signedInAs = QtWidgets.QLabel("Signed in as: ")
        signedInAs.setAlignment(QtCore.Qt.AlignLeft)
        layout_curr.addWidget(signedInAs)
        self.signedInAs = QtWidgets.QLabel("[NONE]")
        self.signedInAs.setAlignment(QtCore.Qt.AlignRight)
        layout_curr.addWidget(self.signedInAs)
        layout_sign.addLayout(layout_curr)
        line_sep = QtWidgets.QFrame()
        line_sep.setFrameShape(QtWidgets.QFrame.HLine)
        line_sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout_sign.addWidget(line_sep)
        layout_switch = QtWidgets.QHBoxLayout()
        self.signerInit = QtWidgets.QLabel(f"Initials: <b>N/A</b>")
        layout_switch.addWidget(self.signerInit)
        switch_user = QtWidgets.QPushButton("Switch User")
        switch_user.clicked.connect(self.switch_user_at_sign_time)
        layout_switch.addWidget(switch_user)
        layout_sign.addLayout(layout_switch)
        # self.sign = QtWidgets.QLineEdit() # declared prior
        self.sign.installEventFilter(self)
        layout_sign.addWidget(self.sign)
        self.sign_do_not_ask = QtWidgets.QCheckBox(
            "Do not ask again this session")
        self.sign_do_not_ask.setEnabled(False)
        if UserProfiles.checkDevMode()[0]:  # DevMode enabled
            auto_sign_key = None
            session_key = None
            if os.path.exists(Constants.auto_sign_key_path):
                with open(Constants.auto_sign_key_path, 'r') as f:
                    auto_sign_key = f.readline()
            session_key_path = os.path.join(
                Constants.user_profiles_path, "session.key")
            if os.path.exists(session_key_path):
                with open(session_key_path, 'r') as f:
                    session_key = f.readline()
            if auto_sign_key == session_key and session_key != None:
                self.sign_do_not_ask.setChecked(True)
            else:
                self.sign_do_not_ask.setChecked(False)
                if os.path.exists(Constants.auto_sign_key_path):
                    os.remove(Constants.auto_sign_key_path)
            layout_sign.addWidget(self.sign_do_not_ask)
        self.sign_ok = QtWidgets.QPushButton("OK")
        self.sign_ok.clicked.connect(self.signForm.hide)
        self.sign_ok.clicked.connect(self.confirm)
        self.sign_ok.setDefault(True)
        self.sign_ok.setAutoDefault(True)
        self.sign_cancel = QtWidgets.QPushButton("Cancel")
        self.sign_cancel.clicked.connect(self.signForm.hide)
        layout_ok_cancel = QtWidgets.QHBoxLayout()
        layout_ok_cancel.addWidget(self.sign_ok)
        layout_ok_cancel.addWidget(self.sign_cancel)
        layout_sign.addLayout(layout_ok_cancel)
        self.signForm.setLayout(layout_sign)
        # END CAPTURE SIGNATURE CODE

        self.btn = QtWidgets.QPushButton("Save")
        self.btn.pressed.connect(self.confirm)
        layout_v.addWidget(self.btn)

        self.setLayout(layout_v)
        self.setWindowTitle("Enter Run Info")

        ###### scannow widget for batch number ######
        # note: this must be after self.setLayout() #
        self.l_scannow = QtWidgets.QWidget(self)    #
        self.l_scannow.setVisible(False)            #
        #############################################

        QtWidgets.QShortcut(QtGui.QKeySequence(
            QtCore.Qt.Key_Enter), self, activated=self.confirm)
        QtWidgets.QShortcut(QtGui.QKeySequence(
            QtCore.Qt.Key_Return), self, activated=self.confirm)
        QtWidgets.QShortcut(QtGui.QKeySequence(
            QtCore.Qt.Key_Escape), self, activated=self.close)

        self.recallFromXML()

        # Pre-populate number of channels when saving run using cached run-mode fill predictor result
        if self.post_run:
            # TODO: Using below code yields fixed '3' channels when `forecast_end_time > 0`
            #       This behavior seems like an incomplete implementation. Is this right?
            self.t_channels.setValue(
                getattr(self.parent, 'num_channels', 3)
                if not (hasattr(self.parent, 'forecast_end_time') and self.parent.forecast_end_time > 0)
                else 3
            )

        self.highlight_timer = QtCore.QTimer()  # for highlight check
        self.highlight_timer.timeout.connect(self.highlight_manual_entry)
        self.highlight_timer.setSingleShot(True)

        tb_elems = [self.t_runname, self.t_batch, self.t0,
                    self.t1, self.t2, self.t3, self.t4, self.t5,
                    self.t6, self.t8, self.t12, self.t14]
        self.reset_actions = []
        for tb in tb_elems:
            tb.textChanged.connect(self.detect_change)
            if tb in [self.t1, self.t2, self.t5]:
                tb.textChanged.connect(self.queue_highlight_check)
                # create it, so we can harvest its icon
                tb.setClearButtonEnabled(True)
                self.reset_actions.append(tb.addAction(tb.findChild(
                    QtWidgets.QToolButton).icon(), QtWidgets.QLineEdit.TrailingPosition))
                # disable it, and never use it again
                tb.setClearButtonEnabled(False)
                self.reset_actions[-1].triggered.connect(tb.clear)
                self.reset_actions[-1].triggered.connect(
                    self.clear_manual_entry)
                # self.reset_actions[-1].hovered.connect(QtWidgets.QToolTip.showText(tb.pos(), "Clear manual entry", tb))
        self.t_channels.valueChanged.connect(self.highlight_channels_box)
        self.highlight_manual_entry()  # run now
        self.highlight_channels_box()  # run now
        self.g1.buttonClicked.connect(self.detect_change)
        self.q_recall.stateChanged.connect(self.detect_change)
        self.notes.textChanged.connect(self.detect_change)
        self.c9.currentTextChanged.connect(self.detect_change)
        self.c10.currentTextChanged.connect(self.detect_change)
        self.c11.currentTextChanged.connect(self.detect_change)
        self.c13.currentTextChanged.connect(self.detect_change)
        self.t_channels.valueChanged.connect(self.detect_change)

        if self.post_run:
            self.t_batch.setFocus()

    def showScanNow(self):
        self.l_scannow.resize(self.t_batch.size())
        self.l_scannow.move(self.t_batch.pos())
        self.l_scannow.setObjectName("scannow")
        self.l_scannow.setStyleSheet(
            "#scannow { background-color: #F5FE49; border: 1px solid #7A7A7A; }")
        self.h_scannow = QtWidgets.QHBoxLayout()
        self.h_scannow.setContentsMargins(3, 0, 6, 0)
        self.t_scannow = QtWidgets.QLabel("Scan or enter now!")
        self.h_scannow.addWidget(self.t_scannow)
        self.h_scannow.addStretch()
        self.i_scannow = QtWidgets.QLabel()
        self.i_scannow.setPixmap(QtGui.QPixmap(os.path.join(Architecture.get_path(), "QATCH", "icons", "scan.png"))
                                 .scaledToHeight(self.l_scannow.height()-2))
        self.h_scannow.addWidget(self.i_scannow)
        self.l_scannow.setLayout(self.h_scannow)
        self.l_scannow.setVisible(True)
        self.t_batch.textEdited.connect(self.l_scannow.hide)

    def flashScanNow(self):
        if self.l_scannow.isVisible():
            if not self.t_batch.hasFocus():
                self.l_scannow.hide()
            elif self.t_scannow.styleSheet() == '':
                self.t_scannow.setStyleSheet("color: #F5FE49;")
                QtCore.QTimer.singleShot(250, self.flashScanNow)
            else:
                self.t_scannow.setStyleSheet('')
                QtCore.QTimer.singleShot(500, self.flashScanNow)

    def setRuns(self, count, idx):
        self.run_count = count
        self.run_idx = idx
        show_single_fields = False
        if self.run_count == 1:
            if self.post_run:
                QtCore.QTimer.singleShot(500, self.showScanNow)
                QtCore.QTimer.singleShot(1000, self.flashScanNow)
            show_single_fields = True
        else:
            run_name = self.t_runname.text()
            self.run_port = run_name[run_name.rindex("_"):]
            self.recall_xml = self.recall_xml[:-4] + self.run_port + ".xml"
            recall_parts = os.path.split(self.recall_xml)
            self.recall_xml = os.path.join(
                recall_parts[0], "recall", recall_parts[1])
            self.recallFromXML()
        self.l_runname.setVisible(show_single_fields)
        self.t_runname.setVisible(show_single_fields)
        self.h_runname.setVisible(show_single_fields)
        self.l_batch.setVisible(show_single_fields)
        self.t_batch.setVisible(show_single_fields)
        self.h_batch.setVisible(show_single_fields)
        self.notes.setVisible(show_single_fields)
        self.q_recall.setVisible(show_single_fields)
        self.btn.setVisible(show_single_fields)

    def getRunParams(self):
        run_name = self.run_name
        run_path = self.run_path
        recall_from = self.recall_xml
        run_ruling = self.run_ruling
        user_name = self.username
        return run_name, run_path, recall_from, run_ruling, user_name

    def setHiddenFields(self, run_name, batch_num, notes_txt, do_recall):
        self.t_runname.setText(run_name + self.run_port)
        self.t_batch.setText(batch_num)
        self.notes.setPlainText(notes_txt)
        self.q_recall.setChecked(do_recall)

    def recallFromXML(self):
        recalled = False
        auto_st = 0
        auto_ca = 0
        auto_dn = 0
        auto_nc = 0
        try:
            if secure_open.file_exists(self.recall_xml):
                xml_text = ""
                # secure_open(self.recall_xml, 'r') as f:
                with open(self.recall_xml, 'r') as f:
                    xml_text = f.read()
                doc = minidom.parseString(xml_text)
                params = doc.getElementsByTagName(
                    "params")

                # search for first "fill_type" that is "auto"
                for i in range(len(params)):
                    param = params[i]  # scan n-th element

                    if param.nodeType == param.TEXT_NODE:
                        continue  # only process elements

                    for p in param.childNodes:
                        if p.nodeType == p.TEXT_NODE:
                            continue  # only process elements

                        name = p.getAttribute("name")
                        if name == "fill_type":
                            input = p.getAttribute("input")
                            if input == "auto":
                                value = p.getAttribute("value")
                                auto_nc = value  # save for later
                                break

                    if auto_nc != 0:
                        break  # found value, skip to last element processing

                param = params[-1]  # most recent element

                for p in param.childNodes:
                    if p.nodeType == p.TEXT_NODE:
                        continue  # only process elements

                    name = p.getAttribute("name")
                    value = p.getAttribute("value")

                    if name == "bioformulation":
                        bval = eval(value)
                        # if bval:
                        #     self.b1.click()
                        # else:
                        #     self.b2.click()
                        self.b1.setChecked(bval)
                        self.b2.setChecked(not bval)
                    # if name == "protein":
                    #     bval = eval(value)
                    #     self.b3.setChecked(bval)
                    #     self.b4.setChecked(not bval)
                    if not self.post_run:
                        # only if recalling Run Info from prior run in Analyze
                        if name == "run_name":
                            self.run_name = value
                            self.t_runname.setText(self.run_name)
                        if name == "batch_number":
                            self.batch_found = p.getAttribute(
                                "found").lower() == "true"
                            self.t_batch.setText(value)
                        if name == "notes":
                            if value != self.notes.placeholderText():
                                self.notes.setPlainText(str(value)
                                                        # unescape new lines
                                                        .replace('\\n', '\n')
                                                        .replace("''", '"'))  # unescape double quotes
                            try:
                                notes_path = os.path.join(
                                    os.path.dirname(self.run_path), "notes.txt")
                                notes_txt = self.notes.toPlainText()
                                if os.path.exists(notes_path):
                                    with open(notes_path, 'r') as f:
                                        file_txt = "\n".join(
                                            f.read().splitlines())
                                    if file_txt != notes_txt:
                                        if Constants.import_notes_from_txt_file:
                                            Log.w(
                                                "Importing modified Notes.txt file to Run Info...")
                                            self.notes.setPlainText(file_txt)
                                        else:
                                            Log.w(
                                                "Notes.txt seems to have been modified outside of the application!")
                                            Log.w(
                                                "Please only make changes to Notes from within this application...")
                                            Log.w(
                                                "If you want to keep the file contents please copy & paste it now.")
                                            os.open(notes_path)
                                        self.detect_change()
                            except Exception as e:
                                Log.e("ERROR:", e)
                    if name == "solvent":
                        self.t0.setText(value)
                        self.lookup_completer()  # store auto_st, auto_ca, auto_dn
                        auto_st = float(self.t1.text())
                        auto_ca = float(self.t2.text())
                        auto_dn = float(self.t5.text())
                    if name == "surfactant":
                        self.t3.setText(value)
                    if name == "concentration":
                        self.t4.setText(value)
                    if name == "surface_tension":
                        self.t1.setText(value)
                        input = p.getAttribute("input")
                        if input == "auto":
                            auto_st = value
                        # auto_st = value if input == "auto" else 0
                    if name == "contact_angle":
                        self.t2.setText(value)
                        input = p.getAttribute("input")
                        if input == "auto":
                            auto_ca = value
                        # auto_ca = value if input == "auto" else 0
                    if name == "density":
                        self.t5.setText(value)
                        input = p.getAttribute("input")
                        if input == "auto":
                            auto_dn = value
                        # auto_dn = value if input == "auto" else 0

                    # new parameters for revamped run info

                    new_excipient = False
                    new_excipient_type = ""

                    if name == "protein_type":
                        if value not in self.excipient_proteins and value.casefold() != "none":
                            Log.w(
                                f"Adding new Protein Type: \"{value}\"")
                            self.excipient_proteins.append(value)
                            self.excipient_proteins = sorted(
                                self.excipient_proteins, key=str.casefold)
                            self.populate_excipient_proteins()
                            new_excipient = True
                            new_excipient_type = "Protein"
                        self.c10.setCurrentText(value)
                    if name == "protein_concentration":
                        self.t12.setText(value)
                    if name == "buffer_type":
                        if value not in self.excipient_buffers and value.casefold() != "none":
                            Log.w(
                                f"Adding new Buffer Type: \"{value}\"")
                            self.excipient_buffers.append(value)
                            self.excipient_buffers = sorted(
                                self.excipient_buffers, key=str.casefold)
                            self.populate_excipient_buffers()
                            new_excipient = True
                            new_excipient_type = "Buffer"
                        self.c13.setCurrentText(value)
                    if name == "buffer_concentration":
                        self.t14.setText(value)
                    if name == "surfactant_type":
                        if value not in self.excipient_surfactants and value.casefold() != "none":
                            Log.w(
                                f"Unknown Surfactant Type: \"{value}\" not in list")
                            self.excipient_surfactants.append(value)
                            self.excipient_surfactants = sorted(
                                self.excipient_surfactants, key=str.casefold)
                            self.populate_excipient_surfactants()
                            new_excipient = True
                            new_excipient_type = "Surfactant"
                        self.c9.setCurrentText(value)
                    if name == "surfactant_concentration":
                        self.t6.setText(value)
                    if name == "stabilizer_type":
                        if value not in self.excipient_stabilizers and value.casefold() != "none":
                            Log.w(
                                f"Unknown Stabilizer Type: \"{value}\" not in list")
                            self.excipient_stabilizers.append(value)
                            self.excipient_stabilizers = sorted(
                                self.excipient_stabilizers, key=str.casefold)
                            self.populate_excipient_stabilizers()
                            new_excipient = True
                            new_excipient_type = "Stabilizer"
                        self.c11.setCurrentText(value)
                    if name == "stabilizer_concentration":
                        self.t8.setText(value)

                    if new_excipient:
                        self.excipient_db.add_base_excipient(
                            new_excipient_type,
                            value)

                    # Set the fill type to the recalled number of channels from the XML
                    # file.
                    if name == "fill_type":
                        self.t_channels.setValue(int(value))

                if len(param.childNodes) == 0:
                    # uncheck "Remember for next time"
                    self.q_recall.setChecked(False)
                recalled = True
        except:
            Log.e("Failed to recall info from saved file.")

            import sys
            from traceback import format_tb

            limit = None
            t, v, tb = sys.exc_info()
            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        self.show_hide_gui(None)  # enable/disable elements

        # specify auto/manual inputs when recalled data
        if recalled:
            self.auto_st = float(auto_st)
            self.auto_ca = float(auto_ca)
            self.auto_dn = float(auto_dn)
            self.auto_nc = int(auto_nc)
        return recalled

    def prevent_duplicate_scans(self):
        current_text = self.t_batch.text()
        min_batch_num_len = 3  # characters, i.e. 'MMx'
        given_batch_num_len = len(current_text)
        split_at_idx = int(given_batch_num_len / 2)
        if split_at_idx > min_batch_num_len:
            if given_batch_num_len % 2 == 0:  # even only
                first_half = current_text[:split_at_idx]
                second_half = current_text[split_at_idx:]
                Log.d(
                    f"prevent_duplicate_scans(): Checking '{first_half}|{second_half}'")
                if first_half == second_half:
                    Log.w(f"Duplicate scan ignored: {second_half}")
                    self.t_batch.setText(first_half)

    def queue_highlight_check(self):
        if self.highlight_timer.isActive():
            self.highlight_timer.stop()
        self.highlight_timer.start(50)

    def clear_manual_entry(self):
        self.highlight_manual_entry(True)

    def highlight_manual_entry(self, force_clear=False):
        try:
            # Log.w("Checking entries...")
            allow_reset = True
            if self.b1.isChecked():
                if len(self.t3.text()) == 0 or len(self.t4.text()) == 0:
                    allow_reset = False
            elif self.b2.isChecked():
                if len(self.t0.text()) == 0:
                    allow_reset = False
            else:
                allow_reset = False
            # if any([self.t1.hasFocus(), self.t2.hasFocus(), self.t5.hasFocus()]):
            #     allow_reset = False
            if allow_reset:
                if len(self.t1.text().strip()) == 0:
                    if not force_clear:
                        return
                    self.t1.setText("{:3.3f}".format(self.auto_st))
                if len(self.t2.text().strip()) == 0:
                    if not force_clear:
                        return
                    self.t2.setText("{:2.1f}".format(self.auto_ca))
                if len(self.t5.text().strip()) == 0:
                    if not force_clear:
                        return
                    self.t5.setText("{:1.3f}".format(self.auto_dn))
            manual_st = float(self.t1.text()) != float(
                f"{self.auto_st:3.3f}") if allow_reset else True
            manual_ca = float(self.t2.text()) != float(
                f"{self.auto_ca:2.1f}") if allow_reset else True
            manual_dn = float(self.t5.text()) != float(
                f"{self.auto_dn:1.3f}") if allow_reset else True
            self.t1.setStyleSheet(
                "border: 2px solid black;" if manual_st else "background-color: #eee;")
            self.t2.setStyleSheet(
                "border: 2px solid black;" if manual_ca else "background-color: #eee;")
            self.t5.setStyleSheet(
                "border: 2px solid black;" if manual_dn else "background-color: #eee;")
            self.reset_actions[0].setVisible(
                manual_st if allow_reset else False)
            self.reset_actions[1].setVisible(
                manual_ca if allow_reset else False)
            self.reset_actions[2].setVisible(
                manual_dn if allow_reset else False)
        except Exception as e:
            Log.e(f"Invalid parameter: {e}")

    def highlight_channels_box(self):
        num_channels = int(self.t_channels.text()) if len(
            self.t_channels.text()) else 3
        manual_nc = (num_channels != self.auto_nc) and (self.auto_nc != 0)
        if manual_nc:
            self.t_channels.setPalette(
                QtWidgets.QApplication.palette())  # reset background
            self.f_channels.setStyleSheet(
                "QFrame { border: 1px solid black; }")  # set border
        else:
            palette = self.t_channels.palette()
            palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#eeeeee"))
            self.f_channels.setStyleSheet("")  # reset border
            self.t_channels.setPalette(palette)  # set background

    def switch_user_at_sign_time(self):
        from QATCH.common.userProfiles import UserProfiles, UserRoles
        new_username, new_initials, new_userrole = UserProfiles.change(
            UserRoles.ANALYZE)
        if UserProfiles.check(UserRoles(new_userrole), UserRoles.ANALYZE):
            if self.username != new_username:
                self.username = new_username
                self.initials = new_initials
                self.signedInAs.setText(self.username)
                self.signerInit.setText(f"Initials: <b>{self.initials}</b>")
                self.parent.signature_received = False
                self.parent.signature_required = True
                self.sign.setReadOnly(False)
                self.sign.setMaxLength(4)
                self.sign.clear()

                Log.d("User name changed. Changing sign-in user info.")
                self.parent.ControlsWin.username.setText(
                    f"User: {new_username}")
                self.parent.ControlsWin.userrole = UserRoles(new_userrole)
                self.parent.ControlsWin.signinout.setText("&Sign Out")
                self.parent.ControlsWin.ui1.tool_User.setText(new_username)
                self.parent.AnalyzeProc.tool_User.setText(new_username)
                if self.parent.ControlsWin.userrole != UserRoles.ADMIN:
                    self.parent.ControlsWin.manage.setText(
                        "&Change Password...")
            else:
                Log.d("User switched users to the same user profile. Nothing to change.")
            # PopUp.warning(self, Constants.app_title, "User has been switched.\n\nPlease sign now.")
        # elif new_username == None and new_initials == None and new_userrole == 0:
        else:
            if new_username == None and not UserProfiles.session_info()[0]:
                Log.d("User session invalidated. Switch users credentials incorrect.")
                self.parent.ControlsWin.username.setText("User: [NONE]")
                self.parent.ControlsWin.userrole = UserRoles.NONE
                self.parent.ControlsWin.signinout.setText("&Sign In")
                self.parent.ControlsWin.manage.setText("&Manage Users...")
                self.parent.ControlsWin.ui1.tool_User.setText("Anonymous")
                self.parent.AnalyzeProc.tool_User.setText("Anonymous")
                PopUp.warning(self, Constants.app_title,
                              "User has not been switched.\n\nReason: Not authenticated.")
            if new_username != None and UserProfiles.session_info()[0]:
                Log.d("User name changed. Changing sign-in user info.")
                self.parent.ControlsWin.username.setText(
                    f"User: {new_username}")
                self.parent.ControlsWin.userrole = UserRoles(new_userrole)
                self.parent.ControlsWin.signinout.setText("&Sign Out")
                self.parent.ControlsWin.ui1.tool_User.setText(new_username)
                self.parent.AnalyzeProc.tool_User.setText(new_username)
                if self.parent.ControlsWin.userrole != UserRoles.ADMIN:
                    self.parent.ControlsWin.manage.setText(
                        "&Change Password...")
                PopUp.warning(self, Constants.app_title,
                              "User has not been switched.\n\nReason: Not authorized.")

            Log.d("User did not authenticate for role to switch users.")

    def detect_change(self):
        self.unsaved_changes = True

    def sign_edit(self):
        if self.sign.text().upper() == self.initials:
            sign_text = f"{self.username} ({self.sign.text().upper()})"
            self.sign.setMaxLength(len(sign_text))
            self.sign.setText(sign_text)
            self.sign.setReadOnly(True)
            self.signed.setText(f"CAPTURE by\t=")
            self.parent.signed_at = dt.datetime.now().isoformat()
            self.parent.signature_received = True
            self.sign_do_not_ask.setEnabled(True)

    def text_transform(self):
        text = self.sign.text()
        if len(text) in [1, 2, 3, 4]:
            # will not fire 'textEdited' signal again
            self.sign.setText(text.upper())

    def show(self):
        super(QueryRunInfo, self).show()
        width = 350
        height = self.height()
        area = QtWidgets.QDesktopWidget().availableGeometry()
        left = int((area.width() - width) / 2)
        top = int((area.height() - height) / 2)
        self.setGeometry(left, top, width, height)

    def new_protein_type(self, text: str):
        if text.casefold() == "add new...":
            self.add_protein_type = QtWidgets.QWidget()
            self.add_protein_type.setWindowTitle("Protein Types")
            layout = QtWidgets.QVBoxLayout()
            label = QtWidgets.QLabel("Available Protein Types:")
            self.protein_types_multiline = QtWidgets.QPlainTextEdit()
            self.protein_types_multiline.setPlainText(
                "\n".join(self.excipient_proteins))
            save = QtWidgets.QPushButton("Save")
            save.clicked.connect(self.save_excipient_proteins)
            layout.addWidget(label)
            layout.addWidget(self.protein_types_multiline)
            layout.addWidget(save)
            self.add_protein_type.setLayout(layout)
            self.add_protein_type.show()
            self.protein_types_multiline.setFocus()
            self.protein_types_multiline.moveCursor(
                QtGui.QTextCursor.MoveOperation.End)
        elif text.casefold() == "none":
            self.t12.setText("0")  # clear Protein Concentration
            self.t12.setEnabled(False)
            self.b2.click()  # check "no" to bioformulation question
        else:
            self.t12.setEnabled(True)
            pass  # do nothing if any other value was selected

    def new_buffer_type(self, text: str):
        if text.casefold() == "add new...":
            self.add_buffer_type = QtWidgets.QWidget()
            self.add_buffer_type.setWindowTitle("Buffer Types")
            layout = QtWidgets.QVBoxLayout()
            label = QtWidgets.QLabel("Available Buffer Types:")
            self.buffer_types_multiline = QtWidgets.QPlainTextEdit()
            self.buffer_types_multiline.setPlainText(
                "\n".join(self.excipient_buffers))
            save = QtWidgets.QPushButton("Save")
            save.clicked.connect(self.save_excipient_buffers)
            layout.addWidget(label)
            layout.addWidget(self.buffer_types_multiline)
            layout.addWidget(save)
            self.add_buffer_type.setLayout(layout)
            self.add_buffer_type.show()
            self.buffer_types_multiline.setFocus()
            self.buffer_types_multiline.moveCursor(
                QtGui.QTextCursor.MoveOperation.End)
        elif text.casefold() == "none":
            self.t14.setText("0")  # clear Buffer Concentration
            self.t14.setEnabled(False)
        else:
            self.t14.setEnabled(True)
            pass  # do nothing if any other value was selected

    def new_surfactant_type(self, text: str):
        if text.casefold() == "add new...":
            self.add_surfactant_type = QtWidgets.QWidget()
            self.add_surfactant_type.setWindowTitle("Surfactant Types")
            layout = QtWidgets.QVBoxLayout()
            label = QtWidgets.QLabel("Available Surfactant Types:")
            self.surfactant_types_multiline = QtWidgets.QPlainTextEdit()
            self.surfactant_types_multiline.setPlainText(
                "\n".join(self.excipient_surfactants))
            save = QtWidgets.QPushButton("Save")
            save.clicked.connect(self.save_excipient_surfactants)
            layout.addWidget(label)
            layout.addWidget(self.surfactant_types_multiline)
            layout.addWidget(save)
            self.add_surfactant_type.setLayout(layout)
            self.add_surfactant_type.show()
            self.surfactant_types_multiline.setFocus()
            self.surfactant_types_multiline.moveCursor(
                QtGui.QTextCursor.MoveOperation.End)
        elif text.casefold() == "none":
            self.t6.setText("0")  # clear Surfactant Concentration
            self.t6.setEnabled(False)
        else:
            self.t6.setEnabled(True)
            pass  # do nothing if any other value was selected

    def new_stabilizer_type(self, text: str):
        if text.casefold() == "add new...":
            self.add_stabilizer_type = QtWidgets.QWidget()
            self.add_stabilizer_type.setWindowTitle("Stabilizer Types")
            layout = QtWidgets.QVBoxLayout()
            label = QtWidgets.QLabel("Available Stabilizer Types:")
            self.stabilizer_types_multiline = QtWidgets.QPlainTextEdit()
            self.stabilizer_types_multiline.setPlainText(
                "\n".join(self.excipient_stabilizers))
            save = QtWidgets.QPushButton("Save")
            save.clicked.connect(self.save_excipient_stabilizers)
            layout.addWidget(label)
            layout.addWidget(self.stabilizer_types_multiline)
            layout.addWidget(save)
            self.add_stabilizer_type.setLayout(layout)
            self.add_stabilizer_type.show()
            self.stabilizer_types_multiline.setFocus()
            self.stabilizer_types_multiline.moveCursor(
                QtGui.QTextCursor.MoveOperation.End)
        elif text.casefold() == "none":
            self.t8.setText("0")  # clear Stabilizer Concentration
            self.t8.setEnabled(False)
        else:
            self.t8.setEnabled(True)
            pass  # do nothing if any other value was selected

    def load_all_excipient_types(self):
        self.excipient_proteins = []
        self.excipient_buffers = []
        self.excipient_surfactants = []
        self.excipient_stabilizers = []
        excipients = self.excipient_db.list_base_excipients()

        # Create default excipients list (if none exist)
        if len(excipients) == 0:
            self.excipient_db.add_base_excipient("Buffer", "Acetate")
            self.excipient_db.add_base_excipient("Buffer", "Histidine")
            self.excipient_db.add_base_excipient("Buffer", "PBS")
            self.excipient_db.add_base_excipient("Surfactant", "TWEEN20")
            self.excipient_db.add_base_excipient("Surfactant", "TWEEN80")
            self.excipient_db.add_base_excipient("Stabilizer", "Sucrose")
            self.excipient_db.add_base_excipient("Stabilizer", "Trehalose")
            excipients = self.excipient_db.list_base_excipients()  # reload

        for e in excipients:
            if e['etype'] == "Protein":
                self.excipient_proteins.append(e['name'])
            if e['etype'] == "Buffer":
                self.excipient_buffers.append(e['name'])
            if e['etype'] == "Surfactant":
                self.excipient_surfactants.append(e['name'])
            if e['etype'] == "Stabilizer":
                self.excipient_stabilizers.append(e['name'])
        # this is case-sensitive, which is not what we want:
        # self.excipient_proteins.sort()
        # self.excipient_surfactants.sort()
        # self.excipient_stabilizers.sort()
        # this is using a case-insensitive sorting method:
        self.excipient_proteins = sorted(
            self.excipient_proteins, key=str.casefold)
        self.excipient_buffers = sorted(
            self.excipient_buffers, key=str.casefold)
        self.excipient_surfactants = sorted(
            self.excipient_surfactants, key=str.casefold)
        self.excipient_stabilizers = sorted(
            self.excipient_stabilizers, key=str.casefold)
        Log.d("Proteins:", self.excipient_proteins)
        Log.d("Buffers:", self.excipient_buffers)
        Log.d("Surfactants:", self.excipient_surfactants)
        Log.d("Stabilizers:", self.excipient_stabilizers)

    def save_excipient_proteins(self):
        old_proteins = self.excipient_proteins.copy()
        new_proteins = self.protein_types_multiline.toPlainText().splitlines()
        for name in new_proteins:
            name = name.strip()
            if name not in old_proteins and len(name):
                self.excipient_db.add_base_excipient("Protein", name)
                self.excipient_proteins.append(name)
        for name in old_proteins:
            if name not in new_proteins:
                self.excipient_db.delete_base_excipient("Protein", name)
                self.excipient_proteins.remove(name)
        # self.excipient_proteins.sort()
        self.excipient_proteins = sorted(
            self.excipient_proteins, key=str.casefold)
        self.populate_excipient_proteins()
        self.add_protein_type.close()

    def save_excipient_buffers(self):
        old_buffers = self.excipient_buffers.copy()
        new_buffers = self.buffer_types_multiline.toPlainText().splitlines()
        for name in new_buffers:
            name = name.strip()
            if name not in old_buffers and len(name):
                self.excipient_db.add_base_excipient("Buffer", name)
                self.excipient_buffers.append(name)
        for name in old_buffers:
            if name not in new_buffers:
                self.excipient_db.delete_base_excipient("Buffer", name)
                self.excipient_buffers.remove(name)
        # self.excipient_buffers.sort()
        self.excipient_buffers = sorted(
            self.excipient_buffers, key=str.casefold)
        self.populate_excipient_buffers()
        self.add_buffer_type.close()

    def save_excipient_surfactants(self):
        old_surfactants = self.excipient_surfactants.copy()
        new_surfactants = self.surfactant_types_multiline.toPlainText().splitlines()
        for name in new_surfactants:
            name = name.strip()
            if name not in old_surfactants and len(name):
                self.excipient_db.add_base_excipient("Surfactant", name)
                self.excipient_surfactants.append(name)
        for name in old_surfactants:
            if name not in new_surfactants:
                self.excipient_db.delete_base_excipient("Surfactant", name)
                self.excipient_surfactants.remove(name)
        # self.excipient_surfactants.sort()
        self.excipient_surfactants = sorted(
            self.excipient_surfactants, key=str.casefold)
        self.populate_excipient_surfactants()
        self.add_surfactant_type.close()

    def save_excipient_stabilizers(self):
        old_stabilizers = self.excipient_stabilizers.copy()
        new_stabilizers = self.stabilizer_types_multiline.toPlainText().splitlines()
        for name in new_stabilizers:
            name = name.strip()
            if name not in old_stabilizers and len(name):
                self.excipient_db.add_base_excipient("Stabilizer", name)
                self.excipient_stabilizers.append(name)
        for name in old_stabilizers:
            if name not in new_stabilizers:
                self.excipient_db.delete_base_excipient("Stabilizer", name)
                self.excipient_stabilizers.remove(name)
        # self.excipient_stabilizers.sort()
        self.excipient_stabilizers = sorted(
            self.excipient_stabilizers, key=str.casefold)
        self.populate_excipient_stabilizers()
        self.add_stabilizer_type.close()

    def populate_excipient_proteins(self):
        try:
            num_items = self.c10.count()
            self.c10.clear()
            self.c10.addItem("none")
            self.c10.addItems(self.excipient_proteins)
            self.c10.addItem("add new...")
            if num_items:
                # select newly entered value (last in list)
                # TODO: since we `sort()`, "newest" is not "last"
                self.c10.setCurrentText(self.excipient_proteins[-1])
            else:
                self.c10.setCurrentIndex(0)  # initial load value: none
        except:
            Log.e("Failed to update proteins list after saving.")

    def populate_excipient_buffers(self):
        try:
            num_items = self.c13.count()
            self.c13.clear()
            self.c13.addItem("none")
            self.c13.addItems(self.excipient_buffers)
            self.c13.addItem("add new...")
            if num_items:
                # select newly entered value (last in list)
                # TODO: since we `sort()`, "newest" is not "last"
                self.c13.setCurrentText(self.excipient_buffers[-1])
            else:
                self.c13.setCurrentIndex(0)  # initial load value: none
        except:
            Log.e("Failed to update buffers list after saving.")

    def populate_excipient_surfactants(self):
        try:
            num_items = self.c9.count()
            self.c9.clear()
            self.c9.addItem("none")
            self.c9.addItems(self.excipient_surfactants)
            self.c9.addItem("add new...")
            if num_items:
                # select newly entered value (last in list)
                # TODO: since we `sort()`, "newest" is not "last"
                self.c9.setCurrentText(self.excipient_surfactants[-1])
            else:
                self.c9.setCurrentIndex(0)  # initial load value: none
        except:
            Log.e("Failed to update surfactants list after saving.")

    def populate_excipient_stabilizers(self):
        try:
            num_items = self.c11.count()
            self.c11.clear()
            self.c11.addItem("none")
            self.c11.addItems(self.excipient_stabilizers)
            self.c11.addItem("add new...")
            if num_items:
                # select newly entered value (last in list)
                # TODO: since we `sort()`, "newest" is not "last"
                self.c11.setCurrentText(self.excipient_stabilizers[-1])
            else:
                self.c11.setCurrentIndex(0)  # initial load value: none
        except:
            Log.e("Failed to update stabilizers list after saving.")

    # def load_surfactant_types(self):
    #     try:
    #         path_to_types_file = os.path.join(
    #             Constants.local_app_data_path, "settings", "surfactantTypes.txt")
    #         if not os.path.isfile(path_to_types_file):
    #             self.save_surfactant_types()  # load and create file with defaults
    #         else:
    #             with open(path_to_types_file, "r") as f:
    #                 self.available_surfactant_types = f.read.splitlines()
    #     except:
    #         Log.e("Failed to load surfactant types list from file.")

    # def save_surfactant_types(self):
    #     try:
    #         self.available_surfactant_types = self.surfactant_types_multiline.toPlainText().splitlines()
    #     except:
    #         Log.w("Creating default surfactant types list...")
    #         self.available_surfactant_types = ["none", "tween-20", "tween-80"]
    #     try:
    #         path_to_types_file = os.path.join(
    #             Constants.local_app_data_path, "settings", "surfactantTypes.txt")
    #         with open(path_to_types_file, "w") as f:
    #             f.writelines("\n".join(self.available_surfactant_types))
    #     except:
    #         Log.e("Failed to save surfactant types list to file.")
    #     try:
    #         num_items = self.c5.count()
    #         self.c5.clear()
    #         self.c5.addItems(self.available_surfactant_types)
    #         self.c5.addItem("add new...")
    #         if num_items:
    #             # select newly entered value (last in list)
    #             self.c5.setCurrentText(self.available_surfactant_types[-1])
    #         else:
    #             self.c5.setCurrentIndex(0)  # initial load value: none
    #     except:
    #         Log.e("Failed to update surfactant list after saving.")

    def lookup_completer(self):
        try:
            solvent = self.t0.text()
            if solvent in self.fluids:
                idx = self.fluids.index(solvent)
                surface_tension = 72  # self.surface_tensions[idx]
                contact_angle = 55  # 20
                density = self.densities[idx]
                # special_CAs = ["water", "deuterium", "glycerol"]
                # for s in special_CAs:
                #     if solvent.lower().find(s) >= 0:
                #         contact_angle = 55
                self.t1.setText("{:3.3f}".format(surface_tension))
                self.t2.setText("{:2.1f}".format(contact_angle))
                self.t5.setText("{:1.3f}".format(density))
                self.auto_st = float(self.t1.text()) if len(
                    self.t1.text()) else 0  # surface_tension
                self.auto_ca = float(self.t2.text()) if len(
                    self.t2.text()) else 0  # contact_angle
                self.auto_dn = float(self.t5.text()) if len(
                    self.t5.text()) else 0  # density
            else:
                self.t1.clear()
                self.t2.clear()
                self.t5.clear()
        except Exception as e:
            Log.e("ERROR:", e)
            Log.e(
                f"Failed to lookup parametres for solvent '{self.t0.text()}'.\nPlease try again, or enter parameters manually.")

    def enforce_completer(self):
        solvent = self.t0.text()
        if len(solvent.strip()) < 3:  # length of shortest valid solvent string in list
            self.t0.clear()
            return
        if not solvent in self.fluids:
            Log.w(
                f"Unknown solvent '{self.t0.text()}' entered.\nPlease try again, or enter parameters manually.")
            # self.t0.clear()

    def show_hide_gui(self, object):
        curr_state = None
        is_bioformulation = None
        if self.isVisible():
            curr_state = not self.t0.isEnabled()
        if self.b1.isChecked():
            is_bioformulation = True
        if self.b2.isChecked():
            is_bioformulation = False

        # Always do this on form load, only ignore if after "show" call
        if is_bioformulation != curr_state:  # only if value actually changed
            if is_bioformulation != True:
                self.t3.clear()
                self.t4.clear()
                # setting protein type to "none" will disable protein contentration
                # self.c10.setCurrentIndex(0)
            else:
                # self.t0.clear()
                pass

            self.t0.setEnabled(is_bioformulation == False)  # solvent
            # surfactant (hidden)
            self.t3.setEnabled(is_bioformulation == True)
            self.t4.setEnabled(is_bioformulation == True)  # protein (hidden)

            self.groupSolvent.setVisible(
                is_bioformulation == False)  # solvent group
            self.groupProtein.setVisible(
                is_bioformulation == True)  # protein group
            self.groupBuffer.setVisible(
                is_bioformulation == True)  # buffer group
            self.groupSurfactant.setVisible(
                is_bioformulation == True)  # surfactant group
            self.groupStabilizer.setVisible(
                is_bioformulation == True)  # stabilizer group

            if curr_state is not None:
                # resize vertically to fit fields (if visible)
                # NOTE: use timer to add to scheduler after redraw event
                QtCore.QTimer.singleShot(
                    1, lambda: self.resize(self.width(), self.minimumHeight()))

            if object == None:
                self.auto_st = float(self.t1.text()) if len(
                    self.t1.text()) else 0
                self.auto_ca = float(self.t2.text()) if len(
                    self.t2.text()) else 0
                self.auto_dn = float(self.t5.text()) if len(
                    self.t5.text()) else 0
            elif curr_state == None:
                return  # Run Info not visible, stop here
            elif is_bioformulation != None:
                if is_bioformulation:
                    self.calc_params()
                else:
                    self.lookup_completer()
            else:  # form is blank
                self.t1.clear()
                self.t2.clear()
                self.t5.clear()

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
            "border: 1px solid black;" if not found else "background-color: #eee;")
        # detect when AUDIT says 'found = false' but now it is found
        if self.batch_found != found:
            self.batch_found = found
            self.detect_change()

    def calc_params(self):
        from QATCH.processors.Analyze import AnalyzeProcess

        if self.t3.text() != "0":
            self.t3.setText("0")  # Surfactant locked
        if self.t4.text() != "0":
            self.t4.setText("0")  # Concentration locked

        try:
            surfactant = float(self.t3.text()) if len(self.t3.text()) else 0
            concentration = float(self.t4.text()) if len(self.t4.text()) else 0
            protein_concentration = float(
                self.t12.text()) if len(self.t12.text()) else 0
            stabilizer_type = self.c11.currentText().casefold()  # i.e. "sucrose"
            stabilizer_concentration = float(
                self.t8.text()) if len(self.t8.text()) else 0
        except Exception as e:
            return

        input_error = False
        if (len(self.t3.text()) == 0 or len(self.t4.text()) == 0 or
                len(self.t12.text()) == 0 or len(self.t8.text()) == 0 or
                not self.isVisible()):
            input_error = True
        if not input_error:
            # perform additional validations only if fields are not blank
            if not self.t3.hasAcceptableInput():
                Log.e("Input Error: Surfactant must be between {} and {}."
                      .format(
                          self.validSurfactant.bottom(),
                          self.validSurfactant.top()))
                input_error = True
            if not self.t4.hasAcceptableInput():
                Log.e("Input Error: Concentration must be between {} and {}."
                      .format(
                          self.validConcentration.bottom(),
                          self.validConcentration.top()))
                input_error = True
            if not self.t12.hasAcceptableInput():
                Log.e("Input Error: Protein Concentration must be between {} and {}."
                      .format(
                          self.validProteinConcentration.bottom(),
                          self.validProteinConcentration.top()))
                input_error = True
            if not self.t8.hasAcceptableInput():
                Log.e("Input Error: Stabilizer Concentration must be between {} and {}."
                      .format(
                          self.validStabilizerConcentration.bottom(),
                          self.validStabilizerConcentration.top()))
                input_error = True
        if input_error:
            if self.isVisible():
                self.t1.clear()  # surface tension (hidden)
                self.t2.clear()  # contact angle (hidden)
                self.t5.clear()  # density
            return

        try:
            # Log.d(f"passing in {surfactant} and {concentration}")
            surface_tension = AnalyzeProcess.Lookup_ST(
                surfactant, concentration)
            contact_angle = AnalyzeProcess.Lookup_CA(surfactant, concentration)
            density = AnalyzeProcess.Lookup_DN(surfactant=surfactant,
                                               concentration=protein_concentration,
                                               stabilizer_type=stabilizer_type,
                                               stabilizer_concentration=stabilizer_concentration)
            self.t1.setText("{:3.3f}".format(surface_tension))
            self.t2.setText("{:2.1f}".format(contact_angle))
            self.t5.setText("{:1.3f}".format(density))
            self.auto_st = float(self.t1.text()) if len(
                self.t1.text()) else 0  # surface_tension
            self.auto_ca = float(self.t2.text()) if len(
                self.t2.text()) else 0  # contact_angle
            self.auto_dn = float(self.t5.text()) if len(
                self.t5.text()) else 0  # density
        except Exception as e:
            Log.e("ERROR:", e)
            Log.e("Lookup Error: Failed to estimate ST and/or CA.")
            self.t1.clear()  # surface tension (hidden)
            self.t2.clear()  # contact angle (hidden)
            self.t5.clear()  # density

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress and obj is self.sign and self.sign.hasFocus():
            if event.key() in [QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return, QtCore.Qt.Key_Space]:
                if self.parent.signature_received:
                    self.sign_ok.clicked.emit()
            if event.key() == QtCore.Qt.Key_Escape:
                self.sign_cancel.clicked.emit()
        return super().eventFilter(obj, event)

    def confirm(self, force: bool = False):
        """
        On confirmation click, the corresponding run_info tag is updated match the run information
        parameterized by the user in the 'Run Info' form of the Analyze window.
        Modifications to the run_name cause a rewrite of the entire run directory and
        corresponding data files to reflect the modified run_name.

        Args:
            force (bool) : (Optional) Flag to force the writing of the run XML file regardless of errors
                or other preventative measure.

        Returns:
            bool : True if confirmation and writing of run_info was successful.  On errors, Flase is returned
                indicating a failed writing attempt.
        """
        from QATCH.processors.Analyze import AnalyzeProcess

        # Parameter initialization
        surfactant = 0  # float(self.t3.text()) if len(self.t3.text()) else 0
        concentration = float(self.t4.text()) if len(self.t4.text()) else 0
        st = AnalyzeProcess.Lookup_ST(surfactant, concentration)
        ca = float(self.t2.text()) if len(self.t2.text()) else 0
        density = float(self.t5.text()) if len(self.t5.text()) else 0

        manual_st = (st != self.auto_st)
        manual_ca = (ca != self.auto_ca)
        manual_dn = (density != self.auto_dn)

        # Get the number of channels from the textbox.  Default to full fill if
        # they cannot be processed.
        num_channels = int(self.t_channels.text()) if len(
            self.t_channels.text()) else 3
        manual_nc = (num_channels != self.auto_nc) and (self.auto_nc != 0)

        # Form input error checking for valid Surfactant, Concentration, Surface Tension,
        # Contact Angle, and Density. Errors are logged to the user and the input_error flag
        # is set to True.
        input_error = False
        if self.t3.isEnabled() and not self.t3.hasAcceptableInput():
            Log.e(tags=TAG, msg="Input Error: Surfactant must be between {} and {}."
                  .format(
                      self.validSurfactant.bottom(),
                      self.validSurfactant.top()))
            input_error = True
        if self.t4.isEnabled() and not self.t4.hasAcceptableInput():
            Log.e(tag=TAG, msg="Input Error: Concentration must be between {} and {}."
                  .format(
                      self.validConcentration.bottom(),
                      self.validConcentration.top()))
            input_error = True
        if not self.t1.hasAcceptableInput():
            Log.e(tag=TAG, msg="Input Error: Surface Tension must be between {} and {}."
                  .format(
                      self.validSurfaceTension.bottom(),
                      self.validSurfaceTension.top()))
            input_error = True
        if not self.t2.hasAcceptableInput():
            Log.e(tag=TAG, msg="Input Error: Contact Angle must be between {} and {}."
                  .format(
                      self.validContactAngle.bottom(),
                      self.validContactAngle.top()))
            input_error = True
        if not self.t5.hasAcceptableInput():
            Log.e(tag=TAG, msg="Input Error: Density must be between {} and {}."
                  .format(
                      self.validDensity.bottom(),
                      self.validDensity.top()))
            input_error = True
        if not self.t12.hasAcceptableInput():
            Log.e("Input Error: Protein Concentration must be between {} and {}."
                  .format(
                      self.validProteinConcentration.bottom(),
                      self.validProteinConcentration.top()))
            input_error = True
        if not self.t14.hasAcceptableInput():
            Log.e("Input Error: Buffer Concentration must be between {} and {}."
                  .format(
                      self.validBufferConcentration.bottom(),
                      self.validBufferConcentration.top()))
            input_error = True
        if not self.t6.hasAcceptableInput():
            Log.e("Input Error: Surfactant Concentration must be between {} and {}."
                  .format(
                      self.validSurfactantConcentration.bottom(),
                      self.validSurfactantConcentration.top()))
            input_error = True
        if not self.t8.hasAcceptableInput():
            Log.e("Input Error: Stabilizer Concentration must be between {} and {}."
                  .format(
                      self.validStabilizerConcentration.bottom(),
                      self.validStabilizerConcentration.top()))
            input_error = True
        if self.c10.currentText().casefold() == "none" and self.c10.isEnabled() and self.c10.isVisible():
            Log.e(
                "Input Error: You must provide a Protein Type if this is a bioformulation.")
            Log.e(
                "Either select a Protein Type other than \"none\" or click \"no\" for \"Is this a bioformulation?\"")
            input_error = True
        if self.t12.text() == "0" and self.t12.isEnabled() and self.t12.isVisible():
            Log.e(
                "Input Error: Protein Concentration should be non-zero when Protein Type is not \"none\".")
            input_error = True
        if self.t14.text() == "0" and self.t14.isEnabled() and self.t14.isVisible():
            Log.e(
                "Input Error: Buffer Concentration should be non-zero when Buffer Type is not \"none\".")
            input_error = True
        if self.t6.text() == "0" and self.t6.isEnabled() and self.t6.isVisible():
            Log.e(
                "Input Error: Surfactant Concentration should be non-zero when Surfactant Type is not \"none\".")
            input_error = True
        if self.t8.text() == "0" and self.t8.isEnabled() and self.t8.isVisible():
            Log.e(
                "Input Error: Stabilizer Concentration should be non-zero when Stabilizer Type is not \"none\".")
            input_error = True

        # If the force parameter is set to True, input errors are ignored.
        if force:
            Log.w(tag=TAG, msg="Forcing XML write regardless of input errors!")
            input_error = False
        if input_error:
            Log.w(tag=TAG, msg="Input error: Not saving Run Info.")
            return False

        # Error checking for valid signature per captured run: If the do not ask option
        # is checked, then signatures are ignored for this session.
        if self.parent.signature_received == False and self.sign_do_not_ask.isChecked():
            Log.w(
                tag=TAG, msg=f"Signing CAPTURE with initials {self.initials} (not asking again)")
            self.parent.signed_at = dt.datetime.now().isoformat()
            self.parent.signature_received = True  # Do not ask again this session

        # Error checking for valid signature per captured run: If the signature is still required
        # the user should be prompted to provide a valid signature as a valid batch parameter. With the
        # force flag enabled, this step is ignored.
        if self.parent.signature_required and not self.parent.signature_received:  # missing initials
            if force or self.run_idx != 0:
                Log.w(
                    tag=TAG, msg=f"Auto-signing CAPTURE with initials {self.initials}")
                self.parent.signed_at = dt.datetime.now().isoformat()
            else:
                if self.run_idx == 0 and self.batch_found == False and self.batch_warned == False:
                    if not PopUp.question(self, Constants.app_title,
                                          "Batch Number not found!\nAn invalid Batch Number will lead to less accurate Analyze results.\n\n" +
                                          "Please confirm you entered the correct value and/or\ncheck for updates to your batch parameters resource file.\n\n" +
                                          "Are you sure you want to save this info?", False):
                        return False  # do not save, allow further changes, user doesn't want to save with invalid Batch Number
                    self.batch_warned = True

                # Check 'Run Info' form for unsaved changes.  Notify the user if they are about to
                # exit the form without saving edits.  The user may ignore this warning an
                # and close without saving edits.
                if self.unsaved_changes:
                    if self.signForm.isVisible():
                        self.signForm.hide()
                    self.signedInAs.setText(self.username)
                    self.signerInit.setText(
                        f"Initials: <b>{self.initials}</b>")
                    screen = QtWidgets.QDesktopWidget().availableGeometry()
                    left = int(
                        (screen.width() - self.signForm.sizeHint().width()) / 2) + 50
                    top = int(
                        (screen.height() - self.signForm.sizeHint().height()) / 2) - 50
                    self.signForm.move(left, top)
                    self.signForm.setVisible(True)
                    self.sign.setFocus()
                    Log.d(tag=TAG, msg="Saving Run Info, requesting signature.")
                else:
                    Log.d(tag=TAG, msg="Nothing to save, closing Run Info.")
                    self.close()  # nothing to save
                return False

        # If sign_do_not_ask attribute is checked, provide the latest session key to the
        # user to allow fo rmodification of the XML.
        if self.sign_do_not_ask.isChecked():
            session_key_path = os.path.join(
                Constants.user_profiles_path, "session.key")
            if os.path.exists(session_key_path):
                with open(session_key_path, 'r') as f:
                    session_key = f.readline()
                if not os.path.exists(Constants.auto_sign_key_path):
                    with open(Constants.auto_sign_key_path, 'w') as f:
                        f.write(session_key)

        # Update the run_name to the name in the 'Run name' text box of the
        # Run Info window, after removing leading and trailing whitespace.
        name_invalid = False
        self.run_name_changed = True if self.run_name != self.t_runname.text() else False
        self.run_name = self.t_runname.text()
        for invalidChar in Constants.invalidChars:
            if invalidChar in self.run_name:
                name_invalid = True
            self.run_name = self.run_name.replace(invalidChar, '')
        self.run_name = self.run_name.strip().replace(
            ' ', '_')  # word spaces -> underscores
        if name_invalid:
            if not PopUp.question(self, "Invalid Characters", f"Run name changed to:\n{self.run_name}\n\nDo you accept?"):
                return False  # allow further changes
        if self.run_name == self.t_runname.text():
            # run name was different, but after normalization they're the same again
            self.run_name_changed = False
        else:
            self.t_runname.setText(self.run_name)

        # Mode of operation for the XML file at the xml_path attribute.
        # If the path already exists, the action is modification of parameters
        # in which caase, the corresponding XML document is parsed for usage later while
        # the 'name' field in the 'run_info' header tag is updated to match the
        # parameterized name by the user.
        #
        # If the xml_path is new, the action is to CAPTURE i.e. write a new XML file
        # for a new run.  This mode of operation writes the entire run_info tag and all
        # child tags of run_info to a new XML document object.
        if secure_open.file_exists(self.xml_path):
            audit_action = "PARAMS"
            run = minidom.parse(self.xml_path)
            xml = run.documentElement
        else:
            audit_action = "CAPTURE"
            run = minidom.Document()
            xml = run.createElement('run_info')
            run.appendChild(xml)

            try:
                dev_name = FileStorage.DEV_get_active(self.run_idx + 1)
            except Exception as e:
                Log.e(
                    tag=TAG, msg=f"Unable to get active device for RUN_IDX ={self.run_idx}")
                dev_name = "UNKNOWN"
                Log.e(tag=TAG, msg=f"ERROR: {e}")

            # Set machine, device, name, and ruling field for top-level run_info tag.
            xml.setAttribute('machine', Architecture.get_os_name())
            xml.setAttribute('device', dev_name)
            xml.setAttribute('name', self.run_name)
            xml.setAttribute('ruling', self.run_ruling)

            # BUILD REMAINING XML DOCUMENT #
            metrics = run.createElement('metrics')
            xml.appendChild(metrics)

            try:
                if self.run_path.endswith(".zip"):
                    Log.e(tag=TAG, msg=f"ZIP file passed as 'run_path' incorrectly!")
                    raise Exception()

                if self.run_path.endswith(".csv"):
                    with secure_open(self.run_path, 'r', "capture") as file:
                        header_line = file.readline()
                        first_line = file.readline()
                        samples = 1
                        for last_line in file:
                            samples += 1

                        if isinstance(header_line, bytes):
                            header_line = header_line.decode()
                        if isinstance(first_line, bytes):
                            first_line = first_line.decode()
                        if isinstance(last_line, bytes):
                            last_line = last_line.decode()

                        first = first_line.split(',')
                        last = last_line.split(',')
                start = dt.datetime.strptime(
                    f"{first[0]} {first[1]}", "%Y-%m-%d %H:%M:%S").isoformat()
                stop = dt.datetime.strptime(
                    f"{last[0]} {last[1]}", "%Y-%m-%d %H:%M:%S").isoformat()
                duration = float(last[2])
                duration_units = "seconds"
                if duration > 60.0:
                    duration /= 60.0
                    duration_units = "minutes"
                samples = str(samples)
                est_start_time = self.parent.forecast_start_time
                est_end_time = self.parent.forecast_end_time
                if est_start_time < 0:
                    est_start_time = 0.0
                if est_end_time < 0:
                    est_end_time = 0.0
                Log.d(tag=TAG, msg=f"{start}, {stop}, {duration}, {samples}")

                # Get time of last cal - based on file timestamp
                cal_file_path = Constants.cvs_peakfrequencies_path
                cal_file_path = FileStorage.DEV_populate_path(
                    cal_file_path, self.run_idx + 1)
                timestamp = os.path.getmtime(cal_file_path)
                last_modified = dt.datetime.fromtimestamp(timestamp)
                cal_time = last_modified.isoformat().split('.')[0]

                metric0 = run.createElement('metric')
                metric0.setAttribute('name', 'calibrated')
                metric0.setAttribute('value', cal_time)
                metrics.appendChild(metric0)

                metric1 = run.createElement('metric')
                metric1.setAttribute('name', 'start')
                metric1.setAttribute('value', start)
                metrics.appendChild(metric1)

                metric2 = run.createElement('metric')
                metric2.setAttribute('name', 'stop')
                metric2.setAttribute('value', stop)
                metrics.appendChild(metric2)

                metric3 = run.createElement('metric')
                metric3.setAttribute('name', 'duration')
                metric3.setAttribute('value', f"{duration:2.4f}")
                metric3.setAttribute('units', duration_units)
                metrics.appendChild(metric3)

                metric4 = run.createElement('metric')
                metric4.setAttribute('name', 'samples')
                metric4.setAttribute('value', samples)
                metrics.appendChild(metric4)

                # ------------ Forecast start and end time ------------ #
                # metric5 = run.createElement('metric')
                # metric5.setAttribute('name', 'est_start_time')
                # metric5.setAttribute('value', est_start_time)
                # metrics.appendChild(metric5)

                # metric6 = run.createElement('metric')
                # metric6.setAttribute('name', 'est_end_time')
                # metric6.setAttribute('value', est_end_time)
                # metrics.appendChild(metric6)
                # ----------------------------------------------------- #
            except Exception as e:
                Log.e(
                    "Metrics Error: Failed to open/parse CSV file for XML file run info metrics.")
                # raise e

        # create or append new audits element
        try:
            audits = xml.getElementsByTagName('audits')[-1]
        except:
            audits = run.createElement('audits')
            xml.appendChild(audits)

        # create or append new params element
        recorded_at = self.parent.signed_at if self.parent.signature_required else dt.datetime.now().isoformat()
        params = run.createElement('params')
        params.setAttribute('recorded', recorded_at)
        xml.appendChild(params)

        param1 = run.createElement('param')
        param1.setAttribute('name', 'bioformulation')
        param1.setAttribute('value', str(self.b1.isChecked()))
        params.appendChild(param1)

        param_runname = run.createElement('param')
        param_runname.setAttribute('name', 'run_name')
        param_runname.setAttribute('value', self.t_runname.text())
        params.appendChild(param_runname)

        param_batch = run.createElement('param')
        param_batch.setAttribute('name', 'batch_number')
        param_batch.setAttribute('value', self.t_batch.text())
        param_batch.setAttribute('found', str(self.batch_found))
        params.appendChild(param_batch)

        param_notes = run.createElement('param')
        param_notes.setAttribute('name', 'notes')
        # we must escape new lines, double quotes are converted to '&quot;' automatically
        param_notes.setAttribute('value', self.notes.toPlainText()
                                 .replace('\n', '\\n')  # escape new lines
                                 .replace('"', "''"))  # escape double quotes
        param_notes.setAttribute(
            'source', 'single' if self.run_count == 1 else f"multi_{self.run_count}")
        params.appendChild(param_notes)

        try:
            if Constants.export_notes_to_txt_file:
                notes_path = os.path.join(
                    os.path.dirname(self.run_path), "notes.txt")
                notes_txt = self.notes.toPlainText()
                if notes_txt != self.notes.placeholderText() and len(notes_txt) > 0:
                    with open(notes_path, 'w') as f:
                        f.write(notes_txt)
                elif os.path.exists(notes_path):
                    os.remove(notes_path)
        except Exception as e:
            Log.e("ERROR:", e)

        if self.b2.isChecked():  # is NOT bioformulation
            param2 = run.createElement('param')
            param2.setAttribute('name', 'solvent')
            param2.setAttribute('value', self.t0.text())
            param2.setAttribute(
                'input', 'auto' if self.t0.text() in self.fluids else 'manual')
            params.appendChild(param2)

        if self.b1.isChecked():  # IS bioformulation
            param3 = run.createElement('param')
            param3.setAttribute('name', 'surfactant')
            param3.setAttribute('value', "{0:0.{1}f}".format(
                surfactant, self.validSurfactant.decimals()))
            param3.setAttribute('units', 'mg/mL')
            param3.setAttribute('input', 'manual')
            params.appendChild(param3)

            param4 = run.createElement('param')
            param4.setAttribute('name', 'concentration')
            param4.setAttribute('value', "{0:0.{1}f}".format(
                concentration, self.validConcentration.decimals()))
            param4.setAttribute('units', '%w')
            param4.setAttribute('input', 'manual')
            params.appendChild(param4)

            # new parameters for revamped run info

            param8 = run.createElement('param')
            param8.setAttribute('name', 'protein_type')
            param8.setAttribute('value', self.c10.currentText())
            params.appendChild(param8)

            param9 = run.createElement('param')
            param9.setAttribute('name', 'protein_concentration')
            param9.setAttribute('value', self.t12.text())
            param9.setAttribute('units', 'mg/mL')
            params.appendChild(param9)

            param14 = run.createElement('param')
            param14.setAttribute('name', 'buffer_type')
            param14.setAttribute('value', self.c13.currentText())
            params.appendChild(param14)

            param15 = run.createElement('param')
            param15.setAttribute('name', 'buffer_concentration')
            param15.setAttribute('value', self.t14.text())
            params.appendChild(param15)

            param10 = run.createElement('param')
            param10.setAttribute('name', 'surfactant_type')
            param10.setAttribute('value', self.c9.currentText())
            params.appendChild(param10)

            param11 = run.createElement('param')
            param11.setAttribute('name', 'surfactant_concentration')
            param11.setAttribute('value', self.t6.text())
            param11.setAttribute('units', '%w')
            params.appendChild(param11)

            param12 = run.createElement('param')
            param12.setAttribute('name', 'stabilizer_type')
            param12.setAttribute('value', self.c11.currentText())
            params.appendChild(param12)

            param13 = run.createElement('param')
            param13.setAttribute('name', 'stabilizer_concentration')
            param13.setAttribute('value', self.t8.text())
            param13.setAttribute('units', 'M')
            params.appendChild(param13)

        param5 = run.createElement('param')
        param5.setAttribute('name', 'surface_tension')
        param5.setAttribute('value', "{0:0.{1}f}".format(
            st, self.validSurfaceTension.decimals()))
        param5.setAttribute('units', 'mN/m')
        param5.setAttribute('input', 'manual' if manual_st else 'auto')
        params.appendChild(param5)

        param6 = run.createElement('param')
        param6.setAttribute('name', 'contact_angle')
        param6.setAttribute('value', "{0:0.{1}f}".format(
            ca, self.validContactAngle.decimals()))
        param6.setAttribute('units', 'degrees')
        param6.setAttribute('input', 'manual' if manual_ca else 'auto')
        params.appendChild(param6)

        param7 = run.createElement('param')
        param7.setAttribute('name', 'density')
        param7.setAttribute('value', "{0:0.{1}f}".format(
            density, self.validDensity.decimals()))
        param7.setAttribute('units', 'g/cm^3')
        param7.setAttribute('input', 'manual' if manual_dn else 'auto')
        params.appendChild(param7)

        # Add the fill_type parameter to the XML with options for number of channels
        # and if the value was auto generated or manually set.
        param14 = run.createElement('param')
        param14.setAttribute('name', 'fill_type')
        param14.setAttribute('value', str(num_channels))
        param14.setAttribute('input', 'manual' if manual_nc else 'auto')
        params.appendChild(param14)

        # add hashes for security and verification
        if not os.path.exists(self.xml_path):

            hash = hashlib.sha256()
            for name, value in xml.attributes.items():
                hash.update(name.encode())
                hash.update(value.encode())
            signature = hash.hexdigest()
            xml.setAttribute('signature', signature)

            hash = hashlib.sha256()
            for m in metrics.childNodes:
                for name, value in m.attributes.items():
                    hash.update(name.encode())
                    hash.update(value.encode())
            signature = hash.hexdigest()
            metrics.setAttribute('signature', signature)

        hash = hashlib.sha256()
        for p in params.childNodes:
            for name, value in p.attributes.items():
                hash.update(name.encode())
                hash.update(value.encode())
        signature = hash.hexdigest()
        params.setAttribute('signature', signature)

        if self.parent.signature_required:
            from QATCH.common.userProfiles import UserProfiles
            valid, infos = UserProfiles.session_info()
            if valid:
                Log.d(f"Found valid session: {infos}")
                username = infos[0]
                initials = infos[1]
                salt = UserProfiles.find(username, initials)[1][:-4]
                userrole = infos[2]
            else:
                Log.w(
                    f"Found invalid session: searching for user ({self.username}, {self.initials})")
                username = self.username
                initials = self.initials
                salt = UserProfiles.find(username, initials)[1][:-4]
                userrole = UserProfiles.get_user_info(f"{salt}.xml")[2]

            timestamp = self.parent.signed_at
            machine = Architecture.get_os_name()
            hash = hashlib.sha256()
            hash.update(salt.encode())  # aka 'profile'
            hash.update(audit_action.encode())
            hash.update(timestamp.encode())
            hash.update(machine.encode())
            hash.update(username.encode())
            hash.update(initials.encode())
            hash.update(userrole.encode())
            signature = hash.hexdigest()

            audit1 = run.createElement('audit')
            audit1.setAttribute('profile', salt)
            audit1.setAttribute('action', audit_action)
            audit1.setAttribute('recorded', timestamp)
            audit1.setAttribute('machine', machine)
            audit1.setAttribute('username', username)
            audit1.setAttribute('initials', initials)
            audit1.setAttribute('role', userrole)
            audit1.setAttribute('signature', signature)
            audits.appendChild(audit1)
        else:
            pass  # leave 'audits' block as empty

        if not self.post_run:
            # Get date from XML for secure file writing.
            metrics = xml.getElementsByTagName('metric')
            stop_value = None
            for metric in metrics:
                if metric.getAttribute('name') == 'stop':
                    stop_value = metric.getAttribute('value')
                    break
            stop_datetime = datetime.date.fromisoformat(
                stop_value.split('T')[0])
            # Update the run data files and directory to reflect changes made to
            # the run name in the RunInfo window.
            updated_name, new_name, old_name = self.update_run_name(
                self.xml_path, self.run_name, secure=True, date=stop_datetime)
            if not updated_name:
                Log.e(tag=TAG, msg="Could not update directory due path error.")
                # return False
            # os.makedirs(os.path.split(self.xml_path)[0], exist_ok=True)
            # secure_open(self.xml_path, 'w', "audit") as f:
            elif old_name != new_name:
                # self.run_name = new_name
                if xml.hasAttribute('name'):
                    xml.setAttribute('name', new_name)
                    hash = hashlib.sha256()
                    for name, value in xml.attributes.items():
                        hash.update(name.encode())
                        hash.update(value.encode())
                    signature = hash.hexdigest()
                    xml.setAttribute('signature', signature)
                    Log.d(
                        tag=TAG, msg=f"Updated 'name' field to: {new_name}")
                else:
                    Log.e(tag=TAG, msg="No 'name' field found.")

        with open(self.xml_path, 'w') as f:
            xml_str = run.toxml()  # .encode() #prettyxml(indent ="\t")
            f.write(xml_str)
            Log.i(f"Created XML file: {self.xml_path}")

        if self.q_recall.isEnabled():
            run = minidom.Document()
            xml = run.createElement('run_info')
            xml.setAttribute('name', 'recall')
            run.appendChild(xml)
            if self.q_recall.isChecked():  # remember for next time
                Log.i("Run info remembered for next time.")
            else:
                params = run.createElement('params')  # blank it
            xml.appendChild(params)
            os.makedirs(os.path.split(self.recall_xml)[0], exist_ok=True)
            # secure_open(self.recall_xml, 'w') as f:
            with open(self.recall_xml, 'w') as f:
                f.write(run.toxml())

        if not self.post_run:
            if updated_name:
                Log.d(tag=TAG, msg=f"Emitting {self.xml_path}")
                self.updated_run.emit(
                    self.xml_path, new_name, old_name, str(stop_datetime))
                self.updated_xml_path.emit(self.xml_path)

            if hasattr(self.parent, 'num_channels') and num_channels != self.parent.num_channels:
                Log.d("Number of fill channels changed by user in Run Info.")
                Log.w(
                    "Fill channels count changed! Re-run \"Predict\" to update points.")
                self.parent.num_channels = num_channels

        self.unsaved_changes = False
        self.close()
        return True

    def update_run_name(self, previous_xml_path: str, new_name: str, date: datetime.date, secure: bool = False):
        """
        Renames the the run directory and corresponding run data files to the updated name parameterized
        by the RunInfo window.

        TODO: Discuss how to have this match in the XML file.  Currently, the XML can contain an invalid
        file name and will display in the dropdown.  I can just remove this if its more of a nuisance than
        of benefit. [AJR: If it's not readable data or is 'invalid' and cannot be processed, remove it]

        Args:
            previous_xml_path (str): The file path to the XML file whose directory is to be renamed.
            new_name (str): The new name for the directory and the files within it.
            date (datetime.date): The stop date of the run to rename.
            secure (bool): (Optional) Flag for secure directory creation. Set to False by default.

        Returns:
            str: The name of the new directory if the operation was successful, None otherwise.
            str: The name of the renamed file without the any additional tags such as _3rd, _tec, _lower, etc.
            str: The name of the old base directory that is being renamed

        Raises:
            ValueError: If the `previous_xml_path` is not a valid file or is not within the base directory.
            ValueError: If the `new_name` is invalid (contains special characters or path traversal).
            PermissionError: If the function is denied permission to access directories or rename files.
            OSError: If any OS-level error occurs during renaming or file operations.
        """
        import pyzipper
        from io import BytesIO

        def rename_file(file_name: str, new_name: str):
            """
            A helper function to rename files based on their suffixes.

            Args:
                file_name (str): The file name prefix to update each type of run file to.
                new_name (str): The new prefix for the file.

            Returns:
                tuple: The name of the file with a new prefix and a flag indicating if it is XML.
                    If the file does not have a matching suffix, (None, False) is returned.
            """
            xml_suffixes = ['_3rd.xml', '.xml']
            csv_suffixes = [
                '_3rd_cal.csv', '_cal.csv', '_3rd_poi.csv', '_poi.csv',
                '_3rd_tec.csv', '_tec.csv', '_3rd.csv', '_3rd_lower.csv', '_lower.csv', '.csv'
            ]
            crc_suffixes = [
                '_3rd_tec.crc', '_tec.crc', '_3rd.crc', '_3rd_lower.crc', '_lower.crc'
            ]

            for suffix in xml_suffixes:
                if file_name.endswith(suffix):
                    return f"{new_name}{suffix}", True

            for suffix in csv_suffixes + crc_suffixes:
                if file_name.endswith(suffix):
                    return f"{new_name}{suffix}", False

            return None, False

        try:
            # If file name is the same, it is fine to resave.  If the path already exists elsewhere,
            # return with an error.
            if not previous_xml_path.endswith(new_name) and not os.path.isfile(previous_xml_path):
                Log.e(
                    tag=TAG, msg=f"Previous XML path {previous_xml_path} does not exist or is not a file.")
                return None, None, None

            # Prevents file creation outside of logged_data directory.
            # if not os.path.abspath(previous_xml_path).startswith(Constants.log_export_path):
            #     Log.e(
            #         tag=TAG, msg=f"Operation outside of secure directory is not allowed.")
            #     return False

            parent_dir = os.path.dirname(previous_xml_path)
            grandparent_dir = os.path.dirname(parent_dir)

            # TODO: Modify regex to fit naming convention for files. [AJR: Is this still TODO?]
            # Validate new_name for security (e.g., no special characters, no path traversal)
            # This could work: r'^[\w\s\-.]+$' or ^[\w\-.]+(\s[\w\-.]+)*$ for trailing and leading spaces
            # and not re.match(r'^[\w\-.]+(\s[\w\-.]+)*$', new_name):
            if secure:
                name_invalid = False
                for invalidChar in Constants.invalidChars:
                    if invalidChar in new_name:
                        name_invalid = True
                    new_name = new_name.replace(invalidChar, '')
                new_name = new_name.strip().replace(' ', '_')  # word spaces -> underscores
                if name_invalid:
                    Log.w(
                        tag=TAG, msg=f"Modified run name to \"{new_name}\". Invalid characters: {Constants.invalidChars}")
                # return None, None, None

            # Form new directory path and validate it is within grandparent_dir to avoid path traversal
            new_dir = os.path.join(grandparent_dir, new_name)
            if secure and not os.path.abspath(new_dir).startswith(os.path.abspath(grandparent_dir)):
                Log.w(
                    tag=TAG, msg=f"Path traversal attempt detected in new name: {new_name}")
                return None, None, None
             # Symlink avoidance by rejecting symbolic links in directory hierarchy
            # NOTE: The PYCODE releases do actually have a hard symbolic link for the 'logged_data' folder
            # We can confirm that this rejection of symlinks does not break renaming for PYCODE builds; however,
            # it may when the naming structure folder config for the logged_data folder is only one dir deep.
            if secure and any(os.path.islink(d) for d in [previous_xml_path, parent_dir, grandparent_dir]):
                Log.e(
                    tag=TAG, msg="Symbolic links are not allowed in the directory path.")
                return None, None, None

            # Check if new directory path is valid
            new_dir = os.path.join(grandparent_dir, new_name)
            old_name = os.path.basename(parent_dir)
            # Solves case change issue in run names.
            case_change = old_name.lower() == new_name.lower() and old_name != new_name
            if os.path.exists(new_dir) and not case_change:
                Log.i(
                    tag=TAG, msg=f"Path {new_dir} already exists, no action taken.")
                return None, None, None

            os.rename(parent_dir, new_dir)
            Log.i(tag=TAG, msg=f"Updating directory {parent_dir} to {new_dir}")

            # List files in the new directory
            try:
                files = os.listdir(new_dir)
            except PermissionError:
                Log.e(
                    tag=TAG, msg=f"Permission denied: Cannot access {new_dir}")
                return None, None, None

            is_xml = False
            # Rename each file to match the new directory name
            for root, _, files in os.walk(new_dir):
                for file_name in files:
                    file_path = os.path.join(root, file_name)

                    # Set a temporary directory to write renamed files too.  This directory is
                    # renamed to the capture.zip directory at the end of the process.
                    temp_dir = None
                    if file_name.endswith('capture.zip'):

                        # Get a secure list of the contents of a capture.zip archive using the
                        # path to the zip folder.
                        capture_contents = secure_open.get_namelist(
                            zip_path=file_path)
                        # Set the path of the temporary archive to write renamed files to.
                        temp_dir = os.path.join(root, "~capture.zip")

                        # For each file in the secure archive, read it as bytes and then write
                        # the bytes to the temporary zip directory.
                        for capture_file in capture_contents:
                            capture_path = os.path.join(
                                root, capture_file)
                            with secure_open(capture_path, "r", "capture") as f:
                                f_bytes = BytesIO(f.read())

                                # Create each new file name for the contents of the secure
                                # archive directory.
                                new_file_name, _ = rename_file(
                                    capture_file, new_name)

                            # For each file, securely write it to the temporary location using the
                            # date of the stop time along with the name of the renamed run.
                            with pyzipper.AESZipFile(temp_dir, 'a',
                                                     compression=pyzipper.ZIP_DEFLATED,
                                                     allowZip64=True,
                                                     encryption=pyzipper.WZ_AES) as zf:
                                friendly_name = f"{new_name} ({date})"
                                zf.comment = friendly_name.encode()
                                enabled, _, _ = UserProfiles.checkDevMode()

                                # TODO: If already password protected, dev mode should not allow the decryption
                                # and rewriting of already encrypted files.
                                if UserProfiles.count() > 0 and enabled == False:
                                    # create a protected archive
                                    zf.setpassword(hashlib.sha256(
                                        zf.comment).hexdigest().encode())
                                else:
                                    zf.setencryption(None)
                                    if enabled:
                                        Log.d(
                                            tag=TAG, msg="Developer Mode is ENABLED - NOT encrypting ZIP file")

                                # Write the bytes string to the new zip file location.
                                zf.writestr(
                                    zinfo_or_arcname=new_file_name, data=f_bytes.read())

                        # Remove the old capture.zip archive and rename the temporary capture.zip archive to capture.zip.
                        os.remove(file_path)
                        os.rename(temp_dir, file_path)
                    else:
                        # Process the rest of the files in the renamed direcotry.
                        new_file_name, is_xml = rename_file(
                            file_name, new_name)
                        if new_file_name:
                            new_file_path = os.path.join(
                                root, new_file_name)
                            if os.path.exists(new_file_path):
                                Log.d(
                                    tag=TAG, msg=f"File {new_file_path} already exists. Skipping rename for {new_file_path}")
                                continue
                            os.rename(file_path, new_file_path)
                            Log.d(
                                tag=TAG, msg=f"Renamed file {file_path} to {new_file_path}")
                            if is_xml:
                                self.xml_path = new_file_path
                                self.recall_xml = new_file_path
                                xml_name_to_write = os.path.splitext(
                                    os.path.basename(new_file_path))[0]
            return xml_name_to_write, new_name, old_name
        except Exception as e:
            Log.e(tag=TAG, msg=f"An unexpected error occurred: {e}")
            return None, None, None

    def closeEvent(self, event):
        if self.unsaved_changes:
            res = PopUp.question(
                self, Constants.app_title, "You have unsaved changes!\n\nAre you sure you want to close this window?", False)
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
            if self.run_idx == 0 and self.batch_found == False and self.batch_warned == False:  # invalid Batch Number
                if not PopUp.question(self, Constants.app_title,
                                      "Batch Number not found!\nAn invalid Batch Number will lead to less accurate Analyze results.\n\n" +
                                      "Please confirm you entered the correct value and/or\ncheck for updates to your batch parameters resource file.\n\n" +
                                      "Are you sure you want to close without updating this info?", False):
                    event.ignore()  # do not close, allow further changes, user doesn't want to close with invalid Batch Number
                    return
                self.batch_warned = True
            # pre-hide so callers connected to 'finished' signal can see the "hidden" widget
            self.setVisible(False)
            self.finished.emit()
