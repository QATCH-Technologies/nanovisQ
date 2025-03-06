from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserProfiles
from QATCH.core.constants import Constants
from QATCH.ui.popUp import PopUp
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDesktopWidget
from threading import Thread
from xml.dom import minidom
import time
import datetime
import send2trash
import shutil
import subprocess
import os
import zipfile

TAG1 = "[Export]"
TAG2 = "[Import]"

###############################################################################
# Export dialog
###############################################################################


class Ui_Export(QtWidgets.QWidget):
    usb_add = QtCore.pyqtSignal()
    usb_remove = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(str, int, str, int)
    freeze_gui = QtCore.pyqtSignal(bool)
    confirmed = False
    exported = False
    do_close = False
    drive = None
    source_subfolder = ""
    chk1 = False
    chk2 = False

    def __init__(self, type="item", parent=None):
        super(Ui_Export, self).__init__(parent)

        USE_FULLSCREEN = (QDesktopWidget().availableGeometry().width() == 2880)
        self.setMinimumSize(500, 500)
        # self.move(500, 50)

        self.layout = QtWidgets.QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QtWidgets.QTabWidget()
        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()
        self.tabAdv = QtWidgets.QWidget()
        self.tab3 = QtWidgets.QWidget()
        # self.tabs.resize(300,200)

        # Add tabs
        self.tabs.addTab(self.tab1, "Import")
        self.tabs.addTab(self.tab2, "Export")
        self.tabs.addTab(self.tabAdv, "Advanced")
        self.tabs.addTab(self.tab3, "History")

        self.lbl1 = QtWidgets.QLabel("Import from...")
        self.btn6 = QtWidgets.QPushButton("Folder")
        self.btn6.pressed.connect(self.select_import_folder)
        self.btn8 = QtWidgets.QPushButton("ZIP")
        self.btn8.pressed.connect(self.select_file_source)
        self.btn7 = QtWidgets.QLineEdit("[NONE]")
        self.btn7.setReadOnly(True)

        layout_h10 = QtWidgets.QHBoxLayout()
        layout_h10.addWidget(self.lbl1)
        layout_h10.addWidget(self.btn6)
        layout_h10.addWidget(self.btn8)

        self.existingImport = QtWidgets.QLabel("Existing files:")
        self.doOverwrite2 = QtWidgets.QCheckBox("Replace")
        self.doMerge2 = QtWidgets.QCheckBox("Merge")
        self.doSkip2 = QtWidgets.QCheckBox("Skip")
        self.doMerge2.setChecked(True)

        self.btnGroup4 = QtWidgets.QButtonGroup()
        self.btnGroup4.addButton(self.doMerge2, 2)
        self.btnGroup4.addButton(self.doOverwrite2, 1)
        self.btnGroup4.addButton(self.doSkip2, 3)
        self.btnGroup4.setExclusive(True)

        layout_h12 = QtWidgets.QHBoxLayout()
        layout_h12.addWidget(self.existingImport)
        layout_h12.addWidget(self.doMerge2)
        layout_h12.addWidget(self.doOverwrite2)
        layout_h12.addWidget(self.doSkip2)

        layout_v6 = QtWidgets.QVBoxLayout()
        layout_v6.addLayout(layout_h10)
        layout_v6.addWidget(self.btn7)
        layout_v6.addLayout(layout_h12)

        self.groupbox4 = QtWidgets.QGroupBox("Import Location")
        self.groupbox4.setCheckable(False)
        self.groupbox4.setChecked(False)
        self.groupbox4.setLayout(layout_v6)

        self.archiveInfo = QtWidgets.QTextEdit()
        self.archiveInfo.setReadOnly(True)
        self.archiveInfo.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)

        layout_v5 = QtWidgets.QVBoxLayout()
        layout_v5.addWidget(self.archiveInfo)

        self.groupbox5 = QtWidgets.QGroupBox("Data to Import")
        self.groupbox5.setCheckable(True)
        self.groupbox5.setChecked(False)
        self.groupbox5.setLayout(layout_v5)
        self.checkChanged5(self.groupbox5.isChecked())

        self.tb1 = QtWidgets.QLabel()
        self.tb1.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.tb1.setAlignment(QtCore.Qt.AlignCenter)
        self.tb1.setFixedHeight(50)

        self.pb1 = QtWidgets.QLabel()
        self.pb1.setAlignment(QtCore.Qt.AlignCenter)

        self.importNow = QtWidgets.QPushButton("Import")
        self.importNow.pressed.connect(self.doImport)
        self.importCancel = QtWidgets.QPushButton("Cancel")
        self.importCancel.pressed.connect(self.cancel)
        self.importNow.setEnabled(False)
        self.importCancel.setEnabled(False)

        layout_h11 = QtWidgets.QHBoxLayout()
        layout_h11.addWidget(self.importNow)
        layout_h11.addWidget(self.importCancel)

        layout_v1 = QtWidgets.QVBoxLayout()
        layout_v1.addWidget(self.groupbox4)
        layout_v1.addWidget(self.groupbox5)
        # layout_v1.addWidget(self.groupbox1)
        layout_v1.addWidget(self.tb1)
        layout_v1.addWidget(self.pb1)
        layout_v1.addLayout(layout_h11)

        self.tab1.setLayout(layout_v1)

        '''
        # Create first tab
        self.tab1.layout = QtWidgets.QVBoxLayout(self)
        self.pushButton1 = QtWidgets.QPushButton("PyQt5 button")
        self.tab1.layout.addWidget(self.pushButton1)
        self.tab1.setLayout(self.tab1.layout)
        '''

        self.tb = QtWidgets.QLabel()
        self.tb.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.tb.setAlignment(QtCore.Qt.AlignCenter)
        self.tb.setFixedHeight(50)

        self.pb = QtWidgets.QLabel()
        self.pb.setAlignment(QtCore.Qt.AlignCenter)

        self.btn4 = QtWidgets.QPushButton("Export to...")
        self.btn4.pressed.connect(self.select_folder_target)
        self.btn5 = QtWidgets.QLineEdit("[NONE]")
        self.btn5.setReadOnly(True)

        self.btn1 = QtWidgets.QPushButton("&Detect USB")
        self.btn1.pressed.connect(self.detect)
        self.btn2 = QtWidgets.QPushButton("E&ject USB")
        self.btn2.pressed.connect(self.eject)
        self.btn3 = QtWidgets.QPushButton("E&rase Local Data")
        self.btn3.pressed.connect(self.erase)

        if USE_FULLSCREEN:
            self.tb.setFixedHeight(100)
            self.pb.setFixedHeight(25)
            self.btn1.setFixedHeight(50)
            self.btn2.setFixedHeight(50)
            self.btn3.setFixedHeight(50)
            self.btn4.setFixedHeight(50)
            self.btn5.setFixedHeight(50)

        layout_h1 = QtWidgets.QHBoxLayout()
        layout_h1.addWidget(self.btn1)
        layout_h1.addWidget(self.btn2)

        self.groupbox1 = QtWidgets.QGroupBox("Export to USB")
        self.groupbox1.setCheckable(True)
        self.groupbox1.setChecked(False)
        self.groupbox1.setLayout(layout_h1)

        layout_h2 = QtWidgets.QHBoxLayout()
        layout_h2.addWidget(self.btn4)
        layout_h2.addWidget(self.btn5)

        self.groupbox2 = QtWidgets.QGroupBox("Export to Folder")
        self.groupbox2.setCheckable(True)
        self.groupbox2.setChecked(False)
        self.groupbox2.setLayout(layout_h2)

        self.exportAsZIP = QtWidgets.QCheckBox("ZIP Archive")
        self.exportAsFolder = QtWidgets.QCheckBox("Folder")
        self.dateFilter = QtWidgets.QLabel("Export by date:")
        self.filterOff = QtWidgets.QCheckBox("All Dates")
        self.filterToday = QtWidgets.QCheckBox("Today")
        self.filterLastXDays = QtWidgets.QCheckBox("Last:")
        self.filterLastXDays.setFixedWidth(
            self.filterLastXDays.sizeHint().width())
        self.filterNumDays = QtWidgets.QLineEdit("7")
        self.filterNumDays.setAlignment(QtCore.Qt.AlignCenter)
        self.filterValid = QtGui.QIntValidator(1, 31)
        self.filterNumDays.setValidator(self.filterValid)
        self.filterNumDays.setFixedWidth(25)
        self.filterUnits = QtWidgets.QComboBox()
        self.filterUnits.addItems(["Hours", "Days", "Weeks"])
        self.filterUnits.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self.filterUnits.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToContents)
        self.filterUnits.setFixedWidth(self.filterUnits.sizeHint().width())
        self.filterUnits.setCurrentText("Days")
        self.filterOff.setChecked(True)
        self.existingExport = QtWidgets.QLabel("Existing files:")
        self.doOverwrite = QtWidgets.QCheckBox("Replace")
        self.doMerge = QtWidgets.QCheckBox("Merge")
        self.doSkip = QtWidgets.QCheckBox("Skip")
        self.doMerge.setChecked(True)

        self.btnGroup1 = QtWidgets.QButtonGroup()
        self.btnGroup1.addButton(self.exportAsFolder)
        self.btnGroup1.addButton(self.exportAsZIP)
        self.btnGroup1.setExclusive(True)
        self.btnGroup5 = QtWidgets.QButtonGroup()
        self.btnGroup5.addButton(self.filterOff, 0)
        self.btnGroup5.addButton(self.filterToday, 1)
        self.btnGroup5.addButton(self.filterLastXDays, 2)
        self.btnGroup5.setExclusive(True)
        self.btnGroup2 = QtWidgets.QButtonGroup()
        self.btnGroup2.addButton(self.doMerge, 2)
        self.btnGroup2.addButton(self.doOverwrite, 1)
        self.btnGroup2.addButton(self.doSkip, 3)
        self.btnGroup2.setExclusive(True)

        layout_h4 = QtWidgets.QHBoxLayout()
        layout_h4.addWidget(self.exportAsFolder)
        layout_h4.addWidget(self.exportAsZIP)

        layout_filter = QtWidgets.QHBoxLayout()
        layout_filter.addWidget(self.dateFilter, 6)
        layout_filter.addWidget(self.filterOff, 6)
        layout_filter.addWidget(self.filterToday, 6)
        layout_filter.addWidget(self.filterLastXDays, 1)
        layout_filter.addWidget(self.filterNumDays, 1)
        layout_filter.addWidget(self.filterUnits, 1)

        layout_h7 = QtWidgets.QHBoxLayout()
        layout_h7.addWidget(self.existingExport)
        layout_h7.addWidget(self.doMerge)
        layout_h7.addWidget(self.doOverwrite)
        layout_h7.addWidget(self.doSkip)

        self.exportAll = QtWidgets.QCheckBox("All Runs")
        self.selection = QtWidgets.QCheckBox("Selection:")
        self.selectRun = QtWidgets.QPushButton("[ALL]")
        self.exportAll.setChecked(True)
        # self.selectRun.setEnabled(False)

        self.btnGroup3 = QtWidgets.QButtonGroup()
        self.btnGroup3.addButton(self.exportAll)
        self.btnGroup3.addButton(self.selection)
        self.btnGroup3.setExclusive(True)

        layout_h5 = QtWidgets.QHBoxLayout()
        layout_h5.addWidget(self.exportAll)
        layout_h6 = QtWidgets.QHBoxLayout()
        layout_h6.addWidget(self.selection)
        layout_h6.addWidget(self.selectRun)
        layout_h5.addLayout(layout_h6)

        self.exportNameChk = QtWidgets.QLabel("Export name:")
        self.exportNameTxt = QtWidgets.QLineEdit()
        self.exportNameTxt.setAlignment(QtCore.Qt.AlignCenter)
        self.exportUnnamed = QtWidgets.QCheckBox("Include \"_unnamed\" runs")
        self.exportNoName = QtWidgets.QCheckBox("Copy directly to folder")

        layout_h9 = QtWidgets.QHBoxLayout()
        layout_h9.addWidget(self.exportNameChk)
        layout_h9.addWidget(self.exportNameTxt)
        layout_h9.addWidget(self.exportUnnamed)

        layout_v4 = QtWidgets.QVBoxLayout()
        layout_v4.addLayout(layout_h4)
        layout_v4.addLayout(layout_h5)
        layout_v4.addLayout(layout_h9)
        layout_v4.addLayout(layout_filter)
        layout_v4.addLayout(layout_h7)

        # use grid layout (it's cleaner)
        exportGridLayout = QtWidgets.QGridLayout()
        # row 1: export selection
        exportGridLayout.addWidget(QtWidgets.QLabel("Export:"), 1, 1, 1, 1)
        exportGridLayout.addWidget(self.exportAll, 1, 2, 1, 1)
        exportGridLayout.addWidget(self.selection, 1, 3, 1, 1)
        exportGridLayout.addWidget(self.selectRun, 1, 4, 1, 3)
        # row 2: export as
        exportGridLayout.addWidget(QtWidgets.QLabel("Export as:"), 2, 1, 1, 1)
        exportGridLayout.addWidget(self.exportAsZIP, 2, 2, 1, 1)
        # row, col, rspan, cspan
        exportGridLayout.addWidget(self.exportAsFolder, 2, 3, 1, 1)
        exportGridLayout.addWidget(self.exportUnnamed, 2, 4, 1, 3)
        # row 3: export name
        exportGridLayout.addWidget(self.exportNameChk, 3, 1, 1, 1)
        exportGridLayout.addWidget(self.exportNameTxt, 3, 2, 1, 2)
        exportGridLayout.addWidget(self.exportNoName, 3, 4, 1, 3)
        # row 4: Export by date
        exportGridLayout.addWidget(self.dateFilter, 4, 1, 1, 1)
        exportGridLayout.addWidget(self.filterOff, 4, 2, 1, 1)
        exportGridLayout.addWidget(self.filterToday, 4, 3, 1, 1)
        exportGridLayout.addWidget(self.filterLastXDays, 4, 4, 1, 1)
        exportGridLayout.addWidget(self.filterNumDays, 4, 5, 1, 1)
        exportGridLayout.addWidget(self.filterUnits, 4, 6, 1, 1)
        # row 5: existing files
        exportGridLayout.addWidget(self.existingExport, 5, 1, 1, 1)
        exportGridLayout.addWidget(self.doMerge, 5, 2, 1, 1)
        exportGridLayout.addWidget(self.doOverwrite, 5, 3, 1, 1)
        exportGridLayout.addWidget(self.doSkip, 5, 4, 1, 3)

        self.groupbox3 = QtWidgets.QGroupBox("Export Settings")
        self.groupbox3.setCheckable(False)
        self.groupbox3.setChecked(False)
        self.groupbox3.setLayout(exportGridLayout)

        layout_h3 = QtWidgets.QVBoxLayout()
        erase_notice = QtWidgets.QLabel("<h1 style='color: #FF0000;'>WARNING</h1><br/>" +
                                        "<br/>This operation will erase all locally logged data from this machine.<br/>" +
                                        "<br/>NOTE: Erased runs can be recovered from the Recycle Bin (if applicable).<br/>" +
                                        "<br/><b>To fully erase, empty your Recycle Bin too.</b>")
        erase_notice.setAlignment(QtCore.Qt.AlignCenter)
        layout_h3.addWidget(erase_notice)
        layout_h3.addWidget(self.btn3)

        '''
        groupbox3 = QtWidgets.QGroupBox("Erase Local Data")
        # groupbox3.setCheckable(True)
        # groupbox3.setChecked(False)
        groupbox3.setLayout(layout_h3)
        '''

        groupbox3 = QtWidgets.QTabWidget()
        tab4 = QtWidgets.QWidget()
        tab4.setLayout(layout_h3)
        groupbox3.addTab(tab4, "Erase Local Data")
        self.tabAdv.setLayout(layout_h3)

        self.exportNow = QtWidgets.QPushButton("Export")
        self.exportNow.pressed.connect(self.export)
        self.exportCancel = QtWidgets.QPushButton("Cancel")
        self.exportCancel.pressed.connect(self.cancel)

        layout_h8 = QtWidgets.QHBoxLayout()
        layout_h8.addWidget(self.exportNow)
        layout_h8.addWidget(self.exportCancel)

        layout_v = QtWidgets.QVBoxLayout()
        layout_v.addWidget(self.groupbox3)
        layout_v.addWidget(self.groupbox2)
        layout_v.addWidget(self.groupbox1)
        layout_v.addWidget(self.tb)
        layout_v.addWidget(self.pb)
        layout_v.addLayout(layout_h8)

        self.tab2.setLayout(layout_v)

        '''
        # making widget resizable
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        self.scroll.setWidget(content)

        # vertical box layout
        lay = QtWidgets.QVBoxLayout()
        content.setLayout(lay)

        # creating label
        self.label = QtWidgets.QLabel()

        # setting alignment to the text
        self.label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        # making label multi-line
        self.label.setWordWrap(True)

        # adding label to the layout
        lay.addWidget(self.label)

        self.history = self.scroll
        layout_v3 = QtWidgets.QVBoxLayout()
        layout_v3.addWidget(self.history)

        self.tab3.setLayout(layout_v3)
        '''

        self.history = QtWidgets.QTextEdit()
        self.history.setReadOnly(True)
        self.history.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)

        self.clearAllHistory = QtWidgets.QPushButton("Clear All History")
        self.clearAllHistory.pressed.connect(self.do_clearAllHistory)

        layout_v3 = QtWidgets.QVBoxLayout()
        layout_v3.addWidget(self.history)
        layout_v3.addWidget(self.clearAllHistory)
        self.tab3.setLayout(layout_v3)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        # self.layout.addWidget(groupbox3)
        self.setLayout(self.layout)
        self.setWindowTitle("Import/Export Data")

        self.usb_add.connect(self.ui_add)
        self.usb_remove.connect(self.ui_remove)
        self.progress.connect(self.setProgress)
        self.freeze_gui.connect(self.freezeGUI)
        self.tabs.currentChanged.connect(self.tabChanged)
        self.groupbox1.clicked.connect(self.checkChanged1)
        self.groupbox2.clicked.connect(self.checkChanged2)
        self.selection.stateChanged.connect(self.selectChanged)
        self.exportAsZIP.stateChanged.connect(self.exportChanged)
        self.selectRun.pressed.connect(self.select_folder_source)
        self.groupbox5.clicked.connect(self.checkChanged5)
        self.exportNoName.stateChanged.connect(self.noNameChanged)

        self.exportAsZIP.setChecked(True)  # emit signal now that it's set

    def noNameChanged(self, arg):
        self.generateExportName()

    def showNormal(self, tab_idx=0):
        super(Ui_Export, self).hide()
        super(Ui_Export, self).showNormal()
        self.resize(self.minimumSize())

        self.tabs.setCurrentIndex(tab_idx)
        self.generateExportName()

        self.exported = False
        self.do_close = False
        self.drive = None

        self.progress.emit(
            "<b>Insert USB drive...</b><br/>No USB drives detected.", 0, 'b', 0)
        self.progress.emit(
            "<b>Select an input location...</b><br/>No import selected.", 0, 'b', 1)
        self.freeze_gui.emit(True)  # Enable Erase only

        self.main = Thread(target=self.mainTask)
        self.main.start()

    def do_clearAllHistory(self):
        history_path = os.path.join(
            os.getcwd(), Constants.log_export_path, "export_history.log")
        send2trash.send2trash(history_path)
        self.tabChanged(2)

    def tabChanged(self, idx):
        if idx == 2:
            history_path = os.path.join(
                os.getcwd(), Constants.log_export_path, "export_history.log")
            if os.path.exists(history_path):
                self.clearAllHistory.setEnabled(True)
                with open(history_path, 'r') as f:
                    self.history.setText(f.read())
            else:
                self.clearAllHistory.setEnabled(False)
                self.history.setText("No import/export history to show.")

    def checkChanged1(self, chk):
        # Log.d(f"group1 clicked! {chk}")
        self.chk1 = chk
        self.chk2 = False
        if chk and self.groupbox2.isChecked():
            self.groupbox2.setChecked(False)
        self.freezeGUI(True)

    def checkChanged2(self, chk):
        self.chk1 = False
        self.chk2 = chk
        # Log.d(f"group2 clicked! {chk}")
        if chk and self.groupbox1.isChecked():
            self.groupbox1.setChecked(False)
        if chk:
            self.drive = self.btn5.text(
            ) if self.btn5.text() != "[NONE]" else None
        self.freezeGUI(True)

    def checkChanged5(self, chk):
        self.archiveInfo.setEnabled(True)
        if self.groupbox5.isChecked():
            self.groupbox5.setTitle("Data to Import (View: Folders and Files)")
        else:
            self.groupbox5.setTitle("Data to Import (View: Folders only)")
        self.generateImportDescription()

    def selectChanged(self, arg):
        if not self.selection.isChecked():
            self.selectRun.setText("[ALL]")
            self.source_subfolder = ""
            self.generateExportName()
        # if self.selection.isChecked():
            # self.selectRun.setEnabled(True)
        # else:
            # self.selectRun.setEnabled(False)

    def generateImportDescription(self):
        try:
            path_to_import = self.btn7.text()
            if path_to_import == "[NONE]":
                return
            self.importNow.setEnabled(False)
            self.progress.emit(
                "<b>Generating archive info...</b><br/>This may take a few seconds.", 0, 'b', 1)
            path_split = os.path.split(path_to_import)
            # Log.d(f"path = {path_to_import}")
            show_files = self.groupbox5.isChecked()
            if path_split[1].count('.') and path_split[1][-4] == '.':
                # ZIP archive
                with zipfile.ZipFile(path_to_import, 'r') as f:
                    names = f.namelist()
                    names.sort()
                    if names[0].find(Constants.log_export_path) == -1:
                        Log.d(
                            "Archive does not contain \"logged_data\" folder. Must parse XMLs for relative path reconstruction.")
                    to_show = []
                    for item in names:
                        split = os.path.split(item)
                        is_folder = (split[1] == '')
                        if is_folder:
                            split = os.path.split(split[0])
                        else:
                            item = '/' + item
                        if show_files or is_folder:
                            level = item.count("/") - 1
                            tabwidth = 2
                            indent = ' ' * tabwidth * (level)
                            dash = '- '
                            formatted_item = '{}{}{}'.format(
                                indent, dash, split[1])
                            to_show.append(formatted_item)
                    self.archiveInfo.setText("\n".join(to_show))
            else:
                # folder
                test1 = path_to_import.find(Constants.log_export_path) >= 0
                test2 = os.path.exists(os.path.join(
                    path_to_import, Constants.log_export_path))
                # Log.d(f"tests = {test1}, {test2}")
                self.archiveInfo.setText(
                    self.list_files(path_to_import, show_files))
                if test1 or test2:
                    pass
                else:
                    Log.d(
                        "Archive does not contain \"logged_data\" folder. Must parse XMLs for relative path reconstruction.")
            self.progress.emit(
                "<b>Archive info generated!</b><br/>Ready to import.", 100, 'b', 1)
            self.importNow.setEnabled(True)
        except Exception as e:
            self.progress.emit(
                "<b>Error reading archive info!</b><br/>This archive might be corrupt.", 100, 'r', 1)
            self.archiveInfo.setText("Error reading archive: {}".format(e))
            # Log.d(e)
            # raise e

    def list_files(self, startpath, option, tree="", level=0):
        # Log.d(f"sp = {startpath}")
        for root, dirs, files in os.walk(startpath):
            # Log.d(f"root = {root}, dirs = {dirs}, files = {files}")
            # level = root.replace(startpath, '').count(os.sep)
            tabwidth = 2
            indent = ' ' * tabwidth * (level)
            dash = '- '  # * (level)
            line = '{}{}{}\n'.format(indent, dash, os.path.basename(root))
            # Log.d(line) #, end='')
            tree += line
            for dir in dirs:
                tree = self.list_files(os.path.join(
                    startpath, dir), option, tree, level+1)
            if option:
                subindent = ' ' * tabwidth * (level + 1)
                # subdash = '-' * (level + 1)
                for f in files:
                    subline = '{}{}{}\n'.format(subindent, dash, f)
                    # Log.d(subline) #, end='')
                    tree += subline
            if level == 0:
                tree = tree.rstrip()
            return tree

    def generateExportName(self):
        _, selected_folder = os.path.split(self.source_subfolder)
        default_filename = str(datetime.datetime.now()).split(' ')[0].replace(
            ":", "").replace("-", "").replace(" ", "_") + "_QATCH_EXPORT"
        if len(selected_folder) > 0:
            default_filename = selected_folder
        enabled = self.exportNoName.isChecked() == False
        self.exportNameTxt.setEnabled(enabled)
        self.exportNameTxt.setText(default_filename if enabled else "")

    def exportChanged(self, arg):
        if self.exportAsZIP.isChecked():
            self.exportNoName.setEnabled(False)
            if self.exportNoName.isChecked():
                self.exportNoName.setChecked(False)
        else:
            self.exportNoName.setEnabled(True)

    def freezeGUI(self, enable):
        if enable:
            self.ui_toggle(not self.drive == None)
        else:
            self.ui_toggle(False)
        # Log.d(f"en={enable} chk1={self.chk1} chk2={self.chk2} drive='{self.drive}'")
        if enable and (self.chk1 or self.chk2) and (self.drive != None):
            self.exportNow.setEnabled(True)
            self.exportCancel.setEnabled(False)
        else:
            self.exportNow.setEnabled(False)
            self.exportCancel.setEnabled(not enable)
        self.groupbox1.setEnabled(enable)
        self.groupbox2.setEnabled(enable)
        self.groupbox3.setEnabled(enable)
        self.archiveInfo.setEnabled(True)
        self.btn3.setEnabled(enable)

    def ui_toggle(self, enable):
        if self.chk1:
            self.btn1.setEnabled(enable)
            self.btn2.setEnabled(enable)
        else:
            self.btn1.setEnabled(False)
            self.btn2.setEnabled(False)

    def ui_add(self):
        self.ui_toggle(True)
        self.confirmed = False
        self.progress.emit(
            f"<b>Export to USB</b><br/>[{self.drive}] USB drive found! Ready to export.", 0, 'b', 0)
        self.freezeGUI(True)

    def ui_remove(self):
        self.ui_toggle(False)
        if self.confirmed:
            # Drive ejected by user request
            self.progress.emit(None, 66, 'b', 0)
        else:
            self.progress.emit(
                f"[{self.drive}] USB drive removed unexpectedly! Please eject first.", 100, 'b', 0)
        self.confirmed = False
        self.drive = None
        self.freezeGUI(True)

    def diff(self, list1, list2):
        list_difference = [item for item in list1 if item not in list2]
        return list_difference

    def mainTask(self):
        Log.d(TAG1, "Thread started.")
        dl = 'AB_DEFGHIJKLMNOPQRSTUVWXYZ'  # ignore C drive
        drives = ['%s:' % d for d in dl if os.path.exists('%s:' % d)]
        Log.i(TAG1, "Existing drive(s): " + str(drives))
        if len(drives) == 1:  # and drives[0] != "D:":
            Log.i(TAG1, "Using existing drive")
            self.drive = drives[0]
            self.usb_add.emit()
        if len(drives) > 1:
            self.progress.emit(
                "<b>Re-insert USB drive...</b><br/>Multiple USB drives detected.", 0, 'b', 0)
        self.checkMode = None
        while not self.do_close:
            uncheckeddrives = ['%s:' %
                               d for d in dl if os.path.exists('%s:' % d)]
            if self.chk1:
                if self.checkMode != 1:
                    Log.i("Detecting USB drives...")
                    drives = []
                changes = False
                x = self.diff(uncheckeddrives, drives)
                if x:
                    changes = True
                    Log.i(TAG1, "New drive(s):      " + str(x))
                    for d in x:
                        Log.i(TAG1, "New drive introduced")
                        self.drive = d
                        self.usb_add.emit()
                x = self.diff(drives, uncheckeddrives)
                if x:
                    changes = True
                    Log.i(TAG1, "Removed drive(s):  " + str(x))
                    for d in x:
                        Log.i(TAG1, "Drive disconnected")
                        if self.drive == d:
                            self.usb_remove.emit()
                if changes:
                    drives = ['%s:' %
                              d for d in dl if os.path.exists('%s:' % d)]
                self.checkMode = 1
            elif self.chk2:
                if self.checkMode != 2:
                    self.progress.emit(
                        "<b>Export to Folder</b><br/>Configure settings then hit 'Export'.", 0, 'b', 0)
                self.checkMode = 2
            else:
                if self.checkMode != 0:
                    self.progress.emit(
                        "<b>Select an export location...</b><br/>Export to Folder or USB.", 0, 'b', 0)
                self.checkMode = 0
        Log.i(TAG1, "Thread finished.")

    def select_folder_source(self):
        data_path = QtCore.QUrl.fromLocalFile(
            os.path.join(os.getcwd(), Constants.log_export_path))
        select_data = self.select_folder(data_path)
        if select_data == None:
            # self.selectRun.setText("[ALL]")
            # self.source_subfolder = ""
            # Log.d("User aborted folder selection.")
            return
        if not data_path.toLocalFile() in select_data:
            self.selectRun.setText("[ALL]")
            self.source_subfolder = ""
            Log.w("User selected folder not in logged data path.")
            return
        self.selection.setChecked(True)  # force "selection" checked on select
        t = os.path.split(select_data)
        self.selectRun.setText(t[1])
        self.source_subfolder = select_data.replace(
            data_path.toLocalFile(), "").replace("/", Constants.slash)
        if self.source_subfolder[0] == Constants.slash:
            self.source_subfolder = self.source_subfolder[1:]
        if self.source_subfolder[-1] == Constants.slash:
            self.source_subfolder = self.source_subfolder[:-1]
        if self.source_subfolder.count(Constants.slash) == 1:
            dev_or_run = "run:"
        else:
            dev_or_run = "dev:"
        self.selectRun.setText(f"{dev_or_run}{self.selectRun.text()}")
        self.generateExportName()

    def select_folder_target(self):
        top_level = QtCore.QUrl("clsid:0AC0837C-BBF8-452A-850D-79D08E667CA7")
        if self.btn5.text() != "[NONE]":
            top_level = QtCore.QUrl.fromLocalFile(self.btn5.text())
        select_path = self.select_folder(top_level)
        if select_path == None:
            return  # self.btn5.setText("[NONE]")
        self.drive = select_path
        self.btn5.setText(self.drive)
        self.freezeGUI(True)

    def select_import_folder(self):
        top_level = QtCore.QUrl("clsid:0AC0837C-BBF8-452A-850D-79D08E667CA7")
        if self.btn7.text() != "[NONE]":
            path = self.btn7.text()
            path = os.path.split(path)[0]
            # if sp[1].count(".") > 0:
            #    path = sp[0] # drop ZIP file for folder select
            top_level = QtCore.QUrl.fromLocalFile(path)
        select_path = self.select_folder(top_level)
        if select_path == None:
            return  # self.btn5.setText("[NONE]")
        # self.drive = select_path
        self.btn7.setText(select_path)
        self.generateImportDescription()
        # self.freezeGUI(True)

    def select_folder(self, dir):
        folderpath = QtWidgets.QFileDialog.getExistingDirectoryUrl(
            self, 'Select Folder', dir)
        if folderpath.isValid():
            Log.i(TAG1, f"Selected {folderpath.toLocalFile()}")
            return folderpath.toLocalFile()
        else:
            Log.w(TAG1, "User cancelled folder selection request.")
            return None

    def select_file_source(self):
        top_level = QtCore.QUrl("clsid:0AC0837C-BBF8-452A-850D-79D08E667CA7")
        if self.btn7.text() != "[NONE]":
            path = self.btn7.text()
            path = os.path.split(path)[0]
            # if sp[1].count(".") > 0:
            #    path = sp[0] # drop ZIP file for folder select
            top_level = QtCore.QUrl.fromLocalFile(path)
        select_path = self.select_file(top_level)
        if select_path == None:
            return  # self.btn5.setText("[NONE]")
        # self.drive = select_path
        self.btn7.setText(select_path)
        self.generateImportDescription()
        # self.freezeGUI(True)

    def select_file(self, dir):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileUrl(
            self, 'Select File', dir)
        # Log.d(filepath)
        if filepath.isValid():
            Log.i(TAG2, f"Selected {filepath.toLocalFile()}")
            return filepath.toLocalFile()
        else:
            Log.w(TAG2, "User cancelled folder selection request.")
            return None

    def doImport(self):
        self.stop_threads = False
        thread = Thread(target=self.importTask, args=(
            lambda: self.stop_threads, self.btn7.text()))
        thread.start()

    def importTask(self, abort, path):
        self.importNow.setEnabled(False)
        self.importCancel.setEnabled(True)
        try:
            self.progress.emit(
                "<b>Importing archived data...</b><br/>please wait...", 0, 'g', 1)
            time.sleep(2)  # give user time to bail
            if abort():
                self.progress.emit(
                    "<b>Import to local data: Operation cancelled.</b><br/>No import was performed.", 100, 'b', 1)
                Log.w("Import thread killed prematurely!")
                self.importNow.setEnabled(True)
                self.importCancel.setEnabled(False)
                return

            if os.path.isfile(path):
                copied = 0
                skipped = 0
                if not zipfile.is_zipfile(path):
                    self.progress.emit(
                        "<b>Import Error: Invalid ZIP file selected!</b><br/>Please try again...", 100, 'r', 1)
                    Log.e("Selected file does not appear to be a valid ZIP file.")
                    self.importNow.setEnabled(True)
                    self.importCancel.setEnabled(False)
                    return
                with zipfile.ZipFile(path, 'r') as f:
                    zip_filename = os.path.split(path)[1][:-4]
                    self.progress.emit(
                        "<b>Verifying ZIP file integrity...</b><br/>please wait...", 0, 'b', 1)
                    test_result = f.testzip()
                    if test_result != None:
                        self.progress.emit(
                            "<b>Import Error: Corrupt ZIP file selected!</b><br/>Please try again...", 100, 'r', 1)
                        Log.e(
                            "Selected ZIP archive contains a bad file:", test_result)
                        self.importNow.setEnabled(True)
                        self.importCancel.setEnabled(False)
                        return
                    preferences_load_path = UserProfiles.user_preferences._get_load_data_path()
                    if preferences_load_path == "":
                        local_data = os.path.join(
                            os.getcwd(), Constants.log_export_path)
                    else:
                        # If user preferences are set, load from the prefered load path.
                        local_data = os.path.join(
                            os.getcwd(), preferences_load_path)
                    Log.i(f"Import from {path} to {local_data}")
                    all_info = f.infolist()
                    zippedFolders = []
                    zippedFiles = []
                    zippedXMLs = []
                    for info in all_info:
                        if info.filename[-1] == '/':
                            zippedFolders.append(info)
                        else:
                            zippedFiles.append(info)
                            if info.filename[-4:] == ".xml":
                                zippedXMLs.append(info)
                    export_to = {}
                    for idx, xf in enumerate(zippedXMLs):
                        xfp = os.path.split(xf.filename)[0]
                        xml_str = f.read(xf).decode()
                        doc = minidom.parseString(xml_str)
                        metrics = doc.getElementsByTagName("run_info")
                        for m in metrics:
                            device = m.getAttribute("device") if m.hasAttribute(
                                "device") else zip_filename
                        if xf.filename.find(Constants.log_export_path) >= 0:
                            Log.e(
                                f"XML {xf.filename} did not provide the device name. Using cwd() since full-path is in ZIP.")
                            exp = os.getcwd()
                        elif xfp.count('/') == 0:  # or not device in xfp:
                            if device == zip_filename:
                                Log.e(
                                    f"XML {xf.filename} did not provide the device name. Using \"{device}\" as a fallback.")
                            exp = os.path.join(local_data, device)
                        else:
                            exp = local_data
                        export_to[xfp] = exp
                    for idx, xf in enumerate(zippedFiles):
                        xfp = os.path.split(xf.filename)[0]
                        if export_to.get(xfp) == None:
                            if xf.filename.find(Constants.log_export_path) >= 0:
                                Log.e(
                                    f"XML missing for run {xfp}. Using cwd() since full-path is in ZIP.")
                                exp = os.getcwd()
                            elif xfp.count('/') == 0:
                                device = zip_filename
                                Log.e(
                                    f"XML missing for run {xfp}. Using \"{device}\" as a fallback.")
                                exp = os.path.join(local_data, device)
                            else:
                                exp = local_data
                            export_to[xfp] = exp
                    num_files = len(zippedFiles)
                    zip_src = os.path.split(path)[1]
                    for x, zf in enumerate(zippedFiles):
                        pct = min(99, max(1, int(100 * (x + 1) / num_files)))
                        if abort():
                            self.progress.emit(
                                "<b>Import {}: Operation cancelled.</b><br/>Parital import was performed.".format(zip_src), pct, 'b', 1)
                            Log.w("Import thread killed prematurely!")
                            return
                        sp0 = os.path.split(zf.filename)
                        run = os.path.split(sp0[0])[1]
                        try:
                            if run == "_unnamed":
                                run = sp0[1][0:-4]
                        except:
                            pass
                        self.progress.emit(
                            "<b>Importing {}... please wait...</b><br/>Importing '{}'".format(zip_src, run), pct, 'g', 1)
                        d = os.path.join(os.getcwd(), zf.filename)
                        allow_copy = False
                        if self.btnGroup4.checkedId() == 1:
                            allow_copy = True
                        elif not os.path.exists(d):
                            allow_copy = True
                        elif self.btnGroup4.checkedId() == 2:
                            last_modified = datetime.datetime(
                                *zf.date_time).astimezone()
                            exist_modified = datetime.datetime.fromtimestamp(
                                os.stat(d).st_mtime, tz=datetime.timezone.utc)
                            # 2 sec resolution on zf.date_time
                            if last_modified - exist_modified > datetime.timedelta(seconds=2):
                                allow_copy = True
                        # elif self.btnGroup4.checkedId() == 4:
                        #    Log.w("User selected to cancel the copy operation.")
                        item = zf.filename
                        if allow_copy:
                            if item.endswith(".xml"):
                                copied += 1
                            d = f.extract(zf, export_to.get(sp0[0]))
                            last_modified = datetime.datetime(
                                *zf.date_time).astimezone()
                            epoch = datetime.datetime.fromtimestamp(
                                0, tz=datetime.timezone.utc)
                            file_time = (last_modified - epoch).total_seconds()
                            os.utime(d, (file_time, file_time))
                        else:
                            if item.endswith(".xml"):
                                skipped += 1
            else:

                def find_xml_files(directory):
                    xml_files = []
                    all_files = []
                    for root, dirs, files in os.walk(directory):
                        for file in files:
                            if file.endswith(".xml"):
                                xml_files.append(os.path.join(root, file))
                            all_files.append(os.path.join(root, file))
                    return xml_files, all_files

                # if not Constants.log_export_path in path:
                #     path = os.path.join(path, Constants.log_export_path)
                #     if not os.path.exists(path): raise ValueError("No logged data root found in archive.")
                # relative = path[path.rindex(Constants.log_export_path):]
                local_data = os.path.join(
                    os.getcwd(), Constants.log_export_path)  # relative)
                archive_filename = os.path.split(path)[1]
                xml_files, all_files = find_xml_files(path)
                # Log.w(f"xmls: {xml_files}")
                export_to = {}
                for idx, xf in enumerate(xml_files):
                    xfp = os.path.split(xf)[0]
                    relative = xfp.replace(path, "")
                    if len(relative) > 0:
                        if relative[0] == Constants.slash:  # trim leading slash
                            relative = relative[1:]
                        if relative[-1] == Constants.slash:  # trim trailing slash
                            relative = relative[:-1]
                    doc = minidom.parse(xf)
                    metrics = doc.getElementsByTagName("run_info")
                    for m in metrics:
                        found_dev = m.hasAttribute("device")
                        device = m.getAttribute("device") if m.hasAttribute(
                            "device") else archive_filename
                        name = m.getAttribute("name") if m.hasAttribute(
                            "name") else os.path.split(xfp)[1]
                    if relative.find(Constants.log_export_path) >= 0:
                        Log.e(
                            f"XML {xf} did not provide the device name. Using cwd() since full-path is in ZIP.")
                        exp = os.path.join(os.getcwd(), relative)
                    # or not device in xfp:
                    elif relative.count(Constants.slash) == 0:
                        if not found_dev:
                            if name == archive_filename:
                                run_path = os.path.join(path, name)
                                # check if the archive folder root was selected, not the run itself
                                if os.path.exists(run_path):
                                    # the archive has the same name as the run, nothing to do
                                    pass
                                else:  # they selected a run within an archive to import, the archive name is one level up
                                    Log.d(
                                        "A run within an archive folder was selected. The archive name is one higher in the tree.")
                                    archive_filename = os.path.split(os.path.split(path)[0])[
                                        1]  # take one level up
                                    device = archive_filename
                            Log.e(
                                f"XML {xf} did not provide the device name. Using \"{device}\" as a fallback.")
                        exp = os.path.join(local_data, device, name)
                    else:
                        exp = os.path.join(local_data, relative)
                    export_to[xfp] = exp

                for idx, xf in enumerate(all_files):
                    xfp = os.path.split(xf)[0]
                    relative = xfp.replace(path, "")
                    if len(relative) > 0:
                        if relative[0] == Constants.slash:  # trim leading slash
                            relative = relative[1:]
                        if relative[-1] == Constants.slash:  # trim trailing slash
                            relative = relative[:-1]
                    if export_to.get(xfp) == None:
                        if relative.find(Constants.log_export_path) >= 0:
                            Log.e(
                                f"XML missing for run {xfp}. Using cwd() since full-path is in ZIP.")
                            exp = os.path.join(os.getcwd(), relative)
                        elif relative.count(Constants.slash) == 0:
                            device = archive_filename
                            name = os.path.split(xfp)[1]
                            if name == archive_filename:
                                run_path = os.path.join(path, name)
                                # check if the archive folder root was selected, not the run itself
                                if os.path.exists(run_path):
                                    # the archive has the same name as the run, nothing to do
                                    pass
                                else:  # they selected a run within an archive to import, the archive name is one level up
                                    Log.d(
                                        "A run within an archive folder was selected. The archive name is one higher in the tree.")
                                    archive_filename = os.path.split(os.path.split(path)[0])[
                                        1]  # take one level up
                                    device = archive_filename
                            Log.e(
                                f"XML missing for run {xfp}. Using \"{device}\" as a fallback.")

                            exp = os.path.join(local_data, device, name)
                        else:
                            exp = os.path.join(local_data, relative)
                        export_to[xfp] = exp
                Log.i(f"Import from {path} to {local_data}")
                for key, val in export_to.items():
                    path = key
                    local_data = val
                    copied, skipped = self.copytree(
                        path, local_data, self.btnGroup4.checkedId())

            history_path = os.path.join(
                os.getcwd(), Constants.log_export_path, "export_history.log")
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    log_lines = f.read()
            else:
                log_lines = ""
            with open(history_path, 'w') as f:
                f.write(
                    f"<b>Imported {copied} run(s) at {str(datetime.datetime.now()).split('.')[0]}</b><br/>\n")
                f.write(f"<small>from \"{path}\" <br/>\n")
                f.write("to \"{}\"</small><br/>\n".format(local_data))
                f.write(f"<small>Settings: ")
                f.write("Import from {}, ".format(
                    "ZIP" if ".zip" in path else "Folder"))
                f.write(
                    f"{self.btnGroup4.checkedButton().text()} existing files</small><br/>\n")
                if skipped > 0:
                    f.write(
                        f"<small>Skipped {skipped} run(s) since overwrites were disabled.</small><br/>\n")
                f.write(f"<br/>\n")
                f.write(log_lines)  # pre-pend data to log file
            finished_msg = f"<b>Imported {copied} run(s) from archive!</b><br/>Import process complete."
            if skipped > 0:
                finished_msg += f" {skipped} run(s) were skipped."
            self.progress.emit(finished_msg, 100, 'g', 1)
        except Exception as e:
            Log.e(TAG2, "Import error: {}".format(str(e)))
            self.progress.emit("Error importing local data!", 100, 'r', 1)
        self.importNow.setEnabled(True)
        self.importCancel.setEnabled(False)

    def export(self):
        self.stop_threads = False
        if self.btnGroup5.checkedId() == 0:  # all, filtering off
            self.filter_min = 0  # no filter
        elif self.btnGroup5.checkedId() == 1:  # today
            self.filter_min = datetime.datetime.utcnow() - datetime.timedelta(hours=24)
        elif self.btnGroup5.checkedId() == 2:  # last x something
            if self.filterNumDays.hasAcceptableInput():
                date_num = int(self.filterNumDays.text())
                if self.filterUnits.currentText() == "Hours":
                    timedelta = datetime.timedelta(hours=date_num)
                elif self.filterUnits.currentText() == "Days":
                    timedelta = datetime.timedelta(days=date_num)
                elif self.filterUnits.currentText() == "Weeks":
                    timedelta = datetime.timedelta(weeks=date_num)
                else:
                    Log.e(
                        f"Input Error: \"Export by date\" units \"{self.filterUnits.currentText()}\" are not recognized.")
                    return
                self.filter_min = datetime.datetime.utcnow() - timedelta
            else:
                Log.e(
                    f"Input Error: \"Export by date\" range must be between 1 and 31. You entered \"{self.filterNumDays.text()}\".")
                return
        if self.exportAsZIP.isChecked() and len(self.exportNameTxt.text()) == 0:
            self.generateExportName()
            default_filename = self.exportNameTxt.text()
            export_name, ok = QtWidgets.QInputDialog.getText(
                None, 'Export Name', 'File name for this export catalog:', text=default_filename)
            if not ok:
                Log.w(
                    "User cancelled file name request for exporting a ZIP file archive.")
                return
            self.exportNameTxt.setText(export_name)
        export = Thread(target=self.exportTask, args=(
            lambda: self.stop_threads, self.exportNameTxt.text(), self.drive, self.filter_min,))
        export.start()

    def cancel(self):
        self.stop_threads = True
        Log.w("User cancelled an operation")
        self.freezeGUI(True)

    def exportTask(self, abort, name, output_folder, date_filter):
        self.freeze_gui.emit(False)
        try:
            # opft = os.path.split(output_folder)
            # if opft[1] == Constants.log_export_path:
            #    output_folder = opft[0] # don't nest "logged_data" folders
            output_folder = output_folder.replace("/", Constants.slash)
            if len(output_folder) > 2:
                self.drive = output_folder[0:2]
            else:
                output_folder += Constants.slash
            drive_or_folder = "USB drive" if self.chk1 else "folder"
            data_path = os.path.join(os.getcwd(), Constants.log_export_path)
            if Constants.log_export_path in output_folder:
                export_path = os.path.join(output_folder[0:output_folder.rindex(
                    Constants.log_export_path)], Constants.log_export_path)
            else:
                export_path = os.path.join(
                    output_folder, name, Constants.log_export_path)
            # Log.d(output_folder, export_path)
            if self.exportAsZIP.isChecked():
                export_folder = os.path.split(export_path)[0]
                zip_path = export_folder + ".zip"
                if os.path.exists(export_folder):
                    Log.w(
                        TAG1, "WARN: A folder with the same Export Name already exists at this location")
                if os.path.exists(zip_path) and self.btnGroup2.checkedId() != 1:
                    if os.path.exists(export_folder):
                        self.progress.emit("<b>[{}] Export to {}: Folder already exists.</b><br/>Please choose a different Export Name.".format(
                            self.drive, drive_or_folder), 100, 'r', 0)
                        Log.w("Export thread killed prematurely!")
                        return
                    with zipfile.ZipFile(zip_path, 'r') as f:
                        Log.i(TAG1, "Expanding existing ZIP archive")
                        self.progress.emit(
                            "<b>Expanding existing ZIP archive...</b><br/>please wait...", 0, 'g', 0)
                        all_info = f.infolist()
                        zippedFolders = []
                        zippedFiles = []
                        for info in all_info:
                            if info.filename[-1] == '/':
                                zippedFolders.append(info)
                            else:
                                zippedFiles.append(info)
                        for x, zf in enumerate(zippedFiles):
                            d = f.extract(zf, export_folder)
                            last_modified = datetime.datetime(
                                *zf.date_time).astimezone()
                            epoch = datetime.datetime.fromtimestamp(
                                0, tz=datetime.timezone.utc)
                            Log.w(
                                f"s_tz = {last_modified.tzinfo}, d_tz = {epoch.tzinfo}")
                            file_time = (last_modified - epoch).total_seconds()
                            os.utime(d, (file_time, file_time))
            Log.i(
                TAG1, f"[{self.drive}] Exporting to {drive_or_folder} {export_path}...")
            self.progress.emit(
                f"[{self.drive}] Exporting to {drive_or_folder}... please wait...", 0, 'g', 0)
            copied = 0
            skipped = 0
            select_device, select_run = os.path.split(self.source_subfolder)
            if select_device == '':
                select_device = select_run
                select_run = ''
            # walking through the directory tree
            for folder, devices, logs in os.walk(data_path):
                y1 = len(devices)
                z1 = 0
                for x1, device in enumerate(devices):
                    if select_device != '':
                        if select_device != device:
                            continue
                        y1 = 1
                        z1 = x1  # only exporting a single device folder
                    device_path = os.path.join(data_path, device)
                    for folder, runs, files in os.walk(device_path):
                        y2 = len(runs)
                        z2 = 0
                        for x2, run in enumerate(runs):
                            if select_run != '':
                                if select_run != run:
                                    continue
                                y2 = 1
                                z2 = x2 - 0.5  # only exporting a single run folder
                            pct = min(
                                99, max(1, int(100 * (((x1 - z1) + ((x2 - z2) / y2)) / y1))))
                            if abort():
                                self.progress.emit("<b>[{}] Export to {}: Operation cancelled.</b><br/>Parital export was performed.".format(
                                    self.drive, drive_or_folder), pct, 'b', 0)
                                Log.w("Export thread killed prematurely!")
                                return
                            # Log.d(TAG1, "{}% - [{}] Exporting to {} \"{}\"...".format(pct, self.drive, drive_or_folder, run))
                            t_run = run
                            is_unnamed = False
                            try:
                                if t_run == "_unnamed":
                                    is_unnamed = True
                                    t_run = files[0][0:-4]
                            except:
                                pass
                            self.progress.emit("<b>[{}] Exporting to {}... please wait...</b><br/>Exporting '{}'".format(
                                self.drive, drive_or_folder, t_run), pct, 'g', 0)
                            if is_unnamed and not self.exportUnnamed.isChecked():
                                continue  # skip "_unnamed" runs if not opted in for including them
                            src = os.path.join(data_path, device, run)
                            dst = os.path.join(export_path, device, run)
                            if not os.path.exists(src):
                                Log.w(f"Skipping non-existent folder: {src}")
                                continue
                            copied, skipped = self.copytree(
                                src, dst, self.btnGroup2.checkedId(), None, copied, skipped, date_filter)
            # remove nested folders
            Log.d(f"Checking for nested folders at {export_path}")
            top_level = os.path.split(export_path)[0]
            path = export_path  # includes 'logged_data'
            while os.path.exists(path):
                files = [file for file in os.listdir(
                    path) if not os.path.isdir(os.path.join(path, file))]
                folders = [file for file in os.listdir(
                    path) if os.path.isdir(os.path.join(path, file))]
                num_files = len(files)
                num_folders = len(folders)
                if num_files == 0 and num_folders == 1:
                    path = os.path.join(path, folders[0])
                    Log.d(f"Moving into path: {path}")
                    continue
                else:
                    src = path + Constants.slash  # force to directory, not a file
                    if num_files > 1:
                        # force to directory, not a file
                        dst = os.path.join(top_level, os.path.split(path)[
                            1]) + Constants.slash
                    else:
                        dst = top_level
                    Log.d(f"Moving nested folders from {src} to {dst}...")
                    Log.d(
                        f"Deleting empty folder structure at {export_path}...")
                    # Check if the destination folder already exists
                    if not os.path.exists(dst):
                        Log.d(f"Making directory: {dst}")
                        os.makedirs(dst)
                    if not os.path.samefile(src, dst):
                        # Move the contents of the single folder to the parent folder
                        self.copytree(src, dst, self.btnGroup2.checkedId())
                        shutil.rmtree(export_path)
                    else:
                        Log.d(
                            "NOTICE: Nested directory structure points to itself. Leaving as-is.")
                    break  # we've removed all the nesting we can
            if self.exportAsZIP.isChecked():
                self.progress.emit(
                    "<b>Creating ZIP Archive...</b><br/>This may take a while for large exports...", 99, 'g', 0)
                export_path = os.path.split(export_path)[0]
                zip_path = export_path + ".zip"
                if os.path.exists(zip_path):
                    Log.w(TAG1, "Overwriting existing ZIP archive")
                shutil.make_archive(export_path, 'zip', export_path)
                shutil.rmtree(export_path)
            Log.i(TAG1, "DONE - Exported {} run(s) to {}.".format(copied, export_path))
            if skipped > 0:
                reason = "they already existed in the output location" if self.btnGroup5.checkedId(
                ) == 0 else "date filtering was enabled"
                Log.i(TAG1, "Skipped {} run(s) because {}".format(skipped, reason))

            history_path = os.path.join(
                os.getcwd(), Constants.log_export_path, "export_history.log")
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    log_lines = f.read()
            else:
                log_lines = ""
            with open(history_path, 'w') as f:
                f.write(
                    f"<b>Exported {copied} run(s) at {str(datetime.datetime.now()).split('.')[0]}</b><br/>\n")
                f.write(f"<small>from \"{data_path}\" <br/>\n")
                f.write("to \"{}{}\"</small><br/>\n".format(export_path,
                        ".zip" if self.exportAsZIP.isChecked() else ""))
                f.write(f"<small>Settings: ")
                f.write("Export {}{}, ".format(self.btnGroup3.checkedButton().text(
                ), self.selectRun.text() if self.selection.isChecked() else ""))
                f.write(f"{self.btnGroup1.checkedButton().text()}, ")
                f.write(
                    f"{self.btnGroup2.checkedButton().text()} existing files</small><br/>\n")
                if skipped > 0:
                    reason = "overwrites were disabled" if self.btnGroup5.checkedId(
                    ) == 0 else "filtering was enabled"
                    f.write(
                        f"<small>Skipped {skipped} run(s) since {reason}.</small><br/>\n")
                f.write(f"<br/>\n")
                f.write(log_lines)  # pre-pend data to log file
            finished_msg = f"[{self.drive}] Exported to {drive_or_folder}!"
            if drive_or_folder != "folder":
                finished_msg += " Ready to eject."
            self.progress.emit(finished_msg, 100, 'g', 0)
            self.exported = True
        except Exception as e:
            Log.e(TAG1, "Export error: {}".format(str(e)))
            self.progress.emit("Error exporting local data!", 100, 'r', 0)
        self.freeze_gui.emit(True)

    # Copy missing or modified files from 'src' to 'dst'
    # (leave newer or existing files in 'dst' untouched)
    # Use 'symlinks' to indicate overwrite all to output
    def copytree(self, src, dst, symlinks=None, ignore=None, copied=0, skipped=0, date_filter=0):
        # if not os.path.exists(dst):
        #     os.makedirs(dst)
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                copied, skipped = self.copytree(
                    s, d, symlinks, ignore, copied, skipped, date_filter)
            else:
                allow_copy = False
                if symlinks == 1:
                    allow_copy = True
                elif not os.path.exists(d):
                    allow_copy = True
                elif symlinks == 2:
                    last_modified = datetime.datetime.utcfromtimestamp(
                        os.stat(s).st_mtime)
                    exist_modified = datetime.datetime.utcfromtimestamp(
                        os.stat(d).st_mtime)
                    # 2 sec resolution on zf.date_time
                    if last_modified - exist_modified > datetime.timedelta(seconds=2):
                        allow_copy = True
                if allow_copy and date_filter != 0:
                    # check all files in this folder for recency filtering
                    path = src
                    if "_unnamed" in path:  # check this file only, it's an orphan
                        last_modified = datetime.datetime.utcfromtimestamp(
                            os.stat(s).st_mtime)
                    else:
                        files = [file for file in os.listdir(
                            path) if not os.path.isdir(os.path.join(path, file))]
                        epoch = datetime.datetime.utcfromtimestamp(0)
                        last_modified = epoch
                        for f in files:
                            f_path = os.path.join(path, f)
                            st_mtime = datetime.datetime.utcfromtimestamp(
                                os.stat(f_path).st_mtime)
                            if st_mtime > last_modified:
                                last_modified = st_mtime
                    if last_modified < date_filter:  # file older than filter
                        allow_copy = False

                if allow_copy:
                    if not os.path.exists(dst):
                        os.makedirs(dst)
                    if item.endswith(".xml"):
                        copied += 1
                    shutil.copy2(s, d)
                else:
                    if item.endswith(".xml"):
                        skipped += 1
        return copied, skipped

    def detect(self):
        # clear 'drives' in mainTask()
        self.chk1 = False
        self.chk2 = True
        i = 0
        while (self.checkMode != 2 and i < 30):  # 3 sec timeout
            time.sleep(0.1)
            i += 1
        self.chk1 = True
        self.chk2 = False

    def eject(self):
        self.stop_threads = False
        eject = Thread(target=self.ejectTask,
                       args=(lambda: self.stop_threads,))
        eject.start()

    def ejectTask(self, abort):
        self.freeze_gui.emit(False)
        try:
            Log.i(TAG1, f"[{self.drive}] USB drive ejecting...")
            self.progress.emit(
                f"[{self.drive}] USB drive ejecting... please wait...", 33, 'b', 0)
            self.confirmed = True
            time.sleep(1)
            if abort():
                self.progress.emit(
                    f"<b>[{self.drive}] USB drive eject: Operation cancelled.</b>", 0, 'b', 0)
                Log.w("Eject task aborted prematurely!")
                return
            # NOTE: Calling 'os.system' causes a console window to blip and disappear when launched with 'pythonw.exe':
            subprocess.call(
                'powershell $driveEject = New-Object -comObject Shell.Application; $driveEject.Namespace(17).ParseName("""{}""").InvokeVerb("""Eject""")'.format(self.drive), shell=True)
            if self.drive == None:
                tbd = self.tb.text().split()[0]
                Log.i(TAG1, f"[{tbd}] USB drive ejected.")
                self.progress.emit(
                    f"[{tbd}] USB drive ejected. Safe to remove.", 100, 'b', 0)
            else:
                Log.e(TAG1, f"[{self.drive}] USB drive eject failed!")
                self.progress.emit(
                    f"[{self.drive}] USB drive eject failed! Try again.", 66, 'r', 0)
        except Exception as e:
            Log.e(TAG1, "Eject error: {}".format(str(e)))
            self.progress.emit("Error ejecting USB drive!", 100, 'r', 0)
        self.freeze_gui.emit(True)

    def erase(self):
        self.stop_threads = False
        if not self.exported:
            if PopUp.question(self, "Confirm Erase", "You have not exported local data yet.\nAre you sure you want to erase it?"):
                Log.w(
                    TAG1, "Erasing local data without exporting first.\n ***Local data can be recovered from the Recycle Bin.***")
            else:
                Log.w(TAG1, "Erase Aborted by User.")
                return
        else:
            if PopUp.question(self, "Confirm Erase", "Are you sure you want to erase all local data?"):
                Log.i(
                    TAG1, "Erasing local data after exporting first.\n ***Local data can be recovered from the Recycle Bin.***")
            else:
                Log.w(TAG1, "Erase Aborted by User.")
                return
        erase = Thread(target=self.eraseTask,
                       args=(lambda: self.stop_threads,))
        erase.start()

    def eraseTask(self, abort):
        self.freeze_gui.emit(False)
        try:
            data_path = os.path.join(os.getcwd(), Constants.log_export_path)
            Log.i(TAG1, "Erasing local data...")
            self.progress.emit(
                "Erasing local data... please wait...", 0, 'r', 0)
            # walking through the directory tree
            for folder, devices, logs in os.walk(data_path):
                y1 = len(devices)
                for x1, device in enumerate(devices):
                    device_path = os.path.join(data_path, device)
                    for folder, runs, files in os.walk(device_path):
                        y2 = len(runs)
                        for x2, run in enumerate(runs):
                            pct = int(100 * ((x1 + (x2 / y2)) / y1))
                            if abort():
                                self.progress.emit(
                                    "<b>Erase local data: Operation cancelled.</b><br/>See Recycle Bin to restore deleted runs.", pct, 'b', 0)
                                Log.w(
                                    "Erase cancelled by user. Recover any deleted items from your Recycle Bin.")
                                return
                            # Log.d(TAG1, "{}% - Erasing local data \"{}\"...".format(pct, run))
                            self.progress.emit(
                                "<b>Erasing local data... please wait...</b><br/>Erasing '{}'".format(run), pct, 'r', 0)
                            run_path = os.path.join(data_path, device, run)
                            send2trash.send2trash(run_path)
                    send2trash.send2trash(device_path)
            Log.i(TAG1, "DONE - All local data erased.")
            self.progress.emit("All local data erased!", 100, 'g', 0)
        except Exception as e:
            Log.e(TAG1, "Erase error: {}".format(str(e)))
            self.progress.emit("Error erasing local data!", 100, 'r', 0)
        self.freeze_gui.emit(True)

    def setProgress(self, label, pct, color, tab=0):
        if color == 'r':
            c = 'cc0000'
        if color == 'g':
            c = '00cc00'
        if color == 'b':
            c = '0000cc'
        if pct == 0:
            c = 'cccccc'
        if tab == 0:
            tb = self.tb
            pb = self.pb
        else:
            tb = self.tb1
            pb = self.pb1
        if len(label):  # not None or ''
            tb.setText(label)
        if 0 < pct < 100:  # set for 1% to 99%
            pb.setStyleSheet(
                f'background: qlineargradient(x1:{(pct-1)/100} y1:0, x2:{pct/100} y2:0, stop:0 #{c}, stop:1 #cccccc); border: 1px solid #cccccc;')
            pb.setText(f'{pct}%')
        else:  # set for 0% and/or 100%
            pb.setStyleSheet(f'background: #{c}; border: 1px solid #cccccc;')
            pb.clear()

    def closeEvent(self, event):
        self.do_close = True
        self.main.join()
