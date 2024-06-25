from QATCH.common.architecture import Architecture
from QATCH.common.fileStorage import FileStorage, secure_open
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants
from PyQt5 import QtCore, QtGui, QtWidgets
from xml.dom import minidom
import numpy as np
import os
import pyzipper
import shutil
import datetime as dt
import hashlib

TAG = "[PopUp]"

###############################################################################
# Warning dialog module
###############################################################################
class PopUp:

    ###########################################################################
    # Shows a pop-up question dialog with yes and no buttons (unused)
    ###########################################################################
    @staticmethod
    def question_QCM(parent, title, message):
        """
        :param parent: Parent window for the dialog.
        :param title: Title of the dialog :type title: str.
        :param message: Message to be shown in the dialog :type message: str.
        :return: 1 if button1 was pressed, 0 if button2   :rtype: int.
        """
        #ans = QtWidgets.QMessageBox.question(parent, title, message, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        #if ans == QtWidgets.QMessageBox.Yes:
        #    Log.d('Si')
        #    return True
        #elif ans == QtWidgets.QMessageBox.No:
        #    Log.d('No')
        #    return False
        width = 340
        height = 220
        area = QtWidgets.QDesktopWidget().availableGeometry()
        left = int((area.width() - width) / 2)
        top = int((area.height() - height) / 2)
        box = QtWidgets.QMessageBox(parent)
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setWindowTitle(title)
        box.setGeometry(left, top, width, height)
        box.setText(message)
        box.setStandardButtons(QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
        button1 = box.button(QtWidgets.QMessageBox.Yes)
        button1.setText('@10MHz')
        button2 = box.button(QtWidgets.QMessageBox.No)
        button2.setText(' @5MHz')
        box.exec_()

        if box.clickedButton() == button1:
            Log.i(TAG, 'Quartz Crystal Sensor installed on the QATCH Q-1 Device: @10MHz')
            return 1
        elif box.clickedButton() == button2:
            Log.i(TAG, 'Quartz Crystal Sensor installed on the QATCH Q-1 Device: @5MHz')
            return 0

    ###########################################################################
    # Shows a pop-up question dialog with Yes/No (or Ok) buttons
    ###########################################################################
    @staticmethod
    def question_FW(parent, title, message, details = "", onlyOK = False):
        """
        :param parent: Parent window for the dialog.
        :param title: Title of the dialog :type title: str.
        :param message: Message to be shown in the dialog :type message: str.
        :return: 1 if button1 was pressed, 0 if button2   :rtype: int.
        """
        #ans = QtWidgets.QMessageBox.question(parent, title, message, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        #if ans == QtWidgets.QMessageBox.Yes:
        #    Log.d('Si')
        #    return True
        #elif ans == QtWidgets.QMessageBox.No:
        #    Log.d('No')
        #    return False
        width = 340
        height = 220
        area = QtWidgets.QDesktopWidget().availableGeometry()
        left = int((area.width() - width) / 2)
        top = int((area.height() - height) / 2)
        box = QtWidgets.QMessageBox(parent)
        icon_path = os.path.join(Architecture.get_path(), 'QATCH/icons/download_icon.ico')
        box.setIconPixmap(QtGui.QPixmap(icon_path))
        box.setWindowTitle(title)
        box.setGeometry(left, top, width, height)
        box.setText(message)
        box.setDetailedText(details)

        if not onlyOK:
            box.setStandardButtons(QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
            box.setDefaultButton(QtWidgets.QMessageBox.Yes)
            button1 = box.button(QtWidgets.QMessageBox.Yes)
            button1.setText('Yes')
            button2 = box.button(QtWidgets.QMessageBox.No)
            button2.setText('No')
        else:
            box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            box.setDefaultButton(QtWidgets.QMessageBox.Ok)
            button1 = box.button(QtWidgets.QMessageBox.Ok)
            button1.setText('Awesome!')

        box.exec_()

        return box.clickedButton() == button1

    ###########################################################################
    # Shows a Pop up warning dialog with Ok button
    ###########################################################################
    @staticmethod
    def warning(parent, title, message):
        """
        :param parent: Parent window for the dialog.
        :param title: Title of the dialog :type title: str.
        :param message: Message to be shown in the dialog :type message: str.
        """
        QtWidgets.QMessageBox.warning(None, title, message, QtWidgets.QMessageBox.Ok)
        #msgBox=QtWidgets.QMessageBox.warning(parent, title, message, QtWidgets.QMessageBox.Ok)
        #msgBox = QtWidgets.QMessageBox()
        #msgBox.setIconPixmap( QtGui.QPixmap("favicon.png"))
        #msgBox.exec_()

    ###########################################################################
    # Shows a pop-up question dialog with yes and no buttons
    ###########################################################################
    @staticmethod
    def question(parent, title, message, default = False):
        """
        :param parent: Parent window for the dialog.
        :param title: Title of the dialog :type title: str.
        :param message: Message to be shown in the dialog :type message: str.
        :return: True if Yes button was pressed :rtype: bool.
        """
        defaultButton = QtWidgets.QMessageBox.Yes if default else QtWidgets.QMessageBox.No
        ans = QtWidgets.QMessageBox.question(None, title, message, QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No, defaultButton)
        if ans == QtWidgets.QMessageBox.Yes:
            return True
        else:
            return False

    ###########################################################################
    # Shows a Pop up critical dialog with Retry/Ignore buttons
    ###########################################################################
    @staticmethod
    def critical(parent, title, message, details = "", question = False, ok_only = False, btn1_text = "Retry"):
        """
        :param parent: Parent window for the dialog.
        :param title: Title of the dialog :type title: str.
        :param message: Message to be shown in the dialog :type message: str.
        """
        #return (QtWidgets.QMessageBox.critical(parent, title, message, QtWidgets.QMessageBox.Retry, QtWidgets.QMessageBox.Ignore)
        #     == QtWidgets.QMessageBox.Retry)

        width = 340
        height = 220
        area = QtWidgets.QDesktopWidget().availableGeometry()
        left = int((area.width() - width) / 2)
        top = int((area.height() - height) / 2)
        box = QtWidgets.QMessageBox(parent)
        if question:
            box.setIcon(QtWidgets.QMessageBox.Question)
        else:
            box.setIcon(QtWidgets.QMessageBox.Critical)
        box.setWindowTitle(title)
        box.setGeometry(left, top, width, height)
        box.setText(message)
        box.setDetailedText(details)
        if question:
            box.setStandardButtons(QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
            box.setDefaultButton(QtWidgets.QMessageBox.No)
            button1 = box.button(QtWidgets.QMessageBox.Yes)
            button1.setText('Yes')
            button2 = box.button(QtWidgets.QMessageBox.No)
            button2.setText('No')
        elif ok_only:
            box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            box.setDefaultButton(QtWidgets.QMessageBox.Ok)
            button1 = box.button(QtWidgets.QMessageBox.Ok)
            button1.setText('Ok')
        else:
            box.setStandardButtons(QtWidgets.QMessageBox.Retry|QtWidgets.QMessageBox.Ignore)
            box.setDefaultButton(QtWidgets.QMessageBox.Retry)
            button1 = box.button(QtWidgets.QMessageBox.Retry)
            button1.setText(btn1_text)
            button2 = box.button(QtWidgets.QMessageBox.Ignore)
            button2.setText('Ignore')
        box.exec_()

        return box.clickedButton() == button1

    ###########################################################################
    # Shows a Pop up information dialog with Ok button
    ###########################################################################
    @staticmethod
    def information(parent, title, message, details = ""):
        """
        :param parent: Parent window for the dialog.
        :param title: Title of the dialog :type title: str.
        :param message: Message to be shown in the dialog :type message: str.
        """
        QtWidgets.QMessageBox.information(None, title, message, QtWidgets.QMessageBox.Ok)
        #msgBox=QtWidgets.QMessageBox.warning(parent, title, message, QtWidgets.QMessageBox.Ok)
        #msgBox = QtWidgets.QMessageBox()
        #msgBox.setIconPixmap( QtGui.QPixmap("favicon.png"))
        #msgBox.exec_()


class QueryComboBox(QtWidgets.QWidget):
    finished = QtCore.pyqtSignal()
    confirmed = False

    def __init__(self, items, type = "item", parent = None):
        super(QueryComboBox, self).__init__(parent)

        layout_h = QtWidgets.QHBoxLayout()
        self.cb = QtWidgets.QComboBox()
        self.cb.addItems(items)
        self.cb.setCurrentIndex(-1)
        self.btn = QtWidgets.QPushButton("OK")
        self.btn.pressed.connect(self.confirm)
        layout_h.addWidget(self.cb)
        layout_h.addWidget(self.btn)

        layout_v = QtWidgets.QVBoxLayout()
        self.tb = QtWidgets.QLabel()
        vowel = 'aeiou'
        a_n = "an" if type[0].lower() in vowel else "a"
        self.tb.setText("Select {} {}:".format(a_n, type))
        layout_v.addWidget(self.tb)
        layout_v.addLayout(layout_h)

        self.setLayout(layout_v)
        self.setWindowTitle("Select")

    def confirm(self):
        self.confirmed = True
        self.close()

    def closeEvent(self, event):
        if not self.confirmed:
            self.cb.setCurrentIndex(-1)
        self.finished.emit() # queue call for clickedButton
        event.ignore() # keep open until clickedButton call

    def clickedButton(self):
        ret = self.cb.currentIndex()
        self.hide()
        return ret


class QueryRunInfo(QtWidgets.QWidget):
    finished = QtCore.pyqtSignal()

    def __init__(self, run_name, run_path, run_ruling, user_name = "NONE", recall_from = Constants.query_info_recall_path, parent = None):
        super(QueryRunInfo, self).__init__(None)
        self.parent = parent

        self.run_name = run_name
        self.run_path = run_path
        self.xml_path = run_path[0:-4] + ".xml"
        self.recall_xml = recall_from
        self.run_ruling = "good" if run_ruling else "bad"
        self.username = user_name
        self.post_run = self.recall_xml != self.xml_path
        self.unsaved_changes = self.post_run # force save on post-run
        self.batch_found = False

        self.q_batch = QtWidgets.QHBoxLayout() # batch #
        self.l_batch = QtWidgets.QLabel()
        self.l_batch.setText("Batch Number\t=")
        self.q_batch.addWidget(self.l_batch)
        self.t_batch = QtWidgets.QLineEdit()
        self.q_batch.addWidget(self.t_batch)
        self.h_batch = QtWidgets.QLabel()
        self.h_batch.setText("<u>?</u>")
        self.h_batch.setToolTip("<b>Hint:</b> Find this # on the crystal's packaging.")
        self.q_batch.addWidget(self.h_batch)

        self.blankIcon = QtGui.QIcon()
        self.foundIcon = QtGui.QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "checkmark.png"))
        self.missingIcon = QtGui.QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "warning.png"))
        self.t_batchAction = self.t_batch.addAction(self.blankIcon, QtWidgets.QLineEdit.TrailingPosition)
        self.t_batchAction.triggered.connect(self.find_batch_num)
        self.t_batch.textChanged.connect(self.find_batch_num)
        self.t_batch.editingFinished.connect(self.find_batch_num)

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
        self.l2.setText("Solvent\t\t=")
        self.q2.addWidget(self.l2)
        self.t0 = QtWidgets.QLineEdit()

        self.fluids = []
        self.surface_tensions = []
        self.densities = []
        try:
             # auto complete options
            working_resource_path = os.path.join(os.getcwd(), "QATCH/resources/") # prefer working resource path, if exists
            # bundled_resource_path = os.path.join(Architecture.get_path(), "QATCH/resources/") # otherwise, use bundled resource path
            resource_path = working_resource_path # if os.path.exists(working_resource_path) else bundled_resource_path
            data  = np.genfromtxt(os.path.join(resource_path, "lookup_by_solvent.csv"), dtype = 'str', delimiter = '\t', skip_header = 1)
            fluids_with_commas = data[:,0]
            surface_tensions = data[:,1]
            densities = data[:,2]
            for idx,name in enumerate(fluids_with_commas):
                st = float(surface_tensions[idx].strip())
                dn = float(densities[idx].strip())
                for n in name.split(','):
                    if len(n.strip()) > 0:
                        self.fluids.append(n.strip())
                        self.surface_tensions.append(st * 1000) # show as mN/m
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
            completer.setModelSorting(QtWidgets.QCompleter.CaseInsensitivelySortedModel)
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
        self.h0.setToolTip("<b>Hint:</b> If not listed, enter parameters manually.")
        self.q2.addWidget(self.h0)
        self.t0.textChanged.connect(self.lookup_completer)
        self.t0.editingFinished.connect(self.enforce_completer)

        self.q3 = QtWidgets.QHBoxLayout()
        self.l3 = QtWidgets.QLabel()
        self.l3.setText("Surfactant\t=")
        self.q3.addWidget(self.l3)
        self.t3 = QtWidgets.QLineEdit()
        self.validSurfactant = QtGui.QDoubleValidator(0, 1, 5)
        self.validSurfactant.setNotation(QtGui.QDoubleValidator.StandardNotation)
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
        self.validConcentration.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.t4.setValidator(self.validConcentration)
        self.q4.addWidget(self.t4)
        self.h4 = QtWidgets.QLabel()
        self.h4.setText("<u>mg/mL</u>")
        self.h4.setToolTip("<b>Hint:</b> For 100mg/mL enter \"100\".")
        self.q4.addWidget(self.h4)
        self.t4.textChanged.connect(self.calc_params)
        self.t4.editingFinished.connect(self.calc_params)

        self.r1 = QtWidgets.QHBoxLayout()
        self.l6 = QtWidgets.QLabel()
        self.l6.setText("Surface Tension\t=")
        self.r1.addWidget(self.l6)
        self.t1 = QtWidgets.QLineEdit()
        self.validSurfaceTension = QtGui.QDoubleValidator(1, 1000, 3)
        self.validSurfaceTension.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.t1.setValidator(self.validSurfaceTension)
        self.r1.addWidget(self.t1)
        self.h1 = QtWidgets.QLabel()
        self.h1.setText("<u>mN/m</u>")
        self.h1.setToolTip("<b>This field is auto-calculated.</b>\nYou can modify it to a custom value.")
        self.r1.addWidget(self.h1)

        self.r2 = QtWidgets.QHBoxLayout()
        self.l7 = QtWidgets.QLabel()
        self.l7.setText("Contact Angle\t=")
        self.r2.addWidget(self.l7)
        self.t2 = QtWidgets.QLineEdit()
        self.validContactAngle = QtGui.QDoubleValidator(10, 80, 1)
        self.validContactAngle.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.t2.setValidator(self.validContactAngle)
        self.r2.addWidget(self.t2)
        self.h2 = QtWidgets.QLabel()
        self.h2.setText("<u>deg</u>")
        self.h2.setToolTip("<b>This field is auto-calculated.</b>\nYou can modify it to a custom value.")
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
        self.h5.setToolTip("<b>This field is auto-calculated.</b>\nYou can modify it to a custom value.")
        self.r3.addWidget(self.h5)

        layout_v = QtWidgets.QVBoxLayout()
        self.l0 = QtWidgets.QLabel()
        self.l0.setText(f"<b><u>Run Info for \"{self.run_name}\":</b></u>")
        layout_v.addWidget(self.l0)
        layout_v.addLayout(self.q1)
        layout_v.addLayout(self.q_batch)
        layout_v.addLayout(self.q2)
        # layout_v.addLayout(self.q3) # hide Surfactant
        layout_v.addLayout(self.q4)
        self.l5 = QtWidgets.QLabel()
        self.l5.setText("<b><u>Estimated Parameters:</b></u>")
        layout_v.addWidget(self.l5)
        # layout_v.addLayout(self.r1) # hide Surface Tension
        layout_v.addLayout(self.r2)
        layout_v.addLayout(self.r3)
        self.q5 = QtWidgets.QCheckBox("Remember for next run")
        self.q5.setChecked(True)
        self.q5.setEnabled(self.unsaved_changes)
        layout_v.addWidget(self.q5)

        from QATCH.common.userProfiles import UserProfiles
        valid, infos = UserProfiles.session_info()
        if valid:
            Log.d(f"Found valid session: {infos}")
            self.username = infos[0]
            self.initials = infos[1]
        else:
            try:
                # all we can do is trust the user provided a valid username (session is now expired)
                infos = UserProfiles.get_user_info(UserProfiles().find(self.username, None)[1])
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
            self.signed_at = "[NEVER]"
            self.signature_required = True
            self.signature_received = False
        else:
            self.sign = QtWidgets.QLineEdit()
            self.sign.setText("[NONE]")
            self.initials = self.sign.text()
            self.signed_at = self.sign.text()
            self.signature_required = False
            self.signature_received = False

        ### START CAPTURE SIGNATURE CODE:
        # This code also exists in popUp.py in class QueryRunInfo for "ANALYZE SIGNATURE CODE"
        # The following method also is duplicated in both files: 'self.switch_user_at_sign_time'
        # There is duplicated logic code within the submit button handler: 'self.confirm'
        # The method for handling keystroke shortcuts is also duplicated too: 'self.eventFilter'
        self.signForm = QtWidgets.QWidget()
        self.signForm.setWindowFlags(QtCore.Qt.Dialog) # | QtCore.Qt.WindowStaysOnTopHint)
        icon_path = os.path.join(Architecture.get_path(), 'QATCH/icons/sign.png')
        self.signForm.setWindowIcon(QtGui.QIcon(icon_path)) #.png
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
        line_sep.setFrameShape(QtWidgets.QFrame.HLine);
        line_sep.setFrameShadow(QtWidgets.QFrame.Sunken);
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
        ### END CAPTURE SIGNATURE CODE

        self.btn = QtWidgets.QPushButton("Save")
        self.btn.pressed.connect(self.confirm)
        layout_v.addWidget(self.btn)

        self.setLayout(layout_v)
        self.setWindowTitle("Enter Run Info")

        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter), self, activated=self.confirm)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self, activated=self.confirm)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self, activated=self.close)

        recalled = False
        auto_st = 0
        auto_ca = 0
        auto_dn = 0
        try:
            if secure_open.file_exists(self.recall_xml):
                xml_text = ""
                with open(self.recall_xml, 'r') as f: # secure_open(self.recall_xml, 'r') as f:
                    xml_text = f.read()
                doc = minidom.parseString(xml_text)
                params = doc.getElementsByTagName("params")[-1] # most recent element

                for p in params.childNodes:
                    if p.nodeType == p.TEXT_NODE:
                        continue # only process elements

                    name = p.getAttribute("name")
                    value = p.getAttribute("value")

                    if name == "bioformulation":
                        bval = eval(value)
                        self.b1.setChecked(bval)
                        self.b2.setChecked(not bval)
                    # if name == "protein":
                    #     bval = eval(value)
                    #     self.b3.setChecked(bval)
                    #     self.b4.setChecked(not bval)
                    if name == "batch_number":
                        self.batch_found = p.getAttribute("found").lower() == "true"
                        self.t_batch.setText(value)
                    if name == "solvent":
                        self.t0.setText(value)
                        self.lookup_completer() # store auto_st, auto_ca, auto_dn
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

                if len(params.childNodes) == 0:
                    self.q5.setChecked(False) # uncheck "Remember for next time"
                recalled = True
        except:
            Log.e("Failed to recall info from saved file.")

        self.show_hide_gui(None) # enable/disable elements

        # specify auto/manual inputs when recalled data
        if recalled:
            self.auto_st = float(auto_st)
            self.auto_ca = float(auto_ca)
            self.auto_dn = float(auto_dn)

        self.highlight_timer = QtCore.QTimer() # for highlight check
        self.highlight_timer.timeout.connect(self.highlight_manual_entry)
        self.highlight_timer.setSingleShot(True)

        tb_elems = [self.t_batch, self.t0, self.t1, self.t2, self.t3, self.t4, self.t5]
        self.reset_actions = []
        for tb in tb_elems:
            tb.textChanged.connect(self.detect_change)
            if tb in [self.t1, self.t2, self.t5]:
                tb.textChanged.connect(self.queue_highlight_check)
                tb.setClearButtonEnabled(True) # create it, so we can harvest its icon
                self.reset_actions.append(tb.addAction(tb.findChild(QtWidgets.QToolButton).icon(), QtWidgets.QLineEdit.TrailingPosition))
                tb.setClearButtonEnabled(False) # disable it, and never use it again
                self.reset_actions[-1].triggered.connect(tb.clear)
                self.reset_actions[-1].triggered.connect(self.clear_manual_entry)
                # self.reset_actions[-1].hovered.connect(QtWidgets.QToolTip.showText(tb.pos(), "Clear manual entry", tb))
        self.highlight_manual_entry() # run now
        self.g1.buttonClicked.connect(self.detect_change)
        self.q5.stateChanged.connect(self.detect_change)

    def queue_highlight_check(self):
        if self.highlight_timer.isActive():
            self.highlight_timer.stop()
        self.highlight_timer.start(50)

    def clear_manual_entry(self):
        self.highlight_manual_entry(True)

    def highlight_manual_entry(self, force_clear = False):
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
                    if not force_clear: return
                    self.t1.setText("{:3.3f}".format(self.auto_st))
                if len(self.t2.text().strip()) == 0:
                    if not force_clear: return
                    self.t2.setText("{:2.1f}".format(self.auto_ca))
                if len(self.t5.text().strip()) == 0:
                    if not force_clear: return
                    self.t5.setText("{:1.3f}".format(self.auto_dn))
            manual_st = float(self.t1.text()) != float(f"{self.auto_st:3.3f}") if allow_reset else True
            manual_ca = float(self.t2.text()) != float(f"{self.auto_ca:2.1f}") if allow_reset else True
            manual_dn = float(self.t5.text()) != float(f"{self.auto_dn:1.3f}") if allow_reset else True
            self.t1.setStyleSheet("border: 1px solid black;" if manual_st else "background-color: #eee;")
            self.t2.setStyleSheet("border: 1px solid black;" if manual_ca else "background-color: #eee;")
            self.t5.setStyleSheet("border: 1px solid black;" if manual_dn else "background-color: #eee;")
            self.reset_actions[0].setVisible(manual_st if allow_reset else False)
            self.reset_actions[1].setVisible(manual_ca if allow_reset else False)
            self.reset_actions[2].setVisible(manual_dn if allow_reset else False)
        except Exception as e:
            Log.e(f"Invalid parameter: {e}")

    def switch_user_at_sign_time(self):
        from QATCH.common.userProfiles import UserProfiles, UserRoles
        new_username, new_initials, new_userrole = UserProfiles.change(UserRoles.ANALYZE)
        if UserProfiles.check(UserRoles(new_userrole), UserRoles.ANALYZE):
            if self.username != new_username:
                self.username = new_username
                self.initials = new_initials
                self.signedInAs.setText(self.username)
                self.signerInit.setText(f"Initials: <b>{self.initials}</b>")
                self.signature_received = False
                self.signature_required = True
                self.sign.setReadOnly(False)
                self.sign.setMaxLength(4)
                self.sign.clear()

                Log.d("User name changed. Changing sign-in user info.")
                self.parent.ControlsWin.username.setText(f"User: {new_username}")
                self.parent.ControlsWin.userrole = UserRoles(new_userrole)
                self.parent.ControlsWin.signinout.setText("&Sign Out")
                self.parent.ControlsWin.ui1.tool_User.setText(new_username)
                self.parent.AnalyzeProc.tool_User.setText(new_username)
                if self.parent.ControlsWin.userrole != UserRoles.ADMIN:
                    self.parent.ControlsWin.manage.setText("&Change Password...")
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
                PopUp.warning(self, Constants.app_title, "User has not been switched.\n\nReason: Not authenticated.")
            if new_username != None and UserProfiles.session_info()[0]:
                Log.d("User name changed. Changing sign-in user info.")
                self.parent.ControlsWin.username.setText(f"User: {new_username}")
                self.parent.ControlsWin.userrole = UserRoles(new_userrole)
                self.parent.ControlsWin.signinout.setText("&Sign Out")
                self.parent.ControlsWin.ui1.tool_User.setText(new_username)
                self.parent.AnalyzeProc.tool_User.setText(new_username)
                if self.parent.ControlsWin.userrole != UserRoles.ADMIN:
                    self.parent.ControlsWin.manage.setText("&Change Password...")
                PopUp.warning(self, Constants.app_title, "User has not been switched.\n\nReason: Not authorized.")

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
            self.signed_at = dt.datetime.now().isoformat()
            self.signature_received = True

    def text_transform(self):
        text = self.sign.text()
        if len(text) in [1, 2, 3, 4]:
            self.sign.setText(text.upper()) # will not fire 'textEdited' signal again

    def show(self):
        super(QueryRunInfo, self).show()
        width = 350
        height = self.height()
        area = QtWidgets.QDesktopWidget().availableGeometry()
        left = int((area.width() - width) / 2)
        top = int((area.height() - height) / 2)
        self.setGeometry(left, top, width, height)

    def lookup_completer(self):
        try:
            solvent = self.t0.text()
            if solvent in self.fluids:
                idx = self.fluids.index(solvent)
                surface_tension = 72 # self.surface_tensions[idx]
                contact_angle = 55 # 20
                density = self.densities[idx]
                # special_CAs = ["water", "deuterium", "glycerol"]
                # for s in special_CAs:
                #     if solvent.lower().find(s) >= 0:
                #         contact_angle = 55
                self.t1.setText("{:3.3f}".format(surface_tension))
                self.t2.setText("{:2.1f}".format(contact_angle))
                self.t5.setText("{:1.3f}".format(density))
                self.auto_st = float(self.t1.text()) if len(self.t1.text()) else 0 # surface_tension
                self.auto_ca = float(self.t2.text()) if len(self.t2.text()) else 0 # contact_angle
                self.auto_dn = float(self.t5.text()) if len(self.t5.text()) else 0 # density
            else:
                self.t1.clear()
                self.t2.clear()
                self.t5.clear()
        except Exception as e:
            Log.e("ERROR:", e)
            Log.e(f"Failed to lookup parametres for solvent '{self.t0.text()}'.\nPlease try again, or enter parameters manually.")

    def enforce_completer(self):
        solvent = self.t0.text()
        if len(solvent.strip()) < 3: # length of shortest valid solvent string in list
            self.t0.clear()
            return
        if not solvent in self.fluids:
            Log.w(f"Unknown solvent '{self.t0.text()}' entered.\nPlease try again, or enter parameters manually.")
            # self.t0.clear()

    def show_hide_gui(self, object):
        is_bioformulation = None
        if self.b1.isChecked():
            is_bioformulation = True
        if self.b2.isChecked():
            is_bioformulation = False
        if is_bioformulation != True:
            self.t3.clear()
            self.t4.clear()
        else:
            self.t0.clear()

        self.t0.setEnabled(is_bioformulation == False)
        self.t3.setEnabled(is_bioformulation == True)
        self.t4.setEnabled(is_bioformulation == True) #is_protein)

        if object == None:
            self.auto_st = float(self.t1.text()) if len(self.t1.text()) else 0
            self.auto_ca = float(self.t2.text()) if len(self.t2.text()) else 0
            self.auto_dn = float(self.t5.text()) if len(self.t5.text()) else 0
        elif is_bioformulation != None:
            self.calc_params()
        else: # form is blank
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
        # detect when AUDIT says 'found = false' but now it is found
        if self.batch_found != found:
            self.batch_found = found
            self.detect_change()

    def calc_params(self):
        from QATCH.processors.Analyze import AnalyzeProcess

        if self.t3.text() != "0":
            self.t3.setText("0") # Surfactant locked

        try:
            surfactant = float(self.t3.text()) if len(self.t3.text()) else 0
            concentration = float(self.t4.text()) if len(self.t4.text()) else 0
        except Exception as e:
            return

        input_error = False
        if len(self.t3.text()) == 0 or len(self.t4.text()) == 0:
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
        if input_error:
            self.t1.clear()
            self.t2.clear()
            self.t5.clear()
            return

        try:
            #Log.d(f"passing in {surfactant} and {concentration}")
            surface_tension = AnalyzeProcess.Lookup_ST(surfactant, concentration)
            contact_angle = AnalyzeProcess.Lookup_CA(surfactant, concentration)
            density = AnalyzeProcess.Lookup_DN(surfactant, concentration)
            self.t1.setText("{:3.3f}".format(surface_tension))
            self.t2.setText("{:2.1f}".format(contact_angle))
            self.t5.setText("{:1.3f}".format(density))
            self.auto_st = float(self.t1.text()) if len(self.t1.text()) else 0 # surface_tension
            self.auto_ca = float(self.t2.text()) if len(self.t2.text()) else 0 # contact_angle
            self.auto_dn = float(self.t5.text()) if len(self.t5.text()) else 0 # density
        except Exception as e:
            Log.e("ERROR:", e)
            Log.e("Lookup Error: Failed to estimate ST and/or CA.")
            self.t1.clear()
            self.t2.clear()
            self.t5.clear()

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress and obj is self.sign and self.sign.hasFocus():
            if event.key() in [QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return, QtCore.Qt.Key_Space]:
                if self.signature_received:
                    self.sign_ok.clicked.emit()
            if event.key() == QtCore.Qt.Key_Escape:
                self.sign_cancel.clicked.emit()
        return super().eventFilter(obj, event)

    def confirm(self, force=False):
        from QATCH.processors.Analyze import AnalyzeProcess

        surfactant = 0 # float(self.t3.text()) if len(self.t3.text()) else 0
        concentration = float(self.t4.text()) if len(self.t4.text()) else 0
        st = AnalyzeProcess.Lookup_ST(surfactant, concentration)
        ca = float(self.t2.text()) if len(self.t2.text()) else 0
        density = float(self.t5.text()) if len(self.t5.text()) else 0
        manual_st = (st != self.auto_st)
        manual_ca = (ca != self.auto_ca)
        manual_dn = (density != self.auto_dn)

        input_error = False
        if self.t3.isEnabled() and not self.t3.hasAcceptableInput():
            Log.e("Input Error: Surfactant must be between {} and {}."
                .format(
                    self.validSurfactant.bottom(),
                    self.validSurfactant.top()))
            input_error = True
        if self.t4.isEnabled() and not self.t4.hasAcceptableInput():
            Log.e("Input Error: Concentration must be between {} and {}."
                .format(
                    self.validConcentration.bottom(),
                    self.validConcentration.top()))
            input_error = True
        if not self.t1.hasAcceptableInput():
            Log.e("Input Error: Surface Tension must be between {} and {}."
                .format(
                    self.validSurfaceTension.bottom(),
                    self.validSurfaceTension.top()))
            input_error = True
        if not self.t2.hasAcceptableInput():
            Log.e("Input Error: Contact Angle must be between {} and {}."
                .format(
                    self.validContactAngle.bottom(),
                    self.validContactAngle.top()))
            input_error = True
        if not self.t5.hasAcceptableInput():
            Log.e("Input Error: Density must be between {} and {}."
                .format(
                    self.validDensity.bottom(),
                    self.validDensity.top()))
            input_error = True
        if force:
            Log.w("Forcing XML write regardless of input errors!")
            input_error = False
        if input_error:
            Log.w("Input error: Not saving Run Info.")
            return

        if self.signature_required and not self.signature_received: # missing initials
            if force:
                Log.w(f"Auto-signing CAPTURE with initials {self.initials}")
                self.signed_at = dt.datetime.now().isoformat()
            else:
                if not self.batch_found:
                    if not PopUp.question(self, Constants.app_title, 
                                        "Batch Number not found!\nAn invalid Batch Number will lead to less accurate Analyze results.\n\n" +
                                        "Please confirm you entered the correct value and/or\ncheck for updates to your batch parameters resource file.\n\n" +
                                        "Are you sure you want to save this info?", False):
                        return # do not save, allow further changes, user doesn't want to save with invalid Batch Number
                if self.unsaved_changes:
                    if self.signForm.isVisible():
                        self.signForm.hide()
                    self.signedInAs.setText(self.username)
                    self.signerInit.setText(f"Initials: <b>{self.initials}</b>")
                    screen = QtWidgets.QDesktopWidget().availableGeometry()
                    left = int((screen.width() - self.signForm.sizeHint().width()) / 2) + 50
                    top = int((screen.height() - self.signForm.sizeHint().height()) / 2) - 50
                    self.signForm.move(left, top)
                    self.signForm.setVisible(True)
                    self.sign.setFocus()
                    Log.d("Saving Run Info, requesting signature.")
                else:
                    Log.d("Nothing to save, closing Run Info.")
                    self.close() # nothing to save
                return

        if secure_open.file_exists(self.xml_path):
            audit_action = "PARAMS"
            run = minidom.parse(self.xml_path)
            xml = run.documentElement
        else:
            audit_action = "CAPTURE"
            run = minidom.Document()
            xml = run.createElement('run_info')
            run.appendChild(xml)

            xml.setAttribute('machine', Architecture.get_os_name())
            xml.setAttribute('device', FileStorage.DEV_get_active(0)) # TODO: For multiplex, this logs the primary not secondary DEV
            xml.setAttribute('name', self.run_name)
            xml.setAttribute('ruling', self.run_ruling)

            metrics = run.createElement('metrics')
            xml.appendChild(metrics)

            try:
                Log.w(f"run_path: {self.run_path}")
                if self.run_path.endswith(".zip"):
                    Log.e("ZIP file passed as 'run_path' incorrectly!")
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

                start = dt.datetime.strptime(f"{first[0]} {first[1]}", "%Y-%m-%d %H:%M:%S").isoformat()
                stop = dt.datetime.strptime(f"{last[0]} {last[1]}", "%Y-%m-%d %H:%M:%S").isoformat()
                duration = float(last[2])
                duration_units = "seconds"
                if duration > 60.0:
                    duration /= 60.0
                    duration_units = "minutes"
                samples = str(samples)
                Log.d(f"{start}, {stop}, {duration}, {samples}")

                # Get time of last cal - based on file timestamp
                cal_file_path = Constants.cvs_peakfrequencies_path
                cal_file_path = FileStorage.DEV_populate_path(cal_file_path, 0)
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
            except Exception as e:
                raise e
                Log.e("Metrics Error: Failed to open/parse CSV file for XML file run info metrics.")

        # create or append new audits element
        try:
            audits = xml.getElementsByTagName('audits')[-1]
        except:
            audits = run.createElement('audits')
            xml.appendChild(audits)

        # create or append new params element
        recorded_at = self.signed_at if self.signature_required else dt.datetime.now().isoformat()
        params = run.createElement('params')
        params.setAttribute('recorded', recorded_at)
        xml.appendChild(params)

        param1 = run.createElement('param')
        param1.setAttribute('name', 'bioformulation')
        param1.setAttribute('value', str(self.b1.isChecked()))
        params.appendChild(param1)

        param_batch = run.createElement('param')
        param_batch.setAttribute('name', 'batch_number')
        param_batch.setAttribute('value', self.t_batch.text())
        param_batch.setAttribute('found', str(self.batch_found))
        params.appendChild(param_batch)

        if self.b2.isChecked(): # is NOT bioformulation
            param2 = run.createElement('param')
            param2.setAttribute('name', 'solvent')
            param2.setAttribute('value', self.t0.text())
            param2.setAttribute('input', 'auto' if self.t0.text() in self.fluids else 'manual')
            params.appendChild(param2)

        if self.b1.isChecked(): # IS bioformulation
            param3 = run.createElement('param')
            param3.setAttribute('name', 'surfactant')
            param3.setAttribute('value', "{0:0.{1}f}".format(surfactant, self.validSurfactant.decimals()))
            param3.setAttribute('units', 'mg/mL')
            param3.setAttribute('input', 'manual')
            params.appendChild(param3)

            param4 = run.createElement('param')
            param4.setAttribute('name', 'concentration')
            param4.setAttribute('value', "{0:0.{1}f}".format(concentration, self.validConcentration.decimals()))
            param4.setAttribute('units', '%w')
            param4.setAttribute('input', 'manual')
            params.appendChild(param4)

        param5 = run.createElement('param')
        param5.setAttribute('name', 'surface_tension')
        param5.setAttribute('value', "{0:0.{1}f}".format(st, self.validSurfaceTension.decimals()))
        param5.setAttribute('units', 'mN/m')
        param5.setAttribute('input', 'manual' if manual_st else 'auto')
        params.appendChild(param5)

        param6 = run.createElement('param')
        param6.setAttribute('name', 'contact_angle')
        param6.setAttribute('value', "{0:0.{1}f}".format(ca, self.validContactAngle.decimals()))
        param6.setAttribute('units', 'degrees')
        param6.setAttribute('input', 'manual' if manual_ca else 'auto')
        params.appendChild(param6)

        param7 = run.createElement('param')
        param7.setAttribute('name', 'density')
        param7.setAttribute('value', "{0:0.{1}f}".format(density, self.validDensity.decimals()))
        param7.setAttribute('units', 'g/cm^3')
        param7.setAttribute('input', 'manual' if manual_dn else 'auto')
        params.appendChild(param7)

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

        if self.signature_required:
            from QATCH.common.userProfiles import UserProfiles
            valid, infos = UserProfiles.session_info()
            if valid:
                Log.d(f"Found valid session: {infos}")
                username = infos[0]
                initials = infos[1]
                salt = UserProfiles.find(username, initials)[1][:-4]
                userrole = infos[2]
            else:
                Log.w(f"Found invalid session: searching for user ({self.username}, {self.initials})")
                username = self.username
                initials = self.initials
                salt = UserProfiles.find(username, initials)[1][:-4]
                userrole = UserProfiles.get_user_info(f"{salt}.xml")[2]

            timestamp = self.signed_at
            machine = Architecture.get_os_name()
            hash = hashlib.sha256()
            hash.update(salt.encode()) # aka 'profile'
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
            pass # leave 'audits' block as empty

        os.makedirs(os.path.split(self.xml_path)[0], exist_ok=True)

        with open(self.xml_path, 'w') as f: # secure_open(self.xml_path, 'w', "audit") as f:
            xml_str = run.toxml() #.encode() #prettyxml(indent ="\t")
            f.write(xml_str)
            Log.i(f"Created XML file: {self.xml_path}")

        if self.q5.isEnabled():
            run = minidom.Document()
            xml = run.createElement('run_info')
            xml.setAttribute('name', 'recall')
            run.appendChild(xml)
            if self.q5.isChecked(): # remember for next time
                Log.i("Run info remembered for next time.")
            else:
                params = run.createElement('params') # blank it
            xml.appendChild(params)
            with open(self.recall_xml, 'w') as f: # secure_open(self.recall_xml, 'w') as f:
                f.write(run.toxml())

        self.unsaved_changes = False
        self.close()

    def closeEvent(self, event):
        if self.unsaved_changes:
            res =PopUp.question(self, Constants.app_title, "You have unsaved changes!\n\nAre you sure you want to close this window?", False)
            if res:
                if self.post_run:
                    try:
                        self.confirm(force=True)
                    except Exception as e:
                        Log.e(e)
                self.finished.emit()
            else:
                event.ignore()
        else: # closing, with no changes
            if not self.batch_found: # invalid Batch Number
                if not PopUp.question(self, Constants.app_title, 
                                    "Batch Number not found!\nAn invalid Batch Number will lead to less accurate Analyze results.\n\n" +
                                    "Please confirm you entered the correct value and/or\ncheck for updates to your batch parameters resource file.\n\n" +
                                    "Are you sure you want to close without updating this info?", False):
                    event.ignore() # do not close, allow further changes, user doesn't want to close with invalid Batch Number
                    return
            self.finished.emit()
