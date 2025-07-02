try:
    from QATCH.ui.popUp import PopUp
    from QATCH.core.constants import Constants
    from QATCH.common.userProfiles import UserProfiles, UserRoles
    from QATCH.common.logger import Logger as Log
    from QATCH.common.architecture import Architecture
except:
    print("Running VisQAI as standalone app")

    class Log:
        def d(tag, msg=""): print("DEBUG:", tag, msg)
        def i(tag, msg=""): print("INFO:", tag, msg)
        def w(tag, msg=""): print("WARNING:", tag, msg)
        def e(tag, msg=""): print("ERROR:", tag, msg)

from xml.dom import minidom
from numpy import loadtxt
from PyQt5 import QtCore, QtGui, QtWidgets
from random import randint
import copy
import os
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
import hashlib
from scipy.optimize import curve_fit
import datetime as dt
from types import SimpleNamespace
import webbrowser
from PyQt5.QtPrintSupport import QPrinter
from scipy.interpolate import interp1d

try:
    from src.io.file_storage import SecureOpen
    from src.models.formulation import Formulation, ViscosityProfile
    from src.models.ingredient import Protein, Surfactant, Stabilizer, Salt, Buffer
    from src.models.predictor import Predictor
    from src.controller.formulation_controller import FormulationController
    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database, DB_PATH
    from src.processors.sampler import Sampler
    from src.threads.executor import Executor, ExecutionRecord
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.io.file_storage import SecureOpen
    from QATCH.VisQAI.src.models.formulation import Formulation, ViscosityProfile
    from QATCH.VisQAI.src.models.ingredient import Protein, Surfactant, Stabilizer, Salt, Buffer
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.controller.formulation_controller import FormulationController
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.db.db import Database, DB_PATH
    from QATCH.VisQAI.src.processors.sampler import Sampler
    from QATCH.VisQAI.src.threads.executor import Executor, ExecutionRecord
TAG = "[VisQ.AI]"


class HorizontalTabBar(QtWidgets.QTabBar):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        font = self.font()
        font.setPointSize(10)  # Set desired font size
        self.setFont(font)

    def tabSizeHint(self, index):
        sz = super().tabSizeHint(index)
        return QtCore.QSize(sz.width() + 20, 90)  # fixed height

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


class BaseVisQAIWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        """BASE CLASS DEFINITION"""
        super().__init__(parent)

        self.setWindowTitle("VisQ.AI Base Class")
        self.setMinimumSize(900, 600)

        # Create dummy database object for base class.
        self.database = SimpleNamespace(is_open=False)

        # Create dummy UI for main tab widget
        self.tab_widget = QtWidgets.QDialog()

        # Create simple UI for base class when not subscribed.
        self._expired_widget = QtWidgets.QDialog()
        self.expired_layout = QtWidgets.QVBoxLayout(self._expired_widget)
        self.expired_layout.setAlignment(QtCore.Qt.AlignCenter)
        # self.setCentralWidget(self._expired_widget)

        self.expired_label = QtWidgets.QLabel(
            "<b>Your VisQ.AI preview period has ended.</b><br/><br/>" +
            "Please subscribe to regain access on this system.<br/><br/>")
        self.expired_label.setAlignment(QtCore.Qt.AlignCenter)
        self.expired_label.setStyleSheet("font-size: 24px;")

        self.expired_subscribe = QtWidgets.QPushButton("Subscribe")
        self.expired_subscribe.clicked.connect(
            lambda: webbrowser.open("https://qatchtech.com"))

        self.expired_learnmore = QtWidgets.QPushButton("Learn more...")
        self.expired_learnmore.clicked.connect(
            lambda: webbrowser.open("https://qatchtech.com"))

        self.expired_buttons = QtWidgets.QHBoxLayout()
        self.expired_buttons.addStretch(3)
        self.expired_buttons.addWidget(self.expired_subscribe, 1)
        self.expired_buttons.addWidget(self.expired_learnmore, 1)
        self.expired_buttons.addStretch(3)

        self.expired_layout.addWidget(self.expired_label)
        self.expired_layout.addLayout(self.expired_buttons)

        # Create simple UI for base class when not subscribed.
        self._trial_widget = QtWidgets.QDialog()
        self.trial_layout = QtWidgets.QVBoxLayout(self._trial_widget)
        self.trial_layout.setAlignment(QtCore.Qt.AlignCenter)
        # self.setCentralWidget(self._trial_widget)

        self.trial_label = QtWidgets.QLabel(
            "<b>Your VisQ.AI preview has {} days remaining.</b><br/><br/>" +
            "Please subscribe to retain access on this system.<br/><br/>")
        self.trial_label.setAlignment(QtCore.Qt.AlignCenter)
        self.trial_label.setStyleSheet("font-size: 24px;")

        self.trial_subscribe = QtWidgets.QPushButton("Subscribe")
        self.trial_subscribe.clicked.connect(
            lambda: webbrowser.open("https://qatchtech.com"))

        self.trial_dismiss = QtWidgets.QPushButton("Dismiss")
        self.trial_dismiss.clicked.connect(
            lambda: self.setCentralWidget(self.tab_widget))

        self.trial_buttons = QtWidgets.QHBoxLayout()
        self.trial_buttons.addStretch(3)
        self.trial_buttons.addWidget(self.trial_subscribe, 1)
        self.trial_buttons.addWidget(self.trial_dismiss, 1)
        self.trial_buttons.addStretch(3)

        self.trial_layout.addWidget(self.trial_label)
        self.trial_layout.addLayout(self.trial_buttons)

    def check_license(self) -> bool:
        free_preview_period = 90  # days
        # TODO: dummy check, always false for now
        is_valid_license = False
        if not is_valid_license:
            # how long ago did the preview period start?
            if not os.path.exists(DB_PATH):
                Log.e("No VisQAI license or trial found.")
                self.setCentralWidget(self._expired_widget)
                return is_valid_license

            file_stats = os.stat(DB_PATH)

            # Get creation time (st_ctime on Unix is actually change time,
            # but on Windows it's creation time)
            # For true creation time on all platforms, use st_birthtime if available
            if hasattr(file_stats, 'st_birthtime'):
                creation_time = file_stats.st_birthtime  # macOS/BSD
            else:
                creation_time = file_stats.st_ctime  # Windows/Linux

            # Rollback to midnight UTC, day of file creation
            creation_time -= creation_time % 86400

            # Get local time with timezone info
            local_time = dt.datetime.now().astimezone()
            utc_offset = local_time.utcoffset()

            # Adjust file creation time to midnight local time
            creation_time -= utc_offset.total_seconds()

            # Current time
            current_time = dt.datetime.now().timestamp()

            # Calculate time difference in seconds
            time_ago_seconds = current_time - creation_time

            # Compare to allowable age
            time_allowed_secs = dt.timedelta(
                days=free_preview_period).total_seconds()

            if time_ago_seconds >= time_allowed_secs:
                # Trial preview expired
                Log.w("No VisQAI license found; trial preview has expired.")
                self.setCentralWidget(self._expired_widget)
            else:
                self.trial_left = free_preview_period - \
                    dt.timedelta(seconds=time_ago_seconds).days
                Log.i(
                    f"No VisQAI license found; trial preview has {self.trial_left} days remaining.")
                self.trial_label.setText(
                    self.trial_label.text().format(self.trial_left))  # insert # of days remaining
                self.setCentralWidget(self._trial_widget)
        else:
            # valid license found
            Log.i("VisQAI license found and is valid.")
            self.setCentralWidget(self.tab_widget)

        return is_valid_license

    def clear(self):
        """BASE CLASS DEFINITION"""
        pass

    def reset(self):
        """BASE CLASS DEFINITION"""
        pass

    def enable(self, bool=False):
        """BASE CLASS DEFINITION"""
        pass

    def hasUnsavedChanges(self):
        """BASE CLASS DEFINITION"""
        return False


class VisQAIWindow(BaseVisQAIWindow):
    def __init__(self, parent=None):
        super(VisQAIWindow, self).__init__(parent)

        # import typing here to avoid circularity
        from QATCH.ui.mainWindow import MainWindow
        self.parent: MainWindow = parent
        self.setWindowTitle("VisQ.AI Mockup")
        self.setMinimumSize(900, 600)
        self.init_ui()
        self.init_sign()
        self.check_license()  # see BASE CLASS
        self._unsaved_changes = False

        self.select_formulation = Formulation()
        self.predict_formulation = Formulation()

    def init_ui(self):
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setTabBar(HorizontalTabBar())
        self.tab_widget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tab_widget.tabBar().installEventFilter(self)

        # Enable database objects for initial UI load.
        self.enable(True)

        self.tab_widget.addTab(FrameStep1(self, 1),
                               "\u2460 Select Run")  # unicode circled 1
        self.tab_widget.addTab(FrameStep1(self, 2),
                               "\u2461 Suggest Experiments")  # unicode circled 2
        self.tab_widget.addTab(FrameStep1(self, 3),
                               "\u2462 Import Experiments")  # unicode circled 3
        self.tab_widget.addTab(FrameStep2(self, 4),
                               "\u2463 Learn")  # unicode circled 4
        self.tab_widget.addTab(FrameStep1(self, 5),
                               "\u2464 Predict")  # unicode circled 5
        self.tab_widget.addTab(FrameStep2(self, 6),
                               "\u2465 Optimize")  # unicode circled 6

        # Disable database objects after initial UI load.
        self.enable(False)

        # NOTE: central widget set by `check_license()`

        # Signals
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def init_sign(self):
        # START VISQAI SIGNATURE CODE:
        # This code also exists in runInfo.py in class QueryRunInfo for "CAPTURE SIGNATURE CODE"
        # This code also exists in Analyze.py in class VisQAIWindow for "ANALYZE SIGNATURE CODE"
        # The following method also is duplicated in both files: 'self.switch_user_at_sign_time'
        # There is duplicated logic code within the submit button handler: 'self.save_run_infos'
        # The method for handling keystroke shortcuts is also duplicated too: 'self.eventFilter'
        self.signForm = QtWidgets.QDialog()
        self.signForm.setWindowFlags(
            QtCore.Qt.Dialog
        )  # | QtCore.Qt.WindowStaysOnTopHint)
        icon_path = os.path.join(
            Architecture.get_path(), "QATCH/icons/sign.png")
        self.signForm.setWindowIcon(QtGui.QIcon(icon_path))  # .png
        self.signForm.setWindowTitle("Signature")
        self.signForm.setModal(True)
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
        self.sign = QtWidgets.QLineEdit()
        self.sign.installEventFilter(self)
        layout_sign.addWidget(self.sign)
        self.sign_do_not_ask = QtWidgets.QCheckBox(
            "Do not ask again this session")
        self.sign_do_not_ask.setEnabled(False)
        if UserProfiles.checkDevMode()[0]:  # DevMode enabled
            auto_sign_key = None
            session_key = None
            if os.path.exists(Constants.auto_sign_key_path):
                with open(Constants.auto_sign_key_path, "r") as f:
                    auto_sign_key = f.readline()
            session_key_path = os.path.join(
                Constants.user_profiles_path, "session.key")
            if os.path.exists(session_key_path):
                with open(session_key_path, "r") as f:
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
        # self.sign_ok.clicked.connect(self.save_run_info)
        self.sign_ok.setDefault(True)
        self.sign_ok.setAutoDefault(True)
        self.sign_cancel = QtWidgets.QPushButton("Cancel")
        self.sign_cancel.clicked.connect(self.signForm.hide)
        layout_ok_cancel = QtWidgets.QHBoxLayout()
        layout_ok_cancel.addWidget(self.sign_ok)
        layout_ok_cancel.addWidget(self.sign_cancel)
        layout_sign.addLayout(layout_ok_cancel)
        self.signForm.setLayout(layout_sign)
        # END ANALYZE SIGNATURE CODE

        self.sign.textEdited.connect(self.sign_edit)
        self.sign.textEdited.connect(self.text_transform)

    def eventFilter(self, obj, event):
        # Key press on user audit sign form
        if (
            event.type() == QtCore.QEvent.KeyPress
            and obj is self.sign
            and self.sign.hasFocus()
        ):
            if event.key() in [
                QtCore.Qt.Key_Enter,
                QtCore.Qt.Key_Return,
                QtCore.Qt.Key_Space,
            ]:
                if self.parent.signature_received:
                    self.sign_ok.clicked.emit()
            if event.key() == QtCore.Qt.Key_Escape:
                self.sign_cancel.clicked.emit()
        # Mouse click on tab widget tab bar
        if (
            event.type() == QtCore.QEvent.MouseButtonPress
            and obj == self.tab_widget.tabBar()
        ):
            now_step = self.tab_widget.currentIndex() + 1
            tab_step = obj.tabAt(event.pos()) + 1
            if tab_step > 0:
                if tab_step == now_step + 1:
                    if hasattr(self.tab_widget.currentWidget(), "btn_next"):
                        self.tab_widget.currentWidget().btn_next.click()
                        return True  # ignore click, let "Next" btn decide
                # Block tab change based on some condition
                if tab_step in [2, 4, 5]:
                    if tab_step == 2 or tab_step == 5:
                        widget: FrameStep1 = self.tab_widget.widget(
                            0)  # Select
                        if not widget.run_file_run:
                            QtWidgets.QMessageBox.information(
                                None, Constants.app_title,
                                "Please select a run.",
                                QtWidgets.QMessageBox.Ok)
                            return True  # deny tab change
                    if tab_step == 4:
                        widget: FrameStep1 = self.tab_widget.widget(
                            2)  # Import
                        if len(widget.all_files) == 0:
                            QtWidgets.QMessageBox.information(
                                None, Constants.app_title,
                                "Please import at least 1 experiment before proceeding.",
                                QtWidgets.QMessageBox.Ok)
                            return True  # deny tab change
                if hasattr(self.tab_widget.currentWidget(), "btn_next"):
                    self.tab_widget.currentWidget().btn_next.click()
                    # still perform click action
        return super().eventFilter(obj, event)

    def sign_edit(self):
        if self.sign.text().upper() == self.initials:
            sign_text = f"{self.username} ({self.sign.text().upper()})"
            self.sign.setMaxLength(len(sign_text))
            self.sign.setText(sign_text)
            self.sign.setReadOnly(True)
            self.parent.signed_at = dt.datetime.now().isoformat()
            self.parent.signature_received = True
            self.sign_do_not_ask.setEnabled(True)

    def text_transform(self):
        text = self.sign.text()
        if len(text) in [1, 2, 3, 4]:  # are these initials?
            # will not fire 'textEdited' signal again
            self.sign.setText(text.upper())

    def check_user_info(self):
        # get active session info, if available
        active, info = UserProfiles.session_info()
        if active:
            self.parent.signature_required = True
            self.parent.signature_received = False
            self.username, self.initials = info[0], info[1]
        else:
            self.parent.signature_required = False

    def switch_user_at_sign_time(self):
        new_username, new_initials, new_userrole = UserProfiles.change(
            UserRoles.ANALYZE
        )
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
                Log.d(
                    "User switched users to the same user profile. Nothing to change."
                )
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
                PopUp.warning(
                    self,
                    Constants.app_title,
                    "User has not been switched.\n\nReason: Not authenticated.",
                )
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
                PopUp.warning(
                    self,
                    Constants.app_title,
                    "User has not been switched.\n\nReason: Not authorized.",
                )

            Log.d("User did not authenticate for role to switch users.")

    def on_tab_changed(self, index):
        # Purge dataase to disk on tab change
        if self.database.is_open:
            self.database.backup()
        elif not isinstance(self.database, SimpleNamespace):
            Log.w("Database closed: backup failed")

        # Get the current widget and call it's select handler (if exists)
        current_widget = self.tab_widget.widget(index)
        if hasattr(current_widget, 'on_tab_selected') and callable(current_widget.on_tab_selected):
            current_widget.on_tab_selected()

    def clear(self) -> None:
        self._unsaved_changes = False

    def hasUnsavedChanges(self) -> bool:
        return self._unsaved_changes

    def reset(self) -> None:
        self.check_user_info()
        self.signedInAs.setText(self.username)
        self.signerInit.setText(f"Initials: <b>{self.initials}</b>")
        self.parent.signature_received = False
        self.parent.signature_required = True
        self.sign.setReadOnly(False)
        self.sign.setMaxLength(4)
        self.sign.clear()

    def enable(self, enable=False) -> None:
        if not enable:
            # VisQ.AI UI is not in foreground, Mode not selected
            # Do things here to shutdown resources and disable:

            # Close database.
            if self.database.is_open:
                self.database.close()
            elif not isinstance(self.database, SimpleNamespace):
                Log.w("Database closed: write failed")

            # Disable database objects.
            self.database = SimpleNamespace(is_open=False, status="Disabled")
            self.form_ctrl = SimpleNamespace(db=None, status="Disabled")
            self.ing_ctrl = SimpleNamespace(db=None, status="Disabled")
            Log.d("Database objects disabled on VisQ.AI not enabled.")

        else:
            # VisQ.AI UI is now in foreground, Mode is selected
            # Do things here to initialize resources and enable:

            # Create database objects, and open DB from file.
            self.database = Database(parse_file_key=True)
            self.form_ctrl = FormulationController(db=self.database)
            self.ing_ctrl = IngredientController(db=self.database)
            Log.d("Database objects created on VisQ.AI enable.")

            # Emit tab selected code for the currently active tab frame.
            self.tab_widget.currentChanged.emit(self.tab_widget.currentIndex())

            # # Create default user preferences object
            # UserProfiles.user_preferences = UserPreferences(
            #     UserProfiles.get_session_file())
            # prefs = UserProfiles.user_preferences.get_preferences()
            # self.load_data_path = prefs['load_data_path']

    def save_run_info(self, xml_path: str, run_info: list, cancel: bool = False):
        # This will update the run info XML only if there are changes.
        # The user may be asked for an audit signature if required.

        info_tags = ['protein_type', 'protein_concentration',
                     'buffer_type', 'buffer_concentration',
                     'surfactant_type', 'surfactant_concentration',
                     'stabilizer_type', 'stabilizer_concentration',
                     'salt_type', 'salt_concentration']
        required_len = len(info_tags)
        if len(run_info) != required_len:
            Log.e(
                f"There must be {required_len} run info parameters given. Received {len(run_info)}")
            return
        if not os.path.exists(xml_path):
            Log.e(f"XML path not found: {xml_path}")
            return

        run = minidom.parse(xml_path)
        xml = run.documentElement

        existing_params = []
        params = xml.getElementsByTagName(
            "params")[-1]  # most recent element
        params = params.cloneNode(deep=True)
        for p in params.childNodes:
            if p.nodeType == p.TEXT_NODE:
                continue  # only process elements
            name = p.getAttribute("name")
            existing_params.append(name)
            if name in info_tags:
                i = info_tags.index(name)
                value = run_info[i]
                if p.getAttribute("value") != value:
                    p.setAttribute("value", value)
                    self._unsaved_changes = True
        for i in range(required_len):
            name = info_tags[i]
            if name not in existing_params:
                value = run_info[i]
                p = run.createElement('param')
                p.setAttribute('name', name)
                p.setAttribute('value', value)
                params.appendChild(p)
                self._unsaved_changes = True

        if not self._unsaved_changes:
            Log.d("No changes detected, not appending new run info to XML.")
            return
        elif cancel:
            Log.d("User canceling with unsaved changes in table.")
            result = QtWidgets.QMessageBox.question(
                None,
                Constants.app_title,
                "You have unsaved changes!\n\nAre you sure you want to cancel without saving?")
            if result == QtWidgets.QMessageBox.Yes:
                self._unsaved_changes = False
                return

        # Get audit signature from authorized user.
        if self.parent.signature_required and self._unsaved_changes:
            if self.parent.signature_received == False and self.sign_do_not_ask.isChecked():
                Log.w(
                    f"Signing ANALYZE with initials {self.initials} (not asking again)"
                )
                self.parent.signed_at = dt.datetime.now().isoformat()
                self.parent.signature_received = True  # Do not ask again this session
            if not self.parent.signature_received:
                if self.signForm.isVisible():
                    self.signForm.hide()
                self.signedInAs.setText(self.username)
                self.signerInit.setText(f"Initials: <b>{self.initials}</b>")
                screen = QtWidgets.QDesktopWidget().availableGeometry()
                left = int(
                    (screen.width() - self.signForm.sizeHint().width()) / 2) + 50
                top = (
                    int((screen.height() - self.signForm.sizeHint().height()) / 2) - 50
                )
                self.signForm.move(left, top)
                self.signForm.setVisible(True)
                self.sign.setFocus()
                self.signForm.exec_()
                if not self.parent.signature_received:
                    Log.w("User did not sign when requested.")
                    return

        if self.sign_do_not_ask.isChecked():
            session_key_path = os.path.join(
                Constants.user_profiles_path, "session.key")
            if os.path.exists(session_key_path):
                with open(session_key_path, "r") as f:
                    session_key = f.readline()
                if not os.path.exists(Constants.auto_sign_key_path):
                    with open(Constants.auto_sign_key_path, "w") as f:
                        f.write(session_key)

        if self.parent.signature_required:
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

            audit_action = "VISQAI"
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

            audits = xml.getElementsByTagName('audits')[-1]
            audits.appendChild(audit1)
        else:
            pass  # leave 'audits' block as empty

        hash = hashlib.sha256()
        params.setAttribute('recorded', timestamp)
        for p in params.childNodes:
            for name, value in p.attributes.items():
                hash.update(name.encode())
                hash.update(value.encode())
        signature = hash.hexdigest()
        params.setAttribute('signature', signature)

        xml.appendChild(params)

        with open(xml_path, 'w') as f:
            f.write(run.toxml())

        self._unsaved_changes = False


class FrameStep1(QtWidgets.QDialog):
    def __init__(self, parent=None, step=1):
        super().__init__(parent)
        self.parent: VisQAIWindow = parent
        self.step = step

        self.all_files = {}
        self.model_path = None
        self.run_file_run = None
        self.run_file_xml = None
        self.run_file_analyze = None

        self.profile_shears = [1e2, 1e3, 1e4, 1e5, 15000000]
        self.profile_viscos = []

        if step == 1:
            self.setWindowTitle("Select Run")
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

        # Left panel: Run selection
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        form_layout = QtWidgets.QFormLayout()
        if step == 1:
            left_group = QtWidgets.QGroupBox("Select Run")
        elif step == 2:
            left_group = QtWidgets.QGroupBox("Suggested Runs")
        elif step == 3:
            left_group = QtWidgets.QGroupBox("Experiment Runs")
        elif step == 5:
            left_group = QtWidgets.QGroupBox("Predictions")
        left_group_layout = QtWidgets.QVBoxLayout(left_group)
        left_group_layout.addLayout(form_layout)

        # Select model (for step 5: Predict)
        if True:  # step == 5:
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
            select_model_layout = QtWidgets.QHBoxLayout(
                self.select_model_group)
            self.select_model_btn = QtWidgets.QPushButton("Browse...")
            self.select_model_label = QtWidgets.QLineEdit()
            self.select_model_label.setPlaceholderText("No model selected")
            self.select_model_label.setReadOnly(True)
            if step == 1:
                predictor_path = "QATCH/VisQAI/assets/VisQAI-base.zip"
                if os.path.exists(predictor_path):
                    self.model_selected(path=predictor_path)
            select_model_layout.addWidget(self.select_model_btn)
            select_model_layout.addWidget(self.select_model_label)
            select_model_layout.addStretch()

            left_layout.addWidget(self.select_model_group)

        left_layout.addWidget(left_group)

        # Browse run
        self.file_dialog = QtWidgets.QFileDialog()
        self.file_dialog.setOption(
            QtWidgets.QFileDialog.DontUseNativeDialog, True)
        # NOTE: `setDirectory()` called when VisQAI mode is enabled.
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self.file_dialog.setNameFilter("Captured Runs (capture.zip)")
        self.file_dialog.selectNameFilter("Captured Runs (capture.zip)")

        self.select_run = QtWidgets.QPushButton(
            "Add Run..." if step == 3 else "Browse...")
        self.select_label = QtWidgets.QLineEdit()
        self.select_label.setPlaceholderText("No run selected")
        self.select_label.setReadOnly(True)
        # run_select = QtWidgets.QHBoxLayout()
        # run_select.addWidget(self.select_run)
        # run_select.addWidget(self.select_label)

        if step == 1:
            form_layout.addRow(self.select_run, self.select_label)
        elif step == 2 or step == 3 or step == 5:
            self.list_view = QtWidgets.QListView()
            self.list_view.setEditTriggers(
                QtWidgets.QAbstractItemView.NoEditTriggers)
            self.model = QtGui.QStandardItemModel()
            string_list = []
            if False:  # step == 2 or step == 5:
                for i in range(4):
                    base_name = "Suggestion" if step == 2 else "Prediction"
                    string_list.append(f"{base_name} {i+1}")
            for string in string_list:
                self.model.appendRow(QtGui.QStandardItem(string))
            self.list_view_addPlaceholderText()
            self.list_view.setModel(self.model)
            if step == 1 or step == 3:
                form_layout.addRow(self.select_run, self.list_view)
            elif step == 2:
                form_layout.addRow("Experiment:", self.list_view)
            elif step == 5:
                form_layout.addRow("Prediction:", self.list_view)
            if step == 3:
                self.list_view.clicked.connect(self.user_run_clicked)
            else:
                # For steps 2 and 5, pull suggestions/predictions from model
                self.list_view.clicked.connect(
                    lambda: self.feature_table.setData(self.loaded_features[self.list_view.selectedIndexes()[0].row()]) if len(self.loaded_features) else None)
                self.list_view.clicked.connect(self.hide_extended_features)
            self.list_view.clicked.connect(
                lambda: self.btn_update.setEnabled(True))

            add_remove_export_widget = QtWidgets.QWidget()
            add_remove_export_layout = QtWidgets.QHBoxLayout(
                add_remove_export_widget)
            add_remove_export_layout.setContentsMargins(0, 0, 0, 0)
            if step in [2, 5]:
                btn_text = "Suggestion" if step == 2 else "Prediction"
                self.btn_add = QtWidgets.QPushButton(f"Add {btn_text}")
                self.btn_add.clicked.connect(self.add_another_item)
                add_remove_export_layout.addWidget(self.btn_add)
            self.btn_remove = QtWidgets.QPushButton("Remove Selected Run")
            self.btn_remove.clicked.connect(self.user_run_removed)
            add_remove_export_layout.addWidget(self.btn_remove)
            if step in [2, 5]:  # Suggest, Predict
                self.btn_export = QtWidgets.QPushButton("Export as PDF")
                self.btn_export.clicked.connect(self.export_table_data)
                add_remove_export_layout.addWidget(self.btn_export)
            form_layout.addRow("", add_remove_export_widget)

        self.run_notes = QtWidgets.QTextEdit()
        self.run_notes.setPlaceholderText("None")
        self.run_notes.setReadOnly(True)

        # Run information
        self.run_name = QtWidgets.QLabel()
        self.run_date_time = QtWidgets.QLabel()
        self.run_duration = QtWidgets.QLabel()
        self.run_temperature = QtWidgets.QLabel()
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
            form_layout.addRow("Temperature (avg):", self.run_temperature)
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
                "Next Step: Learn")
        elif step == 5:
            self.btn_next = QtWidgets.QPushButton(""
                                                  "Next Step: Optimize")
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
        self.load_all_excipient_types()
        self.default_features = {"Feature": ["Protein Type", "Protein Concentration",
                                             "Protein Molecular Weight", "Protein pI Mean", "Protein pI Range",  # not in Run Info
                                             "Buffer Type", "Buffer Concentration",
                                             "Buffer pH",  # not in Run Info
                                             "Surfactant Type", "Surfactant Concentration",
                                             "Stabilizer Type", "Stabilizer Concentration",
                                             "Salt Type", "Salt Concentration"],
                                 "Value": [self.proteins, "",
                                           "", "", "",  # molecular weight, pI mean, pI range
                                           self.buffers, "",
                                           "",  # buffer pH
                                           self.surfactants, "",
                                           self.stabilizers, "",
                                           self.salts, ""],
                                 "Units": ["", "mg/mL",
                                           "kDa", "", "",  # pI
                                           "", "mM",
                                           "",  # pH
                                           "", "%w",
                                           "", "M",
                                           "", "mM"]}
        self.default_rows, self.default_cols = (len(list(self.default_features.values())[0]),
                                                len(list(self.default_features.keys())))

        self.feature_table = TableView(self.default_features,
                                       self.default_rows, self.default_cols)
        self.feature_table.clear()
        right_group.addWidget(self.feature_table)

        self.btn_update = QtWidgets.QPushButton()
        self.btn_update.setEnabled(False)
        if step == 1:  # Select
            self.btn_update.setText("Save Formulation")
            self.btn_update.clicked.connect(self.save_formulation)
        if step == 2:  # Suggest
            self.btn_update.setText("Decline Suggestion")
            self.btn_update.clicked.connect(self.user_run_removed)
        if step == 3:  # Import
            self.btn_update.setText("Save Formulation")
            self.btn_update.clicked.connect(self.save_formulation)
        # step 4 is not in this class: Learn
        if step == 5:  # Predict
            self.btn_update.setText("Update Prediction")
            self.btn_update.clicked.connect(self.make_predictions)
        right_group.addWidget(self.btn_update)

        self.loaded_features = []

        # # Testing only, create dummy features
        # self.dummy_features = []
        # for i in range(4):
        #     dummy_feature = copy.deepcopy(self.default_features)
        #     value_tags = [0, range(5, 95),
        #                   0, 0, 0,
        #                   range(3), range(5, 95),
        #                   0,
        #                   range(2), range(5, 95),
        #                   range(2), range(5, 95),
        #                   0, range(5, 95)]
        #     for x in range(len(dummy_feature["Value"])):
        #         try:
        #             current_value = dummy_feature["Value"][x]
        #             current_tag = value_tags[x]
        #             if isinstance(current_value, list):
        #                 if isinstance(current_tag, int):
        #                     dummy_feature["Value"][x] = \
        #                         current_value[current_tag]
        #                 else:
        #                     dummy_feature["Value"][x] = current_value[randint(
        #                         current_tag[0], current_tag[-1])]
        #             else:
        #                 if isinstance(current_tag, range):
        #                     dummy_feature["Value"][x] = randint(
        #                         current_tag[0], current_tag[-1])
        #         except Exception as e:
        #             print(e)
        #     # Hide protein and buffer characteristics
        #     # for values in dummy_feature.values():
        #     #     del values[7]  # buffer PH
        #     #     del values[4]  # protein pI range
        #     #     del values[3]  # protein pI mean
        #     #     del values[2]  # protein weight
        #     self.dummy_features.append(dummy_feature)

        self.run_figure = Figure()
        self.run_figure_valid = False
        self.run_canvas = FigureCanvas(self.run_figure)
        v_splitter.addWidget(self.run_canvas)

        # Build main layout
        h_splitter.addWidget(left_widget)
        h_splitter.addWidget(right_widget)
        h_splitter.setSizes([100, 300])
        v_splitter.setSizes([180, 100])

        # Signals
        self.btn_cancel.clicked.connect(
            lambda: self.file_selected(None, cancel=True))
        self.btn_next.clicked.connect(
            getattr(self, f"proceed_to_step_{self.step+1}"))
        self.select_run.clicked.connect(self.file_dialog.show)
        self.file_dialog.fileSelected.connect(self.file_selected)
        if True:  # step == 5:
            self.select_model_btn.clicked.connect(self.model_dialog.show)
            self.model_dialog.fileSelected.connect(self.model_selected)

    def on_tab_selected(self):

        # Set run directory from User Preferences.
        self.file_dialog.setDirectory(Constants.log_prefer_path)

        if self.step == 2:  # Suggest
            # self.load_suggestion()
            pass
        if self.step == 5:  # Predict
            if len(self.loaded_features) == 0:
                self.model.removeRow(0)  # no_item placeholder
                self.add_formulation(self.parent.select_formulation)
        if True:  # self.step == 5:  # Predict
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

    def list_view_addPlaceholderText(self):
        if self.model.rowCount() == 0:
            no_item_text = "No items in list"
            if self.step == 2:
                no_item_text = "No suggestions available"
            if self.step == 3:
                no_item_text = "No experiments selected"
            if self.step == 5:
                no_item_text = "No predictions available"
            no_item = QtGui.QStandardItem(no_item_text)
            no_item.setEnabled(False)
            no_item.setSelectable(False)
            self.model.appendRow(no_item)

    def unique_case_insensitive_sort(self, list):
        seen = set()
        result = []
        for item in list:
            lower_item = item.lower()
            if lower_item not in seen:
                seen.add(lower_item)
                result.append(item)

        # Sort case-insensitive
        result.sort(key=str.lower)
        return result

    def load_all_excipient_types(self):
        self.proteins: list[str] = []
        self.buffers: list[str] = []
        self.surfactants: list[str] = []
        self.stabilizers: list[str] = []
        self.salts: list[str] = []

        ingredients = self.parent.ing_ctrl.get_all_ingredients()
        for i in ingredients:
            if i.name.casefold() == "none":
                continue  # skip "none"
            if i.type == "Protein":
                self.proteins.append(i.name)
            elif i.type == "Buffer":
                self.buffers.append(i.name)
            elif i.type == "Surfactant":
                self.surfactants.append(i.name)
            elif i.type == "Stabilizer":
                self.stabilizers.append(i.name)
            elif i.type == "Salt":
                self.salts.append(i.name)

        # this is case-sensitive, which is not what we want:
        # self.excipient_proteins.sort()
        # self.excipient_surfactants.sort()
        # self.excipient_stabilizers.sort()
        # this is using a case-insensitive sorting method:
        # self.proteins = sorted(
        #     self.proteins, key=str.casefold)
        # self.buffers = sorted(
        #     self.buffers, key=str.casefold)
        # self.surfactants = sorted(
        #     self.surfactants, key=str.casefold)
        # self.stabilizers = sorted(
        #     self.stabilizers, key=str.casefold)
        # self.salts = sorted(
        #     self.salts, key=str.casefold)
        # this is unique, case-insensitive sorting method:
        self.proteins = self.unique_case_insensitive_sort(self.proteins)
        self.buffers = self.unique_case_insensitive_sort(self.buffers)
        self.surfactants = self.unique_case_insensitive_sort(self.surfactants)
        self.stabilizers = self.unique_case_insensitive_sort(self.stabilizers)
        self.salts = self.unique_case_insensitive_sort(self.salts)

        Log.d("Proteins:", self.proteins)
        Log.d("Buffers:", self.buffers)
        Log.d("Surfactants:", self.surfactants)
        Log.d("Stabilizers:", self.stabilizers)
        Log.d("Salts", self.salts)

    def hide_extended_features(self):
        hide_rows = [2, 3, 4, 7]
        for row in hide_rows:
            self.feature_table.hideRow(row)

    def save_formulation(self, cancel: bool = False) -> bool:
        protein_type = self.feature_table.cellWidget(0, 1).currentText()
        protein_conc = self.feature_table.item(1, 1).text()
        protein_weight = self.feature_table.item(2, 1).text()
        protein_pI_mean = self.feature_table.item(3, 1).text()
        protein_pI_range = self.feature_table.item(4, 1).text()
        buffer_type = self.feature_table.cellWidget(5, 1).currentText()
        buffer_conc = self.feature_table.item(6, 1).text()
        buffer_pH = self.feature_table.item(7, 1).text()
        surfactant_type = self.feature_table.cellWidget(8, 1).currentText()
        surfactant_conc = self.feature_table.item(9, 1).text()
        stabilizer_type = self.feature_table.cellWidget(
            10, 1).currentText()
        stabilizer_conc = self.feature_table.item(11, 1).text()
        salt_type = self.feature_table.cellWidget(12, 1).currentText()
        salt_conc = self.feature_table.item(13, 1).text()

        # save run info to XML (if changed, request audit sign)
        if self.step in [1, 3]:  # Select, Import
            self.parent.save_run_info(self.run_file_xml, [
                protein_type, protein_conc,
                buffer_type, buffer_conc,
                surfactant_type, surfactant_conc,
                stabilizer_type, stabilizer_conc,
                salt_type, salt_conc], cancel)
            if self.parent.hasUnsavedChanges():
                if cancel:
                    Log.w("Unsaved changes lost, per user discretion.")
                    return True
                Log.w("There are still unsaved changes. Cannot continue.")
                QtWidgets.QMessageBox.information(
                    None,
                    Constants.app_title,
                    "There are still unsaved changes!\n\n" +
                    "To save: Try again and sign when prompted.\n" +
                    "Click \"Cancel\" to discard these changes.")
                return False
            elif cancel:
                Log.d("User canceled with nothing to save.")
                return True

        protein = self.parent.ing_ctrl.get_protein_by_name(name=protein_type)
        if protein == None:
            protein = self.parent.ing_ctrl.add_protein(
                Protein(enc_id=-1, name=protein_type))

        buffer = self.parent.ing_ctrl.get_buffer_by_name(name=buffer_type)
        if buffer == None:
            buffer = self.parent.ing_ctrl.add_buffer(
                Buffer(enc_id=-1, name=buffer_type))

        surfactant = self.parent.ing_ctrl.get_surfactant_by_name(
            name=surfactant_type)
        if surfactant == None:
            surfactant = self.parent.ing_ctrl.add_surfactant(
                Surfactant(enc_id=-1, name=surfactant_type))

        stabilizer = self.parent.ing_ctrl.get_stabilizer_by_name(
            name=stabilizer_type)
        if stabilizer == None:
            stabilizer = self.parent.ing_ctrl.add_stabilizer(
                Stabilizer(enc_id=-1, name=stabilizer_type))

        salt = self.parent.ing_ctrl.get_salt_by_name(name=salt_type)
        if salt == None:
            salt = self.parent.ing_ctrl.add_salt(
                Salt(enc_id=-1, name=salt_type))

        def is_number(s: str):
            try:
                float(s)
                return True
            except ValueError:
                return False

        # update protein and buffer characteristics
        # bail if any extended features are missing
        if is_number(protein_weight):
            protein.molecular_weight = float(protein_weight)
        elif not protein.molecular_weight:
            Log.e("Missing protein molecular weight!")
            return
        if is_number(protein_pI_mean):
            protein.pI_mean = float(protein_pI_mean)
        elif not protein.pI_mean:
            Log.e("Missing protein pI mean!")
            return
        if is_number(protein_pI_range):
            protein.pI_range = float(protein_pI_range)
        elif not protein.pI_range:
            Log.e("Missing protein pI range!")
            return
        if is_number(buffer_pH):
            buffer.pH = float(buffer_pH)
        elif not buffer.pH:
            Log.e("Missing buffer pH!")
            return

        # if no changes, nothing is done on 'update' call
        self.parent.ing_ctrl.update_protein(protein.id, protein)
        self.parent.ing_ctrl.update_buffer(buffer.id, buffer)

        while len(self.profile_viscos) < len(self.profile_shears):
            self.profile_viscos.append(-1)

        # pull in viscosity profile from run load
        vp = ViscosityProfile(shear_rates=self.profile_shears,
                              viscosities=self.profile_viscos,
                              units='cP')
        vp.is_measured = self.run_figure_valid

        # pull temperaure
        temp = self.run_temperature.text()
        if temp.endswith('C'):
            temp = temp[:-1]  # strip Celsius unit character
        if not is_number(temp):
            temp = "nan"  # not a number, casts to float as nan

        form = Formulation()
        form.set_protein(
            protein=protein, concentration=float(protein_conc), units='mg/mL')
        form.set_buffer(buffer, concentration=float(
            buffer_conc), units='mM')
        form.set_surfactant(surfactant=surfactant,
                            concentration=float(surfactant_conc), units='%w')
        form.set_stabilizer(stabilizer=stabilizer,
                            concentration=float(stabilizer_conc), units='M')
        form.set_salt(salt, concentration=float(salt_conc), units='mM')
        form.set_viscosity_profile(profile=vp)
        form.set_temperature(float(temp))

        form_saved = self.parent.form_ctrl.add_formulation(
            formulation=form)

        if self.step == 1:
            Log.d("Saving selected formulation to parent for later")
            self.parent.select_formulation = form_saved
            # print(self.parent.form_ctrl.get_all_as_dataframe())
        if self.step == 5:
            Log.d("Saving prediction formulation to parent for later")
            self.parent.predict_formulation = form_saved

        return True

    def load_suggestion(self):
        model_name = self.select_model_label.text()
        if hasattr(self, "timer") and self.timer.isActive():
            Log.w("Busy canceling... Please wait...")
            return
        if len(self.select_model_label.text()) == 0 or self.model_path == None:
            Log.e("No model selected. Cannot load suggestions.")
            return
        if not self.parent.database.is_open:
            Log.e("No database connection. Cannot load suggestions.")
            return

        self.progressBar = QtWidgets.QProgressDialog(
            "Suggesting...", "Cancel", 0, 0, self)
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/reset.png')
        self.progressBar.setWindowIcon(QtGui.QIcon(icon_path))
        self.progressBar.setWindowTitle("Busy")
        self.progressBar.setWindowFlag(
            QtCore.Qt.WindowContextHelpButtonHint, False)
        self.progressBar.setWindowFlag(
            QtCore.Qt.WindowStaysOnTopHint, True)
        self.progressBar.setFixedSize(
            int(self.progressBar.width()*1.5), int(self.progressBar.height()*1.1))
        self.progressBar.setModal(True)
        self.progressBar.show()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.check_finished)
        self.timer.start()

        def add_new_suggestion(record: ExecutionRecord):
            if self.progressBar.wasCanceled():
                Log.d("User canceled suggestion. Ignoring results.")
                return

            Log.d("Processing suggestion results!")

            form = record.result
            exception = record.exception
            if exception:
                Log.e(f"ERROR: Failed to suggest: {str(exception)}")
                return

            self.add_formulation(form)

        Log.d("Waiting for suggestion results...")
        self.parent.enable(False)
        self.executor = Executor()
        self.executor.run(
            self,
            method_name="get_new_suggestion",
            asset_name=model_name,
            callback=add_new_suggestion)

    def get_new_suggestion(self, asset_name):
        database = Database(parse_file_key=True)
        sampler = Sampler(asset_name=asset_name,
                          database=database)
        form = sampler.get_next_sample()
        database.close()
        return form

    def add_formulation(self, form: Formulation):
        feature = copy.deepcopy(self.default_features)
        if form.protein:  # NOT an empty Formulation() object
            feature["Value"][0] = form.protein.ingredient.name
            feature["Value"][1] = form.protein.concentration
            feature["Value"][2] = form.protein.ingredient.molecular_weight
            feature["Value"][3] = form.protein.ingredient.pI_mean
            feature["Value"][4] = form.protein.ingredient.pI_range
            feature["Value"][5] = form.buffer.ingredient.name
            feature["Value"][6] = form.buffer.concentration
            feature["Value"][7] = form.buffer.ingredient.pH
            feature["Value"][8] = form.surfactant.ingredient.name
            feature["Value"][9] = form.surfactant.concentration
            feature["Value"][10] = form.stabilizer.ingredient.name
            feature["Value"][11] = form.stabilizer.concentration
            feature["Value"][12] = form.salt.ingredient.name
            feature["Value"][13] = form.salt.concentration

        if len(self.loaded_features) == 0:
            self.model.removeRow(0)  # no_item placeholder
        self.loaded_features.append(feature)
        num = len(self.loaded_features)
        form_type = "Suggestion" if self.step == 2 else "Prediction"
        self.model.appendRow(QtGui.QStandardItem(f"{form_type} {num}"))

    def _get_viscosity_list(self, formulation: Formulation) -> list:
        rate_list = []
        vp = formulation.viscosity_profile
        for rate in [100, 1000, 10000, 100000, 15000000]:
            rate_list.append(vp.get_viscosity(rate))
        return rate_list

    def make_predictions(self):
        if hasattr(self, "timer") and self.timer.isActive():
            Log.w("Busy canceling... Please wait...")
            return
        if len(self.select_model_label.text()) == 0 or self.model_path == None:
            Log.e("No model selected. Cannot make predictions.")
            return
        if not self.parent.database.is_open:
            Log.e("No database connection. Cannot make predictions.")
            return

        if not self.feature_table.allSet():
            message = "Please correct the highlighted fields first."
            QtWidgets.QMessageBox.information(
                None, Constants.app_title, message, QtWidgets.QMessageBox.Ok)
            return

        self.save_formulation()

        self.predictor = Predictor(zip_path=self.model_path)
        select_df = self.parent.select_formulation.to_dataframe(
            encoded=False, training=False)
        predict_df = self.parent.predict_formulation.to_dataframe(
            encoded=False, training=False)

        self.progressBar = QtWidgets.QProgressDialog(
            "Updating...", "Cancel", 0, 0, self)
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/reset.png')
        self.progressBar.setWindowIcon(QtGui.QIcon(icon_path))
        self.progressBar.setWindowTitle("Busy")
        self.progressBar.setWindowFlag(
            QtCore.Qt.WindowContextHelpButtonHint, False)
        self.progressBar.setWindowFlag(
            QtCore.Qt.WindowStaysOnTopHint, True)
        self.progressBar.setFixedSize(
            int(self.progressBar.width()*1.5), int(self.progressBar.height()*1.1))
        self.progressBar.setModal(True)
        self.progressBar.show()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.check_finished)
        self.timer.start()

        def run_prediction_result(record: ExecutionRecord):

            Log.d("Waiting for prediction results...")
            self.progressBar.setLabelText("Predicting...")

            self.executor.run(
                self.predictor,
                method_name="predict_uncertainty",
                data=predict_df,
                callback=get_prediction_result)

        def get_prediction_result(record: ExecutionRecord):

            if self.progressBar.wasCanceled():
                Log.d("User canceled prediction. Ignoring results.")
                return

            Log.d("Processing prediction results!")

            # The returns from this are a predicted viscosity profile [val1,val2,...val5]
            # predicted_vp = self.parent.predictor.predict(data=form_df)

            # The returns from this are a predicted viscosity profile [val1,val2,...val5] and
            # a series of standard deviations for each predicted value.
            predicted_mean_vp, mean_std = record.result
            exception = record.exception
            if exception:
                Log.e(f"ERROR: Prediction exception: {str(exception)}")

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

            def make_plot(name, shear, mean_arr, std_arr, title, color):
                # clear existing plot before making a new one
                self.run_figure_valid = False
                self.run_figure.clear()
                self.run_canvas.draw()

                ax = self.run_figure.add_subplot(111)
                xs, ys = smooth_log_interpolate(shear, mean_arr)
                xs_up, ys_up = smooth_log_interpolate(
                    shear, mean_arr + std_arr)
                xs_dn, ys_dn = smooth_log_interpolate(
                    shear, mean_arr - std_arr)
                ax.plot(xs, ys, '-', lw=2.5, color=color)
                ax.fill_between(xs_dn, ys_dn, ys_up, alpha=0.25, color=color)
                ax.scatter(shear, mean_arr, s=40, color=color, zorder=5)
                ax.set_xlim(xs.min(), xs.max())
                ann = "\n".join(f"{x:.0e}: {m:.1f}±{s:.1f}" for x, m,
                                s in zip(shear, mean_arr, std_arr))
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_ylim(self.calc_limits(yall=np.concat((ys_dn, ys_up))))
                ax.set_xlabel("Shear rate (s⁻¹)", fontsize=10)
                ax.set_ylabel("Viscosity (cP)", fontsize=10)
                ax.grid(True, which="both", ls=":")
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))

                self.run_figure_valid = True
                self.run_canvas.draw()

            # Plot
            make_plot("name", self.profile_shears,
                      predicted_mean_vp[0], mean_std[0], "title", "blue")

        self.executor = Executor()

        if self.parent.select_formulation.viscosity_profile.is_measured:

            # Get the viscosity profile or y target to update with.
            vp = self._get_viscosity_list(self.parent.select_formulation)

            # Target needs to be form np.array([[Viscosity_100, ..., Viscosity_15000000]])
            # Also I have this set so updating does not overwrite the existing model until
            # we figure out how model storage works
            self.executor.run(
                self.predictor,
                method_name="update",
                new_data=select_df,
                new_targets=np.array([vp]),
                epochs=10,
                batch_size=32,
                save=False,
                callback=run_prediction_result)

        else:
            run_prediction_result()

    def check_finished(self):
        record_count = 1
        if self.step == 5 and self.parent.select_formulation.viscosity_profile.is_measured:
            record_count += 1
        if self.executor.active_count() == 0 and len(self.executor.get_task_records()) == record_count:
            self.progressBar.close()  # finished
            self.timer.stop()
            if self.step == 2:
                self.parent.enable(True)

    def proceed_to_step_2(self):
        # Are we ready to proceed?
        # Yes, if and only if:
        #   1. All audits contain valid values
        #   2. All initial features are set
        #   3. Analyze results are valid
        #   4. All formulations saved, and XMLs up-to-date
        if (len(self.run_captured.text()) and
            len(self.run_updated.text()) and
            len(self.run_analyzed.text()) and
                self.feature_table.allSet() and
                self.run_figure_valid):
            # ready to proceed
            if not self.save_formulation():
                return
            if self.parent is not None:
                i = self.parent.tab_widget.currentIndex()
                self.parent.tab_widget.setCurrentIndex(i+1)
                # next_widget: FrameStep1 = self.parent.tab_widget.currentWidget()
                # next_widget.load_suggestions()
            else:
                self.run_notes.setText(
                    "ERROR: self.parent is None.\n" +
                    "Cannot proceed to next step!")
        else:  # not ready
            message = "Please select a run."
            if self.select_label.text():
                message = "Please correct the highlighted fields first."
            QtWidgets.QMessageBox.information(
                None, Constants.app_title, message, QtWidgets.QMessageBox.Ok)

    def proceed_to_step_3(self):
        # ready to proceed
        if self.parent is not None:
            i = self.parent.tab_widget.currentIndex()
            self.parent.tab_widget.setCurrentIndex(i+1)

    def proceed_to_step_4(self):
        # First of all, there must be at least 1 imported experiment
        # For each run in list, must pass the same criteria from Step 1
        #   1. All audits contain valid values
        #   2. All initial features are set
        #   3. Analyze results are valid
        #   4. All formulations saved, and XMLs up-to-date
        if len(self.all_files) == 0:
            QtWidgets.QMessageBox.information(
                None, Constants.app_title,
                "Please import at least 1 experiment before proceeding.",
                QtWidgets.QMessageBox.Ok)
            return
        all_is_good = True
        for file_name, file_path in self.all_files.items():
            self.file_selected(file_path)  # load each run
            if (len(self.run_captured.text()) and
                len(self.run_updated.text()) and
                len(self.run_analyzed.text()) and
                    self.feature_table.allSet() and
                    self.run_figure_valid):
                if not self.save_formulation():
                    return
            else:
                all_is_good = False
                # break # maybe not, if we want to highlight *all* errors on "Next"
        if all_is_good:
            # ready to proceed
            if self.parent is not None:
                i = self.parent.tab_widget.currentIndex()
                self.parent.tab_widget.setCurrentIndex(i+1)
                # next_widget: FrameStep2 = self.parent.tab_widget.currentWidget()
                # next_widget.learn()
            else:
                self.run_notes.setText(
                    "ERROR: self.parent is None.\n" +
                    "Cannot proceed to next step!")
        else:  # not ready
            QtWidgets.QMessageBox.information(
                None, Constants.app_title, "Please correct the highlighted fields first.", QtWidgets.QMessageBox.Ok)

    # NOTE: step_5 would be handled in FrameStep2

    def proceed_to_step_6(self):
        # ready to proceed
        if self.parent is not None:
            i = self.parent.tab_widget.currentIndex()
            self.parent.tab_widget.setCurrentIndex(i+1)
        else:
            self.run_notes.setText(
                "ERROR: self.parent is None.\n" +
                "Cannot proceed to next step!")

    def user_run_clicked(self):
        try:
            self.file_selected(self.all_files[self.model.itemFromIndex(
                self.list_view.selectedIndexes()[0]).text()])
        except IndexError as e:
            if len(self.all_files):
                raise e
            else:  # no files in list, this error can occur when user cliks on the placeholder text
                pass  # ignore the click
        # raise any other exception type

    def add_another_item(self):
        if self.step == 2:
            self.load_suggestion()
        if self.step == 5:
            self.add_formulation(Formulation())

    def user_run_removed(self):
        try:
            selected = self.list_view.selectedIndexes()
            if len(selected) == 0:
                return  # nothing selected, nothing to do
            file_name = self.model.itemFromIndex(selected[0]).text()
            self.all_files.pop(file_name, None)  # remove key from dict
            if self.step in [2, 5]:
                self.loaded_features.pop(selected[0].row())
            self.model.removeRow(selected[0].row())
            self.list_view_addPlaceholderText()
            self.file_selected(None)  # clear selection
        except IndexError as e:
            if len(self.all_files):
                raise e
            else:  # no files in list, this error can occur when user cliks on the placeholder text
                pass  # ignore the click
        # raise any other exception type

    def export_table_data(self):
        info_on_success = False
        open_on_success = True

        default_export_folder = os.path.expanduser(os.path.join(
            "~", "Documents", f"{Constants.app_publisher} {Constants.app_name}", "exported_pdfs"))
        if os.path.exists(os.path.dirname(default_export_folder)):
            os.makedirs(default_export_folder, exist_ok=True)
        else:
            default_export_folder = os.getcwd()

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, "Export as PDF", default_export_folder, "PDF files (*.pdf)"
        )

        if file_path:
            try:
                # Create printer to make PDFs
                printer = QPrinter(QPrinter.PrinterResolution)
                printer.setOutputFormat(QPrinter.PdfFormat)
                printer.setOutputFileName(file_path)
                printer.setPageSize(QPrinter.Letter)

                # Set 1-inch margins on all sides
                # left, top, right, bottom
                margins = QtCore.QMarginsF(1.0, 1.0, 1.0, 1.0)
                printer.setPageMargins(
                    margins.left(), margins.top(), margins.right(), margins.bottom(),
                    QPrinter.Inch)
                painter = QtGui.QPainter(printer)

                # Set font
                font = QtGui.QFont("Times New Roman", 12)
                # NOTE: Bold will be set/unset for header/data rows

                for i in range(len(self.loaded_features)):

                    # Select the item
                    index = self.model.index(i, 0)
                    self.list_view.setCurrentIndex(index)
                    self.list_view.selectionModel().select(
                        index, QtCore.QItemSelectionModel.ClearAndSelect)
                    self.list_view.clicked.emit(index)

                    if i > 0:
                        # Add new page to PDF for each table in list
                        printer.newPage()

                    # Start at top of page
                    y = 0

                    # Page spacing parameters
                    cell_pad_top = 7
                    cell_pad_left = 10
                    row_height = 30

                    # Calculate cell dimensions
                    table_widget = self.feature_table
                    page_rect = printer.pageRect()
                    col_width = page_rect.width() // table_widget.columnCount()

                    # Draw page header/title
                    font.setBold(True)
                    painter.setFont(font)
                    painter.drawText(cell_pad_left, y,
                                     f"Suggested Experiment {i+1}")
                    y += row_height

                    # Draw headers
                    for col in range(table_widget.columnCount()):
                        header = table_widget.horizontalHeaderItem(col)
                        text = header.text() if header else f"Column {col}"
                        border_rect = QtCore.QRect(
                            col * col_width, y, col_width, row_height)
                        text_rect = QtCore.QRect(border_rect)
                        text_rect.adjust(cell_pad_left, cell_pad_top, 0, 0)
                        painter.drawText(text_rect, 0, text)
                        painter.drawRect(border_rect)
                    y += row_height

                    # Draw data rows
                    font.setBold(False)
                    painter.setFont(font)
                    for row in range(table_widget.rowCount()):
                        skip = True
                        for col in range(table_widget.columnCount()):
                            item = table_widget.item(row, col)
                            if item:
                                text = item.text()
                            else:
                                widget = table_widget.cellWidget(row, col)
                                if widget:
                                    text = widget.currentText()
                                else:
                                    continue  # skip blank rows
                            if table_widget.isRowHidden(row):
                                continue  # skip hidden rows
                            skip = False
                            border_rect = QtCore.QRect(
                                col * col_width, y, col_width, row_height)
                            text_rect = QtCore.QRect(border_rect)
                            text_rect.adjust(
                                cell_pad_left, cell_pad_top, 0, 0)
                            painter.drawText(text_rect, 0, text)
                            painter.drawRect(border_rect)
                        if not skip:
                            y += row_height

                painter.end()

                if info_on_success:
                    QtWidgets.QMessageBox.information(
                        None, "Success", "PDF exported successfully!")

                if open_on_success:
                    webbrowser.open(file_path)

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    None, "Error", f"Failed to export PDF: {str(e)}")

    def file_selected(self, path: str | None, cancel: bool = False):
        # If run already loaded, try saving formulation to write any changed Run Info to XML
        if self.run_file_xml and self.step in [1, 3]:
            if not self.feature_table.allSet():
                result = QtWidgets.QMessageBox.question(
                    None,
                    Constants.app_title,
                    "You have missing feature values.\n\nAre you sure you want to reload the features table?")
                if result != QtWidgets.QMessageBox.Yes:
                    return
            elif not self.feature_table.isEmpty():
                Log.i("Saving formulation for fully populated feature table.")
                if not self.save_formulation(cancel):
                    return

        self.run_file_run = path
        self.run_file_xml = None
        self.run_file_analyze = None

        # clear all fields, before repopulating them
        self.select_label.clear()
        self.run_notes.clear()
        self.run_name.clear()
        self.run_date_time.clear()
        self.run_duration.clear()
        self.run_temperature.clear()
        self.run_batch.clear()
        self.run_fill_type.clear()
        self.run_captured.clear()
        self.run_updated.clear()
        self.run_analyzed.clear()
        self.run_figure.clear()
        self.run_figure_valid = False
        self.run_canvas.draw()
        self.feature_table.clear()

        self.btn_update.setEnabled(False)

        if path is None:
            if self.step == 1:  # Select
                self.parent.select_formulation = Formulation()
            if self.step == 3:  # Import Experiments
                self.list_view.clearSelection()
            if self.step == 5:  # Predict
                self.parent.predict_formulation = Formulation()
            if True:  # Always, all tabs
                self.model_selected(None)
            return

        self.btn_update.setEnabled(True)

        namelist = SecureOpen.get_namelist(self.run_file_run)
        for file in namelist:
            if file.endswith(".csv"):
                self.run_file_run = os.path.join(
                    os.path.dirname(self.run_file_run), file)
                break

        self.select_label.setText(
            os.path.basename(os.path.dirname(self.run_file_run)))

        if self.step == 3:
            item = QtGui.QStandardItem(self.select_label.text())

            # Disallow user from selecting same run for Step 1 and Step 3.
            if item.text() == self.parent.tab_widget.widget(0).select_label.text():
                QtCore.QTimer.singleShot(100, lambda: QtWidgets.QMessageBox.information(
                    None,
                    Constants.app_title,
                    "The selected run from Step 1 cannot also be an imported experiment run.",
                    QtWidgets.QMessageBox.Ok))
                return

            found = self.model.findItems(item.text())
            if len(found) == 0:
                if len(self.all_files) == 0:
                    self.model.removeRow(0)  # no_item placeholder
                self.model.appendRow(item)
                new_index = self.model.indexFromItem(item)
                self.list_view.setCurrentIndex(new_index)
                self.all_files[item.text()] = path

        folder = os.path.dirname(self.run_file_run)
        files: list[str] = os.listdir(folder)
        max_index = 0
        for f in files:
            if f.endswith(".xml"):
                self.run_file_xml = os.path.join(folder, f)
            if f.startswith("analyze") and f.endswith(".zip"):
                this_index = int(f[f.index("-")+1:f.index(".")])
                if this_index > max_index:
                    max_index = this_index
                self.run_file_analyze = os.path.join(folder,
                                                     f.replace(str(this_index), str(max_index)))
        if self.run_file_xml == None:
            self.run_notes.setTextBackgroundColor(Color.light_red)
            self.run_notes.setText("ERROR: Cannot find XML file for this run!")
            return
        if self.run_file_analyze == None:
            self.run_notes.setTextBackgroundColor(Color.light_yellow)
            self.run_notes.setText("This run has not been analyzed yet.\n" +
                                   "Please Analyze and try again!")
            return

        doc = minidom.parse(self.run_file_xml)

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

        if xml_params.get("bioformulation", False) != 'True':
            self.run_notes.setTextBackgroundColor(Color.light_red)
            self.run_notes.setText("ERROR: This run is not a bioformulation!")
            return

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

        run_features = copy.deepcopy(self.default_features)
        value_tags = ["protein_type", "protein_concentration",
                      "", "", "",  # molecular weight, pI mean, pI range
                      "buffer_type", "buffer_concentration",
                      "",  # pH
                      "surfactant_type", "surfactant_concentration",
                      "stabilizer_type", "stabilizer_concentration",
                      "salt_type", "salt_concentration"]
        for x, y in enumerate(value_tags):
            try:
                if y == "":
                    continue
                if y in xml_params.keys():
                    # TODO: quick fix for demo
                    value = xml_params[y]
                    if value == "TWEEN80":
                        value = "Tween-80"
                    if value == "TWEEN20":
                        value = "Tween-20"
                    run_features["Value"][x] = value
            except Exception as e:
                print(e)

        if False:  # self.step == 3:
            # Hide protein and buffer characteristics
            for values in run_features.values():
                del values[7]  # buffer PH
                del values[4]  # protein pI range
                del values[3]  # protein pI mean
                del values[2]  # protein weight
        else:
            # Pull protein and buffer characteristics from database (if available)
            protein = self.parent.ing_ctrl.get_protein_by_name(
                name=xml_params.get("protein_type", None))
            if protein != None:
                if protein.molecular_weight != None:
                    run_features["Value"][2] = protein.molecular_weight
                if protein.pI_mean != None:
                    run_features["Value"][3] = protein.pI_mean
                if protein.pI_range != None:
                    run_features["Value"][4] = protein.pI_range
            buffer = self.parent.ing_ctrl.get_buffer_by_name(
                name=xml_params.get("buffer_type", None))
            if buffer != None:
                if buffer.pH != None:
                    run_features["Value"][7] = buffer.pH

        self.feature_table.setData(run_features)

        # Import most recent analysis
        in_shear_rate = []
        in_viscosity = []
        in_temperature = []
        try:
            base_run_name: str = os.path.basename(self.run_file_run)
            base_run_name = base_run_name[:base_run_name.rfind("_")]
            csv_file = os.path.join(os.path.dirname(
                self.run_file_analyze), f"{base_run_name}_analyze_out.csv")
            zip_filename = os.path.splitext(
                os.path.basename(self.run_file_analyze))[0]
            with SecureOpen(csv_file, "r", zip_filename, insecure=True) as f:
                csv_headers = next(f)
                csv_cols = (0, 2, 4)
                data = np.loadtxt(
                    f.readlines(), delimiter=",", skiprows=0, usecols=csv_cols
                )
            in_shear_rate = data[:, 0]
            in_viscosity = data[:, 1]
            in_temperature = data[:, 2]
        except Exception as e:
            print(e)
        pass_to_models = {"shear_rate": in_shear_rate,
                          "viscosity": in_viscosity}

        # self.profile_shears = [1e2, 1e3, 1e4, 1e5, 15000000] # already set
        self.profile_viscos = []
        has_high_shear_pt = in_shear_rate[-1] > 1e6
        has_curve_fit_est = False

        for shear_rate in self.profile_shears:
            viscosity = np.interp(shear_rate, in_shear_rate, in_viscosity,
                                  left=np.nan)
            self.profile_viscos.append(viscosity)

        if np.any(np.isnan(self.profile_viscos)):
            try:
                # Define the logarithmic function to fit
                def logarithmic_func(x, a, b, c):
                    return a * np.log10(x - c) + b

                # Define bounds for parameters [a, b, c]
                lower_bounds = [-np.inf, -np.inf, -np.inf]
                upper_bounds = [np.inf, np.inf, 99]
                fit_bounds = (lower_bounds, upper_bounds)

                # Perform the curve fit (not including high-shear point)
                initial_guess = (2, 1, 0.5)
                popt, pcov = curve_fit(
                    logarithmic_func,
                    in_shear_rate[:-1] if has_high_shear_pt else in_shear_rate,
                    in_viscosity[:-1] if has_high_shear_pt else in_viscosity,
                    p0=initial_guess,
                    bounds=fit_bounds)
                a_fit, b_fit, c_fit = popt
                has_curve_fit_est = True

                # Generate missing points for the profile using curve fitting
                for i in range(len(self.profile_viscos)):
                    if np.isnan(self.profile_viscos[i]):
                        self.profile_viscos[i] = logarithmic_func(
                            self.profile_shears[i], a_fit, b_fit, c_fit)
                    else:
                        break

            except Exception as e:
                Log.w(
                    "Failed to fit logarithmic curve to viscosity profile. Using entirely interpolated data instead.")

                # Generate missing points for the profile using interpolation
                for i in range(len(self.profile_viscos)):
                    if np.isnan(self.profile_viscos[i]):
                        new_value = np.interp(
                            self.profile_shears[i], in_shear_rate, in_viscosity)
                        self.profile_viscos[i] = new_value
                    else:
                        break

        if has_high_shear_pt:
            self.profile_viscos[-1] = in_viscosity[-1]

        expected_point_count = 13
        if len(in_viscosity) > expected_point_count:
            indices_to_drop = list(range(4, len(in_shear_rate)-2))
            in_shear_rate = [item for i, item in enumerate(
                in_shear_rate) if i not in indices_to_drop]
            in_viscosity = [item for i, item in enumerate(
                in_viscosity) if i not in indices_to_drop]

        minidx = np.argmin(self.profile_viscos)
        maxidx = np.argmax(self.profile_viscos)
        Log.i(
            f"Viscosity profile ranges from {self.profile_viscos[minidx]:.2f} to {self.profile_viscos[maxidx]:.2f} cP.")

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
        if len(in_viscosity) > 0:
            xs, ys = smooth_log_interpolate(
                self.profile_shears, self.profile_viscos)
            ax.set_xlim(xs.min(), xs.max())
            ax.set_ylim(self.calc_limits(yall=in_viscosity))
            ax.plot(self.profile_shears, self.profile_viscos,
                    lw=2.5, color="blue")
            ax.scatter(self.profile_shears, self.profile_viscos,
                       s=40, color="blue", zorder=5)
            ax.plot(in_shear_rate, in_viscosity, "b,")
            self.run_figure_valid = True

            DEBUG = False
            if has_curve_fit_est and DEBUG:
                x_trend = np.logspace(2, 5)
                y_trend = logarithmic_func(
                    x_trend, a_fit, b_fit, c_fit)
                y_trend = [1 if np.isnan(x) else x for x in y_trend]
                ax.plot(x_trend, y_trend, color='blue', linewidth=0.5)
        else:
            ax.text(0.5, 0.5, "Invalid Results",
                    transform=ax.transAxes,
                    ha='center', va='center',
                    bbox=dict(facecolor='yellow', edgecolor='black'))
        ax.set_xscale("log")
        ax.set_yscale("log")
        self.run_canvas.draw()

        avg_temp = np.average(data[:, 2])
        if np.isnan(avg_temp):
            self.run_temperature.setText("(Unknown)")
        else:
            self.run_temperature.setText(f"{avg_temp:2.2f}C")

    def calc_limits(self, yall):
        ymin, ymax = 0, 1000
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
        if lower_limit >= upper_limit:
            Log.d(
                "Limits were auto-calculated but are in an invalid range! Using ylim [0, 1000]."
            )
        elif np.isfinite(lower_limit) and np.isfinite(upper_limit):
            Log.d(
                f"Auto-calculated y-range limits for figure are: [{lower_limit}, {upper_limit}]"
            )
            ymin = lower_limit
            ymax = upper_limit
        else:
            Log.d(
                "Limits were auto-calculated but were not finite values! Using ylim [0, 1000]."
            )
        return ymin, ymax

    def model_selected(self, path: str | None):
        self.model_path = path

        if path is None:
            self.select_model_label.clear()
            return

        self.select_model_label.setText(
            path.split('\\')[-1].split('/')[-1].split('.')[0])


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
        self.verticalHeader().setVisible(False)

    def clear(self):
        super().clear()
        self._is_empty = True

    def setData(self, data: dict[str, str]) -> None:
        self.data = data
        self.clear()
        horHeaders = []
        for n, key in enumerate(self.data.keys()):
            horHeaders.append(key)
            for m, item in enumerate(self.data[key]):
                if n == 0 or n == 2:
                    # always treat first and last cols as uneditable text, not a QComboBox
                    newitem = QtWidgets.QTableWidgetItem(str(item))
                elif self._is_number(item) or len(item) == 0:
                    # item is either a number or blank (empty string)
                    newitem = QtWidgets.QTableWidgetItem(str(item))
                else:
                    newitem = QtWidgets.QComboBox()
                    # newitem.addItem("add new...")
                    if isinstance(item, list):
                        newitem.addItem("None")
                        newitem.addItems(item)
                        self.data["Units"][m] = "\u2190"  # unicode left arrow
                        newitem.currentIndexChanged.connect(
                            lambda idx, row=m: self._row_combo_set(row))
                    else:  # str
                        newitem.addItem(item)
                        self.data["Units"][m] = ""  # clear flag
                # disable 1st and last column items (not selectable or editable)
                if n == 0 or n == 2:
                    if n == 0:  # bold 1st column items (headers)
                        font = newitem.font()
                        font.setBold(True)
                        newitem.setFont(font)
                    newitem.setFlags(newitem.flags() &
                                     ~(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEditable))
                if isinstance(newitem, QtWidgets.QWidget):
                    self.setCellWidget(m, n, newitem)
                else:
                    self.setItem(m, n, newitem)
                # Unhide a row if it was hidden
                if self.isRowHidden(m):
                    self.showRow(m)
        self.setHorizontalHeaderLabels(horHeaders)
        header = self.horizontalHeader()
        header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        self._is_empty = False

    def allSet(self) -> bool:
        for n, key in enumerate(self.data.keys()):
            for m, _ in enumerate(self.data[key]):
                item = self.item(m, n)
                if item is None:
                    continue  # QComboBox will return a None item
                if item.background().color().name() in [Color.light_yellow.name(), Color.light_red.name()]:
                    return False
        return True

    def isEmpty(self) -> bool:
        return self._is_empty

    def _row_combo_set(self, idx):
        item = self.item(idx, 2)
        if item is not None:
            self.blockSignals(True)  # prevent recursion
            item.setBackground(QtGui.QBrush(Color.white))
            self.item(idx, 2).setText("")
            self.blockSignals(False)

    def _is_number(self, s: str):
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem):
        row, col, text = item.row(), item.column(), item.text()
        print(f"Cell ({row}, {col}) changed to: {text}")

        if col == 2 and text == "\u2190":  # unicode left arrow
            item.setBackground(QtGui.QBrush(Color.light_yellow))

        if not (item.flags() & QtCore.Qt.ItemFlag.ItemIsEditable):
            # print("skip, disabled")
            return

        now_bg = item.background()
        now_fg = item.foreground()
        new_bg = QtGui.QBrush(now_bg.color())
        new_fg = QtGui.QBrush(now_fg.color())

        if len(text) == 0:
            new_bg.setColor(Color.light_yellow)
            new_fg.setColor(Color.black)
        elif not self._is_number(text):
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


class CollapsibleBox(QtWidgets.QWidget):
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)
        self._collapsed = True

        self.toggle_button = QtWidgets.QToolButton(
            text=title, checkable=True, checked=False
        )

        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonTextBesideIcon
        )
        self.toggle_button.setArrowType(QtCore.Qt.RightArrow)
        self.toggle_button.pressed.connect(self.toggle)

        self.toggle_animation = QtCore.QParallelAnimationGroup(self)

        self.content_area = QtWidgets.QScrollArea(
            maximumHeight=0, minimumHeight=0
        )
        self.content_area.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.content_area.setFrameShape(QtWidgets.QFrame.NoFrame)

        lay = QtWidgets.QVBoxLayout(self)
        # lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b"minimumHeight")
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b"maximumHeight")
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self.content_area, b"maximumHeight")
        )

    def setCollapsed(self, checked):
        if self._collapsed is not checked:
            self._collapsed = checked
            self.repaint()

    def isCollapsed(self):
        return self._collapsed

    def toggle(self):
        self.setCollapsed(not self.isCollapsed())
        # calls self.repaint()

    def repaint(self):
        checked = self._collapsed
        self.toggle_button.setArrowType(
            QtCore.Qt.DownArrow if not checked else QtCore.Qt.RightArrow
        )
        self.toggle_animation.setDirection(
            QtCore.QAbstractAnimation.Forward
            if not checked
            else QtCore.QAbstractAnimation.Backward
        )

        self.toggle_animation.start()

    def setContentLayout(self, layout):
        lay = self.content_area.layout()
        del lay
        self.content_area.setLayout(layout)
        collapsed_height = (
            self.sizeHint().height() - self.content_area.maximumHeight()
        )
        content_height = layout.sizeHint().height()
        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(500)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(
            self.toggle_animation.animationCount() - 1
        )
        content_animation.setDuration(500)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)


if __name__ == '__main__':
    if False:
        _app = QtWidgets.QApplication([])
        _win = VisQAIWindow()
        _win.show()
        _app.exec()
        _app.exit()

    else:
        import sys
        import random

        app = QtWidgets.QApplication(sys.argv)

        w = QtWidgets.QMainWindow()
        w.setCentralWidget(QtWidgets.QWidget())
        dock = QtWidgets.QDockWidget("Collapsible Demo")
        w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        scroll = QtWidgets.QScrollArea()
        dock.setWidget(scroll)
        content = QtWidgets.QWidget()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        vlay = QtWidgets.QVBoxLayout(content)
        vlay.addWidget(QtWidgets.QGroupBox("Buffer"))
        for i in range(1):
            box = CollapsibleBox("Advanced Information")
            vlay.addWidget(box)
            vlay.setContentsMargins(0, 0, 0, 0)
            lay = QtWidgets.QVBoxLayout()
            lay.setContentsMargins(0, 0, 0, 0)
            labels = ["Protein", "Buffer", "Salt"]
            for j in range(3):
                label = QtWidgets.QGroupBox(f"{labels[j]} Information")
                layout = QtWidgets.QFormLayout(label)
                layout.addRow("Type:", QtWidgets.QLineEdit())
                layout.addRow("Concentration:", QtWidgets.QLineEdit())
                # color = QtGui.QColor(*[random.randint(0, 255) for _ in range(3)])
                # label.setStyleSheet(
                #     "background-color: {}; color : white;".format(color.name())
                # )
                # label.setAlignment(QtCore.Qt.AlignCenter)
                lay.addWidget(label)
            box.setContentLayout(lay)
        vlay.addWidget(QtWidgets.QCheckBox("Remember for next time"))
        toggle = QtWidgets.QPushButton("Toggle")
        toggle.clicked.connect(box.toggle)
        open = QtWidgets.QPushButton("Open")
        open.clicked.connect(lambda: box.setCollapsed(False))
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(lambda: box.setCollapsed(True))
        vlay.addWidget(toggle)
        vlay.addWidget(open)
        vlay.addWidget(close)
        vlay.addStretch()
        w.resize(640, 480)
        w.show()
        # box.setCollapsed(False)
        sys.exit(app.exec_())
