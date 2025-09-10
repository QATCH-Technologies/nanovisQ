try:
    from typing import Optional, Tuple
    from QATCH.ui.popUp import PopUp
    from QATCH.core.constants import Constants
    from QATCH.common.userProfiles import UserProfiles, UserRoles
    from QATCH.common.logger import Logger as Log
    from QATCH.common.architecture import Architecture
except (ModuleNotFoundError, ImportError):
    print("Running VisQAI as standalone app")

    class Log:
        def d(tag, msg=""): print("DEBUG:", tag, msg)
        def i(tag, msg=""): print("INFO:", tag, msg)
        def w(tag, msg=""): print("WARNING:", tag, msg)
        def e(tag, msg=""): print("ERROR:", tag, msg)

from xml.dom import minidom
from numpy import loadtxt
from PyQt5 import QtCore, QtGui, QtWidgets

import os
import hashlib
from scipy.optimize import curve_fit
import datetime as dt
from types import SimpleNamespace
import webbrowser

try:
    from QATCH.common.licenseManager import LicenseManager, LicenseStatus
    from src.models.formulation import Formulation
    from src.controller.formulation_controller import FormulationController
    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database, DB_PATH
    from src.view.frame_step1 import FrameStep1
    from src.view.frame_step2 import FrameStep2
    from src.view.horizontal_tab_bar import HorizontalTabBar

except (ModuleNotFoundError, ImportError):
    from QATCH.common.licenseManager import LicenseManager, LicenseStatus
    from QATCH.VisQAI.src.models.formulation import Formulation
    from QATCH.VisQAI.src.controller.formulation_controller import FormulationController
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.db.db import Database, DB_PATH
    from QATCH.VisQAI.src.view.frame_step1 import FrameStep1
    from QATCH.VisQAI.src.view.frame_step2 import FrameStep2
    from QATCH.VisQAI.src.view.horizontal_tab_bar import HorizontalTabBar


class BaseVisQAIWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        """BASE CLASS DEFINITION"""
        super().__init__(parent)
        self.parent = parent

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

    def check_license(self, license_manager: Optional[LicenseManager]) -> bool:
        free_preview_period = 90  # days
        is_valid_license = False
        message = ""
        license_data = {}
        if license_manager is not None:
            try:
                is_valid_license, message, license_data = license_manager.validate_license()
                status_raw = license_data.get('status', LicenseStatus.INACTIVE)
                if isinstance(status_raw, str):
                    try:
                        status = LicenseStatus(status_raw)
                    except ValueError:
                        status = LicenseStatus.INACTIVE
                else:
                    status = status_raw
                expiration_str: str = license_data.get('expiration', '')
                if is_valid_license and expiration_str and status != LicenseStatus.ADMIN:
                    try:
                        iso = expiration_str.replace('Z', '+00:00')
                        expiration_date = dt.datetime.fromisoformat(iso)
                        now = dt.datetime.now()
                        days_remaining = (expiration_date - now).days
                        if days_remaining <= 7:
                            message += f"License expires soon!"
                        elif days_remaining <= 30:
                            message += f"Consider renewal"

                    except (ValueError, TypeError):
                        pass
                if license_manager.cache_enabled:
                    cache_status = license_manager.get_cache_status()
                    if cache_status.get('cache_expired', False):
                        Log.d(
                            f"Cache expired, background refresh: {cache_status.get('refresh_thread_active', False)}")

            except Exception as e:
                Log.e(f"License validation error: {e}")
                is_valid_license = False
                message = f"License check failed: {str(e)}"
                license_data = {}
        else:
            is_valid_license = False
            message = "License manager not initialized!"
            license_data = {}
        Log.i(f"{message}")

        # Handle valid license
        if is_valid_license:
            status = license_data.get('status', 'unknown')
            if status == LicenseStatus.ADMIN:
                self.setCentralWidget(self.tab_widget)
                self._update_status_bar(
                    "Licensed: Administrator", permanent=True)

            elif status == LicenseStatus.ACTIVE:
                self.setCentralWidget(self.tab_widget)
                expiration_str: str = license_data.get('expiration', '')
                if expiration_str:
                    try:
                        iso = expiration_str.replace('Z', '+00:00')
                        expiration_date = dt.datetime.fromisoformat(iso)
                        days_remaining = (expiration_date -
                                          dt.datetime.now()).days
                        self._update_status_bar(
                            f"Licensed: {days_remaining} days remaining", permanent=True)
                    except:
                        self._update_status_bar(
                            "Licensed: Active", permanent=True)

            elif status == LicenseStatus.TRIAL:
                self.setCentralWidget(self.tab_widget)
                expiration_str: str = license_data.get('expiration', '')
                if expiration_str:
                    try:
                        iso = expiration_str.replace('Z', '+00:00')
                        expiration_date = dt.datetime.fromisoformat(iso)
                        days_remaining = (expiration_date -
                                          dt.datetime.now()).days

                        if days_remaining <= 7:
                            self._update_status_bar(f"Trial expires in {days_remaining} days!",
                                                    permanent=True,
                                                    style="color: orange;")
                        else:
                            self._update_status_bar(f"Trial: {days_remaining} days remaining",
                                                    permanent=True)
                    except:
                        self._update_status_bar(
                            "Trial: Active", permanent=True)
            else:
                self.setCentralWidget(self.tab_widget)
                status_text = status.value if isinstance(status, LicenseStatus) else str(status)
                self._update_status_bar(f"Licensed: {status_text}", permanent=True)

            return True

        if not os.path.exists(DB_PATH):
            Log.e(f"No license found and no database exists. {message}")
            self.setCentralWidget(self._expired_widget)
            self._update_status_bar(
                "License Required", permanent=True, style="color: red;")
            return False
        try:
            creation_ts = os.stat(DB_PATH).st_ctime
            creation_dt = dt.datetime.fromtimestamp(creation_ts)
            time_ago_seconds = (dt.datetime.now() - creation_dt).total_seconds()
            time_allowed_secs = dt.timedelta(
                days=free_preview_period).total_seconds()

            if time_ago_seconds >= time_allowed_secs:
                Log.w(f"Free preview period expired. {message}")
                self.setCentralWidget(self._expired_widget)
                self._update_status_bar("Preview Expired - License Required",
                                        permanent=True,
                                        style="color: red;")

                if hasattr(self._expired_widget, 'set_message'):
                    days_over = int(
                        (time_ago_seconds - time_allowed_secs) / 86400)
                    self._expired_widget.set_message(
                        f"Your {free_preview_period}-day preview ended {days_over} days ago.\n"
                        f"Please contact support to purchase a license."
                    )

            else:
                days_elapsed = int(time_ago_seconds / 86400)
                self.trial_left = free_preview_period - days_elapsed

                Log.i(
                    f"Free preview: {self.trial_left} days remaining. {message}")

                if hasattr(self, 'trial_label'):
                    trial_text = self.trial_label.text()
                    if '{}' in trial_text:
                        self.trial_label.setText(
                            trial_text.format(self.trial_left))
                    else:
                        self.trial_label.setText(
                            f"Free Preview: {self.trial_left} days remaining")

                if self.trial_left <= 3:
                    self.setCentralWidget(self._trial_widget)
                    self._update_status_bar(f"Preview expires in {self.trial_left} days!",
                                            permanent=True,
                                            style="color: orange; font-weight: bold;")
                elif self.trial_left <= 7:
                    self.setCentralWidget(self._trial_widget)
                    self._update_status_bar(f"Preview: {self.trial_left} days remaining",
                                            permanent=True,
                                            style="color: orange;")
                else:
                    self.setCentralWidget(self._trial_widget)
                    self._update_status_bar(f"Preview: {self.trial_left} days remaining",
                                            permanent=True)

        except Exception as e:
            Log.e(f"Error calculating preview period: {e}")
            self.setCentralWidget(self._expired_widget)
            self._update_status_bar(
                "License Error", permanent=True, style="color: red;")
            return False

        return False

    def _update_status_bar(self, message: str, permanent: bool = False, style: str = None):
        if hasattr(self, 'statusBar'):
            status_bar = self.statusBar()
            if permanent:
                if not hasattr(self, '_license_status_label'):
                    self._license_status_label = QtWidgets.QLabel()
                    status_bar.addPermanentWidget(self._license_status_label)
                self._license_status_label.setText(message)
                if style:
                    self._license_status_label.setStyleSheet(style)
            else:
                timeout = 5000
                status_bar.showMessage(message, timeout)

    def refresh_license_async(self, license_manager: Optional[LicenseManager]):
        if license_manager is None:
            return

        def refresh_callback():
            try:
                is_valid, message, data = license_manager.refresh_license()

                # Update UI in main thread
                QtCore.QTimer.singleShot(0, lambda lm=license_manager: self.check_license(lm))

            except Exception as e:
                Log.e(f"Background license refresh failed: {e}")

        # Start background thread
        import threading
        thread = threading.Thread(target=refresh_callback, daemon=True)
        Log.i("Refreshing VisQAI license lease")
        thread.start()

    def clear(self):
        """BASE CLASS DEFINITION"""
        pass

    def reset(self):
        """BASE CLASS DEFINITION"""
        pass

    def enable(self, enabled=False):
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
        self.setWindowTitle("VisQ.AI")
        self.setMinimumSize(900, 600)
        self.init_ui()
        self.init_sign()
        self.check_license(
            self.parent._license_manager)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(
            lambda: self.refresh_license_async(self.parent._license_manager)
        )
        self.timer.start(3600000)
        self._unsaved_changes = False

        self.select_formulation: Formulation = Formulation()
        self.import_formulations: list[Formulation] = []
        self.predict_formulation: Formulation = Formulation()

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
        self.username = None
        self.initials = None
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
                widget: FrameStep1 = self.tab_widget.widget(
                    0)  # Select
                if tab_step == now_step + 1 and widget.run_file_run:
                    if hasattr(self.tab_widget.currentWidget(), "btn_next"):
                        self.tab_widget.currentWidget().btn_next.click()
                        return True  # ignore click, let "Next" btn decide
                # Block tab change based on some condition
                if tab_step in [3, 4, 6]:
                    if not widget.run_file_run:
                        QtWidgets.QMessageBox.information(
                            None, Constants.app_title,
                            "Please select a run.",
                            QtWidgets.QMessageBox.Ok)
                        return True  # deny tab change
                    # NOTE: No longer a requirement, but can be added back if needed
                    # if tab_step == 4:
                    #     widget: FrameStep1 = self.tab_widget.widget(
                    #         2)  # Import
                    #     if len(widget.all_files) == 0:
                    #         QtWidgets.QMessageBox.information(
                    #             None, Constants.app_title,
                    #             "Please import at least 1 experiment before proceeding.",
                    #             QtWidgets.QMessageBox.Ok)
                    #         return True  # deny tab change
                if now_step >= tab_step:
                    # do not click next when user is going backwards
                    # (or nowhere) from the currently selected step.
                    pass
                elif hasattr(self.tab_widget.currentWidget(), "btn_next") and widget.run_file_run:
                    # still perform click action
                    self.tab_widget.currentWidget().btn_next.click()
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

    def set_global_model_path(self, model_path: Optional[str] = None):
        # Get each widget and call its model select handler (if exists)
        for index in range(self.tab_widget.count()):
            current_widget = self.tab_widget.widget(index)
            if hasattr(current_widget, 'model_selected') and callable(current_widget.model_selected):
                current_widget.model_selected(model_path)

    def enable(self, enable=False) -> None:
        if not enable:
            # VisQ.AI UI is not in foreground, Mode not selected
            # Do things here to shutdown resources and disable:

            # Disable hourly license check timer.
            if hasattr(self, "timer") and self.timer.isActive():
                self.timer.stop()

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

            # Enable hourly license check timer.
            if hasattr(self, "timer") and not self.timer.isActive():
                self.timer.start(3600000)

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
