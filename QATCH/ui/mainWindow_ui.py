
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDesktopWidget
from pyqtgraph import GraphicsLayoutWidget

from QATCH.core.constants import Constants, OperationType
from QATCH.common.architecture import Architecture, OSType
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserProfiles, UserRoles
from QATCH.ui.popUp import PopUp
from QATCH.ui.drawPlateConfig import WellPlate

import datetime
import logging
import math
import os
# import threading


class Ui_Main(object):

    def setupUi(self, MainWindow0, parent):
        USE_FULLSCREEN = (QDesktopWidget().availableGeometry().width() == 2880)
        self.parent = parent

        MainWindow0.setObjectName("MainWindow0")
        MainWindow0.setMinimumSize(QtCore.QSize(1331, 711))
        self.centralwidget = QtWidgets.QWidget(MainWindow0)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        layout_h = QtWidgets.QHBoxLayout()

        # mode menu add here: Run / Analyze
        modewidget = QtWidgets.QWidget()
        modelayout = QtWidgets.QVBoxLayout()
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/nanovisQ-logo.png')
        logo_icon = QtGui.QPixmap(icon_path).scaledToWidth(100)
        self.logolabel = QtWidgets.QLabel()
        self.logolabel.setStyleSheet("padding-bottom:10px;")
        self.logolabel.setPixmap(logo_icon)
        self.logolabel.resize(logo_icon.width(), logo_icon.height())
        self.mode_mode = QtWidgets.QLabel("<b>MODE</b>")
        self.mode_run = QtWidgets.QLabel("Run")
        self.mode_run.mousePressEvent = self.setRunMode
        self.mode_analyze = QtWidgets.QLabel("Analyze")
        self.mode_analyze.mousePressEvent = self.setAnalyzeMode
        modelayout.setContentsMargins(0, 0, 0, 0)
        modelayout.addWidget(self.logolabel)
        modelayout.addWidget(self.mode_mode)
        modelayout.addWidget(self.mode_run)
        modelayout.addWidget(self.mode_analyze)
        modelayout.addStretch()
        modewidget.setLayout(modelayout)
        self.modemenu = QtWidgets.QScrollArea()
        self.modemenu.setStyleSheet("background: #DDDDDD; color: #333333;")
        self.mode_mode.setStyleSheet("padding: 10px; padding-top: 15px;")
        # StyledPanel | QtWidgets.QFrame.Plain)
        self.modemenu.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.modemenu.setLineWidth(0)
        self.modemenu.setMidLineWidth(0)
        self.modemenu.setWidgetResizable(True)
        self.modemenu.setMinimumSize(QtCore.QSize(100, 700))
        self.modemenu.setWidget(modewidget)

        # user sign-in view frame: TODO
        self.userview = QtWidgets.QScrollArea()
        self.userview.setObjectName("userview")
        self.userview.setStyleSheet(
            "#userview {border: 1px solid #DDDDDD; border-radius: 2px;}")
        self.userview.setFrameShape(
            QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Plain)
        self.userview.setLineWidth(0)
        self.userview.setMidLineWidth(0)
        self.userview.setWidgetResizable(True)
        # TODO: Implement user widget
        self.userview.setWidget(parent.LoginWin.ui5.centralwidget)
        self.userview.setMinimumSize(QtCore.QSize(1000, 122))

        # run mode view frame: Controls and Plots
        runwidget = QtWidgets.QWidget()
        runlayout = QtWidgets.QVBoxLayout()
        runlayout.setContentsMargins(0, 0, 0, 0)
        runlayout.addWidget(parent.ControlsWin.ui1.centralwidget, 1)
        runlayout.addWidget(parent.PlotsWin.ui2.centralwidget, 255)
        runwidget.setLayout(runlayout)
        self.runview = QtWidgets.QScrollArea()
        self.runview.setObjectName("runview")
        self.runview.setStyleSheet(
            "#runview {border: 1px solid #DDDDDD; border-radius: 2px;}")
        self.runview.setFrameShape(
            QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Plain)
        self.runview.setLineWidth(0)
        self.runview.setMidLineWidth(0)
        self.runview.setWidgetResizable(True)
        self.runview.setWidget(runwidget)
        self.runview.setMinimumSize(QtCore.QSize(1000, 122))

        # analyze mode view frame: Analyze
        self.analyze = QtWidgets.QScrollArea()
        self.analyze.setObjectName("analyze")
        self.analyze.setStyleSheet(
            "#analyze {border: 1px solid #DDDDDD; border-radius: 2px;}")
        self.analyze.setFrameShape(
            QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Plain)
        self.analyze.setLineWidth(0)
        self.analyze.setMidLineWidth(0)
        self.analyze.setWidgetResizable(True)
        self.analyze.setWidget(parent.AnalyzeProc)
        self.analyze.setMinimumSize(QtCore.QSize(1000, 122))

        # log view frame: Logger
        self.logview = QtWidgets.QScrollArea()
        self.logview.setObjectName("logview")
        self.logview.setStyleSheet(
            "#logview {border: 1px solid #DDDDDD; border-radius: 2px;}")
        self.logview.setFrameShape(
            QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Plain)
        self.logview.setLineWidth(0)
        self.logview.setMidLineWidth(0)
        self.logview.setWidgetResizable(True)
        self.logview.setWidget(parent.LogWin.ui4.centralwidget)
        self.logview.setMinimumSize(QtCore.QSize(1000, 166))

        layout_h.addWidget(self.modemenu, 1)
        layout_v = QtWidgets.QVBoxLayout()
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        if UserProfiles.count() > 0:
            # NOTE: this widget must not be changed at load time (or else it disappears)
            self.splitter.addWidget(self.userview)
        else:
            self.splitter.addWidget(self.runview)
        self.splitter.addWidget(self.logview)
        self.splitter.setSizes([1000, 1])
        layout_v.addWidget(self.splitter)
        copy_foot = QtWidgets.QLabel("<center>&copy; {} QATCH Technologies. All rights reserved.</center>"
                                     .format(datetime.date.today().year))  # use <center> tag to force HTML parsing of &copy; symbol
        # copy_foot.setAlignment(QtCore.Qt.AlignCenter)
        copy_foot.setContentsMargins(0, 0, 0, 0)
        copy_foot.setFixedHeight(20)
        layout_v.addWidget(copy_foot)
        layout_h.addLayout(layout_v, 255)

        # add collapse/expand icon arrows
        self.splitter.setHandleWidth(10)
        handle = self.splitter.handle(1)
        layout_s = QtWidgets.QHBoxLayout()
        layout_s.setContentsMargins(0, 0, 0, 0)
        layout_s.addStretch()
        self.btnCollapse = QtWidgets.QToolButton(handle)
        self.btnCollapse.setArrowType(QtCore.Qt.DownArrow)
        self.btnCollapse.clicked.connect(
            lambda: self.handleSplitterButton(True))
        layout_s.addWidget(self.btnCollapse)
        self.btnExpand = QtWidgets.QToolButton(handle)
        self.btnExpand.setArrowType(QtCore.Qt.UpArrow)
        self.btnExpand.clicked.connect(
            lambda: self.handleSplitterButton(False))
        layout_s.addWidget(self.btnExpand)
        layout_s.addStretch()
        handle.setLayout(layout_s)
        self.btnExpand.setVisible(False)
        # self.handleSplitterButton(False)
        self.splitter.splitterMoved.connect(self.handleSplitterMoved)

        # self.splitter.replaceWidget(0, self.userview)
        self._force_splitter_mode_set = True
        if UserProfiles.count() > 0:
            self.setNoUserMode(self.mode_mode)
        else:
            # TODO: implement user sign in widget (show accordingly)
            self.setRunMode(self.mode_run)
        self._force_splitter_mode_set = False
        # NOTE: splitter[0] widget must not change at load or else it disappears
        # (ignore the warning: "Trying to replace a widget with itself")

        # retain sizing of view menu toggle elements
        elems = [
            parent.LogWin.ui4.centralwidget,
            parent.PlotsWin.ui2.centralwidget
        ]
        for e in elems:
            not_resize = e.sizePolicy()
            not_resize.setHorizontalStretch(1)
            e.setSizePolicy(not_resize)

        elems = [
            parent.PlotsWin.ui2.plt,
            parent.PlotsWin.ui2.pltB
        ]
        for i, e in enumerate(elems):
            not_resize = e.sizePolicy()
            not_resize.setVerticalStretch(i+2)
            e.setSizePolicy(not_resize)

        self.centralwidget.setLayout(layout_h)
        MainWindow0.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow0)

    def handleSplitterMoved(self, pos, index):
        collapsed = self.splitter.sizes()[-1] == 0
        self.btnCollapse.setVisible(not collapsed)
        self.btnExpand.setVisible(collapsed)

    def handleSplitterButton(self, collapse=True):
        if collapse:
            self.btnCollapse.setVisible(False)
            self.btnExpand.setVisible(True)
            self.splitter.setSizes([1, 0])
        else:
            self.btnCollapse.setVisible(True)
            self.btnExpand.setVisible(False)
            height = self.splitter.height()
            log_size = self.logview.minimumHeight()
            self.splitter.setSizes([height-log_size, log_size])

    def setNoUserMode(self, obj):

        # DO NOT COMMIT
        # import pyautogui
        # pyautogui.typewrite("AKM")
        # pyautogui.press("enter")
        # pyautogui.typewrite("12345678")
        # pyautogui.press("enter")

        if self.splitter.widget(0) == self.userview and not self._force_splitter_mode_set:
            Log.d("User sign-in mode already active. Skipping mode change request.")
            if obj == None:
                return True
            # return # do not return when 'obj != None' to allow mode button styles to be set on 'setupUi()'' call
        if self.parent.AnalyzeProc.hasUnsavedChanges():
            if PopUp.question(self, Constants.app_title, "You have unsaved changes!\n\nAre you sure you want to close this window?", False):
                self.parent.AnalyzeProc.clear()  # lose unsaved changes
        if not self.parent.AnalyzeProc.hasUnsavedChanges():
            self.mode_run.setStyleSheet("padding: 10px; padding-left: 15px;")
            self.mode_analyze.setStyleSheet(
                "padding: 10px; padding-left: 15px;")
            self.splitter.replaceWidget(0, self.userview)
            # login, forgot pw, create user (must match pages in _configure_tutorials() too)
            self.parent.viewTutorialPage([1, 2, 0])
            QtCore.QTimer.singleShot(
                500, self.parent.LoginWin.ui5.user_initials.setFocus)
            if obj == None:
                if not UserProfiles.session_info()[0]:  # user session expired
                    self.parent.LoginWin.ui5.error_expired()
                else:  # user manually logged out
                    self.parent.LoginWin.ui5.error_loggedout()
                # note: 'session_end()' will be called by caller (if need be)
                return True
        else:
            Log.e(
                "Please \"Analyze\" to save or \"Close\" to lose your changes before switching modes.")
        if obj == None:
            return False

    def setRunMode(self, obj):
        if self.splitter.widget(0) == self.runview and not self._force_splitter_mode_set:
            Log.d("Run mode already active. Skipping mode change request.")
            if obj == None:
                return True
            return
        if self.parent.AnalyzeProc.hasUnsavedChanges():
            if PopUp.question(self, Constants.app_title, "You have unsaved changes!\n\nAre you sure you want to close this window?", False):
                self.parent.AnalyzeProc.clear()  # lose unsaved changes
        if not self.parent.AnalyzeProc.hasUnsavedChanges():
            action_role = UserRoles.CAPTURE
            check_result = UserProfiles().check(self.parent.ControlsWin.userrole, action_role)
            if check_result == None:  # user check required, but no user signed in
                Log.w(
                    f"Not signed in: User with role {action_role.name} is required to perform this action.")
                Log.i("Please sign in to continue.")
                self.parent.ControlsWin.set_user_profile()  # prompt for sign-in
                check_result = UserProfiles().check(
                    self.parent.ControlsWin.userrole, action_role)  # check again
            if check_result:
                self.parent._enable_ui(True)
                self.mode_run.setStyleSheet(
                    "padding: 10px; padding-left: 15px; background: #B7D3DC;")
                self.mode_analyze.setStyleSheet(
                    "padding: 10px; padding-left: 15px;")
                self.splitter.replaceWidget(0, self.runview)
                if UserProfiles.count() == 0:
                    # measure, next steps, create accounts (must match pages in _configure_tutorials() too)
                    self.parent.viewTutorialPage([3, 4, 0])
                else:
                    # measure / next steps (must match pages in _configure_tutorials() too, without page 0)
                    self.parent.viewTutorialPage([3, 4])
                if obj == None:
                    return True
            elif check_result == None:
                Log.w(
                    f"ACTION DENIED: User with role {self.parent.ControlsWin.userrole.name} does not have permission to {action_role.name}.")
                Log.e("Please sign in to access Run mode.")
            else:
                Log.w(
                    f"ACTION DENIED: User with role {self.parent.ControlsWin.userrole.name} does not have permission to {action_role.name}.")
                Log.e("You are not authorized to access Run mode.")
        else:
            Log.e(
                "Please \"Analyze\" to save or \"Close\" to lose your changes before switching modes.")
        if obj == None:
            return False

    def setAnalyzeMode(self, obj):
        if self.splitter.widget(0) == self.analyze and not self._force_splitter_mode_set:
            Log.d("Analyze mode already active. Skipping mode change request.")
            if obj == None:
                return True
            return
        if self.parent.ControlsWin.ui1.pButton_Start.isEnabled():
            self.parent.analyze_data()
            action_role = UserRoles.ANALYZE
            check_result = UserProfiles().check(self.parent.ControlsWin.userrole, action_role)
            if check_result:
                self.parent._enable_ui(False)
                self.mode_run.setStyleSheet(
                    "padding: 10px; padding-left: 15px;")
                self.mode_analyze.setStyleSheet(
                    "padding: 10px; padding-left: 15px; background: #B7D3DC;")
                self.splitter.replaceWidget(0, self.analyze)
                self.parent.viewTutorialPage([5, 6])  # analyze / prior results
                if obj == None:
                    return True
            elif check_result == None:
                Log.e("Please sign in to access Analyze mode.")
            else:
                Log.e("You are not authorized to access Analyze mode.")
        else:
            Log.e("Please stop the current run before switching modes.")
        if obj == None:
            return False

    def retranslateUi(self, MainWindow0):
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/qatch-icon.png')
        MainWindow0.setWindowIcon(QtGui.QIcon(icon_path))  # .png
        MainWindow0.setWindowTitle(_translate("MainWindow0", "{} {}".format(
            Constants.app_title, Constants.app_version)))


class Ui_Login(object):
    """
    User Interface for the Login Window.

    This class provides and manages the login interface for the QATCH application.
    It sets up and configures the UI elements for user authentication, including labels,
    text fields, buttons, and password visibility toggles. The class also handles events
    such as text transformation, session validation, and error reporting, ensuring a smooth
    user experience during the sign-in process.

    The UI is dynamically updated to reflect changes such as password visibility toggling,
    session expiration, and invalid credential notifications. Timers are used to clear error
    messages after a brief interval and to periodically check the validity of the user session.
    An action-sign in method is also provided to process and validate user credentials upon
    clicking the sign-in button.

    Attributes:
        parent (QtWidgets.QMainWindow): The parent window that the login UI interacts with.
        caps_lock_on (bool): Indicates if the Caps Lock is active.
        centralwidget (QtWidgets.QWidget): The main widget containing the login interface.
        layout (QtWidgets.QGridLayout): Grid layout manager for arranging UI components.
        user_welcome (QtWidgets.QLabel): Label displaying a welcome message.
        user_label (QtWidgets.QLabel): Label prompting the user to sign in.
        user_initials (QtWidgets.QLineEdit): Text field for entering user initials.
        user_password (QtWidgets.QLineEdit): Text field for entering the password, supporting masked input.
        sign_in (QtWidgets.QPushButton): Button that initiates the sign-in process.
        user_info (QtWidgets.QLabel): Label for displaying informational messages.
        user_error (QtWidgets.QLabel): Label for displaying error messages.
        _errorTimer (QtCore.QTimer): Timer to clear error messages after a set interval.
        _sessionTimer (QtCore.QTimer): Timer to periodically validate the user session.
        visibleIcon (QtGui.QIcon): Icon shown when the password is visible.
        hiddenIcon (QtGui.QIcon): Icon shown when the password is hidden.
        password_shown (bool): Flag indicating whether the password is currently shown in plain text.
        togglepasswordAction (QAction): Action to toggle password visibility.

    Methods:
        setupUi(MainWindow5: QtWidgets.QMainWindow, parent: QtWidgets.QMainWindow) -> None:
            Initializes and arranges all UI components in the main window.
        retranslateUi(MainWindow5: QtWidgets.QMainWindow) -> None:
            Updates the window's icon and title based on localization and application settings.
        on_toggle_password_Action() -> None:
            Toggles the echo mode of the password field between masked and unmasked.
        check_user_session() -> None:
            Checks if the user session is valid; prompts re-authentication if the session has expired.
        error_loggedout() -> None:
            Displays an error message indicating that the user has been signed out.
        error_invalid() -> None:
            Displays an error message for invalid login credentials.
        error_expired() -> None:
            Displays an error message indicating that the user session has expired.
        text_transform() -> None:
            Converts the input text in the user initials field to uppercase.
        action_sign_in() -> None:
            Handles the sign-in process by validating credentials and initiating the user session.

    Example:
        >>> main_window = QtWidgets.QMainWindow()
        >>> parent_window = QtWidgets.QMainWindow()
        >>> login_ui = Ui_Login()
        >>> login_ui.setupUi(main_window, parent_window)
    """

    def setupUi(self, MainWindow5: QtWidgets.QMainWindow, parent: QtWidgets.QMainWindow) -> None:
        """Set up and configure the login user interface for the main window.

        This method initializes and arranges all the UI elements required for a user
        login interface within the provided main window. It creates labels, line edits,
        and buttons, sets their properties (such as size, alignment, placeholder texts,
        and event filters), and organizes them using grid and vertical layouts. In addition,
        the method configures timers for error message handling and session checks, and
        sets up a password visibility toggle with corresponding icons.

        The Caps Lock key state is also checked at initialization to adjust the UI behavior
        if needed.

        Args:
            MainWindow5 (QtWidgets.QMainWindow): The main window instance where the login UI
                is to be set up.
            parent (QtWidgets.QMainWindow): The parent window that may be used for event filtering
                and further interactions. This is stored in the instance as `self.parent`.

        Returns:
            None
        """
        self.parent = parent

        # Variable to check and store the state of the Caps Lock key on init.
        self.caps_lock_on = False  # set on focus of `user_password` field

        MainWindow5.setObjectName("MainWindow5")
        MainWindow5.setMinimumSize(QtCore.QSize(1000, 500))
        MainWindow5.resize(500, 500)
        MainWindow5.setStyleSheet("")
        MainWindow5.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow5)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        # TODO: user grid layout to narrow width and height with spacers
        self.layout = QtWidgets.QGridLayout()
        self.user_welcome = QtWidgets.QLabel(
            "<span style='font-size: 14pt'>Welcome to QATCH nanovisQ<sup>TM</sup> Real-Time GUI </span>")
        self.user_welcome.setAlignment(QtCore.Qt.AlignCenter)
        self.user_welcome.setFixedHeight(50)
        self.user_label = QtWidgets.QLabel(
            "<span style='font-size: 10pt'><b>User Sign-In Required</b></span>")
        self.user_label.setFixedHeight(50)
        self.layout.addWidget(self.user_label, 2, 1, 1,
                              3, QtCore.Qt.AlignCenter)
        self.user_initials = QtWidgets.QLineEdit()
        self.user_initials.textEdited.connect(self.text_transform)
        self.user_initials.setMinimumWidth(190)
        self.user_initials.setPlaceholderText("Initials")
        self.user_initials.setMaxLength(4)
        self.user_initials.installEventFilter(MainWindow5)
        self.layout.addWidget(self.user_initials, 3, 2,
                              1, 1, QtCore.Qt.AlignCenter)
        self.user_password = QtWidgets.QLineEdit()
        self.user_password.setMinimumWidth(190)
        self.user_password.setPlaceholderText("Password")
        self.user_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.user_password.installEventFilter(MainWindow5)
        self.layout.addWidget(self.user_password, 4, 2,
                              1, 1, QtCore.Qt.AlignCenter)
        self.sign_in = QtWidgets.QPushButton("&Sign In")
        self.sign_in.setMinimumWidth(190)
        self.sign_in.clicked.connect(self.action_sign_in)
        self.sign_in.installEventFilter(MainWindow5)
        self.layout.addWidget(self.sign_in, 5, 2, 1, 1, QtCore.Qt.AlignCenter)

        # User information message box on Login Window.
        self.user_info = QtWidgets.QLabel("")
        self.user_info.setStyleSheet("color: #000000; font-weight: bold;")
        # self.user_info.setFixedHeight(50)
        self.layout.addWidget(self.user_info, 6, 1, 1,
                              3, QtCore.Qt.AlignCenter)

        self.user_error = QtWidgets.QLabel("")
        self.user_error.setStyleSheet("color: #ff0000;")
        # self.user_label.setAlignment(QtCore.Qt.AlignCenter)
        # self.user_error.setFixedHeight(50)
        self.layout.addWidget(self.user_error, 7, 1, 1,
                              3, QtCore.Qt.AlignCenter)

        v_layout = QtWidgets.QVBoxLayout()
        v_layout.addStretch()
        v_layout.addWidget(self.user_welcome)
        v_layout.addStretch()
        v_layout.addLayout(self.layout)
        v_layout.addStretch()
        v_layout.addStretch()
        v_layout.addStretch()
        # logo_icon = QtGui.QPixmap("QATCH/icons/Qatch-logo_hi-res.jpg").scaledToWidth(400)
        # self.logolabel = QtWidgets.QLabel()
        # self.logolabel.setStyleSheet("padding-bottom:10px;")
        # self.logolabel.setPixmap(logo_icon)
        # self.logolabel.resize(logo_icon.width(), logo_icon.height())
        # v_layout.addWidget(self.logolabel, alignment=QtCore.Qt.AlignCenter)
        self.centralwidget.setLayout(v_layout)

        self._errorTimer = QtCore.QTimer()
        self._errorTimer.setSingleShot(True)
        self._errorTimer.timeout.connect(self.user_error.clear)

        self._sessionTimer = QtCore.QTimer()
        self._sessionTimer.setSingleShot(True)
        self._sessionTimer.timeout.connect(self.check_user_session)
        self._sessionTimer.setInterval(1000*60*60)  # once an hour

        self.visibleIcon = QtGui.QIcon(os.path.join(
            Architecture.get_path(), "QATCH", "icons", "eye.svg"))
        self.hiddenIcon = QtGui.QIcon(os.path.join(
            Architecture.get_path(), "QATCH", "icons", "hide.svg"))

        # Add the password hide/shown toggle at the end of the edit box.
        self.password_shown = False
        self.togglepasswordAction = self.user_password.addAction(
            self.visibleIcon, QtWidgets.QLineEdit.TrailingPosition)
        self.togglepasswordAction.triggered.connect(
            self.on_toggle_password_Action)

    def on_toggle_password_Action(self) -> None:
        """Toggle the visibility of the password input field.

        This method switches the echo mode of the `user_password` QLineEdit widget between
        normal text display and password (masked) mode. When the password is hidden (echo mode
        set to Password), activating this method will make it visible by setting the echo mode to
        Normal, update the internal state flag `password_shown` to True, and change the toggle icon
        to indicate that clicking it again will hide the password. Conversely, if the password is
        currently visible, the method sets the echo mode back to Password, updates `password_shown` to
        False, and resets the icon to the visible state.

        Returns:
            None
        """
        if not self.password_shown:
            self.user_password.setEchoMode(QtWidgets.QLineEdit.Normal)
            self.password_shown = True
            self.togglepasswordAction.setIcon(self.hiddenIcon)
        else:
            self.user_password.setEchoMode(QtWidgets.QLineEdit.Password)
            self.password_shown = False
            self.togglepasswordAction.setIcon(self.visibleIcon)

    def check_user_session(self) -> None:
        """Check the current user's session and prompt for sign-in if expired.

        This method retrieves session validity and associated information by invoking
        `UserProfiles().session_info()`. If the session is found to be invalid (i.e., the
        user is not signed in or the session has expired), a warning is logged, an
        informational message is provided, and the user is prompted to sign in by calling
        `self.parent.ControlsWin.set_user_profile()`. If the session is still valid, a
        debug message is logged and the session timer is restarted to re-check after one hour.

        Returns:
            None
        """
        valid, infos = UserProfiles().session_info()
        if not valid:  # user check required, but no user signed in
            Log.w(f"User session has expired.")
            Log.i("Please sign in to continue.")
            self.parent.ControlsWin.set_user_profile()  # prompt for sign-in
            # self.parent.LoginWin.ui5.error_expired()
        else:
            Log.d("User session is still valid at the hourly check.")
            self._sessionTimer.start()  # check again in another hour

    def retranslateUi(self, MainWindow5: QtWidgets.QMainWindow) -> None:
        """Update the main window's icon and title with localized content.

        This method retrieves translation functions and sets up the main window's
        appearance by loading an icon from the predefined file path and updating
        the window title with the application title and version. The title is formatted
        to include a "Login" suffix, indicating the purpose of the window.

        Args:
            MainWindow5 (QtWidgets.QMainWindow): The main window instance that will have its
                icon and title updated.
        """
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/qatch-icon.png')
        MainWindow5.setWindowIcon(QtGui.QIcon(icon_path))  # .png
        MainWindow5.setWindowTitle(_translate(
            "MainWindow5", "{} {} - Login".format(Constants.app_title, Constants.app_version)))

    def error_loggedout(self) -> None:
        """Display a logout error message and initiate a timer to clear it.

        This method triggers a timer using `kickErrorTimer` that will clear the displayed error message
        after a predefined duration. It then sets the error message on the `user_error`
        widget to notify the user that they have been signed out.

        Returns:
            None
        """
        self.kickErrorTimer()  # the following text will clear in 10s
        self.user_error.setText("You have been signed out")

    def error_invalid(self) -> None:
        """Display an error message for invalid credentials and schedule its clearance.

        This method triggers a timer via `kickErrorTimer` to clear the error message after a
        predefined interval. It then sets the error message on the
        `user_error` widget to inform the user that the credentials provided are invalid.

        Returns:
            None
        """
        self.kickErrorTimer()  # the following text will clear in 10s
        self.user_error.setText("Invalid credentials")

    def error_expired(self):
        # self.kickErrorTimer() # the following text will clear in 10s
        self.user_error.setText("Your session has expired")

    def kickErrorTimer(self) -> None:
        """Display an error message indicating that the user session has expired.

        This method sets the error message on the `user_error` widget to inform the user that
        their session has expired. The code for initiating a timer to clear the message (via
        `kickErrorTimer`) is commented out, suggesting that the clearance behavior might be
        managed elsewhere or was disabled intentionally.

        Returns:
            None
        """
        if self._errorTimer.isActive():
            Log.d("Error Timer was restarted while running")
            self._errorTimer.stop()
        self._errorTimer.start(10000)

    def text_transform(self):
        """Convert the input text in the user initials field to uppercase.

        This method retrieves the current text from the `user_initials` QLineEdit widget.
        If the text is non-empty, it converts the text to uppercase and updates the widget.
        The update does not re-trigger the 'textEdited' signal, preventing potential recursive calls.

        Returns:
            None
        """
        text = self.user_initials.text()
        if len(text) > 0:
            # will not fire 'textEdited' signal again
            self.user_initials.setText(text.upper())

    def action_sign_in(self) -> None:
        """ Perform the sign-in action for a user given initials and a password.

        This method attempts to sign in a user given that their provided initials and password
        are non-empty and valid.  Upon sign-in, a new user session is opened along with as session timer.
        If the use could not be authenticated, sign-in is skipped and the sign-form is cleared.

        Returns:
            None
        """
        if len(self.user_initials.text()) == 0 and len(self.user_password.text()) == 0:
            Log.d("No initials or password entered. Ignoring Sign-In request.")
            return

        initials = self.user_initials.text().upper()
        pwd = self.user_password.text()
        requiredRole = UserRoles.ANY
        authenticated, filename, params = UserProfiles.auth(
            initials, pwd, requiredRole)
        if authenticated:
            Log.i(
                f"Welcome, {params[0]}! Your assigned role is {params[2].name}.")
            name, init, role = params[0], params[1], params[2].value
            self._sessionTimer.start()  # check session every hour
        else:
            name, init, role = None, None, 0
        self.clear_form()

        if name != None:
            self.parent.ControlsWin.username.setText(f"User: {name}")
            self.parent.ControlsWin.userrole = UserRoles(role)
            self.parent.ControlsWin.signinout.setText("&Sign Out")
            self.parent.ControlsWin.ui1.tool_User.setText(name)
            self.parent.AnalyzeProc.tool_User.setText(name)
            if self.parent.ControlsWin.userrole != UserRoles.ADMIN:
                self.parent.ControlsWin.manage.setText("&Change Password...")

            action_role = UserRoles.CAPTURE
            check_result = UserProfiles().check(self.parent.ControlsWin.userrole, action_role)
            if check_result:
                self.parent.MainWin.ui0.setRunMode(self.user_label)
            else:
                self.parent.MainWin.ui0.setAnalyzeMode(self.user_label)

            action_role = UserRoles.ADMIN
            check_result = UserProfiles().check(self.parent.ControlsWin.userrole, action_role)
            if check_result:
                enabled, error, expires = UserProfiles.checkDevMode()
                if enabled != True and error != False:
                    is_expired = True if expires != "" else False
                    messagebox_description = ("<b>Developer Mode " +
                                              ("has expired" if is_expired else "is invalid") +
                                              " and is no longer active!</b>")  # requires renewal to use.")
                    from QATCH.common.userProfiles import UserProfilesManager, UserConstants
                    if PopUp.question(self.parent, "Developer Mode " + ("Expired" if is_expired else "Error"),
                                      messagebox_description + "<br/>" +
                                      f"Renewal Period: Every {UserConstants.DEV_EXPIRE_LEN} days<br/><br/>" +
                                      "Would you like to renew Developer Mode now?<br/><br/>" +
                                      "<small>NOTE: This setting can be changed in the \"Manage Users\" window.</small>"):
                        temp_upm = UserProfilesManager(self.parent, name)
                        # triggers call to 'toggleDevMode' on state change
                        temp_upm.developerModeChk.setChecked(True)
                        Log.i("Developer Mode renewed!")
                    else:
                        Log.w("Developer Mode NOT renewed!")

            # check for updates, if and only if ADMIN (as required)
            if hasattr(self.parent, "url_download"):
                delattr(self.parent, "url_download")  # require re-ask
            QtCore.QTimer.singleShot(1, self.parent.start_download)
        else:
            self.error_invalid()

    def clear_form(self) -> None:
        """Clears the user initials and password form and user error message box.

        If password is shown, revert the password.  Additionally, revert focus back
        to initials field on clear.

        Returns:
            None
        """
        self.user_initials.clear()
        self.user_password.clear()
        self.user_error.clear()

        if self.password_shown:  # revert to password (if shown)
            self.on_toggle_password_Action()

        if self.user_password.hasFocus():
            self.user_initials.setFocus()

    def update_caps_lock_state(self, caps_lock_state: bool) -> None:
        """
        Method to update the message in the user_info message box
        to inform the user that Caps Lock is on.

        Args:
            caps_lock_state (bool): The state of the Caps Lock on this device

        Returns:
            None
        """
        if caps_lock_state:
            self.user_info.setText("Caps Lock is On")
        else:
            self.user_info.clear()


class Ui_Controls(object):  # QtWidgets.QMainWindow

    def setupUi(self, MainWindow1):
        USE_FULLSCREEN = (QDesktopWidget().availableGeometry().width() == 2880)
        SHOW_SIMPLE_CONTROLS = True
        self.cal_initialized = False

        MainWindow1.setObjectName("MainWindow1")
        # MainWindow1.setGeometry(50, 50, 975, 70)
        # MainWindow1.setFixedSize(980, 150)
        # MainWindow1.resize(550, 50)
        MainWindow1.setMinimumSize(QtCore.QSize(1000, 50))
        if Architecture.get_os() is OSType.macosx:
            MainWindow1.resize(1080, 188)
        elif USE_FULLSCREEN:
            MainWindow1.resize(2880, 390)
            MainWindow1.move(0, 1485)
        else:
            MainWindow1.resize(1503, 175)
            MainWindow1.move(7, 567)
        MainWindow1.setStyleSheet("")
        MainWindow1.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow1)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.Layout_controls = QtWidgets.QGridLayout()
        self.Layout_controls.setObjectName("Layout_controls")

        # frequency/quartz combobox -------------------------------------------
        self.cBox_Speed = QtWidgets.QComboBox()
        self.cBox_Speed.setEditable(False)
        self.cBox_Speed.setObjectName("cBox_Speed")
        if USE_FULLSCREEN:
            self.cBox_Speed.setFixedHeight(50)
        self.Layout_controls.addWidget(self.cBox_Speed, 4, 1, 1, 1)

        # stop button ---------------------------------------------------------
        self.pButton_Stop = QtWidgets.QPushButton()
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/stop_icon.ico')
        self.pButton_Stop.setIcon(QtGui.QIcon(
            QtGui.QPixmap(icon_path)))  # .png
        self.pButton_Stop.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Stop.setObjectName("pButton_Stop")
        if USE_FULLSCREEN:
            self.pButton_Stop.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pButton_Stop, 3, 6, 1, 1)

        # COM port combobox ---------------------------------------------------
        self.cBox_Port = QtWidgets.QComboBox()
        self.cBox_Port.setEditable(False)
        self.cBox_Port.setObjectName("cBox_Port")
        if USE_FULLSCREEN:
            self.cBox_Port.setFixedHeight(50)
        self.Layout_controls.addWidget(self.cBox_Port, 2, 1, 1, 1)

        # Identify button ---------------------------------------------------------
        self.pButton_ID = QtWidgets.QPushButton()
        self.pButton_ID.setToolTip("Identify selected Serial COM Port")
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/identify-icon.png')
        self.pButton_ID.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))  # .png
        self.pButton_ID.setStyleSheet("background:white;padding:3px;")
        if USE_FULLSCREEN:
            self.pButton_ID.setMinimumSize(QtCore.QSize(60, 50))
        else:
            self.pButton_ID.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_ID.setObjectName("pButton_ID")
        self.Layout_controls.addWidget(self.pButton_ID, 2, 2, 1, 1)

        # Refresh button ---------------------------------------------------------
        self.pButton_Refresh = QtWidgets.QPushButton()
        self.pButton_Refresh.setToolTip("Refresh Serial COM Port list")
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/refresh-icon.png')
        self.pButton_Refresh.setIcon(
            QtGui.QIcon(QtGui.QPixmap(icon_path)))  # .png
        self.pButton_Refresh.setStyleSheet(
            "background:white;padding:3px;margin-right:9px;")
        if USE_FULLSCREEN:
            self.pButton_Refresh.setMinimumSize(QtCore.QSize(70, 50))
        else:
            self.pButton_Refresh.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Refresh.setObjectName("pButton_Refresh")
        self.Layout_controls.addWidget(self.pButton_Refresh, 2, 3, 1, 1)

        # Operation mode - source ---------------------------------------------
        self.cBox_Source = QtWidgets.QComboBox()
        self.cBox_Source.setObjectName("cBox_Source")
        if USE_FULLSCREEN:
            self.cBox_Source.setFixedHeight(50)
        self.Layout_controls.addWidget(self.cBox_Source, 2, 0, 1, 1)

        # Frequency hopping checkbox ------------------------------------------
        self.chBox_freqHop = QtWidgets.QCheckBox()
        self.chBox_freqHop.setEnabled(True)
        self.chBox_freqHop.setChecked(False)
        self.chBox_freqHop.setObjectName("chBox_freqHop")
        self.Layout_controls.addWidget(self.chBox_freqHop, 4, 2, 1, 2)

        # Noise correction checkbox ------------------------------------------
        self.chBox_correctNoise = QtWidgets.QCheckBox()
        self.chBox_correctNoise.setEnabled(True)
        self.chBox_correctNoise.setChecked(True)
        self.chBox_correctNoise.setObjectName("chBox_correctNoise")
        # self.chBox_correctNoise.setVisible(False)
        self.Layout_controls.addWidget(self.chBox_correctNoise, 5, 1, 1, 3)

        # start button --------------------------------------------------------
        self.pButton_Start = QtWidgets.QPushButton()
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/start_icon.ico')
        self.pButton_Start.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Start.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Start.setObjectName("pButton_Start")
        if USE_FULLSCREEN:
            self.pButton_Start.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pButton_Start, 2, 6, 1, 1)

        # clear plots button --------------------------------------------------
        self.pButton_Clear = QtWidgets.QPushButton()
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/clear_icon.ico')
        self.pButton_Clear.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Clear.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Clear.setObjectName("pButton_Clear")
        if USE_FULLSCREEN:
            self.pButton_Clear.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pButton_Clear, 2, 5, 1, 1)

        # reference button ----------------------------------------------------
        self.pButton_Reference = QtWidgets.QPushButton()
        # self.pButton_Reference.setIcon(QtGui.QIcon(QtGui.QPixmap("ref_icon.ico")))
        self.pButton_Reference.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Reference.setObjectName("pButton_Reference")
        self.pButton_Reference.setCheckable(True)
        if USE_FULLSCREEN:
            self.pButton_Reference.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pButton_Reference, 3, 5, 1, 1)

        # restore factory defaults --------------------------------------------
        self.pButton_ResetApp = QtWidgets.QPushButton()
        self.pButton_ResetApp.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_ResetApp.setObjectName("pButton_ResetApp")
        if USE_FULLSCREEN:
            self.pButton_ResetApp.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pButton_ResetApp, 4, 5, 1, 1)

        # samples SpinBox -----------------------------------------------------
        self.sBox_Samples = QtWidgets.QSpinBox()
        self.sBox_Samples.setMinimum(1)
        self.sBox_Samples.setMaximum(100000)
        self.sBox_Samples.setProperty("value", 500)
        self.sBox_Samples.setObjectName("sBox_Samples")
        # self.sBox_Samples.setEnabled(False)
        self.sBox_Samples.setVisible(False)
        self.Layout_controls.addWidget(self.sBox_Samples, 2, 4, 1, 1)

        # export file CheckBox ------------------------------------------------
        self.chBox_export = QtWidgets.QCheckBox()
        self.chBox_export.setEnabled(True)
        self.chBox_export.setObjectName("chBox_export")
        self.chBox_export.setVisible(False)
        self.Layout_controls.addWidget(self.chBox_export, 4, 4, 1, 1)

        # temperature Control slider ------------------------------------------
        self.slTemp = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slTemp.setMinimum(8)
        self.slTemp.setMaximum(40)
        self.slTemp.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slTemp.setTickInterval(1)
        self.slTemp.setSingleStep(1)
        self.slTemp.setPageStep(5)
        self.Layout_controls.addWidget(self.slTemp, 3, 4, 1, 1)

        # temperature Control label -------------------------------------------
        self.lTemp = QtWidgets.QLabel()
        # self.lTemp.setStyleSheet('background: #008EC0; padding: 1px;')
        self.lTemp.setText("PV:--.--C SP:--.--C OP:----")
        self.lTemp.setAlignment(QtCore.Qt.AlignCenter)
        self.lTemp.setFont(QtGui.QFont('Consolas', -1))
        # self.lTemp.setFixedHeight(15)
        self.Layout_controls.addWidget(self.lTemp, 2, 4, 1, 1)

        # temperature Control label -------------------------------------------
        self.pTemp = QtWidgets.QPushButton()
        # self.pTemp.setStyleSheet('background: #008EC0; padding: 1px;')
        self.pTemp.setText("Start Temp Control")
        # self.pTemp.setAlignment(QtCore.Qt.AlignCenter)
        if USE_FULLSCREEN:
            self.pTemp.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pTemp, 4, 4, 1, 1)

        # Samples Number / History Buffer Size---------------------------------
        # self.l5 = QtWidgets.QLabel()
        # self.l5.setStyleSheet('background: #008EC0; padding: 1px;')
        # self.l5.setText("<font color=#ffffff > Samples Number / History Buffer Size </font>")
        # self.l5.setFixedHeight(15)
        # self.Layout_controls.addWidget(self.l5, 1, 4, 1, 1)

        # Control Buttons------------------------------------------------------
        self.l = QtWidgets.QLabel()
        self.l.setStyleSheet('background: #008EC0; padding: 1px;')
        self.l.setText("<font color=#ffffff > Control Buttons </font>")
        if USE_FULLSCREEN:
            self.l.setFixedHeight(50)
        # else:
        #    self.l.setFixedHeight(15)
        self.Layout_controls.addWidget(self.l, 1, 5, 1, 2)

        # Operation Mode ------------------------------------------------------
        self.l0 = QtWidgets.QLabel()
        self.l0.setStyleSheet('background: #008EC0; padding: 1px;')
        self.l0.setText("<font color=#ffffff >Operation Mode</font> </a>")
        if USE_FULLSCREEN:
            self.l0.setFixedHeight(50)
        # else:
        #    self.l0.setFixedHeight(15)
        self.Layout_controls.addWidget(self.l0, 1, 0, 1, 1)

        # Resonance Frequency / Quartz Sensor ---------------------------------
        self.l2 = QtWidgets.QLabel()
        self.l2.setStyleSheet('background: #008EC0; padding: 1px;')
        self.l2.setText(
            "<font color=#ffffff > Resonance Frequency / Quartz Sensor </font>")
        if USE_FULLSCREEN:
            self.l2.setFixedHeight(50)
        # else:
        #    self.l2.setFixedHeight(15)
        self.Layout_controls.addWidget(self.l2, 3, 1, 1, 3)

        # Serial COM Port -----------------------------------------------------
        self.l1 = QtWidgets.QLabel()
        self.l1.setStyleSheet('background: #008EC0; padding: 1px;')
        self.l1.setText("<font color=#ffffff > Serial COM Port </font>")
        if USE_FULLSCREEN:
            self.l1.setFixedHeight(50)
        # else:
        #    self.l1.setFixedHeight(15)
        self.Layout_controls.addWidget(self.l1, 1, 1, 1, 3)

        # logo---------------------------------------------------------
        self.l3 = QtWidgets.QLabel()
        self.l3.setAlignment(QtCore.Qt.AlignRight)
        self.Layout_controls.addWidget(self.l3, 4, 7, 1, 1)
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/qatch-logo_full.jpg')
        if USE_FULLSCREEN:
            pixmap = QtGui.QPixmap(icon_path)
            pixmap = pixmap.scaled(
                250, 50, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            self.l3.setPixmap(pixmap)
        else:
            self.l3.setPixmap(QtGui.QPixmap(icon_path))

        # qatch link --------------------------------------------------------
        self.l4 = QtWidgets.QLabel()
        self.Layout_controls.addWidget(self.l4, 3, 7, 1, 1)

        def link(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))
        self.l4.linkActivated.connect(link)
        self.l4.setAlignment(QtCore.Qt.AlignRight)
        self.l4.setText(
            '<a href="https://qatchtech.com/"> <font size=4 color=#008EC0 >qatchtech.com</font>')  # &nbsp;

        # info@qatchtech.com Mail -----------------------------------------------
        self.lmail = QtWidgets.QLabel()
        self.Layout_controls.addWidget(self.lmail, 2, 7, 1, 1)  # 25 40

        def linkmail(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))
        self.lmail.linkActivated.connect(linkmail)
        self.lmail.setAlignment(QtCore.Qt.AlignRight)
        # self.lmail.setAlignment(QtCore.Qt.AlignLeft)
        self.lmail.setText(
            '<a href="mailto:info@qatchtech.com"> <font color=#008EC0 >info@qatchtech.com</font>')

        # software user guide --------------------------------------------------------
        self.lg = QtWidgets.QLabel()
        self.Layout_controls.addWidget(self.lg, 1, 7, 1, 1)  # 30

        def link(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))
        self.lg.linkActivated.connect(link)
        self.lg.setAlignment(QtCore.Qt.AlignRight)
        self.lg.setText('<a href="file://{}/docs/userguide.pdf"> <font color=#008EC0 >User Guide</font>'.format(
            Architecture.get_path()))  # &nbsp;
        #####################################
        '''
        self.ico = QtWidgets.QLabel()
        self.Layout_controls.addWidget(self.ico, 2, 5, 1, 1)
        self.title = QtWidgets.QLabel()
        def link(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))
        self.title.linkActivated.connect(link)
        self.title.setText('<a href="https://openqcm.com/openqcm-q-1-software"> <font color=#008EC0 >user guide</font>')
        self.Layout_controls.addWidget(self.title, 2, 5, 1, 1)
        self.pixmap = QtGui.QPixmap("guide.ico")
        self.ico.setPixmap(self.pixmap)
        self.ico.setAlignment(QtCore.Qt.AlignRight)
        self.title.setMinimumHeight(self.pixmap.height())
        self.title.setAlignment(QtCore.Qt.AlignRight)
        '''
        #####################################
        # Save file -----------------------------------------------------------
        self.infosave = QtWidgets.QLabel()
        self.infosave.setStyleSheet('background: #008EC0; padding: 1px;')
        # self.infosave.setAlignment(QtCore.Qt.AlignCenter)
        if USE_FULLSCREEN:
            self.infosave.setFixedHeight(50)
        # else:
        #    self.infosave.setFixedHeight(15)
        self.infosave.setText(
            "<font color=#ffffff > TEC Temperature Control </font>")
        self.Layout_controls.addWidget(self.infosave, 1, 4, 1, 1)

        # Program Status standby ----------------------------------------------
        self.infostatus = QtWidgets.QLabel()
        self.infostatus.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.infostatus.setAlignment(QtCore.Qt.AlignCenter)
        self.infostatus.setText(
            "<font color=#333333 > Program Status Standby </font>")
        if USE_FULLSCREEN:
            self.infostatus.setFixedHeight(50)
        self.Layout_controls.addWidget(self.infostatus, 5, 5, 1, 2)

        # Infobar -------------------------------------------------------------
        self.infobar = QtWidgets.QLineEdit()
        self.infobar.setReadOnly(True)
        self.infobar_label = QtWidgets.QLabel()
        self.infobar_label.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        # self.infobar_label.setAlignment(QtCore.Qt.AlignCenter)
        self.infobar.textChanged.connect(self.infobar_label.setText)
        if SHOW_SIMPLE_CONTROLS:
            self.infobar.textChanged.connect(self._update_progress_text)
        if USE_FULLSCREEN:
            self.infobar_label.setFixedHeight(50)
        # self.infobar.setText("<font color=#0000ff > Infobar </font>") # WAIT until progressBar exists to trigger signals
        self.Layout_controls.addWidget(self.infobar_label, 0, 0, 1, 7)

        # Multiplex -----------------------------------------------------------
        self.lmp = QtWidgets.QLabel()
        self.lmp.setStyleSheet('background: #008EC0; padding: 1px;')
        self.lmp.setText("<font color=#ffffff > Multiplex Mode </font>")
        if USE_FULLSCREEN:
            self.lmp.setFixedHeight(50)
        # else:
        #    self.lmp.setFixedHeight(15)
        self.Layout_controls.addWidget(self.lmp, 3, 0, 1, 1)

        self.cBox_MultiMode = QtWidgets.QComboBox()
        self.cBox_MultiMode.setObjectName("cBox_MultiMode")
        self.cBox_MultiMode.addItems(
            ["1 Channel", "2 Channels", "3 Channels", "4 Channels"])
        self.cBox_MultiMode.setCurrentIndex(0)  # default 1
        if USE_FULLSCREEN:
            self.cBox_MultiMode.setFixedHeight(50)

        icon_path = os.path.join(Architecture.get_path(), 'QATCH/icons/')
        self.pButton_PlateConfig = QtWidgets.QPushButton(
            QtGui.QIcon(os.path.join(icon_path, 'advanced.png')), "")
        self.pButton_PlateConfig.setToolTip("Plate Configuration...")
        self.pButton_PlateConfig.clicked.connect(self.doPlateConfig)
        self.hBox_MultiConfig = QtWidgets.QHBoxLayout()
        self.hBox_MultiConfig.addWidget(self.cBox_MultiMode, 3)
        self.hBox_MultiConfig.addWidget(self.pButton_PlateConfig, 1)
        self.Layout_controls.addLayout(self.hBox_MultiConfig, 4, 0, 1, 1)

        self.chBox_MultiAuto = QtWidgets.QCheckBox()
        self.chBox_MultiAuto.setEnabled(True)
        self.chBox_MultiAuto.setChecked(True)
        self.chBox_MultiAuto.setObjectName("chBox_MultiAuto")
        self.Layout_controls.addWidget(self.chBox_MultiAuto, 5, 0, 1, 1)

        # Progressbar -------------------------------------------------------------
        styleBar = """
                    QProgressBar
                    {
                     border: 0.5px solid #B8B8B8;
                     border-radius: 1px;
                     text-align: center;
                     color: #333333;
                    }
                     QProgressBar::chunk
                    {
                     background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(184, 184, 184, 200), stop:1 rgba(221, 221, 221, 200));
                    }
                 """  # background:url("openQCM/icons/openqcm-logo.png")
        self.run_progress_bar = QtWidgets.QProgressBar()
        self.run_progress_bar.setGeometry(QtCore.QRect(0, 0, 50, 10))
        self.run_progress_bar.setObjectName("progressBar")
        self.run_progress_bar.setStyleSheet(styleBar)

        self.fill_prediction_progress_bar = QtWidgets.QProgressBar()
        self.fill_prediction_progress_bar.setMinimum(0)
        self.fill_prediction_progress_bar.setMaximum(2)
        self.fill_prediction_progress_bar.setGeometry(
            QtCore.QRect(0, 0, 50, 10))
        self.fill_prediction_progress_bar.setObjectName("fillProgressBar")
        self.fill_prediction_progress_bar.setStyleSheet(styleBar)

        if USE_FULLSCREEN:
            self.run_progress_bar.setFixedHeight(50)
            self.fill_prediction_progress_bar.setFixedHeight(50)
        if SHOW_SIMPLE_CONTROLS:
            self.run_progress_bar.valueChanged.connect(
                self._update_progress_value)

        self.run_progress_bar.setValue(0)
        self.fill_prediction_progress_bar.setValue(0)

        self.fill_prediction_progress_bar.setFormat("Run: %v/%m (No Fill)")

        self.Layout_controls.setColumnStretch(0, 0)
        self.Layout_controls.setColumnStretch(1, 1)
        self.Layout_controls.setColumnStretch(2, 0)
        self.Layout_controls.setColumnStretch(3, 0)
        self.Layout_controls.setColumnStretch(4, 2)
        self.Layout_controls.setColumnStretch(5, 2)
        self.Layout_controls.setColumnStretch(6, 2)
        self.Layout_controls.addWidget(self.run_progress_bar, 0, 7, 1, 1)
        self.gridLayout.addLayout(self.Layout_controls, 7, 1, 1, 1)
        # ---------------------------------------------------------------------

        # define simple layout - only add to central widget if requested
        self.toolLayout = QtWidgets.QVBoxLayout()
        self.toolBar = QtWidgets.QHBoxLayout()

        self.tool_bar = QtWidgets.QToolBar()
        self.tool_bar.setIconSize(QtCore.QSize(50, 30))
        self.tool_bar.setStyleSheet("color: #333333;")

        icon_path = os.path.join(Architecture.get_path(), 'QATCH/icons/')

        icon_init = QtGui.QIcon()
        icon_init.addPixmap(QtGui.QPixmap(os.path.join(
            icon_path, "initialize.png")), QtGui.QIcon.Normal)
        # icon_init.addPixmap(QtGui.QPixmap(os.path.join(icon_path, 'initialize-disabled.png')), QtGui.QIcon.Disabled)
        self.tool_Initialize = QtWidgets.QToolButton()
        self.tool_Initialize.setToolButtonStyle(
            QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_Initialize.setIcon(icon_init)  # normal and disabled pixmaps
        self.tool_Initialize.setText("Initialize")
        self.tool_Initialize.clicked.connect(self.action_initialize)
        self.tool_bar.addWidget(self.tool_Initialize)

        self.tool_bar.addSeparator()

        icon_start = QtGui.QIcon()
        icon_start.addPixmap(QtGui.QPixmap(os.path.join(
            icon_path, "start.png")), QtGui.QIcon.Normal)
        # icon_start.addPixmap(QtGui.QPixmap(os.path.join(icon_path, 'start-disabled.png')), QtGui.QIcon.Disabled)
        self.tool_Start = QtWidgets.QToolButton()
        self.tool_Start.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_Start.setIcon(icon_start)
        self.tool_Start.setText("Start")
        self.tool_Start.clicked.connect(self.action_start)
        self.tool_bar.addWidget(self.tool_Start)

        icon_stop = QtGui.QIcon()
        icon_stop.addPixmap(QtGui.QPixmap(os.path.join(
            icon_path, "stop.png")), QtGui.QIcon.Normal)
        # icon_stop.addPixmap(QtGui.QPixmap(os.path.join(icon_path, 'stop-disabled.png')), QtGui.QIcon.Disabled)
        self.tool_Stop = QtWidgets.QToolButton()
        self.tool_Stop.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_Stop.setIcon(icon_stop)
        self.tool_Stop.setText("Stop")
        self.tool_Stop.clicked.connect(self.action_stop)
        self.tool_bar.addWidget(self.tool_Stop)

        self.tool_bar.addSeparator()

        icon_reset = QtGui.QIcon()
        icon_reset.addPixmap(QtGui.QPixmap(os.path.join(
            icon_path, "reset.png")), QtGui.QIcon.Normal)
        # icon_reset.addPixmap(QtGui.QPixmap(os.path.join(icon_path, 'reset-disabled.png')), QtGui.QIcon.Disabled) # not provided
        self.tool_Reset = QtWidgets.QToolButton()
        self.tool_Reset.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_Reset.setIcon(icon_reset)
        self.tool_Reset.setText("Reset")
        self.tool_Reset.clicked.connect(self.action_reset)
        self.tool_bar.addWidget(self.tool_Reset)

        self.tool_bar.addSeparator()

        self._warningTimer = QtCore.QTimer()
        self._warningTimer.setSingleShot(True)
        self._warningTimer.timeout.connect(self.action_tempcontrol_warning)
        self._warningTimer.setInterval(2000)  # 2 second delay

        icon_temp = QtGui.QIcon()
        icon_temp.addPixmap(QtGui.QPixmap(os.path.join(
            icon_path, "temp.png")), QtGui.QIcon.Normal)
        # icon_temp.addPixmap(QtGui.QPixmap(os.path.join(icon_path, 'temp-disabled.png')), QtGui.QIcon.Disabled)
        self.tool_TempControl = QtWidgets.QToolButton()
        self.tool_TempControl.setToolButtonStyle(
            QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_TempControl.setIcon(icon_temp)
        self.tool_TempControl.setText("Temp Control")
        self.tool_TempControl.setCheckable(True)
        self.tool_TempControl.clicked.connect(self.action_tempcontrol)
        # warn to "stop" before changing when disabled:
        self.tool_TempControl.enterEvent = self.action_tempcontrol_warn_start
        self.tool_TempControl.leaveEvent = self.action_tempcontrol_warn_stop
        # self.tool_TempControl.enterEvent = self.action_tempcontrol_warn_now
        self.tool_bar.addWidget(self.tool_TempControl)

        self.toolBar.addWidget(self.tool_bar)

        self.tempController = QtWidgets.QWidget()
        # warn to "stop" before changing when disabled:
        self.tempController.enterEvent = self.action_tempcontrol_warn_start
        self.tempController.leaveEvent = self.action_tempcontrol_warn_stop
        # self.tempController.enterEvent = self.action_tempcontrol_warn_now
        self.tempController.setMinimumSize(QtCore.QSize(200, 40))
        self.tempController.setFixedWidth(200)
        self.tempLayout = QtWidgets.QVBoxLayout()
        self.tempLayout.setContentsMargins(0, 5, 0, 5)
        self.tempLayout.addWidget(self.lTemp)
        self.tempLayout.addWidget(self.slTemp)
        self.tempController.setLayout(self.tempLayout)
        self.toolBar.addWidget(self.tempController)
        self.tempController.setEnabled(False)

        self.toolBar.addStretch()

        self.tool_bar_2 = QtWidgets.QToolBar()
        self.tool_bar_2.setIconSize(QtCore.QSize(50, 30))
        self.tool_bar_2.setStyleSheet("color: #333333;")

        # self.tool_Advanced = QtWidgets.QLabel("Advanced Settings")
        # self.tool_Advanced.setStyleSheet("color: #0D4AAF; text-decoration: none;")
        # self.tool_Advanced.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        # self.tool_Advanced.mousePressEvent = self.action_advanced
        # self.toolBar.addWidget(self.tool_Advanced)

        icon_advanced = QtGui.QIcon()
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/advanced.png')
        icon_advanced.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal)
        # icon_advanced.addPixmap(QtGui.QPixmap('QATCH/icons/advanced-disabled.png'), QtGui.QIcon.Disabled)
        self.tool_Advanced = QtWidgets.QToolButton()
        self.tool_Advanced.setToolButtonStyle(
            QtCore.Qt.ToolButtonTextUnderIcon)
        # normal and disabled pixmaps
        self.tool_Advanced.setIcon(icon_advanced)
        self.tool_Advanced.setText("Advanced")
        self.tool_Advanced.clicked.connect(self.action_advanced)
        self.tool_bar_2.addWidget(self.tool_Advanced)

        self.tool_bar_2.addSeparator()

        icon_user = QtGui.QIcon()
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/user.png')
        icon_user.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal)
        icon_user.addPixmap(QtGui.QPixmap(
            'QATCH/icons/user.png'), QtGui.QIcon.Disabled)
        self.tool_User = QtWidgets.QToolButton()
        self.tool_User.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_User.setIcon(icon_user)  # normal and disabled pixmaps
        self.tool_User.setText("Anonymous")
        self.tool_User.setEnabled(False)
        # self.tool_User.clicked.connect(self.action_user)
        self.tool_bar_2.addWidget(self.tool_User)

        self.toolBar.addWidget(self.tool_bar_2)

        self.toolBar.setContentsMargins(10, 10, 5, 5)
        self.toolBarWidget = QtWidgets.QWidget()
        self.toolBarWidget.setLayout(self.toolBar)
        self.toolBarWidget.setStyleSheet("background: #DDDDDD;")

        self.toolLayout.addWidget(self.toolBarWidget)
        self.toolLayout.addWidget(self.run_progress_bar)
        self.toolLayout.addWidget(self.fill_prediction_progress_bar)

        if SHOW_SIMPLE_CONTROLS:
            # Remove bottom margin, leaving the rest as "default"
            self.toolLayout.setContentsMargins(11, 11, 11, 0)
            self.centralwidget.setLayout(self.toolLayout)

            self.Layout_controls.removeWidget(self.infosave)  # tec controller
            self.Layout_controls.removeWidget(self.lTemp)  # label
            self.Layout_controls.removeWidget(self.slTemp)  # slider
            self.Layout_controls.removeWidget(self.pTemp)  # start/stop button
            self.Layout_controls.removeWidget(self.run_progress_bar)
            self.Layout_controls.removeWidget(self.lg)  # user guide
            self.Layout_controls.removeWidget(self.lmail)  # email
            self.Layout_controls.removeWidget(self.l4)  # website
            self.Layout_controls.removeWidget(self.l3)  # logo

            self.advancedwidget = QtWidgets.QWidget()
            self.advancedwidget.setWindowFlags(
                QtCore.Qt.Dialog | QtCore.Qt.WindowStaysOnTopHint)
            self.advancedwidget.setWhatsThis(
                "These settings are for Advanced Users ONLY!")
            warningWidget = QtWidgets.QLabel(
                f"WARNING: {self.advancedwidget.whatsThis()}")
            warningWidget.setStyleSheet(
                'background: #FF6600; padding: 1px; font-weight: bold;')
            warningLayout = QtWidgets.QVBoxLayout()
            warningLayout.addWidget(warningWidget)
            warningLayout.addLayout(self.gridLayout)
            self.advancedwidget.setLayout(warningLayout)
        else:
            self.centralwidget.setLayout(self.gridLayout)

        MainWindow1.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow1)

    def _update_progress_text(self):
        # get innerText from HTML in infobar
        plain_text = self.infobar.text()
        color = plain_text[plain_text.rindex(
            'color=')+6:plain_text.rindex('color=')+6+7]
        plain_text = plain_text[plain_text.index('>')+1:]
        plain_text = plain_text[plain_text.index('>')+1:]
        plain_text = plain_text[plain_text.index('>')+1:]
        plain_text = plain_text[0:plain_text.rindex('<')]
        # remove any formatting tags: <b>, <i>, <u>
        while plain_text.rfind('<') != plain_text.find('<'):
            plain_text = plain_text[0:plain_text.rindex('<')]
            plain_text = plain_text[plain_text.index('>')+1:]
        if len(plain_text) == 0:
            plain_text = "Progress: Not Started"
        else:
            plain_text = "Status: {}".format(plain_text)
        self.run_progress_bar.setFormat(plain_text)
        styleBar = """
                    QProgressBar
                    {
                     border: 0.5px solid #B8B8B8;
                     border-radius: 1px;
                     text-align: center;
                     color: {COLOR};
                     font-weight: bold;
                    }
                     QProgressBar::chunk
                    {
                     background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(184, 184, 184, 200), stop:1 rgba(221, 221, 221, 200));
                    }
                 """.replace("{COLOR}", color)
        self.run_progress_bar.setStyleSheet(styleBar)

    def _update_progress_value(self):
        if self.cBox_Source.currentIndex() == OperationType.measurement.value:
            pass  # self._update_progress_text() # defer to infobar text, not percentage
        else:
            self.run_progress_bar.setFormat("Progress: %p%")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate(
            "MainWindow", "{} {} - Setup/Control".format(Constants.app_title, Constants.app_version)))
        icon_path = os.path.join(Architecture.get_path(), 'QATCH/icons/')
        MainWindow.setWindowIcon(QtGui.QIcon(
            os.path.join(icon_path, 'qatch-icon.png')))  # .png
        self.advancedwidget.setWindowIcon(QtGui.QIcon(
            os.path.join(icon_path, 'advanced.png')))  # .png
        self.advancedwidget.setWindowTitle(
            _translate("MainWindow2", "Advanced Settings"))
        self.pButton_Stop.setText(_translate("MainWindow", " STOP"))
        self.pButton_Start.setText(_translate("MainWindow", "START"))
        self.pButton_Clear.setText(_translate("MainWindow", "Clear Plots"))
        self.pButton_Reference.setText(
            _translate("MainWindow", "Set/Reset Reference"))
        self.pButton_ResetApp.setText(
            _translate("MainWindow", "Factory Defaults"))
        self.sBox_Samples.setSuffix(_translate("MainWindow", " / 5 min"))
        self.sBox_Samples.setPrefix(_translate("MainWindow", ""))
        self.chBox_export.setText(_translate(
            "MainWindow", "Txt Export Sweep File"))
        self.chBox_freqHop.setText(_translate("MainWindow", "Mode Hop"))
        self.chBox_correctNoise.setText(
            _translate("MainWindow", "Show amplitude curve"))
        self.chBox_MultiAuto.setText(_translate(
            "MainWindow", "Auto-detect channel count"))

    def action_initialize(self):
        if self.pButton_Start.isEnabled():
            self.cBox_Source.setCurrentIndex(OperationType.calibration.value)
            self.fill_prediction_progress_bar.setValue(0)
            self.fill_prediction_progress_bar.setFormat(
                Constants.FILL_TYPE_LABEL_MAP.get(0, ""))
            self.pButton_Start.clicked.emit()
            self.cal_initialized = True

    def action_start(self):
        if self.pButton_Start.isEnabled():
            self.cBox_Source.setCurrentIndex(OperationType.measurement.value)
            self.fill_prediction_progress_bar.setValue(0)
            self.fill_prediction_progress_bar.setFormat(
                Constants.FILL_TYPE_LABEL_MAP.get(0, ""))
            self.pButton_Start.clicked.emit()

    def action_stop(self):
        if self.pButton_Stop.isEnabled():
            self.cal_initialized = False
            self.pButton_Stop.clicked.emit()

    def action_reset(self):
        if self.tool_TempControl.isChecked():
            self.tool_TempControl.setChecked(False)
            self.tool_TempControl.clicked.emit()  # if running, stop
        self.slTemp.setValue(25)
        if self.pButton_Start.isEnabled():
            self.pButton_Clear.clicked.emit()
            self.pButton_Refresh.clicked.emit()
        self.infostatus.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.infostatus.setText(
            "<font color=#333333 > Program Status Standby </font>")
        self.cal_initialized = False
        self.tool_Start.setEnabled(False)
        self.fill_prediction_progress_bar.setValue(0)
        self.fill_prediction_progress_bar.setFormat(
            Constants.FILL_TYPE_LABEL_MAP.get(0, ""))
        # at least one device connected
        self.tool_TempControl.setEnabled(self.cBox_Port.count() > 1)

    def action_tempcontrol(self):
        self.tempController.setEnabled(self.tool_TempControl.isChecked())
        if self.tool_TempControl.isChecked():
            if self.pTemp.text().find("Stop") < 0:  # not found (i.e. "start" or "resume")
                self.pTemp.clicked.emit()
                self.slTemp.setFocus()
        else:
            if self.pTemp.text().find("Stop") >= 0:  # found (i.e. currently running, not locked)
                self.pTemp.clicked.emit()

    def action_tempcontrol_warn_start(self, event):
        self.event_windowPos = event.windowPos()
        self._warningTimer.start()

    def action_tempcontrol_warn_stop(self, event):
        self._warningTimer.stop()

    def action_tempcontrol_warn_now(self, event):
        self.event_windowPos = event.windowPos()
        self.action_tempcontrol_warning()

    def action_tempcontrol_warning(self):
        if self.tool_TempControl.isChecked() and not self.tool_TempControl.isEnabled():
            # Temp Control is checked (running) and not enabled (during measurement run)
            # Log.e("window pos:", self.event_windowPos)
            # Log.e("widget pos:", self.tempController.mapToGlobal(QtCore.QPoint(0, 0)))
            Log.w("WARNING: Temp Control mode cannot be changed during an active run.")
            if self.event_windowPos.x() >= self.tempController.mapToGlobal(QtCore.QPoint(0, 0)).x():
                Log.w(
                    "To adjust Temp Control: Press \"Stop\" first, then adjust setpoint accordingly.")
            else:
                Log.w(
                    "To stop Temp Control: Press \"Stop\" first, then click \"Temp Control\" button.")

        # TODO: Not implemented; only show once per measurement run (maybe not the best idea)
        # else:
        #     if hasattr(self, "cached_warning_adjust"):
        #         # pass

    def action_advanced(self, obj):
        if self.advancedwidget.isVisible():
            self.advancedwidget.hide()
        self.advancedwidget.move(0, 0)
        self.advancedwidget.show()
        # make plate config button square
        self.pButton_PlateConfig.setFixedWidth(
            self.pButton_PlateConfig.height())
        # QtWidgets.QWhatsThis.enterWhatsThisMode()
        # QtWidgets.QWhatsThis.showText(
        #     QtCore.QPoint(int(self.advancedwidget.width() / 2), int(self.advancedwidget.height() * (2/3))),
        #     self.advancedwidget.whatsThis(),
        #     self.advancedwidget)

    def doPlateConfig(self):
        if hasattr(self, "wellPlateUI"):
            if self.wellPlateUI.isVisible():
                # close if already open, don't bother to ask to save unsaved changes (TODO)
                self.wellPlateUI.close()

        # Dynamically specify plate dimensions and number of devices connected to constructor
        # num port currently detected / connected
        num_ports = self.cBox_Port.count() - 1
        i = self.cBox_Port.currentText()
        i = 0 if i.find(":") == -1 else int(i.split(":")[0], base=16)
        if i % 9 == i:  # 4x1 system
            well_width = 4  # number of well on a single device sensor for a multiplex device
            well_height = 1  # num of multiplex devices, ceil
        else:           # 4x6 system
            well_width = 6
            well_height = 4
        num_channels = self.cBox_MultiMode.currentIndex() + 1  # user define device count
        if num_ports not in [well_width, well_height] and num_ports != 1:
            PopUp.warning(self, "Plate Configuration",
                          f"<b>Multiplex device(s) are required for plate configuration.</b><br/>" +
                          f"You must have exactly 4 device ports connected for this mode.<br/>" +
                          f"Currently connected device port count is: {num_ports} (not 4)")
        else:
            # creation of widget also shows UI to user
            self.wellPlateUI = WellPlate(well_width, well_height, num_channels)


#######################################################################################################################

class Ui_Plots(object):
    def setupUi(self, MainWindow2):
        USE_FULLSCREEN = (QDesktopWidget().availableGeometry().width() == 2880)

        MainWindow2.setObjectName("MainWindow2")
        # MainWindow2.setGeometry(100, 100, 890, 750)
        # MainWindow2.setFixedSize(1091, 770)
        # MainWindow2.resize(1091, 770)
        MainWindow2.setMinimumSize(QtCore.QSize(1000, 250))
        if USE_FULLSCREEN:
            MainWindow2.resize(1701, 1435)
            MainWindow2.move(0, 0)
        else:
            MainWindow2.move(692, 0)
        MainWindow2.setStyleSheet("")
        MainWindow2.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow2)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        # Remove top margin, leaving the rest as "default"
        self.gridLayout.setContentsMargins(11, 0, 11, 11)
        self.Layout_graphs = QtWidgets.QSplitter(
            QtCore.Qt.Horizontal)  # QGridLayout()
        self.Layout_graphs.setObjectName("Layout_graphs")

        self.plt = GraphicsLayoutWidget(self.centralwidget)
        self.pltB = GraphicsLayoutWidget(self.centralwidget)

        self.plt.setAutoFillBackground(False)
        self.plt.setStyleSheet("border: 0px;")
        self.plt.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.plt.setFrameShadow(QtWidgets.QFrame.Plain)
        self.plt.setLineWidth(0)
        self.plt.setObjectName("plt")
        self.plt.setMinimumWidth(333)

        self.pltB.setAutoFillBackground(False)
        self.pltB.setStyleSheet("border: 0px;")
        self.pltB.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.pltB.setFrameShadow(QtWidgets.QFrame.Plain)
        self.pltB.setLineWidth(0)
        self.pltB.setObjectName("pltB")
        self.pltB.setMinimumWidth(666)

        """
        self.label = QtWidgets.QLabel()
        self.Layout_graphs.addWidget(self.label, 0, 0, 1, 1)
        def link1(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))
        self.label.linkActivated.connect(link1)
        self.label.setText('<a href="https://openqcm.com/"> <font color=#333333 >Open-source Python application for displaying, processing and storing real-time data from openQCM Q-1 Device</font> </a>')
        """

        self.Layout_graphs.addWidget(self.pltB)
        self.Layout_graphs.addWidget(self.plt)
        width = self.Layout_graphs.width()
        self.Layout_graphs.setSizes([int(width*2/3), int(width*1/3)])

        # add collapse/expand icon arrows
        self.Layout_graphs.setHandleWidth(10)
        handle = self.Layout_graphs.handle(1)
        layout_s = QtWidgets.QVBoxLayout()
        layout_s.setContentsMargins(0, 0, 0, 0)
        layout_s.addStretch()
        self.btnCollapse = QtWidgets.QToolButton(handle)
        self.btnCollapse.setArrowType(QtCore.Qt.RightArrow)
        self.btnCollapse.clicked.connect(
            lambda: self.handleSplitterButton(True))
        layout_s.addWidget(self.btnCollapse)
        self.btnExpand = QtWidgets.QToolButton(handle)
        self.btnExpand.setArrowType(QtCore.Qt.LeftArrow)
        self.btnExpand.clicked.connect(
            lambda: self.handleSplitterButton(False))
        layout_s.addWidget(self.btnExpand)
        layout_s.addStretch()
        handle.setLayout(layout_s)
        self.btnExpand.setVisible(False)
        # self.handleSplitterButton(False)
        self.Layout_graphs.splitterMoved.connect(self.handleSplitterMoved)

        self.gridLayout.addWidget(self.Layout_graphs, 2, 1, 1, 1)
        MainWindow2.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow2)

    def handleSplitterMoved(self, pos, index):
        collapsed = self.Layout_graphs.sizes()[-1] == 0
        self.btnCollapse.setVisible(not collapsed)
        self.btnExpand.setVisible(collapsed)

    def handleSplitterButton(self, collapse=True):
        if collapse:
            self.btnCollapse.setVisible(False)
            self.btnExpand.setVisible(True)
            self.Layout_graphs.setSizes([1, 0])
        else:
            self.btnCollapse.setVisible(True)
            self.btnExpand.setVisible(False)
            width = self.Layout_graphs.width()
            self.Layout_graphs.setSizes([int(width*2/3), int(width*1/3)])

    def retranslateUi(self, MainWindow2):
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/qatch-icon.png')
        MainWindow2.setWindowIcon(QtGui.QIcon(icon_path))  # .png
        MainWindow2.setWindowTitle(_translate(
            "MainWindow2", "{} {} - Plots".format(Constants.app_title, Constants.app_version)))


############################################################################################################

class Ui_Info(object):
    def setupUi(self, MainWindow3):
        # MainWindow3.setObjectName("MainWindow3")
        # MainWindow3.setGeometry(500, 50, 100, 500)
        # MainWindow3.setFixedSize(100, 500)
        # MainWindow3.resize(100, 500)
        # MainWindow3.setMinimumSize(QtCore.QSize(0, 0))
        MainWindow3.setStyleSheet("")
        MainWindow3.setTabShape(QtWidgets.QTabWidget.Rounded)
        MainWindow3.setMinimumSize(QtCore.QSize(268, 518))
        MainWindow3.move(820, 0)
        self.centralwidget = QtWidgets.QWidget(MainWindow3)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")

        # Setup Information -------------------------------------------------------------------
        self.info1 = QtWidgets.QLabel()
        self.info1.setStyleSheet('background: #008EC0; padding: 1px;')
        self.info1.setText("<font color=#ffffff > Setup Information&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</font>")
        # self.info1.setFixedWidth(250)
        # self.info1.setFixedHeight(15)
        self.gridLayout_2.addWidget(self.info1, 0, 0, 1, 1)

        # Device Setup -------------------------------------------------------------------------
        self.info1a = QtWidgets.QLabel()
        self.info1a.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.info1a.setText("<font color=#0000ff > Device Setup</font>")
        # self.info1a.setFixedWidth(250)
        # self.info1a.setFixedHeight(22)
        self.gridLayout_2.addWidget(self.info1a, 1, 0, 1, 1)

        # Operation Mode -----------------------------------------------------------------------
        self.info11 = QtWidgets.QLabel()
        self.info11.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.info11.setText("<font color=#0000ff > Operation Mode </font>")
        # self.info11.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.info11, 2, 0, 1, 1)

        # Data Information ---------------------------------------------------------------------
        self.info = QtWidgets.QLabel()
        self.info.setStyleSheet('background: #008EC0; padding: 1px;')
        self.info.setText(
            "<font color=#ffffff > Data Information&nbsp;</font>")
        # self.info.setFixedWidth(250)
        # self.info.setFixedHeight(15)
        self.gridLayout_2.addWidget(self.info, 3, 0, 1, 1)

        # Selected Frequency -------------------------------------------------------------------
        self.info2 = QtWidgets.QLabel()
        self.info2.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.info2.setText("<font color=#0000ff > Selected Frequency </font>")
        # self.info2.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.info2, 4, 0, 1, 1)

        # Frequency Value ----------------------------------------------------------------------
        self.info6 = QtWidgets.QLabel()
        self.info6.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.info6.setText("<font color=#0000ff > Frequency Value </font>")
        # self.info6.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.info6, 5, 0, 1, 1)

        # Start Frequency ----------------------------------------------------------------------
        self.info3 = QtWidgets.QLabel()
        self.info3.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.info3.setText("<font color=#0000ff > Start Frequency </font>")
        # self.info3.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.info3, 6, 0, 1, 1)

        # Stop Frequency -----------------------------------------------------------------------
        self.info4 = QtWidgets.QLabel()
        self.info4.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.info4.setText("<font color=#0000ff > Stop Frequency </font>")
        # self.info4.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.info4, 7, 0, 1, 1)

        # Frequency Range----------------------------------------------------------------------
        self.info4a = QtWidgets.QLabel()
        self.info4a.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.info4a.setText("<font color=#0000ff > Frequency Range </font>")
        # self.info4a.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.info4a, 8, 0, 1, 1)

        # Sample Rate----------------------------------------------------------------------
        self.info5 = QtWidgets.QLabel()
        self.info5.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.info5.setText("<font color=#0000ff > Sample Rate </font>")
        # self.info5.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.info5, 9, 0, 1, 1)

        # Sample Number----------------------------------------------------------------------
        self.info7 = QtWidgets.QLabel()
        self.info7.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.info7.setText("<font color=#0000ff > Sample Number </font>")
        # self.info7.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.info7, 10, 0, 1, 1)

        # Reference Settings -------------------------------------------------------------------
        self.inforef = QtWidgets.QLabel()
        self.inforef.setStyleSheet('background: #008EC0; padding: 1px;')
        # self.inforef1.setAlignment(QtCore.Qt.AlignCenter)
        self.inforef.setText(
            "<font color=#ffffff > Reference Settings </font>")
        # self.inforef.setFixedHeight(15)
        # self.inforef.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.inforef, 11, 0, 1, 1)

        # Ref. Frequency -----------------------------------------------------------------------
        self.inforef1 = QtWidgets.QLabel()
        self.inforef1.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        # self.inforef1.setAlignment(QtCore.Qt.AlignCenter)
        self.inforef1.setText("<font color=#0000ff > Ref. Frequency </font>")
        # self.inforef1.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.inforef1, 12, 0, 1, 1)
        # Ref. Dissipation -----------------------------------------------------------------------

        self.inforef2 = QtWidgets.QLabel()
        self.inforef2.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        # self.inforef2.setAlignment(QtCore.Qt.AlignCenter)
        self.inforef2.setText("<font color=#0000ff > Ref. Dissipation </font>")
        # self.inforef2.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.inforef2, 13, 0, 1, 1)

        # Current Data ---------------------------------------------------------------------------
        self.l8 = QtWidgets.QLabel()
        self.l8.setStyleSheet('background: #008EC0; padding: 1px;')
        self.l8.setText("<font color=#ffffff > Current Data </font>")
        # self.l8.setFixedHeight(15)
        # self.l8.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.l8, 14, 0, 1, 1)

        # Resonance Frequency -------------------------------------------------------------------
        self.l7 = QtWidgets.QLabel()
        self.l7.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.l7.setText("<font color=#0000ff >  Resonance Frequency </font>")
        # self.l7.setAlignment(QtCore.Qt.AlignCenter)
        # self.l7.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.l7, 15, 0, 1, 1)

        # Dissipation ---------------------------------------------------------------------------
        self.l6 = QtWidgets.QLabel()
        self.l6.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.l6.setText("<font color=#0000ff > Dissipation  </font>")
        # self.l6.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.l6, 16, 0, 1, 1)

        # Temperature ---------------------------------------------------------------------------
        self.l6a = QtWidgets.QLabel()
        self.l6a.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.l6a.setText("<font color=#0000ff >  Temperature </font>")
        # self.l6a.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.l6a, 17, 0, 1, 1)

        # Info from QATCH website -------------------------------------------------------------
        self.lweb = QtWidgets.QLabel()
        self.lweb.setStyleSheet('background: #008EC0; padding: 1px;')
        self.lweb.setText("<font color=#ffffff > Check for Updates </font>")
        # self.lweb.setFixedHeight(15)
        # self.lweb.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.lweb, 18, 0, 1, 1)

        # Check internet connection -------------------------------------------------------------
        '''self.lweb2 = QtWidgets.QLabel()
        self.lweb2.setStyleSheet('background: white; padding: 1px; border: 1px solid #cccccc')
        self.lweb2.setText("<font color=#0000ff > Checking your internet connection </font>")
        # self.lweb2.setFixedHeight(20)
        #self.lweb2.setFixedWidth(250)
        self.gridLayout_2.addWidget(self.lweb2, 19, 0, 1, 1)'''

        # Software update status ----------------------------------------------------------------
        self.lweb3 = QtWidgets.QLabel()
        self.lweb3.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.lweb3.setText("<font color=#0000ff > Update Status </font>")
        # self.lweb3.setFixedHeight(16)
        # self.lweb3.setFixedWidth(300)
        self.gridLayout_2.addWidget(self.lweb3, 20, 0, 1, 1)

        # Download button -----------------------------------------------------------------------
        self.pButton_Download = QtWidgets.QPushButton(self.centralwidget)
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/refresh-icon.png')
        self.pButton_Download.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Download.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Download.setObjectName("pButton_Download")
        self.pButton_Download.setFixedWidth(145)
        self.gridLayout_2.addWidget(
            self.pButton_Download, 21, 0, 1, 1, QtCore.Qt.AlignRight)
        ##########################################################################################

        self.gridLayout.addLayout(self.gridLayout_2, 3, 1, 1, 1)
        MainWindow3.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow3)
        QtCore.QMetaObject.connectSlotsByName(MainWindow3)

    def retranslateUi(self, MainWindow3):
        _translate = QtCore.QCoreApplication.translate
        self.pButton_Download.setText(
            _translate("MainWindow3", " Check Again"))
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/qatch-icon.png')
        MainWindow3.setWindowIcon(QtGui.QIcon(icon_path))  # .png
        MainWindow3.setWindowTitle(_translate("MainWindow3", "Information"))


class Ui_Logger(object):
    def setupUi(self, MainWindow4):
        MainWindow4.setMinimumSize(QtCore.QSize(1000, 100))
        MainWindow4.move(0, 0)
        self.centralwidget = QtWidgets.QWidget(MainWindow4)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)

        logTextBox = QTextEditLogger(self.centralwidget)

        # log to text box
        logTextBox.setFormatter(
            logging.Formatter(
                fmt='%(asctime)s %(levelname)s %(message)s',
                datefmt=None))
        logging.getLogger("QATCH").addHandler(logTextBox)
        Log._show_user_info()

        MainWindow4.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow4)
        QtCore.QMetaObject.connectSlotsByName(MainWindow4)

    def retranslateUi(self, MainWindow4):
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/qatch-icon.png')
        MainWindow4.setWindowIcon(QtGui.QIcon(icon_path))  # .png
        MainWindow4.setWindowTitle(_translate(
            "MainWindow4", "{} {} - Console".format(Constants.app_title, Constants.app_version)))


class QTextEditLogger(logging.Handler, QtCore.QObject):
    appendInfoText = QtCore.pyqtSignal(str)
    appendDebugText = QtCore.pyqtSignal(str)
    forceRepaintEvents = False
    progressMode = False

    def __init__(self, parent):
        super().__init__()
        QtCore.QObject.__init__(self)

        # Initialize tab screen
        self.tabs = QtWidgets.QTabWidget()
        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()

        # Add tabs
        self.logInfo = QtWidgets.QTextEdit(parent)
        self.logInfo.setReadOnly(True)
        layout_v1 = QtWidgets.QVBoxLayout()
        layout_v1.addWidget(self.logInfo)
        self.tab1.setLayout(layout_v1)
        self.tabs.addTab(self.tab1, "Info")

        self.logDebug = QtWidgets.QTextEdit(parent)
        self.logDebug.setReadOnly(True)
        layout_v2 = QtWidgets.QVBoxLayout()
        layout_v2.addWidget(self.logDebug)
        self.tab2.setLayout(layout_v2)
        self.tabs.addTab(self.tab2, "Debug")

        self.tabs.setTabPosition(QtWidgets.QTabWidget.East)

        layout_v = QtWidgets.QVBoxLayout()
        layout_v.addWidget(self.tabs)
        parent.setLayout(layout_v)

        self.appendInfoText.connect(self.appendToInfo)  # logInfo.insertHtml)
        # logDebug.insertHtml)
        self.appendDebugText.connect(self.appendToDebug)
        self.last_record_msg = None

    def appendToInfo(self, html):
        if self.forceRepaintEvents and "[Device] ERROR:" in html:
            return  # do not show serial errors during firmware update on info console
        self.logInfo.moveCursor(QtGui.QTextCursor.End,
                                QtGui.QTextCursor.MoveAnchor)
        if self.progressMode:
            # replace the most recent line with this new html line
            self.logInfo.textCursor().deletePreviousChar()
            self.logInfo.moveCursor(
                QtGui.QTextCursor.StartOfLine, QtGui.QTextCursor.MoveAnchor)
            self.logInfo.moveCursor(
                QtGui.QTextCursor.End, QtGui.QTextCursor.KeepAnchor)
            self.logInfo.textCursor().removeSelectedText()
        self.logInfo.insertHtml(html)
        self.logInfo.ensureCursorVisible()
        if self.forceRepaintEvents:
            self.logInfo.repaint()

    def appendToDebug(self, html):
        self.logDebug.moveCursor(
            QtGui.QTextCursor.End, QtGui.QTextCursor.MoveAnchor)
        self.logDebug.insertHtml(html)
        self.logDebug.ensureCursorVisible()
        if self.forceRepaintEvents:
            self.logDebug.repaint()
        if "GUI: Clear console window" in html:
            self.logInfo.clear()
        if "GUI: Force repaint events" in html:
            self.forceRepaintEvents = True
        if "GUI: Normal repaint events" in html:
            self.forceRepaintEvents = False
        if "GUI: Toggle progress mode" in html:
            self.progressMode = not self.progressMode

    def emit(self, record):
        msg = self.format(record)
        if msg == self.last_record_msg:
            # This must be print(), not Log.d(), or else an endless loop could occur!
            print(msg, "(duplicate record ignored)")
            return  # ignore duplicate records when they are handled back-to-back
        self.last_record_msg = msg
        msg = msg[msg.index(' ')+1:]  # trim date from console
        html_fmt = "<font style='font-family:\"Lucida Console\",\"Courier New\",monospace;color:{};font-weight:{};'>{}</font><br/><br/>"
        color = "black" if record.levelno <= logging.INFO else "red"
        weight = "normal" if record.levelno <= logging.WARNING else "bold"
        time_only = msg[0:msg.index(',')]
        padding = "&nbsp;&nbsp;&nbsp;" if weight == "normal" else "&nbsp;&nbsp;"
        msg_info = time_only + padding + record.msg
        msg_debug = msg
        html_info = html_fmt.format(color, weight, msg_info)
        html_debug = html_fmt.format(color, weight, msg_debug)
        if record.levelno >= logging.INFO:
            self.appendInfoText.emit(html_info)
        self.appendDebugText.emit(html_debug)
