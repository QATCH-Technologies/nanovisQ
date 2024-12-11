from QATCH.ui.mainWindow_ui import Ui_Controls, Ui_Info, Ui_Plots, Ui_Logger, Ui_Main, Ui_Login
from pyqtgraph import AxisItem
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from QATCH.core.worker import Worker
from QATCH.core.constants import Constants, OperationType, UpdateEngines
from QATCH.ui.popUp import PopUp, QueryComboBox
from QATCH.ui.runInfo import QueryRunInfo, RunInfoWindow
from QATCH.ui.export import Ui_Export
from QATCH.ui.configure_data import UIConfigureData
from QATCH.common.logger import Logger as Log
from QATCH.common.fileStorage import FileStorage
from QATCH.common.fileManager import FileManager
from QATCH.common.findDevices import Discovery
from QATCH.common.fwUpdater import FW_Updater
from QATCH.common.architecture import Architecture, OSType
from QATCH.common.tutorials import TutorialPages
from QATCH.common.userProfiles import UserProfiles, UserRoles, UserProfilesManager
from QATCH.processors.Analyze import AnalyzeProcess
from time import time, mktime, strftime, strptime, localtime
from dateutil import parser
import threading
import multiprocessing
import datetime
from serial import serialutil
from QATCH.processors.Device import serial  # real device hardware
from xml.dom import minidom
import numpy as np
import sys
import os
import io
import pyzipper
import hashlib
import requests
import stat
import subprocess

TAG = ""  # "[MainWindow]"
ADMIN_OPTION_CMDS = 1

##########################################################################################
# Package that handles the UIs elements and connects to worker service to execute processes
##########################################################################################


class _MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent):
        super().__init__()
        parent.ControlsWin._createMenu(self)
        self.ui0 = Ui_Main()
        self.ui0.setupUi(self, parent)

    def closeEvent(self, event):
        # Log.d(" Exit Real-Time Plot GUI")
        res = PopUp.question(self, Constants.app_title,
                             "Are you sure you want to quit QATCH Q-1 application now?", True)
        if res:
            # self.close()
            QtWidgets.QApplication.quit()
        else:
            event.ignore()


# ------------------------------------------------------------------------------
class LoginWindow(QtWidgets.QMainWindow):

    def __init__(self, parent):
        super().__init__()
        self.ui5 = Ui_Login()
        self.ui5.setupUi(self, parent)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress:
            # Log.i(f"Key {event.key()} pressed!")
            if event.key() in [QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return]:
                if len(self.ui5.user_password.text()) == 0:
                    self.ui5.user_password.setFocus()
                else:
                    self.ui5.action_sign_in()
            if event.key() == QtCore.Qt.Key_Escape:
                self.ui5.clear_form()
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        # Log.d(" Exit Real-Time Plot GUI")
        res = PopUp.question(self, Constants.app_title,
                             "Are you sure you want to quit QATCH Q-1 application now?", True)
        if res:
            # self.close()
            QtWidgets.QApplication.quit()
        else:
            event.ignore()


# ------------------------------------------------------------------------------
class LoggerWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui4 = Ui_Logger()
        self.ui4.setupUi(self)

    def closeEvent(self, event):
        # Log.d(" Exit Real-Time Plot GUI")
        res = PopUp.question(self, Constants.app_title,
                             "Are you sure you want to quit QATCH Q-1 application now?", True)
        if res:
            # self.close()
            QtWidgets.QApplication.quit()
        else:
            event.ignore()


# ------------------------------------------------------------------------------
class PlotsWindow(QtWidgets.QMainWindow):

    def __init__(self, samples=Constants.argument_default_samples):
        super().__init__()
        self.ui2 = Ui_Plots()
        self.ui2.setupUi(self)
    '''
    def closeEvent(self, event):
        #Log.d(" Exit Real-Time Plot GUI")
        res =PopUp.question(self, Constants.app_title, "Are you sure you want to quit QATCH Q-1 application now?")
        if res:
           #self.close()
           QtWidgets.QApplication.quit()
        else:
           event.ignore()#pass
        #sys.exit(0)
    '''

    def closeEvent(self, event):
        # Log.d(" Exit Real-Time Plot GUI")
        res = PopUp.question(self, Constants.app_title,
                             "Are you sure you want to quit QATCH Q-1 application now?", True)
        if res:
            # self.close()
            QtWidgets.QApplication.quit()
        else:
            event.ignore()


# ------------------------------------------------------------------------------
class InfoWindow(QtWidgets.QMainWindow):

    def __init__(self, samples=Constants.argument_default_samples):
        super().__init__()
        self.ui3 = Ui_Info()
        self.ui3.setupUi(self)

    def closeEvent(self, event):
        # Log.d(" Exit Real-Time Plot GUI")
        res = PopUp.question(self, Constants.app_title,
                             "Are you sure you want to quit QATCH Q-1 application now?", True)
        if res:
            # self.close()
            QtWidgets.QApplication.quit()
        else:
            event.ignore()

# ------------------------------------------------------------------------------


class ControlsWindow(QtWidgets.QMainWindow):

    def __init__(self, parent, samples=Constants.argument_default_samples):
        self.parent = parent
        super().__init__()
        self.ui1 = Ui_Controls()
        self.ui1.setupUi(self)
        self.ui_export = Ui_Export()
        self.ui_configure_data = UIConfigureData()
        self.current_timer = QtCore.QTimer()
        self.current_timer.timeout.connect(self.double_toggle_plots)
        UserProfiles().session_end()

    def _createMenu(self, target):
        self.menubar = []
        self.menubar.append(target.menuBar().addMenu("&Options"))
        self.menubar[0].addAction('&Analyze Data', self.analyze_data)
        self.menubar[0].addAction('&Import Data', self.import_data)
        self.menubar[0].addAction('&Export Data', self.export_data)
        self.menubar[0].addAction('&Configure Data', self.configure_data)
        self.menubar[0].addAction('&Find Devices', self.scan_subnets)
        self.menubar[0].addAction('E&xit', self.close)
        self.menubar.append(target.menuBar().addMenu("&Users"))
        self.username = self.menubar[1].addAction('User: [NONE]')
        self.username.setEnabled(False)
        self.signinout = self.menubar[1].addAction(
            '&Sign In', self.set_user_profile)
        self.manage = self.menubar[1].addAction(
            '&Manage Users...', self.manage_user_profiles)
        self.userrole = UserRoles.NONE
        self.menubar.append(target.menuBar().addMenu("&View"))
        self.chk1 = self.menubar[2].addAction('&Console', self.toggle_console)
        self.chk1.setCheckable(True)
        self.chk1.setChecked(self.parent.AppSettings.value(
            "viewState_Console", "True").lower() == "true")
        self.chk2 = self.menubar[2].addAction(
            '&Amplitude', self.toggle_amplitude)
        self.chk2.setCheckable(True)
        self.chk2.setChecked(self.parent.AppSettings.value(
            "viewState_Amplitude", "True").lower() == "true")
        self.chk3 = self.menubar[2].addAction(
            '&Temperature', self.toggle_temperature)
        self.chk3.setCheckable(True)
        self.chk3.setChecked(self.parent.AppSettings.value(
            "viewState_Temperature", "True").lower() == "true")
        self.chk4 = self.menubar[2].addAction(
            '&Resonance/Dissipation', self.toggle_RandD)
        self.chk4.setCheckable(True)
        self.chk4.setChecked(self.parent.AppSettings.value(
            "viewState_Resonance_Dissipation", "True").lower() == "true")
        self.menubar.append(target.menuBar().addMenu("&Help"))
        self.chk5 = self.menubar[3].addAction(
            'View &Tutorials', self.view_tutorials)
        self.chk5.setCheckable(False)
        self.menubar.append(self.menubar[3].addMenu('View &Documentation'))
        self.menubar[4].addAction('&Release Notes', self.release_notes)
        self.menubar[4].addAction('&FW Change Log', self.fw_change_log)
        self.menubar[4].addAction('&SW Change Log', self.sw_change_log)
        self.menubar[3].addAction('View &License', self.view_license)
        self.menubar[3].addAction('View &User Guide', self.view_user_guide)
        self.menubar[3].addAction('&Check for Updates', self.check_for_updates)
        self.menubar[3].addSeparator()
        sw_version = self.menubar[3].addAction('SW {}_{} ({})'.format(
            Constants.app_version,
            "exe" if getattr(sys, 'frozen', False) else "py",
            Constants.app_date))
        sw_version.setEnabled(False)
        from QATCH.QModel.__init__ import __version__ as QModel_version
        from QATCH.QModel.__init__ import __release__ as QModel_release
        q_version = self.menubar[3].addAction('QModel v{} ({})'.format(
            QModel_version, QModel_release))
        q_version.setEnabled(False)

        # update application UI states to reflect viewStates from AppSettings
        if not self.chk1.isChecked():
            QtCore.QTimer.singleShot(100, self.toggle_console)
        if not self.chk2.isChecked():
            QtCore.QTimer.singleShot(100, self.toggle_amplitude)
        if not self.chk3.isChecked():
            QtCore.QTimer.singleShot(100, self.toggle_temperature)
        if not self.chk4.isChecked():
            QtCore.QTimer.singleShot(100, self.toggle_RandD)

    def analyze_data(self):
        self.parent.MainWin.ui0.setAnalyzeMode(self)

    def import_data(self):
        self.ui_export.showNormal(0)

    def export_data(self):
        self.ui_export.showNormal(1)

    def configure_data(self):
        self.ui_configure_data.show()

    def scan_subnets(self):
        Discovery().scanSubnets()
        self.parent._port_list_refresh()

    def set_user_profile(self):
        action = self.signinout.text().lower().replace('&', '')
        if action == "sign in":
            if UserProfiles().count() == 0:
                self.manage_user_profiles()
                return
            name, init, role = UserProfiles.change()
            if name != None:
                self.username.setText(f"User: {name}")
                self.userrole = UserRoles(role)
                self.signinout.setText("&Sign Out")
                self.ui1.tool_User.setText(name)
                self.parent.AnalyzeProc.tool_User.setText(name)
                if self.userrole != UserRoles.ADMIN:
                    self.manage.setText("&Change Password...")
        else:
            if self.parent.MainWin.ui0.setNoUserMode(None):
                UserProfiles().session_end()
                name = self.username.text()[6:]
                Log.i(f"Goodbye, {name}! You have been signed out.")
                self.username.setText("User: [NONE]")
                self.userrole = UserRoles.NONE
                self.signinout.setText("&Sign In")
                self.manage.setText("&Manage Users...")
                self.ui1.tool_User.setText("Anonymous")
                self.parent.AnalyzeProc.tool_User.setText("Anonymous")
            else:
                Log.d("User has unsaved changes in Analyze mode. Sign out aborted.")

    def manage_user_profiles(self):
        # dissallow user management if Analyze mode still has unsaved changes (after prompt to close)
        # still logged in, but user cannot stay in Analyze mode
        if not self.parent.MainWin.ui0.setRunMode(None):
            Log.d("User has unsaved changes in Analyze mode. Manage users aborted.")
            return

        # dissallow user management if current mode is busy
        if not self.parent.ControlsWin.ui1.pButton_Start.isEnabled():
            PopUp.warning(self, "Action Not Allowed",
                          "User info cannot be changed during an active capture.\n" +
                          "Please 'Stop' the measurement before attempting this action.")
            return

        if self.userrole != UserRoles.ADMIN and self.userrole != UserRoles.NONE:
            # change password, and return
            name = self.username.text()[6:]
            found, filename = UserProfiles.find(name, None)
            if filename != None:
                UserProfiles.change_password(filename)
                return
            else:
                Log.e("Attempted to change password, but user was not found!")

        name = self.username.text()[6:]
        allow, admin = UserProfiles().manage(name, self.userrole)

        if admin == None and not UserProfiles.session_info()[0]:
            if name != "[NONE]":
                Log.i(f"Goodbye, {name}! You have been signed out.")
            self.username.setText("User: [NONE]")
            self.userrole = UserRoles.NONE
            self.signinout.setText("&Sign In")
            self.manage.setText("&Manage Users...")
            self.ui1.tool_User.setText("Anonymous")
            self.parent.AnalyzeProc.tool_User.setText("Anonymous")
            self.parent.MainWin.ui0.setNoUserMode(None)
        if admin != name and admin != None:
            Log.d("User name changed. Changing sign-in user info.")
            self.username.setText(f"User: {admin}")
            self.userrole = UserRoles.ADMIN
            self.signinout.setText("&Sign Out")
            self.manage.setText("&Manage Users...")
            self.ui1.tool_User.setText(admin)
            self.parent.AnalyzeProc.tool_User.setText(admin)

        if allow:
            self.manageUsersUI = UserProfilesManager(self, admin)
            self.manageUsersUI.show()

    def toggle_console(self):
        if self.current_timer.isActive():
            self.current_timer.stop()
        if not self.chk1.isChecked():
            Log.d("Hiding Console window")
            self.parent.MainWin.ui0.logview.setVisible(False)
            # self.parent.LogWin.ui4.centralwidget.setVisible(False)
        else:
            Log.d("Showing Console window")
            self.parent.MainWin.ui0.logview.setVisible(True)
            # self.parent.LogWin.ui4.centralwidget.setVisible(True)
        self.parent.AppSettings.setValue(
            "viewState_Console", self.chk1.isChecked())

    def toggle_amplitude(self):
        tc = self.show_top_plot()
        if not self.chk2.isChecked():
            Log.d("Hiding Amplitude plot(s)")
            for i, p in enumerate(self.parent._plt0_arr):
                if p == None:
                    continue
                p.setVisible(False)
                self.parent._plt0_arr[i] = p
        else:
            Log.d("Showing Amplitude plot(s)")
            for i, p in enumerate(self.parent._plt0_arr):
                if p == None:
                    continue
                p.setVisible(True)
                self.parent._plt0_arr[i] = p
        self.hide_top_plot(tc)
        self.parent.AppSettings.setValue(
            "viewState_Amplitude", self.chk2.isChecked())

    def toggle_temperature(self):
        tc = self.show_top_plot()
        if not self.chk3.isChecked():
            Log.d("Hiding Temperature plot")
            self.parent._plt4.setVisible(False)
        else:
            Log.d("Showing Temperature plot")
            self.parent._plt4.setVisible(True)
        self.hide_top_plot(tc)
        self.parent.AppSettings.setValue(
            "viewState_Temperature", self.chk3.isChecked())

    def toggle_RandD(self):
        if self.current_timer.isActive():
            self.current_timer.stop()
        if not self.chk4.isChecked():
            Log.d("Hiding Resonance/Dissipation plot(s)")
            self.parent.PlotsWin.ui2.pltB.setVisible(False)
        else:
            Log.d("Showing Resonance/Dissipation plot(s)")
            self.parent.PlotsWin.ui2.pltB.setVisible(True)
        self.parent.AppSettings.setValue(
            "viewState_Resonance_Dissipation", self.chk4.isChecked())

    def show_top_plot(self):
        toggle_console = False
        if self.chk2.isChecked() or self.chk3.isChecked():
            Log.d("Showing top plots window")
            toggle_console = self.parent.PlotsWin.ui2.plt.isVisible() == False
            self.parent.PlotsWin.ui2.plt.setVisible(True)
        return toggle_console

    def hide_top_plot(self, toggle_console):
        if self.chk2.isChecked() or self.chk3.isChecked():
            if toggle_console:
                if self.current_timer.isActive():
                    self.current_timer.stop()
                self.current_timer.setSingleShot(True)
                self.current_timer.start(100)
        else:
            Log.d("Hiding top plots window")
            self.parent.PlotsWin.ui2.plt.setVisible(False)

    def double_toggle_plots(self):
        Log.d("Toggling console window (for sizing)")
        self.chk4.setChecked(not self.chk4.isChecked())
        self.toggle_RandD()
        self.chk4.setChecked(not self.chk4.isChecked())
        QtCore.QTimer.singleShot(0, self.toggle_RandD)

    def view_tutorials(self):
        self.parent.TutorialWin.setVisible(
            not self.parent.TutorialWin.isVisible())
        self.chk5.setChecked(self.parent.TutorialWin.isVisible())

    def open_file(self, filepath, relative_to_cwd=True):
        try:
            if relative_to_cwd:
                fullpath = os.path.join(Architecture.get_path(), filepath)
            os_type = Architecture.get_os()
            if os_type == OSType.macosx:    # macOS
                subprocess.call(('open', fullpath))
            elif os_type == OSType.windows:  # Windows
                os.startfile(fullpath)
            elif os_type == OSType.linux:   # linux
                subprocess.call(('xdg-open', fullpath))
            else:                           # other variants
                Log.w("Unknown OS Type:", os_type)
                Log.w("Assuming Linux variant...")
                subprocess.call(('xdg-open', fullpath))
        except:
            Log.e(TAG, f"ERROR: Cannot open \"{os.path.split(fullpath)[1]}\"")

    def release_notes(self):
        self.open_file(f"docs/Release Notes {Constants.app_version}.pdf")

    def fw_change_log(self):
        self.open_file(
            f"QATCH_Q-1_FW_py_{Constants.best_fw_version}/FW Change Control Doc.pdf")

    def sw_change_log(self):
        self.open_file("QATCH/SW Change Control Doc.pdf")

    def view_license(self):
        self.open_file("docs/gpl.txt")
        self.open_file("docs/LICENSE.txt")

    def view_user_guide(self):
        self.open_file("docs/userguide.pdf")

    def check_for_updates(self):
        if hasattr(self.parent, "url_download"):
            delattr(self.parent, "url_download")
        color, status = self.parent.start_download(True)
        if color == "#ff0000":
            if status == "ERROR":
                PopUp.warning(self, "Check for Updates",
                              "An error occurred checking for updates.")
            if status == "OFFLINE":
                PopUp.warning(self, "Check for Updates",
                              "Unable to check online for updates.")
        elif color != "#00ff00":
            technicality = " available " if color == "#00c600" else " supported "
            PopUp.information(self, "Check for Updates",
                              f"You are running the latest{technicality}version.")

    def closeEvent(self, event):
        # Log.d(" Exit Setup/Control GUI")
        if hasattr(self, "close_no_confirm"):
            res = True
        else:
            res = PopUp.question(
                self, Constants.app_title, "Are you sure you want to quit QATCH Q-1 application now?", True)
        if res:
            # self.close()
            QtWidgets.QApplication.quit()
        else:
            event.ignore()


class Rename_Output_Files(QtCore.QObject):
    finished = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.finished.connect(self.indicate_done)
        self.DockingWidgets = []
        self.bThread = []
        self.bWorker = []

    def indicate_analyzing(self):
        color_err = '#333333'
        labelbar = "Run Stopped. Analyzing run..."
        self.infobar_write(color_err, labelbar)

    def indicate_saving(self):
        color_err = '#333333'
        labelbar = "Run Stopped. Saving run..."
        self.infobar_write(color_err, labelbar)

    def indicate_finalizing(self):
        color_err = '#333333'
        labelbar = "Run Stopped. Requesting run info..."
        self.infobar_write(color_err, labelbar)

    def indicate_error(self):
        color_err = '#ff0000'
        labelbar = "Run Stopped. Error saving data!"
        self.infobar_write(color_err, labelbar)

    def indicate_done(self):
        # are we really done?
        remaining_threads = 0
        for i in range(len(self.bThread)):
            if self.bWorker[i].isVisible():
                remaining_threads += 1
        if remaining_threads > 0:
            Log.d(
                f"Waiting on {remaining_threads} Run Info threads to close...")
            return  # not done yet
        color_err = '#333333'
        labelbar = "Run Stopped. Saved to local data."
        self.infobar_write(color_err, labelbar)

    def infobar_write(self, color_err, labelbar):
        self.parent.ControlsWin.ui1.infobar.setText(
            "<font color=#0000ff> Infobar </font><font color={}>{}</font>".format(color_err, labelbar))

    #######################################################################
    # Prompt user for run name(s) and rename new output file(s) accordingly
    #######################################################################
    def run(self) -> None:
        """
        Prompt user for run name(s) and rename new output file(s) accordingly.

        Args:
            None

        Logs:
            I/O warnings and 

        Returns:
            None
        """
        try:
            # Check if new files were generated.  If no new files generated, log as a warnings
            # and emit the finished signal to parent along with the error.
            if not os.path.exists(Constants.new_files_path):
                Log.w(TAG, "WARNING: No new files were generated by this run!")
                self.finished.connect(self.indicate_error)
                self.finished.emit()
                return

            # Copy and rename log files from the temporary directory in alphabetical order
            # to include a user-defined run name string
            content = []
            with open(Constants.new_files_path, 'r') as file:
                self.indicate_analyzing()
                new_file_time = strftime(
                    Constants.csv_default_prefix, localtime())
                content = file.readlines()
                content.sort()

            # Current device directory.
            current_directory = ""

            # Sub-device directory or the name of the run.
            run_directory = ""

            # Pop-up input text from user.
            input_text = ""

            # Pop-up status.
            status_ok = False

            # Additional_info_control request.
            ask_for_info = False

            # Loop through the paths of each data file re-writing them to
            # the user defined directory.
            for line in content:
                old_path = line.rstrip()
                path_parts = os.path.split(old_path)
                this_dir = path_parts[0]
                this_file = path_parts[1]
                this_name = this_file[:this_file.rindex('_')]
                copy_file = None

                # Skip if the new directory is the same as the temporary directory.
                if this_dir != current_directory:
                    current_directory = this_dir
                    path_split = os.path.split(this_dir)
                    path_root = path_split[0]
                    dev_name = path_split[1]
                    try:
                        _dev_pid = 0
                        _dev_name = dev_name
                        if _dev_name.count('_') > 0:
                            _dev_parts = _dev_name.split('_')
                            _dev_pid = int(_dev_parts[0], base=16) % 9
                            # do not override 'dev_name'
                            _dev_name = _dev_parts[1]

                        # Retrieve device info based on the deivce name and port-id
                        dev_info = FileStorage.DEV_info_get(
                            _dev_pid, _dev_name)
                        if 'NAME' in dev_info:
                            if dev_info['NAME'] != _dev_name:
                                dev_name = dev_info['NAME']
                    except:
                        Log.e(
                            TAG, f"Unable to lookup device info for: {dev_name}")

                    # Controls saving with errors.
                    force_save = True
                    is_good = AnalyzeProcess.Model_Data(old_path)

                    # If run is not deemed "good" report this to the user asking if they should save the
                    # force save with bad data.
                    if input_text == "" and not is_good:
                        force_save = PopUp.critical(self.parent,
                                                    "Capture Run",
                                                    "Are you sure you want to save this run?",
                                                    "This software contains AI technology that is trained to detect run quality and it has determined this run does not match the standard characteristics of a complete run. Analysis of this run data may enocunter issues. If so, trying again with a different crystal may yield better results.",
                                                    True)
                    self.indicate_saving()

                    # Prompt user for a runname.
                    while True:
                        if input_text == "":
                            if force_save:
                                input_text, status_ok = QtWidgets.QInputDialog.getText(self.parent, 'Name this run...',
                                                                                       # for device "{}":'.format(dev_name),
                                                                                       'Enter a name for this run:',
                                                                                       text=input_text)
                            else:
                                status_ok = False  # bad run, don't save with custom name

                        # Fetch run and run_parent directories based on user preferences.
                        run_directory = UserProfiles.user_preferences.get_file_save_path(
                            runname=input_text, device_id=dev_name, port_id=_dev_pid)
                        run_parent_directory = UserProfiles.user_preferences.get_folder_save_path(
                            runname=input_text, device_id=dev_name, port_id=_dev_pid)

                        if status_ok:
                            ask_for_info = True
                            # Remove any invalid characters from user input
                            invalid_characters = "\\/:*?\"'<>|"
                            for character in invalid_characters:
                                input_text = input_text.replace(
                                    character, '')
                            # Potentially remove this vvvvv
                            if _dev_pid != 0:  # append Port ID 1-4 for 4x1, ID A1-D6 for 4x6
                                if self.has_active_multi_port():  # 4x6 system
                                    # mask in port, e.g. "A" -> "A1"
                                    _dev_pid = (
                                        (_dev_pid + 9) << 4) | self.get_active_multi_port()
                                # else: 4x1 system, nothing to do
                                port_id = self._portIDfromIndex(_dev_pid)
                                run_directory = UserProfiles.user_preferences.get_file_save_path(
                                    runname=input_text, device_id=dev_name, port_id=port_id)

                            # Raise exception if runname retrieved from user is empty.
                            try:
                                if len(input_text) == 0:
                                    raise Exception(
                                        "No text entered. Please try again.")

                                # Using Device folder path from UserPreferences class.
                                # os.makedirs(os.path.join(
                                #     path_root, dev_name, run_directory), exist_ok=False)

                                os.makedirs(os.path.join(
                                    path_root, run_parent_directory, run_directory), exist_ok=False)
                                # break (done below)
                            except:
                                if len(input_text) > 0:
                                    PopUp.warning(self.parent,
                                                  "Duplicate Run Name",
                                                  "A run with this name already exists...\nPlease try again with a different name.")
                                else:
                                    PopUp.warning(self.parent,
                                                  "Enter a Run Name",
                                                  "Please enter a run name to save this run...\nPlease try again with a valid name.")
                                input_text = ""
                                continue  # no break (try again)
                            input_text = input_text.strip().replace(' ', '_')  # word spaces -> underscores
                        else:
                            ask_for_info = False
                            run_directory = "_unnamed"
                            os.makedirs(os.path.join(
                                path_root, run_parent_directory, run_directory), exist_ok=True)
                            input_text = new_file_time  # uniquify
                            if not force_save:
                                input_text += "_BAD"
                        break

                # Update run path to the new run path under run parent directory.
                new_run_path = os.path.join(
                    path_root, run_parent_directory, run_directory, this_file.replace(this_name, input_text))
                try:
                    os.rename(old_path, new_run_path)
                    Log.i(
                        ' Renamed "{}" ->\n         "{}"'.format(old_path, new_run_path))
                    copy_file = new_run_path
                except Exception as e:
                    # raise e
                    Log.e(' ERROR: Failed to rename "{}" to "{}"!!!'.format(
                        old_path, new_run_path))
                    self.finished.connect(self.indicate_error)
                    if os.path.isfile(old_path):
                        copy_file = old_path
                    if os.path.isfile(new_run_path):
                        copy_file = new_run_path
                old_path_parts = os.path.split(old_path)
                try:
                    # only try if empty
                    if len(os.listdir(old_path_parts[0])) == 0:
                        # delete old path folder (throws error if not empty)
                        os.rmdir(old_path_parts[0])
                except:
                    Log.e(
                        ' ERROR: Failed to clean-up after renaming "{}"!!!'.format(old_path_parts[1]))
                    self.finished.connect(self.indicate_error)

                if copy_file != None:  # require access controls
                    file_parts = os.path.split(copy_file)
                    folder = os.path.split(file_parts[0])[0]
                    if run_directory == "_unnamed":
                        zn = copy_file[:copy_file.rfind("_")] + ".zip"
                    else:
                        zn = os.path.join(file_parts[0], f"capture.zip")
                    archive_file = file_parts[1]
                    crc_file = archive_file[:-4] + ".crc"

                    # Create a new zip file with a password
                    with pyzipper.AESZipFile(zn, 'a',
                                             compression=pyzipper.ZIP_DEFLATED,
                                             allowZip64=True,
                                             encryption=pyzipper.WZ_AES) as zf:
                        # Add a protected file to the zip archive
                        friendly_name = f"{run_directory} ({datetime.date.today()})"
                        zf.comment = friendly_name.encode()  # run name

                        enabled, error, expires = UserProfiles.checkDevMode()
                        if enabled == False and (error == True or expires != ""):
                            PopUp.warning(self, "Developer Mode Expired",
                                          "Developer Mode has expired and this data capture will now be encrypted.\n" +
                                          "An admin must renew or disable \"Developer Mode\" to suppress this warning.")

                        if UserProfiles.count() > 0 and enabled == False:
                            # create a protected archive
                            zf.setpassword(hashlib.sha256(
                                zf.comment).hexdigest().encode())
                        else:
                            zf.setencryption(None)
                            if enabled:
                                Log.w(
                                    "Developer Mode is ENABLED - NOT encrypting ZIP file")

                        zf.write(copy_file, arcname=archive_file)
                        if archive_file.endswith(".csv"):
                            zf.writestr(crc_file, str(
                                hex(zf.getinfo(archive_file).CRC)))

                    os.remove(copy_file)

                if ask_for_info:
                    ask_for_info = False
                    self.indicate_finalizing()
                    self.bThread.append(QtCore.QThread())
                    user_name = None if self.parent == None else self.parent.ControlsWin.username.text()[
                        6:]
                    # TODO: more secure to pass user_hash (filename)
                    self.bWorker.append(QueryRunInfo(
                        run_directory, new_run_path, is_good, user_name, parent=self.parent))
                    self.bThread[-1].started.connect(self.bWorker[-1].show)
                    self.bWorker[-1].finished.connect(self.bThread[-1].quit)
                    # add here
                    self.bWorker[-1].finished.connect(self.indicate_done)
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

        except:
            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb
            a_list = ['Traceback (most recent call last):']
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        finally:
            if os.path.exists(Constants.new_files_path):
                os.remove(Constants.new_files_path)
            self.finished.emit()

    def _portIDfromIndex(self, pid):  # convert ASCII byte to character
        # For 4x1 system: expect pid 1-4, return "1" thru "4"
        # For 4x6 system: expect pid 0xA1-0xD6, return "A1" thru "D6"
        return hex(pid)[2:].upper()

    def has_active_multi_port(self):
        i = self.parent.ControlsWin.ui1.cBox_Port.currentText()
        i = 0 if i.find(":") == -1 else int(i.split(":")[0], base=16)
        if i != i % 9:  # 4x6 system detected, PID A-D, not 1-4
            return os.path.exists("plate-config.json")
        return False

    def get_active_multi_port(self):
        # returns 1-6, depending on active 4x6 port on MUX of active device
        return self.parent.active_multi_ch
        # NOTE: setting 'active_multi_ch' is not implemented, only defined as 1 (always)

# ------------------------------------------------------------------------------


class MainWindow(QtWidgets.QMainWindow):

    ReadyToShow = False
    set_cal1 = "0"
    set_cal2 = "0"

    ###########################################################################
    # Initializes methods, values and sets the UI
    ###########################################################################
    def __init__(self, samples=Constants.argument_default_samples):

        # :param samples: Default samples shown in the plot :type samples: int.
        # to be always placed at the beginning, initializes some important methods
        QtWidgets.QMainWindow.__init__(self)

        # Check application settings global variable to get/set elsewhere
        self.AppSettings = QtCore.QSettings(
            Constants.app_publisher, Constants.app_name)

        # Sets up the user interface from the generated Python script using Qt Designer
        # Instantiates Ui classes

        # Calls setupUi method of created instance
        self.LogWin = LoggerWindow()
        self.ControlsWin = ControlsWindow(self)
        # self.ControlsWin.move(270, 550)
        self.PlotsWin = PlotsWindow()
        # self.PlotsWin.move(270,0) #GUI position (x,y) on the screen
        self.InfoWin = InfoWindow()
        self.LoginWin = LoginWindow(self)
        # self.InfoWin.move(1082, 0)
        # self.ControlsWin.show()
        # self.PlotsWin.show()
        # self.InfoWin.show()
        self.AnalyzeProc = AnalyzeProcess(self)

        self.tecWorker = TECTask()

        self.MainWin = _MainWindow(self)

        Log.d("AppSettings = ", self.AppSettings.fileName())

        # Shared variables, initial values
        # amplitude / calibration curve (split for multi below)
        self._plt0_1 = None
        self._plt0_2 = None
        self._plt0_3 = None
        self._plt0_4 = None
        self._plt0_arr = [self._plt0_1, self._plt0_2,
                          self._plt0_3, self._plt0_4]  # array of all _plt0 plots
        self._plt1 = None  # phase (not used)
        self._plt2_1 = None  # resonance (split for multi below)
        self._plt2_2 = None
        self._plt2_3 = None
        self._plt2_4 = None
        self._plt2_arr = [self._plt2_1, self._plt2_2,
                          self._plt2_3, self._plt2_4]  # array of all _plt2 plots
        self._plt3_1 = None  # dissipation (split for multi below)
        self._plt3_2 = None
        self._plt3_3 = None
        self._plt3_4 = None
        self._plt3_arr = [self._plt3_1, self._plt3_2,
                          self._plt3_3, self._plt3_4]  # array of all _plt3 plots
        self._plt4 = None  # temperature (combined)
        self.multiplex_plots = 1
        # TODO: update this variable on write to MUX state of primary device
        self.active_multi_ch = 1
        self._timer_plot = None
        self._readFREQ = None
        self._QCS_installed = None
        self._ser_control = 0
        self._ser_error1 = 0
        self._ser_error2 = 0
        self._ser_err_usb = 0
        self._identifying = False
        self._no_ports_found = 0
        self._selected_port = ''

        # Reference variables
        self._text4 = [None, None, None, None]
        self._drop_applied = [False, False, False, False]
        self._run_finished = [False, False, False, False]
        self._baselinedata = [[[0, 0], [0, 0]], [
            [0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
        self._last_y_range = [[[0, 0], [0, 0]], [
            [0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
        self._last_y_delta = [0, 0, 0, 0]
        self._reference_flag = False
        self._vector_reference_frequency = [None]
        self._vector_reference_dissipation = [None]
        self._vector_1 = None
        self._vector_2 = None

        # Instantiates a Worker class
        self.worker = Worker()

        # Initiates an Upgrader class
        self.fwUpdater = FW_Updater()

        # Restore mode index for speed comboBox
        self.restore_mode_idx = np.full(len(OperationType), -1, int)

        # Populates comboBox for sources
        self.ControlsWin.ui1.cBox_Source.addItems(Constants.app_sources)

        # Configures specific elements of the PyQtGraph plots
        self._configure_plot()

        # Configures specific elements of the QTimers
        self._configure_timers()

        # Configures the connections between signals and UI elements
        self._configure_signals()

        # Configures the required permissions on the filesystem
        self._configure_filesystem()

        # Configures the tutorials interface for user interaction
        self._configure_tutorials()

        # Populates combo box for serial ports
        self._source_changed()
        self.ControlsWin.ui1.cBox_Source.setCurrentIndex(
            OperationType.calibration.value)
        self.ControlsWin.ui1.sBox_Samples.setValue(samples - 1)  # samples

        # Populate TEC temperature initial value
        self.ControlsWin.ui1.slTemp.setValue(25)  # start at ambient

        # enable ui
        self._enable_ui(True)
        ###################################################################################################################################
        if UserProfiles.count() == 0:
            QtCore.QTimer.singleShot(3000, self.start_download)
        # Gets the QCS installed on the device (not used now)
        # self._QCS_installed = PopUp.question_QCM(self, Constants.app_title, "Please choose the Quartz Crystal Resonator installed on the openQCM-1 Device (default 5MHz if exit)")

        # Safety check
        if (Constants.avg_in or Constants.avg_out) < 5:
            PopUp.warning(self, "Averaging Disabled", "WARNING: avg_in and/or avg_out are set to unsupported values that disable averaging." +
                                                      "\n\nThis seems unintentional and may result in unreliable measurement performance.")

        # self.MainWin.showMaximized()
        self.ReadyToShow = True

    def analyze_data(self, data_device=None, data_folder=None, data_file=None):
        action_role = UserRoles.ANALYZE
        check_result = UserProfiles().check(self.ControlsWin.userrole, action_role)
        if check_result == None:  # user check required, but no user signed in
            Log.w(
                f"Not signed in: User with role {action_role.name} is required to perform this action.")
            Log.i("Please sign in to continue.")
            self.ControlsWin.set_user_profile()  # prompt for sign-in
            check_result = UserProfiles().check(
                self.ControlsWin.userrole, action_role)  # check again
        if not check_result:  # no user signed in or user not authorized
            Log.w(
                f"ACTION DENIED: User with role {self.ControlsWin.userrole.name} does not have permission to {action_role.name}.")
            return  # deny action

        self.data_device = data_device
        self.data_folder = data_folder
        self.data_run = data_file
        if data_device == None:
            for _, dirs, _ in os.walk(os.path.join(os.getcwd(), Constants.log_export_path)):
                self.data_devices = dirs  # show all available devices in logged data
                break
            self.AnalyzeProc.scan_for_most_recent_run = True
            self.AnalyzeProc.hide()
            self.AnalyzeProc.reset()
            self.AnalyzeProc.showMaximized()
            self.AnalyzeProc.check_user_info()
            if len(self.data_devices) > 0:
                pass
            else:
                Log.w("No data devices available for selection.")
            return
        if data_folder == None:
            self.data_folders = FileStorage.DEV_get_logged_data_folders(
                data_device)
            if len(self.data_folders) > 0:
                self.aThread = QtCore.QThread()
                self.aWorker = QueryComboBox(self.data_folders, "run")
                self.aThread.started.connect(self.aWorker.show)
                self.aWorker.finished.connect(self.aThread.quit)
                self.aWorker.finished.connect(
                    self.analyze_data_get_data_folder)
                self.aThread.start()
            else:
                Log.w("No data folders available for selection.")
            return
        if data_file == None:
            self.data_files = FileStorage.DEV_get_logged_data_files(
                data_device, data_folder)
            if "capture.zip" in self.data_files:
                zn = os.path.join(Constants.log_export_path,
                                  data_device, data_folder, "capture.zip")
                if FileManager.file_exists(zn):
                    with pyzipper.AESZipFile(zn, 'r',
                                             compression=pyzipper.ZIP_DEFLATED,
                                             allowZip64=True,
                                             encryption=pyzipper.WZ_AES) as zf:
                        # Add a protected file to the zip archive
                        try:
                            zf.testzip()
                        except:
                            zf.setpassword(hashlib.sha256(
                                zf.comment).hexdigest().encode())
                        self.data_files = zf.namelist()
                        self.data_files = [
                            x for x in self.data_files if not x.endswith(".crc")]
            self.data_files = [
                x for x in self.data_files if not x.endswith("_tec.csv")]
            self.data_files = [
                x for x in self.data_files if not x.endswith("_poi.csv")]
            self.data_files = [
                x for x in self.data_files if not x.endswith(".txt")]
            self.data_files = [
                x for x in self.data_files if not x.endswith(".xml")]
            self.data_files = [
                x for x in self.data_files if not x.endswith(".pdf")]
            self.data_files = [
                x for x in self.data_files if not x.endswith(".zip")]
            self.data_files = [x for x in self.data_files if not x.endswith(
                Constants.export_file_format)]
            if len(self.data_files) == 1:
                self.data_file = self.data_files[0]
                Log.i("Selected data file = {}".format(self.data_file))
                # continue analysis
                self.analyze_data(self.data_device,
                                  self.data_folder, self.data_file)
            elif "_3rd.csv" in self.data_files[0] or "_3rd.csv" in self.data_files[-1]:
                if "_3rd.csv" in self.data_files[0]:
                    self.data_file = self.data_files[0]
                else:
                    self.data_file = self.data_files[1]
                Log.i(
                    f"Found {len(self.data_files)} modes in this run: {self.data_files}")
                Log.i("Selected data file = {}".format(self.data_file))
                # continue analysis
                self.analyze_data(self.data_device,
                                  self.data_folder, self.data_file)
            elif len(self.data_files) > 1:
                self.aThread = QtCore.QThread()
                self.aWorker = QueryComboBox(self.data_files, "log file")
                self.aThread.started.connect(self.aWorker.show)
                self.aWorker.finished.connect(self.aThread.quit)
                self.aWorker.finished.connect(self.analyze_data_get_data_file)
                self.aThread.start()
            else:
                Log.w("No data files available for selection.")
            return

        # do analysis here, we've been given a device, folder and file to get data from
        data_path = os.path.join(
            Constants.log_export_path, data_device, data_folder, data_file)

        is_good = True  # AnalyzeProcess.Model_Data(data_path)
        if not is_good:
            if not PopUp.critical(self,
                                  "Bad Run",
                                  "This run appears to be bad...\nPlease try again with a different crystal.\n\nDo you want to analyze this run anyways?",
                                  "This software contains a machine learning model that has been trained to detect run quality and it has determined this run looks bad.",
                                  True):
                Log.w("Analysis aborted by user due to bad run detection.")
                return

        # self.AnalyzeProc.cBox_Devices.clear()
        # self.AnalyzeProc.cBox_Runs.clear()
        # self.AnalyzeProc.cBox_Devices.addItems(self.data_devices)
        # self.AnalyzeProc.cBox_Devices.setFixedWidth(self.AnalyzeProc.cBox_Devices.sizeHint().width())
        # set by device change handler: self.AnalyzeProc.cBox_Runs.addItems(self.data_folders)
        # self.AnalyzeProc.cBox_Devices.setCurrentText(data_device)
        # self.AnalyzeProc.cBox_Runs.setCurrentText(data_folder)
        try:
            xml_path = data_path[0:-3] + "xml"
            # set always, even if not found
            self.AnalyzeProc.setXmlPath(xml_path)
            if os.path.exists(xml_path):
                doc = minidom.parse(xml_path)
                metrics = doc.getElementsByTagName("metric")
                for m in metrics:
                    name = m.getAttribute("name")
                    value = m.getAttribute("value")
                    if name == "start":
                        captured = value[0:value.find("T")]
            else:
                zn = os.path.join(os.path.split(xml_path)[0], "capture.zip")
                # Log.e(f"zn: {zn}")
                if FileManager.file_exists(zn):
                    with pyzipper.AESZipFile(zn, 'r') as zf:
                        captured = zf.comment.decode()[-11:-1]
                else:
                    raise Exception("No XML exists for this run.")
        except Exception as e:
            Log.e("ERROR:", e)
            captured = os.path.getctime(data_path)
            captured = datetime.datetime.fromtimestamp(captured)
            captured = captured.strftime("%Y-%m-%d")  # %H:%M:%S")
        self.AnalyzeProc.text_Created.setText(
            "Loaded: {} ({})".format(data_folder, captured))

        self.AnalyzeProc.Analyze_Data(data_path)

    def analyze_data_get_data_device(self):
        idx = self.aWorker.clickedButton()
        if idx >= 0:
            self.data_device = self.data_devices[idx]
            Log.i("Selected data device = {}".format(self.data_device))
            self.analyze_data(self.data_device)  # continue analysis
        else:
            self.data_device = self.data_folder = self.data_file = None
            Log.w("User aborted data device selection.")

    def analyze_data_get_data_folder(self):
        idx = self.aWorker.clickedButton()
        if idx >= 0:
            self.data_folder = self.data_folders[idx]
            Log.i("Selected data folder = {}".format(self.data_folder))
            # continue analysis
            self.analyze_data(self.data_device, self.data_folder)
        else:
            self.data_device = self.data_folder = self.data_file = None
            Log.w("User aborted data folder selection.")

    def analyze_data_get_data_file(self):
        idx = self.aWorker.clickedButton()
        if idx >= 0:
            self.data_file = self.data_files[idx]
            Log.i("Selected data file = {}".format(self.data_file))
            self.analyze_data(self.data_device, self.data_folder,
                              self.data_file)  # continue analysis
        else:
            self.data_device = self.data_folder = self.data_file = None
            Log.w("User aborted data file selection.")

    def auto_update_view_windows(self):
        chk1 = self.ControlsWin.chk1.isChecked()  # console
        chk2 = self.ControlsWin.chk2.isChecked()  # amplitude
        chk3 = self.ControlsWin.chk3.isChecked()  # temperature
        chk4 = self.ControlsWin.chk4.isChecked()  # resonance/dissipation
        if self._get_source() == OperationType.calibration:
            self.ControlsWin.chk2.setChecked(True)
            self.ControlsWin.chk3.setChecked(False)
            self.ControlsWin.chk4.setChecked(False)
        elif self._get_source() == OperationType.measurement:
            self.ControlsWin.chk2.setChecked(True)
            self.ControlsWin.chk3.setChecked(False)
            self.ControlsWin.chk4.setChecked(True)
        if chk1 != self.ControlsWin.chk1.isChecked():
            self.ControlsWin.toggle_console()
        if chk2 != self.ControlsWin.chk2.isChecked():
            self.ControlsWin.toggle_amplitude()
        if chk3 != self.ControlsWin.chk3.isChecked():
            self.ControlsWin.toggle_temperature()
        if chk4 != self.ControlsWin.chk4.isChecked():
            self.ControlsWin.toggle_RandD()

    ###########################################################################
    # Starts the acquisition of the selected serial port
    ###########################################################################

    def start(self):
        action_role = UserRoles.CAPTURE
        check_result = UserProfiles().check(self.ControlsWin.userrole, action_role)
        if check_result == None:  # user check required, but no user signed in
            Log.w(
                f"Not signed in: User with role {action_role.name} is required to perform this action.")
            Log.i("Please sign in to continue.")
            self.ControlsWin.set_user_profile()  # prompt for sign-in
            check_result = UserProfiles().check(
                self.ControlsWin.userrole, action_role)  # check again
        if not check_result:  # no user signed in or user not authorized
            Log.w(
                f"ACTION DENIED: User with role {self.ControlsWin.userrole.name} does not have permission to {action_role.name}.")
            return  # deny action

        Log.d("GUI: Clear console window")
        # NOTE: Calling 'os.system' causes a console window to blink in and disappear when launched with 'pythonw.exe':
        # os.system('cls' if os.name == 'nt' else 'clear')

        # If starting run, uncheck "Identify" button and close port (if open)
        self._port_identify_stop()

        # Disable UI elements for run
        self._enable_ui(False)

        # This function is connected to the clicked signal of the Start button.
        Log.i(TAG, "Clicked START")

        # Focus plots window (useful if hidden)
        self.PlotsWin.raise_()

        ### BEGIN HANDLE ORPHANED FILES ###
        # Move any existing orphaned CSV files in the output directories from prior Returns
        # This is to prevent new files from being missed and not renamed at the end of a run
        try:
            if os.path.exists(Constants.new_files_path):
                os.remove(Constants.new_files_path)
            import glob
            # get a recursive list of file paths that matches pattern including sub directories
            fileList = glob.glob(os.path.join(
                os.getcwd(), Constants.csv_export_path, '*/*.csv'))
            # Iterate over the list of filepaths & remove each file.
            for old_path in fileList:
                subDir = "_unnamed"
                path_parts = os.path.split(old_path)
                # filenames in list will be left alone
                if path_parts[1] in ["output_tec.csv"]:
                    continue  # only ignore certain filenames
                Log.w(TAG, "WARNING: Found an orphaned output file!")
                os.makedirs(os.path.join(path_parts[0], subDir), exist_ok=True)
                new_file_time = strftime(
                    Constants.csv_default_prefix, localtime())  # uniquify
                new_path = os.path.join(
                    path_parts[0], subDir, "{}_{}".format(new_file_time, path_parts[1]))
                try:
                    os.rename(old_path, new_path)
                    Log.i(' Renamed "{}" ->\n         "{}"'.format(old_path, new_path))
                except:
                    Log.e(' ERROR: Failed to rename "{}" to "{}"!!!'.format(
                        old_path, new_path))
        except:
            Log.e(
                TAG, "ERROR: Failed to move prior created files. Some new files may not be renamed.")
        ### END HANDLE ORPHANED FILES ###

        # Add TEC output file to new files list (if exists)
        tec_log_path = FileStorage.DEV_populate_path(Constants.tec_log_path, 0)
        if os.path.exists(tec_log_path) and not self.tecWorker._tec_state == "OFF":
            with open(Constants.new_files_path, 'a') as tempFile:
                tempFile.write(tec_log_path + "\n")

        selected_port = self.ControlsWin.ui1.cBox_Port.currentData()
        if selected_port == None:
            selected_port = ''  # Dissallow None
        if selected_port == "CMD_DEV_INFO":
            selected_port = ''  # Dissallow Action

        if selected_port == '':
            self.ControlsWin.ui1.pButton_Refresh.clicked.emit()  # look for devices

            now_port = self.ControlsWin.ui1.cBox_Port.currentData()
            if now_port == None:
                now_port = ''  # Dissallow None
            if now_port == "CMD_DEV_INFO":
                now_port = ''  # Dissallow Action

            if now_port == '' or now_port == None:
                if self.ControlsWin.ui1.cBox_Port.count() > 2:  # multiple devices
                    Log.e(
                        "Multiple devices detected. Please select a device and try again.")
                    PopUp.warning(
                        self, Constants.app_title, "Multiple devices detected. Please select a device in Advanced Settings and try again.")
                else:
                    Log.e(
                        "No device is detected. Please connect a device and try again.")
                    PopUp.warning(
                        self, Constants.app_title, "No device is detected. Please connect a device and try again.")
                self._enable_ui(True)
                return
            else:
                selected_port = now_port

        if self.multiplex_plots > 1:
            selected_port = []
            for i in range(self.multiplex_plots):
                if i < self.ControlsWin.ui1.cBox_Port.count() - 1:
                    selected_port.append(
                        self.ControlsWin.ui1.cBox_Port.itemData(i))

        if self._get_source() == OperationType.measurement:
            enabled, error, expires = UserProfiles.checkDevMode()
            if enabled == False and (error == True or expires != ""):
                PopUp.warning(self, "Developer Mode Expired",
                              "Developer Mode has expired and this data capture will be encrypted.\n" +
                              "An admin must renew or disable \"Developer Mode\" to suppress this warning.")

        if self._get_source() == OperationType.measurement:
            is_recent, age_in_mins = self._get_cal_age()
            if not is_recent:
                Log.w(TAG, "Initialize was last performed {} minute{} ago.".format(
                    age_in_mins, "" if age_in_mins == 1 else "s"))
                if PopUp.question(self, "Initialize Recommended", "Initialize has not been performed recently.\nWould you like to initialize now?", True):
                    self.ControlsWin.ui1.cBox_Source.setCurrentIndex(
                        OperationType.calibration.value)
            else:
                Log.i(TAG, "Initialize was last performed {} minute{} ago.".format(
                    age_in_mins, "" if age_in_mins == 1 else "s"))
            # application EXE hangs on close if we do not check before doing a measurement run every single time
            self.fwUpdater.checkAgain()
        else:
            # Check for and remove any invalid calibration files in root of config folder on CAL start
            paths = [Constants.csv_calibration_path,
                     Constants.cvs_peakfrequencies_path]
            for p in paths:
                path = p.replace(Constants.tbd_active_device_name_path, '')
                if os.path.exists(path):
                    Log.w(
                        f"Removing invalid initialization file from 'config' root: {path}")
                    os.remove(path)

        # Instantiates process
        self.worker.config(QCS_on=self._QCS_installed,
                           port=selected_port,
                           pid=self.ControlsWin.ui1.cBox_Port.currentIndex() + 1,
                           speed=self.ControlsWin.ui1.cBox_Speed.currentText(),
                           samples=self.ControlsWin.ui1.sBox_Samples.value() + 1,
                           source=self._get_source(),
                           export_enabled=self.ControlsWin.ui1.chBox_export.isChecked(),
                           freq_hopping=self.ControlsWin.ui1.chBox_freqHop.isChecked(),
                           reconstruct=self.ControlsWin.ui1.chBox_correctNoise.isChecked())

        # Check for firmware updates (only if not yet checked this instance)
        try:
            do_continue = self.fwUpdater.run(self)
            if self.fwUpdater._port_changed:
                self.ControlsWin.ui1.pButton_Refresh.clicked.emit()
                selected_port = self.fwUpdater._port
                self.worker._port = selected_port
            if not do_continue:
                Log.e(
                    "Firmware is incompatible for running with this version of software.")
                self._enable_ui(True)
                return
        except Exception as e:
            Log.e(f"Error during pre-run FW check: {e}")
            self._enable_ui(True)
            return

        self.setMultiMode()

        if self._get_source() == OperationType.measurement:
            color_err = "#333333"
            labelbar = "Starting..."
            self.ControlsWin.ui1.infobar.setText(
                "<font color=#0000ff> Infobar </font><font color={}>{}</font>".format(color_err, labelbar))
            self.ControlsWin.ui1._update_progress_text()
            self.ControlsWin.ui1.progressBar.repaint()

            # set variable to preload tensorflow module, if desired
            # hide info/warning logs from tf # lazy load
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            import logging
            logging.getLogger("tensorflow").setLevel(
                logging.ERROR)  # suppress AutoGraph warnings
            if Constants.preload_tensorflow and Constants.Tensorflow_predict:
                # load tensorflow library once per session
                Log.d("GUI: Force repaint events")
                Log.w("Loading tensorflow modules...")
                import tensorflow as tf  # lazy load
                Log.d("LOADED: tensorflow as tf")
                Log.d("GUI: Normal repaint events")

        worker_check = self.worker.start()
        if worker_check == 1:
            # Gets frequency range
            self._readFREQ = self.worker.get_frequency_range()
            # Duplicate frequencies
            self._text4 = [None, None, None, None]
            self._drop_applied = [False, False, False, False]
            self._run_finished = [False, False, False, False]
            self._baselinedata = [[[0, 0], [0, 0]], [
                [0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
            self._last_y_range = [[[0, 0], [0, 0]], [
                [0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
            self._last_y_delta = [0, 0, 0, 0]
            self._reference_flag = False
            self._vector_reference_frequency = [list(self._readFREQ)]
            self._reference_value_frequency = [0]
            self._reference_value_dissipation = [0]
            self._labelref1 = "not set"
            self._labelref2 = "not set"
            # progressbar variables
            self._completed = 0
            self._ser_control = 0
            # error variables
            self._ser_error1 = 0
            self._ser_error2 = 0
            self._ser_err_usb = 0
            ##### other useful location #########
            # self.get_web_info()
            #####

            # check for invalid range and abort if bad
            if self._readFREQ[0] >= self._readFREQ[-1]:
                Log.e(
                    "Invalid frequency range calculated. Re-Initialize and try again. Aborting run...")
                self._enable_ui(True)
                return

            # AJR 2023-03-07: disabled (for now)
            # self.auto_update_view_windows()

            if self._get_source() == OperationType.measurement:
                overtones_number = len(
                    self.worker.get_source_speeds(OperationType.measurement))

                # set the quartz sensor
                if overtones_number == 5:
                    label_quartz = "@5MHz_QCM"
                elif overtones_number == 3:
                    label_quartz = "@10MHz_QCM"
                elif overtones_number == 2:
                    label_quartz = "@5MHz_QCM"

                self.InfoWin.ui3.info1a.setText(
                    "<font color=#0000ff > Device Setup </font>" + label_quartz)
                label11 = "Measurement Qatch Q-1"
                self.InfoWin.ui3.info11.setText(
                    "<font color=#0000ff > Operation Mode </font>" + label11)
                self._overtone_name, self._overtone_value, self._fStep = self.worker.get_overtone()
                label6 = str(int(self._overtone_value))+" Hz"
                self.InfoWin.ui3.info6.setText(
                    "<font color=#0000ff > Frequency Value </font>" + label6)
                # +" "+ str(self._overtone_value)+"Hz"
                label2 = str(self._overtone_name)
                self.InfoWin.ui3.info2.setText(
                    "<font color=#0000ff > Selected Frequency </font>" + label2)
                label3 = str(int(self._readFREQ[0]))+" Hz"
                self.InfoWin.ui3.info3.setText(
                    "<font color=#0000ff > Start Frequency </font>" + label3)
                label4 = str(int(self._readFREQ[-1]))+" Hz"
                self.InfoWin.ui3.info4.setText(
                    "<font color=#0000ff > Stop Frequency </font>" + label4)
                label4a = str(int(self._readFREQ[-1]-self._readFREQ[0]))+" Hz"
                self.InfoWin.ui3.info4a.setText(
                    "<font color=#0000ff > Frequency Range </font>" + label4a)
                label5 = str(int(self._fStep))+" Hz"
                self.InfoWin.ui3.info5.setText(
                    "<font color=#0000ff > Sample Rate </font>" + label5)
                label7 = str(Constants.argument_default_samples-1)
                self.InfoWin.ui3.info7.setText(
                    "<font color=#0000ff > Sample Number </font>" + label7)

            elif self._get_source() == OperationType.calibration:
                label_quartz = self.ControlsWin.ui1.cBox_Speed.currentText()
                self.InfoWin.ui3.info1a.setText(
                    "<font color=#0000ff > Device Setup </font>" + label_quartz)
                label11 = "Calibration Qatch Q-1"
                self.InfoWin.ui3.info11.setText(
                    "<font color=#0000ff > Operation Mode </font>" + label11)
                label6 = "Overall Frequency Range"
                self.InfoWin.ui3.info6.setText(
                    "<font color=#0000ff > Frequency Value </font>" + label6)
                label2 = "Overall Frequency Range"
                self.InfoWin.ui3.info2.setText(
                    "<font color=#0000ff > Selected Frequency </font>" + label2)
                label3 = str(Constants.calibration_frequency_start)+" Hz"
                self.InfoWin.ui3.info3.setText(
                    "<font color=#0000ff > Start Frequency </font>" + label3)
                label4 = str(Constants.calibration_frequency_stop)+" Hz"
                self.InfoWin.ui3.info4.setText(
                    "<font color=#0000ff > Stop Frequency </font>" + label4)
                label4a = str(int(Constants.calibration_frequency_stop -
                              Constants.calibration_frequency_start))+" Hz"
                self.InfoWin.ui3.info4a.setText(
                    "<font color=#0000ff > Frequency Range </font>" + label4a)
                label5 = str(int(Constants.calibration_fStep))+" Hz"
                self.InfoWin.ui3.info5.setText(
                    "<font color=#0000ff > Sample Rate </font>" + label5)
                label7 = str(Constants.calibration_default_samples-1)
                self.InfoWin.ui3.info7.setText(
                    "<font color=#0000ff > Sample Number </font>" + label7)
                self.stop_flag = 0
            #
            self._timer_plot.start(Constants.plot_update_ms)
            # self._timer_plot.timeout.connect(self._update_plot) #moved from _configure_timers mothod

            if self._get_source() == OperationType.calibration:
                self.ControlsWin.ui1.pButton_Clear.setEnabled(False)  # insert
                self.ControlsWin.ui1.pButton_Reference.setEnabled(
                    False)  # insert
        elif worker_check == 0:
            Log.w(TAG, "Warning: port is not available")
            PopUp.warning(self, Constants.app_title, "Warning: Selected Port [{}] is not available!".format(
                self.ControlsWin.ui1.cBox_Port.currentText()))
            self._enable_ui(True)
        elif worker_check == -1:
            Log.w(TAG, "Warning: No peak magnitudes found. Rerun Initialize.")
            PopUp.warning(self, Constants.app_title, "Warning: No peak magnitudes found. Rerun Initialize on port {}".format(
                self.ControlsWin.ui1.cBox_Port.currentText()))
            self._enable_ui(True)

    ###########################################################################
    # Stops the acquisition of the selected serial port
    ###########################################################################

    def stop(self):

        # This function is connected to the clicked signal of the Stop button.
        self.ControlsWin.ui1.infostatus.setStyleSheet(
            'background: white; padding: 1px; border: 1px solid #cccccc')
        self.ControlsWin.ui1.infostatus.setText(
            "<font color=#333333 > Program Status Standby</font>")
        self.ControlsWin.ui1.infobar_label.setText(
            "<font color=#0000ff > Infobar </font>Stopped")
        self.InfoWin.ui3.inforef1.setText(
            "<font color=#0000ff > Ref. Frequency </font>")
        self.InfoWin.ui3.inforef2.setText(
            "<font color=#0000ff > Ref. Dissipation </font>")
        self.ControlsWin.ui1.progressBar.setValue(0)
        Log.i(TAG, "Clicked STOP")
        self._timer_plot.stop()
        self._enable_ui(True)
        self.worker.stop()

        if self._get_source() == OperationType.measurement:
            # start rename operation in separate thread (non-blocking)
            self.renameThread = QtCore.QThread()
            self.renameWorker = Rename_Output_Files(self)
            self.renameThread.started.connect(self.renameWorker.run)
            self.renameWorker.finished.connect(self.renameThread.quit)
            self.renameThread.start()

            for i in range(len(self._drop_applied)):
                if self._text4[i] != None:
                    self._text4[i].setText(' ')  # clear plot status message

    ###########################################################################
    # Overrides the QTCloseEvent,is connected to the close button of the window
    ###########################################################################

    def closeEvent(self, evnt):

        # :param evnt: QT evnt.
        if self.worker.is_running():
            Log.i(
                TAG, "Window closed without stopping the capture, application will stop...")
            self.stop()
            # self.ControlsWin.close()
            # self.PlotsWin.close()
            # self.InfoWin.close()
            # evnt.accept()

        if hasattr(self, "tecThread"):
            if self.tecThread.isRunning() or self.tecWorker._tec_locked:
                Log.w("Stopping TEC before closing...")
                self.tecWorker._tec_update("OFF")

    ###########################################################################
    # Enables or disables the UI elements of the window.
    ###########################################################################

    def _enable_ui(self, enabled):

        # :param enabled: The value to be set for the UI elements :type enabled: bool
        self.ControlsWin.ui1.cBox_Port.setEnabled(enabled)
        self.ControlsWin.ui1.pButton_ID.setEnabled(enabled)
        self.ControlsWin.ui1.pButton_Refresh.setEnabled(enabled)
        self.ControlsWin.ui1.cBox_Speed.setEnabled(enabled)
        self.ControlsWin.ui1.pButton_Start.setEnabled(enabled)
        self.ControlsWin.ui1.chBox_freqHop.setEnabled(enabled)
        self.ControlsWin.ui1.chBox_correctNoise.setEnabled(enabled)
        self.ControlsWin.ui1.chBox_export.setEnabled(enabled)
        self.ControlsWin.ui1.cBox_Source.setEnabled(enabled)
        self.ControlsWin.ui1.cBox_MultiMode.setEnabled(enabled)
        self.ControlsWin.ui1.pButton_PlateConfig.setEnabled(enabled)
        self.ControlsWin.ui1.chBox_MultiAuto.setEnabled(enabled)
        # self.ControlsWin.ui1.lTemp.setEnabled(enabled)
        self.ControlsWin.ui1.slTemp.setEnabled(enabled)
        self.ControlsWin.ui1.pTemp.setEnabled(enabled)
        self.tecWorker.set_slider_enable(enabled)

        if self.ControlsWin.ui1.pTemp.text() == "Stop Temp Control" and not enabled:
            self.ControlsWin.ui1.pTemp.setText("Temp Control Locked")
            self.tecWorker._tec_update_now = True
            self.tecWorker.update_now.emit()

        if self.ControlsWin.ui1.pTemp.text() == "Temp Control Locked" and enabled:
            self.ControlsWin.ui1.pTemp.setText("Stop Temp Control")
            self.tecWorker._tec_update_now = True
            QtCore.QTimer.singleShot(1000, self.tecWorker.update_now.emit)
            # self.tecWorker.update_now.emit()

        self.ControlsWin.ui1.pButton_Stop.setEnabled(not enabled)
        self.ControlsWin.ui1.sBox_Samples.setEnabled(not enabled)  # insert
        self.ControlsWin.ui1.pButton_Clear.setEnabled(not enabled)
        self.ControlsWin.ui1.pButton_Reference.setEnabled(not enabled)
        self.ControlsWin.ui1.pButton_Reference.setChecked(
            False)  # clear toggle state

        enable_start = enabled
        if enable_start:
            is_recent, _ = self._get_cal_age()
            enable_start = is_recent

        enable_stop = False if enabled else True
        if enable_stop:
            enable_stop = self.ControlsWin.ui1.cBox_Source.currentIndex(
            ) == OperationType.measurement.value

        enable_temp = enabled
        if enable_temp:
            # at least one device connected
            enable_temp = self.ControlsWin.ui1.cBox_Port.count() > 1

        self.ControlsWin.ui1.tool_Initialize.setEnabled(enabled)
        self.ControlsWin.ui1.tool_Start.setEnabled(enable_start)
        self.ControlsWin.ui1.tool_Stop.setEnabled(enable_stop)
        self.ControlsWin.ui1.tool_Reset.setEnabled(enabled)
        self.ControlsWin.ui1.tool_TempControl.setEnabled(enable_temp)
        # self.ControlsWin.ui1.tool_Advanced.setEnabled(enabled)

        # For more details on "bug in PyQt under macOS"... http://stackoverflow.com/a/60074600
        # required due to bug in PyQt under macOS
        self.ControlsWin.ui1.pButton_Start.repaint()
        # required due to bug in PyQt under macOS
        self.ControlsWin.ui1.pButton_Stop.repaint()

    ###########################################################################
    # Configures specific elements of the PyQtGraph plots.
    ###########################################################################

    def _configure_plot(self):

        ###############################################################################
        #  Provides a date-time aware axis
        ###############################################################################
        class DateAxis(AxisItem):
            def __init__(self, *args, **kwargs):
                super(DateAxis, self).__init__(*args, **kwargs)

            def tickStrings(self, values, scale, spacing):
                try:
                    # If less than 1 hour: display as "MM:SS" format.
                    # If equal or over 1 hour: display as "HH:MM:SS".
                    z = [datetime.datetime.utcfromtimestamp(float(value)).strftime("%M:%S")
                         if datetime.datetime.utcfromtimestamp(float(value)).strftime("%H") == "00"
                         else datetime.datetime.utcfromtimestamp(float(value)).strftime("%H:%M:%S")
                         for value in values]
                except:
                    z = ''
                return z

        # ----------------------------------------------------------------------
        # set background color
        self.PlotsWin.ui2.plt.setBackground(background='#FFFFFF')
        self.PlotsWin.ui2.pltB.setBackground(background='#FFFFFF')

        # ----------------------------------------------------------------------
        # defines the graph title
        title1 = "Real-Time Plot: Amplitude"
        title2 = "Real-Time Plot: Resonance Frequency / Dissipation"
        title3 = "Real-Time Plot: Temperature"
        # --------------------------------------------------------------------------------------------------------------
        # Configures elements of the PyQtGraph plots: amplitude
        self.PlotsWin.ui2.plt.setAntialiasing(True)
        self.PlotsWin.ui2.pltB.setAntialiasing(True)
        for i, p in enumerate(self._plt0_arr):
            if i >= self.multiplex_plots:
                # save back to global, otherwise local gets trashed
                self._plt0_arr[i] = None
                continue
            span = 1
            visible = True
            x = int(i % 2)  # 0 -> 0     1 -> 1      2 -> 0      3 -> 1
            y = int(i / 2) + 1  # 0 -> 0     1 -> 0      2 -> 1      3 -> 1
            if self._get_source() == OperationType.measurement:
                visible = True if i == 0 else False
                if i == 0:
                    span = 2
                    y = 0
            p = self.PlotsWin.ui2.plt.addPlot(
                col=x, row=y, colspan=span, title=title1+f" {i+1}", **{'font-size': '10pt'})
            p.showGrid(x=True, y=True)
            p.setVisible(visible)
            p.setLabel('bottom', 'Frequency', units='Hz')
            p.setLabel('left', 'Amplitude', units='dB',
                       color=Constants.plot_colors[0], **{'font-size': '10pt'})
            # set size policy for show/hide
            not_resize = p.sizePolicy()
            not_resize.setHorizontalStretch(1)
            p.setSizePolicy(not_resize)
            # save back to global, otherwise local gets trashed
            self._plt0_arr[i] = p

        # --------------------------------------------------------------------------------------------------------------
        # Configures elements of the PyQtGraph plots: resonance
        self._yaxis = []
        self._xaxis = []
        for i, p in enumerate(self._plt2_arr):
            if i >= self.multiplex_plots:
                # save back to global, otherwise local gets trashed
                self._plt2_arr[i] = None
                continue
            x = int(i % 2)  # 0 -> 0     1 -> 1      2 -> 0      3 -> 1
            y = int(i / 2)  # 0 -> 0     1 -> 0      2 -> 1      3 -> 1
            self._yaxis.append(AxisItem(orientation='left'))
            self._xaxis.append(DateAxis(orientation='bottom'))
            p = self.PlotsWin.ui2.pltB.addPlot(col=x, row=y, title=title2+f" {i+1}", **{
                                               'font-size': '12pt'}, axisItems={"bottom": self._xaxis[i], "left": self._yaxis[i]})
            p.showGrid(x=True, y=True)
            p.setLabel('bottom', 'Time', units='s')
            p.setLabel('left', 'Resonance Frequency', units='Hz',
                       color=Constants.plot_colors[2], **{'font-size': '10pt'})
            # save back to global, otherwise local gets trashed
            self._plt2_arr[i] = p

        # --------------------------------------------------------------------------------------------------------------
        # Configures elements of the PyQtGraph plots: Multiple Plot resonance frequency and dissipation
        for i, p in enumerate(self._plt3_arr):
            if i >= self.multiplex_plots:
                # save back to global, otherwise local gets trashed
                self._plt3_arr[i] = None
                continue
            _p2 = self._plt2_arr[i]
            p = pg.ViewBox()
            _p2.showAxis('right')
            _p2.scene().addItem(p)
            _p2.getAxis('right').setGrid(False)
            _p2.getAxis('right').linkToView(p)
            p.setXLink(_p2)
            _p2.enableAutoRange(axis='y', enable=True)
            p.enableAutoRange(axis='y', enable=True)
            _p2.setLabel('bottom', 'Time', units='s')
            _p2.setLabel('right', 'Dissipation', units='',
                         color=Constants.plot_colors[3], **{'font-size': '10pt'})
            # save back to global, otherwise local gets trashed
            self._plt2_arr[i] = _p2
            # save back to global, otherwise local gets trashed
            self._plt3_arr[i] = p

        self._annotate_welcome_text()

        # -----------------------------------------------------------------------------------------------------------------
        # Configures elements of the PyQtGraph plots: temperature
        self._plt4 = self.PlotsWin.ui2.plt.addPlot(row=3, col=0, colspan=2, title=title3, axisItems={
                                                   'bottom': DateAxis(orientation='bottom')})
        self._plt4.showGrid(x=True, y=True)
        self._plt4.setLabel('bottom', 'Time', units='s')
        self._plt4.setLabel('left', 'Temperature', units='°C',
                            color=Constants.plot_colors[4], **{'font-size': '10pt'})
        # set size policy for show/hide
        not_resize = self._plt4.sizePolicy()
        not_resize.setHorizontalStretch(1)
        self._plt4.setSizePolicy(not_resize)

        if self._plt1 is None and not Constants.plot_show_phase is False:
            self._configure_add_phase()

    ###########################################################################
    # Configures text comment and info
    ###########################################################################

    def _annotate_welcome_text(self):
        if self._plt2_arr[1] != None:
            self._plt2_arr[1].clear()
        else:
            self._plt2_arr[0].clear()
        font_size = 10 if self.multiplex_plots == 1 else 9
        self._text1 = pg.TextItem('', (51, 51, 51), anchor=(0.5, 0.5))
        self._text1.setHtml(
            "<span style='font-size: 14pt'>Welcome to QATCH nanovisQ<sup>TM</sup> Real-Time GUI </span>")
        self._text2 = pg.TextItem('', (51, 51, 51), anchor=(0.5, 0.5))
        self._text2.setHtml(
            f"<span style='font-size: {font_size}pt'>Don't forget to initialize (in air) your quartz device <b><i>before</i></b> starting. </span>")
        self._text3 = pg.TextItem('', (51, 51, 51), anchor=(0.5, 0.5))
        self._text3.setHtml(
            f"<span style='font-size: {font_size}pt'>Wait to apply the drop to your quartz device until <b><i>after</i></b> hitting \"Start\". </span>")
        if UserProfiles.count() == 0:
            self._text1.setPos(0.5, 0.65)
            self._text2.setPos(0.5, 0.35)
            self._text3.setPos(0.5, 0.25)
            if self._plt2_arr[1] != None:
                self._plt2_arr[1].addItem(self._text1, ignoreBounds=True)
            else:
                self._plt2_arr[0].addItem(self._text1, ignoreBounds=True)

        else:  # user sign in form shown
            self._text2.setPos(0.5, 0.55)
            self._text3.setPos(0.5, 0.45)
        if self._plt2_arr[1] != None:
            self._plt2_arr[1].addItem(self._text2, ignoreBounds=True)
            self._plt2_arr[1].addItem(self._text3, ignoreBounds=True)
        else:
            self._plt2_arr[0].addItem(self._text2, ignoreBounds=True)
            self._plt2_arr[0].addItem(self._text3, ignoreBounds=True)

    ###########################################################################
    # Configures phase-specific elements of the PyQtGraph plots
    ###########################################################################

    def _configure_add_phase(self):
        # --------------------------------------------------------------------------------------------------------------
        # Configures elements of the PyQtGraph plots: Multiple Plot amplitude and phase
        Log.w("Phase is no longer supported with multiplex software.")
        return
        self._plt0.setTitle("Real-Time Plot: Amplitude / Phase")
        self._plt1 = pg.ViewBox()
        self._plt0.showAxis('right')
        self._plt0.scene().addItem(self._plt1)
        self._plt0.getAxis('right').linkToView(self._plt1)
        self._plt1.setXLink(self._plt0)
        self._plt0.enableAutoRange(axis='y', enable=True)
        self._plt1.enableAutoRange(axis='y', enable=True)
        self._plt0.setLabel('right', 'Phase', units='deg',
                            color=Constants.plot_colors[1], **{'font-size': '10pt'})

    ###########################################################################
    # Configures specific elements of the QTimers
    ###########################################################################
    def _configure_timers(self):

        self._timer_plot = QtCore.QTimer(self)
        self._timer_plot.timeout.connect(self._update_plot)
        # self._timer_plot.timeout.connect(self._update_plot) #moved to start method

    ###########################################################################
    # Configures the connections between signals and UI elements
    ###########################################################################

    def _configure_signals(self):

        self.ControlsWin.ui1.pButton_Start.clicked.connect(self.start)
        self.ControlsWin.ui1.pButton_Stop.clicked.connect(self.stop)
        self.ControlsWin.ui1.pButton_Clear.clicked.connect(self.clear)
        self.ControlsWin.ui1.pButton_Reference.clicked.connect(self.reference)
        self.ControlsWin.ui1.pButton_ResetApp.clicked.connect(
            self.factory_defaults)
        self.ControlsWin.ui1.pButton_ID.clicked.connect(self._port_identify)
        self.ControlsWin.ui1.pButton_Refresh.clicked.connect(
            self._port_list_refresh)
        self.ControlsWin.ui1.sBox_Samples.valueChanged.connect(
            self._update_sample_size)
        self.ControlsWin.ui1.slTemp.valueChanged.connect(self._update_tec_temp)
        self.ControlsWin.ui1.slTemp.sliderReleased.connect(
            self._update_tec_temp)
        self.ControlsWin.ui1.cBox_Source.currentIndexChanged.connect(
            self._source_changed)
        self.ControlsWin.ui1.cBox_Port.currentIndexChanged.connect(
            self._port_changed)
        self.ControlsWin.ui1.cBox_MultiMode.currentIndexChanged.connect(
            self.setMultiMode)
        self.ControlsWin.ui1.pTemp.clicked.connect(self._enable_tec)
        # --------
        self.InfoWin.ui3.pButton_Download.clicked.connect(self.start_download)

    ###########################################################################
    # Configures the required permissions on the filesystem
    ###########################################################################

    def _configure_filesystem(self):
        # set permissions on local application data folders and files
        try:
            local_app_data_path = os.path.split(
                Constants.local_app_data_path)[0]
            path_to_logged_data = os.path.join(
                os.getcwd(), Constants.log_export_path)
            path_to_mydocs_data = os.path.join(
                os.getcwd(), Constants.app_publisher)
            if Architecture.get_os() == OSType.windows:
                # NOTE: Calling 'os.system' causes a console window to blip and disappear when launched with 'pythonw.exe':
                subprocess.call(
                    f"cd {local_app_data_path} & attrib -r -a -s -h /s /d", shell=True)
                subprocess.call(
                    f"cd {path_to_logged_data} & attrib -r -a -s -h /s /d", shell=True)
                subprocess.call(
                    f"cd {path_to_mydocs_data} & attrib -r -a -s -h /s /d", shell=True)
            else:
                os.chmod(local_app_data_path, stat.S_IRWXU |
                         stat.S_IRWXG | stat.S_IRWXO)
                for root, dirs, files in os.walk(local_app_data_path):
                    # set perms on sub-directories
                    for momo in dirs:
                        os.chmod(os.path.join(root, momo),
                                 stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                    # set perms on files
                    for momo in files:
                        os.chmod(os.path.join(root, momo),
                                 stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        except:
            Log.e(
                "ERROR: Unable to set file permissions on local application data folders and files.")

    ###########################################################################
    # Configures the tutorials interface for user interaction
    ###########################################################################

    def _configure_tutorials(self):
        # creating dock widget
        self.TutorialWin = QtWidgets.QDockWidget("Help: Tutorials", self)
        self.TutorialWidget = QtWidgets.QWidget(self)
        self.TutorialScroll = QtWidgets.QScrollArea(self)
        self.TutorialText = QtWidgets.QLabel(self)
        self.TutorialCheckbox = QtWidgets.QCheckBox(
            "Show these tutorials on startup", self)
        # stylesheet applies to titlebar and all children widgets
        self.TutorialWin.setStyleSheet("background-color: #A9E1FA;")
        # self.TutorialTitle.setWordWrap(True)
        scroll_widget = QtWidgets.QWidget(self)
        scroll_layout = QtWidgets.QVBoxLayout()
        scroll_layout.setContentsMargins(0, 0, 5, 0)
        scroll_layout.addWidget(self.TutorialText)
        scroll_layout.addStretch()
        scroll_widget.setLayout(scroll_layout)
        # Scroll Area Properties
        self.TutorialScroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.TutorialScroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # self.TutorialScroll.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.TutorialScroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.TutorialScroll.setWidgetResizable(True)
        self.TutorialScroll.setWidget(scroll_widget)
        self.TutorialText.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.TutorialText.setWordWrap(True)
        # apply layout to tutorials widget
        top_layout = QtWidgets.QVBoxLayout()
        top_layout.setContentsMargins(10, 10, 5, 10)
        # layout.addWidget(self.TutorialTitle)
        top_layout.addWidget(self.TutorialScroll)
        # layout.addStretch()
        top_layout.addWidget(self.TutorialCheckbox)
        # set size of docker, min overrides auto-width
        # self.TutorialWin.setMinimumSize(400, 100)
        self.TutorialWidget.setLayout(top_layout)
        self.TutorialWin.setWidget(self.TutorialWidget)
        # set widget to the dock
        self.MainWin.addDockWidget(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
            self.TutorialWin)
        # check application settings and set visibility on startup
        if self.AppSettings.contains("showTutorialsOnStartup"):
            if not self.AppSettings.value("showTutorialsOnStartup").lower() == "true":
                self.TutorialWin.setVisible(False)
        else:
            self.AppSettings.setValue("showTutorialsOnStartup", True)
        # update checkbox to reflect application settings state
        self.ControlsWin.chk5.setChecked(self.AppSettings.value(
            "showTutorialsOnStartup").lower() == "true")
        self.TutorialCheckbox.setChecked(self.AppSettings.value(
            "showTutorialsOnStartup").lower() == "true")
        self.TutorialCheckbox.clicked.connect(self.toggleTutorialOnStartup)
        # force widget to minimum width, determined by length of checkbox text
        min_size = self.TutorialWin.minimumSizeHint()
        self.TutorialWidget.setMinimumSize(
            min_size.width()+42, min_size.height())
        # self.TutorialText.setMinimumSize(min_size.width(), min_size.height())

        if UserProfiles.count() == 0:
            # insert crystal, next steps, create accounts
            self.viewTutorialPage([3, 4, 0])
        else:
            # login to begin, forgot password, create accounts
            self.viewTutorialPage([1, 2, 0])

    def viewTutorialPage(self, ids):
        if not hasattr(self, "TutorialText"):
            Log.d("Tutorials widget is not initialized yet. Ignoring page set.")
            return
        if not isinstance(ids, list):
            ids = [ids]
        for id in ids.copy():
            subset_ids = []
            # floating point precision errors cause issue if you just do simple addition without rounding
            step = round(id + 0.1, 2)
            while True:
                if isinstance(id, int) and step in TutorialPages.keys():
                    subset_ids.append(step)
                    # floating point precision errors cause issue if you just do simple addition without rounding
                    step = round(step + 0.1, 2)
                else:
                    break
            if len(subset_ids) > 0:
                base = ids.index(id)
                for y, sub in enumerate(subset_ids):
                    ids.insert(base+y+1, sub)
        tutorial_html = ""
        last_id = ids[-1]
        for id in ids:
            try:
                if id in TutorialPages.keys():
                    title, text = TutorialPages[id]
                else:
                    title, text = "Missing Tutorial", f"Sorry, the requested tutorial ID {id} cannot be found!"
            except:
                title, text = "Tutorial Error", f"Sorry, there was an error showing the requested tutorial ID {id}!"
            tutorial_html += f"<b>{title}</b><br/>{text}"
            if id != last_id:
                tutorial_html += "<br/><br/>"
        self.TutorialText.setText(tutorial_html)

    def toggleTutorialOnStartup(self):
        currentState = self.TutorialCheckbox.isChecked()
        self.AppSettings.setValue("showTutorialsOnStartup", currentState)

    ###########################################################################
    # Indentify device COM port by finding LED on device
    ###########################################################################

    def _port_identify(self):

        # Open or close the port accordingly
        selected_port = self.ControlsWin.ui1.cBox_Port.currentData()
        if selected_port == None:
            selected_port = ''  # Dissallow None
        if selected_port == "CMD_DEV_INFO":
            selected_port = ''  # Dissallow Action
        friendly_port_name = selected_port.split(':')[0]

        if not self._identifying:
            Log.i(TAG, f"Identifying port {friendly_port_name}...")
            self.ControlsWin.ui1.pButton_ID.setStyleSheet(
                "background: yellow;")
            self._identifying = True
            if True:  # not ';' in selected_port: # for NET only, call 'IDENTIFY' command
                # selected_port.count('.') == 3:
                if not len(selected_port) == 0:
                    IDENTIFY_serial = serial.Serial()
                    IDENTIFY_serial.port = selected_port
                    IDENTIFY_serial.write("identify\n".encode())
                    resp_bytes = IDENTIFY_serial.read().strip()
                    if not resp_bytes == b'1':
                        IDENTIFY_serial.write("identify\n".encode())
                        resp_bytes = IDENTIFY_serial.read().strip()
                    IDENTIFY_serial.close()
            # check version and write device info (if needed)
            self.worker._port = selected_port  # used in run()
            self.fwUpdater.run(self)
            if self.fwUpdater._port_changed:
                self.ControlsWin.ui1.pButton_Refresh.clicked.emit()
                selected_port = self.fwUpdater._port
                self.worker._port = selected_port
            # leave port open after update check to keep LED blinking
            self.fwUpdater.open(selected_port)
        else:
            Log.i(TAG, f"Identifying port {friendly_port_name}... done!")
            self.ControlsWin.ui1.pButton_ID.setStyleSheet("background: white;")
            self._identifying = False
            # close port to stop LED blink
            self.fwUpdater.close()
            if True:  # not ';' in selected_port: # for NET only, call 'IDENTIFY' command
                # selected_port.count('.') == 3:
                if not len(selected_port) == 0:
                    IDENTIFY_serial = serial.Serial()
                    IDENTIFY_serial.port = selected_port
                    IDENTIFY_serial.write("identify\n".encode())
                    resp_bytes = IDENTIFY_serial.read().strip()
                    if not resp_bytes == b'0':
                        IDENTIFY_serial.write("identify\n".encode())
                        resp_bytes = IDENTIFY_serial.read().strip()
                    IDENTIFY_serial.close()

    ###########################################################################
    # Stop port identify task (if running)
    ###########################################################################

    def _port_identify_stop(self):
        if self._identifying:
            self._port_identify()

    ###########################################################################
    # Updates the sample size of the plot (now not used)
    ###########################################################################

    def _port_changed(self):

        # If port changed, uncheck "Identify" button and close port (if open)
        self._port_identify_stop()

        # Program Position ID if asked
        if self.ControlsWin.ui1.cBox_Port.currentData() == "CMD_DEV_INFO":
            self._configure_device_info()
            return  # skip the rest of this method

        # Inform Upgrader class to perform version check on next run
        self.fwUpdater.checkAgain()

        # Set active port to update speeds in real-time
        self._selected_port = self.ControlsWin.ui1.cBox_Port.currentData()
        if self._selected_port == None:
            self._selected_port = ''  # Dissallow None
        if self._selected_port == "CMD_DEV_INFO":
            self._selected_port = ''  # Dissallow Action

        if self._selected_port.find(';') > 0:
            self._selected_port = self._selected_port.split(
                ';')[0]  # use COM port; never Ethernet
        device_list = FileStorage.DEV_get_device_list()
        usb_dev = []
        for i, dev_name in device_list:
            dev_info = FileStorage.DEV_info_get(i, dev_name)
            if 'USB' in dev_info and 'PORT' in dev_info:
                if dev_info['PORT'] == self._selected_port:
                    usb_dev.append([i, dev_info['USB']])
                    # keep going, search for dups (conflicts in config)
        if len(usb_dev) == 0:
            Log.d(
                f"No matching config found for port \"{self._selected_port.split(':')[0]}\". New device?")
        elif len(usb_dev) == 1:
            i, dev_name = usb_dev.pop()
            config_folder_name = dev_name if i == 0 else f"{i}_{dev_name}"
            Log.d(
                f"Matching config found: port \"{self._selected_port.split(':')[0]}\" is device \"{config_folder_name}\".")
            FileStorage.DEV_set_active(i, dev_name)
        # multiple dev infos contain the same port (conflicts should resolve later)
        else:
            Log.d(
                f"Multiple matching configs found for port \"{self._selected_port.split(':')[0]}\":")
            Log.d(f"Matching devices are: [{', '.join('_'.join(usb_dev))}]")
            # use root, trigger FW check conflict correction
            FileStorage.DEV_set_active(None, '')
            # Log.w(f"There is conflicting device info for port \"{self._selected_port}\"")
            # Log.w(f"Conflicts should automatically resolve once the port is FW checked.")
        self._refresh_speeds()

    def setMultiMode(self):
        try:
            self.multiplex_plots = max(
                1, min(4, 1 + self.ControlsWin.ui1.cBox_MultiMode.currentIndex()))
            self.PlotsWin.ui2.plt.clear()
            self.PlotsWin.ui2.pltB.clear()
            self.clear()  # erase any saved data shown on plots
            self._configure_plot()  # re-draw plots with new count
        except Exception as e:
            Log.e("ERROR: Unable to set count of multiplex plots.")
            Log.e("Details: " + str(e))

    ###########################################################################
    # Run user through the device information configuration prompts
    ###########################################################################

    def _configure_device_info(self):
        if not len(self._selected_port) == 0:
            # configure device info for selected device
            ok_name = self._configure_device_name()
            ok_pid, dif = self._configure_device_pid()

            self.tecWorker.set_port(self._selected_port)
            self.tecWorker._tec_update()  # Force read to update SW cached offsets
            self.set_cal1 = self.tecWorker._tec_offset1  # Store offsets to locals
            self.set_cal2 = self.tecWorker._tec_offset2

            ok_cal = self._configure_device_temp_cal_1()
            ok_cal = self._configure_device_temp_cal_2()

            # restore selected port back to device
            restore_port_idx = self.ControlsWin.ui1.cBox_Port.findData(
                self._selected_port)
            self.ControlsWin.ui1.cBox_Port.setCurrentIndex(restore_port_idx)

            if ok_pid:
                if dif != None:
                    try:
                        os.remove(dif)
                    except:
                        Log.e("Failed to delete file:", dif)
                # force parse and/or write device info (to update name and/or pid)
                self.fwUpdater.checkAgain()
                self.worker._port = self._selected_port  # used in run()
                self.fwUpdater.run(self)
            elif ok_name:  # (needed only if PID not changed too)
                self._refresh_ports()  # update name in port list
            # elif ok_cal: do nothing
        else:
            Log.w("NOTICE: Please select a device from the port list first!!")
            Log.w("This configuration action applies to the SELECTED device.")

    def _configure_device_name(self):
        friendly_name = self._selected_port
        dev_handle = None
        device_list = FileStorage.DEV_get_device_list()
        for i, dev_name in device_list:
            dev_info = FileStorage.DEV_info_get(i, dev_name)
            if 'NAME' in dev_info and 'PORT' in dev_info:
                if dev_info['PORT'] == self._selected_port:
                    friendly_name = dev_info['NAME']
                    dev_handle = dev_name
                    break

        text, ok = QtWidgets.QInputDialog.getText(self,
                                                  self.ControlsWin.ui1.cBox_Port.currentText(),
                                                  "Enter a name for device '{}':".format(
                                                      dev_handle),
                                                  text=friendly_name)
        if ok:
            # remove any invalid characters from user input
            invalidChars = "\\/:*?\"'<>|"
            for invalidChar in invalidChars:
                text = text.replace(invalidChar, '')
            text = text.strip().replace(' ', '_')  # word spaces -> underscores
            # limit length of input
            text = text[:12] if len(text) > 12 else text
            text = text.upper()  # make user input uppercase
            try:
                if text == '':
                    text = dev_info['USB']
                Log.i("Set on device '{}': NAME = {}".format(dev_handle, text))
                dev_file = os.path.join(
                    Constants.csv_calibration_export_path,
                    dev_handle,
                    "{}.{}".format(Constants.txt_device_info_filename,
                                   Constants.txt_extension))
                dev_lines = []
                with open(dev_file, 'r') as file:
                    dev_lines = file.readlines()
                    dev_lines[0] = "NAME: {}\n".format(text)
                with open(dev_file, 'w') as file:
                    file.writelines(dev_lines)
                Log.i("Program 'Name' operation was successful!")
            except:
                Log.e("Failed to update name entered by user.")
        else:
            Log.w("Program 'Name' operation aborted by user.")
        return ok

    def _configure_device_pid(self):
        friendly_name = self._selected_port
        dev_handle = None
        pid_old = 0xFF  # default: unassigned
        pid_new = 0xFF
        device_list = FileStorage.DEV_get_device_list()
        for i, dev_name in device_list:
            dev_info = FileStorage.DEV_info_get(i, dev_name)
            if 'NAME' in dev_info and 'PORT' in dev_info:
                if dev_info['PORT'] == self._selected_port:
                    friendly_name = dev_info['NAME']
                    dev_handle = dev_name
                    if 'PID' in dev_info:
                        pid_old = int(dev_info['PID'], base=16)
                    break

        # confirm PID in DEV_INFO matches COM Port listed text
        try:
            idx = self.ControlsWin.ui1.cBox_Port.findData(self._selected_port)
            if idx >= 0:
                device_text = self.ControlsWin.ui1.cBox_Port.itemText(idx)
                if ":" in device_text:
                    dev_i = int(device_text.split(":")[0], base=16)
                    if dev_i != pid_old:
                        Log.e(
                            f'Conflicting device info, using PID as {dev_i} instead of reported {pid_old}!')
                        pid_old = int(dev_i, base=16)
        except:
            Log.e("ERROR: Unable to check if PID in COM Port list matches DEV_INFO.")

        text, ok = QtWidgets.QInputDialog.getText(self,
                                                  self.ControlsWin.ui1.cBox_Port.currentText(),
                                                  "Enter a 'Position ID' for device '{}':".format(
                                                      friendly_name),
                                                  text=hex(pid_old)[2:].upper())
        if ok:
            try:
                pid_new = int(text, base=16)
                # valid values: 1-4, A-D
                if not pid_new in [0x1, 0x2, 0x3, 0x4, 0xA, 0xB, 0xC, 0xD]:
                    Log.w("Out-of-range PID entered by user. Using default: 0xFF")
                    pid_new = 0xFF
            except:
                Log.w("Non-numeric PID entered by user. Using default: 0xFF")
                pid_new = 0xFF
            Log.i("Set on device '{}': PID = {}".format(dev_handle, pid_new))
            if pid_new != pid_old:
                if self.setEEPROM(self._selected_port, 0, pid_new):
                    Log.i("Device EEPROM write PID success!")
                Log.i("Program 'Position ID' operation was successful!")
            else:
                Log.w("Program 'Position ID' operation resulted in no change!")
                ok = False
        else:
            Log.w("Program 'Position ID' operation aborted by user.")
        if ok:  # pid changed
            try:
                # Configure serial port (assume baud to check before update)
                _serial = serial.Serial()
                _serial.port = self._selected_port
                _serial.baudrate = Constants.serial_default_speed  # 115200
                _serial.stopbits = serial.STOPBITS_ONE
                _serial.bytesize = serial.EIGHTBITS
                _serial.timeout = Constants.serial_timeout_ms
                _serial.write_timeout = Constants.serial_writetimeout_ms
                _serial.open()
                _serial.write(b'MULTI INIT 0\n')
                _serial.close()
            except:
                Log.e("Unable to refresh LCD. PID error may be stale.")
            try:
                dev_name = dev_handle
                i_old = 0 if pid_old == 0xFF else pid_old
                dev_folder_old = "{}_{}".format(
                    i_old, dev_name) if i_old > 0 else dev_name
                dev_info_file_old = os.path.join(
                    Constants.csv_calibration_export_path, dev_folder_old, f"{Constants.txt_device_info_filename}.txt")
                if os.path.exists(dev_info_file_old):
                    Log.d(
                        f"Queueing removal of stale DEV_INFO file for {dev_name} with PID {pid_new}...")
                    return ok, dev_info_file_old
            except:
                Log.e("Unable to check for stale DEV_INFO file removal.")
        return ok, None

    def _configure_device_temp_cal_1(self):
        friendly_name = self._selected_port
        dev_handle = None
        device_list = FileStorage.DEV_get_device_list()
        for i, dev_name in device_list:
            dev_info = FileStorage.DEV_info_get(i, dev_name)
            if 'NAME' in dev_info and 'PORT' in dev_info:
                if dev_info['PORT'] == self._selected_port:
                    friendly_name = dev_info['NAME']
                    dev_handle = dev_name
                    break

        text, ok = QtWidgets.QInputDialog.getText(self,
                                                  self.ControlsWin.ui1.cBox_Port.currentText(),
                                                  "Enter a constant 'TEMP CAL' for device '{}':\n(Applies all the time)".format(
                                                      friendly_name),
                                                  text=self.set_cal1)
        if ok:
            try:
                cal_new = int(float(text) * 20.0)
                if cal_new < 0:
                    cal_new = (~(-cal_new)) & 0xFF
                if not cal_new in range(0, 0xFF):
                    Log.w("Out-of-range CAL1 entered by user. Using default: 0xFF")
                    cal_new = 0xFF
            except:
                Log.w("Non-numeric CAL1 entered by user. Using default: 0xFF")
                cal_new = 0xFF
            if cal_new == 0xFF:
                self.set_cal1 = "0"
            else:
                self.set_cal1 = text
            Log.i("Set on device '{}': CAL1 = {} ({}C)".format(
                dev_handle, cal_new, text))
            if self.setEEPROM(self._selected_port, 1, cal_new):
                Log.i("Device EEPROM write CAL1 success!")
            Log.i("Program 'TEMP CAL1' operation was successful!")
        else:
            Log.w("Program 'TEMP CAL1' operation aborted by user.")
        return ok

    def _configure_device_temp_cal_2(self):
        friendly_name = self._selected_port
        dev_handle = None
        device_list = FileStorage.DEV_get_device_list()
        for i, dev_name in device_list:
            dev_info = FileStorage.DEV_info_get(i, dev_name)
            if 'NAME' in dev_info and 'PORT' in dev_info:
                if dev_info['PORT'] == self._selected_port:
                    friendly_name = dev_info['NAME']
                    dev_handle = dev_name
                    break

        text, ok = QtWidgets.QInputDialog.getText(self,
                                                  self.ControlsWin.ui1.cBox_Port.currentText(),
                                                  "Enter a running 'TEMP CAL' for device '{}':\n(Applies during measurements ONLY)".format(
                                                      friendly_name),
                                                  text=self.set_cal2)
        if ok:
            try:
                cal_new = int(float(text) * 20.0)
                if cal_new < 0:
                    cal_new = (~(-cal_new)) & 0xFF
                if not cal_new in range(0, 0xFF):
                    Log.w("Out-of-range CAL2 entered by user. Using default: 0xFF")
                    cal_new = 0xFF
            except:
                Log.w("Non-numeric CAL2 entered by user. Using default: 0xFF")
                cal_new = 0xFF
            if cal_new == 0xFF:
                self.set_cal2 = "0"
            else:
                self.set_cal2 = text
            Log.i("Set on device '{}': CAL2 = {} ({}C)".format(
                dev_handle, cal_new, text))
            if self.setEEPROM(self._selected_port, 3, cal_new):
                Log.i("Device EEPROM write CAL2 success!")
            Log.i("Program 'TEMP CAL2' operation was successful!")
        else:
            Log.w("Program 'TEMP CAL2' operation aborted by user.")
        return ok

    ###########################################################################
    # Updates the EEPROM on port at address with value
    ###########################################################################

    def setEEPROM(self, port, address, value):
        success = False

        # Attempt to open port and print errors (if any)
        EEPROM_serial = serial.Serial()
        try:
            # Configure serial port (assume baud to check before update)
            EEPROM_serial.port = port
            EEPROM_serial.baudrate = Constants.serial_default_speed  # 115200
            EEPROM_serial.stopbits = serial.STOPBITS_ONE
            EEPROM_serial.bytesize = serial.EIGHTBITS
            EEPROM_serial.timeout = Constants.serial_timeout_ms
            EEPROM_serial.write_timeout = Constants.serial_writetimeout_ms
            EEPROM_serial.open()

            # Write EEPROM address with value on the device
            EEPROM_serial.write(
                "EEPROM {} {}\n".format(address, value).encode())
            timeoutAt = time() + 3
            temp_reply = ""
            lines_in_reply = 2
            # timeout needed if old FW
            while temp_reply.count('\n') < lines_in_reply and time() < timeoutAt:
                # timeout needed if old FW:
                while EEPROM_serial.in_waiting == 0 and time() < timeoutAt:
                    pass
                temp_reply += EEPROM_serial.read(
                    EEPROM_serial.in_waiting).decode()

            if time() < timeoutAt:
                success = True
            else:
                Log.e(TAG, "ERROR: Timeout during program of EEPROM controller.")
        except:
            Log.e(TAG, "ERROR: Failure reading port to program EEPROM controller.")
        finally:
            if EEPROM_serial.is_open:
                EEPROM_serial.close()
        return success

    ###########################################################################
    # Updates the sample size of the plot (now not used)
    ###########################################################################
    def _update_sample_size(self):

        # This function is connected to the valueChanged signal of the sample Spin Box.
        if self.worker is not None:
            # Log.i(TAG, "Changing sample size")
            self.worker.reset_buffers(
                self.ControlsWin.ui1.sBox_Samples.value() + 1)

    ###########################################################################
    # Updates and redraws the graphics in the plot.
    ###########################################################################

    def _update_plot(self):

        # This function is connected to the timeout signal of a QTimer
        self.worker.consume_logger()
        self.worker.consume_queue0()
        self.worker.consume_queue1()
        self.worker.consume_queue2()
        self.worker.consume_queue3()
        self.worker.consume_queue4()
        self.worker.consume_queue5()
        self.worker.consume_queue6()

        labelbar = "Status: Unknown"

        # MEASUREMENT: dynamic frequency and dissipation labels at run-time
        ###################################################################
        if self._get_source() == OperationType.measurement:
            self._readFREQ = self.worker.get_value0_buffer(0)
            vector1 = self.worker.get_d1_buffer(0)
            vector2 = self.worker.get_d2_buffer(0)
            vectortemp = self.worker.get_d3_buffer(0)
            vectoramb = self.worker.get_d4_buffer(0)
            self._ser_error1, self._ser_error2, self._ser_error3, self._ser_error4, self._ser_control, self._ser_err_usb = self.worker.get_ser_error()

            if self._ser_err_usb > 0:
                if self.worker.is_running():
                    Log.i(
                        TAG, "Port closed without stopping the capture, application will stop...")
                    self.stop()
                PopUp.warning(self, Constants.app_title,
                              "Warning: Disconnected Device!")

            if vectortemp.any():
                # TEMPERATURE: Update TEC temperature and power (if running)
                ############################################################
                if self.tecWorker._tec_locked:
                    self.tecWorker._tec_temp = vectortemp[0]
                    self.tecWorker._tec_power = self.worker.get_value2_buffer(
                        0)
                    sp = self.tecWorker._tec_setpoint
                    pv = self.tecWorker._tec_temp
                    op = self.tecWorker._tec_power
                    if sp == 0.00:
                        sp = 0.25
                    if op == 0:
                        new_l1 = "[AUTO-OFF ERROR]" if np.isnan(
                            pv) else "[AUTO-OFF TIMEOUT]"
                    else:
                        new_l1 = "PV:{0:2.2f}C SP:{1:2.2f}C OP:{2:+04.0f}".format(
                            pv, sp, op)
                    self.ControlsWin.ui1.lTemp.setText(new_l1)
                    bgcolor = "yellow"
                    if op == 0:
                        bgcolor = "red" if np.isnan(pv) else "yellow"
                    elif abs(pv - sp) <= 1.0:
                        bgcolor = "lightgreen"
                    self.ControlsWin.ui1.lTemp.setStyleSheet(
                        "background-color: {}".format(bgcolor))
                    self.ControlsWin.ui1.lTemp.repaint()

            if vector1.any():
                # progressbar
                if self._ser_control <= Constants.environment:
                    self._completed = self._ser_control*2

                if vector1[0] == 0 and not self._ser_error1 and not self._ser_error2:
                    label1 = 'processing...'
                    label2 = 'processing...'
                    label3 = 'processing...'
                    labelstatus = 'Processing'
                    self.ControlsWin.ui1.infostatus.setStyleSheet(
                        'background: #ffff00; padding: 1px; border: 1px solid #cccccc')  # ff8000
                    color_err = '#333333'
                    labelbar = 'Please wait, processing early data...'

                elif (vector1[0] == 0 and (self._ser_error1 or self._ser_error2)):
                    if self._ser_error1 and self._ser_error2:
                        label1 = ""
                        label2 = ""
                        label3 = ""
                        labelstatus = 'Warning'
                        color_err = '#ff0000'
                        labelbar = 'Warning: unable to apply half-power bandwidth method, lower and upper cut-off frequency not found'
                        self.ControlsWin.ui1.infostatus.setStyleSheet(
                            'background: #ff0000; padding: 1px; border: 1px solid #cccccc')
                    elif self._ser_error1:
                        label1 = ""
                        label2 = ""
                        label3 = ""
                        labelstatus = 'Warning'
                        color_err = '#ff0000'
                        labelbar = 'Warning: unable to apply half-power bandwidth method, lower cut-off frequency (left side) not found'
                        self.ControlsWin.ui1.infostatus.setStyleSheet(
                            'background: #ff0000; padding: 1px; border: 1px solid #cccccc')
                    elif self._ser_error2:
                        label1 = ""
                        label2 = ""
                        label3 = ""
                        labelstatus = 'Warning'
                        color_err = '#ff0000'
                        labelbar = 'Warning: unable to apply half-power bandwidth method, upper cut-off frequency (right side) not found'
                        self.ControlsWin.ui1.infostatus.setStyleSheet(
                            'background: #ff0000; padding: 1px; border: 1px solid #cccccc')
                else:
                    if not self._ser_error1 and not self._ser_error2 and not self._ser_error3 and not self._ser_error4:
                        if not self._reference_flag:
                            d1 = float("{0:.2f}".format(vector1[0]))
                            d2 = float("{0:.4f}".format(vector2[0]*1e6))
                            d3 = float("{0:.2f}".format(vectortemp[0]))
                        else:
                            a1 = vector1[0]-self._reference_value_frequency[0]
                            a2 = vector2[0] - \
                                self._reference_value_dissipation[0]
                            d1 = float("{0:.2f}".format(a1))
                            d2 = float("{0:.4f}".format(a2*1e6))
                            d3 = float("{0:.2f}".format(vectortemp[0]))
                        if not vector2[0] in (Constants.max_dissipation_1st_mode,
                                              Constants.max_dissipation_3rd_mode,
                                              Constants.max_dissipation_5th_mode):
                            label1 = str(d1) + " Hz"
                            label2 = str(d2) + "e-06"
                            label3 = str(d3) + " °C"
                            labelstatus = 'Monitoring'
                            color_err = '#008000'
                            self.ControlsWin.ui1.infostatus.setStyleSheet(
                                'background: #00ff80; padding: 1px; border: 1px solid #cccccc')

                            if not all(self._drop_applied):
                                for i, p in enumerate(self._plt2_arr):
                                    if p == None:
                                        self._drop_applied[i] = True
                                        self._run_finished[i] = True
                                        continue
                                    try:
                                        vector0 = self.worker.get_t2_buffer(i)
                                        time_running = vector0[0]
                                        if time_running == 0:
                                            labelbar = 'Waiting for start...'
                                            continue
                                        if time_running < 3.0:
                                            labelbar = 'Capturing data... Calibrating baselines for first 3 seconds... please wait...'
                                            # next(x for x,y in list(vector0) if y <= 1.0)
                                            idx = int(len(list(vector0)) / 3)
                                            if idx > 0:
                                                self._baseline_freq_avg = np.average(
                                                    vector1[:-idx])
                                                self._baseline_freq_noise = np.amax(
                                                    vector1[:-idx]) - np.amin(vector1[:-idx])
                                                self._baseline_diss_avg = np.average(
                                                    vector2[:-idx])
                                                self._baseline_diss_noise = np.amax(
                                                    vector2[:-idx]) - np.amin(vector2[:-idx])
                                        else:
                                            labelbar = 'Capturing data... Apply drop when ready...'
                                            if (abs(vector1[0] - self._baseline_freq_avg) > 10 * self._baseline_freq_noise
                                                    and abs(vector2[0] - self._baseline_diss_avg) > 10 * self._baseline_diss_noise):
                                                self._drop_applied[i] = True
                                    except:
                                        Log.e(
                                            "Error 'calibrating baselines' for drop detection. Apply drop when ready.")
                                        self._drop_applied[i] = True
                            else:
                                labelbar = 'Capturing data... Drop applied! Wait for exit... Press "Stop" when run is finished.'

                        else:
                            label1 = str(d1) + " Hz"
                            label2 = "-"
                            label3 = str(d3) + " °C"
                            labelstatus = 'Warning'
                            color_err = '#ff8000'
                            labelbar = ('Warning: sensor dissipation calculation is not considered accurate above ' +
                                        str("{0:.0f}e-06".format(vector2[0]*1e6)) + ' for the this mode')
                            self.ControlsWin.ui1.infostatus.setStyleSheet(
                                'background: #00ff80; padding: 1px; border: 1px solid #cccccc')
                    else:
                        if self._ser_error1 and self._ser_error2:
                            label1 = "-"
                            label2 = "-"
                            label3 = "-"
                            labelstatus = 'Warning'
                            color_err = '#ff0000'
                            labelbar = 'Warning: unable to apply half-power bandwidth method, lower and upper cut-off frequency not found'
                            self.ControlsWin.ui1.infostatus.setStyleSheet(
                                'background: #ff0000; padding: 1px; border: 1px solid #cccccc')
                        elif self._ser_error1:
                            label1 = "-"
                            label2 = "-"
                            label3 = "-"
                            labelstatus = 'Warning'
                            color_err = '#ff0000'
                            peak_name = " "
                            if self._ser_error1 == 1:
                                peak_name = " upper "
                            if self._ser_error1 == 2:
                                peak_name = " lower "
                            labelbar = 'Warning: unable to track the{}peak frequency, it has drifted more than maximum drift (left) allows'.format(
                                peak_name)
                            self.ControlsWin.ui1.infostatus.setStyleSheet(
                                'background: #ff0000; padding: 1px; border: 1px solid #cccccc')
                        elif self._ser_error2:
                            label1 = "-"
                            label2 = "-"
                            label3 = "-"
                            labelstatus = 'Warning'
                            color_err = '#ff0000'
                            peak_name = " "
                            if self._ser_error1 == 1:
                                peak_name = " upper "
                            if self._ser_error1 == 2:
                                peak_name = " lower "
                            labelbar = 'Warning: unable to track the{}peak frequency, it has drifted more than maximum drift (right) allows'.format(
                                peak_name)
                            self.ControlsWin.ui1.infostatus.setStyleSheet(
                                'background: #ff0000; padding: 1px; border: 1px solid #cccccc')
                        elif self._ser_error3:
                            label1 = "-"
                            label2 = "-"
                            label3 = "-"
                            labelstatus = 'Warning'
                            color_err = '#ff0000'
                            labelbar = 'Warning: failed to construct sine wave, approximating reconstruction with derivatives only'
                            self.ControlsWin.ui1.infostatus.setStyleSheet(
                                'background: #ff0000; padding: 1px; border: 1px solid #cccccc')
                        elif self._ser_error4:
                            label1 = "-"
                            label2 = "-"
                            label3 = "-"
                            labelstatus = 'Warning'
                            color_err = '#ff0000'
                            labelbar = 'Warning: failed to reconstruct signal wave, unable to apply noise correction reconstruction'
                            self.ControlsWin.ui1.infostatus.setStyleSheet(
                                'background: #ff0000; padding: 1px; border: 1px solid #cccccc')

                label8 = str(
                    int(self._readFREQ[self._readFREQ.size // 2]))+" Hz"
                self.InfoWin.ui3.info6.setText(
                    "<font color=#0000ff > Frequency Value </font>" + label8)
                label7 = str(int(self._readFREQ[0]))+" Hz"
                self.InfoWin.ui3.info3.setText(
                    "<font color=#0000ff > Start Frequency </font>" + label7)
                label6 = str(int(self._readFREQ[-1]))+" Hz"
                self.InfoWin.ui3.info4.setText(
                    "<font color=#0000ff > Stop Frequency </font>" + label6)
                label5 = str(int(self._readFREQ[-1]-self._readFREQ[0]))+" Hz"
                self.InfoWin.ui3.info4a.setText(
                    "<font color=#0000ff > Frequency Range </font>" + label5)
                label4 = str(int(self._readFREQ[1]-self._readFREQ[0]))+" Hz"
                self.InfoWin.ui3.info5.setText(
                    "<font color=#0000ff > Sample Rate </font>" + label4)

                self.InfoWin.ui3.l6a.setText(
                    "<font color=#0000ff > Temperature </font>" + label3)
                self.InfoWin.ui3.l6.setText(
                    "<font color=#0000ff > Dissipation </font>" + label2)
                self.InfoWin.ui3.l7.setText(
                    "<font color=#0000ff > Resonance Frequency </font>" + label1)
                self.ControlsWin.ui1.infostatus.setText(
                    "<font color=#333333 > Program Status </font>" + labelstatus)
                self.ControlsWin.ui1.infobar.setText(
                    "<font color=#0000ff> Infobar </font><font color={}>{}</font>".format(color_err, labelbar))
                # progressbar
                self.ControlsWin.ui1.progressBar.setValue(
                    int((self._ser_control / 10) % 100))

        # CALIBRATION: dynamic info in infobar at run-time
        ##################################################
        elif self._get_source() == OperationType.calibration:
            # flag for terminating calibration
            stop_flag = 0
            self.ControlsWin.ui1.pButton_Stop.setEnabled(False)
            vector1 = self.worker.get_value1_buffer(0)
            vector1_any = 1 if vector1.any() else 0
            # vector2[0] and vector3[0] flag error
            vector2 = self.worker.get_t3_buffer(0)
            vector3 = self.worker.get_d3_buffer(0)

            label1 = 'not available'
            label2 = 'not available'
            label3 = 'not available'
            labelstatus = 'Calibration Processing'
            color_err = '#333333'
            labelbar = 'The operation will take a few seconds to complete... please wait...'
            self.ControlsWin.ui1.infostatus.setStyleSheet(
                'background: #ffff00; padding: 1px; border: 1px solid #cccccc')

            # progressbar
            error1, error2, error3, on_sample, _, _ = self.worker.get_ser_error()
            if on_sample < Constants.calibration_default_samples:
                self._completed = (
                    on_sample / Constants.calibration_default_samples)*100
            ###########################################
            # calibration buffer empty
            if vector1_any == 0 and vector3[0] == 1:
                label1 = 'not available'
                label2 = 'not available'
                label3 = 'not available'
                color_err = '#ff0000'
                labelstatus = 'Calibration Warning'
                self.ControlsWin.ui1.infostatus.setStyleSheet(
                    'background: #ff0000; padding: 1px; border: 1px solid #cccccc')
                labelbar = 'Initialize Warning: empty buffer! Please, repeat the Initialize after disconnecting/reconnecting Device!'
                stop_flag = 1
            # calibration buffer empty and ValueError from the serial port
            elif vector1_any == 0 and vector2[0] == 1:
                label1 = 'not available'
                label2 = 'not available'
                label3 = 'not available'
                color_err = '#ff0000'
                labelstatus = 'Calibration Warning'
                self.ControlsWin.ui1.infostatus.setStyleSheet(
                    'background: #ff0000; padding: 1px; border: 1px solid #cccccc')
                labelbar = 'Initialize Warning: empty buffer/ValueError! Please, repeat the Initialize after disconnecting/reconnecting Device!'
                stop_flag = 1
            # calibration buffer not empty
            elif vector1_any != 0:
                label1 = 'not available'
                label2 = 'not available'
                label3 = 'not available'
                labelstatus = 'Calibration Processing'
                color_err = '#333333'
                labelbar = 'The operation will take a few seconds to complete... please wait...'
                if vector2[0] == 0 and vector3[0] == 0:
                    labelstatus = 'Calibration Success'
                    self.ControlsWin.ui1.infostatus.setStyleSheet(
                        'background: #00ff80; padding: 1px; border: 1px solid #cccccc')
                    color_err = '#008000'
                    labelbar = 'Initialize Success for baseline correction! Ready to measure. Press "Start" then apply drop.'
                    stop_flag = 1

                elif vector2[0] == 1 or vector3[0] == 1:
                    color_err = '#ff0000'
                    labelstatus = 'Calibration Warning'
                    self.ControlsWin.ui1.infostatus.setStyleSheet(
                        'background: #ff0000; padding: 1px; border: 1px solid #cccccc')
                    if vector2[0] == 1:
                        labelbar = 'Initialize Warning: ValueError or generic error during signal acquisition. Please, repeat the Initialize'
                        stop_flag = 1
                    elif vector3[0] == 1:
                        labelbar = 'Initialize Warning: unable to identify fundamental peak or apply peak detection algorithm. Please, repeat the Initialize!'
                        stop_flag = 1

            # progressbar -------------
            self.ControlsWin.ui1.progressBar.setValue(
                0 if stop_flag else int(self._completed+1))  # dwight ver const was 10
            self.InfoWin.ui3.l6a.setText(
                "<font color=#0000ff>  Temperature </font>" + label3)
            self.InfoWin.ui3.l6.setText(
                "<font color=#0000ff>  Dissipation </font>" + label2)
            self.InfoWin.ui3.l7.setText(
                "<font color=#0000ff>  Resonance Frequency </font>" + label1)
            self.ControlsWin.ui1.infostatus.setText(
                "<font color=#333333> Program Status </font>" + labelstatus)
            self.ControlsWin.ui1.infobar.setText(
                "<font color=#0000ff> Infobar </font><font color={}>{}</font>".format(color_err, labelbar))

            # terminate the  calibration (simulate clicked stop)
            if stop_flag == 1:
                self.stop_flag += 1
                if isinstance(self.worker._port, str) or self.stop_flag >= len(self.worker._port):
                    self._timer_plot.stop()
                    self._enable_ui(True)
                    self.worker.stop()

        ############################################################################################################################
        # REFERENCE SET
        ############################################################################################################################
        if self._reference_flag:
            for i, p in enumerate(self._plt2_arr):
                if p == None:
                    continue
                p.setLabel('left', 'Resonance Frequency', units='Hz',
                           color=Constants.plot_colors[6], **{'font-size': '10pt'})
                p.setLabel('right', 'Dissipation', units='',
                           color=Constants.plot_colors[7], **{'font-size': '10pt'})
                # save back to global, otherwise local gets trashed
                self._plt2_arr[i] = p
            self.InfoWin.ui3.inforef1.setText(
                "<font color=#0000ff > Ref. Frequency </font>" + self._labelref1)
            self.InfoWin.ui3.inforef2.setText(
                "<font color=#0000ff > Ref. Dissipation </font>" + self._labelref2)

            ###################################################################
            # Amplitude and phase multiple Plot
            def updateViews1():
                for i, p in enumerate(self._plt0_arr):
                    if p == None:
                        continue
                    p.clear()
                    self._plt0_arr[i] = p
                if not self._plt1 is None:
                    Log.w("Phase is no longer supported with multiplex software.")
                    return
                    self._plt1.clear()
                    self._plt1.setGeometry(self._plt0.vb.sceneBoundingRect())
                    self._plt1.linkedViewChanged(
                        self._plt0.vb, self._plt1.XAxis)

            # updates for multiple plot y-axes
            updateViews1()
            for i, p in enumerate(self._plt0_arr):
                if p == None:
                    continue
                # p.vb.sigResized.connect(updateViews1)
                p.setLimits(yMax=None, yMin=None,
                            minYRange=None, maxYRange=None)
                p.enableAutoRange(axis='x', enable=True)
                p.enableAutoRange(axis='y', enable=True)
                p.plot(x=self.worker.get_value0_buffer(
                    i), y=self.worker.get_value1_buffer(i), pen=Constants.plot_colors[0])
                # save back to global, otherwise local gets trashed
                self._plt0_arr[i] = p
            if not self._plt1 is None:
                Log.w("Phase is no longer supported with multiplex software.")
                # self._plt1.addItem(pg.PlotCurveItem(x=self._readFREQ,y=self.worker.get_value2_buffer(i),pen=Constants.plot_colors[1]))

            ###################################################################
            # Resonance frequency and dissipation multiple Plot
            def updateViews2():
                for i, p in enumerate(self._plt3_arr):
                    if p == None:
                        continue
                    _p2 = self._plt2_arr[i]
                    _p2.clear()
                    p.clear()
                    p.setGeometry(_p2.vb.sceneBoundingRect())
                    p.linkedViewChanged(_p2.vb, p.XAxis)
                    # save back to global, otherwise local gets trashed
                    self._plt2_arr[i] = _p2
                    # save back to global, otherwise local gets trashed
                    self._plt3_arr[i] = p

            # updates for multiple plot y-axes
            updateViews2()

            for i, p in enumerate(self._plt2_arr):
                if p == None:
                    continue
                # p.vb.sigResized.connect(updateViews2)
                self._vector_1 = np.array(self.worker.get_d1_buffer(
                    i))-self._reference_value_frequency[i]
                p.plot(x=self.worker.get_t1_buffer(i), y=self._vector_1,
                       pen=Constants.plot_colors[6])  # 2

                # Prevent the user from zooming/panning out of this specified region
                if self._get_source() == OperationType.measurement:
                    p.setLimits(
                        yMax=self._vector_reference_frequency[i][-1], yMin=self._vector_reference_frequency[i][0], minYRange=5)
                    p.enableAutoRange(axis='x', enable=True)
                    p.enableAutoRange(axis='y', enable=True)

                # save back to global, otherwise local gets trashed
                self._plt2_arr[i] = p

            for i, p in enumerate(self._plt3_arr):
                if p == None:
                    continue
                self._vector_2 = np.array(self.worker.get_d2_buffer(
                    i))-self._reference_value_dissipation[i]
                p.addItem(pg.PlotCurveItem(self.worker.get_t2_buffer(
                    i), self._vector_2, pen=Constants.plot_colors[7]))

                # Prevent the user from zooming/panning out of this specified region
                if self._get_source() == OperationType.measurement:
                    p.setLimits(
                        yMax=self._vector_reference_dissipation[i][-1], yMin=self._vector_reference_dissipation[i][0], minYRange=1e-7)
                    p.enableAutoRange(axis='x', enable=True)
                    p.enableAutoRange(axis='y', enable=True)

                # save back to global, otherwise local gets trashed
                self._plt3_arr[i] = p

            # Prevent the user from zooming/panning out of this specified region
            if self._get_source() == OperationType.measurement:
                self._plt4.setLimits(yMax=50, yMin=-10)
                self._plt4.enableAutoRange(axis='x', enable=True)
                self._plt4.enableAutoRange(axis='y', enable=True)

            ###################################################################
            # Temperature plot
            self._plt4.clear()
            self._plt4.plot(x=self.worker.get_t3_buffer(
                i), y=self.worker.get_d3_buffer(i), pen=Constants.plot_colors[4])
            # self._plt4.addItem(pg.PlotCurveItem(self.worker.get_t3_buffer(i),self.worker.get_d4_buffer(i),pen=Constants.plot_colors[1])) # hide ambient

        ###########################################################################################################################
        # REFERENCE NOT SET
        ###########################################################################################################################
        else:

            for i, p in enumerate(self._plt2_arr):
                if p == None:
                    continue
                p.setLabel('left', 'Resonance Frequency', units='Hz',
                           color=Constants.plot_colors[2], **{'font-size': '10pt'})
                p.setLabel('right', 'Dissipation', units='',
                           color=Constants.plot_colors[3], **{'font-size': '10pt'})
                self._plt2_arr[i] = p
            self.InfoWin.ui3.inforef1.setText(
                "<font color=#0000ff > Ref. Frequency </font>" + self._labelref1)
            self.InfoWin.ui3.inforef2.setText(
                "<font color=#0000ff > Ref. Dissipation </font>" + self._labelref2)

            # limit number of display points to last num
            numPoints = 12000

            ###################################################################
            # Amplitude and phase multiple Plot
            def updateViews1():
                for i, p in enumerate(self._plt0_arr):
                    if p == None:
                        continue
                    p.clear()
                    self._plt0_arr[i] = p
                if not self._plt1 is None:
                    Log.w("Phase is no longer supported with multiplex software.")
                    return
                    self._plt1.clear()
                    self._plt1.setGeometry(self._plt0.vb.sceneBoundingRect())
                    self._plt1.linkedViewChanged(
                        self._plt0.vb, self._plt1.XAxis)
            # updates for multiple plot y-axes
            updateViews1()
            self._amps = {}  # empty dict for showing combined amplitudes
            for i, p in enumerate(self._plt0_arr):
                if p == None:
                    continue
                _plt2 = self._plt2_arr[i]
                _plt3 = self._plt3_arr[i]
                self._readFREQ = self.worker.get_value0_buffer(i)
                vector1 = self.worker.get_d1_buffer(i)

                if self._get_source() == OperationType.measurement and self.multiplex_plots > 1:
                    self._amps[i] = self.worker.get_value1_buffer(i)
                else:
                    # p.vb.sigResized.connect(updateViews1)
                    p.setLimits(yMax=None, yMin=None,
                                minYRange=None, maxYRange=None)
                    p.enableAutoRange(axis='x', enable=True)
                    p.enableAutoRange(axis='y', enable=True)
                    p.plot(x=self._readFREQ, y=self.worker.get_value1_buffer(
                        i), pen=Constants.plot_colors[0])
                if not self._plt1 is None:
                    Log.w("Phase is no longer supported with multiplex software.")
                    # self._plt1.addItem(pg.PlotCurveItem(x=self._readFREQ,y=self.worker.get_value2_buffer(i),pen=Constants.plot_colors[1]))

                if self._get_source() == OperationType.measurement:
                    ###################################################################
                    # Resonance frequency and dissipation multiple Plot
                    def updateViews2(_plt2, _plt3):
                        _plt2.clear()
                        _plt3.clear()
                        _plt3.setGeometry(_plt2.vb.sceneBoundingRect())
                        _plt3.linkedViewChanged(_plt2.vb, _plt3.XAxis)
                    # updates for multiple plot y-axes
                    updateViews2(_plt2, _plt3)
                    # _plt2.vb.sigResized.connect(updateViews2)
                    _plt2.plot(x=self.worker.get_t1_buffer(i)[:numPoints], y=self.worker.get_d1_buffer(
                        i)[:numPoints], pen=Constants.plot_colors[2])

                    # Add apply drop message to those still pending drop
                    try:
                        if self._text4[i] == None:
                            # 'size' and 'bold' retained when calling 'setText()'
                            self._text4[i] = pg.LabelItem(
                                size='11pt', bold=True)
                            self._text4[i].setParentItem(_plt2.graphicsItem())
                            self._text4[i].anchor(
                                itemPos=(0.5, 0.25), parentPos=(0.5, 0.25))
                        elif not self._drop_applied[i]:
                            vector0 = self.worker.get_t2_buffer(i)
                            time_running = vector0[0]
                            if time_running == 0:
                                labelbar = 'Waiting for start...'
                                continue
                            if time_running < 3.0:
                                self._text4[i].setText(
                                    'Calibrating...', color=(0, 0, 200))
                                self._baselinedata[i] = [
                                    [np.amin(self.worker.get_d1_buffer(i)[:numPoints]), np.amax(
                                        self.worker.get_d1_buffer(i)[:numPoints])],
                                    [np.amin(self.worker.get_d2_buffer(i)[:numPoints]), np.amax(self.worker.get_d2_buffer(i)[:numPoints])]]
                            else:
                                self._text4[i].setText(
                                    'Apply drop now!', color=(0, 200, 0))
                        else:
                            time_running = _plt2.getViewBox().viewRange()[0][1]
                            current_y_range = [_plt2.getViewBox().viewRange()[
                                1], _plt3.viewRange()[1]]
                            current_deltas = np.diff(current_y_range)[:, 0]
                            last_y_deltas = np.diff(
                                self._last_y_range[i])[:, 0]
                            baseline_deltas = np.diff(
                                self._baselinedata[i])[:, 0]
                            # self._last_y_range[i] != current_y_range:
                            if any(np.subtract(current_deltas, last_y_deltas) > baseline_deltas):
                                self._last_y_range[i] = current_y_range
                                # time of last delta
                                self._last_y_delta[i] = time_running
                                # hide message once drop is applied
                                self._text4[i].setText(' ')
                                self._run_finished[i] = False
                            elif time_running - self._last_y_delta[i] > 10.0 and not self._run_finished[i]:
                                self._run_finished[i] = True
                                Log.d(
                                    f"The dataset on Plot {i} appears to have finished (Y-Axis stablized for 10 seconds)")
                                # TODO 2024-05-24: (Temporary) Disable showing "Press Stop" on UI until accuracy is improved
                                # if all(self._run_finished):
                                #     status_text = 'Press \"Stop\"'
                                # else:
                                #     status_text = "Run Complete, waiting on other channels..."
                                # self._text4[i].setText(status_text, color=(200, 0, 0)) # range unchanging / probably finished
                    except Exception as e:
                        Log.e("Error handling plot status label text!")
                        # Log.e(e)

                    # Prevent the user from zooming/panning out of this specified region
                    if self._get_source() == OperationType.measurement:
                        _ymin = np.amin(vector1)
                        _ymax = np.amax(vector1)
                        if _ymax - _ymin < Constants.plot_min_range_freq:
                            _yavg = int(np.average(vector1))
                            _ymin = _yavg - (Constants.plot_min_range_freq / 2)
                            _ymax = _yavg + (Constants.plot_min_range_freq / 2)

                        _plt2.setLimits(
                            yMax=_ymax, yMin=_ymin, minYRange=Constants.plot_min_range_freq, maxYRange=10000)
                        _plt3.setLimits(yMax=(
                            self._readFREQ[-1]-self._readFREQ[0])/self._readFREQ[0], yMin=0, minYRange=Constants.plot_min_range_diss)

                        _plt2.enableAutoRange(axis='x', enable=True)
                        _plt3.enableAutoRange(axis='x', enable=True)

                        _plt2.enableAutoRange(axis='y', enable=True)
                        _plt3.enableAutoRange(axis='y', enable=True)

                    _plt3.addItem(pg.PlotCurveItem(self.worker.get_t2_buffer(i)[
                                  :numPoints], self.worker.get_d2_buffer(i)[:numPoints], pen=Constants.plot_colors[3]))

                # save back to global, otherwise local gets trashed
                self._plt0_arr[i] = p
                # save back to global, otherwise local gets trashed
                self._plt2_arr[i] = _plt2
                # save back to global, otherwise local gets trashed
                self._plt3_arr[i] = _plt3

                # Prevent the user from zooming/panning out of this specified region
                if self._get_source() == OperationType.measurement:
                    self._plt4.setLimits(yMax=50, yMin=-10, minYRange=0.5)
                    self._plt4.enableAutoRange(axis='x', enable=True)
                    self._plt4.enableAutoRange(axis='y', enable=True)

            if self._get_source() == OperationType.measurement and self.multiplex_plots > 1:
                # get the largest values for each index
                i = 0  # first Amplitude plot is the main one
                combined_amps = self._amps[i]
                for a in range(1, len(self._amps)):
                    combined_amps = np.maximum(combined_amps, self._amps[a])
                p = self._plt0_arr[i]
                p.setLimits(yMax=None, yMin=None,
                            minYRange=None, maxYRange=None)
                p.enableAutoRange(axis='x', enable=True)
                p.enableAutoRange(axis='y', enable=True)
                p.plot(x=self._readFREQ, y=combined_amps,
                       pen=Constants.plot_colors[0])
                self.labels = []
                for j in range(len(self._amps)):
                    am = np.argmax(self._amps[j])
                    item = pg.TextItem(
                        text=str(j+1), color=(0, 0, 0), anchor=(0, 1))
                    item.setPos(self._readFREQ[am], combined_amps[am])
                    self.labels.append(item)
                    p.addItem(self.labels[j])
                # save back to global, otherwise local gets trashed
                self._plt0_arr[i] = p

            ###################################################################
            # Temperature plot
            self._plt4.clear()
            self._plt4.plot(x=self.worker.get_t3_buffer(0)[:numPoints], y=self.worker.get_d3_buffer(
                0)[:numPoints], pen=Constants.plot_colors[4])
            # self._plt4.addItem(pg.PlotCurveItem(self.worker.get_t3_buffer(i),self.worker.get_d4_buffer(i),pen=Constants.plot_colors[1])) # hide ambient

    ###########################################################################################################################################

    ###########################################################################
    # Updates the list of ports and reports findings
    ###########################################################################

    def _port_list_refresh(self):

        # Capture ports list before changing it
        before_count = self.ControlsWin.ui1.cBox_Port.count()
        before_items = [self.ControlsWin.ui1.cBox_Port.itemData(
            i) for i in range(before_count)]

        # Update ports list
        self._source_changed()

        # Get differences from before and after
        after_count = self.ControlsWin.ui1.cBox_Port.count()
        after_items = [self.ControlsWin.ui1.cBox_Port.itemData(
            i) for i in range(after_count)]
        differences = set(before_items) ^ set(after_items)

        # Compare before and after, report changes
        title = "Refresh COM Port List"

        added = differences & set(after_items)
        if len(added) > 0:  # port(s) added
            added_str = str(sorted(added)).translate(
                {ord(i): None for i in "[{',}]"})
            if len(added) == 1:
                # PopUp.question(self, title, "Found new port: {}\nWould you like to use it?".format(added_str), default=True)
                use_port = True
                Log.w(
                    f"Found new port: {[t.split(':')[0] for t in added_str.split()]}")
                if use_port:
                    i = self.ControlsWin.ui1.cBox_Port.findData(added_str)
                    self.ControlsWin.ui1.cBox_Port.setCurrentIndex(i)
                    self._port_changed()
            else:
                if len(list(added)[0]) > 6:
                    added_str = added_str.replace(" ", "\n")
                # PopUp.information(self, title, "{} port(s) added:\n{}".format(len(added), added_str))
                Log.w(
                    f"{len(added)} port(s) added: {[t.split(':')[0] for t in added_str.split()]}")

        removed = differences & set(before_items)
        if len(removed) > 0:  # port(s) removed
            removed_str = str(sorted(removed)).translate(
                {ord(i): None for i in "[{',}]"})
            if len(list(removed)[0]) > 6:
                removed_str = removed_str.replace(" ", "\n")
            # PopUp.information(self, title, "{} port(s) removed:\n{}".format(len(removed), removed_str))
            Log.w(
                f"{len(removed)} port(s) removed: {[t.split(':')[0] for t in removed_str.split()]}")

        if after_count == ADMIN_OPTION_CMDS and self._no_ports_found > 1:
            # no ports connected, ask if initial flash desired
            PopUp.warning(self, "Device Not Detected", "WARNING: No device port is detected. Please connect a device first and try again.\n\n" +
                                "If a device is connected, it is not enumerating a COM port. Please seek assistance from support.")
            # NOTE: The required "tools" for recovery are not included anymore. This is no longer possible.
            # PopUp.question(self, "Having Trouble?", "No working devices detected.\n\n" +
            do_flash = False
            #      "Would you like to restore functionality to an unresponsive device?")
            if do_flash:
                PopUp.warning(self, "Manual Firmware Update Process",
                                    "Once the \"PROGRAM\" button is pushed and the HW TYPE is known this tool will attempt to flash the firmware.\n\n" +
                                    "NOTICE: YOU MUST PUSH THE \"PROGRAM\" BUTTON ON THE DEVICE BEFORE PROCEEDING.\n\n" +
                                    "Next Step: You will be asked to provide the HW TYPE of the Teensy device.")
                self.fwUpdater.doUpdate(self, None)
        elif len(differences) == 0:  # no port list changes
            # PopUp.information(self, title, "No port changes detected.")
            if after_count == ADMIN_OPTION_CMDS:
                self._no_ports_found += 1
        else:
            self._no_ports_found = 0

    ###########################################################################
    # Updates the source and depending boxes on change
    ###########################################################################

    def _source_changed(self):

        # It is connected to the indexValueChanged signal of the Source ComboBox.
        try:
            Log.i(TAG, "Scanning the source: {}".format(
                Constants.app_sources[self._get_source().value]))
        except:
            Log.e(TAG, "Scanning the source: [UNKNOWN]")

        self._refresh_ports()
        self._refresh_speeds()

    ###########################################################################
    # Updates the ports and depending boxes on change
    ###########################################################################

    def _refresh_ports(self):

        selected_port = self.ControlsWin.ui1.cBox_Port.currentData()
        if selected_port == None:
            selected_port = ''  # Dissallow None
        if selected_port == "CMD_DEV_INFO":
            selected_port = ''  # Dissallow Action

        # Clears boxes before adding new
        self.ControlsWin.ui1.cBox_Port.clear()

        # Gets the current source type
        source = self._get_source()

        self.ControlsWin.ui1.infobar.setText(
            "<font color=#0000ff> Infobar </font><font color={}>{}</font>".format("#333333", "Searching for devices... please wait..."))
        self.ControlsWin.ui1.infobar.repaint()

        ports = self.worker.get_source_ports(source)

        self.ControlsWin.ui1.infobar.setText(
            "<font color=#0000ff> Infobar </font><font color={}>{}</font>".format("#333333", ""))

        # Check for device info and update port names accordingly
        port_names = list(ports)  # copy list
        device_list = FileStorage.DEV_get_device_list()
        device_ports = []
        dev_pids = []
        for i, dev_name in device_list:
            dev_info = FileStorage.DEV_info_get(i, dev_name)
            if 'IP' in dev_info and not dev_info['IP'] == "0.0.0.0":
                net_exists = Discovery().ping(dev_info['IP'])
            if 'NAME' in dev_info and 'PORT' in dev_info:
                try:
                    device_ports.append(dev_info['PORT'])
                    if not dev_info['PORT'] in port_names:
                        continue  # do not throw exception for debugging
                    i = port_names.index(dev_info['PORT'])
                    if 'PID' in dev_info and not dev_info['PID'] == "FF":
                        if not dev_info['PID'] in dev_pids:
                            dev_pids.append(dev_info['PID'])
                        # shorthand, for dropdown menu only (colon; no underscore)
                        port_names[i] = "{}:{}".format(
                            dev_info['PID'], dev_info['NAME'])
                    elif dev_info['NAME'] != dev_name:
                        port_names[i] = dev_info['NAME']
                    elif 'COM' in dev_info['PORT']:
                        port_names[i] = "{} ({})".format(
                            dev_info['NAME'], dev_info['PORT'])
                    elif ':' in dev_info['PORT']:
                        port_names[i] = dev_info['NAME']
                    else:
                        port_names[i] = "{} ({})".format(
                            dev_info['NAME'], "COM" + str((10 + i)))
                    if 'IP' in dev_info and not dev_info['IP'] == "0.0.0.0":
                        ports[i] += f";{dev_info['IP']}"
                        if ' (' in port_names[i]:
                            port_names[i] = "{} ({})".format(
                                port_names[i][0:port_names[i].index(' (')], dev_info['IP'])
                        else:
                            port_names[i] += f" ({dev_info['IP']})"
                        if not net_exists:
                            Log.e(
                                f"ERROR: Failed to ping device {dev_info['IP']}")
                except ValueError:
                    pass  # device not connected, ignore it
                except:
                    Log.w(TAG, "WARN: Error while generating port names list.")

        for port_name in list(ports):
            if not port_name in device_ports:
                # found connected port with no associated device info in config folder
                # force parse and/or write device info (to update name and/or pid)
                Log.d(
                    f"New device found: querying device info for {port_name.split(':')[0]}...")
                self.fwUpdater.checkAgain()
                self.worker._port = port_name  # used in run()
                # do NOT ask to update if not ReadyToShow
                ret = self.fwUpdater.run(self, self.ReadyToShow)
                if ret == True or ret >= 0:  # not a failed check
                    Log.d("Device info queried. Waiting to refresh ports on next call.")
                    return  # fwUpdater.run() calls _refresh_ports() when devinfo written, stop stop here
                    # NOTE: Each subsequent call to _refresh_ports() will parse one pending device info.

        for net_dev in Discovery().doDiscover(full_query=True):
            [build, version, date, hw, ip, mac, usb, uid] = net_dev
            found_new = True
            found_exist = False
            for port in ports:              # find if IP already in port list
                if ip in port:
                    found_new = False
                    break
            if found_new:
                for name in port_names:     # find if DEV already in port list
                    if usb in name:
                        found_exist = True
                        found_new = False
                        break
            if found_exist:
                for i in range(len(port_names)):    # add IP to existing DEV
                    if usb in port_names[i]:
                        ports[i] += f";{ip}"
            if found_new:                           # add new IP & DEV to list
                ports.append(ip)
                port_names.append("{} ({})".format(usb, ip))

        # usb and ethernet icons
        icon_path = os.path.join(Architecture.get_path(), 'QATCH/icons/')
        usb_icon = QtGui.QIcon(os.path.join(icon_path, 'usb-icon.png'))  # png
        ethernet_icon = QtGui.QIcon(os.path.join(
            icon_path, 'ethernet-icon.png'))  # png

        if ports is not None:
            for i in range(len(ports)):
                port_parts = port_names[i].split(' (')
                dev_name = port_parts[0]
                is_networked = port_parts[-1].count(".") == 3
                port_icon = ethernet_icon if is_networked else usb_icon
                self.ControlsWin.ui1.cBox_Port.addItem(
                    port_icon, dev_name, ports[i])
            self.ControlsWin.ui1.cBox_Port.model().sort(0)
            self.ControlsWin.ui1.cBox_Port.addItem(
                "⚙️  Configure...", "CMD_DEV_INFO")

        # Log.d(selected_port, ports)
        selected_port_parts = selected_port.split(';')
        common_port = ''
        for port in ports:
            port_parts = port.split(';')
            for p in port_parts:
                for sp in selected_port_parts:
                    # Log.d(f"compare {sp} to {p}")
                    if sp == p:
                        common_port = ";".join(port_parts)
                        break
                if common_port != '':
                    break
            if common_port != '':
                break
        if common_port != '':
            Log.d(f"found pre-selected port = {common_port.split(':')[0]}")
            selected_port = common_port

        if selected_port in ports:
            i = self.ControlsWin.ui1.cBox_Port.findData(selected_port)
            self.ControlsWin.ui1.cBox_Port.setCurrentIndex(i)

            # RESET PORT MSGBOX
            try:
                _serial = serial.Serial()
                _serial.port = selected_port
                _serial.baudrate = Constants.serial_default_speed  # 115200
                _serial.stopbits = serial.STOPBITS_ONE
                _serial.bytesize = serial.EIGHTBITS
                _serial.timeout = Constants.serial_timeout_ms
                _serial.write_timeout = Constants.serial_writetimeout_ms
                _serial.open()
                _serial.write(b'MSGBOX\n')
                _serial.close()
            except Exception as e:
                Log.e("ERROR: Unable to clear messagebox on device.")
                Log.e(e)

        else:
            self.ControlsWin.ui1.cBox_Port.setCurrentIndex(-1)
            if len(selected_port) > 0:
                Log.w(
                    f"The selected port ({selected_port.split(':')[0]}) is no longer available.")
                Log.w("Please check connection or select a new port.")
                # PopUp.warning(self, "Missing COM Port",
                #     "The selected port ({}) is no longer available.\n\n".format(selected_port) +
                #     "Please check connection or select a new port.")
            elif len(ports) == 1 or len(ports) == len(dev_pids):
                self.ControlsWin.ui1.cBox_Port.setCurrentIndex(0)
                self._port_changed()

        restore_idx = self.ControlsWin.ui1.cBox_MultiMode.currentIndex()
        self.ControlsWin.ui1.cBox_MultiMode.clear()
        multi_channel_count = 4*1
        if "A" in dev_pids:
            multi_channel_count = 4*6
        multi_channel_items = [
            f"{i+1} Channel" + ("s" if i > 0 else "") for i in range(multi_channel_count)]
        self.ControlsWin.ui1.cBox_MultiMode.addItems(multi_channel_items)
        if self.ControlsWin.ui1.chBox_MultiAuto.isChecked():
            idx = max(0, min(len(dev_pids), multi_channel_count) - 1)
            Log.d(f"Auto-Detect Channel Count: {idx + 1}")
        else:
            if self.ControlsWin.ui1.cBox_MultiMode.count() > restore_idx:
                idx = restore_idx
            else:
                Log.w("Too few channels to restore prior selection.")
                idx = self.ControlsWin.ui1.cBox_MultiMode.count() - 1
        self.ControlsWin.ui1.cBox_MultiMode.setCurrentIndex(idx)
        for i in range(self.ControlsWin.ui1.cBox_MultiMode.count()):
            if i < self.ControlsWin.ui1.cBox_Port.count() * (6 if "A" in dev_pids else 1) - 1:
                enable = True
            else:
                enable = False
            self.ControlsWin.ui1.cBox_MultiMode.model().item(i).setEnabled(enable)

    ###########################################################################
    # Updates the speeds and depending boxes on change
    ###########################################################################

    def _refresh_speeds(self):

        # Gets the current source type
        source = self._get_source()
        i = self.ControlsWin.ui1.cBox_Port.currentText()
        i = 0 if i.find(":") == -1 else int(i.split(":")[0], base=16) % 9
        speeds = self.worker.get_source_speeds(source, i)

        # Store and get the restore index
        if len(speeds) == self.ControlsWin.ui1.cBox_Speed.count():
            idx = source.value
        else:
            idx = (source.value + 1) % 2  # 0 -> 1, 1 -> 0
        if not self.ControlsWin.ui1.cBox_Speed.currentText() == "0":
            self.restore_mode_idx[idx] = self.ControlsWin.ui1.cBox_Speed.currentIndex(
            )
        # do not use 'idx' here
        selected_index = self.restore_mode_idx[source.value]

        # Set defaults if no restore index set
        if selected_index == -1:
            if source == OperationType.measurement:
                selected_index = 0 if len(speeds) == 1 else len(speeds) - 2
            else:
                selected_index = 0

        # Clears boxes before adding new
        self.ControlsWin.ui1.cBox_Speed.clear()

        if speeds is not None:
            self.ControlsWin.ui1.cBox_Speed.addItems(speeds)

        if selected_index < self.ControlsWin.ui1.cBox_Speed.count():
            self.ControlsWin.ui1.cBox_Speed.setCurrentIndex(selected_index)

    ###########################################################################
    # Gets the current source type
    ###########################################################################

    def _get_source(self):

        # :rtype: OperationType.
        return OperationType(self.ControlsWin.ui1.cBox_Source.currentIndex())

    def _get_cal_age(self):
        is_recent = False
        age_in_mins = -1
        try:
            if self.ControlsWin.ui1.cal_initialized:  # been initialized this session
                is_recent = True
                for i in range(len(self.worker._port)):
                    j = i
                    if isinstance(self.worker._port, list):
                        j += 1
                    else:
                        # TODO: doesn't work if connected PIDs: FF, 1, 2, 3, 4
                        j = self.ControlsWin.ui1.cBox_Port.currentIndex() + 1
                    # Check age of calibration file, and ask for new cal if older than 15 mins
                    cal_file_path = Constants.cvs_peakfrequencies_path
                    cal_file_path = FileStorage.DEV_populate_path(
                        cal_file_path, j)
                    timestamp = os.path.getmtime(
                        cal_file_path)  # may throw OSError
                    last_modified = datetime.datetime.fromtimestamp(timestamp)
                    last_cal_age = datetime.datetime.now() - last_modified
                    FIFTEEN_MINS = datetime.timedelta(minutes=15)
                    # mask in this device cal state to the top-level vars
                    is_recent &= last_cal_age < FIFTEEN_MINS
                    age_of_this = int(last_cal_age.total_seconds()/60)
                    if age_of_this > age_in_mins:
                        age_in_mins = age_of_this
                    if not isinstance(self.worker._port, list):
                        break
        except OSError as e:
            Log.w("No calibration file found. Please initialize again.")
        except Exception as e:
            Log.e("ERROR in _get_cal_age():", str(e))
        return is_recent, age_in_mins

    ###########################################################################
    # Cleans history plot
    ###########################################################################
    def clear(self):
        elems = self._plt0_arr + [self._plt1] + \
            self._plt2_arr + self._plt3_arr + [self._plt4]
        for e in elems:
            if e != None:
                e.clear()
                e.setLimits(yMin=None, yMax=None,
                            minYRange=None, maxYRange=None)
                e.setXRange(min=0, max=1)
                e.setYRange(min=0, max=1)
        if self._plt2_arr[0] != None:
            self._annotate_welcome_text()
        self.ControlsWin.ui1.progressBar.setValue(0)

    ###########################################################################
    # Reference set/reset
    ###########################################################################

    def reference(self):
        import numpy as np
        # import sys
        self._reference_value_frequency.clear()
        self._vector_reference_frequency.clear()
        self._reference_value_dissipation.clear()
        self._vector_reference_dissipation.clear()
        for i, p in enumerate(self._plt2_arr):
            if p == None:
                continue
            # _plt2 = self._plt2_arr[i]
            # _plt3 = self._plt3_arr[i]
            self._readFREQ = self.worker.get_value0_buffer(i)
            support = self.worker.get_d1_buffer(i)
            if support.any():
                if support[0] != 0:
                    ref_vector1 = [
                        c for c in self.worker.get_d1_buffer(i) if ~np.isnan(c)]
                    ref_vector2 = [
                        c for c in self.worker.get_d2_buffer(i) if ~np.isnan(c)]
                    self._reference_value_frequency.append(ref_vector1[0])
                    self._reference_value_dissipation.append(ref_vector2[0])
                    # sys.stdout.write("\033[K") #clear line
                    if self._reference_flag:
                        self._reference_flag = False
                        Log.i(TAG, "Reference reset!   ")
                        self._labelref1 = "not set"
                        self._labelref2 = "not set"
                    else:
                        self._reference_flag = True
                        d1 = float("{0:.2f}".format(
                            self._reference_value_frequency[i]))
                        d2 = float("{0:.4f}".format(
                            self._reference_value_dissipation[i]*1e6))
                        self._labelref1 = str(d1) + "Hz"
                        self._labelref2 = str(d2) + "e-06"
                        Log.i(TAG, "Reference set!     ")
                        self._vector_reference_frequency.append(
                            [s - self._reference_value_frequency[i] for s in self._readFREQ])
                        xs = np.array(np.linspace(
                            0, ((self._readFREQ[-1]-self._readFREQ[0])/self._readFREQ[0]), len(self._readFREQ)))
                        self._vector_reference_dissipation.append(
                            xs-self._reference_value_dissipation[i])

    def factory_defaults(self):

        action_role = UserRoles.ADMIN
        check_result = UserProfiles().check(self.ControlsWin.userrole, action_role)

        if not check_result:
            PopUp.critical(self, "Insufficient User Rights",
                           "A user with ADMIN rights is required to perform this action.\nPlease try again with sufficient user rights.", ok_only=True)
            Log.w("Factory Default aborted: Insufficient User Rights.")
            return

        do_default = PopUp.question(self, "Restore Factory Defaults",
                                    "WARNING: Are you sure you want to restore factory defaults?\n\nThis will invalidate the 'config' folder and delete all registry keys that are associated with this application.")

        if not do_default:
            Log.d("Factory Default aborted: User Canceled Confirmation.")
            return

        try:
            # invalidate 'config' folder
            i = 1
            while os.path.exists(Constants.csv_calibration_export_path):
                try:
                    os.rename(Constants.csv_calibration_export_path,
                              f"{Constants.csv_calibration_export_path}-{i}")
                except:
                    Log.d(
                        "Folder already exists, looking for non-existent directory:", i)
                    i += 1
            Log.w("Factory Default: Invalidated 'config' folder successfully.")

            # restore viewStates to True
            self.ControlsWin.chk1.setChecked(True)
            self.ControlsWin.chk2.setChecked(True)
            self.ControlsWin.chk3.setChecked(True)
            self.ControlsWin.chk4.setChecked(True)
            Log.w("Factory Default: Restore viewStates in application.")

            # reload UI states to match viewStates
            self.ControlsWin.toggle_console()
            self.ControlsWin.toggle_amplitude()
            self.ControlsWin.toggle_temperature()
            self.ControlsWin.toggle_RandD()
            Log.w("Factory Default: Reloaded UI states to sync.")

            # remove all keys
            self.AppSettings.clear()
            Log.w("Factory Default: Removed all application registry keys.")

            Log.i("Factory Default finished successfully.")

        except Exception as e:
            Log.e("Factory Default: Error occurred!", e)

    ###########################################################################
    # TEC Temperature Update Task
    ###########################################################################

    def _update_tec_temp(self):  # wired to slider change

        self.tecWorker.set_slider_down(
            self.ControlsWin.ui1.slTemp.isSliderDown())
        self.tecWorker.set_slider_value(self.ControlsWin.ui1.slTemp.value())
        self.tecWorker.set_slider_enable(
            self.ControlsWin.ui1.slTemp.isEnabled())

        pv = self.tecWorker._tec_temp
        sp = self.ControlsWin.ui1.slTemp.value()
        op = self.tecWorker._tec_power

        if sp == 0.00:
            sp = 0.25

        if self.tecWorker._tec_state == "OFF":
            # control is OFF: update GUI only, not TEC
            new_l1 = "PV:--.--C SP:{1:2.2f}C OP:[OFF]".format(pv, sp, op)
            self.ControlsWin.ui1.lTemp.setText(new_l1)
            self.tecWorker._tec_update_now = False
        elif self.ControlsWin.ui1.slTemp.isSliderDown():
            # control is ON: update GUI only, not TEC
            new_l1 = "PV:{0:2.2f}C SP:{1:2.2f}C OP:{2:+04.0f}".format(
                pv, sp, op)
            self.ControlsWin.ui1.lTemp.setText(new_l1)
            self.tecWorker._tec_update_now = False
        else:
            # control is ON: update TEC now (which updates GUI internally)
            self.tecWorker._tec_update_now = True
            self.tecWorker.update_now.emit()

    def _enable_tec(self):  # wired to on/off button
        if self.tecWorker._tec_state == "OFF" or self.tecWorker._tec_locked:
            # check version and write device info (if needed)
            selected_port = self.ControlsWin.ui1.cBox_Port.currentData()
            if selected_port == None:
                selected_port = ''  # Dissallow None
            if selected_port == "CMD_DEV_INFO":
                selected_port = ''  # Dissallow Action

            if self.multiplex_plots > 1:
                selected_port = []
                for i in range(self.multiplex_plots):
                    if i < self.ControlsWin.ui1.cBox_Port.count() - 1:
                        selected_port.append(
                            self.ControlsWin.ui1.cBox_Port.itemData(i))

            self.worker._port = selected_port  # used in run()
            do_continue = self.fwUpdater.run(self)
            if self.fwUpdater._port_changed:
                self.ControlsWin.ui1.pButton_Refresh.clicked.emit()
                selected_port = self.fwUpdater._port
                self.worker._port = selected_port
            if not do_continue:
                return

            # Restore single selected port (even when multiplex mode)
            selected_port = self.ControlsWin.ui1.cBox_Port.currentData()
            if selected_port == None:
                selected_port = ''  # Dissallow None
            if selected_port == "CMD_DEV_INFO":
                selected_port = ''  # Dissallow Action

            if len(self._selected_port) == 0:
                Log.e(
                    f"ERROR: No active device is currently available for TEC status updates.")
                Log.e(
                    "Please connect a device, hit \"Reset\", and try \"Temp Control\" again.")
                return

            # turn temp control on
            self.ControlsWin.ui1.pTemp.setText("Stop Temp Control")

            if hasattr(self, "tecThread"):
                if self.tecThread.isRunning():
                    Log.d("Waiting for Temp Control to stop.")
                    # queue thread for 'quit' on next update
                    self.tecWorker._tec_stop_thread = True
                    self.tecWorker._tec_update_now = False  # invalidate flag to update TEC again
                    self.tecWorker.update_now.emit()  # force next update of task to happen now
                    self.tecThread.wait()  # wait for thread to quit, gracefully
            Log.d("Starting new TEC thread.")
            self.tecThread = QtCore.QThread()
            self.tecWorker = TECTask()
            self.tecWorker.set_port(selected_port)
            self.tecWorker.set_slider_down(
                self.ControlsWin.ui1.slTemp.isSliderDown())
            self.tecWorker.set_slider_value(
                self.ControlsWin.ui1.slTemp.value())
            self.tecWorker.set_slider_enable(
                self.ControlsWin.ui1.slTemp.isEnabled())
            self.tecWorker.moveToThread(self.tecThread)
            self.tecThread.worker = self.tecWorker
            self.tecThread.started.connect(self.tecWorker.run)
            self.tecWorker.finished.connect(self.tecThread.quit)
            self.tecWorker.auto_off.connect(self.tec_auto_off)
            self.tecWorker.volt_err.connect(self.tec_volt_err)
            self.tecWorker.finished.connect(self.tec_stopped)
            self.tecWorker.lTemp_setText.connect(
                lambda str: self.ControlsWin.ui1.lTemp.setText(str))
            self.tecWorker.lTemp_setStyleSheet.connect(
                lambda str: self.ControlsWin.ui1.lTemp.setStyleSheet(str))
            self.tecWorker.infobar_setText.connect(
                lambda str: self.ControlsWin.ui1.infobar.setText(str))
            self.tecThread.start()

            # disable task timeout if running in development mode
            enabled, error, expires = UserProfiles.checkDevMode()
            if enabled:
                self.tecWorker._task_timeout = np.inf

        else:
            # turn temp control off
            self.ControlsWin.ui1.pTemp.setText("Start Temp Control")
            self.tecWorker._tec_update("OFF")
            self.tecWorker._tec_stop_thread = True
            self.tecWorker._tec_update_now = False  # invalidate flag to update TEC again
            self.tecWorker.update_now.emit()
            self.ControlsWin.ui1.lTemp.setStyleSheet("")

    def tec_auto_off(self):
        self.ControlsWin.ui1.pTemp.setText("Resume Temp Control")
        self.ControlsWin.ui1.tool_TempControl.setChecked(False)
        self.ControlsWin.ui1.tempController.setEnabled(False)

    def tec_volt_err(self):
        if not PopUp.critical(self,
                              "Power Issue Detected",
                              "<b>POWER ISSUE</b>: VOLTAGE ERROR DETECTED!<br/>" +
                              "Please confirm external voltage supply is powered.<br/>" +
                              "Ignore this warning at your own risk. May cause damage!",
                              details=f"Voltage Detected: {self.tecWorker._tec_voltage}\nExpected: 5V",
                              btn1_text="Ok"):
            # force auto-restart, user accepts liability of incorrect voltage
            self.tecWorker._tec_stop_thread = False
            self.tecWorker._tec_update_now = True

    def tec_stopped(self):
        sp = self.ControlsWin.ui1.slTemp.value()
        new_l1 = "PV:--.--C SP:{0:2.2f}C OP:[OFF]".format(sp)  # pv, sp, op)
        self.ControlsWin.ui1.lTemp.setText(new_l1)
        self.ControlsWin.ui1.lTemp.setStyleSheet("")
        Log.d("TEC Thead and Worker have finished.")

    ########################################################################################################
    # Gets information from QATCH webpage and enables download button if new version software is available
    ########################################################################################################

    def get_web_info(self, return_info):

        try:
            Log.i(TAG, 'Checking online for updates...')

            color = '#ffa500'
            labelweb2 = 'ONLINE'
            labelweb3 = 'UNKNOWN!'

            if "v2.3" in Constants.app_version:
                branch = "v2.3x"
            elif "v2.4" in Constants.app_version:
                branch = "v2.4x"
            elif "v2.5" in Constants.app_version:
                branch = "v2.5x"
            elif "v2.6" in Constants.app_version:
                branch = "v2.6x"
            else:
                # skip to 'except' case below
                raise NameError('Unknown branch, not recognized')

            if hasattr(self, "url_download"):
                color, labelweb3 = self.update_found(self.build_descr)
                return color, labelweb2

            if return_info:
                color, labelweb2 = self.update_check()  # blocking

                if not hasattr(self, 'url_download'):
                    if self.res_download:
                        if self.update_resources(branch):
                            labelweb3 = 'Resources updated'
                        else:
                            labelweb3 = 'Resources out-of-date'
                    else:
                        labelweb3 = 'UP-TO-DATE!'
                    self.ask_for_update = False
                else:
                    color, labelweb3 = self.update_found(self.build_descr)

                if hasattr(self, "_dbx_connection"):
                    self._dbx_connection.close()
                return color, labelweb2

            self.web_thread = threading.Thread(
                target=self.update_check)  # non-blocking
            self.web_thread.start()

            # ping periodically for task to finish
            QtCore.QTimer.singleShot(1000, self.update_ping)

        except Exception as e:
            Log.e("Update Task error:", e)

    def update_ping(self):
        # periodic check for update task completion
        try:
            if "v2.3" in Constants.app_version:
                branch = "v2.3x"
            elif "v2.4" in Constants.app_version:
                branch = "v2.4x"
            elif "v2.5" in Constants.app_version:
                branch = "v2.5x"
            elif "v2.6" in Constants.app_version:
                branch = "v2.6x"
            else:
                # skip to 'except' case below
                raise NameError('Unknown branch, not recognized')

            if self.web_thread.is_alive():
                Log.d("Waiting on update check...")
                QtCore.QTimer.singleShot(1000, self.update_ping)
                return

            if not hasattr(self, 'url_download'):
                if self.res_download:
                    if self.update_resources(branch):
                        labelweb3 = 'Resources updated'
                    else:
                        labelweb3 = 'Resources out-of-date'
                else:
                    labelweb3 = 'UP-TO-DATE!'
                self.ask_for_update = False
            else:
                color, labelweb3 = self.update_found(self.build_descr)

            if hasattr(self, "_dbx_connection"):
                self._dbx_connection.close()

        except Exception as e:
            Log.e("Update Task error:", e)

    def update_found(self, v):
        color = '#ff0000'
        # if there's a space, skip the first word; otherwise, use the whole string
        labelweb3 = '{} available!'.format(v)
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(
            Architecture.get_path(), 'QATCH/icons/download_icon.ico')
        self.InfoWin.ui3.pButton_Download.setIcon(
            QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.InfoWin.ui3.pButton_Download.setText(
            _translate("MainWindow3", " Download ZIP"))

        # re-import each time to update settings from file
        from QATCH.common.userProfiles import UserConstants
        check_result = True
        if UserConstants.REQ_ADMIN_UPDATES:
            action_role = UserRoles.ADMIN
            check_result = UserProfiles().check(self.ControlsWin.userrole, action_role)

        if hasattr(self, "ask_for_update"):
            Log.i("Update Status:", labelweb3)
            if check_result == True:
                if PopUp.question_FW(self,
                                     "QATCH Update Available!",
                                     "A new software version is available!\nWould you like to download it now?",
                                     "Running SW: {} ({})\nRecommended: {}\n".format(Constants.app_version, Constants.app_date, v) +
                                     "Filename: {}\n\nPlease save your work before updating.".format(os.path.basename(self.url_download["path"]))):
                    self.start_download()
            else:
                Log.w(
                    "A software update is available! Please ask your administrator to install update.")
            color = "#00ff00"  # do now show popup indicating "You are running the latest version"
        self.ask_for_update = True

        return color, labelweb3

    def update_check(self):
        # Get latest info from Dropbox webpage
        try:
            color = '#ffa500'
            labelweb2 = 'ONLINE'
            labelweb3 = 'UNKNOWN!'

            # init 'res_download' if not set
            if not hasattr(self, "res_download"):
                self.res_download = False

            import dropbox
            import base64
            import json

            access_path = os.path.join(
                Constants.local_app_data_path, "tokens", "dbx_access_token.pem")
            expires_at = datetime.datetime.fromtimestamp(
                0)  # mark expired if no file found
            # check if stored access token is still valid
            if os.path.exists(access_path):
                Log.d("Checking expiration of access token.")
                with open(access_path, 'r') as f:
                    data = json.load(f)
                    access_token = data['access_token']
                    expires_in = data['expires_in']
                    captured = os.path.getmtime(access_path)
                    captured = datetime.datetime.fromtimestamp(captured)
                    # consider expired if within 1 min of expiration:
                    expires_at = captured + \
                        datetime.timedelta(seconds=int(expires_in), minutes=-1)
                    Log.d(f"Expires at: {expires_at}")
            if datetime.datetime.now() >= expires_at:
                # stored access_token is expired, attempt to get a new one
                # prefer working resource path, if exists
                working_resource_path = os.path.join(
                    os.getcwd(), "QATCH/resources/")
                bundled_resource_path = os.path.join(Architecture.get_path(
                ), "QATCH/resources/")  # otherwise, use bundled resource path
                # resource_path = working_resource_path if os.path.exists(working_resource_path) else bundled_resource_path
                # prefer working keystore, if it exists
                keystore = os.path.join(
                    working_resource_path, "dbx_key_store.zip")
                if not os.path.exists(keystore):
                    # use bundled keystore, if none in working folder
                    keystore = os.path.join(
                        bundled_resource_path, "dbx_key_store.zip")
                if os.path.exists(keystore):
                    Log.d("Retrieving keys from Dropbox Keystore.")
                    with pyzipper.AESZipFile(keystore, 'r',
                                             compression=pyzipper.ZIP_DEFLATED,
                                             allowZip64=True,
                                             encryption=pyzipper.WZ_AES) as zf:
                        # Add a protected file to the zip archive
                        zf.setpassword(hashlib.sha256(
                            zf.comment).hexdigest().encode())
                        app_authorization = zf.read(
                            "app_authorization").decode()
                        refresh_token = zf.read("refresh_token").decode()
                else:
                    raise Exception(
                        "Dropbox Keystore file does not exist. Cannot automatically check for updates.")
                Log.d("Retrieving refreshed access token from Dropbox.")
                headers = {'Authorization': f'Basic {app_authorization}',
                           'content-type': 'application/x-www-form-urlencoded'}
                r = requests.post("https://api.dropbox.com/oauth2/token", headers=headers,
                                  data=f"refresh_token={refresh_token}&grant_type=refresh_token")
                if r.status_code != requests.codes.ok:
                    Log.e(r.status_code, r.json())
                r.raise_for_status()  # throw exception if not Status Code 200 - OK
                access_token = r.json()['access_token']
                # store retrieved access_token for later
                FileManager.create_dir(os.path.split(access_path)[0])
                with open(access_path, 'w') as f:
                    f.write(r.text)
                    Log.d("Saved access token for later.")

            if "v2.3" in Constants.app_version:
                branch = "v2.3x"
            elif "v2.4" in Constants.app_version:
                branch = "v2.4x"
            elif "v2.5" in Constants.app_version:
                branch = "v2.5x"
            elif "v2.6" in Constants.app_version:
                branch = "v2.6x"
            else:
                # skip to 'except' case below
                raise NameError('Unknown branch, not recognized')

            # determing bundle type: Python vs EXE
            if getattr(sys, 'frozen', False):
                require_EXE = True
            else:
                require_EXE = False

            # determine current build type: beta vs release
            running_beta_build = branch.replace(
                'x', 'b') in Constants.app_version
            running_release_build = branch.replace(
                'x', 'r') in Constants.app_version
            if running_beta_build == running_release_build:
                Log.e(
                    "Unable to determine whether running beta or release build. Assuming 'beta' and continuing.")
                running_beta_build = True  # if we got here, logically both had to be 'False'
                running_release_build = False  # just to be safe

            self._dbx_connection = dropbox.Dropbox(access_token)

            try:
                all_targets_path = f'/targets.csv'
                metadata, response = self._dbx_connection.files_download(
                    all_targets_path)
                targets = {}
                response_data = ""
                for line in response.iter_lines():
                    response_data += f"{line.decode()}\n"
                    uuid, uuip = line.decode().strip().split(',')
                    targets[uuid] = uuip
                response.close()  # release connection, we are done reading 'targets'
                # get unique user id (PC name) and universal user ip (WANIP address)
                uuid = Architecture.get_os_name()
                r = requests.get("https://checkip.amazonaws.com")
                r.raise_for_status()
                uuip = r.text.strip()
                if not uuid in targets.keys():  # UUID not registered in targets, add it
                    response_data += f"{uuid},{uuip}\n"
                    self._dbx_connection.files_upload(response_data.encode(
                    ), all_targets_path, dropbox.files.WriteMode.overwrite)
                elif targets[uuid] != uuip:  # UUIP has changed for existing UUID, update it
                    start_idx = response_data.index(uuid)
                    end_idx = response_data.index('\n', start_idx)
                    old_entry = response_data[start_idx:end_idx+1]
                    new_entry = f"{uuid},{uuip}\n"
                    response_data = response_data.replace(old_entry, new_entry)
                    self._dbx_connection.files_upload(response_data.encode(
                    ), all_targets_path, dropbox.files.WriteMode.overwrite)
            except Exception as e:
                Log.w("Unable to check the list of targets on Dropbox.")
                Log.d("Exception details:", e)

            builds = []
            if Constants.UpdateEngine == UpdateEngines.DropboxAPI:
                for entry in self._dbx_connection.files_list_folder(f'/{branch}', recursive=True).entries:
                    if (
                        entry.name.lower().startswith("nanovisq") and
                        entry.name.lower().endswith(".zip") and
                        entry.name.lower().find("installer") == -1 and
                        # build type matches (python vs EXE)
                        (entry.name.lower().find("exe") >= 0) == require_EXE
                    ):
                        build = {"date": entry.server_modified,
                                 # .replace("_exe", "").replace("_py", ""),
                                 "name": entry.name,
                                 "path": entry.path_display,
                                 "size": entry.size}  # ,
                        builds.append(build)
            if Constants.UpdateEngine == UpdateEngines.GitHub:
                latest_release_url = Constants.UpdateGitRepo + "/releases/latest"
                # url redirects to latest tag
                resp = requests.get(latest_release_url)
                tags_url = resp.url.replace("tag", "download") + "/tags.txt"
                resp = requests.get(tags_url)
                if resp.ok:
                    all_tags = resp.content.decode().split()[
                        ::-1]  # newest to oldest
                    date_order = parser.parse(resp.headers['Last-Modified'])
                    for entry in all_tags:
                        build_type = "exe" if require_EXE else "py"
                        date_order -= datetime.timedelta(days=1)
                        build = {"date": date_order,  # updated to actual release date when calling target_build()
                                 "name": f"nanovisQ_SW_{entry}_{build_type}.zip",
                                 "path": latest_release_url.replace("latest", f"download/{entry}/nanovisQ_SW_{entry}_{build_type}.zip"),
                                 "size": 0}
                        builds.append(build)

            def get_dates(build):
                return build.get('date')

            def target_build(path):
                try:
                    targets = []
                    response = None
                    actual_date = None
                    install_check_path = f"{os.path.split(path)[0]}/installer.checksum"
                    build_targets_path = f"{os.path.split(path)[0]}/targets.csv"
                    if Constants.UpdateEngine == UpdateEngines.DropboxAPI:
                        metadata, response = self._dbx_connection.files_download(
                            build_targets_path)
                    if Constants.UpdateEngine == UpdateEngines.GitHub:
                        metadata = requests.get(install_check_path)
                        response = requests.get(build_targets_path)
                        if metadata.ok:
                            # prefer time from 'installer.checksum'
                            actual_date = parser.parse(
                                metadata.headers['Last-Modified'])
                        elif response.ok:
                            # fallback to time from 'targets.csv'
                            actual_date = parser.parse(
                                response.headers['Last-Modified'])
                    if response:
                        for line in response.iter_lines():
                            targets.append(line.decode().strip())
                        response.close()  # release connection, we are done reading 'targets'
                except Exception as e:
                    targets = ["ALL"]
                if "ALL" in targets or Architecture.get_os_name() in targets:
                    return True, actual_date
                else:
                    return False, actual_date
            # sort by name (descending order)
            builds.sort(key=get_dates, reverse=True)

            # find most recent build based on allowed build type: beta vs release
            enabled, error, expires = UserProfiles.checkDevMode()
            # bundled EXE, running 'r' or not in dev
            release_builds_only = (require_EXE and (
                running_release_build or not enabled))
            most_recent = None
            if release_builds_only:
                # find most recent *release* build, skipping any newer *beta* builds
                # raise an Exception if no *release* builds are found on the server!
                for build in builds:
                    is_release_build = branch.replace(
                        'x', 'r') in build["name"]
                    is_target_build, _d = target_build(build["path"])
                    if _d:
                        build["date"] = _d
                    if is_release_build and is_target_build:
                        most_recent = build
                        break  # found, stop searching
            elif len(builds) > 0:
                for build in builds:
                    is_release_build = branch.replace(
                        'x', 'r') in build["name"]
                    is_target_build, _d = target_build(build["path"])
                    if _d:
                        build["date"] = _d
                    if is_release_build and enabled and not running_release_build:
                        # Do NOT offer a B-dev build an update to R build
                        # (user must turn off dev mode 1st to get R build)
                        continue  # keep searching
                    if is_target_build:
                        most_recent = build
                        break  # found, stop searching
            if most_recent == None:
                Log.d(f'requires EXE: {require_EXE}')
                Log.d(f'release ONLY: {release_builds_only}')
                Log.d(f'found builds: {builds}')
                raise NameError("No compatible builds available on server!")

            # import all to dictionary, sorted by date
            # download the most recent build if version is different and/or date is newer than current build
            ts1 = most_recent["date"].date()
            ts2 = datetime.datetime.fromisoformat(Constants.app_date).date()
            Log.d(f'ts1 = {ts1}, ts2 = {ts2}')

            if Constants.app_version in most_recent["name"] and ts1 == ts2:
                # we're running the latest version (with files)
                if hasattr(self, "url_download"):
                    delattr(self, "url_download")
                color = '#00c600'
            elif ts1 >= ts2 or (running_beta_build and release_builds_only):
                # most recent build is newer than current build - or -
                # current version is a *beta* build and we are only allowed to run *release* builds
                self.ask_for_update = True
                self.url_download = most_recent
                version_info_descr = branch[:-1]  # drop 'x' from 'v2.6x'
                self.build_descr = f'{most_recent["name"][most_recent["name"].rfind(version_info_descr):-4]} ({ts1})'
                self.build_descr = self.build_descr.replace("_exe", "").replace(
                    "_py", "")  # remove '_exe' and '_py' from build description
                # color, labelweb3 = self.update_found(self.build_descr)

            if not hasattr(self, 'url_download'):
                if self.update_resources_check(branch):
                    labelweb3 = 'Resources out-of-date'
                    # self.res_download = True
                else:
                    labelweb3 = 'UP-TO-DATE!'

                if ts1 < ts2:  # and 'self.url_download' not defined
                    Log.w(
                        "*Most recent build on server is not newer than this build. Ignoring it.")
                    labelweb3 += '*'
            elif ts1 < ts2:  # and 'self.url_download' is defined
                Log.e(
                    "*Most recent RELEASE build on server is not newer but you are running a BETA build against policy.")
                Log.e(
                    "Please downgrade your BETA build back to the most recent RELEASE build (or enable DEV MODE again).")
                labelweb3 += '*'

        # catch network-related errors
        except requests.exceptions.RequestException as e:
            Log.d('Update Error:', e)
            color = '#ff0000'
            labelweb2 = 'OFFLINE'
            labelweb3 = 'Offline, cannot check'

        # catch unknown branch error
        except NameError as e:
            Log.e('Update Error:', e)
            # do nothing else, labels are good

        # catch all other errors (non-network)
        except Exception as e:
            Log.e('Update Error:', e)
            color = '#ff0000'
            labelweb2 = 'ERROR'
            labelweb3 = 'Error, check failed'

            # a general exception has occurred, print stack traceback
            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb
            a_list = ['Traceback (most recent call last):']
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        Log.i(TAG, 'Checking your internet connection {} '.format(labelweb2))

        self.InfoWin.ui3.lweb3.setText(
            "<font color=#0000ff > Update Status &nbsp;&nbsp;&nbsp;</font><font color={}>{}</font>".format(color, labelweb3))
        if hasattr(self, 'ask_for_update') == False or self.ask_for_update == False:
            Log.i("Update Status:", labelweb3)
        if not hasattr(self, 'ask_for_update'):
            self.ask_for_update = False
        return color, labelweb2

    def update_resources_check(self, branch):
        try:
            download_resources = False
            current_version = "0,None"
            latest_version = "-1,Unknown"
            server_file_size = 0

            self.res_download = download_resources
            self.res_current_version = current_version
            self.res_latest_version = latest_version
            self.res_files = []

            remote_resource_path = f'/{branch}/resources/'
            working_resource_path = os.path.join(
                os.getcwd(), "QATCH/resources/")
            bundled_resource_path = os.path.join(
                Architecture.get_path(), "QATCH/resources/")
            remote_file_compare = os.path.join(
                remote_resource_path, "lookup_resources.csv")
            working_file_compare = os.path.join(
                working_resource_path, "lookup_resources.csv")
            bundled_file_compare = os.path.join(
                bundled_resource_path, "lookup_resources.csv")

            resources = []
            if Constants.UpdateEngine == UpdateEngines.DropboxAPI:
                for resource in self._dbx_connection.files_list_folder(remote_resource_path, recursive=False).entries:
                    resources.append(resource.name)
            if Constants.UpdateEngine == UpdateEngines.GitHub:
                resources = os.listdir(bundled_resource_path)
            self.res_files = resources.copy()

            try:
                if os.path.exists(bundled_file_compare):
                    Log.d(
                        "Looking for updated bundled resources to unpack in local working directory...")
                    if (os.path.exists(working_file_compare) == False or
                            os.stat(working_file_compare).st_size < os.stat(bundled_file_compare).st_size):
                        Log.d(
                            "Found newer locally bundled resources! Extracting to working directory...")
                        os.makedirs(working_resource_path, exist_ok=True)
                        for root, dirs, files in os.walk(bundled_resource_path):
                            for file in files:
                                if file in resources:
                                    # only export resources listed on server, not all locally bundled resources
                                    bundled_resource_path_file = os.path.join(
                                        bundled_resource_path, file)
                                    working_resource_path_file = os.path.join(
                                        working_resource_path, file)
                                    if os.path.exists(working_resource_path_file):
                                        Log.d(
                                            f"Deleting {working_resource_path_file}...")
                                        os.remove(working_resource_path_file)
                                    Log.d(
                                        f"Extracting {file}... to {working_resource_path}")
                                    os.rename(bundled_resource_path_file,
                                              working_resource_path_file)
                        Log.d(
                            "Extracted all bundled resource files to working directory.")
                    else:
                        Log.d(
                            "Resources are up-to-date compared to the locally bundled files. Checking server for updates.")
                else:
                    Log.w(
                        "Unable to find any locally bundled resource files. Nothing to extract.")
            except Exception as e:
                Log.e(
                    "An error occurred while extracting bundled resources to working directory. Checking server for updates.")

            if Constants.UpdateEngine == UpdateEngines.DropboxAPI:
                metadata, response = self._dbx_connection.files_download(
                    remote_file_compare)
                for line in response.iter_lines():
                    latest_version = line.decode().strip()  # last line saved
                server_file_size = metadata.size
            if Constants.UpdateEngine == UpdateEngines.GitHub:
                resource_file_url = Constants.UpdateGitRepo + "/raw/" + \
                    Constants.UpdateGitBranch + \
                    remote_file_compare.replace(branch, "QATCH")
                response = requests.get(resource_file_url)
                if response.ok:
                    latest_version = response.text.split(
                    )[-1].strip()  # last line saved
                    server_file_size = int(
                        response.headers['Content-Length'])+112
            response.close()  # release connection, we are done reading 'remote file compare'
            if os.path.exists(working_file_compare):
                if server_file_size != os.stat(working_file_compare).st_size:
                    with open(working_file_compare, 'r') as f:
                        lines = f.read().splitlines()
                        current_version = lines[-1].strip()
                    if latest_version != current_version:
                        download_resources = True
            else:
                os.makedirs(working_resource_path, exist_ok=True)
                download_resources = True

            self.res_download = download_resources
            self.res_current_version = current_version
            self.res_latest_version = latest_version

            return download_resources

        except Exception as e:
            Log.e("ERROR: Unable to update resources!")
            Log.e("Reason: " + str(e))
            return False

    def update_resources(self, branch):
        try:
            remote_resource_path = f'/{branch}/resources/'
            working_resource_path = os.path.join(
                os.getcwd(), "QATCH/resources/")
            working_file_compare = os.path.join(
                working_resource_path, "lookup_resources.csv")

            download_resources = self.res_download
            resources = self.res_files
            current_version = self.res_current_version
            latest_version = self.res_latest_version

            if download_resources:
                if not PopUp.question_FW(self,
                                         "QATCH Update Available!",
                                         "A new resource bundle is available!\nWould you like to download it now?",
                                         "Current version: {}\nRecommended: {}\n".format(current_version, latest_version) +
                                         "Remote path: {}\n\nPlease save your work before updating.".format(remote_resource_path)):
                    download_resources = False  # abort, do now download resources, user does not want it
            else:
                Log.d(
                    f"Resource files are up-to-date! Latest Version: {latest_version}")
                return

            if download_resources:
                # Show progress bar
                self.progressBar = QtWidgets.QProgressDialog(
                    f"Downloading resources...", "Cancel", 0, 100, self)
                icon_path = os.path.join(
                    Architecture.get_path(), 'QATCH/icons/download_icon.ico')
                self.progressBar.setWindowIcon(QtGui.QIcon(icon_path))
                self.progressBar.setWindowTitle("QATCH nanovisQ")
                self.progressBar.setWindowFlag(
                    QtCore.Qt.WindowContextHelpButtonHint, False)
                self.progressBar.setWindowFlag(
                    QtCore.Qt.WindowStaysOnTopHint, True)
                self.progressBar.setFixedSize(
                    int(self.progressBar.width()*1.5), int(self.progressBar.height()*1.1))
                self.progressBar.show()
                _cancel = False

                # DOWNLOAD RESOURCES FROM SERVER
                detail_text = []
                num_files = len(resources)
                for i, resource in enumerate(resources):
                    pct = int(100 * i / num_files)
                    self.progressBar.setValue(pct)
                    self.progressBar.repaint()  # force update
                    # process repaint and cancel signals in synchronous thread
                    QtCore.QCoreApplication.processEvents()
                    if self.progressBar.wasCanceled():
                        # stop prematurely due to user cancel
                        _cancel = True
                        break
                    detail_text.append(f"Updating \"{resource}\"...")
                    file_local = os.path.join(working_resource_path, resource)
                    file_remote = os.path.join(remote_resource_path, resource)
                    if os.path.exists(file_local):
                        Log.d(f"Deleting {file_local}...")
                        os.remove(file_local)
                    Log.d(f"Downloading {file_remote} to {file_local}...")
                    if Constants.UpdateEngine == UpdateEngines.DropboxAPI:
                        self._dbx_connection.files_download_to_file(
                            file_local, file_remote)
                    if Constants.UpdateEngine == UpdateEngines.GitHub:
                        git_mapped_url = Constants.UpdateGitRepo + '/raw/' + \
                            Constants.UpdateGitBranch + \
                            file_remote.replace(branch, "QATCH")
                        response = requests.get(git_mapped_url)
                        response.raise_for_status()
                        with open(file_local, 'wb') as f:
                            f.write(response.content)
                        response.close()
                    detail_text[-1] += "\tDONE!"

                # Update finished
                self.progressBar.close()  # finish progress bar @ 100%
                if _cancel:
                    # Partial success
                    Log.w(
                        f"User canceled resource download: Updated {i} of {num_files} files.")
                    with open(working_file_compare, "a") as f:
                        f.write("* (Partial download)")
                    Log.w("Marked resources for future update again.")
                    PopUp.question_FW(self, "Resource Update Canceled",
                                      "Resource files are partially updated to their latest versions.\nPlease update again to fully update.",
                                      "Updated {} of {} files from \"{}\" to \"{}\".\n\nDETAILS:\n{}".format(i, num_files, current_version, latest_version, "\n".join(detail_text)), True)
                else:
                    # Update successful
                    PopUp.question_FW(self, "Resource Update Successful",
                                      "Resource files are now updated to their latest versions.",
                                      "Updated all {} files from \"{}\" to \"{}\".\n\nDETAILS:\n{}".format(num_files, current_version, latest_version, "\n".join(detail_text)), True)

            else:
                Log.d("User declined resource update. Skipping download.")

        except Exception as e:
            Log.e("ERROR: Unable to update resources!")
            Log.e("Reason: " + str(e))
            raise e

    ###########################################################################
    # Opens webpage for download
    ###########################################################################

    def start_download(self, return_info=False):
        do_launch_inline = False
        if hasattr(self, "url_download"):
            date = self.url_download["date"]
            name = self.url_download["name"]
            path = self.url_download["path"]
            size = self.url_download["size"]

            # self.MainWin.ui0.splitter.setSizes([0, 1]) # make log window entire splitter view

            # Log.d("GUI: Force repaint events") # no longer needed
            Log.w(f"Most Recent Build: {name}")
            # Log.w(f"Update Filename: {os.path.basename(path)}")
            Log.w("Downloading latest software... please wait...")
            Log.w("The updated application will launch when ready.")

            # auto-extract build and launch it when user says yes
            cur_install_path = os.path.dirname(sys.argv[0])
            if '\\' in cur_install_path or '/' in cur_install_path:
                pass  # found directories in install path, not relative to CWD()
            elif getattr(sys, 'frozen', False):  # frozen cannot do relative path
                cur_install_path = os.path.dirname(sys.executable)
            # argv[0] contains a file name only, no relative path
            elif len(cur_install_path) == 0:
                cur_install_path = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__)))
            if name[0:-4].endswith("_py") and os.path.split(cur_install_path)[1] == "QATCH":
                cur_install_path = os.path.dirname(cur_install_path)
            save_to = os.path.join(os.path.dirname(
                cur_install_path), name[0:-4], name)
            new_install_path = os.path.dirname(save_to)
            if not os.path.exists(new_install_path):
                # make folder first
                os.makedirs(new_install_path, exist_ok=True)
                do_install = True
            else:
                Log.w(f"Default path already exists: {new_install_path}")
                do_install = PopUp.question(self, "QATCH Software Conflict",
                                            "WARNING: The default path for this new build already exists.\n\n" +
                                            "Would you like to re-install or repair this installation?")

            setup_finished = False
            if do_install:
                if not os.path.exists(save_to):
                    # def progressTask(file_local, file_remote):
                    #     md = self._dbx_connection.files_download_to_file(file_local, file_remote)
                    # from threading import Thread
                    # progressTaskHandle = Thread(target=progressTask, args=(save_to, path,))
                    # progressTaskHandle.start()

                    TAG1 = "[SW Updater]"
                    Log.d(f"Saving new build to: {save_to}")
                    Log.w(f"{size} bytes; pkg: {os.path.basename(path)}")
                    Log.i(TAG1, "Download started.")
                    Log.d("GUI: Toggle progress mode")

                    self.do_install = do_install
                    self.setup_finished = setup_finished
                    self.save_to = save_to
                    self.new_install_path = new_install_path
                    self.do_launch_inline = do_launch_inline

                    self.progressBar = QtWidgets.QProgressDialog(
                        f"Downloading SW {os.path.basename(new_install_path)}...", "Cancel", 0, 100, self)
                    icon_path = os.path.join(
                        Architecture.get_path(), 'QATCH/icons/download_icon.ico')
                    self.progressBar.setWindowIcon(QtGui.QIcon(icon_path))
                    self.progressBar.setWindowTitle(
                        f"Downloading SW {os.path.basename(new_install_path)}")
                    self.progressBar.setWindowFlag(
                        QtCore.Qt.WindowContextHelpButtonHint, False)
                    self.progressBar.setWindowFlag(
                        QtCore.Qt.WindowStaysOnTopHint, True)
                    self.progressBar.canceled.disconnect()
                    self.progressBar.canceled.connect(self.install_cancel)
                    self.progressBar.setFixedSize(
                        int(self.progressBar.width()*1.5), int(self.progressBar.height()*1.1))

                    self.upd_thread = QtCore.QThread()
                    self.updater = self.newUpdaterTask(path, save_to, size)
                    self.updater.moveToThread(self.upd_thread)

                    self.upd_thread.started.connect(self.updater.run)
                    self.updater.finished.connect(self.upd_thread.quit)
                    self.updater.finished.connect(self.updater.deleteLater)
                    self.upd_thread.finished.connect(
                        self.upd_thread.deleteLater)
                    self.updater.progress.connect(self.update_progress)
                    self.updater.finished.connect(self.progressBar.reset)
                    self.updater.finished.connect(self.post_download_check)
                    self.updater.exception.connect(self.show_exception)

                    self.upd_thread.start()
                    self.progressBar.exec_()  # wait here until finished installing

                    # self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
                    if self.updater._cancel:
                        Log.w("Download canceled.")
                        box = QtWidgets.QMessageBox(None)
                        box.setIcon(QtWidgets.QMessageBox.Information)
                        # box.setWindowIcon(QtGui.QIcon(tempicon))
                        box.setWindowTitle("Download Software Update")
                        box.setText("Download canceled.")
                        box.setDetailedText("")
                        box.setStandardButtons(QtWidgets.QMessageBox.Ok)
                        box.exec_()

                    # NOTE: post_download_check() called upon task completion (but not on cancel)
                else:
                    engine = "Dropbox" if Constants.UpdateEngine == UpdateEngines.DropboxAPI else "GitHub"
                    Log.d(
                        f"Build {name} already downloaded from {engine}. Ready to update.")
                    self.post_download_check(
                        do_install, setup_finished, save_to, new_install_path, do_launch_inline)
            else:
                self.post_download_check(
                    do_install, setup_finished, save_to, new_install_path, do_launch_inline)

            # threading.Thread(target=self.ControlsWin.close,).start()
            return None, None
        else:
            return self.get_web_info(return_info)

    def newUpdaterTask(self, src, dest, size):
        _updaterTask = None
        if Constants.UpdateEngine == UpdateEngines.DropboxAPI:
            # InstallWorker(proceed, copy_src, copy_dst)
            _updaterTask = UpdaterTask_Dbx(
                self._dbx_connection, dest, src, size)
        if Constants.UpdateEngine == UpdateEngines.GitHub:
            _updaterTask = UpdaterTask_Git(dest, src, size)
        if not _updaterTask:
            Log.e("No valid UpdateEngine found. Cannot check for updates.")
        return _updaterTask

    def thread_is_finished(self):
        Log.e("Web Thread is finished!")

    def update_progress(self, label_str, progress_pct):
        # size = self.url_download["size"]
        # last_pct = self.progressBar.value()
        # save_to = self.save_to
        # need_repaint = False
        # TAG1 = "[SW Updater]"
        if label_str != None and self.updater._cancel == False:
            # need_repaint = True
            self.progressBar.setLabelText(label_str)
        if progress_pct != None:
            # need_repaint = True
            self.progressBar.setValue(progress_pct)
        # if need_repaint:
        #     self.progressBar.repaint()

    def install_cancel(self):
        self.update_progress("Canceling installation...", 99)
        cancelButton = self.progressBar.findChild(QtWidgets.QPushButton)
        cancelButton.setEnabled(False)
        self.updater.cancel()
        # self.updater.wait()
        # self.progressBar.close()

    def show_exception(self, except_str: str = None):
        box = QtWidgets.QMessageBox(None)
        box.setIcon(QtWidgets.QMessageBox.Critical)
        # box.setWindowIcon(QtGui.QIcon(tempicon))
        box.setWindowTitle("Download Software Update")
        box.setText("<b>An unhandled exception has occurred during execution:</b><br/>" +
                    f"{except_str}<br/><br/>" +
                    "See details for more information.")
        # box.setDetailedText(log_stream.getvalue())
        box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        box.exec_()

    def post_download_check(self, do_install=None, setup_finished=None, save_to=None, new_install_path=None, do_launch_inline=None):
        do_install = do_install if not do_install is None else self.do_install
        setup_finished = setup_finished if not setup_finished is None else self.setup_finished
        save_to = save_to if not save_to is None else self.save_to
        new_install_path = new_install_path if not new_install_path is None else self.new_install_path
        do_launch_inline = do_launch_inline if not do_launch_inline is None else self.do_launch_inline

        if os.path.exists(save_to):
            # Extract ZIP and launch new build
            with pyzipper.AESZipFile(save_to, 'r') as zf:
                zf.extractall(os.path.dirname(new_install_path))
            os.remove(save_to)

            Log.w("Launching setup script for new build...")
            start_new_build = os.path.join(new_install_path, "launch.bat")

            if do_launch_inline:
                with subprocess.Popen(start_new_build, cwd=new_install_path,
                                      stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                      universal_newlines=True) as proc:

                    # Cannot send input after starting communication (must do it now)
                    Log.d("INPUT: Sending inputs to automate the launch script.")
                    # say yes to 'choice' (up to two times, but do 5 to future-proof it)
                    proc._stdin_write("YYYYY")

                    # monitor the script output for application launch
                    line = ""
                    while proc.poll() == None:  # still running
                        line += proc.stdout.read(1)
                        if "Application will launch" in line:
                            Log.i("FINISHED:", "Launching application...")
                            setup_finished = True
                            proc.kill()  # abort!
                            break  # stop monitoring
                        if line.endswith('\n'):
                            Log.i(line.strip())
                            line = ""  # clear, ready for next line

                    # print any errors, but only if the script terminated prematurely
                    has_errors = False
                    error_lines = 0
                    for line in proc.stderr:
                        if error_lines == 0:
                            error_lines += 1
                            Log.i("")  # blank line
                            Log.i("Warnings & Errors Encountered:")
                        if "ERROR" in line:
                            Log.e(line.strip())
                            if not "Access is denied" in line:  # ignore WinError 5: future dev needed to avoid this error
                                has_errors = True
                        else:
                            Log.w(line.strip())
                    # let BAT file script die...

                if has_errors:
                    PopUp.warning(self, "QATCH Update Warning",
                                  "Setup finished, but encountered one or more errors.\nSee log for details.")

                # finally, remove the setup file (if finished)
                try:
                    if setup_finished:
                        os.remove(start_new_build)
                except:
                    Log.e("ERROR: Failed to delete setup script upon completion.")
            else:
                setup_finished = True
        else:
            engine = "Dropbox" if Constants.UpdateEngine == UpdateEngines.DropboxAPI else "GitHub"
            Log.e(
                f"ERROR: File {save_to} does not exist. Failed to download from {engine} server.")

        if setup_finished or not do_install:
            if PopUp.question_FW(self, "QATCH Software Ready!", "Would you like to close this instance and\nlaunch the new application now?"):

                # launch new instance
                launch_file = "QATCH nanovisQ.lnk" if do_launch_inline else "launch.bat"
                start_new_build = os.path.join(new_install_path, launch_file)
                os.startfile(start_new_build, cwd=new_install_path,)

                # close application (prompting user first)
                self.ControlsWin.close_no_confirm = True
                QtCore.QTimer.singleShot(1000, self.ControlsWin.close)

        elif hasattr(self, "updater") and self.updater._cancel:
            Log.w(
                "User aborted software download by clicking cancel. Failed to update software.")
        else:
            PopUp.critical(self, "QATCH Software Update Failed!",
                           "Setup of new application encountered a fatal error.\nSee log for details.", ok_only=True)

        # Log.d("GUI: Normal repaint events") # no longer needed


class UpdaterProcess_Dbx(multiprocessing.Process):
    def __init__(self, dbx_conn, local, remote):
        super().__init__()
        self._dbx_conn = dbx_conn
        self._local = local
        self._remote = remote

    def run(self):
        self.progressTask(self._local, self._remote)

    def progressTask(self, file_local, file_remote):
        md = self._dbx_conn.files_download_to_file(file_local, file_remote)


class UpdaterTask_Dbx(QtCore.QThread):
    finished = QtCore.pyqtSignal()
    exception = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str, int)
    _cancel = False

    def __init__(self, dbx_conn, local_file, remote_file, total_size):
        super().__init__()
        self._dbx_connection = dbx_conn
        self.local_file = local_file
        self.remote_file = remote_file
        self.total_size = total_size

    def cancel(self):
        Log.d("GUI: Toggle progress mode")
        # Log.w("Process kill request")
        self._cancel = True
        # self.progressTaskHandle.terminate()
        self.progressTaskHandle.kill()
        # self._dbx_connection.close() # force abort of active file download

    def run(self):
        try:
            TAG1 = "[UpdaterTask]"
            save_to = self.local_file
            path = self.remote_file
            size = self.total_size
            last_pct = -1

            self.progressTaskHandle = UpdaterProcess_Dbx(
                self._dbx_connection, save_to, path)
            self.progressTaskHandle.start()

            while True:
                try:
                    curr_size = os.path.getsize(save_to)
                except FileNotFoundError as e:
                    curr_size = 0
                except Exception as e:
                    curr_size = 0
                    Log.e(TAG1, f"ERROR: {e}")
                    self.exception.emit(str(e))
                pct = int(100 * curr_size / size)
                if pct != last_pct or curr_size == size:
                    status_str = f"Download Progress: {curr_size} / {size} bytes ({pct}%)"
                    if curr_size == 0:
                        status_str = f"Starting Download: {os.path.basename(path)} ({pct}%)"
                    Log.i(TAG1, status_str)
                    self.progress.emit(
                        status_str[:status_str.rfind(' (')], pct)
                    need_repaint = True
                    last_pct = pct
                if curr_size == size or self._cancel or not self.progressTaskHandle.is_alive():
                    break
            if not self._cancel:
                Log.d("GUI: Toggle progress mode")
                Log.i(TAG1, "Finshed downloading!")
            else:
                self.progressTaskHandle.join()
                if os.path.exists(save_to):
                    Log.d(f"Removing partial file download: {save_to}")
                    os.remove(save_to)
        except Exception as e:
            Log.e(TAG1, f"ERROR: {e}")
            self.exception.emit(str(e))
        self.finished.emit()


class UpdaterTask_Git(QtCore.QThread):
    finished = QtCore.pyqtSignal()
    exception = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str, int)
    _cancel = False

    def __init__(self, local_file, remote_file, total_size):
        super().__init__()
        self.local_file = local_file
        self.remote_file = remote_file
        self.total_size = total_size

    def cancel(self):
        Log.d("GUI: Toggle progress mode")
        # Log.w("Process kill request")
        self._cancel = True

    def run(self):
        try:
            TAG1 = "[UpdaterTask]"
            save_to = self.local_file
            path = self.remote_file
            size = self.total_size
            last_pct = -1

            status_str = f"Starting Download: {os.path.basename(path)} (0%)"
            Log.i(TAG1, status_str)
            self.progress.emit(status_str[:status_str.rfind(' (')], 0)

            file_remote = path
            file_local = save_to
            with requests.get(file_remote, stream=True) as r:
                r.raise_for_status()
                size = int(r.headers['Content-Length'])
                with open(file_local, 'wb') as f:
                    curr_size = 0
                    for chunk in r.iter_content(chunk_size=8192):
                        if self._cancel:
                            break

                        f.write(chunk)
                        curr_size += len(chunk)

                        pct = int(100 * curr_size / size)
                        if pct != last_pct or curr_size == size:
                            status_str = f"Download Progress: {curr_size} / {size} bytes ({pct}%)"
                            Log.i(TAG1, status_str)
                            self.progress.emit(
                                status_str[:status_str.rfind(' (')], pct)
                            last_pct = pct

            if not self._cancel:
                Log.d("GUI: Toggle progress mode")
                Log.i(TAG1, "Finshed downloading!")
            else:
                if os.path.exists(save_to):
                    Log.d(f"Removing partial file download: {save_to}")
                    os.remove(save_to)
        except Exception as e:
            Log.e(TAG1, f"ERROR: {e}")
            self.exception.emit(str(e))
        self.finished.emit()


class TECTask(QtCore.QThread):
    update_now = QtCore.pyqtSignal()
    auto_off = QtCore.pyqtSignal()
    volt_err = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()
    lTemp_setText = QtCore.pyqtSignal(str)
    lTemp_setStyleSheet = QtCore.pyqtSignal(str)
    infobar_setText = QtCore.pyqtSignal(str)

    port = None

    slider_value = 0
    slider_down = False
    slider_enable = False

    _tec_initialized = False
    _tec_state = "OFF"
    _tec_cycling = False
    _tec_status = "CYCLE"
    _tec_setpoint = -1
    _tec_temp = 0
    _tec_power = -1
    _tec_voltage = "0V (0)"
    _tec_voltage_error_seen = False
    _tec_offset1 = "0"
    _tec_offset2 = "0"
    _tec_locked = False
    _tec_update_now = False
    _tec_stop_thread = False
    _tec_debug = False

    _task_timer = None
    _task_rate = 5000
    _task_timeout = 15*60*1000 / _task_rate
    _task_counter = 0
    _task_active = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self._task_timer = QtCore.QTimer(self)
        self._task_timer.timeout.connect(self.task)
        self.update_now.connect(self.task_update)

    def set_port(self, value):
        self.port = value

    def set_slider_value(self, value):
        self.slider_value = value

    def set_slider_down(self, value):
        self.slider_down = value

    def set_slider_enable(self, value):
        self.slider_enable = value

    def run(self):
        Log.i(TAG, "Temp Control started".format(
            strftime('%Y-%m-%d %H:%M:%S', localtime())))
        self.infobar_setText.emit("<font color=#0000ff> Infobar </font><font color={}>{}</font>".format("#333333",
                                                                                                        "Temp Control started."))

        self._task_active = True
        self._task_timer.start(self._task_rate)
        self.task()  # fire immediately

    def task(self):
        try:
            if not self._task_active:
                return
            if True:  # was while()
                try:
                    sp = ""  # only update TEC if changed
                    if (self.slider_value != self._tec_setpoint and not self.slider_down):
                        Log.d("Scheduling TEC for immediate update (out-of-sync)!")
                        self._tec_update_now = True
                    if self._tec_update_now and not self._tec_locked:
                        sp = self.slider_value
                    if self.slider_enable:
                        Log.d(
                            f"{self._task_counter:.0f}/{self._task_timeout:.0f}: Querying TEC status...")
                        self._tec_update(sp)
                        self._tec_locked = False
                    elif not self._tec_stop_thread:
                        if not self._tec_locked:
                            Log.d("Temp Control is locked while main thead is busy!")
                        self._tec_locked = True
                        return  # stop task silently
                    self._tec_update_now = False
                    if self._tec_update_now == False:
                        sp = self.slider_value
                    pv = self._tec_temp
                    op = self._tec_power
                    self._task_counter += 1
                    if sp == 0.00:
                        sp = 0.25
                    if op == 0 or self._task_counter > self._task_timeout:
                        if self._tec_voltage_error_seen:
                            new_l1 = "[VOLTAGE ERROR]"
                            self._tec_voltage_error_seen = False
                            self.volt_err.emit()
                        else:
                            new_l1 = "[AUTO-OFF ERROR]" if np.isnan(
                                pv) else "[AUTO-OFF TIMEOUT]"
                            self._tec_update("OFF")
                            self._task_stop()
                            self.auto_off.emit()
                    else:
                        new_l1 = "PV:{0:2.2f}C SP:{1:2.2f}C OP:{2:+04.0f}".format(
                            pv, sp, op)
                    self.lTemp_setText.emit(new_l1)
                    bgcolor = "yellow"
                    if op == 0 or self._task_counter > self._task_timeout:
                        self._tec_stop_thread = True
                        self._tec_update_now = False  # invalidate flag to update TEC again
                        bgcolor = "red" if np.isnan(pv) else "yellow"
                    else:
                        if self._tec_status == "CYCLE":
                            # Log.i(TAG, "{0}: TEC setpoint is rapid cycle. Wait for READY!".format(strftime('%Y-%m-%d %H:%M:%S', localtime())))
                            self.infobar_setText.emit("<font color=#0000ff> Infobar </font><font color={}><b>{}</b></font>".format("#ff0000",
                                                                                                                                   "Temp Control is cycling to target. Wait for READY! (this may take a few minutes)"))

                        if self._tec_status == "WAIT":
                            self.infobar_setText.emit("<font color=#0000ff> Infobar </font><font color={}><b>{}</b></font>".format("#ff9900",
                                                                                                                                   "Temp Control is stabilizing. Wait for READY! (about one minute remaining)"))

                        if self._tec_status == "CLOSE":
                            self.infobar_setText.emit("<font color=#0000ff> Infobar </font><font color={}><b>{}</b></font>".format("#ff9900",
                                                                                                                                   "Temp Control is about ready. Wait for READY! (only a few seconds left)"))

                        if self._tec_status == "STABLE":
                            bgcolor = "lightgreen"
                            # Log.i(TAG, "{0}: TEC setpoint has stabilized. Ready for START!".format(strftime('%Y-%m-%d %H:%M:%S', localtime())))
                            self.infobar_setText.emit("<font color=#0000ff> Infobar </font><font color={}><b>{}</b></font>".format("#009900",
                                                                                                                                   "Temp Control has stabilized. Ready for START!"))

                        if self._tec_status == "ERROR":
                            bgcolor = "red"
                            Log.e(TAG, "TEC status is in an unkown state. Please restart Temp Control.".format(
                                strftime('%Y-%m-%d %H:%M:%S', localtime())))

                    self.lTemp_setStyleSheet.emit(
                        "background-color: {}".format(bgcolor))
                except Exception as e:
                    Log.e(TAG, "ERROR: Port read error during TEC task".format(
                        strftime('%Y-%m-%d %H:%M:%S', localtime())))
                    if self._tec_debug:
                        raise e  # debug only
                finally:
                    pass

        except:
            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb
            a_list = ['Traceback (most recent call last):']
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        finally:
            pass

    def task_update(self):
        self._task_counter = 0
        if self._tec_stop_thread:
            self._task_stop()  # stop immediately
        if self._tec_update_now:
            self.task()  # fire immediately

    def _task_stop(self):
        Log.i(TAG, "Temp Control stopped".format(
            strftime('%Y-%m-%d %H:%M:%S', localtime())))
        self.infobar_setText.emit("<font color=#0000ff> Infobar </font><font color={}>{}</font>".format("#333333",
                                                                                                        "Temp Control stopped."))
        self._task_active = False
        self.finished.emit()  # stop task

    def _tec_update(self, dac=""):
        # Open, write, read and close the port accordingly
        selected_port = self.port
        if selected_port == None:
            selected_port = ''  # Dissallow None
        if selected_port == "CMD_DEV_INFO":
            selected_port = ''  # Dissallow Action

        if self._tec_initialized == False:
            self._tec_initialized = True

            if len(selected_port) == 0:
                Log.e(
                    f"ERROR: No active device is currently available for TEC status updates.")
                Log.e(
                    "Please connect a device, hit \"Reset\", and try \"Temp Control\" again.")
                self._tec_stop_thread = True  # queue thread for 'quit' on next update
                self._tec_update_now = False  # invalidate flag to update TEC again
                self.auto_off.emit()  # toggle off button state now
                # fire next instance soon-ish
                QtCore.QTimer.singleShot(500, self.task_update)
                return

            if not self._is_port_available(selected_port):
                Log.e(
                    f"ERROR: The selected device \"{selected_port}\" is no longer available.")
                Log.e(
                    "Please \"Reset\" to detect devices and then try \"Temp Control\" again.")
                self._tec_stop_thread = True  # queue thread for 'quit' on next update
                self._tec_update_now = False  # invalidate flag to update TEC again
                self.auto_off.emit()  # toggle off button state now
                # fire next instance soon-ish
                QtCore.QTimer.singleShot(500, self.task_update)
                return

            # set offsetB, offsetH and offsetC prior to initialization
            if hasattr(Constants, 'temp_offset_both'):
                self._tec_update("=OFFSET {0:2.2f}".format(
                    Constants.temp_offset_both))
                Log.d(TAG, '{1}: Set offsetB={0:+02.2f}'.format(
                    Constants.temp_offset_both, strftime('%Y-%m-%d %H:%M:%S', localtime())))
            if self._tec_stop_thread:
                return
            if hasattr(Constants, 'temp_offset_heat'):
                self._tec_update(
                    "+OFFSET {0:2.2f}".format(Constants.temp_offset_heat))
                Log.d(TAG, '{1}: Set offsetH={0:+02.2f}'.format(
                    Constants.temp_offset_heat, strftime('%Y-%m-%d %H:%M:%S', localtime())))
            if self._tec_stop_thread:
                return
            if hasattr(Constants, 'temp_offset_cool'):
                self._tec_update(
                    "-OFFSET {0:2.2f}".format(Constants.temp_offset_cool))
                Log.d(TAG, '{1}: Set offsetC={0:+02.2f}'.format(
                    Constants.temp_offset_cool, strftime('%Y-%m-%d %H:%M:%S', localtime())))
            if self._tec_stop_thread:
                return

            if (
                hasattr(Constants, 'tune_pid_cp') and
                hasattr(Constants, 'tune_pid_ci') and
                hasattr(Constants, 'tune_pid_cd') and
                hasattr(Constants, 'tune_pid_hp') and
                hasattr(Constants, 'tune_pid_hi') and
                hasattr(Constants, 'tune_pid_hd')
            ):
                self._tec_update("TUNE {0:.3g},{1:.3g},{2:.3g},{3:.3g},{4:.3g},{5:.3g}".format(
                    Constants.tune_pid_cp, Constants.tune_pid_ci, Constants.tune_pid_cd,  # cool PID
                    Constants.tune_pid_hp, Constants.tune_pid_hi, Constants.tune_pid_hd))  # heat PID
                Log.w(TAG, '{6}: Set PID tuning per Constants.py parameters: {0:.3g},{1:.3g},{2:.3g},{3:.3g},{4:.3g},{5:.3g}'.format(
                    Constants.tune_pid_cp, Constants.tune_pid_ci, Constants.tune_pid_cd,  # cool PID
                    Constants.tune_pid_hp, Constants.tune_pid_hi, Constants.tune_pid_hd,  # heat PID
                    strftime('%Y-%m-%d %H:%M:%S', localtime())))
            if self._tec_stop_thread:
                return

        # Attempt to open port and print errors (if any)
        TEC_serial = serial.Serial()
        try:
            # Configure serial port (assume baud to check before update)
            TEC_serial.port = selected_port
            TEC_serial.baudrate = Constants.serial_default_speed  # 115200
            TEC_serial.stopbits = serial.STOPBITS_ONE
            TEC_serial.bytesize = serial.EIGHTBITS
            TEC_serial.timeout = Constants.serial_timeout_ms
            TEC_serial.write_timeout = Constants.serial_writetimeout_ms
            TEC_serial.open()

            # Handle special values, only send to TEC FW if not the current setpoint
            if str(dac).isnumeric():
                if int(dac) < 0 or int(dac) > 60:
                    dac = "OFF"  # turn off if temp is outside valid range
                if int(dac) == 0:
                    dac = 0.25  # temp sensor clips at 0, so target 0.25C
                Log.i(TAG, "Temp Control setpoint: {1}C".format(
                    strftime('%Y-%m-%d %H:%M:%S', localtime()), dac))
                self.infobar_setText.emit("<font color=#0000ff> Infobar </font><font color={}><b>{}</b></font>".format("#ff0000",
                                                                                                                       "Cycling temperature to Temp Control setpoint... please wait..."))
                self._tec_cycling = True

            # Read and show the TEC temp status from the device
            TEC_serial.write("temp {}\n".format(dac).encode())
            timeoutAt = time() + 3
            temp_reply = ""
            lines_in_reply = 11
            # timeout needed if old FW
            while temp_reply.count('\n') < lines_in_reply and time() < timeoutAt:
                while TEC_serial.in_waiting == 0 and time() < timeoutAt:  # timeout needed if old FW:
                    pass
                temp_reply += TEC_serial.read(TEC_serial.in_waiting).decode()

            if time() < timeoutAt:
                temp_reply = temp_reply.split('\n')
                actual_lines = len(temp_reply)
                # ends with blank line (+1)
                sl = actual_lines - (lines_in_reply+1)
                status_line = temp_reply[sl+0]          # line 1
                setpoint_line = temp_reply[sl+1]        # line 2
                power_line = temp_reply[sl+2]           # line 3
                voltage_line = temp_reply[sl+3]         # line 4
                stable_total_line = temp_reply[sl+4]    # line 5
                min_max_line = temp_reply[sl+5]         # line 6
                temp_status_line = temp_reply[sl+6]     # line 7
                ambient_line = temp_reply[sl+7]         # line 8
                temp_line = temp_reply[sl+8]            # line 9
                offsets_line = temp_reply[sl+9]         # line 10 (unused)
                tune_pid_line = temp_reply[sl+10]       # line 11 (unused)
                status_text = status_line.split(':')[1].strip()
                # cycle, wait, close, stable, error
                cycle_stable = status_text.split(',')[0]
                heat_cool_off = status_text.split(',')[1]
                setpoint_val = float(setpoint_line.split(':')[1].strip())
                power_val = int(power_line.split(':')[1].strip())
                voltage_val = voltage_line.split(':')[1].strip()
                voltage_volts = float(voltage_val.split()[0][0:-1])
                voltage_raw = int(voltage_val.split()[1][1:-1])
                stable_total = stable_total_line.split(":")[1].strip()
                stable = int(stable_total.split(',')[0])
                total = int(stable_total.split(',')[1])
                min_max = min_max_line.split(":")[1].strip()
                min = float(min_max.split(',')[0])
                max = float(min_max.split(',')[1])
                if min == 50:
                    min = -1
                if max == 0:
                    max = -1
                temp_status = temp_status_line.split(":")[1].strip()
                ambient = float(ambient_line.split(':')[1].strip())
                temp = float(temp_line.split(':')[1].strip())
                offsets_text = offsets_line.split(':')[1].strip().split(',')
                tune_pid_text = tune_pid_line.split(':')[1].strip().split(',')
                # throw variables into global module scope
                self._tec_status = cycle_stable
                self._tec_state = heat_cool_off
                if "COOL" == self._tec_state:
                    power_val = -power_val
                self._tec_setpoint = setpoint_val
                self._tec_temp = temp
                self._tec_power = power_val
                self._tec_voltage = voltage_val
                self._tec_offset1 = offsets_text[0]  # first: A (always)
                self._tec_offset2 = offsets_text[-1]  # last: M (measure)
                self._tec_pid_tune = []
                for kp in tune_pid_text:
                    self._tec_pid_tune.append(float(kp))

                if "VOLTAGE" == self._tec_state:
                    self._tec_voltage_error_seen = True
                    Log.e(f"External voltage is out of bounds: {voltage_val}")

                # Append to log file for temperature controller
                # checks the path for the header insertion
                tec_log_path = FileStorage.DEV_populate_path(
                    Constants.tec_log_path, 0)
                os.makedirs(os.path.split(tec_log_path)[0], exist_ok=True)
                header_exists = os.path.exists(tec_log_path)
                with open(tec_log_path, 'a') as tempFile:
                    if not header_exists:
                        tempFile.write(
                            "Date/Time,Command,Status/Mode,Power(raw),Stable/Total(sec),Min/Max(C),Temp(C),Ambient(C)\n")
                    log_line = "{},{},{},{},{},{},{},{}\n".format(
                        strftime('%Y-%m-%d %H:%M:%S', localtime()
                                 ),             # Date/Time
                        "SET '{}'".format(
                            dac) if not dac == "" else "GET",     # Command
                        # Status/Mode
                        "{}/{}".format(cycle_stable, heat_cool_off),
                        # Power(raw)
                        power_val,
                        # Stable/Total(sec)
                        "{}/{}".format(stable, total),
                        # Min/Max(C)
                        "{}/{}".format(min, max),
                        # Temp(C)
                        temp,
                        ambient)                                                # Ambient(C)
                    tempFile.write(log_line)
            else:
                Log.e(TAG, "ERROR: Timeout during check and/or update to TEC controller.".format(
                    strftime('%Y-%m-%d %H:%M:%S', localtime())))
        except serialutil.SerialException as e:
            self._tec_state = "OFF" if dac == "OFF" else "ERROR"
            # Ignore the following exception types:
            # errno exception           cause
            ##################################################
            #   2   FileNotFoundError   scanning the source
            #  13   PermissionError     active run in progress
            if not any([s in str(e) for s in ["FileNotFoundError", "PermissionError"]]):
                Log.e(TAG, "ERROR: Serial exception reading port to check and/or update TEC controller.".format(
                    strftime('%Y-%m-%d %H:%M:%S', localtime())))
                if self._tec_debug:
                    raise e  # debug only
        except PermissionError as e:
            self._tec_state = "OFF" if dac == "OFF" else "ERROR"
            if "OFFSET" in dac:  # unable to open port during initialize phase is a hard-stop condition
                # if dac == "OFF" else "ERROR" # always off here, since we are OFF in 500ms
                self._tec_state = "OFF"
                Log.e(
                    f"ERROR: The selected device \"{selected_port}\" cannot be opened. Is the port already open in another program?")
                Log.e(
                    "Please close the device port, hit \"Reset\", and try \"Temp Control\" again.")
                self._tec_stop_thread = True  # queue thread for 'quit' on next update
                self._tec_update_now = False  # invalidate flag to update TEC again
                self.auto_off.emit()  # toggle off button state now
                # fire next instance soon-ish
                QtCore.QTimer.singleShot(500, self.task_update)
        except Exception as e:
            self._tec_state = "OFF" if dac == "OFF" else "ERROR"
            if not any([s in str(e) for s in ["Permission"]]):
                Log.e(TAG, "ERROR: Failure reading port to check and/or update TEC controller.".format(
                    strftime('%Y-%m-%d %H:%M:%S', localtime())))
            else:
                Log.e(TAG, "ERROR: File permission error occurred while logging TEC controller data.".format(
                    strftime('%Y-%m-%d %H:%M:%S', localtime())))
            if self._tec_debug:
                raise e  # debug only
        finally:
            if TEC_serial.is_open:
                TEC_serial.close()

    ###########################################################################
    # Automatically selects the serial ports for Teensy (macox/windows)
    ###########################################################################
    @staticmethod
    def get_ports():
        return serial.enumerate()
        from QATCH.common.architecture import Architecture, OSType
        from serial.tools import list_ports
        if Architecture.get_os() is OSType.macosx:
            import glob
            return glob.glob("/dev/tty.usbmodem*")
        elif Architecture.get_os() is OSType.linux:
            import glob
            return glob.glob("/dev/ttyACM*")
        else:
            found_ports = []
            port_connected = []
            found = False
            ports_avaiable = list(list_ports.comports())
            for port in ports_avaiable:
                if port[2].startswith("USB VID:PID=16C0:0483"):
                    found = True
                    port_connected.append(port[0])
            if found:
                found_ports = port_connected
            return found_ports

    ###########################################################################
    # Checks if the serial port is currently connected
    ###########################################################################
    def _is_port_available(self, port):
        """
        :param port: Port name to be verified.
        :return: True if the port is connected to the host :rtype: bool.
        """
        # dm = Discovery()
        # if self._serial.net_port == None:
        #     net_exists = False
        # else:
        #     net_exists = dm.ping(self._serial.net_port)
        for p in self.get_ports():
            if p == port:
                return True
        # if port == None:
        #     if len(dm.doDiscover()) > 0:
        #         return net_exists
        return False
