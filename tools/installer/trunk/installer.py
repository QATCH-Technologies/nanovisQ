# import ctypes
# from ctypes import wintypes as w

import logging
import os
import shutil
import sys
import requests

# from datetime import date
from io import StringIO, open
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication

import win32api
import win32con
import win32com.client
import win32process

# NOTE: When changing version, also modify `version.rc` file
INSTALLER_VERSION = "v1.0.2.1"

CHECKSUM_METHOD = "hashlib" # enter 'hashlib' or 'certutil'

if CHECKSUM_METHOD == "hashlib":
    import hashlib

app = QApplication(sys.argv)

frozen = 'not'
if getattr(sys, 'frozen', False):
    # we are running in a bundle
    frozen = 'ever so'
    bundle_dir = sys._MEIPASS
else:
    # we are running in a normal Python environment
    bundle_dir = os.path.dirname(os.path.abspath(__file__))

tempicon = os.path.join(bundle_dir, "favicon.ico")

datapath = os.path.expandvars("%LOCALAPPDATA%")
datapath = os.path.join(datapath, "QATCH")
iconpath = os.path.join(datapath, "nanovisQ")
iconfile = os.path.join(iconpath, "favicon.ico")
iconresc = iconfile.replace('\\', '/')

userpath = os.path.expandvars("%USERPROFILE%")
nanopath = os.path.join(userpath, "QATCH nanovisQ")
dinifile = os.path.join(nanopath, "desktop.ini")

log_stream = StringIO()
Log = logging.getLogger(__name__)


class QatchInstaller(QtWidgets.QMessageBox):

    def __init__(self):

        super(QtWidgets.QDialog, self).__init__()
        self.setup_logging()

        # ctypes.windll.kernel32.SetConsoleTitleW("QATCH nanovisQ SW Installer - command line")

        # user32 = ctypes.WinDLL('user32')
        # user32.GetForegroundWindow.argtypes = ()
        # user32.GetForegroundWindow.restype = w.HWND
        # user32.ShowWindow.argtypes = w.HWND,w.BOOL
        # user32.ShowWindow.restype = w.BOOL

        # # From https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-showwindow
        # SW_MAXIMIZE = 3
        # SW_MINIMIZE = 6

        # hWnd = user32.GetForegroundWindow()
        # user32.ShowWindow(hWnd, SW_MINIMIZE)

        Log.debug( "=== DEBUG INFORMATIONS ===")
        Log.debug( f'we are {frozen} frozen')
        Log.debug( f'bundle dir is {bundle_dir}' )
        Log.debug( f'sys.argv[0] is {sys.argv[0]}' )
        Log.debug( f'sys.executable is {sys.executable}' )
        Log.debug( f'os.getcwd is {os.getcwd()}' )

        self.is_finished = False


    def setup_logging(self):
        log_format_file = logging.Formatter(
            fmt = '%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
            datefmt = None)
        log_format_console = logging.Formatter(
            fmt = '%(asctime)s\t%(levelname)s\t%(message)s',
            datefmt = '%Y-%m-%d %I:%M:%S %p')
        log_format_buffer = logging.Formatter(
            fmt = '%(levelname)s\t%(message)s',
            datefmt = None)

        Log.setLevel(logging.DEBUG)

        log_path = os.path.join(os.getcwd(), f"{self.__class__.__name__}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(log_format_file)
        file_handler.setLevel(logging.DEBUG)
        Log.addHandler(file_handler)

        if sys.stdout is None:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format_console)
        console_handler.setLevel(logging.INFO)
        Log.addHandler(console_handler)

        buffer_handler = logging.StreamHandler(log_stream)
        buffer_handler.setFormatter(log_format_buffer)
        buffer_handler.setLevel(logging.WARNING)
        Log.addHandler(buffer_handler)

        #Logger.d("Added handlers successfully")
        Log.debug("Added logging handlers")


    def run(self):
        while True:

            if self.is_finished:
                # spin your wheels forever until the main QATCH app is closed, 
                # then this installer will close too
                handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION, False, self.app_pid)

                # Wait for process to exit (blocking)
                Log.warning("Waiting for main app to close before closing the installer...")
                while True:
                    exit_code = win32process.GetExitCodeProcess(handle)
                    # STILL_ACTIVE = 259
                    if exit_code != 259:
                        break  # app closed
                Log.warning("Process has closed; installer closing...")

                # Close installer, now that application has closed
                win32api.CloseHandle(handle)
                os._exit(0)

            self.response = None
            self.exception = None

            try:
                # self.app = QApplication(sys.argv)
                # box = QtWidgets.QMessageBox(None)
                self.setWindowIcon(QtGui.QIcon(tempicon))
                self.setIconPixmap(QtGui.QPixmap(tempicon))
                self.setWindowTitle("QATCH nanovisQ | Install/Uninstall")
                # self.setGeometry(left, top, width, height)
                self.setText("<b>Thank you for using QATCH nanovisQ software!</b><br/><br/>" +
                            "How would you like to modify QATCH nanovisQ?<br/><br/>" +
                            "Please select an option to proceed.<br/><br/>")
                
                self.setDetailedText("")
                self.setStandardButtons(QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No|QtWidgets.QMessageBox.Cancel)
                self.setDefaultButton(QtWidgets.QMessageBox.Yes)

                # Create radio buttons
                self.install_options = QtWidgets.QGroupBox("Install options")
                self.online = QtWidgets.QRadioButton("Check online for latest versions")
                self.offline = QtWidgets.QRadioButton("Install from a software bundle")
                self.online.setChecked(True)

                self.install_layout = QtWidgets.QVBoxLayout(self.install_options)
                self.install_layout.addWidget(self.online)
                self.install_layout.addWidget(self.offline)

                # Add them to the layout
                self.layout().addWidget(self.install_options, 1, 2, 1, 1)

                # Add version label to button row
                self.button_label = QtWidgets.QLabel(INSTALLER_VERSION)
                self.button_label.setContentsMargins(0, 10, 0, 0)
                self.button_label.setEnabled(False)

                # Find the row where buttons are (usually last row)
                row_count = self.layout().rowCount()
                self.layout().addWidget(self.button_label, row_count - 1, 0, QtCore.Qt.AlignLeft)

                install = self.button(QtWidgets.QMessageBox.Yes)
                install.setText('Install')
                uninstall = self.button(QtWidgets.QMessageBox.No)
                uninstall.setText('Uninstall')
                cancel = self.button(QtWidgets.QMessageBox.Cancel)
                self.exec_()
                # QtWidgets.QMessageBox.question(self, "QATCH Installer", "Would you like to install or uninstall the QATCH nanovisQ software from this computer?", )

                if self.clickedButton() == install:
                    if self.online.isChecked():
                        self.install_online()
                    else:
                        self.install_offline()

                if self.clickedButton() == uninstall:
                    self.uninstall()

                if self.clickedButton() == cancel:
                    break

                # Garbage collect combobox
                widgetAt = self.findChild(QtWidgets.QGroupBox)
                self.install_options.hide()
                self.layout().removeWidget(widgetAt)
                widgetAt.deleteLater()

                # Garbage collect version label
                widgetsAt = self.findChildren(QtWidgets.QLabel)
                for w in widgetsAt:
                    if w == self.button_label:
                        widgetAt = w
                        break
                self.button_label.hide()
                self.layout().removeWidget(widgetAt)
                widgetAt.deleteLater()


            except Exception as e:
                # Log.error("An unhandled exception occurred:")
                # Log.error(f"{e.__class__.__name__}: {e}")

                limit = None
                t, v, tb = sys.exc_info()
                from traceback import format_tb
                a_list = ['Traceback (most recent call last):']
                a_list = a_list + format_tb(tb, limit)
                a_list.append(f"{t.__name__}: {str(v)}")
                for line in a_list:
                    Log.error(line)

                # box = QtWidgets.QMessageBox(None)
                self.setIcon(QtWidgets.QMessageBox.Critical)
                self.setWindowIcon(QtGui.QIcon(tempicon))
                self.setWindowTitle("QATCH nanovisQ | Unhandled Exception")
                self.setText("An unhandled exception has occurred during execution.\nSee details for more information.")
                self.setDetailedText(log_stream.getvalue())
                self.setStandardButtons(QtWidgets.QMessageBox.Ok)
                self.exec_()

            # Clear log_stream (if not empty) for next instance
            if log_stream.getvalue() != "":
                log_stream.truncate(0)
                log_stream.seek(0)


    def on_request_response(self, response):
        self.response = response
        self.worker_finished = True


    def on_request_progress(self, message, percentage):
        self.progressBar.setLabelText(message)
        if 0 <= percentage <= 100:
            self.progressBar.setRange(0, 100)
            self.progressBar.setValue(percentage)
        else:
            self.progressBar.setRange(0, 0)
            self.progressBar.setValue(0)

    def on_launcher_finish(self, success, pid):
        Log.info(f"PID: {pid}")
        self.app_pid = pid
        self.worker_finished = True
        if success and pid > 0:
            self.is_finished = True  # wait in run() for main app to close
        else:
            Log.error("Process either failed to launch, or returned an invalid process handle.")
            self.show_exception("Failed to launch application.")

    def install_online(self):

        # Flag to skip offline install step if online step fails or is canceled.
        run_offline_installer = True

        try:
            GIT_ROOT = 'https://github.com/QATCH-Technologies/nanovisQ/releases'
            SHOW_BETA = False

            # Create progress dialog for version query progress
            self.progressBar = QtWidgets.QProgressDialog(f"QATCH nanovisQ", "Cancel", 0, 100, self)
            self.progressBar.setWindowIcon(QtGui.QIcon(tempicon))
            self.progressBar.setWindowTitle(f"Download QATCH nanovisQ software")
            self.progressBar.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
            self.progressBar.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
            # self.progressBar.setWindowFlag(QtCore.Qt.Dialog, True)
            # self.progressBar.setCancelButton(None)
            self.progressBar.canceled.disconnect()
            self.progressBar.canceled.connect(self.download_cancel)
            self.progressBar.setFixedSize(int(self.progressBar.width()*1.5), int(self.progressBar.height()*1.1))
            self.progressBar.show()

            # Get list of versions from server (read "tags.txt" from "latest" url)
            self.worker = RequestWorker(url=f'{GIT_ROOT}/latest', 
                                        allow_redirects=True)
            self.worker.finished.connect(self.on_request_response)
            # self.worker.finished.connect(self.worker.deleteLater)
            self.worker.exception.connect(self.show_exception)
            self.worker.progress.connect(self.on_request_progress)

            self.worker_finished = False
            self.worker.start()
            while not self.worker_finished:
                 # wait here until finished querying
                QApplication.processEvents()

            response = self.response
            response.raise_for_status()

            latest_tag = None
            if response.history:
                latest_tag = response.url.split("/")[-1]
                Log.debug(f"Latest tag: {latest_tag}")
            else:
                raise requests.HTTPError("No redirect where one was expected.")

            releases = None
            if latest_tag and latest_tag.startswith('v'):
                # Get the list of tags available for the latest release
                self.worker = RequestWorker(url=f'{GIT_ROOT}/download/{latest_tag}/tags.txt')
                self.worker.finished.connect(self.on_request_response)
                # self.worker.finished.connect(self.worker.deleteLater)
                self.worker.exception.connect(self.show_exception)
                self.worker.progress.connect(self.on_request_progress)

                self.worker_finished = False
                self.worker.start()
                while not self.worker_finished:
                    # wait here until finished querying
                    QApplication.processEvents()

                response = self.response
                response.raise_for_status()

                releases = response.content.decode().splitlines()
                releases.reverse()

                if not SHOW_BETA:
                    for tag in releases.copy():
                        if tag.find('r') == -1 or not tag.startswith('v'):
                            releases.remove(tag)
            else:
                raise requests.HTTPError("Latest tag is not a valid version.")

            # Hide progress dialog for version query progress
            self.progressBar.reset()

            selected_version = None
            if releases and len(releases):
                Log.debug(f"Found releases: {releases}")
                latest_label = " (latest)"
                releases[0] += latest_label

                self.setWindowIcon(QtGui.QIcon(tempicon))
                self.setWindowTitle("QATCH nanovisQ | Install Version")
                self.setText("Select a software version to install:")
                self.setIcon(QtWidgets.QMessageBox.Question)
                self.setStandardButtons(QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
                button1 = self.button(QtWidgets.QMessageBox.Yes)
                button1.setText('Install')
                button2 = self.button(QtWidgets.QMessageBox.No)
                button2.setText('Cancel')
                self.setDefaultButton(QtWidgets.QMessageBox.Yes)
                self.comboBox = QtWidgets.QComboBox(None)
                self.comboBox.addItems(releases)
                self.comboBox.setCurrentIndex(0)
                self.layout().addWidget(self.comboBox, 1, 2, 1, 1) # widget, row, col, rowSpan, colSpan
                self.exec_() # wait for user interaction
                
                # remove custom combobox widget from layout
                selected_version = self.comboBox.currentText().replace(latest_label, "")
                widgetAt = self.findChild(QtWidgets.QComboBox)
                self.comboBox.hide()
                self.layout().removeWidget(widgetAt)
                widgetAt.deleteLater()

                if self.clickedButton() != button1:
                    Log.warning("User canceled or did not make a selection.")

                    # box = QtWidgets.QMessageBox(None)
                    self.setIcon(QtWidgets.QMessageBox.Critical)
                    self.setWindowIcon(QtGui.QIcon(tempicon))
                    self.setWindowTitle("QATCH nanovisQ | Abort")
                    self.setText("User canceled install. Install aborted.")
                    self.setDetailedText(log_stream.getvalue())
                    self.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    self.exec_()
                    
                    raise UserWarning(False)  # skip download and install
                else:
                    Log.info(f"Selected version to install: {selected_version}")
            else:
                raise requests.HTTPError("No valid releases found on server.")

            # Show progress dialog for download progress
            self.progressBar.show()
            
            if selected_version:
                # Download bundle to CWD
                self.worker = RequestWorker(
                    url=f'{GIT_ROOT}/download/{selected_version}/nanovisQ_SW_{selected_version}_exe.zip', 
                    stream=True)
                self.worker.finished.connect(self.on_request_response)
                # self.worker.finished.connect(self.worker.deleteLater)
                self.worker.exception.connect(self.show_exception)
                self.worker.progress.connect(self.on_request_progress)

                self.worker_finished = False
                self.worker.start()
                while not self.worker_finished:
                    # wait here until finished querying
                    QApplication.processEvents()

                response = self.response
                response.raise_for_status()

                file_local = os.path.join(os.getcwd(), f"nanovisQ_SW_{selected_version}_exe.zip")
                size = int(response.headers['Content-Length'])
                if os.path.exists(file_local):
                    # Check if the file size matches; if so, ask if they want to download again.
                    if os.path.getsize(file_local) == size:
                        Log.debug(f"Software bundle {selected_version} is already downloaded.")
                        # Hide progress dialog for version query progress
                        self.progressBar.reset()
                        ans = QtWidgets.QMessageBox.question(
                            None,
                            "QATCH nanovisQ | Re-Download",
                            f"This software version is already downloaded:\n{selected_version}\n\nDo you want to download it again?",
                            buttons=QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No|QtWidgets.QMessageBox.Cancel,
                            defaultButton=QtWidgets.QMessageBox.No)
                        if ans == QtWidgets.QMessageBox.Yes:
                            Log.debug("User wants to download the software bundle again.")
                            pass  # continue to delete and re-download file
                        if ans == QtWidgets.QMessageBox.No:
                            Log.debug("User wants to use the existing software bundle.")
                            raise UserWarning(True)  # skip download, continue to offline installer
                        if ans == QtWidgets.QMessageBox.Cancel:
                            Log.debug("User wants to cancel the download and install entirely.")

                            # box = QtWidgets.QMessageBox(None)
                            self.setIcon(QtWidgets.QMessageBox.Critical)
                            self.setWindowIcon(QtGui.QIcon(tempicon))
                            self.setWindowTitle("QATCH nanovisQ | Abort")
                            self.setText("User canceled download. Install aborted.")
                            self.setDetailedText(log_stream.getvalue())
                            self.setStandardButtons(QtWidgets.QMessageBox.Ok)
                            self.exec_()

                            raise UserWarning(False)  # skip download and install
                        # Show progress dialog for version query progress
                        self.progressBar.show()
                    Log.debug(f"Deleting {file_local}...")
                    os.remove(file_local)

                self.worker = RequestWorker(response=response, file_local=file_local)
                # self.worker.finished.connect(self.on_request_response)
                # self.worker.finished.connect(self.worker.deleteLater)
                self.worker.exception.connect(self.show_exception)
                self.worker.progress.connect(self.on_request_progress)

                self.worker.start()
                self.progressBar.exec_()  # wait here until finished downloading

                if "cancel" in self.progressBar.labelText().lower():
                    Log.debug("User hit cancel on the download progress bar.")

                    # box = QtWidgets.QMessageBox(None)
                    self.setIcon(QtWidgets.QMessageBox.Critical)
                    self.setWindowIcon(QtGui.QIcon(tempicon))
                    self.setWindowTitle("QATCH nanovisQ | Abort")
                    self.setText("User canceled download. Install aborted.")
                    self.setDetailedText(log_stream.getvalue())
                    self.setStandardButtons(QtWidgets.QMessageBox.Ok)
                    self.exec_()

                    raise UserWarning(False)  # skip download and install
            else:
                raise requests.HTTPError("Selected version to install not specified.")

        except requests.HTTPError as http:
            Log.error(f"Requests error: {http}")
            run_offline_installer = False

        except UserWarning as w:
            pass  # user hit cancel (or skipped re-download of existing bundle)
            run_offline_installer = w.args[0]

        finally:
            # Hide progress bar
            self.progressBar.reset()
        
        # Call offline installer to install downloaded bundle
        if run_offline_installer:
            self.install_offline(search_for_bundles=True)

    def install_offline(self, search_for_bundles: bool = False):

        ### create application data directory
        os.makedirs(iconpath, exist_ok=True)

        ### move favicon.ico to data path
        shutil.copy(tempicon, iconfile)

        ### create software directory in user's home path
        os.makedirs(nanopath, exist_ok=True)

        try:
            # specify software path as normal to write desktop.ini file (if exists)
            win32api.SetFileAttributes(nanopath, win32con.FILE_ATTRIBUTE_NORMAL)

            ### create desktop.ini for custom icon
            with open(dinifile, "w") as dini:
                dini.write("[.ShellClassInfo]\n")
                dini.write(f"IconResource={iconresc},0")

            ### specify path as read-only to tell windows to read desktop.ini file
            win32api.SetFileAttributes(nanopath, win32con.FILE_ATTRIBUTE_READONLY)

        except Exception as e:
            Log.error("ERROR: Unable to create desktop.ini file and set a custom folder icon.")
            Log.error("Error Reason: {e}")

        copy_src = os.getcwd()
        copy_dst = nanopath
        found = False

        if search_for_bundles:
            most_recent = 0
            for file in os.listdir(copy_src):
                if os.path.isfile(file) and file.find("nanovisQ") >= 0 and file.endswith("_exe.zip"):
                    modified_at = os.path.getmtime(file)
                    if modified_at > most_recent:
                        most_recent = modified_at
                        copy_src = os.path.join(copy_src, file)
                        copy_dst = os.path.join(nanopath, file[0:-4]) # drop '.zip'
                        found = True

        if not found:
            fname = QtWidgets.QFileDialog.getOpenFileName(None,
                                                          'Select QATCH software ZIP package...',
                                                          copy_src,
                                                          "Zip files (*.zip)")
            fullfile = os.path.abspath(fname[0]) # drop file filter and format slashes to system standard
            file = os.path.basename(fullfile) # just the filename, no path
            if os.path.isfile(fullfile) and file.find("nanovisQ") >= 0 and file.endswith("_exe.zip"):
                copy_src = fullfile # already conformed to abspath() format
                copy_dst = os.path.join(nanopath, file[0:-4]) # drop '.zip'
                found = True

        proceed = False
        if found:
            version = os.path.split(copy_dst)[1]
            if os.path.exists(copy_dst):
                ans = QtWidgets.QMessageBox.question(None,
                                                     "QATCH nanovisQ | Re-Install",
                                                     f"This software version is already installed:\n{version}\n\nDo you want to re-install it?",
                                                     buttons=QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No|QtWidgets.QMessageBox.Cancel,
                                                     defaultButton=QtWidgets.QMessageBox.No)
            else:
                ans = QtWidgets.QMessageBox.question(None,
                                                     "QATCH nanovisQ | Re-Install",
                                                     f"The following software version is ready to install:\n{version}\n\nDo you want to install it?",
                                                     buttons=QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No|QtWidgets.QMessageBox.Cancel,
                                                     defaultButton=QtWidgets.QMessageBox.Yes)
            if ans == QtWidgets.QMessageBox.Yes:
                proceed = True

            if ans == QtWidgets.QMessageBox.No:
                Log.warning("User declined install.")

                # box = QtWidgets.QMessageBox(None)
                self.setIcon(QtWidgets.QMessageBox.Critical)
                self.setWindowIcon(QtGui.QIcon(tempicon))
                self.setWindowTitle("QATCH nanovisQ | Abort")
                self.setText("User declined install. Install aborted.")
                self.setDetailedText(log_stream.getvalue())
                self.setStandardButtons(QtWidgets.QMessageBox.Ok)
                self.exec_()

            if ans == QtWidgets.QMessageBox.Cancel:
                Log.warning("User aborted install.")

                # box = QtWidgets.QMessageBox(None)
                self.setIcon(QtWidgets.QMessageBox.Critical)
                self.setWindowIcon(QtGui.QIcon(tempicon))
                self.setWindowTitle("QATCH nanovisQ | Abort")
                self.setText("User aborted install. Install aborted.")
                self.setDetailedText(log_stream.getvalue())
                self.setStandardButtons(QtWidgets.QMessageBox.Ok)
                self.exec_()

                return
        else:
            Log.error("No software bundle found. Cannot install.")

            # box = QtWidgets.QMessageBox(None)
            self.setIcon(QtWidgets.QMessageBox.Critical)
            self.setWindowIcon(QtGui.QIcon(tempicon))
            self.setWindowTitle("QATCH nanovisQ | No Build Found")
            self.setText("No software bundle found. Cannot install.")
            self.setDetailedText("Could not find a valid ZIP file containing \"nanovisQ\" in the filename. Please try again.")
            self.setStandardButtons(QtWidgets.QMessageBox.Ok)
            self.exec_()
        
            return

        ## START OF WORKER TASK ###

        self.progressBar = QtWidgets.QProgressDialog(f"QATCH nanovisQ", "Cancel", 0, 100, self)
        self.progressBar.setWindowIcon(QtGui.QIcon(tempicon))
        self.progressBar.setWindowTitle(f"Installing {version}")
        self.progressBar.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.progressBar.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        # self.progressBar.setWindowFlag(QtCore.Qt.Dialog, True)
        # self.progressBar.setCancelButton(None)
        self.progressBar.canceled.disconnect()
        self.progressBar.canceled.connect(self.install_cancel)
        self.progressBar.setFixedSize(int(self.progressBar.width()*1.5), int(self.progressBar.height()*1.1))
        # self.progressBar.show()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.setInterval(50) # set in update_progress
        self.timer.start()

        self.worker = InstallWorker(proceed, copy_src, copy_dst)
        self.worker.finished.connect(self.progressBar.reset)
        self.worker.finished.connect(self.post_install_check)
        self.worker.finished.connect(self.timer.stop)
        self.worker.finished.connect(self.close)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.exception.connect(self.show_exception)
        self.worker.progress.connect(self.update_progress)

        self.worker.start()
        self.progressBar.exec_() # wait here until finished installing

        # self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        if self.worker._cancel:
            self.setIcon(QtWidgets.QMessageBox.Information)
            self.setWindowIcon(QtGui.QIcon(tempicon))
            self.setWindowTitle("QATCH nanovisQ | Canceled")
            self.setText("Installation canceled.")
            self.setDetailedText("")
            self.setStandardButtons(QtWidgets.QMessageBox.Ok)
            self.exec_()

       
    def download_cancel(self):
        self.update_progress("Canceling installation...", 99)
        cancelButton = self.progressBar.findChild(QtWidgets.QPushButton)
        cancelButton.setEnabled(False)
        self.worker.cancel()
        

    def install_cancel(self):
        self.update_progress("Canceling installation...", 99)
        cancelButton = self.progressBar.findChild(QtWidgets.QPushButton)
        cancelButton.setEnabled(False)
        self.worker.cancel()


    def show_exception(self, except_str : str = None):
        self.worker_finished = True

        box = QtWidgets.QMessageBox(None)
        box.setIcon(QtWidgets.QMessageBox.Critical)
        box.setWindowIcon(QtGui.QIcon(tempicon))
        box.setWindowTitle("QATCH nanovisQ | Unhandled Exception")
        box.setText("<b>An unhandled exception has occurred during execution:</b><br/>" +
                    f"{except_str}<br/><br/>" +
                    "See details for more information.")
        box.setDetailedText(log_stream.getvalue())
        box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        box.exec_()


    def post_install_check(self):

        success = self.worker.success
        copy_dst = self.worker.copy_dst
        version = os.path.split(copy_dst)[1]

        if success and os.path.exists(copy_dst) and log_stream.getvalue() == "":
            QtWidgets.QMessageBox.information(None, 
                                                    "QATCH nanovisQ | Install", 
                                                    "Installer was successful.",
                                                    buttons=QtWidgets.QMessageBox.Ok)
                                                    # "Installer was successful.\n\nWould you like to launch it now?", 
                                                    # buttons=QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)

            desktop = None
            userdir = os.path.expanduser('~')
            option1 = os.path.join(userdir, "Desktop")
            option2 = os.path.join(userdir, "OneDrive", "Desktop")
            if os.path.exists(option1):
                desktop = option1
            elif os.path.exists(option2):
                desktop = option2
            else:
                Log.warning("Unable to find desktop shortcut. Attempting 'launch.bat' launcher.")
            launcher = os.path.join(copy_dst, "launch.bat")
            if desktop == None:
                if not os.path.exists(launcher):
                    launcher = os.path.join(copy_dst, "QATCH nanovisQ.lnk")
                if not os.path.exists(launcher):
                    launcher = os.path.join(copy_dst, "QATCH nanovisQ.exe")
                Log.info("Launching application...")
                os.startfile(launcher, cwd=copy_dst,)
            else:
                _nanopath = os.path.join(copy_dst, "QATCH nanovisQ.exe")
                linkpath = os.path.join(desktop, "QATCH nanovisQ.lnk")
                shell = win32com.client.Dispatch("WScript.Shell")
                shortcut = shell.CreateShortCut(linkpath)
                shortcut.TargetPath = _nanopath
                shortcut.Arguments = "" # explicity remove '-m QATCH' if upgrading PY install to EXE
                shortcut.Description = version #"(C) QATCH Technologies LLC"
                shortcut.IconLocation = f"{_nanopath},0"
                shortcut.WorkingDirectory = os.path.join(os.path.dirname(desktop), "Documents", "QATCH nanovisQ")
                shortcut.save() # create shortcut link
                shutil.copy(linkpath, copy_dst) # copy to local build directory too
                os.remove(launcher) # delete launch.bat
                if not os.path.exists(shortcut.WorkingDirectory):
                    Log.info("Working Directory does not exist. Creating now.")
                    Log.info(f"Working Directory: {shortcut.WorkingDirectory}")
                    os.makedirs(shortcut.WorkingDirectory)
                if QtWidgets.QMessageBox.question(None,
                                                    "QATCH nanovisQ | Launch",
                                                    "Would you like to launch the new version now?",
                                                    buttons=QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No,
                                                    defaultButton=QtWidgets.QMessageBox.Yes) == QtWidgets.QMessageBox.Yes:
                    Log.info("Launching application...")
                    pb = QtWidgets.QProgressDialog(os.path.basename(copy_dst), None, 0, 100, self)
                    pb.setWindowIcon(QtGui.QIcon(tempicon))
                    pb.setWindowTitle(f"Launching {os.path.basename(copy_dst).replace('QATCH', '')}...               ")
                    pb.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
                    pb.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
                    pb.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
                    pb.setCancelButton(None)
                    pb.canceled.disconnect()
                    pb.setFixedSize(int(pb.width()*1.5), int(pb.height()*1.1)) # QApplication.style().pixelMetric(QtWidgets.QStyle.PM_TitleBarHeight))
                    pb.show()
                    pb.setRange(0, 0)
                    # os.startfile(_nanopath, cwd=shortcut.WorkingDirectory,)
                    self.launcher = LaunchWorker(filepath=_nanopath, cwd=shortcut.WorkingDirectory,)
                    self.launcher.finished.connect(self.on_launcher_finish)
                    self.launcher.finished.connect(self.launcher.deleteLater)
                    self.launcher.exception.connect(self.show_exception)
                    # self.launcher.progress.connect(self.on_request_progress)

                    self.worker_finished = False
                    self.elapsed_timer.restart()
                    while self.elapsed_timer.elapsed() < 3000:
                        # wait at least 3 seconds to avoid quick window flashes
                        QApplication.processEvents()
                    self.launcher.start()
                    while not self.worker_finished:
                        # wait here until thread finished
                        QApplication.processEvents()
                    # os.startfile(_nanopath, cwd=shortcut.WorkingDirectory,)
                    # # CREATE_NEW_PROCESS_GROUP prevents it from being killed with parent
                    # DETACHED_PROCESS = 0x00000008
                    # CREATE_NEW_PROCESS_GROUP = 0x00000200
                    # # CREATE_NO_WINDOW = 0x08000000
                    # import subprocess
                    # subprocess.Popen(
                    #     f'START "" /D "{shortcut.WorkingDirectory}" "{_nanopath}"', shell=True,
                    #     creationflags=CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS | subprocess.CREATE_BREAKAWAY_FROM_JOB | subprocess.CREATE_NEW_CONSOLE)
                    # QtCore.QProcess.startDetached(os.path.normpath(_nanopath), [], os.path.normpath(shortcut.WorkingDirectory))
                    # def launch_detached(filepath, working_dir=None):
                    #     import ctypes
                    #     if working_dir is None:
                    #         working_dir = os.path.dirname(filepath)
                        
                    #     # ShellExecuteW - more detached than CreateProcess
                    #     result = ctypes.windll.shell32.ShellExecuteW(
                    #         None,           # hwnd
                    #         "open",         # operation
                    #         filepath,       # file
                    #         None,           # parameters
                    #         working_dir,    # directory
                    #         1               # SW_SHOWNORMAL
                    #     )
                    #     return result > 32  # Success if > 32
                    # def launch_detached(filepath, working_dir=None):
                    #     if working_dir is None:
                    #         working_dir = os.path.dirname(os.path.abspath(filepath))
                        
                    #     # Normalize paths
                    #     filepath = os.path.normpath(filepath)
                    #     working_dir = os.path.normpath(working_dir)
                        
                    #     success, pid = QtCore.QProcess.startDetached(filepath, [], working_dir)
                    #     Log.info(f"PID: {pid}")
                    #     self.app_pid = pid
                    #     return success and pid > 0
                    # launch_detached(_nanopath, shortcut.WorkingDirectory)
                    # self.elapsed_timer.restart()
                    # while self.elapsed_timer.elapsed() < 3000:
                    #     # wait at least 3 seconds to avoid quick window flashes
                    #     QApplication.processEvents()
                    # self.launcher.start()
                    # while not self.launcher.isRunning():
                    #     # wait here until thread starts
                    #     QApplication.processEvents()
                    # while not self.launcher.isFinished():
                    #     # wait here until thread finished
                    #     QApplication.processEvents()
                    pb.reset()
                    # os._exit(0) # app.quit() # stop installer on app launch
                    # self.is_finished = True
                else:
                    QtWidgets.QMessageBox.information(None,
                                                        "QATCH nanovisQ | Launch",
                                                        "Use the desktop link to launch the software when ready.",
                                                        buttons=QtWidgets.QMessageBox.Ok)
            
        else:
            # box = QtWidgets.QMessageBox(None)
            self.setIcon(QtWidgets.QMessageBox.Critical)
            self.setWindowTitle("QATCH nanovisQ | Install Error")
            self.setText("Installer was not successful.\nSee details for more information.")
            self.setDetailedText(log_stream.getvalue())
            self.setStandardButtons(QtWidgets.QMessageBox.Ok)
            self.exec_()
            # QtWidgets.QMessageBox.critical(None, 
            #                                "QATCH nanovisQ | Install Error", 
            #                                "Installer was not successful.\nSee details for warning(s)/error(s).",
            #                                buttons=QtWidgets.QMessageBox.Ok)
            # QtWidgets.QMessageBox.critical(None,
            #                                "QATCH nanovisQ | Warnings and Errors",
            #                                log_stream.getvalue(),
            #                                buttons=QtWidgets.QMessageBox.Ok)

        Log.info("Install finished.")


    def update_progress(self, label_str : str = None, progress_pct : int = None):

        need_repaint = False
        if label_str != None and self.worker._cancel == False:
            need_repaint = True
            self.progressBar.setLabelText(label_str)
        if progress_pct != None:
            need_repaint = True
            self.progressBar.setValue(progress_pct)
        elif self.progressBar.value() < 50:
            # self.timer.setInterval(50) # one speed for copy phase
            try:
                path = self.worker.copy_dst
                actual = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
                expected = os.path.getsize(self.worker.copy_src)
                pct = max(self.progressBar.minimum(), min(self.progressBar.maximum()-1, int(((100 * actual) / expected) / 2)))
                self.progressBar.setValue(pct) # self.progressBar.value() + 1)
                need_repaint = True
                # Log.info(f"Copy dest: {path}")
                # Log.info(f"Actual size: {actual}b")
                # Log.info(f"Expected size: {expected}b")
                # Log.info(f"Percent: {pct}%")
            except Exception as e:
                # Log.info(f"Exception: {e}")
                pass
            # if self.progressBar.value() == 50:
            #     self.progressBar.setLabelText(self.progressBar.labelText() + " (takes a few seconds)")
        if not  hasattr(self, "elapsed_timer"):
            # Create the timer and start it immediately
            self.elapsed_timer = QtCore.QElapsedTimer()
            self.elapsed_timer.start()
        elif self.elapsed_timer.elapsed() > 1000:
            if self.progressBar.value() < self.progressBar.maximum():
                self.progressBar.setValue(self.progressBar.value() + 1)
            else:
                self.progressBar.setRange(0, 0)  # indeterminate
                self.progressBar.setLabelText("This is taking longer than usual...")
            need_repaint = True
        # elif self.progressBar.value() >= 51 and self.progressBar.value() < 99:
        #     need_repaint = True
        #     # self.timer.setInterval(200) # another speed for checksum
        #     self.progressBar.setValue(self.progressBar.value() + 1)
        #     if self.progressBar.value() == 99:
        #         self.progressBar.setLabelText(self.progressBar.labelText() + " (takes a minute)")
        if need_repaint:
            self.elapsed_timer.restart()
            self.progressBar.repaint()


    @staticmethod
    def recursive_remove(folder):

        try:
            folders = []
            for root, dirs, files in os.walk(folder):
                for name in files:
                    Log.info(f"Remove file: {os.path.join(root, name)}")
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    folders.append(os.path.join(root, name))
            for name in folders:
                # removedirs handles nested directories
                dups = len([True for i in folders if i.startswith(name)])
                if not dups == 1: # skip parent dirs (with dups)
                    Log.debug(f"Skip folder: {name} - it will be deleted later")
                else:
                    Log.info(f"Remove folder: {name}")
                    try:
                        os.removedirs(name)
                    except Exception as e:
                        Log.error(f"Error removing folder: {name}")
                        Log.error(f"Error Details: {e}")
            if os.path.exists(folder):
                Log.info(f"Remove folder: {folder}")
                try:
                    os.removedirs(folder)
                except Exception as e:
                    Log.error(f"Error removing folder: {folder}")
                    Log.error(f"Error Details: {e}")

        except Exception as e:
            # Log.error("An unhandled exception occurred:")
            # Log.error(f"{e.__class__.__name__}: {e}")

            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb
            a_list = ['Traceback (most recent call last):']
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.error(line)


    def uninstall(self):

        number_of_builds_installed = 0
        installed_versions = []  
        if os.path.isdir(nanopath):    
            for file in os.listdir(nanopath):
                fullfile = os.path.join(nanopath, file)
                if os.path.isdir(fullfile):
                    number_of_builds_installed += 1
                    installed_versions.append(file)
        Log.info(f"Number of builds found: {number_of_builds_installed}")

        ask_for_specific_version = False
        if number_of_builds_installed > 1:
            ans = QtWidgets.QMessageBox.question(None, 
                                                 "QATCH nanovisQ | Uninstall All", 
                                                 "<b>Would you like to delete all installed versions?</b><br/>" +
                                                 "If you only want to uninstall a single version, select 'No'.<br/><br/>" +
                                                 f"'Yes': All <b>{number_of_builds_installed}</b> installed versions will be uninstalled.<br/>" +
                                                 "'No': You will be prompted which version to uninstall.", 
                                                 buttons=QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No|QtWidgets.QMessageBox.Cancel,
                                                 defaultButton=QtWidgets.QMessageBox.No)
            if ans == QtWidgets.QMessageBox.Yes:
                pass
            if ans == QtWidgets.QMessageBox.No:
                ask_for_specific_version = True
            if ans == QtWidgets.QMessageBox.Cancel:
                Log.warning("Uninstall canceled. No changes were made.")

                # box = QtWidgets.QMessageBox(None)
                self.setIcon(QtWidgets.QMessageBox.Critical)
                self.setWindowIcon(QtGui.QIcon(tempicon))
                self.setWindowTitle("QATCH nanovisQ | Abort")
                self.setText("User canceled uninstall. Uninstall aborted.")
                self.setDetailedText(log_stream.getvalue())
                self.setStandardButtons(QtWidgets.QMessageBox.Ok)
                self.exec_()
                
                return
            
        if ask_for_specific_version:
            # box = QtWidgets.QMessageBox(None)
            self.setWindowIcon(QtGui.QIcon(tempicon))
            self.setWindowTitle("QATCH nanovisQ | Uninstall Version")
            self.setText("Select a software version to uninstall:")
            self.setIcon(QtWidgets.QMessageBox.Question)
            self.setStandardButtons(QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
            button1 = self.button(QtWidgets.QMessageBox.Yes)
            button1.setText('Uninstall')
            button2 = self.button(QtWidgets.QMessageBox.No)
            button2.setText('Cancel')
            self.setDefaultButton(QtWidgets.QMessageBox.Yes)
            self.comboBox = QtWidgets.QComboBox(None)
            self.comboBox.addItems(installed_versions)
            self.comboBox.setCurrentIndex(0)
            self.layout().addWidget(self.comboBox, 1, 2, 1, 1) # widget, row, col, rowSpan, colSpan
            self.exec_() # wait for user interaction
            
            # remove custom combobox widget from layout
            selected_version = self.comboBox.currentText()
            widgetAt = self.findChild(QtWidgets.QComboBox)
            self.comboBox.hide()
            self.layout().removeWidget(widgetAt)
            widgetAt.deleteLater()

            if self.clickedButton() != button1:
                Log.warning("User canceled or did not make a selection.")

                # box = QtWidgets.QMessageBox(None)
                self.setIcon(QtWidgets.QMessageBox.Critical)
                self.setWindowIcon(QtGui.QIcon(tempicon))
                self.setWindowTitle("QATCH nanovisQ | Abort")
                self.setText("User canceled uninstall. Uninstall aborted.")
                self.setDetailedText(log_stream.getvalue())
                self.setStandardButtons(QtWidgets.QMessageBox.Ok)
                self.exec_()
                
                return
            else:
                Log.info(f"Selected version to uninstall: {selected_version}")
                _nanopath = os.path.join(nanopath, selected_version)

            ans = QtWidgets.QMessageBox.No # do not uninstall data if only uninstalling single version
        else:
            _nanopath = nanopath

            ans = QtWidgets.QMessageBox.question(None, 
                                                 "QATCH nanovisQ | Uninstall", 
                                                 "<b>Would you like to delete all application data too?</b><br/>" +
                                                 "If you plan to re-install, you probably want to select 'No'.<br/><br/>" +
                                                 "This data includes any user profiles, configs, and/or saved preferences.<br/>" +
                                                 "This data does <B>NOT</B> include any previously logged data or results.", 
                                                 buttons=QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No|QtWidgets.QMessageBox.Cancel,
                                                 defaultButton=QtWidgets.QMessageBox.No)
        
        if ans == QtWidgets.QMessageBox.Yes:
            uninstall_data = True
        if ans == QtWidgets.QMessageBox.No:
            uninstall_data = False
        if ans == QtWidgets.QMessageBox.Cancel:
            Log.warning("Uninstall canceled. No changes were made.")

            # box = QtWidgets.QMessageBox(None)
            self.setIcon(QtWidgets.QMessageBox.Critical)
            self.setWindowIcon(QtGui.QIcon(tempicon))
            self.setWindowTitle("QATCH nanovisQ | Abort")
            self.setText("User canceled uninstall. Uninstall aborted.")
            self.setDetailedText(log_stream.getvalue())
            self.setStandardButtons(QtWidgets.QMessageBox.Ok)
            self.exec_()
            return

        self.setWindowIcon(QtGui.QIcon(tempicon))
        self.setWindowTitle("QATCH nanovisQ | Confirm Uninstall")
        if ask_for_specific_version:
            self.setText(f"Are you sure you want to uninstall version:\n{selected_version}?")
        elif number_of_builds_installed == 1:
            self.setText(f"Are you sure you want to uninstall version:\n{installed_versions[0]}?")
        else:
            self.setText(f"Are you sure you want to uninstall {number_of_builds_installed} versions:\n" + 
                         "\n".join(installed_versions) + "?")
        self.setIcon(QtWidgets.QMessageBox.Question)
        self.setStandardButtons(QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
        self.setDefaultButton(QtWidgets.QMessageBox.Yes)
        uninstall_confirmed = self.button(QtWidgets.QMessageBox.Yes)
        self.exec_()

        if self.clickedButton() != uninstall_confirmed:
            Log.warning("User did not consent to confirmation request. No changes made.")

            # box = QtWidgets.QMessageBox(None)
            self.setIcon(QtWidgets.QMessageBox.Critical)
            self.setWindowIcon(QtGui.QIcon(tempicon))
            self.setWindowTitle("QATCH nanovisQ | Abort")
            self.setText("User canceled uninstall. Uninstall aborted.")
            self.setDetailedText(log_stream.getvalue())
            self.setStandardButtons(QtWidgets.QMessageBox.Ok)
            self.exec_()

            return

        # Uninstall application data (if exists, and requested)
        if os.path.exists(iconpath):
            if uninstall_data:
                QatchInstaller.recursive_remove(iconpath)
                Log.info("Uninstalled application data.")
            else:
                Log.info("Retaining application data, per user request.")
        else:
            Log.warning("No application data found. Nothing to uninstall here.")

        # Uninstall software files (if exists)
        if os.path.exists(_nanopath):
            # TODO: If more than 1 version being uninstalled, show progress window
            # specify software path as normal to write desktop.ini file (if exists)
            win32api.SetFileAttributes(_nanopath, win32con.FILE_ATTRIBUTE_NORMAL)
            QatchInstaller.recursive_remove(_nanopath)
            Log.info("Uninstalled software files.")
        else:
            Log.warning("No software files found. Nothing to uninstall here.")

        # Update cached version info
        if ask_for_specific_version:
            installed_versions.remove(selected_version)
            number_of_builds_installed -= 1
        else:
            installed_versions = []
            number_of_builds_installed = 0

        # Inspect desktop shortcut and update as necessary
        desktop = None
        require_reinstall = False
        userdir = os.path.expanduser('~')
        option1 = os.path.join(userdir, "Desktop")
        option2 = os.path.join(userdir, "OneDrive", "Desktop")
        if os.path.exists(option1):
            desktop = option1
        elif os.path.exists(option2):
            desktop = option2
        else:
            Log.warning("Unable to find desktop shortcut. Please delete manually (if exists).")
        if desktop != None:
            linkpath = os.path.join(desktop, "QATCH nanovisQ.lnk")
            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(linkpath)
            targetpath = shortcut.TargetPath
            target = os.path.dirname(targetpath)
            if os.path.abspath(target) == os.path.abspath(_nanopath):
                # we just uninstalled the active version, remove desktop link
                Log.info("Removing desktop shortcut for uninstalled version.")
                os.remove(linkpath)
                if number_of_builds_installed > 0:
                    Log.info("Please re-install shortcut to restore.")
                    require_reinstall = True
            elif os.path.exists(targetpath):
                Log.debug("Leaving desktop shortcut, it's for another (still installed) version.")
            else:
                # remove desktop link, target path missing
                Log.info("Removing desktop shortcut. Target path EXE is missing.")
                os.remove(linkpath)
                if number_of_builds_installed > 0:
                    Log.info("Please re-install shortcut to restore.")
                    require_reinstall = True     

        # Re-install desktop shortcut link (if required)
        if require_reinstall:
            Log.debug("Re-installation is required to restore a working desktop shortcut.")

            # box = QtWidgets.QMessageBox(None)
            self.setWindowIcon(QtGui.QIcon(tempicon))
            self.setWindowTitle("QATCH nanovisQ | Restore Shortcut")
            self.setText("Which version should the desktop shortcut launch?")
            self.setIcon(QtWidgets.QMessageBox.Question)
            self.setStandardButtons(QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
            button1 = self.button(QtWidgets.QMessageBox.Yes)
            button1.setText('Create')
            button2 = self.button(QtWidgets.QMessageBox.No)
            button2.setText('Skip')
            self.setDefaultButton(QtWidgets.QMessageBox.Yes)
            self.comboBox = QtWidgets.QComboBox(None)
            self.comboBox.addItems(installed_versions)
            self.comboBox.setCurrentIndex(number_of_builds_installed - 1)
            self.layout().addWidget(self.comboBox, 1, 2, 1, 1) # widget, row, col, rowSpan, colSpan
            self.exec_() # wait for user interaction

            # remove custom combobox widget from layout
            selected_version = self.comboBox.currentText()
            widgetAt = self.findChild(QtWidgets.QComboBox)
            self.comboBox.hide()
            self.layout().removeWidget(widgetAt)
            widgetAt.deleteLater()
            
            if self.clickedButton() != button1:
                Log.debug("User skipped or did not make a selection.")
            else:
                Log.info(f"Selected version for shortcut launch: {selected_version}")
                _nanopath = os.path.join(nanopath, selected_version, "QATCH nanovisQ.exe")
                shortcut.TargetPath = _nanopath
                shortcut.Arguments = "" # explicity remove '-m QATCH' if upgrading PY install to EXE
                shortcut.Description = selected_version #f"(C) {date.today().year} QATCH Technologies LLC"
                shortcut.IconLocation = f"{_nanopath},0"
                shortcut.WorkingDirectory = os.path.join(os.path.dirname(desktop), "Documents", "QATCH nanovisQ")
                shortcut.save()
                Log.info(f"Created desktop link for {selected_version}!")

        if log_stream.getvalue() != "": # success, no warnings or errors
            # box = QtWidgets.QMessageBox(None)
            self.setIcon(QtWidgets.QMessageBox.Information)
            self.setWindowIcon(QtGui.QIcon(tempicon))
            self.setWindowTitle("QATCH nanovisQ | Uninstall Error")
            self.setText("Uninstaller was successful.")
            self.setStandardButtons(QtWidgets.QMessageBox.Ok)
            self.exec_()
        else:
            # box = QtWidgets.QMessageBox(None)
            self.setIcon(QtWidgets.QMessageBox.Critical)
            self.setWindowIcon(QtGui.QIcon(tempicon))
            self.setWindowTitle("QATCH nanovisQ | Uninstall Error")
            self.setText("Uninstaller was not successful.\nSee details for more information.")
            self.setDetailedText(log_stream.getvalue())
            self.setStandardButtons(QtWidgets.QMessageBox.Ok)
            self.exec_()
        # QtWidgets.QMessageBox.information(None, 
        #                                   "QATCH nanovisQ | Uninstall Error", 
        #                                   "Uninstaller finished.\nSee console output for detail(s).", 
        #                                   buttons=QtWidgets.QMessageBox.Ok)
        
        Log.info("Uninstall finished.")


class RequestWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(requests.Response)
    exception = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str, int)
    
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        
        self.success = False
        self._cancel = False
        
    def cancel(self):
        self._cancel = True

    def update_progress(self, str, pct):
        self.progress.emit(str, pct)

    def run(self):
        try:
            if self.kwargs.get("response") and self.kwargs.get("file_local"):
                response: requests.Response = self.kwargs.pop("response")
                file_local: str = self.kwargs.pop("file_local")
                last_pct = -1
                size = int(response.headers['Content-Length'])
                with open(file_local, 'wb') as f:
                        Log.debug(f"Writing {file_local}...")
                        curr_size = 0
                        for chunk in response.iter_content(chunk_size=65536):
                            if self._cancel:
                                break
                            if chunk:
                                f.write(chunk)

                                curr_size += len(chunk)
                                pct = int(100 * curr_size / size)
                                if pct != last_pct or curr_size == size:
                                    status_str = f"Download Progress: {curr_size} / {size} bytes ({pct}%)"
                                    Log.debug(status_str)
                                    # Update progress dialog for download progress
                                    self.update_progress(
                                        status_str[:status_str.rfind(' (')], pct)
                                    last_pct = pct
                if self._cancel:
                    Log.warning("User canceled software bundle download.")
                    self.update_progress("Canceling download...", 99)
                    os.remove(file_local)
                    self.update_progress("Download canceled.", 100)
                else:
                    Log.info(f"Downloaded {os.path.basename(file_local)}")
                    self.update_progress("Download finished.", 100)
                    self._success = True
                # self.finished.emit(response)
            else:
                self.update_progress("Checking online software bundles...", -1)
                self.finished.emit(
                    requests.get(**self.kwargs))
                self._success = True
            
        except Exception as e:
            # Log.error("An unhandled exception occurred:")
            # Log.error(f"{e.__class__.__name__}: {e}")

            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb
            a_list = ['Traceback (most recent call last):']
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.error(line)

            # show exception message box
            self.exception.emit(f"{t.__name__}: {str(v)}")


class InstallWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal()
    exception = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str, int)
    # missing = QtCore.pyqtSignal()

    def __init__(self, proceed, copy_src, copy_dst):
        super().__init__()
        
        self.proceed = proceed
        self.copy_src = copy_src
        self.copy_dst = copy_dst
        
        self.success = False
        self._cancel = False

        
    def cancel(self):
        self._cancel = True


    def update_progress(self, str, pct):
        self.progress.emit(str, pct)


    def run(self):
        try:
            proceed = self.proceed
            copy_src = self.copy_src
            copy_dst = self.copy_dst        
                
            self.success = False
            if proceed:
                self.update_progress("Preparing installation...", 1)
                
                Log.debug(f"Copying from {copy_src} to {copy_dst}...")
                # we already have permission to blow away any existing installation
                if os.path.isdir(copy_dst):
                    QatchInstaller.recursive_remove(copy_dst)
                self.update_progress("Copying application files...", 2)
                Log.info("Copying new application files...")
                shutil.unpack_archive(copy_src, copy_dst)
                # shutil.copytree(copy_src, copy_dst, dirs_exist_ok=True)
                self.success = True

            # abort early if cancel pending
            if self._cancel: self.success = False

            # if not errors up to this point
            exe_path = os.path.join(copy_dst, "QATCH nanovisQ.exe")
            if self.success and os.path.exists(exe_path) and log_stream.getvalue() == "":
                self.update_progress("Verifying application checksum...", 51)
                Log.info("Verifying application checksum...")
                checksum_path = os.path.join(copy_dst, "app.checksum")
                if not os.path.exists(checksum_path):
                    self.success = False
                    Log.error("No app.checksum file provided. Unable to verify EXE.")
                    Log.error("Per security best practives, this version will now be uninstalled.")
                    Log.error("Please contact support if this problem persists.")
                    QatchInstaller.recursive_remove(copy_dst)
                    return
                else:
                    # app.checksum file exists, check it
                    with open(checksum_path, "r") as f:
                        expect_md5 = f.readline().strip()
                    # self.update_progress("Verifying application checksum...", 60)

                    ### PRIMARY METHOD: WORKS, BUT FASTER, MAYBE?
                    if CHECKSUM_METHOD == "certutil":
                        calculated_path = os.path.join(copy_dst, "calc.checksum")
                        os.system(f'certutil -hashfile "{exe_path}" MD5 | find /i /v "md5" | find /i /v "certutil" > "{calculated_path}"')
                        if os.path.exists(calculated_path):
                            with open(calculated_path, "r") as f:
                                actual_md5 = f.readline().strip()

                    ### ALTERNATIVE METHOD: WORKS, BUT SLOWER, MAYBE?
                    if CHECKSUM_METHOD == "hashlib":
                        exe_size = os.stat(exe_path).st_size
                        # self.update_progress("Verifying application (may take a bit)...", 50)
                        step_size = 50 / (exe_size / 8192)
                        this_step = 0
                        upd_inter = int(1 / step_size) # update interval (1 update per every 1%)
                        with open(exe_path, "rb", buffering=8192) as f:
                            file_hash = hashlib.md5()
                            while chunk := f.read(8192):
                                this_step += 1
                                if this_step % upd_inter == 1:
                                    percentage = 50 + int(step_size * this_step)
                                    self.update_progress("Verifying application checksum...", percentage)
                                file_hash.update(chunk)
                            actual_md5 = file_hash.hexdigest()
                        # print(step_size)
                        # print(this_step)

                    Log.debug(f"Expect MD5 = {expect_md5}")
                    Log.debug(f"Actual MD5 = {actual_md5}")
                    if expect_md5 == actual_md5:
                        Log.info("App checksum verified!")
                        os.remove(checksum_path)
                        if CHECKSUM_METHOD == 'certutil':
                            os.remove(calculated_path)
                    else:
                        self.success = False
                        Log.error("App checksum mismatch!")
                        Log.error("Per security best practives, this version will now be uninstalled.")
                        Log.error("Please contact support if this problem persists.")
                        QatchInstaller.recursive_remove(copy_dst)
                        return
            
            missing = False
            if not os.path.exists(exe_path):
                missing = True
                self._cancel = True
                self.success = False
                Log.error(f"Missing file: {exe_path}")
                Log.warning("Is the application in a subfolder of the bundled ZIP file?")

            if self._cancel:
                self.update_progress("Canceling installation...", 99)
                QatchInstaller.recursive_remove(copy_dst)
                self.update_progress("Install canceled.", 100)
            else:
                self.update_progress("Install finished and verified!", 100)

            if missing:
                raise FileNotFoundError("The application executable could not be found after unpacking bundle. Unable to verify build checksum. Installation failed.")
        

        except Exception as e:
            # Log.error("An unhandled exception occurred:")
            # Log.error(f"{e.__class__.__name__}: {e}")

            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb
            a_list = ['Traceback (most recent call last):']
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.error(line)

            # show exception message box
            self.exception.emit(f"{t.__name__}: {str(v)}")
        
        self.finished.emit()


class LaunchWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(bool, int)
    exception = QtCore.pyqtSignal(str)
    # progress = QtCore.pyqtSignal(str, int)
    # missing = QtCore.pyqtSignal()

    def __init__(self, filepath, cwd):
        super().__init__()
        self.filepath = filepath
        self.cwd = cwd
    
        
    # def cancel(self):
    #     self._cancel = True


    # def update_progress(self, str, pct):
    #     self.progress.emit(str, pct)


    def run(self):
        success = False
        pid = 0
        
        try:
            filepath = self.filepath
            working_dir = self.cwd

            if working_dir is None:
                working_dir = os.path.dirname(os.path.abspath(filepath))
            
            # Normalize paths
            filepath = os.path.normpath(filepath)
            working_dir = os.path.normpath(working_dir)
            
            success, pid = QtCore.QProcess.startDetached(filepath, [], working_dir)

            # self.finished.emit(success, pid)
            # Log.info(f"PID: {pid}")
            # self.app_pid = pid
            # return success and pid > 0

        except Exception as e:
            # Log.error("An unhandled exception occurred:")
            # Log.error(f"{e.__class__.__name__}: {e}")

            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb
            a_list = ['Traceback (most recent call last):']
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.error(line)

            # show exception message box
            self.exception.emit(f"{t.__name__}: {str(v)}")
        
        self.finished.emit(success, pid)


if __name__ == '__main__':
    QatchInstaller().run()