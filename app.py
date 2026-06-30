import ctypes
import os  # add
import subprocess
import sys
import time
from multiprocessing import freeze_support

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication

from QATCH.common.architecture import Architecture, OSType
from QATCH.common.arguments import Arguments
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants, MinimalPython
from QATCH.ui.widgets import QatchSplashScreen

# from QATCH.ui import mainWindow # lazy load

try:
    import logging

    # suppress ERROR if not bundled in EXE
    logging.getLogger("pyi_splash").setLevel(logging.CRITICAL)
    # if splash binaries are not bundled with a compiled EXE, this import will fail
    import pyi_splash

    # this is just a sanity check to confirm the splash module
    USE_PYI_SPLASH = False  # pyi_splash.is_alive() jj
    # restore to default level, it's active
    logging.getLogger("pyi_splash").setLevel(logging.WARNING)
except:
    USE_PYI_SPLASH = False

if not USE_PYI_SPLASH:
    if len(sys.argv) > 1 and sys.argv[1] == "--splash":
        # This block only executes inside the subprocess
        app = QApplication(sys.argv)
        splash = QatchSplashScreen()
        sys.exit(app.exec_())

TAG = ""  # "[Application]"


###############################################################################
# Main Application
###############################################################################
class QATCH:

    ###########################################################################
    # Initializing values for application
    ###########################################################################
    def __init__(self, argv=sys.argv):

        self.win = None
        self.flashSplashShow()

        if getattr(sys, "frozen", False):
            userpath = os.path.expandvars("%USERPROFILE%")
            docspath = os.path.join(userpath, "Documents", "QATCH nanovisQ")
            if os.path.isdir(docspath):
                if os.path.normcase(os.getcwd()) != os.path.normcase(docspath):
                    try:
                        os.chdir(docspath)
                    except OSError as ose:
                        raise ose

        print("Launching application...")
        if Architecture.get_os() is OSType.windows:
            myappid = "{} {} {} ({})".format(
                Constants.app_publisher,
                Constants.app_name,
                Constants.app_version,
                Constants.app_date,
            )  # arbitrary string, required for Windows Toolbar to display QATCH icon
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            ctypes.windll.kernel32.SetConsoleTitleW("QATCH Q-1 Real-Time GUI - command line")
        self._args = self._init_logger()
        QtCore.QCoreApplication.setOrganizationName("QATCH")
        QtCore.QCoreApplication.setApplicationName("nanovisQ")
        self._app = QApplication(argv)

        # Apply the persisted light/dark theme app-wide before any window
        # (including the pre-login sign-in screen) is constructed.
        from QATCH.ui.styles.theme_manager import ThemeManager

        ThemeManager.instance().apply_app_stylesheet(self._app)

    def flashSplashShow(self):
        build_info = f" {Constants.app_title}\n Version: {Constants.app_version}\n Build Date: {Constants.app_date}\n"

        if USE_PYI_SPLASH:
            # Update the text on the splash screen
            build_info = "\n".join(
                [f"                                          {s}" for s in build_info.split("\n")]
            )
            pyi_splash.update_text(build_info)
        else:
            try:
                if getattr(sys, "frozen", False):
                    # Launch the exact same executable, but pass the --splash flag
                    # This safely tells the child copy to only run the splash screen loop
                    self.splash_process = subprocess.Popen([sys.executable, "--splash"])
                else:
                    # Launch a completely separate Python instance for splash screen process
                    self.splash_process = subprocess.Popen([sys.executable, "app.py", "--splash"])
            except Exception as e:
                Log.e("Failed to launch splash screen process:", e)

        # Close SplashScreen after app is loaded
        # (min 3 sec, timer will wait longer for app load, if needed)
        # time.sleep(2)
        #     QtCore.QTimer.singleShot(3000, self.flashSplashHide)
        self.start = time.time()

    def flashSplashHide(self):
        while time.time() - self.start < 3 and self.win.ReadyToShow == False:
            # Log.w("Waiting for splash delay")
            pass

        while time.time() - self.start < 9 and not hasattr(self.win, "ask_for_update"):
            pass

        if USE_PYI_SPLASH:
            # Close the splash screen. It does not matter when the call
            # to this function is made, the splash screen remains open until
            # this function is called or the Python program is terminated.
            pyi_splash.close()
        else:
            if hasattr(self, "splash_process"):
                try:
                    self.splash_process.terminate()
                except PermissionError as e:
                    Log.e("Failed to terminate splash screen:", e)

        # if Architecture.get_os() is OSType.windows:
        #     kernel32 = ctypes.WinDLL('kernel32')
        #     user32 = ctypes.WinDLL('user32')
        #     SW_HIDE = 0
        #     hWnd = kernel32.GetConsoleWindow()
        #     if hWnd:
        #         user32.ShowWindow(hWnd, SW_HIDE)

        self.win.MainWin.showMaximized()
        self.win.MainWin.activateWindow()

        if hasattr(self.win, "ask_for_update") and self.win.ask_for_update:
            self.win.start_download()
        ##

    ###########################################################################
    # Runs the application
    ###########################################################################
    def run(self):
        # lazy load imports
        from QATCH.ui import mainWindow

        if Architecture.is_python_version(MinimalPython.major, minor=MinimalPython.minor):
            Log.i(TAG, "Application started")

            self.win = mainWindow.MainWindow(samples=self._args.get_user_samples())
            # win.setWindowTitle("{} - {}".format(Constants.app_title, Constants.app_version))
            # win.move(500, 20) #GUI position (x,y) on the screen
            # win.show()
            self.flashSplashHide()
            # self.gui_ready = True
            self._app.exec()
            Log.i(TAG, "Finishing Application...")
            Log.i(TAG, "Application closed")
            self.win.close()
        else:
            self._fail()
            time.sleep(5)
        self.close()

    ###########################################################################
    # Closes application
    ###########################################################################
    def close(self):
        self._app.exit()
        Log.close()
        os._exit(0)

    ###########################################################################
    # Initializing logger
    ###########################################################################
    @staticmethod
    def _init_logger():
        sys.stderr = Log()
        Log.create()  # initialize file and console handlers
        args = Arguments()
        args.create()
        args.set_user_log_level()
        return args

    ###########################################################################
    # Specifies the minimal Python version required
    ###########################################################################
    @staticmethod
    def _fail():
        txt = str(
            "Application requires Python {}.{} to run".format(
                MinimalPython.major, MinimalPython.minor
            )
        )
        Log.e(TAG, txt)


if __name__ == "__main__":
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    if hasattr(QtCore.Qt, "AA_ShareOpenGLContexts"):  # Needed to load web modules
        QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts, True)
    freeze_support()

    QATCH().run()
