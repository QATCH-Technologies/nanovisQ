"""
QATCH.app.py

Main entry point for the QATCH nanovisQ application.

Handles application startup, operating system-specific configurations, custom logging,
and theming initialization. Manages the application lifecycle, including launching
the initial splash screen as a separate subprocess and tearing it down once the main
window is ready.
"""

import ctypes
import os
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

TAG = "[Application]"
try:
    import logging

    # suppress ERROR if not bundled in EXE
    logging.getLogger("pyi_splash").setLevel(logging.CRITICAL)

    # if splash binaries are not bundled with a compiled EXE, this import will fail
    import pyi_splash

    # this is just a sanity check to confirm the splash module
    USE_PYI_SPLASH = False  # pyi_splash.is_alive()

    # restore to default level, it's active
    logging.getLogger("pyi_splash").setLevel(logging.WARNING)
except ImportError:
    USE_PYI_SPLASH = False

if not USE_PYI_SPLASH and len(sys.argv) > 1 and sys.argv[1] == "--splash":
    # This block only executes inside the subprocess. It has no
    # console attached when launched from a frozen/windowed build, so
    # without this try/except a construction failure here would just
    # silently kill the subprocess with no visible trace at all -
    # log specific UI construction regressions so they are diagnosable.
    try:
        app = QApplication(sys.argv)
        splash = QatchSplashScreen()
        sys.exit(app.exec_())
    except (RuntimeError, ValueError, TypeError) as splash_exc:
        Log.e(
            TAG,
            f"Splash screen subprocess failed to start due to instantiation error: {splash_exc}",
        )
        sys.exit(1)


class QATCH:
    """Main application class for QATCH nanovisQ.

    Manages the initialization of the Qt application, logging, command-line arguments,
    and the main user interface lifecycle.
    """

    def __init__(self, argv: list = sys.argv) -> None:
        """Initializes the QATCH application setup and environment.

        Triggers the splash screen immediately, configures the working directory for frozen
        builds, and sets the Windows AppUserModelID so the QATCH icon displays correctly on
        the toolbar. It also initializes the logger, sets QCoreApplication metadata,
        and applies the application stylesheet via the ThemeManager.

        Args:
            argv (list, optional): Command-line arguments passed to the application.
                Defaults to sys.argv.
        """

        self.win = None
        self.flashSplashShow()

        if getattr(sys, "frozen", False):
            userpath = os.path.expandvars("%USERPROFILE%")
            docspath = os.path.join(userpath, "Documents", "QATCH nanovisQ")
            if os.path.isdir(docspath) and os.path.normcase(os.getcwd()) != os.path.normcase(
                docspath
            ):
                os.chdir(docspath)

        if Architecture.get_os() is OSType.windows:
            myappid = f"{Constants.app_publisher} {Constants.app_name} {Constants.app_version} ({Constants.app_date})"  # arbitrary string, required for Windows Toolbar to display QATCH icon
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            ctypes.windll.kernel32.SetConsoleTitleW("QATCH Q-1 Real-Time GUI - command line")
        self._args = self._init_logger()
        QtCore.QCoreApplication.setOrganizationName("QATCH")
        QtCore.QCoreApplication.setApplicationName("nanovisQ")
        self._app = QApplication(argv)

        from QATCH.ui.styles.theme_manager import ThemeManager

        ThemeManager.instance().apply_app_stylesheet(self._app)

    def flashSplashShow(self) -> None:
        """Displays the application splash screen.

        If bundled as a compiled executable with pyi_splash, it updates the splash screen text
        directly. Otherwise, it spawns a separate subprocess (using the `--splash`
        flag) to run the animated splash screen concurrently without blocking the main thread.
        """
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
            except OSError as e:
                Log.e(TAG, f"Failed to launch splash screen process: {e}")

        # Close SplashScreen after app is loaded
        self.start = time.time()

    def flashSplashHide(self) -> None:
        """Hides and terminates the splash screen.

        Waits in a non-blocking loop until the main window indicates it is ready to show and
        has loaded the update attributes. Once ready, it cleanly closes pyi_splash
        or terminates the splash subprocess, shows the main mode window maximized, and starts
        update downloads if requested.
        """
        while time.time() - self.start < 3 and (self.win is None or not self.win.ReadyToShow):
            time.sleep(0.02)

        while time.time() - self.start < 9 and not hasattr(self.win, "ask_for_update"):
            time.sleep(0.02)

        if USE_PYI_SPLASH:
            pyi_splash.close()
        else:
            if hasattr(self, "splash_process"):
                try:
                    self.splash_process.terminate()
                except PermissionError as e:
                    Log.e(TAG, f"Failed to terminate splash screen: {e}")

        if self.win is not None:
            self.win.mode_window.showMaximized()
            self.win.mode_window.activateWindow()

        if self.win is not None and getattr(self.win, "ask_for_update", False):
            self.win.start_download()

    def run(self) -> None:
        """Executes the main application loop.

        Validates that the required minimal Python version is being used. If valid,
        it instantiates the MainWindow with user samples, hides the splash screen, starts the
        Qt event loop, and eventually closes the application. If invalid, it logs a
        failure and terminates.
        """
        from QATCH.ui import main_window

        if Architecture.is_python_version(MinimalPython.major, minor=MinimalPython.minor):
            Log.i(TAG, "Application started")
            samples = self._args.get_user_samples() if self._args is not None else 51
            self.win = main_window.MainWindow(samples=samples)
            self.flashSplashHide()
            self._app.exec()

            Log.i(TAG, "Finishing Application...")
            Log.i(TAG, "Application closed")

            if self.win is not None:
                self.win.close()
        else:
            self._fail()
            time.sleep(5)

        self.close()

    def close(self) -> None:
        """Closes the application and releases resources.

        Exits the Qt application event loop, cleanly closes the logger, and aggressively exits
        the process with status 0.
        """
        self._app.exit()
        Log.close()
        os._exit(0)

    @staticmethod
    def _init_logger() -> Arguments:
        """Initializes the system logger and parses command-line arguments.

        Redirects standard error to the custom logger, creates file and console logging handlers,
        and sets the user log level.

        Returns:
            Arguments: The parsed argument object.
        """
        sys.stderr = Log()
        Log.create()  # initialize file and console handlers
        args = Arguments()
        args.create()
        args.set_user_log_level()
        return args

    @staticmethod
    def _fail():
        """Logs a failure message regarding an unsupported Python version.

        Prints an error specifying the minimal major and minor Python version required to
        run the application.
        """
        Log.e(
            TAG, f"Application requires Python {MinimalPython.major}.{MinimalPython.minor} to run."
        )


if __name__ == "__main__":
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)  # type: ignore
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)  # type: ignore
    if hasattr(QtCore.Qt, "AA_ShareOpenGLContexts"):  # Needed to load web modules
        QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts, True)  # type: ignore
    freeze_support()

    QATCH().run()
