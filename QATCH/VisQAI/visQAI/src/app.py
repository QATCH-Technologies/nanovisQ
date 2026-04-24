import os
import sys
import types
from types import SimpleNamespace

from PyQt5 import QtCore, QtWidgets

# --- 1. PATH FIX ---
# This ensures Python can find 'src' modules when running from inside 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# --- 2. DEFINE MOCK CLASSES ---


class MockLicenseManager:
    """Mocks the QATCH LicenseManager."""

    def __init__(self):
        self.cache_enabled = False

    def validate_license(self):
        return (
            True,
            "Standalone Valid",
            {"status": "ADMIN", "expiration": "2099-12-31T23:59:59Z"},
        )

    def get_cache_status(self):
        return {"cache_expired": False}


class MockLicenseStatus:
    INACTIVE = "INACTIVE"
    ADMIN = "ADMIN"
    ACTIVE = "ACTIVE"
    TRIAL = "TRIAL"

    def __new__(cls, value):
        return value


class MockUserProfiles:
    @staticmethod
    def checkDevMode():
        return (True, "Dev Mode")

    @staticmethod
    def session_info():
        return True, ["StandaloneUser", "SU", "ADMIN"]

    @staticmethod
    def check(role1, role2):
        return True

    @staticmethod
    def change(role):
        return "StandaloneUser", "SU", 2


class Constants:
    app_title = "VisQ.AI Standalone"
    user_profiles_path = "./"
    auto_sign_key_path = "./autosign.key"

    # --- ADDED PATHS TO FIX ATTRIBUTE ERROR ---
    log_prefer_path = "."  # Default directory for file dialogs
    log_path = "./logs"  # Common logging path
    data_path = "./data"  # Common data path


class Architecture:
    @staticmethod
    def get_path():
        return "."

    @staticmethod
    def get_os_name():
        return "Windows"


class PopUp:
    @staticmethod
    def warning(parent, title, msg):
        print(f"POPUP WARNING: {title} - {msg}")


class Logger:
    @staticmethod
    def d(tag, msg=""):
        print(f"DEBUG: {tag} {msg}")

    @staticmethod
    def i(tag, msg=""):
        print(f"INFO: {tag} {msg}")

    @staticmethod
    def w(tag, msg=""):
        print(f"WARN: {tag} {msg}")

    @staticmethod
    def e(tag, msg=""):
        print(f"ERROR: {tag} {msg}")


# --- 3. DEFINE MAIN WINDOW MOCK (MOVED UP) ---
# We define this BEFORE registration so we can assign it to the mock module
class MockMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self._license_manager = MockLicenseManager()
        self.signature_required = False
        self.signature_received = False
        self.signed_at = None

        self.MainWin = SimpleNamespace(
            ui0=SimpleNamespace(
                floating_widget=SimpleNamespace(setActiveItem=lambda x: None)
            )
        )
        self.ControlsWin = SimpleNamespace(
            username=QtWidgets.QLabel(),
            signinout=QtWidgets.QPushButton(),
            manage=QtWidgets.QPushButton(),
            ui1=SimpleNamespace(tool_User=QtWidgets.QLabel()),
            userrole=2,  # ADMIN
        )
        self.AnalyzeProc = SimpleNamespace(tool_User=QtWidgets.QLabel())


# --- 4. REGISTER MOCKS IN SYS.MODULES ---

# Base packages
qatch_mock = types.ModuleType("QATCH")
qatch_common = types.ModuleType("QATCH.common")
qatch_ui = types.ModuleType("QATCH.ui")
qatch_core = types.ModuleType("QATCH.core")

sys.modules["QATCH"] = qatch_mock
sys.modules["QATCH.common"] = qatch_common
sys.modules["QATCH.ui"] = qatch_ui
sys.modules["QATCH.core"] = qatch_core

# 4a. Register LICENSE MANAGER
license_mod = types.ModuleType("licenseManager")
license_mod.LicenseManager = MockLicenseManager
license_mod.LicenseStatus = MockLicenseStatus
sys.modules["QATCH.common.licenseManager"] = license_mod

# 4b. Register USER PROFILES
user_prof_mod = types.ModuleType("userProfiles")
user_prof_mod.UserProfiles = MockUserProfiles
user_prof_mod.UserRoles = SimpleNamespace(ANALYZE=1, ADMIN=2, NONE=0)
sys.modules["QATCH.common.userProfiles"] = user_prof_mod

# 4c. Register ARCHITECTURE
arch_mod = types.ModuleType("architecture")
arch_mod.Architecture = Architecture
sys.modules["QATCH.common.architecture"] = arch_mod

# 4d. Register LOGGER
log_mod = types.ModuleType("logger")
log_mod.Logger = Logger
sys.modules["QATCH.common.logger"] = log_mod

# 4e. Register CONSTANTS
const_mod = types.ModuleType("constants")
const_mod.Constants = Constants
sys.modules["QATCH.core.constants"] = const_mod

# 4f. Register POPUP
popup_mod = types.ModuleType("popUp")
popup_mod.PopUp = PopUp
sys.modules["QATCH.ui.popUp"] = popup_mod

# 4g. Register MAIN WINDOW (THE FIX)
# This creates QATCH.ui.mainWindow and puts our MockMainWindow class inside it
mw_mod = types.ModuleType("mainWindow")
mw_mod.MainWindow = MockMainWindow
sys.modules["QATCH.ui.mainWindow"] = mw_mod

# --- 5. MAIN EXECUTION ---
from src.view.main_window import VisQAIWindow


def main():
    app = QtWidgets.QApplication(sys.argv)

    parent_mock = MockMainWindow()

    window = VisQAIWindow(parent=parent_mock)
    window.show()
    window.enable(True)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
