import sys
import json
import time
import threading
import pyautogui
import pygetwindow as gw
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QGroupBox,
    QCheckBox,
    QLabel,
)

CONFIG_FILE = "credentials.json"


class AutoLoginApp(QWidget):
    status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.found = False
        self.status_format_str = "Idle."
        self.status_time = None
        self.status_signal.connect(self.update_status)
        self.init_ui()
        self.load_credentials()

    def init_ui(self):
        self.setWindowTitle("QATCH Auto-Login Tool")

        self.setWindowFlags(
            Qt.Window
            | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowCloseButtonHint
        )

        layout = QVBoxLayout()

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)

        self.save_button = QPushButton("Save Credentials")
        self.save_button.clicked.connect(self.save_credentials)

        options_layout = QVBoxLayout()

        self.options_group = QGroupBox("Options")

        self.auto_login = QCheckBox("Auto-login on launch")
        self.auto_login.setChecked(True)

        self.auto_sign = QCheckBox("Auto-sign audit logs")
        self.auto_sign.setChecked(True)

        options_layout.addWidget(self.auto_login)
        options_layout.addWidget(self.auto_sign)

        self.options_group.setLayout(options_layout)

        self.toggle_button = QPushButton("Start")
        self.toggle_button.clicked.connect(self.toggle_automation)

        self.status_text = QLabel(self.status_format_str)
        self.status_text.setEnabled(False)

        layout.addWidget(QLabel("Username"))
        layout.addWidget(self.username_input)
        layout.addWidget(QLabel("Password"))
        layout.addWidget(self.password_input)
        layout.addWidget(self.save_button)
        layout.addWidget(self.options_group)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.status_text)

        self.setLayout(layout)

    def save_credentials(self):
        data = {
            "username": self.username_input.text(),
            "password": self.password_input.text(),
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f)

        orig_text = self.save_button.text()
        self.save_button.setText("Saved!")
        QTimer.singleShot(3000, lambda: self.save_button.setText(orig_text))

    def load_credentials(self):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                self.username_input.setText(data.get("username", ""))
                self.password_input.setText(data.get("password", ""))
        except FileNotFoundError:
            pass

    def get_credentials(self):
        return {
            "username": self.username_input.text(),
            "password": self.password_input.text(),
        }

    def toggle_automation(self):
        self.running = not self.running
        self.toggle_button.setText("Stop" if self.running else "Start")

        if self.running:
            threading.Thread(target=self.watch_for_windows, daemon=True).start()

    def update_status(self, status_text: str):
        self.status_text.setText(status_text)

    def watch_for_windows(self):
        self.status_format_str = "Running. No action performed."
        self.found = self.find_window_startswith("QATCH nanovisQ")

        while self.running:

            if self.auto_login.isChecked():
                windows = gw.getWindowsWithTitle("Sign In")
                if windows:
                    win = windows[0]
                else:
                    win = self.find_window_startswith("QATCH nanovisQ")
                if win:
                    win.activate()

                    self.status_format_str = "Running. <i>Login event pending.</i>"
                    self.status_signal.emit(self.status_format_str)

                    if not self.wait_for_focus(win, timeout=15):
                        self.status_format_str = "Running. <b>Login error {} ago.</b>"
                        self.status_time = time.time()
                        continue

                    creds = self.get_credentials()

                    pyautogui.write(creds["username"])
                    pyautogui.press("enter")
                    time.sleep(0.5)

                    pyautogui.write(creds["password"])
                    pyautogui.press("enter")

                    self.status_format_str = "Running. Login event {} ago."
                    self.status_time = time.time()

                    time.sleep(1)  # avoid repeated triggering

            if self.auto_sign.isChecked():
                windows = gw.getWindowsWithTitle("Signature")
                if windows:
                    win = windows[0]
                else:
                    win = None
                if win:
                    win.activate()

                    self.status_format_str = "Running. <i>Audit event pending.</i>"
                    self.status_signal.emit(self.status_format_str)

                    if not self.wait_for_focus(win, timeout=15):
                        self.status_format_str = "Running. <b>Audit error {} ago.</b>"
                        self.status_time = time.time()
                        continue

                    creds = self.get_credentials()

                    pyautogui.write(creds["username"])
                    pyautogui.press("enter")

                    self.status_format_str = "Running. Audit event {} ago."
                    self.status_time = time.time()

                    time.sleep(1)  # avoid repeated triggering

            elapsed = 0
            time_fmt_str = "???"
            if self.status_time:
                elapsed = int(time.time() - self.status_time)
                time_fmt_str = "{}s"
                if elapsed > 60:
                    elapsed = int(elapsed / 60)
                    time_fmt_str = "{}m"
                if elapsed > 60:
                    elapsed = int(elapsed / 60)
                    time_fmt_str = "{}h"
                if elapsed > 60:
                    elapsed = int(elapsed / 24)
                    time_fmt_str = "{}d"

            self.status_signal.emit(
                self.status_format_str.format(time_fmt_str.format(elapsed))
            )

            time.sleep(1)

        self.status_format_str = "Idle."
        self.status_signal.emit(self.status_format_str)

    def find_window_startswith(self, prefix):
        for win in gw.getAllWindows():
            if win.title and win.title.startswith(prefix):
                if self.found:
                    return None
                self.found = True
                return win
        self.found = False
        return None

    def wait_for_focus(self, win, timeout=5):
        start = time.time()
        while time.time() - start < timeout:
            if win.isActive:
                return True
            time.sleep(0.1)
        return False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AutoLoginApp()
    window.show()
    sys.exit(app.exec_())
