"""
QATCH.ui.mainWindow_ui

The mainWindow_ui module handles the drawing, UI control elements,
and UI actions for the main window of the main application.

Author(s)
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)
    Others...

Date:
    2026-01-26
"""

import os
import sys
from time import monotonic, sleep
from typing import Optional
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QDesktopWidget,
)

from QATCH.common.architecture import Architecture, OSType
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants, OperationType, UserRoles
from QATCH.ui.widgets.well_plate_widget import WellPlate
from QATCH.ui.popUp import PopUp
from QATCH.ui.components.number_icon_button import NumberIconButton
from QATCH.ui.components.run_controls_button import RunControls
from QATCH.ui.widgets.export_widget import Ui_Export
from QATCH.ui.widgets.user_preferences_widget import UserPreferencesWidget
from QATCH.ui.widgets.user_profiles_manager_widget import UserProfilesManagerWidget
from QATCH.common.userProfiles import UserProfiles
from QATCH.common.deviceFingerprint import DeviceFingerprint
from QATCH.common.fileStorage import FileStorage
from QATCH.common.findDevices import Discovery
from QATCH.common.licenseManager import LicenseManager
from QATCH.processors.Device import serial
from QATCH.ui.widgets.advanced_main_widget import (
    AdvancedMainWidget,
    GlassWarningLabel,
)
from QATCH.ui.components.glass_push_button import GlassPushButton
from QATCH.ui.components.glass_toggle import GlassToggle
from QATCH.ui.components.animated_combo_box import AnimatedComboBox

# ---------------------------------------------------------------------------
# Glass-morphism primitives
# ---------------------------------------------------------------------------

TAG = "[ControlsWindow]"


class ControlsWindow(QtWidgets.QMainWindow):
    def __init__(self, parent, samples=Constants.argument_default_samples):
        self.parent = parent
        super().__init__()
        self.ui1 = UIControls()
        self.ui1.setupUi(self)
        self.ui_export = Ui_Export()
        # self.ui_configure_data = UIConfigureData()
        self.ui_preferences = UserPreferencesWidget(self)
        # self.userrole
        self.current_timer = QtCore.QTimer()
        # self.current_timer.timeout.connect(self.double_toggle_plots)
        UserProfiles().session_end()

    def _createMenu(self, target):
        self.menubar = []
        self.menubar.append(target.menuBar().addMenu("&Options"))
        self.menubar[0].addAction("&Analyze Data", self.analyze_data)
        self.menubar[0].addAction("&Import Data", self.import_data)
        self.menubar[0].addAction("&Export Data", self.export_data)
        self.menubar[0].addAction("&Recover Data", self.recover_data)
        # self.menubar[0].addAction('&Configure Data', self.configure_data)
        self.menubar[0].addAction("&Preferences", self.preferences)
        self.menubar[0].addAction("&Find Devices", self.scan_subnets)
        self.menubar[0].addAction("E&xit", self.close)
        self.menubar.append(target.menuBar().addMenu("&Users"))
        self.username = self.menubar[1].addAction("User: [NONE]")
        self.username.setEnabled(False)
        self.signinout = self.menubar[1].addAction("&Sign In", self.set_user_profile)
        self.menubar[1].addAction("Select &directory...", self.set_working_directory)
        self.manage = self.menubar[1].addAction("&Manage Users...", self.manage_user_profiles)
        self.userrole = UserRoles.NONE
        self.menubar.append(target.menuBar().addMenu("&View"))
        self.modebar = self.menubar[2].addMenu("&Mode")
        self.modebar.addAction("&1: Run", lambda: self.parent.MainWin.ui0._set_run_mode(None))
        self.modebar.addAction(
            "&2: Analyze", lambda: self.parent.MainWin.ui0._set_analyze_mode(None)
        )
        if Constants.show_visQ_in_R_builds:
            self.modebar.addAction(
                "&3: VisQ.AI", lambda: self.parent.MainWin.ui0._set_learn_mode(None)
            )
        self.chk1 = self.menubar[2].addAction("&Console", self.toggle_console)
        self.chk1.setCheckable(True)
        self.chk1.setChecked(
            self.parent.AppSettings.value("viewState_Console", "True").lower() == "true"
        )
        self.chk2 = self.menubar[2].addAction("&Amplitude", self.toggle_amplitude)
        self.chk2.setCheckable(True)
        self.chk2.setChecked(
            self.parent.AppSettings.value("viewState_Amplitude", "True").lower() == "true"
        )
        self.chk3 = self.menubar[2].addAction("&Temperature", self.toggle_temperature)
        self.chk3.setCheckable(True)
        self.chk3.setChecked(
            self.parent.AppSettings.value("viewState_Temperature", "True").lower() == "true"
        )
        self.chk4 = self.menubar[2].addAction("&Resonance/Dissipation", self.toggle_RandD)
        self.chk4.setCheckable(True)
        self.chk4.setChecked(
            self.parent.AppSettings.value("viewState_Resonance_Dissipation", "True").lower()
            == "true"
        )
        self.menubar.append(target.menuBar().addMenu("&Help"))
        self.chk5 = self.menubar[3].addAction("View &Tutorials", self.view_tutorials)
        self.chk5.setCheckable(False)
        self.menubar.append(self.menubar[3].addMenu("View &Documentation"))
        self.menubar[4].addAction("&Release Notes", self.release_notes)
        self.menubar[4].addAction("&FW Change Log", self.fw_change_log)
        self.menubar[4].addAction("&SW Change Log", self.sw_change_log)
        self.menubar[3].addAction("View &License", self.view_license)
        self.menubar[3].addAction("View &User Guide", self.view_user_guide)
        self.menubar[3].addAction("&Check for Updates", self.check_for_updates)
        self.menubar[3].addSeparator()
        from QATCH.models.ModelData import __release__ as ModelData_release
        from QATCH.models.ModelData import __version__ as ModelData_version
        from QATCH.QModel.src.models.static_v4_fusion.__init__ import (
            __release__ as QModel4_release,
        )
        from QATCH.QModel.src.models.static_v4_fusion.__init__ import (
            __version__ as QModel4_version,
        )
        from QATCH.QModel.src.models.v6_yolo.__init__ import (
            __release__ as QModel6_release,
        )
        from QATCH.QModel.src.models.v6_yolo.__init__ import (
            __version__ as QModel6_version,
        )

        qmodel_versions_menu = self.menubar[3].addMenu("Model versions (3 available)")
        self.menubar.append(qmodel_versions_menu)
        self.q_version_v1 = self.menubar[5].addAction(
            "ModelData v{} ({})".format(ModelData_version, ModelData_release),
            lambda: self.parent.AnalyzeProc.set_new_prediction_model(
                Constants.list_predict_models[0]
            ),
        )
        self.q_version_v1.setCheckable(True)
        self.q_version_v4 = self.menubar[5].addAction(
            "QModel Fusion v{} ({})".format(QModel4_version, QModel4_release),
            lambda: self.parent.AnalyzeProc.set_new_prediction_model(
                Constants.list_predict_models[1]
            ),
        )
        self.q_version_v4.setCheckable(True)
        self.q_version_v6 = self.menubar[5].addAction(
            "QModel YOLO26 v{} ({})".format(QModel6_version, QModel6_release),
            lambda: self.parent.AnalyzeProc.set_new_prediction_model(
                Constants.list_predict_models[2]
            ),
        )
        self.q_version_v6.setCheckable(True)
        if Constants.QModel6_predict:
            self.q_version_v6.setChecked(True)
        elif Constants.QModel4_predict:
            self.q_version_v4.setChecked(True)
        elif Constants.ModelData_predict:
            self.q_version_v1.setChecked(True)
        else:
            Log.w(TAG, "No model selected on startup")
        self.menubar[3].addSeparator()
        sw_version = self.menubar[3].addAction(
            "SW {}_{} ({})".format(
                Constants.app_version,
                "exe" if getattr(sys, "frozen", False) else "py",
                Constants.app_date,
            )
        )
        sw_version.setEnabled(False)
        fingerprint_txt = DeviceFingerprint.get_key()
        if fingerprint_txt is not None:
            self.menubar[3].addSeparator()
            fingerprint_action = self.menubar[3].addAction(fingerprint_txt)
            fingerprint_action.setToolTip("Click to copy to clipboard")
            fingerprint_action.triggered.connect(
                lambda: QtWidgets.QApplication.clipboard().setText(fingerprint_txt)
            )

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
        self.parent.MainWin.ui0._set_analyze_mode(self)

    def import_data(self):
        self.ui_export.showNormal(0)

    def export_data(self):
        self.ui_export.showNormal(1)

    def recover_data(self):
        self.ui_export.showNormal(2)

    # def configure_data(self):
    #     self.ui_configure_data.show()

    def preferences(self):
        self.ui_preferences.showNormal(0)

    def scan_subnets(self):
        Discovery().scanSubnets()
        self.parent._port_list_refresh()

    def set_working_directory(self):
        # loads global/user prefs
        self.ui_preferences.toggle_global_preferences()
        # force sync read/write
        self.ui_preferences.sync_write_with_load.setChecked(True)
        # ask user for working directory
        result = self.ui_preferences.open_load_file_dialog()
        # auto-save, assuming user gave us a new directory AND submit button is enabled
        if result and self.ui_preferences.submit_button.isEnabled():
            self.ui_preferences.submit_button.click()
        else:
            Log.w("Working directory not changed.")

    def set_user_profile(self):
        action = self.signinout.text().lower().replace("&", "")
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
            if self.parent.MainWin.ui0._set_no_user_mode(None):
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
        if not self.parent.MainWin.ui0._set_run_mode(None):
            Log.d("User has unsaved changes in Analyze mode. Manage users aborted.")
            return

        # dissallow user management if current mode is busy
        if not self.parent.ControlsWin.ui1.pButton_Start.isEnabled():
            PopUp.warning(
                self,
                "Action Not Allowed",
                "User info cannot be changed during an active capture.\n"
                + "Please 'Stop' the measurement before attempting this action.",
            )
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

        if admin is None and not UserProfiles.session_info()[0]:
            if name != "[NONE]":
                Log.i(f"Goodbye, {name}! You have been signed out.")
            self.username.setText("User: [NONE]")
            self.userrole = UserRoles.NONE
            self.signinout.setText("&Sign In")
            self.manage.setText("&Manage Users...")
            self.ui1.tool_User.setText("Anonymous")
            self.parent.AnalyzeProc.tool_User.setText("Anonymous")
            self.parent.MainWin.ui0._set_no_user_mode(None)
        if admin != name and admin != None:
            Log.d("User name changed. Changing sign-in user info.")
            self.username.setText(f"User: {admin}")
            self.userrole = UserRoles.ADMIN
            self.signinout.setText("&Sign Out")
            self.manage.setText("&Manage Users...")
            self.ui1.tool_User.setText(admin)
            self.parent.AnalyzeProc.tool_User.setText(admin)

        if allow:
            # Parent to the overarching MainWin, NOT the thin ControlsWindow
            if hasattr(self.parent, "MainWin"):
                overlay_parent = self.parent.MainWin.centralWidget() or self.parent.MainWin
            else:
                overlay_parent = self.centralWidget() or self

            Log.d(
                f"[ControlsWindow] manage_user_profiles: showing overlay, parent={overlay_parent}"
            )
            self.user_manager = UserProfilesManagerWidget(parent=overlay_parent, admin_name=admin)
            # Size to the actual overlay parent (MainWin central widget), not the
            # thin ControlsWindow — otherwise the overlay first appears at the
            # controls bar's small rect and then snaps to full size.
            try:
                self.user_manager.setGeometry(overlay_parent.rect())
            except Exception:
                pass
            self.user_manager.show()
        else:
            Log.d(f"[ControlsWindow] manage_user_profiles: allow=False, overlay not shown")

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
        self.parent.AppSettings.setValue("viewState_Console", self.chk1.isChecked())

    def toggle_amplitude(self):
        tc = self.show_top_plot()
        if not self.chk2.isChecked():
            Log.d("Hiding Amplitude plot(s)")
            for i, p in enumerate(self.parent._plt0_arr):
                if p is None:
                    continue
                p.setVisible(False)
                self.parent._plt0_arr[i] = p
        else:
            Log.d("Showing Amplitude plot(s)")
            for i, p in enumerate(self.parent._plt0_arr):
                if p is None:
                    continue
                p.setVisible(True)
                self.parent._plt0_arr[i] = p
        self.hide_top_plot(tc)
        self.parent.AppSettings.setValue("viewState_Amplitude", self.chk2.isChecked())

    def toggle_temperature(self):
        tc = self.show_top_plot()
        if not self.chk3.isChecked():
            Log.d("Hiding Temperature plot")
            self.parent._plt4.setVisible(False)
        else:
            Log.d("Showing Temperature plot")
            self.parent._plt4.setVisible(True)
        self.hide_top_plot(tc)
        self.parent.AppSettings.setValue("viewState_Temperature", self.chk3.isChecked())

    def toggle_RandD(self):
        if self.current_timer.isActive():
            self.current_timer.stop()
        if not self.chk4.isChecked():
            Log.d("Hiding Resonance/Dissipation plot(s)")
            self.parent.PlotsWin.ui2.pltB.setVisible(False)
        else:
            Log.d("Showing Resonance/Dissipation plot(s)")
            self.parent.PlotsWin.ui2.pltB.setVisible(True)
        self.parent.AppSettings.setValue("viewState_Resonance_Dissipation", self.chk4.isChecked())

    def show_top_plot(self):
        toggle_console = False
        if self.chk2.isChecked() or self.chk3.isChecked():
            Log.d("Showing top plots window")
            toggle_console = self.parent.PlotsWin.ui2.plt.isVisible() == False
            self.parent.PlotsWin.ui2.plt.setVisible(True)
        return toggle_console

    def hide_top_plot(self, toggle_console):
        if self.chk2.isChecked() or self.chk3.isChecked():
            # Remove the timer hack completely:
            # if toggle_console:
            #     if self.current_timer.isActive():
            #         self.current_timer.stop()
            #     self.current_timer.setSingleShot(True)
            #     self.current_timer.start(100)

            # Instead, optionally trigger a proper layout refresh if PyQt needs a nudge
            if toggle_console:
                self.parent.PlotsWin.layout().activate()  # Or update() / adjustSize()
        else:
            Log.d("Hiding top plots window")
            self.parent.PlotsWin.ui2.plt.setVisible(False)

    # def double_toggle_plots(self):
    #     Log.d("Toggling console window (for sizing)")
    #     self.chk4.setChecked(not self.chk4.isChecked())
    #     self.toggle_RandD()
    #     self.chk4.setChecked(not self.chk4.isChecked())
    #     QtCore.QTimer.singleShot(0, self.toggle_RandD)

    def view_tutorials(self):
        self.parent.TutorialWin.setVisible(not self.parent.TutorialWin.isVisible())
        self.chk5.setChecked(self.parent.TutorialWin.isVisible())

    def open_file(self, filepath, relative_to_cwd=True):
        try:
            if relative_to_cwd:
                fullpath = os.path.join(Architecture.get_path(), filepath)
            os_type = Architecture.get_os()
            if os_type == OSType.macosx:  # macOS
                subprocess.call(("open", fullpath))
            elif os_type == OSType.windows:  # Windows
                os.startfile(fullpath)
            elif os_type == OSType.linux:  # linux
                subprocess.call(("xdg-open", fullpath))
            else:  # other variants
                Log.w("Unknown OS Type:", os_type)
                Log.w("Assuming Linux variant...")
                subprocess.call(("xdg-open", fullpath))
        except:
            Log.e(TAG, f'ERROR: Cannot open "{os.path.split(fullpath)[1]}"')

    def release_notes(self):
        self.open_file(f"docs/Release Notes {Constants.app_version}.pdf")

    def fw_change_log(self):
        self.open_file(f"QATCH_Q-1_FW_py_{Constants.best_fw_version}/FW Change Control Doc.pdf")

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
        if hasattr(self.parent, "_license_manager"):
            lm: LicenseManager = self.parent._license_manager
            if hasattr(lm, "refresh_license") and callable(lm.refresh_license):
                lm.refresh_license()
        color, status = self.parent.start_download(True)
        if color == "#ff0000":
            if status == "ERROR":
                PopUp.warning(self, "Check for Updates", "An error occurred checking for updates.")
            if status == "OFFLINE":
                PopUp.warning(self, "Check for Updates", "Unable to check online for updates.")
        elif color != "#00ff00":
            technicality = " available " if color == "#00c600" else " supported "
            PopUp.information(
                self,
                "Check for Updates",
                f"You are running the latest{technicality}version.",
            )

    def closeEvent(self, event):
        # Log.d(" Exit Setup/Control GUI")
        if hasattr(self, "close_no_confirm"):
            res = True
        else:
            res = PopUp.question(
                self,
                Constants.app_title,
                "Are you sure you want to quit QATCH Q-1 application now?",
                True,
            )
        if res:
            # self.close()
            QtWidgets.QApplication.quit()
        else:
            event.ignore()


class GlassControlsWidget(QtWidgets.QWidget):
    """Frosted-glass container that provides the toolbar's gradient backdrop.

    Renders the same cool-blue gradient palette used by GlassCard in
    ui_login when no live backdrop is available, overlaid with the standard
    white-tint, shimmer, and dual-border glass language.
    """

    _RADIUS: float = 10.0

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        rect_f = QtCore.QRectF(self.rect())

        # Clip to rounded rectangle
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)

        # Match modeMenuScrollArea: rgba(255,255,255,160) on the app gradient
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 160))

        # Faint cool tint — same underlying gradient tone as #E4EBF1
        p.fillRect(self.rect(), QtGui.QColor(228, 235, 241, 18))

        # Soft top shimmer
        shimmer = QtGui.QLinearGradient(0, 0, 0, 32)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 50))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Border — matches the sidebar's rgba(255,255,255,220) border
        p.setClipping(False)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)
        p.setPen(QtGui.QPen(QtGui.QColor(200, 210, 220, 80), 1.0))
        p.drawRoundedRect(
            rect_f.adjusted(1.5, 1.5, -1.5, -1.5),
            self._RADIUS - 1.5,
            self._RADIUS - 1.5,
        )

        p.end()


class GlassHeaderLabel(QtWidgets.QLabel):
    """Section-header label rendered as a brand-blue glass panel.

    Replaces the legacy solid ``background: #008EC0`` headers.  The
    hand-painted background carries the same shimmer/border pipeline as
    GlassCard while maintaining the QATCH cool-blue identity.
    """

    _RADIUS: float = 4.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        # Text colour and padding only — background handled in paintEvent
        self.setStyleSheet(
            "QLabel { color: rgba(255, 255, 255, 230); "
            "padding: 2px 6px; font-weight: bold; background: transparent; }"
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        rect_f = QtCore.QRectF(self.rect())
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)

        # Brand-blue gradient base
        grad = QtGui.QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0.0, QtGui.QColor(0, 118, 174))
        grad.setColorAt(1.0, QtGui.QColor(0, 158, 210))
        p.fillRect(self.rect(), QtGui.QBrush(grad))

        # Glass tints
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 45))
        p.fillRect(self.rect(), QtGui.QColor(180, 220, 245, 30))

        # Top shimmer
        shimmer = QtGui.QLinearGradient(0, 0, 0, self.height() * 0.65)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 55))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Borders
        p.setClipping(False)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(80, 160, 215, 130), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 130), 1.0))
        p.drawRoundedRect(
            rect_f.adjusted(1.5, 1.5, -1.5, -1.5),
            self._RADIUS - 1.5,
            self._RADIUS - 1.5,
        )

        p.end()
        # Render text via base class (respects alignment, QSS color)
        super().paintEvent(event)


class GlassStatusLabel(QtWidgets.QLabel):
    """Frosted-white glass panel for status and info displays.

    Replaces the legacy ``background: white; border: 1px solid #cccccc``
    status labels with a translucent glass treatment that integrates
    seamlessly with GlassControlsWidget.
    """

    _RADIUS: float = 5.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setStyleSheet(
            "QLabel { color: rgba(28, 40, 52, 210); " "padding: 2px 6px; background: transparent; }"
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing)

        rect_f = QtCore.QRectF(self.rect())
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)

        # Frosted white glass base
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 155))
        p.fillRect(self.rect(), QtGui.QColor(210, 225, 240, 40))

        # Top shimmer
        shimmer = QtGui.QLinearGradient(0, 0, 0, 36)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 80))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Borders
        p.setClipping(False)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(120, 160, 200, 110), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 160), 1.0))
        p.drawRoundedRect(
            rect_f.adjusted(1.5, 1.5, -1.5, -1.5),
            self._RADIUS - 1.5,
            self._RADIUS - 1.5,
        )

        p.end()
        super().paintEvent(event)


# ---------------------------------------------------------------------------
# Shared QSS fragments
# ---------------------------------------------------------------------------

_GLASS_BUTTON_QSS = """
    QPushButton {{
        background: transparent;
        color: rgba(30, 40, 55, 200);
        border: 1px solid transparent;
        border-radius: 4px;
        padding: {padding};
        font-size: 12px;
    }}
    QPushButton:hover {{
        background: rgba(229, 229, 229, 150);
        border: 1px solid transparent;
    }}
    QPushButton:pressed {{
        background: rgba(229, 229, 229, 200);
        border: 1px solid transparent;
    }}
    QPushButton:disabled {{
        color: rgba(30, 40, 55, 90);
        background: transparent;
        border: 1px solid transparent;
    }}
"""

_GLASS_TOOLBAR_QSS = """
    QToolBar {
        background: transparent;
        border: none;
        spacing: 2px;
    }
    QToolButton {
        background: transparent;
        color: rgba(30, 40, 55, 200);
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 4px 8px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 12px;
    }
    QToolButton:hover {
        background: rgba(229, 229, 229, 150);
        border: 1px solid transparent;
    }
    QToolButton:pressed {
        background: rgba(229, 229, 229, 200);
        border: 1px solid transparent;
    }
    QToolButton:checked {
        background: transparent;
        border: 1px solid transparent;
    }
    QToolButton:disabled {
        color: rgba(30, 40, 55, 90);
        background: transparent;
        border: 1px solid transparent;
    }
    QToolBar::separator {
        background: rgba(0, 0, 0, 22);
        width: 1px;
        margin: 5px 4px;
    }
"""

_GLASS_PROGRESSBAR_QSS = """
    QProgressBar {
        border: 1px solid rgba(0, 0, 0, 25);
        border-radius: 4px;
        text-align: center;
        color: rgba(30, 40, 55, 200);
        background: rgba(255, 255, 255, 120);
        font-weight: bold;
    }
    QProgressBar::chunk {
        background: qlineargradient(
            spread:pad, x1:0, y1:0, x2:1, y2:0,
            stop:0 rgba(10, 163, 230, 130),
            stop:1 rgba(10, 163, 230, 90)
        );
        border-radius: 3px;
    }
"""

_GLASS_TEMP_CONTROLLER_QSS = """
    QWidget#tempController {
        background: rgba(229, 229, 229, 80);
        border: none;
        border-radius: 6px;
    }
    QFrame#tempPidInfo {
        background: rgba(255, 255, 255, 120);
        border: 1px solid rgba(255, 255, 255, 200);
        border-radius: 5px;
    }
    QLabel#tempPidHeader {
        background: transparent;
        border: none;
        color: rgba(0, 118, 174, 220);
        font-weight: bold;
        font-size: 8pt;
        letter-spacing: 0.4px;
    }
    QLabel#tempStatusBanner {
        background: rgba(150, 155, 160, 120);
        color: rgba(30, 40, 55, 160);
        border: 1px solid rgba(255, 255, 255, 160);
        border-radius: 3px;
        padding: 0 6px;
        font-weight: bold;
    }
    QLabel {
        background: transparent;
        border: none;
        color: rgba(30, 40, 55, 200);
    }
    QSlider::groove:horizontal {
        height: 4px;
        background: rgba(0, 0, 0, 30);
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #0AA3E6;
        border: 1px solid rgba(0, 130, 200, 200);
        width: 12px;
        height: 12px;
        margin: -4px 0;
        border-radius: 6px;
    }
    QSlider::sub-page:horizontal {
        background: rgba(10, 163, 230, 120);
        border-radius: 2px;
    }
    QSlider::handle:horizontal:disabled {
        background: rgba(150, 170, 190, 140);
        border: 1px solid rgba(0, 0, 0, 30);
    }
"""


# ---------------------------------------------------------------------------
# Account dropdown — glass popup showing current user info
# ---------------------------------------------------------------------------


class _AvatarLabel(QtWidgets.QWidget):
    """Circular avatar rendered with QATCH brand-blue gradient + user initials."""

    def __init__(self, initials: str, parent=None) -> None:
        super().__init__(parent)
        self._initials = initials[:2].upper() if initials else "?"
        self.setAutoFillBackground(False)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing)
        r = min(self.width(), self.height()) - 2
        x = (self.width() - r) / 2
        y = (self.height() - r) / 2
        rect = QtCore.QRectF(x, y, r, r)

        grad = QtGui.QRadialGradient(rect.center(), r / 2)
        grad.setColorAt(0.0, QtGui.QColor(0, 158, 210))
        grad.setColorAt(1.0, QtGui.QColor(0, 100, 160))
        p.setBrush(QtGui.QBrush(grad))
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 90), 1.5))
        p.drawEllipse(rect)

        # Shimmer half-circle
        shimmer = QtGui.QLinearGradient(0, float(rect.top()), 0, float(rect.center().y()))
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 55))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.setBrush(QtGui.QBrush(shimmer))
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(rect)

        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QtGui.QColor(255, 255, 255, 235))
        p.drawText(rect.toRect(), QtCore.Qt.AlignCenter, self._initials)
        p.end()


class _GlassAccountInnerPanel(QtWidgets.QWidget):
    """Inner glass-morphism panel for the account popup.

    Paints the frosted-glass background with rounded corners.  The outer
    :class:`GlassAccountPopup` applies a :class:`QGraphicsDropShadowEffect`
    to this widget so the shadow follows the painted alpha mask, producing
    a soft, rounded drop shadow.  This mirrors the pattern used by
    ``RecoveryFilterWidget`` to avoid the rectangular OS popup outline.
    """

    _RADIUS: float = 10.0

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        rect_f = QtCore.QRectF(self.rect())
        _R = self._RADIUS

        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, _R, _R)
        p.setClipPath(clip)

        # Frosted white base — slightly higher alpha than before because the
        # outer widget is fully transparent (no manual shadow underlay)
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 235))
        p.fillRect(self.rect(), QtGui.QColor(228, 235, 241, 28))

        # Top shimmer
        shimmer = QtGui.QLinearGradient(0, 0, 0, 44)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 80))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Dual borders (outer warm white, inner cool grey)
        p.setClipping(False)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), _R, _R)
        p.setPen(QtGui.QPen(QtGui.QColor(200, 210, 220, 90), 1.0))
        p.drawRoundedRect(rect_f.adjusted(1.5, 1.5, -1.5, -1.5), _R - 1.5, _R - 1.5)

        p.end()


class GlassAccountPopup(QtWidgets.QWidget):
    """Frosted-glass dropdown panel for the Account toolbar button.

    Displays the active user's avatar, full name, and role badge.  Admin users
    additionally see a "Manage Users…" shortcut.  The popup uses ``Qt.Popup``
    so it closes automatically on any outside click.

    Implementation notes
    --------------------
    The popup is built as a transparent outer ``QWidget`` (this class) wrapping
    an inner :class:`_GlassAccountInnerPanel`.  The outer widget reserves margin
    space around the inner panel so a :class:`QGraphicsDropShadowEffect` applied
    to the inner panel renders a soft, rounded shadow that follows the panel's
    border-radius — exactly the trick used by ``RecoveryFilterWidget`` to fix
    the sharp shadow corners produced by manual painted shadows.

    The popup also tracks its main window: when the main window is resized or
    moved, the popup closes itself so it never floats outside the application.
    """

    # Margins reserved around the inner panel for the drop shadow.  Bottom is
    # larger to accommodate the shadow's positive Y offset.
    _SHADOW_MARGIN_L = 22
    _SHADOW_MARGIN_T = 18
    _SHADOW_MARGIN_R = 22
    _SHADOW_MARGIN_B = 26

    def __init__(
        self,
        open_manager_cb=None,
        sign_out_cb=None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(
            parent,
            QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint | QtCore.Qt.NoDropShadowWindowHint,
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

        self._open_manager_cb = open_manager_cb
        self._sign_out_cb = sign_out_cb
        self._main_window: Optional[QtWidgets.QWidget] = None  # set by show_anchored_to

        # -- outer container with shadow margins --
        self._panel = _GlassAccountInnerPanel(self)
        self._panel.setObjectName("AccountPopupInner")

        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(
            self._SHADOW_MARGIN_L,
            self._SHADOW_MARGIN_T,
            self._SHADOW_MARGIN_R,
            self._SHADOW_MARGIN_B,
        )
        outer_layout.setSpacing(0)
        outer_layout.addWidget(self._panel)

        # Soft drop shadow that follows the inner panel's painted alpha mask
        shadow = QtWidgets.QGraphicsDropShadowEffect(self._panel)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 4)
        shadow.setColor(QtGui.QColor(0, 20, 40, 110))
        self._panel.setGraphicsEffect(shadow)

        # -- resolve current session info (lazy import avoids circular deps) --
        # session_info() returns: [name, initials, role.name, created, modified, accessed]
        accessed: Optional[str] = None
        try:
            from QATCH.common.userProfiles import UserProfiles, UserRoles  # noqa: PLC0415

            is_valid, user_info = UserProfiles.session_info()
            if is_valid and user_info:
                name = user_info[0] or "Unknown"
                initials = user_info[1] or "?"
                role_name = user_info[2] or "NONE"
                # Index 5 = "accessed" timestamp ("Today, HH:MM:SS" or "YYYY-MM-DD HH:MM:SS")
                accessed = user_info[5] if len(user_info) > 5 else None
            else:
                name, initials, role_name = "Anonymous", "?", "NONE"
            is_admin = role_name == UserRoles.ADMIN.name
            is_signed_in = is_valid
        except Exception:
            name, initials, role_name = "Anonymous", "?", "NONE"
            is_admin = False
            is_signed_in = False

        # -- inner panel layout (all visible content lives here) --
        layout = QtWidgets.QVBoxLayout(self._panel)
        layout.setContentsMargins(14, 14, 14, 12)
        layout.setSpacing(8)

        # Avatar + name/role column
        header_row = QtWidgets.QHBoxLayout()
        header_row.setSpacing(12)

        avatar = _AvatarLabel(initials)
        avatar.setFixedSize(44, 44)
        header_row.addWidget(avatar, 0, QtCore.Qt.AlignTop)

        info_col = QtWidgets.QVBoxLayout()
        info_col.setSpacing(3)
        info_col.setContentsMargins(0, 1, 0, 0)

        name_lbl = QtWidgets.QLabel(name)
        name_lbl.setStyleSheet(
            "color: rgba(28,40,52,235); font-weight: bold; font-size: 13px; "
            "background: transparent; border: none;"
        )
        info_col.addWidget(name_lbl)

        # Subtle initials line under the name
        initials_lbl = QtWidgets.QLabel(f"Initials: {initials}")
        initials_lbl.setStyleSheet(
            "color: rgba(70, 90, 110, 180); font-size: 10px; "
            "background: transparent; border: none;"
        )
        info_col.addWidget(initials_lbl)

        _role_palette = {
            "ADMIN": ("rgba(0,118,174,215)", "white"),
            "OPERATE": ("rgba(40,155,75,200)", "white"),
            "ANALYZE": ("rgba(130,80,200,200)", "white"),
            "CAPTURE": ("rgba(200,125,0,200)", "white"),
        }
        bg, fg = _role_palette.get(role_name, ("rgba(140,150,160,160)", "rgba(28,40,52,180)"))
        role_badge = QtWidgets.QLabel(role_name)
        role_badge.setFixedHeight(17)
        role_badge.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 3px; "
            "padding: 1px 6px; font-size: 10px; font-weight: bold; border: none;"
        )
        # Wrap the badge so it doesn't stretch to full column width
        role_row = QtWidgets.QHBoxLayout()
        role_row.setContentsMargins(0, 2, 0, 0)
        role_row.setSpacing(0)
        role_row.addWidget(role_badge)
        role_row.addStretch()
        info_col.addLayout(role_row)

        header_row.addLayout(info_col, 1)
        layout.addLayout(header_row)

        # Last sign-in / status line
        if is_signed_in and accessed:
            last_lbl = QtWidgets.QLabel(f"Last access: {accessed}")
            last_lbl.setStyleSheet(
                "color: rgba(70, 90, 110, 175); font-size: 10px; "
                "background: transparent; border: none; padding-left: 1px;"
            )
            layout.addWidget(last_lbl)
        elif not is_signed_in:
            status_lbl = QtWidgets.QLabel("No active session")
            status_lbl.setStyleSheet(
                "color: rgba(140, 90, 30, 200); font-size: 10px; font-style: italic; "
                "background: transparent; border: none; padding-left: 1px;"
            )
            layout.addWidget(status_lbl)

        show_manage = is_admin
        show_sign_out = is_signed_in
        if show_manage or show_sign_out:
            # Hairline divider
            divider = QtWidgets.QFrame()
            divider.setFrameShape(QtWidgets.QFrame.HLine)
            divider.setStyleSheet(
                "QFrame { background: rgba(200,210,220,130); border: none; max-height: 1px; }"
            )
            layout.addWidget(divider)

        if show_manage:
            icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "user-circle.svg")
            manage_btn = GlassPushButton("  Manage Users…")
            manage_btn.setIcon(QtGui.QIcon(icon_path))
            manage_btn.setIconSize(QtCore.QSize(14, 14))
            manage_btn.setStyleSheet("""
                QPushButton {
                    background: rgba(0, 118, 174, 18);
                    color: rgba(0, 118, 174, 230);
                    border: 1px solid rgba(0, 118, 174, 55);
                    border-radius: 5px;
                    padding: 6px 10px;
                    text-align: left;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover  {
                    background: rgba(0, 142, 192, 35);
                    border: 1px solid rgba(0, 118, 174, 110);
                }
                QPushButton:pressed {
                    background: rgba(0, 118, 174, 70);
                    border: 1px solid rgba(0, 118, 174, 160);
                }
            """)
            manage_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            manage_btn.clicked.connect(self._on_manage_users)
            layout.addWidget(manage_btn)

        if show_sign_out:
            icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "sign-out.svg")
            sign_out_btn = GlassPushButton("  Sign Out")
            if os.path.exists(icon_path):
                sign_out_btn.setIcon(QtGui.QIcon(icon_path))
                sign_out_btn.setIconSize(QtCore.QSize(14, 14))
            sign_out_btn.setStyleSheet("""
                QPushButton {
                    background: rgba(200, 70, 40, 16);
                    color: rgba(170, 55, 30, 235);
                    border: 1px solid rgba(200, 70, 40, 60);
                    border-radius: 5px;
                    padding: 6px 10px;
                    text-align: left;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover  {
                    background: rgba(210, 80, 45, 38);
                    border: 1px solid rgba(200, 70, 40, 120);
                }
                QPushButton:pressed {
                    background: rgba(200, 70, 40, 75);
                    border: 1px solid rgba(200, 70, 40, 170);
                }
            """)
            sign_out_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            sign_out_btn.clicked.connect(self._on_sign_out)
            layout.addWidget(sign_out_btn)

        self._panel.setMinimumWidth(230)

    # -- public API -----------------------------------------------------------

    def show_anchored_to(
        self,
        anchor: QtWidgets.QWidget,
        main_window: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Show the popup pinned to ``anchor`` and constrained to ``main_window``.

        The popup's *visible* right edge aligns with the anchor button's right
        edge; the visible top edge sits 2 px below the anchor's bottom.  If the
        popup would extend past the main window's frame, its position is clamped
        so the visible panel stays inside the main window.  When the main window
        is later resized or moved while the popup is open, the popup closes
        itself to avoid floating outside the application.
        """
        self._main_window = main_window
        self.adjustSize()

        size = self.sizeHint()
        popup_w, popup_h = size.width(), size.height()

        # Anchor at the bottom-right corner of the button (in screen coords)
        anchor_br = anchor.mapToGlobal(QtCore.QPoint(anchor.width(), anchor.height()))

        # Position so the *visible* panel right edge aligns with the button's
        # right edge, 2 px below the button.  Account for the transparent
        # shadow margins on the outer widget.
        x = anchor_br.x() + self._SHADOW_MARGIN_R - popup_w
        y = anchor_br.y() + 2 - self._SHADOW_MARGIN_T

        # Clamp so the visible panel stays inside the main window
        x, y = self._clamp_to_main_window(x, y, popup_w, popup_h, anchor)
        self.move(x, y)

        # Track resize/move events on the main window so the popup never
        # ends up floating outside the application after a resize.
        if self._main_window is not None:
            self._main_window.installEventFilter(self)

        self.show()

    # -- positioning helpers --------------------------------------------------

    def _visible_rect_for(self, x: int, y: int, w: int, h: int) -> QtCore.QRect:
        """Return the *visible* panel rect for an outer-widget position.

        The outer widget reserves transparent shadow margins, so the visible
        rect is the outer rect minus those margins.
        """
        return QtCore.QRect(
            x + self._SHADOW_MARGIN_L,
            y + self._SHADOW_MARGIN_T,
            w - self._SHADOW_MARGIN_L - self._SHADOW_MARGIN_R,
            h - self._SHADOW_MARGIN_T - self._SHADOW_MARGIN_B,
        )

    def _clamp_to_main_window(
        self,
        x: int,
        y: int,
        popup_w: int,
        popup_h: int,
        anchor: QtWidgets.QWidget,
    ) -> tuple:
        """Adjust ``(x, y)`` so the visible panel stays inside the main window.

        Falls back to the anchor's screen geometry if no main window is set.
        """
        # Prefer the anchor widget's own top-level window (content geometry, screen
        # coords) so the popup is always clamped against the window that actually
        # contains the button — regardless of which QWidget was passed as
        # main_window.  Fall back to main_window, then the screen.
        top_level = anchor.window() if anchor is not None else None
        if top_level is not None:
            bounds = top_level.geometry()
        elif self._main_window is not None:
            bounds = self._main_window.geometry()
        else:
            screen = QtWidgets.QApplication.screenAt(anchor.mapToGlobal(QtCore.QPoint(0, 0)))
            bounds = screen.availableGeometry() if screen is not None else QtCore.QRect()

        if bounds.isNull():
            return x, y

        visible = self._visible_rect_for(x, y, popup_w, popup_h)

        # Horizontal clamp
        if visible.right() > bounds.right():
            x -= visible.right() - bounds.right()
            visible = self._visible_rect_for(x, y, popup_w, popup_h)
        if visible.left() < bounds.left():
            x += bounds.left() - visible.left()
            visible = self._visible_rect_for(x, y, popup_w, popup_h)

        # Vertical clamp — if the popup spills off the bottom, flip it above
        # the anchor button.
        if visible.bottom() > bounds.bottom():
            anchor_top = anchor.mapToGlobal(QtCore.QPoint(0, 0)).y()
            y_above = anchor_top - 2 - popup_h + self._SHADOW_MARGIN_B
            visible_above = self._visible_rect_for(x, y_above, popup_w, popup_h)
            if visible_above.top() >= bounds.top():
                y = y_above
            else:
                # Neither orientation fits — just clamp to the bottom edge
                y -= visible.bottom() - bounds.bottom()

        return x, y

    # -- event handling -------------------------------------------------------

    def eventFilter(  # noqa: N802 — Qt naming
        self, watched: QtCore.QObject, event: QtCore.QEvent
    ) -> bool:
        """Close the popup if the main window is resized or moved.

        The popup is positioned in screen coordinates against the anchor at the
        time of show.  Re-anchoring on every resize would race the layout
        engine, so the safer behaviour is to dismiss the popup and let the user
        re-open it once the new window geometry has settled.
        """
        if watched is self._main_window and event.type() in (
            QtCore.QEvent.Resize,
            QtCore.QEvent.Move,
            QtCore.QEvent.WindowStateChange,
        ):
            self.close()
        return super().eventFilter(watched, event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 — Qt naming
        if self._main_window is not None:
            try:
                self._main_window.removeEventFilter(self)
            except Exception:
                pass
            self._main_window = None
        super().closeEvent(event)

    # -- slots ----------------------------------------------------------------

    def _on_manage_users(self) -> None:
        self.close()
        if self._open_manager_cb:
            self._open_manager_cb()

    def _on_sign_out(self) -> None:
        self.close()
        if self._sign_out_cb:
            self._sign_out_cb()


# ---------------------------------------------------------------------------
# Temperature label — emits textUpdated so the display panel can react
# ---------------------------------------------------------------------------


class TemperatureLabel(QtWidgets.QLabel):
    """QLabel that fires textUpdated whenever setText() is called.

    Keeps full backward compatibility (callers use setText as normal) while
    letting the new split-display panel observe changes without polling.
    """

    textUpdated = QtCore.pyqtSignal(str)

    def setText(self, text: str) -> None:
        super().setText(text)
        self.textUpdated.emit(text)


# ---------------------------------------------------------------------------
# Helpers for glass toggles (label + switch rows, radio-group compatibility)
# ---------------------------------------------------------------------------


class _SectionHeader(QtWidgets.QLabel):
    """Soft, muted section header mirroring the account dropdown's typography.

    Replaces the heavy blue GlassHeaderLabel pills inside the advanced panel
    with quiet uppercase gray text, so the panel reads as clean grouped
    sections rather than a grid of competing colored bars.
    """

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text.upper(), parent)
        self.setStyleSheet(
            "QLabel { color: rgba(70, 90, 110, 200); font-size: 10px; "
            "font-weight: bold; letter-spacing: 1px; background: transparent; "
            "border: none; padding: 0px 1px; }"
        )


def _hairline() -> QtWidgets.QFrame:
    """A 1px divider matching the account dropdown's hairline separators."""
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.HLine)
    line.setStyleSheet(
        "QFrame { background: rgba(200, 210, 220, 130); border: none; max-height: 1px; }"
    )
    return line


class LabeledToggle(QtWidgets.QWidget):
    """A GlassToggle paired with a text label in a horizontal row.

    Exposes the subset of the QCheckBox API used by the rest of the app
    (``isChecked``, ``setChecked``, ``setEnabled``, ``setText``, ``toggled``)
    so it can stand in for a checkbox without touching call sites.
    """

    def __init__(self, text: str = "", parent=None, *, label_left: bool = False) -> None:
        super().__init__(parent)
        self.toggle = GlassToggle(self)
        self.label = QtWidgets.QLabel(text, self)
        self.label.setStyleSheet(
            "background: transparent; border: none; color: rgba(30, 40, 55, 215);"
        )
        self.label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        if label_left:
            lay.addWidget(self.label)
            lay.addWidget(self.toggle)
            lay.addStretch()
        else:
            lay.addWidget(self.toggle)
            lay.addWidget(self.label)
            lay.addStretch()

        # Re-expose the toggle's signal as our own.
        self.toggled = self.toggle.toggled

    # -- QCheckBox-compatible surface --------------------------------------
    def isChecked(self) -> bool:
        return self.toggle.isChecked()

    def setChecked(self, checked: bool) -> None:
        self.toggle.setChecked(checked)

    def setText(self, text: str) -> None:
        self.label.setText(text)

    def text(self) -> str:
        return self.label.text()

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self.toggle.setEnabled(enabled)
        self.label.setEnabled(enabled)


class _ToggleButtonGroupShim:
    """Minimal stand-in for the old QButtonGroup over the auto-lock radios.

    The single auto-lock GlassToggle represents Automatic(on)/Manual(off).
    This shim preserves the only QButtonGroup method the app used:
    ``checkedId()`` returning 1 for Automatic and 0 for Manual.
    """

    def __init__(self, toggle: GlassToggle) -> None:
        self._toggle = toggle

    def checkedId(self) -> int:
        return 1 if self._toggle.isChecked() else 0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class UIControls:  # QtWidgets.QMainWindow

    def setupUi(self, MainWindow1):
        USE_FULLSCREEN = QDesktopWidget().availableGeometry().width() == 2880
        SHOW_SIMPLE_CONTROLS = True
        self.cal_initialized = False
        self.parent = MainWindow1

        MainWindow1.setObjectName("MainWindow1")
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

        # Shared chevron icon for the glass (animated) combo boxes.
        self._combo_chevron = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "down-chevron.svg"
        )

        # frequency/quartz combobox -------------------------------------------
        self.cBox_Speed = AnimatedComboBox(icon_path=self._combo_chevron)
        self.cBox_Speed.setEditable(False)
        self.cBox_Speed.setObjectName("cBox_Speed")
        if USE_FULLSCREEN:
            self.cBox_Speed.setFixedHeight(50)
        self.Layout_controls.addWidget(self.cBox_Speed, 4, 1, 1, 1)

        # stop button ---------------------------------------------------------
        # Shared control-button sizing (thick enough for icon + label).
        _CTRL_BTN_H = 40
        _CTRL_ICON = QtCore.QSize(20, 20)
        self.pButton_Stop = GlassPushButton(variant="neutral")
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "stop-filled.svg")
        self.pButton_Stop.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Stop.setIconSize(_CTRL_ICON)
        self.pButton_Stop.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Stop.setFixedHeight(_CTRL_BTN_H)
        self.pButton_Stop.setObjectName("pButton_Stop")
        self.pButton_Stop.set_icon_left(True)
        self.Layout_controls.addWidget(self.pButton_Stop, 3, 6, 1, 1)

        # COM port combobox ---------------------------------------------------
        self.cBox_Port = AnimatedComboBox(icon_path=self._combo_chevron)
        self.cBox_Port.setEditable(False)
        self.cBox_Port.setObjectName("cBox_Port")
        if USE_FULLSCREEN:
            self.cBox_Port.setFixedHeight(50)
        self.Layout_controls.addWidget(self.cBox_Port, 2, 1, 1, 1)

        # Identify button
        _CIRCLE_D = 34  # diameter for circular icon buttons
        self.pButton_ID = GlassPushButton(variant="default")
        self.pButton_ID.setToolTip("Identify selected Serial COM Port")
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "search.svg")
        self.pButton_ID.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_ID.setIconSize(QtCore.QSize(18, 18))
        self.pButton_ID.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pButton_ID.setFixedSize(_CIRCLE_D, _CIRCLE_D)  # square -> circle
        self.pButton_ID.setObjectName("pButton_ID")
        self.Layout_controls.addWidget(self.pButton_ID, 2, 2, 1, 1)

        # Refresh button
        self.pButton_Refresh = GlassPushButton(variant="default")
        self.pButton_Refresh.setToolTip("Refresh Serial COM Port list")
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "refresh-cw.svg")
        self.pButton_Refresh.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Refresh.setIconSize(QtCore.QSize(18, 18))
        self.pButton_Refresh.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pButton_Refresh.setFixedSize(_CIRCLE_D, _CIRCLE_D)  # square -> circle
        self.pButton_Refresh.setObjectName("pButton_Refresh")
        self.Layout_controls.addWidget(self.pButton_Refresh, 2, 3, 1, 1)

        # Configure button — replaces the in-dropdown "Configure..." item.
        # Wired to the main window's device-info handler in mainWindow.py.
        self.pButton_Configure = GlassPushButton(variant="default")
        self.pButton_Configure.setToolTip("Configure device / position info")
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "gear.svg")
        self.pButton_Configure.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Configure.setIconSize(QtCore.QSize(18, 18))
        self.pButton_Configure.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pButton_Configure.setFixedSize(_CIRCLE_D, _CIRCLE_D)  # square -> circle
        self.pButton_Configure.setObjectName("pButton_Configure")
        self.Layout_controls.addWidget(self.pButton_Configure, 2, 4, 1, 1)

        # Operation mode - source ---------------------------------------------
        self.cBox_Source = AnimatedComboBox(icon_path=self._combo_chevron)
        self.cBox_Source.setObjectName("cBox_Source")
        if USE_FULLSCREEN:
            self.cBox_Source.setFixedHeight(50)
        self.Layout_controls.addWidget(self.cBox_Source, 2, 0, 1, 1)

        # Frequency hopping toggle --------------------------------------------
        self.chBox_freqHop = LabeledToggle("Mode Hop")
        self.chBox_freqHop.setEnabled(True)
        self.chBox_freqHop.setChecked(False)
        self.chBox_freqHop.setObjectName("chBox_freqHop")
        self.Layout_controls.addWidget(self.chBox_freqHop, 4, 2, 1, 2)

        # Noise correction toggle ---------------------------------------------
        self.chBox_correctNoise = LabeledToggle("Show amplitude curve")
        self.chBox_correctNoise.setEnabled(True)
        self.chBox_correctNoise.setChecked(True)
        self.chBox_correctNoise.setObjectName("chBox_correctNoise")
        self.Layout_controls.addWidget(self.chBox_correctNoise, 5, 1, 1, 3)

        # Cartridge Auto-Lock -------------------------------------------------
        self.l9 = GlassHeaderLabel("Cartridge Auto-Lock")
        if USE_FULLSCREEN:
            self.l9.setFixedHeight(50)
        self.Layout_controls.addWidget(self.l9, 1, 4, 1, 1)

        # Cartridge Controls (single glass toggle: Manual <-> Automatic) -----
        self.toggle_Cartridge = GlassToggle()
        self.toggle_Cartridge.setToolTip("""
            <b><u>Auto-Lock Mode:</u></b><br/>
            <b>Automatic</b> (on): locks before init/run; useful if the user forgets.<br/>
            <b>Manual</b> (off): you control lock position; must lock before init/run.
            """)
        self.toggle_Cartridge.setChecked(True)  # default: Automatic

        self.lbl_lock_manual = QtWidgets.QLabel("Manual")
        self.lbl_lock_auto = QtWidgets.QLabel("Automatic")
        for _lbl in (self.lbl_lock_manual, self.lbl_lock_auto):
            _lbl.setStyleSheet(
                "background: transparent; border: none; color: rgba(30, 40, 55, 215);"
            )
            _lbl.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

        self.layMode = QtWidgets.QHBoxLayout()
        self.layMode.setContentsMargins(0, 0, 0, 0)
        self.layMode.setSpacing(8)
        self.layMode.addWidget(self.lbl_lock_manual)
        self.layMode.addWidget(self.toggle_Cartridge)
        self.layMode.addWidget(self.lbl_lock_auto)
        self.layMode.addStretch()
        self.grpMode = QtWidgets.QGroupBox("Auto-Lock Mode:")
        self.grpMode.setLayout(self.layMode)
        self.Layout_controls.addWidget(self.grpMode, 2, 4, 3, 1)

        # Backward-compatibility shims so existing call sites keep working:
        #   rCartridgeMode.checkedId() -> 1 (Automatic) / 0 (Manual)
        #   rButton_Automatic / rButton_Manual -> .setEnabled() proxies
        self.rCartridgeMode = _ToggleButtonGroupShim(self.toggle_Cartridge)
        self.rButton_Automatic = self.toggle_Cartridge
        self.rButton_Manual = self.toggle_Cartridge

        # start button --------------------------------------------------------
        self.pButton_Start = GlassPushButton(variant="neutral")
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "play-filled.svg")
        self.pButton_Start.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Start.setIconSize(_CTRL_ICON)
        self.pButton_Start.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Start.setFixedHeight(_CTRL_BTN_H)
        self.pButton_Start.setObjectName("pButton_Start")
        self.pButton_Start.set_icon_left(True)
        self.Layout_controls.addWidget(self.pButton_Start, 2, 6, 1, 1)

        # Add signal for Run Controls UI to handle START from Advanced menu ---
        self.pButton_Start.clicked.connect(
            lambda: (
                self.run_controls.set_running(True)
                if (
                    (OperationType(self.cBox_Source.currentIndex()) == OperationType.measurement)
                    and hasattr(self, "run_controls")
                )
                else None
            )
        )

        # clear plots button --------------------------------------------------
        self.pButton_Clear = GlassPushButton(variant="neutral")
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "clear-plot.svg")
        self.pButton_Clear.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Clear.setIconSize(_CTRL_ICON)
        self.pButton_Clear.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Clear.setFixedHeight(_CTRL_BTN_H)
        self.pButton_Clear.setObjectName("pButton_Clear")
        self.pButton_Clear.set_icon_left(True)
        self.Layout_controls.addWidget(self.pButton_Clear, 2, 5, 1, 1)

        # Plot mode toggle (Absolute <-> Reference) -------------------------
        # Replaces the old Set/Reset Reference push button. Toggle ON = Reference
        # mode, OFF = Absolute mode. pButton_Reference is kept as an alias so the
        # existing mainWindow wiring (setEnabled / setChecked / clicked) is
        # preserved without changes.
        self.toggle_PlotMode = GlassToggle()
        self.toggle_PlotMode.setToolTip(
            "<b>Plot Mode</b><br/>Off: Absolute &nbsp;|&nbsp; On: Reference"
        )
        self.toggle_PlotMode.setChecked(False)  # default: Absolute

        self.lbl_plot_absolute = QtWidgets.QLabel("Absolute")
        self.lbl_plot_reference = QtWidgets.QLabel("Reference")
        for _lbl in (self.lbl_plot_absolute, self.lbl_plot_reference):
            _lbl.setStyleSheet(
                "background: transparent; border: none; color: rgba(30, 40, 55, 215);"
            )
            _lbl.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

        # Alias: existing code refers to pButton_Reference for enable/check/click.
        self.pButton_Reference = self.toggle_PlotMode

        # restore factory defaults --------------------------------------------
        self.pButton_ResetApp = GlassPushButton(variant="neutral")
        self.pButton_ResetApp.setIconSize(_CTRL_ICON)
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "factory-reset.svg")
        self.pButton_ResetApp.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_ResetApp.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_ResetApp.setFixedHeight(_CTRL_BTN_H)
        self.pButton_ResetApp.setObjectName("pButton_ResetApp")
        self.pButton_ResetApp.set_icon_left(True)
        self.Layout_controls.addWidget(self.pButton_ResetApp, 4, 5, 1, 1)

        # samples SpinBox -----------------------------------------------------
        self.sBox_Samples = QtWidgets.QSpinBox()
        self.sBox_Samples.setMinimum(1)
        self.sBox_Samples.setMaximum(100000)
        self.sBox_Samples.setProperty("value", 500)
        self.sBox_Samples.setObjectName("sBox_Samples")
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

        # temperature Control label (hidden data conduit — kept for compat) ----
        self.lTemp = TemperatureLabel()
        self.lTemp.setText("PV:--.--C SP:--.--C OP:----")
        self.lTemp.setAlignment(QtCore.Qt.AlignCenter)
        self.lTemp.setFont(QtGui.QFont("Consolas", -1))
        self.lTemp.hide()
        self.Layout_controls.addWidget(self.lTemp, 2, 4, 1, 1)

        # temperature Control button ------------------------------------------
        self.pTemp = QtWidgets.QPushButton()
        self.pTemp.setText("Start Temp Control")
        if USE_FULLSCREEN:
            self.pTemp.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pTemp, 4, 4, 1, 1)

        # Control Buttons ------------------------------------------------------
        self.l = GlassHeaderLabel("Control Buttons")
        if USE_FULLSCREEN:
            self.l.setFixedHeight(50)
        self.Layout_controls.addWidget(self.l, 1, 5, 1, 2)

        # Operation Mode -------------------------------------------------------
        self.l0 = GlassHeaderLabel("Operation Mode")
        if USE_FULLSCREEN:
            self.l0.setFixedHeight(50)
        self.Layout_controls.addWidget(self.l0, 1, 0, 1, 1)

        # Resonance Frequency / Quartz Sensor ---------------------------------
        self.l2 = GlassHeaderLabel("Resonance Frequency / Quartz Sensor")
        if USE_FULLSCREEN:
            self.l2.setFixedHeight(50)
        self.Layout_controls.addWidget(self.l2, 3, 1, 1, 3)

        # Serial COM Port -----------------------------------------------------
        self.l1 = GlassHeaderLabel("Serial COM Port")
        if USE_FULLSCREEN:
            self.l1.setFixedHeight(50)
        self.Layout_controls.addWidget(self.l1, 1, 1, 1, 3)

        # logo ----------------------------------------------------------------
        self.l3 = QtWidgets.QLabel()
        self.l3.setAlignment(QtCore.Qt.AlignRight)
        self.Layout_controls.addWidget(self.l3, 4, 7, 1, 1)
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/qatch-logo_full.jpg")
        if USE_FULLSCREEN:
            pixmap = QtGui.QPixmap(icon_path)
            pixmap = pixmap.scaled(250, 50, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            self.l3.setPixmap(pixmap)
        else:
            self.l3.setPixmap(QtGui.QPixmap(icon_path))

        # qatch link ----------------------------------------------------------
        self.l4 = QtWidgets.QLabel()
        self.Layout_controls.addWidget(self.l4, 3, 7, 1, 1)

        def link(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))

        self.l4.linkActivated.connect(link)
        self.l4.setAlignment(QtCore.Qt.AlignRight)
        self.l4.setText(
            '<a href="https://qatchtech.com/"> <font size=4 color=#008EC0 >qatchtech.com</font>'
        )

        # info@qatchtech.com Mail -----------------------------------------------
        self.lmail = QtWidgets.QLabel()
        self.Layout_controls.addWidget(self.lmail, 2, 7, 1, 1)

        def linkmail(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))

        self.lmail.linkActivated.connect(linkmail)
        self.lmail.setAlignment(QtCore.Qt.AlignRight)
        self.lmail.setText(
            '<a href="mailto:info@qatchtech.com"> <font color=#008EC0 >info@qatchtech.com</font>'
        )

        # software user guide --------------------------------------------------------
        self.lg = QtWidgets.QLabel()
        self.Layout_controls.addWidget(self.lg, 1, 7, 1, 1)

        def link(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))

        self.lg.linkActivated.connect(link)
        self.lg.setAlignment(QtCore.Qt.AlignRight)
        self.lg.setText(
            '<a href="file://{}/docs/userguide.pdf"> <font color=#008EC0 >User Guide</font>'.format(
                Architecture.get_path()
            )
        )

        # Save file / TEC Temperature Control header --------------------------
        self.infosave = GlassHeaderLabel("TEC Temperature Control")
        if USE_FULLSCREEN:
            self.infosave.setFixedHeight(50)
        self.Layout_controls.addWidget(self.infosave, 1, 4, 1, 1)

        # Program Status standby ----------------------------------------------
        self.infostatus = GlassStatusLabel()
        self.infostatus.setAlignment(QtCore.Qt.AlignCenter)
        self.infostatus.setText("Program Status Standby")
        if USE_FULLSCREEN:
            self.infostatus.setFixedHeight(50)
        self.Layout_controls.addWidget(self.infostatus, 5, 5, 1, 2)

        # Infobar -------------------------------------------------------------
        self.infobar = QtWidgets.QLineEdit()
        self.infobar.setReadOnly(True)
        self.infobar_label = GlassStatusLabel()
        self.infobar.textChanged.connect(self.infobar_label.setText)
        if SHOW_SIMPLE_CONTROLS:
            self.infobar.textChanged.connect(self._update_progress_text)
        if USE_FULLSCREEN:
            self.infobar_label.setFixedHeight(50)
        self.Layout_controls.addWidget(self.infobar_label, 0, 0, 1, 7)

        # Multiplex -----------------------------------------------------------
        self.lmp = GlassHeaderLabel("Multiplex Mode")
        if USE_FULLSCREEN:
            self.lmp.setFixedHeight(50)
        self.Layout_controls.addWidget(self.lmp, 3, 0, 1, 1)

        self.cBox_MultiMode = AnimatedComboBox(icon_path=self._combo_chevron)
        self.cBox_MultiMode.setObjectName("cBox_MultiMode")
        self.cBox_MultiMode.addItems(["1 Channel", "2 Channels", "3 Channels", "4 Channels"])
        self.cBox_MultiMode.setCurrentIndex(0)
        if USE_FULLSCREEN:
            self.cBox_MultiMode.setFixedHeight(50)

        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons")
        self.pButton_PlateConfig = GlassPushButton(variant="default")
        self.pButton_PlateConfig.setIcon(QtGui.QIcon(os.path.join(icon_path, "gear.svg")))
        self.pButton_PlateConfig.setIconSize(QtCore.QSize(18, 18))
        self.pButton_PlateConfig.setToolTip("Plate Configuration...")
        self.pButton_PlateConfig.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pButton_PlateConfig.setFixedSize(_CIRCLE_D, _CIRCLE_D)  # square -> circle
        self.pButton_PlateConfig.clicked.connect(self.doPlateConfig)
        self.hBox_MultiConfig = QtWidgets.QHBoxLayout()
        self.hBox_MultiConfig.addWidget(self.cBox_MultiMode, 3)
        self.hBox_MultiConfig.addWidget(self.pButton_PlateConfig, 1)
        self.Layout_controls.addLayout(self.hBox_MultiConfig, 4, 0, 1, 1)

        # Disable Plate Configuration when only a single channel is selected or
        # available — a 1-channel setup has no plate layout to configure.
        self.cBox_MultiMode.currentIndexChanged.connect(self._update_plate_config_enabled)
        self._update_plate_config_enabled()

        self.chBox_MultiAuto = LabeledToggle("Auto-detect channel count")
        self.chBox_MultiAuto.setEnabled(True)
        self.chBox_MultiAuto.setChecked(True)
        self.chBox_MultiAuto.setObjectName("chBox_MultiAuto")
        self.Layout_controls.addWidget(self.chBox_MultiAuto, 5, 0, 1, 1)

        # Progressbar ---------------------------------------------------------
        self.run_progress_bar = QtWidgets.QProgressBar()
        self.run_progress_bar.setGeometry(QtCore.QRect(0, 0, 50, 10))
        self.run_progress_bar.setObjectName("progressBar")
        self.run_progress_bar.setStyleSheet(_GLASS_PROGRESSBAR_QSS)

        if USE_FULLSCREEN:
            self.run_progress_bar.setFixedHeight(50)
        if SHOW_SIMPLE_CONTROLS:
            self.run_progress_bar.valueChanged.connect(self._update_progress_value)

        self.run_progress_bar.setValue(0)
        self.run_progress_bar.setHidden(True)

        self.Layout_controls.setColumnStretch(0, 0)
        self.Layout_controls.setColumnStretch(1, 1)
        self.Layout_controls.setColumnStretch(2, 0)
        self.Layout_controls.setColumnStretch(3, 0)
        self.Layout_controls.setColumnStretch(4, 2)
        self.Layout_controls.setColumnStretch(5, 2)
        self.Layout_controls.setColumnStretch(6, 2)
        self.Layout_controls.addWidget(self.run_progress_bar, 0, 7, 1, 1)
        self.gridLayout.addLayout(self.Layout_controls, 7, 1, 1, 1)

        # ---- Simple / toolbar layout ----------------------------------------

        self.toolLayout = QtWidgets.QVBoxLayout()
        self.toolBar = QtWidgets.QHBoxLayout()

        self.tool_bar = QtWidgets.QToolBar()
        self.tool_bar.setIconSize(QtCore.QSize(50, 30))
        self.tool_bar.setStyleSheet(_GLASS_TOOLBAR_QSS)

        self.tool_NextPortRow = NumberIconButton()
        self.tool_NextPortRow.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_NextPortRow.setText("Next Port")
        self.tool_NextPortRow.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.tool_NextPortRow.clicked.connect(self.action_next_port)
        self.action_NextPortRow = self.tool_bar.addWidget(self.tool_NextPortRow)

        self.action_NextPortSep = self.tool_bar.addSeparator()

        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons")

        icon_init = QtGui.QIcon()
        icon_init.addPixmap(
            QtGui.QPixmap(os.path.join(icon_path, "speedometer.svg")), QtGui.QIcon.Normal
        )
        self.tool_Initialize = QtWidgets.QToolButton()
        self.tool_Initialize.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_Initialize.setIcon(icon_init)
        self.tool_Initialize.setText("Initialize")
        self.tool_Initialize.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.tool_Initialize.clicked.connect(self.action_initialize)
        self.tool_bar.addWidget(self.tool_Initialize)

        self.tool_bar.addSeparator()

        # RunControls composite widget ----------------------------------------
        self.run_controls = RunControls()
        self.run_controls.startRequested.connect(self.action_start)
        self.run_controls.stopRequested.connect(self.action_stop)
        self.run_controls.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.run_controls.setEnabled(False)
        self.tool_Start = self.run_controls  # backward-compat alias
        self.tool_Stop = self.run_controls
        self.tool_bar.addWidget(self.run_controls)
        self.tool_bar.addSeparator()

        icon_reset = QtGui.QIcon()
        icon_reset.addPixmap(
            QtGui.QPixmap(os.path.join(icon_path, "reset.svg")), QtGui.QIcon.Normal
        )
        self.tool_Reset = QtWidgets.QToolButton()
        self.tool_Reset.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_Reset.setIcon(icon_reset)
        self.tool_Reset.setText("Reset")
        self.tool_Reset.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.tool_Reset.clicked.connect(self.action_reset)
        self.tool_bar.addWidget(self.tool_Reset)

        self.tool_bar.addSeparator()

        self._warningTimer = QtCore.QTimer()
        self._warningTimer.setSingleShot(True)
        self._warningTimer.timeout.connect(self.action_tempcontrol_warning)
        self._warningTimer.setInterval(2000)

        icon_temp = QtGui.QIcon()
        icon_temp.addPixmap(
            QtGui.QPixmap(os.path.join(icon_path, "temperature-control.svg")), QtGui.QIcon.Normal
        )
        self.tool_TempControl = QtWidgets.QToolButton()
        self.tool_TempControl.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_TempControl.setIcon(icon_temp)
        self.tool_TempControl.setText("Temp Control")
        self.tool_TempControl.setCheckable(True)
        self.tool_TempControl.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.tool_TempControl.clicked.connect(self.action_tempcontrol)
        self.tool_TempControl.enterEvent = self.action_tempcontrol_warn_start
        self.tool_TempControl.leaveEvent = self.action_tempcontrol_warn_stop
        self.tool_bar.addWidget(self.tool_TempControl)

        self.toolBar.addWidget(self.tool_bar)

        # Temperature controller widget — starts collapsed, expands on toggle ----
        self.tempController = QtWidgets.QWidget()
        self.tempController.setObjectName("tempController")
        self.tempController.enterEvent = self.action_tempcontrol_warn_start
        self.tempController.leaveEvent = self.action_tempcontrol_warn_stop
        self.tempController.setMinimumWidth(0)
        self.tempController.setMaximumWidth(0)  # collapsed until activated
        self.tempController.setStyleSheet(_GLASS_TEMP_CONTROLLER_QSS)

        # Status banner — coloured background + descriptive text. Sits ABOVE the
        # slider on the left side of the panel, matching the wireframe.
        self.tempStatusBar = QtWidgets.QLabel("Offline")
        self.tempStatusBar.setObjectName("tempStatusBanner")
        self.tempStatusBar.setFixedHeight(18)
        self.tempStatusBar.setAlignment(QtCore.Qt.AlignCenter)
        _status_font = QtGui.QFont()
        _status_font.setPointSize(7)
        _status_font.setBold(True)
        self.tempStatusBar.setFont(_status_font)

        # Left column: status (top) above slider (bottom)
        left_col = QtWidgets.QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(4)
        left_col.addWidget(self.tempStatusBar)
        left_col.addWidget(self.slTemp)

        # PID Info panel (right side) — header + PV / SP / OP value stack
        value_font = QtGui.QFont("Consolas", 7)
        self.lPV = QtWidgets.QLabel("PV  --.--°C")
        self.lSP = QtWidgets.QLabel("SP  --.--°C")
        self.lOP = QtWidgets.QLabel("OP  ----")
        for lbl in (self.lPV, self.lSP, self.lOP):
            lbl.setFont(value_font)
            lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            lbl.setStyleSheet("background: transparent; border: none; color: rgba(30,40,55,210);")

        self.tempPidInfo = QtWidgets.QFrame()
        self.tempPidInfo.setObjectName("tempPidInfo")
        pid_layout = QtWidgets.QVBoxLayout(self.tempPidInfo)
        pid_layout.setContentsMargins(8, 4, 8, 4)
        pid_layout.setSpacing(1)

        pid_header = QtWidgets.QLabel("PID INFO")
        pid_header.setObjectName("tempPidHeader")
        pid_header.setAlignment(QtCore.Qt.AlignCenter)
        pid_layout.addWidget(pid_header)
        pid_layout.addWidget(self.lPV)
        pid_layout.addWidget(self.lSP)
        pid_layout.addWidget(self.lOP)

        # Assemble panel — [Status / Slider stacked on left]  |  [PID Info on right]
        self.tempLayout = QtWidgets.QHBoxLayout()
        self.tempLayout.setContentsMargins(8, 6, 8, 6)
        self.tempLayout.setSpacing(8)
        self.tempLayout.addLayout(left_col, 1)
        self.tempLayout.addWidget(self.tempPidInfo, 0)
        self.tempController.setLayout(self.tempLayout)
        self.toolBar.addWidget(self.tempController)

        # Set initial chevron on the toolbar button (collapsed → "›")
        self._set_temp_arrow(expand=False)

        # Wire live temperature updates to the display panel
        self.lTemp.textUpdated.connect(self._update_temp_display)

        self.toolBar.addStretch()

        self.tool_bar_2 = QtWidgets.QToolBar()
        self.tool_bar_2.setIconSize(QtCore.QSize(50, 30))
        self.tool_bar_2.setStyleSheet(_GLASS_TOOLBAR_QSS)

        icon_advanced = QtGui.QIcon()
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "gear.svg")
        icon_advanced.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal)
        self.tool_Advanced = QtWidgets.QToolButton()
        self.tool_Advanced.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_Advanced.setIcon(icon_advanced)
        self.tool_Advanced.setText("Advanced")
        self.tool_Advanced.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.tool_Advanced.clicked.connect(self.action_advanced)
        self.tool_bar_2.addWidget(self.tool_Advanced)

        self.tool_bar_2.addSeparator()

        icon_user = QtGui.QIcon()
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "user-circle.svg")
        icon_user.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal)
        icon_user.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Disabled)
        self.tool_User = QtWidgets.QToolButton()
        self.tool_User.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_User.setIcon(icon_user)
        self.tool_User.setText("Account")
        self.tool_User.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.tool_User.setEnabled(self._is_user_signed_in())
        self.tool_User.clicked.connect(self._toggle_account_popup)
        self.tool_bar_2.addWidget(self.tool_User)

        self.toolBar.addWidget(self.tool_bar_2)

        self.toolBar.setContentsMargins(8, 4, 8, 4)

        # Glass container for the entire toolbar row --------------------------
        self.toolBarWidget = GlassControlsWidget()
        self.toolBarWidget.setLayout(self.toolBar)

        self.toolLayout.addWidget(self.toolBarWidget)
        self.toolLayout.addWidget(self.run_progress_bar)

        if SHOW_SIMPLE_CONTROLS:
            self.toolLayout.setContentsMargins(6, 6, 6, 0)
            self.centralwidget.setLayout(self.toolLayout)

            self.Layout_controls.removeWidget(self.infosave)
            self.Layout_controls.removeWidget(self.lTemp)
            self.Layout_controls.removeWidget(self.slTemp)
            self.Layout_controls.removeWidget(self.pTemp)
            self.Layout_controls.removeWidget(self.run_progress_bar)
            self.Layout_controls.removeWidget(self.lg)
            self.Layout_controls.removeWidget(self.lmail)
            self.Layout_controls.removeWidget(self.l4)
            self.Layout_controls.removeWidget(self.l3)
            self.Layout_controls.removeWidget(self.infostatus)

            # The advanced container, warning banner, and popup are owned by
            # AdvancedMainWidget. Build a clean, sectioned layout (mirroring the
            # account dropdown) from the existing widgets and hand that to the
            # container. Built eagerly (hidden) so other code can rely on
            # advanced_container existing.
            self._advanced_controls_layout = self._build_advanced_layout()
            self.advanced_container = AdvancedMainWidget.build_container(
                self._advanced_controls_layout
            )
            self._advanced_content_container = self.advanced_container
        else:
            self.centralwidget.setLayout(self.gridLayout)

        # QLineEdit icons, trailing position
        self.blankIcon = QtGui.QIcon()
        self.savedIcon = QtGui.QIcon(
            os.path.join(Architecture.get_path(), "QATCH", "icons", "checkmark-circle.svg")
        )
        self.unsavedIcon = QtGui.QIcon(
            os.path.join(Architecture.get_path(), "QATCH", "icons", "warning.svg")
        )

        # --- Device Info container, widgets and layout ---
        self.device_info_container = QtWidgets.QWidget()
        self.device_info_container.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.device_info_container.setStyleSheet("background: transparent;")

        # --- Header: Back Button & Banner ---
        self.back_btn = QtWidgets.QPushButton()
        self.back_btn.setIcon(QtGui.QIcon("left-arrow.svg"))
        self.back_btn.setIconSize(QtCore.QSize(20, 20))
        self.back_btn.setFixedSize(36, 36)
        self.back_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.back_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 40);
                border: 1px solid rgba(255, 255, 255, 100);
                border-radius: 18px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 80);
                border: 1px solid rgba(255, 255, 255, 150);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 30);
            }
        """)
        # Assuming the back button performs the same action as closing the editor
        self.back_btn.clicked.connect(self.on_device_config_editor_close)

        self.ConfigBannerWidget = GlassWarningLabel("Configuration Editor for Device")

        # Create a header layout that perfectly centers the banner
        # by balancing the back button with a dummy widget on the right.
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        dummy_spacer = QtWidgets.QWidget()
        dummy_spacer.setFixedSize(36, 36)

        header_layout.addWidget(self.back_btn)
        header_layout.addStretch()
        header_layout.addWidget(self.ConfigBannerWidget)
        header_layout.addStretch()
        header_layout.addWidget(dummy_spacer)

        # --- Input validators ---
        self.validDeviceName = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r'[^\\/:*?"\'<>|]{1,12}')
        )
        self.validDevicePid = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"[0-9A-Fa-f]{1,2}")
        )
        self.validTempOffset = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(
                r"-?(?:[0-5](?:\.\d{0,2})?|6(?:\.(?:[0-2]\d?|3[0-5]?))?|6\.?|\.\d{1,2})"
            )
        )
        self.validPogoPosition = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"[0-5]?[0-9]|60")
        )
        self.validPogoDelayMs = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"[0-1]?[0-9]?[0-9]|2[0-4][0-9]|25[0-4]")
        )

        # Unified translucent style for text inputs
        glass_input_style = """
            QLineEdit {
                background: rgba(255, 255, 255, 60);
                border: 1px solid rgba(255, 255, 255, 120);
                border-radius: 6px;
                padding: 4px 8px;
                color: rgba(30, 40, 55, 220);
                min-width: 150px;
            }
            QLineEdit:focus {
                background: rgba(255, 255, 255, 180);
                border: 1px solid rgba(0, 120, 215, 150);
            }
        """

        # Row 0L: Device Name
        self.device_name_input = QtWidgets.QLineEdit()
        self.device_name_input.setValidator(self.validDeviceName)
        self.device_name_input.setStyleSheet(glass_input_style)
        self.device_name_action = self.device_name_input.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.device_name_input.textEdited.connect(
            lambda text, action=self.device_name_action: self.on_text_edit(text, action)
        )

        # Row 0R: Device Position ID
        self.device_pid_input = QtWidgets.QLineEdit()
        self.device_pid_input.setValidator(self.validDevicePid)
        self.device_pid_input.setStyleSheet(glass_input_style)
        self.device_pid_action = self.device_pid_input.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.device_pid_input.textEdited.connect(
            lambda text, action=self.device_pid_action: self.on_text_edit(text, action)
        )

        self.device_config_default = GlassPushButton("Default")
        self.device_config_default.clicked.connect(self.on_device_config_default)
        self.device_config_save = GlassPushButton("Save")
        self.device_config_save.clicked.connect(self.on_device_config_save)
        self.device_config_reset = GlassPushButton("Reset")
        self.device_config_reset.clicked.connect(self.on_device_config_reset)

        # Row 1L: Constant Temperature Calibration
        self.temp_cal_always_input = QtWidgets.QLineEdit()
        self.temp_cal_always_input.setValidator(self.validTempOffset)
        self.temp_cal_always_input.setStyleSheet(glass_input_style)
        self.temp_cal_always_action = self.temp_cal_always_input.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.temp_cal_always_input.textEdited.connect(
            lambda text, action=self.temp_cal_always_action: self.on_text_edit(text, action)
        )
        self.temp_cal_always_input.editingFinished.connect(
            lambda widget=self.temp_cal_always_input: self.on_edit_finish(widget)
        )

        # Row 1R: Running Temperature Calibration
        self.temp_cal_measure_input = QtWidgets.QLineEdit()
        self.temp_cal_measure_input.setValidator(self.validTempOffset)
        self.temp_cal_measure_input.setStyleSheet(glass_input_style)
        self.temp_cal_measure_action = self.temp_cal_measure_input.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.temp_cal_measure_input.textEdited.connect(
            lambda text, action=self.temp_cal_measure_action: self.on_text_edit(text, action)
        )
        self.temp_cal_measure_input.editingFinished.connect(
            lambda widget=self.temp_cal_measure_input: self.on_edit_finish(widget)
        )

        self.temp_cal_default = GlassPushButton("Default")
        self.temp_cal_default.clicked.connect(self.on_temp_cal_default)
        self.temp_cal_save = GlassPushButton("Save")
        self.temp_cal_save.clicked.connect(self.on_temp_cal_save)
        self.temp_cal_reset = GlassPushButton("Reset")
        self.temp_cal_reset.clicked.connect(self.on_temp_cal_reset)

        # Row 2L: Lid Pogo Distance
        self.lid_pogo_distance_input = QtWidgets.QLineEdit()
        self.lid_pogo_distance_input.setValidator(self.validPogoPosition)
        self.lid_pogo_distance_input.setStyleSheet(glass_input_style)
        self.lid_pogo_distance_action = self.lid_pogo_distance_input.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.lid_pogo_distance_input.textEdited.connect(
            lambda text, action=self.lid_pogo_distance_action: self.on_text_edit(text, action)
        )

        # Row 2R: Lid Pogo Delay
        self.lid_pogo_delay_input = QtWidgets.QLineEdit()
        self.lid_pogo_delay_input.setValidator(self.validPogoDelayMs)
        self.lid_pogo_delay_input.setStyleSheet(glass_input_style)
        self.lid_pogo_delay_action = self.lid_pogo_delay_input.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.lid_pogo_delay_input.textEdited.connect(
            lambda text, action=self.lid_pogo_delay_action: self.on_text_edit(text, action)
        )

        self.lid_pogo_default = GlassPushButton("Default")
        self.lid_pogo_default.clicked.connect(self.on_lid_pogo_default)
        self.lid_pogo_save = GlassPushButton("Save")
        self.lid_pogo_save.clicked.connect(self.on_lid_pogo_save)
        self.lid_pogo_reset = GlassPushButton("Reset")
        self.lid_pogo_reset.clicked.connect(self.on_lid_pogo_reset)

        # --- Custom Translucent Cards ---
        def create_glass_card(title_text):
            card = QtWidgets.QFrame()
            card.setStyleSheet("""
                QFrame {
                    background: rgba(255, 255, 255, 30);
                    border: 1px solid rgba(255, 255, 255, 90);
                    border-radius: 12px;
                }
                QLabel { 
                    border: none; 
                    background: transparent; 
                    color: rgba(40, 50, 65, 210);
                }
            """)
            layout = QtWidgets.QVBoxLayout(card)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)

            title = QtWidgets.QLabel(title_text.upper())
            title.setStyleSheet("font-weight: bold; font-size: 10pt; letter-spacing: 1px;")
            layout.addWidget(title)

            form_layout = QtWidgets.QGridLayout()
            form_layout.setContentsMargins(0, 0, 0, 0)
            form_layout.setSpacing(12)
            layout.addLayout(form_layout)

            btn_layout = QtWidgets.QHBoxLayout()
            btn_layout.setContentsMargins(0, 8, 0, 0)
            btn_layout.setSpacing(8)
            layout.addLayout(btn_layout)

            return card, form_layout, btn_layout

        # 1. Device Config Card
        self.device_config_card, dev_form, dev_btns = create_glass_card("Device Configuration")
        dev_form.addWidget(QtWidgets.QLabel("Device Name:"), 0, 0)
        dev_form.addWidget(self.device_name_input, 0, 1)
        dev_form.addWidget(QtWidgets.QLabel("Position ID:"), 1, 0)
        dev_form.addWidget(self.device_pid_input, 1, 1)
        dev_form.setColumnStretch(2, 1)  # Prevent inputs from expanding horizontally

        dev_btns.addWidget(self.device_config_default)
        dev_btns.addWidget(self.device_config_save)
        dev_btns.addWidget(self.device_config_reset)
        dev_btns.addStretch()

        # 2. Temp Cal Card
        self.temp_cal_card, temp_form, temp_btns = create_glass_card("Temperature Calibration")
        temp_form.addWidget(QtWidgets.QLabel("\u2206T_always:"), 0, 0)
        temp_form.addWidget(self.temp_cal_always_input, 0, 1)
        temp_form.addWidget(QtWidgets.QLabel("\u2206T_measure:"), 1, 0)
        temp_form.addWidget(self.temp_cal_measure_input, 1, 1)
        temp_form.setColumnStretch(2, 1)

        temp_btns.addWidget(self.temp_cal_default)
        temp_btns.addWidget(self.temp_cal_save)
        temp_btns.addWidget(self.temp_cal_reset)
        temp_btns.addStretch()

        # 3. Lid Pogo Card
        self.lid_pogo_cal_card, pogo_form, pogo_btns = create_glass_card("Lid POGO Calibration")
        pogo_form.addWidget(QtWidgets.QLabel("Servo Steps:"), 0, 0)
        pogo_form.addWidget(self.lid_pogo_distance_input, 0, 1)
        pogo_form.addWidget(QtWidgets.QLabel("Servo Delay:"), 1, 0)
        pogo_form.addWidget(self.lid_pogo_delay_input, 1, 1)
        pogo_form.setColumnStretch(2, 1)

        pogo_btns.addWidget(self.lid_pogo_default)
        pogo_btns.addWidget(self.lid_pogo_save)
        pogo_btns.addWidget(self.lid_pogo_reset)
        pogo_btns.addStretch()

        # --- Main Vertical Layout Assembly ---
        self.deviceLayout = QtWidgets.QVBoxLayout()
        self.deviceLayout.setSpacing(16)
        self.deviceLayout.addWidget(self.device_config_card)
        self.deviceLayout.addWidget(self.temp_cal_card)
        self.deviceLayout.addWidget(self.lid_pogo_cal_card)

        # Bottom Close Button (Optional, keeping it if you still want it alongside back)
        self.close_btn = GlassPushButton("Close Configuration Editor")
        self.close_btn.clicked.connect(self.on_device_config_editor_close)
        self.close_btn.setFixedHeight(28)

        # Final Banner Layout combining everything
        bannerLayout = QtWidgets.QVBoxLayout(self.device_info_container)
        bannerLayout.setContentsMargins(24, 24, 24, 24)
        bannerLayout.setSpacing(24)

        bannerLayout.addLayout(header_layout)  # Added Header here
        bannerLayout.addStretch()
        bannerLayout.addLayout(self.deviceLayout)  # Row layout goes here
        bannerLayout.addStretch()

        # Center the close button slightly distinct from the layout stretch
        close_btn_layout = QtWidgets.QHBoxLayout()
        close_btn_layout.addStretch()
        close_btn_layout.addWidget(self.close_btn)
        close_btn_layout.addStretch()
        bannerLayout.addLayout(close_btn_layout)

        # Hide it initially; the popup will show it when anchored
        self.device_info_container.hide()

        MainWindow1.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow1)

    def on_text_edit(self, text, action):
        if len(text):
            action.setIcon(self.unsavedIcon)
            action.setIconText("unsaved")
        else:
            action.setIcon(self.blankIcon)
            action.setIconText("blank")

    def on_edit_finish(self, widget):
        try:
            text = widget.text()
            if len(text):
                rounded_05 = round(round(float(text) * 20) / 20, 2)
                widget.setText(f"{rounded_05:2.02f}")
        except (ValueError, TypeError):
            Log.e("Invalid input, resetting field to blank.")
            widget.setText("")

    def on_device_config_default(self):
        default_name = self.ConfigBannerWidget.text().split()[-1]
        default_pid = "FF"

        if self.device_name_input.text() != default_name:
            self.device_name_input.setText(default_name)
            self.device_name_action.setIcon(self.unsavedIcon)
            self.device_name_action.setIconText("unsaved")
        if self.device_pid_input.text() != default_pid:
            self.device_pid_input.setText(default_pid)
            self.device_pid_action.setIcon(self.unsavedIcon)
            self.device_pid_action.setIconText("unsaved")

    def on_device_config_save(self):
        ok_name = False
        ok_pid = False
        if self.device_name_action.iconText() == "unsaved":
            if self.device_name_input.hasAcceptableInput():
                text = self.device_name_input.text()
                Log.d("Save device name =", text)
                ok_name = self.save_device_name_input(text)
                if ok_name:
                    self.device_name_action.setIcon(self.savedIcon)
                    self.device_name_action.setIconText("saved")
            else:
                Log.e(
                    f"Invalid 'Device Name' input: {self.device_name_input.text()} (out of valid range)"
                )
        if self.device_pid_action.iconText() == "unsaved":
            if self.device_pid_input.hasAcceptableInput():
                text = self.device_pid_input.text()
                Log.d("Save device pid =", text)
                ok_pid, dif = self.save_device_pid_input(text)
                if ok_pid:
                    self.device_pid_action.setIcon(self.savedIcon)
                    self.device_pid_action.setIconText("saved")
            else:
                Log.e(
                    f"Invalid 'Position ID' input: {self.device_pid_input.text()} (out of valid range)"
                )

        mainWindow = self.parent.parent
        if ok_pid:
            if dif != None:
                try:
                    os.remove(dif)
                except:
                    Log.e("Failed to delete file:", dif)
            # force parse and/or write device info (to update name and/or pid)
            # mainWindow.fwUpdater.checkAgain()
            # mainWindow.worker._port = mainWindow._selected_port  # used in run()
            # mainWindow.fwUpdater.run(mainWindow)
            QtCore.QTimer.singleShot(
                1000, lambda: not mainWindow._identifying and mainWindow._port_identify()
            )
            QtCore.QTimer.singleShot(
                4000, lambda: mainWindow._identifying and mainWindow._port_identify()
            )
        elif ok_name:  # (needed only if PID not changed too)
            mainWindow._refresh_ports()  # update name in port list
        # elif ok_cal: do nothing

    def on_device_config_reset(self):
        if self.device_name_action.iconText() != "saved":
            Log.d("Reset device name")
            self.device_name_input.clear()
            # self.device_name_input.setEnabled(False)
            self.device_name_input.setPlaceholderText("Querying...")
            self.device_name_action.setIcon(self.blankIcon)
            self.device_name_action.setIconText("blank")
            QtCore.QTimer.singleShot(500, self.reset_device_name_input)
        if self.device_pid_action.iconText() != "saved":
            Log.d("Reset device pid")
            self.device_pid_input.clear()
            # self.device_pid_input.setEnabled(False)
            self.device_pid_input.setPlaceholderText("Querying...")
            self.device_pid_action.setIcon(self.blankIcon)
            self.device_pid_action.setIconText("blank")
            QtCore.QTimer.singleShot(1000, self.reset_device_pid_input)

    def on_temp_cal_default(self):
        default_always = "0.00"
        default_measure = "0.00"

        if self.temp_cal_always_input.text() != default_always:
            self.temp_cal_always_input.setText(default_always)
            self.temp_cal_always_action.setIcon(self.unsavedIcon)
            self.temp_cal_always_action.setIconText("unsaved")
        if self.temp_cal_measure_input.text() != default_measure:
            self.temp_cal_measure_input.setText(default_measure)
            self.temp_cal_measure_action.setIcon(self.unsavedIcon)
            self.temp_cal_measure_action.setIconText("unsaved")

    def on_temp_cal_save(self):
        if self.temp_cal_always_action.iconText() == "unsaved":
            if self.temp_cal_always_input.hasAcceptableInput():
                text = self.temp_cal_always_input.text()
                Log.d("Save T_always =", text)
                if self.save_temp_cal_always_input(text):
                    self.temp_cal_always_action.setIcon(self.savedIcon)
                    self.temp_cal_always_action.setIconText("saved")
            else:
                Log.e(
                    f"Invalid T_always input: {self.temp_cal_always_input.text()} (out of valid range)"
                )
        if self.temp_cal_measure_action.iconText() == "unsaved":
            if self.temp_cal_measure_input.hasAcceptableInput():
                text = self.temp_cal_measure_input.text()
                Log.d("Save T_measure =", text)
                if self.save_temp_cal_measure_input(text):
                    self.temp_cal_measure_action.setIcon(self.savedIcon)
                    self.temp_cal_measure_action.setIconText("saved")
            else:
                Log.e(
                    f"Invalid T_measure input: {self.temp_cal_measure_input.text()} (out of valid range)"
                )

    def on_temp_cal_reset(self):
        if self.temp_cal_always_action.iconText() != "saved":
            Log.d("Reset T_always")
            self.temp_cal_always_input.clear()
            # self.temp_cal_always_input.setEnabled(False)
            self.temp_cal_always_input.setPlaceholderText("Querying...")
            self.temp_cal_always_action.setIcon(self.blankIcon)
            self.temp_cal_always_action.setIconText("blank")
            QtCore.QTimer.singleShot(1, self.reset_temp_cal_always_input)
        if self.temp_cal_measure_action.iconText() != "saved":
            Log.d("Reset T_measure")
            self.temp_cal_measure_input.clear()
            # self.temp_cal_measure_input.setEnabled(False)
            self.temp_cal_measure_input.setPlaceholderText("Querying...")
            self.temp_cal_measure_action.setIcon(self.blankIcon)
            self.temp_cal_measure_action.setIconText("blank")
            QtCore.QTimer.singleShot(3000, self.reset_temp_cal_measure_input)

    def on_lid_pogo_default(self):
        default_distance = "30"
        default_delay = "30"

        if self.lid_pogo_distance_input.text() != default_distance:
            self.lid_pogo_distance_input.setText(default_distance)
            self.lid_pogo_distance_action.setIcon(self.unsavedIcon)
            self.lid_pogo_distance_action.setIconText("unsaved")
        if self.lid_pogo_delay_input.text() != default_delay:
            self.lid_pogo_delay_input.setText(default_delay)
            self.lid_pogo_delay_action.setIcon(self.unsavedIcon)
            self.lid_pogo_delay_action.setIconText("unsaved")

    def on_lid_pogo_save(self):
        send_lid_cal_cmd = False
        form_error = False  # blocking if True
        if self.lid_pogo_distance_action.iconText() == "unsaved":
            if self.lid_pogo_distance_input.hasAcceptableInput():
                text = self.lid_pogo_distance_input.text()
                Log.d("Save lid pogo distance =", text)
                self.lid_pogo_distance_action.setIcon(self.savedIcon)
                self.lid_pogo_distance_action.setIconText("saved")
                send_lid_cal_cmd = True
            else:
                Log.e(
                    f"Invalid 'Servo Steps' input: {self.lid_pogo_distance_input.text()} (out of valid range)"
                )
                form_error = True
        if self.lid_pogo_delay_action.iconText() == "unsaved":
            if self.lid_pogo_delay_input.hasAcceptableInput():
                text = self.lid_pogo_delay_input.text()
                Log.d("Save lid pogo delay =", text)
                self.lid_pogo_delay_action.setIcon(self.savedIcon)
                self.lid_pogo_delay_action.setIconText("saved")
                send_lid_cal_cmd = True
            else:
                Log.e(
                    f"Invalid 'Servo Delay' input: {self.lid_pogo_delay_input.text()} (out of valid range)"
                )
                form_error = True
        if send_lid_cal_cmd and not form_error:
            self.save_lid_pogo_calibration()

    def on_lid_pogo_reset(self):
        get_lid_cal = False
        if self.lid_pogo_distance_action.iconText() != "saved":
            Log.d("Reset lid pogo distance")
            self.lid_pogo_distance_input.clear()
            # self.lid_pogo_distance_input.setEnabled(False)
            self.lid_pogo_distance_input.setPlaceholderText("Querying...")
            self.lid_pogo_distance_action.setIcon(self.blankIcon)
            self.lid_pogo_distance_action.setIconText("blank")
            get_lid_cal = True
        if self.lid_pogo_delay_action.iconText() != "saved":
            Log.d("Reset lid pogo delay")
            self.lid_pogo_delay_input.clear()
            # self.lid_pogo_delay_input.setEnabled(False)
            self.lid_pogo_delay_input.setPlaceholderText("Querying...")
            self.lid_pogo_delay_action.setIcon(self.blankIcon)
            self.lid_pogo_delay_action.setIconText("blank")
            get_lid_cal = True
        if get_lid_cal:
            QtCore.QTimer.singleShot(1500, self.get_lid_pogo_calibration)

    def save_device_name_input(self, text):
        mainWindow = self.parent.parent
        dev_handle = None
        device_list = FileStorage.DEV_get_device_list()
        for i, dev_name in device_list:
            dev_info = FileStorage.DEV_info_get(i, dev_name)
            if "NAME" in dev_info and "PORT" in dev_info:
                if dev_info["PORT"] == mainWindow._selected_port:
                    dev_handle = dev_name
                    break
        # remove any invalid characters from user input
        # invalidChars = "\\/:*?\"'<>|"
        for invalidChar in Constants.invalidChars:
            text = text.replace(invalidChar, "")
        text = text.strip().replace(" ", "_")  # word spaces -> underscores
        # limit length of input
        text = text[:12] if len(text) > 12 else text
        text = text.upper()  # make user input uppercase
        try:
            if text == "":
                text = dev_info["USB"]
            Log.i("Set on device '{}': NAME = {}".format(dev_handle, text))
            if text != self.device_name_input.text():
                self.device_name_input.setText(text)
            dev_file = os.path.join(
                Constants.csv_calibration_export_path,
                dev_handle,
                "{}.{}".format(Constants.txt_device_info_filename, Constants.txt_extension),
            )
            dev_lines = []
            with open(dev_file, "r") as file:
                dev_lines = file.readlines()
                dev_lines[0] = "NAME: {}\n".format(text)
            with open(dev_file, "w") as file:
                file.writelines(dev_lines)
            Log.i("Program 'Name' operation was successful!")
            return True
        except:
            Log.e("Failed to update name entered by user.")
            return False

    def reset_device_name_input(self):
        mainWindow = self.parent.parent

        friendly_name = mainWindow._selected_port
        # dev_handle = None
        device_list = FileStorage.DEV_get_device_list()
        for i, dev_name in device_list:
            dev_info = FileStorage.DEV_info_get(i, dev_name)
            if "NAME" in dev_info and "PORT" in dev_info:
                if dev_info["PORT"] == mainWindow._selected_port:
                    friendly_name = dev_info["NAME"]
                    # dev_handle = dev_name
                    break

        self.device_name_input.setText(friendly_name)
        self.device_name_input.setPlaceholderText(None)
        self.device_name_action.setIcon(self.savedIcon)
        self.device_name_action.setIconText("saved")

    def save_device_pid_input(self, text):
        mainWindow = self.parent.parent
        dev_handle = None
        pid_old = 0xFF  # default: unassigned
        device_list = FileStorage.DEV_get_device_list()
        for i, dev_name in device_list:
            dev_info = FileStorage.DEV_info_get(i, dev_name)
            if "NAME" in dev_info and "PORT" in dev_info:
                if dev_info["PORT"] == mainWindow._selected_port:
                    dev_handle = dev_name
                    if "PID" in dev_info:
                        pid_old = int(dev_info["PID"], base=16)
                    break
        try:
            pid_new = int(text, base=16)
            # valid values: 1-4, A-D, 0x00 (single),
            #               0x80 (flux controller), 0xFF (default, single)
            if pid_new not in [
                0x1,
                0x2,
                0x3,
                0x4,
                0xA,
                0xB,
                0xC,
                0xD,
                0x00,
                0x80,
                0xFF,
            ]:
                Log.w("Out-of-range PID entered by user. Using default: 0xFF")
                pid_new = 0xFF
        except:
            Log.w("Non-numeric PID entered by user. Using default: 0xFF")
            pid_new = 0xFF
        Log.i("Set on device '{}': PID = {}".format(dev_handle, pid_new))
        pid_str = hex(pid_new)[2:].upper()
        if pid_str != self.device_pid_input.text():
            self.device_pid_input.setText(pid_str)
        if pid_new != pid_old:
            if mainWindow.setEEPROM(mainWindow._selected_port, 0, pid_new):
                Log.i("Device EEPROM write PID success!")
            Log.i("Program 'Position ID' operation was successful!")
            ok = True
        else:
            Log.w("Program 'Position ID' operation resulted in no change!")
            ok = False

        if ok:  # pid changed
            try:
                # Configure serial port (assume baud to check before update)
                _serial = serial.Serial()
                _serial.port = mainWindow._selected_port
                _serial.baudrate = Constants.serial_default_speed  # 115200
                _serial.stopbits = serial.STOPBITS_ONE
                _serial.bytesize = serial.EIGHTBITS
                _serial.timeout = Constants.serial_timeout_ms
                _serial.write_timeout = Constants.serial_writetimeout_ms
                _serial.open()
                _serial.write(b"MULTI INIT 0\n")
                _serial.close()
            except:
                Log.e("Unable to refresh LCD. PID error may be stale.")
            try:
                dev_name = dev_handle
                i_old = 0 if pid_old == 0xFF else pid_old
                dev_folder_old = "{}_{}".format(i_old, dev_name) if i_old > 0 else dev_name
                dev_info_file_old = os.path.join(
                    Constants.csv_calibration_export_path,
                    dev_folder_old,
                    f"{Constants.txt_device_info_filename}.txt",
                )
                if os.path.exists(dev_info_file_old):
                    Log.d(
                        f"Queueing removal of stale DEV_INFO file for {dev_name} with PID {pid_new}..."
                    )
                    return ok, dev_info_file_old
            except:
                Log.e("Unable to check for stale DEV_INFO file removal.")
        return ok, None

    def reset_device_pid_input(self):
        mainWindow = self.parent.parent

        # friendly_name = mainWindow._selected_port
        # dev_handle = None
        pid_old = 0xFF  # default: unassigned
        device_list = FileStorage.DEV_get_device_list()
        for i, dev_name in device_list:
            dev_info = FileStorage.DEV_info_get(i, dev_name)
            if "NAME" in dev_info and "PORT" in dev_info:
                if dev_info["PORT"] == mainWindow._selected_port:
                    # friendly_name = dev_info["NAME"]
                    # dev_handle = dev_name
                    if "PID" in dev_info:
                        pid_old = int(dev_info["PID"], base=16)
                    break

        # confirm PID in DEV_INFO matches COM Port listed text
        try:
            idx = mainWindow.ControlsWin.ui1.cBox_Port.findData(mainWindow._selected_port)
            if idx >= 0:
                device_text = mainWindow.ControlsWin.ui1.cBox_Port.itemText(idx)
                if ":" in device_text:
                    dev_i = int(device_text.split(":")[0], base=16)
                    if dev_i != pid_old:
                        Log.e(
                            f"Conflicting device info, using PID as {dev_i} instead of reported {pid_old}!"
                        )
                        pid_old = int(dev_i, base=16)
        except:
            Log.e("ERROR: Unable to check if PID in COM Port list matches DEV_INFO.")

        pid_str = hex(pid_old)[2:].upper()

        self.device_pid_input.setText(pid_str)
        self.device_pid_input.setPlaceholderText(None)
        self.device_pid_action.setIcon(self.savedIcon)
        self.device_pid_action.setIconText("saved")

    def save_temp_cal_always_input(self, text):
        mainWindow = self.parent.parent
        dev_handle = None
        device_list = FileStorage.DEV_get_device_list()
        for i, dev_name in device_list:
            dev_info = FileStorage.DEV_info_get(i, dev_name)
            if "NAME" in dev_info and "PORT" in dev_info:
                if dev_info["PORT"] == mainWindow._selected_port:
                    dev_handle = dev_name
                    break
        # Save CAL1 to EEPROM
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
        # if cal_new == 0xFF:
        #     set_cal1 = "0"
        # else:
        #     set_cal1 = text
        Log.i("Set on device '{}': CAL1 = {} ({}C)".format(dev_handle, cal_new, text))
        if mainWindow.setEEPROM(mainWindow._selected_port, 1, cal_new):
            Log.i("Device EEPROM write CAL1 success!")
            success = True
        else:
            Log.e("Failed to write EEPROM address for CAL1.")
            success = False
        if success:
            Log.i("Program 'TEMP CAL1' operation was successful!")
        else:
            Log.e("Program 'TEMP CAL1' operation was NOT successful!")
        return success

    def reset_temp_cal_always_input(self, skip_delay=False):
        mainWindow = self.parent.parent
        start_time = monotonic()

        tec_update_required = True
        if mainWindow.tecWorker.last_reply():
            if start_time - mainWindow.tecWorker.last_reply() < 10:
                tec_update_required = False

        if tec_update_required:
            Log.i("Updating TEC parameters...")
            mainWindow.tecWorker.set_port(mainWindow._selected_port)
            mainWindow.tecWorker._tec_update()  # Force read to update SW cached offsets

        if tec_update_required and not skip_delay:
            # Wait up to 2 seconds, less TEC query processing delay
            wait_msecs = int(1000 * (2 - (monotonic() - start_time)))
            if wait_msecs > 0:
                QtCore.QTimer.singleShot(
                    wait_msecs,
                    lambda: self.reset_temp_cal_always_input(skip_delay=True),
                )
                return

        set_cal1 = mainWindow.tecWorker._tec_offset1

        self.temp_cal_always_input.setText(set_cal1)
        self.temp_cal_always_input.setPlaceholderText(None)
        self.temp_cal_always_action.setIcon(self.savedIcon)
        self.temp_cal_always_action.setIconText("saved")

    def save_temp_cal_measure_input(self, text):
        mainWindow = self.parent.parent
        dev_handle = None
        device_list = FileStorage.DEV_get_device_list()
        for i, dev_name in device_list:
            dev_info = FileStorage.DEV_info_get(i, dev_name)
            if "NAME" in dev_info and "PORT" in dev_info:
                if dev_info["PORT"] == mainWindow._selected_port:
                    dev_handle = dev_name
                    break
        # Save CAL2 to EEPROM
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
        # if cal_new == 0xFF:
        #     set_cal2 = "0"
        # else:
        #     set_cal2 = text
        Log.i("Set on device '{}': CAL2 = {} ({}C)".format(dev_handle, cal_new, text))
        if mainWindow.setEEPROM(mainWindow._selected_port, 3, cal_new):
            Log.i("Device EEPROM write CAL2 success!")
            success = True
        else:
            Log.e("Failed to write EEPROM address for CAL2")
            success = False
        if success:
            Log.i("Program 'TEMP CAL2' operation was successful!")
        else:
            Log.e("Program 'TEMP CAL2' operation was NOT successful!")
        return success

    def reset_temp_cal_measure_input(self):
        mainWindow = self.parent.parent
        start_time = monotonic()

        tec_update_required = True
        if mainWindow.tecWorker.last_reply():
            if start_time - mainWindow.tecWorker.last_reply() < 10:
                tec_update_required = False

        if tec_update_required:
            Log.i("Updating TEC parameters...")
            mainWindow.tecWorker.set_port(mainWindow._selected_port)
            mainWindow.tecWorker._tec_update()  # Force read to update SW cached offsets

        # No additional delay required on cal2
        # sleep(0)

        set_cal2 = mainWindow.tecWorker._tec_offset2

        self.temp_cal_measure_input.setText(set_cal2)
        self.temp_cal_measure_input.setPlaceholderText(None)
        self.temp_cal_measure_action.setIcon(self.savedIcon)
        self.temp_cal_measure_action.setIconText("saved")

    def save_lid_pogo_calibration(self):
        mainWindow = self.parent.parent
        cal_start = 100
        cal_stop = cal_start + int(self.lid_pogo_distance_input.text())
        cal_delay = int(self.lid_pogo_delay_input.text())
        response = None
        try:
            # Configure serial port (assume baud to check before update)
            LIDCAL_serial = serial.Serial()
            LIDCAL_serial.port = mainWindow._selected_port
            LIDCAL_serial.baudrate = Constants.serial_default_speed  # 115200
            LIDCAL_serial.stopbits = serial.STOPBITS_ONE
            LIDCAL_serial.bytesize = serial.EIGHTBITS
            LIDCAL_serial.timeout = Constants.serial_timeout_ms
            LIDCAL_serial.write_timeout = Constants.serial_writetimeout_ms
            LIDCAL_serial.open()
            LIDCAL_serial.write(f"LID CAL {cal_start},{cal_stop},{cal_delay}\n".encode())
            LIDCAL_serial.close()
        except:
            Log.e("Unable to get LID CAL. No reply from device.")

    def get_lid_pogo_calibration(self):
        mainWindow = self.parent.parent
        pogo_distance = 30
        pogo_delay = 30
        response = None

        try:
            # Configure serial port (assume baud to check before update)
            LIDCAL_serial = serial.Serial()
            LIDCAL_serial.port = mainWindow._selected_port
            LIDCAL_serial.baudrate = Constants.serial_default_speed  # 115200
            LIDCAL_serial.stopbits = serial.STOPBITS_ONE
            LIDCAL_serial.bytesize = serial.EIGHTBITS
            LIDCAL_serial.timeout = Constants.serial_timeout_ms
            LIDCAL_serial.write_timeout = Constants.serial_writetimeout_ms
            LIDCAL_serial.open()
            LIDCAL_serial.write(b"LID CAL\n")
            response = LIDCAL_serial.read_until()
            LIDCAL_serial.close()
        except:
            Log.e("Unable to get LID CAL. No reply from device.")

        try:
            if response:
                lid_cal_split = response.decode().strip().split()[-1]
                lid_cal_params = lid_cal_split.split(",")
                pogo_distance = abs(int(lid_cal_params[1]) - int(lid_cal_params[0]))
                pogo_delay = int(lid_cal_params[-1])
        except:
            Log.e("Unable to parse LID CAL reply.")

        if self.lid_pogo_distance_action.iconText() == "blank":
            self.lid_pogo_distance_input.setText(str(pogo_distance))
            self.lid_pogo_distance_input.setPlaceholderText(None)
            self.lid_pogo_distance_action.setIcon(self.savedIcon)
            self.lid_pogo_distance_action.setIconText("saved")
        if self.lid_pogo_delay_action.iconText() == "blank":
            self.lid_pogo_delay_input.setText(str(pogo_delay))
            self.lid_pogo_delay_input.setPlaceholderText(None)
            self.lid_pogo_delay_action.setIcon(self.savedIcon)
            self.lid_pogo_delay_action.setIconText("saved")

    def blank_device_config_icon_text(self):
        self.device_name_action.setIconText("blank")
        self.device_pid_action.setIconText("blank")
        self.temp_cal_always_action.setIconText("blank")
        self.temp_cal_measure_action.setIconText("blank")
        self.lid_pogo_distance_action.setIconText("blank")
        self.lid_pogo_delay_action.setIconText("blank")

    def on_device_config_editor_close(self):
        actions = [
            self.device_name_action,
            self.device_pid_action,
            self.temp_cal_always_action,
            self.temp_cal_measure_action,
            self.lid_pogo_distance_action,
            self.lid_pogo_delay_action,
        ]
        unsaved_input = False
        for action in actions:
            if action.iconText() == "unsaved":
                Log.w(
                    "You have unsaved device configuration input. Please Save or Reset unsaved input before closing."
                )
                unsaved_input = True
                break
        if not unsaved_input:
            self.device_info_container.parent().parent().hide()

    def _update_progress_text(self):
        # get innerText from HTML in infobar
        plain_text = self.infobar.text()
        color = plain_text[plain_text.rindex("color=") + 6 : plain_text.rindex("color=") + 6 + 7]
        plain_text = plain_text[plain_text.index(">") + 1 :]
        plain_text = plain_text[plain_text.index(">") + 1 :]
        plain_text = plain_text[plain_text.index(">") + 1 :]
        plain_text = plain_text[0 : plain_text.rindex("<")]
        # remove any formatting tags: <b>, <i>, <u>
        while plain_text.rfind("<") != plain_text.find("<"):
            plain_text = plain_text[0 : plain_text.rindex("<")]
            plain_text = plain_text[plain_text.index(">") + 1 :]
        if len(plain_text) == 0:
            plain_text = "Progress: Not Started"
        else:
            plain_text = "Status: {}".format(plain_text)
        self.run_progress_bar.setFormat(plain_text)
        styleBar = _GLASS_PROGRESSBAR_QSS.replace(
            "color: #1a3050;", "color: {COLOR}; font-weight: bold;"
        ).replace("{COLOR}", color)
        self.run_progress_bar.setStyleSheet(styleBar)

    def _update_progress_value(self):
        if self.cBox_Source.currentIndex() == OperationType.measurement.value:
            pass
        else:
            self.run_progress_bar.setFormat("Progress: %p%")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(
            _translate(
                "MainWindow",
                "{} {} - Setup/Control".format(Constants.app_title, Constants.app_version),
            )
        )
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons")
        MainWindow.setWindowIcon(QtGui.QIcon(os.path.join(icon_path, "qatch-icon.png")))
        # NOTE: the advanced container is now created/owned by AdvancedMainWidget
        # when the popup opens; it is frameless and embedded, so the former
        # window icon/title on it never displayed and have been removed.
        self.pButton_Stop.setText(_translate("MainWindow", " STOP"))
        self.pButton_Start.setText(_translate("MainWindow", "START"))
        self.pButton_Clear.setText(_translate("MainWindow", "Clear Plots"))
        # pButton_Reference is now the Absolute/Reference GlassToggle (no text).
        self.pButton_ResetApp.setText(_translate("MainWindow", "Factory Defaults"))
        self.sBox_Samples.setSuffix(_translate("MainWindow", " / 5 min"))
        self.sBox_Samples.setPrefix(_translate("MainWindow", ""))
        self.chBox_export.setText(_translate("MainWindow", "Txt Export Sweep File"))
        self.chBox_freqHop.setText(_translate("MainWindow", "Mode Hop"))
        self.chBox_correctNoise.setText(_translate("MainWindow", "Show amplitude curve"))
        self.chBox_MultiAuto.setText(_translate("MainWindow", "Auto-detect channel count"))

    def action_next_port(self):
        """Method to handle advancing to the next port."""
        try:
            self.action_NextPortRow.setEnabled(False)

            controller_port = None
            for i in range(self.cBox_Port.count()):
                if self.cBox_Port.itemText(i).startswith("80:"):
                    controller_port = self.cBox_Port.itemData(i)
                    break

            if controller_port is None:
                Log.e("FLUX controller not found. Is it connected and powered on?")
                self.tool_NextPortRow.setIconError()
                self.action_NextPortRow.setEnabled(True)
                return

            next_port_num = self.tool_NextPortRow.value()

            if hasattr(self, "fluxThread"):
                if self.fluxThread.isRunning():
                    Log.d("Waiting for FLUX controller to stop.")
                    if not self.fluxThread.wait(msecs=3000):
                        Log.w(
                            "Prior Flux controller thread still busy; skipping new Next Port request."
                        )
                        self.tool_NextPortRow.setEnabled(True)
                        return
            Log.d("Starting FLUX controller thread.")
            self.fluxThread = QtCore.QThread()
            self.fluxWorker = FLUXControl()
            self.fluxWorker.set_ports(controller=controller_port, next_port=next_port_num)
            self.fluxWorker.moveToThread(self.fluxThread)
            self.fluxThread.worker = self.fluxWorker
            self.fluxThread.started.connect(self.fluxWorker.run)
            self.fluxWorker.finished.connect(self.fluxThread.quit)
            self.fluxWorker.result.connect(self.next_port_result)
            self.fluxThread.start()

        except Exception as e:
            Log.e(f"action_next_port ERROR: {e}")
            self.tool_NextPortRow.setIconError()
            self.action_NextPortRow.setEnabled(True)

    def next_port_result(self, success):
        try:
            self.action_NextPortRow.setEnabled(True)

            if success:
                self.parent.parent.active_multi_ch = self.tool_NextPortRow.value()
                self.parent.parent.set_multi_mode()
            else:
                self.tool_NextPortRow.setIconError()

                if PopUp.critical(
                    self,
                    "Next Port Failed",
                    "ERROR: Flux controller failed to move to the next port.",
                    btn1_text="Reset",
                ):
                    self.tool_NextPortRow.click()

        except Exception as e:
            Log.e(f"next_port_result ERROR: {e}")

    def action_initialize(self):
        """Method to handle initialization UI actions."""
        if self.pButton_Start.isEnabled():
            self.cBox_Source.setCurrentIndex(OperationType.calibration.value)
            if hasattr(self, "run_controls"):
                self.run_controls.set_running(False)
                self.run_controls.update_progress(0, 5, "Ready")
                self.run_controls.setEnabled(False)
            self.pButton_Start.clicked.emit()
            self.cal_initialized = True

    def action_start(self):
        """Method to handle start UI actions."""
        if self.pButton_Start.isEnabled():
            self.cBox_Source.setCurrentIndex(OperationType.measurement.value)
            self.pButton_Start.clicked.emit()

    def action_stop(self):
        """Method to handle stop UI actions."""
        if self.pButton_Stop.isEnabled():
            self.cal_initialized = False
            self.pButton_Stop.clicked.emit()
            num_devices = getattr(self, "multiplex_plots", 1)
            for i in range(num_devices):
                self.parent.parent.PlotsWin.ui2.left_pane.set_device_state(i, "idle")

    def action_reset(self):
        """Method to handle reset UI actions."""
        if self.tool_TempControl.isChecked():
            self.tool_TempControl.setChecked(False)
            self.tool_TempControl.clicked.emit()  # triggers action_tempcontrol → collapse
        self.slTemp.setValue(25)
        if self.pButton_Start.isEnabled():
            self.pButton_Clear.clicked.emit()
            self.pButton_Refresh.clicked.emit()
        self.infostatus.setText("Program Status Standby")

        self.cal_initialized = False
        if hasattr(self, "run_controls"):
            self.run_controls.set_running(False)
            self.run_controls.update_progress(0, 5, "Idle")
            self.run_controls.setEnabled(False)

        self.tool_TempControl.setEnabled(self.cBox_Port.count() > 0)
        num_devices = getattr(self, "multiplex_plots", 1)
        for i in range(num_devices):
            self.parent.parent.PlotsWin.ui2.left_pane.set_device_state(i, "idle")

    def _animate_temp_controller(self, expand: bool) -> None:
        """Smoothly expand or collapse the temperature controller panel."""
        _EXPANDED_W = 280
        start = self.tempController.maximumWidth()
        if start > _EXPANDED_W:
            start = _EXPANDED_W
        end = _EXPANDED_W if expand else 0

        if start == end:
            return

        self._set_temp_arrow(expand)

        self._temp_anim = QtCore.QPropertyAnimation(self.tempController, b"maximumWidth")
        self._temp_anim.setDuration(220)
        self._temp_anim.setStartValue(start)
        self._temp_anim.setEndValue(end)
        self._temp_anim.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self._temp_anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)

    def _toggle_temp_controller(self) -> None:
        """Programmatically toggle the panel via the toolbar button.

        Kept for backward compatibility — external callers (e.g. shortcuts) can
        still use this to flip the temperature controller open/closed.
        """
        self.tool_TempControl.setChecked(not self.tool_TempControl.isChecked())
        self.action_tempcontrol()

    def _set_temp_arrow(self, expand: bool) -> None:
        """Update the toolbar button's chevron to indicate panel state.

        Per the wireframe, the chevron lives on the toolbar button itself —
        ``Temp Control ›`` when collapsed (clicking expands), ``Temp Control ‹``
        when expanded (clicking collapses).  No in-panel arrow strip is used.
        """
        if hasattr(self, "tool_TempControl"):
            self.tool_TempControl.setText("Temp Control ‹" if expand else "Temp Control ›")

    def _update_temp_display(self, text: str) -> None:
        """Parse the combined temp string and update the split display + status bar.

        Expected format: ``"PV:25.03C SP:25.00C OP:[50%]"``
        Falls back gracefully when values are placeholder dashes.
        """
        parts: dict = {}
        for segment in text.split():
            if ":" in segment:
                key, val = segment.split(":", 1)
                parts[key] = val

        pv_str = parts.get("PV", "--.--C")
        sp_str = parts.get("SP", "--.--C")
        op_str = parts.get("OP", "----")

        self.lPV.setText(f"PV  {pv_str}")
        self.lSP.setText(f"SP  {sp_str}")
        self.lOP.setText(f"OP  {op_str}")

        # Determine status colour and descriptive label
        try:
            pv = float(pv_str.rstrip("C"))
            sp = float(sp_str.rstrip("C"))
            if abs(pv - sp) <= 0.5:
                status_text = "Ready"
                bg_colour = "rgba(60, 200, 90, 220)"
                text_colour = "rgba(255, 255, 255, 230)"
            elif pv < sp:
                status_text = "Heating to setpoint..."
                bg_colour = "rgba(240, 190, 0, 220)"
                text_colour = "rgba(30, 20, 0, 200)"
            else:
                status_text = "Cooling to setpoint..."
                bg_colour = "rgba(240, 140, 0, 220)"
                text_colour = "rgba(30, 20, 0, 200)"
        except ValueError:
            status_text = "Offline"
            bg_colour = "rgba(150, 155, 160, 120)"
            text_colour = "rgba(30, 40, 55, 160)"

        self.tempStatusBar.setText(status_text)
        self.tempStatusBar.setStyleSheet(
            f"QLabel {{ background: {bg_colour}; color: {text_colour}; "
            "border: 1px solid rgba(255, 255, 255, 160); "
            "border-radius: 3px; padding: 0 6px; font-weight: bold; }}"
        )

    def action_tempcontrol(self):
        is_checked = self.tool_TempControl.isChecked()
        self._animate_temp_controller(is_checked)
        if is_checked:
            if self.pTemp.text().find("Stop") < 0:
                self.pTemp.clicked.emit()
            # Focus the slider after the panel finishes expanding
            QtCore.QTimer.singleShot(230, self.slTemp.setFocus)
        else:
            if self.pTemp.text().find("Stop") >= 0:
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
            Log.w("WARNING: Temp Control mode cannot be changed during an active run.")
            if self.event_windowPos.x() >= self.tempController.mapToGlobal(QtCore.QPoint(0, 0)).x():
                Log.w(
                    'To adjust Temp Control: Press "Stop" first, then adjust setpoint accordingly.'
                )
            else:
                Log.w('To stop Temp Control: Press "Stop" first, then click "Temp Control" button.')

    def _build_advanced_layout(self) -> QtWidgets.QLayout:
        """Assemble the advanced-panel widgets into a clean, sectioned layout.

        Mirrors the account dropdown's visual language: soft muted section
        headers (not blue pills), hairline dividers, and generous, consistent
        spacing. Reuses the existing widget instances so all signal wiring and
        external references remain valid; only their parent layout changes.
        """

        def section(title, *rows, stretch_last=False):
            """A titled vertical group: header, hairline, then content rows."""
            col = QtWidgets.QVBoxLayout()
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(6)
            col.addWidget(_SectionHeader(title))
            col.addWidget(_hairline())
            for row in rows:
                if isinstance(row, QtWidgets.QLayout):
                    col.addLayout(row)
                else:
                    col.addWidget(row)
            if not stretch_last:
                col.addStretch()
            return col

        def hrow(*widgets, spacing=6):
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(spacing)
            for w in widgets:
                if isinstance(w, QtWidgets.QLayout):
                    row.addLayout(w)
                else:
                    row.addWidget(w)
            return row

        # Detach the cartridge toggle from its old QGroupBox so we can restyle.
        self.grpMode.setParent(None)

        # ---- Left column: connection + signal settings ----
        op_section = section("Operation Mode", self.cBox_Source)

        # Port row: shrunken dropdown + inline ID / Refresh / Configure buttons.
        self.cBox_Port.setMinimumWidth(0)
        self.cBox_Port.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        port_row = hrow(
            self.cBox_Port,
            self.pButton_ID,
            self.pButton_Refresh,
            self.pButton_Configure,
        )
        port_section = section("Serial COM Port", port_row)

        res_row = hrow(self.cBox_Speed, self.chBox_freqHop)
        res_section = section("Resonance Frequency / Quartz Sensor", res_row)

        # Rebuild the multiplex row fresh (don't re-parent the old sub-layout).
        multi_row = hrow(self.cBox_MultiMode, self.pButton_PlateConfig)
        multi_section = section("Multiplex Mode", multi_row, self.chBox_MultiAuto)

        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(14)
        left_col.addLayout(op_section)
        left_col.addLayout(port_section)
        left_col.addLayout(res_section)
        left_col.addLayout(multi_section)
        left_col.addWidget(self.chBox_correctNoise)
        left_col.addStretch()

        # ---- Right column: cartridge auto-lock, plot mode, control buttons ----
        lock_row = hrow(self.lbl_lock_manual, self.toggle_Cartridge, self.lbl_lock_auto, spacing=8)

        # Plotting mode toggle (Absolute <-> Reference), its own subsection.
        plot_mode_row = hrow(
            self.lbl_plot_absolute,
            self.toggle_PlotMode,
            self.lbl_plot_reference,
            spacing=8,
        )

        btns = QtWidgets.QVBoxLayout()
        btns.setSpacing(8)
        btns.addWidget(self.pButton_Start)
        btns.addWidget(self.pButton_Stop)
        btns.addWidget(self.pButton_Clear)
        btns.addWidget(self.pButton_ResetApp)

        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(14)
        right_col.addLayout(section("Cartridge Auto-Lock", lock_row, stretch_last=False))
        right_col.addLayout(section("Plot Mode", plot_mode_row, stretch_last=False))
        right_col.addLayout(section("Control Buttons", btns, stretch_last=False))
        right_col.addStretch()

        # ---- Assemble columns ----
        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(22)
        columns.addLayout(left_col, 3)
        columns.addLayout(right_col, 2)

        # ---- Outer: status row on top, then the columns ----
        # ---- Outer: columns, then a quiet status readout at the BOTTOM ----
        # Instead of a blocked-out bar at the top, the infobar becomes a low-key
        # readout line at the foot of the panel (icon-free, borderless, muted).
        self.infobar_readout = QtWidgets.QLabel()
        self.infobar_readout.setObjectName("infobarReadout")
        self.infobar_readout.setStyleSheet(
            "QLabel#infobarReadout { color: rgba(70, 90, 110, 190); font-size: 10px; "
            "background: transparent; border: none; padding: 0px 2px; }"
        )
        self.infobar_readout.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)

        def _mirror_status(t):
            # Strip the legacy "Infobar" prefix/markup; the readout is labelled
            # "Status:" now, so the old prefix is redundant noise.
            clean = self._strip_infobar_prefix(t)
            self.infobar_readout.setText(f"Status:  {clean}" if clean else "Status:  Ready")

        # Mirror live status text from the existing infobar line edit.
        self.infobar.textChanged.connect(_mirror_status)
        _mirror_status(self.infobar.text())

        outer = QtWidgets.QVBoxLayout()
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(10)
        outer.addLayout(columns)
        outer.addWidget(_hairline())
        outer.addWidget(self.infobar_readout)

        return outer

    def action_advanced(self, obj=None):
        main_window = getattr(self, "parent", None)
        popup = AdvancedMainWidget.toggle(
            owner=self,
            anchor=self.tool_Advanced,
            controls_layout=self._advanced_controls_layout,
            main_window=main_window,
        )
        if popup is None:
            return  # toggled closed
        # Keep a handle to the container the popup built (size queries, etc.).
        self.advanced_container = popup.content_container

    # -- Account dropdown -----------------------------------------------------

    @staticmethod
    def _is_user_signed_in() -> bool:
        """Return True if a valid user session is currently active."""
        try:
            from QATCH.common.userProfiles import UserProfiles  # noqa: PLC0415

            is_valid, _ = UserProfiles.session_info()
            return bool(is_valid)
        except Exception:
            return False

    def refresh_user_button_state(self) -> None:
        """Enable or disable the Account toolbar button based on the current session.

        Call this after a user signs in or signs out so the button reflects the
        live authentication state.  The button is disabled (greyed-out) when no
        session is active, and re-enabled once a valid session is established.
        """
        signed_in = self._is_user_signed_in()
        if hasattr(self, "tool_User"):
            self.tool_User.setEnabled(signed_in)
            # If a popup is open and the user just signed out, close it
            if not signed_in:
                popup = getattr(self, "_account_popup", None)
                if popup is not None:
                    try:
                        popup.close()
                    except Exception:
                        pass

    def _toggle_account_popup(self) -> None:
        """Show the glass account-info popup anchored below the Account button.

        The popup uses ``Qt.Popup`` so Qt closes it automatically on any outside
        click, including a click on the Account button itself — which then
        triggers this slot again to re-show a fresh popup with up-to-date info.
        We keep a reference on ``self`` so the widget isn't garbage-collected
        while it's animating in.  Position is clamped to the main window and
        the popup auto-closes if the main window is resized.
        """
        # Tear down any prior popup to avoid stacking translucent windows
        prev = getattr(self, "_account_popup", None)
        if prev is not None:
            try:
                prev.close()
                prev.deleteLater()
            except Exception:
                pass

        self._account_popup = GlassAccountPopup(
            open_manager_cb=self._open_user_manager,
            sign_out_cb=self._sign_out_current_user,
        )
        # Anchor below the Account button; clamp to main window; auto-close on resize
        main_window = getattr(self, "parent", None)
        self._account_popup.show_anchored_to(self.tool_User, main_window=main_window)

    def _sign_out_current_user(self) -> None:
        """Sign out the current user and refresh the UI state accordingly."""
        try:
            if self.parent.parent.MainWin.ui0._set_no_user_mode(None):
                UserProfiles().session_end()
                name = self.parent.username.text()[6:]
                Log.i(f"Goodbye, {name}! You have been signed out.")
                self.parent.username.setText("User: [NONE]")
                self.parent.userrole = UserRoles.NONE
                self.parent.signinout.setText("&Sign In")
                self.parent.manage.setText("&Manage Users...")
                self.parent.ui1.tool_User.setText("Anonymous")
                self.parent.parent.AnalyzeProc.tool_User.setText("Anonymous")
                self.refresh_user_button_state()
            else:
                Log.d("User has unsaved changes in Analyze mode. Sign out aborted.")
        except Exception as exc:
            Log.e(f"Error signing out user: {exc}")

    def _open_user_manager(self) -> None:
        """Open the User Profiles Manager overlay (admin-only).

        The manager is now a glassmorphic child overlay of the main window —
        it covers the parent widget directly rather than opening an OS-managed
        window.  Geometry and z-order are handled internally by the widget, so
        no post-construction anchoring or activateWindow() calls are needed.
        """
        try:
            is_valid, user_info = UserProfiles.session_info()
            if not (is_valid and user_info and user_info[2] == UserRoles.ADMIN.name):
                Log.w("User Profiles Manager: an admin session is required to manage users.")
                return

            admin_name = user_info[0] or ""
            # Step up to MainWindow (self.parent.parent) to target the large MainWin
            main_app = getattr(self.parent, "parent", None)

            if main_app is not None and hasattr(main_app, "MainWin"):
                parent_win = main_app.MainWin.centralWidget() or main_app.MainWin
            else:
                # Fallback if MainWin isn't found
                parent_win = getattr(self, "parent", None)
                if parent_win is not None and hasattr(parent_win, "centralWidget"):
                    parent_win = parent_win.centralWidget() or parent_win

            Log.d(f"[UIControls] _open_user_manager: resolved parent_win={parent_win}")

            # If a manager overlay is already visible, just raise it
            existing = getattr(self, "_user_profiles_manager", None)
            if existing is not None:
                if existing.isVisible():
                    existing.raise_()
                    return
                # Hidden / closed but still in memory — clean up before re-creating
                try:
                    existing.deleteLater()
                except Exception:
                    pass

            self._user_profiles_manager = UserProfilesManagerWidget(
                parent=parent_win, admin_name=admin_name
            )
            # Size to the parent before showing so the overlay never appears at
            # its default (small) size for a frame — that transient frame is the
            # "dialog window" flash.
            try:
                self._user_profiles_manager.setGeometry(parent_win.rect())
            except Exception:
                pass
            self._user_profiles_manager.show()
        except Exception as exc:
            Log.e(f"UIControls._open_user_manager error: {exc}")

    def _anchor_user_manager_to_button(self, manager: QtWidgets.QWidget) -> None:
        """DEPRECATED — no longer called.

        The UserProfilesManagerWidget is now a glassmorphic overlay child of
        the main window and manages its own geometry via ``_refit_to_parent``.
        This method can be safely removed in a future cleanup pass.
        """
        pass

    @staticmethod
    def _strip_infobar_prefix(text: str) -> str:
        """Remove the legacy 'Infobar' prefix and HTML markup from status text.

        Status messages historically embedded a blue '<font>Infobar</font>'
        prefix. The readout is now labelled 'Status:', so strip both the markup
        and a leading 'Infobar' token, returning clean plain text.
        """
        if not text:
            return ""
        import re  # noqa: PLC0415

        # Drop HTML tags, collapse whitespace.
        plain = re.sub(r"<[^>]+>", "", text)
        plain = re.sub(r"\s+", " ", plain).strip()
        # Remove a leading "Infobar" token if present.
        plain = re.sub(r"^infobar[:\s-]*", "", plain, flags=re.IGNORECASE).strip()
        return plain

    def _update_plate_config_enabled(self, *args):
        """Enable Plate Config only when more than one channel is in play.

        The button is disabled when the Multiplex Mode is '1 Channel' (index 0)
        or when only a single channel option exists in the dropdown — there is
        no multi-well plate to configure in those cases.
        """
        if not hasattr(self, "pButton_PlateConfig"):
            return
        only_one_option = self.cBox_MultiMode.count() <= 1
        single_channel = self.cBox_MultiMode.currentIndex() <= 0
        self.pButton_PlateConfig.setEnabled(not (only_one_option or single_channel))

    def doPlateConfig(self):
        if hasattr(self, "wellPlateUI"):
            if self.wellPlateUI.isVisible():
                self.wellPlateUI.close()

        # The "Configure..." sentinel was removed from this dropdown (it is now
        # a dedicated button), so the device count is the full item count.
        num_ports = self.cBox_Port.count()
        if num_ports == 5:
            num_ports = 4
        i = self.cBox_Port.currentText()
        i = 0 if i.find(":") == -1 else int(i.split(":")[0], base=16)
        if i % 9 == i:
            well_width = 4
            well_height = 1
        else:
            well_width = 6
            well_height = 4
        num_channels = self.cBox_MultiMode.currentIndex() + 1
        if num_ports not in [well_width, well_height] or num_ports == 1:
            PopUp.warning(
                self.parent,
                "Plate Configuration",
                f"<b>Multiplex device(s) are required for plate configuration.</b><br/>"
                + f"You must have exactly 4 device ports connected for this mode.<br/>"
                + f"Currently connected device port count is: {num_ports} (not 4)",
            )
        else:
            self.wellPlateUI = WellPlate(well_width, well_height, num_channels)
