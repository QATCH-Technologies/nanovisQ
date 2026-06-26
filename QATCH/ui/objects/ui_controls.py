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
from QATCH.ui.widgets.data_management_widget import DataManagementWidget
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
)
from QATCH.ui.components.glass_warning_label import GlassWarningLabel
from QATCH.ui.components.glass_push_button import GlassPushButton
from QATCH.ui.components.glass_toggle import GlassToggle
from QATCH.ui.components.animated_combo_box import AnimatedComboBox
from QATCH.ui.components.animated_spin_box import AnimatedDoubleSpinBox

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
        self.data_management_widget = None
        # self.ui_configure_data = UIConfigureData()
        self.ui_preferences = UserPreferencesWidget(self)
        # self.userrole
        self.current_timer = QtCore.QTimer()
        # self.current_timer.timeout.connect(self.double_toggle_plots)
        UserProfiles().session_end()

    def _createMenu(self, target):
        self._menu_target = target  # real menu bar lives on MainWin, not here
        self.menubar = []
        self.menubar.append(target.menuBar().addMenu("&Options"))
        self.act_analyze_data = self.menubar[0].addAction("&Analyze Data", self.analyze_data)
        self.act_import_data = self.menubar[0].addAction("&Import Data", self.import_data)
        self.act_export_data = self.menubar[0].addAction("&Export Data", self.export_data)
        self.act_recover_data = self.menubar[0].addAction("&Recover Data", self.recover_data)
        # self.menubar[0].addAction('&Configure Data', self.configure_data)
        self.act_preferences = self.menubar[0].addAction("&Preferences", self.preferences)
        self.act_find_devices = self.menubar[0].addAction("&Find Devices", self.scan_subnets)
        self.menubar[0].addAction("E&xit", self.close)
        self.menubar.append(target.menuBar().addMenu("&Users"))
        self.username = self.menubar[1].addAction("User: [NONE]")
        self.username.setEnabled(False)
        self.signinout = self.menubar[1].addAction("&Sign In", self.set_user_profile)
        self.act_select_directory = self.menubar[1].addAction(
            "Select &directory...", self.set_working_directory
        )
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
        self.act_check_updates = self.menubar[3].addAction(
            "&Check for Updates", self.check_for_updates
        )
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

        # Glass-rounded popup setup, matching the proven AnimatedComboBox
        # pattern (animated_combo_box.py) instead of WA_TranslucentBackground:
        # true per-pixel alpha on a native Windows popup composites as a
        # muddy grey-brown smear, and QSS border-radius never clips the
        # window's own (rectangular) shape, leaving square corners poking
        # out from under the rounded paint. An opaque QSS background plus an
        # explicit rounded QRegion mask avoids both.
        for menu in (*self.menubar, self.modebar):
            menu.setAttribute(QtCore.Qt.WA_TranslucentBackground, False)
            menu.setWindowFlag(QtCore.Qt.NoDropShadowWindowHint, True)
            menu.aboutToShow.connect(lambda m=menu: self._mask_menu_to_rounded_rect(m))

    @staticmethod
    def _mask_menu_to_rounded_rect(menu: QtWidgets.QMenu, radius: int = 10) -> None:
        """Clips `menu` to a rounded-rect window mask once it has its real size.

        Deferred a tick because the menu's size isn't final until just after
        it's shown (see AnimatedComboBox._apply_popup_mask for the same need).
        """

        def _apply() -> None:
            if not menu.isVisible():
                return
            rect = menu.rect()
            path = QtGui.QPainterPath()
            path.addRoundedRect(QtCore.QRectF(rect), radius, radius)
            menu.setMask(QtGui.QRegion(path.toFillPolygon().toPolygon()))

        QtCore.QTimer.singleShot(0, _apply)

    def set_signed_in_menu_state(self, signed_in: bool) -> None:
        """Locks down the menu bar while no user is signed in.

        The glass login overlay is the supported sign-in gate, so while it's
        showing, Options/Users/View are restricted to bare essentials
        (Options keeps only Exit). Help stays available for read-only
        reference, except update-checking and switching prediction model
        versions, which are treated as operator actions.

        Args:
            signed_in (bool): True to restore full menu access, False to
                lock it down to the signed-out subset.
        """
        # Options: only Exit stays enabled while signed out.
        for action in (
            self.act_analyze_data,
            self.act_import_data,
            self.act_export_data,
            self.act_recover_data,
            self.act_preferences,
            self.act_find_devices,
        ):
            action.setEnabled(signed_in)

        # Users: fully locked down while signed out.
        self.signinout.setEnabled(signed_in)
        self.act_select_directory.setEnabled(signed_in)
        self.manage.setEnabled(signed_in)

        # View: fully locked down while signed out.
        self.modebar.menuAction().setEnabled(signed_in)
        self.chk1.setEnabled(signed_in)
        self.chk2.setEnabled(signed_in)
        self.chk3.setEnabled(signed_in)
        self.chk4.setEnabled(signed_in)

        # Help: enabled regardless, except updates and model-version switching.
        self.act_check_updates.setEnabled(signed_in)
        self.menubar[5].menuAction().setEnabled(signed_in)  # "Model versions" submenu

        self._apply_menu_bar_theme(signed_in)

        # The Account toolbar button (top-right, next to Advanced) must be
        # re-enabled the moment a session starts — previously nothing called
        # this on sign-in, so it stayed disabled (built disabled, before any
        # session existed) until a *second* launch happened to start with an
        # already-valid session from the prior run.
        self.ui1.refresh_user_button_state()

    def _apply_menu_bar_theme(self, signed_in: bool) -> None:
        """Tints the native menu bar and its dropdowns to match the glass
        theme instead of the stark native default. While the sign-in gate
        is up, this applies the *same* ~30% dimming the login overlay
        applies to the blurred dashboard to the bar's own light tone — not
        an unrelated dark color — so it reads as part of the same dimmed
        scene. The normal light/glass tone returns once signed in.

        QMenu (the dropdowns) is styled in the *same* stylesheet as
        QMenuBar — a QMenu created via addMenu() is a logical child of the
        bar in Qt's object tree, so it inherits this stylesheet too, even
        though it pops up as its own top-level window.

        Args:
            signed_in (bool): Which palette to apply.
        """
        target = getattr(self, "_menu_target", None)
        if target is None:
            return

        if signed_in:
            target.menuBar().setStyleSheet("""
                QMenuBar {
                    background: rgba(233, 239, 244, 255);
                    color: rgba(50, 60, 70, 230);
                    border: none;
                }
                QMenuBar::item { background: transparent; padding: 4px 10px; }
                QMenuBar::item:selected { background: rgba(10, 163, 230, 60); border-radius: 4px; }
                QMenuBar::item:disabled { color: rgba(120, 130, 140, 140); }

                QMenu {
                    background: rgb(233, 239, 244);
                    border: 1px solid rgba(255, 255, 255, 230);
                    border-radius: 10px;
                    padding: 6px;
                }
                QMenu::item {
                    background: transparent;
                    color: rgba(50, 60, 70, 230);
                    padding: 6px 26px 6px 14px;
                    border-radius: 6px;
                }
                QMenu::item:selected { background: rgba(10, 163, 230, 60); }
                QMenu::item:disabled { color: rgba(120, 130, 140, 140); }
                QMenu::separator {
                    height: 1px;
                    background: rgba(120, 130, 140, 70);
                    margin: 4px 10px;
                }
                QMenu::indicator { width: 14px; height: 14px; }
            """)
        else:
            target.menuBar().setStyleSheet("""
                QMenuBar {
                    background: rgba(163, 167, 171, 255);
                    color: rgba(40, 48, 56, 235);
                    border: none;
                }
                QMenuBar::item { background: transparent; padding: 4px 10px; }
                QMenuBar::item:selected { background: rgba(255, 255, 255, 60); border-radius: 4px; }
                QMenuBar::item:disabled { color: rgba(90, 98, 106, 150); }

                QMenu {
                    background: rgb(163, 167, 171);
                    border: 1px solid rgba(255, 255, 255, 90);
                    border-radius: 10px;
                    padding: 6px;
                }
                QMenu::item {
                    background: transparent;
                    color: rgba(40, 48, 56, 235);
                    padding: 6px 26px 6px 14px;
                    border-radius: 6px;
                }
                QMenu::item:selected { background: rgba(255, 255, 255, 60); }
                QMenu::item:disabled { color: rgba(90, 98, 106, 150); }
                QMenu::separator {
                    height: 1px;
                    background: rgba(255, 255, 255, 80);
                    margin: 4px 10px;
                }
                QMenu::indicator { width: 14px; height: 14px; }
            """)

    def _data_overlay_parent(self):
        """Parent to the overarching MainWin (full app window), not the thin
        ControlsWindow — otherwise the overlay is locked to the controls bar."""
        if hasattr(self.parent, "MainWin"):
            return self.parent.MainWin.centralWidget() or self.parent.MainWin
        return self.centralWidget() or self

    def _open_data_management(self, mode):
        parent = self._data_overlay_parent()
        # (Re)build if missing or parented to a stale widget.
        if self.data_management_widget is None or self.data_management_widget.parent is not parent:
            self.data_management_widget = DataManagementWidget(parent=parent)
        # Size to the full overlay parent up front so it doesn't pop small then snap.
        try:
            self.data_management_widget.setGeometry(parent.rect())
        except Exception:
            pass
        self.data_management_widget.open_mode(mode)

    def analyze_data(self):
        self.parent.MainWin.ui0._set_analyze_mode(self)

    def import_data(self):
        self._open_data_management("import")

    def export_data(self):
        self._open_data_management("export")

    def recover_data(self):
        self._open_data_management("recover")

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
        # Disallow user management if the current mode is busy or has unsaved
        # changes — but otherwise leave the active mode untouched. The manager
        # is a glassmorphic overlay now, not a modal dialog, so it no longer
        # needs to force-switch to Run mode to open.
        if not self.parent.MainWin.ui0._check_mode_change_allowed():
            Log.d("User has unsaved changes in Analyze mode. Manage users aborted.")
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
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        rect_f = QtCore.QRectF(self.rect())

        # Clip to rounded rectangle
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)

        # Frosted base — identical to GlassContainer
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 160))
        p.fillRect(self.rect(), QtGui.QColor(228, 235, 241, 18))

        # Top shimmer — three-stop, 40px (matches GlassContainer)
        shimmer = QtGui.QLinearGradient(0, 0, 0, 40)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 100))
        shimmer.setColorAt(0.5, QtGui.QColor(255, 255, 255, 20))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Bottom vignette (was missing)
        vg = QtGui.QLinearGradient(0, self.height() - 30, 0, self.height())
        vg.setColorAt(0.0, QtGui.QColor(200, 218, 240, 0))
        vg.setColorAt(1.0, QtGui.QColor(200, 218, 240, 18))
        p.fillRect(self.rect(), QtGui.QBrush(vg))

        # Borders — outer bright rim + inner cool-grey inset
        p.setClipping(False)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 230), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)
        p.setPen(QtGui.QPen(QtGui.QColor(190, 210, 235, 70), 1.0))
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
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
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
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
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
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
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
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
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
        p.drawText(rect.toRect(), QtCore.Qt.AlignmentFlag.AlignCenter, self._initials)
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
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)

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
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
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
            QtCore.Qt.WindowType.Popup
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.NoDropShadowWindowHint,
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
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

        # -- entrance animation (matches the advanced menu) --
        # Fade the whole popup window in (setWindowOpacity, NOT a graphics
        # effect — an opacity effect on this panel would clash with the drop
        # shadow above and cause the same ghosting seen in the advanced panel),
        # paired with a brief downward slide so it eases out from the anchor.
        self._enter_fade = QtCore.QVariantAnimation(self)
        self._enter_fade.setDuration(200)
        self._enter_fade.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._enter_fade.setStartValue(0.0)
        self._enter_fade.setEndValue(1.0)
        self._enter_fade.valueChanged.connect(lambda v: self.setWindowOpacity(float(v)))
        self._enter_fade.finished.connect(lambda: self.setWindowOpacity(1.0))

        self._enter_slide = QtCore.QPropertyAnimation(self, b"pos", self)
        self._enter_slide.setDuration(220)
        self._enter_slide.setEasingCurve(QtCore.QEasingCurve.OutCubic)

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
            manage_btn = GlassPushButton(" Manage Users…")
            manage_btn.setIcon(QtGui.QIcon(icon_path))
            manage_btn.setIconSize(QtCore.QSize(16, 16))
            manage_btn.setStyleSheet("""
                QPushButton {
                    background: rgba(0, 118, 174, 18);
                    color: rgba(0, 118, 174, 230);
                    border: 1px solid rgba(0, 118, 174, 55);
                    border-radius: 5px;
                    padding: 8px 14px 8px 12px;
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
            manage_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
            manage_btn.clicked.connect(self._on_manage_users)
            layout.addWidget(manage_btn)

        if show_sign_out:
            icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "sign-out.svg")
            sign_out_btn = GlassPushButton(" Sign Out")
            if os.path.exists(icon_path):
                sign_out_btn.setIcon(QtGui.QIcon(icon_path))
                sign_out_btn.setIconSize(QtCore.QSize(16, 16))
            sign_out_btn.setStyleSheet("""
                QPushButton {
                    background: rgba(200, 70, 40, 16);
                    color: rgba(170, 55, 30, 235);
                    border: 1px solid rgba(200, 70, 40, 60);
                    border-radius: 5px;
                    padding: 8px 14px 8px 12px;
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
            sign_out_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
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

        # Track resize/move events on the main window so the popup never
        # ends up floating outside the application after a resize.
        if self._main_window is not None:
            self._main_window.installEventFilter(self)

        # Entrance animation: start slightly above the final position and fade
        # in as it slides down to (x, y) — same feel as the advanced menu.
        final_pos = QtCore.QPoint(x, y)
        start_pos = QtCore.QPoint(x, y - 12)
        self.move(start_pos)
        self.setWindowOpacity(0.0)
        self.show()

        self._enter_slide.stop()
        self._enter_slide.setStartValue(start_pos)
        self._enter_slide.setEndValue(final_pos)
        self._enter_slide.start()

        self._enter_fade.stop()
        self._enter_fade.start()

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
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.Move,
            QtCore.QEvent.Type.WindowStateChange,
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


class _DeviceConfigTitle(QtWidgets.QLabel):
    """Title label for the device-config perspective that stays banner-compatible.

    The device perspective replaced the old ``GlassWarningLabel`` banner with a
    clean left-aligned title. External code (mainWindow.py) still drives that
    former banner through ``.text()`` / ``.setText()`` to stamp the connected
    device handle onto it (e.g. ``"Configuration Editor for Device 1A"`` and a
    later ``.endswith(dev_handle)`` check).

    To keep that contract intact while showing a tidy title, this widget stores
    the raw banner string verbatim — ``text()`` returns exactly what was set, so
    every existing string operation behaves as before — but the VISIBLE text is
    a friendlier rendering: ``"Device Configuration"`` plus the trailing handle
    when one is present.
    """

    _PREFIX = "Configuration Editor for Device"
    _DISPLAY_BASE = "Device Configuration"

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(parent)
        self._raw_text = ""
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 (Qt override)
        self._raw_text = text if text is not None else ""
        super().setText(self._render(self._raw_text))

    def text(self) -> str:  # noqa: N802 (Qt override)
        # Return the raw banner string so external endswith()/split() logic
        # keeps operating on the same value the old banner exposed.
        return self._raw_text

    def _render(self, raw: str) -> str:
        """Map a raw banner string to the friendly visible title.

        Anything trailing the legacy prefix (the device handle) is rendered as a
        small themed "chip" (bordered, filled, colored) so it reads clearly as
        the connected device's name rather than plain trailing text. Without a
        handle, just the base title is shown.
        """
        handle = ""
        if raw.startswith(self._PREFIX):
            handle = raw[len(self._PREFIX) :].strip()
        base = (
            f"<span style='color: rgba(28,40,52,235); font-size:14px; "
            f"font-weight:bold;'>{self._DISPLAY_BASE}</span>"
        )
        if handle:
            chip = (
                "<span style='"
                "background: rgba(10,163,230,38); "
                "color: rgba(12,110,160,255); "
                "border: 1px solid rgba(10,163,230,120); "
                "border-radius: 7px; "
                "padding: 1px 7px; "
                "font-size: 12px; font-weight: bold; "
                "letter-spacing: 0.5px;"
                f"'>&nbsp;{handle}&nbsp;</span>"
            )
            # The em-space keeps a small gap between the title and the chip.
            return f"{base}&#8195;{chip}"
        return base


class _SavedStateDot(QtWidgets.QWidget):
    """A small glowing status dot that reflects a field's save state.

    States, each a themed color with a soft outer glow:
        * ``"querying"`` — red, pulsing (waiting on the device for a value)
        * ``"blank"``    — quiet gray (no pending change, no value yet)
        * ``"unsaved"``  — amber, gently pulsing to draw the eye
        * ``"saved"``    — green, steady

    The dot is purely visual; the authoritative state still lives on the
    paired field action's ``iconText()`` (so all existing save/reset logic
    keeps working). Call :meth:`set_state` whenever that text changes.

    A one-shot :meth:`flash` pulse is used to nag the user about an unsaved
    field when they try to leave the device view.
    """

    _COLORS = {
        "blank": QtGui.QColor(150, 165, 180),
        "unsaved": QtGui.QColor(240, 170, 50),
        "saved": QtGui.QColor(60, 190, 120),
        "querying": QtGui.QColor(228, 70, 70),
    }
    # States that should pulse continuously.
    _PULSING = ("unsaved", "querying")
    _SIZE = 14

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(self._SIZE, self._SIZE)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._state = "blank"
        self._glow = 0.0  # 0..1 animated glow strength (pulsing when active)
        self._flash = 0.0  # 0..1 one-shot attention flash overlay

        # Continuous gentle pulse used while in a pulsing state.
        self._pulse = QtCore.QVariantAnimation(self)
        self._pulse.setStartValue(0.25)
        self._pulse.setEndValue(1.0)
        self._pulse.setDuration(900)
        self._pulse.setEasingCurve(QtCore.QEasingCurve.InOutSine)
        self._pulse.setLoopCount(-1)
        self._pulse.valueChanged.connect(self._on_pulse)

        # One-shot attention flash (used by flash()).
        self._flash_anim = QtCore.QVariantAnimation(self)
        self._flash_anim.setStartValue(1.0)
        self._flash_anim.setEndValue(0.0)
        self._flash_anim.setDuration(520)
        self._flash_anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._flash_anim.valueChanged.connect(self._on_flash)

    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        if state not in self._COLORS:
            state = "blank"
        if state == self._state:
            return  # idempotent: avoid restarting the pulse on repeat calls
        self._state = state
        if state in self._PULSING:
            self._pulse.stop()
            self._pulse.start()
        else:
            self._pulse.stop()
            self._glow = 1.0 if state == "saved" else 0.0
        self.update()

    def flash(self, times: int = 3) -> None:
        """Play a brief attention pulse (used to nag about unsaved changes)."""
        # Restart the one-shot flash; if not already unsaved-pulsing, this still
        # reads as a distinct nudge because of the brighter overlay.
        self._flash_anim.stop()
        self._flash_anim.setLoopCount(max(1, times))
        self._flash_anim.start()

    def _on_pulse(self, v) -> None:
        self._glow = float(v)
        self.update()

    def _on_flash(self, v) -> None:
        self._flash = float(v)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        c = QtGui.QColor(self._COLORS[self._state])
        cx, cy = self.width() / 2.0, self.height() / 2.0

        # Outer glow halo (strength driven by pulse / flash).
        glow = max(self._glow, self._flash)
        if glow > 0.01:
            halo = QtGui.QColor(c)
            halo.setAlphaF(0.35 * glow)
            radius = 4.0 + 3.0 * glow
            grad = QtGui.QRadialGradient(cx, cy, radius)
            grad.setColorAt(0.0, halo)
            transparent = QtGui.QColor(c)
            transparent.setAlpha(0)
            grad.setColorAt(1.0, transparent)
            p.setBrush(QtGui.QBrush(grad))
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

        # Core dot.
        core_r = 3.4
        p.setBrush(QtGui.QBrush(c))
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 170), 1.0))
        p.drawEllipse(QtCore.QPointF(cx, cy), core_r, core_r)
        p.end()


class _BorderlessActionButton(QtWidgets.QPushButton):
    """A flat, borderless text button that lightens on hover.

    Used for the per-field Save / Reset / Default actions so they read as quiet
    inline links rather than heavy glass pills. No border or fill at rest; a
    subtle translucent gray wash appears on hover and deepens on press.
    """

    def __init__(self, text: str = "", parent=None, *, tone: str = "neutral") -> None:
        super().__init__(text, parent)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setFlat(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        # Tones let "Save" read a touch more primary than Reset/Default.
        _text_color = {
            "neutral": "rgba(60, 78, 96, 230)",
            "primary": "rgba(20, 120, 180, 235)",
        }.get(tone, "rgba(60, 78, 96, 230)")
        self.setStyleSheet(
            f"""
            QPushButton {{
                border: none;
                background: transparent;
                color: {_text_color};
                font-size: 12px;
                font-weight: 600;
                padding: 4px 10px;
                border-radius: 7px;
            }}
            QPushButton:hover {{
                background: rgba(120, 140, 160, 45);
            }}
            QPushButton:pressed {{
                background: rgba(120, 140, 160, 80);
            }}
            QPushButton:disabled {{
                color: rgba(120, 135, 150, 120);
                background: transparent;
            }}
            """
        )


class _FieldStateAction:
    """A minimal stand-in for a QLineEdit trailing QAction.

    The device-config save/reset logic tracks each field's state purely through
    a QAction's ``iconText()`` ("blank" / "unsaved" / "saved") and occasionally
    ``setIcon()``. A bare QWidget like :class:`_RangeSliderField` has no such
    line-edit action, so this lightweight object provides the same tiny surface
    (``setIcon`` / ``setIconText`` / ``iconText``) and nothing else. It is used
    as the authoritative state holder for the slider fields, keeping every
    existing call site working without special-casing.
    """

    def __init__(self) -> None:
        self._icon_text = "blank"
        self._icon = None

    def setIcon(self, icon) -> None:  # noqa: N802 (Qt-like API)
        self._icon = icon

    def icon(self):
        return self._icon

    def setIconText(self, text: str) -> None:  # noqa: N802 (Qt-like API)
        self._icon_text = text or ""

    def iconText(self) -> str:  # noqa: N802 (Qt-like API)
        return self._icon_text


class _RangeSliderField(QtWidgets.QWidget):
    """A horizontal slider paired with a live, editable numeric box.

    Replaces the preset dropdown + read-only box for the Lid POGO calibration
    fields. Dragging the slider updates the number box; typing in the number box
    (or stepping it) moves the slider. A single ``valueChanged(int)`` signal is
    emitted on any change so callers can mark the field unsaved.

    The numeric box exposes ``text()`` / ``setText()`` and the slider range is
    configurable, so the surrounding save/reset/default logic only needs to read
    and write a string value (mirroring the old read-only QLineEdit contract).
    """

    valueChanged = QtCore.pyqtSignal(int)

    def __init__(
        self,
        minimum: int,
        maximum: int,
        suffix: str = "",
        parent=None,
        up_icon_path: str = "",
        down_icon_path: str = "",
    ) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setStyleSheet("background: transparent;")
        self._suffix = suffix
        self._guard = False  # re-entrancy guard for cross-updates

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(5)
        self.slider.setStyleSheet(_RANGE_SLIDER_QSS)
        self.slider.setMinimumWidth(120)
        self.slider.valueChanged.connect(self._on_slider)

        # Animated spin box (integer-configured: 0 decimals, step 1) with custom
        # up/down chevrons, matching the temperature-calibration inputs.
        self.spin = AnimatedDoubleSpinBox(
            up_icon_path=up_icon_path,
            down_icon_path=down_icon_path,
        )
        self.spin.setDecimals(0)
        self.spin.setMinimum(float(minimum))
        self.spin.setMaximum(float(maximum))
        self.spin.setSingleStep(1)
        self.spin.setSuffix(suffix)
        self.spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.spin.setFixedWidth(78)
        self.spin.valueChanged.connect(self._on_spin)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin, 0)

    # -- QLineEdit-compatible action shim ----------------------------------
    def addAction(self, *args, **kwargs):  # noqa: N802 (Qt override/overload)
        """Mimic ``QLineEdit.addAction(icon, position)``.

        The device-config code adds a trailing action to each input solely to
        carry save-state via ``iconText()``. This widget has no line-edit, so we
        return a lightweight :class:`_FieldStateAction` instead of touching the
        QWidget action list (which only accepts real QActions and would raise on
        a QIcon). If called with a real QAction (the QWidget signature), defer
        to the base implementation.
        """
        if args and isinstance(args[0], QtGui.QIcon):
            return _FieldStateAction()
        return super().addAction(*args, **kwargs)

    # -- cross-wiring ------------------------------------------------------
    def _on_slider(self, v: int) -> None:
        if self._guard:
            return
        self._guard = True
        self.spin.setValue(v)
        self._guard = False
        self.valueChanged.emit(v)

    def _on_spin(self, v) -> None:
        if self._guard:
            return
        iv = int(round(float(v)))
        self._guard = True
        self.slider.setValue(iv)
        self._guard = False
        self.valueChanged.emit(iv)

    # -- string-compatible accessors (mirror the old QLineEdit contract) ---
    def value(self) -> int:
        return int(round(float(self.spin.value())))

    def setValue(self, v: int) -> None:  # noqa: N802
        self._guard = True
        try:
            iv = int(round(float(v)))
        except (TypeError, ValueError):
            iv = self.slider.minimum()
        self.slider.setValue(iv)
        self.spin.setValue(float(iv))
        self._guard = False

    def text(self) -> str:  # noqa: N802
        return str(self.value())

    def setText(self, text: str) -> None:  # noqa: N802
        try:
            self.setValue(int(round(float(str(text).strip()))))
        except (TypeError, ValueError):
            pass

    def hasAcceptableInput(self) -> bool:  # noqa: N802
        return self.slider.minimum() <= self.value() <= self.slider.maximum()

    def setPlaceholderText(self, text: str) -> None:  # noqa: N802
        # SpinBox has no placeholder; accept the call for API parity (no-op).
        return

    def clear(self) -> None:
        self.setValue(self.slider.minimum())


# Themed QSS for the Lid POGO range sliders + their numeric boxes.
_RANGE_SLIDER_QSS = """
    QSlider::groove:horizontal {
        height: 4px;
        background: rgba(0, 0, 0, 30);
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #0AA3E6;
        border: 1px solid rgba(0, 130, 200, 200);
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }
    QSlider::handle:horizontal:hover {
        background: #1FB3F0;
    }
    QSlider::sub-page:horizontal {
        background: rgba(10, 163, 230, 140);
        border-radius: 2px;
    }
"""

_RANGE_SPIN_QSS = """
    QSpinBox {
        background: rgba(255, 255, 255, 60);
        border: 1px solid rgba(255, 255, 255, 120);
        border-radius: 12px;
        padding: 4px 8px;
        color: rgba(30, 40, 55, 220);
    }
    QSpinBox:focus {
        background: rgba(255, 255, 255, 180);
        border: 1px solid rgba(0, 120, 215, 150);
    }
"""


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


class _PerspectiveAnimator(QtCore.QObject):
    """Plays a gentle entrance on a perspective container each time it is shown.

    IMPORTANT: this deliberately does NOT use QGraphicsOpacityEffect. Wrapping a
    container that holds custom-painted children (glass combos, toggles,
    buttons) in a graphics effect caches them into an offscreen pixmap, which
    causes ghosting, duplicated section labels, and widgets vanishing on hover.
    Instead the fade is applied to the top-level popup window via
    setWindowOpacity (no pixmap caching), paired with a brief top-margin slide.
    """

    def __init__(self, container: QtWidgets.QWidget) -> None:
        super().__init__(container)
        self._container = container

        # Slide the whole popup window DOWN into place (start 12px above the
        # final position), matching the account menu. Animating the inner
        # top-margin instead made content rise up from the bottom, which read
        # as a bottom-up animation.
        self._slide = QtCore.QVariantAnimation(self)
        self._slide.setDuration(220)
        self._slide.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._slide.setStartValue(0.0)
        self._slide.setEndValue(1.0)
        self._slide.valueChanged.connect(self._apply_slide)
        self._slide_from = None
        self._slide_to = None
        self._slide_offset = 12  # px above final at the start

        self._fade = QtCore.QVariantAnimation(self)
        self._fade.setDuration(200)
        self._fade.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.valueChanged.connect(self._apply_fade)
        self._fade.finished.connect(self._finish_fade)

        container.installEventFilter(self)

    def _apply_slide(self, t) -> None:
        win = self._container.window()
        if win is None or self._slide_to is None:
            return
        x = self._slide_to.x()
        y = int(self._slide_from.y() + (self._slide_to.y() - self._slide_from.y()) * float(t))
        win.move(x, y)

    def _apply_fade(self, v) -> None:
        win = self._container.window()
        if win is not None:
            win.setWindowOpacity(float(v))

    def _finish_fade(self) -> None:
        # Always settle fully opaque and at the exact final position.
        win = self._container.window()
        if win is not None:
            win.setWindowOpacity(1.0)
            if self._slide_to is not None:
                win.move(self._slide_to)

    def eventFilter(self, obj, event) -> bool:
        if obj is self._container and event.type() == QtCore.QEvent.Show:
            # The popup may not be moved to its final anchored position until
            # after this Show event (set_content_widget shows the container
            # before show_anchored_to moves the window). Defer one tick so we
            # capture the settled final position, then play the top-down slide.
            self._fade.stop()
            self._fade.start()
            QtCore.QTimer.singleShot(0, self._begin_slide)
        return super().eventFilter(obj, event)

    def _begin_slide(self) -> None:
        win = self._container.window()
        if win is None or not win.isVisible():
            return
        final_pos = win.pos()
        self._slide_to = QtCore.QPoint(final_pos)
        self._slide_from = QtCore.QPoint(final_pos.x(), final_pos.y() - self._slide_offset)
        win.move(self._slide_from)
        self._slide.stop()
        self._slide.start()


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
        self.pButton_ID.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.pButton_ID.setFixedSize(_CIRCLE_D, _CIRCLE_D)  # square -> circle
        self.pButton_ID.setObjectName("pButton_ID")
        self.Layout_controls.addWidget(self.pButton_ID, 2, 2, 1, 1)

        # Refresh button
        self.pButton_Refresh = GlassPushButton(variant="default")
        self.pButton_Refresh.setToolTip("Refresh Serial COM Port list")
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "refresh-cw.svg")
        self.pButton_Refresh.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Refresh.setIconSize(QtCore.QSize(18, 18))
        self.pButton_Refresh.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
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
        self.pButton_Configure.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.pButton_Configure.setFixedSize(_CIRCLE_D, _CIRCLE_D)  # square -> circle
        self.pButton_Configure.setObjectName("pButton_Configure")
        self.Layout_controls.addWidget(self.pButton_Configure, 2, 4, 1, 1)
        # Reflect device-connection state ('No device connected' when empty).
        self.cBox_Port.currentIndexChanged.connect(self._update_configure_enabled)
        self._update_configure_enabled()

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
        self.lTemp.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
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
            pixmap = pixmap.scaled(
                250, 50, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.FastTransformation
            )
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
        self.infostatus.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
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
        self.pButton_PlateConfig.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
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
        self.tool_NextPortRow.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
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
        self.tool_Initialize.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.tool_Initialize.clicked.connect(self.action_initialize)
        self.tool_bar.addWidget(self.tool_Initialize)

        self.tool_bar.addSeparator()

        # RunControls composite widget ----------------------------------------
        self.run_controls = RunControls()
        self.run_controls.startRequested.connect(self.action_start)
        self.run_controls.stopRequested.connect(self.action_stop)
        self.run_controls.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
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
        self.tool_Reset.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
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
        self.tool_TempControl.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
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
        self.tempStatusBar.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
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
            lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
            lbl.setStyleSheet("background: transparent; border: none; color: rgba(30,40,55,210);")

        self.tempPidInfo = QtWidgets.QFrame()
        self.tempPidInfo.setObjectName("tempPidInfo")
        pid_layout = QtWidgets.QVBoxLayout(self.tempPidInfo)
        pid_layout.setContentsMargins(8, 4, 8, 4)
        pid_layout.setSpacing(1)

        pid_header = QtWidgets.QLabel("PID INFO")
        pid_header.setObjectName("tempPidHeader")
        pid_header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
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
        self.tool_Advanced.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
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
        self.tool_User.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
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
            self.toolLayout.setContentsMargins(0, 0, 0, 0)
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
            # Smooth fade + slide entrance for the advanced perspective.
            self._install_perspective_animation(self.advanced_container)
        else:
            self.centralwidget.setLayout(self.gridLayout)

        # QLineEdit icons, trailing position.
        # The in-field checkmark / warning icons have been removed: the glowing
        # status dot beside each field now conveys save state. These remain as
        # (empty) icons so the existing ``setIcon(...)`` call sites are harmless
        # no-ops while the authoritative ``iconText()`` state machine is intact.
        self.blankIcon = QtGui.QIcon()
        self.savedIcon = QtGui.QIcon()
        self.unsavedIcon = QtGui.QIcon()

        # --- Device Info container, widgets and layout ---
        self.device_info_container = QtWidgets.QWidget()
        self.device_info_container.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True
        )
        self.device_info_container.setStyleSheet("background: transparent;")
        # The device perspective uses the same sectioned, two-column layout as
        # the advanced view, so it sizes to its content and shares the advanced
        # view's footprint rather than forcing an oversized floor.
        self.device_info_container.setMinimumWidth(500)
        self.device_info_container.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )

        # --- Header: Back Button + Title (mirrors the Advanced view) ---
        # The advanced perspective uses a left-aligned title row (icon + bold
        # title + hover-info icon). The device perspective matches that, with a
        # circular back button leading the row so the user can return to the
        # advanced menu. Dismissing via this button slides the panel left.
        self.back_btn = QtWidgets.QPushButton()
        self.back_btn.setIcon(
            QtGui.QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "left-arrow.svg"))
        )
        self.back_btn.setIconSize(QtCore.QSize(18, 18))
        self.back_btn.setFixedSize(32, 32)
        self.back_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.back_btn.setToolTip("Back to Advanced Options")
        self.back_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 40);
                border: 1px solid rgba(255, 255, 255, 100);
                border-radius: 16px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 80);
                border: 1px solid rgba(255, 255, 255, 150);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 30);
            }
        """)
        self.back_btn.clicked.connect(self.on_device_config_editor_close)

        # Title icon (gear, matching the advanced header's leading glyph).
        _dev_icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")
        self.device_config_icon = QtWidgets.QLabel()
        self.device_config_icon.setFixedSize(18, 18)
        self.device_config_icon.setScaledContents(True)
        self.device_config_icon.setStyleSheet("background: transparent; border: none;")
        _dev_gear = QtGui.QPixmap(os.path.join(_dev_icons_dir, "gear.svg"))
        if not _dev_gear.isNull():
            self.device_config_icon.setPixmap(_dev_gear)

        # Bold title, same typography as the advanced "Advanced Options" title.
        # This widget doubles as the legacy ``ConfigBannerWidget``: external
        # code (mainWindow.py) reads/writes the device handle through
        # ``.text()`` / ``.setText()`` exactly as it did with the old
        # GlassWarningLabel banner ("Configuration Editor for Device <handle>").
        # _DeviceConfigTitle stores that raw banner string verbatim (so
        # ``.text()`` / ``.endswith(dev_handle)`` keep working) while DISPLAYING
        # a clean, left-aligned title with the handle appended.
        self.device_config_title = _DeviceConfigTitle("Configuration Editor for Device")
        self.device_config_title.setStyleSheet(
            "QLabel { color: rgba(28, 40, 52, 235); font-size: 14px; "
            "font-weight: bold; background: transparent; border: none; }"
        )
        # Backwards-compatible alias for external references.
        self.ConfigBannerWidget = self.device_config_title

        # Hover-info icon mirroring the advanced view's _InfoIcon usage.
        self._device_info_text = (
            "Configuration editor for the connected device — set its name, "
            "position ID, temperature calibration, and lid POGO timing."
        )
        try:
            from QATCH.ui.widgets.advanced_main_widget import _InfoIcon  # noqa: PLC0415

            self.device_config_info = _InfoIcon(
                os.path.join(_dev_icons_dir, "warning-circle.svg"),
                tooltip=self._device_info_text,
            )
        except Exception:
            # Fallback: a plain static info glyph if _InfoIcon isn't importable.
            self.device_config_info = QtWidgets.QLabel()
            self.device_config_info.setFixedSize(16, 16)
            self.device_config_info.setScaledContents(True)
            self.device_config_info.setStyleSheet("background: transparent; border: none;")
            _info_pix = QtGui.QPixmap(os.path.join(_dev_icons_dir, "warning-circle.svg"))
            if not _info_pix.isNull():
                self.device_config_info.setPixmap(_info_pix)
            self.device_config_info.setToolTip(self._device_info_text)

        # Left-aligned header row: back button, gear, title, info, then stretch.
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(2, 0, 2, 0)
        header_layout.setSpacing(8)
        header_layout.addWidget(self.back_btn, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(self.device_config_icon, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(self.device_config_title, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(self.device_config_info, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        header_layout.addStretch()

        # --- Input validators ---
        self.validDeviceName = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r'[^\\/:*?"\'<>|]{1,12}')
        )  # 1-12 character string without forbidden characters
        self.validDevicePid = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"[0-9A-Fa-f]{1,2}")
        )  # HEX values '00' thru 'FF'
        self.validTempOffset = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(
                r"-?(?:[0-5](?:\.\d{0,2})?|6(?:\.(?:[0-2]\d?|3[0-5]?))?|6\.?|\.\d{1,2})"
            )
        )  # Decimals -6.35 thru 6.35 (inclusive) with precision 2
        self.validPogoPosition = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"[0-9]?[0-9]|1[0-7][0-9]|180")
        )  # 0 thru 180 degrees (rotation)
        self.validPogoDelayMs = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"[0-1]?[0-9]?[0-9]|2[0-4][0-9]|25[0-4]")
        )  # 0 thru 254 ms per step

        # Unified translucent style for text inputs
        glass_icon_style = """
            QLineEdit {
                background: rgba(255, 255, 255, 60);
                border: 1px solid rgba(255, 255, 255, 120);
                border-radius: 6px;
                padding: 4px 8px;
                color: rgba(30, 40, 55, 220);
            }
        """
        glass_input_style = glass_icon_style + """
            QLineEdit:focus {
                background: rgba(255, 255, 255, 180);
                border: 1px solid rgba(0, 120, 215, 150);
            }
        """
        # Pill-shaped variant (rounded ends) for the device text inputs so they
        # match the rounded glass combos / spin boxes used elsewhere.
        glass_pill_style = """
            QLineEdit {
                background: rgba(255, 255, 255, 60);
                border: 1px solid rgba(255, 255, 255, 120);
                border-radius: 14px;
                padding: 5px 12px;
                color: rgba(30, 40, 55, 220);
            }
            QLineEdit:focus {
                background: rgba(255, 255, 255, 180);
                border: 1px solid rgba(0, 120, 215, 150);
            }
        """
        # NOTE: the former ``glass_spinbox_style`` QSS block (which themed the
        # native QDoubleSpinBox up/down buttons) has been removed. The temp-cal
        # inputs are now AnimatedDoubleSpinBox instances, which carry their own
        # glass styling and custom animated chevrons.

        # Maps each field's QLineEdit "action" (the authoritative save-state
        # holder, via iconText) to its glowing status dot so the dot can mirror
        # state. Also maps action -> input widget so the field can pulse when it
        # has unsaved changes and the user tries to leave.
        self._field_dots = {}
        self._field_widgets = {}

        # A lightweight poller keeps every status dot in sync with its action's
        # iconText regardless of which code path changed it (timer-delayed reset
        # helpers, external mainWindow calls, etc.). set_state() is idempotent,
        # so this is effectively free when nothing changed.
        self._dot_sync_timer = QtCore.QTimer()
        self._dot_sync_timer.setInterval(150)
        self._dot_sync_timer.timeout.connect(self._sync_all_dots)
        self._dot_sync_timer.start()

        def _make_dot(action, widget):
            dot = _SavedStateDot()
            self._field_dots[action] = dot
            self._field_widgets[action] = widget
            return dot

        # Row 0L: Device Name
        self.device_name_input = QtWidgets.QLineEdit()
        self.device_name_input.setValidator(self.validDeviceName)
        self.device_name_input.setStyleSheet(glass_pill_style)
        self.device_name_input.setMinimumWidth(160)
        self.device_name_action = self.device_name_input.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.device_name_input.textEdited.connect(
            lambda text, action=self.device_name_action: self.on_text_edit(text, action)
        )
        self.device_name_dot = _make_dot(self.device_name_action, self.device_name_input)

        # Row 0R: Device Position ID
        self.device_pid_input = QtWidgets.QLineEdit()
        self.device_pid_input.setValidator(self.validDevicePid)
        self.device_pid_input.setStyleSheet(glass_pill_style)
        self.device_pid_input.setMinimumWidth(160)
        self.device_pid_action = self.device_pid_input.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.device_pid_input.textEdited.connect(
            lambda text, action=self.device_pid_action: self.on_text_edit(text, action)
        )
        self.device_pid_dot = _make_dot(self.device_pid_action, self.device_pid_input)

        self.device_config_default = _BorderlessActionButton("Default")
        self.device_config_default.clicked.connect(self.on_device_config_default)
        self.device_config_save = _BorderlessActionButton("Save", tone="primary")
        self.device_config_save.clicked.connect(self.on_device_config_save)
        self.device_config_reset = _BorderlessActionButton("Reset")
        self.device_config_reset.clicked.connect(self.on_device_config_reset)

        # Row 1L: Constant Temperature Calibration
        self.temp_cal_always_input = AnimatedDoubleSpinBox(
            up_icon_path=os.path.join(Architecture.get_path(), "QATCH", "icons", "up-chevron.svg"),
            down_icon_path=os.path.join(
                Architecture.get_path(), "QATCH", "icons", "down-chevron.svg"
            ),
        )
        self.temp_cal_always_input.setDecimals(2)
        self.temp_cal_always_input.setRange(-6.35, 6.35)
        self.temp_cal_always_input.setSingleStep(0.25)
        # self.temp_cal_always_input.setValidator(self.validTempOffset)
        self.temp_cal_always_input.setSuffix(" °C")
        self.temp_cal_always_input.setMinimumWidth(142)
        self.temp_cal_always_icon = QtWidgets.QLineEdit()
        self.temp_cal_always_icon.setStyleSheet(glass_icon_style)
        self.temp_cal_always_icon.setFixedWidth(self.temp_cal_always_icon.sizeHint().height())
        self.temp_cal_always_icon.setReadOnly(True)
        self.temp_cal_always_action = self.temp_cal_always_icon.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.temp_cal_always_input.textChanged.connect(
            lambda text, action=self.temp_cal_always_action: self.on_text_edit(text, action)
        )
        self.temp_cal_always_input.editingFinished.connect(
            lambda widget=self.temp_cal_always_input: self.on_edit_finish(widget)
        )
        self.temp_cal_always_dot = _make_dot(
            self.temp_cal_always_action, self.temp_cal_always_input
        )

        # Row 1R: Running Temperature Calibration
        self.temp_cal_measure_input = AnimatedDoubleSpinBox(
            up_icon_path=os.path.join(Architecture.get_path(), "QATCH", "icons", "up-chevron.svg"),
            down_icon_path=os.path.join(
                Architecture.get_path(), "QATCH", "icons", "down-chevron.svg"
            ),
        )
        self.temp_cal_measure_input.setDecimals(2)
        self.temp_cal_measure_input.setRange(-6.35, 6.35)
        self.temp_cal_measure_input.setSingleStep(0.25)
        # self.temp_cal_measure_input.setValidator(self.validTempOffset)
        self.temp_cal_measure_input.setSuffix(" °C")
        self.temp_cal_measure_input.setMinimumWidth(142)
        self.temp_cal_measure_icon = QtWidgets.QLineEdit()
        self.temp_cal_measure_icon.setStyleSheet(glass_icon_style)
        self.temp_cal_measure_icon.setFixedWidth(self.temp_cal_measure_icon.sizeHint().height())
        self.temp_cal_measure_icon.setReadOnly(True)
        self.temp_cal_measure_action = self.temp_cal_measure_icon.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.temp_cal_measure_input.textChanged.connect(
            lambda text, action=self.temp_cal_measure_action: self.on_text_edit(text, action)
        )
        self.temp_cal_measure_input.editingFinished.connect(
            lambda widget=self.temp_cal_measure_input: self.on_edit_finish(widget)
        )
        self.temp_cal_measure_dot = _make_dot(
            self.temp_cal_measure_action, self.temp_cal_measure_input
        )

        self.temp_cal_default = _BorderlessActionButton("Default")
        self.temp_cal_default.clicked.connect(self.on_temp_cal_default)
        self.temp_cal_save = _BorderlessActionButton("Save", tone="primary")
        self.temp_cal_save.clicked.connect(self.on_temp_cal_save)
        self.temp_cal_reset = _BorderlessActionButton("Reset")
        self.temp_cal_reset.clicked.connect(self.on_temp_cal_reset)

        # Row 2L: Lid Pogo Distance (Servo Steps) — range slider + number box.
        # The named presets dropdown has been replaced by a direct 10–50 range
        # slider paired with a live numeric box (custom values allowed anywhere
        # in range). A hidden combo is retained ONLY so any external code that
        # references ``lid_pogo_distance_combo`` keeps working.
        self.lid_pogo_distance_field = _RangeSliderField(
            10,
            50,
            suffix="",
            up_icon_path=os.path.join(Architecture.get_path(), "QATCH", "icons", "up-chevron.svg"),
            down_icon_path=os.path.join(
                Architecture.get_path(), "QATCH", "icons", "down-chevron.svg"
            ),
        )
        self.lid_pogo_distance_input = self.lid_pogo_distance_field  # string-compatible alias
        self.lid_pogo_distance_combo = AnimatedComboBox(icon_path=self._combo_chevron)
        self.lid_pogo_distance_combo.addItems(["Most", "More", "Normal", "Less", "Least", "Custom"])
        self.lid_pogo_distance_combo.setCurrentIndex(2)
        self.lid_pogo_distance_combo.hide()
        self.lid_pogo_distance_values = {
            "Most": 50,
            "More": 40,
            "Normal": 30,
            "Less": 20,
            "Least": 10,
        }
        self.lid_pogo_distance_field.setValue(30)
        self.lid_pogo_distance_action = self.lid_pogo_distance_input.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.lid_pogo_distance_field.valueChanged.connect(
            lambda v, action=self.lid_pogo_distance_action: self.on_text_edit(str(v), action)
        )
        self.lid_pogo_distance_dot = _make_dot(
            self.lid_pogo_distance_action, self.lid_pogo_distance_field
        )

        # Row 2R: Lid Pogo Delay (Servo Delay) — matching range slider 0–254 ms.
        self.lid_pogo_delay_field = _RangeSliderField(
            0,
            254,
            suffix="",
            up_icon_path=os.path.join(Architecture.get_path(), "QATCH", "icons", "up-chevron.svg"),
            down_icon_path=os.path.join(
                Architecture.get_path(), "QATCH", "icons", "down-chevron.svg"
            ),
        )
        self.lid_pogo_delay_input = self.lid_pogo_delay_field  # string-compatible alias
        self.lid_pogo_delay_combo = AnimatedComboBox(icon_path=self._combo_chevron)
        self.lid_pogo_delay_combo.addItems(
            ["Fastest", "Fast", "Normal", "Slow", "Slowest", "Custom"]
        )
        self.lid_pogo_delay_combo.setCurrentIndex(2)
        self.lid_pogo_delay_combo.hide()
        self.lid_pogo_delay_values = {
            "Fastest": 10,
            "Fast": 20,
            "Normal": 30,
            "Slow": 40,
            "Slowest": 50,
        }
        self.lid_pogo_delay_field.setValue(30)
        self.lid_pogo_delay_action = self.lid_pogo_delay_input.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.lid_pogo_delay_field.valueChanged.connect(
            lambda v, action=self.lid_pogo_delay_action: self.on_text_edit(str(v), action)
        )
        self.lid_pogo_delay_dot = _make_dot(
            self.lid_pogo_delay_action, self.lid_pogo_delay_field
        )

        self.lid_pogo_default = _BorderlessActionButton("Default")
        self.lid_pogo_default.clicked.connect(self.on_lid_pogo_default)
        self.lid_pogo_save = _BorderlessActionButton("Save", tone="primary")
        self.lid_pogo_save.clicked.connect(self.on_lid_pogo_save)
        self.lid_pogo_reset = _BorderlessActionButton("Reset")
        self.lid_pogo_reset.clicked.connect(self.on_lid_pogo_reset)

        # --- Sectioned layout (mirrors the Advanced perspective) ---
        # The device perspective uses the SAME visual language as the advanced
        # view: quiet uppercase _SectionHeader labels, hairline dividers, and
        # consistent spacing. Each editable field carries a glowing status dot
        # (gray → amber → green) reflecting its save state.
        def dev_section(title, *rows):
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
            col.addStretch()
            return col

        def dev_field_row(label_html, field_widget, dot, *extra_widgets):
            """One field row: [status dot] label : [widget(s)].

            The leading dot makes each field's save state legible at a glance.
            """
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            row.addWidget(dot, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
            lbl = QtWidgets.QLabel(label_html)
            lbl.setMinimumWidth(86)
            row.addWidget(lbl, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(field_widget, 1)
            for w in extra_widgets:
                row.addWidget(w, 0)
            return row

        def dev_btn_row(*buttons):
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 4, 0, 0)
            row.setSpacing(4)
            row.addStretch()
            for b in buttons:
                row.addWidget(b)
            return row

        # 1. Device Configuration
        device_section = dev_section(
            "Device Configuration",
            dev_field_row("Device Name:", self.device_name_input, self.device_name_dot),
            dev_field_row("Position ID:", self.device_pid_input, self.device_pid_dot),
            dev_btn_row(
                self.device_config_default,
                self.device_config_reset,
                self.device_config_save,
            ),
        )

        # 2. Temperature Calibration (ΔT with proper subscripts via rich text).
        temp_section = dev_section(
            "Temperature Calibration",
            dev_field_row(
                "\u0394T<sub>always</sub>:",
                self.temp_cal_always_input,
                self.temp_cal_always_dot,
                self.temp_cal_always_icon,
            ),
            dev_field_row(
                "\u0394T<sub>measure</sub>:",
                self.temp_cal_measure_input,
                self.temp_cal_measure_dot,
                self.temp_cal_measure_icon,
            ),
            dev_btn_row(
                self.temp_cal_default,
                self.temp_cal_reset,
                self.temp_cal_save,
            ),
        )

        # 3. Lid POGO Calibration — range sliders (full width, underneath).
        pogo_section = dev_section(
            "Lid POGO Calibration",
            dev_field_row("Servo Steps:", self.lid_pogo_distance_field, self.lid_pogo_distance_dot),
            dev_field_row("Servo Delay:", self.lid_pogo_delay_field, self.lid_pogo_delay_dot),
            dev_btn_row(
                self.lid_pogo_default,
                self.lid_pogo_reset,
                self.lid_pogo_save,
            ),
        )

        # --- Top row: Device Config + Temperature Calibration side-by-side ---
        dev_top_row = QtWidgets.QHBoxLayout()
        dev_top_row.setSpacing(22)
        dev_top_row.addLayout(device_section, 1)
        dev_top_row.addLayout(temp_section, 1)

        # --- Global action buttons (Default All / Reset All / Save All) ---
        # These operate across every section at once; the per-section buttons
        # remain for targeted edits.
        self.device_default_all = _BorderlessActionButton("Default All")
        self.device_default_all.clicked.connect(self.on_device_default_all)
        self.device_reset_all = _BorderlessActionButton("Reset All")
        self.device_reset_all.clicked.connect(self.on_device_reset_all)
        self.device_save_all = _BorderlessActionButton("Save All", tone="primary")
        self.device_save_all.clicked.connect(self.on_device_save_all)

        all_btn_row = QtWidgets.QHBoxLayout()
        all_btn_row.setContentsMargins(0, 0, 0, 0)
        all_btn_row.setSpacing(10)
        all_btn_row.addStretch()
        all_btn_row.addWidget(self.device_default_all)
        all_btn_row.addWidget(self.device_reset_all)
        all_btn_row.addWidget(self.device_save_all)

        # Final layout: header row, the two top sections, the POGO section
        # underneath (full width), a divider, then the global action buttons.
        bannerLayout = QtWidgets.QVBoxLayout(self.device_info_container)
        bannerLayout.setContentsMargins(2, 2, 2, 2)
        bannerLayout.setSpacing(12)
        bannerLayout.addLayout(header_layout)  # back + gear + title + info
        bannerLayout.addLayout(dev_top_row)
        bannerLayout.addLayout(pogo_section)
        bannerLayout.addStretch(1)
        bannerLayout.addWidget(_hairline())
        bannerLayout.addLayout(all_btn_row)

        # Hide it initially; the popup hosts and reveals it via a slide.
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
        self._sync_dot(action)

    def _sync_dot(self, action):
        """Update the glowing status dot paired with ``action`` to its state."""
        dot = getattr(self, "_field_dots", {}).get(action)
        if dot is not None:
            dot.set_state(action.iconText() or "blank")

    def _sync_all_dots(self):
        """Refresh every field dot from its action's current iconText."""
        for action, dot in getattr(self, "_field_dots", {}).items():
            dot.set_state(action.iconText() or "blank")

    def _has_unsaved_device_changes(self) -> bool:
        """True if any device field currently holds unsaved input."""
        for action in getattr(self, "_field_dots", {}):
            if action.iconText() == "unsaved":
                return True
        return False

    def _pulse_unsaved_device_fields(self) -> None:
        """Flash the dot AND the input of every field with unsaved changes.

        Used when the user tries to slide back to the advanced view while edits
        are pending, to draw attention to exactly which fields need attention.
        """
        for action, dot in getattr(self, "_field_dots", {}).items():
            if action.iconText() == "unsaved":
                dot.flash(times=3)
                widget = getattr(self, "_field_widgets", {}).get(action)
                if widget is not None:
                    self._pulse_widget_border(widget)

    def _pulse_widget_border(self, widget) -> None:
        """Briefly pulse an amber border on ``widget`` to flag unsaved input."""
        # _RangeSliderField pulses its inner spin box; everything else pulses
        # its own stylesheet border via a transient amber outline.
        target = getattr(widget, "spin", widget)
        base_qss = target.styleSheet()
        amber = (
            target.__class__.__name__ + " { border: 1px solid rgba(240,170,50,230); "
            "border-radius: 12px; }"
        )
        # Simple 3-blink using timers so we don't depend on a QSS property anim.
        def _on():
            target.setStyleSheet(base_qss + amber)

        def _off():
            target.setStyleSheet(base_qss)

        for i in range(3):
            QtCore.QTimer.singleShot(i * 320, _on)
            QtCore.QTimer.singleShot(i * 320 + 160, _off)

    def on_distance_edit(self, text, action):
        self.on_text_edit(text, action)
        if text == "Custom":
            self.lid_pogo_distance_input.setReadOnly(False)
        else:
            if text in self.lid_pogo_distance_values.keys():
                val = self.lid_pogo_distance_values[text]
            else:
                val = 30
            self.lid_pogo_distance_input.setReadOnly(True)
            self.lid_pogo_distance_input.setText(str(val))

    def on_delay_edit(self, text, action):
        self.on_text_edit(text, action)
        if text == "Custom":
            self.lid_pogo_delay_input.setReadOnly(False)
        else:
            if text in self.lid_pogo_delay_values.keys():
                val = self.lid_pogo_delay_values[text]
            else:
                val = 30
            self.lid_pogo_delay_input.setReadOnly(True)
            self.lid_pogo_delay_input.setText(str(val))

    def on_edit_finish(self, widget: QtWidgets.QDoubleSpinBox):
        return  # do nothing, no need for 0.05 rounding
        try:
            text = widget.value()
            rounded_05 = round(round(float(text) * 20) / 20, 2)
            widget.setValue(rounded_05)
        except (ValueError, TypeError):
            Log.e("Invalid input, resetting field to zero.")
            widget.setValue(0.00)

    def on_device_config_default(self):
        # ConfigBannerWidget.text() returns the raw banner string (handle-aware),
        # so the device default name is its last token: the device handle when
        # one is set, or "Device" from the base "...for Device" string otherwise.
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
        self._sync_all_dots()

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
        self._sync_all_dots()

    def on_device_config_reset(self):
        if self.device_name_action.iconText() != "saved":
            Log.d("Reset device name")
            self.device_name_input.clear()
            # self.device_name_input.setEnabled(False)
            self.device_name_input.setPlaceholderText("Querying...")
            self.device_name_action.setIcon(self.blankIcon)
            self.device_name_action.setIconText("querying")
            QtCore.QTimer.singleShot(500, self.reset_device_name_input)
        if self.device_pid_action.iconText() != "saved":
            Log.d("Reset device pid")
            self.device_pid_input.clear()
            # self.device_pid_input.setEnabled(False)
            self.device_pid_input.setPlaceholderText("Querying...")
            self.device_pid_action.setIcon(self.blankIcon)
            self.device_pid_action.setIconText("querying")
            QtCore.QTimer.singleShot(1000, self.reset_device_pid_input)
        self._sync_all_dots()

    def on_temp_cal_default(self):
        default_always = "0.00"
        default_measure = "0.00"

        if self.temp_cal_always_input.text() != default_always:
            self.temp_cal_always_input.setValue(
                self.temp_cal_always_input.valueFromText(default_always)
            )
            self.temp_cal_always_action.setIcon(self.unsavedIcon)
            self.temp_cal_always_action.setIconText("unsaved")
        if self.temp_cal_measure_input.text() != default_measure:
            self.temp_cal_measure_input.setValue(
                self.temp_cal_measure_input.valueFromText(default_measure)
            )
            self.temp_cal_measure_action.setIcon(self.unsavedIcon)
            self.temp_cal_measure_action.setIconText("unsaved")
        self._sync_all_dots()

    def on_temp_cal_save(self):
        if self.temp_cal_always_action.iconText() == "unsaved":
            if self.temp_cal_always_input.hasAcceptableInput():
                text = self.temp_cal_always_input.value()
                text = f"{text:2.02f}"
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
                text = self.temp_cal_measure_input.value()
                text = f"{text:2.02f}"
                Log.d("Save T_measure =", text)
                if self.save_temp_cal_measure_input(text):
                    self.temp_cal_measure_action.setIcon(self.savedIcon)
                    self.temp_cal_measure_action.setIconText("saved")
            else:
                Log.e(
                    f"Invalid T_measure input: {self.temp_cal_measure_input.text()} (out of valid range)"
                )
        self._sync_all_dots()

    def on_temp_cal_reset(self):
        if self.temp_cal_always_action.iconText() != "saved":
            Log.d("Reset T_always")
            self.temp_cal_always_input.clear()
            # self.temp_cal_always_input.setEnabled(False)
            # self.temp_cal_always_input.setPlaceholderText("Querying...")
            self.temp_cal_always_action.setIcon(self.blankIcon)
            self.temp_cal_always_action.setIconText("querying")
            QtCore.QTimer.singleShot(1, self.reset_temp_cal_always_input)
        if self.temp_cal_measure_action.iconText() != "saved":
            Log.d("Reset T_measure")
            self.temp_cal_measure_input.clear()
            # self.temp_cal_measure_input.setEnabled(False)
            # self.temp_cal_measure_input.setPlaceholderText("Querying...")
            self.temp_cal_measure_action.setIcon(self.blankIcon)
            self.temp_cal_measure_action.setIconText("querying")
            QtCore.QTimer.singleShot(3000, self.reset_temp_cal_measure_input)
        self._sync_all_dots()

    def on_lid_pogo_default(self):
        default_distance = str(self.lid_pogo_distance_values["Normal"])
        default_delay = str(self.lid_pogo_delay_values["Normal"])

        if self.lid_pogo_distance_input.text() != default_distance:
            self.lid_pogo_distance_input.setText(default_distance)
            self.lid_pogo_distance_action.setIcon(self.unsavedIcon)
            self.lid_pogo_distance_action.setIconText("unsaved")
        if self.lid_pogo_delay_input.text() != default_delay:
            self.lid_pogo_delay_input.setText(default_delay)
            self.lid_pogo_delay_action.setIcon(self.unsavedIcon)
            self.lid_pogo_delay_action.setIconText("unsaved")
        self._sync_all_dots()

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
        self._sync_all_dots()

    def on_lid_pogo_reset(self):
        get_lid_cal = False
        if self.lid_pogo_distance_action.iconText() != "saved":
            Log.d("Reset lid pogo distance")
            self.lid_pogo_distance_input.clear()
            # self.lid_pogo_distance_input.setEnabled(False)
            self.lid_pogo_distance_input.setPlaceholderText("...")
            self.lid_pogo_distance_action.setIcon(self.blankIcon)
            self.lid_pogo_distance_action.setIconText("querying")
            get_lid_cal = True
        if self.lid_pogo_delay_action.iconText() != "saved":
            Log.d("Reset lid pogo delay")
            self.lid_pogo_delay_input.clear()
            # self.lid_pogo_delay_input.setEnabled(False)
            self.lid_pogo_delay_input.setPlaceholderText("...")
            self.lid_pogo_delay_action.setIcon(self.blankIcon)
            self.lid_pogo_delay_action.setIconText("querying")
            get_lid_cal = True
        if get_lid_cal:
            QtCore.QTimer.singleShot(1500, self.get_lid_pogo_calibration)
        self._sync_all_dots()

    # -- Global (All) actions -------------------------------------------------
    def on_device_default_all(self):
        """Apply 'Default' to every section at once."""
        self.on_device_config_default()
        self.on_temp_cal_default()
        self.on_lid_pogo_default()
        self._sync_all_dots()

    def on_device_reset_all(self):
        """Apply 'Reset' to every section at once."""
        self.on_device_config_reset()
        self.on_temp_cal_reset()
        self.on_lid_pogo_reset()
        self._sync_all_dots()

    def on_device_save_all(self):
        """Apply 'Save' to every section at once."""
        self.on_device_config_save()
        self.on_temp_cal_save()
        self.on_lid_pogo_save()
        self._sync_all_dots()

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

        self.temp_cal_always_input.setValue(self.temp_cal_always_input.valueFromText(set_cal1))
        # self.temp_cal_always_input.setPlaceholderText(None)
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

        self.temp_cal_measure_input.setValue(self.temp_cal_measure_input.valueFromText(set_cal2))
        # self.temp_cal_measure_input.setPlaceholderText(None)
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

        if self.lid_pogo_distance_action.iconText() in ("blank", "querying"):
            idx = self.lid_pogo_distance_combo.count() - 1  # Custom
            for i, val in enumerate(self.lid_pogo_distance_values.values()):
                if val == pogo_distance:
                    idx = i
                    break
            self.lid_pogo_distance_combo.setCurrentIndex(idx)
            self.lid_pogo_distance_input.setText(str(pogo_distance))
            self.lid_pogo_distance_input.setPlaceholderText(None)
            self.lid_pogo_distance_action.setIcon(self.savedIcon)
            self.lid_pogo_distance_action.setIconText("saved")
        if self.lid_pogo_delay_action.iconText() in ("blank", "querying"):
            idx = self.lid_pogo_delay_combo.count() - 1  # Custom
            for i, val in enumerate(self.lid_pogo_delay_values.values()):
                if val == pogo_delay:
                    idx = i
                    break
            self.lid_pogo_delay_combo.setCurrentIndex(idx)
            self.lid_pogo_delay_input.setText(str(pogo_delay))
            self.lid_pogo_delay_input.setPlaceholderText(None)
            self.lid_pogo_delay_action.setIcon(self.savedIcon)
            self.lid_pogo_delay_action.setIconText("saved")
        self._sync_all_dots()

    def blank_device_config_icon_text(self):
        self.device_name_action.setIconText("blank")
        self.device_pid_action.setIconText("blank")
        self.temp_cal_always_action.setIconText("blank")
        self.temp_cal_measure_action.setIconText("blank")
        self.lid_pogo_distance_action.setIconText("blank")
        self.lid_pogo_delay_action.setIconText("blank")
        self._sync_all_dots()

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
        if unsaved_input:
            # Don't leave: pulse every field that still has unsaved changes (dot
            # flash + amber border blink) so the user sees exactly what's pending.
            self._pulse_unsaved_device_fields()
            return
        if not unsaved_input:
            # Slide the perspective back RIGHT to the advanced view within the
            # SAME popup surface (the advanced content slides back into view and
            # the device content slides off to the right). This reads as a back-
            # navigation rather than the whole popup window sliding away.
            popup = getattr(self, "_advanced_popup", None)
            if popup is not None and popup.isVisible():
                popup.show_advanced_perspective(animated=True)
            else:
                # Fallback: no live popup (e.g. opened standalone) — just hide.
                container = self.device_info_container
                host = container.window() if container is not None else None
                if host is not None:
                    host.hide()
                elif container is not None:
                    container.hide()

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
            "QLabel { "
            f"background: {bg_colour}; color: {text_colour}; "
            "border: 1px solid rgba(255, 255, 255, 160); "
            "border-radius: 3px; padding: 0 6px; font-weight: bold; }"
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

    def _install_perspective_animation(self, container: QtWidgets.QWidget) -> None:
        """Attach a fade + slide-up entrance animation that plays on each show.

        Both the advanced and device-info perspectives use this so switching
        between them (or opening either) animates in smoothly instead of
        snapping. The fade uses the popup window's opacity (not a graphics
        effect) plus a transient top content-margin slide.

        UIControls is a plain class (not a QObject), so a dedicated
        _PerspectiveAnimator QObject is installed as the container's event
        filter to detect show events.
        """
        animator = _PerspectiveAnimator(container)
        # Keep a reference so it isn't garbage-collected.
        self._perspective_animators = getattr(self, "_perspective_animators", [])
        self._perspective_animators.append(animator)

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
        self.infobar_readout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignLeft
        )

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

    def _ensure_advanced_popup(self):
        """Return the live advanced popup, opening it (on the advanced view) if needed."""
        popup = getattr(self, "_advanced_popup", None)
        if popup is not None and popup.isVisible():
            return popup
        main_window = getattr(self, "parent", None)
        popup = AdvancedMainWidget.toggle(
            owner=self,
            anchor=self.tool_Advanced,
            controls_layout=self._advanced_controls_layout,
            main_window=main_window,
        )
        if popup is not None:
            self.advanced_container = popup.content_container
        return popup

    def show_device_config_editor(self):
        """Open the advanced popup (if needed) and slide to the device view.

        This is the entry point external code (e.g. mainWindow) calls when the
        user chooses to configure the connected device. The device-config
        perspective lives as the second page inside the advanced popup, so the
        advanced content slides left and the device content slides into view as
        part of the same glass surface.
        """
        popup = self._ensure_advanced_popup()
        if popup is None:
            return None
        # Make sure the device perspective is registered as the second page.
        if self.device_info_container is not None:
            popup.set_device_perspective(self.device_info_container)
        popup.show_device_perspective(animated=True)
        return popup

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

    def _update_configure_enabled(self, *args):
        """Reflect device-connection state on the Configure button.

        With no device connected there is nothing to configure, so the button
        is disabled and its tooltip reads 'No device connected' instead of the
        normal configure prompt (it previously looked clickable but did
        nothing). When a device is present it is re-enabled with the usual
        tooltip.

        Counts only real device ports: any legacy 'Configure...' sentinel item
        (itemData == 'CMD_DEV_INFO') is excluded so its presence does not make
        an empty port list look connected.
        """
        if not hasattr(self, "pButton_Configure"):
            return
        real_ports = 0
        for i in range(self.cBox_Port.count()):
            if self.cBox_Port.itemData(i) != "CMD_DEV_INFO":
                real_ports += 1
        connected = real_ports > 0
        self.pButton_Configure.setEnabled(connected)
        self.pButton_Configure.setToolTip(
            "Configure device / position info" if connected else "No device connected"
        )

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