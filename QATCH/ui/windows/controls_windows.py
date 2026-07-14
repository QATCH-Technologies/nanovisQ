"""controls_windows.py

Defines `ControlsWindow`, the top-level `QMainWindow` that hosts the QATCH
control bar and owns all application-level menus, user session management, and
view-toggle logic.

`ControlsWindow` composes `UIControls` via the Qt Designer mixin pattern:
`UIControls.setup_ui` is called once during construction to build the control
bar widgets, and the remaining methods on this class implement the menu actions,
session workflow, and layout helpers that sit above the widget layer.

Responsibilities:
    - Menu bar construction (Options / Users / View / Help) and dynamic theming
      to match the glass design language and dim-while-signed-out state.
    - User session lifecycle: sign-in, sign-out, profile management, and role-
      gated menu access via `set_signed_in_menu_state`.
    - Data management overlay access for analyze, import, export, and recover
      workflows via `DataManagementWidget`.
    - View toggles for the console, amplitude, temperature, and
      resonance/dissipation panels, with persistence to `AppSettings`.
    - OS-native file opening for release notes, changelogs, license, and user
      guide documents.
    - Software update checks with contextual pop-up feedback.
    - Close-event interception with a user confirmation prompt.

Author(s):
    Alexander J. Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-07-01
"""

import os
import subprocess
import sys
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Optional, cast

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture, OSType
from QATCH.common.deviceFingerprint import DeviceFingerprint
from QATCH.common.findDevices import Discovery
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserProfiles
from QATCH.core.constants import Constants, UserRoles
from QATCH.ui.dialogs.pop_up_dialog import PopUp
from QATCH.ui.interfaces import UIControls
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css
from QATCH.ui.widgets import (
    DataManagementWidget,
    UserPreferencesWidget,
    UserProfilesManagerWidget,
)
from QATCH.ui.windows.base_window import BaseWindow

if TYPE_CHECKING:
    from QATCH.ui.main_window import MainWindow

TAG = "[ControlsWindow]"


class ControlsWindow(BaseWindow):
    def __init__(
        self,
        parent: "MainWindow",
        samples: int = Constants.argument_default_samples,
    ) -> None:
        """Initializes the main widget and its child components.

        Sets up the user interface controls, user preferences, and initializes
        the internal timer. Also ensures any previous user sessions are ended.

        Args:
            parent: The parent widget or window that owns this instance.
            samples: The initial sample data or configuration. Defaults to
                `Constants.argument_default_samples`.

        Attributes:k
            parent (MainWindow): Reference to the parent UI component.
            ui (UIControls): The main user interface controls object.
            data_management_widget (Optional[DataManagementWidget]): The widget
                handling data management operations. Initialized as None.
            ui_preferences (UserPreferencesWidget): The widget managing user
                settings and preferences.
            current_timer (QtCore.QTimer): Timer for handling timed events or
                refresh loops within the UI.
        """
        self.parent: "MainWindow" = parent
        super().__init__()

        self.ui: UIControls = UIControls()
        self.ui.setup_ui(self)

        self.data_management_widget: Optional[Any] = None
        # Lazily created on first open (see preferences()) - same reasoning
        # as data_management_widget: it needs to be parented to the full
        # app window's central widget, which may not exist yet this early
        # in startup.
        self.ui_preferences: Optional[UserPreferencesWidget] = None
        self.current_timer: QtCore.QTimer = QtCore.QTimer()

        UserProfiles().session_end()

    def _create_menu(self, target: QtWidgets.QMainWindow) -> None:
        """Constructs and configures the application's top-level menu bar.

        This method builds the Options, Users, View, and Help menus, populates
        them with their respective actions, and connects them to internal methods.
        It also restores user view preferences from application settings and applies
        custom visual styling to the menus.

        Args:
            target (QtWidgets.QMainWindow): The window or widget that
                owns and displays the menu bar.

        Attributes:
            _menu_target (Any): Reference to the target hosting the menu.
            menubar (List[QtWidgets.QMenu]): Collection of the primary top-level
                menus added to the target's menu bar.
            act_analyze_data (QtWidgets.QAction): Action to trigger data analysis.
            act_import_data (QtWidgets.QAction): Action to import data files.
            act_export_data (QtWidgets.QAction): Action to export data files.
            act_recover_data (QtWidgets.QAction): Action to recover data.
            act_preferences (QtWidgets.QAction): Action to open user preferences.
            act_find_devices (QtWidgets.QAction): Action to scan subnets for devices.
            username (QtWidgets.QAction): Display action for the current user.
            signinout (QtWidgets.QAction): Action to toggle user sign in/out.
            act_select_directory (QtWidgets.QAction): Action to set working directory.
            manage (QtWidgets.QAction): Action to open the user profile manager.
            userrole (int/Enum): The current user's role/permission level.
            modebar (QtWidgets.QMenu): Sub-menu for selecting application modes.
            chk1 (QtWidgets.QAction): Toggle action for the Console view.
            chk2 (QtWidgets.QAction): Toggle action for the Amplitude view.
            chk3 (QtWidgets.QAction): Toggle action for the Temperature view.
            chk4 (QtWidgets.QAction): Toggle action for the Resonance/Dissipation view.
            chk5 (QtWidgets.QAction): Action to view tutorials.
            act_check_updates (QtWidgets.QAction): Action to check for software updates.
            qmodel_tweed_version (QtWidgets.QAction): Toggle for QModel Tweed prediction model.
            qmodel_indus_version (QtWidgets.QAction): Toggle for QModel Indus prediction model.
            q_version_volta (QtWidgets.QAction): Toggle for QModel Volta prediction model.
            q_version_onyx (QtWidgets.QAction): Toggle for QModel Onyx prediction model.
        """
        self._menu_target = target  # real menu bar lives on MainWin, not here
        self.menubar = []
        menu_bar = target.menuBar()
        if menu_bar is None:
            Log.e(TAG, "Error mounting menubar to main window.  Menubar is `None`")
            raise
        self.menubar.append(menu_bar.addMenu("&Options"))
        self.act_analyze_data = self.menubar[0].addAction("&Analyze Data", self.analyze_data)
        self.act_import_data = self.menubar[0].addAction("&Import Data", self.import_data)
        self.act_export_data = self.menubar[0].addAction("&Export Data", self.export_data)
        self.act_recover_data = self.menubar[0].addAction("&Recover Data", self.recover_data)
        self.act_preferences = self.menubar[0].addAction("&Preferences", self.preferences)
        self.act_find_devices = self.menubar[0].addAction("&Find Devices", self.scan_subnets)
        self.menubar[0].addAction("E&xit", self.close)
        self.menubar.append(menu_bar.addMenu("&Users"))
        self.username = self.menubar[1].addAction("User: [NONE]")
        self.username.setEnabled(False)
        self.signinout = self.menubar[1].addAction("&Sign In", self.set_user_profile)
        self.act_select_directory = self.menubar[1].addAction(
            "Select &directory...", self.set_working_directory
        )
        self.manage = self.menubar[1].addAction("&Manage Users...", self.manage_user_profiles)
        self.userrole = UserRoles.NONE
        self.menubar.append(menu_bar.addMenu("&View"))
        self.modebar = self.menubar[2].addMenu("&Mode")
        self.modebar.addAction("&1: Run", lambda: self.parent.mode_window.ui._set_run_mode(None))
        self.modebar.addAction(
            "&2: Analyze", lambda: self.parent.mode_window.ui._set_analyze_mode(None)
        )
        if Constants.show_visQ_in_R_builds:
            self.modebar.addAction(
                "&3: VisQ.AI", lambda: self.parent.mode_window.ui._set_learn_mode(None)
            )
        self.chk1 = self.menubar[2].addAction("&Console", self.toggle_console)
        self.chk1.setCheckable(True)
        self.chk1.setChecked(
            self.parent.app_settings.value("viewState_Console", "True").lower() == "true"
        )
        self.chk2 = self.menubar[2].addAction("&Amplitude", self.toggle_amplitude)
        self.chk2.setCheckable(True)
        self.chk2.setChecked(
            self.parent.app_settings.value("viewState_Amplitude", "True").lower() == "true"
        )
        self.chk3 = self.menubar[2].addAction("&Temperature", self.toggle_temperature)
        self.chk3.setCheckable(True)
        self.chk3.setChecked(
            self.parent.app_settings.value("viewState_Temperature", "True").lower() == "true"
        )
        self.chk4 = self.menubar[2].addAction(
            "&Resonance/Dissipation", self.toggle_resonance_dissipation
        )
        self.chk4.setCheckable(True)
        self.chk4.setChecked(
            self.parent.app_settings.value("viewState_Resonance_Dissipation", "True").lower()
            == "true"
        )
        self.menubar.append(menu_bar.addMenu("&Help"))
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
        from QATCH.QModel.models.qmodel_indus.__init__ import (
            __release__ as qmodel4_release,
        )
        from QATCH.QModel.models.qmodel_indus.__init__ import (
            __version__ as qmodel4_version,
        )
        from QATCH.QModel.models.qmodel_onyx.__init__ import (
            __release__ as qmodel7_release,
        )
        from QATCH.QModel.models.qmodel_onyx.__init__ import (
            __version__ as qmodel7_version,
        )
        from QATCH.QModel.models.qmodel_tweed.tweed import (
            __release__ as model_data_release,
        )
        from QATCH.QModel.models.qmodel_tweed.tweed import (
            __version__ as model_data_version,
        )
        from QATCH.QModel.models.qmodel_volta.__init__ import (
            __release__ as qmodel6_release,
        )
        from QATCH.QModel.models.qmodel_volta.__init__ import (
            __version__ as qmodel6_version,
        )

        qmodel_versions_menu = self.menubar[3].addMenu("Model versions (4 available)")
        self.menubar.append(qmodel_versions_menu)
        self.qmodel_tweed_version = self.menubar[5].addAction(
            "Tweed v{} ({})".format(
                ".".join(str(model_data_version).split(".")[:2]), model_data_release
            ),
            lambda: self.parent.analyze_window.ui.set_new_prediction_model(
                Constants.list_predict_models[0]
            ),
        )
        self.qmodel_tweed_version.setCheckable(True)

        self.qmodel_indus_version = self.menubar[5].addAction(
            "Indus v{} ({})".format(".".join(str(qmodel4_version).split(".")[:2]), qmodel4_release),
            lambda: self.parent.analyze_window.ui.set_new_prediction_model(
                Constants.list_predict_models[1]
            ),
        )
        self.qmodel_indus_version.setCheckable(True)

        self.qmodel_volta_version = self.menubar[5].addAction(
            "Volta v{} ({})".format(".".join(str(qmodel6_version).split(".")[:2]), qmodel6_release),
            lambda: self.parent.analyze_window.ui.set_new_prediction_model(
                Constants.list_predict_models[2]
            ),
        )
        self.qmodel_volta_version.setCheckable(True)

        self.qmodel_onyx_version = self.menubar[5].addAction(
            "Onyx v{} ({})".format(".".join(str(qmodel7_version).split(".")[:2]), qmodel7_release),
            lambda: self.parent.analyze_window.ui.set_new_prediction_model(
                Constants.list_predict_models[3]
            ),
        )
        self.qmodel_onyx_version.setCheckable(True)
        self.qmodel_onyx_version.setCheckable(True)
        if Constants.qmodel_onyx_predict:
            self.qmodel_onyx_version.setChecked(True)
        elif Constants.qmodel_volta_predict:
            self.qmodel_volta_version.setChecked(True)
        elif Constants.qmodel_indus_predict:
            self.qmodel_indus_version.setChecked(True)
        elif Constants.qmodel_tweed_predict:
            self.qmodel_tweed_version.setChecked(True)
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
                lambda: cast(QtGui.QClipboard, QtWidgets.QApplication.clipboard()).setText(
                    fingerprint_txt
                )
            )

        # update application UI states to reflect viewStates from AppSettings
        if not self.chk1.isChecked():
            QtCore.QTimer.singleShot(100, self.toggle_console)
        if not self.chk2.isChecked():
            QtCore.QTimer.singleShot(100, self.toggle_amplitude)
        if not self.chk3.isChecked():
            QtCore.QTimer.singleShot(100, self.toggle_temperature)
        if not self.chk4.isChecked():
            QtCore.QTimer.singleShot(100, self.toggle_resonance_dissipation)

        # Re-apply the menu bar theme whenever light/dark mode changes.
        ThemeManager.instance().themeChanged.connect(
            lambda _: self._apply_menu_bar_theme(getattr(self, "_signed_in_state", True))
        )

        for menu in (*self.menubar, self.modebar):
            menu.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, False)
            menu.setWindowFlag(QtCore.Qt.WindowType.NoDropShadowWindowHint, True)
            menu.aboutToShow.connect(lambda m=menu: self._mask_menu_to_rounded_rect(m))

    @staticmethod
    def _mask_menu_to_rounded_rect(menu: QtWidgets.QMenu, radius: int = 10) -> None:
        """Applies a rounded-rectangle mask to a menu after it is rendered.

        Because a widget's geometry is often calculated or adjusted by the window
        manager after the `aboutToShow` event, this method defers the masking
        operation until the next event loop iteration. This ensures the menu
        has achieved its final dimensions before the clipping region is applied.

        Args:
            menu (QtWidgets.QMenu): The menu widget to which the rounded
                mask will be applied.
            radius (int, optional): The corner radius of the mask in pixels.
                Defaults to 10.
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

        # Users: fully disabled while signed out.
        self.signinout.setEnabled(signed_in)
        self.act_select_directory.setEnabled(signed_in)
        self.manage.setEnabled(signed_in)

        # View: fully disabled while signed out.
        self.modebar.menuAction().setEnabled(signed_in)
        self.chk1.setEnabled(signed_in)
        self.chk2.setEnabled(signed_in)
        self.chk3.setEnabled(signed_in)
        self.chk4.setEnabled(signed_in)

        # Help: enabled regardless, except updates and model-version switching.
        self.act_check_updates.setEnabled(signed_in)
        self.menubar[5].menuAction().setEnabled(signed_in)  # "Model versions" submenu
        self._apply_menu_bar_theme(signed_in)
        self.ui.refresh_user_button_state()

    def _apply_menu_bar_theme(self, signed_in: bool) -> None:
        """Tints the native menu bar and its dropdowns to match the glass
        theme instead of the stark native default. While the sign-in gate
        is up, this applies the same ~30% dimming the login overlay
        applies to the blurred dashboard to the bar's own light tone - not
        an unrelated dark color - so it reads as part of the same dimmed
        scene. The normal light/glass tone returns once signed in.

        QMenu (the dropdowns) is styled in the *same* stylesheet as
        QMenuBar - a QMenu created via addMenu() is a logical child of the
        bar in Qt's object tree, so it inherits this stylesheet too, even
        though it pops up as its own top-level window.

        Args:
            signed_in (bool): Which palette to apply.
        """
        target: QtWidgets.QMainWindow | None = getattr(self, "_menu_target", None)
        if target is None:
            return

        menu_bar = target.menuBar()
        if menu_bar is None:
            return

        self._signed_in_state = signed_in
        tok = ThemeManager.instance().tokens()
        if signed_in:
            bg = tok_css(tok["menubar_bg"])
            text = tok_css(tok["menubar_text"])
            hover = tok_css(tok["menubar_item_hover_bg"])
            disabled = tok_css(tok["menubar_item_disabled_text"])
            border = tok_css(tok["menubar_border"])
            sep = tok_css(tok["menubar_separator"])
        else:
            bg = tok_css(tok["menubar_dim_bg"])
            text = tok_css(tok["menubar_dim_text"])
            hover = tok_css(tok["menubar_dim_item_hover_bg"])
            disabled = tok_css(tok["menubar_dim_item_disabled_text"])
            border = tok_css(tok["menubar_dim_border"])
            sep = tok_css(tok["menubar_dim_separator"])

        menu_bar.setStyleSheet(f"""
            QMenuBar {{
                background: {bg};
                color: {text};
                border: none;
            }}
            QMenuBar::item {{ background: transparent; padding: 4px 10px; }}
            QMenuBar::item:selected {{ background: {hover}; border-radius: 4px; }}
            QMenuBar::item:disabled {{ color: {disabled}; }}

            QMenu {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 10px;
                padding: 6px;
            }}
            QMenu::item {{
                background: transparent;
                color: {text};
                padding: 6px 26px 6px 14px;
                border-radius: 6px;
            }}
            QMenu::item:selected {{ background: {hover}; }}
            QMenu::item:disabled {{ color: {disabled}; }}
            QMenu::separator {{
                height: 1px;
                background: {sep};
                margin: 4px 10px;
            }}
            QMenu::indicator {{ width: 14px; height: 14px; }}
        """)

    def _data_overlay_parent(self) -> QtWidgets.QWidget:
        """Determines the appropriate parent for UI data overlays.

        This ensures overlays are anchored to the full application window
        (`MainWin`) rather than the controls window, preventing the overlay
        from being clipped or restricted to the controls bar area.

        Returns:
            QtWidgets.QWidget: The central widget of the main window if available,
                otherwise the main window itself or the current object.
        """
        if hasattr(self.parent, "mode_window"):
            return self.parent.mode_window.centralWidget() or self.parent.mode_window
        return self.centralWidget() or self

    def _open_data_management(self, mode: str) -> None:
        """Initializes and displays the data management widget overlay.

        This method ensures the data management widget is properly parented
        to the main application window. It recreates the widget if it has not
        been initialized or if the parent has changed. It also pre-sizes the
        widget to match the parent's geometry before opening the specified mode
        to prevent visual "snapping" or resizing artifacts.

        Args:
            mode (str): The operational mode to initialize within the
                `DataManagementWidget`.
        """
        parent = self._data_overlay_parent()
        if self.data_management_widget is None or self.data_management_widget.parent is not parent:
            self.data_management_widget = DataManagementWidget(parent=parent)
        with suppress(Exception):
            self.data_management_widget.setGeometry(parent.rect())
        self.data_management_widget.open_mode(mode)

    def analyze_data(self) -> None:
        """Sets the application to analysis mode.

        Triggers the main UI controller to switch the workspace to the analysis
        view, allowing the user to inspect and interpret processed data.
        """
        self.parent.mode_window.ui._set_analyze_mode(self)

    def import_data(self) -> None:
        """Opens the data management interface in 'import' mode."""
        self._open_data_management("import")

    def export_data(self) -> None:
        """Opens the data management interface in 'export' mode."""
        self._open_data_management("export")

    def recover_data(self) -> None:
        """Opens the data management interface in 'recover' mode."""
        self._open_data_management("recover")

    def _ensure_preferences_widget(self) -> UserPreferencesWidget:
        """Lazily creates (or re-parents) the user preferences overlay.

        Mirrors `_open_data_management`'s lazy-create-and-cache pattern so
        the overlay is parented to the full application window instead of
        just the controls bar, and re-fitted to the current window size
        before use (in case the app window resized since it was created).
        Shared by `preferences()` (visible open) and `set_working_directory()`
        (headless use of the same widget to drive a directory-picker + save).
        """
        parent = self._data_overlay_parent()
        if self.ui_preferences is None or self.ui_preferences.parent is not parent:
            self.ui_preferences = UserPreferencesWidget(self, parent=parent)
        with suppress(Exception):
            self.ui_preferences.setGeometry(parent.rect())
        return self.ui_preferences

    def preferences(self) -> None:
        """Displays the user preferences overlay, creating it on first use."""
        self._ensure_preferences_widget().showNormal(0)

    def scan_subnets(self) -> None:
        """Initiates a network scan for connected devices and refreshes the port list.

        This method triggers the discovery service to identify active hardware on
        subnets and subsequently updates the parent application's view of available
        communication ports.
        """
        Discovery().scanSubnets()
        self.parent._port_list_refresh()

    def set_working_directory(self) -> None:
        """Prompts the user to select a new working directory and updates preferences.

        This method synchronizes global preferences, enforces a read/write
        synchronization state, and opens a file dialog for directory selection.
        If a valid selection is made and the interface is ready, it triggers
        an automatic save by programmatically clicking the submit button.
        """
        # Loads global/user preferences
        prefs = self._ensure_preferences_widget()
        prefs.toggle_global_preferences()
        prefs.sync_write_with_load.setChecked(True)
        result = prefs.open_load_file_dialog()
        if result and prefs.submit_button.isEnabled():
            prefs.submit_button.click()
        else:
            Log.w(TAG, "Working directory not changed.")

    def set_user_profile(self) -> None:
        """Toggles the application user session state between signed-in and signed-out.

        This method manages the transition between authentication states:
        - If signing in: Prompts the user to select a profile. If no profiles exist,
        it redirects to the user management interface. Updates UI elements and
        permissions upon a successful login.
        - If signing out: Ensures the application state allows for sign-out (e.g.,
        no unsaved changes in Analyze mode), resets user-specific UI labels,
        and clears the current session.

        Attributes:
            userrole (UserRoles): Updated based on the authenticated user's role.
        """
        action = self.signinout.text().lower().replace("&", "")
        if action == "sign in":
            # Handle first-time login where no profiles exist
            if UserProfiles().count() == 0:
                self.manage_user_profiles()
                return
            name, init, role = UserProfiles.change()
            if name is not None:
                self.username.setText(f"User: {name}")
                self.userrole = UserRoles(role)
                self.signinout.setText("&Sign Out")
                self.ui.tool_User.setText(name)
                self.parent.analyze_window.ui.tool_User.setText(name)

                # Update management action context
                if self.userrole != UserRoles.ADMIN:
                    self.manage.setText("&Change Password...")
        else:  # Action is "Sign Out"
            if self.parent.mode_window.ui._set_no_user_mode(None):
                UserProfiles().session_end()
                name = self.username.text()[6:]
                Log.i(f"Goodbye, {name}! You have been signed out.")

                # Reset UI to anonymous state
                self.username.setText("User: [NONE]")
                self.userrole = UserRoles.NONE
                self.signinout.setText("&Sign In")
                self.manage.setText("&Manage Users...")
                self.ui.tool_User.setText("Anonymous")
                self.parent.analyze_window.ui.tool_User.setText("Anonymous")
            else:
                Log.d("User has unsaved changes in Analyze mode. Sign out aborted.")

    def manage_user_profiles(self):
        """Handles the user profile management workflow.

        This method verifies if the application state allows for user modifications
        (i.e., checking for unsaved changes). It handles two primary flows:
        password changes for non-admin users, and the full management interface
        for administrators. It also manages the transition of the application UI
        state if a user is modified or removed during the management session.

        Attributes:
            user_manager (UserProfilesManagerWidget): The overlay
                widget used for managing user profiles.
        """
        # Disallow user management if the current mode is busy or has unsaved changes
        if not self.parent.mode_window.ui._check_mode_change_allowed():
            Log.d("User has unsaved changes in Analyze mode. Manage users aborted.")
            return

        # Handle password change for non-admin users
        if self.userrole != UserRoles.ADMIN and self.userrole != UserRoles.NONE:
            name = self.username.text()[6:]
            found, filename = UserProfiles.find(name, None)
            if filename is not None:
                UserProfiles.change_password(filename)
                return
            else:
                Log.e("Attempted to change password, but user was not found!")

        name = self.username.text()[6:]
        allow, admin = UserProfiles().manage(name, self.userrole)

        # Handle session cleanup if user was deleted
        if admin is None and not UserProfiles.session_info()[0]:
            if name != "[NONE]":
                Log.i(f"Goodbye, {name}! You have been signed out.")
            self.username.setText("User: [NONE]")
            self.userrole = UserRoles.NONE
            self.signinout.setText("&Sign In")
            self.manage.setText("&Manage Users...")
            self.ui.tool_User.setText("Anonymous")
            self.parent.analyze_window.ui.tool_User.setText("Anonymous")
            self.parent.mode_window.ui._set_no_user_mode(None)

        # Update UI if user information changed
        if admin != name and admin is not None:
            Log.d("User name changed. Changing sign-in user info.")
            self.username.setText(f"User: {admin}")
            self.userrole = UserRoles.ADMIN
            self.signinout.setText("&Sign Out")
            self.manage.setText("&Manage Users...")
            self.ui.tool_User.setText(admin)
            self.parent.analyze_window.ui.tool_User.setText(admin)

        # Display the management overlay
        if allow:
            if hasattr(self.parent, "mode_window"):
                overlay_parent = self.parent.mode_window.centralWidget() or self.parent.mode_window
            else:
                overlay_parent = self.centralWidget() or self
            self.user_manager = UserProfilesManagerWidget(parent=overlay_parent, admin_name=admin)
            with suppress(Exception):
                self.user_manager.setGeometry(overlay_parent.rect())
            self.user_manager.show()

    def toggle_console(self) -> None:
        """Toggles the visibility of the application console/log view.

        This method updates the visibility state of the console widget based on
        the checkbox state, stops any active background timers to prevent
        rendering conflicts during the toggle, and persists the user's view
        preference to the application settings.
        """
        if self.current_timer.isActive():
            self.current_timer.stop()

        is_visible: bool = self.chk1.isChecked()

        if not is_visible:
            self.parent.mode_window.ui.logview.setVisible(False)
        else:
            self.parent.mode_window.ui.logview.setVisible(True)
        self.parent.app_settings.setValue("viewState_Console", is_visible)

    def toggle_amplitude(self) -> None:
        """Toggles the visibility of the Amplitude plots.

        This method manages the display state of a collection of amplitude plots.
        It synchronizes the `setVisible` state across the plot array, handles
        the visibility of top-level plot containers, and persists the user's
        view preference to the application settings.
        """
        # Capture state of top-level plot containers
        tc = self.show_top_plot()
        is_visible: bool = self.chk2.isChecked()
        # Update visibility for each plot in the array
        for i, p in enumerate(self.parent._plt0_arr):
            if p is None:
                continue
            p.setVisible(is_visible)
            self.parent._plt0_arr[i] = p

        # Restore container layout
        self.hide_top_plot(tc)
        self.parent.app_settings.setValue("viewState_Amplitude", is_visible)

    def toggle_temperature(self) -> None:
        """Toggles the visibility of the temperature plot.

        This method manages the display state of the temperature visualization widget.
        It wraps the toggle logic with top-plot container handlers to ensure layout
        consistency and persists the user's preference to application settings.
        """
        # Capture state of top-level plot containers
        tc = self.show_top_plot()
        is_visible: bool = self.chk3.isChecked()
        if self.parent._plt4 is not None:
            self.parent._plt4.setVisible(is_visible)
        else:
            Log.e(TAG, "Cannot toggle temperature plot, temperature plot is `None`.")
        # Restore container layout
        self.hide_top_plot(tc)
        self.parent.app_settings.setValue("viewState_Temperature", is_visible)

    def toggle_resonance_dissipation(self) -> None:
        """Toggles the visibility of the Resonance/Dissipation plots.

        This method manages the display state of the Resonance/Dissipation
        visualization widget. It stops any active background timers to avoid
        rendering conflicts, updates the visibility of the plot component,
        and persists the user's view preference to application settings.
        """
        # Ensure background tasks are paused to avoid UI conflicts during resize/hide
        if self.current_timer.isActive():
            self.current_timer.stop()

        is_visible: bool = self.chk4.isChecked()

        if not is_visible:
            self.parent.plots_window.ui.pltB.setVisible(False)
        else:
            self.parent.plots_window.ui.pltB.setVisible(True)
        self.parent.app_settings.setValue("viewState_Resonance_Dissipation", is_visible)

    def show_top_plot(self) -> bool:
        """Ensures the top-level plot container is visible.

        This method checks if any amplitude or temperature plots are enabled. If
        they are, it ensures the primary plot container (`plt`) is visible. It
        returns a flag indicating whether it had to manually toggle the visibility,
        which is used to restore the original state later.

        Returns:
            bool: True if the plot container was previously hidden and was
                manually enabled by this method, False otherwise.
        """
        toggle_console: bool = False

        # Check if any associated plots require the top container to be visible
        if self.chk2.isChecked() or self.chk3.isChecked():
            toggle_console = self.parent.plots_window.ui.plt.isVisible() is False
            self.parent.plots_window.ui.plt.setVisible(True)
        return toggle_console

    def hide_top_plot(self, toggle_console: bool) -> None:
        """Restores the visibility state of the top-level plot container.

        This method complements `show_top_plot`. If the container was toggled
        visible to accommodate a child plot, this method checks if that container
        can now be hidden, or triggers a layout update to ensure the UI remains
        properly aligned after a visibility change.

        Args:
            toggle_console (bool): A flag indicating whether this method was
                responsible for originally showing the plot container.
        """
        if self.chk2.isChecked() or self.chk3.isChecked():
            if toggle_console:
                layout = self.parent.plots_window.layout()
                if layout is not None:
                    layout.activate()
        else:
            self.parent.plots_window.ui.plt.setVisible(False)

    def view_tutorials(self) -> None:
        """Toggles the visibility of the tutorials window.

        This method synchronizes the visibility state of the tutorial window
        with its current display status and updates the associated menu checkbox
        to reflect whether the window is currently visible to the user.
        """
        # Toggle the visibility of the tutorials window
        is_visible: bool = self.parent.TutorialWin.isVisible()
        self.parent.TutorialWin.setVisible(not is_visible)
        self.chk5.setChecked(self.parent.TutorialWin.isVisible())

    def open_file(self, filepath: str, relative_to_cwd: bool = True) -> None:
        """Opens a file or directory using the operating system's default handler.

        This method detects the current host OS and delegates the file opening
        process to the appropriate system command.

        Args:
            filepath (str): The path to the file or directory to be opened.
            relative_to_cwd (bool, optional): If True, joins the path with the
                architecture's base directory. Defaults to True.
        """
        fullpath: str = filepath
        try:
            if relative_to_cwd:
                fullpath = os.path.join(Architecture.get_path(), filepath)
            os_type = Architecture.get_os()
            if os_type == OSType.macosx:  # macOS
                subprocess.call(("open", fullpath))
            elif os_type == OSType.windows:  # Windows
                os.startfile(fullpath)
            elif os_type == OSType.linux:  # Linux
                subprocess.call(("xdg-open", fullpath))
            else:  # Fallback for unknown variants
                Log.w(f"Unknown OS Type: {os_type}")
                Log.w("Assuming Linux variant...")
                subprocess.call(("xdg-open", fullpath))
        except Exception as e:
            filename = os.path.split(fullpath)[1]
            Log.e(TAG, f'ERROR: Cannot open "{filename}": {str(e)}')

    def release_notes(self) -> None:
        """Opens the PDF release notes for the current application version."""
        rn_path = os.path.join("docs", f"Release Notes {Constants.app_version}.pdf")
        self.open_file(rn_path)

    def fw_change_log(self) -> None:
        """Opens the firmware change control document for the active firmware version."""
        fw_change_log_path = os.path.join(
            f"QATCH_Q-1_FW_py_{Constants.best_fw_version}", "FW Change Control Doc.pdf"
        )
        self.open_file(fw_change_log_path)

    def sw_change_log(self) -> None:
        """Opens the software change control document."""
        sw_change_log_path = os.path.join("QATCH", "SW Change Control Doc.pdf")
        self.open_file(sw_change_log_path)

    def view_license(self) -> None:
        """Opens the GPL license and the primary application LICENSE file.

        This method sequentially triggers the operating system's default
        handler to open both the GPL text file and the application-specific
        LICENSE document.
        """
        gpl_path = os.path.join("docs", "gpl.txt")
        license_path = os.path.join("docs", "LICENSE.txt")
        self.open_file(gpl_path)
        self.open_file(license_path)

    def view_user_guide(self):
        """Opens the application user guide in the default PDF viewer."""
        user_guide_path = os.path.join("docs", "userguide.pdf")
        self.open_file(user_guide_path)

    def check_for_updates(self) -> None:
        """Initiates an application update check.

        This method clears existing download URLs, triggers a license refresh
        if the license manager is available, and executes a download check
        via the parent application. It then evaluates the status returned by the
        parent and displays appropriate feedback to the user via pop-up notifications.
        """
        # Clean up existing download context
        if hasattr(self.parent, "url_download"):
            delattr(self.parent, "url_download")

        # Refresh license status if applicable
        if hasattr(self.parent, "_license_manager"):
            lm: Any = self.parent._license_manager
            if hasattr(lm, "refresh_license") and callable(lm.refresh_license):
                lm.refresh_license()

        # Initiate update check
        color, status = self.parent.start_download(True)

        # Handle error scenarios
        if color == "#ff0000":
            if status == "ERROR":
                PopUp.warning(self, "Check for Updates", "An error occurred checking for updates.")
            elif status == "OFFLINE":
                PopUp.warning(self, "Check for Updates", "Unable to check online for updates.")

        # Handle up-to-date scenarios (non-error, non-success-update colors)
        elif color != "#00ff00":
            technicality: str = " available " if color == "#00c600" else " supported "
            PopUp.information(
                self,
                "Check for Updates",
                f"You are running the latest{technicality}version.",
            )
