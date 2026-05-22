import os
import sys
import hashlib
import datetime as dt
from xml.dom import minidom
from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.logger import Logger as Log
from QATCH.common.fileManager import FileManager
from QATCH.core.constants import Constants
from QATCH.ui.popUp import PopUp
from QATCH.common.userProfiles import UserProfiles, UserRoles, UserConstants
from QATCH.common.architecture import Architecture

TAG = "[UserProfilesManager]"


class UserProfilesManagerWidget(QtWidgets.QWidget):

    class TableView(QtWidgets.QTableWidget):
        def __init__(self, *args):
            super(UserProfilesManagerWidget.TableView, self).__init__(*args)
            self.setObjectName("userProfilesTable")
            self.setShowGrid(False)
            self.verticalHeader().setVisible(False)
            self.setAlternatingRowColors(True)
            self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
            self.setFrameShape(QtWidgets.QFrame.NoFrame)

            self.setStyleSheet("""
                QTableWidget {
                    background-color: transparent;
                    border: none;
                    gridline-color: transparent;
                }
                QTableWidget::item {
                    padding: 6px;
                    border-bottom: 1px solid rgba(255, 255, 255, 90);
                }
                QTableWidget::item:selected {
                    background-color: rgba(10, 163, 230, 40); 
                    color: black;
                }
                QTableWidget::item:alternate { 
                    background-color: rgba(255, 255, 255, 50); 
                }
                QHeaderView::section {
                    background-color: rgba(255, 255, 255, 120);
                    padding: 10px;
                    border: none;
                    border-bottom: 1px solid rgba(255, 255, 255, 220);
                    border-right: 1px solid rgba(255, 255, 255, 150);
                    font-weight: bold;
                    color: #333;
                }
                QHeaderView::section:hover { 
                    background-color: rgba(255, 255, 255, 180); 
                }
            """)

    def __init__(self, parent=None, admin_name=None):
        super(UserProfilesManagerWidget, self).__init__(parent)

        # ==========================================
        # 🔧 INSERT YOUR CUSTOM ICON PATHS HERE
        # ==========================================
        self.ICON_ADD = os.path.join(Architecture.get_path(), "QATCH", "icons", "add.svg")
        self.ICON_AUDIT = os.path.join(Architecture.get_path(), "QATCH", "icons", "audit.svg")
        self.ICON_DELETE = os.path.join(Architecture.get_path(), "QATCH", "icons", "delete.svg")
        self.ICON_REFRESH = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "refresh-cw.svg"
        )
        # ==========================================

        self.parent = parent
        self.admin_file = UserProfiles.find(admin_name, None)[1]
        self.is_audit_mode = False
        self.all_selected = False

        # --- Overlay / glassmorphic setup ---
        # WA_TranslucentBackground is top-level windows only — on a child widget
        # it makes the whole widget invisible. For child widgets the correct
        # approach is to disable Qt's auto-fill so our paintEvent draws the scrim
        # directly over the parent's backing store.
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        # Install event-filter on parent and top-level window so we refit on resize.
        if parent is not None:
            parent.installEventFilter(self)
            top = parent.window()
            if top is not parent:
                top.installEventFilter(self)
        Log.d(f"{TAG} overlay created, parent={parent}")

        self.base_layout = QtWidgets.QVBoxLayout(self)
        # Margins define the inset of the glass panel from the overlay edges (17.5% each side)
        self._panel_margin_pct = 0.175
        self.base_layout.setContentsMargins(0, 0, 0, 0)

        self.glass_frame = QtWidgets.QFrame(self)
        self.glass_frame.setObjectName("userview")
        self.glass_frame.setStyleSheet("""
            QFrame#userview {
                background: rgba(255, 255, 255, 160);
                border: 1px solid rgba(255, 255, 255, 220);
                border-radius: 12px;
            }
        """)
        self._apply_shadow(self.glass_frame, blur_radius=40, alpha=60, offset=(0, 12))

        self.main_layout = QtWidgets.QVBoxLayout(self.glass_frame)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)

        # --- Top Control Bar (Glass Pill Taskbar) ---
        self.taskbar_frame = QtWidgets.QFrame()
        self.taskbar_frame.setObjectName("taskbarFrame")
        self.taskbar_frame.setFixedHeight(55)
        self.taskbar_frame.setStyleSheet("""
            QFrame#taskbarFrame {
                background: rgba(255, 255, 255, 120);
                border: 1px solid rgba(255, 255, 255, 200);
                border-radius: 27px;
            }
        """)
        self._apply_shadow(self.taskbar_frame, blur_radius=15, alpha=20, offset=(0, 4))

        self.top_bar = QtWidgets.QHBoxLayout(self.taskbar_frame)
        self.top_bar.setContentsMargins(20, 0, 20, 0)
        self.top_bar.setSpacing(15)

        self.btn_back = QtWidgets.QPushButton("← Back")
        self.btn_back.setStyleSheet("""
            QPushButton { 
                background: rgba(108, 117, 125, 200); 
                color: white; 
                border-radius: 15px; 
                padding: 6px 15px; 
                font-weight: bold; 
                border: 1px solid rgba(255, 255, 255, 100);
            }
            QPushButton:hover { background: rgba(90, 98, 104, 255); }
        """)
        self.btn_back.setVisible(False)
        self.btn_back.clicked.connect(self.back_to_users)

        # --- Search Bar ---
        self.search_bar = QtWidgets.QLineEdit()
        self.search_bar.setPlaceholderText("🔍 Search Initials or Name...")
        self.search_bar.setFixedWidth(250)
        self.search_bar.setFixedHeight(34)
        self.search_bar.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 180);
                border: 1px solid rgba(255, 255, 255, 240);
                border-radius: 17px;
                padding: 0px 15px;
                color: #333;
            }
            QLineEdit:focus { 
                background: rgba(255, 255, 255, 255);
                border: 1px solid #0AA3E6; 
            }
        """)
        self.search_bar.textChanged.connect(self.filter_users)

        # --- Enhanced Symbolic Toolbar Buttons ---
        # Added a slight default background so actions are more visible, and crisp borders
        base_action_style = """
            QPushButton { 
                background: rgba(255, 255, 255, 140); 
                border: 1px solid rgba(255, 255, 255, 200); 
                border-radius: 17px; 
                min-width: 34px;
                max-width: 34px;
                min-height: 34px;
                max-height: 34px;
            }
            QPushButton:hover { 
                background: rgba(255, 255, 255, 220); 
                border: 1px solid #0AA3E6; 
            }
            QPushButton:pressed { 
                background: rgba(255, 255, 255, 100); 
            }
        """

        # Ensure icon size scales well inside the 34x34 button
        icon_size = QtCore.QSize(18, 18)

        self.btn_add = QtWidgets.QPushButton("")
        self.btn_add.setIcon(QtGui.QIcon(self.ICON_ADD))
        self.btn_add.setIconSize(icon_size)
        self.btn_add.setToolTip("Add New User")
        self.btn_add.setStyleSheet(base_action_style)
        self.btn_add.clicked.connect(self.add_user)

        self.btn_audit_selected = QtWidgets.QPushButton("")
        self.btn_audit_selected.setIcon(QtGui.QIcon(self.ICON_AUDIT))
        self.btn_audit_selected.setIconSize(icon_size)
        self.btn_audit_selected.setToolTip("Audit Selected Users")
        self.btn_audit_selected.setStyleSheet(base_action_style)
        self.btn_audit_selected.clicked.connect(lambda: self.audit_selected())

        self.btn_delete_selected = QtWidgets.QPushButton("")
        self.btn_delete_selected.setIcon(QtGui.QIcon(self.ICON_DELETE))
        self.btn_delete_selected.setIconSize(icon_size)
        self.btn_delete_selected.setToolTip("Delete Selected Users")
        # Distinct red danger styling for delete
        self.btn_delete_selected.setStyleSheet("""
            QPushButton { 
                background: rgba(220, 53, 69, 0.1); 
                border: 1px solid rgba(220, 53, 69, 0.3); 
                border-radius: 17px; 
                min-width: 34px; max-width: 34px; min-height: 34px; max-height: 34px;
            }
            QPushButton:hover { 
                background: rgba(220, 53, 69, 0.25); 
                border: 1px solid rgba(220, 53, 69, 0.6); 
            }
            QPushButton:pressed { background: rgba(220, 53, 69, 0.4); }
        """)
        self.btn_delete_selected.clicked.connect(lambda: self.delete_selected())

        self.btn_refresh = QtWidgets.QPushButton("")
        self.btn_refresh.setIcon(QtGui.QIcon(self.ICON_REFRESH))
        self.btn_refresh.setIconSize(icon_size)
        self.btn_refresh.setToolTip("Refresh Table")
        self.btn_refresh.setStyleSheet(base_action_style)
        self.btn_refresh.clicked.connect(self.update_table_data)

        self.btn_close = QtWidgets.QPushButton("✕")
        self.btn_close.setFixedSize(34, 34)
        self.btn_close.setToolTip("Close")
        self.btn_close.setStyleSheet("""
            QPushButton {
                background: rgba(220, 53, 69, 0.15);
                border: 1px solid rgba(220, 53, 69, 0.35);
                border-radius: 17px;
                color: #C82333;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background: rgba(220, 53, 69, 0.30);
                border: 1px solid rgba(220, 53, 69, 0.65);
            }
            QPushButton:pressed { background: rgba(220, 53, 69, 0.50); }
        """)
        self.btn_close.clicked.connect(self.close)

        # Assemble taskbar (Title removed)
        self.top_bar.addWidget(self.btn_back)
        self.top_bar.addWidget(self.search_bar)
        self.top_bar.addStretch()  # Pushes the actions to the right
        self.top_bar.addWidget(self.btn_add)
        self.top_bar.addWidget(self.btn_audit_selected)
        self.top_bar.addWidget(self.btn_delete_selected)
        self.top_bar.addWidget(self.btn_refresh)
        self.top_bar.addWidget(self.btn_close)

        # --- Table View ---
        self.table = self.TableView()
        self.table.itemChanged.connect(self.handle_inline_edit)
        self.table.horizontalHeader().sectionClicked.connect(self.handle_header_click)

        # --- Bottom Settings Bar ---
        self.settings_layout = QtWidgets.QHBoxLayout()

        self.developerModeChk = QtWidgets.QCheckBox("Enable Developer Mode (unencrypted)")
        enabled, error, expires = UserProfiles.checkDevMode()
        self.developerModeChk.setChecked(enabled)
        color = "#D32F2F" if error else ("#388E3C" if enabled else "#444")
        self.developerModeChk.setStyleSheet(
            f"QCheckBox {{ color: {color}; font-weight: bold; background: transparent; }}"
        )
        self.developerModeChk.stateChanged.connect(self.toggleDevMode)

        self.reqAdminUpd_chkbox = QtWidgets.QCheckBox("Require Admin for Updates")
        self.reqAdminUpd_chkbox.setChecked(UserConstants.REQ_ADMIN_UPDATES)
        self.reqAdminUpd_chkbox.setStyleSheet("QCheckBox { color: #444; background: transparent; }")
        self.reqAdminUpd_chkbox.stateChanged.connect(self.toggleReqAdminUpdates)

        self.settings_layout.addWidget(self.developerModeChk)
        self.settings_layout.addStretch()
        self.settings_layout.addWidget(self.reqAdminUpd_chkbox)

        self.main_layout.addWidget(self.taskbar_frame)
        self.main_layout.addWidget(self.table)
        self.main_layout.addLayout(self.settings_layout)

        self.base_layout.addWidget(self.glass_frame)
        self.setLayout(self.base_layout)

        self.update_table_data()

    def _apply_shadow(self, widget, blur_radius=15, alpha=40, offset=(0, 4)):
        """Helper method to add depth to glass components."""
        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(blur_radius)
        shadow.setColor(QtGui.QColor(0, 0, 0, alpha))
        shadow.setOffset(offset[0], offset[1])
        widget.setGraphicsEffect(shadow)

    # ------------------------------------------------------------------
    # Overlay geometry management
    # ------------------------------------------------------------------

    def _refit_to_parent(self):
        """Resize/reposition self to fill the parent's content area, then
        inset the glass panel via layout margins (never via direct setGeometry,
        which would be immediately overwritten by the layout manager)."""
        if self.parent is None:
            # No Qt parent — fall back to covering the primary screen
            geo = QtWidgets.QApplication.primaryScreen().availableGeometry()
            self.setGeometry(geo)
            Log.w(f"{TAG} _refit_to_parent: no parent, using screen geometry {geo}")
        else:
            geo = self.parent.rect()
            self.setGeometry(geo)
            Log.d(f"{TAG} _refit_to_parent: geometry set to {geo}")
        w, h = self.width(), self.height()
        mx = int(w * self._panel_margin_pct)
        my = int(h * self._panel_margin_pct)
        # Drive the glass panel position through the layout — not setGeometry —
        # so the layout manager and our sizing never fight each other.
        self.base_layout.setContentsMargins(mx, my, mx, my)
        Log.d(f"{TAG} _refit_to_parent: glass panel margins mx={mx} my={my}")

    def showEvent(self, event):
        """Fill parent geometry and raise to front each time we show."""
        Log.d(f"{TAG} showEvent fired, parent={self.parent}")
        self._refit_to_parent()
        self.raise_()
        super().showEvent(event)

    def hideEvent(self, event):
        super().hideEvent(event)

    def eventFilter(self, obj, event):
        """Track parent/window resize so the overlay always covers the content area."""
        if event.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Move):
            if self.isVisible():
                self._refit_to_parent()
        return super().eventFilter(obj, event)

    def paintEvent(self, event):
        """Draw a semi-transparent dark scrim over the whole overlay area."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        # Scrim: dark translucent fill across the full widget
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 110))
        painter.end()

    def mousePressEvent(self, event):
        """Clicking outside the glass panel dismisses the overlay."""
        if not self.glass_frame.geometry().contains(event.pos()):
            self.close()
        else:
            super().mousePressEvent(event)

    def create_role_banner(self, role_name):
        """
        Creates a beautifully styled 'glass' label for user roles.
        Call this during your table population loop and insert it using:
        self.table.setCellWidget(row, col_index, self.create_role_banner(role))
        """
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        badge = QtWidgets.QLabel(role_name)

        # Color coding logic based on role (Customize these as needed)
        role_upper = str(role_name).upper()
        if "ADMIN" in role_upper:
            bg_color = "rgba(220, 53, 69, 0.15)"  # Red tint
            border_color = "rgba(220, 53, 69, 0.4)"
            text_color = "#C82333"
        elif "AUDIT" in role_upper or "MANAGER" in role_upper:
            bg_color = "rgba(255, 193, 7, 0.2)"  # Yellow/Gold tint
            border_color = "rgba(255, 193, 7, 0.5)"
            text_color = "#E0A800"
        else:
            bg_color = "rgba(10, 163, 230, 0.15)"  # Blue tint (Theme default)
            border_color = "rgba(10, 163, 230, 0.4)"
            text_color = "#0AA3E6"

        badge.setStyleSheet(f"""
            QLabel {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                color: {text_color};
                border-radius: 10px;
                padding: 2px 10px;
                font-weight: bold;
                font-size: 11px;
            }}
        """)

        layout.addWidget(badge)
        return container

    # --- Data & Table Management ---

    def update_table_data(self):
        if getattr(self, "is_audit_mode", False):
            return
        self.table.blockSignals(True)
        self.all_selected = False

        user_files, user_info = UserProfiles.get_all_user_info()
        data = {
            "Initials": [i[1] for i in user_info],
            "Name": [i[0] for i in user_info],
            "Role": [i[2] for i in user_info],
            "Created": [i[3] for i in user_info],
            "Accessed": [i[5] for i in user_info],
        }

        # ☐ represents unchecked state for Select All in header
        horHeaders = ["☐", "Initials", "Name", "Role", "Created", "Accessed", "Actions"]
        self.table.clear()
        self.table.setRowCount(len(user_info))
        self.table.setColumnCount(len(horHeaders))
        self.table.setHorizontalHeaderLabels(horHeaders)

        roles_list = [e.name for e in UserRoles][1:]
        if UserRoles.OPERATE.name in roles_list:
            roles_list[roles_list.index(UserRoles.OPERATE.name)] += " (Capture & Analyze)"

        for row in range(len(user_info)):
            initials = data["Initials"][row]

            # 0. Checkbox
            chk_item = QtWidgets.QTableWidgetItem()
            chk_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            chk_item.setCheckState(QtCore.Qt.Unchecked)
            self.table.setItem(row, 0, chk_item)

            # 1. Initials
            item_init = QtWidgets.QTableWidgetItem(initials)
            item_init.setTextAlignment(QtCore.Qt.AlignCenter)
            item_init.setData(QtCore.Qt.UserRole, initials)
            self.table.setItem(row, 1, item_init)

            # 2. Name
            item_name = QtWidgets.QTableWidgetItem(data["Name"][row])
            item_name.setTextAlignment(QtCore.Qt.AlignCenter)
            item_name.setData(QtCore.Qt.UserRole, initials)
            self.table.setItem(row, 2, item_name)

            # 3. Role
            role_combo = QtWidgets.QComboBox()
            role_combo.addItems(roles_list)
            curr_role = data["Role"][row]
            idx = role_combo.findText(curr_role, QtCore.Qt.MatchContains)
            if idx >= 0:
                role_combo.setCurrentIndex(idx)
            self._style_role_combo(role_combo, curr_role)
            role_combo.currentTextChanged.connect(
                lambda text, r=initials, cb=role_combo: self.handle_role_change(r, text, cb)
            )

            combo_container = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(combo_container)
            lay.setContentsMargins(10, 4, 10, 4)
            lay.addWidget(role_combo)
            self.table.setCellWidget(row, 3, combo_container)

            # 4 & 5. Dates
            item_created = QtWidgets.QTableWidgetItem(data["Created"][row])
            item_created.setFlags(QtCore.Qt.ItemIsEnabled)
            item_created.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row, 4, item_created)

            item_accessed = QtWidgets.QTableWidgetItem(data["Accessed"][row])
            item_accessed.setFlags(QtCore.Qt.ItemIsEnabled)
            item_accessed.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row, 5, item_accessed)

            # 6. Actions
            action_widget = QtWidgets.QWidget()
            action_layout = QtWidgets.QHBoxLayout(action_widget)
            action_layout.setContentsMargins(5, 2, 5, 2)
            action_layout.setSpacing(5)

            btn_pwd = QtWidgets.QPushButton("🔑")
            btn_pwd.setToolTip("Reset Password")
            btn_pwd.setFixedSize(28, 28)
            btn_pwd.setStyleSheet(
                "background: transparent; border: 1px solid #FFC107; border-radius: 4px;"
            )
            btn_pwd.clicked.connect(lambda _, i=initials: self.change_password(i))

            btn_audit = QtWidgets.QPushButton("📋")
            btn_audit.setToolTip("Audit User")
            btn_audit.setFixedSize(28, 28)
            btn_audit.setStyleSheet(
                "background: transparent; border: 1px solid #6C757D; border-radius: 4px;"
            )
            btn_audit.clicked.connect(lambda _, i=initials: self.audit_selected([i]))

            btn_delete = QtWidgets.QPushButton("🗑️")
            btn_delete.setToolTip("Delete User")
            btn_delete.setFixedSize(28, 28)
            btn_delete.setStyleSheet(
                "background: transparent; border: 1px solid #DC3545; border-radius: 4px;"
            )
            btn_delete.clicked.connect(lambda _, i=initials: self.delete_selected([i]))

            action_layout.addStretch()
            action_layout.addWidget(btn_pwd)
            action_layout.addWidget(btn_audit)
            action_layout.addWidget(btn_delete)
            action_layout.addStretch()
            self.table.setCellWidget(row, 6, action_widget)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QtWidgets.QHeaderView.Stretch)

        # Re-apply any active search filter
        self.filter_users(self.search_bar.text())
        self.table.blockSignals(False)

    # --- Filtering and Header Selection ---

    def filter_users(self, text):
        text = text.lower()
        for row in range(self.table.rowCount()):
            init_item = self.table.item(row, 1)
            name_item = self.table.item(row, 2)
            if init_item and name_item:
                match = text in init_item.text().lower() or text in name_item.text().lower()
                self.table.setRowHidden(row, not match)

    def handle_header_click(self, index):
        if index == 0:
            self.all_selected = not self.all_selected

            # Determine the correct custom icon path based on state
            icon_path = (
                os.path.join(Architecture.get_path(), "QATCH", "icons", "select-multiple.svg")
                if self.all_selected
                else os.path.join(
                    Architecture.get_path(), "QATCH", "icons", "unselect-multiple.svg"
                )
            )

            header_item = self.table.horizontalHeaderItem(0)
            if header_item:
                header_item = QtWidgets.QTableWidgetItem("")
                header_item.setIcon(
                    QtGui.QIcon(Architecture.get_path(), "QATCH", "icons", "unselect-multiple.svg")
                )
                self.table.setHorizontalHeaderItem(0, header_item)

            self.table.blockSignals(True)
            state = QtCore.Qt.Checked if self.all_selected else QtCore.Qt.Unchecked
            for row in range(self.table.rowCount()):
                if not self.table.isRowHidden(row):  # Only select visible/filtered rows
                    item = self.table.item(row, 0)
                    if item:
                        item.setCheckState(state)
            self.table.blockSignals(False)

    # --- Editing Logic ---

    def handle_inline_edit(self, item):
        col = item.column()
        if col not in [1, 2]:
            return

        original_initials = item.data(QtCore.Qt.UserRole)
        new_val = item.text().strip()
        found, filename = UserProfiles.find(None, original_initials)
        if not found:
            return

        if col == 1:
            new_val = new_val.upper()
            if " " in new_val or len(new_val) < 2 or len(new_val) > 4:
                PopUp.warning(
                    self, "Invalid Input", "Initials must be 2-4 characters with no spaces."
                )
                self.update_table_data()
                return
            self._update_user_xml(filename, new_initials=new_val)

        elif col == 2:
            new_val = new_val.title()
            if " " not in new_val or len(new_val) < 4:
                PopUp.warning(self, "Invalid Input", "Please enter a valid First and Last name.")
                self.update_table_data()
                return
            self._update_user_xml(filename, new_name=new_val)

    def handle_role_change(self, target_initials, new_role_str, combo_widget):
        self._style_role_combo(combo_widget, new_role_str)
        found, filename = UserProfiles.find(None, target_initials)
        if found:
            if self.admin_file == filename:
                PopUp.warning(
                    self,
                    "Security Restriction",
                    "You cannot change the role of the currently active ADMIN.",
                )
                self.update_table_data()
                return
            new_role_enum = UserRoles[new_role_str.split()[0]]
            self._update_user_xml(filename, new_role=new_role_enum)

    def _style_role_combo(self, combo, role_text):
        role_upper = role_text.upper()
        if "ADMIN" in role_upper:
            bg_color = "#0AA3E6"
            color = "white"
        elif "OPERATE" in role_upper:
            bg_color = "#28A745"
            color = "white"
        elif "CAPTURE" in role_upper:
            bg_color = "#FFC107"
            color = "black"
        elif "ANALYZE" in role_upper:
            bg_color = "#DC3545"
            color = "white"
        else:
            bg_color = "#6C757D"
            color = "white"

        combo.setStyleSheet(f"""
            QComboBox {{ background-color: {bg_color}; color: {color}; border-radius: 12px; padding: 2px 10px; font-weight: bold; font-size: 11px; border: none; }}
            QComboBox::drop-down {{ border: none; }}
        """)

    def _update_user_xml(
        self, filename, new_initials=None, new_name=None, new_role=None, new_pwd_plain=None
    ):
        file = os.path.join(UserProfiles.PATH, filename)
        doc = minidom.parse(file)
        xml = doc.documentElement
        up = doc.getElementsByTagName("secure_user_info")

        curr_pwd, curr_init, curr_name, curr_role_enum = "", "", "", UserRoles.INVALID
        for u in up:
            try:
                curr_role_enum = UserRoles(int(u.getAttribute("role")))
            except:
                pass
            curr_pwd = u.getAttribute("password")
            curr_init = u.getAttribute("initials")
            curr_name = u.getAttribute("name")
            if u.attributes["password"].value.find("X") < 0:
                u.attributes["password"].value += "X"

        final_init = new_initials if new_initials else curr_init
        final_name = new_name if new_name else curr_name
        final_role = new_role if new_role else curr_role_enum

        salt = filename[:-4]
        hash_obj = hashlib.sha256()

        if new_pwd_plain:
            h2 = hashlib.sha256()
            h2.update(salt.encode())
            h2.update(new_pwd_plain.encode())
            final_pwd = h2.hexdigest()
        else:
            final_pwd = curr_pwd

        hash_obj.update(salt.encode())
        hash_obj.update(final_name.encode())
        hash_obj.update(final_init.encode())
        hash_obj.update(final_role.name.encode())
        hash_obj.update(final_pwd.encode())
        signature = hash_obj.hexdigest()

        info = doc.createElement("secure_user_info")
        xml.appendChild(info)
        info.setAttribute("name", final_name)
        info.setAttribute("initials", final_init)
        info.setAttribute("role", str(final_role.value))
        info.setAttribute("password", final_pwd)
        info.setAttribute("signature", signature)

        ts_type, ts_val = "modified", dt.datetime.now().isoformat()
        ts_hash = hashlib.sha256()
        ts_hash.update(salt.encode())
        ts_hash.update(ts_type.encode())
        ts_hash.update(ts_val.encode())

        ts1 = doc.createElement("timestamp")
        xml.appendChild(ts1)
        ts1.setAttribute("type", ts_type)
        ts1.setAttribute("value", ts_val)
        ts1.setAttribute("signature", ts_hash.hexdigest())

        try:
            with open(file, "w") as f:
                f.write(doc.toxml(encoding="ascii").decode(encoding="utf-8", errors="ignore"))
            self.update_table_data()
        except Exception as e:
            PopUp.critical(self, "Save Failed", str(e), ok_only=True)

    def change_password(self, target_initials):
        found, filename = UserProfiles.find(None, target_initials)
        if not found:
            return
        pwd, ok = QtWidgets.QInputDialog.getText(
            None,
            "Reset Password",
            f"New Password for {target_initials}:",
            QtWidgets.QLineEdit.Password,
        )
        if ok and len(pwd) >= 8:
            self._update_user_xml(filename, new_pwd_plain=pwd)
        elif ok:
            PopUp.warning(self, "Invalid", "Passwords must be at least 8 characters.")

    # --- Unified Audit Engine ---

    def back_to_users(self):
        self.is_audit_mode = False
        self.view_title.setText("Users")
        self.btn_back.setVisible(False)
        self.search_bar.setVisible(True)
        self.btn_audit_selected.setVisible(True)
        self.btn_delete_selected.setVisible(True)
        self.btn_add.setVisible(True)
        self.btn_refresh.setVisible(True)
        self.update_table_data()

    def audit_selected(self, override_list=None):
        initials_list = []
        if override_list:
            initials_list = override_list
        else:
            for row in range(self.table.rowCount()):
                chk_item = self.table.item(row, 0)
                if (
                    chk_item
                    and chk_item.checkState() == QtCore.Qt.Checked
                    and not self.table.isRowHidden(row)
                ):
                    initials_list.append(self.table.item(row, 1).text())

        if not initials_list:
            PopUp.warning(self, "No Selection", "Please select at least one user to audit.")
            return

        self.is_audit_mode = True
        title = "Unified Audit" if len(initials_list) > 1 else f"Audit Log: {initials_list[0]}"
        self.view_title.setText(title)

        self.search_bar.setVisible(False)
        self.btn_back.setVisible(True)
        self.btn_add.setVisible(False)
        self.btn_audit_selected.setVisible(False)
        self.btn_delete_selected.setVisible(False)
        self.btn_refresh.setVisible(False)

        self.table.blockSignals(True)
        self.table.clear()

        all_audit_entries = []
        overall_failure = False

        for target in initials_list:
            entries, failed = self._parse_single_audit(target)
            all_audit_entries.extend(entries)
            if failed:
                overall_failure = True

        all_audit_entries.sort(key=lambda x: x["raw_time"])

        self.table.setRowCount(len(all_audit_entries))
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Timestamp", "User", "Result", "Audit Notes"])

        for row, entry in enumerate(all_audit_entries):
            items = [entry["time_display"], entry["user"], entry["result"], entry["notes"]]
            for col, text in enumerate(items):
                is_error = text.startswith("*") and text.endswith("*")
                clean_text = text[1:-1] if is_error else text

                item = QtWidgets.QTableWidgetItem(clean_text)
                item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                item.setTextAlignment(QtCore.Qt.AlignCenter if col < 3 else QtCore.Qt.AlignLeft)

                if is_error:
                    item.setForeground(QtGui.QBrush(QtGui.QColor(255, 60, 60)))
                self.table.setItem(row, col, item)

        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.blockSignals(False)

        if overall_failure:
            PopUp.warning(
                self, "Audit Result", "There are security failures in this unified audit log!"
            )

    def _parse_single_audit(self, target_initials):
        found, filename = UserProfiles.find(None, target_initials)
        entries = []
        if not found:
            return entries, True

        file = os.path.join(UserProfiles.PATH, filename)
        if not os.path.isfile(file):
            return entries, True

        audit_failed = False
        xml_doc = minidom.parse(file)
        top = xml_doc.documentElement

        last_role, last_password, last_initials, last_name = "NONE", "NONE", "NONE", "NONE"
        passwords = 0

        for child in top.childNodes:
            audit_pass, change_record = False, False
            change_role, change_password, change_initials, change_name = False, False, False, False
            type_str, time_str, signature = "unknown", "unknown", None
            notes = "[empty]"

            salt = filename[:-4]
            hash_obj = hashlib.sha256()
            hash_obj.update(salt.encode())

            for name, value in child.attributes.items():
                if name == "signature":
                    signature = hash_obj.hexdigest()
                    if change_record:
                        changed = []
                        if change_role:
                            changed.append(f"ROLE={last_role}")
                        if change_password:
                            changed.append(f"PASSWORD")
                        if change_initials:
                            changed.append(f"INITIALS={last_initials}")
                        if change_name:
                            changed.append(f"NAME={last_name}")
                        if not changed:
                            changed.append("NOTHING")
                        notes = "{}: " + ", ".join(changed)
                    else:
                        if (
                            type_str != "accessed"
                            and len(entries) > 0
                            and entries[-1]["raw_time"] == "[empty]"
                        ):
                            entries[-1]["raw_time"] = time_str
                            entries[-1]["time_display"] = time_str
                            entries[-1]["notes"] = entries[-1]["notes"].format(type_str.upper())
                        notes = f"[{type_str.upper()}]"

                    audit_pass = signature == value
                else:
                    try:
                        val = UserRoles(int(value)).name if name == "role" else value
                        if name == "password":
                            if "X" in value:
                                val = value[:-1]
                            else:
                                passwords += 1
                        if name == "type":
                            type_str = val
                        if name == "value":
                            time_str = val
                        if name in ["role", "password", "initials", "name"]:
                            change_record = True
                            if name == "role":
                                change_role = val != last_role
                                last_role = val
                            if name == "password":
                                change_password = val != last_password
                                last_password = val
                            if name == "initials":
                                change_initials = val != last_initials
                                last_initials = val
                            if name == "name":
                                change_name = val != last_name
                                last_name = val
                    except:
                        val = value
                    hash_obj.update(val.encode())

            if not signature:
                audit_pass = False
                notes = "MISSING signature"

            is_err = not audit_pass
            if is_err:
                audit_failed = True

            entries.append(
                {
                    "raw_time": time_str,
                    "time_display": f"*{time_str}*" if is_err else time_str,
                    "user": target_initials,
                    "result": "*FAIL*" if is_err else "PASS",
                    "notes": f"*{notes}*" if is_err else notes,
                }
            )

        if passwords != 1:
            audit_failed = True
            entries.append(
                {
                    "raw_time": "Z_NOW",
                    "time_display": "*now*",
                    "user": target_initials,
                    "result": "*FAIL*",
                    "notes": "*More than one valid user info element!*",
                }
            )

        return entries, audit_failed

    # --- Deletion Logic ---

    def delete_selected(self, override_list=None):
        initials_list = []
        if override_list:
            initials_list = override_list
        else:
            for row in range(self.table.rowCount()):
                chk_item = self.table.item(row, 0)
                if (
                    chk_item
                    and chk_item.checkState() == QtCore.Qt.Checked
                    and not self.table.isRowHidden(row)
                ):
                    initials_list.append(self.table.item(row, 1).text())

        if not initials_list:
            PopUp.warning(self, "No Selection", "Please select at least one user to delete.")
            return

        if len(initials_list) > 1:
            if not PopUp.question(
                self,
                "Confirm Bulk Delete",
                f"Are you sure you want to permanently delete {len(initials_list)} selected users?",
            ):
                return
        else:
            if not PopUp.question(
                self, "Confirm Delete", f"Are you sure you want to delete user {initials_list[0]}?"
            ):
                return

        admin_initials = UserProfiles.get_user_info(self.admin_file)[1]

        for init in initials_list:
            if init == admin_initials:
                if UserProfiles.count() > len(initials_list):
                    PopUp.warning(
                        self,
                        "Action Denied",
                        "Cannot delete active ADMIN user while other accounts remain. Aborting ADMIN deletion.",
                    )
                    continue
                else:
                    if PopUp.question(
                        self,
                        "Remove Last User",
                        "WARNING: You are deleting the active ADMIN account. This will sign you out and cannot be undone.\nProceed?",
                    ):
                        self._archive_user_file(init)
                        UserProfiles.session_end()
                        self.parent.username.setText("User: [NONE]")
                        self.parent.userrole = UserRoles.NONE
                        self.parent.signinout.setText("&Sign In")
                        self.parent.manage.setText("&Manage Users...")
                        self.close()
                        return
                    continue
            self._archive_user_file(init)

        self.update_table_data()

    def _archive_user_file(self, initials):
        found, filename = UserProfiles.find(None, initials)
        if not found:
            return
        file = os.path.join(UserProfiles.PATH, filename)
        ts_type, ts_val = "deleted", dt.datetime.now().isoformat()
        salt = filename[:-4]
        hash_obj = hashlib.sha256()
        hash_obj.update(salt.encode())
        hash_obj.update(ts_type.encode())
        hash_obj.update(ts_val.encode())

        try:
            with open(file, "rb+") as f:
                f.seek(-15, 2)
                f.write(
                    f'<timestamp type="{ts_type}" value="{ts_val}" signature="{hash_obj.hexdigest()}"/></user_profile>'.encode()
                )

            archive_to = os.path.join(os.path.split(file)[0], "archived")
            FileManager.create_dir(archive_to)
            os.rename(file, os.path.join(archive_to, os.path.split(file)[1]))
        except Exception as e:
            Log.e(f"Failed to archive user {initials}: {str(e)}")

    def add_user(self):
        roles = [e.name for e in UserRoles][1:]
        if UserRoles.OPERATE.name in roles:
            roles[roles.index(UserRoles.OPERATE.name)] += " (Capture & Analyze)"

        role_str, ok = QtWidgets.QInputDialog().getItem(None, "Add User", "Role:", roles, 0, False)
        if ok:
            UserProfiles.create_new_user(UserRoles[role_str.split()[0]])
            self.update_table_data()

    def toggleDevMode(self, arg):
        try:
            dev_path = os.path.join(Constants.local_app_data_path, ".dev_mode")
            if self.developerModeChk.isChecked():
                with open(dev_path, "w") as dev:
                    expires_at = str(
                        (dt.datetime.now() + dt.timedelta(days=UserConstants.DEV_EXPIRE_LEN)).date()
                    )
                    hexify_str1 = expires_at.encode().hex()
                    hexify_str2 = Constants.app_date.encode().hex()
                    encode_key = "DEADBEEFDEADBEEFDEAD"
                    dev.write(
                        bytes([(ord(a) ^ ord(b)) for a, b in zip(hexify_str1, encode_key)]).decode()
                        + "\n"
                    )
                    dev.write(
                        bytes([(ord(a) ^ ord(b)) for a, b in zip(hexify_str2, encode_key)]).decode()
                    )

                    self.developerModeChk.setStyleSheet("color:#388E3C; font-weight: bold;")
                    PopUp.information(
                        self,
                        "Developer Mode Status",
                        f"Developer Mode: ENABLED.\nExpires on {expires_at}",
                    )
            else:
                if os.path.exists(dev_path):
                    os.remove(dev_path)
                self.developerModeChk.setStyleSheet("color:#444; font-weight: bold;")
        except Exception as e:
            Log.e(f"Error updating Developer Mode: {str(e)}")

    def toggleReqAdminUpdates(self, arg):
        try:
            if not os.path.isfile(Constants.user_constants_path):
                os.makedirs(os.path.split(Constants.user_constants_path)[0], exist_ok=True)
            with open(Constants.user_constants_path, "w") as uc:
                UserConstants.REQ_ADMIN_UPDATES = self.reqAdminUpd_chkbox.isChecked()
                uc.write(f"REQ_ADMIN_UPDATES = {UserConstants.REQ_ADMIN_UPDATES}")
        except Exception as e:
            Log.e(f"Failed to save user constants: {str(e)}")
