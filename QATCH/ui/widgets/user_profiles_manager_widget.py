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
            sp = self.sizePolicy()
            sp.setRetainSizeWhenHidden(True)
            self.setSizePolicy(sp)
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
                QHeaderView {
                    border-top-left-radius: 9px;
                    border-top-right-radius: 9px;
                    border: none;
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
                QHeaderView::section:first {
                    border-top-left-radius: 9px;
                }
                QHeaderView::section:last {
                    border-top-right-radius: 9px;
                    border-right: none;
                }
                QHeaderView::section:hover { 
                    background-color: rgba(255, 255, 255, 180); 
                }
            """)

    def __init__(self, parent=None, admin_name=None):
        super(UserProfilesManagerWidget, self).__init__(parent)

        # ==========================================
        #  INSERT YOUR CUSTOM ICON PATHS HERE
        # ==========================================
        self.ICON_ADD = os.path.join(Architecture.get_path(), "QATCH", "icons", "add.svg")
        self.ICON_AUDIT = os.path.join(Architecture.get_path(), "QATCH", "icons", "audit.svg")
        self.ICON_DELETE = os.path.join(Architecture.get_path(), "QATCH", "icons", "delete.svg")
        self.ICON_REFRESH = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "refresh-cw.svg"
        )
        self.ICON_PWD = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "reset-password.svg"
        )
        self.ICON_SEARCH = os.path.join(Architecture.get_path(), "QATCH", "icons", "search.svg")
        self.ICON_BACK = os.path.join(Architecture.get_path(), "QATCH", "icons", "left-arrow.svg")

        self.ICON_EXPAND = os.path.join(Architecture.get_path(), "QATCH", "icons", "expand.svg")
        self.ICON_COLLAPSE = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "collapse.svg"
        )  # <-- Add this line
        self.ICON_CLEAR = os.path.join(Architecture.get_path(), "QATCH", "icons", "clear.svg")
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
                background: rgba(255, 255, 255, 215);
                border: 1.5px solid rgba(255, 255, 255, 230);
                border-radius: 12px;
            }
        """)
        # NOTE: intentionally no _apply_shadow here — QGraphicsDropShadowEffect on
        # a parent frame causes native QComboBox popups to render with a square clip
        # shadow. The border above provides sufficient visual separation.

        # Animation state
        self._scrim_alpha = 0  # paintEvent reads this; animated on open/close
        self._panel_alpha = 215  # glass_frame background alpha; animated on open/close
        self._closing = False  # guards closeEvent re-entry during fade-out
        # NOTE: no QGraphicsOpacityEffect here — installing one on a child widget whose
        # parent overrides paintEvent with WA_NoSystemBackground causes Qt's effect
        # compositor to try to grab the widget pixmap while our painter is already
        # active → "Painter not active" / "A paint device can only be painted by one
        # painter at a time" spam and invisible content.  We animate via stylesheet
        # background-alpha instead, which is safe at any call site.

        self.main_layout = QtWidgets.QVBoxLayout(self.glass_frame)
        self.main_layout.setContentsMargins(20, 38, 20, 20)
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

        self.btn_back = QtWidgets.QPushButton(" Back")
        self.btn_back.setIcon(QtGui.QIcon(self.ICON_BACK))
        self.btn_back.setIconSize(QtCore.QSize(16, 16))
        self.btn_back.setStyleSheet("""
            QPushButton { 
                background: rgba(255, 255, 255, 140); 
                color: #333; 
                border-radius: 17px; 
                padding: 0px 15px; 
                font-weight: bold; 
                border: 1px solid rgba(255, 255, 255, 200);
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
        """)
        self.btn_back.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_back.setVisible(False)
        self.btn_back.clicked.connect(self.back_to_users)

        # --- Search Bar ---
        self.search_bar = QtWidgets.QLineEdit()
        self.search_bar.setPlaceholderText("Search Initials or Name...")
        self.search_bar.addAction(
            QtGui.QIcon(self.ICON_SEARCH), QtWidgets.QLineEdit.LeadingPosition
        )
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

        # --- Audit / mode title label (replaces search bar in audit mode) ---
        self.view_title = QtWidgets.QLabel("User Management")
        self.view_title.setVisible(False)
        self.view_title.setStyleSheet("""
            QLabel {
                color: #1a1a2e;
                font-size: 13px;
                font-weight: bold;
                background: transparent;
                padding: 0px 6px;
            }
        """)

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
        self.btn_add.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_add.clicked.connect(self.add_user)

        self.btn_audit_selected = QtWidgets.QPushButton("")
        self.btn_audit_selected.setIcon(QtGui.QIcon(self.ICON_AUDIT))
        self.btn_audit_selected.setIconSize(icon_size)
        self.btn_audit_selected.setToolTip("Audit Selected Users")
        self.btn_audit_selected.setStyleSheet(base_action_style)
        self.btn_audit_selected.setCursor(QtCore.Qt.PointingHandCursor)
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
        self.btn_delete_selected.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_delete_selected.clicked.connect(lambda: self.delete_selected())

        self.btn_refresh = QtWidgets.QPushButton("")
        self.btn_refresh.setIcon(QtGui.QIcon(self.ICON_REFRESH))
        self.btn_refresh.setIconSize(icon_size)
        self.btn_refresh.setToolTip("Refresh Table")
        self.btn_refresh.setStyleSheet(base_action_style)
        self.btn_refresh.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_refresh.clicked.connect(self._animate_refresh_spin)

        self._default_margin_pct = 0.175
        button_size = 16
        icon_size = QtCore.QSize(10, 10)

        # Fullscreen Toggle (Green Dot)
        self.btn_fullscreen = QtWidgets.QPushButton("", self)
        self.btn_fullscreen.setFixedSize(button_size, button_size)
        self.btn_fullscreen.setIcon(QtGui.QIcon(self.ICON_EXPAND))
        self.btn_fullscreen.setIconSize(icon_size)
        self.btn_fullscreen.setToolTip("Toggle Fullscreen")
        self.btn_fullscreen.setStyleSheet("""
            QPushButton {
                background: rgba(39, 201, 63, 140);
                border: 1px solid rgba(39, 201, 63, 190);
                border-radius: 8px;
            }
            QPushButton:hover { background: rgba(39, 201, 63, 255); }
        """)
        self.btn_fullscreen.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)

        # Close Window (Red Dot)
        self.btn_close = QtWidgets.QPushButton("", self)
        self.btn_close.setFixedSize(button_size, button_size)
        self.btn_close.setIcon(QtGui.QIcon(self.ICON_CLEAR))
        self.btn_close.setIconSize(icon_size)
        self.btn_close.setToolTip("Close")
        self.btn_close.setStyleSheet("""
            QPushButton {
                background: rgba(255, 95, 86, 140);
                border: 1px solid rgba(255, 95, 86, 190);
                border-radius: 8px;
            }
            QPushButton:hover { background: rgba(255, 95, 86, 255); }
        """)
        self.btn_close.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_close.clicked.connect(self.close)
        self.btn_close.raise_()

        # Assemble taskbar — add, audit, refresh, delete (no close; panel click-outside handles dismiss)
        self.top_bar.addWidget(self.btn_back)
        self.top_bar.addWidget(self.view_title)
        self.top_bar.addWidget(self.search_bar)
        self.top_bar.addStretch()
        self.top_bar.addWidget(self.btn_add)
        self.top_bar.addWidget(self.btn_audit_selected)
        self.top_bar.addWidget(self.btn_refresh)
        self.top_bar.addWidget(self.btn_delete_selected)

        # --- Table View (wrapped in a rounded-corner container frame) ---
        self.table = self.TableView()
        self.table.itemChanged.connect(self.handle_inline_edit)
        self.table.horizontalHeader().sectionClicked.connect(self.handle_header_click)

        self.table_container = QtWidgets.QFrame()
        self.table_container.setObjectName("tableContainer")
        self.table_container.setStyleSheet("""
            QFrame#tableContainer {
                background: rgba(255, 255, 255, 30);
                border: 1px solid rgba(200, 210, 220, 110);
                border-radius: 10px;
            }
        """)
        # Retain layout space when hidden so the transition grab works correctly
        _sp = self.table_container.sizePolicy()
        _sp.setRetainSizeWhenHidden(True)
        self.table_container.setSizePolicy(_sp)

        _container_layout = QtWidgets.QVBoxLayout(self.table_container)
        _container_layout.setContentsMargins(0, 0, 0, 0)
        _container_layout.setSpacing(0)
        _container_layout.addWidget(self.table)

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
        self.main_layout.addWidget(self.table_container)
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

    def _animate_refresh_spin(self):
        """Animates a single 360-degree spin on the refresh icon, slow with deceleration."""
        if (
            hasattr(self, "refresh_anim")
            and self.refresh_anim.state() == QtCore.QAbstractAnimation.Running
        ):
            return

        self.refresh_anim = QtCore.QVariantAnimation(self)
        self.refresh_anim.setDuration(900)  # slow, satisfying rotation
        self.refresh_anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)  # decelerates naturally
        self.refresh_anim.setStartValue(0.0)
        self.refresh_anim.setEndValue(360.0)

        orig_pixmap = QtGui.QIcon(self.ICON_REFRESH).pixmap(18, 18)
        w, h = orig_pixmap.width(), orig_pixmap.height()

        def update_icon(angle):
            transform = (
                QtGui.QTransform().translate(w / 2, h / 2).rotate(angle).translate(-w / 2, -h / 2)
            )
            rotated = orig_pixmap.transformed(transform, QtCore.Qt.SmoothTransformation)
            self.btn_refresh.setIcon(QtGui.QIcon(rotated))

        def finish_refresh():
            self.btn_refresh.setIcon(QtGui.QIcon(self.ICON_REFRESH))
            self.update_table_data()

        self.refresh_anim.valueChanged.connect(update_icon)
        self.refresh_anim.finished.connect(finish_refresh)
        self.refresh_anim.start()

    def _animate_table_transition(self, update_func, direction="right"):
        """Slides the table content horizontally while swapping data — no processEvents stutter."""
        # Stop any in-flight transition cleanly before starting a new one
        if (
            hasattr(self, "transition_group")
            and self.transition_group.state() == QtCore.QAbstractAnimation.Running
        ):
            self.transition_group.stop()

        # 1. Snapshot the current visible state BEFORE any data change
        old_pixmap = self.table_container.grab()
        tbl_geo = self.table_container.geometry()  # geometry in glass_frame coords

        # 2. Hide container so the data rebuild is invisible, then run the update
        self.table_container.setVisible(False)
        update_func()

        # 3. Defer the rest by one event-loop tick so the layout has settled.
        #    No processEvents() — that's the source of the original stutter.
        def _start():
            # Briefly show to force a layout pass + render for the grab, then hide again.
            # This is synchronous — no paint event fires between show and hide.
            self.table_container.setVisible(True)
            new_pixmap = self.table_container.grab()
            self.table_container.setVisible(False)

            # Build overlay labels as glass_frame children so they clip correctly
            old_overlay = QtWidgets.QLabel(self.glass_frame)
            old_overlay.setPixmap(old_pixmap)
            old_overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
            old_overlay.setGeometry(tbl_geo)
            old_overlay.show()
            old_overlay.raise_()

            w = tbl_geo.width()
            start_pos = tbl_geo.topLeft()
            if direction == "right":
                old_end = QtCore.QPoint(start_pos.x() + w, start_pos.y())
                new_start = QtCore.QPoint(start_pos.x() - w, start_pos.y())
            else:
                old_end = QtCore.QPoint(start_pos.x() - w, start_pos.y())
                new_start = QtCore.QPoint(start_pos.x() + w, start_pos.y())

            new_overlay = QtWidgets.QLabel(self.glass_frame)
            new_overlay.setPixmap(new_pixmap)
            new_overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
            new_overlay.setGeometry(QtCore.QRect(new_start, tbl_geo.size()))
            new_overlay.show()
            new_overlay.raise_()

            # OutQuint: fast start → smooth deceleration — no mid-animation hitching
            anim_old = QtCore.QPropertyAnimation(old_overlay, b"pos")
            anim_old.setDuration(320)
            anim_old.setEasingCurve(QtCore.QEasingCurve.OutQuint)
            anim_old.setStartValue(start_pos)
            anim_old.setEndValue(old_end)

            anim_new = QtCore.QPropertyAnimation(new_overlay, b"pos")
            anim_new.setDuration(320)
            anim_new.setEasingCurve(QtCore.QEasingCurve.OutQuint)
            anim_new.setStartValue(new_start)
            anim_new.setEndValue(start_pos)

            self.transition_group = QtCore.QParallelAnimationGroup(self)
            self.transition_group.addAnimation(anim_old)
            self.transition_group.addAnimation(anim_new)

            def _cleanup():
                old_overlay.deleteLater()
                new_overlay.deleteLater()
                self.table_container.setVisible(True)

            self.transition_group.finished.connect(_cleanup)
            self.transition_group.start()

        QtCore.QTimer.singleShot(0, _start)

    # ------------------------------------------------------------------
    # Overlay geometry management
    # ------------------------------------------------------------------

    def _refit_to_parent(self):
        """Resize/reposition self to fill the parent's content area."""
        if self.parent is None:
            geo = QtWidgets.QApplication.primaryScreen().availableGeometry()
            self.setGeometry(geo)
        else:
            geo = self.parent.rect()
            self.setGeometry(geo)

        # If a transition animation is actively running, let it handle the layout
        if hasattr(self, "anim") and self.anim.state() == QtCore.QAbstractAnimation.Running:
            return

        w, h = self.width(), self.height()

        if getattr(self, "_is_fullscreen", False):
            mx = my = 0
            alpha, radius, border = getattr(self, "_panel_alpha", 255), 0, 0.0
        else:
            mx = int(w * getattr(self, "_default_margin_pct", 0.175))
            my = int(h * getattr(self, "_default_margin_pct", 0.175))
            alpha, radius, border = getattr(self, "_panel_alpha", 215), 12, 1.5

        self.base_layout.setContentsMargins(mx, my, mx, my)
        self.glass_frame.setStyleSheet(f"""
            QFrame#userview {{ 
                background: rgba(255, 255, 255, {alpha}); 
                border: {border}px solid rgba(255, 255, 255, 230); 
                border-radius: {radius}px; 
            }}
        """)

        self._update_dots_position(mx, my)

    def _update_dots_position(self, mx, my):
        """Helper to position dots safely inside the glass panel."""
        w = self.width()
        btn_sz = 16  # Matched to the new button size
        padding_top = 12
        padding_right = 16
        spacing = 8

        close_x = w - mx - padding_right - btn_sz
        close_y = my + padding_top
        self.btn_close.setGeometry(close_x, close_y, btn_sz, btn_sz)
        self.btn_close.raise_()

        fs_x = close_x - spacing - btn_sz
        fs_y = close_y
        self.btn_fullscreen.setGeometry(fs_x, fs_y, btn_sz, btn_sz)
        self.btn_fullscreen.raise_()

    def toggle_fullscreen(self):
        """Toggles between inset glass panel and a full-bleed opaque window with animation."""
        if hasattr(self, "anim") and self.anim.state() == QtCore.QAbstractAnimation.Running:
            self.anim.stop()

        self._is_fullscreen = not getattr(self, "_is_fullscreen", False)

        # --- NEW: Swap the icon based on the new state ---
        icon_path = self.ICON_COLLAPSE if self._is_fullscreen else self.ICON_EXPAND
        self.btn_fullscreen.setIcon(QtGui.QIcon(icon_path))
        # -------------------------------------------------

        w = self.width()
        start_m = self.base_layout.contentsMargins().left()

        # Determine targets and instantly apply the static CSS to avoid layout engine thrashing
        if self._is_fullscreen:
            target_m = 0
            self._panel_alpha = 255  # <--- FIX: Update the internal state tracker
            self.glass_frame.setStyleSheet("""
                QFrame#userview { 
                    background: rgba(255, 255, 255, 255); 
                    border: 0px solid rgba(255, 255, 255, 230); 
                    border-radius: 0px; 
                }
            """)
        else:
            target_m = int(w * getattr(self, "_default_margin_pct", 0.175))
            self._panel_alpha = 215  # <--- FIX: Update the internal state tracker
            self.glass_frame.setStyleSheet("""
                QFrame#userview { 
                    background: rgba(255, 255, 255, 215); 
                    border: 1.5px solid rgba(255, 255, 255, 230); 
                    border-radius: 12px; 
                }
            """)

        # QVariantAnimation drives the layout margin float over 250ms
        self.anim = QtCore.QVariantAnimation(self)
        self.anim.setDuration(250)
        self.anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self.anim.setStartValue(float(start_m))
        self.anim.setEndValue(float(target_m))

        self.anim.valueChanged.connect(self._apply_margin_step)
        self.anim.finished.connect(self._refit_to_parent)
        self.anim.start()

    def _apply_margin_step(self, current_mx):
        """Fired every frame by the expand/collapse animation.

        The animated value is the *horizontal* margin (derived from width).
        The vertical margin is kept proportional to height independently so the
        glass panel maintains its intended inset regardless of window aspect ratio.
        """
        w = self.width()
        h = self.height()
        mx = int(current_mx)
        # Back-compute the fractional progress so we can apply it to height too
        pct = (current_mx / w) if w > 0 else self._default_margin_pct
        my = int(h * pct)
        self.base_layout.setContentsMargins(mx, my, mx, my)
        self._update_dots_position(mx, my)

    def _apply_animation_step(
        self, progress, start_m, target_m, start_a, target_a, start_r, target_r, start_b, target_b
    ):
        """Fired every frame to update geometry and stylesheet."""
        current_m = int(start_m + (target_m - start_m) * progress)
        current_a = int(start_a + (target_a - start_a) * progress)
        current_r = int(start_r + (target_r - start_r) * progress)
        current_b = start_b + (target_b - start_b) * progress

        self.base_layout.setContentsMargins(current_m, current_m, current_m, current_m)
        self.glass_frame.setStyleSheet(f"""
            QFrame#userview {{ 
                background: rgba(255, 255, 255, {current_a}); 
                border: {current_b:.1f}px solid rgba(255, 255, 255, 230); 
                border-radius: {current_r}px; 
            }}
        """)
        self._update_dots_position(current_m, current_m)

    def showEvent(self, event):
        """Fill parent geometry, raise to front, then fade in."""
        Log.d(f"{TAG} showEvent fired, parent={self.parent}")
        self._refit_to_parent()
        self.raise_()
        self._animate_open()
        super().showEvent(event)

    def hideEvent(self, event):
        super().hideEvent(event)

    def closeEvent(self, event):
        """Intercept close to run a fade-out first; re-enters with _closing=True to accept."""
        if self._closing:
            event.accept()
            return
        # If a close animation is already in flight let it finish
        if (
            hasattr(self, "_close_anim")
            and self._close_anim.state() == QtCore.QAbstractAnimation.Running
        ):
            event.ignore()
            return
        event.ignore()
        self._animate_close()

    # ------------------------------------------------------------------
    # Open / close animations
    # ------------------------------------------------------------------

    def _set_panel_alpha(self, alpha):
        """Apply glass-frame background alpha through the stylesheet (no graphics effect)."""
        self._panel_alpha = int(alpha)
        is_fs = getattr(self, "_is_fullscreen", False)
        if is_fs:
            bg, border, radius = self._panel_alpha, 0, 0
        else:
            bg, border, radius = self._panel_alpha, 1.5, 12
        self.glass_frame.setStyleSheet(f"""
            QFrame#userview {{
                background: rgba(255, 255, 255, {bg});
                border: {border}px solid rgba(255, 255, 255, 230);
                border-radius: {radius}px;
            }}
        """)

    def _animate_open(self):
        """Fade the scrim in over 200 ms. The glass panel appears immediately at full
        opacity — no per-frame setStyleSheet call, so the animation is pure fillRect
        speed and stays perfectly smooth."""
        if (
            hasattr(self, "_close_anim")
            and self._close_anim.state() == QtCore.QAbstractAnimation.Running
        ):
            self._close_anim.stop()

        # Snap panel to full opacity instantly (no stylesheet animation per frame)
        target_alpha = 255 if getattr(self, "_is_fullscreen", False) else 215
        self._panel_alpha = target_alpha
        self._set_panel_alpha(target_alpha)

        self._scrim_alpha = 0
        self.update()

        self._open_anim = QtCore.QVariantAnimation(self)
        self._open_anim.setDuration(200)
        self._open_anim.setEasingCurve(QtCore.QEasingCurve.OutQuad)
        self._open_anim.setStartValue(0)
        self._open_anim.setEndValue(65)

        def _step(v):
            self._scrim_alpha = int(v)
            self.update()  # triggers paintEvent → single fillRect, very cheap

        self._open_anim.valueChanged.connect(_step)
        self._open_anim.start()

    def _animate_close(self):
        """Fade the scrim out over 180 ms then perform the real close.
        Panel stays fully opaque — only the scrim animates for the same reason as open."""
        if (
            hasattr(self, "_open_anim")
            and self._open_anim.state() == QtCore.QAbstractAnimation.Running
        ):
            self._open_anim.stop()

        start_scrim = self._scrim_alpha

        self._close_anim = QtCore.QVariantAnimation(self)
        self._close_anim.setDuration(180)
        self._close_anim.setEasingCurve(QtCore.QEasingCurve.InQuad)
        self._close_anim.setStartValue(start_scrim)
        self._close_anim.setEndValue(0)

        def _step(v):
            self._scrim_alpha = int(v)
            self.update()

        self._close_anim.valueChanged.connect(_step)
        self._close_anim.finished.connect(self._do_close)
        self._close_anim.start()

    def _do_close(self):
        """Actually perform the close after the fade-out animation completes."""
        self._closing = True
        self.close()  # closeEvent sees _closing=True → accepts → Qt calls hide()
        self._closing = False

    def eventFilter(self, obj, event):
        """Track parent/window resize so the overlay always covers the content area."""
        if event.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Move):
            if self.isVisible():
                self._refit_to_parent()
        return super().eventFilter(obj, event)

    def paintEvent(self, event):
        """Draw a semi-transparent dark scrim over the whole overlay area.
        Alpha is driven by _scrim_alpha so it participates in open/close animations."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, self._scrim_alpha))
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
        horHeaders = ["", "Initials", "Name", "Role", "Created", "Accessed", "Actions"]
        self.table.clear()
        self.table.setRowCount(len(user_info))
        self.table.setColumnCount(len(horHeaders))
        self.table.setHorizontalHeaderLabels(horHeaders)

        # Initialise col-0 header icon to match the reset all_selected = False state
        _chk_hdr = QtWidgets.QTableWidgetItem("")
        _chk_hdr.setIcon(
            QtGui.QIcon(
                os.path.join(Architecture.get_path(), "QATCH", "icons", "unselect-multiple.svg")
            )
        )
        self.table.setHorizontalHeaderItem(0, _chk_hdr)

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
            combo_container.setStyleSheet("background: transparent;")
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

            icon_size_table = QtCore.QSize(16, 16)

            btn_pwd = QtWidgets.QPushButton("")
            btn_pwd.setIcon(QtGui.QIcon(self.ICON_PWD))
            btn_pwd.setIconSize(icon_size_table)
            btn_pwd.setToolTip("Reset Password")
            btn_pwd.setFixedSize(28, 28)
            btn_pwd.setCursor(QtCore.Qt.PointingHandCursor)
            btn_pwd.setStyleSheet("""
                QPushButton {
                    background: rgba(255, 193, 7, 0.12);
                    border: 1px solid rgba(255, 193, 7, 0.5);
                    border-radius: 14px;
                }
                QPushButton:hover { background: rgba(255, 193, 7, 0.28); }
                QPushButton:pressed { background: rgba(255, 193, 7, 0.45); }
            """)
            btn_pwd.clicked.connect(lambda _, i=initials: self.change_password(i))

            btn_audit = QtWidgets.QPushButton("")
            btn_audit.setIcon(QtGui.QIcon(self.ICON_AUDIT))
            btn_audit.setIconSize(icon_size_table)
            btn_audit.setToolTip("Audit User")
            btn_audit.setFixedSize(28, 28)
            btn_audit.setCursor(QtCore.Qt.PointingHandCursor)
            btn_audit.setStyleSheet("""
                QPushButton {
                    background: rgba(108, 117, 125, 0.12);
                    border: 1px solid rgba(108, 117, 125, 0.45);
                    border-radius: 14px;
                }
                QPushButton:hover { background: rgba(108, 117, 125, 0.28); }
                QPushButton:pressed { background: rgba(108, 117, 125, 0.45); }
            """)
            btn_audit.clicked.connect(lambda _, i=initials: self.audit_selected([i]))

            btn_delete = QtWidgets.QPushButton("")
            btn_delete.setIcon(QtGui.QIcon(self.ICON_DELETE))
            btn_delete.setIconSize(icon_size_table)
            btn_delete.setToolTip("Delete User")
            btn_delete.setFixedSize(28, 28)
            btn_delete.setCursor(QtCore.Qt.PointingHandCursor)
            btn_delete.setStyleSheet("""
                QPushButton {
                    background: rgba(220, 53, 69, 0.12);
                    border: 1px solid rgba(220, 53, 69, 0.45);
                    border-radius: 14px;
                }
                QPushButton:hover { background: rgba(220, 53, 69, 0.28); }
                QPushButton:pressed { background: rgba(220, 53, 69, 0.45); }
            """)
            btn_delete.clicked.connect(lambda _, i=initials: self.delete_selected([i]))

            action_layout.addStretch()
            action_layout.addWidget(btn_pwd)
            action_layout.addWidget(btn_audit)
            action_layout.addWidget(btn_delete)
            action_layout.addStretch()
            self.table.setCellWidget(row, 6, action_widget)

        header = self.table.horizontalHeader()
        # Checkbox col — fixed narrow
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)
        header.resizeSection(0, 36)
        # Initials — snug to content
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        # Name — takes up remaining space
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        # Role combo — fixed wide enough for text + dropdown
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Fixed)
        header.resizeSection(3, 155)
        # Created / Accessed — snug to content
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)
        # Actions — fixed width for three 28px buttons + padding
        header.setSectionResizeMode(6, QtWidgets.QHeaderView.Fixed)
        header.resizeSection(6, 115)

        # Comfortable row height so cell widgets breathe
        self.table.verticalHeader().setDefaultSectionSize(54)

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

            # Icon reflects the NEW state: when all selected show unselect option, and vice-versa
            icon_path = (
                os.path.join(Architecture.get_path(), "QATCH", "icons", "select-multiple.svg")
                if self.all_selected
                else os.path.join(
                    Architecture.get_path(), "QATCH", "icons", "unselect-multiple.svg"
                )
            )

            header_item = QtWidgets.QTableWidgetItem("")
            header_item.setIcon(QtGui.QIcon(icon_path))
            self.table.setHorizontalHeaderItem(0, header_item)

            self.table.blockSignals(True)
            state = QtCore.Qt.Checked if self.all_selected else QtCore.Qt.Unchecked
            for row in range(self.table.rowCount()):
                if not self.table.isRowHidden(row):
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

            # Reject duplicate initials (case-insensitive, skip self)
            _, all_user_info = UserProfiles.get_all_user_info()
            taken = [i[1].upper() for i in all_user_info]
            if new_val in taken and new_val != original_initials.upper():
                # Flash the cell red to signal the conflict
                item.setBackground(QtGui.QBrush(QtGui.QColor(220, 53, 69, 80)))
                item.setForeground(QtGui.QBrush(QtGui.QColor(180, 0, 0)))
                PopUp.warning(
                    self,
                    "Duplicate Initials",
                    f"Initials '{new_val}' are already in use.\nPlease choose a unique value.",
                )
                # Revert to original without re-triggering the signal
                self.table.blockSignals(True)
                item.setText(original_initials)
                item.setBackground(QtGui.QBrush())
                item.setForeground(QtGui.QBrush())
                self.table.blockSignals(False)
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
            bg_color = "rgba(220, 53, 69, 0.12)"
            hover_bg = "rgba(220, 53, 69, 0.22)"
            border_color = "rgba(220, 53, 69, 0.45)"
            text_color = "#C82333"
        elif "OPERATE" in role_upper:
            bg_color = "rgba(40, 167, 69, 0.12)"
            hover_bg = "rgba(40, 167, 69, 0.22)"
            border_color = "rgba(40, 167, 69, 0.45)"
            text_color = "#1E7E34"
        elif "CAPTURE" in role_upper:
            bg_color = "rgba(255, 193, 7, 0.12)"
            hover_bg = "rgba(255, 193, 7, 0.22)"
            border_color = "rgba(255, 193, 7, 0.50)"
            text_color = "#B38600"
        elif "ANALYZE" in role_upper:
            bg_color = "rgba(111, 66, 193, 0.12)"
            hover_bg = "rgba(111, 66, 193, 0.22)"
            border_color = "rgba(111, 66, 193, 0.45)"
            text_color = "#6F42C1"
        else:
            bg_color = "rgba(108, 117, 125, 0.12)"
            hover_bg = "rgba(108, 117, 125, 0.22)"
            border_color = "rgba(108, 117, 125, 0.45)"
            text_color = "#495057"

        combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {bg_color};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 10px;
                padding: 2px 24px 2px 10px;
                font-weight: bold;
                font-size: 11px;
            }}
            QComboBox:hover {{
                background-color: {hover_bg};
                border: 1px solid {text_color};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border: none;
                border-left: 1px solid {border_color};
                border-top-right-radius: 10px;
                border-bottom-right-radius: 10px;
            }}
            QComboBox::down-arrow {{
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {text_color};
                width: 0; height: 0;
            }}
            QComboBox QAbstractItemView {{
                background-color: rgba(255, 255, 255, 250);
                border: 1px solid {border_color};
                border-radius: 6px;
                outline: none;
                selection-background-color: transparent;
                selection-color: #333;
                color: #333;
                padding: 2px;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 4px 10px;
                color: #333;
                background-color: transparent;
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: rgba(10, 163, 230, 25);
                color: #1a1a2e;
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: transparent;
                color: #333;
            }}
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
        self._animate_table_transition(self._do_back_to_users, direction="right")

    def _do_back_to_users(self):
        self.is_audit_mode = False
        self.view_title.setText("User Management")
        self.view_title.setVisible(False)
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

        self._animate_table_transition(
            lambda: self._do_audit_selected(initials_list), direction="left"
        )

    def _do_audit_selected(self, initials_list):
        self.is_audit_mode = True

        display_names = ", ".join(initials_list[:5])
        if len(initials_list) > 5:
            display_names += ", ..."
        title = f"Audit Log: {display_names}"

        self.view_title.setText(title)
        self.view_title.setVisible(True)

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
            # Defer popup so the animation can finish playing smoothly
            QtCore.QTimer.singleShot(
                400,
                lambda: PopUp.warning(
                    self, "Audit Result", "There are security failures in this unified audit log!"
                ),
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

            # change_record entries (secure_user_info) have no time attribute of their own.
            # Use the "[empty]" sentinel so the immediately following timestamp element
            # can backfill raw_time and time_display (see the timestamp branch above).
            if change_record:
                raw_t = "[empty]"
                disp_t = "[pending]"
            else:
                raw_t = time_str
                disp_t = f"*{time_str}*" if is_err else time_str

            entries.append(
                {
                    "raw_time": raw_t,
                    "time_display": disp_t,
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
        """Stage 1: gather target list then hand off to inline symbolic confirmation."""
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

        self._show_delete_confirmation(initials_list)

    def _show_delete_confirmation(self, initials_list):
        """Tint the full row red via viewport overlays then show confirmation.

        Single delete  → animated slide-in: action cell is replaced with Cancel / Confirm
                         buttons that expand left from width=0, replacing all action buttons.
        Multi delete   → action cells remain; a Cancel / Confirm pair slides into the top
                         bar from the right, replacing the delete button."""
        self._pending_delete = list(initials_list)
        is_bulk = len(initials_list) > 1

        for row in range(self.table.rowCount()):
            init_item = self.table.item(row, 1)
            if not (init_item and init_item.text() in initials_list):
                continue

            if not is_bulk:
                # Single: animated expand-from-right confirm widget in the action cell
                self._install_single_delete_confirm(row, initials_list)

        # Overlay the full row(s) in red — works for both single and bulk, and is
        # immune to QSS alternating-row overrides and cell-widget child coverage.
        self._add_delete_row_overlays(initials_list)

        if is_bulk:
            self._show_topbar_delete_confirm(initials_list)

    # ------------------------------------------------------------------
    # Single-delete animated confirm widget
    # ------------------------------------------------------------------

    def _install_single_delete_confirm(self, row, initials_list):
        """Replace col-6 with Cancel+Confirm buttons that slide in from the right."""
        confirm_w = QtWidgets.QWidget()
        confirm_w.setStyleSheet("background: transparent;")
        cl = QtWidgets.QHBoxLayout(confirm_w)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.setSpacing(8)

        btn_cancel = QtWidgets.QPushButton("")
        btn_cancel.setIcon(QtGui.QIcon(self.ICON_CLEAR))
        btn_cancel.setIconSize(QtCore.QSize(12, 12))
        btn_cancel.setFixedSize(26, 26)
        btn_cancel.setCursor(QtCore.Qt.PointingHandCursor)
        btn_cancel.setToolTip("Cancel")
        btn_cancel.setStyleSheet("""
            QPushButton {
                background: rgba(108,117,125,0.15);
                border: 1px solid rgba(108,117,125,0.40);
                border-radius: 13px;
            }
            QPushButton:hover  { background: rgba(108,117,125,0.30); }
            QPushButton:pressed{ background: rgba(108,117,125,0.50); }
        """)
        btn_cancel.clicked.connect(self._cancel_delete)

        btn_confirm = QtWidgets.QPushButton("")
        btn_confirm.setIcon(QtGui.QIcon(self.ICON_DELETE))
        btn_confirm.setIconSize(QtCore.QSize(12, 12))
        btn_confirm.setFixedSize(26, 26)
        btn_confirm.setCursor(QtCore.Qt.PointingHandCursor)
        btn_confirm.setToolTip("Confirm delete")
        btn_confirm.setStyleSheet("""
            QPushButton {
                background: rgba(220,53,69,0.18);
                border: 1px solid rgba(220,53,69,0.50);
                border-radius: 13px;
            }
            QPushButton:hover  { background: rgba(220,53,69,0.35); }
            QPushButton:pressed{ background: rgba(220,53,69,0.55); }
        """)
        btn_confirm.clicked.connect(
            lambda _c=False, il=list(initials_list): self._confirm_delete(il)
        )

        # Right-aligned so the buttons appear from the right edge as width expands left
        cl.addStretch()
        cl.addWidget(btn_cancel)
        cl.addWidget(btn_confirm)

        # Start collapsed — the setCellWidget swap is invisible at width 0
        confirm_w.setMaximumWidth(0)
        self.table.setCellWidget(row, 6, confirm_w)

        col_width = self.table.columnWidth(6)
        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(220)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim.setStartValue(0)
        anim.setEndValue(col_width)
        anim.valueChanged.connect(lambda v: confirm_w.setMaximumWidth(int(v)))
        anim.start()
        self._single_delete_anim = anim  # keep reference alive

    # ------------------------------------------------------------------
    # Full-row red overlay helpers (viewport widgets, immune to QSS)
    # ------------------------------------------------------------------

    def _add_delete_row_overlays(self, initials_list):
        """Place a semi-transparent red overlay widget over every affected row.

        Overlays are children of the table viewport so they clip correctly, and
        have WA_TransparentForMouseEvents so all clicks pass through to the table.
        They are tracked in self._delete_overlays as (widget, row) tuples so they
        can be repositioned on scroll and removed on cancel/confirm."""
        self._remove_delete_overlays()
        vp = self.table.viewport()
        self._delete_overlays = []  # list of (QWidget overlay, int row)

        for row in range(self.table.rowCount()):
            if self.table.isRowHidden(row):
                continue
            init_item = self.table.item(row, 1)
            if not (init_item and init_item.text() in initials_list):
                continue

            row_rect = self._get_row_full_rect(row)
            if not row_rect.isValid():
                continue

            overlay = QtWidgets.QWidget(vp)
            overlay.setGeometry(row_rect)
            overlay.setStyleSheet("background: rgba(220, 53, 69, 50);")
            overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
            overlay.show()
            overlay.raise_()
            self._delete_overlays.append((overlay, row))

        # Wire scroll repositioning once
        if not getattr(self, "_overlay_scroll_connected", False):
            self.table.verticalScrollBar().valueChanged.connect(self._reposition_delete_overlays)
            self.table.horizontalScrollBar().valueChanged.connect(self._reposition_delete_overlays)
            self._overlay_scroll_connected = True

    def _get_row_full_rect(self, row):
        """Return the bounding rect of the entire row in viewport coordinates."""
        model = self.table.model()
        n_cols = self.table.columnCount()
        # Find first visible column
        first_col = next((c for c in range(n_cols) if not self.table.isColumnHidden(c)), None)
        last_col = next(
            (c for c in range(n_cols - 1, -1, -1) if not self.table.isColumnHidden(c)), None
        )
        if first_col is None or last_col is None:
            return QtCore.QRect()
        r1 = self.table.visualRect(model.index(row, first_col))
        r2 = self.table.visualRect(model.index(row, last_col))
        return r1.united(r2)

    def _reposition_delete_overlays(self):
        """Called on table scroll — move each overlay to its row's current position."""
        for overlay, row in getattr(self, "_delete_overlays", []):
            if self.table.isRowHidden(row):
                overlay.setVisible(False)
            else:
                row_rect = self._get_row_full_rect(row)
                overlay.setGeometry(row_rect)
                overlay.setVisible(row_rect.isValid())

    def _remove_delete_overlays(self):
        """Destroy all pending-delete row overlays."""
        for overlay, _row in getattr(self, "_delete_overlays", []):
            overlay.deleteLater()
        self._delete_overlays = []

    # ------------------------------------------------------------------
    # Top-bar bulk-delete confirmation (slides in / out)
    # ------------------------------------------------------------------

    def _show_topbar_delete_confirm(self, initials_list):
        """Append a Cancel + Confirm widget to the top bar and animate its width 0 → full.
        The QHBoxLayout stretch contracts, so the existing action buttons shift left."""
        icon_sz = QtCore.QSize(18, 18)

        btn_cancel = QtWidgets.QPushButton("", self.taskbar_frame)
        btn_cancel.setIcon(QtGui.QIcon(self.ICON_CLEAR))
        btn_cancel.setIconSize(icon_sz)
        btn_cancel.setToolTip("Cancel")
        btn_cancel.setCursor(QtCore.Qt.PointingHandCursor)
        btn_cancel.setStyleSheet("""
            QPushButton {
                background: rgba(108,117,125,0.15);
                border: 1px solid rgba(108,117,125,0.40);
                border-radius: 17px;
                min-width: 34px; max-width: 34px;
                min-height: 34px; max-height: 34px;
            }
            QPushButton:hover  { background: rgba(108,117,125,0.30);
                                 border: 1px solid rgba(108,117,125,0.65); }
            QPushButton:pressed{ background: rgba(108,117,125,0.50); }
        """)
        btn_cancel.clicked.connect(self._cancel_delete)

        btn_confirm = QtWidgets.QPushButton("", self.taskbar_frame)
        btn_confirm.setIcon(QtGui.QIcon(self.ICON_DELETE))
        btn_confirm.setIconSize(icon_sz)
        btn_confirm.setToolTip(f"Confirm delete  ({len(initials_list)} users)")
        btn_confirm.setCursor(QtCore.Qt.PointingHandCursor)
        btn_confirm.setStyleSheet("""
            QPushButton {
                background: rgba(220,53,69,0.18);
                border: 1px solid rgba(220,53,69,0.50);
                border-radius: 17px;
                min-width: 34px; max-width: 34px;
                min-height: 34px; max-height: 34px;
            }
            QPushButton:hover  { background: rgba(220,53,69,0.35);
                                 border: 1px solid rgba(220,53,69,0.80); }
            QPushButton:pressed{ background: rgba(220,53,69,0.55); }
        """)
        btn_confirm.clicked.connect(
            lambda _c=False, il=list(initials_list): self._confirm_delete(il)
        )

        # Single container so we animate one widget width instead of two
        container = QtWidgets.QWidget(self.taskbar_frame)
        container.setStyleSheet("background: transparent;")
        cl = QtWidgets.QHBoxLayout(container)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(self.top_bar.spacing())
        cl.addWidget(btn_cancel)
        cl.addWidget(btn_confirm)

        # Start collapsed so the slide-in is visible
        container.setMaximumWidth(0)
        self.top_bar.addWidget(container)
        self._topbar_confirm_widget = container

        target_w = 34 + self.top_bar.spacing() + 34  # cancel + gap + confirm

        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(220)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim.setStartValue(0)
        anim.setEndValue(target_w)
        anim.valueChanged.connect(lambda v: container.setMaximumWidth(int(v)))
        anim.start()
        self._topbar_confirm_anim = anim  # keep reference alive

    def _hide_topbar_delete_confirm(self):
        """Collapse and remove the top-bar confirmation container (if present)."""
        widget = getattr(self, "_topbar_confirm_widget", None)
        if widget is None:
            return
        # Stop any in-progress expand animation
        anim_in = getattr(self, "_topbar_confirm_anim", None)
        if anim_in and anim_in.state() == QtCore.QAbstractAnimation.Running:
            anim_in.stop()

        start_w = widget.maximumWidth()
        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(160)
        anim.setEasingCurve(QtCore.QEasingCurve.InCubic)
        anim.setStartValue(start_w)
        anim.setEndValue(0)
        anim.valueChanged.connect(lambda v: widget.setMaximumWidth(int(v)))

        def _done():
            self.top_bar.removeWidget(widget)
            widget.deleteLater()
            self._topbar_confirm_widget = None

        anim.finished.connect(_done)
        anim.start()
        self._topbar_confirm_anim = anim

    # ------------------------------------------------------------------
    # Delete confirmation callbacks
    # ------------------------------------------------------------------

    def _cancel_delete(self):
        """Discard the pending confirmation and restore normal state."""
        self._pending_delete = []
        self._hide_topbar_delete_confirm()
        self.update_table_data()

    def _confirm_delete(self, initials_list):
        """Stage 2: confirmed — collapse the top-bar widget then run row animation."""
        self._pending_delete = []
        self._hide_topbar_delete_confirm()
        self._animate_delete_rows(initials_list)

    def _animate_delete_rows(self, initials_list):
        """Phase 1: deepen the red wash on the full row (text + widget cells).
        Phase 2: after 160 ms collapse each row height 0 (bottom-to-top visual).
        Phase 3: archive files and refresh."""
        affected_rows = [
            row
            for row in range(self.table.rowCount())
            if self.table.item(row, 1) and self.table.item(row, 1).text() in initials_list
        ]

        if not affected_rows:
            self._execute_delete(initials_list)
            return

        # Phase 1 — deepen tint across all cell types
        for row in affected_rows:
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item:
                    item.setBackground(QtGui.QBrush(QtGui.QColor(220, 53, 69, 90)))
            role_cw = self.table.cellWidget(row, 3)
            if role_cw:
                role_cw.setStyleSheet("background: rgba(220, 53, 69, 55);")
            cw = self.table.cellWidget(row, 6)
            if cw:
                cw.setStyleSheet("background: rgba(220, 53, 69, 55);")

        # Phase 2 — collapse rows simultaneously after the flash settles
        row_height = self.table.rowHeight(affected_rows[0])
        self._delete_anims = []
        completed = [0]
        total = len(affected_rows)

        for row in affected_rows:
            anim = QtCore.QVariantAnimation(self)
            anim.setDuration(280)
            anim.setEasingCurve(QtCore.QEasingCurve.InQuart)
            anim.setStartValue(float(row_height))
            anim.setEndValue(0.0)

            def _make_step(r):
                def _step(v):
                    self.table.setRowHeight(r, max(0, int(v)))

                return _step

            def _make_finish():
                def _finish():
                    completed[0] += 1
                    if completed[0] >= total:
                        self._execute_delete(initials_list)

                return _finish

            anim.valueChanged.connect(_make_step(row))
            anim.finished.connect(_make_finish())
            self._delete_anims.append(anim)

        QtCore.QTimer.singleShot(160, lambda: [a.start() for a in self._delete_anims])

    def _execute_delete(self, initials_list):
        """Archive user files after the animation completes; self-admin edge case preserved."""
        admin_initials = UserProfiles.get_user_info(self.admin_file)[1]

        for init in initials_list:
            if init == admin_initials:
                if UserProfiles.count() > len(initials_list):
                    PopUp.warning(
                        self,
                        "Action Denied",
                        "Cannot delete active ADMIN user while other accounts remain."
                        " Aborting ADMIN deletion.",
                    )
                    continue
                else:
                    if PopUp.question(
                        self,
                        "Remove Last User",
                        "WARNING: You are deleting the active ADMIN account.\n"
                        "This will sign you out and cannot be undone.\nProceed?",
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
