"""
Things that I need to update:
- Refresh should blur users when loading
- add filter button filter by creation dates, accessed dates, and roles

- fix full screen animation to be smooth
- Closing user management from full screen also needs the close animation.

- Initials, name, role, created, and accessed should have sort by options on the header row
- Make name and initials fields more clearly editable

- Bug in audit view where if the 'Timestamp' cell is double clicked it brings up a select all option.

- Optional: pretty print creation dates, access dates and audit logs
"""

import datetime as dt
import hashlib
import os
from xml.dom import minidom

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.common.fileManager import FileManager
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserConstants, UserProfiles
from QATCH.core.constants import Constants, UserRoles
from QATCH.ui.components import (
    AnimatedComboBox,
    GlassLineEdit,
    GlassPushButton,
    GlassToggle,
)
from QATCH.ui.popUp import PopUp
from QATCH.ui.widgets.reset_password_widget import ResetPasswordWidget

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

            # --- NEW: Pointing hand cursor for the select/unselect all header ---
            self.horizontalHeader().setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

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
        self.ICON_USERS = os.path.join(Architecture.get_path(), "QATCH", "icons", "users.svg")
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
        # WA_TranslucentBackground is top-level windows only - on a child widget
        # it makes the whole widget invisible. For child widgets the correct
        # approach is to disable Qt's auto-fill so our paintEvent draws the scrim
        # directly over the parent's backing store.
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
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
        # NOTE: intentionally no _apply_shadow here - QGraphicsDropShadowEffect on
        # a parent frame causes native QComboBox popups to render with a square clip
        # shadow. The border above provides sufficient visual separation.

        # Animation state
        self._scrim_alpha = 0  # paintEvent reads this; animated on open/close
        self._panel_alpha = 215  # glass_frame background alpha; fixed per fullscreen state
        self._closing = False  # guards closeEvent re-entry during fade-out
        self._close_in_progress = False
        self._fade_anim = None  # single reusable open/close fade (like DataManagementWidget)

        # Fade via a composited opacity effect (animating the property is a
        # paint-time multiply) instead of re-setting the stylesheet each frame,
        # which would reparse + repolish the entire glass subtree per frame.
        # Start at 0.0 so the very first paint after glass_frame.show() in
        # _reveal() is transparent — prevents the one-frame white flash that
        # occurs between show() and _animate_open() setting opacity to 0.
        self._glass_opacity = QtWidgets.QGraphicsOpacityEffect(self.glass_frame)
        self._glass_opacity.setOpacity(0.0)
        self.glass_frame.setGraphicsEffect(self._glass_opacity)

        self.main_layout = QtWidgets.QVBoxLayout(self.glass_frame)
        self.main_layout.setContentsMargins(20, 12, 20, 20)
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
        # NOTE: no QGraphicsDropShadowEffect here - glass_frame (an ancestor)
        # carries a QGraphicsOpacityEffect for fading, and Qt doesn't compose
        # nested widget graphics effects reliably (see data_management_widget.py).
        # The border above provides sufficient visual separation.
        self.header_layout = QtWidgets.QHBoxLayout()
        self.header_layout.setContentsMargins(0, 0, 0, 0)

        self.window_icon_label = QtWidgets.QLabel()
        self.window_icon_label.setPixmap(QtGui.QIcon(self.ICON_USERS).pixmap(16, 16))

        self.window_title_label = QtWidgets.QLabel("User Management")
        self.window_title_label.setStyleSheet("""
            QLabel {
                color: #333;
                font-weight: bold;
                font-size: 13px;
                background: transparent;
            }
        """)

        self.header_layout.addWidget(self.window_icon_label)
        self.header_layout.addWidget(self.window_title_label)
        self.header_layout.addStretch()
        self.top_bar = QtWidgets.QHBoxLayout(self.taskbar_frame)
        self.top_bar.setContentsMargins(20, 0, 20, 0)
        self.top_bar.setSpacing(15)

        self.btn_back = GlassPushButton(" Back", variant="default")
        self.btn_back.setIcon(QtGui.QIcon(self.ICON_BACK))
        self.btn_back.setIconSize(QtCore.QSize(16, 16))
        self.btn_back.setFixedHeight(34)
        self.btn_back.setVisible(False)
        self.btn_back.clicked.connect(self.back_to_users)

        # --- Search Bar ---
        self.search_bar = GlassLineEdit()
        self.search_bar.setPlaceholderText("Search Initials or Name...")
        self.search_bar.addAction(
            QtGui.QIcon(self.ICON_SEARCH), QtWidgets.QLineEdit.LeadingPosition
        )
        self.search_bar.setFixedWidth(250)
        self.search_bar.setFixedHeight(34)
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

        # Ensure icon size scales well inside the 34x34 button
        icon_size = QtCore.QSize(18, 18)

        self.btn_add = GlassPushButton(" Add", variant="default")
        self.btn_add.setIcon(QtGui.QIcon(self.ICON_ADD))
        self.btn_add.setIconSize(icon_size)
        self.btn_add.setToolTip("Add New User")
        self.btn_add.setFixedHeight(34)
        self.btn_add.clicked.connect(self.add_user)

        self.btn_audit_selected = GlassPushButton(" Audit", variant="default")
        self.btn_audit_selected.setIcon(QtGui.QIcon(self.ICON_AUDIT))
        self.btn_audit_selected.setIconSize(icon_size)
        self.btn_audit_selected.setToolTip("Audit Selected Users")
        self.btn_audit_selected.setFixedHeight(34)
        self.btn_audit_selected.clicked.connect(lambda: self.audit_selected())

        self.btn_delete_selected = GlassPushButton(" Delete", variant="danger")
        self.btn_delete_selected.setIcon(QtGui.QIcon(self.ICON_DELETE))
        self.btn_delete_selected.setIconSize(icon_size)
        self.btn_delete_selected.setToolTip("Delete Selected Users")
        self.btn_delete_selected.setFixedHeight(34)
        self.btn_delete_selected.clicked.connect(lambda: self.delete_selected())

        self.btn_refresh = GlassPushButton(" Refresh", variant="default")
        self.btn_refresh.setIcon(QtGui.QIcon(self.ICON_REFRESH))
        self.btn_refresh.setIconSize(icon_size)
        self.btn_refresh.setToolTip("Refresh Table")
        self.btn_refresh.setFixedHeight(34)
        self.btn_refresh.clicked.connect(self._animate_refresh_spin)

        self._default_margin_pct = 0.175
        button_size = 28
        icon_size = QtCore.QSize(14, 14)

        # Fullscreen Toggle (Transparent icon button - matches create_user_widget btn_close style)
        _FS_NORMAL = QtGui.QColor(110, 120, 130, 190)
        _FS_HOVER = QtGui.QColor(185, 190, 200, 230)
        self._fs_normal_icon = self._tinted_icon(self.ICON_EXPAND, _FS_NORMAL, size=14)
        self._fs_hover_icon = self._tinted_icon(self.ICON_EXPAND, _FS_HOVER, size=14)

        self.btn_fullscreen = QtWidgets.QPushButton("", self)
        self.btn_fullscreen.setFixedSize(button_size, button_size)
        self.btn_fullscreen.setIcon(self._fs_normal_icon)
        self.btn_fullscreen.setIconSize(icon_size)
        self.btn_fullscreen.setToolTip("Toggle Fullscreen")
        self.btn_fullscreen.setStyleSheet("""
            QPushButton { background: transparent; border: none; }
        """)
        self.btn_fullscreen.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_fullscreen.installEventFilter(self)
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)

        # Close Window (Transparent icon button - matches create_user_widget btn_close style)
        self.btn_close = QtWidgets.QPushButton("x", self)
        self.btn_close.setFixedSize(button_size, button_size)
        self.btn_close.setToolTip("Close")
        self.btn_close.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: rgba(110, 120, 130, 190);
                font-size: 18px;
                font-weight: bold;
                padding-bottom: 2px;
            }
            QPushButton:hover   { color: rgba(210, 55, 55, 230); }
            QPushButton:pressed { color: rgba(160, 30, 30, 255); }
        """)
        self.btn_close.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_close.clicked.connect(self.close)
        self.btn_close.raise_()
        # Assemble taskbar - add, audit, refresh, delete (no close; panel click-outside handles dismiss)
        self.top_section_layout = QtWidgets.QVBoxLayout()
        self.top_section_layout.setSpacing(10)

        self.top_section_layout.addLayout(self.header_layout)
        self.top_section_layout.addWidget(self.taskbar_frame)
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
        self.table.viewport().installEventFilter(self)
        _container_layout = QtWidgets.QVBoxLayout(self.table_container)
        _container_layout.setContentsMargins(0, 0, 0, 0)
        _container_layout.setSpacing(0)
        _container_layout.addWidget(self.table)

        # --- Bottom Settings Bar ---
        self.settings_layout = QtWidgets.QHBoxLayout()
        self.settings_layout.setSpacing(0)

        # ── Developer Mode toggle ──────────────────────────────────────
        enabled, error, expires = UserProfiles.checkDevMode()

        self.developerModeChk = GlassToggle()
        # setChecked before connecting so the handler doesn't fire on init
        self.developerModeChk.setChecked(enabled)

        dev_label_color = "#D32F2F" if error else ("#388E3C" if enabled else "#555")
        self.dev_mode_label = QtWidgets.QLabel("Enable Developer Mode")
        self.dev_mode_label.setStyleSheet(f"""
            QLabel {{
                color: {dev_label_color};
                font-weight: bold;
                font-size: 11px;
                background: transparent;
            }}
        """)

        # Inline expiry label - shown only when dev mode is active
        expiry_text = ""
        expiry_color = "#D32F2F" if error else "#388E3C"
        if enabled and expires:
            expiry_text = f"Expired: {expires}" if error else f"Expires: {expires}"
        self.dev_expiry_label = QtWidgets.QLabel(expiry_text)
        self.dev_expiry_label.setVisible(bool(expiry_text))
        self.dev_expiry_label.setStyleSheet(f"""
            QLabel {{
                color: {expiry_color};
                font-size: 10px;
                background: transparent;
                padding-left: 6px;
            }}
        """)

        # Connect after initial state is fully applied
        self.developerModeChk.toggled.connect(self.toggle_dev_mode)

        # ── Require Admin for Updates toggle ──────────────────────────
        self.reqAdminUpd_chkbox = GlassToggle()
        self.reqAdminUpd_chkbox.setChecked(UserConstants.REQ_ADMIN_UPDATES)
        self.reqAdminUpd_chkbox.toggled.connect(self.toggle_req_admin_updates)

        self.admin_label = QtWidgets.QLabel("Require Admin for Updates")
        self.admin_label.setStyleSheet("""
            QLabel {
                color: #555;
                font-size: 11px;
                background: transparent;
            }
        """)

        # ── Assemble ─────────────────────────────────────────────────
        # Dev mode group - absorbs all leftover space so the admin group
        # stays pinned to the right regardless of expiry label visibility.
        # Parent these grouping widgets to glass_frame at construction. A
        # parentless QWidget is a top-level window, and setting
        # WA_TranslucentBackground on a top-level widget forces Qt to realize its
        # native window backing IMMEDIATELY - before the widget is reparented
        # into settings_layout. That momentary translucent top-level is the
        # empty window frame that flashes on open. Giving them a parent up front
        # means they are in-layout children from the start and never get a
        # native window of their own. (The reference DataManagementWidget never
        # creates parentless translucent intermediates, which is why it doesn't
        # exhibit the flash.)
        dev_group = QtWidgets.QWidget(self.glass_frame)
        dev_group.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        dev_layout = QtWidgets.QHBoxLayout(dev_group)
        dev_layout.setContentsMargins(0, 0, 0, 0)
        dev_layout.setSpacing(0)
        dev_layout.addWidget(self.developerModeChk)
        dev_layout.addSpacing(8)
        dev_layout.addWidget(self.dev_mode_label)
        dev_layout.addSpacing(6)
        dev_layout.addWidget(self.dev_expiry_label)
        dev_layout.addStretch()  # internal stretch - soaks up excess left-side space

        # Admin group - fixed to the right edge
        admin_group = QtWidgets.QWidget(self.glass_frame)
        admin_group.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        admin_layout = QtWidgets.QHBoxLayout(admin_group)
        admin_layout.setContentsMargins(0, 0, 0, 0)
        admin_layout.setSpacing(0)
        admin_layout.addWidget(self.admin_label)
        admin_layout.addSpacing(8)
        admin_layout.addWidget(self.reqAdminUpd_chkbox)

        # Stretch factor 1 on dev_group, 0 on admin_group - dev expands,
        # admin never moves.
        self.settings_layout.addWidget(dev_group, 1)
        self.settings_layout.addWidget(admin_group, 0)
        self.main_layout.addLayout(self.top_section_layout)

        self.main_layout.addWidget(self.table_container)
        self.main_layout.addLayout(self.settings_layout)

        self.base_layout.addWidget(self.glass_frame)
        self.setLayout(self.base_layout)

        self.update_table_data()

        # Start hidden and pre-sized to the parent's content area. If the widget
        # is ever shown without its geometry first being set, Qt briefly paints
        # it as a small top-level window (the "dialog flash"). Fitting here, at
        # construction, means the geometry is already correct before any show.
        # The glass panel is also kept hidden until the reveal tick so no
        # unsettled frame of it can paint.
        self._revealed = False
        self.hide()
        self._scrim_alpha = 0
        self.glass_frame.hide()
        self._refit_to_parent()

    def _apply_shadow(self, widget, blur_radius=15, alpha=40, offset=(0, 4)):
        """Helper method to add depth to glass components."""
        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(blur_radius)
        shadow.setColor(QtGui.QColor(0, 0, 0, alpha))
        shadow.setOffset(offset[0], offset[1])
        widget.setGraphicsEffect(shadow)

    @staticmethod
    def _tinted_icon(path: str, color: QtGui.QColor, size: int = 14) -> QtGui.QIcon:
        """Returns a copy of the icon at *path* fully painted in *color*.

        Uses SourceAtop composition so the tint respects the original
        alpha channel - transparent SVG areas stay transparent.
        """
        src = QtGui.QIcon(path).pixmap(size, size)
        dst = QtGui.QPixmap(src.size())
        dst.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(dst)
        p.drawPixmap(0, 0, src)
        p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
        p.fillRect(dst.rect(), color)
        p.end()
        return QtGui.QIcon(dst)

    def _animate_refresh_spin(self):
        """Animates a single 360-degree spin on the refresh icon, smooth with parabolic acceleration."""
        if (
            hasattr(self, "refresh_anim")
            and self.refresh_anim.state() == QtCore.QAbstractAnimation.Running
        ):
            return

        self.refresh_anim = QtCore.QVariantAnimation(self)
        self.refresh_anim.setDuration(900)  # slow, satisfying rotation
        # InOutCubic provides the parabolic curve: slow start -> fast middle -> slow stop
        self.refresh_anim.setEasingCurve(QtCore.QEasingCurve.InOutCubic)
        self.refresh_anim.setStartValue(0.0)
        self.refresh_anim.setEndValue(360.0)

        orig_pixmap = QtGui.QIcon(self.ICON_REFRESH).pixmap(18, 18)
        w, h = orig_pixmap.width(), orig_pixmap.height()

        def update_icon(angle):
            # Create a fixed-size transparent canvas so the bounding box NEVER expands
            rotated_pixmap = QtGui.QPixmap(w, h)
            rotated_pixmap.fill(QtGui.QColor(0, 0, 0, 0))  # Fully transparent

            # Use QPainter to draw the rotated original inside our fixed canvas
            painter = QtGui.QPainter(rotated_pixmap)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)

            # Move the origin to the center, rotate, and move it back
            painter.translate(w / 2, h / 2)
            painter.rotate(angle)
            painter.translate(-w / 2, -h / 2)

            painter.drawPixmap(0, 0, orig_pixmap)
            painter.end()

            self.btn_refresh.setIcon(QtGui.QIcon(rotated_pixmap))

        def finish_refresh():
            self.btn_refresh.setIcon(QtGui.QIcon(self.ICON_REFRESH))
            self.update_table_data()

        self.refresh_anim.valueChanged.connect(update_icon)
        self.refresh_anim.finished.connect(finish_refresh)
        self.refresh_anim.start()

    def _animate_table_transition(self, update_func, direction="right"):
        """Slides the table content horizontally while swapping data - no processEvents stutter."""
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
        #    No processEvents() - that's the source of the original stutter.
        def _start():
            # Briefly show to force a layout pass + render for the grab, then hide again.
            # This is synchronous - no paint event fires between show and hide.
            self.table_container.setVisible(True)
            new_pixmap = self.table_container.grab()
            self.table_container.setVisible(False)

            # Clipping wrapper: a plain child frame sized exactly to the table
            # area, with clipped children enabled. The sliding pixmaps are
            # parented INTO this wrapper, so anything that slides past the table
            # bounds is masked instead of bleeding over the panel's padded
            # margins and rounded corners (the "persistent edge items" bug).
            clip = QtWidgets.QFrame(self.glass_frame)
            clip.setObjectName("slideClip")
            clip.setStyleSheet("QFrame#slideClip { background: transparent; border: none; }")
            clip.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            clip.setGeometry(tbl_geo)
            clip.setAttribute(QtCore.Qt.WA_StyledBackground, True)
            # Force hard clipping of child widgets to the wrapper's rect.
            clip.setMask(QtGui.QRegion(0, 0, tbl_geo.width(), tbl_geo.height()))
            clip.show()
            clip.raise_()

            w = tbl_geo.width()
            # Inside the clip wrapper everything is in wrapper-local coords,
            # so the resting position is (0, 0).
            rest_pos = QtCore.QPoint(0, 0)
            if direction == "right":
                old_end = QtCore.QPoint(w, 0)
                new_start = QtCore.QPoint(-w, 0)
            else:
                old_end = QtCore.QPoint(-w, 0)
                new_start = QtCore.QPoint(w, 0)

            old_overlay = QtWidgets.QLabel(clip)
            old_overlay.setPixmap(old_pixmap)
            old_overlay.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            old_overlay.setGeometry(QtCore.QRect(rest_pos, tbl_geo.size()))
            old_overlay.show()

            new_overlay = QtWidgets.QLabel(clip)
            new_overlay.setPixmap(new_pixmap)
            new_overlay.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            new_overlay.setGeometry(QtCore.QRect(new_start, tbl_geo.size()))
            new_overlay.show()
            new_overlay.raise_()

            # OutQuint: fast start → smooth deceleration - no mid-animation hitching
            anim_old = QtCore.QPropertyAnimation(old_overlay, b"pos")
            anim_old.setDuration(320)
            anim_old.setEasingCurve(QtCore.QEasingCurve.OutQuint)
            anim_old.setStartValue(rest_pos)
            anim_old.setEndValue(old_end)

            anim_new = QtCore.QPropertyAnimation(new_overlay, b"pos")
            anim_new.setDuration(320)
            anim_new.setEasingCurve(QtCore.QEasingCurve.OutQuint)
            anim_new.setStartValue(new_start)
            anim_new.setEndValue(rest_pos)

            self.transition_group = QtCore.QParallelAnimationGroup(self)
            self.transition_group.addAnimation(anim_old)
            self.transition_group.addAnimation(anim_new)

            def _cleanup():
                # Reveal the real container FIRST so there is never a frame where
                # neither the overlays nor the live table is painted, then drop
                # the whole clip wrapper (and its child overlays) in one shot.
                self.table_container.setVisible(True)
                clip.hide()
                clip.deleteLater()

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
        """Toggle between inset glass panel and full-bleed window.

        Smoothness strategy: animate the REAL glass_frame geometry every frame
        (via base_layout margins) and let the table's flexible columns STRETCH
        continuously so they redistribute fluidly as the width changes, instead
        of jumping. Earlier approaches failed two ways: a scaled pixmap zoomed
        (aspect-ratio stretch distorted every child), and Fixed/ResizeToContents
        columns snapped because they don't interpolate. Setting all flexible
        columns to Stretch for the duration makes the reflow itself the smooth
        animation; the natural column modes are restored once at the end.
        """
        if hasattr(self, "anim") and self.anim.state() == QtCore.QAbstractAnimation.Running:
            self.anim.stop()

        self._is_fullscreen = not getattr(self, "_is_fullscreen", False)

        _icon_path = self.ICON_COLLAPSE if self._is_fullscreen else self.ICON_EXPAND
        self._fs_normal_icon = self._tinted_icon(
            _icon_path, QtGui.QColor(110, 120, 130, 190), size=14
        )
        self._fs_hover_icon = self._tinted_icon(
            _icon_path, QtGui.QColor(185, 190, 200, 230), size=14
        )
        if self.btn_fullscreen.underMouse():
            self.btn_fullscreen.setIcon(self._fs_hover_icon)
        else:
            self.btn_fullscreen.setIcon(self._fs_normal_icon)

        w = self.width()
        h = self.height()
        pct = getattr(self, "_default_margin_pct", 0.175)

        start_mx = self.base_layout.contentsMargins().left()
        start_my = self.base_layout.contentsMargins().top()

        if self._is_fullscreen:
            target_mx, target_my = 0, 0
            start_alpha, target_alpha = getattr(self, "_panel_alpha", 215), 255
            start_radius, target_radius = 12, 0
            start_border, target_border = 1.5, 0.0
        else:
            target_mx = int(w * pct)
            target_my = int(h * pct)
            start_alpha, target_alpha = getattr(self, "_panel_alpha", 255), 215
            start_radius, target_radius = 0, 12
            start_border, target_border = 0.0, 1.5

        # Lock the table into its FINAL column modes for the whole motion so
        # the modes never change (no end-snap). Only the single Stretch column
        # flexes, interpolating smoothly as the frame width animates.
        self._set_columns_fluid()

        self.anim = QtCore.QVariantAnimation(self)
        self.anim.setDuration(280)
        self.anim.setEasingCurve(QtCore.QEasingCurve.InOutCubic)
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)

        def _step(t):
            mx = int(start_mx + (target_mx - start_mx) * t)
            my = int(start_my + (target_my - start_my) * t)
            a = int(start_alpha + (target_alpha - start_alpha) * t)
            r = int(start_radius + (target_radius - start_radius) * t)
            b = start_border + (target_border - start_border) * t
            self._panel_alpha = a
            self.base_layout.setContentsMargins(mx, my, mx, my)
            self.glass_frame.setStyleSheet(f"""
                QFrame#userview {{
                    background: rgba(255, 255, 255, {a});
                    border: {b:.1f}px solid rgba(255, 255, 255, 230);
                    border-radius: {r}px;
                }}
            """)
            self._update_dots_position(mx, my)

        self.anim.valueChanged.connect(_step)

        def _finish():
            self._panel_alpha = target_alpha
            self.base_layout.setContentsMargins(target_mx, target_my, target_mx, target_my)
            self._set_panel_alpha(target_alpha)
            self._restore_column_modes()  # back to natural widths at final size
            self._update_dots_position(target_mx, target_my)

        self.anim.finished.connect(_finish)
        self.anim.start()

    def _set_columns_fluid(self):
        """Apply the FINAL column resize modes for the duration of the motion.

        The key to a snap-free animation is that the column modes never change
        between the motion and the settled state: the fixed/content columns hold
        steady while a single Stretch column absorbs the width delta, so it
        interpolates smoothly as the table resizes. Using uniform Stretch during
        the motion and then restoring different per-column modes at the end
        caused the visible width jump (the "snap"). Reusing the final modes here
        means the animated state and end state are identical - no jump.
        """
        self._restore_column_modes()

    def _restore_column_modes(self):
        """Re-apply the correct header resize modes after a fullscreen animation.

        Columns are frozen to Fixed widths during the animation to suppress
        per-frame Stretch/ResizeToContents recalculation.  This method restores
        them once at the settled final panel size so they expand/contract correctly.
        """
        header = self.table.horizontalHeader()
        if getattr(self, "is_audit_mode", False):
            # Audit view: all columns auto-fit, last one stretches
            header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
            header.setStretchLastSection(True)
        else:
            # User management view: mirror the modes set in update_table_data
            header.setStretchLastSection(False)
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)
            header.resizeSection(0, 36)
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
            header.setSectionResizeMode(3, QtWidgets.QHeaderView.Fixed)
            header.resizeSection(3, 155)
            header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(6, QtWidgets.QHeaderView.Fixed)
            header.resizeSection(6, 115)

    def _apply_margin_step(self, progress):
        """Fired every frame by the expand/collapse animation.

        *progress* is a normalized 0→1 value. Both the horizontal and vertical
        margins are interpolated independently between their true start and
        target pixel values (stashed in toggle_fullscreen). Deriving the
        vertical margin from the horizontal ratio - the old approach - produced
        non-proportional motion and the "strange resizing" on expand.
        """
        start_mx, start_my = getattr(self, "_fs_margin_start", (0, 0))
        target_mx, target_my = getattr(self, "_fs_margin_target", (0, 0))
        mx = int(start_mx + (target_mx - start_mx) * progress)
        my = int(start_my + (target_my - start_my) * progress)
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

    def setVisible(self, visible):
        """Fit to parent and keep the panel hidden until layout settles - kills
        the 'tiny blank dialog' flash on open.

        UIControls calls .show()/.setVisible(True) on this overlay. The entire
        show sequence is deferred to a singleShot(0) so that the overlay and its
        glass panel become visible in one atomic event-loop tick — no intermediate
        frame where the overlay is visible without the panel, which is what caused
        the floating-button flash in earlier versions.
        """
        if visible and not self.isVisible():
            self._scrim_alpha = 0
            self._panel_alpha = 0  # panel starts invisible; fades in
            # Pre-hide the panel and pre-fit geometry while we're still invisible,
            # so the layout has settled before the first paint.
            self.glass_frame.hide()
            self._refit_to_parent()

            def _reveal():
                # Fit once more now that the event loop has had one iteration to
                # settle pending geometry changes from the caller's stack frame.
                self._refit_to_parent()
                _layout = self.layout()
                if _layout is not None:
                    _layout.activate()
                self._revealed = True
                # Guarantee opacity=0 so the glass panel's very first painted
                # frame is transparent regardless of any earlier state change.
                self._set_glass_opacity(0.0)
                # Show the overlay and panel atomically in this same callback so
                # there is no event-loop iteration (and therefore no paint pass)
                # between them — this eliminates the one-frame button-without-panel
                # flash that happened when super().setVisible(True) was called
                # before this singleShot, leaving the overlay visible but panelless
                # while the event loop processed queued paints.
                QtWidgets.QWidget.setVisible(self, True)
                self._refit_to_parent()
                _layout2 = self.layout()
                if _layout2 is not None:
                    _layout2.activate()
                self.glass_frame.show()
                self.glass_frame.raise_()
                self.btn_close.raise_()
                self.btn_fullscreen.raise_()
                self.update()
                self._animate_open()

            QtCore.QTimer.singleShot(0, _reveal)
            return
        super().setVisible(visible)

    def showEvent(self, event):
        """Raise to front. Geometry is fitted and painting is gated by
        setVisible(); the fade is kicked off from there once layout settles."""
        Log.d(f"{TAG} showEvent fired, parent={self.parent}")
        self._refit_to_parent()
        self.raise_()
        super().showEvent(event)

    def hideEvent(self, event):
        super().hideEvent(event)

    def closeEvent(self, event):
        """Intercept close to run a fade-out first; re-enters with _closing=True to accept."""
        if self._closing:
            event.accept()
            return
        anim = getattr(self, "_fade_anim", None)
        if anim is not None and anim.state() == QtCore.QAbstractAnimation.Running:
            # A close fade is already running; let it finish.
            if self._close_in_progress:
                event.ignore()
                return
        event.ignore()
        self._close_in_progress = True
        self._animate_close()

    # ------------------------------------------------------------------
    # Open / close animations
    # ------------------------------------------------------------------

    def _set_glass_opacity(self, frac):
        frac = max(0.0, min(1.0, float(frac)))
        if self._glass_opacity is not None:
            self._glass_opacity.setOpacity(frac)

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

    def _stop_anim(self):
        """Stop and fully tear down any running fade so it can't keep firing."""
        anim = getattr(self, "_fade_anim", None)
        if anim is not None:
            try:
                anim.stop()
                anim.valueChanged.disconnect()
                anim.finished.disconnect()
            except (TypeError, RuntimeError):
                pass
            anim.deleteLater()
            self._fade_anim = None

    def _run_fade(self, scrim_from, scrim_to, op_from, op_to, duration, easing, on_done=None):
        """Animate the scrim alpha (cheap paintEvent) and the glass opacity
        effect (composited) from fixed endpoints. No per-frame stylesheet."""
        self._stop_anim()
        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(duration)
        anim.setEasingCurve(easing)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)

        def _step(t):
            self._scrim_alpha = int(scrim_from + (scrim_to - scrim_from) * t)
            self._set_glass_opacity(op_from + (op_to - op_from) * t)
            self.update()  # repaint the scrim only

        def _settle():
            self._scrim_alpha = int(scrim_to)
            self._set_glass_opacity(op_to)
            self.update()
            done_anim = self._fade_anim
            self._fade_anim = None
            if done_anim is not None:
                done_anim.deleteLater()
            if on_done is not None:
                on_done()

        anim.valueChanged.connect(_step)
        anim.finished.connect(_settle)
        self._fade_anim = anim
        anim.start()

    def _animate_open(self):
        """Fade the scrim and the glass panel in together over 200 ms.

        The panel's own alpha/border/radius are fixed at their target values
        immediately (matching DataManagementWidget); the entrance fade itself
        is purely a composited opacity ramp on glass_frame, so it never
        re-polishes the stylesheet per frame.
        """
        target_alpha = 255 if getattr(self, "_is_fullscreen", False) else 215
        self._panel_alpha = target_alpha
        self._set_panel_alpha(target_alpha)
        self._scrim_alpha = 0
        self._set_glass_opacity(0.0)
        self.update()
        self._run_fade(
            scrim_from=0,
            scrim_to=65,
            op_from=0.0,
            op_to=1.0,
            duration=200,
            easing=QtCore.QEasingCurve.OutQuad,
        )

    def _animate_close(self):
        """Fade the scrim and the glass panel out together, then perform the
        real close. Identical whether inset or fullscreen - the panel's own
        opacity fades along with the scrim, so no pixmap-proxy collapse is
        needed to avoid an "instant pop" at fullscreen.
        """
        # A fullscreen expand/collapse may still be in flight - stop it cleanly.
        if hasattr(self, "anim") and self.anim.state() == QtCore.QAbstractAnimation.Running:
            self.anim.stop()

        cur_op = self._glass_opacity.opacity() if self._glass_opacity else 1.0
        self._run_fade(
            scrim_from=self._scrim_alpha,
            scrim_to=0,
            op_from=cur_op,
            op_to=0.0,
            duration=180,
            easing=QtCore.QEasingCurve.InQuad,
            on_done=self._do_close,
        )

    def _do_close(self):
        """Actually perform the close after the fade-out animation completes."""
        self._stop_anim()
        self._closing = True
        self.close()  # closeEvent sees _closing=True → accepts → Qt calls hide()
        self._closing = False
        self._close_in_progress = False
        # Reset to inset state so the next open always starts clean, regardless
        # of whether we closed from fullscreen. The live frame is restored to a
        # visible, inset, semi-transparent panel here.
        self.glass_frame.show()
        if getattr(self, "_is_fullscreen", False):
            self._is_fullscreen = False
            self._fs_normal_icon = self._tinted_icon(
                self.ICON_EXPAND, QtGui.QColor(110, 120, 130, 190), size=14
            )
            self._fs_hover_icon = self._tinted_icon(
                self.ICON_EXPAND, QtGui.QColor(185, 190, 200, 230), size=14
            )
            self.btn_fullscreen.setIcon(self._fs_normal_icon)
        self._panel_alpha = 215
        self._set_panel_alpha(215)
        self._set_glass_opacity(1.0)
        self._refit_to_parent()

    def eventFilter(self, obj, event):
        """Track parent/window resize so the overlay always covers the content area,
        and listen for table clicks to cancel pending deletions."""

        # --- 1. Handle Table Viewport Clicks (Cancel Delete) ---
        if hasattr(self, "table") and obj == self.table.viewport():
            if event.type() == QtCore.QEvent.MouseButtonPress:
                if getattr(self, "_pending_delete", []):
                    self._cancel_delete()

        if event.type() in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Move):
            if getattr(self, "_pending_delete", []):
                self._cancel_delete()

            if self.isVisible():
                self._refit_to_parent()

        if hasattr(self, "btn_fullscreen") and obj is self.btn_fullscreen:
            if event.type() == QtCore.QEvent.Enter:
                self.btn_fullscreen.setIcon(self._fs_hover_icon)
            elif event.type() == QtCore.QEvent.Leave:
                self.btn_fullscreen.setIcon(self._fs_normal_icon)

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
        layout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)

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
            item_init.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_init.setData(QtCore.Qt.UserRole, initials)
            self.table.setItem(row, 1, item_init)

            # 2. Name
            item_name = QtWidgets.QTableWidgetItem(data["Name"][row])
            item_name.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_name.setData(QtCore.Qt.UserRole, initials)
            self.table.setItem(row, 2, item_name)

            # 3. Role
            down_arrow_path = os.path.join(
                Architecture.get_path(), "QATCH", "icons", "down-arrow.svg"
            )
            role_combo = AnimatedComboBox(down_arrow_path)
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
            item_created.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 4, item_created)

            item_accessed = QtWidgets.QTableWidgetItem(data["Accessed"][row])
            item_accessed.setFlags(QtCore.Qt.ItemIsEnabled)
            item_accessed.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 5, item_accessed)

            # 6. Actions
            action_widget = QtWidgets.QWidget()
            action_layout = QtWidgets.QHBoxLayout(action_widget)
            action_layout.setContentsMargins(5, 2, 5, 2)
            action_layout.setSpacing(5)

            icon_size_table = QtCore.QSize(16, 16)

            btn_pwd = GlassPushButton("", variant="default")
            btn_pwd.setIcon(QtGui.QIcon(self.ICON_PWD))
            btn_pwd.setIconSize(icon_size_table)
            btn_pwd.setToolTip("Reset Password")
            btn_pwd.setFixedSize(28, 28)
            btn_pwd.clicked.connect(lambda _, i=initials: self.change_password(i))

            btn_audit = GlassPushButton("", variant="default")
            btn_audit.setIcon(QtGui.QIcon(self.ICON_AUDIT))
            btn_audit.setIconSize(icon_size_table)
            btn_audit.setToolTip("Audit User")
            btn_audit.setFixedSize(28, 28)
            btn_audit.clicked.connect(lambda _, i=initials: self.audit_selected([i]))

            btn_delete = GlassPushButton("", variant="danger")
            btn_delete.setIcon(QtGui.QIcon(self.ICON_DELETE))
            btn_delete.setIconSize(icon_size_table)
            btn_delete.setToolTip("Delete User")
            btn_delete.setFixedSize(28, 28)
            btn_delete.clicked.connect(lambda _, i=initials: self.delete_selected([i]))

            action_layout.addStretch()
            action_layout.addWidget(btn_pwd)
            action_layout.addWidget(btn_audit)
            action_layout.addWidget(btn_delete)
            action_layout.addStretch()
            self.table.setCellWidget(row, 6, action_widget)

        header = self.table.horizontalHeader()
        # Reset stretchLastSection unconditionally - audit mode sets it True and
        # the column-mode calls below do not clear it, leaving col-6 stretched on return.
        header.setStretchLastSection(False)
        # Checkbox col - fixed narrow
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)
        header.resizeSection(0, 36)
        # Initials - snug to content
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        # Name - takes up remaining space
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        # Role combo - fixed wide enough for text + dropdown
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Fixed)
        header.resizeSection(3, 155)
        # Created / Accessed - snug to content
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)
        # Actions - fixed width for three 28px buttons + padding
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

        # --- Dynamically dye the SVG arrow to match the role text color ---
        # Pre-render the tint into the pixmap itself (same technique as
        # _tinted_icon) rather than a QGraphicsColorizeEffect: glass_frame
        # carries a QGraphicsOpacityEffect for the open/close fade, and Qt
        # doesn't compose nested widget graphics effects reliably - every
        # role combo in the table got its own colorize effect, which spammed
        # "QPainter::begin: A paint device can only be painted by one painter
        # at a time" / "Painter not active" during the fade and corrupted the
        # panel's first painted frame.
        if hasattr(combo, "arrow_lbl"):
            base = getattr(combo, "_base_pixmap", None)
            if base is not None and not base.isNull():
                tinted = QtGui.QPixmap(base.size())
                tinted.fill(QtCore.Qt.transparent)
                p = QtGui.QPainter(tinted)
                p.drawPixmap(0, 0, base)
                p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
                p.fillRect(tinted.rect(), QtGui.QColor(text_color))
                p.end()
                combo._base_pixmap = tinted
                combo.arrow_lbl.setPixmap(tinted)

        combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {bg_color};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 10px;
                padding: 2px 28px 2px 10px; /* Slight right padding increase to accommodate label */
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
                width: 24px; /* Widened slightly to fit the AnimatedComboBox label */
                border: none;
                border-left: 1px solid {border_color};
                border-top-right-radius: 10px;
                border-bottom-right-radius: 10px;
            }}
            QComboBox::down-arrow {{
                image: none; /* Handled dynamically by AnimatedComboBox */
            }}
            
            /* --- Dropdown Viewport Fixes --- */
            QComboBox QAbstractItemView {{
                background-color: rgba(255, 255, 255, 250);
                border: 1px solid {border_color};
                border-radius: 6px;
                outline: none;
                selection-background-color: transparent;
                selection-color: #333;
                color: #333;
                padding: 4px;
            }}
            QComboBox QAbstractItemView::viewport {{
                background: transparent;
                border-radius: 6px;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 4px 10px;
                min-height: 20px;
                color: #333;
                background-color: transparent;
                border-radius: 4px;
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

        user_data = UserProfiles.get_user_info(os.path.join(UserProfiles.PATH, filename))

        overlay = ResetPasswordWidget(
            name=user_data[0],
            initials=user_data[1],
            role=user_data[2],
            parent=self,
        )
        overlay.resize(self.size())
        overlay.show()
        overlay.raise_()
        # password_confirmed is emitted before the close animation starts,
        # so the filename closure is still valid when the lambda fires.
        overlay.password_confirmed.connect(
            lambda pwd, fn=filename: self._update_user_xml(fn, new_pwd_plain=pwd)
        )

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
            # Shake the button instead of showing a popup
            self._shake_widget(self.btn_audit_selected)
            return

        self._animate_table_transition(
            lambda: self._do_audit_selected(initials_list), direction="left"
        )

    def _shake_widget(self, widget: QtWidgets.QWidget) -> None:
        """Animates a widget with a rapid horizontal jiggle to indicate an error."""
        # Wrap in a try-except block just to be absolutely safe against C++ deletions
        try:
            if (
                hasattr(widget, "_shake_anim")
                and widget._shake_anim.state() == QtCore.QAbstractAnimation.Running
            ):
                return
        except RuntimeError:
            pass  # The C++ object was deleted, meaning it's not running anyway

        start_pos = widget.pos()
        anim = QtCore.QPropertyAnimation(widget, b"pos", self)
        anim.setDuration(350)

        # Keyframes for left-right jiggle
        anim.setKeyValueAt(0.0, start_pos)
        anim.setKeyValueAt(0.1, start_pos + QtCore.QPoint(-6, 0))
        anim.setKeyValueAt(0.3, start_pos + QtCore.QPoint(6, 0))
        anim.setKeyValueAt(0.5, start_pos + QtCore.QPoint(-4, 0))
        anim.setKeyValueAt(0.7, start_pos + QtCore.QPoint(4, 0))
        anim.setKeyValueAt(0.9, start_pos + QtCore.QPoint(-2, 0))
        anim.setKeyValueAt(1.0, start_pos)

        widget._shake_anim = anim

        # FIX: Removed QtCore.QAbstractAnimation.DeleteWhenStopped
        # Now Python's garbage collector will handle memory cleanup safely
        # when the _shake_anim reference is overwritten.
        anim.start()

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
                item.setTextAlignment(
                    QtCore.Qt.AlignmentFlag.AlignCenter if col < 3 else QtCore.Qt.AlignLeft
                )

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
        is_topbar_action = False

        if override_list:
            initials_list = override_list
        else:
            is_topbar_action = True
            for row in range(self.table.rowCount()):
                chk_item = self.table.item(row, 0)
                if (
                    chk_item
                    and chk_item.checkState() == QtCore.Qt.Checked
                    and not self.table.isRowHidden(row)
                ):
                    initials_list.append(self.table.item(row, 1).text())

        if not initials_list:
            # Shake the button instead of showing a popup
            self._shake_widget(self.btn_delete_selected)
            return

        self._show_delete_confirmation(initials_list, is_topbar_action)

    def _set_table_actions_enabled(self, enabled: bool):
        """Enables or disables the action buttons in all table rows."""
        for row in range(self.table.rowCount()):
            action_widget = self.table.cellWidget(row, 6)
            if action_widget:
                action_widget.setEnabled(enabled)

    def _show_delete_confirmation(self, initials_list, is_topbar_action=False):
        """Tint the full row red via viewport overlays then show confirmation."""
        self._pending_delete = list(initials_list)

        # --- NEW: Disable all row action buttons while any confirmation is active ---
        self._set_table_actions_enabled(False)

        # If triggered from the top bar, ALWAYS use the top bar confirmation
        is_bulk = is_topbar_action or (len(initials_list) > 1)

        for row in range(self.table.rowCount()):
            init_item = self.table.item(row, 1)
            if not (init_item and init_item.text() in initials_list):
                continue

            if not is_bulk:
                # Single: animated expand-from-right confirm widget in the action cell
                # (Because this is a brand new widget replacement, it ignores the False lock above)
                self._install_single_delete_confirm(row, initials_list)

        # Overlay the full row(s) in red - works for both single and bulk
        self._add_delete_row_overlays(initials_list)

        if is_bulk:
            self._show_topbar_delete_confirm(initials_list)

    # ------------------------------------------------------------------
    # Single-delete animated confirm widget
    # ------------------------------------------------------------------
    def _install_single_delete_confirm(self, row, initials_list):
        """Replace col-6 with a layout where Cancel slides out from the Delete button."""
        confirm_w = QtWidgets.QWidget()
        confirm_w.setStyleSheet("background: transparent;")
        cl = QtWidgets.QHBoxLayout(confirm_w)
        cl.setContentsMargins(5, 2, 5, 2)
        cl.setSpacing(5)
        # Right-align so the confirm button stays stationary on the right
        cl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        # Cancel button
        btn_cancel = GlassPushButton("", variant="neutral")
        btn_cancel.setIcon(QtGui.QIcon(self.ICON_CLEAR))
        btn_cancel.setIconSize(QtCore.QSize(16, 16))
        btn_cancel.setFixedSize(0, 0)
        btn_cancel.clicked.connect(self._cancel_delete)

        # Save reference for the hide animation
        self._single_delete_cancel_btn = btn_cancel

        # Confirm button
        btn_confirm = GlassPushButton("", variant="danger_confirm")
        btn_confirm.setIcon(QtGui.QIcon(self.ICON_DELETE))
        btn_confirm.setIconSize(QtCore.QSize(16, 16))
        btn_confirm.setFixedSize(28, 28)

        cl.addWidget(btn_cancel)
        cl.addWidget(btn_confirm)

        self.table.setCellWidget(row, 6, confirm_w)

        # Animate the cancel button opening to full 28px width
        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(220)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        anim.setStartValue(0.0)
        anim.setEndValue(28.0)

        def _expand_single_cancel(v):
            s = int(v)
            btn_cancel.setFixedSize(s, s)

        anim.valueChanged.connect(_expand_single_cancel)
        anim.start()
        self._single_delete_anim = anim

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
            overlay.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
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
        """Called on table scroll - move each overlay to its row's current position."""
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
        """Insert a Cancel button into the top bar and animate its width to push other buttons left."""
        icon_sz = QtCore.QSize(18, 18)

        self.btn_delete_selected.setText("")
        self.btn_delete_selected.setToolTip(f"Confirm delete  ({len(initials_list)} users)")
        self.btn_delete_selected.set_variant("danger_confirm")
        self.btn_delete_selected.setFixedSize(34, 34)
        # Swap the connection safely
        try:
            self.btn_delete_selected.clicked.disconnect()
        except RuntimeError:
            pass
        self.btn_delete_selected.clicked.connect(
            lambda _c=False, il=list(initials_list): self._confirm_delete(il)
        )

        self.btn_cancel_bulk = GlassPushButton("", self.taskbar_frame, variant="neutral")
        self.btn_cancel_bulk.setIcon(QtGui.QIcon(self.ICON_CLEAR))
        self.btn_cancel_bulk.setIconSize(icon_sz)
        self.btn_cancel_bulk.setToolTip("Cancel")
        self.btn_cancel_bulk.setFixedSize(0, 0)
        self.btn_cancel_bulk.clicked.connect(self._cancel_delete)

        # 3. Insert it right before the delete button
        idx = self.top_bar.indexOf(self.btn_delete_selected)
        self.top_bar.insertWidget(idx, self.btn_cancel_bulk)

        # 4. Animate the Cancel button opening
        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(220)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        anim.setStartValue(0.0)
        anim.setEndValue(34.0)

        def _expand_bulk_cancel(v):
            s = int(v)
            self.btn_cancel_bulk.setFixedSize(s, s)

        anim.valueChanged.connect(_expand_bulk_cancel)
        anim.start()
        self._topbar_confirm_anim = anim

    def _hide_single_delete_confirm(self):
        """Collapse the single-delete cancel button, then restore the action buttons."""
        btn_cancel = getattr(self, "_single_delete_cancel_btn", None)
        if not btn_cancel:
            return

        # Stop any opening animation if it's currently running
        anim_in = getattr(self, "_single_delete_anim", None)
        if anim_in and anim_in.state() == QtCore.QAbstractAnimation.State.Running:
            anim_in.stop()

        start_w = btn_cancel.width()  # equals height since both were set together
        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(160)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.InCubic)
        anim.setStartValue(float(start_w))
        anim.setEndValue(0.0)

        def _collapse_single_cancel(v):
            s = int(v)
            btn_cancel.setFixedSize(s, s)

        anim.valueChanged.connect(_collapse_single_cancel)

        def _cleanup():
            self._single_delete_cancel_btn = None
            self.update_table_data()  # Restores standard action buttons

        anim.finished.connect(_cleanup)
        anim.start()
        self._single_delete_anim = anim

    def _hide_topbar_delete_confirm(self):
        """Collapse the cancel button, then restore the delete button to its normal state."""
        if not hasattr(self, "btn_cancel_bulk") or not self.btn_cancel_bulk:
            return

        anim_in = getattr(self, "_topbar_confirm_anim", None)
        if anim_in and anim_in.state() == QtCore.QAbstractAnimation.State.Running:
            anim_in.stop()

        # Restore the normal delete button logic, text, and width
        self.btn_delete_selected.setText(" Delete")
        self.btn_delete_selected.setToolTip("Delete Selected Users")
        self.btn_delete_selected.set_variant("danger")
        self.btn_delete_selected.setMinimumWidth(0)
        self.btn_delete_selected.setMaximumWidth(16777215)
        self.btn_delete_selected.setFixedHeight(34)
        try:
            self.btn_delete_selected.clicked.disconnect()
        except RuntimeError:
            pass
        self.btn_delete_selected.clicked.connect(lambda: self.delete_selected())

        # Animate the cancel button closing
        start_w = self.btn_cancel_bulk.width()
        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(160)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.InCubic)
        anim.setStartValue(float(start_w))
        anim.setEndValue(0.0)

        def _collapse_bulk_cancel(v):
            s = int(v)
            self.btn_cancel_bulk.setFixedSize(s, s)

        anim.valueChanged.connect(_collapse_bulk_cancel)

        def _cleanup():
            self.top_bar.removeWidget(self.btn_cancel_bulk)
            self.btn_cancel_bulk.deleteLater()
            self.btn_cancel_bulk = None

        anim.finished.connect(_cleanup)
        anim.start()
        self._topbar_confirm_anim = anim

    # ------------------------------------------------------------------
    # Delete confirmation callbacks
    # ------------------------------------------------------------------

    def _cancel_delete(self):
        """Discard the pending confirmation, remove overlays, and trigger collapse animations."""
        self._pending_delete = []

        # Instantly remove the red row highlights
        self._remove_delete_overlays()

        # --- NEW: Instantly re-enable all other row action buttons ---
        self._set_table_actions_enabled(True)

        # Trigger the slide-out animations instead of instantly redrawing the table
        if getattr(self, "btn_cancel_bulk", None):
            self._hide_topbar_delete_confirm()
        elif getattr(self, "_single_delete_cancel_btn", None):
            self._hide_single_delete_confirm()

    def _confirm_delete(self, initials_list):
        """Stage 2: confirmed - instantly discard topbar UI, then run row collapse animation."""
        self._pending_delete = []

        if hasattr(self, "btn_cancel_bulk") and self.btn_cancel_bulk:
            anim_in = getattr(self, "_topbar_confirm_anim", None)
            if anim_in and anim_in.state() == QtCore.QAbstractAnimation.State.Running:
                anim_in.stop()
            self.top_bar.removeWidget(self.btn_cancel_bulk)
            self.btn_cancel_bulk.deleteLater()
            self.btn_cancel_bulk = None

            # Restore delete button state now that the bulk confirm UI is gone
            self.btn_delete_selected.setText(" Delete")
            self.btn_delete_selected.setToolTip("Delete Selected Users")
            self.btn_delete_selected.setStyleSheet("""
                QPushButton { 
                    background: rgba(220, 53, 69, 0.1); 
                    color: #B02A37;
                    font-weight: bold;
                    border: 1px solid rgba(220, 53, 69, 0.3); 
                    border-radius: 17px; 
                    padding: 0px 14px;
                    min-height: 34px; max-height: 34px;
                    min-width: 0px; max-width: 16777215px; /* Free the width constraints */
                }
                QPushButton:hover { background: rgba(220, 53, 69, 0.25); border: 1px solid rgba(220, 53, 69, 0.6); }
                QPushButton:pressed { background: rgba(220, 53, 69, 0.4); }
            """)
            try:
                self.btn_delete_selected.clicked.disconnect()
            except RuntimeError:
                pass
            self.btn_delete_selected.clicked.connect(lambda: self.delete_selected())

        self._animate_delete_rows(initials_list)

    def _animate_delete_rows(self, initials_list):
        """Phase 1: Retain viewport overlays to guarantee perfect edge-to-edge red tint.
        Phase 2: Immediately collapse each row's height to 0 while dynamically shrinking the overlays.
        Phase 3: Clean up overlays, archive files, and refresh."""
        affected_rows = [
            row
            for row in range(self.table.rowCount())
            if self.table.item(row, 1) and self.table.item(row, 1).text() in initials_list
        ]

        if not affected_rows:
            self._remove_delete_overlays()
            self._execute_delete(initials_list)
            return

        # Phase 1: DO NOT remove the overlays here! The QWidget overlays placed by
        # _show_delete_confirmation provide a perfect, unbroken red tint that ignores
        # QSS overrides. We will keep them alive and animate them down to 0 height.

        # Phase 2 - collapse rows and dynamically shrink the overlays
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
                    # Force the perfect red overlays to shrink in lockstep with the row
                    self._reposition_delete_overlays()

                return _step

            def _make_finish():
                def _finish():
                    completed[0] += 1
                    if completed[0] >= total:
                        # Phase 3 - Clean up the overlays right BEFORE the table redraws
                        self._remove_delete_overlays()
                        self._execute_delete(initials_list)

                return _finish

            anim.valueChanged.connect(_make_step(row))
            anim.finished.connect(_make_finish())
            self._delete_anims.append(anim)

        # Start the collapse animations
        QtCore.QTimer.singleShot(0, lambda: [a.start() for a in self._delete_anims])

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
        """Spawns the user creation overlay and updates the table on completion."""
        UserProfiles.create_new_user()
        self.update_table_data()

    def toggle_dev_mode(self, arg):
        try:
            dev_path = os.path.join(Constants.local_app_data_path, ".dev_mode")
            if self.developerModeChk.isChecked():
                expires_at = str(
                    (dt.datetime.now() + dt.timedelta(days=UserConstants.DEV_EXPIRE_LEN)).date()
                )
                with open(dev_path, "w") as dev:
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
                self.dev_mode_label.setStyleSheet("""
                    QLabel { color: #388E3C; font-weight: bold; font-size: 11px; background: transparent; }
                """)
                self.dev_expiry_label.setStyleSheet("""
                    QLabel { color: #388E3C; font-size: 10px; background: transparent; padding-left: 6px; }
                """)
                self.dev_expiry_label.setText(f"Expires: {expires_at}")
                self.dev_expiry_label.setVisible(True)
            else:
                if os.path.exists(dev_path):
                    os.remove(dev_path)
                self.dev_mode_label.setStyleSheet("""
                    QLabel { color: #555; font-weight: bold; font-size: 11px; background: transparent; }
                """)
                self.dev_expiry_label.setVisible(False)
                self.dev_expiry_label.setText("")
        except Exception as e:
            Log.e(f"Error updating Developer Mode: {str(e)}")

    def toggle_req_admin_updates(self, arg):
        try:
            if not os.path.isfile(Constants.user_constants_path):
                os.makedirs(os.path.split(Constants.user_constants_path)[0], exist_ok=True)
            with open(Constants.user_constants_path, "w") as uc:
                UserConstants.REQ_ADMIN_UPDATES = self.reqAdminUpd_chkbox.isChecked()
                uc.write(f"REQ_ADMIN_UPDATES = {UserConstants.REQ_ADMIN_UPDATES}")
        except Exception as e:
            Log.e(f"Failed to save user constants: {str(e)}")
