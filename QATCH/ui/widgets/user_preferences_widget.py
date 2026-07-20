"""
QATCH.ui.widgets.user_preferences_widget

Glassmorphic "Preferences" overlay - same overlay shell convention as
`QATCH.ui.widgets.data_management_widget.DataManagementWidget` and
`QATCH.ui.widgets.user_profiles_manager_widget.UserProfilesManagerWidget`:
a child widget reparented over the app's central widget, dimmed scrim +
fade-in/out glass panel, click-outside-to-dismiss, and a `showNormal()`
compatibility shim so existing call sites in
`QATCH.ui.windows.controls_windows` / `QATCH.ui.interfaces.ui_controls`
don't need to change shape. The overlay lifecycle itself (init scaffolding,
fade animations, parent-tracking geometry fit, scrim paint, click-outside
dismiss) is shared with those two widgets via
`QATCH.ui.components.overlay_shell.OverlayLifecycleMixin` - only this
widget's own content (a vertical `SegmentedControl` sidebar over section
pages, no fullscreen toggle) lives here. Every panel is styled from the
shared flat control system (QATCH.ui.styles.theme_manager /
QATCH.ui.components) so it matches the rest of the app and stays correct
across light/dark theme changes.
"""

from __future__ import annotations

import os
from datetime import datetime

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.common.fileStorage import FileStorage
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserPreferences, UserProfiles
from QATCH.core.constants import Constants, UserRoles
from QATCH.ui.components import (
    QATCHLineEdit,
    QATCHOptionCard,
    QATCHOptionCardGroup,
    QATCHPushButton,
    QATCHToggle,
    SegmentedControl,
)
from QATCH.ui.components.overlay_shell import (
    FULLSCREEN_ANIM_EASING,
    OverlayLifecycleMixin,
    rebuild_fullscreen_icons,
    run_stack_slide,
    run_variant_animation,
    teardown_stack_slide,
)
from QATCH.ui.dialogs.pop_up_dialog import PopUp
from QATCH.ui.styles.theme_manager import (
    ThemeManager,
    ThemeMode,
    caption_label_qss,
    desc_label_qss,
    field_label_qss,
    glass_panel_qss,
    hairline_qss,
    info_wash_card_qss,
    mono_preview_qss,
    surface_panel_qss,
    tok_css,
)

TAG = "[Preferences]"
SELECT_TAG_PROMPT = Constants.select_tag_prompt
SUBFOLDER_FIELD = Constants.subfolder_field

_ICONS_DIR = os.path.join(Architecture.get_path(), "QATCH", "icons")
_ICON_PREFERENCES = os.path.join(_ICONS_DIR, "preferences.svg")
_ICON_DATETIME = os.path.join(_ICONS_DIR, "date-range.svg")
_ICON_FILENAMING = os.path.join(_ICONS_DIR, "barcode.svg")
_ICON_DATAPATHS = os.path.join(_ICONS_DIR, "import-export.svg")
_ICON_APPEARANCE = os.path.join(_ICONS_DIR, "color-palette.svg")


class UserPreferencesWidget(OverlayLifecycleMixin, QtWidgets.QWidget):
    def __init__(self, controller, parent=None):
        """Args:
        controller: Business-logic owner (the `ControlsWindow` instance) -
            used only for `.userrole` in `check_user_role()`. Kept distinct
            from `parent` (the Qt overlay parent, stored as `self.parent`
            per the `DataManagementWidget`/`UserProfilesManagerWidget`
            convention) since the two are no longer the same widget now
            that this is an overlay reparented onto the app's central
            widget instead of a standalone top-level window.
        parent: The overlay host widget (typically the main window's
            central widget) this panel is reparented onto and fitted to.
        """
        super().__init__(parent)
        self._controller = controller
        self.ICON_EXPAND = os.path.join(_ICONS_DIR, "expand.svg")
        self.ICON_COLLAPSE = os.path.join(_ICONS_DIR, "collapse.svg")

        # Assume non-admin user role until proven otherwise
        self._is_admin = False  # call 'check_user_role()' to set

        # Create default user preferences object (initially "global" only)
        UserProfiles.user_preferences = UserPreferences(UserProfiles.get_session_file())
        UserProfiles.user_preferences.set_preferences()

        # Initialize the _updating flag to avoid recursion
        self._updating = False  # Initialize the _updating flag

        # Widgets registered here get their QSS re-applied on theme change
        # (see _on_theme_changed) since each was styled with a one-shot
        # setStyleSheet() call rather than a live-token paintEvent.
        self._wells: list[tuple[QtWidgets.QFrame, str]] = []
        self._captions: list[QtWidgets.QLabel] = []
        self._descs: list[QtWidgets.QLabel] = []
        self._field_labels: list[QtWidgets.QLabel] = []
        self._hairlines: list[QtWidgets.QFrame] = []
        self._preview_boxes: list[QtWidgets.QFrame] = []
        self._preview_labels: list[QtWidgets.QLabel] = []

        # Slide-transition state for section switches (see
        # overlay_shell.run_stack_slide).
        self._slide_group = None
        self._slide_clip = None
        self._slide_stack = None

        # Overlay scaffolding (scrim/panel state, base_layout + glass_frame +
        # opacity effect + main_layout, parent-resize event filter) - shared
        # with DataManagementWidget/UserProfilesManagerWidget via
        # OverlayLifecycleMixin.
        self._init_overlay_shell(
            parent,
            "prefsview",
            panel_alpha=215,
            margin_pct=0.175,
            content_margins=(20, 14, 20, 20),
            content_spacing=14,
        )

        self.main_layout.addLayout(
            self._build_overlay_header(_ICON_PREFERENCES, "Preferences", fullscreen=True)
        )

        self._section_keys = ["datetime", "filenaming", "datapaths", "appearance"]
        nav_modes = [
            ("datetime", "Date/Time", _ICON_DATETIME),
            ("filenaming", "File Naming", _ICON_FILENAMING),
            ("datapaths", "Data Paths", _ICON_DATAPATHS),
            ("appearance", "Appearance", _ICON_APPEARANCE),
        ]
        self.nav = SegmentedControl(nav_modes, orientation=QtCore.Qt.Vertical)
        self.nav.modeChanged.connect(self.handle_section_change)

        self.content_stack = QtWidgets.QStackedWidget()
        # Let the section slide truly hide() the stack during a transition
        # while it still reserves its layout space - same technique as
        # DataManagementWidget's content_stack (see its _build_stack for the
        # full rationale: without this, hide()/show() collapses the stack to
        # zero height and forces a repaint of the whole panel around it).
        stack_policy = self.content_stack.sizePolicy()
        stack_policy.setRetainSizeWhenHidden(True)
        self.content_stack.setSizePolicy(stack_policy)
        self.content_stack.addWidget(self._build_datetime_page())
        self.content_stack.addWidget(self._build_filenaming_page())
        self.content_stack.addWidget(self._build_datapaths_page())
        self.content_stack.addWidget(self._build_appearance_page())

        body_row = QtWidgets.QHBoxLayout()
        # No gap: the nav sits directly against the active section well so
        # the highlighted row and the content card read as adjacent, related
        # surfaces instead of two panels floating apart.
        body_row.setSpacing(0)
        body_row.addWidget(self.nav)
        body_row.addWidget(self.content_stack, 1)
        self.main_layout.addLayout(body_row, 1)

        self.main_layout.addLayout(self._build_action_bar())

        # Manually fire the folder-sync handler on UI initialization.
        # NOTE: This must be after ALL write-directory layout objects exist.
        self.toggle_folder_sync(self.sync_write_with_load.isChecked())

        # Start hidden + pre-fitted to avoid the tiny-panel flash.
        self._finish_overlay_shell()

        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_datetime_page(self) -> QtWidgets.QWidget:
        well = self._well("dtWell")
        lay = QtWidgets.QVBoxLayout(well)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(14)

        lay.addWidget(self._caption("Date & Time Format"))

        date_row = QtWidgets.QHBoxLayout()
        date_row.setSpacing(10)
        date_label = self._field_label("Date Format")
        date_label.setFixedWidth(110)
        date_row.addWidget(date_label)
        self.date_format_combo = QtWidgets.QComboBox()
        self.date_format_combo.addItems(Constants.date_formats)
        self.date_format_combo.setFixedHeight(34)
        date_row.addWidget(self.date_format_combo, 1)
        lay.addLayout(date_row)

        time_row = QtWidgets.QHBoxLayout()
        time_row.setSpacing(10)
        time_label = self._field_label("Time Format")
        time_label.setFixedWidth(110)
        time_row.addWidget(time_label)
        self.time_format_combo = QtWidgets.QComboBox()
        self.time_format_combo.addItems(Constants.time_formats)
        self.time_format_combo.setFixedHeight(34)
        time_row.addWidget(self.time_format_combo, 1)
        lay.addLayout(time_row)

        lay.addWidget(
            self._desc("HH = 24-hour clock · hh = 12-hour clock with an AM/PM (A) suffix.")
        )
        lay.addStretch()

        self.preview_date_time_button = QATCHPushButton(
            "Preview Date && Time Format", variant="secondary"
        )
        self.preview_date_time_button.setFixedHeight(34)
        self.preview_date_time_button.clicked.connect(self.preview_date_time_format)
        lay.addWidget(self.preview_date_time_button)

        self.preview_date_time_label = QtWidgets.QLabel("Preview will appear here.")
        lay.addWidget(self._preview_box(self.preview_date_time_label))

        return well

    def _build_filenaming_page(self) -> QtWidgets.QWidget:
        well = self._well("fileWell")
        lay = QtWidgets.QVBoxLayout(well)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(14)

        # Tags available
        self.tags = Constants.valid_tags
        self.selected_tags = []

        lay.addWidget(self._caption("File Name Format"))
        lay.addWidget(
            self._desc("Choose the tags used to build each exported file name, in order.")
        )

        self.file_format_container = QtWidgets.QHBoxLayout()
        self.file_format_container.setSpacing(8)
        self.file_format_combos = []
        self.add_dropdown(self.file_format_container)
        lay.addLayout(self.file_format_container)
        lay.addLayout(self._build_format_controls_row(self.file_format_container))
        self.file_delimiter_combo = self._last_delimiter_combo

        lay.addWidget(self._hairline())

        lay.addWidget(self._caption("Folder Name Format"))
        lay.addWidget(
            self._desc("Choose the tags used to build the destination subfolder path.")
        )

        self.folder_format_container = QtWidgets.QHBoxLayout()
        self.folder_format_container.setSpacing(8)
        self.folder_format_combos = []
        self.add_dropdown(self.folder_format_container)
        lay.addLayout(self.folder_format_container)
        lay.addLayout(self._build_format_controls_row(self.folder_format_container))
        self.folder_delimiter_combo = self._last_delimiter_combo

        lay.addStretch()

        self.preview_button = QATCHPushButton("Preview File && Folder Format", variant="secondary")
        self.preview_button.setFixedHeight(34)
        self.preview_button.clicked.connect(self.preview_format)
        lay.addWidget(self.preview_button)

        self.preview_label = QtWidgets.QLabel("Preview will appear here.")
        lay.addWidget(self._preview_box(self.preview_label))

        return well

    def _build_format_controls_row(self, container: QtWidgets.QHBoxLayout) -> QtWidgets.QHBoxLayout:
        """Builds the "+"/"-"/delimiter control row beneath a tag-dropdown
        container (file or folder format). Stashes the delimiter combo on
        `self._last_delimiter_combo` for the caller to pick up, since each
        row needs its own combo but this helper is shared by both."""
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        row.setAlignment(QtCore.Qt.AlignLeft)

        add_button = QATCHPushButton("+", variant="secondary")
        add_button.setFixedSize(40, 30)
        add_button.clicked.connect(lambda: self.add_dropdown(container))
        row.addWidget(add_button)

        remove_button = QATCHPushButton("-", variant="secondary")
        remove_button.setFixedSize(40, 30)
        remove_button.clicked.connect(lambda: self.remove_last_dropdown(container))
        row.addWidget(remove_button)

        row.addSpacing(6)
        delim_label = self._desc("Delimiter")
        row.addWidget(delim_label)

        delimiter_combo = QtWidgets.QComboBox()
        delimiter_combo.setFixedSize(56, 30)
        delimiter_combo.addItems(Constants.path_delimiters)
        row.addWidget(delimiter_combo)
        self._last_delimiter_combo = delimiter_combo

        row.addStretch()
        return row

    def _build_datapaths_page(self) -> QtWidgets.QWidget:
        well = self._well("pathsWell")
        lay = QtWidgets.QVBoxLayout(well)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(10)

        lay.addWidget(self._caption("Load Directory"))
        lay.addWidget(self._desc("Where runs are read from when opening or analyzing data."))

        load_row = QtWidgets.QHBoxLayout()
        load_row.setSpacing(8)
        self.load_directory_input = QATCHLineEdit()
        self.load_directory_input.setFixedHeight(34)
        load_row.addWidget(self.load_directory_input, 1)
        load_browse_button = QATCHPushButton("Browse…", variant="secondary")
        load_browse_button.setFixedHeight(34)
        load_browse_button.clicked.connect(self.open_load_file_dialog)
        load_row.addWidget(load_browse_button)
        lay.addLayout(load_row)

        lay.addWidget(self._hairline())

        write_caption_row = QtWidgets.QHBoxLayout()
        write_caption_row.addWidget(self._caption("Write Directory"))
        write_caption_row.addStretch()
        self.sync_write_with_load = QATCHToggle()
        write_caption_row.addWidget(self.sync_write_with_load)
        write_caption_row.addWidget(self._desc("Same as load directory"))
        lay.addLayout(write_caption_row)
        lay.addWidget(self._desc("Destination for newly captured runs."))

        write_row = QtWidgets.QHBoxLayout()
        write_row.setSpacing(8)
        self.write_directory_input = QATCHLineEdit()
        self.write_directory_input.setFixedHeight(34)
        write_row.addWidget(self.write_directory_input, 1)
        self.write_browse_button = QATCHPushButton("Browse…", variant="secondary")
        self.write_browse_button.setFixedHeight(34)
        self.write_browse_button.clicked.connect(self.open_write_file_dialog)
        write_row.addWidget(self.write_browse_button)
        lay.addLayout(write_row)

        self.sync_write_with_load.toggled.connect(self.toggle_folder_sync)

        lay.addStretch()
        return well

    def _build_appearance_page(self) -> QtWidgets.QWidget:
        well = self._well("appearanceWell")
        lay = QtWidgets.QVBoxLayout(well)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(14)

        lay.addWidget(self._caption("Theme"))

        cards_row = QtWidgets.QHBoxLayout()
        cards_row.setSpacing(12)
        light_card = QATCHOptionCard(
            "Light", "Bright canvas for well-lit rooms.", show_radio=True
        )
        dark_card = QATCHOptionCard(
            "Dark", "Low-glare canvas for dim rooms.", show_radio=True
        )
        cards_row.addWidget(light_card, 1)
        cards_row.addWidget(dark_card, 1)
        lay.addLayout(cards_row)

        self._theme_cards = QATCHOptionCardGroup()
        self._theme_cards.addCard(light_card, "light")
        self._theme_cards.addCard(dark_card, "dark")
        # Set the initial selection BEFORE connecting `toggled`, so this
        # doesn't fire an app-wide theme re-application during construction
        # (mirrors the old combo box's setCurrentText-before-connect order).
        self._theme_cards.setCheckedId(ThemeManager.instance().mode().value)
        self._theme_cards.toggled.connect(lambda card, checked: self.change_theme(card.text()))

        lay.addWidget(
            self._desc(
                "Applies instantly and app-wide, including the sign-in screen. "
                "Saved separately from your other preferences."
            )
        )
        lay.addStretch()
        return well

    def _build_action_bar(self) -> QtWidgets.QHBoxLayout:
        action_bar = QtWidgets.QHBoxLayout()
        action_bar.setSpacing(12)

        self.global_pref_toggle = QATCHToggle()
        action_bar.addWidget(self.global_pref_toggle)

        global_text_col = QtWidgets.QVBoxLayout()
        global_text_col.setContentsMargins(0, 0, 0, 0)
        global_text_col.setSpacing(1)
        global_text_col.addWidget(self._field_label("Use global preferences"))
        self.global_pref_label = self._desc("Off — settings apply to you only")
        global_text_col.addWidget(self.global_pref_label)
        action_bar.addLayout(global_text_col)

        action_bar.addStretch()

        reset_button = QATCHPushButton("Reset to Default", variant="secondary")
        reset_button.setFixedHeight(34)
        reset_button.clicked.connect(self.reset_to_default_preferences)
        action_bar.addWidget(reset_button)

        self.submit_button = QATCHPushButton("Save Preferences", variant="primary")
        self.submit_button.setFixedHeight(34)
        self.submit_button.setMinimumWidth(150)
        self.submit_button.clicked.connect(self.save_preferences)
        action_bar.addWidget(self.submit_button)

        self.global_pref_toggle.toggled.connect(self.toggle_global_preferences)

        return action_bar

    # ------------------------------------------------------------------
    # Themed-widget factories (each registers itself for _on_theme_changed)
    # ------------------------------------------------------------------
    def _well(self, object_name: str) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName(object_name)
        # radius=11 matches SegmentedControl's vertical checked-row radius
        # (self._radius - 5, with self._radius == 16) for visual consistency
        # between the nav's highlighted row and the section well beside it.
        frame.setStyleSheet(surface_panel_qss(object_name, radius=11))
        self._wells.append((frame, object_name))
        return frame

    def _caption(self, text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text.upper())
        lbl.setStyleSheet(caption_label_qss())
        self._captions.append(lbl)
        return lbl

    def _desc(self, text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(desc_label_qss())
        self._descs.append(lbl)
        return lbl

    def _field_label(self, text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet(field_label_qss())
        self._field_labels.append(lbl)
        return lbl

    def _hairline(self) -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFixedHeight(1)
        line.setStyleSheet(hairline_qss())
        self._hairlines.append(line)
        return line

    def _preview_box(self, label: QtWidgets.QLabel) -> QtWidgets.QFrame:
        box = QtWidgets.QFrame()
        box.setObjectName("previewBox")
        box.setStyleSheet(info_wash_card_qss("previewBox"))

        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(13, 10, 13, 10)
        lay.setSpacing(5)
        lay.addWidget(self._caption("Preview"))

        label.setWordWrap(True)
        label.setStyleSheet(mono_preview_qss())
        self._preview_labels.append(label)
        lay.addWidget(label)

        self._preview_boxes.append(box)
        return box

    def _sync_theme_cards(self, mode: str) -> None:
        if self._theme_cards.checkedId() != mode:
            self._theme_cards.setCheckedId(mode)

    def _on_theme_changed(self, mode: str) -> None:
        self._apply_panel_appearance(self._current_margin_frac())
        self._refresh_header_theme()
        for frame, name in self._wells:
            frame.setStyleSheet(surface_panel_qss(name, radius=11))
        for lbl in self._captions:
            lbl.setStyleSheet(caption_label_qss())
        for lbl in self._descs:
            lbl.setStyleSheet(desc_label_qss())
        for lbl in self._field_labels:
            lbl.setStyleSheet(field_label_qss())
        for line in self._hairlines:
            line.setStyleSheet(hairline_qss())
        for box in self._preview_boxes:
            box.setStyleSheet(info_wash_card_qss("previewBox"))
        for lbl in self._preview_labels:
            lbl.setStyleSheet(mono_preview_qss())
        self._sync_theme_cards(mode)

    # ------------------------------------------------------------------
    # Overlay shell hooks (see QATCH.ui.components.overlay_shell for the
    # shared init scaffolding, fade engine, geometry fit, and event
    # handlers this widget inherits from OverlayLifecycleMixin)
    # ------------------------------------------------------------------
    def _apply_panel_appearance(self, frac: float) -> None:
        """Interpolates the glass panel's alpha/border/radius continuously
        with `frac`, matching the live margin so the fullscreen toggle reads
        as one smooth motion instead of snapping partway through - same
        recipe as DataManagementWidget/UserProfilesManagerWidget. frac == 0
        -> fullscreen (flush, no radius/border, opaque); frac ==
        `_default_margin_pct` -> fully inset."""
        p = 0.0 if self._default_margin_pct <= 0 else min(1.0, frac / self._default_margin_pct)
        alpha = int(255 + (215 - 255) * p)
        border = 1.5 * p
        radius = 12.0 * p
        self._panel_alpha = alpha
        self.glass_frame.setStyleSheet(glass_panel_qss("prefsview", alpha, border, radius))

    def _rebuild_fs_icons(self) -> None:
        """Rebuilds the fullscreen button's normal/hover icon pixmaps for the
        icon matching the current fullscreen state - shared with
        DataManagementWidget/UserProfilesManagerWidget via
        `overlay_shell.rebuild_fullscreen_icons`."""
        rebuild_fullscreen_icons(self, self.ICON_EXPAND, self.ICON_COLLAPSE)

    def toggle_fullscreen(self) -> None:
        self._is_fullscreen = not self._is_fullscreen
        self._rebuild_fs_icons()

        target = 0.0 if self._is_fullscreen else self._default_margin_pct
        start = self._default_margin_pct if self._is_fullscreen else 0.0

        def _step(t):
            frac = start + (target - start) * t
            self._apply_margin_frac(frac)

        run_variant_animation(
            self, "_fs_anim", duration=240, easing=FULLSCREEN_ANIM_EASING, on_step=_step,
        )

    def _animate_close(self) -> None:
        """A fullscreen expand/collapse may still be in flight when the
        overlay is dismissed - stop it cleanly before the shared fade-out
        takes over."""
        fs_anim = getattr(self, "_fs_anim", None)
        if fs_anim is not None and fs_anim.state() == QtCore.QAbstractAnimation.Running:
            fs_anim.stop()
        super()._animate_close()

    def _do_close(self) -> None:
        """Resets to inset state so the next open always starts clean,
        regardless of whether this closed from fullscreen."""
        teardown_stack_slide(self)
        if getattr(self, "_is_fullscreen", False):
            self._is_fullscreen = False
            self._rebuild_fs_icons()
        super()._do_close()

    # ------------------------------------------------------------------
    # Public API (called from controls_windows.py / ui_controls.py)
    # ------------------------------------------------------------------
    def showNormal(self, tab_idx=0):
        """Compatibility shim for the old standalone-window call sites
        (`controls_windows.py`'s `preferences()`, `ui_controls.py`'s
        `_open_user_preferences`) - opens the overlay on a specific
        section instead of restoring a native window."""
        # Reset labels to un-previewed states
        self.preview_date_time_label.setText("Preview will appear here.")
        self.preview_label.setText("Preview will appear here.")

        self.check_user_role()  # updates self._is_admin
        if UserProfiles().count() > 0 and self._is_admin is not None:
            if not self.global_pref_toggle.isChecked():
                # Toggle on first, forcing change handler to emit
                self.global_pref_toggle.setChecked(True)
            self.global_pref_toggle.setChecked(False)
            self.global_pref_toggle.setEnabled(True)
        else:  # is None:
            if self.global_pref_toggle.isChecked():
                # Toggle off first, forcing change handler to emit
                self.global_pref_toggle.setChecked(False)
            self.global_pref_toggle.setChecked(True)
            self.global_pref_toggle.setEnabled(False)

        if 0 <= tab_idx < len(self._section_keys):
            self.nav.set_active(self._section_keys[tab_idx])
        else:
            self.nav.set_active(self._section_keys[0])

        self.setVisible(True)

    def check_user_role(self):
        self._is_admin = UserProfiles.check(self._controller.userrole, UserRoles.ADMIN)

    def handle_section_change(self, key: str) -> None:
        """Handle sidebar-section change and load preferences if needed."""
        if key in self._section_keys:
            new_index = self._section_keys.index(key)
            cur_index = self.content_stack.currentIndex()
            if new_index != cur_index:
                # Direction matches the sidebar's top-to-bottom ordering -
                # picking a section below the current one slides the new
                # page in from below (old exits up), and vice versa. Same
                # `run_stack_slide` helper DataManagementWidget's mode
                # switch uses, just driven by this widget's own nav order.
                run_stack_slide(
                    self,
                    self.content_stack,
                    self.content_stack.widget(cur_index),
                    self.content_stack.widget(new_index),
                    axis="y",
                    forward=new_index > cur_index,
                )
        if self.global_pref_toggle.isChecked():
            self.load_global_preferences()
        elif hasattr(UserProfiles.user_preferences, "_user_preferences_path"):
            self.load_user_preferences()

    def open_load_file_dialog(self) -> bool:
        # Open file dialog for directory selection
        selected_directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory", self.load_directory_input.text()
        )
        if selected_directory:
            # Set the selected directory path to the input field
            self.load_directory_input.setText(selected_directory)
            # Sync with write folder (if sync is checked)
            if self.sync_write_with_load.isChecked():
                self.write_directory_input.setText(selected_directory)
            return True
        return False

    def open_write_file_dialog(self) -> bool:
        # Open file dialog for directory selection
        selected_directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory", self.write_directory_input.text()
        )
        if selected_directory:
            # Set the selected directory path to the input field
            self.write_directory_input.setText(selected_directory)
            return True
        return False

    def change_theme(self, theme_text: str) -> None:
        """Switches the app-wide light/dark theme immediately.

        Unlike the other sections, this isn't staged behind "Save
        Preferences" - ThemeManager applies and persists the change itself
        the moment a theme card is selected.
        """
        ThemeManager.instance().set_mode(ThemeMode(theme_text.lower()))

    def toggle_folder_sync(self, checked: bool):
        is_synced = bool(checked)
        self.write_directory_input.setEnabled(not is_synced)
        self.write_browse_button.setEnabled(not is_synced)
        if is_synced:
            self.write_directory_input.setText(self.load_directory_input.text())
        if self.sync_write_with_load.isChecked() != is_synced:
            self.sync_write_with_load.setChecked(is_synced)

    def add_dropdown(self, layout):
        """Add a new dropdown to the layout."""
        # Determine the correct layout based on the section (file or folder)
        if layout == self.file_format_container:
            combo_list = self.file_format_combos
        else:
            combo_list = self.folder_format_combos

        if len(combo_list) < len(self.tags):  # Ensure no more than 7 dropdowns
            combo = QtWidgets.QComboBox()
            combo.setFixedHeight(34)
            combo.setMinimumWidth(92)
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            # Add the "Select tag" placeholder as the first item
            combo.addItem(SELECT_TAG_PROMPT)
            # # Add "Subfolder" field as the 2nd item for folder format only
            # if combo_list != self.file_format_combos:
            #     combo.addItem(SUBFOLDER_FIELD)
            # Filter out already selected tags from the available options
            available_tags = [tag for tag in self.tags if tag not in self.selected_tags]
            combo.addItems(available_tags)
            layout.addWidget(combo)
            combo_list.append(combo)  # Add the combo to the respective list

    def remove_last_dropdown(self, layout):
        """Remove the last dropdown from the given layout, if there is more than one."""
        # Ensure there's more than one dropdown in the layout before allowing removal
        if layout == self.file_format_container and len(self.file_format_combos) > 1:
            self.remove_dropdown(self.file_format_combos[-1], layout)
        elif layout == self.folder_format_container and len(self.folder_format_combos) > 1:
            self.remove_dropdown(self.folder_format_combos[-1], layout)

    def remove_dropdown(self, combo, layout):
        """Remove a dropdown and its corresponding remove button."""
        # Determine the correct combo list based on the section (file or folder)
        if layout == self.file_format_container:
            combo_list = self.file_format_combos
        else:
            combo_list = self.folder_format_combos

        # Remove the combo from the list and layout
        if combo in combo_list:
            combo_list.remove(combo)
            layout.removeWidget(combo)

            # Delete the combo box widget itself
            combo.deleteLater()

            # Update the selected tags list to allow the tag to be selected again
            current_tag = combo.currentText()
            if current_tag != SELECT_TAG_PROMPT and current_tag in self.selected_tags:
                self.selected_tags.remove(current_tag)

    def toggle_global_preferences(self):
        is_checked = self.global_pref_toggle.isChecked()
        self.check_user_role()  # updates self._is_admin

        # Update the status subtitle
        self.global_pref_label.setText(
            "On — applies to every user on this instrument"
            if is_checked
            else "Off — settings apply to you only"
        )

        # Load preferences
        if is_checked:
            self.load_global_preferences()
            if self._is_admin is None:
                self.submit_button.setEnabled(False)
            else:
                # Allows global pref file write when there are no user profiles in the system
                # thanks to the '_is_admin' flag returning True when no user profiles exist
                self.submit_button.setEnabled(True)
        else:  # not checked
            if hasattr(UserProfiles.user_preferences, "_user_preferences_path"):
                self.load_user_preferences()
            self.submit_button.setEnabled(True)

        return  # skip the rest of this method, we only disable the submit button now

        if not self._is_admin:
            # Disable/Enable Date and Time dropdowns
            self.date_format_combo.setEnabled(not is_checked)
            self.time_format_combo.setEnabled(not is_checked)

            # Disable/Enable File Format Section
            self.file_delimiter_combo.setEnabled(not is_checked)
            for combo in self.file_format_combos:
                # Disable all file format dropdowns
                combo.setEnabled(not is_checked)

            # Disable/Enable Folder Format Section
            self.folder_delimiter_combo.setEnabled(not is_checked)
            for combo in self.folder_format_combos:
                # Disable all folder format dropdowns
                combo.setEnabled(not is_checked)

            # Explicitly disable add/remove buttons
            self.disable_add_remove_buttons(is_checked)

            # Ensure all dropdowns in the layouts are disabled dynamically
            self.disable_all_combos_in_layout(self.file_format_container, not is_checked)
            self.disable_all_combos_in_layout(self.folder_format_container, not is_checked)

    def disable_add_remove_buttons(self, disable):
        """Disable add and remove buttons for file and folder format layouts."""
        for button in self.findChildren(QtWidgets.QPushButton):
            if button.text() in ["+", "-"]:
                button.setEnabled(not disable)

    def disable_all_combos_in_layout(self, layout, enable):
        """Disable all QComboBox widgets in a layout."""
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item:
                widget = item.widget()
                if isinstance(widget, QtWidgets.QComboBox):
                    widget.setEnabled(not enable)
                elif isinstance(widget, QtWidgets.QWidget) and widget.layout():
                    # Recursively disable combo boxes in nested layouts
                    self.disable_all_combos_in_layout(widget.layout(), enable)

    def preview_date_time_format(self):
        """Preview the date and time format based on selected options."""
        date_format = self.date_format_combo.currentText()
        time_format = self.time_format_combo.currentText()

        current_datetime = datetime.now()

        formatted_date = current_datetime.strftime(Constants.date_conversion.get(date_format))
        formatted_time = current_datetime.strftime(Constants.time_conversion.get(time_format))

        preview_text = f"Date: {formatted_date}\nTime: {formatted_time}"
        self.preview_date_time_label.setText(preview_text)

    def load_global_preferences(self):
        UserProfiles.user_preferences.set_use_global(use_global=True)
        global_preferences = UserProfiles.user_preferences.load_global_preferences()
        UserProfiles.user_preferences.set_preferences()
        # Update the folder sync state based on the dictionary
        paths_synced = global_preferences["load_data_path"] == global_preferences["write_data_path"]
        self.toggle_folder_sync(paths_synced)
        # Reset the load and write paths based on the dictionary
        self.load_directory_input.setText(global_preferences["load_data_path"])
        self.write_directory_input.setText(global_preferences["write_data_path"])
        # Reset the date and time format dropdowns based on the dictionary
        self.date_format_combo.setCurrentText(global_preferences["date_format"])
        self.time_format_combo.setCurrentText(global_preferences["time_format"])

        # Reset file format
        file_format = global_preferences["filename_format"].split(
            global_preferences["filename_format_delimiter"]
        )
        self.set_file_format_dropdowns(file_format, global_preferences["filename_format_delimiter"])

        # Reset folder format
        folder_format = global_preferences["folder_format"].split(
            global_preferences["folder_format_delimiter"]
        )
        self.set_folder_format_dropdowns(
            folder_format, Constants.default_preferences["folder_format_delimiter"]
        )

        # Reset delimiters
        self.file_delimiter_combo.setCurrentText(global_preferences["filename_format_delimiter"])
        self.folder_delimiter_combo.setCurrentText(global_preferences["folder_format_delimiter"])
        # Reset global preferences toggle
        self.global_pref_toggle.setChecked(True)
        self.global_pref_label.setText("On — applies to every user on this instrument")

    def load_user_preferences(self):
        UserProfiles.user_preferences.set_use_global(use_global=False)
        user_preferences = UserProfiles.user_preferences.load_user_preferences()
        UserProfiles.user_preferences.set_preferences()
        # Update the folder sync state based on the dictionary
        paths_synced = user_preferences["load_data_path"] == user_preferences["write_data_path"]
        self.toggle_folder_sync(paths_synced)
        # Reset the load and write paths based on the dictionary
        self.load_directory_input.setText(user_preferences["load_data_path"])
        self.write_directory_input.setText(user_preferences["write_data_path"])
        # Reset the date and time format dropdowns based on the dictionary
        self.date_format_combo.setCurrentText(user_preferences["date_format"])
        self.time_format_combo.setCurrentText(user_preferences["time_format"])

        # Reset file format
        file_format = user_preferences["filename_format"].split(
            user_preferences["filename_format_delimiter"]
        )
        self.set_file_format_dropdowns(file_format, user_preferences["filename_format_delimiter"])

        # Reset folder format
        folder_format = user_preferences["folder_format"].split(
            user_preferences["folder_format_delimiter"]
        )
        self.set_folder_format_dropdowns(
            folder_format, Constants.default_preferences["folder_format_delimiter"]
        )

        # Reset delimiters
        self.file_delimiter_combo.setCurrentText(user_preferences["filename_format_delimiter"])
        self.folder_delimiter_combo.setCurrentText(user_preferences["folder_format_delimiter"])

        # Reset global preferences toggle
        self.global_pref_toggle.setChecked(False)
        self.global_pref_label.setText("Off — settings apply to you only")

    def show_error_dialog(self, title, message):
        """Show a themed error dialog with the specified title and message."""
        PopUp.critical(self, title, message, ok_only=True)

    def preview_format(self):
        """Preview the file and folder format based on selected tags."""
        file_format = [combo.currentText() for combo in self.file_format_combos]
        folder_format = [combo.currentText() for combo in self.folder_format_combos]
        file_delimiter = self.file_delimiter_combo.currentText()
        folder_delimiter = self.folder_delimiter_combo.currentText()
        port_id = 0
        device_id = FileStorage.DEV_get_active(0)
        if device_id == "" or device_id is None:
            device_id = "12345678"
            Log.w(
                TAG, f"Failed to retrieve active 'Device'. Using ID \"{device_id}\" as an example."
            )
        if "_" in device_id:  # for multiplex devices, parse port_id from device_id
            port_id, device_id = device_id.split("_", 1)
        port_id = int(str(port_id), 16)
        if port_id == "" or port_id is None:
            Log.w(
                TAG, f"Failed to retrieve active 'Port ID'. Using ID \"{port_id}\" as an example."
            )
            port_id = "1"
        if port_id != port_id % 9:  # 4x6 system detected, PID A-D, not 1-4
            # convert int(10) -> int(161)
            # where int(10) refers to Port A, assuming active channel 1 of 6
            port_id = (port_id << 4) + 0x01

        # Generate a preview string based on the selected format
        file_preview = UserProfiles.user_preferences._build_save_path(
            file_format, "Runname", file_delimiter, device_id, port_id
        )
        folder_preview = UserProfiles.user_preferences._build_save_path(
            folder_format, "Runname", folder_delimiter, device_id, port_id
        )
        file_preview = file_preview.strip("-_ ")
        folder_preview = folder_preview.strip("-_ ")
        preview_text = (
            f"File Format Preview: {file_preview}\nFolder Format Preview: {folder_preview}"
        )
        self.preview_label.setText(preview_text)

    def save_preferences(self):
        load_data_path = self.load_directory_input.text()
        write_data_path = self.write_directory_input.text()
        self.check_user_role()  # updates self._is_admin

        # Check if load path exists
        if not os.path.exists(load_data_path):
            self.show_error_dialog(
                "Load Path Error", f"The specified load path does not exist:\n{load_data_path}"
            )
            return
        # Check if write path exists
        if not os.path.exists(write_data_path):
            self.show_error_dialog(
                "Write Path Error", f"The specified write path does not exist:\n{write_data_path}"
            )
            return
        # Check if paths are directories
        if not os.path.isdir(load_data_path):
            self.show_error_dialog(
                "Load Path Error", f"The specified load path is not a directory:\n{load_data_path}"
            )
            return
        if not os.path.isdir(write_data_path):
            self.show_error_dialog(
                "Write Path Error",
                f"The specified write path is not a directory:\n{write_data_path}",
            )
            return

        date_format = self.date_format_combo.currentText()
        time_format = self.time_format_combo.currentText()

        # Get file and folder formats
        file_format = [combo.currentText() for combo in self.file_format_combos]
        folder_format = [combo.currentText() for combo in self.folder_format_combos]
        file_delimiter = self.file_delimiter_combo.currentText()
        folder_delimiter = self.folder_delimiter_combo.currentText()

        # Check "Port" exists in format dropdowns
        if "Port" not in file_format:
            self.show_error_dialog(
                "File Format Error",
                'The "Port" tag must exist in the file format to create unique paths for multiplex runs.\n'
                + "For single runs, the tag's value will be blank (unused).",
            )
            return
        # if not "Port" in folder_format:
        #     self.show_error_dialog(
        #         "Folder Format Error",
        #         "The \"Port\" tag must exist in the folder format to create unique paths for multiplex runs.\n" +
        #         "For single runs, the tag's value will be blank (unused).")
        #     return

        # Check tag placeholder text does not exist in format dropdowns
        if Constants.select_tag_prompt in file_format:
            self.show_error_dialog(
                "File Format Error",
                f'The "{Constants.select_tag_prompt}" placeholder is not a valid tag.\n'
                + "Please remove it from the file format and try again.",
            )
            return
        if Constants.select_tag_prompt in folder_format:
            self.show_error_dialog(
                "Folder Format Error",
                f'The "{Constants.select_tag_prompt}" placeholder is not a valid tag.\n'
                + "Please remove it from the folder format and try again.",
            )
            return

        file_format_pattern = ""
        folder_format_pattern = ""
        for i, tok in enumerate(file_format):
            file_format_pattern = file_format_pattern + tok
            if i < len(file_format) - 1:
                file_format_pattern = file_format_pattern + file_delimiter
        for i, tok in enumerate(folder_format):
            folder_format_pattern = folder_format_pattern + tok
            if i < len(folder_format) - 1:
                folder_format_pattern = folder_format_pattern + folder_delimiter

        # Save preferences
        UserProfiles.user_preferences._set_date_format(date_format=date_format)
        UserProfiles.user_preferences._set_time_format(time_format=time_format)
        UserProfiles.user_preferences._set_file_delimiter(file_delimiter=file_delimiter)
        UserProfiles.user_preferences._set_folder_delimiter(folder_delimiter=folder_delimiter)
        UserProfiles.user_preferences._set_file_format_pattern(
            file_format_pattern=file_format_pattern
        )
        UserProfiles.user_preferences._set_folder_format_pattern(
            folder_format_pattern=folder_format_pattern
        )
        UserProfiles.user_preferences._set_load_data_path(load_data_path=load_data_path)
        UserProfiles.user_preferences._set_write_data_path(write_data_path=write_data_path)

        write_globals = False
        if hasattr(UserProfiles.user_preferences, "_user_preferences_path"):
            UserProfiles.user_preferences.write_user_preferences()
            if self._is_admin and self.global_pref_toggle.isChecked():
                # Admin user is modifying global prefs, update both files
                write_globals = True
        elif self._is_admin:
            # No user profiles exist, read/write global pref file only
            write_globals = True
        if write_globals:
            UserProfiles.user_preferences.write_global_preferences()

        # Confirm preferences were saved
        PopUp.information(
            self, "Preferences Saved", "Your preferences have been successfully saved."
        )

    def reset_to_default_preferences(self):
        """Reset preferences to their default values based on a dictionary."""
        # Update folder sync state based on the dictionary
        paths_synced = (
            Constants.default_preferences["load_data_path"]
            == Constants.default_preferences["write_data_path"]
        )
        self.toggle_folder_sync(paths_synced)
        # Reset the load and write paths based on the dictionary
        self.load_directory_input.setText(Constants.default_preferences["load_data_path"])
        self.write_directory_input.setText(Constants.default_preferences["write_data_path"])
        # Reset the date and time format dropdowns based on the dictionary
        self.date_format_combo.setCurrentText(Constants.default_preferences["date_format"])
        self.time_format_combo.setCurrentText(Constants.default_preferences["time_format"])

        # Reset file format
        file_format = Constants.default_preferences["filename_format"].split(
            Constants.default_preferences["filename_format_delimiter"]
        )
        self.set_file_format_dropdowns(
            file_format, Constants.default_preferences["filename_format_delimiter"]
        )

        # Reset folder format
        folder_format = Constants.default_preferences["folder_format"].split(
            Constants.default_preferences["folder_format_delimiter"]
        )
        self.set_folder_format_dropdowns(
            folder_format, Constants.default_preferences["folder_format_delimiter"]
        )

        # Reset delimiters
        self.file_delimiter_combo.setCurrentText(
            Constants.default_preferences["filename_format_delimiter"]
        )
        self.folder_delimiter_combo.setCurrentText(
            Constants.default_preferences["folder_format_delimiter"]
        )

    def set_file_format_dropdowns(self, file_format, delimiter):
        """Sets the file format dropdowns based on the provided file format list."""
        # Remove all existing dropdowns
        for combo in self.file_format_combos:
            combo.deleteLater()
        self.file_format_combos.clear()
        for i, format_item in enumerate(file_format):
            if i >= len(self.file_format_combos):
                self.add_dropdown(self.file_format_container)
            self.file_format_combos[i].setCurrentText(format_item)
        # Ensure delimiters are correctly set
        self.file_delimiter_combo.setCurrentText(delimiter)

    def set_folder_format_dropdowns(self, folder_format, delimiter):
        """Sets the folder format dropdowns based on the provided folder format list."""
        for combo in self.folder_format_combos:
            combo.deleteLater()
        self.folder_format_combos.clear()
        for i, format_item in enumerate(folder_format):
            if i >= len(self.folder_format_combos):
                self.add_dropdown(self.folder_format_container)
            self.folder_format_combos[i].setCurrentText(format_item)
        # Ensure delimiters are correctly set
        self.folder_delimiter_combo.setCurrentText(delimiter)
