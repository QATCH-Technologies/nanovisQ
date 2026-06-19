"""Export mode — select runs + destination (folder or USB) and export data.

PORT FROM export_widget.Ui_Export:
    build()                  <- tab2 (groupbox1 USB, groupbox2 Folder,
                                groupbox3 Export Settings, groupbox6 CSV fields,
                                tb/pb progress, exportNow/exportCancel)
    select_folder_source     -> _select_run
    select_folder_target     -> _select_target / _refresh_target
    generateExportName       -> _generate_name
    exportChanged            -> _on_format_changed
    noNameChanged            -> _on_noname_changed
    selectChanged            -> _on_selection_changed
    checkChanged1/2          -> _on_dest_changed (USB/Folder destination segment)
    export                   -> _do_export (validates filter, then run_task)
    exportTask               -> _export_task(abort, name, target, date_filter)   [STUB]
    appendRunToCsvReport      -> _append_run_to_csv                              [STUB]
    copytree                 -> shared with import (port in refinement)          [STUB]

LAYOUT PASS: all controls are built and wired to their state handlers,
preserving the original semantics:
  * USB vs Folder targets are mutually exclusive (chk1 / chk2), and selecting
    one clears the other; `drive` tracks the active target.
  * Export-as CSV/ZIP/Folder is exclusive; CSV enables the field picker and
    relabels Merge/Skip -> Append/Abort.
  * Run scope is All vs Selection (a logged-data subfolder).
  * Date filter is Off / Today / Last-N {Hours,Days,Weeks}.
  * Existing-files policy is Replace / Merge / Skip (ids 1 / 2 / 3).

The heavy `_export_task` body (CSV report generation, ZIP packaging, copytree,
history write) is stubbed with porting markers for the refinement pass.
"""

import os
import datetime
from datetime import timezone as tz

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.core.constants import Constants
from QATCH.common.logger import Logger as Log

from QATCH.ui.widgets.data_mode_base import DataModeWidget
from QATCH.ui.components.glass_push_button import GlassPushButton
from QATCH.ui.components.glass_line_edit import GlassLineEdit

try:
    from QATCH.VisQAI.src.view.components.checkable_combo_box import CheckableComboBox
except Exception:  # pragma: no cover - optional dependency at layout time
    CheckableComboBox = None

TAG = "[DataExport]"

# Existing-files policy ids — preserved from the original btnGroup2.
POLICY_REPLACE = 1
POLICY_MERGE = 2
POLICY_SKIP = 3

# Date-filter ids — preserved from the original btnGroup5.
FILTER_OFF = 0
FILTER_TODAY = 1
FILTER_LAST = 2

CSV_FIELDS = [
    "Run Name",
    "Average Viscosity",
    "Std Dev",
    "Viscosity Profile",
    "Temp",
    "Formulation",
    "Notes",
]


class ExportMode(DataModeWidget):
    MODE_KEY = "export"
    MODE_LABEL = "Export"

    # ------------------------------------------------------------------
    #  Build
    # ------------------------------------------------------------------
    def build(self):
        # Shared-state mirrors of the original flags.
        self._chk_usb = False  # exporting to USB
        self._chk_folder = False  # exporting to folder
        self._source_subfolder = ""  # selected run/device subpath ("" = all)
        self._filter_min = 0  # computed date floor at export time

        self._build_target_card()
        self._build_settings_card()
        self._build_csv_card()
        self._build_status_and_actions()

        # Apply initial enable-state (CSV default, folder destination default).
        self.rb_csv.setChecked(True)
        self._on_format_changed()
        self._set_destination("folder")

    # ---- Target card (USB / Folder) -----------------------------------
    def _build_target_card(self):
        card = self._card("Export Destination")
        lay = card.layout()

        # Destination: frosted segmented control (USB vs Folder).
        dest_caption = QtWidgets.QLabel("Export to")
        dest_caption.setStyleSheet(self._caption_qss())
        lay.addWidget(dest_caption)

        self.dest_segment = QtWidgets.QFrame()
        self.dest_segment.setObjectName("destSegment")
        self.dest_segment.setStyleSheet(self._segment_frame_qss())
        dseg = QtWidgets.QHBoxLayout(self.dest_segment)
        dseg.setContentsMargins(4, 4, 4, 4)
        dseg.setSpacing(4)
        self.dest_group = QtWidgets.QButtonGroup(self)
        self.btn_dest_usb = self._segment_button("USB Drive", "Export to a removable USB drive")
        self.btn_dest_folder = self._segment_button("Folder", "Export to a folder on this machine")
        self.dest_group.addButton(self.btn_dest_usb, 0)
        self.dest_group.addButton(self.btn_dest_folder, 1)
        dseg.addWidget(self.btn_dest_usb)
        dseg.addWidget(self.btn_dest_folder)
        self.dest_group.buttonToggled.connect(self._on_dest_changed)
        lay.addWidget(self.dest_segment)

        # Target row: read-only target field + grouped action pickers. The
        # contents adapt to the destination (Detect/Eject for USB; Choose for
        # folder), but both live in the same frosted picker box for consistency.
        target_caption = QtWidgets.QLabel("Target")
        target_caption.setStyleSheet(self._caption_qss())
        lay.addWidget(target_caption)

        target_row = QtWidgets.QHBoxLayout()
        target_row.setContentsMargins(0, 0, 0, 0)
        target_row.setSpacing(8)
        self.target_field = GlassLineEdit()
        self.target_field.setText("[NONE]")
        self.target_field.setReadOnly(True)
        self.target_field.setMinimumHeight(34)

        picker_box = QtWidgets.QFrame()
        picker_box.setObjectName("pickerBox")
        picker_box.setStyleSheet(self._picker_box_qss())
        picker_lay = QtWidgets.QHBoxLayout(picker_box)
        picker_lay.setContentsMargins(4, 4, 4, 4)
        picker_lay.setSpacing(4)
        # USB actions
        self.btn_detect = GlassPushButton(" Detect", variant="default")
        self.btn_detect.setFixedHeight(28)
        self.btn_detect.setIcon(self._icon("usb.svg"))
        self.btn_detect.clicked.connect(self._do_detect)
        self.btn_eject = GlassPushButton(" Eject", variant="default")
        self.btn_eject.setFixedHeight(28)
        self.btn_eject.clicked.connect(self._do_eject)
        # Folder action
        self.btn_target = GlassPushButton(" Choose…", variant="default")
        self.btn_target.setFixedHeight(28)
        self.btn_target.setIcon(self._icon("folder.svg"))
        self.btn_target.clicked.connect(self._select_target)
        picker_lay.addWidget(self.btn_detect)
        picker_lay.addWidget(self.btn_eject)
        picker_lay.addWidget(self.btn_target)

        target_row.addWidget(self.target_field, 1)
        target_row.addWidget(picker_box)
        lay.addLayout(target_row)

        self.root.addWidget(card)

    # ---- Settings card ------------------------------------------------
    def _build_settings_card(self):
        card = self._card("Export Settings")
        lay = card.layout()

        # Run scope: All vs Selection (+ choose button).
        lay.addWidget(self._caption("Runs to export"))
        scope_row = QtWidgets.QHBoxLayout()
        scope_row.setContentsMargins(0, 0, 0, 0)
        scope_row.setSpacing(8)
        self.scope_segment, self.scope_group, (self.btn_scope_all, self.btn_scope_sel) = (
            self._make_segment([("All Runs", ""), ("Selection", "")])
        )
        self.btn_scope_all.setChecked(True)
        self.scope_group.buttonToggled.connect(self._on_selection_changed)
        self.btn_select_run = GlassPushButton(" [ALL]", variant="default")
        self.btn_select_run.setFixedHeight(34)
        self.btn_select_run.setIcon(self._icon("folder.svg"))
        self.btn_select_run.clicked.connect(self._select_run)
        scope_row.addWidget(self.scope_segment, 1)
        scope_row.addWidget(self.btn_select_run)
        lay.addLayout(scope_row)

        # Export format: CSV / ZIP / Folder.
        lay.addWidget(self._caption("Export as"))
        self.format_segment, self.format_group, (self.rb_csv, self.rb_zip, self.rb_folder_fmt) = (
            self._make_segment(
                [
                    ("CSV Report", "A single spreadsheet report"),
                    ("ZIP Archive", "One compressed archive of runs"),
                    ("Folder", "A plain folder of run files"),
                ]
            )
        )
        self.format_group.buttonToggled.connect(self._on_format_changed)
        lay.addWidget(self.format_segment)

        # Export name + "copy directly to folder".
        lay.addWidget(self._caption("Export name"))
        name_row = QtWidgets.QHBoxLayout()
        name_row.setContentsMargins(0, 0, 0, 0)
        name_row.setSpacing(8)
        self.name_field = GlassLineEdit()
        self.name_field.setMinimumHeight(34)
        name_row.addWidget(self.name_field, 1)
        self.chk_noname = QtWidgets.QCheckBox("Copy directly to folder")
        self.chk_noname.setStyleSheet(self._radio_qss())
        self.chk_noname.stateChanged.connect(self._on_noname_changed)
        name_row.addWidget(self.chk_noname)
        lay.addLayout(name_row)

        # Date filter: All / Today / Last N {units}.
        lay.addWidget(self._caption("Export by date"))
        date_row = QtWidgets.QHBoxLayout()
        date_row.setContentsMargins(0, 0, 0, 0)
        date_row.setSpacing(8)
        (
            self.filter_segment,
            self.filter_group,
            (self.rb_filter_off, self.rb_filter_today, self.rb_filter_last),
        ) = self._make_segment(
            [("All Dates", ""), ("Today", ""), ("Last…", "")],
            ids=[FILTER_OFF, FILTER_TODAY, FILTER_LAST],
        )
        self.rb_filter_off.setChecked(True)
        self.filter_num = QtWidgets.QLineEdit("7")
        self.filter_num.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.filter_num.setValidator(QtGui.QIntValidator(1, 31))
        self.filter_num.setFixedWidth(40)
        self.filter_num.setMinimumHeight(34)
        self.filter_units = QtWidgets.QComboBox()
        self.filter_units.addItems(["Hours", "Days", "Weeks"])
        self.filter_units.setCurrentText("Days")
        self.filter_units.setMinimumHeight(34)
        date_row.addWidget(self.filter_segment, 1)
        date_row.addWidget(self.filter_num)
        date_row.addWidget(self.filter_units)
        lay.addLayout(date_row)

        # Existing-files policy: Merge / Replace / Skip.
        lay.addWidget(self._caption("When a file already exists"))
        self.policy_segment, self.policy_group, (self.rb_merge, self.rb_replace, self.rb_skip) = (
            self._make_segment(
                [
                    ("Merge", "Keep newer versions"),
                    ("Replace", "Overwrite existing"),
                    ("Skip", "Leave existing untouched"),
                ],
                ids=[POLICY_MERGE, POLICY_REPLACE, POLICY_SKIP],
            )
        )
        self.rb_merge.setChecked(True)
        lay.addWidget(self.policy_segment)

        self.root.addWidget(card)

    # ---- CSV fields card ----------------------------------------------
    def _build_csv_card(self):
        self.csv_card = self._card("CSV Report Fields")
        lay = self.csv_card.layout()
        lay.addWidget(self._caption("Columns to include"))
        if CheckableComboBox is not None:
            self.combo_csv_cols = CheckableComboBox(self)
            self.combo_csv_cols.addItems(CSV_FIELDS)
            for i in range(self.combo_csv_cols.count()):
                self.combo_csv_cols.model().item(i, 0).setCheckState(QtCore.Qt.Checked)
            self.combo_csv_cols.check_items()
            self.combo_csv_cols.model().item(0, 0).setEnabled(False)  # Run Name required
            self.combo_csv_cols.setMinimumHeight(34)
            lay.addWidget(self.combo_csv_cols)
        else:
            self.combo_csv_cols = None
            lbl = QtWidgets.QLabel("CSV field picker unavailable in this context.")
            lbl.setStyleSheet(self._desc_qss())
            lay.addWidget(lbl)
        self.root.addWidget(self.csv_card)

    # ---- Status + actions ---------------------------------------------
    def _build_status_and_actions(self):
        self.status_label = QtWidgets.QLabel("Insert USB drive or choose a folder to begin.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "QLabel { color: rgba(30, 42, 56, 210); font-size: 12px; "
            "background: rgba(255,255,255,40); border: 1px solid rgba(255,255,255,120); "
            "border-radius: 8px; padding: 8px; }"
        )

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        self.btn_cancel = GlassPushButton(" Cancel", variant="default")
        self.btn_cancel.setFixedHeight(34)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.services.request_abort)
        self.btn_export = GlassPushButton(" Export", variant="default")
        self.btn_export.setFixedHeight(34)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._do_export)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_export)

        self.root.addWidget(self.status_label)
        self.root.addLayout(row)

    # ------------------------------------------------------------------
    #  Shared-service hooks
    # ------------------------------------------------------------------
    def on_enter(self):
        self._generate_name()
        self._refresh_target(no_ask=True)
        # Pick up any drive the shared loop already found.
        self.services.usb_add.connect(self._on_usb_add)
        self.services.usb_remove.connect(self._on_usb_remove)
        self._update_export_enabled()

    def on_freeze(self, frozen: bool):
        # Mirror freezeGUI's enable/disable (Erase-only handled elsewhere).
        for w in (self.dest_segment, self.btn_export):
            w.setDisabled(frozen)

    def on_progress(self, label, pct, color):
        if label:
            self.status_label.setText(label)

    # ------------------------------------------------------------------
    #  USB / folder destination state (preserves chk_usb/chk_folder/drive)
    # ------------------------------------------------------------------
    def _set_destination(self, kind):
        """Programmatically set the destination ('usb' or 'folder')."""
        if kind == "usb":
            self.btn_dest_usb.setChecked(True)
        else:
            self.btn_dest_folder.setChecked(True)

    def _on_dest_changed(self, button, checked):
        if not checked:
            return
        is_usb = button is self.btn_dest_usb
        self._chk_usb = is_usb
        self._chk_folder = not is_usb

        # Swap which picker actions are relevant for the destination.
        self.btn_detect.setVisible(is_usb)
        self.btn_eject.setVisible(is_usb)
        self.btn_target.setVisible(not is_usb)

        if is_usb:
            # USB: drive is supplied by detection (shared loop). Reflect current.
            drive = self._drive()
            self.target_field.setText(drive if drive else "[NONE]")
            self.status_label.setText("Insert or detect a USB drive to export.")
        else:
            # Folder: target is whatever was chosen / defaulted.
            t = self.target_field.text()
            self._set_drive(t if t and t != "[NONE]" else None)
            self._refresh_target(no_ask=True)
        self._update_export_enabled()

    def _on_usb_add(self):
        drive = getattr(self.services, "drive", None)
        if self._chk_usb:
            self.target_field.setText(drive if drive else "[NONE]")
        self.status_label.setText(f"[{drive}] USB drive found! Ready to export.")
        self._update_export_enabled()

    def _on_usb_remove(self):
        self.status_label.setText("USB drive removed. Please eject first next time.")
        self._set_drive(None)
        if self._chk_usb:
            self.target_field.setText("[NONE]")
        self._update_export_enabled()

    def _set_drive(self, value):
        # The shared service owns the canonical 'drive'; mirror locally too.
        try:
            self.services.drive = value
        except Exception:
            pass

    def _drive(self):
        return getattr(self.services, "drive", None)

    def _update_export_enabled(self):
        ready = (self._chk_usb or self._chk_folder) and self._drive() is not None
        self.btn_export.setEnabled(bool(ready))

    # ------------------------------------------------------------------
    #  Selection / target pickers
    # ------------------------------------------------------------------
    def _select_target(self):
        start = QtCore.QUrl.fromLocalFile(
            os.path.join(os.path.dirname(Constants.log_prefer_path), "export")
        )
        cur = self.target_field.text()
        if cur and cur != "[NONE]":
            start = QtCore.QUrl.fromLocalFile(cur)
        folder = QtWidgets.QFileDialog.getExistingDirectoryUrl(self, "Select Folder", start)
        if not folder.isValid():
            Log.w(TAG, "User cancelled target folder selection.")
            return
        path = folder.toLocalFile()
        self._set_drive(path)
        self.target_field.setText(path)
        self._update_export_enabled()

    def _refresh_target(self, no_ask=True):
        """Default the folder target (old select_folder_target(no_ask=True))."""
        cur = self.target_field.text()
        if cur and cur != "[NONE]":
            target = cur
        else:
            target = os.path.join(os.path.dirname(Constants.log_prefer_path), "export")
        if no_ask:
            try:
                os.makedirs(target, exist_ok=True)
            except OSError as e:
                Log.e(TAG, f"Export target not accessible: {target}: {e}")
                self._set_drive(None)
                self.target_field.setText("[NONE]")
                return
            if self._chk_folder:
                self._set_drive(target)
                self.target_field.setText(target)
        self._update_export_enabled()

    def _select_run(self):
        data_root = Constants.log_prefer_path
        start = QtCore.QUrl.fromLocalFile(data_root)
        folder = QtWidgets.QFileDialog.getExistingDirectoryUrl(self, "Select Folder", start)
        if not folder.isValid():
            return
        selected = folder.toLocalFile()
        if data_root not in selected:
            self.btn_select_run.setText(" [ALL]")
            self._source_subfolder = ""
            Log.w(TAG, "Selected folder not in logged data path.")
            return
        self.btn_scope_sel.setChecked(True)
        sub = selected.replace(data_root, "").replace("/", Constants.slash)
        sub = sub.strip(Constants.slash)
        self._source_subfolder = sub
        leaf = os.path.split(selected)[1]
        kind = "run:" if sub.count(Constants.slash) == 1 else "dev:"
        self.btn_select_run.setText(f" {kind}{leaf}")
        self._generate_name()

    def _on_selection_changed(self, *_):
        if self.btn_scope_all.isChecked():
            self.btn_select_run.setText(" [ALL]")
            self._source_subfolder = ""
            self._generate_name()

    # ------------------------------------------------------------------
    #  Name / format handlers
    # ------------------------------------------------------------------
    def _generate_name(self):
        _, leaf = os.path.split(self._source_subfolder)
        default = (
            str(datetime.datetime.now())
            .split(" ")[0]
            .replace(":", "")
            .replace("-", "")
            .replace(" ", "_")
            + "_QATCH_EXPORT"
        )
        if leaf:
            default = leaf
        enabled = not self.chk_noname.isChecked()
        self.name_field.setEnabled(enabled)
        self.name_field.setText(default if enabled else "")

    def _on_noname_changed(self, *_):
        self._generate_name()

    def _on_format_changed(self, *_):
        is_csv = self.rb_csv.isChecked()
        self.csv_card.setEnabled(is_csv)
        # CSV relabels the merge/skip policy to Append/Abort (semantics differ).
        if is_csv:
            self.rb_merge.setText("Append")
            self.rb_skip.setText("Abort")
        else:
            self.rb_merge.setText("Merge")
            self.rb_skip.setText("Skip")
        # "Copy directly to folder" only valid when exporting as a Folder.
        if not self.rb_folder_fmt.isChecked():
            self.chk_noname.setEnabled(False)
            if self.chk_noname.isChecked():
                self.chk_noname.setChecked(False)
        else:
            self.chk_noname.setEnabled(True)

    # ------------------------------------------------------------------
    #  USB detect / eject (delegate to shared service)
    # ------------------------------------------------------------------
    def _do_detect(self):
        trigger = getattr(self.services, "request_detect", None)
        if callable(trigger):
            trigger()
        else:
            Log.d(f"{TAG} detect requested (shared loop handles enumeration)")

    def _do_eject(self):
        # Eject task lives with the shared worker; reuse if exposed.
        ejector = getattr(self.services, "eject", None)
        if callable(ejector):
            ejector()
        else:
            Log.d(f"{TAG} eject requested (no shared ejector wired yet)")

    # ------------------------------------------------------------------
    #  Export
    # ------------------------------------------------------------------
    def _compute_filter_min(self):
        """Resolve the date floor from the filter controls. Returns the floor
        (0 = no filter) or raises ValueError on bad input."""
        fid = self.filter_group.checkedId()
        if fid == FILTER_OFF:
            return 0
        if fid == FILTER_TODAY:
            now = datetime.datetime.now(tz.utc)
            local_midnight = now.astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
            return local_midnight.astimezone(tz.utc)
        # FILTER_LAST
        if not self.filter_num.hasAcceptableInput():
            raise ValueError(
                f'"Export by date" range must be 1-31; you entered "{self.filter_num.text()}".'
            )
        n = int(self.filter_num.text())
        unit = self.filter_units.currentText()
        delta = {
            "Hours": datetime.timedelta(hours=n),
            "Days": datetime.timedelta(days=n),
            "Weeks": datetime.timedelta(weeks=n),
        }.get(unit)
        if delta is None:
            raise ValueError(f'Unrecognized date units "{unit}".')
        return datetime.datetime.now(tz.utc) - delta

    def _do_export(self):
        try:
            self._filter_min = self._compute_filter_min()
        except ValueError as e:
            Log.e(TAG, f"Input Error: {e}")
            self.status_label.setText(str(e))
            return

        # ZIP with no name -> prompt for one (matches original).
        if self.rb_zip.isChecked() and not self.name_field.text():
            self._generate_name()
            name, ok = QtWidgets.QInputDialog.getText(
                self,
                "Export Name",
                "File name for this export catalog:",
                text=self.name_field.text(),
            )
            if not ok:
                Log.w(TAG, "User cancelled export name request.")
                return
            self.name_field.setText(name)

        name = self.name_field.text()
        target = self._drive()
        date_filter = self._filter_min
        self.services.run_task(lambda abort: self._export_task(abort, name, target, date_filter))

    def _set_running(self, running):
        QtCore.QMetaObject.invokeMethod(
            self.btn_export,
            "setEnabled",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(bool, not running),
        )
        QtCore.QMetaObject.invokeMethod(
            self.btn_cancel,
            "setEnabled",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(bool, running),
        )

    def _policy_id(self):
        return self.policy_group.checkedId()

    def _export_task(self, abort, name, target, date_filter):
        """STUB — port export_widget.Ui_Export.exportTask here.

        Responsibilities to port (refinement pass):
          - normalize target -> drive/export_path
          - if CSV: build column list from combo_csv_cols, walk runs,
            _append_run_to_csv per run, write the CSV report
          - if ZIP: package export_path into <name>.zip
          - if Folder: copytree(data, export_path, policy, date_filter)
          - honor run scope (_source_subfolder) and date_filter
          - poll `abort`, emit progress on channel self.MODE_KEY
          - on success: write history entry, mark exported, set ready-to-eject
        """
        self._set_running(True)
        try:
            self.services.set_freeze(False)
            self.services.emit_progress(
                self.MODE_KEY, "Export not yet implemented (layout pass).", 0, "b"
            )
            # PORT: real export work goes here.
        except Exception as e:
            Log.e(TAG, f"Export error: {e}")
            self.services.emit_progress(self.MODE_KEY, "Error exporting data!", 100, "r")
        finally:
            self.services.set_freeze(True)
            self._set_running(False)

    def _append_run_to_csv(self, run, cols, date_filter=0):
        """STUB — port export_widget.Ui_Export.appendRunToCsvReport."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    #  Styling helpers (mirror data_mode_import)
    # ------------------------------------------------------------------
    def _caption(self, text):
        w = QtWidgets.QLabel(text)
        w.setStyleSheet(self._caption_qss())
        return w

    @staticmethod
    def _desc_qss():
        return (
            "QLabel { color: rgba(60, 72, 88, 190); font-size: 12px; " "background: transparent; }"
        )

    @staticmethod
    def _caption_qss():
        return (
            "QLabel { color: rgba(60, 72, 88, 160); font-size: 10px; "
            "font-weight: 600; text-transform: uppercase; "
            "letter-spacing: 0.5px; background: transparent; }"
        )

    @staticmethod
    def _radio_qss():
        return (
            "QRadioButton, QCheckBox { color: rgba(40, 50, 65, 200); "
            "font-size: 12px; background: transparent; }"
        )

    @staticmethod
    def _segment_frame_qss():
        return """
            QFrame { background: rgba(255, 255, 255, 60);
                     border: 1px solid rgba(255, 255, 255, 150);
                     border-radius: 8px; }
        """

    @staticmethod
    def _picker_box_qss():
        return """
            QFrame#pickerBox { background: rgba(255, 255, 255, 60);
                               border: 1px solid rgba(255, 255, 255, 150);
                               border-radius: 8px; }
        """

    def _segment_button(self, text, tooltip=""):
        btn = QtWidgets.QToolButton()
        btn.setText(text)
        btn.setCheckable(True)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn.setFixedHeight(28)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        if tooltip:
            btn.setToolTip(tooltip)
        btn.setStyleSheet(self._segment_qss())
        return btn

    @staticmethod
    def _segment_qss():
        return """
            QToolButton {
                background: transparent; border: none; border-radius: 6px;
                color: rgba(40, 50, 65, 190); font-size: 12px; font-weight: 600;
                padding: 0px 10px;
            }
            QToolButton:hover   { background: rgba(255, 255, 255, 80); }
            QToolButton:checked {
                background: rgba(255, 255, 255, 235);
                color: rgba(0, 118, 174, 230);
            }
        """

    def _make_segment(self, items, ids=None):
        """Build a frosted segmented control.

        items : list of (label, tooltip). ids: optional explicit button ids.
        Returns (frame, QButtonGroup, tuple_of_buttons). Exclusive selection.
        """
        frame = QtWidgets.QFrame()
        frame.setObjectName("segFrame")
        frame.setStyleSheet(
            "QFrame#segFrame { background: rgba(255,255,255,60); "
            "border: 1px solid rgba(255,255,255,150); border-radius: 8px; }"
        )
        lay = QtWidgets.QHBoxLayout(frame)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        group = QtWidgets.QButtonGroup(frame)
        group.setExclusive(True)
        buttons = []
        for i, (label, tip) in enumerate(items):
            btn = self._segment_button(label, tip)
            bid = ids[i] if ids is not None else i
            group.addButton(btn, bid)
            lay.addWidget(btn)
            buttons.append(btn)
        return frame, group, tuple(buttons)

    def _icon(self, name):
        try:
            from QATCH.common.architecture import Architecture

            path = os.path.join(Architecture.get_path(), "QATCH", "icons", name)
            if os.path.exists(path):
                return QtGui.QIcon(path)
        except Exception:
            pass
        return QtGui.QIcon()

    def _card(self, title):
        card = QtWidgets.QFrame()
        card.setObjectName("glassPanel")
        card.setStyleSheet("""
            QFrame#glassPanel {
                background: rgba(255, 255, 255, 30);
                border: 1px solid rgba(200, 210, 220, 110);
                border-radius: 10px;
            }
        """)
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        header = QtWidgets.QLabel(title)
        header.setStyleSheet(
            "QLabel { color: #333; font-size: 12px; font-weight: bold; "
            "background: transparent; }"
        )
        lay.addWidget(header)
        return card
