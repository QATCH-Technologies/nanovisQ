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
    export                   -> _do_export (validates range, then run_task)
    exportTask               -> _export_task(abort, name, target, min, max)
    appendRunToCsvReport      -> _append_run_to_csv (+ _build_csv_row)
    copytree                 -> _copytree
    (CSV header expansion)    -> _expand_csv_cols
    (nested-folder cleanup)   -> _flatten_nested
    (export_history.log)      -> _write_history

LAYOUT (responsive redesign):
  The previous layout stacked every caption on its own line above a full-width
  segmented control, producing a tall column that overflowed and collided at the
  minimized overlay width (~17.5% inset). This pass:

    * Wraps the whole page in a QScrollArea so content is ALWAYS reachable at any
      overlay size — nothing clips or overlaps regardless of height.
    * Uses a compact "field" pattern (caption over control, packed tightly) so
      each setting takes one short row instead of two tall ones.
    * Lays the Export Settings out in a responsive grid that collapses from two
      columns to one below a width threshold, keeping controls legible when the
      overlay is small and using the space efficiently when it is large.
    * Pins the status readout + primary actions in a footer below the scroll
      area, so the Export button is always visible.

  All original semantics are preserved verbatim — only widget *arrangement*
  changed. The state handlers, signals, and validation are untouched:
    * USB vs Folder targets are mutually exclusive; selecting one clears the
      other; `drive` tracks the active target.
    * Export-as CSV/ZIP/Folder is exclusive; CSV enables the field picker and
      relabels Merge/Skip -> Append/Abort.
    * Run scope is All vs Selection (a logged-data subfolder).
    * Date filter is Off / Today / Last-N {Hours,Days,Weeks}.
    * Existing-files policy is Merge / Replace / Skip (ids 2 / 1 / 3).

The export pipeline is a full port of ``export_widget.Ui_Export``: CSV report
generation (with column expansion and per-run parsing), ZIP packaging, folder
copytree with existing-files policy + date filtering, nested-folder flattening,
and the export-history log entry.
"""

import os
import csv
import shutil
import zipfile
import datetime
from datetime import timezone as tz

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.core.constants import Constants
from QATCH.common.logger import Logger as Log

from QATCH.ui.widgets.data_mode_base import DataModeWidget
from QATCH.ui.components.glass_push_button import GlassPushButton
from QATCH.ui.components.glass_line_edit import GlassLineEdit

# Parser for reading run capture archives when building the CSV report. Imported
# lazily-safe: if VisQAI isn't importable at layout time, CSV export degrades to
# a clear error rather than crashing the whole module import.
try:
    from QATCH.VisQAI.src.io.parser import Parser
except Exception:  # pragma: no cover - optional at layout time
    Parser = None

TAG = "[DataExport]"

# Existing-files policy ids — preserved from the original btnGroup2.
POLICY_REPLACE = 1
POLICY_MERGE = 2
POLICY_SKIP = 3

# Below this content width the settings grid collapses to a single column so
# paired controls never get crushed at the minimized overlay size.
RESPONSIVE_BREAKPOINT = 560

CSV_FIELDS = [
    "Run Name",
    "Average Viscosity",
    "Std Dev",
    "Viscosity Profile",
    "Temp",
    "Formulation",
    "Notes",
]

# Components a "Formulation" column expands into (one CSV column each), in the
# same order as the original Ui_Export.exportTask.
_FORMULATION_COMPONENTS = [
    "Protein",
    "Stabilizer",
    "Buffer",
    "Surfactant",
    "Salt",
    "Excipient",
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
        self._filter_max = None  # computed date ceiling (None = open-ended)
        # Preserve the original's hidden "Include _unnamed runs" opt-in (default
        # off). No visible control today; flip via _export_unnamed if needed.
        self._export_unnamed = False
        self._exported = False  # set True after a successful export
        self.csv_report_path = None  # path of the CSV report being written

        # Tracks the live column mode of the settings grid so resizeEvent only
        # re-lays-out when the breakpoint is actually crossed.
        self._settings_two_col = None

        # --- Scrollable content host -----------------------------------
        # Everything that can grow lives inside a transparent scroll area; the
        # action bar + status line stay pinned below it so Export is always
        # reachable. This is the core fix for the overflow/overlap at small
        # overlay sizes.
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setObjectName("exportScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll.setStyleSheet(self._scroll_qss())
        self.scroll.viewport().setStyleSheet("background: transparent;")

        self.scroll_host = QtWidgets.QWidget()
        self.scroll_host.setObjectName("exportScrollHost")
        self.scroll_host.setStyleSheet("QWidget#exportScrollHost { background: transparent; }")
        self.content = QtWidgets.QVBoxLayout(self.scroll_host)
        self.content.setContentsMargins(2, 2, 6, 2)  # right pad = room for scrollbar
        self.content.setSpacing(12)

        self._build_target_card()
        self._build_settings_card()
        self._build_csv_card()
        self.content.addStretch(1)

        self.scroll.setWidget(self.scroll_host)
        self.root.addWidget(self.scroll, 1)

        # Pinned footer: status readout + primary actions, always on-screen.
        self._build_status_and_actions()

        # Apply initial enable-state (CSV default, folder destination default).
        self.rb_csv.setChecked(True)
        self._on_format_changed()
        self._set_destination("folder")

    # ---- Target card (USB / Folder) -----------------------------------
    def _build_target_card(self):
        card = self._card("Export Destination", "Where the exported data is written")
        lay = card.body

        # Destination + target share one tidy block: a segmented control on
        # top, then a single picker row that adapts to the chosen destination.
        lay.addWidget(self._caption("Export to"))

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
        lay.addWidget(self._caption("Target"))

        target_row = QtWidgets.QHBoxLayout()
        target_row.setContentsMargins(0, 0, 0, 0)
        target_row.setSpacing(8)
        self.target_field = GlassLineEdit()
        self.target_field.setText("[NONE]")
        self.target_field.setReadOnly(True)
        self.target_field.setMinimumHeight(34)
        self.target_field.setToolTip("Current export destination")

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
        target_row.addWidget(picker_box, 0)
        lay.addLayout(target_row)

        self.content.addWidget(card)

    # ---- Settings card ------------------------------------------------
    def _build_settings_card(self):
        card = self._card("Export Settings", "Choose what gets exported and how")
        lay = card.body

        # --- Export name (prominent, full-width, top of the card) -------
        # Pulled out of the responsive grid and given a larger label + its own
        # banded row so it reads as the primary thing you set before exporting.
        name_band = QtWidgets.QFrame()
        name_band.setObjectName("nameBand")
        name_band.setStyleSheet("""
            QFrame#nameBand {
                background: rgba(255, 255, 255, 55);
                border: 1px solid rgba(255, 255, 255, 140);
                border-radius: 10px;
            }
        """)
        nb = QtWidgets.QVBoxLayout(name_band)
        nb.setContentsMargins(12, 10, 12, 10)
        nb.setSpacing(6)
        name_label = QtWidgets.QLabel("Export Name")
        name_label.setStyleSheet(
            "QLabel { color: rgba(30, 42, 56, 230); font-size: 13px; "
            "font-weight: 700; background: transparent; }"
        )
        nb.addWidget(name_label)

        name_inner = QtWidgets.QHBoxLayout()
        name_inner.setContentsMargins(0, 0, 0, 0)
        name_inner.setSpacing(8)
        self.name_field = GlassLineEdit()
        self.name_field.setMinimumHeight(38)  # taller than grid fields = prominence
        name_inner.addWidget(self.name_field, 1)
        self.chk_noname = QtWidgets.QCheckBox("Copy directly to folder")
        self.chk_noname.setStyleSheet(self._radio_qss())
        self.chk_noname.setToolTip(
            "Write run files straight into the target with no wrapper folder"
        )
        self.chk_noname.stateChanged.connect(self._on_noname_changed)
        name_inner.addWidget(self.chk_noname, 0)
        nb.addLayout(name_inner)
        lay.addWidget(name_band)
        lay.addSpacing(4)

        # The rest of the settings build as self-contained "field" units dropped
        # into a responsive grid that re-flows between 1 and 2 columns.

        # --- Runs to export: All vs Selection (+ choose button) ---------
        self.scope_segment, self.scope_group, (self.btn_scope_all, self.btn_scope_sel) = (
            self._make_segment([("All Runs", ""), ("Selection", "")])
        )
        self.btn_scope_all.setChecked(True)
        self.scope_group.buttonToggled.connect(self._on_selection_changed)
        self.btn_select_run = GlassPushButton(" All", variant="default")
        self.btn_select_run.setFixedHeight(34)
        self.btn_select_run.setIcon(self._icon("folder.svg"))
        self.btn_select_run.clicked.connect(self._select_run)
        scope_inner = QtWidgets.QHBoxLayout()
        scope_inner.setContentsMargins(0, 0, 0, 0)
        scope_inner.setSpacing(8)
        scope_inner.addWidget(self.scope_segment, 1)
        scope_inner.addWidget(self.btn_select_run, 0)
        self.field_scope = self._field("Runs to export", scope_inner)

        # --- Export format: CSV / ZIP / Folder -------------------------
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
        self.field_format = self._field("Export as", self.format_segment)

        # --- Export by date: an explicit Start → End date range. --------
        # This replaces the old All / Today / Last… mode segment entirely; any
        # of those can be expressed as a range, so the range IS the behavior.
        # An empty range (start == minimum sentinel) means "all dates".
        today = QtCore.QDate.currentDate()
        self.date_start = QtWidgets.QDateEdit()
        self.date_start.setCalendarPopup(True)
        self.date_start.setDisplayFormat("yyyy-MM-dd")
        self.date_start.setDate(today.addMonths(-1))
        self.date_start.setMaximumDate(today)
        self.date_start.setMinimumHeight(34)
        self.date_start.setStyleSheet(self._date_qss())
        self.date_end = QtWidgets.QDateEdit()
        self.date_end.setCalendarPopup(True)
        self.date_end.setDisplayFormat("yyyy-MM-dd")
        self.date_end.setDate(today)
        self.date_end.setMaximumDate(today)
        self.date_end.setMinimumHeight(34)
        self.date_end.setStyleSheet(self._date_qss())
        # Keep the window coherent: start can't exceed end, end can't precede start.
        self.date_start.dateChanged.connect(lambda d: self.date_end.setMinimumDate(d))
        self.date_end.dateChanged.connect(
            lambda d: self.date_start.setMaximumDate(min(d, QtCore.QDate.currentDate()))
        )
        self.date_end.setMinimumDate(self.date_start.date())

        date_inner = QtWidgets.QHBoxLayout()
        date_inner.setContentsMargins(0, 0, 0, 0)
        date_inner.setSpacing(8)
        from_lbl = QtWidgets.QLabel("From")
        from_lbl.setStyleSheet(self._inline_lbl_qss())
        to_lbl = QtWidgets.QLabel("to")
        to_lbl.setStyleSheet(self._inline_lbl_qss())
        date_inner.addWidget(from_lbl, 0)
        date_inner.addWidget(self.date_start, 1)
        date_inner.addWidget(to_lbl, 0)
        date_inner.addWidget(self.date_end, 1)
        self.field_date = self._field("Export by date range", date_inner)

        # --- Existing-files policy: Merge / Replace / Skip -------------
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
        self.field_policy = self._field("When a file already exists", self.policy_segment)

        # Responsive grid. Wide layout pairs settings two-up; narrow stacks them.
        self.settings_grid = QtWidgets.QGridLayout()
        self.settings_grid.setContentsMargins(0, 0, 0, 0)
        self.settings_grid.setHorizontalSpacing(16)
        self.settings_grid.setVerticalSpacing(12)
        self.settings_grid.setColumnStretch(0, 1)
        self.settings_grid.setColumnStretch(1, 1)
        self._settings_fields = [
            self.field_scope,
            self.field_format,
            self.field_date,
            self.field_policy,
        ]
        lay.addLayout(self.settings_grid)
        self._relayout_grid(force=True)

        self.content.addWidget(card)

    # ---- CSV fields card ----------------------------------------------
    def _build_csv_card(self):
        self.csv_card = self._card("CSV Report Fields", "Tick the columns to include in the report")
        lay = self.csv_card.body
        lay.addWidget(self._caption("Columns to include"))

        # A checkbox per column, laid out in a responsive grid (re-flows on
        # resize). "Run Name" is required, so its box is checked + disabled.
        self.csv_checks = {}
        self.csv_checks_grid = QtWidgets.QGridLayout()
        self.csv_checks_grid.setContentsMargins(0, 2, 0, 0)
        self.csv_checks_grid.setHorizontalSpacing(18)
        self.csv_checks_grid.setVerticalSpacing(8)
        for field in CSV_FIELDS:
            cb = QtWidgets.QCheckBox(field)
            cb.setStyleSheet(self._check_qss())
            cb.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            cb.setChecked(True)
            if field == "Run Name":
                cb.setEnabled(False)  # always present
                cb.setToolTip("Run Name is always included")
            self.csv_checks[field] = cb
        lay.addLayout(self.csv_checks_grid)
        self._relayout_csv_checks(force=True)
        self.content.addWidget(self.csv_card)

    def _relayout_csv_checks(self, force=False):
        """Flow the column checkboxes into 1–3 columns based on width."""
        avail = self.scroll.viewport().width() if hasattr(self, "scroll") else self.width()
        cols = 3 if avail >= RESPONSIVE_BREAKPOINT else (2 if avail >= 360 else 1)
        if not force and cols == getattr(self, "_csv_check_cols", None):
            return
        self._csv_check_cols = cols
        while self.csv_checks_grid.count():
            item = self.csv_checks_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(self.csv_card)
        for c in range(3):
            self.csv_checks_grid.setColumnStretch(c, 1 if c < cols else 0)
        for idx, field in enumerate(CSV_FIELDS):
            r, c = divmod(idx, cols)
            self.csv_checks_grid.addWidget(self.csv_checks[field], r, c)
        for cb in self.csv_checks.values():
            cb.show()

    def _selected_csv_cols(self):
        """The ordered list of columns the user has ticked (Run Name always
        first/included). Replaces the old combo_csv_cols.check_items() read."""
        return [f for f in CSV_FIELDS if f == "Run Name" or self.csv_checks[f].isChecked()]

    # ---- Status + actions ---------------------------------------------
    def _build_status_and_actions(self):
        footer = QtWidgets.QFrame()
        footer.setObjectName("exportFooter")
        footer.setStyleSheet("QFrame#exportFooter { background: transparent; border: none; }")
        flay = QtWidgets.QVBoxLayout(footer)
        flay.setContentsMargins(0, 4, 0, 0)
        flay.setSpacing(8)

        # Slim progress bar (matches the Import UI). Visible only while an export
        # is running; driven by the pct emitted on the export channel. Replaces
        # the old inline status readout.
        self.export_progress = QtWidgets.QProgressBar()
        self.export_progress.setObjectName("exportProgress")
        self.export_progress.setRange(0, 100)
        self.export_progress.setValue(0)
        self.export_progress.setTextVisible(False)
        self.export_progress.setFixedHeight(3)
        self.export_progress.setVisible(False)
        self.export_progress.setStyleSheet("""
            QProgressBar#exportProgress {
                background: rgba(255, 255, 255, 35);
                border: none;
                border-radius: 1px;
            }
            QProgressBar#exportProgress::chunk {
                background: rgba(0, 118, 174, 120);
                border-radius: 1px;
            }
        """)
        flay.addWidget(self.export_progress)

        # Buttons match the Import UI: compact, content-sized (no minimum
        # width), both the default glass variant, sitting at the right end of
        # the row.
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.btn_cancel = GlassPushButton(" Cancel", variant="default")
        self.btn_cancel.setFixedHeight(34)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.services.request_abort)
        self.btn_export = GlassPushButton(" Export", variant="default")
        self.btn_export.setFixedHeight(34)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._do_export)

        row.addStretch(1)
        row.addWidget(self.btn_cancel, 0)
        row.addWidget(self.btn_export, 0)

        flay.addLayout(row)
        self.root.addWidget(footer, 0)

    # ------------------------------------------------------------------
    #  Responsive grid relayout
    # ------------------------------------------------------------------
    def _relayout_grid(self, force=False):
        """Place the settings fields in 1 or 2 columns based on content width."""
        avail = self.scroll.viewport().width() if hasattr(self, "scroll") else self.width()
        two_col = avail >= RESPONSIVE_BREAKPOINT
        if not force and two_col == self._settings_two_col:
            return
        self._settings_two_col = two_col

        # Detach existing items without deleting the field widgets.
        while self.settings_grid.count():
            item = self.settings_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(self.scroll_host)

        if two_col:
            self.settings_grid.setColumnStretch(0, 1)
            self.settings_grid.setColumnStretch(1, 1)
            for idx, field in enumerate(self._settings_fields):
                r, c = divmod(idx, 2)
                self.settings_grid.addWidget(field, r, c)
        else:
            self.settings_grid.setColumnStretch(0, 1)
            self.settings_grid.setColumnStretch(1, 0)
            for idx, field in enumerate(self._settings_fields):
                self.settings_grid.addWidget(field, idx, 0, 1, 2)
        for field in self._settings_fields:
            field.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "settings_grid"):
            self._relayout_grid()
        if hasattr(self, "csv_checks_grid"):
            self._relayout_csv_checks()

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
        self._relayout_grid(force=True)
        self._relayout_csv_checks(force=True)

    def on_freeze(self, frozen: bool):
        # Mirror freezeGUI's enable/disable (Erase-only handled elsewhere).
        for w in (self.dest_segment, self.btn_export):
            w.setDisabled(frozen)

    def on_progress(self, label, pct, color):
        # The slim bar conveys progress; text labels are logged, not shown.
        try:
            self.export_progress.setValue(max(0, min(100, int(pct))))
        except Exception:
            pass

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
            # Switching to USB: clear any folder target and adopt the currently
            # detected USB drive (if the shared loop already found one).
            drive = self._drive()
            self.target_field.setText(drive if drive else "[NONE]")
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
        Log.i(TAG, f"[{drive}] USB drive found! Ready to export.")
        self._update_export_enabled()

    def _on_usb_remove(self):
        Log.w(TAG, "USB drive removed. Please eject first next time.")
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
            self.btn_select_run.setText(" All")
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
            self.btn_select_run.setText(" All")
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
        # Dim the disabled CSV card so the active format reads clearly.
        self.csv_card.setProperty("dimmed", not is_csv)
        self.csv_card.style().unpolish(self.csv_card)
        self.csv_card.style().polish(self.csv_card)
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
    @staticmethod
    def _qdate_to_utc_floor(qdate):
        """Local midnight at the START of `qdate`, as an aware UTC datetime."""
        local = datetime.datetime(qdate.year(), qdate.month(), qdate.day(), 0, 0, 0).astimezone()
        return local.astimezone(tz.utc)

    @staticmethod
    def _qdate_to_utc_ceiling(qdate):
        """Local midnight at the END of `qdate` (start of next day), UTC aware."""
        nxt = qdate.addDays(1)
        local = datetime.datetime(nxt.year(), nxt.month(), nxt.day(), 0, 0, 0).astimezone()
        return local.astimezone(tz.utc)

    def _compute_filter_min(self):
        """Date floor: local midnight at the START of the selected start date,
        as an aware UTC datetime. Raises ValueError if start is after end."""
        start, end = self.date_start.date(), self.date_end.date()
        if start > end:
            raise ValueError('"Export by date range" start date must be on or before the end date.')
        return self._qdate_to_utc_floor(start)

    def _compute_filter_max(self):
        """Date ceiling: local midnight at the END of the selected end date
        (i.e. start of the following day) so the end day is inclusive."""
        return self._qdate_to_utc_ceiling(self.date_end.date())

    def _do_export(self):
        try:
            self._filter_min = self._compute_filter_min()
            self._filter_max = self._compute_filter_max()
        except ValueError as e:
            Log.e(TAG, f"Input Error: {e}")
            QtWidgets.QMessageBox.warning(self, "Export by date range", str(e))
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
        date_filter_max = self._filter_max
        self.services.run_task(
            lambda abort: self._export_task(abort, name, target, date_filter, date_filter_max)
        )

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
        # Reset to 0 on start, then show/hide the slim bar (worker-thread safe).
        if running:
            QtCore.QMetaObject.invokeMethod(
                self.export_progress,
                "setValue",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(int, 0),
            )
        QtCore.QMetaObject.invokeMethod(
            self.export_progress,
            "setVisible",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(bool, running),
        )

    def _policy_id(self):
        return self.policy_group.checkedId()

    # ------------------------------------------------------------------
    #  CSV column expansion (port of the header-building block in exportTask)
    # ------------------------------------------------------------------
    def _expand_csv_cols(self):
        """Expand the user-ticked CSV fields into concrete report columns.

        Mirrors Ui_Export.exportTask:
          * "Temp"            -> "Temperature"
          * "Formulation"     -> one column per component (Protein, Stabilizer,
                                 Buffer, Surfactant, Salt, Excipient), named
                                 "Formulation_<Component>"
          * "Viscosity Profile" and all others pass through unchanged.
        """
        cols = []
        for field in self._selected_csv_cols():
            if field == "Temp":
                cols.append("Temperature")
            elif field == "Formulation":
                for comp in _FORMULATION_COMPONENTS:
                    cols.append(f"Formulation_{comp}")
            else:
                cols.append(field)
        return cols

    # ------------------------------------------------------------------
    #  Export task — full port of Ui_Export.exportTask
    # ------------------------------------------------------------------
    def _export_task(self, abort, name, output_folder, date_filter, date_filter_max=None):
        """Export the selected runs to the chosen destination.

        Faithful port of ``export_widget.Ui_Export.exportTask`` covering all
        three output formats (CSV report / ZIP archive / plain Folder), the
        existing-files policy, run-scope selection, date filtering, nested-folder
        flattening, and the export-history log entry.

        ``abort`` is a no-arg callable returning True when the user cancels.
        ``date_filter`` is the lower bound (0 = no filter); ``date_filter_max``
        is the optional upper bound from the date-range picker (None = open).
        """
        self._set_running(True)
        self.services.set_freeze(False)
        is_csv = self.rb_csv.isChecked()
        is_zip = self.rb_zip.isChecked()
        try:
            output_folder = output_folder.replace("/", Constants.slash)
            if len(output_folder) > 2:
                drive = output_folder[0:2]
            else:
                drive = output_folder
                output_folder += Constants.slash
            self._set_drive(drive)
            drive_or_folder = "USB drive" if self._chk_usb else "folder"
            data_path = os.path.join(Constants.log_prefer_path)

            nested_marker = f"{Constants.slash}{Constants.log_export_path}{Constants.slash}"
            if nested_marker in output_folder:
                export_path = os.path.join(
                    output_folder[0 : output_folder.rindex(nested_marker)],
                    Constants.log_export_path,
                )
            else:
                export_path = os.path.join(output_folder, name, Constants.log_export_path)

            # --- CSV report header -------------------------------------
            csv_report_cols = []
            if is_csv:
                if Parser is None:
                    self.services.emit_progress(
                        self.MODE_KEY,
                        "CSV export unavailable: run parser not loaded.",
                        100,
                        "r",
                    )
                    return
                csv_report_cols = self._expand_csv_cols()
                Log.d(TAG, f"CSV report cols: {csv_report_cols}")
                export_folder = os.path.split(export_path)[0]
                self.csv_report_path = export_folder + ".csv"
                policy = self._policy_id()
                if os.path.exists(self.csv_report_path):
                    if policy == POLICY_SKIP:  # "Abort" in CSV mode
                        Log.e(TAG, "CSV report already exists; user selected Abort.")
                        self.services.emit_progress(
                            self.MODE_KEY,
                            "CSV report already exists. Export aborted.",
                            100,
                            "r",
                        )
                        return
                    if policy == POLICY_MERGE:  # "Append" in CSV mode
                        with open(self.csv_report_path, "r", newline="") as f:
                            existing_header = next(csv.reader(f), [])
                        if ",".join(csv_report_cols) != ",".join(existing_header):
                            Log.e(TAG, "Existing CSV columns differ; cannot Append.")
                            self.services.emit_progress(
                                self.MODE_KEY,
                                "Existing CSV has different columns. Cannot append.",
                                100,
                                "r",
                            )
                            return
                        Log.d(TAG, "CSV columns match; appending to existing file.")
                    if policy == POLICY_REPLACE:
                        Log.w(TAG, "Replacing existing CSV report file.")
                        os.remove(self.csv_report_path)
                if not os.path.exists(self.csv_report_path):
                    os.makedirs(os.path.dirname(self.csv_report_path), exist_ok=True)
                    with open(self.csv_report_path, "w", newline="") as f:
                        csv.writer(f).writerow(csv_report_cols)

            # --- ZIP: expand an existing archive so we can merge into it
            if is_zip:
                export_folder = os.path.split(export_path)[0]
                zip_path = export_folder + ".zip"
                if os.path.exists(export_folder):
                    Log.w(TAG, "A folder with the same Export Name already exists here.")
                if os.path.exists(zip_path) and self._policy_id() != POLICY_REPLACE:
                    if os.path.exists(export_folder):
                        self.services.emit_progress(
                            self.MODE_KEY,
                            f"[{drive}] Export to {drive_or_folder}: folder already "
                            "exists. Choose a different Export Name.",
                            100,
                            "r",
                        )
                        Log.w(TAG, "Export aborted: target folder exists.")
                        return
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        Log.i(TAG, "Expanding existing ZIP archive.")
                        self.services.emit_progress(
                            self.MODE_KEY, "Expanding existing ZIP archive…", 0, "g"
                        )
                        for info in zf.infolist():
                            if info.filename.endswith("/"):
                                continue
                            extracted = zf.extract(info, export_folder)
                            last_modified = datetime.datetime(*info.date_time).astimezone()
                            epoch = datetime.datetime.fromtimestamp(0, tz=tz.utc)
                            file_time = (last_modified - epoch).total_seconds()
                            os.utime(extracted, (file_time, file_time))

            Log.i(TAG, f"[{drive}] Exporting to {drive_or_folder} {export_path}…")
            self.services.emit_progress(
                self.MODE_KEY, f"[{drive}] Exporting to {drive_or_folder}… please wait…", 0, "g"
            )

            copied = 0
            skipped = 0
            select_device, select_run = os.path.split(self._source_subfolder)
            if select_device == "":
                select_device = select_run
                select_run = ""

            # --- Walk the data tree, exporting each matching run --------
            for _folder, devices, _logs in os.walk(data_path):
                y1 = len(devices)
                z1 = 0
                for x1, device in enumerate(devices):
                    if select_device != "":
                        if select_device != device:
                            continue
                        y1 = 1
                        z1 = x1
                    device_path = os.path.join(data_path, device)
                    for _f2, runs, _files in os.walk(device_path):
                        y2 = len(runs) or 1
                        z2 = 0
                        for x2, run in enumerate(runs):
                            if select_run != "":
                                if select_run != run:
                                    continue
                                y2 = 1
                                z2 = x2 - 0.5
                            pct = min(99, max(1, int(100 * (((x1 - z1) + ((x2 - z2) / y2)) / y1))))
                            if abort.is_set():
                                self.services.emit_progress(
                                    self.MODE_KEY,
                                    f"[{drive}] Export cancelled. Partial export performed.",
                                    pct,
                                    "b",
                                )
                                Log.w(TAG, "Export cancelled by user.")
                                return
                            t_run = run
                            is_unnamed = False
                            try:
                                run_files = os.listdir(os.path.join(device_path, run))
                                if t_run == "_unnamed":
                                    is_unnamed = True
                                    t_run = run_files[0][0:-4]
                                if device == "_unnamed":
                                    is_unnamed = True
                            except Exception:
                                pass
                            self.services.emit_progress(
                                self.MODE_KEY,
                                f"[{drive}] Exporting to {drive_or_folder}… exporting '{t_run}'",
                                pct,
                                "g",
                            )
                            if is_unnamed and not self._export_unnamed:
                                continue
                            src = os.path.join(data_path, device, run)
                            dst = os.path.join(export_path, device, run)
                            if not os.path.exists(src):
                                Log.w(TAG, f"Skipping non-existent folder: {src}")
                                continue
                            if is_csv:
                                if self._append_run_to_csv(
                                    src, csv_report_cols, date_filter, date_filter_max
                                ):
                                    copied += 1
                                else:
                                    skipped += 1
                            else:
                                copied, skipped = self._copytree(
                                    src,
                                    dst,
                                    self._policy_id(),
                                    copied,
                                    skipped,
                                    date_filter,
                                    date_filter_max,
                                )

            # --- Flatten nested folders (folder/ZIP output only) -------
            if not is_csv:
                self._flatten_nested(export_path)

            # --- ZIP packaging -----------------------------------------
            if is_zip:
                self.services.emit_progress(
                    self.MODE_KEY,
                    "Creating ZIP archive… this may take a while for large exports…",
                    99,
                    "g",
                )
                export_path = os.path.split(export_path)[0]
                zip_path = export_path + ".zip"
                if os.path.exists(zip_path):
                    Log.w(TAG, "Overwriting existing ZIP archive.")
                shutil.make_archive(export_path, "zip", export_path)
                shutil.rmtree(export_path)

            Log.i(TAG, f"DONE — exported {copied} run(s) to {export_path}.")
            if skipped > 0:
                if is_csv:
                    reason = "there were errors with the analyze results"
                elif date_filter == 0:
                    reason = "they already existed in the output location"
                else:
                    reason = "date filtering was enabled"
                Log.i(TAG, f"Skipped {skipped} run(s) because {reason}.")

            self._write_history(data_path, export_path, copied, skipped, is_zip, date_filter)

            finished_msg = f"[{drive}] Exported to {drive_or_folder}!"
            if drive_or_folder != "folder":
                finished_msg += " Ready to eject."
            self.services.emit_progress(self.MODE_KEY, finished_msg, 100, "g")
            self._exported = True
            # Mirror onto the shared service so a later Advanced-mode "erase"
            # can tell whether local data has been exported yet (matches the
            # original Ui_Export.exported shared flag).
            try:
                self.services.exported = True
            except Exception:
                pass
        except Exception as e:
            Log.e(TAG, f"Export error: {e}")
            self.services.emit_progress(self.MODE_KEY, "Error exporting local data!", 100, "r")
        finally:
            self.services.set_freeze(True)
            self._set_running(False)

    # ------------------------------------------------------------------
    #  Nested-folder flattening (port of the flatten block in exportTask)
    # ------------------------------------------------------------------
    def _flatten_nested(self, export_path):
        """Collapse single-child folder chains created under export_path.

        Direct port of the 'remove nested folders' loop in the original.
        """
        Log.d(TAG, f"Checking for nested folders at {export_path}")
        top_level = os.path.split(export_path)[0]
        path = export_path
        while os.path.exists(path):
            entries = os.listdir(path)
            files = [f for f in entries if not os.path.isdir(os.path.join(path, f))]
            folders = [f for f in entries if os.path.isdir(os.path.join(path, f))]
            if len(files) == 0 and len(folders) == 1:
                path = os.path.join(path, folders[0])
                Log.d(TAG, f"Moving into path: {path}")
                continue
            src = path + Constants.slash
            if len(files) > 1:
                dst = os.path.join(top_level, os.path.split(path)[1]) + Constants.slash
            else:
                dst = top_level
            Log.d(TAG, f"Moving nested folders from {src} to {dst}…")
            if not os.path.exists(dst):
                os.makedirs(dst)
            if not os.path.samefile(src, dst):
                self._copytree(src, dst, self._policy_id())
                shutil.rmtree(export_path)
            else:
                Log.d(TAG, "Nested directory points to itself; leaving as-is.")
            break

    # ------------------------------------------------------------------
    #  CSV row builder — full port of Ui_Export.appendRunToCsvReport
    # ------------------------------------------------------------------
    def _append_run_to_csv(self, run, cols, date_filter=0, date_filter_max=None):
        """Parse one run and append a row to the open CSV report.

        Returns True on success, False if the run was skipped (date filtered,
        not analyzed, missing data, or a conversion error). Faithful port of
        the original, with the date-range upper bound added.
        """
        run_name = os.path.basename(run)
        viscosity_profile = []
        average_viscosity = np.nan
        std_dev = np.nan
        temperature = np.nan
        formulation = None
        notes = "Unknown"
        success = True

        try:
            files = os.listdir(run)

            # Date filtering: use the newest file mtime in the run folder.
            if date_filter != 0 or date_filter_max is not None:
                epoch = datetime.datetime.fromtimestamp(0, tz=tz.utc)
                last_modified = epoch
                for f in files:
                    st_mtime = datetime.datetime.fromtimestamp(
                        os.stat(os.path.join(run, f)).st_mtime, tz=tz.utc
                    )
                    if st_mtime > last_modified:
                        last_modified = st_mtime
                if date_filter != 0 and last_modified < date_filter:
                    return False  # older than the floor
                if date_filter_max is not None and last_modified >= date_filter_max:
                    return False  # newer than the ceiling

            Log.i(TAG, f"Exporting {run} to CSV Report…")
            file_path = os.path.join(run, "capture.zip")
            if os.path.exists(file_path):
                parser = Parser(file_path)

                if "Run Name" in cols:
                    parsed_name = parser.get_run_name()
                    if parsed_name:
                        run_name = parsed_name

                if "Notes" in cols:
                    notes = parser.get_run_notes()
                    if notes:
                        notes = (
                            notes.strip()
                            .encode(encoding="ascii", errors="xmlcharrefreplace")
                            .decode(encoding="utf-8", errors="ignore")
                        )

                require_formulation = any(
                    col in ["Temperature", "Viscosity Profile", "Average Viscosity", "Std Dev"]
                    or col.startswith("Formulation_")
                    for col in cols
                )
                if require_formulation:
                    formulation = parser.get_formulation()

                if "Temperature" in cols:
                    if formulation and formulation.temperature:
                        temperature = formulation.temperature

                require_vp = any(
                    col in ["Viscosity Profile", "Average Viscosity", "Std Dev"] for col in cols
                )
                if require_vp:
                    if formulation and formulation.viscosity_profile:
                        viscosity_profile = formulation.viscosity_profile.viscosities
                    else:
                        raise FileNotFoundError(
                            "Run has no measured Viscosity Profile. Has it been analyzed?"
                        )

                if "Average Viscosity" in cols:
                    average_viscosity = np.average(viscosity_profile)
                if "Std Dev" in cols:
                    std_dev = np.std(viscosity_profile)
            else:
                Log.e(TAG, f"Run {os.path.basename(run)} has no run data file. Cannot export!")
                success = False
        except FileNotFoundError:
            Log.e(
                TAG,
                f"Run {run_name} has not been analyzed and cannot be exported. "
                "Please analyze before exporting to CSV.",
            )
            success = False
        except Exception as e:
            Log.e(TAG, f"Run {os.path.basename(run)} error ({e}). Cannot export!")
            success = False

        try:
            row = self._build_csv_row(
                cols,
                run_name,
                viscosity_profile,
                average_viscosity,
                std_dev,
                temperature,
                formulation,
                notes,
            )
            if row is None:
                success = False
                row = []

            # Round floats to 2 dp where possible (Notes stays raw text).
            def is_float(value):
                try:
                    float(str(value))
                    return True
                except (ValueError, TypeError):
                    return False

            try:
                for i, val in enumerate(row):
                    if isinstance(val, list):
                        for j, inner in enumerate(val):
                            val[j] = float(f"{inner:2.2f}") if is_float(inner) else str(inner)
                    row[i] = f"{val:2.2f}" if is_float(val) and cols[i] != "Notes" else str(val)
            except Exception:
                Log.w(TAG, "Could not round row; using raw values.")
                row = [str(v) for v in row]

            if len(cols) != len(row):
                Log.e(
                    TAG,
                    f"Run {os.path.basename(run)} column count mismatch "
                    f"({len(cols)} != {len(row)}). Cannot export!",
                )
                success = False

            if self.csv_report_path:
                with open(self.csv_report_path, "a", newline="") as f:
                    csv.writer(f).writerow(row)
            else:
                raise ValueError("CSV file path not set; cannot export row")
        except Exception as e:
            Log.e(TAG, f"Run {os.path.basename(run)} could not be written to CSV ({e}).")
            success = False

        return success

    @staticmethod
    def _formulation_cell(formulation, attr):
        """Format one formulation component as '<conc> <units> <name>' or ''."""
        if not formulation:
            return ""
        component = getattr(formulation, attr, None)
        if component and component.ingredient.name != "None":
            return f"{component.concentration} {component.units} {component.ingredient.name}"
        return ""

    def _build_csv_row(
        self,
        cols,
        run_name,
        viscosity_profile,
        average_viscosity,
        std_dev,
        temperature,
        formulation,
        notes,
    ):
        """Assemble the CSV row values in column order. Returns None on an
        unknown column (signals failure to the caller)."""
        comp_attr = {
            "Formulation_Protein": "protein",
            "Formulation_Stabilizer": "stabilizer",
            "Formulation_Buffer": "buffer",
            "Formulation_Surfactant": "surfactant",
            "Formulation_Salt": "salt",
            "Formulation_Excipient": "excipient",
        }
        row = []
        for col in cols:
            if col == "Run Name":
                row.append(run_name)
            elif col == "Viscosity Profile":
                row.append(viscosity_profile)
            elif col == "Average Viscosity":
                row.append(average_viscosity)
            elif col == "Std Dev":
                row.append(std_dev)
            elif col == "Temperature":
                row.append(temperature)
            elif col in comp_attr:
                row.append(self._formulation_cell(formulation, comp_attr[col]))
            elif col == "Notes":
                row.append(notes if notes else "")
            else:
                row.append("Unknown")
                return None
        return row

    # ------------------------------------------------------------------
    #  copytree — full port of Ui_Export.copytree
    # ------------------------------------------------------------------
    def _copytree(self, src, dst, policy, copied=0, skipped=0, date_filter=0, date_filter_max=None):
        """Recursively copy ``src`` into ``dst`` honoring the existing-files
        policy and date filtering. Counts .xml files copied vs skipped.

        ``policy``: POLICY_REPLACE overwrites all; POLICY_MERGE overwrites only
        when the source is >2s newer; POLICY_SKIP leaves existing files. New
        files are always copied. Direct port of the original.
        """
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                copied, skipped = self._copytree(
                    s, d, policy, copied, skipped, date_filter, date_filter_max
                )
                continue
            allow_copy = False
            if policy == POLICY_REPLACE:
                allow_copy = True
            elif not os.path.exists(d):
                allow_copy = True
            elif policy == POLICY_MERGE:
                last_mod = datetime.datetime.fromtimestamp(os.stat(s).st_mtime, tz=tz.utc)
                exist_mod = datetime.datetime.fromtimestamp(os.stat(d).st_mtime, tz=tz.utc)
                if last_mod - exist_mod > datetime.timedelta(seconds=2):
                    allow_copy = True

            if allow_copy and (date_filter != 0 or date_filter_max is not None):
                # Recency filter: newest file mtime in the source folder.
                if "_unnamed" in src:
                    last_modified = datetime.datetime.fromtimestamp(os.stat(s).st_mtime, tz=tz.utc)
                else:
                    epoch = datetime.datetime.fromtimestamp(0, tz=tz.utc)
                    last_modified = epoch
                    for f in os.listdir(src):
                        fp = os.path.join(src, f)
                        if os.path.isdir(fp):
                            continue
                        st_mtime = datetime.datetime.fromtimestamp(os.stat(fp).st_mtime, tz=tz.utc)
                        if st_mtime > last_modified:
                            last_modified = st_mtime
                if date_filter != 0 and last_modified < date_filter:
                    allow_copy = False
                if date_filter_max is not None and last_modified >= date_filter_max:
                    allow_copy = False

            if allow_copy:
                if not os.path.exists(dst):
                    os.makedirs(dst)
                if item.endswith(".xml"):
                    copied += 1
                shutil.copy2(s, d)
            else:
                if item.endswith(".xml"):
                    skipped += 1
        return copied, skipped

    # ------------------------------------------------------------------
    #  Export history log (port of the history block in exportTask)
    # ------------------------------------------------------------------
    def _write_history(self, data_path, export_path, copied, skipped, is_zip, date_filter):
        """Prepend an HTML entry to export_history.log (same format the History
        view parses)."""
        history_path = os.path.join(os.getcwd(), Constants.log_export_path, "export_history.log")
        try:
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    log_lines = f.read()
            else:
                log_lines = ""

            scope_btn = self.scope_group.checkedButton()
            scope_text = scope_btn.text() if scope_btn else "All Runs"
            scope_detail = self.btn_select_run.text() if self.btn_scope_sel.isChecked() else ""
            fmt_btn = self.format_group.checkedButton()
            fmt_text = fmt_btn.text() if fmt_btn else "CSV Report"
            policy_btn = self.policy_group.checkedButton()
            policy_text = policy_btn.text() if policy_btn else "Merge"
            ts = str(datetime.datetime.now()).split(".")[0]

            with open(history_path, "w") as f:
                f.write(f"<b>Exported {copied} run(s) at {ts}</b><br/>\n")
                f.write(f'<small>from "{data_path}" <br/>\n')
                f.write('to "{}{}"</small><br/>\n'.format(export_path, ".zip" if is_zip else ""))
                f.write("<small>Settings: ")
                f.write(f"Export {scope_text}{scope_detail}, ")
                f.write(f"{fmt_text}, ")
                f.write(f"{policy_text} existing files</small><br/>\n")
                if skipped > 0:
                    reason = (
                        "overwrites were disabled" if date_filter == 0 else "filtering was enabled"
                    )
                    f.write(f"<small>Skipped {skipped} run(s) since {reason}.</small><br/>\n")
                f.write("<br/>\n")
                f.write(log_lines)  # prepend
        except Exception as e:
            Log.e(TAG, f"Failed writing export history: {e}")

    # ------------------------------------------------------------------
    #  Layout helpers
    # ------------------------------------------------------------------
    def _field(self, caption_text, control):
        """A compact caption-over-control unit used as a grid cell.

        `control` may be a QWidget or a QLayout. Returns a QWidget so it can be
        placed into the responsive grid and shown/hidden as one.
        """
        wrap = QtWidgets.QWidget()
        wrap.setStyleSheet("background: transparent;")
        v = QtWidgets.QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(5)
        v.addWidget(self._caption(caption_text))
        if isinstance(control, QtWidgets.QLayout):
            v.addLayout(control)
        else:
            v.addWidget(control)
        return wrap

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

    @staticmethod
    def _scroll_qss():
        return """
            QScrollArea#exportScroll { background: transparent; border: none; }
            QScrollBar:vertical {
                background: transparent; width: 8px; margin: 2px 0 2px 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(120, 134, 150, 110);
                border-radius: 4px; min-height: 28px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(90, 104, 120, 150);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        """

    @staticmethod
    def _inline_lbl_qss():
        return (
            "QLabel { color: rgba(60, 72, 88, 200); font-size: 12px; " "background: transparent; }"
        )

    @staticmethod
    def _check_qss():
        """Glass-friendly checkbox styling for the CSV column pickers."""
        return """
            QCheckBox {
                color: rgba(40, 50, 65, 215); font-size: 12px;
                background: transparent; spacing: 8px; padding: 2px 0px;
            }
            QCheckBox:disabled { color: rgba(40, 50, 65, 120); }
            QCheckBox::indicator {
                width: 16px; height: 16px; border-radius: 5px;
                border: 1px solid rgba(120, 130, 145, 170);
                background: rgba(255, 255, 255, 160);
            }
            QCheckBox::indicator:hover {
                border: 1px solid rgba(10, 163, 230, 200);
            }
            QCheckBox::indicator:checked {
                background: rgba(10, 163, 230, 230);
                border: 1px solid rgba(10, 163, 230, 235);
                image: url(none);
            }
            QCheckBox::indicator:checked:disabled {
                background: rgba(10, 163, 230, 120);
                border: 1px solid rgba(10, 163, 230, 120);
            }
        """

    @staticmethod
    def _date_qss():
        """Glass styling for QDateEdit that mirrors the line-edit/combo look."""
        return """
            QDateEdit {
                background: rgba(255, 255, 255, 150);
                border: 1px solid rgba(120, 130, 145, 150);
                border-radius: 14px; padding-left: 12px; padding-right: 6px;
                color: rgb(40, 50, 62); font-weight: bold; min-height: 26px;
            }
            QDateEdit:hover {
                background: rgba(255, 255, 255, 200);
                border: 1px solid rgba(90, 100, 115, 190);
            }
            QDateEdit:focus {
                background: rgba(255, 255, 255, 225);
                border: 1px solid rgba(10, 163, 230, 200);
            }
            QDateEdit::drop-down {
                border: none; background: transparent; width: 22px;
                subcontrol-position: center right; margin-right: 4px;
            }
            QDateEdit::down-arrow {
                width: 0px; height: 0px;
            }
            QCalendarWidget QWidget { alternate-background-color: rgba(235,240,246,255); }
            QCalendarWidget QAbstractItemView:enabled {
                color: rgb(40, 50, 62);
                selection-background-color: rgba(10, 163, 230, 60);
                selection-color: rgb(20, 30, 40);
            }
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
            QToolButton:disabled { color: rgba(40, 50, 65, 90); }
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

    def _card(self, title, subtitle=""):
        """A frosted glass panel. Returns the QFrame with a `.body` QVBoxLayout
        for callers to populate (header + optional subtitle are pre-added)."""
        card = QtWidgets.QFrame()
        card.setObjectName("glassPanel")
        card.setStyleSheet("""
            QFrame#glassPanel {
                background: rgba(255, 255, 255, 30);
                border: 1px solid rgba(200, 210, 220, 110);
                border-radius: 10px;
            }
            QFrame#glassPanel[dimmed="true"] {
                background: rgba(255, 255, 255, 14);
                border: 1px solid rgba(200, 210, 220, 70);
            }
        """)
        outer = QtWidgets.QVBoxLayout(card)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)

        header = QtWidgets.QLabel(title)
        header.setStyleSheet(
            "QLabel { color: #333; font-size: 12px; font-weight: bold; "
            "background: transparent; }"
        )
        outer.addWidget(header)
        if subtitle:
            sub = QtWidgets.QLabel(subtitle)
            sub.setStyleSheet(self._desc_qss())
            sub.setWordWrap(True)
            outer.addWidget(sub)

        # Body holds the actual controls; callers append here.
        card.body = QtWidgets.QVBoxLayout()
        card.body.setContentsMargins(0, 2, 0, 0)
        card.body.setSpacing(8)
        outer.addLayout(card.body)
        return card
