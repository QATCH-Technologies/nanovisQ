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
from QATCH.ui.components.glass_option_card import GlassOptionCard, GlassOptionCardGroup

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


class _Stepper(QtWidgets.QWidget):
    """Horizontal numbered-step indicator for the Export wizard.

    Circles connected by thin rule lines; the current step is filled solid,
    reached-but-not-current steps are outlined/tinted, future steps are plain
    gray. Clicking a step the user has already reached jumps back to it.
    """

    stepClicked = QtCore.pyqtSignal(int)

    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self._current = 0
        self._max_reached = 0
        self._circles = []
        self._captions = []
        self._lines = []

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 8)
        outer.setSpacing(0)

        # Circles and connecting lines live in their OWN grid row (row 0),
        # with captions in a separate row (row 1) below. Keeping captions out
        # of row 0 means row 0's height is just the circle's height, so a
        # vertically-centered line lands on the circle's true center instead
        # of the midpoint of "circle + caption" as a combined column.
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(0)
        grid.setVerticalSpacing(4)

        col = 0
        for i, label in enumerate(labels):
            if i > 0:
                line = QtWidgets.QFrame()
                line.setFixedHeight(2)
                line.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                grid.addWidget(line, 0, col, QtCore.Qt.AlignVCenter)
                grid.setColumnStretch(col, 1)
                self._lines.append(line)
                col += 1

            circle = QtWidgets.QToolButton()
            circle.setText(str(i + 1))
            circle.setFixedSize(26, 26)
            circle.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            circle.clicked.connect(lambda _=False, idx=i: self._on_clicked(idx))
            grid.addWidget(circle, 0, col, QtCore.Qt.AlignCenter)

            cap = QtWidgets.QLabel(label)
            cap.setAlignment(QtCore.Qt.AlignCenter)
            grid.addWidget(cap, 1, col, QtCore.Qt.AlignHCenter)

            self._circles.append(circle)
            self._captions.append(cap)
            grid.setColumnStretch(col, 0)
            col += 1

        outer.addLayout(grid)
        self._restyle()

    def _on_clicked(self, idx):
        if idx <= self._max_reached:
            self.stepClicked.emit(idx)

    def set_current(self, index):
        self._current = index
        self._max_reached = max(self._max_reached, index)
        self._restyle()

    def reset(self):
        """Clear "reached" progress entirely — used when the wizard itself
        resets, so old steps don't keep showing as done/clickable."""
        self._current = 0
        self._max_reached = 0
        self._restyle()

    def _restyle(self):
        # setStyleSheet() alone can leave a stale rendered pixmap behind for
        # QSS-styled QToolButtons during rapid restyles (each step click
        # restyles two circles at once) — an explicit unpolish/polish +
        # update() forces an immediate, clean repaint instead of a "ghost"
        # of the previous state lingering under the new one.
        for i, circle in enumerate(self._circles):
            if i == self._current:
                state = "current"
            elif i <= self._max_reached:
                state = "done"
            else:
                state = "future"
            circle.setStyleSheet(self._circle_qss(state))
            circle.style().unpolish(circle)
            circle.style().polish(circle)
            circle.update()

            cap = self._captions[i]
            cap.setStyleSheet(self._caption_qss(active=(i == self._current)))
            cap.style().unpolish(cap)
            cap.style().polish(cap)
            cap.update()
        for i, line in enumerate(self._lines):
            line.setStyleSheet(self._line_qss(done=(i < self._max_reached)))
            line.style().unpolish(line)
            line.style().polish(line)
            line.update()

    @staticmethod
    def _circle_qss(state):
        if state == "current":
            body = "background: rgba(0, 118, 174, 235); color: white; border: none;"
        elif state == "done":
            body = (
                "background: rgba(10, 163, 230, 90); color: rgba(0, 90, 135, 245); "
                "border: 1px solid rgba(0, 118, 174, 150);"
            )
        else:
            body = (
                "background: rgba(255, 255, 255, 90); color: rgba(60, 72, 88, 170); "
                "border: 1px solid rgba(180, 195, 210, 170);"
            )
        return f"QToolButton {{ {body} border-radius: 13px; font-weight: 700; font-size: 12px; }}"

    @staticmethod
    def _caption_qss(active):
        # Weight is constant (700) regardless of state — varying it between
        # active/inactive changes the text's rendered width slightly, which
        # made the whole bar visibly jitter/resize on every step transition.
        # Only colour differentiates the active step now.
        color = "rgba(0, 90, 135, 245)" if active else "rgba(60, 72, 88, 160)"
        return (
            f"QLabel {{ color: {color}; font-size: 10px; font-weight: 700; "
            "background: transparent; }"
        )

    @staticmethod
    def _line_qss(done):
        color = "rgba(0, 118, 174, 150)" if done else "rgba(190, 200, 212, 140)"
        return f"QFrame {{ background: {color}; border: none; }}"


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
        self._task_running = False  # drives Cancel's dual abort/reset behavior

        # Tracks the live column mode of the responsive grids so resizeEvent
        # only re-lays-out when a breakpoint is actually crossed.
        self._settings_two_col = None

        # --- Stepper + step pages ---------------------------------------
        self._step_labels = ["Destination", "Scope", "Fields", "Review"]
        self.stepper = _Stepper(self._step_labels)
        self.stepper.stepClicked.connect(self._go_to_step)
        self.root.addWidget(self.stepper)

        self.step_stack = QtWidgets.QStackedWidget()
        self.dest_scroll = self._make_step_scroll(self._build_destination_page())
        self.scope_scroll = self._make_step_scroll(self._build_scope_page())
        self.fields_scroll = self._make_step_scroll(self._build_fields_page())
        self.review_scroll = self._make_step_scroll(self._build_review_page())
        for scroll in (self.dest_scroll, self.scope_scroll, self.fields_scroll, self.review_scroll):
            self.step_stack.addWidget(scroll)
        self.root.addWidget(self.step_stack, 1)

        # Pinned footer: progress bar + Cancel / Back / Next-or-Export.
        self._build_status_and_actions()

        # Apply initial enable-state (CSV default, folder destination default).
        self.rb_csv.setChecked(True)
        self._on_format_changed()
        self._set_destination("folder")
        self._step = 0
        self.stepper.set_current(0)
        self.btn_back.setEnabled(False)

    @staticmethod
    def _make_step_scroll(content_widget):
        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("exportScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet(ExportMode._scroll_qss())
        scroll.viewport().setStyleSheet("background: transparent;")
        scroll.setWidget(content_widget)
        return scroll

    @staticmethod
    def _step_host():
        host = QtWidgets.QWidget()
        host.setObjectName("exportScrollHost")
        host.setStyleSheet("QWidget#exportScrollHost { background: transparent; }")
        outer = QtWidgets.QVBoxLayout(host)
        outer.setContentsMargins(2, 2, 6, 2)  # right pad = room for scrollbar
        outer.setSpacing(12)
        return host, outer

    # ---- Step 0: Destination -------------------------------------------
    def _build_destination_page(self):
        host, outer = self._step_host()
        card = self._card("Export Destination", "Where the exported data is written")
        lay = card.body

        # Destination is a pair of labelled cards (radio-style); target picker
        # adapts to whichever one is selected.
        lay.addWidget(self._caption("Export to"))
        self.dest_group = GlassOptionCardGroup(self)
        dest_row = QtWidgets.QHBoxLayout()
        dest_row.setContentsMargins(0, 0, 0, 0)
        dest_row.setSpacing(8)
        self.card_usb = GlassOptionCard("USB Drive", "Removable storage", show_radio=True)
        self.card_folder = GlassOptionCard(
            "Folder on this PC", "Local or network path", show_radio=True
        )
        self.dest_group.addCard(self.card_usb, 0)
        self.dest_group.addCard(self.card_folder, 1)
        self.dest_group.toggled.connect(self._on_dest_changed)
        dest_row.addWidget(self.card_usb, 1)
        dest_row.addWidget(self.card_folder, 1)
        lay.addLayout(dest_row)

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
        # USB actions — borderless (ghost variant: blue-accent hover wash,
        # matching the app's accent colour) since they already sit inside
        # pickerBox's own frosted border; a border per-button too would be
        # one too many. Fixed size policy keeps them from being stretched by
        # the layout's surplus space while a sibling slides open/closed.
        self.btn_detect = GlassPushButton(" Detect", variant="ghost")
        self.btn_detect.setFixedHeight(28)
        self.btn_detect.setIcon(self._icon("usb.svg"))
        self.btn_detect.clicked.connect(self._do_detect)
        self.btn_detect.set_border_visible(False)
        self.btn_eject = GlassPushButton(" Eject", variant="ghost")
        self.btn_eject.setFixedHeight(28)
        self.btn_eject.clicked.connect(self._do_eject)
        self.btn_eject.set_border_visible(False)
        # Folder action
        self.btn_target = GlassPushButton(" Choose…", variant="ghost")
        self.btn_target.setFixedHeight(28)
        self.btn_target.setIcon(self._icon("folder.svg"))
        self.btn_target.clicked.connect(self._select_target)
        self.btn_target.set_border_visible(False)
        for btn in (self.btn_detect, self.btn_eject, self.btn_target):
            btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.sep_detect_eject = self._picker_separator()
        self.sep_eject_choose = self._picker_separator()
        picker_lay.addWidget(self.btn_detect)
        picker_lay.addWidget(self.sep_detect_eject)
        picker_lay.addWidget(self.btn_eject)
        picker_lay.addWidget(self.sep_eject_choose)
        picker_lay.addWidget(self.btn_target)

        # Detect/Eject (and their separators) only apply to a USB
        # destination; start collapsed (Folder is the default) and slide
        # open/closed on _on_dest_changed.
        for btn in (self.btn_detect, self.btn_eject):
            btn._natural_w = btn.sizeHint().width()
            btn.setMaximumWidth(0)
            btn.setVisible(False)
        self.sep_detect_eject.setVisible(False)
        self.sep_eject_choose.setVisible(False)

        target_row.addWidget(self.target_field, 1)
        target_row.addWidget(picker_box, 0)
        lay.addLayout(target_row)

        lay.addSpacing(4)
        self.chk_dated_subfolder = QtWidgets.QCheckBox("Create a dated subfolder for this export")
        self.chk_dated_subfolder.setStyleSheet(self._radio_qss())
        self.chk_dated_subfolder.setChecked(True)
        self.chk_dated_subfolder.setToolTip(
            "When unchecked, files are copied directly into the target with no wrapper folder."
        )
        self.chk_dated_subfolder.stateChanged.connect(self._on_dated_subfolder_changed)
        lay.addWidget(self.chk_dated_subfolder)

        outer.addWidget(card)
        outer.addStretch(1)
        return host

    # ---- Step 1: Scope --------------------------------------------------
    def _build_scope_page(self):
        host, outer = self._step_host()
        self.scope_host = host
        card = self._card("Export Scope", "Choose what gets exported and how")
        lay = card.body

        # --- Export name (prominent, full-width, top of the card) -------
        # Pulled out of the responsive grid and given a larger label + its own
        # banded row so it reads as the primary thing you set before exporting.
        name_band = QtWidgets.QFrame()
        name_band.setObjectName("nameBand")
        name_band.setStyleSheet("""
            QFrame#nameBand {
                background: rgba(222, 238, 248, 130);
                border: 1px solid rgba(140, 170, 195, 190);
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
        self.name_field = GlassLineEdit()
        self.name_field.setMinimumHeight(38)  # taller than grid fields = prominence
        nb.addWidget(self.name_field)
        lay.addWidget(name_band)
        lay.addSpacing(4)

        # The rest of the settings build as self-contained "field" units dropped
        # into a responsive grid that re-flows between 1 and 2 columns.

        # --- Runs to export: All vs Selection (+ choose button) ---------
        self.scope_group = GlassOptionCardGroup(self)
        self.btn_scope_all = GlassOptionCard("All Runs", "Export every run")
        self.btn_scope_sel = GlassOptionCard("Selection", "Choose a device or run")
        self.scope_group.addCard(self.btn_scope_all, 0)
        self.scope_group.addCard(self.btn_scope_sel, 1)
        self.btn_scope_all.setChecked(True)
        self.scope_group.toggled.connect(self._on_selection_changed)
        self.btn_select_run = GlassPushButton(" All", variant="default")
        self.btn_select_run.setFixedHeight(34)
        self.btn_select_run.setIcon(self._icon("folder.svg"))
        self.btn_select_run.clicked.connect(self._select_run)
        scope_cards = QtWidgets.QHBoxLayout()
        scope_cards.setContentsMargins(0, 0, 0, 0)
        scope_cards.setSpacing(8)
        scope_cards.addWidget(self.btn_scope_all, 1)
        scope_cards.addWidget(self.btn_scope_sel, 1)
        scope_inner = QtWidgets.QVBoxLayout()
        scope_inner.setContentsMargins(0, 0, 0, 0)
        scope_inner.setSpacing(8)
        scope_inner.addLayout(scope_cards)
        scope_inner.addWidget(self.btn_select_run)
        self.field_scope = self._field("Runs to export", scope_inner)

        # --- Export format: CSV / ZIP / Folder -------------------------
        self.format_group = GlassOptionCardGroup(self)
        self.rb_csv = GlassOptionCard("CSV Report", "A single spreadsheet report")
        self.rb_zip = GlassOptionCard("ZIP Archive", "One compressed archive of runs")
        self.rb_folder_fmt = GlassOptionCard("Folder", "A plain folder of run files")
        self.format_group.addCard(self.rb_csv, 0)
        self.format_group.addCard(self.rb_zip, 1)
        self.format_group.addCard(self.rb_folder_fmt, 2)
        self.format_group.toggled.connect(self._on_format_changed)
        format_row = QtWidgets.QHBoxLayout()
        format_row.setContentsMargins(0, 0, 0, 0)
        format_row.setSpacing(8)
        for c in (self.rb_csv, self.rb_zip, self.rb_folder_fmt):
            format_row.addWidget(c, 1)
        self.field_format = self._field("Export as", format_row)

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

        # Responsive grid. Wide layout pairs settings two-up; narrow stacks them.
        self.scope_grid = QtWidgets.QGridLayout()
        self.scope_grid.setContentsMargins(0, 0, 0, 0)
        self.scope_grid.setHorizontalSpacing(16)
        self.scope_grid.setVerticalSpacing(12)
        self.scope_grid.setColumnStretch(0, 1)
        self.scope_grid.setColumnStretch(1, 1)
        self._scope_fields = [self.field_scope, self.field_format, self.field_date]
        lay.addLayout(self.scope_grid)
        self._relayout_grid(force=True)

        outer.addWidget(card)
        outer.addStretch(1)
        return host

    # ---- Step 2: Fields -------------------------------------------------
    def _build_fields_page(self):
        host, outer = self._step_host()

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
        outer.addWidget(self.csv_card)

        # --- Existing-files policy: Merge / Replace / Skip -------------
        policy_card = self._card("Existing Files", "What to do when a file already exists")
        self.policy_group = GlassOptionCardGroup(self)
        self.rb_merge = GlassOptionCard("Merge", "Keep newer versions")
        self.rb_replace = GlassOptionCard("Replace", "Overwrite existing")
        self.rb_skip = GlassOptionCard("Skip", "Leave existing untouched")
        self.policy_group.addCard(self.rb_merge, POLICY_MERGE)
        self.policy_group.addCard(self.rb_replace, POLICY_REPLACE)
        self.policy_group.addCard(self.rb_skip, POLICY_SKIP)
        self.rb_merge.setChecked(True)
        policy_row = QtWidgets.QHBoxLayout()
        policy_row.setContentsMargins(0, 0, 0, 0)
        policy_row.setSpacing(8)
        for c in (self.rb_merge, self.rb_replace, self.rb_skip):
            policy_row.addWidget(c, 1)
        policy_card.body.addLayout(policy_row)
        outer.addWidget(policy_card)

        outer.addStretch(1)
        return host

    # ---- Step 3: Review --------------------------------------------------
    def _build_review_page(self):
        host, outer = self._step_host()
        card = self._card("Review", "Confirm your selections, then export")
        self.review_rows_lay = QtWidgets.QVBoxLayout()
        self.review_rows_lay.setContentsMargins(0, 0, 0, 0)
        self.review_rows_lay.setSpacing(14)
        card.body.addLayout(self.review_rows_lay)
        outer.addWidget(card)
        outer.addStretch(1)
        return host

    def _refresh_review(self):
        while self.review_rows_lay.count():
            item = self.review_rows_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        dest_card = self.dest_group.checkedButton()
        scope_btn = self.scope_group.checkedButton()
        scope_detail = (
            f" ({self.btn_select_run.text().strip()})" if self.btn_scope_sel.isChecked() else ""
        )
        fmt_btn = self.format_group.checkedButton()
        policy_btn = self.policy_group.checkedButton()
        date_text = (
            f"{self.date_start.date().toString('yyyy-MM-dd')} → "
            f"{self.date_end.date().toString('yyyy-MM-dd')}"
        )

        sections = [
            (
                "Destination",
                [
                    ("Destination", dest_card.text() if dest_card else "Folder on this PC"),
                    ("Target", self.target_field.text()),
                    ("Dated subfolder", "Yes" if self.chk_dated_subfolder.isChecked() else "No"),
                ],
            ),
            (
                "Scope",
                [
                    ("Export name", self.name_field.text() or "(none — copied directly)"),
                    (
                        "Runs to export",
                        (scope_btn.text() if scope_btn else "All Runs") + scope_detail,
                    ),
                    ("Date range", date_text),
                ],
            ),
            (
                "Fields",
                [
                    ("Export as", fmt_btn.text() if fmt_btn else "CSV Report"),
                    ("Existing files", policy_btn.text() if policy_btn else "Merge"),
                ]
                + (
                    [("CSV columns", ", ".join(self._selected_csv_cols()))]
                    if self.rb_csv.isChecked()
                    else []
                ),
            ),
        ]

        for idx, (heading, rows) in enumerate(sections):
            if idx > 0:
                divider = QtWidgets.QFrame()
                divider.setFixedHeight(1)
                divider.setStyleSheet("background: rgba(210, 218, 228, 150); border: none;")
                self.review_rows_lay.addWidget(divider)
            self.review_rows_lay.addWidget(self._build_review_section(heading, rows))

    def _build_review_section(self, heading, rows):
        section = QtWidgets.QWidget()
        section.setStyleSheet("background: transparent;")
        slay = QtWidgets.QVBoxLayout(section)
        slay.setContentsMargins(0, 0, 0, 0)
        slay.setSpacing(8)

        head = QtWidgets.QLabel(heading.upper())
        head.setStyleSheet(self._caption_qss())
        slay.addWidget(head)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        for i, (label, value) in enumerate(rows):
            r, c = divmod(i, 2)
            grid.addWidget(self._review_field(label, value), r, c)
        slay.addLayout(grid)
        return section

    @staticmethod
    def _review_field(label, value):
        block = QtWidgets.QWidget()
        block.setStyleSheet("background: transparent;")
        blay = QtWidgets.QVBoxLayout(block)
        blay.setContentsMargins(0, 0, 0, 0)
        blay.setSpacing(2)
        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet(
            "QLabel { color: rgba(60, 72, 88, 160); font-size: 10px; font-weight: 600; "
            "text-transform: uppercase; letter-spacing: 0.5px; background: transparent; }"
        )
        val = QtWidgets.QLabel(str(value))
        val.setWordWrap(True)
        val.setStyleSheet(
            "QLabel { color: rgba(20, 30, 42, 235); font-size: 13px; font-weight: 600; "
            "background: transparent; }"
        )
        blay.addWidget(lbl)
        blay.addWidget(val)
        return block

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

        # Cancel sits on the left (always available — aborts a running task, or
        # resets the wizard to step 1 when idle); Back/Next-or-Export on the
        # right, matching the wireframe's footer.
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.btn_cancel = GlassPushButton(" Cancel", variant="default")
        self.btn_cancel.setFixedHeight(34)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        self.btn_back = GlassPushButton(" Back", variant="default")
        self.btn_back.setFixedHeight(34)
        self.btn_back.clicked.connect(self._go_back)
        self.btn_next = GlassPushButton(" Next →", variant="primary")
        self.btn_next.setFixedHeight(34)
        self.btn_next.clicked.connect(self._go_next_or_export)

        row.addWidget(self.btn_cancel, 0)
        row.addStretch(1)
        row.addWidget(self.btn_back, 0)
        row.addWidget(self.btn_next, 0)

        flay.addLayout(row)
        self.root.addWidget(footer, 0)

    # ------------------------------------------------------------------
    #  Wizard step navigation
    # ------------------------------------------------------------------
    def _go_to_step(self, index):
        if index == self._step:
            return
        going_forward = index > self._step
        old_index = self._step

        # Prep the destination page's content BEFORE sliding so its captured
        # pixmap (used by the slide) reflects the latest selections, not a
        # stale snapshot from whenever that page was last shown.
        if index == len(self._step_labels) - 1:
            self._refresh_review()

        self._step = index
        self.stepper.set_current(index)
        self.btn_back.setEnabled(index > 0)
        if index == 0:
            self._update_export_enabled()
        else:
            self.btn_next.setEnabled(True)
        self.btn_next.setText(" Export" if index == len(self._step_labels) - 1 else " Next →")

        self._slide_step(old_index, index, going_forward)

    def _go_back(self):
        if self._step > 0:
            self._go_to_step(self._step - 1)

    def _go_next_or_export(self):
        if self._step == len(self._step_labels) - 1:
            self._do_export()
            return
        if self._step == 0 and self._drive() is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Export Destination",
                "Choose a target folder or USB drive before continuing.",
            )
            return
        if self._step == 1:
            try:
                self._compute_filter_min()
                self._compute_filter_max()
            except ValueError as e:
                QtWidgets.QMessageBox.warning(self, "Export by date range", str(e))
                return
        self._go_to_step(self._step + 1)

    def _on_cancel_clicked(self):
        if self._task_running:
            self.services.request_abort()
        else:
            self._reset_state()

    def _reset_state(self):
        """Reset every wizard field back to its defaults and return to step
        0 — used after a completed export, and by Cancel when idle."""
        self._source_subfolder = ""
        self.btn_select_run.setText(" All")
        self.scope_group.setCheckedId(0)  # All Runs

        self.format_group.setCheckedId(0)  # CSV Report
        self._on_format_changed()

        today = QtCore.QDate.currentDate()
        self.date_start.setDate(today.addMonths(-1))
        self.date_end.setDate(today)

        self.policy_group.setCheckedId(POLICY_MERGE)

        for cb in self.csv_checks.values():
            cb.setChecked(True)

        self.chk_dated_subfolder.setChecked(True)
        self._generate_name()

        self.dest_group.setCheckedId(1)  # Folder on this PC
        self._refresh_target(no_ask=True)

        self._exported = False
        # Explicit, not just via _go_to_step(0): that call is a no-op when
        # already on step 0, which would otherwise leave old steps marked
        # "done" (still highlighted/clickable) after a reset.
        self.stepper.reset()
        self._go_to_step(0)

    # ------------------------------------------------------------------
    #  Step slide transition (pixmap-proxy cross-slide, same technique as
    #  DataManagementWidget._slide_to — adapted to horizontal/Next-Back).
    # ------------------------------------------------------------------
    def _teardown_step_slide(self):
        """Stop any running slide and destroy its proxy clip immediately, so
        a rapid double Next/Back click can't leave a stale page snapshot
        painting over the live stack."""
        group = getattr(self, "_step_slide_group", None)
        if group is not None:
            try:
                group.stop()
                group.deleteLater()
            except RuntimeError:
                pass
            self._step_slide_group = None
        clip = getattr(self, "_step_slide_clip", None)
        if clip is not None:
            try:
                clip.hide()
                clip.setParent(None)
                clip.deleteLater()
            except RuntimeError:
                pass
            self._step_slide_clip = None
        if hasattr(self, "step_stack"):
            self.step_stack.setGraphicsEffect(None)
            self.step_stack.show()

    def _slide_step(self, old_index, new_index, going_forward):
        """Cross-slide the outgoing/incoming step pages horizontally: Next
        slides the new page in from the right (old exits left); Back is the
        mirror image."""
        self._teardown_step_slide()

        stack = self.step_stack
        old_widget = stack.widget(old_index)
        new_widget = stack.widget(new_index)
        size = stack.size()
        if old_widget is None or new_widget is None or size.width() <= 0 or size.height() <= 0:
            stack.setCurrentIndex(new_index)  # layout not settled — fall back to instant
            return

        old_pix = old_widget.grab()
        stack.setCurrentIndex(new_index)
        new_widget.resize(size)
        new_pix = new_widget.grab()

        clip = QtWidgets.QFrame(self)
        clip.setObjectName("stepSlideClip")
        clip.setStyleSheet("QFrame#stepSlideClip { background: transparent; border: none; }")
        clip.setGeometry(stack.geometry())
        clip.show()
        clip.raise_()  # paint over the (still-visible, still-laid-out) live stack
        self._step_slide_clip = clip

        w = size.width()
        rest = QtCore.QPoint(0, 0)
        if going_forward:
            old_end = QtCore.QPoint(-w, 0)  # old exits left
            new_start = QtCore.QPoint(w, 0)  # new enters from the right
        else:
            old_end = QtCore.QPoint(w, 0)  # old exits right
            new_start = QtCore.QPoint(-w, 0)  # new enters from the left

        old_lbl = QtWidgets.QLabel(clip)
        old_lbl.setPixmap(old_pix)
        old_lbl.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        old_lbl.setGeometry(QtCore.QRect(rest, size))
        old_lbl.show()

        new_lbl = QtWidgets.QLabel(clip)
        new_lbl.setPixmap(new_pix)
        new_lbl.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        new_lbl.setGeometry(QtCore.QRect(new_start, size))
        new_lbl.show()
        new_lbl.raise_()

        # Make the live stack fully transparent — NOT hidden — for the
        # duration. setVisible(False) would collapse its reserved space in
        # the shared QVBoxLayout (stepper above, footer below), snapping
        # them around (the earlier "stepper sliding"/jumpy-layout bug).
        # Opacity 0 has no effect on layout — it just stops the live stack
        # (already showing the new page underneath the proxies) from
        # bleeding through at the proxies' edges.
        stack_effect = QtWidgets.QGraphicsOpacityEffect(stack)
        stack.setGraphicsEffect(stack_effect)
        stack_effect.setOpacity(0.0)

        # Both animations share the EXACT same duration/curve so old and new
        # move in lockstep — old's trailing edge and new's leading edge stay
        # joined at every frame (a "push"), with no momentary gap that would
        # let the live stack (already showing the new page) flash through.
        anim_old = QtCore.QPropertyAnimation(old_lbl, b"pos", self)
        anim_old.setDuration(220)
        anim_old.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim_old.setStartValue(rest)
        anim_old.setEndValue(old_end)

        anim_new = QtCore.QPropertyAnimation(new_lbl, b"pos", self)
        anim_new.setDuration(220)
        anim_new.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim_new.setStartValue(new_start)
        anim_new.setEndValue(rest)

        group = QtCore.QParallelAnimationGroup(self)
        group.addAnimation(anim_old)
        group.addAnimation(anim_new)

        def _finish():
            stack.setGraphicsEffect(None)
            clip.hide()
            clip.setParent(None)
            clip.deleteLater()
            self._step_slide_clip = None
            self._step_slide_group = None

        group.finished.connect(_finish)
        self._step_slide_group = group
        group.start()

    # ------------------------------------------------------------------
    #  Responsive grid relayout
    # ------------------------------------------------------------------
    def _relayout_grid(self, force=False):
        """Place the Scope step's fields in 1 or 2 columns based on width."""
        avail = self.scope_scroll.viewport().width() if hasattr(self, "scope_scroll") else self.width()
        two_col = avail >= RESPONSIVE_BREAKPOINT
        if not force and two_col == self._settings_two_col:
            return
        self._settings_two_col = two_col

        # Detach existing items without deleting the field widgets.
        while self.scope_grid.count():
            item = self.scope_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(self.scope_host)

        if two_col:
            self.scope_grid.setColumnStretch(0, 1)
            self.scope_grid.setColumnStretch(1, 1)
            for idx, field in enumerate(self._scope_fields):
                r, c = divmod(idx, 2)
                self.scope_grid.addWidget(field, r, c)
        else:
            self.scope_grid.setColumnStretch(0, 1)
            self.scope_grid.setColumnStretch(1, 0)
            for idx, field in enumerate(self._scope_fields):
                self.scope_grid.addWidget(field, idx, 0, 1, 2)
        for field in self._scope_fields:
            field.show()

    def _relayout_csv_checks(self, force=False):
        """Flow the column checkboxes into 1–3 columns based on width."""
        avail = (
            self.fields_scroll.viewport().width() if hasattr(self, "fields_scroll") else self.width()
        )
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "scope_grid"):
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
        # Destination cards aren't gated here: they live on their own wizard
        # step the user can't reach mid-export anyway, and disabling them
        # caused a visible "stuck disabled" flash on the Destination step
        # right as a completed export resets back to it (the success signal
        # can arrive on the GUI thread before the freeze-clearing one does).
        for w in (self.btn_next, self.btn_back):
            w.setDisabled(frozen)

    def on_progress(self, label, pct, color):
        # The slim bar conveys progress; text labels are logged, not shown.
        try:
            self.export_progress.setValue(max(0, min(100, int(pct))))
        except Exception:
            pass
        # pct == 100 with the "success" colour is unique to _export_task's
        # final "Exported to ..." message (cancelled/error finish at 100
        # with "b"/"r" instead) — reset the wizard back to a clean slate
        # once an export genuinely completes.
        if pct == 100 and color == "g":
            self._reset_state()

    # ------------------------------------------------------------------
    #  USB / folder destination state (preserves chk_usb/chk_folder/drive)
    # ------------------------------------------------------------------
    def _set_destination(self, kind):
        """Programmatically set the destination ('usb' or 'folder')."""
        self.dest_group.setCheckedId(0 if kind == "usb" else 1)

    def _on_dest_changed(self, card, checked):
        if not checked:
            return
        is_usb = card is self.card_usb
        self._chk_usb = is_usb
        self._chk_folder = not is_usb

        # Detect/Eject are USB-only extras; Choose… stays available for both
        # destinations so there's always a manual way to set/override the
        # target — auto-detection isn't guaranteed to find a drive. Slide
        # Detect/Eject open/closed rather than an abrupt show/hide; their
        # separators track the same state (no point separating from a
        # button that isn't there).
        self._slide_button(self.btn_detect, is_usb)
        self._slide_button(self.btn_eject, is_usb)
        self.sep_detect_eject.setVisible(is_usb)
        self.sep_eject_choose.setVisible(is_usb)
        self.btn_target.setVisible(True)

        if is_usb:
            # Switching to USB: adopt the currently detected USB drive if the
            # shared loop already found one; otherwise leave the field as-is
            # so a manually-chosen target (via Choose…) isn't clobbered.
            drive = self._drive()
            if drive:
                self.target_field.setText(drive)
        else:
            # Folder: target is whatever was chosen / defaulted.
            t = self.target_field.text()
            self._set_drive(t if t and t != "[NONE]" else None)
            self._refresh_target(no_ask=True)
        self._update_export_enabled()

    def _slide_button(self, widget, show):
        """Slide a USB-only action button open (show=True) or closed
        (show=False) by animating maximumWidth — a real QWidget property
        that QHBoxLayout respects every frame, so siblings reflow smoothly
        as it grows/shrinks (unlike animating geometry directly, which only
        the layout itself is allowed to set)."""
        natural_w = getattr(widget, "_natural_w", widget.sizeHint().width())
        anim = QtCore.QPropertyAnimation(widget, b"maximumWidth", self)
        anim.setDuration(200)
        if show:
            widget.setVisible(True)
            anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
            anim.setStartValue(widget.maximumWidth())
            anim.setEndValue(natural_w)
        else:
            anim.setEasingCurve(QtCore.QEasingCurve.InCubic)
            anim.setStartValue(widget.maximumWidth())
            anim.setEndValue(0)
            anim.finished.connect(lambda: widget.setVisible(False))

        self._dest_anims = [
            a for a in getattr(self, "_dest_anims", []) if a.state() == QtCore.QAbstractAnimation.Running
        ]
        self._dest_anims.append(anim)
        anim.start()

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
        # Live-gates Next while the user is on the Destination step; steps
        # 1-3 validate (and warn) at click-time instead in _go_next_or_export.
        ready = (self._chk_usb or self._chk_folder) and self._drive() is not None
        if getattr(self, "_step", 0) == 0 and hasattr(self, "btn_next"):
            self.btn_next.setEnabled(bool(ready))

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
        self.scope_group.setCheckedId(1)  # Selection — setChecked() alone wouldn't
        # enforce exclusivity or notify the group (only the click path / setCheckedId do).
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
        enabled = self.chk_dated_subfolder.isChecked()
        self.name_field.setEnabled(enabled)
        self.name_field.setText(default if enabled else "")

    def _on_dated_subfolder_changed(self, *_):
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
        # The dated-subfolder toggle stays freely editable regardless of
        # format: it lives on the Destination step, before the user has even
        # reached this Scope step's format choice, so locking/forcing it here
        # would make a control they already set appear to break for no
        # visible reason. Unchecking it for CSV/ZIP just changes the report's
        # default file name (falls back to the target folder's name).

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
        self._task_running = running  # plain bool; safe to set from the worker thread
        QtCore.QMetaObject.invokeMethod(
            self.btn_next,
            "setEnabled",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(bool, not running),
        )
        QtCore.QMetaObject.invokeMethod(
            self.btn_back,
            "setEnabled",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(bool, not running and self._step > 0),
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
    def _picker_box_qss():
        # No fill/border of its own — the thin separators between Detect,
        # Eject, and Choose… (shown only while Detect/Eject are expanded)
        # are the only visual division here now.
        return "QFrame#pickerBox { background: transparent; border: none; }"

    @staticmethod
    def _picker_separator():
        """A thin vertical divider between borderless pickerBox actions."""
        sep = QtWidgets.QFrame()
        sep.setFixedWidth(1)
        sep.setFixedHeight(18)
        sep.setStyleSheet("background: rgba(170, 182, 196, 150); border: none;")
        return sep

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

    def _date_qss(self):
        """Glass styling for QDateEdit that mirrors the line-edit/combo look."""
        icon_path = self._icon_file_path("date-range.svg")
        drop_image = f"image: url({icon_path});" if icon_path else ""
        return f"""
            QDateEdit {{
                background: rgba(255, 255, 255, 150);
                border: 1px solid rgba(120, 130, 145, 150);
                border-radius: 14px; padding-left: 12px; padding-right: 6px;
                color: rgb(40, 50, 62); font-weight: bold; min-height: 26px;
            }}
            QDateEdit:hover {{
                background: rgba(255, 255, 255, 200);
                border: 1px solid rgba(90, 100, 115, 190);
            }}
            QDateEdit:focus {{
                background: rgba(255, 255, 255, 225);
                border: 1px solid rgba(10, 163, 230, 200);
            }}
            QDateEdit::drop-down {{
                border: none; background: transparent; width: 24px;
                subcontrol-position: center right; margin-right: 4px;
            }}
            QDateEdit::down-arrow {{
                {drop_image}
                width: 14px; height: 14px;
            }}
            QCalendarWidget QWidget {{ alternate-background-color: rgba(235,240,246,255); }}
            QCalendarWidget QAbstractItemView:enabled {{
                color: rgb(40, 50, 62);
                selection-background-color: rgba(10, 163, 230, 60);
                selection-color: rgb(20, 30, 40);
            }}
        """

    def _icon(self, name):
        path = self._icon_file_path(name)
        return QtGui.QIcon(path) if path else QtGui.QIcon()

    @staticmethod
    def _icon_file_path(name):
        try:
            from QATCH.common.architecture import Architecture

            path = os.path.join(Architecture.get_path(), "QATCH", "icons", name)
            if os.path.exists(path):
                return path.replace("\\", "/")
        except Exception:
            pass
        return ""

    def _card(self, title, subtitle=""):
        """A frosted glass panel. Returns the QFrame with a `.body` QVBoxLayout
        for callers to populate (header + optional subtitle are pre-added)."""
        card = QtWidgets.QFrame()
        card.setObjectName("glassPanel")
        card.setStyleSheet("""
            QFrame#glassPanel {
                background: rgba(255, 255, 255, 110);
                border: 1px solid rgba(218, 224, 232, 170);
                border-radius: 10px;
            }
            QFrame#glassPanel[dimmed="true"] {
                background: rgba(255, 255, 255, 55);
                border: 1px solid rgba(218, 224, 232, 100);
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
