"""
QATCH.ui.data_mode_export.py

Export mode for selecting runs, destinations, and export settings.

Provides the Export mode page used by the data-management overlay. The module
ports the export functionality formerly implemented by
`export_widget.Ui_Export` into the shared :class:`DataModeWidget` /
:class:`DataServices` architecture.

The mode supports selecting logged runs and exporting them to a folder or
removable USB destination in CSV, ZIP, or folder format. It preserves the
original export semantics, including existing-file handling, date filtering,
CSV field selection and expansion, ZIP packaging, nested-folder cleanup, and
export-history logging.

The page uses a responsive layout designed to remain usable at the compact
width of the data-management overlay. Its content is contained within a
scrollable area, settings are arranged using compact field controls and a
responsive grid, and the status readout and primary actions remain pinned
below the scrolling content.

All cross-mode task execution, cancellation, USB state, progress reporting,
and GUI-freeze coordination are delegated to :class:`DataServices`; this
module is responsible for Export-specific UI state and processing.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-21
"""

import csv
import datetime
import os
import shutil
import zipfile
from datetime import timezone as tz

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants
from QATCH.ui.components import (
    QATCHLineEdit,
    QATCHOptionCard,
    QATCHOptionCardGroup,
    QATCHPushButton,
)
from QATCH.ui.components.stepper import Stepper as _Stepper
from QATCH.ui.styles.theme_manager import (
    ThemeManager,
    caption_label_qss,
    desc_label_qss,
    hairline_qss,
    tok_css,
)
from QATCH.ui.widgets.data_mode_base import DataModeWidget

try:
    from QATCH.VisQAI.src.io.parser import Parser
except Exception:
    Parser = None

TAG = "[DataExport]"

POLICY_REPLACE = 1
POLICY_MERGE = 2
POLICY_SKIP = 3
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

_FORMULATION_COMPONENTS = [
    "Protein",
    "Stabilizer",
    "Buffer",
    "Surfactant",
    "Salt",
    "Excipient",
]


class _FlowLayout(QtWidgets.QLayout):
    """Flow layout that arranges child items left-to-right with row wrapping.

    Lays out items sequentially across the available width and automatically
    starts a new row when the next item would exceed the available space.
    The layout reports a height based on the required number of wrapped rows,
    allowing it to participate correctly in Qt's height-for-width layout
    system.

    This layout is used for CSV column chips, where variable-width labels are
    better represented by a natural flowing arrangement than by a fixed-column
    grid.

    Attributes:
        _items (list): Layout items managed by the flow layout.
    """

    def __init__(self, parent=None, margin=0, spacing=8) -> None:
        """Initialize the flow layout.

        Args:
            parent (QtWidgets.QWidget, optional): Parent layout or widget.
                Defaults to None.
            margin (int, optional): Uniform margin applied around the layout.
                Defaults to 0.
            spacing (int, optional): Horizontal and vertical spacing between
                items. Defaults to 8.
        """
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def addItem(self, item) -> None:
        """Add a layout item to the flow layout.

        Args:
            item (QtWidgets.QLayoutItem): Layout item to append.
        """

        self._items.append(item)
        self.invalidate()

    def count(self) -> int:
        """Return the number of items managed by the layout.

        Returns:
            int: Number of layout items.
        """
        return len(self._items)

    def itemAt(self, index) -> QtWidgets.QLayoutItem:
        """Return the layout item at the specified index.

        Args:
            index (int): Zero-based item index.

        Returns:
            QtWidgets.QLayoutItem | None: The requested item, or `None` if
            the index is outside the layout's valid range.
        """
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index) -> QtWidgets.QLayoutItem:
        """Remove and return the layout item at the specified index.

        Args:
            index (int): Zero-based item index.

        Returns:
            QtWidgets.QLayoutItem | None: The removed item, or `None` if
            the index is outside the layout's valid range.
        """
        item = self._items.pop(index) if 0 <= index < len(self._items) else None
        if item is not None:
            self.invalidate()
        return item

    def expandingDirections(self) -> QtCore.Qt.Orientations:
        """Return the layout's expanding directions.

        Returns:
            QtCore.Qt.Orientations: Empty orientations because the layout does
            not request expansion in either direction.
        """
        return QtCore.Qt.Orientations(QtCore.Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:
        """Indicate that the layout's height depends on its available width.

        Returns:
            bool: Always `True` because items wrap into additional rows as
            the available width decreases.
        """
        return True

    def heightForWidth(self, width) -> int:
        """Calculate the height required to lay out items at a given width.

        Args:
            width (int): Available layout width.

        Returns:
            int: Required layout height including margins and wrapped rows.
        """
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect) -> None:
        """Position all managed items within the supplied geometry.

        Args:
            rect (QtCore.QRect): Geometry available to the layout.
        """
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QtCore.QSize:
        """Return the preferred size of the layout.

        Returns:
            QtCore.QSize: Minimum size required by the managed items.
        """
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:
        """Calculate the minimum size required by all layout items.

        Returns:
            QtCore.QSize: Bounding minimum size including layout margins.
        """
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        left, top, right, bottom = self.getContentsMargins()
        return size + QtCore.QSize(left + right, top + bottom)

    def _do_layout(self, rect: QtCore.QRect, test_only: bool) -> int:
        """Lay out items sequentially and wrap them onto additional rows.

        Args:
            rect (QtCore.QRect): Rectangle available for laying out items.
            test_only (bool): If `True`, calculate the required height
                without changing item geometries.

        Returns:
            int: Height required to lay out all items within `rect`.
        """
        left, top, right, bottom = self.getContentsMargins()
        effective = rect.adjusted(left, top, -right, -bottom)
        x, y = effective.x(), effective.y()
        line_height = 0
        spacing = self.spacing()

        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + spacing
            if next_x - spacing > effective.right() and line_height > 0:
                x = effective.x()
                y += line_height + spacing
                next_x = x + hint.width() + spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + bottom


class _ToggleChip(QtWidgets.QPushButton):
    """Checkable pill-shaped control for selecting CSV export columns.

    The chip presents a compact, wrapping-friendly representation of a CSV
    field. Selected fields display a check-mark prefix and accent styling,
    while unselected fields display a plus prefix with the standard flat
    surface and border styling.

    The widget manages its own visual state and automatically reapplies its
    styling when the checked state changes or when the application theme is
    changed.

    Attributes:
        _label (str): Display label identifying the CSV field represented by
            the chip.
    """

    def __init__(self, label, parent=None) -> None:
        """Initialize the CSV column toggle chip.

        The chip starts in the checked state and immediately applies the
        current theme styling. It also subscribes to theme changes so its
        appearance remains synchronized with the application theme.

        Args:
            label (str): Text displayed on the chip to identify the CSV
                column.
            parent (QtWidgets.QWidget, optional): Parent widget. Defaults to
                None.
        """
        super().__init__(parent)
        self._label = label
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setChecked(True)
        self.toggled.connect(self._restyle)
        self._restyle()
        ThemeManager.instance().themeChanged.connect(lambda _: self._restyle())

    def label(self) -> str:
        """Return the CSV column label represented by this chip.

        Returns:
            str: The field label assigned when the chip was created.
        """
        return self._label

    def _restyle(self, *_) -> None:
        """Update the chip text and appearance for its current state.

        Selected chips use the accent color and a highlighted surface, while
        unselected chips use the standard flat surface and text colors.
        Disabled-state styling is also applied for both states. All colors
        are obtained from the current theme tokens so the chip remains
        consistent across light and dark themes.

        Args:
            *_: Ignored signal arguments supplied when connected to the
                `toggled` signal or theme-change callback.
        """
        checked = self.isChecked()
        tok = ThemeManager.instance().tokens()
        self.setText(("✓ " if checked else "+ ") + self._label)
        if checked:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {tok_css(tok['flat_accent_weak'])};
                    border: 1.5px solid {tok_css(tok['flat_accent'])};
                    border-radius: 13px; padding: 5px 12px;
                    color: {tok_css(tok['flat_accent'])}; font-size: 12px; font-weight: 700;
                }}
                QPushButton:hover {{ background: {tok_css(tok['flat_accent_ring'])}; }}
                QPushButton:disabled {{
                    background: {tok_css(tok['flat_accent_weak'])};
                    border: 1.5px solid {tok_css(tok['flat_accent_ring'])};
                    color: {tok_css(tok['flat_text_muted'])};
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {tok_css(tok['flat_surface'])};
                    border: 1px solid {tok_css(tok['flat_border_strong'])};
                    border-radius: 13px; padding: 5px 12px;
                    color: {tok_css(tok['flat_text'])}; font-size: 12px; font-weight: 600;
                }}
                QPushButton:hover {{
                    background: {tok_css(tok['flat_surface2'])};
                    border: 1px solid {tok_css(tok['flat_border_strong'])};
                }}
                QPushButton:disabled {{
                    background: {tok_css(tok['flat_surface'])};
                    border: 1px solid {tok_css(tok['flat_border'])};
                    color: {tok_css(tok['flat_text_muted'])};
                }}
            """)


class ExportMode(DataModeWidget):
    """Data-management mode for configuring and executing data exports.

    Provides the complete export workflow for selecting an output destination,
    defining the run scope, choosing CSV fields, reviewing the configuration,
    and executing the export operation. The interface is organized as a
    four-step workflow:

    * Destination: Select USB or folder output and configure the export format.
    * Scope: Select all available runs or a specific run/device subfolder and
      configure date filtering.
    * Fields: Select the CSV columns to include when exporting CSV data.
    * Review: Verify the configured export options before starting the task.

    The mode delegates shared task execution, USB detection, cancellation,
    progress reporting, and GUI-freeze coordination to the inherited
    `DataModeWidget` / `DataServices` infrastructure. Export-specific state,
    validation, file processing, and presentation remain owned by this class.

    The widget maintains compatibility with the original export workflow while
    using a responsive step-based interface. A pinned footer provides progress
    feedback and navigation controls without requiring the user to scroll away
    from the primary actions.

    Class Attributes:
        MODE_KEY (str): Named service channel used for export progress routing.
        MODE_LABEL (str): Human-readable label displayed by the mode selector.

    Attributes:
        _chk_usb (bool): Whether USB export is currently selected.
        _chk_folder (bool): Whether folder export is currently selected.
        _source_subfolder (str): Selected source run or device subfolder.
            An empty string indicates that all available runs are selected.
        _filter_min (int): Computed lower bound for the active date filter.
        _filter_max (int | None): Computed upper bound for the active date
            filter, or None when the range is open-ended.
        _export_unnamed (bool): Whether unnamed runs are included in exports.
        _exported (bool): Whether the most recent export completed successfully.
        csv_report_path (str | None): Path to the CSV report currently being
            generated, when applicable.
        _task_running (bool): Whether an export task is currently active.
        _csv_card_opacity: Optional visual effect used to dim the CSV settings
            card when CSV-specific controls are unavailable.
        _cards (list): Collection of export settings cards that require
            coordinated theme updates.
        _settings_two_col (bool | None): Tracks the current responsive
            two-column breakpoint state for the settings layout.
        _step_labels (list[str]): Labels displayed by the export workflow
            stepper.
        stepper (_Stepper): Workflow stepper used to navigate between export
            stages.
        step_stack (QtWidgets.QStackedWidget): Container holding the four
            export workflow pages.
    """

    MODE_KEY = "export"
    MODE_LABEL = "Export"

    def build(self):
        """Construct the export workflow and initialize its default state.

        Creates the four-step export interface, including the destination,
        scope, CSV fields, and review pages. A pinned footer containing the
        progress indicator and navigation/action buttons is added below the
        step pages.

        The export format defaults to CSV and the destination defaults to a
        folder. The initial step is selected and the appropriate controls are
        enabled or disabled according to those defaults. Theme handling is
        also initialized after the interface has been constructed.
        """
        # Shared-state mirrors of the original flags.
        self._chk_usb = False  # exporting to USB
        self._chk_folder = False  # exporting to folder
        self._source_subfolder = ""  # selected run/device subpath ("" = all)
        self._filter_min = 0  # computed date floor at export time
        self._filter_max = None  # computed date ceiling (None = open-ended)
        self._export_unnamed = False
        self._exported = False  # set True after a successful export
        self.csv_report_path = None  # path of the CSV report being written
        self._task_running = False  # drives Cancel's dual abort/reset behavior
        self._csv_card_opacity = None  # lazily-created dim effect for the CSV card
        self._cards = []  # every GlassPanel from self._card(), restyled on theme change.
        self._settings_two_col = None

        # Stepper, step pages
        self._step_labels = ["Destination", "Scope", "Fields", "Review"]
        self.stepper = _Stepper(self._step_labels)
        self.stepper.stepClicked.connect(self._go_to_step)
        self.root.addWidget(self.stepper)

        self.step_stack = QtWidgets.QStackedWidget()
        stack_policy = self.step_stack.sizePolicy()
        stack_policy.setRetainSizeWhenHidden(True)
        self.step_stack.setSizePolicy(stack_policy)
        self.dest_scroll = self._make_step_scroll(self._build_destination_page())
        self.scope_scroll = self._make_step_scroll(self._build_scope_page())
        self.fields_scroll = self._make_step_scroll(self._build_fields_page())
        self.review_scroll = self._make_step_scroll(self._build_review_page())
        for scroll in (self.dest_scroll, self.scope_scroll, self.fields_scroll, self.review_scroll):
            self.step_stack.addWidget(scroll)
        self.root.addWidget(self.step_stack, 1)

        # Footer
        self._build_status_and_actions()

        # Apply initial enable-state
        self.rb_csv.setChecked(True)
        self._on_format_changed()
        self._set_destination("folder")
        self._step = 0
        self.stepper.set_current(0)
        self.btn_back.setEnabled(False)

        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        """Refresh the export interface when the application theme changes.

        Args:
            _mode (str): Identifier for the newly activated theme mode. The
                value is not used directly because the current theme is
                obtained from `ThemeManager` by `_apply_theme`.
        """
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Apply the current theme to all export-mode UI components.

        Refreshes theme-dependent colors, borders, typography, separators,
        date controls, review elements, progress-bar styling, and export
        setting cards using the active `ThemeManager` tokens.

        The method also refreshes the review section so dynamically generated
        review cards are rebuilt using the current theme rather than retaining
        styling from the previous theme.

        Returns:
            None
        """
        tok = ThemeManager.instance().tokens()

        for card in self._cards:
            self._restyle_card(card)
        if hasattr(self, "_fields_hairline"):
            self._fields_hairline.setStyleSheet(hairline_qss())

        self._review_heading.setStyleSheet(
            f"QLabel {{ color: {tok_css(tok['flat_text'])}; font-size: 14px; "
            "font-weight: bold; background: transparent; }"
        )
        self.csv_count_label.setStyleSheet(caption_label_qss())

        self._date_box.setStyleSheet(
            f"QFrame#dateRangeBox {{ background: {tok_css(tok['flat_surface2'])}; "
            f"border: 1px solid {tok_css(tok['flat_border'])}; border-radius: 8px; }}"
        )
        self.chk_dated_subfolder.setStyleSheet(self._radio_qss())
        self.chk_date_range.setStyleSheet(self._radio_qss())
        self.date_start.setStyleSheet(self._date_qss())
        self.date_end.setStyleSheet(self._date_qss())
        self.lbl_date_from.setStyleSheet(self._inline_lbl_qss())
        self.lbl_date_to.setStyleSheet(self._inline_lbl_qss())
        self.sep_detect_eject.setStyleSheet(
            f"background: {tok_css(tok['flat_border'])}; border: none;"
        )
        self.sep_eject_choose.setStyleSheet(
            f"background: {tok_css(tok['flat_border'])}; border: none;"
        )

        self.review_banner.setStyleSheet(
            f"QFrame#reviewBanner {{ background: {tok_css(tok['flat_accent_weak'])}; "
            f"border: 1px solid {tok_css(tok['flat_accent'])}; border-radius: 10px; }}"
        )
        self._banner_icon.setStyleSheet(
            f"QLabel {{ color: {tok_css(tok['flat_accent'])}; font-size: 14px; "
            "font-weight: 800; background: transparent; }"
        )
        self.review_banner_text.setStyleSheet(
            f"QLabel {{ color: {tok_css(tok['flat_accent'])}; font-size: 12px; "
            "font-weight: 600; background: transparent; }"
        )

        self.export_progress.setStyleSheet(f"""
            QProgressBar#exportProgress {{
                background: {tok_css(tok['flat_surface2'])};
                border: none;
                border-radius: 1px;
            }}
            QProgressBar#exportProgress::chunk {{
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 {tok_css(tok['ctrl_progress_chunk_start'])},
                    stop:1 {tok_css(tok['ctrl_progress_chunk_end'])}
                );
                border-radius: 1px;
            }}
        """)
        self._refresh_review()

    @staticmethod
    def _make_step_scroll(content_widget) -> QtWidgets.QScrollArea:
        """Create a scrollable container for an export workflow step.

        Args:
            content_widget (QtWidgets.QWidget): Widget containing the controls
                and layout for the step.

        Returns:
            QtWidgets.QScrollArea: Configured scroll area with transparent
            styling, a hidden horizontal scrollbar, and an automatically
            displayed vertical scrollbar when required.
        """
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
    def _step_host() -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
        """Create the transparent host used inside an export step scroll area.

        Returns:
            tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]: The transparent
            host widget and its outer vertical layout.
        """
        host = QtWidgets.QWidget()
        host.setObjectName("exportScrollHost")
        host.setStyleSheet("QWidget#exportScrollHost { background: transparent; }")
        outer = QtWidgets.QVBoxLayout(host)
        outer.setContentsMargins(2, 2, 6, 2)
        outer.setSpacing(12)
        return host, outer

    def _build_destination_page(self) -> QtWidgets.QWidget:
        """Build the destination-selection step of the export workflow.

        Creates controls for selecting USB or local-folder export, choosing
        or detecting the destination target, ejecting a selected USB drive,
        and optionally creating a dated export subfolder.

        Returns:
            QtWidgets.QWidget: Scrollable-step host containing the destination
            controls.
        """
        host, outer = self._step_host()
        card = self._card("Export Destination", "Where the exported data is written")
        lay = card.body

        # Destination is a pair of labelled cards
        lay.addWidget(self._caption("Export to"))
        self.dest_group = QATCHOptionCardGroup(self)
        dest_row = QtWidgets.QHBoxLayout()
        dest_row.setContentsMargins(0, 0, 0, 0)
        dest_row.setSpacing(8)
        self.card_usb = QATCHOptionCard("USB Drive", "Removable storage", show_radio=True)
        self.card_folder = QATCHOptionCard(
            "Folder on this PC", "Local or network path", show_radio=True
        )
        self.dest_group.addCard(self.card_usb, 0)
        self.dest_group.addCard(self.card_folder, 1)
        self.dest_group.toggled.connect(self._on_dest_changed)
        dest_row.addWidget(self.card_usb, 1)
        dest_row.addWidget(self.card_folder, 1)
        lay.addLayout(dest_row)

        # Target row
        lay.addWidget(self._caption("Target"))

        target_row = QtWidgets.QHBoxLayout()
        target_row.setContentsMargins(0, 0, 0, 0)
        target_row.setSpacing(8)
        self.target_field = QATCHLineEdit()
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
        self.btn_detect = QATCHPushButton(" Detect", variant="ghost")
        self.btn_detect.setFixedHeight(28)
        self.btn_detect.setIcon(self._icon("usb.svg"))
        self.btn_detect.clicked.connect(self._do_detect)
        self.btn_detect.set_border_visible(False)
        self.btn_eject = QATCHPushButton(" Eject", variant="ghost")
        self.btn_eject.setFixedHeight(28)
        self.btn_eject.clicked.connect(self._do_eject)
        self.btn_eject.set_border_visible(False)
        # Folder action
        self.btn_target = QATCHPushButton(" Choose…", variant="ghost")
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

        # Detect/Eject
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

    def _build_scope_page(self) -> QtWidgets.QWidget:
        """Builds the export scope configuration page.

        Creates the controls used to configure the export name, run selection,
        optional date filtering, and output format. The run-selection and export-
        format controls are arranged in a responsive two-column layout that
        collapses to a single column when space is limited.

        Returns:
            QtWidgets.QWidget: The host widget containing the completed export
                scope page.
        """
        host, outer = self._step_host()
        self.scope_host = host
        card = self._card("Export Scope", "Choose what gets exported and how")
        lay = card.body

        # Export name
        self.name_field = QATCHLineEdit()
        self.name_field.setMinimumHeight(34)
        lay.addWidget(self._field("Export name", self.name_field))
        lay.addSpacing(4)

        # Which runs
        self.scope_group = QATCHOptionCardGroup(self)
        self.btn_scope_all = QATCHOptionCard("All runs", "Export every run in the database")
        self.btn_scope_sel = QATCHOptionCard("Selected runs", "Pick specific devices or runs")
        self.scope_group.addCard(self.btn_scope_all, 0)
        self.scope_group.addCard(self.btn_scope_sel, 1)
        self.btn_scope_all.setChecked(True)
        self.scope_group.toggled.connect(self._on_selection_changed)
        self.btn_select_run = QATCHPushButton(" Choose…", variant="ghost")
        self.btn_select_run.setFixedHeight(28)
        self.btn_select_run.setIcon(self._icon("folder.svg"))
        self.btn_select_run.clicked.connect(self._select_run)
        self.btn_select_run.setVisible(False)  # only relevant once "Selected runs" is active
        scope_cards = QtWidgets.QVBoxLayout()
        scope_cards.setContentsMargins(0, 0, 0, 0)
        scope_cards.setSpacing(8)
        scope_cards.addWidget(self.btn_scope_all)
        scope_cards.addWidget(self.btn_scope_sel)

        # Date range
        date_box = QtWidgets.QFrame()
        date_box.setObjectName("dateRangeBox")
        self._date_box = date_box
        db = QtWidgets.QVBoxLayout(date_box)
        db.setContentsMargins(10, 8, 10, 8)
        db.setSpacing(6)
        self.chk_date_range = QtWidgets.QCheckBox("Limit to a date range")
        self.chk_date_range.setStyleSheet(self._radio_qss())
        self.chk_date_range.setChecked(True)
        self.chk_date_range.stateChanged.connect(self._on_date_range_toggled)
        db.addWidget(self.chk_date_range)

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
        self.date_start.dateChanged.connect(lambda d: self.date_end.setMinimumDate(d))
        self.date_end.dateChanged.connect(
            lambda d: self.date_start.setMaximumDate(min(d, QtCore.QDate.currentDate()))
        )
        self.date_end.setMinimumDate(self.date_start.date())

        date_inner = QtWidgets.QHBoxLayout()
        date_inner.setContentsMargins(20, 0, 0, 0)
        date_inner.setSpacing(8)
        self.lbl_date_from = QtWidgets.QLabel("From")
        self.lbl_date_from.setStyleSheet(self._inline_lbl_qss())
        self.lbl_date_to = QtWidgets.QLabel("to")
        self.lbl_date_to.setStyleSheet(self._inline_lbl_qss())
        date_inner.addWidget(self.lbl_date_from, 0)
        date_inner.addWidget(self.date_start, 1)
        date_inner.addWidget(self.lbl_date_to, 0)
        date_inner.addWidget(self.date_end, 1)
        db.addLayout(date_inner)

        which_inner = QtWidgets.QVBoxLayout()
        which_inner.setContentsMargins(0, 0, 0, 0)
        which_inner.setSpacing(8)
        which_inner.addLayout(scope_cards)
        which_inner.addWidget(self.btn_select_run)
        which_inner.addWidget(date_box)
        self.field_which = self._field("Which runs", which_inner)

        # Export as CSV / ZIP / Folder
        self.format_group = QATCHOptionCardGroup(self)
        self.rb_csv = QATCHOptionCard("CSV Report", "A single spreadsheet - choose columns next")
        self.rb_zip = QATCHOptionCard("ZIP Archive", "One compressed archive of raw run files")
        self.rb_folder_fmt = QATCHOptionCard("Folder", "A plain folder of run files")
        self.format_group.addCard(self.rb_csv, 0)
        self.format_group.addCard(self.rb_zip, 1)
        self.format_group.addCard(self.rb_folder_fmt, 2)
        self.format_group.toggled.connect(self._on_format_changed)
        format_col = QtWidgets.QVBoxLayout()
        format_col.setContentsMargins(0, 0, 0, 0)
        format_col.setSpacing(8)
        for c in (self.rb_csv, self.rb_zip, self.rb_folder_fmt):
            format_col.addWidget(c)
        self.field_export_as = self._field("Export as", format_col)

        self.scope_grid = QtWidgets.QGridLayout()
        self.scope_grid.setContentsMargins(0, 0, 0, 0)
        self.scope_grid.setHorizontalSpacing(20)
        self.scope_grid.setVerticalSpacing(12)
        self.scope_grid.setColumnStretch(0, 1)
        self.scope_grid.setColumnStretch(1, 1)
        self._scope_fields = [self.field_which, self.field_export_as]
        lay.addLayout(self.scope_grid)
        self._relayout_grid(force=True)

        outer.addWidget(card)
        outer.addStretch(1)
        return host

    def _build_fields_page(self) -> QtWidgets.QWidget:
        """Builds the export fields and existing-file policy page.

        Creates the CSV report field-selection controls, including bulk
        select/clear actions and a flow layout of toggle chips for individual
        columns. The required `Run Name` field is always included and cannot
        be disabled.

        Also creates the existing-file handling controls, allowing the user to
        choose whether matching files are merged, replaced, or skipped during
        export.

        Returns:
            QtWidgets.QWidget: The host widget containing the completed fields
                configuration page.
        """
        host, outer = self._step_host()

        header_actions = QtWidgets.QHBoxLayout()
        header_actions.setContentsMargins(0, 0, 0, 0)
        header_actions.setSpacing(4)
        self.btn_csv_select_all = QATCHPushButton(" Select all", variant="ghost")
        self.btn_csv_select_all.setFixedHeight(24)
        self.btn_csv_select_all.set_border_visible(False)
        self.btn_csv_select_all.clicked.connect(self._on_csv_select_all)
        self.btn_csv_clear = QATCHPushButton(" Clear", variant="ghost")
        self.btn_csv_clear.setFixedHeight(24)
        self.btn_csv_clear.set_border_visible(False)
        self.btn_csv_clear.clicked.connect(self._on_csv_clear)
        header_actions.addWidget(self.btn_csv_select_all)
        header_actions.addWidget(self.btn_csv_clear)

        self.csv_card = self._card(
            "CSV Report Fields",
            "Tap the columns to include in the report.",
            header_right=header_actions,
        )
        lay = self.csv_card.body

        cols_row = QtWidgets.QHBoxLayout()
        cols_row.setContentsMargins(0, 0, 0, 0)
        cols_row.addWidget(self._caption("Columns to include"))
        cols_row.addStretch(1)
        self.csv_count_label = QtWidgets.QLabel()
        self.csv_count_label.setStyleSheet(caption_label_qss())
        cols_row.addWidget(self.csv_count_label)
        lay.addLayout(cols_row)

        # Columns flow as toggle chips
        self.csv_chips = {}
        chip_host = QtWidgets.QWidget()
        chip_host.setStyleSheet("background: transparent;")
        self.csv_chip_flow = _FlowLayout(chip_host, margin=0, spacing=8)
        for field in CSV_FIELDS:
            chip = _ToggleChip(field)
            if field == "Run Name":
                chip.setEnabled(False)  # always present
                chip.setToolTip("Run Name is always included")
            else:
                chip.toggled.connect(self._update_csv_count)
            self.csv_chips[field] = chip
            self.csv_chip_flow.addWidget(chip)
        lay.addWidget(chip_host)
        self._update_csv_count()
        outer.addWidget(self.csv_card)

        self._fields_hairline = self._hairline()
        outer.addWidget(self._fields_hairline)

        # Existing-files policy
        policy_card = self._card("Existing Files")
        policy_card.body.addWidget(self._caption("When a file already exists"))
        policy_desc = QtWidgets.QLabel(
            "Applies if the export name matches a file already in the destination."
        )
        policy_desc.setStyleSheet(desc_label_qss())
        policy_desc.setWordWrap(True)
        policy_card.body.addWidget(policy_desc)

        self.policy_group = QATCHOptionCardGroup(self)
        self.rb_merge = QATCHOptionCard("Merge", "Keep newer versions")
        self.rb_replace = QATCHOptionCard("Replace", "Overwrite existing")
        self.rb_skip = QATCHOptionCard("Skip", "Leave existing untouched")
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

    def _on_csv_select_all(self) -> None:
        """Selects all optional CSV report fields.

        Iterates over the available CSV field chips and checks every selectable
        field. The required `Run Name` field is left unchanged because it is
        always included in the export.
        """
        for field, chip in self.csv_chips.items():
            if field != "Run Name":
                chip.setChecked(True)

    def _on_csv_clear(self) -> None:
        """Clears all optional CSV report field selections.

        Unchecks every selectable CSV field chip while leaving the required
        `Run Name` field checked and unchanged.
        """
        for field, chip in self.csv_chips.items():
            if field != "Run Name":
                chip.setChecked(False)

    def _update_csv_count(self, *_) -> None:
        """Updates the CSV field-selection count label.

        Recalculates the number of currently selected CSV columns and updates
        the associated label to display the selected count relative to the
        total number of available fields.

        Args:
            *_: Ignored signal arguments. Accepts arbitrary positional arguments
                so the method can be connected directly to Qt signals that emit
                values.
        """
        total = len(CSV_FIELDS)
        selected = len(self._selected_csv_cols())
        self.csv_count_label.setText(f"{selected} of {total} selected")

    def _build_review_page(self) -> QtWidgets.QWidget:
        """Builds the final review and export page.

        Creates the review page heading, explanatory subtitle, dynamically
        populated review-card layout, and a confirmation banner summarizing
        the export action. The page allows users to verify their selections
        before proceeding with the export and provides the visual structure
        used for editing previously configured options.

        Returns:
            QtWidgets.QWidget: The host widget containing the completed review
                and export page.
        """
        host, outer = self._step_host()

        heading = QtWidgets.QLabel("Review & export")
        self._review_heading = heading
        outer.addWidget(heading)
        subtitle = QtWidgets.QLabel(
            "Confirm your selections, then export. Tap Edit on any card to change it."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(desc_label_qss())
        outer.addWidget(subtitle)
        outer.addSpacing(2)

        self.review_cards_lay = QtWidgets.QVBoxLayout()
        self.review_cards_lay.setContentsMargins(0, 0, 0, 0)
        self.review_cards_lay.setSpacing(12)
        outer.addLayout(self.review_cards_lay)

        # "Ready to export" banner
        self.review_banner = QtWidgets.QFrame()
        self.review_banner.setObjectName("reviewBanner")
        banner_lay = QtWidgets.QHBoxLayout(self.review_banner)
        banner_lay.setContentsMargins(12, 10, 12, 10)
        banner_lay.setSpacing(8)
        self._banner_icon = QtWidgets.QLabel("✓")
        banner_icon = self._banner_icon
        self.review_banner_text = QtWidgets.QLabel()
        self.review_banner_text.setWordWrap(True)
        banner_lay.addWidget(banner_icon, 0)
        banner_lay.addWidget(self.review_banner_text, 1)
        outer.addWidget(self.review_banner)
        outer.addSpacing(12)

        outer.addStretch(1)
        return host

    def _refresh_review(self) -> None:
        """Rebuilds the review summary from the current export settings.

        Clears the existing review cards and reconstructs them using the
        currently selected destination, scope, date range, export format,
        CSV fields, and existing-file policy. The review banner is also
        updated to provide a concise summary of the pending export operation.

        The method is intended to be called whenever the underlying export
        configuration changes or the review page needs to be refreshed.
        """
        while self.review_cards_lay.count():
            item = self.review_cards_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        dest_card = self.dest_group.checkedButton()
        scope_btn = self.scope_group.checkedButton()
        scope_detail = (
            f" - {self.btn_select_run.text().strip()}" if self.btn_scope_sel.isChecked() else ""
        )
        fmt_btn = self.format_group.checkedButton()
        policy_btn = self.policy_group.checkedButton()
        run_count = self._count_scoped_runs()
        date_text = (
            f"{self.date_start.date().toString('yyyy-MM-dd')} → "
            f"{self.date_end.date().toString('yyyy-MM-dd')}"
            if self.chk_date_range.isChecked()
            else "All dates"
        )

        # Step indices line up with self._step_labels: Destination, Scope, Fields, Review.
        sections = [
            (
                "Destination",
                0,
                [
                    ("Target", dest_card.text() if dest_card else "Folder on this PC"),
                    ("Path", self.target_field.text()),
                    ("Dated subfolder", "Yes" if self.chk_dated_subfolder.isChecked() else "No"),
                ],
            ),
            (
                "Scope",
                1,
                [
                    ("Export name", self.name_field.text() or "(none - copied directly)"),
                    (
                        "Runs",
                        (scope_btn.text() if scope_btn else "All runs")
                        + scope_detail
                        + f" · {run_count} total",
                    ),
                    ("Date range", date_text),
                ],
            ),
            (
                "Format & Fields",
                2,
                [
                    ("Export as", fmt_btn.text() if fmt_btn else "CSV Report"),
                    ("If file exists", policy_btn.text() if policy_btn else "Merge"),
                ]
                + (
                    [("Columns", ", ".join(self._selected_csv_cols()))]
                    if self.rb_csv.isChecked()
                    else []
                ),
            ),
        ]

        for i, (heading, step_index, rows) in enumerate(sections):
            if i > 0:
                self.review_cards_lay.addWidget(self._hairline())
            self.review_cards_lay.addWidget(self._build_review_card(heading, step_index, rows))

        fmt_phrase = {0: "one CSV report", 1: "a ZIP archive", 2: "a folder of files"}.get(
            self.format_group.checkedId(), "a report"
        )
        dest_label = dest_card.text() if dest_card else "Folder on this PC"
        plural = "" if run_count == 1 else "s"
        self.review_banner_text.setText(
            f"Ready to export {run_count} run{plural} as {fmt_phrase} to {dest_label}."
        )

    def _build_review_card(self, title: str, step_index: int, rows: list) -> QtWidgets.QFrame:
        """Builds a grouped review section for the export summary.

        Creates a borderless review card containing a section heading, an
        `Edit` action that navigates back to the corresponding configuration
        step, and a two-column grid of label/value fields.

        Args:
            title: Section title displayed above the review fields.
            step_index: Index of the configuration step to open when the user
                activates the `Edit` button.
            rows: Sequence of `(label, value)` pairs to display in the review
                field grid.

        Returns:
            QtWidgets.QFrame: The completed review section widget.
        """
        card = QtWidgets.QFrame()
        card.setObjectName("reviewCard")
        card.setStyleSheet("QFrame#reviewCard { background: transparent; border: none; }")
        clay = QtWidgets.QVBoxLayout(card)
        clay.setContentsMargins(14, 12, 14, 12)
        clay.setSpacing(8)

        head_row = QtWidgets.QHBoxLayout()
        head_row.setContentsMargins(0, 0, 0, 0)
        head_row.setSpacing(8)
        cap = QtWidgets.QLabel(title.upper())
        cap.setStyleSheet(caption_label_qss())
        head_row.addWidget(cap)
        head_row.addStretch(1)
        edit_btn = QATCHPushButton(" Edit", variant="ghost")
        edit_btn.setFixedHeight(24)
        edit_btn.set_border_visible(False)
        edit_btn.clicked.connect(lambda _=False, idx=step_index: self._go_to_step(idx))
        head_row.addWidget(edit_btn)
        clay.addLayout(head_row)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        for i, (label, value) in enumerate(rows):
            r, c = divmod(i, 2)
            grid.addWidget(self._review_field(label, value), r, c)
        clay.addLayout(grid)
        return card

    def _count_scoped_runs(self) -> int:
        """Counts runs matching the current export scope.

        Counts run directories according to the currently selected device/run
        scope, unnamed-run export policy, and optional date range. The method
        only inspects directory structure and run metadata needed for date
        filtering; it does not parse the contents of individual run files.

        Returns:
            int: Number of runs matching the current export scope. Returns `0`
                if the configured data directory cannot be accessed or no runs
                satisfy the current filters.
        """
        data_path = Constants.log_prefer_path
        select_device, select_run = os.path.split(self._source_subfolder)
        if select_device == "":
            select_device = select_run
            select_run = ""
        try:
            date_filter = self._compute_filter_min()
            date_filter_max = self._compute_filter_max()
        except ValueError:
            date_filter, date_filter_max = 0, None

        count = 0
        try:
            devices = os.listdir(data_path)
        except OSError:
            return 0
        for device in devices:
            if select_device and select_device != device:
                continue
            device_path = os.path.join(data_path, device)
            if not os.path.isdir(device_path):
                continue
            try:
                runs = os.listdir(device_path)
            except OSError:
                continue
            for run in runs:
                if select_run and select_run != run:
                    continue
                run_path = os.path.join(device_path, run)
                if not os.path.isdir(run_path):
                    continue
                is_unnamed = run == "_unnamed" or device == "_unnamed"
                if is_unnamed and not self._export_unnamed:
                    continue
                if date_filter != 0 or date_filter_max is not None:
                    if not self._run_in_date_range(run_path, date_filter, date_filter_max):
                        continue
                count += 1
        return count

    @staticmethod
    def _run_in_date_range(run_path, date_filter, date_filter_max):
        """True if the newest file mtime in `run_path` falls within the
        given [date_filter, date_filter_max) window."""
        try:
            files = os.listdir(run_path)
        except OSError:
            return False
        last_modified = None
        for f in files:
            try:
                mtime = os.stat(os.path.join(run_path, f)).st_mtime
            except OSError:
                continue
            ts = datetime.datetime.fromtimestamp(mtime, tz=tz.utc)
            if last_modified is None or ts > last_modified:
                last_modified = ts
        if last_modified is None:
            return False
        if date_filter != 0 and last_modified < date_filter:
            return False
        if date_filter_max is not None and last_modified >= date_filter_max:
            return False
        return True

    @staticmethod
    def _review_field(label: str, value: float) -> QtWidgets.QWidget:
        """Builds a label/value field for the export review summary.

        Creates a compact vertical field containing a muted, uppercase label
        and a prominently styled value. The value supports word wrapping so
        longer paths, column lists, or other review details remain readable.

        Args:
            label: Descriptive label displayed above the value.
            value: Value to display in the review field. It is converted to a
                string before being assigned to the label.

        Returns:
            QtWidgets.QWidget: A widget containing the styled label and value.
        """
        tok = ThemeManager.instance().tokens()
        block = QtWidgets.QWidget()
        block.setStyleSheet("background: transparent;")
        blay = QtWidgets.QVBoxLayout(block)
        blay.setContentsMargins(0, 0, 0, 0)
        blay.setSpacing(2)
        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet(
            f"QLabel {{ color: {tok_css(tok['flat_text_muted'])}; font-size: 10px; "
            "font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; "
            "background: transparent; }"
        )
        val = QtWidgets.QLabel(str(value))
        val.setWordWrap(True)
        val.setStyleSheet(
            f"QLabel {{ color: {tok_css(tok['flat_text'])}; font-size: 13px; "
            "font-weight: 600; background: transparent; }"
        )
        blay.addWidget(lbl)
        blay.addWidget(val)
        return block

    def _selected_csv_cols(self) -> list[str]:
        """Returns the currently selected CSV report columns in defined order.

        The required `Run Name` column is always included and appears first,
        regardless of the state of its associated chip. Optional columns are
        included when their corresponding toggle chip is checked.

        Returns:
            list[str]: Ordered list of CSV column names selected for export.
        """
        return [f for f in CSV_FIELDS if f == "Run Name" or self.csv_chips[f].isChecked()]

    def _build_status_and_actions(self):
        """Builds the export progress indicator and wizard action controls.

        Creates the footer containing a slim progress bar and the navigation
        buttons used to cancel, move backward, advance through the wizard, or
        initiate the export. The progress bar is hidden while no export is
        running and is updated through the export task's progress reporting.

        The cancel action is available in both idle and active states, where it
        either resets the wizard or aborts the current export task.
        """
        footer = QtWidgets.QFrame()
        footer.setObjectName("exportFooter")
        footer.setStyleSheet("QFrame#exportFooter { background: transparent; border: none; }")
        flay = QtWidgets.QVBoxLayout(footer)
        flay.setContentsMargins(0, 4, 0, 0)
        flay.setSpacing(8)

        # Progress bar
        self.export_progress = QtWidgets.QProgressBar()
        self.export_progress.setObjectName("exportProgress")
        self.export_progress.setRange(0, 100)
        self.export_progress.setValue(0)
        self.export_progress.setTextVisible(False)
        self.export_progress.setFixedHeight(3)
        self.export_progress.setVisible(False)
        flay.addWidget(self.export_progress)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.btn_cancel = QATCHPushButton(" Cancel", variant="default")
        self.btn_cancel.setFixedHeight(34)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        self.btn_back = QATCHPushButton(" Back", variant="default")
        self.btn_back.setFixedHeight(34)
        self.btn_back.clicked.connect(self._go_back)
        self.btn_next = QATCHPushButton(" Next →", variant="primary")
        self.btn_next.setFixedHeight(34)
        self.btn_next.clicked.connect(self._go_next_or_export)

        row.addWidget(self.btn_cancel, 0)
        row.addStretch(1)
        row.addWidget(self.btn_back, 0)
        row.addWidget(self.btn_next, 0)

        flay.addLayout(row)
        self.root.addWidget(footer, 0)

    def _go_to_step(self, index: int) -> None:
        """Navigates to a specified export wizard step.

        Updates the current step, navigation controls, and stepper state before
        animating the transition from the current page to the requested page.
        When navigating to the review step, refreshes its contents first so the
        transition reflects the latest export selections.

        Args:
            index: Zero-based index of the destination wizard step.
        """
        if index == self._step:
            return
        going_forward = index > self._step
        old_index = self._step

        if index == len(self._step_labels) - 1:
            self._refresh_review()

        self._step = index
        self.stepper.set_current(index)
        self.btn_back.setEnabled(index > 0)
        if index == 0:
            self._update_export_enabled()
        else:
            self.btn_next.setEnabled(True)
        self.btn_next.setText(" Export" if index == len(self._step_labels) - 1 else " Next")

        self._slide_step(old_index, index, going_forward)

    def _go_back(self) -> None:
        """Navigates to the previous export wizard step.

        Does nothing when the wizard is already on its first step.
        """
        if self._step > 0:
            self._go_to_step(self._step - 1)

    def _go_next_or_export(self) -> None:
        """Advances the wizard or starts the export operation.

        On the final review step, initiates the export. On earlier steps,
        validates the required destination or date-range configuration before
        advancing to the next step. Displays a warning dialog when validation
        fails.
        """
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

    def _on_cancel_clicked(self) -> None:
        """Handles the export wizard's Cancel action.

        Requests an abort from the export service when an export task is
        running. When idle, resets the wizard to its default configuration.
        """
        if self._task_running:
            self.services.request_abort()
        else:
            self._reset_state()

    def _reset_state(self) -> None:
        """Resets the export wizard to its default configuration.

        Restores the source selection, export format, date range, existing-file
        policy, CSV field selections, dated-subfolder option, generated export
        name, destination, and internal export state. The wizard is returned to
        the first step and its stepper is explicitly reset so previously
        completed steps are no longer marked as complete.

        This method is used after a completed export and when Cancel is pressed
        while the wizard is idle.
        """
        self._source_subfolder = ""
        self.btn_select_run.setText(" Choose…")
        self.btn_select_run.setVisible(False)
        self.scope_group.setCheckedId(0)  # All Runs

        self.format_group.setCheckedId(0)  # CSV Report
        self._on_format_changed()

        today = QtCore.QDate.currentDate()
        self.date_start.setDate(today.addMonths(-1))
        self.date_end.setDate(today)
        self.chk_date_range.setChecked(True)

        self.policy_group.setCheckedId(POLICY_MERGE)

        for chip in self.csv_chips.values():
            chip.setChecked(True)

        self.chk_dated_subfolder.setChecked(True)
        self._generate_name()

        self.dest_group.setCheckedId(1)  # Folder on this PC
        self._refresh_target(no_ask=True)

        self._exported = False
        self.stepper.reset()
        self._go_to_step(0)

    def _teardown_step_slide(self) -> None:
        """Stops and cleans up any active step-slide animation.

        Immediately stops the current slide animation and removes its temporary
        proxy clip. This prevents stale page snapshots from remaining visible
        when navigation occurs again before a previous slide has finished.

        The live step stack is also restored to its normal visible state and any
        graphics effect applied during the animation is removed.
        """
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

    def _slide_step(self, old_index: int, new_index: int, going_forward: bool) -> None:
        """Animates a horizontal transition between export wizard steps.

        Captures the outgoing and incoming pages as pixmaps and animates them
        together inside a temporary clipping frame. When moving forward, the
        current page exits to the left while the new page enters from the right;
        navigating backward reverses the direction.

        Any existing slide animation is torn down before starting a new one.
        The live step stack is hidden for the duration of the animation to
        prevent the newly selected page from bleeding through the animated
        snapshots.

        Args:
            old_index: Zero-based index of the currently displayed step.
            new_index: Zero-based index of the destination step.
            going_forward: Whether navigation is moving toward a later step.
                If `False`, the transition is animated in the reverse
                direction.

        Notes:
            If either page is unavailable or the step stack has not yet been
            laid out with a valid size, the method falls back to an immediate
            step change without animation.
        """
        self._teardown_step_slide()

        stack = self.step_stack
        old_widget = stack.widget(old_index)
        new_widget = stack.widget(new_index)
        size = stack.size()
        if old_widget is None or new_widget is None or size.width() <= 0 or size.height() <= 0:
            stack.setCurrentIndex(new_index)  # layout not settled
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
        clip.raise_()  # paint over the live stack
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
        old_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        old_lbl.setGeometry(QtCore.QRect(rest, size))
        old_lbl.show()

        new_lbl = QtWidgets.QLabel(clip)
        new_lbl.setPixmap(new_pix)
        new_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        new_lbl.setGeometry(QtCore.QRect(new_start, size))
        new_lbl.show()
        new_lbl.raise_()

        # Hide the live stack for the duration
        stack.hide()
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
            stack.show()
            clip.hide()
            clip.setParent(None)
            clip.deleteLater()
            self._step_slide_clip = None
            self._step_slide_group = None

        group.finished.connect(_finish)
        self._step_slide_group = group
        group.start()

    def _relayout_grid(self, force: bool = False) -> None:
        """Reflows the Scope step fields between one and two columns.

        Determines the available content width and switches the Scope step
        layout between a two-column arrangement for wider views and a
        single-column arrangement for narrower views. Existing field widgets
        are detached and reinserted into the grid without being destroyed.

        Args:
            force: Whether to rebuild the grid even when the current column
                configuration already matches the available width.
        """
        avail = (
            self.scope_scroll.viewport().width() if hasattr(self, "scope_scroll") else self.width()
        )
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

    def resizeEvent(self, event) -> None:
        """Handles resizing of the export wizard.

        Performs the standard Qt resize handling and updates the Scope step's
        responsive field grid when the widget dimensions change.

        Args:
            event: Qt resize event containing the widget's new dimensions.
        """
        super().resizeEvent(event)
        if hasattr(self, "scope_grid"):
            self._relayout_grid()

    def on_enter(self) -> None:
        """Initializes the export UI when the wizard becomes active.

        Generates the default export name, refreshes the current destination,
        connects USB device add/remove signals, updates export availability,
        and forces an initial responsive layout of the Scope step.
        """
        self._generate_name()
        self._refresh_target(no_ask=True)
        self.services.usb_add.connect(self._on_usb_add)
        self.services.usb_remove.connect(self._on_usb_remove)
        self._update_export_enabled()
        self._relayout_grid(force=True)

    def on_freeze(self, frozen: bool) -> None:
        """Updates navigation controls when the export UI is frozen.

        Disables or re-enables the Back and Next/Export buttons to mirror the
        application's frozen GUI state while leaving destination controls
        unchanged.

        Args:
            frozen: Whether the export UI should be disabled while an operation
                is in progress.
        """
        for w in (self.btn_next, self.btn_back):
            w.setDisabled(frozen)

    def on_progress(self, label: str, pct: float, color: str) -> None:
        """Updates export progress and handles successful completion.

        Updates the export progress bar from the reported percentage. When the
        progress reaches 100 percent with the success color, treats the export
        as successfully completed and resets the wizard to its default state.

        Args:
            label: Status message associated with the progress update.
            pct: Export completion percentage.
            color: Status color code identifying the result state. A value of
                `"g"` at 100 percent indicates successful completion.
        """
        try:
            self.export_progress.setValue(max(0, min(100, int(pct))))
        except Exception:
            pass
        if pct == 100 and color == "g":
            self._reset_state()

    def _set_destination(self, kind):
        """Selects the export destination type programmatically.

        Args:
            kind: Destination type to select. `"usb"` selects the USB
                destination; any other value selects the local folder
                destination.
        """
        self.dest_group.setCheckedId(0 if kind == "usb" else 1)

    def _on_dest_changed(self, card, checked: bool) -> None:
        """Handles changes to the selected export destination.

        Updates the internal USB/folder destination state, adjusts the
        visibility of USB-specific controls, and synchronizes the target path
        with the selected destination. When USB is selected, an already
        detected drive is adopted when available; otherwise the existing target
        is preserved. When a folder is selected, the current target is restored
        or refreshed as appropriate.

        Args:
            card: Destination option card whose checked state changed.
            checked: Whether the destination card is now selected. Unchecked
                cards are ignored.
        """
        if not checked:
            return
        is_usb = card is self.card_usb
        self._chk_usb = is_usb
        self._chk_folder = not is_usb
        self._slide_button(self.btn_detect, is_usb)
        self._slide_button(self.btn_eject, is_usb)
        self.sep_detect_eject.setVisible(is_usb)
        self.sep_eject_choose.setVisible(is_usb)
        self.btn_target.setVisible(True)

        if is_usb:
            drive = getattr(self.services, "usb_drive", None)
            if drive:
                self.target_field.setText(drive)
                self._set_drive(drive)
        else:
            t = self.target_field.text()
            self._set_drive(t if t and t != "[NONE]" else None)
            self._refresh_target(no_ask=True)
        self._update_export_enabled()

    def _slide_button(self, widget: QtWidgets.QWidget, show: bool) -> None:
        """Animates the visibility of a destination action button.

        Expands or collapses the button by animating its `maximumWidth` so the
        surrounding layout can reflow smoothly. When hiding the button, it is
        made invisible after the collapse animation completes.

        Args:
            widget: Button widget whose width should be animated.
            show: Whether the button should be expanded and shown. If `False`,
                the button is collapsed and hidden when the animation finishes.
        """
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
            a
            for a in getattr(self, "_dest_anims", [])
            if a.state() == QtCore.QAbstractAnimation.Running
        ]
        self._dest_anims.append(anim)
        anim.start()

    def _on_usb_add(self) -> None:
        """Handles detection of a newly available USB drive.

        Updates the export target to the detected USB drive when USB is the
        currently selected destination. A folder destination is left unchanged
        so that automatic USB detection cannot overwrite a target selected by
        the user.

        Also logs the detection event and refreshes the export button's enabled
        state.
        """
        drive = getattr(self.services, "usb_drive", None)
        if self._chk_usb:
            self.target_field.setText(drive if drive else "[NONE]")
            self._set_drive(drive)
        Log.i(TAG, f"[{drive}] USB drive found! Ready to export.")
        self._update_export_enabled()

    def _on_usb_remove(self) -> None:
        """Handles removal of the currently detected USB drive.

        Logs the removal event and clears the active export target when USB is
        the selected destination. Folder destinations are left unchanged.
        The export button state is refreshed after the target is updated.
        """
        Log.w(TAG, "USB drive removed. Please eject first next time.")
        if self._chk_usb:
            self._set_drive(None)
            self.target_field.setText("[NONE]")
        self._update_export_enabled()

    def _set_drive(self, value):
        """Sets the shared export destination path.

        Updates the canonical `drive` value maintained by the shared service.
        Errors from the service assignment are ignored so a failure to mirror
        the value does not interrupt the export UI.

        Args:
            value: Destination path to assign, or `None` when no destination
                is currently available.
        """
        try:
            self.services.drive = value
        except Exception:
            pass

    def _drive(self) -> str | None:
        """Returns the currently configured export destination.

        Returns:
            str | None: The destination path maintained by the shared service,
                or `None` when no destination is configured.
        """
        return getattr(self.services, "drive", None)

    def _update_export_enabled(self) -> None:
        """Updates whether export navigation is available.

        Enables the Next button while on the Destination step only when a valid
        destination type is selected and a destination path is available.
        Later wizard steps perform their validation when the user attempts to
        advance.
        """
        ready = (self._chk_usb or self._chk_folder) and self._drive() is not None
        if getattr(self, "_step", 0) == 0 and hasattr(self, "btn_next"):
            self.btn_next.setEnabled(bool(ready))

    def _select_target(self) -> None:
        """Prompts the user to select a local export destination folder.

        Opens a directory-selection dialog using the existing target as the
        initial location when available, otherwise falling back to the default
        export directory. When a folder is selected, updates both the shared
        destination and target field, then refreshes the export button state.

        If the dialog is cancelled, the current destination remains unchanged.
        """
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

    def _refresh_target(self, no_ask: bool = True) -> None:
        """Refreshes and validates the default export folder target.

        Uses the existing target when available, otherwise falls back to the
        application's default export directory. When `no_ask` is enabled,
        ensures the target directory exists and updates the shared destination
        when the Folder destination is active.

        If the target cannot be created or accessed, clears the shared
        destination and marks the target field as unavailable.

        Args:
            no_ask: Whether to refresh the target without prompting the user.
                Defaults to `True`.
        """
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

    def _select_run(self) -> None:
        """Prompts the user to select a device or run directory for export.

        Opens a directory-selection dialog rooted at the logged-data directory.
        Valid selections are converted to a relative source path, switch the
        scope to `Selected runs`, update the selection button label, and
        regenerate the export name.

        Selections outside the configured logged-data directory are rejected and
        leave the current run selection unchanged.

        """
        data_root = Constants.log_prefer_path
        start = QtCore.QUrl.fromLocalFile(data_root)
        folder = QtWidgets.QFileDialog.getExistingDirectoryUrl(self, "Select Folder", start)
        if not folder.isValid():
            return
        selected = folder.toLocalFile()
        if data_root not in selected:
            self.btn_select_run.setText(" Choose…")
            self._source_subfolder = ""
            Log.w(TAG, "Selected folder not in logged data path.")
            return
        self.scope_group.setCheckedId(1)
        sub = selected.replace(data_root, "").replace("/", Constants.slash)
        sub = sub.strip(Constants.slash)
        self._source_subfolder = sub
        leaf = os.path.split(selected)[1]
        kind = "run:" if sub.count(Constants.slash) == 1 else "dev:"
        self.btn_select_run.setText(f" {kind}{leaf}")
        self._generate_name()

    def _on_selection_changed(self, *_) -> None:
        """Updates controls when the run-selection mode changes.

        Shows the run-selection button when `Selected runs` is active. When
        `All runs` is selected, clears the current source subfolder, restores
        the default selection label, and regenerates the export name.

        Args:
            *_: Ignored signal arguments emitted by the option-card group.
        """
        self.btn_select_run.setVisible(self.btn_scope_sel.isChecked())
        if self.btn_scope_all.isChecked():
            self.btn_select_run.setText(" Choose…")
            self._source_subfolder = ""
            self._generate_name()

    def _on_date_range_toggled(self, *_) -> None:
        """Enables or disables the date-range controls.

        Synchronizes the enabled state of the start/end date editors and their
        associated labels with the `Limit to a date range` checkbox.

        Args:
            *_: Ignored signal arguments emitted by the date-range checkbox.
        """
        enabled = self.chk_date_range.isChecked()
        for w in (self.date_start, self.date_end, self.lbl_date_from, self.lbl_date_to):
            w.setEnabled(enabled)

    def _generate_name(self) -> None:
        """Generates and applies the default export name.

        Uses the selected run's leaf directory name when a specific source
        subfolder is selected. Otherwise, generates a date-based name using the
        current date followed by the `_QATCH_EXPORT` suffix.

        The export name field is enabled only when dated subfolders are enabled;
        otherwise the field is cleared because no explicit subfolder name is
        required.
        """
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

    def _on_dated_subfolder_changed(self, *_) -> None:
        """Regenerates the export name after the dated-subfolder setting changes.

        Args:
            *_: Ignored signal arguments emitted by the dated-subfolder control.
        """
        self._generate_name()

    def _on_format_changed(self, *_) -> None:
        """Updates the UI for the selected export format.

        Enables or disables the CSV field-selection card based on whether CSV
        export is selected, applies visual dimming when the card is inactive,
        and updates the existing-file policy labels to reflect the semantics
        of the selected format.

        For CSV exports, the `Merge` and `Skip` policies are relabeled as
        `Append` and `Cancel` respectively. Non-CSV formats restore the
        original `Merge` and `Skip` labels and descriptions.

        Args:
            *_: Ignored signal arguments emitted by the export-format option
                group.
        """
        is_csv = self.rb_csv.isChecked()
        self.csv_card.setEnabled(is_csv)
        # Dim the disabled CSV card so the active format reads clearly.
        if not is_csv:
            if self._csv_card_opacity is None:
                self._csv_card_opacity = QtWidgets.QGraphicsOpacityEffect(self.csv_card)
            self._csv_card_opacity.setOpacity(0.5)
            self.csv_card.setGraphicsEffect(self._csv_card_opacity)
        else:
            self.csv_card.setGraphicsEffect(None)
            self._csv_card_opacity = None
        # CSV relabels the merge/skip policy to Append/Cancel (semantics differ).
        if is_csv:
            self.rb_merge.setText("Append")
            self.rb_merge.setDescription("Add rows to the file")
            self.rb_skip.setText("Cancel")
            self.rb_skip.setDescription("Leave existing untouched")
        else:
            self.rb_merge.setText("Merge")
            self.rb_merge.setDescription("Keep newer versions")
            self.rb_skip.setText("Skip")
            self.rb_skip.setDescription("Leave existing untouched")

    def _do_detect(self) -> None:
        """Requests USB device detection from the shared service.

        Invokes the shared USB detection trigger when available. If the service
        does not expose a callable detection handler, logs the request instead,
        leaving USB enumeration to the shared service loop.
        """
        trigger = getattr(self.services, "request_detect", None)
        if callable(trigger):
            trigger()
        else:
            Log.d(f"{TAG} detect requested (shared loop handles enumeration)")

    def _do_eject(self) -> None:
        """Requests ejection of the currently selected USB device.

        Invokes the shared service ejector when available. If no ejector is
        exposed by the service, logs the request without performing an eject
        operation.
        """
        ejector = getattr(self.services, "eject", None)
        if callable(ejector):
            ejector()
        else:
            Log.d(f"{TAG} eject requested (no shared ejector wired yet)")

    @staticmethod
    def _qdate_to_utc_floor(qdate) -> datetime.datetime:
        """Converts a Qt date to the UTC start of its local calendar day.

        Interprets the supplied date at local midnight and converts that instant
        to an aware UTC datetime.

        Args:
            qdate: Qt date to convert.

        Returns:
            datetime.datetime: Timezone-aware UTC datetime representing local
                midnight at the beginning of `qdate`.
        """
        local = datetime.datetime(qdate.year(), qdate.month(), qdate.day(), 0, 0, 0).astimezone()
        return local.astimezone(tz.utc)

    @staticmethod
    def _qdate_to_utc_ceiling(qdate) -> datetime.datetime:
        """Converts a Qt date to the UTC start of the following local day.

        Interprets the supplied date as ending at local midnight immediately
        after the date and converts that instant to an aware UTC datetime. This
        provides an exclusive upper bound while keeping the selected end date
        inclusive.

        Args:
            qdate: Qt date whose following local midnight should be converted.

        Returns:
            datetime.datetime: Timezone-aware UTC datetime representing local
                midnight at the beginning of the day after `qdate`.
        """
        nxt = qdate.addDays(1)
        local = datetime.datetime(nxt.year(), nxt.month(), nxt.day(), 0, 0, 0).astimezone()
        return local.astimezone(tz.utc)

    def _compute_filter_min(self) -> datetime.datetime:
        """Computes the inclusive lower bound for the selected date range.

        Converts the selected start date's local midnight to an aware UTC
        datetime. When date filtering is disabled, returns `0` to indicate
        that no lower date bound should be applied.

        Raises:
            ValueError: If the selected start date occurs after the selected
                end date.

        Returns:
            datetime.datetime | int: UTC datetime representing the beginning of
                the selected start date, or `0` when date filtering is disabled.
        """
        if not self.chk_date_range.isChecked():
            return 0
        start, end = self.date_start.date(), self.date_end.date()
        if start > end:
            raise ValueError(
                '"Limit to a date range" start date must be on or before the end date.'
            )
        return self._qdate_to_utc_floor(start)

    def _compute_filter_max(self) -> datetime.datetime:
        """Computes the exclusive upper bound for the selected date range.

        Converts midnight at the start of the day following the selected end
        date to an aware UTC datetime, making the selected end date fully
        inclusive. When date filtering is disabled, returns `None` to indicate
        that no upper date bound should be applied.

        Returns:
            datetime.datetime | None: UTC datetime representing the start of the
                day after the selected end date, or `None` when date filtering
                is disabled.
        """
        if not self.chk_date_range.isChecked():
            return None
        return self._qdate_to_utc_ceiling(self.date_end.date())

    def _do_export(self) -> None:
        """Validates the export configuration and starts the export task.

        Computes and stores the active date-range filters, prompts for an export
        name when a ZIP export has no name, and submits the export operation to
        the shared task service.

        If the date range is invalid or the user cancels the ZIP export-name
        prompt, no export task is started.
        """
        try:
            self._filter_min = self._compute_filter_min()
            self._filter_max = self._compute_filter_max()
        except ValueError as e:
            Log.e(TAG, f"Input Error: {e}")
            QtWidgets.QMessageBox.warning(self, "Export by date range", str(e))
            return
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

    def _set_running(self, running: bool) -> None:
        """Updates the wizard UI to reflect export task activity.

        Tracks whether an export task is running and asynchronously updates the
        navigation buttons and progress bar through Qt's queued invocation
        mechanism. The Next/Export and Back buttons are disabled while an export
        is active, and the progress bar is reset and shown when a task starts.

        Args:
            running: Whether an export task is currently running.
        """
        self._task_running = running
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

    def _policy_id(self) -> int:
        """Returns the identifier of the selected existing-file policy.

        Returns:
            int: Identifier associated with the currently selected policy card.
        """
        return self.policy_group.checkedId()

    def _expand_csv_cols(self) -> list[str]:
        """Expands selected CSV field names into concrete report columns.

        Converts user-facing field selections into the column names written to
        the exported CSV report. `Temp` is mapped to `Temperature` and
        `Formulation` expands into one column for each configured formulation
        component. All other selected fields are passed through unchanged.

        Returns:
            list[str]: Ordered list of concrete CSV report column names.
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

    def _export_task(
        self,
        abort,
        name: str,
        output_folder: str,
        date_filter,
        date_filter_max=None,
    ):
        """Exports the selected runs to the configured destination.

        Performs the complete export operation for CSV reports, ZIP archives,
        and plain folders. Handles existing-file policies, run and date-range
        filtering, unnamed-run handling, nested-folder flattening, ZIP
        packaging, progress reporting, cancellation, and export-history
        logging.

        The task updates the shared service with progress and completion state
        and ensures the running/frozen UI state is restored when the operation
        finishes.

        Args:
            abort: Cancellation event or compatible object whose `is_set()`
                method returns `True` when the export should be aborted.
            name: Export name used to construct the destination path.
            output_folder: Base folder or drive path where the export should be
                written.
            date_filter: Inclusive lower date bound. `0` disables the lower
                date filter.
            date_filter_max: Optional exclusive upper date bound. `None`
                leaves the upper date range open.

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

            # CSV report header
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
                    if policy == POLICY_SKIP:  # "Cancel" in CSV mode
                        Log.e(TAG, "CSV report already exists; user selected Cancel.")
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

            # ZIP: expand an existing archive so we can merge into it
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

            # -- Walk the data tree, exporting each matching run
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

            # Flatten nested folders
            if not is_csv:
                self._flatten_nested(export_path)

            # ZIP packaging
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

            Log.i(TAG, f"DONE - exported {copied} run(s) to {export_path}.")
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

    def _flatten_nested(self, export_path: str) -> None:
        """Collapses unnecessary single-child folder chains in an export.

        Walks downward through directories containing no files and exactly one
        child directory, then moves the resulting contents toward the export's
        top-level directory. This removes redundant nesting introduced while
        exporting runs and preserves the configured existing-file policy when
        copying contents.

        Args:
            export_path: Root path of the exported directory tree to flatten.
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

    def _append_run_to_csv(self, run: str, cols: list, date_filter=0, date_filter_max=None):
        """Parses a run and appends its data as a row to the CSV report.

        Reads the run's capture data, applies the configured date-range filter,
        extracts only the information required by the requested columns, builds
        the corresponding CSV row, rounds numeric values, and appends the row to
        the active report.

        Runs that fall outside the date range, lack required analysis data, have
        missing capture files, or encounter parsing/conversion errors are
        skipped and reported as unsuccessful.

        Args:
            run: Path to the run directory to export.
            cols: Ordered list of concrete CSV column names to generate.
            date_filter: Inclusive lower UTC datetime bound. `0` disables the
                lower date filter.
            date_filter_max: Optional exclusive upper UTC datetime bound.
                `None` disables the upper date filter.

        Returns:
            bool: `True` if the run was successfully written to the CSV report;
                `False` if the run was skipped or could not be exported.
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

            # Date filtering
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

            # Round floats to 2 dp where possible
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
    def _formulation_cell(formulation, attr: str) -> str:
        """Formats a formulation component for a CSV cell.

        Extracts the requested formulation component and formats it as
        `"<concentration> <units> <ingredient name>"`. Missing formulations,
        missing components, or components whose ingredient name is `"None"`
        produce an empty string.

        Args:
            formulation: Formulation object containing component attributes.
            attr: Name of the formulation component attribute to retrieve.

        Returns:
            str: Formatted formulation component, or an empty string when no
                usable component is available.
        """
        if not formulation:
            return ""
        component = getattr(formulation, attr, None)
        if component and component.ingredient.name != "None":
            return f"{component.concentration} {component.units} {component.ingredient.name}"
        return ""

    def _build_csv_row(
        self,
        cols: list,
        run_name: str,
        viscosity_profile,
        average_viscosity: float,
        std_dev: float,
        temperature: float,
        formulation,
        notes: str,
    ) -> list | None:
        """Builds a CSV row in the requested column order.

        Maps concrete CSV column names to their corresponding run, viscosity,
        temperature, formulation, and notes values. Formulation component
        columns are formatted through :meth:`_formulation_cell`.

        Args:
            cols: Ordered list of concrete CSV column names.
            run_name: Name of the run.
            viscosity_profile: Viscosity profile values for the run.
            average_viscosity: Average viscosity calculated from the profile.
            std_dev: Standard deviation of the viscosity profile.
            temperature: Run temperature.
            formulation: Parsed formulation object, if available.
            notes: Run notes text.

        Returns:
            list | None: CSV row values in the same order as `cols`, or
                `None` when an unknown column is encountered.
        """
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

    def _copytree(
        self,
        src: str,
        dst: str,
        policy: int,
        copied: int = 0,
        skipped: int = 0,
        date_filter: int = 0,
        date_filter_max=None,
    ) -> tuple[int, int]:
        """Recursively copies files while applying export policies and filters.

        Traverses the source directory tree and copies files into the destination
        according to the configured existing-file policy. Existing files can be
        replaced unconditionally, merged when the source is sufficiently newer,
        or left untouched. Optional date filtering is applied using source file
        modification times.

        XML files are counted separately to track the number of copied and
        skipped run-related files.

        Args:
            src: Source directory to copy.
            dst: Destination directory.
            policy: Existing-file policy. `POLICY_REPLACE` overwrites existing
                files, `POLICY_MERGE` replaces files that are more than two
                seconds newer, and other policies leave existing files untouched.
            copied: Running count of copied XML files.
            skipped: Running count of skipped XML files.
            date_filter: Inclusive lower UTC datetime bound. `0` disables the
                lower date filter.
            date_filter_max: Optional exclusive upper UTC datetime bound.
                `None` disables the upper date filter.

        Returns:
            tuple[int, int]: Updated `(copied, skipped)` XML file counts.
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
                # Recency filter
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

    def _write_history(
        self,
        data_path: str,
        export_path: str,
        copied: int,
        skipped: int,
        is_zip: bool,
        date_filter,
    ) -> None:
        """Prepends an HTML-formatted entry to the export history log.

        Records the completed export operation using the format consumed by the
        application's History view. The entry includes the export timestamp,
        source and destination paths, scope, format, existing-file policy, and
        any skipped-run information.

        Args:
            data_path: Root path containing the exported run data.
            export_path: Destination path used for the export.
            copied: Number of runs successfully exported.
            skipped: Number of runs skipped during export.
            is_zip: Whether the export was packaged as a ZIP archive.
            date_filter: Lower date-filter bound. `0` indicates that date
                filtering was not enabled.
        """
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

    def _field(
        self,
        caption_text: str,
        control: QtWidgets.QWidget | QtWidgets.QLayout,
    ) -> QtWidgets.QWidget:
        """Builds a compact caption-and-control field container.

        Creates a vertically arranged widget containing a caption followed by
        either a child widget or an existing layout. The resulting wrapper can
        be inserted into the responsive settings grid and treated as a single
        layout unit.

        Args:
            caption_text: Text displayed as the field caption.
            control: QWidget or QLayout containing the field's interactive
                control(s).

        Returns:
            QtWidgets.QWidget: Wrapper containing the caption and supplied
                control or layout.
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

    def _caption(self, text: str) -> QtWidgets.QLabel:
        """Creates a styled caption label.

        Args:
            text: Caption text to display.

        Returns:
            QtWidgets.QLabel: Label styled using the shared caption-label
                stylesheet.
        """
        w = QtWidgets.QLabel(text)
        w.setStyleSheet(caption_label_qss())
        return w

    @staticmethod
    def _hairline() -> QtWidgets.QFrame:
        """Create a subtle horizontal divider for stacked sections.

        Creates a 1-pixel-high horizontal frame using the shared hairline
        stylesheet. The resulting divider is intended for use between
        borderless sections and follows the same visual treatment as the
        separators used by `UserPreferencesWidget`.

        Returns:
            QtWidgets.QFrame: A themed 1-pixel horizontal divider.
        """
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFixedHeight(1)
        line.setStyleSheet(hairline_qss())
        return line

    @staticmethod
    def _radio_qss() -> str:
        """Build the stylesheet for radio buttons and checkboxes.

        Retrieves the current theme tokens and generates a stylesheet that
        applies the flat text color, compact font size, and transparent
        background used by radio buttons and checkboxes.

        Returns:
            str: A Qt stylesheet for `QRadioButton` and `QCheckBox` widgets.
        """
        tok = ThemeManager.instance().tokens()
        return (
            f"QRadioButton, QCheckBox {{ color: {tok_css(tok['flat_text'])}; "
            "font-size: 12px; background: transparent; }"
        )

    @staticmethod
    def _picker_box_qss() -> str:
        """Build the stylesheet for the picker action container.

        Returns a transparent, borderless stylesheet so the picker container
        does not introduce its own visual surface. Visual separation between
        picker actions is provided by the individual separators shown when
        the relevant actions are expanded.

        Returns:
            str: A Qt stylesheet for the `pickerBox` frame.
        """
        return "QFrame#pickerBox { background: transparent; border: none; }"

    @staticmethod
    def _picker_separator() -> QtWidgets.QFrame:
        """Create a thin vertical divider between picker actions.

        Creates a fixed-size vertical frame using the current theme's flat
        border color. The divider is intended to visually separate the
        borderless picker actions such as Detect, Eject, and Choose.

        Returns:
            QtWidgets.QFrame: A themed 1-pixel-wide vertical divider.
        """
        tok = ThemeManager.instance().tokens()
        sep = QtWidgets.QFrame()
        sep.setFixedWidth(1)
        sep.setFixedHeight(18)
        sep.setStyleSheet(f"background: {tok_css(tok['flat_border'])}; border: none;")
        return sep

    @staticmethod
    def _scroll_qss() -> str:
        """Build the stylesheet for the export step scroll area.

        Applies a transparent background and removes the scroll area's border.
        The scrollbar handle styling is provided by the application's global
        `QScrollBar` stylesheet.

        Returns:
            str: A Qt stylesheet for the `exportScroll` scroll area.
        """
        return "QScrollArea#exportScroll { background: transparent; border: none; }"

    @staticmethod
    def _inline_lbl_qss() -> str:
        """Build the stylesheet for inline labels.

        Retrieves the current theme tokens and creates a transparent label
        style using the application's flat text color and compact 12-pixel
        font size.

        Returns:
            str: A Qt stylesheet for `QLabel` widgets.
        """
        tok = ThemeManager.instance().tokens()
        return (
            f"QLabel {{ color: {tok_css(tok['flat_text'])}; font-size: 12px; "
            "background: transparent; }"
        )

    def _date_qss(self) -> str:
        """Build the themed stylesheet for the date editor.

        Creates the styled appearance used by `QDateEdit` controls,
        including themed backgrounds, borders, text, hover and focus states,
        and the calendar drop-down icon. The associated calendar popup is
        styled using the current combo-box theme tokens.

        Returns:
            str: A Qt stylesheet for the `QDateEdit` control and its calendar
            popup.
        """
        tok = ThemeManager.instance().tokens()
        icon_path = self._icon_file_path("date-range.svg")
        drop_image = f"image: url({icon_path});" if icon_path else ""
        return f"""
            QDateEdit {{
                background: {tok_css(tok['combo_bg'])};
                border: 1px solid {tok_css(tok['combo_border'])};
                border-radius: 14px; padding-left: 12px; padding-right: 6px;
                color: {tok_css(tok['combo_text'])}; font-weight: bold; min-height: 26px;
            }}
            QDateEdit:hover {{
                background: {tok_css(tok['combo_bg_hover'])};
                border: 1px solid {tok_css(tok['combo_border_hover'])};
            }}
            QDateEdit:focus {{
                background: {tok_css(tok['combo_bg_focus'])};
                border: 1px solid {tok_css(tok['combo_border_focus'])};
            }}
            QDateEdit::drop-down {{
                border: none; background: transparent; width: 24px;
                subcontrol-position: center right; margin-right: 4px;
            }}
            QDateEdit::down-arrow {{
                {drop_image}
                width: 14px; height: 14px;
            }}
            QCalendarWidget QWidget {{
                alternate-background-color: {tok_css(tok['combo_popup_bg'])};
                background-color: {tok_css(tok['combo_popup_bg'])};
            }}
            QCalendarWidget QAbstractItemView:enabled {{
                color: {tok_css(tok['combo_text'])};
                selection-background-color: {tok_css(tok['combo_selection_bg'])};
                selection-color: {tok_css(tok['combo_selection_text'])};
            }}
        """

    def _icon(self, name: str) -> QtGui.QIcon:
        """Create an icon from a named application icon resource.

        Resolves the icon file path using :meth:`_icon_file_path` and creates a
        `QIcon` from the resolved path. If the icon cannot be located, returns
        an empty icon.

        Args:
            name (str): The filename of the icon resource.

        Returns:
            QtGui.QIcon: The resolved application icon, or an empty icon when
            the resource cannot be found.
        """
        path = self._icon_file_path(name)
        return QtGui.QIcon(path) if path else QtGui.QIcon()

    @staticmethod
    def _icon_file_path(name: str):
        """Resolve the filesystem path for an application icon.

        Uses the application's architecture path to locate an icon within the
        `QATCH/icons` directory. Path separators are normalized for use by
        Qt styles and resources. Any lookup or filesystem error is suppressed
        and treated as a missing icon.

        Args:
            name (str): The filename of the icon resource.

        Returns:
            str: The normalized icon file path if it exists; otherwise an empty
            string.
        """
        try:
            from QATCH.common.architecture import Architecture

            path = os.path.join(Architecture.get_path(), "QATCH", "icons", name)
            if os.path.exists(path):
                return path.replace("\\", "/")
        except Exception:
            pass
        return ""

    def _card(
        self,
        title: str,
        subtitle: str = "",
        header_right: str | None = None,
    ) -> QtWidgets.QFrame:
        """Create a borderless content section with a populated header.

        Creates a transparent, borderless frame containing a title row, an
        optional subtitle, and a body layout for caller-supplied controls. An
        optional widget or layout can be placed on the right side of the title
        row for actions or other auxiliary controls.

        The card intentionally does not provide its own background or border.
        The surrounding step container supplies the shared content-pane
        surface, avoiding nested bordered panels.

        Args:
            title (str): Text displayed in the section header.
            subtitle (str, optional): Descriptive text displayed beneath the
                title. Defaults to an empty string.
            header_right (QtWidgets.QWidget or QtWidgets.QLayout, optional):
                A widget or layout to place at the right side of the header.
                Defaults to `None`.

        Returns:
            QtWidgets.QFrame: The constructed content section. Its `body`
            attribute contains a `QVBoxLayout` for adding section controls.
        """
        card = QtWidgets.QFrame()
        card.setObjectName("dataCard")
        card.setStyleSheet("QFrame#dataCard { background: transparent; border: none; }")
        outer = QtWidgets.QVBoxLayout(card)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)

        head_row = QtWidgets.QHBoxLayout()
        head_row.setContentsMargins(0, 0, 0, 0)
        head_row.setSpacing(8)
        header = QtWidgets.QLabel(title)
        card._header_lbl = header
        head_row.addWidget(header)
        head_row.addStretch(1)
        if header_right is not None:
            if isinstance(header_right, QtWidgets.QLayout):
                head_row.addLayout(header_right)
            else:
                head_row.addWidget(header_right)
        outer.addLayout(head_row)
        if subtitle:
            sub = QtWidgets.QLabel(subtitle)
            sub.setWordWrap(True)
            card._sub_lbl = sub
            outer.addWidget(sub)
        else:
            card._sub_lbl = None
        card.body = QtWidgets.QVBoxLayout()
        card.body.setContentsMargins(0, 2, 0, 0)
        card.body.setSpacing(8)
        outer.addLayout(card.body)
        self._restyle_card(card)
        self._cards.append(card)
        return card

    @staticmethod
    def _restyle_card(card) -> None:
        """Apply the current theme styling to a content card.

        Updates the card header and optional subtitle using the current theme
        tokens. The header receives the flat text color and bold compact
        typography, while the subtitle uses the shared description-label
        stylesheet.

        Args:
            card (QtWidgets.QFrame): The content card whose header and subtitle
                styling should be refreshed.
        """
        tok = ThemeManager.instance().tokens()
        card._header_lbl.setStyleSheet(
            f"QLabel {{ color: {tok_css(tok['flat_text'])}; font-size: 12px; "
            "font-weight: bold; background: transparent; }"
        )
        if card._sub_lbl is not None:
            card._sub_lbl.setStyleSheet(desc_label_qss())
