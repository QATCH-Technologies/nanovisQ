"""
visualization_panel.py

Interactive pyqtgraph-based viscosity visualization panel for the VisQAI dashboard.

Provides ``VisualizationPanel``, a self-contained ``QWidget`` that renders
shear-rate vs. viscosity profiles (standard mode) or true-vs-predicted parity
plots (parity mode) with rich interactive features: hover/click annotations,
crosshair, log/linear axis toggling, confidence-interval fill, measured-data
overlay, and keyboard-free zoom / pan controls.

All axis ticks and axis-name labels are drawn manually as pooled ``TextItem``
overlays so they survive pyqtgraph's ``setLogMode`` and remain correctly
positioned after any zoom or pan operation.  The axis label pool is managed
by ``update_internal_axes``, which is always called through the
``axis_debounce`` timer (50 ms single-shot) to coalesce rapid range-change
events into a single redraw.

Plot modes:

* **standard** - one viscosity-profile curve per series, with optional CI fill,
  measured-data dashed overlay, and CP-overlay scatter points at
  ``STANDARD_SHEAR_RATES``.
* **parity** - true vs. predicted viscosity scatter per series with a ``y = x``
  reference line.


Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.3
"""

import os

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import numpy as np
import pyqtgraph as pg
from scipy import interpolate

try:
    TAG = ["VisualizationPanel (HEADLESS)"]

    class Log:
        @staticmethod
        def d(TAG, msg=""):
            print("DEBUG:", TAG, msg)

        @staticmethod
        def i(TAG, msg=""):
            print("INFO:", TAG, msg)

        @staticmethod
        def w(TAG, msg=""):
            print("WARNING:", TAG, msg)

        @staticmethod
        def e(TAG, msg=""):
            print("ERROR:", TAG, msg)

    from architecture import Architecture
    from src.models.formulation import ViscosityProfile
    from styles.style_loader import load_stylesheet
except (ModuleNotFoundError, ImportError):
    TAG = "[VisualizationPanel]"
    from QATCH.VisQAI.src.models.formulation import ViscosityProfile
    from QATCH.VisQAI.src.view.styles.style_loader import load_stylesheet
    from QATCH.common.architecture import Architecture


class VisualizationPanel(QtWidgets.QWidget):
    """Interactive shear-rate / viscosity plot panel with dual render modes.

    Manages the full lifecycle of a pyqtgraph ``PlotWidget`` embedded inside
    a ``QGridLayout`` stack shared with a translucent loading overlay and an
    empty-state placeholder label.  Floating action buttons (options, home,
    zoom-in, zoom-out) are parented directly to the ``PlotWidget`` and
    repositioned on every ``Resize`` event via ``eventFilter``.

    All axis tick labels and axis-name labels are drawn as pooled
    ``pg.TextItem`` objects managed by ``update_internal_axes``; pyqtgraph's
    native axis widgets are hidden so the custom labels have full control over
    positioning in both log and linear coordinate spaces.

    Attributes:
        STANDARD_SHEAR_RATES (list[int]): Class-level list of five canonical
            shear-rate values used for CP-overlay scatter point placement.
        data_series (list[dict]): Current set of viscosity-profile data
            packages in standard mode.  Each dict must contain ``"x"`` and
            ``"y"`` arrays; optional keys include ``"color"``, ``"config_name"``,
            ``"lower"``, ``"upper"``, and ``"measured_y"``.
        measured_scatter_items (list[pg.ScatterPlotItem]): All measured-data
            scatter markers currently on the plot, used for hover detection.
        predicted_scatter_items (list[pg.ScatterPlotItem]): All predicted-data
            scatter markers currently on the plot, used for hover detection.
        measured_text_annotations (dict[float, pg.TextItem]): Mapping of
            shear-rate value to the ``TextItem`` label for each measured CP
            overlay point, enabling per-point toggle on click.
        predicted_text_annotations (dict[float, pg.TextItem]): Mapping of
            shear-rate value to the ``TextItem`` label for each predicted CP
            overlay point, enabling per-point toggle on click.
        series_plot_items (list[dict]): One dict per series (plus the parity
            line in parity mode), each with keys ``"lines"``, ``"fills"``,
            ``"scatters"``, and ``"texts"`` holding the corresponding
            ``PlotItem`` objects for batch show/hide operations.
        series_hidden (list[bool]): Parallel visibility-state list for
            ``series_plot_items``; ``True`` means the series is currently
            hidden.
        axis_text_pool (list[pg.TextItem]): Reusable pool of ``TextItem``
            objects used by ``update_internal_axes`` to avoid allocating new
            items on every range change.
        hovered_scatter (pg.ScatterPlotItem | None): The scatter item currently
            under the cursor, or ``None`` when no scatter point is hovered.
        axis_debounce (QtCore.QTimer): 50 ms single-shot timer that calls
            ``update_internal_axes``; started whenever the view range changes
            to coalesce rapid resize/pan/zoom events.
    """

    STANDARD_SHEAR_RATES = [100, 1000, 10000, 100000, 15000000]
    eval_point_clicked = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        """Initialise the panel, configure pyqtgraph, and build the full UI.

        Sets up all instance state lists and dicts to empty/``None``, enables
        pyqtgraph antialiasing globally, initialises the 50 ms debounce timer
        wired to ``update_internal_axes``, then delegates full construction to
        ``init_ui``.

        Args:
            parent (QtWidgets.QWidget | None): Optional Qt parent widget.
                Defaults to ``None``.
        """
        super().__init__(parent)
        self.data_series = []
        self.measured_scatter_items = []
        self.predicted_scatter_items = []
        self.measured_text_annotations = {}
        self.predicted_text_annotations = {}
        self.series_plot_items = []
        self.series_hidden = []

        self.axis_text_pool = []
        self.hovered_scatter = None

        self.setStyleSheet(load_stylesheet())
        pg.setConfigOptions(antialias=True)

        self.axis_debounce = QtCore.QTimer()
        self.axis_debounce.setSingleShot(True)
        self.axis_debounce.setInterval(50)
        self.axis_debounce.timeout.connect(self.update_internal_axes)

        self.init_ui()

    def init_ui(self):
        """Build the complete widget hierarchy and wire all signals.

        Constructs three layers stacked in a ``QGridLayout`` (``plot_stack``):

        1. **PlotWidget** (``plot_widget``)
        2. **Overlay** (``overlay_widget``)
        3. **Placeholder** (``placeholder_label``)

        Four floating ``QPushButton`` widgets (``btn_opts``, ``btn_home``,
        ``btn_zoom_in``, ``btn_zoom_out``) are created via
        ``_create_floating_button`` and parented to ``plot_widget``.  An
        ``eventFilter`` on ``plot_widget`` repositions them on every
        ``Resize`` event.  ``_init_controls`` is called last to create all
        ``QAction`` toggles and shear-range spin boxes.
        """
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.graph_container = QtWidgets.QFrame()
        self.graph_container.setStyleSheet("background-color: transparent; border: none;")

        graph_layout = QtWidgets.QVBoxLayout(self.graph_container)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(0)

        self.plot_stack = QtWidgets.QWidget()
        self.stack_layout = QtWidgets.QGridLayout(self.plot_stack)
        self.stack_layout.setContentsMargins(0, 0, 0, 0)

        # Plot widget
        self.plot_widget = pg.PlotWidget(title="")
        self.plot_widget.setBackground(None)
        self.plot_widget.getPlotItem().setMenuEnabled(False)

        self.plot_widget.plotItem.showAxis("top", False)
        self.plot_widget.plotItem.showAxis("right", False)
        self.plot_widget.plotItem.showAxis("left", False)
        self.plot_widget.plotItem.showAxis("bottom", False)
        self.plot_widget.getPlotItem().setContentsMargins(0, 0, 0, 0)

        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)

        self.legend = self.plot_widget.addLegend()
        self.legend.anchor((1, 0), (1, 0), offset=(-70, 20))

        self.legend.setBrush(pg.mkBrush(255, 255, 255, 255))
        self.legend.setPen(pg.mkPen(None))
        self.legend.labelTextSize = "9pt"

        self.plot_widget.installEventFilter(self)
        self.vLine = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#6b7280", width=1, style=Qt.DashLine)
        )
        self.hLine = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen("#6b7280", width=1, style=Qt.DashLine)
        )
        self.vLine.setVisible(False)
        self.hLine.setVisible(False)
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)

        self.proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved
        )
        self.plot_widget.scene().sigMouseClicked.connect(self.on_plot_click)

        self.plot_widget.sigRangeChanged.connect(lambda: self.axis_debounce.start())

        # Overlays
        self.overlay_widget = QtWidgets.QFrame()
        self.overlay_widget.setStyleSheet("background-color: rgba(255, 255, 255, 180);")
        self.overlay_widget.setVisible(False)
        overlay_layout = QtWidgets.QVBoxLayout(self.overlay_widget)
        overlay_layout.setAlignment(Qt.AlignCenter)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setFixedWidth(300)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar { border: none; background-color: #e5e7eb; border-radius: 3px; }
            QProgressBar::chunk { background-color: #3b82f6; border-radius: 3px; }
            """
        )
        overlay_layout.addWidget(self.progress_bar)

        self.loading_label = QtWidgets.QLabel("Calculating...")
        self.loading_label.setStyleSheet("color: #4b5563; font-weight: 600;")
        self.loading_label.setAlignment(Qt.AlignCenter)
        overlay_layout.addWidget(self.loading_label)

        self.btn_cancel_loading = QtWidgets.QPushButton("Cancel")
        self.btn_cancel_loading.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel_loading.setFixedWidth(100)
        self.btn_cancel_loading.setStyleSheet(
            "QPushButton {"
            "  color: #374151; background-color: #f3f4f6;"
            "  border: 1px solid #d1d5db; border-radius: 6px;"
            "  padding: 5px 14px; font-weight: 600; font-size: 11px;"
            "}"
            "QPushButton:hover  { background-color: #e5e7eb; }"
            "QPushButton:pressed { background-color: #d1d5db; }"
        )
        self.btn_cancel_loading.setVisible(False)
        overlay_layout.addSpacing(8)
        overlay_layout.addWidget(self.btn_cancel_loading, 0, Qt.AlignCenter)
        self._cancel_loading_cb = None

        # Placeholder
        self.placeholder_label = QtWidgets.QLabel(
            "No data to display.\nRun a prediction or import data to view profiles."
        )
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet(
            "color: #9ca3af; font-size: 13pt; font-weight: 500; background-color: transparent;"
        )
        self.placeholder_label.setAttribute(
            Qt.WA_TransparentForMouseEvents
        )  # Prevents blocking plot interactions

        self.stack_layout.addWidget(self.plot_widget, 0, 0)
        self.stack_layout.addWidget(self.placeholder_label, 0, 0)
        self.stack_layout.addWidget(self.overlay_widget, 0, 0)

        graph_layout.addWidget(self.plot_stack)
        layout.addWidget(self.graph_container)

        # Controls
        # Options Button
        self.btn_opts = self._create_floating_button(
            os.path.join(
                Architecture.get_path(),
                "QATCH",
                "VisQAI",
                "src",
                "view",
                "icons",
                "configure-svgrepo-com.svg",
            ),
            "Graph Options",
        )
        self.btn_opts.clicked.connect(self.show_options_menu)

        # Home Button
        self.btn_home = self._create_floating_button(
            os.path.join(
                Architecture.get_path(),
                "QATCH",
                "VisQAI",
                "src",
                "view",
                "icons",
                "home2-svgrepo-com.svg",
            ),
            "Reset View",
        )
        self.btn_home.clicked.connect(self.reset_view)

        # Zoom In
        self.btn_zoom_in = self._create_floating_button(
            os.path.join(
                Architecture.get_path(),
                "QATCH",
                "VisQAI",
                "src",
                "view",
                "icons",
                "add-plus-svgrepo-com.svg",
            ),
            "Zoom In",
        )
        self.btn_zoom_in.clicked.connect(self.zoom_in)

        # Zoom Out
        self.btn_zoom_out = self._create_floating_button(
            os.path.join(
                Architecture.get_path(),
                "QATCH",
                "VisQAI",
                "src",
                "view",
                "icons",
                "minus-svgrepo-com.svg",
            ),
            "Zoom Out",
        )
        self.btn_zoom_out.clicked.connect(self.zoom_out)

        self._init_controls()

    def _create_floating_button(self, icon_name, tooltip):
        """Create a styled circular floating action button parented to ``plot_widget``.

        Attempts to load the SVG icon from *icon_name*; falls back to a
        single Unicode glyph (``+``, ``-``, ``⌂``, or ``⚙``) derived from a
        keyword match in the filename when the path does not exist.  Applies a
        white pill style with a subtle drop shadow.

        Args:
            icon_name (str): Absolute or relative path to the SVG icon file.
            tooltip (str): Tooltip string shown on hover.

        Returns:
            QtWidgets.QPushButton: Fully styled 36 x 36 px button parented to
                ``plot_widget``.
        """
        btn = QtWidgets.QPushButton("", self.plot_widget)
        btn.setFixedSize(36, 36)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(tooltip)

        icon_path = os.path.join(Architecture.get_path(), icon_name)
        if os.path.exists(icon_path):
            btn.setIcon(QtGui.QIcon(icon_path))
            btn.setIconSize(QtCore.QSize(20, 20))
        else:
            if "plus" in icon_name:
                btn.setText("+")
            elif "minus" in icon_name:
                btn.setText("-")
            elif "home" in icon_name:
                btn.setText("⌂")
            elif "configure" in icon_name:
                btn.setText("⚙")

        btn.setStyleSheet(
            """
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 18px;
                color: #555;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #f9fafb;
                border-color: #d1d5db;
                color: #17C4F3;
            }
            """
        )
        shadow = QtWidgets.QGraphicsDropShadowEffect(btn)
        shadow.setBlurRadius(8)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QtGui.QColor(0, 0, 0, 20))
        btn.setGraphicsEffect(shadow)
        return btn

    def _init_controls(self):
        """Create all ``QAction`` toggles and shear-range spin boxes.

        Instantiates and configures the following checkable ``QAction`` objects
        (all wired to ``update_plot`` or related slots unless noted):

        * ``act_log_x`` - Log Scale X (default: checked / on).
        * ``act_log_y`` - Log Scale Y (default: unchecked / off).
        * ``act_axis_labels`` - Show Axis Labels (default: checked; debounced).
        * ``act_crosshairs`` - Show Crosshairs (default: unchecked; wired to
          ``toggle_crosshairs``).
        * ``act_ci`` - Show Confidence Interval (default: checked).
        * ``act_cp`` - Show CP Overlay (default: checked; wired to
          ``_toggle_cp_overlay``).
        * ``act_measured`` - Show Measured Profile (default: unchecked and
          disabled until measured data is loaded).

        Also creates ``spin_min_shear`` (default 100 s⁻¹) and
        ``spin_max_shear`` (default 15 000 000 s⁻¹) ``QDoubleSpinBox``
        widgets, both wired to ``update_plot``.
        """
        self.act_log_x = QtWidgets.QAction("Log Scale X", self, checkable=True)
        self.act_log_x.setChecked(True)
        self.act_log_x.toggled.connect(self.update_plot)

        self.act_log_y = QtWidgets.QAction("Log Scale Y", self, checkable=True)
        self.act_log_y.setChecked(False)
        self.act_log_y.toggled.connect(self.update_plot)

        self.act_axis_labels = QtWidgets.QAction("Show Axis Labels", self, checkable=True)
        self.act_axis_labels.setChecked(True)
        self.act_axis_labels.toggled.connect(lambda: self.axis_debounce.start())

        self.act_crosshairs = QtWidgets.QAction("Show Crosshairs", self, checkable=True)
        self.act_crosshairs.setChecked(False)
        self.act_crosshairs.toggled.connect(self.toggle_crosshairs)

        self.act_ci = QtWidgets.QAction("Show Confidence Interval", self, checkable=True)
        self.act_ci.setChecked(True)
        self.act_ci.toggled.connect(self.update_plot)

        self.act_cp = QtWidgets.QAction("Show CP Overlay", self, checkable=True)
        self.act_cp.setChecked(True)
        self.act_cp.toggled.connect(self._toggle_cp_overlay)

        self.act_measured = QtWidgets.QAction("Show Measured Profile", self, checkable=True)
        self.act_measured.setChecked(False)
        self.act_measured.setEnabled(False)
        self.act_measured.toggled.connect(self.update_plot)

        _spin_style = """
            QDoubleSpinBox {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 3px 6px;
                color: #24292f;
                font-size: 12px;
                min-width: 110px;
            }
            QDoubleSpinBox:hover {
                border-color: #2596be;
            }
            QDoubleSpinBox:focus {
                border-color: #2596be;
                outline: none;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 18px;
                border: none;
                background: transparent;
            }
            QDoubleSpinBox::up-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 5px solid #6b7280;
                width: 0; height: 0;
            }
            QDoubleSpinBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #6b7280;
                width: 0; height: 0;
            }
        """

        self.spin_min_shear = QtWidgets.QDoubleSpinBox()
        self.spin_min_shear.setRange(0, 15000000)
        self.spin_min_shear.setValue(100)
        self.spin_min_shear.setDecimals(0)
        self.spin_min_shear.setSingleStep(1000)
        self.spin_min_shear.setStyleSheet(_spin_style)
        self.spin_min_shear.valueChanged.connect(self.update_plot)

        self.spin_max_shear = QtWidgets.QDoubleSpinBox()
        self.spin_max_shear.setRange(0, 15000000)
        self.spin_max_shear.setValue(15000000)
        self.spin_max_shear.setDecimals(0)
        self.spin_max_shear.setSingleStep(10000)
        self.spin_max_shear.setStyleSheet(_spin_style)
        self.spin_max_shear.valueChanged.connect(self.update_plot)

    def toggle_crosshairs(self, checked):
        """Show or hide the dashed crosshair lines on the plot.

        Args:
            checked (bool): ``True`` to make ``vLine`` and ``hLine`` visible;
                ``False`` to hide them.
        """
        self.vLine.setVisible(checked)
        self.hLine.setVisible(checked)

    def show_options_menu(self):
        """Display the graph-options context menu anchored below ``btn_opts``.

        Builds a styled ``QMenu`` containing all ``QAction`` toggles from
        ``_init_controls`` (separated into logical groups) and a
        ``QWidgetAction`` embedding the min/max shear-rate spin boxes in a
        two-row ``QGridLayout``.  Executes the menu modally at the bottom-left
        corner of ``btn_opts``.
        """
        menu = QtWidgets.QMenu(self)

        menu.setStyleSheet(
            """
            QMenu {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 6px;
            }
            QMenu::item {
                padding: 8px 16px;
                border-radius: 4px;
                color: #24292f;
                background-color: transparent;
            }
            QMenu::item:selected {
                background-color: #e6f7fd;
                color: #2596be;
            }
            QMenu::separator {
                height: 1px;
                background: #e5e7eb;
                margin: 4px 8px;
            }
        """
        )

        menu.addAction(self.act_log_x)
        menu.addAction(self.act_log_y)
        menu.addSeparator()
        menu.addAction(self.act_axis_labels)
        menu.addAction(self.act_crosshairs)
        menu.addAction(self.act_ci)
        menu.addAction(self.act_cp)
        menu.addAction(self.act_measured)
        menu.addSeparator()

        range_widget = QtWidgets.QWidget()
        range_layout = QtWidgets.QGridLayout(range_widget)
        range_layout.setContentsMargins(10, 2, 10, 2)
        range_layout.addWidget(QtWidgets.QLabel("Min Shear:"), 0, 0)
        range_layout.addWidget(self.spin_min_shear, 0, 1)
        range_layout.addWidget(QtWidgets.QLabel("Max Shear:"), 1, 0)
        range_layout.addWidget(self.spin_max_shear, 1, 1)
        range_action = QtWidgets.QWidgetAction(menu)
        range_action.setDefaultWidget(range_widget)
        menu.addAction(range_action)

        menu.exec_(self.btn_opts.mapToGlobal(QtCore.QPoint(0, self.btn_opts.height())))

    def update_internal_axes(self):
        """Redraw all custom tick labels and axis-name overlays.

        Called via ``axis_debounce`` after any view-range change.  Hides all
        pooled ``TextItem`` objects, then recomputes tick positions for both
        axes using the inner ``get_tick_values`` helper:

        * **Log mode** - integer powers of 10 spanning the view range;
          auto-stepped when the span exceeds 20 decades.
        * **Linear mode** - order-of-magnitude-aligned steps refined to keep
          the label count between ~5 and ~15.

        Tick labels are formatted as ``10^N`` for powers of 10, or decimal
        otherwise.  ``TextItem`` objects are retrieved from or appended to
        ``axis_text_pool`` to avoid per-frame allocation.

        When ``act_axis_labels`` is checked, centred X and Y axis-name labels
        are also placed 1.5 % inside the view boundary in their respective
        dimensions.  In parity mode the axis names reflect true/predicted
        viscosity rather than shear rate / viscosity.
        """
        for item in self.axis_text_pool:
            item.hide()

        vb = self.plot_widget.plotItem.vb
        x_range = vb.viewRange()[0]
        y_range = vb.viewRange()[1]

        log_x = self.act_log_x.isChecked()
        log_y = self.act_log_y.isChecked()
        _mode = getattr(self, "plot_mode", "standard")
        if _mode == "parity":
            log_x = getattr(self, "parity_log_visc", False)
            log_y = getattr(self, "parity_log_visc", False)

        def get_tick_values(min_v, max_v, is_log):
            if is_log:
                start_exp = int(np.floor(min_v))
                end_exp = int(np.ceil(max_v))
                if end_exp - start_exp > 20:
                    step = int((end_exp - start_exp) / 10)
                    return [float(i) for i in range(start_exp, end_exp + 1, step)]
                return [float(i) for i in range(start_exp, end_exp + 1)]
            else:
                span = max_v - min_v
                if span <= 1e-9:
                    return []
                step = 10 ** int(np.floor(np.log10(span)) - 1)
                if span / step > 15:
                    step *= 2
                if span / step > 15:
                    step *= 2.5
                start = np.ceil(min_v / step) * step
                return np.arange(start, max_v, step * 2)

        x_ticks = get_tick_values(x_range[0], x_range[1], log_x)
        y_ticks = get_tick_values(y_range[0], y_range[1], log_y)
        y_pos_for_x_labels = y_range[0] + (y_range[1] - y_range[0]) * 0.06
        x_pos_for_y_labels = x_range[0] + (x_range[1] - x_range[0]) * 0.06

        pool_index = 0

        def get_item():
            nonlocal pool_index
            if pool_index < len(self.axis_text_pool):
                item = self.axis_text_pool[pool_index]
                item.show()
                pool_index += 1
                return item
            else:
                item = pg.TextItem("", color="#9ca3af", anchor=(0.5, 1))
                item.setFont(QtGui.QFont("Arial", 8))
                self.plot_widget.addItem(item, ignoreBounds=True)
                self.axis_text_pool.append(item)
                pool_index += 1
                return item

        for x_val in x_ticks:
            if x_val < x_range[0] or x_val > x_range[1]:
                continue

            val_display = 10**x_val if log_x else x_val
            text = (
                f"{val_display:.0e}"
                if abs(val_display) >= 1000 or abs(val_display) < 0.01
                else f"{val_display:.1f}"
            )
            if "e" in text:
                base, power = text.split("e")
                text = f"10^{int(power)}" if base == "1" else text

            label = get_item()
            label.setText(text)
            label.setAnchor((0.5, 1))
            label.setAngle(0)
            label.setPos(x_val, y_pos_for_x_labels)

        for y_val in y_ticks:
            if y_val < y_range[0] or y_val > y_range[1]:
                continue

            val_display = 10**y_val if log_y else y_val
            text = f"{val_display:.1f}"

            label = get_item()
            label.setText(text)
            label.setAnchor((0, 0.5))
            label.setAngle(0)
            label.setPos(x_pos_for_y_labels, y_val)

        if self.act_axis_labels.isChecked():
            x_center = (x_range[0] + x_range[1]) / 2
            y_center = (y_range[0] + y_range[1]) / 2

            mode = getattr(self, "plot_mode", "standard")
            if mode == "parity":
                x_label_text = "Log True Viscosity (cP)" if log_x else "True Viscosity (cP)"
                y_label_text = (
                    "Log Predicted Viscosity (cP)" if log_y else "Predicted Viscosity (cP)"
                )
            else:
                x_label_text = "Log Shear Rate (1/s)" if log_x else "Shear Rate (1/s)"
                y_label_text = "Log Viscosity (cP)" if log_y else "Viscosity (cP)"

            x_name = get_item()
            x_name.setText(x_label_text)
            x_name.setAnchor((0.5, 1))
            x_name.setAngle(0)
            x_name.setPos(x_center, y_range[0] + (y_range[1] - y_range[0]) * 0.015)
            x_name.setColor("#6b7280")
            x_name.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Bold))

            y_name = get_item()
            y_name.setText(y_label_text)
            y_name.setAnchor((0.5, 0))
            y_name.setAngle(90)
            y_name.setPos(x_range[0] + (x_range[1] - x_range[0]) * 0.015, y_center)
            y_name.setColor("#6b7280")
            y_name.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Bold))

    def set_plot_title(self, title_text):
        """Set the plot title to *title_text* with consistent HTML styling.

        Wraps *title_text* in a ``<span>`` tag applying the dashboard's
        standard title style (``#374151``, 11 pt, weight 600).  Does nothing
        if ``plot_widget`` has not yet been created.

        Args:
            title_text (str): Plain text for the plot title.
        """
        if hasattr(self, "plot_widget"):
            self.plot_widget.setTitle(
                f"<span style='color: #374151; font-size: 11pt; font-weight: 600;'>{title_text}</span>"
            )

    def show_loading(self, cancel_callback=None):
        """Show the translucent loading overlay with an animated progress bar.

        Makes ``overlay_widget`` visible and starts ``anim_timer`` (10 ms
        interval) to increment the progress bar from 0 to 90 % in the
        foreground.  ``QApplication.processEvents()`` is called once
        immediately so the overlay renders before any heavy computation begins.

        When *cancel_callback* is provided, ``btn_cancel_loading`` is shown
        and wired so that clicking it disables the button, changes its text
        to ``"Cancelling…"``, and invokes the callback (e.g.
        ``worker.requestInterruption``).  Any previous connection on the
        button is disconnected first to prevent signal accumulation across
        multiple overlay lifecycles.  Pass ``None`` to hide the cancel button
        (appropriate for fast synchronous predictions).

        Args:
            cancel_callback (callable | None): Zero-argument callable invoked
                when the user clicks "Cancel".  Defaults to ``None``.
        """
        self._cancel_loading_cb = cancel_callback
        try:
            self.btn_cancel_loading.clicked.disconnect()
        except TypeError:
            pass
        if cancel_callback is not None:

            def _on_cancel():
                self.btn_cancel_loading.setEnabled(False)
                self.btn_cancel_loading.setText("Cancelling…")
                cancel_callback()

            self.btn_cancel_loading.clicked.connect(_on_cancel)
            self.btn_cancel_loading.setEnabled(True)
            self.btn_cancel_loading.setText("Cancel")
            self.btn_cancel_loading.setVisible(True)
        else:
            self.btn_cancel_loading.setVisible(False)

        self.overlay_widget.setVisible(True)
        self.progress_bar.setValue(0)
        self.anim_timer = QtCore.QTimer()
        self.anim_timer.timeout.connect(self._animate_step)
        self.anim_timer.start(10)
        QtWidgets.QApplication.processEvents()

    def _animate_step(self):
        """Advance the loading progress bar by one step, capping at 90 %.

        Connected to ``anim_timer.timeout``.  Increments the progress bar
        value by 1 on each tick until it reaches 90, leaving the final 10 %
        for ``hide_loading`` to fill when the operation completes.
        """
        val = self.progress_bar.value()
        if val < 90:
            self.progress_bar.setValue(val + 1)

    def hide_loading(self):
        """Stop the progress animation, complete the bar to 100 %, and hide the overlay.

        Stops ``anim_timer``, sets the progress bar to 100 %, then hides and
        fully resets ``btn_cancel_loading`` (disconnects signals, restores
        text and enabled state) so it does not bleed into subsequent overlay
        uses.  Calls ``QApplication.processEvents()`` to flush the final
        visual state before hiding ``overlay_widget``.
        """
        if hasattr(self, "anim_timer"):
            self.anim_timer.stop()
        self.progress_bar.setValue(100)
        if hasattr(self, "btn_cancel_loading"):
            try:
                self.btn_cancel_loading.clicked.disconnect()
            except TypeError:
                pass
            self.btn_cancel_loading.setText("Cancel")
            self.btn_cancel_loading.setEnabled(True)
            self.btn_cancel_loading.setVisible(False)
        self._cancel_loading_cb = None
        QtWidgets.QApplication.processEvents()
        self.overlay_widget.setVisible(False)

    def eventFilter(self, source, event):
        """Intercept ``plot_widget`` resize events to reposition floating buttons.

        When *source* is ``plot_widget`` and the event type is ``Resize``,
        calls ``_reposition_overlay_buttons`` and starts ``axis_debounce`` to
        refresh tick labels at the new size.  All other events are forwarded
        to the base-class handler.

        Args:
            source (QtCore.QObject): The object that generated the event.
            event (QtCore.QEvent): The event to filter.

        Returns:
            bool: The result of ``super().eventFilter(source, event)``.
        """
        if source == self.plot_widget and event.type() == QtCore.QEvent.Resize:
            self._reposition_overlay_buttons()
            self.axis_debounce.start()
        return super().eventFilter(source, event)

    def _reposition_overlay_buttons(self):
        """Lay out the four floating action buttons in a right-aligned vertical column.

        Positions ``btn_opts``, ``btn_home``, ``btn_zoom_in``, and
        ``btn_zoom_out`` with 20 px right margin, 20 px top margin, and 10 px
        inter-button spacing, all measured from the top-right corner of
        ``plot_widget``.  The x position is clamped to a minimum of 0 to
        prevent buttons from escaping the widget's left edge during extreme
        resize events.
        """
        margin_right = 20
        margin_top = 20
        spacing = 10
        btn_height = 36

        x = self.plot_widget.width() - self.btn_opts.width() - margin_right

        y = margin_top
        self.btn_opts.move(max(0, x), y)

        y += btn_height + spacing
        self.btn_home.move(max(0, x), y)

        y += btn_height + spacing
        self.btn_zoom_in.move(max(0, x), y)

        y += btn_height + spacing
        self.btn_zoom_out.move(max(0, x), y)

    def zoom_in(self):
        """Scale the view box inward by a factor of 0.8 on both axes.

        Delegates to ``ViewBox.scaleBy((0.8, 0.8))``, which zooms in by
        reducing the visible range symmetrically around the current centre.
        """
        self.plot_widget.plotItem.vb.scaleBy((0.8, 0.8))

    def zoom_out(self):
        """Scale the view box outward by a factor of 1.25 on both axes.

        Delegates to ``ViewBox.scaleBy((1.25, 1.25))``, which zooms out by
        expanding the visible range symmetrically around the current centre.
        """
        self.plot_widget.plotItem.vb.scaleBy((1.25, 1.25))

    def reset_view(self):
        """Reset the plot view to its intelligently snapped global limits.

        Calls ``_apply_global_limits`` to recompute and set the Y range from
        the current data, then starts ``axis_debounce`` to refresh tick
        labels at the restored view range.
        """
        self._apply_global_limits()
        self.axis_debounce.start()

    def set_data(self, data):
        """Load one or more viscosity-profile data packages and switch to standard mode.

        Accepts a single data dict or a list of dicts.  ``None`` values are
        filtered out.  Initialises ``series_hidden`` to all-``False``.
        Enables and checks ``act_measured`` when any series contains a non-``None``
        ``"measured_y"`` key; disables and unchecks it otherwise.  Calls
        ``update_plot`` to render the new data.

        Args:
            data (dict | list[dict] | None): One or more viscosity-profile
                data packages.  Each dict requires at minimum ``"x"`` (shear
                rates) and ``"y"`` (predicted viscosities); optional keys are
                ``"color"``, ``"config_name"``, ``"lower"``, ``"upper"``
                (confidence-interval bounds), and ``"measured_y"``.
        """
        self.plot_mode = "standard"
        if isinstance(data, list):
            self.data_series = [d for d in data if d is not None]
        elif data is not None:
            self.data_series = [data]
        else:
            self.data_series = []

        if not self.data_series:
            self.set_plot_title("")

        self.series_hidden = [False] * len(self.data_series)
        has_measured = any(
            "measured_y" in d and d["measured_y"] is not None for d in self.data_series
        )

        self.act_measured.setEnabled(has_measured)

        if has_measured:
            self.act_measured.setChecked(True)
        else:
            self.act_measured.setChecked(False)

        self.update_plot()

    def set_parity_data(self, parity_data, log_visc):
        """Load parity-plot data and switch to parity render mode.

        Stores *parity_data* and *log_visc*, sets ``plot_mode`` to
        ``"parity"``, then calls ``update_plot`` to render the parity scatter.

        Args:
            parity_data (list[dict]): List of per-series parity dicts.  Each
                dict must contain a ``"points"`` list of point dicts (keys
                ``"true"``, ``"pred"``, ``"shear"``, ``"config_name"``), a
                ``"color"`` hex string, and a ``"config_name"`` string.
            log_visc (bool): When ``True``, point coordinates are
                log10-transformed before plotting and axis labels read
                "Log … Viscosity (cP)".
        """
        self.plot_mode = "parity"
        self.parity_data = parity_data
        self.parity_log_visc = log_visc
        self.update_plot()

    def update_plot(self):
        """Clear and fully redraw the plot from current data and option states.

        Clears ``plot_widget``, resets all scatter/annotation/series tracking
        lists, hides all pooled axis text items, and re-adds the crosshair
        lines at their current visibility state.

        Dispatch:

        * **Parity mode** - delegates to ``_plot_parity``; returns immediately
          after triggering ``axis_debounce``.
        * **Standard mode, no data** - shows ``placeholder_label`` and
          returns.
        * **Standard mode, with data** - iterates ``data_series``, calling
          ``_plot_single_series`` for each and storing the returned items dict
          in ``series_plot_items``; hidden series are immediately concealed
          via ``_set_series_items_visible``.  After all series are rendered,
          ``_apply_global_limits`` sets the view range and ``axis_debounce``
          is started; ``_rebuild_standard_legend`` populates the legend.
        """
        self.plot_widget.clear()

        self.measured_scatter_items = []
        self.predicted_scatter_items = []
        self.measured_text_annotations = {}
        self.predicted_text_annotations = {}
        self.series_plot_items = []

        for item in self.axis_text_pool:
            item.hide()
        self.axis_text_pool = []

        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)
        self.vLine.setVisible(self.act_crosshairs.isChecked())
        self.hLine.setVisible(self.act_crosshairs.isChecked())
        self.legend.clear()

        mode = getattr(self, "plot_mode", "standard")
        if mode == "parity":
            self.placeholder_label.hide()
            self._plot_parity()
            self.axis_debounce.start()
            return

        if not self.data_series:
            self.placeholder_label.show()
            return

        self.placeholder_label.hide()

        while len(self.series_hidden) < len(self.data_series):
            self.series_hidden.append(False)
        self.series_hidden = self.series_hidden[: len(self.data_series)]

        for i, data_package in enumerate(self.data_series):
            QtWidgets.QApplication.processEvents()
            items = self._plot_single_series(data_package, index=i)
            self.series_plot_items.append(items)

            if self.series_hidden[i]:
                self._set_series_items_visible(items, False)

        self._apply_global_limits()
        self.axis_debounce.start()
        self._rebuild_standard_legend()

    def _plot_parity(self):
        """Render the parity (true vs. predicted viscosity) scatter plot.

        Disables ``setLogMode`` on the ``PlotWidget`` because all coordinates
        are pre-logged manually when ``parity_log_visc`` is ``True``.

        Steps:

        1. Synchronises ``series_hidden`` length with ``parity_data``.
        2. Computes global min/max across all point ``"true"`` and ``"pred"``
           values to size the ``y = x`` reference line.
        3. Plots a dashed grey ``y = x`` line covering the data range with
           5 % padding, registered as a non-hideable entry in
           ``series_plot_items``.
        4. For each parity series, builds a ``ScatterPlotItem`` with hover
           styling and per-point ``TextItem`` annotations (hidden by default)
           that show config name, shear rate, true viscosity, and predicted
           viscosity.  Annotations are referenced from ``pt["_annotation"]``
           so the click handler can reach them without a secondary lookup.
        5. Sets view-box limits and range symmetrically on both axes.
        """
        self.plot_widget.setLogMode(x=False, y=False)

        while len(self.series_hidden) < len(self.parity_data):
            self.series_hidden.append(False)
        self.series_hidden = self.series_hidden[: len(self.parity_data)]

        min_val = float("inf")
        max_val = float("-inf")
        for series in self.parity_data:
            for pt in series["points"]:
                min_val = min(min_val, pt["true"], pt["pred"])
                max_val = max(max_val, pt["true"], pt["pred"])

        if min_val == float("inf"):
            return

        # Plot y=x parity line
        line_min = max(min_val * 0.8, 1e-10) if self.parity_log_visc else min_val * 0.8
        line_max = max_val * 1.2
        val_min = np.log10(line_min) if self.parity_log_visc else line_min
        val_max = np.log10(line_max) if self.parity_log_visc else line_max

        parity_line = self.plot_widget.plot(
            [val_min, val_max],
            [val_min, val_max],
            pen=pg.mkPen("#9ca3af", width=2, style=Qt.DashLine),
            name="y = x",
        )

        self.series_plot_items.append(
            {"lines": [parity_line], "fills": [], "scatters": [], "texts": []}
        )

        # Plot scattered points per series
        for i, series in enumerate(self.parity_data):
            spots = []
            annotations = []
            for pt in series["points"]:
                px = np.log10(max(pt["true"], 1e-10)) if self.parity_log_visc else pt["true"]
                py = np.log10(max(pt["pred"], 1e-10)) if self.parity_log_visc else pt["pred"]
                spots.append({"pos": (px, py), "data": pt})

                # Build label
                shear_label = f"{int(pt['shear']):,}" if pt["shear"] < 1e6 else f"{pt['shear']:.2e}"
                label_text = (
                    f"{series['config_name']}\n"
                    f"Shear: {shear_label} 1/s\n"
                    f"True:  {pt['true']:.4f} cP\n"
                    f"Pred:  {pt['pred']:.4f} cP"
                )
                text_item = pg.TextItem(
                    label_text,
                    color=series["color"],
                    anchor=(0.0, 1.0),
                    border=pg.mkPen(series["color"], width=1),
                    fill=pg.mkBrush(255, 255, 255, 220),
                )
                text_item.setPos(px, py)
                text_item.setZValue(20)
                text_item.setVisible(False)
                self.plot_widget.addItem(text_item)
                annotations.append(text_item)
                pt["_annotation"] = text_item

            if spots:
                scatter = pg.ScatterPlotItem(
                    spots=spots,
                    symbol="o",
                    size=14,
                    brush=pg.mkBrush("w"),
                    pen=pg.mkPen(series["color"], width=2),
                    hoverable=True,
                    hoverSize=20,
                    hoverBrush=pg.mkBrush(series["color"]),
                    hoverPen=pg.mkPen(series["color"], width=2),
                )

                # Connect click event
                scatter.sigClicked.connect(self._on_parity_scatter_clicked)

                self.plot_widget.addItem(scatter)

                items_dict = {
                    "lines": [],
                    "fills": [],
                    "scatters": [scatter],
                    "texts": annotations,
                }
                self.series_plot_items.append(items_dict)

                if self.series_hidden[i]:
                    self._set_series_items_visible(items_dict, False)

        self.plot_widget.plotItem.vb.setLimits(
            xMin=val_min, xMax=val_max, yMin=val_min, yMax=val_max
        )
        self.plot_widget.plotItem.vb.setRange(
            xRange=[val_min, val_max], yRange=[val_min, val_max], padding=0.05
        )

    def _on_parity_scatter_clicked(self, plot, points):
        """Toggle the annotation ``TextItem`` for the clicked parity scatter point.

        Acts on the first point only when multiple overlapping points are
        clicked.  Retrieves the annotation from ``pt["_annotation"]`` and
        toggles its visibility.  Emits ``eval_point_clicked`` with the raw
        point dict so the dashboard can silently expand and scroll to the
        corresponding prediction card.

        Args:
            plot (pg.ScatterPlotItem): The scatter item that emitted the
                signal.
            points (list[pg.SpotItem]): List of clicked spot items.

        Emits:
            eval_point_clicked (dict): The raw point dict for the first
                clicked point.
        """
        if len(points) == 0:
            return
        # When multiple overlapping points are clicked, act on the first one only
        data = points[0].data()
        if not data:
            return
        ann = data.get("_annotation")
        if ann is not None:
            ann.setVisible(not ann.isVisible())
        self.eval_point_clicked.emit(data)

    def _plot_single_series(self, data, index):
        """Render one viscosity-profile series and return its plot-item dict.

        Attempts to construct a ``ViscosityProfile`` from the series data for
        smooth interpolation over 200 log-spaced shear rates within
        ``[spin_min_shear, spin_max_shear]``; falls back to a simple
        shear-range mask on the raw arrays if the profile raises.

        Renders in this order (each conditional on option states):

        1. **CI fill** - ``FillBetweenItem`` between interpolated lower/upper
           bounds, alpha-scaled by series count.
        2. **Measured overlay** - dashed line and CP scatter points at
           ``STANDARD_SHEAR_RATES``, drawn only when ``act_measured`` is
           checked and enabled.
        3. **Predicted line** - solid line and CP scatter points, skipped for
           pure measured-only series.

        Args:
            data (dict): A viscosity-profile data package as described in
                ``set_data``.
            index (int): Zero-based series index used for default colour
                selection when ``data["color"]`` is absent.

        Returns:
            dict: A series-items dict with keys ``"lines"``, ``"fills"``,
                ``"scatters"``, and ``"texts"`` containing the corresponding
                ``PlotItem`` objects for this series.
        """
        series_items = {
            "lines": [],
            "fills": [],
            "scatters": [],
            "texts": [],
        }

        raw_color = data.get("color")
        default_colors = ["#2596be", "#be4d25", "#25be4d", "#be2596", "#96be25"]
        main_color = raw_color if raw_color else default_colors[index % len(default_colors)]
        min_shear = self.spin_min_shear.value()
        max_shear = self.spin_max_shear.value()

        x_full = np.array(data["x"])
        y_full = np.array(data["y"])
        has_ci = "lower" in data and "upper" in data
        lower_full = np.array(data["lower"]) if has_ci else np.array([])
        upper_full = np.array(data["upper"]) if has_ci else np.array([])
        n_dense = 200
        dense_sr = np.logspace(
            np.log10(max(min_shear, 1)),
            np.log10(max(max_shear, 2)),
            n_dense,
        )

        vp_pred = None
        try:
            vp_pred = ViscosityProfile(list(x_full.astype(float)), list(y_full.astype(float)))
            x = dense_sr
            y = np.array([vp_pred.get_viscosity(sr) for sr in dense_sr])

            if has_ci and len(lower_full) == len(x_full):
                vp_lower = ViscosityProfile(
                    list(x_full.astype(float)), list(lower_full.astype(float))
                )
                vp_upper = ViscosityProfile(
                    list(x_full.astype(float)), list(upper_full.astype(float))
                )
                lower = np.array([vp_lower.get_viscosity(sr) for sr in dense_sr])
                upper = np.array([vp_upper.get_viscosity(sr) for sr in dense_sr])
            else:
                lower = np.array([])
                upper = np.array([])
                has_ci = False
        except Exception as e:
            Log.w(TAG, f"ViscosityProfile interpolation failed, falling back: {e}")
            mask = (x_full >= min_shear) & (x_full <= max_shear)
            x, y = x_full[mask], y_full[mask]
            lower = lower_full[mask] if has_ci and len(lower_full) == len(x_full) else np.array([])
            upper = upper_full[mask] if has_ci and len(upper_full) == len(x_full) else np.array([])

        if len(x) == 0:
            return series_items

        log_x = self.act_log_x.isChecked()
        log_y = self.act_log_y.isChecked()
        self.plot_widget.setLogMode(x=log_x, y=log_y)

        if self.act_ci.isChecked() and has_ci and len(lower) > 0:
            if log_x or log_y:
                x_ci = np.log10(np.maximum(x, 1e-10)) if log_x else x
                lower_ci = np.log10(np.maximum(lower, 1e-10)) if log_y else lower
                upper_ci = np.log10(np.maximum(upper, 1e-10)) if log_y else upper
            else:
                x_ci, lower_ci, upper_ci = x, lower, upper

            ci_color = QtGui.QColor(main_color)
            ci_color.setAlpha(20 if len(self.data_series) > 1 else 40)
            fill = pg.FillBetweenItem(
                pg.PlotDataItem(x_ci, lower_ci),
                pg.PlotDataItem(x_ci, upper_ci),
                brush=pg.mkBrush(ci_color),
            )
            self.plot_widget.addItem(fill)
            series_items["fills"].append(fill)

        measured_data = data.get("measured_y")
        if (
            self.act_measured.isChecked()
            and self.act_measured.isEnabled()
            and measured_data is not None
        ):
            try:
                meas_arr = np.array(measured_data, dtype=float)
                vp_meas = None

                if (
                    ViscosityProfile is not None
                    and len(x_full) >= 2
                    and len(meas_arr) == len(x_full)
                ):
                    try:
                        vp_meas = ViscosityProfile(
                            list(x_full.astype(float)), list(meas_arr.astype(float))
                        )
                        meas_x = dense_sr
                        meas_y = np.array([vp_meas.get_viscosity(sr) for sr in dense_sr])
                    except Exception:
                        mask = (x_full >= min_shear) & (x_full <= max_shear)
                        meas_x = x_full[mask]
                        meas_y = meas_arr[mask] if len(meas_arr) == len(x_full) else meas_arr
                else:
                    mask = (x_full >= min_shear) & (x_full <= max_shear)
                    meas_x = x_full[mask]
                    meas_y = meas_arr[mask] if len(meas_arr) == len(mask) else meas_arr

                meas_line = self.plot_widget.plot(
                    meas_x,
                    meas_y,
                    pen=pg.mkPen(main_color, width=2, style=Qt.DashLine),
                )
                series_items["lines"].append(meas_line)

                sc, tx = self._generate_scatter_points(
                    vp_meas,
                    meas_x,
                    meas_y,
                    min_shear,
                    max_shear,
                    log_x,
                    log_y,
                    main_color,
                    "measured",
                )
                series_items["scatters"].extend(sc)
                series_items["texts"].extend(tx)
            except Exception as e:
                Log.e(TAG, f"Error plotting measured data: {e}")

        is_measured_only = data.get("measured", False) and not has_ci
        if not is_measured_only:
            pred_line = self.plot_widget.plot(x, y, pen=pg.mkPen(main_color, width=3))
            series_items["lines"].append(pred_line)

            sc, tx = self._generate_scatter_points(
                vp_pred,
                x,
                y,
                min_shear,
                max_shear,
                log_x,
                log_y,
                main_color,
                "predicted",
            )
            series_items["scatters"].extend(sc)
            series_items["texts"].extend(tx)

        return series_items

    def _generate_scatter_points(
        self, vp, x, y_curve, min_shear, max_shear, log_x, log_y, color, point_type
    ):
        """Create CP-overlay scatter markers and text annotations at standard shear rates.

        Iterates over ``STANDARD_SHEAR_RATES``, keeping only those within
        ``[min_shear, max_shear]``.  For each qualifying rate, viscosity is
        read from *vp* if available; otherwise linear interpolation on
        (*x*, *y_curve*) is used as a fallback.

        Each scatter point is a 12 px white-filled circle with a coloured
        border that expands to 18 px with a filled brush on hover.  Per-point
        metadata (shear rate, data-space x/y, point type) is stored on the
        ``ScatterPlotItem`` as ``_point_data``, and hover state as
        ``_original_size``, ``_original_brush``, ``_hover_size``, and
        ``_hover_brush`` for use by ``_apply_scatter_hover`` /
        ``_reset_scatter_appearance``.

        Each text annotation is a ``TextItem`` showing the shear rate and
        viscosity, positioned 8 % above the scatter point in data coordinates.
        Visibility mirrors the current ``act_cp`` state at creation time.
        Annotations are registered in ``measured_text_annotations`` or
        ``predicted_text_annotations`` by shear rate for click-toggle access.

        Args:
            vp (ViscosityProfile | None): Interpolation object for the series;
                ``None`` falls back to ``scipy.interpolate.interp1d``.
            x (np.ndarray): Shear-rate array in the view's current coordinate
                space (possibly log-transformed by ``setLogMode``).
            y_curve (np.ndarray): Viscosity array parallel to *x*.
            min_shear (float): Lower shear-rate bound from ``spin_min_shear``.
            max_shear (float): Upper shear-rate bound from ``spin_max_shear``.
            log_x (bool): Whether the X axis is in log mode.
            log_y (bool): Whether the Y axis is in log mode.
            color (str): Hex colour string for border, hover brush, and text.
            point_type (str): ``"measured"`` or ``"predicted"``; controls
                which annotation dict and scatter list the items are appended
                to.

        Returns:
            tuple[list[pg.ScatterPlotItem], list[pg.TextItem]]: Two parallel
                lists of scatter markers and their text annotations.
        """
        if not color:
            color = "#2596be"

        scatter_items_out = []
        text_items_out = []

        target_shear_rates = []
        target_viscosities = []

        for shear_rate in self.STANDARD_SHEAR_RATES:
            if min_shear <= shear_rate <= max_shear:
                try:
                    if vp is not None:
                        # Extract directly from object without interpolating raw lines to avoid tail discontinuity
                        viscosity = float(vp.get_viscosity(shear_rate))
                    else:
                        # Fallback just in case VP fails setup
                        interp_func = interpolate.interp1d(
                            x, y_curve, kind="linear", fill_value="extrapolate"
                        )
                        viscosity = float(interp_func(shear_rate))

                    target_shear_rates.append(shear_rate)
                    target_viscosities.append(viscosity)
                except Exception:
                    continue

        if not target_shear_rates:
            return scatter_items_out, text_items_out

        show_labels = self.act_cp.isChecked()

        for shear_rate, viscosity in zip(target_shear_rates, target_viscosities):
            scatter_x = np.log10(max(shear_rate, 1e-10)) if log_x else shear_rate
            scatter_y = np.log10(max(viscosity, 1e-10)) if log_y else viscosity

            scatter = pg.ScatterPlotItem(
                x=[scatter_x],
                y=[scatter_y],
                symbol="o",
                size=12,
                brush=pg.mkBrush("w"),
                pen=pg.mkPen(color, width=2),
                hoverable=True,
                tip=None,
            )

            scatter._point_data = {
                "shear_rate": shear_rate,
                "x": shear_rate,
                "y": viscosity,
                "type": point_type,
            }
            scatter._original_size = 12
            scatter._original_brush = pg.mkBrush("w")
            scatter._hover_size = 18
            scatter._hover_brush = pg.mkBrush(color)
            scatter.setZValue(10)
            self.plot_widget.addItem(scatter)
            scatter_items_out.append(scatter)

            if point_type == "measured":
                self.measured_scatter_items.append(scatter)
            else:
                self.predicted_scatter_items.append(scatter)

            text_x = scatter_x
            offset_visc = viscosity * 1.08
            text_y = np.log10(max(offset_visc, 1e-10)) if log_y else offset_visc

            sr_label = f"{int(shear_rate):,}" if shear_rate < 1e6 else f"{shear_rate:.2e}"
            label_text = f"{sr_label} 1/s\n{viscosity:.2f} cP"

            text_item = pg.TextItem(
                label_text,
                color=color,
                anchor=(0.5, 0),
                border=pg.mkPen(color, width=1),
                fill=pg.mkBrush(255, 255, 255, 220),
            )
            text_item.setPos(text_x, text_y)
            text_item.setZValue(15)
            text_item.setVisible(show_labels)
            self.plot_widget.addItem(text_item)
            text_items_out.append(text_item)

            if point_type == "measured":
                self.measured_text_annotations[shear_rate] = text_item
            else:
                self.predicted_text_annotations[shear_rate] = text_item

        return scatter_items_out, text_items_out

    def _toggle_cp_overlay(self, checked):
        """Show or hide all CP-overlay text annotations in one pass.

        Iterates over all values in ``measured_text_annotations`` and
        ``predicted_text_annotations`` and sets each ``TextItem``'s visibility
        to *checked*.

        Args:
            checked (bool): ``True`` to show all annotations; ``False`` to
                hide them.
        """
        all_texts = list(self.measured_text_annotations.values()) + list(
            self.predicted_text_annotations.values()
        )
        for item in all_texts:
            item.setVisible(checked)

    def _set_series_items_visible(self, items_dict, visible):
        """Set the visibility of every plot item belonging to one series.

        Applies *visible* to all lines and fills unconditionally.  For text
        annotations, visibility is the logical AND of *visible* and the current
        ``act_cp`` checked state, so annotations stay hidden when the CP
        overlay is globally off even if the series itself is shown.

        Args:
            items_dict (dict): A series-items dict with keys ``"lines"``,
                ``"fills"``, ``"scatters"``, and ``"texts"``.
            visible (bool): Target visibility state.
        """

        for line in items_dict.get("lines", []):
            line.setVisible(visible)
        for fill in items_dict.get("fills", []):
            fill.setVisible(visible)
        for sc in items_dict.get("scatters", []):
            sc.setVisible(visible)

        cp_on = self.act_cp.isChecked()
        for tx in items_dict.get("texts", []):
            tx.setVisible(visible and cp_on)

    def _rebuild_standard_legend(self):
        """ "Repopulate the legend with exactly two fixed type-indicator entries.

        Clears the legend and adds:

        * **Predicted** - solid, 3 px wide, neutral grey (``#555555``).
        * **Measured** - dashed, 2 px wide, neutral grey; only added when
          ``act_measured`` is both checked and enabled.

        No per-series names are shown in the legend; series are distinguished
        solely by colour on the plot itself.
        """
        self.legend.clear()
        pred_item = pg.PlotDataItem(pen=pg.mkPen("#555555", width=3))
        self.legend.addItem(pred_item, "Predicted")
        if self.act_measured.isChecked() and self.act_measured.isEnabled():
            meas_item = pg.PlotDataItem(pen=pg.mkPen("#555555", width=2, style=Qt.DashLine))
            self.legend.addItem(meas_item, "Measured")

    def toggle_card_series(self, card_name: str) -> bool:
        """Toggle the plot visibility of the series identified by *card_name*.

        Searches ``data_series`` for the first entry whose ``"config_name"``
        matches *card_name``, flips its ``series_hidden`` flag, and calls
        ``_set_series_items_visible`` on the corresponding entry in
        ``series_plot_items``.  Called by the dashboard when a card's
        "Hide from Plot" action fires.

        Args:
            card_name (str): The ``config_name`` value of the series to toggle.

        Returns:
            bool: The new hidden state for the matched series (``True`` =
                now hidden).  Returns ``False`` if no matching series is found.
        """
        for i, data in enumerate(self.data_series):
            if data.get("config_name", "") == card_name:
                while len(self.series_hidden) <= i:
                    self.series_hidden.append(False)
                new_hidden = not self.series_hidden[i]
                self.series_hidden[i] = new_hidden
                if i < len(self.series_plot_items):
                    self._set_series_items_visible(self.series_plot_items[i], not new_hidden)
                return new_hidden
        return False

    def _toggle_series_visibility(self, series_index):
        """Toggle the visibility of the series at *series_index* and update the legend.

        Flips the ``series_hidden`` flag at *series_index*, applies the new
        state via ``_set_series_items_visible``, then calls
        ``_make_legend_clickable`` to refresh legend item styling.  Does
        nothing if *series_index* is out of range for ``series_plot_items``.

        Args:
            series_index (int): Zero-based index into ``series_plot_items``.
        """
        if series_index >= len(self.series_plot_items):
            return
        currently_hidden = (
            series_index < len(self.series_hidden) and self.series_hidden[series_index]
        )
        new_hidden = not currently_hidden

        while len(self.series_hidden) <= series_index:
            self.series_hidden.append(False)
        self.series_hidden[series_index] = new_hidden
        self._set_series_items_visible(self.series_plot_items[series_index], not new_hidden)
        self._make_legend_clickable()

    def _apply_global_limits(self):
        """Compute and apply an intelligently snapped view range from the current data.

        Returns immediately when ``data_series`` is empty.

        **X axis** - always fixed to the hardware shear-rate domain
        (100 - 15 000 000 s⁻¹) with 5 % logarithmic padding, converted to
        log10 coordinates when ``act_log_x`` is checked.

        **Y axis** - scanned across all visible series (predicted Y, CI bands
        when ``act_ci`` is checked, and measured Y when ``act_measured`` is
        checked).  Non-positive values are excluded in log mode to prevent
        ``log10(0)`` errors.  A fallback range of ``[0, 100]`` (linear) or
        ``[-1, 3]`` (log) is applied when no valid Y data is found.

        **Snapping** (applied after global min/max are known):

        * *Log Y* - snaps to the nearest integer powers of 10; expands by ±1
          decade when min equals max (flat data).
        * *Linear Y* - computes the order of magnitude of the data span, then
          refines the grid step (÷5 if span ≤ 2 steps; ÷2 if span ≤ 5 steps),
          floors/ceils to the nearest step, expands boundaries outward by one
          step when data exactly hits the edge, and clamps the lower bound to
          zero when all data is non-negative.

        Disables pyqtgraph's built-in auto-ranging before setting custom limits
        to prevent it from overriding the snapped range on subsequent
        interactions.
        """
        if not self.data_series:
            return

        log_x = self.act_log_x.isChecked()
        log_y = self.act_log_y.isChecked()

        # X Axis Boundaries
        limit_x_min = 100
        limit_x_max = 15000000

        log_min = np.log10(limit_x_min)
        log_max = np.log10(limit_x_max)
        log_range = log_max - log_min
        padding_x = log_range * 0.05
        padded_min = 10 ** (log_min - padding_x)
        padded_max = 10 ** (log_max + padding_x)
        vb_x_min = np.log10(padded_min) if log_x else padded_min
        vb_x_max = np.log10(padded_max) if log_x else padded_max

        # Hard limits for panning/zooming to prevent PyQtGraph NoneType errors
        vb_y_min = -10.0 if log_y else 0.0
        vb_y_max = 300.0 if log_y else 1e300

        # Y Axis Boundaries
        global_min_y = float("inf")
        global_max_y = float("-inf")
        found_data = False

        for data in self.data_series:
            y = np.array(data["y"])
            if len(y) == 0:
                continue

            # In log scale, we must filter out values <= 0 before getting min/max
            if log_y:
                y_valid = y[y > 0]
                if len(y_valid) > 0:
                    global_min_y = min(global_min_y, np.min(y_valid))
                    global_max_y = max(global_max_y, np.max(y_valid))
            else:
                global_min_y = min(global_min_y, np.min(y))
                global_max_y = max(global_max_y, np.max(y))

            found_data = True

            # Factor in Confidence Intervals
            if self.act_ci.isChecked() and "lower" in data:
                if log_y:
                    low_valid = np.array(data["lower"])[np.array(data["lower"]) > 0]
                    up_valid = np.array(data["upper"])[np.array(data["upper"]) > 0]
                    if len(low_valid) > 0:
                        global_min_y = min(global_min_y, np.min(low_valid))
                    if len(up_valid) > 0:
                        global_max_y = max(global_max_y, np.max(up_valid))
                else:
                    global_min_y = min(global_min_y, np.min(data["lower"]))
                    global_max_y = max(global_max_y, np.max(data["upper"]))

            # Factor in Measured Points
            if (
                self.act_measured.isChecked()
                and "measured_y" in data
                and data["measured_y"] is not None
            ):
                meas = np.array(data["measured_y"])
                if log_y:
                    meas_valid = meas[meas > 0]
                    if len(meas_valid) > 0:
                        global_min_y = min(global_min_y, np.min(meas_valid))
                        global_max_y = max(global_max_y, np.max(meas_valid))
                else:
                    global_min_y = min(global_min_y, np.min(meas))
                    global_max_y = max(global_max_y, np.max(meas))

        self.plot_widget.plotItem.vb.disableAutoRange(axis="y")
        self.plot_widget.plotItem.vb.setLimits(
            xMin=vb_x_min, xMax=vb_x_max, yMin=vb_y_min, yMax=vb_y_max
        )

        # Fallback if no valid Y data was found
        if not found_data:
            fallback_y_min = -1.0 if log_y else 0.0
            fallback_y_max = 3.0 if log_y else 100.0
            self.plot_widget.plotItem.vb.setYRange(fallback_y_min, fallback_y_max, padding=0)
            return

        # Intelligent View Range Snapping
        if log_y:
            # Snap to the nearest integer powers of 10
            y_min_target = np.floor(np.log10(global_min_y))
            y_max_target = np.ceil(np.log10(global_max_y))
            if y_max_target == y_min_target:
                y_max_target += 1
                y_min_target -= 1

            self.plot_widget.plotItem.vb.setYRange(y_min_target, y_max_target, padding=0)

        else:
            # Calculate the Order of Magnitude of the data spread to find a clean step interval
            span = global_max_y - global_min_y
            if span == 0:
                span = abs(global_max_y) if global_max_y != 0 else 1.0

            oom = 10 ** np.floor(np.log10(span))
            step = oom
            if span / step <= 2:
                step /= 5
            elif span / step <= 5:
                step /= 2

            # Snap to linear demarcations
            y_min_target = np.floor(global_min_y / step) * step
            y_max_target = np.ceil(global_max_y / step) * step
            if y_min_target == global_min_y:
                y_min_target -= step
            if y_max_target == global_max_y:
                y_max_target += step
            if global_min_y >= 0 and y_min_target < 0:
                y_min_target = 0

            self.plot_widget.plotItem.vb.setYRange(y_min_target, y_max_target, padding=0)

    def mouse_moved(self, evt):
        """Handle throttled mouse-move events from the ``SignalProxy``.

        When crosshairs are enabled (``act_crosshairs`` checked), maps the
        scene position to view coordinates, updates ``vLine`` and ``hLine``
        positions, and delegates hover detection to ``_check_scatter_hover``.
        When crosshairs are disabled, delegates directly to
        ``_check_scatter_hover_event``.  Resets hover state when the cursor
        leaves the plot bounding rect.

        Args:
            evt (tuple): Single-element tuple wrapping the ``QPointF`` scene
                position, as emitted by ``pg.SignalProxy``.
        """
        if not self.act_crosshairs.isChecked():
            self._check_scatter_hover_event(evt)
            return
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
            self._check_scatter_hover(mousePoint.x(), mousePoint.y())
        else:
            self._reset_scatter_hover()

    def _check_scatter_hover_event(self, evt):
        """Unpack a ``SignalProxy`` event tuple and forward to ``_check_scatter_hover``.

        Maps the scene position to view coordinates and calls
        ``_check_scatter_hover`` when the position is inside the plot bounding
        rect; calls ``_reset_scatter_hover`` otherwise.

        Args:
            evt (tuple): Single-element tuple wrapping the ``QPointF`` scene
                position, as emitted by ``pg.SignalProxy``.
        """
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            self._check_scatter_hover(mousePoint.x(), mousePoint.y())
        else:
            self._reset_scatter_hover()

    def _check_scatter_hover(self, mouse_x_view, mouse_y_view):
        """Detect whether the cursor is within 2 % tolerance of any scatter point.

        Computes per-axis tolerance as 2 % of the current view range, then
        iterates all items in ``measured_scatter_items`` and
        ``predicted_scatter_items``.  Point coordinates are log10-transformed
        before comparison when the corresponding axis is in log mode (clamped
        to 1e-10 to avoid domain errors).

        When the hovered item changes, ``_reset_scatter_appearance`` is called
        on the previously hovered item and ``_apply_scatter_hover`` on the
        newly hovered one.  ``hovered_scatter`` is updated to reflect the
        current state.

        Args:
            mouse_x_view (float): Cursor X coordinate in view (data) space.
            mouse_y_view (float): Cursor Y coordinate in view (data) space.
        """

        log_x = self.act_log_x.isChecked()
        log_y = self.act_log_y.isChecked()
        view_range_x = self.plot_widget.plotItem.vb.viewRange()[0]
        view_range_y = self.plot_widget.plotItem.vb.viewRange()[1]
        tolerance_x = (view_range_x[1] - view_range_x[0]) * 0.02
        tolerance_y = (view_range_y[1] - view_range_y[0]) * 0.02

        hovered_item = None
        all_items = self.measured_scatter_items + self.predicted_scatter_items
        for scatter in all_items:
            if not hasattr(scatter, "_point_data"):
                continue
            point_data = scatter._point_data
            px = np.log10(max(point_data["x"], 1e-10)) if log_x else point_data["x"]
            py = np.log10(max(point_data["y"], 1e-10)) if log_y else point_data["y"]
            if abs(mouse_x_view - px) < tolerance_x and abs(mouse_y_view - py) < tolerance_y:
                hovered_item = scatter
                break

        if hovered_item != self.hovered_scatter:
            if self.hovered_scatter:
                self._reset_scatter_appearance(self.hovered_scatter)
            if hovered_item:
                self._apply_scatter_hover(hovered_item)
            self.hovered_scatter = hovered_item

    def _reset_scatter_hover(self):
        """Clear the hover state for the currently hovered scatter point, if any.

        Calls ``_reset_scatter_appearance`` on ``hovered_scatter`` and sets it
        to ``None``.  Safe to call when no scatter is hovered.
        """
        if self.hovered_scatter:
            self._reset_scatter_appearance(self.hovered_scatter)
            self.hovered_scatter = None

    def _apply_scatter_hover(self, scatter):
        """Enlarge *scatter* and switch its brush to the hover style.

        Sets the scatter's size to ``_hover_size`` and brush to
        ``_hover_brush`` (both stored on the item at creation time), and
        changes the plot cursor to ``PointingHandCursor`` to signal
        clickability.

        Args:
            scatter (pg.ScatterPlotItem): The scatter item to apply hover
                styling to.
        """
        scatter.setSize(scatter._hover_size)
        scatter.setBrush(scatter._hover_brush)
        self.plot_widget.setCursor(Qt.PointingHandCursor)

    def _reset_scatter_appearance(self, scatter):
        """Restore *scatter* to its default (non-hover) size and brush.

        Sets the scatter's size to ``_original_size`` and brush to
        ``_original_brush``, and restores the plot cursor to
        ``ArrowCursor``.

        Args:
            scatter (pg.ScatterPlotItem): The scatter item to restore.
        """
        scatter.setSize(scatter._original_size)
        scatter.setBrush(scatter._original_brush)
        self.plot_widget.setCursor(Qt.ArrowCursor)

    def on_plot_click(self, event):
        """Toggle the text annotation for the scatter point nearest the click position.

        Ignored in parity mode (annotation toggling is handled by
        ``_on_parity_scatter_clicked``) and for non-left-button clicks.
        Maps the click scene position to view coordinates and searches
        ``measured_scatter_items`` and ``predicted_scatter_items`` for a point
        within the same 2 % per-axis tolerance used by ``_check_scatter_hover``.
        On a hit, looks up the corresponding ``TextItem`` in
        ``measured_text_annotations`` or ``predicted_text_annotations`` by
        shear rate and toggles its visibility.

        Args:
            event (pg.GraphicsScene.mouseEvents.MouseClickEvent): The click
                event emitted by ``plot_widget.scene().sigMouseClicked``.
        """
        if getattr(self, "plot_mode", "standard") == "parity":
            return

        if event.button() != Qt.LeftButton:
            return
        pos = event.scenePos()
        if not self.plot_widget.sceneBoundingRect().contains(pos):
            return

        mousePoint = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        cx, cy = mousePoint.x(), mousePoint.y()

        log_x = self.act_log_x.isChecked()
        log_y = self.act_log_y.isChecked()
        view_range_x = self.plot_widget.plotItem.vb.viewRange()[0]
        view_range_y = self.plot_widget.plotItem.vb.viewRange()[1]
        tolerance_x = (view_range_x[1] - view_range_x[0]) * 0.02
        tolerance_y = (view_range_y[1] - view_range_y[0]) * 0.02

        all_items = self.measured_scatter_items + self.predicted_scatter_items
        for scatter in all_items:
            point_data = scatter._point_data
            px = np.log10(max(point_data["x"], 1e-10)) if log_x else point_data["x"]
            py = np.log10(max(point_data["y"], 1e-10)) if log_y else point_data["y"]
            if abs(cx - px) < tolerance_x and abs(cy - py) < tolerance_y:
                sr = point_data["shear_rate"]
                target_dict = (
                    self.measured_text_annotations
                    if point_data["type"] == "measured"
                    else self.predicted_text_annotations
                )
                if sr in target_dict:
                    item = target_dict[sr]
                    item.setVisible(not item.isVisible())
                return
