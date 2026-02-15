import os

import numpy as np
import pyqtgraph as pg
from architecture import Architecture
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from scipy import interpolate
from styles.style_loader import load_stylesheet


class VisualizationPanel(QtWidgets.QWidget):
    STANDARD_SHEAR_RATES = [100, 1000, 10000, 100000, 15000000]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_data = None
        self.measured_scatter_items = []
        self.predicted_scatter_items = []
        self.measured_text_annotations = {}
        self.predicted_text_annotations = {}

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
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.graph_container = QtWidgets.QFrame()
        self.graph_container.setStyleSheet(
            "background-color: transparent; border: none;"
        )

        graph_layout = QtWidgets.QVBoxLayout(self.graph_container)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(0)

        self.plot_stack = QtWidgets.QWidget()
        self.stack_layout = QtWidgets.QGridLayout(self.plot_stack)
        self.stack_layout.setContentsMargins(0, 0, 0, 0)

        # --- THE PLOT WIDGET ---
        self.plot_widget = pg.PlotWidget(title="")
        self.plot_widget.setBackground(None)
        self.plot_widget.getPlotItem().setMenuEnabled(False)

        self.plot_widget.plotItem.showAxis("top", False)
        self.plot_widget.plotItem.showAxis("right", False)
        self.plot_widget.plotItem.showAxis("left", False)
        self.plot_widget.plotItem.showAxis("bottom", False)
        self.plot_widget.getPlotItem().setContentsMargins(0, 0, 0, 0)

        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)

        self.legend = self.plot_widget.addLegend(offset=(20, 20))
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

        # --- OVERLAY ---
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

        self.stack_layout.addWidget(self.plot_widget, 0, 0)
        self.stack_layout.addWidget(self.overlay_widget, 0, 0)

        graph_layout.addWidget(self.plot_stack)
        layout.addWidget(self.graph_container)

        # --- FLOATING CONTROLS ---
        # 1. Options Button
        self.btn_opts = self._create_floating_button(
            "icons/configure-svgrepo-com.svg", "Graph Options"
        )
        self.btn_opts.clicked.connect(self.show_options_menu)

        # 2. Home Button (Reset View)
        self.btn_home = self._create_floating_button(
            "icons/home2-svgrepo-com.svg", "Reset View"
        )
        self.btn_home.clicked.connect(self.reset_view)

        # 3. Zoom In
        self.btn_zoom_in = self._create_floating_button(
            "icons/add-plus-svgrepo-com.svg", "Zoom In"
        )
        self.btn_zoom_in.clicked.connect(self.zoom_in)

        # 4. Zoom Out
        self.btn_zoom_out = self._create_floating_button(
            "icons/minus-svgrepo-com.svgg", "Zoom Out"
        )
        self.btn_zoom_out.clicked.connect(self.zoom_out)

        self._init_controls()

    def _create_floating_button(self, icon_name, tooltip):
        """Helper to create consistent floating buttons."""
        btn = QtWidgets.QPushButton("", self.plot_widget)
        btn.setFixedSize(36, 36)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(tooltip)

        # Try to load icon, fallback to text if missing
        icon_path = os.path.join(Architecture.get_path(), icon_name)
        if os.path.exists(icon_path):
            btn.setIcon(QtGui.QIcon(icon_path))
            btn.setIconSize(QtCore.QSize(20, 20))
        else:
            # Simple text fallback if icon missing
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
        self.act_log_x = QtWidgets.QAction("Log Scale X", self, checkable=True)
        self.act_log_x.setChecked(True)
        self.act_log_x.toggled.connect(self.update_plot)

        self.act_log_y = QtWidgets.QAction("Log Scale Y", self, checkable=True)
        self.act_log_y.setChecked(False)
        self.act_log_y.toggled.connect(self.update_plot)

        self.act_axis_labels = QtWidgets.QAction(
            "Show Axis Labels", self, checkable=True
        )
        self.act_axis_labels.setChecked(True)
        self.act_axis_labels.toggled.connect(lambda: self.axis_debounce.start())

        self.act_crosshairs = QtWidgets.QAction("Show Crosshairs", self, checkable=True)
        self.act_crosshairs.setChecked(False)
        self.act_crosshairs.toggled.connect(self.toggle_crosshairs)

        self.act_ci = QtWidgets.QAction(
            "Show Confidence Interval", self, checkable=True
        )
        self.act_ci.setChecked(True)
        self.act_ci.toggled.connect(self.update_plot)

        self.act_cp = QtWidgets.QAction("Show CP Overlay", self, checkable=True)
        self.act_cp.setChecked(False)
        self.act_cp.toggled.connect(self.update_plot)

        self.act_measured = QtWidgets.QAction(
            "Show Measured Profile", self, checkable=True
        )
        self.act_measured.setChecked(True)
        self.act_measured.setEnabled(False)
        self.act_measured.toggled.connect(self.update_plot)

        self.act_smooth = QtWidgets.QAction("Smooth Curves", self, checkable=True)
        self.act_smooth.setChecked(False)
        self.act_smooth.toggled.connect(self.update_plot)

        self.spin_min_shear = QtWidgets.QDoubleSpinBox()
        self.spin_min_shear.setRange(0, 15000000)
        self.spin_min_shear.setValue(100)
        self.spin_min_shear.setDecimals(0)
        self.spin_min_shear.setSingleStep(1000)
        self.spin_min_shear.valueChanged.connect(self.update_plot)

        self.spin_max_shear = QtWidgets.QDoubleSpinBox()
        self.spin_max_shear.setRange(0, 15000000)
        self.spin_max_shear.setValue(15000000)
        self.spin_max_shear.setDecimals(0)
        self.spin_max_shear.setSingleStep(10000)
        self.spin_max_shear.valueChanged.connect(self.update_plot)

        self.btn_hypothesis = QtWidgets.QPushButton("Add Hypothesis")
        self.btn_hypothesis.clicked.connect(self.open_hypothesis_dialog)

    def toggle_crosshairs(self, checked):
        self.vLine.setVisible(checked)
        self.hLine.setVisible(checked)

    def show_options_menu(self):
        menu = QtWidgets.QMenu(self)

        # Apply consistent styling to match the theme
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
        menu.addAction(self.act_smooth)
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
        menu.addSeparator()

        hyp_action = QtWidgets.QWidgetAction(menu)
        hyp_btn_widget = QtWidgets.QWidget()
        hyp_layout = QtWidgets.QVBoxLayout(hyp_btn_widget)
        hyp_layout.setContentsMargins(10, 5, 10, 5)
        hyp_layout.addWidget(self.btn_hypothesis)
        hyp_action.setDefaultWidget(hyp_btn_widget)
        menu.addAction(hyp_action)

        menu.exec_(self.btn_opts.mapToGlobal(QtCore.QPoint(0, self.btn_opts.height())))

    def update_internal_axes(self):
        # 1. Hide all existing items in the pool
        for item in self.axis_text_pool:
            item.hide()

        if not self.act_axis_labels.isChecked():
            return

        vb = self.plot_widget.plotItem.vb
        x_range = vb.viewRange()[0]
        y_range = vb.viewRange()[1]

        log_x = self.act_log_x.isChecked()
        log_y = self.act_log_y.isChecked()

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

        y_pos_for_x_labels = y_range[0] + (y_range[1] - y_range[0]) * 0.02
        x_pos_for_y_labels = x_range[0] + (x_range[1] - x_range[0]) * 0.02

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
            label.setPos(x_val, y_pos_for_x_labels)

        for y_val in y_ticks:
            if y_val < y_range[0] or y_val > y_range[1]:
                continue

            val_display = 10**y_val if log_y else y_val
            text = f"{val_display:.1f}"

            label = get_item()
            label.setText(text)
            label.setAnchor((0, 0.5))
            label.setPos(x_pos_for_y_labels, y_val)

    def set_plot_title(self, title_text):
        if hasattr(self, "plot_widget"):
            self.plot_widget.setTitle(
                f"<span style='color: #374151; font-size: 11pt; font-weight: 600;'>{title_text}</span>"
            )

    def show_loading(self):
        self.overlay_widget.setVisible(True)
        self.progress_bar.setValue(0)
        self.anim_timer = QtCore.QTimer()
        self.anim_timer.timeout.connect(self._animate_step)
        self.anim_timer.start(10)

    def _animate_step(self):
        val = self.progress_bar.value()
        if val < 90:
            self.progress_bar.setValue(val + 1)

    def hide_loading(self):
        if hasattr(self, "anim_timer"):
            self.anim_timer.stop()
        self.overlay_widget.setVisible(False)

    def eventFilter(self, source, event):
        if source == self.plot_widget and event.type() == QtCore.QEvent.Resize:
            self._reposition_overlay_buttons()
            self.axis_debounce.start()
        return super().eventFilter(source, event)

    def _reposition_overlay_buttons(self):
        """Stacks floating buttons vertically on the right side."""
        margin_right = 20
        margin_top = 20
        spacing = 10
        btn_height = 36

        # Base X coordinate (aligned to right)
        x = self.plot_widget.width() - self.btn_opts.width() - margin_right

        # 1. Options (Top)
        y = margin_top
        self.btn_opts.move(max(0, x), y)

        # 2. Home (Below Options)
        y += btn_height + spacing
        self.btn_home.move(max(0, x), y)

        # 3. Zoom In (Below Home)
        y += btn_height + spacing
        self.btn_zoom_in.move(max(0, x), y)

        # 4. Zoom Out (Below Zoom In)
        y += btn_height + spacing
        self.btn_zoom_out.move(max(0, x), y)

    def open_hypothesis_dialog(self):
        QtWidgets.QMessageBox.information(self, "Add Hypothesis", "Placeholder")

    def zoom_in(self):
        """Zooms in by 20%."""
        self.plot_widget.plotItem.vb.scaleBy((0.8, 0.8))

    def zoom_out(self):
        """Zooms out by 25%."""
        self.plot_widget.plotItem.vb.scaleBy((1.25, 1.25))

    def reset_view(self):
        """Resets the view to fit the current data."""
        if not self.last_data:
            return

        x_full = np.array(self.last_data["x"])
        mask = (x_full >= self.spin_min_shear.value()) & (
            x_full <= self.spin_max_shear.value()
        )
        x = x_full[mask]
        y = np.array(self.last_data["y"])[mask]
        lower = np.array(self.last_data["lower"])[mask]
        upper = np.array(self.last_data["upper"])[mask]

        log_x = self.act_log_x.isChecked()
        log_y = self.act_log_y.isChecked()

        self._apply_axis_limits(log_x, log_y, y, lower, upper)
        self.axis_debounce.start()

    def set_data(self, data):
        self.last_data = data
        has_measured = "measured_y" in data and data["measured_y"] is not None
        self.act_measured.setEnabled(has_measured)
        if not has_measured:
            self.act_measured.setChecked(False)
        self.update_plot()

    def update_plot(self):
        if not self.last_data:
            return

        self.plot_widget.clear()

        # Reset items
        self.measured_scatter_items = []
        self.predicted_scatter_items = []
        self.measured_text_annotations = {}
        self.predicted_text_annotations = {}

        # Hide pool items
        for item in self.axis_text_pool:
            item.hide()
        self.axis_text_pool = []

        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)
        self.vLine.setVisible(self.act_crosshairs.isChecked())
        self.hLine.setVisible(self.act_crosshairs.isChecked())

        raw_color = self.last_data.get("color")
        main_color = raw_color if raw_color else "#2596be"

        x_full = np.array(self.last_data["x"])
        min_shear = self.spin_min_shear.value()
        max_shear = self.spin_max_shear.value()
        mask = (x_full >= min_shear) & (x_full <= max_shear)

        x_original = x_full[mask]
        y_original = np.array(self.last_data["y"])[mask]
        lower_original = np.array(self.last_data["lower"])[mask]
        upper_original = np.array(self.last_data["upper"])[mask]

        if len(x_original) == 0:
            return

        # Extend the range slightly for interpolation to prevent edge cutoff
        # Use logarithmic extension since we're dealing with log-scale shear rates
        log_min = np.log10(min_shear)
        log_max = np.log10(max_shear)
        log_range = log_max - log_min
        extension = log_range * 0.05  # 5% extension on each side in log space

        extended_min = 10 ** (log_min - extension)
        extended_max = 10 ** (log_max + extension)

        # Create extended x range for plotting
        num_points = len(x_original)
        x_extended = np.logspace(
            np.log10(extended_min), np.log10(extended_max), num_points + 10
        )

        # For smoothing, we'll work with original data then extrapolate
        x = x_original.copy()
        y = y_original.copy()
        lower = lower_original.copy()
        upper = upper_original.copy()

        # Apply smoothing if enabled - use spline interpolation through 5 standard points
        if self.act_smooth.isChecked() and len(x) > 1:
            # Get values at the 5 standard shear rates that fall within our range
            standard_points_x = []
            standard_points_y = []
            standard_points_lower = []
            standard_points_upper = []

            # Make sure x is sorted for interpolation
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]
            lower_sorted = lower[sort_idx]
            upper_sorted = upper[sort_idx]

            for shear_rate in self.STANDARD_SHEAR_RATES:
                if min_shear <= shear_rate <= max_shear:
                    # Check if shear_rate is within the data range
                    if x_sorted[0] <= shear_rate <= x_sorted[-1]:
                        # Interpolate at this shear rate
                        val_y = np.interp(shear_rate, x_sorted, y_sorted)
                        val_lower = np.interp(shear_rate, x_sorted, lower_sorted)
                        val_upper = np.interp(shear_rate, x_sorted, upper_sorted)

                        standard_points_x.append(shear_rate)
                        standard_points_y.append(val_y)
                        standard_points_lower.append(val_lower)
                        standard_points_upper.append(val_upper)

            # If we have at least 2 points, create smooth spline through them
            if len(standard_points_x) >= 2:
                # Convert to numpy arrays
                sp_x = np.array(standard_points_x)
                sp_y = np.array(standard_points_y)
                sp_lower = np.array(standard_points_lower)
                sp_upper = np.array(standard_points_upper)

                try:
                    # Create cubic splines through the standard points with extrapolation
                    if len(sp_x) == 2:
                        # For 2 points, use linear interpolation
                        spline_y = interpolate.interp1d(
                            sp_x, sp_y, kind="linear", fill_value="extrapolate"
                        )
                        spline_lower = interpolate.interp1d(
                            sp_x, sp_lower, kind="linear", fill_value="extrapolate"
                        )
                        spline_upper = interpolate.interp1d(
                            sp_x, sp_upper, kind="linear", fill_value="extrapolate"
                        )
                    else:
                        # For 3+ points, use cubic spline
                        spline_y = interpolate.CubicSpline(
                            sp_x, sp_y, bc_type="natural", extrapolate=True
                        )
                        spline_lower = interpolate.CubicSpline(
                            sp_x, sp_lower, bc_type="natural", extrapolate=True
                        )
                        spline_upper = interpolate.CubicSpline(
                            sp_x, sp_upper, bc_type="natural", extrapolate=True
                        )

                    # Evaluate the splines at extended x positions for smooth plotting
                    y_new = spline_y(x_extended)
                    lower_new = spline_lower(x_extended)
                    upper_new = spline_upper(x_extended)

                    # Only use smoothed values if they're reasonable (no NaN or inf)
                    if not (np.any(np.isnan(y_new)) or np.any(np.isinf(y_new))):
                        x = x_extended
                        y = y_new
                        lower = lower_new
                        upper = upper_new
                except Exception as e:
                    # If spline fails, fall back to original data
                    print(f"Smoothing failed: {e}")

        log_x = self.act_log_x.isChecked()
        log_y = self.act_log_y.isChecked()
        self.plot_widget.setLogMode(x=log_x, y=log_y)

        # 1. CI
        if self.act_ci.isChecked():
            if log_x or log_y:
                x_ci = np.log10(np.maximum(x, 1e-10)) if log_x else x
                lower_ci = np.log10(np.maximum(lower, 1e-10)) if log_y else lower
                upper_ci = np.log10(np.maximum(upper, 1e-10)) if log_y else upper
            else:
                x_ci = x
                lower_ci = lower
                upper_ci = upper

            ci_color = QtGui.QColor(main_color)
            ci_color.setAlpha(40)
            fill = pg.FillBetweenItem(
                pg.PlotDataItem(x_ci, lower_ci),
                pg.PlotDataItem(x_ci, upper_ci),
                brush=pg.mkBrush(ci_color),
            )
            self.plot_widget.addItem(fill)

        # 2. Measured
        measured_data = self.last_data.get("measured_y")
        if (
            self.act_measured.isChecked()
            and self.act_measured.isEnabled()
            and measured_data is not None
        ):
            meas_y_original = np.array(measured_data)[mask]
            meas_y = meas_y_original.copy()
            meas_x = x_original.copy()

            # Apply smoothing if enabled
            if self.act_smooth.isChecked() and len(meas_x) > 1:
                standard_points_x = []
                standard_points_meas_y = []

                # Get sorted data for interpolation
                sort_idx = np.argsort(meas_x)
                x_sorted = meas_x[sort_idx]
                meas_y_sorted = meas_y[sort_idx]

                for shear_rate in self.STANDARD_SHEAR_RATES:
                    if min_shear <= shear_rate <= max_shear:
                        if x_sorted[0] <= shear_rate <= x_sorted[-1]:
                            val_meas = np.interp(shear_rate, x_sorted, meas_y_sorted)
                            standard_points_x.append(shear_rate)
                            standard_points_meas_y.append(val_meas)

                if len(standard_points_x) >= 2:
                    sp_x = np.array(standard_points_x)
                    sp_meas = np.array(standard_points_meas_y)

                    try:
                        if len(sp_x) == 2:
                            spline_meas = interpolate.interp1d(
                                sp_x, sp_meas, kind="linear", fill_value="extrapolate"
                            )
                        else:
                            spline_meas = interpolate.CubicSpline(
                                sp_x, sp_meas, bc_type="natural", extrapolate=True
                            )

                        # Evaluate at extended x positions
                        meas_y_new = spline_meas(x_extended)

                        if not (
                            np.any(np.isnan(meas_y_new)) or np.any(np.isinf(meas_y_new))
                        ):
                            meas_x = x_extended
                            meas_y = meas_y_new
                    except Exception as e:
                        print(f"Measured smoothing failed: {e}")

            self.plot_widget.plot(
                meas_x,
                meas_y,
                pen=pg.mkPen(main_color, width=2, style=Qt.DashLine),
                name="Measured",
            )
            self._generate_scatter_points(
                x_original,
                meas_y_original,
                min_shear,
                max_shear,
                log_x,
                log_y,
                main_color,
                "measured",
            )

        # 3. Predicted
        self.plot_widget.plot(x, y, pen=pg.mkPen(main_color, width=3), name="Predicted")
        self._generate_scatter_points(
            x_original,
            y_original,
            min_shear,
            max_shear,
            log_x,
            log_y,
            main_color,
            "predicted",
        )

        # 4. CP Overlay - Show labels for scatter points
        if self.act_cp.isChecked():
            # Show all text annotations for scatter points
            for text_item in self.measured_text_annotations.values():
                text_item.show()
            for text_item in self.predicted_text_annotations.values():
                text_item.show()

        self._apply_axis_limits(log_x, log_y, y, lower, upper)
        self.axis_debounce.start()

    def _generate_scatter_points(
        self, x, y_curve, min_shear, max_shear, log_x, log_y, color, point_type
    ):
        if not color:
            color = "#2596be"

        target_shear_rates = []
        target_viscosities = []

        for shear_rate in self.STANDARD_SHEAR_RATES:
            if min_shear <= shear_rate <= max_shear:
                try:
                    interp_func = interpolate.interp1d(
                        x, y_curve, kind="linear", fill_value="extrapolate"
                    )
                    viscosity = float(interp_func(shear_rate))
                    target_shear_rates.append(shear_rate)
                    target_viscosities.append(viscosity)
                except Exception:
                    continue

        if not target_shear_rates:
            return

        base_qcolor = QtGui.QColor(color)

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

            if point_type == "measured":
                self.measured_scatter_items.append(scatter)
            else:
                self.predicted_scatter_items.append(scatter)

            text_x = scatter_x
            offset_visc = viscosity * 1.08
            text_y = np.log10(max(offset_visc, 1e-10)) if log_y else offset_visc

            text_item = pg.TextItem(
                f"{viscosity:.2f} cP",
                color=color,
                anchor=(0.5, 0),
                border=pg.mkPen(color, width=1),
                fill=pg.mkBrush(255, 255, 255, 220),
            )
            text_item.setPos(text_x, text_y)
            text_item.setZValue(15)
            text_item.hide()
            self.plot_widget.addItem(text_item)

            if point_type == "measured":
                self.measured_text_annotations[shear_rate] = text_item
            else:
                self.predicted_text_annotations[shear_rate] = text_item

    def _apply_axis_limits(self, log_x, log_y, y, lower, upper):
        limit_x_min = 100
        limit_x_max = 15000000

        # Add 5% padding in log space to prevent edge cutoff
        log_min = np.log10(limit_x_min)
        log_max = np.log10(limit_x_max)
        log_range = log_max - log_min
        padding = log_range * 0.05

        padded_min = 10 ** (log_min - padding)
        padded_max = 10 ** (log_max + padding)

        vb_x_min = np.log10(padded_min) if log_x else padded_min
        vb_x_max = np.log10(padded_max) if log_x else padded_max

        vb_y_min = -10.0 if log_y else 0.0
        vb_y_max = 300.0 if log_y else 1e300

        self.plot_widget.plotItem.vb.setLimits(
            xMin=vb_x_min, xMax=vb_x_max, yMin=vb_y_min, yMax=vb_y_max
        )

        if not log_y and len(y) > 0:
            y_min = np.min(y)
            y_max = np.max(y)
            if self.act_ci.isChecked() and len(lower) > 0:
                y_min = min(y_min, np.min(lower))
                y_max = max(y_max, np.max(upper))
            y_range = y_max - y_min
            if y_range > 0:
                self.plot_widget.plotItem.vb.setYRange(
                    max(0, y_min - y_range * 0.1), y_max + y_range * 0.1, padding=0
                )
            else:
                self.plot_widget.plotItem.vb.enableAutoRange(axis="y")
        else:
            self.plot_widget.plotItem.vb.enableAutoRange(axis="y")

    def mouse_moved(self, evt):
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
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            self._check_scatter_hover(mousePoint.x(), mousePoint.y())
        else:
            self._reset_scatter_hover()

    def _check_scatter_hover(self, mouse_x_view, mouse_y_view):
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
            if (
                abs(mouse_x_view - px) < tolerance_x
                and abs(mouse_y_view - py) < tolerance_y
            ):
                hovered_item = scatter
                break

        if hovered_item != self.hovered_scatter:
            if self.hovered_scatter:
                self._reset_scatter_appearance(self.hovered_scatter)
            if hovered_item:
                self._apply_scatter_hover(hovered_item)
            self.hovered_scatter = hovered_item

    def _reset_scatter_hover(self):
        if self.hovered_scatter:
            self._reset_scatter_appearance(self.hovered_scatter)
            self.hovered_scatter = None

    def _apply_scatter_hover(self, scatter):
        scatter.setSize(scatter._hover_size)
        scatter.setBrush(scatter._hover_brush)
        self.plot_widget.setCursor(Qt.PointingHandCursor)

    def _reset_scatter_appearance(self, scatter):
        scatter.setSize(scatter._original_size)
        scatter.setBrush(scatter._original_brush)
        self.plot_widget.setCursor(Qt.ArrowCursor)

    def on_plot_click(self, event):
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
