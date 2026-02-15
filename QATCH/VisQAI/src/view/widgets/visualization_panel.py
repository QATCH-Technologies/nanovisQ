import os

import numpy as np
import pyqtgraph as pg
from architecture import Architecture
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
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
        self.axis_text_items = []  # Store custom axis labels
        self.hovered_scatter = None

        self.setStyleSheet(load_stylesheet())
        pg.setConfigOptions(antialias=True)

        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Container
        self.graph_container = QtWidgets.QFrame()
        self.graph_container.setStyleSheet(
            "background-color: transparent; border: none;"
        )

        graph_layout = QtWidgets.QVBoxLayout(self.graph_container)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(0)

        # Stacked Layout
        self.plot_stack = QtWidgets.QWidget()
        self.stack_layout = QtWidgets.QGridLayout(self.plot_stack)
        self.stack_layout.setContentsMargins(0, 0, 0, 0)

        # --- THE PLOT WIDGET ---
        self.plot_widget = pg.PlotWidget(title="")
        self.plot_widget.setBackground(None)
        self.plot_widget.getPlotItem().setMenuEnabled(False)

        # 1. Remove standard borders/axes to get full bleed
        self.plot_widget.plotItem.showAxis("top", False)
        self.plot_widget.plotItem.showAxis("right", False)
        self.plot_widget.plotItem.showAxis("left", False)  # Hiding standard Left
        self.plot_widget.plotItem.showAxis("bottom", False)  # Hiding standard Bottom

        # 2. Set margins to 0 to touch edges
        self.plot_widget.getPlotItem().setContentsMargins(0, 0, 0, 0)

        # 3. Grid (Subtle)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)

        # Legend
        self.legend = self.plot_widget.addLegend(offset=(20, 20))
        self.legend.setBrush(pg.mkBrush(255, 255, 255, 150))
        self.legend.setPen(pg.mkPen(None))
        self.legend.labelTextSize = "9pt"

        self.plot_widget.installEventFilter(self)

        # Crosshairs
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

        # Connect zoom/pan events to update internal axis labels
        self.plot_widget.sigRangeChanged.connect(self.update_internal_axes)

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

        # Options Button
        self.btn_opts = QtWidgets.QPushButton("", self.plot_widget)
        self.btn_opts.setFixedSize(36, 36)
        self.btn_opts.setCursor(Qt.PointingHandCursor)
        self.btn_opts.setToolTip("Graph Options")
        icon_path = os.path.join(
            Architecture.get_path(), "icons/configure-svgrepo-com.svg"
        )
        self.btn_opts.setIcon(QtGui.QIcon(icon_path))
        self.btn_opts.setIconSize(QtCore.QSize(20, 20))
        self.btn_opts.setStyleSheet(
            """
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 18px;
                color: #555;
            }
            QPushButton:hover {
                background-color: #f9fafb;
                border-color: #d1d5db;
                color: #17C4F3;
            }
            """
        )
        shadow = QtWidgets.QGraphicsDropShadowEffect(self.btn_opts)
        shadow.setBlurRadius(8)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QtGui.QColor(0, 0, 0, 20))
        self.btn_opts.setGraphicsEffect(shadow)
        self.btn_opts.clicked.connect(self.show_options_menu)

        self._init_controls()

    def _init_controls(self):
        # 1. Scale
        self.act_log_x = QtWidgets.QAction("Log Scale X", self, checkable=True)
        self.act_log_x.setChecked(True)
        self.act_log_x.toggled.connect(self.update_plot)

        self.act_log_y = QtWidgets.QAction("Log Scale Y", self, checkable=True)
        self.act_log_y.setChecked(False)
        self.act_log_y.toggled.connect(self.update_plot)

        # 2. Visibility
        self.act_axis_labels = QtWidgets.QAction(
            "Show Axis Labels", self, checkable=True
        )
        self.act_axis_labels.setChecked(True)
        self.act_axis_labels.toggled.connect(self.update_internal_axes)

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

        # 3. Smoothing
        self.act_smooth = QtWidgets.QAction("Smooth Curves", self, checkable=True)
        self.act_smooth.setChecked(False)
        self.act_smooth.toggled.connect(self.update_plot)

        self.slider_smooth = QtWidgets.QSlider(Qt.Horizontal)
        self.slider_smooth.setRange(1, 50)
        self.slider_smooth.setValue(10)
        self.slider_smooth.setFixedWidth(120)
        self.slider_smooth.valueChanged.connect(
            lambda: self.update_plot() if self.act_smooth.isChecked() else None
        )

        # 4. Ranges
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

        # Scaling
        menu.addAction(self.act_log_x)
        menu.addAction(self.act_log_y)
        menu.addSeparator()

        # Visuals
        menu.addAction(self.act_axis_labels)  # NEW OPTION
        menu.addAction(self.act_crosshairs)
        menu.addAction(self.act_ci)
        menu.addAction(self.act_cp)
        menu.addAction(self.act_measured)
        menu.addSeparator()

        # Smoothing
        menu.addAction(self.act_smooth)
        smooth_widget = QtWidgets.QWidget()
        smooth_layout = QtWidgets.QHBoxLayout(smooth_widget)
        smooth_layout.setContentsMargins(20, 0, 20, 0)
        smooth_layout.addWidget(QtWidgets.QLabel("Strength:"))
        smooth_layout.addWidget(self.slider_smooth)
        smooth_action = QtWidgets.QWidgetAction(menu)
        smooth_action.setDefaultWidget(smooth_widget)
        menu.addAction(smooth_action)
        menu.addSeparator()

        # Ranges
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

        # Actions
        hyp_action = QtWidgets.QWidgetAction(menu)
        hyp_btn_widget = QtWidgets.QWidget()
        hyp_layout = QtWidgets.QVBoxLayout(hyp_btn_widget)
        hyp_layout.setContentsMargins(10, 5, 10, 5)
        hyp_layout.addWidget(self.btn_hypothesis)
        hyp_action.setDefaultWidget(hyp_btn_widget)
        menu.addAction(hyp_action)

        menu.exec_(self.btn_opts.mapToGlobal(QtCore.QPoint(0, self.btn_opts.height())))

    def update_internal_axes(self):
        """Draws axis labels inside the plot area."""
        # 1. Clear old labels
        for item in self.axis_text_items:
            self.plot_widget.removeItem(item)
        self.axis_text_items.clear()

        if not self.act_axis_labels.isChecked():
            return

        # 2. Get current visible range
        vb = self.plot_widget.plotItem.vb
        x_range = vb.viewRange()[0]
        y_range = vb.viewRange()[1]

        log_x = self.act_log_x.isChecked()
        log_y = self.act_log_y.isChecked()

        # 3. Helper to determine tick spacing
        def get_tick_values(min_v, max_v, is_log):
            if is_log:
                # For log scale, ticks at powers of 10
                start_exp = int(np.floor(min_v))
                end_exp = int(np.ceil(max_v))
                return [float(i) for i in range(start_exp, end_exp + 1)]
            else:
                # Linear scale ticks
                span = max_v - min_v
                if span == 0:
                    return []
                step = 10 ** int(np.floor(np.log10(span)) - 1)
                if span / step < 5:
                    step /= 2
                start = np.ceil(min_v / step) * step
                return np.arange(start, max_v, step * 2)  # *2 for cleaner look

        x_ticks = get_tick_values(x_range[0], x_range[1], log_x)
        y_ticks = get_tick_values(y_range[0], y_range[1], log_y)

        # 4. Draw X-Axis Labels (Bottom)
        # Position them slightly above the bottom edge of the view
        y_pos_for_x_labels = y_range[0] + (y_range[1] - y_range[0]) * 0.02

        for x_val in x_ticks:
            if x_val < x_range[0] or x_val > x_range[1]:
                continue

            val_display = 10**x_val if log_x else x_val
            text = (
                f"{val_display:.0e}"
                if abs(val_display) >= 1000 or abs(val_display) < 0.01
                else f"{val_display:.1f}"
            )

            # Clean up 1e+02 -> 100 style if preferred, or keep scientific
            if "e" in text:
                base, power = text.split("e")
                text = f"10^{int(power)}" if base == "1" else text

            label = pg.TextItem(text, color="#9ca3af", anchor=(0.5, 1))
            label.setPos(x_val, y_pos_for_x_labels)
            label.setFont(QtGui.QFont("Arial", 8))
            self.plot_widget.addItem(label)
            self.axis_text_items.append(label)

        # 5. Draw Y-Axis Labels (Left)
        # Position slightly to the right of the left edge
        x_pos_for_y_labels = x_range[0] + (x_range[1] - x_range[0]) * 0.02

        for y_val in y_ticks:
            if y_val < y_range[0] or y_val > y_range[1]:
                continue

            val_display = 10**y_val if log_y else y_val
            text = f"{val_display:.1f}"

            label = pg.TextItem(text, color="#9ca3af", anchor=(0, 0.5))
            label.setPos(x_pos_for_y_labels, y_val)
            label.setFont(QtGui.QFont("Arial", 8))
            self.plot_widget.addItem(label)
            self.axis_text_items.append(label)

    # ... (Rest of show_loading, hide_loading, eventFilter, set_plot_title same as before) ...

    def set_plot_title(self, title_text):
        if hasattr(self, "plot_widget"):
            # Overlay title inside top-left if possible, or keep as standard title
            # Standard title adds a margin. For full bleed, we might prefer a TextItem.
            # For now, let's keep standard title but styled minimally.
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
            self._reposition_overlay_button()
            self.update_internal_axes()  # Re-calc labels on resize
        return super().eventFilter(source, event)

    def _reposition_overlay_button(self):
        margin_right = 20
        margin_top = 20
        x = self.plot_widget.width() - self.btn_opts.width() - margin_right
        y = margin_top
        self.btn_opts.move(max(0, x), y)

    def open_hypothesis_dialog(self):
        QtWidgets.QMessageBox.information(self, "Add Hypothesis", "Placeholder")

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
        # IMPORTANT: Clearing removes axis labels, so we must add specific lists to be managed
        # But clear() removes TextItems too.
        self.axis_text_items = []

        # Re-add crosshairs
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)
        self.vLine.setVisible(self.act_crosshairs.isChecked())
        self.hLine.setVisible(self.act_crosshairs.isChecked())

        self.measured_scatter_items = []
        self.predicted_scatter_items = []
        self.measured_text_annotations = {}
        self.predicted_text_annotations = {}

        raw_color = self.last_data.get("color")
        main_color = raw_color if raw_color else "#2596be"

        x_full = np.array(self.last_data["x"])
        min_shear = self.spin_min_shear.value()
        max_shear = self.spin_max_shear.value()
        mask = (x_full >= min_shear) & (x_full <= max_shear)

        x = x_full[mask]
        y = np.array(self.last_data["y"])[mask]
        lower = np.array(self.last_data["lower"])[mask]
        upper = np.array(self.last_data["upper"])[mask]

        if len(x) == 0:
            return

        if self.act_smooth.isChecked():
            sigma = self.slider_smooth.value() / 10.0
            y = gaussian_filter1d(y, sigma)
            lower = gaussian_filter1d(lower, sigma)
            upper = gaussian_filter1d(upper, sigma)

        log_x = self.act_log_x.isChecked()
        log_y = self.act_log_y.isChecked()
        self.plot_widget.setLogMode(x=log_x, y=log_y)

        # 1. Confidence Interval
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
            meas_y = np.array(measured_data)[mask]
            if self.act_smooth.isChecked():
                sigma = self.slider_smooth.value() / 10.0
                meas_y = gaussian_filter1d(meas_y, sigma)

            self.plot_widget.plot(
                x,
                meas_y,
                pen=pg.mkPen(main_color, width=2, style=Qt.DashLine),
                name="Measured",
            )
            self._generate_scatter_points(
                x, meas_y, min_shear, max_shear, log_x, log_y, main_color, "measured"
            )

        # 3. Predicted
        self.plot_widget.plot(x, y, pen=pg.mkPen(main_color, width=3), name="Predicted")
        self._generate_scatter_points(
            x, y, min_shear, max_shear, log_x, log_y, main_color, "predicted"
        )

        # 4. CP Overlay
        if self.act_cp.isChecked():
            cp_y = y * (1 + (np.random.rand(len(y)) - 0.5) * 0.05)
            if self.act_smooth.isChecked():
                sigma = self.slider_smooth.value() / 10.0
                cp_y = gaussian_filter1d(cp_y, sigma)
            self.plot_widget.plot(
                x,
                cp_y,
                pen=None,
                symbol="t1",
                symbolSize=10,
                symbolBrush="#4caf50",
                name="CP Measure",
            )

        self._apply_axis_limits(log_x, log_y, y, lower, upper)
        self.update_internal_axes()  # Draw labels after plot update

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
        hover_qcolor = base_qcolor.lighter(150)

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
        # Limits setup...
        limit_x_min = 100
        limit_x_max = 15000000
        vb_x_min = np.log10(limit_x_min) if log_x else limit_x_min
        vb_x_max = np.log10(limit_x_max) if log_x else limit_x_max

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

    # ... (mouse_moved, _check_scatter_hover*, on_plot_click unchanged) ...
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
