import os

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from scipy import interpolate

try:
    from architecture import Architecture
    from styles.style_loader import load_stylesheet
except (ModuleNotFoundError, ImportError):
    from QATCH.common.architecture import Architecture
    from QATCH.VisQAI.src.view.styles.style_loader import load_stylesheet

try:
    from src.models.formulation import ViscosityProfile
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.formulation import ViscosityProfile


class VisualizationPanel(QtWidgets.QWidget):
    STANDARD_SHEAR_RATES = [100, 1000, 10000, 100000, 15000000]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_series = []
        self.measured_scatter_items = []
        self.predicted_scatter_items = []
        self.measured_text_annotations = {}
        self.predicted_text_annotations = {}
        self.series_plot_items = []  # list of dicts, one per series, for hide/show
        self.series_hidden = []  # bool per series

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

        # 2. Home Button (Reset View)
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

        # 3. Zoom In
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

        # 4. Zoom Out
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
        """Helper to create consistent floating buttons."""
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
        self.act_cp.setChecked(True)
        self.act_cp.toggled.connect(self._toggle_cp_overlay)

        self.act_measured = QtWidgets.QAction(
            "Show Measured Profile", self, checkable=True
        )
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
        self.vLine.setVisible(checked)
        self.hLine.setVisible(checked)

    def show_options_menu(self):
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
        for item in self.axis_text_pool:
            item.hide()

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

        # Labels positioned nicely inside view ranges
        if self.act_axis_labels.isChecked():
            x_center = (x_range[0] + x_range[1]) / 2
            x_label_text = "Log Shear Rate (1/s)" if log_x else "Shear Rate (1/s)"
            x_name = get_item()
            x_name.setText(x_label_text)
            x_name.setAnchor((0.5, 0))
            x_name.setAngle(0)
            x_name.setPos(x_center, y_range[0] + (y_range[1] - y_range[0]) * 0.05)
            x_name.setColor("#6b7280")
            x_name.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Bold))

            y_center = (y_range[0] + y_range[1]) / 2
            y_label_text = "Log Viscosity (cP)" if log_y else "Viscosity (cP)"
            y_name = get_item()
            y_name.setText(y_label_text)
            y_name.setAnchor((0.5, 1))
            y_name.setAngle(90)
            y_name.setPos(x_range[0] + (x_range[1] - x_range[0]) * 0.05, y_center)
            y_name.setColor("#6b7280")
            y_name.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Bold))

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
        QtWidgets.QApplication.processEvents()  # Flush events so GUI displays instantly

    def _animate_step(self):
        val = self.progress_bar.value()
        if val < 90:
            self.progress_bar.setValue(val + 1)

    def hide_loading(self):
        if hasattr(self, "anim_timer"):
            self.anim_timer.stop()
        self.progress_bar.setValue(100)
        QtWidgets.QApplication.processEvents()
        self.overlay_widget.setVisible(False)

    def eventFilter(self, source, event):
        if source == self.plot_widget and event.type() == QtCore.QEvent.Resize:
            self._reposition_overlay_buttons()
            self.axis_debounce.start()
        return super().eventFilter(source, event)

    def _reposition_overlay_buttons(self):
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
        self.plot_widget.plotItem.vb.scaleBy((0.8, 0.8))

    def zoom_out(self):
        self.plot_widget.plotItem.vb.scaleBy((1.25, 1.25))

    def reset_view(self):
        self._apply_global_limits()
        self.axis_debounce.start()

    def set_data(self, data):
        if isinstance(data, list):
            self.data_series = [d for d in data if d is not None]
        elif data is not None:
            self.data_series = [data]
        else:
            self.data_series = []

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

    def update_plot(self):
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

        if not self.data_series:
            return

        while len(self.series_hidden) < len(self.data_series):
            self.series_hidden.append(False)
        self.series_hidden = self.series_hidden[: len(self.data_series)]

        for i, data_package in enumerate(self.data_series):
            # Process events to allow smooth progress bar animation across series
            QtWidgets.QApplication.processEvents()
            items = self._plot_single_series(data_package, index=i)
            self.series_plot_items.append(items)

            if self.series_hidden[i]:
                self._set_series_items_visible(items, False)

        self._apply_global_limits()
        self.axis_debounce.start()
        self._make_legend_clickable()

    def _plot_single_series(self, data, index):
        series_items = {
            "lines": [],
            "fills": [],
            "scatters": [],
            "texts": [],
        }

        raw_color = data.get("color")
        default_colors = ["#2596be", "#be4d25", "#25be4d", "#be2596", "#96be25"]
        main_color = (
            raw_color if raw_color else default_colors[index % len(default_colors)]
        )
        series_name = data.get("config_name", f"Series {index + 1}")

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
            vp_pred = ViscosityProfile(
                list(x_full.astype(float)), list(y_full.astype(float))
            )
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
            print(f"ViscosityProfile interpolation failed, falling back: {e}")
            mask = (x_full >= min_shear) & (x_full <= max_shear)
            x, y = x_full[mask], y_full[mask]
            lower = (
                lower_full[mask]
                if has_ci and len(lower_full) == len(x_full)
                else np.array([])
            )
            upper = (
                upper_full[mask]
                if has_ci and len(upper_full) == len(x_full)
                else np.array([])
            )

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
                        meas_y = np.array(
                            [vp_meas.get_viscosity(sr) for sr in dense_sr]
                        )
                    except Exception:
                        mask = (x_full >= min_shear) & (x_full <= max_shear)
                        meas_x = x_full[mask]
                        meas_y = (
                            meas_arr[mask] if len(meas_arr) == len(x_full) else meas_arr
                        )
                else:
                    mask = (x_full >= min_shear) & (x_full <= max_shear)
                    meas_x = x_full[mask]
                    meas_y = meas_arr[mask] if len(meas_arr) == len(mask) else meas_arr

                meas_line = self.plot_widget.plot(
                    meas_x,
                    meas_y,
                    pen=pg.mkPen(main_color, width=2, style=Qt.DashLine),
                    name=f"{series_name} (Meas)",
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
                print(f"Error plotting measured data: {e}")

        is_measured_only = data.get("measured", False) and not has_ci
        if not is_measured_only:
            pred_line = self.plot_widget.plot(
                x, y, pen=pg.mkPen(main_color, width=3), name=series_name
            )
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

            sr_label = (
                f"{int(shear_rate):,}" if shear_rate < 1e6 else f"{shear_rate:.2e}"
            )
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
        all_texts = list(self.measured_text_annotations.values()) + list(
            self.predicted_text_annotations.values()
        )
        for item in all_texts:
            item.setVisible(checked)

    def _set_series_items_visible(self, items_dict, visible):
        for line in items_dict.get("lines", []):
            line.setVisible(visible)
        for fill in items_dict.get("fills", []):
            fill.setVisible(visible)
        for sc in items_dict.get("scatters", []):
            sc.setVisible(visible)

        cp_on = self.act_cp.isChecked()
        for tx in items_dict.get("texts", []):
            tx.setVisible(visible and cp_on)

    def _make_legend_clickable(self):
        legend = self.legend
        try:
            for i, (sample, label) in enumerate(legend.items):
                sample._series_index = i
                label._series_index = i

                def _make_handler(idx):
                    def handler(event):
                        self._toggle_series_visibility(idx)
                        event.accept()

                    return handler

                sample.mousePressEvent = _make_handler(i)
                label.mousePressEvent = _make_handler(i)

                is_hidden = i < len(self.series_hidden) and self.series_hidden[i]
                label.setAttr("color", "#9ca3af" if is_hidden else "#24292f")
        except Exception as e:
            print(f"Could not make legend clickable: {e}")

    def _toggle_series_visibility(self, series_index):
        if series_index >= len(self.series_plot_items):
            return
        currently_hidden = (
            series_index < len(self.series_hidden) and self.series_hidden[series_index]
        )
        new_hidden = not currently_hidden

        while len(self.series_hidden) <= series_index:
            self.series_hidden.append(False)
        self.series_hidden[series_index] = new_hidden
        self._set_series_items_visible(
            self.series_plot_items[series_index], not new_hidden
        )
        self._make_legend_clickable()

    def _apply_global_limits(self):
        if not self.data_series:
            return

        log_x = self.act_log_x.isChecked()
        log_y = self.act_log_y.isChecked()

        limit_x_min = 100
        limit_x_max = 15000000

        log_min = np.log10(limit_x_min)
        log_max = np.log10(limit_x_max)
        log_range = log_max - log_min
        padding = log_range * 0.05
        padded_min = 10 ** (log_min - padding)
        padded_max = 10 ** (log_max + padding)
        vb_x_min = np.log10(padded_min) if log_x else padded_min
        vb_x_max = np.log10(padded_max) if log_x else padded_max

        global_min_y = float("inf")
        global_max_y = float("-inf")

        found_data = False
        for data in self.data_series:
            y = np.array(data["y"])
            if len(y) == 0:
                continue
            found_data = True

            global_min_y = min(global_min_y, np.min(y))
            global_max_y = max(global_max_y, np.max(y))

            if self.act_ci.isChecked() and "lower" in data:
                global_min_y = min(global_min_y, np.min(data["lower"]))
                global_max_y = max(global_max_y, np.max(data["upper"]))

        vb_y_min = -10.0 if log_y else 0.0
        vb_y_max = 300.0 if log_y else 1e300

        self.plot_widget.plotItem.vb.setLimits(
            xMin=vb_x_min, xMax=vb_x_max, yMin=vb_y_min, yMax=vb_y_max
        )

        if not log_y and found_data:
            y_range = global_max_y - global_min_y
            if y_range > 0:
                self.plot_widget.plotItem.vb.setYRange(
                    max(0, global_min_y - y_range * 0.1),
                    global_max_y + y_range * 0.1,
                    padding=0,
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
