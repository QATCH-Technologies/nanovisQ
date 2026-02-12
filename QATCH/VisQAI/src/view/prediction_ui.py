import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg

# --- MATPLOTLIB IMPORTS ---
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- PYQT IMPORTS ---
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

# Configure pyqtgraph for a white background theme
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
pg.setConfigOptions(antialias=True)
try:
    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database
    from src.models.formulation import Formulation
    from src.models.predictor import Predictor

    BACKEND_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    BACKEND_AVAILABLE = False

    class Formulation:
        pass

    class Predictor:
        pass

    class IngredientController:
        def get_all_ingredients(self):
            return []

    class Database:
        def __init__(self, parse_file_key=False):
            pass

    class Log:
        @staticmethod
        def i(tag, msg):
            print(f"INFO [{tag}]: {msg}")

        @staticmethod
        def e(tag, msg):
            print(f"ERROR [{tag}]: {msg}")


TAG = "[ViscosityUI]"
# LIGHT_STYLE_SHEET = """
# /* ---------------------------------------------------------------------------
#    1. Container Styling (Seamless Integration)
#    --------------------------------------------------------------------------- */
# /* Set background to transparent so it matches the parent app's design */
# QWidget#leftPanel {
#     background-color: transparent;
#     border-right: 1px solid #e1e4e8; /* Very subtle vertical separator line */
# }

# QScrollArea {
#     background-color: transparent;
#     border: none;
# }

# QWidget#scrollContent {
#     background-color: transparent;
# }

# /* ---------------------------------------------------------------------------
#    2. Prediction Cards (Clean "Card" Look)
#    --------------------------------------------------------------------------- */
# QFrame[class="card"] {
#     background-color: #ffffff;
#     border: 1px solid #d1d5db;      /* Soft grey border to define the card */
#     border-radius: 6px;             /* Modern rounded corners */

#     /* Spacing between cards */
#     margin-top: 4px;
#     margin-bottom: 8px;
#     margin-left: 2px;
#     margin-right: 4px; /* Slight right margin so it doesn't touch the scrollbar */
# }

# /* Hover Effect: Subtle blue highlight */
# QFrame[class="card"]:hover {
#     border: 1px solid #00adee;
# }

# /* Measured State: Subtle Green */
# QFrame[class="card"][measured="true"] {
#     background-color: #f6fffa;      /* Very faint green background */
#     border: 1px solid #4caf50;
# }

# /* ---------------------------------------------------------------------------
#    3. Inner Widget Styling (Scoped)
#    --------------------------------------------------------------------------- */
# /* Scoped to avoid messing up global app styles */
# QFrame[class="card"] QLabel {
#     color: #24292f;
# }

# QFrame[class="card"] QLineEdit,
# QFrame[class="card"] QComboBox,
# QFrame[class="card"] QSpinBox,
# QFrame[class="card"] QDoubleSpinBox {
#     background-color: #ffffff;
#     border: 1px solid #d1d5db;
#     border-radius: 4px;
#     padding: 4px;
#     color: #24292f;
# }

# /* Inputs on focus */
# QFrame[class="card"] QLineEdit:focus,
# QFrame[class="card"] QComboBox:focus,
# QFrame[class="card"] QSpinBox:focus {
#     border: 1px solid #00adee;
# }

# QFrame[class="card"] QPushButton {
#     background-color: #ffffff;
#     border: 1px solid #d1d5db;
#     border-radius: 4px;
#     padding: 5px 10px;
#     font-weight: 500;
# }

# QFrame[class="card"] QPushButton:hover {
#     background-color: #f6f8fa;
#     border-color: #00adee;
#     color: #00adee;
# }

# /* Header Text */
# QLabel[class="header-title"] {
#     font-weight: 600;
#     color: #00adee;
#     font-size: 10pt;
# }

# /* Divider Line */
# QFrame[class="divider"] {
#     color: #eaecef;
# }
# """
LIGHT_STYLE_SHEET = """
/* ---------------------------------------------------------------------------
   1. Container Styling
   --------------------------------------------------------------------------- */
QWidget#leftPanel {
    background-color: transparent; 
    border-right: 1px solid #e1e4e8;
}
QScrollArea {
    background-color: transparent;
    border: none;
}
QWidget#scrollContent {
    background-color: transparent;
}

/* ---------------------------------------------------------------------------
   2. Prediction Cards
   --------------------------------------------------------------------------- */
QFrame[class="card"] {
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    margin-top: 4px;
    margin-bottom: 8px;
    margin-left: 2px;
    margin-right: 4px;
}
QFrame[class="card"]:hover {
    border: 1px solid #00adee;
}
QFrame[class="card"][measured="true"] {
    background-color: #f6fffa;
    border: 1px solid #4caf50;
}

/* ---------------------------------------------------------------------------
   3. Typography & Headers
   --------------------------------------------------------------------------- */
QFrame[class="card"] QLabel {
    color: #24292f;
}
QLabel[class="header-title"] {
    font-weight: 600;
    color: #00adee;
    font-size: 10pt;
}
QFrame[class="divider"] {
    color: #eaecef;
}

/* ---------------------------------------------------------------------------
   4. SLEEK INPUTS (Dropdowns & SpinBoxes)
   --------------------------------------------------------------------------- */
/* Common Input Styling */
QFrame[class="card"] QLineEdit, 
QFrame[class="card"] QComboBox, 
QFrame[class="card"] QDoubleSpinBox,
QFrame[class="card"] QSpinBox {
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 24px;
    color: #24292f;
    selection-background-color: #00adee;
}

/* Focus State */
QFrame[class="card"] QLineEdit:focus, 
QFrame[class="card"] QComboBox:focus, 
QFrame[class="card"] QDoubleSpinBox:focus,
QFrame[class="card"] QSpinBox:focus {
    border: 1px solid #00adee;
    background-color: #ffffff;
}

/* -- ComboBox Arrow Fix (Using PNG) -- */
QFrame[class="card"] QComboBox {
    padding-right: 20px;
}
QFrame[class="card"] QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid #d1d5db;
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
    background-color: #e9ecef; /* Visible Grey Background */
}
QFrame[class="card"] QComboBox::drop-down:hover {
    background-color: #d0e8ff;
}
/* Down Arrow (Base64 PNG) */
QFrame[class="card"] QComboBox::down-arrow {
    width: 12px;
    height: 12px;
    image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAMCAYAAABWdVznAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy89olMNAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAp0lEQVR4nJWRQQ6CMBBFH41u3Gu8BxzFA6gRDwcXYMv+H8SgWwwhxrqZJhUBw0+aNJ3fmTczifeeJVoBZFm2C/cZvSQ1SZqmW+AG9MB7wuyANbB3ku5AZY+bieOAStLDWYYT0M3gdMA5lEJSA+TAc8TcAhfz4KJACdTWS1AP1JLKuBmsigeOA7TOcPn5EKFdDa0F8oASNDb7AjhYgmIYTJZu2v23fOsDkms9tVLaAEMAAAAASUVORK5CYII=);
}


/* -- SpinBox Arrow Fix (Using PNG) -- */
QFrame[class="card"] QDoubleSpinBox,
QFrame[class="card"] QSpinBox {
    padding-right: 24px;
}

/* Up Button */
QFrame[class="card"] QDoubleSpinBox::up-button, 
QFrame[class="card"] QSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid #d1d5db;
    border-bottom: 1px solid #d1d5db;
    border-top-right-radius: 3px;
    background-color: #e9ecef; /* Visible Grey Background */
}

/* Down Button */
QFrame[class="card"] QDoubleSpinBox::down-button, 
QFrame[class="card"] QSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 22px;
    border-left: 1px solid #d1d5db;
    border-bottom-right-radius: 3px;
    background-color: #e9ecef; /* Visible Grey Background */
}

/* Hover Effects */
QFrame[class="card"] QDoubleSpinBox::up-button:hover,
QFrame[class="card"] QDoubleSpinBox::down-button:hover,
QFrame[class="card"] QSpinBox::up-button:hover,
QFrame[class="card"] QSpinBox::down-button:hover {
    background-color: #d0e8ff;
}

/* Pressed Effects */
QFrame[class="card"] QDoubleSpinBox::up-button:pressed,
QFrame[class="card"] QDoubleSpinBox::down-button:pressed,
QFrame[class="card"] QSpinBox::up-button:pressed,
QFrame[class="card"] QSpinBox::down-button:pressed {
    background-color: #a6d4fa;
}

/* Up Arrow (Base64 PNG) */
QFrame[class="card"] QDoubleSpinBox::up-arrow, 
QFrame[class="card"] QSpinBox::up-arrow {
    width: 12px;
    height: 12px;
    image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAMCAYAAABWdVznAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy89olMNAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAn0lEQVR4nJWSsQ3CMBBFnyOIUlEhxAJM4KsRJQyQjAf0rHIbsABKSQvK0QTLuhAJ//L77P++zsHMKFFVNA0svCEiAbgBpqrtPwkdcAROItL5w5B3EJENcAdWo/UEdqraTxJGlCvQZA82wHkOqQX2QJ15NXAQkdQlmNkvFK+E9k24OBSvhBZijGvgAbyAYeZCBSyBbY402YnTW1X7UPo1Pp1RL5R2IHWwAAAAAElFTkSuQmCC);
}

/* Down Arrow (Base64 PNG) */
QFrame[class="card"] QDoubleSpinBox::down-arrow, 
QFrame[class="card"] QSpinBox::down-arrow {
    width: 12px;
    height: 12px;
    image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAMCAYAAABWdVznAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy89olMNAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAp0lEQVR4nJWRQQ6CMBBFH41u3Gu8BxzFA6gRDwcXYMv+H8SgWwwhxrqZJhUBw0+aNJ3fmTczifeeJVoBZFm2C/cZvSQ1SZqmW+AG9MB7wuyANbB3ku5AZY+bieOAStLDWYYT0M3gdMA5lEJSA+TAc8TcAhfz4KJACdTWS1AP1JLKuBmsigeOA7TOcPn5EKFdDa0F8oASNDb7AjhYgmIYTJZu2v23fOsDkms9tVLaAEMAAAAASUVORK5CYII=);
}

/* ---------------------------------------------------------------------------
   5. Buttons
   --------------------------------------------------------------------------- */
QFrame[class="card"] QPushButton {
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    padding: 5px 12px;
    font-weight: 500;
}
QFrame[class="card"] QPushButton:hover {
    background-color: #f6f8fa;
    border-color: #00adee;
    color: #00adee;
}
QFrame[class="card"] QPushButton[class="primary"] {
    background-color: #f1f8ff;
    color: #00adee;
    border: 1px solid #00adee;
}
QFrame[class="card"] QPushButton[class="primary"]:hover {
    background-color: #00adee;
    color: #ffffff;
}
"""


class ModelOptionsDialog(QtWidgets.QDialog):
    def __init__(self, current_params, is_measured=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Inference Options")
        self.resize(350, 250)
        self.setModal(True)

        self.setStyleSheet(
            """
            QDialog { background-color: #ffffff; }
            QLabel { color: #333; }
            QDoubleSpinBox, QSpinBox { 
                border: 1px solid #ccc; border-radius: 4px; padding: 4px; min-height: 20px; 
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbb; background: white; height: 6px; border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #00adee; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: white; border: 1px solid #888; width: 14px; margin: -5px 0; border-radius: 7px;
            }
            /* Style for the Reset Button */
            QPushButton#btnReset {
                color: #d32f2f; background: transparent; border: none; text-align: left;
            }
            QPushButton#btnReset:hover {
                text-decoration: underline; background: #ffebee; border-radius: 4px;
            }
        """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # -- Header --
        lbl_header = QtWidgets.QLabel("Hyperparameters")
        lbl_header.setStyleSheet("font-weight: bold; font-size: 11pt; color: #00adee;")
        layout.addWidget(lbl_header)

        # -- Form Layout --
        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(10)

        # Learning Rate
        self.spin_lr = QtWidgets.QDoubleSpinBox()
        self.spin_lr.setRange(0.0001, 1.0)
        self.spin_lr.setSingleStep(0.001)
        self.spin_lr.setDecimals(4)
        self.spin_lr.setValue(current_params.get("lr", 0.01))
        self.spin_lr.setReadOnly(is_measured)
        form_layout.addRow("Learning Rate:", self.spin_lr)

        # Steps
        self.spin_steps = QtWidgets.QSpinBox()
        self.spin_steps.setRange(1, 10000)
        self.spin_steps.setSingleStep(10)
        self.spin_steps.setValue(current_params.get("steps", 50))
        self.spin_steps.setReadOnly(is_measured)
        form_layout.addRow("Inference Steps:", self.spin_steps)

        layout.addLayout(form_layout)

        # -- CI Slider Section --
        ci_group = QtWidgets.QVBoxLayout()
        ci_group.setSpacing(5)

        ci_header_layout = QtWidgets.QHBoxLayout()
        ci_header_layout.addWidget(QtWidgets.QLabel("Confidence Interval:"))
        self.lbl_ci_val = QtWidgets.QLabel(f"{current_params.get('ci', 95)}%")
        self.lbl_ci_val.setStyleSheet("font-weight: bold; color: #555;")
        self.lbl_ci_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        ci_header_layout.addWidget(self.lbl_ci_val)

        ci_group.addLayout(ci_header_layout)

        self.slider_ci = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_ci.setRange(0, 100)
        self.slider_ci.setValue(current_params.get("ci", 95))
        self.slider_ci.setEnabled(not is_measured)
        self.slider_ci.valueChanged.connect(lambda v: self.lbl_ci_val.setText(f"{v}%"))
        ci_group.addWidget(self.slider_ci)

        layout.addLayout(ci_group)
        layout.addStretch()

        # -- Footer Buttons (Reset + OK/Cancel) --
        btn_layout = QtWidgets.QHBoxLayout()

        # Reset Button (Left aligned)
        self.btn_reset = QtWidgets.QPushButton("Reset to Defaults")
        self.btn_reset.setObjectName("btnReset")
        self.btn_reset.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_reset.clicked.connect(self.reset_defaults)
        # Disable reset if card is measured/read-only
        self.btn_reset.setEnabled(not is_measured)

        btn_layout.addWidget(self.btn_reset)
        btn_layout.addStretch()

        # Standard Dialog Buttons (Right aligned)
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        btn_layout.addWidget(btn_box)

        layout.addLayout(btn_layout)

    def reset_defaults(self):
        """Resets the UI elements to default values."""
        self.spin_lr.setValue(0.01)
        self.spin_steps.setValue(50)
        self.slider_ci.setValue(95)

    def get_settings(self):
        return {
            "lr": self.spin_lr.value(),
            "steps": self.spin_steps.value(),
            "ci": self.slider_ci.value(),
        }


class DragHandle(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(20)
        self.setFixedHeight(40)  # FIX: Lock height to match the header row
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.setStyleSheet("background: transparent;")

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        painter.setBrush(QtGui.QColor("#777777"))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)

        dot_size = 4
        spacing = 4

        # Center the dots within the fixed 40px height
        start_x = (self.width() - (dot_size * 2 + spacing)) / 2
        start_y = 15

        for row in range(3):
            for col in range(2):
                x = start_x + col * (dot_size + spacing)
                y = start_y + row * (dot_size + spacing)
                painter.drawEllipse(int(x), int(y), dot_size, dot_size)


class PredictionConfigCard(QtWidgets.QFrame):
    removed = QtCore.Signal(object)
    run_requested = QtCore.Signal(dict)
    save_requested = QtCore.Signal(dict)
    expanded = QtCore.Signal(object)

    def __init__(
        self,
        default_name,
        ingredients_data,
        ingredient_types,
        ingredient_units,
        parent=None,
    ):
        super().__init__(parent)
        self.setProperty("class", "card")

        # Shadow
        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QtGui.QColor(0, 0, 0, 30))
        self.setGraphicsEffect(shadow)

        self.animation = QtCore.QPropertyAnimation(self, b"maximumHeight")
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuad)

        self.ingredients_master = ingredients_data
        self.INGREDIENT_TYPES = ingredient_types
        self.INGREDIENT_UNITS = ingredient_units

        self.active_ingredients = {}
        # Store ML Params in memory since widgets are gone
        self.ml_params = {"lr": 0.01, "steps": 50, "ci": 95}
        self.use_in_icl = True
        self.last_results = None
        self.is_expanded = True
        self.is_measured = False
        self.notes_visible = False

        self.debounce_timer = QtCore.QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(300)
        self.debounce_timer.timeout.connect(self.emit_run_request)

        self._init_ui(default_name)
        self._connect_auto_updates()

    def _init_ui(self, default_name):
        root_layout = QtWidgets.QHBoxLayout(self)
        root_layout.setContentsMargins(5, 5, 15, 5)
        root_layout.setSpacing(5)

        # 1. Left Drag Handle
        self.drag_handle = DragHandle()
        root_layout.addWidget(self.drag_handle, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        self.drag_handle.setStyleSheet("background: transparent; margin-top: 5px;")

        # 2. Central Container
        self.center_widget = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(self.center_widget)
        center_layout.setContentsMargins(0, 10, 0, 5)
        center_layout.setSpacing(10)
        root_layout.addWidget(self.center_widget, stretch=1)

        # --- Header Section ---
        self.header_frame = QtWidgets.QFrame()
        header_layout = QtWidgets.QHBoxLayout(self.header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        self.name_input = QtWidgets.QLineEdit(default_name)
        self.name_input.setPlaceholderText("Prediction Name")
        self.name_input.setProperty("class", "title-input")

        self.lbl_measured = QtWidgets.QLabel("✓ Measured Data")
        self.lbl_measured.setProperty("class", "badge-success")
        self.lbl_measured.setVisible(False)

        # Delete Button
        self.btn_delete = QtWidgets.QPushButton()
        self.btn_delete.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TrashIcon)
        )
        self.btn_delete.setFixedWidth(30)
        self.btn_delete.setFlat(True)
        self.btn_delete.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_delete.setToolTip("Delete Prediction")
        self.btn_delete.setStyleSheet(
            """
            QPushButton { border: none; background: transparent; padding: 4px; border-radius: 4px; }
            QPushButton:hover { background: #ffebee; border: 1px solid #ffcdd2; }
        """
        )
        self.btn_delete.clicked.connect(lambda: self.removed.emit(self))

        # Hamburger Menu
        self.btn_options = QtWidgets.QPushButton("☰")
        self.btn_options.setFixedWidth(30)
        self.btn_options.setFlat(True)
        self.btn_options.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_options.setStyleSheet(
            """
            QPushButton::menu-indicator { image: none; }
            QPushButton { 
                border: none; background: transparent; color: #555; 
                font-size: 16px; font-weight: bold; padding-bottom: 3px; border-radius: 4px;
            }
            QPushButton:hover { color: #00adee; background: #e3f2fd; }
        """
        )

        self.options_menu = QtWidgets.QMenu(self)
        self.options_menu.addAction("Save model")
        self.options_menu.addAction("Save model as...")
        self.options_menu.addSeparator()
        self.options_menu.addAction("Export Formulation").triggered.connect(
            self.export_formulation
        )
        self.act_clear = self.options_menu.addAction("Clear Formulation")
        self.act_clear.triggered.connect(self.clear_formulation)
        self._update_clear_state()
        self.options_menu.addSeparator()
        self.act_use_icl = self.options_menu.addAction("Use in ICL")
        self.act_use_icl.setCheckable(True)
        self.act_use_icl.setEnabled(False)
        self.act_use_icl.toggled.connect(self.set_icl_usage)

        # -- Model Options Action --
        self.act_model_opts = self.options_menu.addAction("Model Options")
        self.act_model_opts.triggered.connect(self.open_model_options)

        self.btn_options.setMenu(self.options_menu)

        header_layout.addWidget(self.name_input, stretch=1)
        header_layout.addWidget(self.lbl_measured)
        header_layout.addWidget(self.btn_delete)
        header_layout.addWidget(self.btn_options)

        center_layout.addWidget(self.header_frame)

        # --- Content Body ---
        self.content_frame = QtWidgets.QFrame()
        self.content_frame.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_LayoutUsesWidgetRect
        )
        content_layout = QtWidgets.QVBoxLayout(self.content_frame)
        content_layout.setContentsMargins(5, 0, 5, 0)
        content_layout.setSpacing(15)

        # Model Selection
        model_layout = QtWidgets.QHBoxLayout()
        model_label = QtWidgets.QLabel("Model:")
        model_label.setStyleSheet("font-weight: bold; color: #555;")
        self.model_display = QtWidgets.QLineEdit("VisQAI-ICL_v1_nightly")
        self.model_display.setReadOnly(True)
        self.model_display.setPlaceholderText("No model selected")
        self.model_display.setProperty("class", "sleek")
        self.btn_select_model = QtWidgets.QPushButton("...")
        self.btn_select_model.setFixedWidth(40)
        self.btn_select_model.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_select_model.setToolTip("Select Model")
        self.btn_select_model.clicked.connect(self.browse_model_file)

        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_display, stretch=1)
        model_layout.addWidget(self.btn_select_model)

        content_layout.addLayout(model_layout)
        # -------------------------------

        self._add_divider(content_layout)

        # Ingredients
        self._add_header_with_help(
            content_layout,
            "Formulation Composition",
            "Define the components in your formulation (i.e. Protein, Buffer, Surfactant, Stabilizer, Salt, Excipient).",
        )
        self.ing_container_layout = QtWidgets.QVBoxLayout()
        self.ing_container_layout.setSpacing(5)
        content_layout.addLayout(self.ing_container_layout)

        self.btn_add_ing = QtWidgets.QPushButton("+ Add Component")
        self.btn_add_ing.setProperty("class", "primary")
        self.btn_add_ing.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_add_ing.clicked.connect(self.show_add_menu)
        content_layout.addWidget(self.btn_add_ing)

        self._add_divider(content_layout)

        # Environment
        self._add_header_with_help(
            content_layout,
            "Environment",
            "Set environmental conditions (i.e. Temperature in C)",
        )
        temp_layout = QtWidgets.QHBoxLayout()
        temp_lbl = QtWidgets.QLabel("Temperature:")
        temp_lbl.setFixedWidth(100)
        temp_layout.addWidget(temp_lbl)

        self.slider_temp = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_temp.setRange(0, 100)
        self.slider_temp.setValue(25)
        self.slider_temp.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        self.spin_temp = QtWidgets.QDoubleSpinBox()
        self.spin_temp.setProperty("class", "sleek")
        self.spin_temp.setRange(0.0, 100.0)
        self.spin_temp.setValue(25.0)
        self.spin_temp.setSuffix(" °C")
        self.spin_temp.setFixedWidth(90)
        self.spin_temp.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.slider_temp.valueChanged.connect(
            lambda v: self.spin_temp.setValue(float(v))
        )
        self.spin_temp.valueChanged.connect(lambda v: self.slider_temp.setValue(int(v)))

        temp_layout.addWidget(self.slider_temp)
        temp_layout.addWidget(self.spin_temp)
        content_layout.addLayout(temp_layout)

        self._add_divider(content_layout)

        lbl_notes = QtWidgets.QLabel("Notes:")
        lbl_notes.setStyleSheet("font-weight: bold; color: #555;")
        content_layout.addWidget(lbl_notes)

        # 2. Add the text edit (Always visible now)
        self.notes_edit = QtWidgets.QTextEdit()
        self.notes_edit.setPlaceholderText("Enter notes about this run...")
        self.notes_edit.setMaximumHeight(80)  # Fixed height since it doesn't animate
        self.notes_edit.setStyleSheet(
            "border: 1px solid #d1d5db; background: #fcfcfc; border-radius: 4px;"
        )

        content_layout.addWidget(self.notes_edit)

        center_layout.addWidget(self.content_frame)

        # Footer
        self.footer_frame = QtWidgets.QFrame()
        footer_layout = QtWidgets.QHBoxLayout(self.footer_frame)
        footer_layout.setContentsMargins(0, 5, 0, 0)
        footer_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.btn_toggle = QtWidgets.QToolButton()
        self.btn_toggle.setArrowType(QtCore.Qt.ArrowType.UpArrow)
        self.btn_toggle.clicked.connect(self.toggle_content)
        self.btn_toggle.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_toggle.setStyleSheet("border: none; background: transparent;")

        footer_layout.addWidget(self.btn_toggle)
        center_layout.addWidget(self.footer_frame)

    def _update_clear_state(self):
        """Disables Clear action if data is imported OR if no ingredients exist."""
        # Safety check in case UI isn't fully init yet
        if not hasattr(self, "act_clear"):
            return

        # Condition 1: Imported Data -> Always Disabled
        if self.is_measured:
            self.act_clear.setEnabled(False)
            return

        # Condition 2: Empty Formulation -> Disabled
        has_ingredients = len(self.active_ingredients) > 0
        self.act_clear.setEnabled(has_ingredients)

    def open_model_options(self):
        dlg = ModelOptionsDialog(self.ml_params, self.is_measured, self)
        if dlg.exec_() == QtWidgets.QDialog.DialogCode.Accepted:
            # Update internal params
            new_settings = dlg.get_settings()
            self.ml_params.update(new_settings)
            # Trigger re-run
            self.trigger_update()

    def _connect_auto_updates(self):
        self.name_input.textChanged.connect(self.trigger_update)
        self.model_display.textChanged.connect(self.trigger_update)
        self.spin_temp.valueChanged.connect(self.trigger_update)

    def set_icl_usage(self, checked):
        self.use_in_icl = checked
        self.trigger_update()

    def _add_divider(self, layout):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        line.setLineWidth(1)
        line.setProperty("class", "divider")
        layout.addWidget(line)

    def _add_header_with_help(self, layout, title, help_text):
        container = QtWidgets.QWidget()
        h_layout = QtWidgets.QHBoxLayout(container)
        h_layout.setContentsMargins(0, 5, 0, 5)
        h_layout.setSpacing(8)
        lbl = QtWidgets.QLabel(title)
        lbl.setProperty("class", "header-title")
        btn_help = QtWidgets.QToolButton()
        btn_help.setText("?")
        btn_help.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn_help.setProperty("class", "help-btn")
        btn_help.clicked.connect(
            lambda: QtWidgets.QMessageBox.information(self, title, help_text)
        )
        h_layout.addWidget(lbl)
        h_layout.addWidget(btn_help)
        h_layout.addStretch()
        layout.addWidget(container)

    def set_measured(self, is_measured: bool):
        self.set_measured_state(is_measured)

    def set_measured_state(self, is_measured: bool):
        self.is_measured = is_measured
        self.lbl_measured.setVisible(is_measured)
        self.setProperty("measured", is_measured)
        self.style().unpolish(self)
        self.style().polish(self)

        lock_state = is_measured

        self.name_input.setReadOnly(lock_state)
        self.btn_add_ing.setVisible(not lock_state)

        for combo, spin in self.active_ingredients.values():
            combo.setEnabled(not lock_state)
            spin.setReadOnly(lock_state)
        self.slider_temp.setEnabled(not lock_state)
        self.spin_temp.setReadOnly(lock_state)
        self.btn_select_model.setEnabled(not lock_state)
        self.notes_edit.setReadOnly(lock_state)

    def toggle_content(self):
        if not self.is_expanded:
            self.expanded.emit(self)
            self.emit_run_request()

        self.is_expanded = not self.is_expanded
        arrow = (
            QtCore.Qt.ArrowType.UpArrow
            if self.is_expanded
            else QtCore.Qt.ArrowType.DownArrow
        )
        self.btn_toggle.setArrowType(arrow)

        if not hasattr(self, "_anim_accordion"):
            self._anim_accordion = QtCore.QPropertyAnimation(
                self.content_frame, b"maximumHeight"
            )
            self._anim_accordion.setDuration(200)
            self._anim_accordion.setEasingCurve(QtCore.QEasingCurve.InOutQuad)

        try:
            self._anim_accordion.finished.disconnect()
        except TypeError:
            pass

        if self.is_expanded:
            self.content_frame.setVisible(True)
            self.content_frame.setMaximumHeight(16777215)
            target_height = self.content_frame.sizeHint().height()
            self._anim_accordion.setStartValue(0)
            self._anim_accordion.setEndValue(target_height)
            self._anim_accordion.finished.connect(
                lambda: self.content_frame.setMaximumHeight(16777215)
            )
        else:
            self._anim_accordion.setStartValue(self.content_frame.height())
            self._anim_accordion.setEndValue(0)
            self._anim_accordion.finished.connect(
                lambda: self.content_frame.setVisible(False)
            )
        self._anim_accordion.start()

    def collapse(self):
        if not self.is_expanded:
            return
        self.is_expanded = False
        self.btn_toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow)

        if not hasattr(self, "_anim_accordion"):
            self._anim_accordion = QtCore.QPropertyAnimation(
                self.content_frame, b"maximumHeight"
            )
            self._anim_accordion.setDuration(300)
            self._anim_accordion.setEasingCurve(QtCore.QEasingCurve.InOutQuad)

        try:
            self._anim_accordion.finished.disconnect()
        except TypeError:
            pass
        self._anim_accordion.setStartValue(self.content_frame.height())
        self._anim_accordion.setEndValue(0)
        self._anim_accordion.finished.connect(
            lambda: self.content_frame.setVisible(False)
        )
        self._anim_accordion.start()

    def set_results(self, data):
        """Stores the simulation/measured data for export."""
        self.last_results = data

    def browse_model_file(self):
        """Opens file dialog to select a model."""
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "VisQAI Models (*.visq)",
        )
        if fname:
            file_info = QtCore.QFileInfo(fname)
            display_name = file_info.fileName()

            # Update the display
            self.model_display.setText(display_name)

            # (Optional) Store full path if you need it later
            self.selected_model_path = fname

            # Trigger run
            self.trigger_update()

    def show_add_menu(self):
        menu = QtWidgets.QMenu(self.btn_add_ing)
        menu.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        present_types = list(self.active_ingredients.keys())
        available = [t for t in self.INGREDIENT_TYPES if t not in present_types]
        if not available:
            action = menu.addAction("All types added")
            action.setEnabled(False)
        else:
            for t in available:
                action = menu.addAction(t)
                action.triggered.connect(
                    lambda checked, type_name=t: self.add_ingredient_row(type_name)
                )
        pos = self.btn_add_ing.mapToGlobal(QtCore.QPoint(0, self.btn_add_ing.height()))
        menu.exec(pos)

    def add_ingredient_row(self, ing_type):
        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)  # Add slight spacing between elements

        # Label
        lbl = QtWidgets.QLabel(f"{ing_type}:")
        lbl.setFixedWidth(80)
        lbl.setStyleSheet("color: #555; font-weight: 500;")

        # ComboBox
        combo = QtWidgets.QComboBox()
        combo.setProperty("class", "sleek")
        items = self.ingredients_master.get(ing_type, [])
        combo.addItem("None", None)
        for item in items:
            combo.addItem(item.name, item)

        # SpinBox
        spin = QtWidgets.QDoubleSpinBox()
        combo.setProperty("class", "sleek")
        spin.setRange(0, 1000)
        spin.setSingleStep(1.0)
        spin.setSuffix(f" {self.INGREDIENT_UNITS.get(ing_type,'')}")
        spin.setFixedWidth(90)

        # Connect Signals
        combo.currentTextChanged.connect(self.trigger_update)
        spin.valueChanged.connect(self.trigger_update)

        # --- IMPROVED REMOVE BUTTON ---
        btn_rem = QtWidgets.QPushButton()
        btn_rem.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TrashIcon)
        )
        btn_rem.setFixedSize(24, 24)  # Small, square button
        btn_rem.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn_rem.setToolTip("Remove Component")

        # Style: Transparent gray normally, Red background on hover
        btn_rem.setStyleSheet(
            """
            QPushButton {
                border: none;
                background-color: transparent;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #ffebee; /* Light Red */
                border: 1px solid #ffcdd2;
            }
        """
        )

        btn_rem.clicked.connect(
            lambda: self.remove_ingredient_row(ing_type, row_widget)
        )
        # ------------------------------

        row_layout.addWidget(lbl)
        row_layout.addWidget(combo, stretch=1)
        row_layout.addWidget(spin)
        row_layout.addWidget(btn_rem)

        self.ing_container_layout.addWidget(row_widget)
        self.active_ingredients[ing_type] = (combo, spin)

        # Update Clear State since we added an item
        self._update_clear_state()

    def trigger_update(self):
        if self.is_expanded:
            self.debounce_timer.start()

    def emit_run_request(self):
        config = self.get_configuration()
        self.run_requested.emit(config)

    def remove_ingredient_row(self, ing_type, widget):
        if ing_type in self.active_ingredients:
            del self.active_ingredients[ing_type]
        self._update_clear_state()
        self.trigger_update()
        if widget:
            anim = QtCore.QPropertyAnimation(widget, b"maximumHeight", widget)
            anim.setDuration(100)
            anim.setEasingCurve(QtCore.QEasingCurve.InQuad)
            anim.setStartValue(widget.height())
            anim.setEndValue(0)
            anim.finished.connect(widget.deleteLater)
            anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def get_configuration(self):
        formulation = {}
        for t, (combo, spin) in self.active_ingredients.items():
            formulation[t] = {
                "component": combo.currentText(),
                "concentration": spin.value(),
            }

        config = {
            "name": self.name_input.text(),
            "model": self.model_display.text(),
            "temp": self.spin_temp.value(),
            "formulation": formulation,
            "measured": self.is_measured,
            "notes": self.notes_edit.toPlainText(),
            "use_in_icl": self.use_in_icl,
        }
        config.update(self.ml_params)
        return config

    def load_data(self, data):
        if "name" in data:
            self.name_input.setText(data["name"])
        if "model" in data:
            self.model_display.setText(data["model"])
        if "temp" in data:
            self.spin_temp.setValue(float(data["temp"]))

        # Update ML Params storage directly
        if "lr" in data:
            self.ml_params["lr"] = float(data["lr"])
        if "steps" in data:
            self.ml_params["steps"] = int(data["steps"])
        if "ci" in data:
            self.ml_params["ci"] = int(data["ci"])
        if "use_in_icl" in data:
            self.use_in_icl = data["use_in_icl"]
            print(self.use_in_icl)
        if "measured" in data:
            self.set_measured(data["measured"])
        if "notes" in data:
            self.notes_edit.setText(data["notes"])
        if "formulation" in data:
            for ing_type, details in data["formulation"].items():
                self.add_ingredient_row(ing_type)
                if ing_type in self.active_ingredients:
                    combo, spin = self.active_ingredients[ing_type]
                    spin.setValue(float(details.get("concentration", 0.0)))
                    comp_name = details.get("component")
                    if comp_name:
                        idx = combo.findText(comp_name)
                        if idx >= 0:
                            combo.setCurrentIndex(idx)

    def clear_formulation(self):
        """Clears all ingredients and resets environment to default."""
        if self.is_measured:
            QtWidgets.QMessageBox.warning(
                self, "Action Denied", "Cannot clear imported data."
            )
            return
        for ing_type in list(self.active_ingredients.keys()):
            combo, _ = self.active_ingredients[ing_type]
            row_widget = combo.parentWidget()

            # Use existing remove logic
            self.remove_ingredient_row(ing_type, row_widget)
        self.slider_temp.setValue(25)
        self.spin_temp.setValue(25.0)

        self.trigger_update()

    def export_formulation(self):
        """Exports the formulation and results to a CSV file."""
        import csv

        # 1. Validation
        if not self.last_results:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No data available to export.\nPlease run a prediction or import data first.",
            )
            return

        # 2. Get Filename
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Formulation",
            f"{self.name_input.text()}.csv",
            "CSV Files (*.csv)",
        )
        if not fname:
            return

        try:
            with open(fname, "w", newline="") as f:
                writer = csv.writer(f)

                # --- HEADER / METADATA ---
                writer.writerow(["--- Metadata ---"])
                writer.writerow(["Name", self.name_input.text()])
                writer.writerow(["Model", self.model_combo.currentText()])
                writer.writerow(["Temperature (C)", self.spin_temp.value()])
                writer.writerow(
                    ["Confidence Interval (%)", self.ml_params.get("ci", 95)]
                )
                writer.writerow(
                    [
                        "Date",
                        QtCore.QDateTime.currentDateTime().toString(
                            QtCore.Qt.DateFormat.ISODate
                        ),
                    ]
                )
                writer.writerow(["Notes", self.notes_edit.toPlainText()])
                writer.writerow([])  # Spacer

                # --- FORMULATION ---
                writer.writerow(["--- Formulation Composition ---"])
                writer.writerow(["Type", "Component", "Concentration", "Unit"])

                for ing_type, (combo, spin) in self.active_ingredients.items():
                    unit = self.INGREDIENT_UNITS.get(ing_type, "")
                    writer.writerow(
                        [ing_type, combo.currentText(), f"{spin.value():.2f}", unit]
                    )
                writer.writerow([])  # Spacer

                # --- VISCOSITY PROFILE ---
                writer.writerow(["--- Viscosity Profile ---"])

                # Retrieve Data arrays
                xs = self.last_results.get("x", [])

                # Columns depend on state (Measured vs Predicted)
                if self.is_measured:
                    # Case: Imported/Measured Data
                    writer.writerow(
                        [
                            "Shear Rate (1/s)",
                            "Measured Viscosity (cP)",
                            "Lower CI (cP)",
                            "Upper CI (cP)",
                        ]
                    )

                    meas_y = self.last_results.get("measured_y", [])
                    lower = self.last_results.get("lower", [])
                    upper = self.last_results.get("upper", [])

                    # Handle missing CI in pure imports if necessary
                    if len(lower) != len(xs):
                        lower = [0] * len(xs)
                    if len(upper) != len(xs):
                        upper = [0] * len(xs)
                    if meas_y is None:
                        meas_y = [0] * len(xs)

                    for x, m, l, u in zip(xs, meas_y, lower, upper):
                        writer.writerow(
                            [f"{x:.4f}", f"{m:.4f}", f"{l:.4f}", f"{u:.4f}"]
                        )

                else:
                    # Case: Prediction
                    writer.writerow(
                        [
                            "Shear Rate (1/s)",
                            "Predicted Viscosity (cP)",
                            "Lower CI (cP)",
                            "Upper CI (cP)",
                        ]
                    )

                    pred_y = self.last_results.get("y", [])
                    lower = self.last_results.get("lower", [])
                    upper = self.last_results.get("upper", [])

                    for x, y, l, u in zip(xs, pred_y, lower, upper):
                        writer.writerow(
                            [f"{x:.4f}", f"{y:.4f}", f"{l:.4f}", f"{u:.4f}"]
                        )

            QtWidgets.QMessageBox.information(
                self, "Export Successful", f"Successfully exported to:\n{fname}"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Export Error", f"Failed to export file:\n{str(e)}"
            )


class VisualizationPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_data = None
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Splitter
        viz_splitter = QtWidgets.QSplitter(Qt.Vertical)

        # --- Graph Area Container ---
        self.graph_container = QtWidgets.QWidget()
        graph_layout = QtWidgets.QVBoxLayout(self.graph_container)
        graph_layout.setContentsMargins(0, 0, 0, 0)

        # 1. Hover Label (Top)
        self.hover_label = QtWidgets.QLabel("Hover over graph")
        self.hover_label.setStyleSheet("font-weight: bold; color: #333; padding: 2px;")
        self.hover_label.setAlignment(Qt.AlignCenter)
        graph_layout.addWidget(self.hover_label)

        # 2. Stacked Plot Area (Plot + Overlay)
        self.plot_stack = QtWidgets.QWidget()
        self.stack_layout = QtWidgets.QGridLayout(self.plot_stack)
        self.stack_layout.setContentsMargins(0, 0, 0, 0)

        # Layer A: The Plot
        self.plot_widget = pg.PlotWidget(title="Viscosity Profile")
        self.plot_widget.setLabel("left", "Viscosity", units="cP")
        self.plot_widget.setLabel("bottom", "Shear Rate", units="1/s")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()

        # [Re-add plot event filters and crosshairs here from previous code...]
        self.plot_widget.installEventFilter(self)
        self.vLine = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("k", style=Qt.DashLine)
        )
        self.hLine = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen("k", style=Qt.DashLine)
        )
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)
        self.proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved
        )

        # [MODIFIED SECTION: Overlay & Progress Bar]
        self.overlay_widget = QtWidgets.QFrame()
        self.overlay_widget.setStyleSheet("background-color: rgba(255, 255, 255, 180);")
        self.overlay_widget.setVisible(False)

        overlay_layout = QtWidgets.QVBoxLayout(self.overlay_widget)
        overlay_layout.setAlignment(Qt.AlignCenter)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setFixedWidth(300)
        self.progress_bar.setFixedHeight(25)

        # [NEW] Enable Text
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setFormat("Running Inference... %p%")  # T
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #999;
                border-radius: 4px;
                background-color: #fff;
                color: #333; /* Text color */
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #0078D7;
                border-radius: 3px;
            }
        """
        )
        overlay_layout.addWidget(self.progress_bar)

        # Add both to the same grid cell (0,0) so they stack
        self.stack_layout.addWidget(self.plot_widget, 0, 0)
        self.stack_layout.addWidget(self.overlay_widget, 0, 0)

        graph_layout.addWidget(self.plot_stack)
        viz_splitter.addWidget(self.graph_container)

        # --- Table Area ---
        self.results_table = QtWidgets.QTableWidget()
        self.results_table.setColumnCount(4)

        viz_splitter.addWidget(self.results_table)
        viz_splitter.setStretchFactor(0, 3)
        viz_splitter.setStretchFactor(1, 1)

        layout.addWidget(viz_splitter)

        # [Re-add Option Button logic]
        self.btn_opts = QtWidgets.QPushButton("⚙ Graph Options", self.plot_widget)
        self.btn_opts.setCursor(Qt.ArrowCursor)
        self.btn_opts.clicked.connect(self.show_options_menu)
        self._init_controls()

    def _init_controls(self):
        """Initialize actions and widgets for the options menu."""
        # Toggles
        self.act_log_x = QtWidgets.QAction("Log Scale X", self, checkable=True)
        self.act_log_x.setChecked(True)
        self.act_log_x.toggled.connect(self.update_plot)

        self.act_log_y = QtWidgets.QAction("Log Scale Y", self, checkable=True)
        self.act_log_y.setChecked(False)
        self.act_log_y.toggled.connect(self.update_plot)

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

        # Range Selector (Min / Max SpinBoxes)
        # Using SpinBoxes allows variable ranges beyond just fixed combo items
        self.spin_min_shear = QtWidgets.QDoubleSpinBox()
        self.spin_min_shear.setRange(0, 15000000)
        self.spin_min_shear.setValue(100)
        self.spin_min_shear.setGroupSeparatorShown(True)
        self.spin_min_shear.setDecimals(0)
        self.spin_min_shear.setSingleStep(1000)
        self.spin_min_shear.valueChanged.connect(self.update_plot)

        self.spin_max_shear = QtWidgets.QDoubleSpinBox()
        self.spin_max_shear.setRange(0, 15000000)
        self.spin_max_shear.setValue(15000000)
        self.spin_max_shear.setGroupSeparatorShown(True)
        self.spin_max_shear.setDecimals(0)
        self.spin_max_shear.setSingleStep(10000)
        self.spin_max_shear.valueChanged.connect(self.update_plot)

        # Hypothesis Button
        self.btn_hypothesis = QtWidgets.QPushButton("Add Hypothesis")
        self.btn_hypothesis.clicked.connect(self.open_hypothesis_dialog)

    def set_plot_title(self, title_text):
        """Updates the title of the plot widget."""
        # Check if plot_widget exists to avoid errors during init
        if hasattr(self, "plot_widget"):
            self.plot_widget.setTitle(f"Viscosity Profile: {title_text}")

    def update_table(self):
        if not self.last_data:
            return

        x, y = self.last_data["x"], self.last_data["y"]
        l, u = self.last_data["lower"], self.last_data["upper"]

        # Safe retrieval of measured data
        meas = self.last_data.get("measured_y")
        if meas is None:
            meas = [None] * len(x)

        self.results_table.setRowCount(len(x))

        for i, (sr, visc, low, high, m_visc) in enumerate(zip(x, y, l, u, meas)):
            # Format Shear Rate
            sr_txt = f"{sr:.2e}" if sr >= 10000 else f"{sr:.1f}"
            self.results_table.setItem(i, 0, QtWidgets.QTableWidgetItem(sr_txt))

            # Format Predicted Viscosity
            self.results_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{visc:.2f}"))

            # Format Interval
            self.results_table.setItem(
                i, 2, QtWidgets.QTableWidgetItem(f"{low:.2f} - {high:.2f}")
            )

            # Format Measured
            if m_visc is not None:
                # Handle potential numpy scalars or strings
                try:
                    val = float(m_visc)
                    m_text = f"{val:.2f}"
                except (ValueError, TypeError):
                    m_text = "-"
            else:
                m_text = "-"

            self.results_table.setItem(i, 3, QtWidgets.QTableWidgetItem(m_text))

    def show_loading(self):
        self.overlay_widget.setVisible(True)
        self.progress_bar.setValue(0)

        self.anim_timer = QtCore.QTimer()
        self.anim_timer.timeout.connect(self._animate_step)
        self.anim_timer.start(10)

    def _animate_step(self):
        """Increments progress bar."""
        val = self.progress_bar.value()
        if val < 90:  # Cap at 90% until process actually finishes
            self.progress_bar.setValue(val + 1)

    def hide_loading(self):
        """Stops animation and hides overlay."""
        if hasattr(self, "anim_timer"):
            self.anim_timer.stop()
        self.overlay_widget.setVisible(False)

    def eventFilter(self, source, event):
        """Handle resize events to keep the overlay button in the top right."""
        if source == self.plot_widget and event.type() == QtCore.QEvent.Resize:
            self._reposition_overlay_button()
        return super().eventFilter(source, event)

    def _reposition_overlay_button(self):
        """Moves button to Top-Right with a defined margin to prevent overrun."""
        margin_right = 25  # Offset from the right edge
        margin_top = 10  # Offset from the top edge

        # Calculate x position
        x = self.plot_widget.width() - self.btn_opts.width() - margin_right
        y = margin_top

        # Ensure x doesn't go negative if the window is extremely small
        if x < 0:
            x = 0

        self.btn_opts.move(x, y)

    def show_options_menu(self):
        menu = QtWidgets.QMenu(self)

        menu.addAction(self.act_log_x)
        menu.addAction(self.act_log_y)
        menu.addSeparator()
        menu.addAction(self.act_ci)
        menu.addAction(self.act_cp)
        menu.addAction(self.act_measured)
        menu.addSeparator()

        # Range Selectors
        range_widget = QtWidgets.QWidget()
        range_layout = QtWidgets.QGridLayout(range_widget)
        range_layout.setContentsMargins(10, 2, 10, 2)

        range_layout.addWidget(QtWidgets.QLabel("Min Shear (1/s):"), 0, 0)
        range_layout.addWidget(self.spin_min_shear, 0, 1)

        range_layout.addWidget(QtWidgets.QLabel("Max Shear (1/s):"), 1, 0)
        range_layout.addWidget(self.spin_max_shear, 1, 1)

        range_action = QtWidgets.QWidgetAction(menu)
        range_action.setDefaultWidget(range_widget)
        menu.addAction(range_action)

        menu.addSeparator()

        # Hypothesis Button
        hyp_action = QtWidgets.QWidgetAction(menu)
        hyp_btn_widget = QtWidgets.QWidget()
        hyp_layout = QtWidgets.QVBoxLayout(hyp_btn_widget)
        hyp_layout.setContentsMargins(10, 5, 10, 5)
        hyp_layout.addWidget(self.btn_hypothesis)
        hyp_action.setDefaultWidget(hyp_btn_widget)
        menu.addAction(hyp_action)

        # Show menu below the button
        menu.exec_(self.btn_opts.mapToGlobal(QtCore.QPoint(0, self.btn_opts.height())))

    def open_hypothesis_dialog(self):
        QtWidgets.QMessageBox.information(
            self, "Add Hypothesis", "Hypothesis dialog placeholder."
        )

    def set_data(self, data):
        self.last_data = data
        has_measured = "measured_y" in data and data["measured_y"] is not None
        self.act_measured.setEnabled(has_measured)
        if not has_measured:
            self.act_measured.setChecked(False)
        self.update_plot()
        self.update_table()

    def update_plot(self):
        if not self.last_data:
            return

        self.plot_widget.clear()
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)

        x_full = np.array(self.last_data["x"])

        # Filter based on Range Selector
        min_shear = self.spin_min_shear.value()
        max_shear = self.spin_max_shear.value()

        mask = (x_full >= min_shear) & (x_full <= max_shear)

        x = x_full[mask]
        y = np.array(self.last_data["y"])[mask]
        lower = np.array(self.last_data["lower"])[mask]
        upper = np.array(self.last_data["upper"])[mask]

        if len(x) == 0:
            return

        # 1. Plot CI
        if self.act_ci.isChecked():
            fill = pg.FillBetweenItem(
                pg.PlotDataItem(x, lower),
                pg.PlotDataItem(x, upper),
                brush=pg.mkBrush(0, 0, 255, 50),
            )
            self.plot_widget.addItem(fill)

        # 2. Plot Measured
        measured_data = self.last_data.get("measured_y")
        if (
            self.act_measured.isChecked()
            and self.act_measured.isEnabled()
            and measured_data is not None
        ):
            meas_y = np.array(measured_data)[mask]
            self.plot_widget.plot(
                x,
                meas_y,
                pen=pg.mkPen("r", width=2, style=Qt.DashLine),
                symbol="x",
                symbolBrush="r",
                name="Measured",
            )

        # 3. Plot Predicted
        self.plot_widget.plot(
            x,
            y,
            pen=pg.mkPen("b", width=2),
            symbol="o",
            symbolSize=6,
            symbolBrush="b",
            name="Predicted",
        )

        # 4. Plot CP
        if self.act_cp.isChecked():
            cp_y = y * (1 + (np.random.rand(len(y)) - 0.5) * 0.05)
            self.plot_widget.plot(
                x,
                cp_y,
                pen=None,
                symbol="t1",
                symbolSize=10,
                symbolBrush="g",
                name="CP Measure",
            )

        # --- UPDATE AXIS SETTINGS ---
        log_x = self.act_log_x.isChecked()
        log_y = self.act_log_y.isChecked()

        # 1. Set Log Mode first
        self.plot_widget.setLogMode(x=log_x, y=log_y)

        # 2. Define X Limits (Fixed Requirement)
        limit_x_min = 100
        limit_x_max = 15000000

        if log_x:
            vb_x_min = np.log10(limit_x_min)
            vb_x_max = np.log10(limit_x_max)
        else:
            vb_x_min = limit_x_min
            vb_x_max = limit_x_max

        # 3. Define Y Limits (No Maximum)
        # [FIX] We replace 'None' with concrete large numbers to avoid TypeError
        if log_y:
            # Log Mode:
            # Min: -10 (corresponds to 1e-10, effectively 0)
            # Max: 300 (corresponds to 1e300, near max float limit)
            vb_y_min = -10.0
            vb_y_max = 300.0
        else:
            # Linear Mode:
            # Min: 0
            # Max: 1e300 (Huge float, effectively infinite)
            vb_y_min = 0.0
            vb_y_max = 1e300

        # 4. Apply Limits
        # Explicitly pass all 4 values as floats. No 'None' allowed.
        self.plot_widget.plotItem.vb.setLimits(
            xMin=vb_x_min, xMax=vb_x_max, yMin=vb_y_min, yMax=vb_y_max
        )

    def mouse_moved(self, evt):
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x_val = mousePoint.x()
            y_val = mousePoint.y()

            # Fix OverflowError when zooming out too far in Log mode
            if self.act_log_x.isChecked():
                try:
                    # Prevent overflow by clamping or checking exponent bounds
                    if x_val > 300:  # 10^300 is near limit
                        x_val = float("inf")
                    else:
                        x_val = 10**x_val
                except OverflowError:
                    x_val = float("inf")

            if self.act_log_y.isChecked():
                try:
                    if y_val > 300:
                        y_val = float("inf")
                    else:
                        y_val = 10**y_val
                except OverflowError:
                    y_val = float("inf")

            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())

            def fmt(v):
                if v == float("inf"):
                    return "Inf"
                if abs(v) >= 10000 or (abs(v) < 0.01 and v != 0):
                    return f"{v:.2e}"
                return f"{v:.2f}"

            self.hover_label.setText(
                f"Shear Rate: {fmt(x_val)} 1/s  |  Viscosity: {fmt(y_val)} cP"
            )


class PredictionUI(QtWidgets.QWidget):
    INGREDIENT_TYPES = [
        "Protein",
        "Buffer",
        "Surfactant",
        "Stabilizer",
        "Excipient",
        "Salt",
    ]
    INGREDIENT_UNITS = {
        "Protein": "mg/mL",
        "Buffer": "mM",
        "Surfactant": "%w",
        "Stabilizer": "M",
        "Excipient": "mM",
        "Salt": "mM",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ingredients_by_type = {}
        self._load_mock_data()
        self.init_ui()
        self.setStyleSheet(LIGHT_STYLE_SHEET)

    def _load_mock_data(self):
        mk_obj = lambda n, t: type("obj", (object,), {"name": n, "type": t})
        self.ingredients_by_type["Protein"] = [
            mk_obj("Ibalizumab", "Protein"),
            mk_obj("mAb-1", "Protein"),
        ]
        self.ingredients_by_type["Buffer"] = [
            mk_obj("Histidine", "Buffer"),
            mk_obj("Acetate", "Buffer"),
        ]
        self.ingredients_by_type["Salt"] = [
            mk_obj("NaCl", "Salt"),
            mk_obj("KCl", "Salt"),
        ]
        self.ingredients_by_type["Surfactant"] = [
            mk_obj("PS20", "Surfactant"),
            mk_obj("PS80", "Surfactant"),
        ]
        self.ingredients_by_type["Excipient"] = [
            mk_obj("Sucrose", "Excipient"),
            mk_obj("Arginine", "Excipient"),
        ]
        for t in self.INGREDIENT_TYPES:
            if t not in self.ingredients_by_type:
                self.ingredients_by_type[t] = []

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        splitter = QtWidgets.QSplitter(Qt.Horizontal)

        left_widget = QtWidgets.QWidget()
        left_widget.setObjectName("leftPanel")
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton("New Prediction")
        btn_add.clicked.connect(lambda: self.add_prediction_card(None))
        btn_import = QtWidgets.QPushButton("Import Data...")
        btn_import.clicked.connect(self.import_data_file)
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_import)
        left_layout.addLayout(btn_layout)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.cards_container = QtWidgets.QWidget()
        self.cards_container.setObjectName("scrollContent")
        self.cards_layout = QtWidgets.QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(15, 15, 15, 15)
        self.cards_layout.setSpacing(10)
        self.cards_layout.addStretch()

        self.scroll_area.setWidget(self.cards_container)
        left_layout.addWidget(self.scroll_area)
        splitter.addWidget(left_widget)

        self.viz_panel = VisualizationPanel()
        splitter.addWidget(self.viz_panel)

        splitter.setSizes([450, 700])
        main_layout.addWidget(splitter)
        self.current_task = None

        self.add_prediction_card()

    def add_prediction_card(self, data=None):
        if data and "name" in data:
            name = data["name"]
        else:
            current_count = max(0, self.cards_layout.count() - 1)
            name = f"Prediction {current_count + 1}"
        card = PredictionConfigCard(
            default_name=name,
            ingredients_data=self.ingredients_by_type,
            ingredient_types=self.INGREDIENT_TYPES,
            ingredient_units=self.INGREDIENT_UNITS,
        )
        card.removed.connect(self.remove_card)
        card.run_requested.connect(self.run_prediction)
        card.expanded.connect(self.on_card_expanded)
        insert_idx = max(0, self.cards_layout.count() - 1)
        self.cards_layout.insertWidget(insert_idx, card)
        if data:
            if hasattr(card, "load_data"):
                card.load_data(data)
            if data.get("measured", False):
                card.set_measured_state(True)
            card.emit_run_request()
        self.on_card_expanded(card)
        QtCore.QTimer.singleShot(100, lambda: self._scroll_to_card(card))

    def _scroll_to_card(self, card_widget):
        """Helper to ensure the new card is visible in the scroll area."""
        self.scroll_area.ensureWidgetVisible(card_widget, 0, 0)

    def on_card_expanded(self, active_card):
        for i in range(self.cards_layout.count()):
            item = self.cards_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, PredictionConfigCard) and widget is not active_card:
                widget.collapse()

    def remove_card(self, card_widget):
        card_widget.setDisabled(True)
        anim = QtCore.QPropertyAnimation(card_widget, b"maximumHeight", card_widget)
        anim.setDuration(200)
        anim.setStartValue(card_widget.height())
        anim.setEndValue(0)
        anim.setEasingCurve(QtCore.QEasingCurve.InBack)

        def cleanup():
            self.cards_layout.removeWidget(card_widget)
            card_widget.deleteLater()

        anim.finished.connect(cleanup)
        anim.start()

    def import_data_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Run Data",
            "",  # Default directory
            "CSV Files (*.csv);;All Files (*)",
        )
        if not fname:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

        try:

            imported_config = {
                "name": f"Imported: {QtCore.QFileInfo(fname).baseName()}",
                "measured": True,
                "use_in_icl": True,
                "notes": "Imported dataset.",
            }
            self.add_prediction_card(imported_config)
            self.run_prediction(imported_config)

        finally:

            QtWidgets.QApplication.restoreOverrideCursor()

    def run_prediction(self, config=None):
        """
        Runs prediction using the robust PredictionThread subclass.
        """
        sender_card = self.sender()
        if isinstance(sender_card, PredictionConfigCard):
            self.running_card = sender_card
        else:
            self.running_card = None
        if self.current_task is not None and self.current_task.isRunning():
            print("Stopping previous task...")
            self.current_task.stop()

        # 2. Visual Feedback
        name = config.get("name", "Unknown Sample") if config else "Unknown Sample"
        self.viz_panel.set_plot_title(f"Calculating: {name}...")
        self.viz_panel.show_loading()

        # 3. Create & Start New Thread
        # We assign it to self.current_task to keep the Python object alive
        self.current_task = PredictionThread(config)

        # 4. Connect Signals
        self.current_task.data_ready.connect(self._on_prediction_finished)

        # Clean up reference when done (Optional, but good for memory)
        self.current_task.finished.connect(self._on_task_complete)

        # 5. Start
        self.current_task.start()

    def _on_prediction_finished(self, data_package):
        final_name = data_package.get("config_name", "Unknown")
        self.viz_panel.set_plot_title(final_name)
        self.viz_panel.set_data(data_package)
        self.viz_panel.hide_loading()
        if hasattr(self, "running_card") and self.running_card:
            self.running_card.set_results(data_package)

    def _on_task_complete(self):
        """Called when thread naturally finishes."""
        # We don't set self.current_task to None here immediately
        # to avoid race conditions, but it's safe to leave it bound.
        pass

    def closeEvent(self, event):
        """
        Guaranteed cleanup on close.
        """
        if self.current_task is not None and self.current_task.isRunning():
            print("Closing application: Stopping background thread...")
            self.current_task.stop()

        super().closeEvent(event)


class PredictionThread(QtCore.QThread):
    """
    A robust QThread subclass.
    Combines the thread and logic into one object to prevent Garbage Collection errors.
    """

    # Define the signal that sends data back to the main UI
    data_ready = QtCore.Signal(dict)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._is_running = True  # Logic flag

    def run(self):
        """
        The code that runs in the background.
        """
        import time

        import numpy as np

        # 1. Simulate Calculation (Safe to sleep here)
        time.sleep(0.8)

        # 2. Check if we were told to stop during the sleep
        if not self._is_running:
            return

        # 3. The Math Logic
        shear_rates = np.logspace(1, 6, 20)
        K = 50.0
        n = 0.6

        if self.config:
            temp_mod = float(self.config.get("temp", 25)) / 25.0
            K = K / max(0.1, temp_mod)

        viscosity = K * np.power(shear_rates, n - 1)

        measured_y = None
        if self.config and self.config.get("measured", False):
            noise = np.random.normal(0, 2, len(shear_rates))
            measured_y = viscosity + noise

        data_package = {
            "x": shear_rates,
            "y": viscosity,
            "upper": viscosity * 1.15,
            "lower": viscosity * 0.85,
            "measured_y": measured_y,
            "config_name": (
                self.config.get("name", "Unknown") if self.config else "Unknown"
            ),
        }

        # 4. Emit Data
        if self._is_running:
            self.data_ready.emit(data_package)

    def stop(self):
        """Call this to stop the thread gracefully."""
        self._is_running = False
        self.quit()
        self.wait(1000)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # app.setStyleSheet(LIGHT_STYLE_SHEET)
    win = PredictionUI()
    win.setWindowTitle("Viscosity AI - Hyperparameter Tuning")
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())
