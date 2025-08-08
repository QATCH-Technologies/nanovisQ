import sys
from typing import Any
from enum import Enum
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit,
    QTabWidget, QTableWidget, QTableWidgetItem, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox, QHeaderView,
    QGridLayout, QListWidget, QFormLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor

# Import the OpentronsFlex and related modules
try:
    from QATCH.FluxControls.src.opentrons_flex import OpentronsFlex
    from QATCH.FluxControls.src.constants import (
        MountPositions, Pipettes, DeckLocations, Axis
    )
    from QATCH.FluxControls.src.standard_labware import (
        StandardLabware, StandardAdapters, StandardAluminumBlocks, StandardReservoirs, StandardTipracks, StandardTubeRacks, StandardWellplates)
    from QATCH.common.logger import Logger as Log
except ImportError:
    class Log:
        @staticmethod
        def d(tag, msg=""): print(f"DEBUG: {tag} {msg}")
        @staticmethod
        def i(tag, msg=""): print(f"INFO: {tag} {msg}")
        @staticmethod
        def w(tag, msg=""): print(f"WARNING: {tag} {msg}")
        @staticmethod
        def e(tag, msg=""): print(f"ERROR: {tag} {msg}")


class RobotWorker(QThread):
    """Worker thread for robot operations to prevent GUI freezing"""
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.operation = None
        self.params = {}
        self.robot = None

    def set_operation(self, operation: str, robot, **params):
        self.operation = operation
        self.robot = robot
        self.params = params

    def run(self):
        try:
            if not self.robot or not self.operation:
                return

            result = None

            # Execute different operations based on type
            if self.operation == "connect":
                mac = self.params.get('mac')
                ip = self.params.get('ip')
                self.robot = OpentronsFlex(mac_address=mac, ip_address=ip)
                result = {"status": "connected",
                          "ip": self.robot._get_robot_ipv4()}

            elif self.operation == "load_pipette":
                pipette = self.params.get('pipette')
                position = self.params.get('position')
                self.robot.load_pipette(pipette, position)
                result = {"status": "pipette_loaded"}

            elif self.operation == "load_labware":
                location = self.params.get('location')
                definition = self.params.get('definition')
                self.robot.load_labware(location, definition)
                result = {"status": "labware_loaded"}

            elif self.operation == "run_protocol":
                protocol_name = self.params.get('protocol_name')
                run_id = self.robot.run_protocol(protocol_name)
                result = {"status": "protocol_running", "run_id": run_id}

            elif self.operation == "upload_protocol":
                filepath = self.params.get('filepath')
                custom_labware = self.params.get('custom_labware', [])
                if custom_labware:
                    response = self.robot.upload_protocol_custom_labware(
                        filepath, *custom_labware)
                else:
                    response = self.robot.upload_protocol(filepath)
                result = {"status": "protocol_uploaded", "response": response}

            elif self.operation == "home":
                self.robot.home()
                result = {"status": "homed"}

            elif self.operation == "lights_on":
                self.robot.lights_on()
                result = {"status": "lights_on"}

            elif self.operation == "lights_off":
                self.robot.lights_off()
                result = {"status": "lights_off"}

            elif self.operation == "pickup_tip":
                labware = self.params.get('labware')
                pipette = self.params.get('pipette')
                self.robot.pickup_tip(labware, pipette)
                result = {"status": "tip_picked"}

            elif self.operation == "aspirate":
                labware = self.params.get('labware')
                pipette = self.params.get('pipette')
                volume = self.params.get('volume')
                flow_rate = self.params.get('flow_rate')
                self.robot.aspirate(labware, pipette, flow_rate, volume)
                result = {"status": "aspirated"}

            elif self.operation == "dispense":
                labware = self.params.get('labware')
                pipette = self.params.get('pipette')
                volume = self.params.get('volume')
                flow_rate = self.params.get('flow_rate')
                self.robot.dispense(labware, pipette, flow_rate, volume)
                result = {"status": "dispensed"}

            elif self.operation == "drop_tip":
                labware = self.params.get('labware')
                pipette = self.params.get('pipette')
                self.robot.drop_tip(labware, pipette)
                result = {"status": "tip_dropped"}

            elif self.operation == "move_to_well":
                labware = self.params.get('labware')
                pipette = self.params.get('pipette')
                self.robot.move_to_well(labware, pipette)
                result = {"status": "moved_to_well"}

            elif self.operation == "pause_run":
                run_id = self.params.get('run_id')
                self.robot.pause_run(run_id)
                result = {"status": "paused"}

            elif self.operation == "resume_run":
                run_id = self.params.get('run_id')
                self.robot.resume_run(run_id)
                result = {"status": "resumed"}

            elif self.operation == "stop_run":
                run_id = self.params.get('run_id')
                self.robot.stop_run(run_id)
                result = {"status": "stopped"}

            if result:
                self.result_ready.emit(result)

        except Exception as e:
            error_msg = f"Operation {self.operation} failed: {str(e)}"
            Log.e("RobotWorker", error_msg)
            self.error_occurred.emit(error_msg)


class OpentronFlexControlPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.robot = None
        self.worker = RobotWorker()
        self.current_run_id = None
        self.loaded_pipettes = {}
        self.loaded_labware = {}
        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        self.setWindowTitle("Opentrons Flex Control Panel")
        self.setGeometry(100, 100, 1400, 900)

        # Set application style
        self.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            background-color: #f5f5f5;
        }
        QPushButton {
            background-color: #00aeee;        /* base color */
            color: white;
            border: none;
            padding: 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #00c4ff;        /* slightly lighter */
        }
        QPushButton:pressed {
            background-color: #009fc4;        /* slightly darker */
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        QTabWidget::pane {
            border: 1px solid #cccccc;
            background-color: white;
        }
    """)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Connection Bar
        connection_group = self.create_connection_group()
        main_layout.addWidget(connection_group)

        # Status Bar
        self.status_bar = QTextEdit()
        self.status_bar.setMaximumHeight(100)
        self.status_bar.setReadOnly(True)
        self.status_bar.setPlaceholderText(
            "System messages will appear here...")
        main_layout.addWidget(self.status_bar)

        # Main Tab Widget
        self.tabs = QTabWidget()

        # Tab 1: Setup & Configuration
        self.setup_tab = self.create_setup_tab()
        self.tabs.addTab(self.setup_tab, "Setup & Configuration")

        # Tab 2: Protocol Management
        self.protocol_tab = self.create_protocol_tab()
        self.tabs.addTab(self.protocol_tab, "Protocol Management")

        # Tab 3: Manual Control
        self.manual_tab = self.create_manual_control_tab()
        self.tabs.addTab(self.manual_tab, "Manual Control")

        # Tab 4: Run Control
        self.run_tab = self.create_run_control_tab()
        self.tabs.addTab(self.run_tab, "Run Control")

        # Tab 5: System
        self.system_tab = self.create_system_tab()
        self.tabs.addTab(self.system_tab, "System")

        main_layout.addWidget(self.tabs)

    def create_connection_group(self):
        group = QGroupBox("Robot Connection")
        layout = QHBoxLayout()

        # MAC Address
        layout.addWidget(QLabel("MAC Address:"))
        self.mac_input = QLineEdit()
        self.mac_input.setPlaceholderText("XX:XX:XX:XX:XX:XX")
        layout.addWidget(self.mac_input)

        # IP Address
        layout.addWidget(QLabel("IP Address (optional):"))
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("192.168.1.XXX")
        layout.addWidget(self.ip_input)

        # Connect Button
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_robot)
        layout.addWidget(self.connect_btn)

        # Connection Status
        self.connection_status = QLabel("Disconnected")
        self.connection_status.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(self.connection_status)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_setup_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Pipette Configuration
        pipette_group = QGroupBox("Pipette Configuration")
        pipette_layout = QGridLayout()

        # Left Mount
        pipette_layout.addWidget(QLabel("Left Mount:"), 0, 0)
        self.left_pipette_combo = QComboBox()
        self.left_pipette_combo.addItems([
            "None",
            "P1000 Single Gen3",
            "P1000 Multi Gen3",
            "P300 Single Gen2",
            "P50 Single Gen3",
            "P50 Multi Gen3"
        ])
        pipette_layout.addWidget(self.left_pipette_combo, 0, 1)
        self.load_left_btn = QPushButton("Load Left")
        self.load_left_btn.clicked.connect(lambda: self.load_pipette("left"))
        pipette_layout.addWidget(self.load_left_btn, 0, 2)

        # Right Mount
        pipette_layout.addWidget(QLabel("Right Mount:"), 1, 0)
        self.right_pipette_combo = QComboBox()
        self.right_pipette_combo.addItems([
            "None",
            "P1000 Single Gen3",
            "P1000 Multi Gen3",
            "P300 Single Gen2",
            "P50 Single Gen3",
            "P50 Multi Gen3"
        ])
        pipette_layout.addWidget(self.right_pipette_combo, 1, 1)
        self.load_right_btn = QPushButton("Load Right")
        self.load_right_btn.clicked.connect(lambda: self.load_pipette("right"))
        pipette_layout.addWidget(self.load_right_btn, 1, 2)

        pipette_group.setLayout(pipette_layout)
        layout.addWidget(pipette_group)

        # Labware Configuration
        labware_group = QGroupBox("Labware Configuration")
        labware_layout = QGridLayout()

        # Deck position selector
        labware_layout.addWidget(QLabel("Deck Position:"), 0, 0)
        self.deck_position_combo = QComboBox()
        positions = [pos.value for pos in DeckLocations]
        self.deck_position_combo.addItems(positions)
        labware_layout.addWidget(self.deck_position_combo, 0, 1)

        # Labware type selector
        labware_layout.addWidget(QLabel("Labware Type:"), 1, 0)
        self.labware_type_combo = QComboBox()
        self.labware_type_combo.addItems([
            "384 Wellplate 40µL (Applied Biosystems MicroAmp)",
            "384 Well Plate 50µL (Biorad)",
            "12 Well Reservoir 15µL (Nest)",
            "Tip Rack 1000µL (GEB 96)",
            "Tip Rack 10µL (GEB 96)",
        ])
        labware_layout.addWidget(self.labware_type_combo, 1, 1)

        self.load_labware_btn = QPushButton("Load Labware")
        self.load_labware_btn.clicked.connect(self.load_labware)
        labware_layout.addWidget(self.load_labware_btn, 2, 0, 1, 2)

        # Deck visualization
        self.deck_table = QTableWidget(4, 4)
        self.deck_table.setHorizontalHeaderLabels(["1", "2", "3", "4"])
        self.deck_table.setVerticalHeaderLabels(["A", "B", "C", "D"])
        self.deck_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.deck_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.deck_table.setMinimumHeight(200)
        labware_layout.addWidget(self.deck_table, 3, 0, 1, 2)

        labware_group.setLayout(labware_layout)
        layout.addWidget(labware_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_protocol_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Upload Protocol
        upload_group = QGroupBox("Upload Protocol")
        upload_layout = QVBoxLayout()

        # Protocol file selection
        file_layout = QHBoxLayout()
        self.protocol_file_label = QLabel("No file selected")
        file_layout.addWidget(self.protocol_file_label)
        self.browse_protocol_btn = QPushButton("Browse Protocol")
        self.browse_protocol_btn.clicked.connect(self.browse_protocol_file)
        file_layout.addWidget(self.browse_protocol_btn)
        upload_layout.addLayout(file_layout)

        # Custom labware files
        self.custom_labware_list = QListWidget()
        self.custom_labware_list.setMaximumHeight(100)
        upload_layout.addWidget(QLabel("Custom Labware Files:"))
        upload_layout.addWidget(self.custom_labware_list)

        labware_btn_layout = QHBoxLayout()
        self.add_labware_btn = QPushButton("Add Labware File")
        self.add_labware_btn.clicked.connect(self.add_custom_labware)
        self.remove_labware_btn = QPushButton("Remove Selected")
        self.remove_labware_btn.clicked.connect(self.remove_custom_labware)
        labware_btn_layout.addWidget(self.add_labware_btn)
        labware_btn_layout.addWidget(self.remove_labware_btn)
        upload_layout.addLayout(labware_btn_layout)

        self.upload_protocol_btn = QPushButton("Upload Protocol")
        self.upload_protocol_btn.clicked.connect(self.upload_protocol)
        upload_layout.addWidget(self.upload_protocol_btn)

        upload_group.setLayout(upload_layout)
        layout.addWidget(upload_group)

        # Available Protocols
        protocols_group = QGroupBox("Available Protocols")
        protocols_layout = QVBoxLayout()

        self.protocols_table = QTableWidget()
        self.protocols_table.setColumnCount(3)
        self.protocols_table.setHorizontalHeaderLabels(
            ["Name", "ID", "Created"])
        self.protocols_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        protocols_layout.addWidget(self.protocols_table)

        protocol_btn_layout = QHBoxLayout()
        self.refresh_protocols_btn = QPushButton("Refresh")
        self.refresh_protocols_btn.clicked.connect(self.refresh_protocols)
        self.run_protocol_btn = QPushButton("Run Selected")
        self.run_protocol_btn.clicked.connect(self.run_selected_protocol)
        self.delete_protocol_btn = QPushButton("Delete Selected")
        self.delete_protocol_btn.clicked.connect(self.delete_selected_protocol)

        protocol_btn_layout.addWidget(self.refresh_protocols_btn)
        protocol_btn_layout.addWidget(self.run_protocol_btn)
        protocol_btn_layout.addWidget(self.delete_protocol_btn)
        protocols_layout.addLayout(protocol_btn_layout)

        protocols_group.setLayout(protocols_layout)
        layout.addWidget(protocols_group)

        tab.setLayout(layout)
        return tab

    def create_manual_control_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Pipette Control
        pipette_control_group = QGroupBox("Pipette Control")
        pipette_layout = QGridLayout()

        # Pipette selector
        pipette_layout.addWidget(QLabel("Select Pipette:"), 0, 0)
        self.manual_pipette_combo = QComboBox()
        self.manual_pipette_combo.addItems(["Left Mount", "Right Mount"])
        pipette_layout.addWidget(self.manual_pipette_combo, 0, 1)

        # Volume control
        pipette_layout.addWidget(QLabel("Volume (µL):"), 1, 0)
        self.volume_spin = QDoubleSpinBox()
        self.volume_spin.setRange(0, 1000)
        self.volume_spin.setDecimals(1)
        pipette_layout.addWidget(self.volume_spin, 1, 1)

        # Flow rate control
        pipette_layout.addWidget(QLabel("Flow Rate (µL/s):"), 2, 0)
        self.flow_rate_spin = QDoubleSpinBox()
        self.flow_rate_spin.setRange(1, 500)
        self.flow_rate_spin.setValue(100)
        pipette_layout.addWidget(self.flow_rate_spin, 2, 1)

        # Action buttons
        action_layout = QGridLayout()
        self.pickup_tip_btn = QPushButton("Pickup Tip")
        self.pickup_tip_btn.clicked.connect(self.pickup_tip)
        action_layout.addWidget(self.pickup_tip_btn, 0, 0)

        self.aspirate_btn = QPushButton("Aspirate")
        self.aspirate_btn.clicked.connect(self.aspirate)
        action_layout.addWidget(self.aspirate_btn, 0, 1)

        self.dispense_btn = QPushButton("Dispense")
        self.dispense_btn.clicked.connect(self.dispense)
        action_layout.addWidget(self.dispense_btn, 1, 0)

        self.blowout_btn = QPushButton("Blowout")
        self.blowout_btn.clicked.connect(self.blowout)
        action_layout.addWidget(self.blowout_btn, 1, 1)

        self.drop_tip_btn = QPushButton("Drop Tip")
        self.drop_tip_btn.clicked.connect(self.drop_tip)
        action_layout.addWidget(self.drop_tip_btn, 2, 0)

        self.move_to_well_btn = QPushButton("Move to Well")
        self.move_to_well_btn.clicked.connect(self.move_to_well)
        action_layout.addWidget(self.move_to_well_btn, 2, 1)

        pipette_layout.addLayout(action_layout, 3, 0, 1, 2)

        pipette_control_group.setLayout(pipette_layout)
        layout.addWidget(pipette_control_group)

        # Movement Control
        movement_group = QGroupBox("Movement Control")
        movement_layout = QGridLayout()

        # Coordinate movement
        movement_layout.addWidget(QLabel("X (mm):"), 0, 0)
        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(-500, 500)
        movement_layout.addWidget(self.x_spin, 0, 1)

        movement_layout.addWidget(QLabel("Y (mm):"), 1, 0)
        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(-500, 500)
        movement_layout.addWidget(self.y_spin, 1, 1)

        movement_layout.addWidget(QLabel("Z (mm):"), 2, 0)
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(0, 500)
        movement_layout.addWidget(self.z_spin, 2, 1)

        self.move_to_coords_btn = QPushButton("Move to Coordinates")
        self.move_to_coords_btn.clicked.connect(self.move_to_coordinates)
        movement_layout.addWidget(self.move_to_coords_btn, 3, 0, 1, 2)

        # Relative movement
        movement_layout.addWidget(QLabel("Relative Distance (mm):"), 4, 0)
        self.relative_distance_spin = QDoubleSpinBox()
        self.relative_distance_spin.setRange(-100, 100)
        movement_layout.addWidget(self.relative_distance_spin, 4, 1)

        movement_layout.addWidget(QLabel("Axis:"), 5, 0)
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["X", "Y", "Z"])
        movement_layout.addWidget(self.axis_combo, 5, 1)

        self.move_relative_btn = QPushButton("Move Relative")
        self.move_relative_btn.clicked.connect(self.move_relative)
        movement_layout.addWidget(self.move_relative_btn, 6, 0, 1, 2)

        movement_group.setLayout(movement_layout)
        layout.addWidget(movement_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_run_control_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Current Run
        current_run_group = QGroupBox("Current Run")
        run_layout = QVBoxLayout()

        # Run info
        info_layout = QHBoxLayout()
        self.run_id_label = QLabel("Run ID: None")
        self.run_status_label = QLabel("Status: Idle")
        info_layout.addWidget(self.run_id_label)
        info_layout.addWidget(self.run_status_label)
        run_layout.addLayout(info_layout)

        # Run controls
        control_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play_run)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_run)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_run)
        self.resume_btn = QPushButton("Resume")
        self.resume_btn.clicked.connect(self.resume_run)

        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.resume_btn)
        control_layout.addWidget(self.stop_btn)

        run_layout.addLayout(control_layout)

        current_run_group.setLayout(run_layout)
        layout.addWidget(current_run_group)

        # Run History
        history_group = QGroupBox("Run History")
        history_layout = QVBoxLayout()

        self.runs_table = QTableWidget()
        self.runs_table.setColumnCount(3)
        self.runs_table.setHorizontalHeaderLabels(
            ["Run ID", "Status", "Created"])
        self.runs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        history_layout.addWidget(self.runs_table)

        history_btn_layout = QHBoxLayout()
        self.refresh_runs_btn = QPushButton("Refresh Runs")
        self.refresh_runs_btn.clicked.connect(self.refresh_runs)
        self.delete_run_btn = QPushButton("Delete Selected")
        self.delete_run_btn.clicked.connect(self.delete_selected_run)

        history_btn_layout.addWidget(self.refresh_runs_btn)
        history_btn_layout.addWidget(self.delete_run_btn)
        history_layout.addLayout(history_btn_layout)

        history_group.setLayout(history_layout)
        layout.addWidget(history_group)

        tab.setLayout(layout)
        return tab

    def create_system_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # System Controls
        system_group = QGroupBox("System Controls")
        system_layout = QGridLayout()

        # Home
        self.home_btn = QPushButton("Home All Axes")
        self.home_btn.clicked.connect(self.home_robot)
        system_layout.addWidget(self.home_btn, 0, 0)

        # Lights
        self.lights_on_btn = QPushButton("Lights On")
        self.lights_on_btn.clicked.connect(self.lights_on)
        system_layout.addWidget(self.lights_on_btn, 1, 0)

        self.lights_off_btn = QPushButton("Lights Off")
        self.lights_off_btn.clicked.connect(self.lights_off)
        system_layout.addWidget(self.lights_off_btn, 1, 1)

        self.flash_lights_btn = QPushButton("Flash Lights")
        self.flash_lights_btn.clicked.connect(self.flash_lights)
        system_layout.addWidget(self.flash_lights_btn, 2, 0)

        self.flash_count_spin = QSpinBox()
        self.flash_count_spin.setRange(1, 10)
        self.flash_count_spin.setValue(3)
        system_layout.addWidget(self.flash_count_spin, 2, 1)

        system_group.setLayout(system_layout)
        layout.addWidget(system_group)

        # System Info
        info_group = QGroupBox("System Information")
        info_layout = QFormLayout()

        self.robot_ip_label = QLabel("Not connected")
        self.robot_mac_label = QLabel("Not connected")
        self.light_status_label = QLabel("Unknown")

        info_layout.addRow("Robot IP:", self.robot_ip_label)
        info_layout.addRow("Robot MAC:", self.robot_mac_label)
        info_layout.addRow("Light Status:", self.light_status_label)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def setup_connections(self):
        """Setup worker thread connections"""
        self.worker.result_ready.connect(self.handle_worker_result)
        self.worker.error_occurred.connect(self.handle_worker_error)
        self.worker.progress_update.connect(self.update_status)

    def log_message(self, message: str, level: str = "INFO"):
        """Add message to status bar"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level}] {message}"
        self.status_bar.append(formatted_msg)

        # Log using QATCH logger
        if level == "ERROR":
            Log.e("Flex Controls", message)
        elif level == "WARNING":
            Log.w("Flex Controls", message)
        elif level == "DEBUG":
            Log.d("Flex Controls", message)
        else:
            Log.i("Flex Controls", message)

    @pyqtSlot(dict)
    def handle_worker_result(self, result: dict):
        """Handle successful worker operation"""
        status = result.get("status", "")

        if status == "connected":
            self.connection_status.setText("Connected")
            self.connection_status.setStyleSheet(
                "color: green; font-weight: bold;")
            self.robot_ip_label.setText(result.get("ip", ""))
            self.robot_mac_label.setText(self.mac_input.text())
            self.enable_controls(True)
            self.log_message(
                f"Successfully connected to robot at {result.get('ip', '')}")
            self.refresh_protocols()

        elif status == "pipette_loaded":
            self.log_message("Pipette loaded successfully")
            self.update_deck_view()

        elif status == "labware_loaded":
            self.log_message("Labware loaded successfully")
            self.update_deck_view()

        elif status == "protocol_running":
            self.current_run_id = result.get("run_id")
            self.run_id_label.setText(f"Run ID: {self.current_run_id}")
            self.run_status_label.setText("Status: Running")
            self.log_message(
                f"Protocol started with run ID: {self.current_run_id}")

        elif status == "protocol_uploaded":
            self.log_message("Protocol uploaded successfully")
            self.refresh_protocols()

        elif status == "homed":
            self.log_message("Robot homed successfully")

        elif status == "lights_on":
            self.light_status_label.setText("On")
            self.log_message("Lights turned on")

        elif status == "lights_off":
            self.light_status_label.setText("Off")
            self.log_message("Lights turned off")

        elif status == "paused":
            self.run_status_label.setText("Status: Paused")
            self.log_message("Run paused")

        elif status == "resumed":
            self.run_status_label.setText("Status: Running")
            self.log_message("Run resumed")

        elif status == "stopped":
            self.run_status_label.setText("Status: Stopped")
            self.log_message("Run stopped")

    @pyqtSlot(str)
    def handle_worker_error(self, error_msg: str):
        """Handle worker operation error"""
        self.log_message(error_msg, "ERROR")
        QMessageBox.critical(self, "Operation Failed", error_msg)

    def update_status(self, message: str):
        """Update status bar with progress message"""
        self.log_message(message)

    def enable_controls(self, enabled: bool):
        """Enable/disable controls based on connection status"""
        # Setup tab
        self.load_left_btn.setEnabled(enabled)
        self.load_right_btn.setEnabled(enabled)
        self.load_labware_btn.setEnabled(enabled)

        # Protocol tab
        self.upload_protocol_btn.setEnabled(enabled)
        self.refresh_protocols_btn.setEnabled(enabled)
        self.run_protocol_btn.setEnabled(enabled)
        self.delete_protocol_btn.setEnabled(enabled)

        # Manual control tab
        self.pickup_tip_btn.setEnabled(enabled)
        self.aspirate_btn.setEnabled(enabled)
        self.dispense_btn.setEnabled(enabled)
        self.blowout_btn.setEnabled(enabled)
        self.drop_tip_btn.setEnabled(enabled)
        self.move_to_well_btn.setEnabled(enabled)
        self.move_to_coords_btn.setEnabled(enabled)
        self.move_relative_btn.setEnabled(enabled)

        # Run control tab
        self.play_btn.setEnabled(enabled)
        self.pause_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(enabled)
        self.resume_btn.setEnabled(enabled)
        self.refresh_runs_btn.setEnabled(enabled)
        self.delete_run_btn.setEnabled(enabled)

        # System tab
        self.home_btn.setEnabled(enabled)
        self.lights_on_btn.setEnabled(enabled)
        self.lights_off_btn.setEnabled(enabled)
        self.flash_lights_btn.setEnabled(enabled)

    def connect_robot(self):
        """Connect to the robot"""
        mac = self.mac_input.text().strip()
        ip = self.ip_input.text().strip() if self.ip_input.text().strip() else None

        if not mac:
            QMessageBox.warning(self, "Connection Error",
                                "Please enter a MAC address")
            return

        self.log_message("Attempting to connect to robot...")

        try:
            # Create robot instance in main thread first
            self.robot = OpentronsFlex(mac_address=mac, ip_address=ip)

            # Update UI
            self.connection_status.setText("Connected")
            self.connection_status.setStyleSheet(
                "color: green; font-weight: bold;")
            self.robot_ip_label.setText(self.robot._get_robot_ipv4())
            self.robot_mac_label.setText(mac)
            self.enable_controls(True)
            self.log_message(
                f"Successfully connected to robot at {self.robot._get_robot_ipv4()}")

            # Initialize worker with robot
            self.worker.robot = self.robot

            # Refresh available protocols
            self.refresh_protocols()

        except Exception as e:
            self.handle_worker_error(f"Connection failed: {str(e)}")

    def load_pipette(self, mount: str):
        """Load pipette on specified mount"""
        if not self.robot:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to robot first")
            return

        if mount == "left":
            pipette_text = self.left_pipette_combo.currentText()
            position = MountPositions.LEFT_MOUNT
        else:
            pipette_text = self.right_pipette_combo.currentText()
            position = MountPositions.RIGHT_MOUNT

        if pipette_text == "None":
            return

        # Map text to pipette enum
        pipette_map = {
            "P1000 Single Gen3": Pipettes.P1000_SINGLE_FLEX,
            "P1000 Multi Gen3": Pipettes.P1000_MULTI_FLEX,
            "P300 Single Gen2": Pipettes.P300_SINGLE_FLEX_GEN2,
            "P50 Single Gen3": Pipettes.P50_SINGLE_FLEX,
            "P50 Multi Gen3": Pipettes.P50_MULTI_FLEX
        }

        pipette = pipette_map.get(pipette_text)
        if not pipette:
            return

        try:
            self.log_message(f"Loading {pipette_text} on {mount} mount...")
            self.robot.load_pipette(pipette, position)
            self.loaded_pipettes[mount] = pipette
            self.log_message(
                f"Successfully loaded {pipette_text} on {mount} mount")
            self.update_deck_view()
        except Exception as e:
            self.handle_worker_error(f"Failed to load pipette: {str(e)}")

    def load_labware(self):
        """Load labware at specified position"""
        if not self.robot:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to robot first")
            return

        position_text = self.deck_position_combo.currentText()
        labware_text = self.labware_type_combo.currentText()

        # Map position text to enum
        position = getattr(DeckLocations, position_text)

        # Map labware text to standard labware
        labware_map = {
            "384 Wellplate 40µL (Applied Biosystems MicroAmp)": StandardWellplates.APPLIED_BIOSYSTEMS_MICROAMP_384_WELLPLATE_40UL,
            "384 Well Plate 50µL (Biorad)": StandardWellplates.BIORAD_348_WELLPLATE_50UL,
            "12 Well Reservoir 15µL (Nest)": StandardReservoirs.NEST_12_RESERVOIR_15ML,
            "Tip Rack 1000µL (GEB 96)": StandardTipracks.GEB_96_TIPRACK_1000UL,
            "Tip Rack 10µL (GEB 96)": StandardTipracks.GEB_96_TIPRACK_10UL,
        }

        labware = labware_map.get(labware_text)
        if not labware:
            return

        try:
            self.log_message(
                f"Loading {labware_text} at position {position_text}...")
            self.robot.load_labware(position, labware)
            self.loaded_labware[position_text] = labware_text
            self.log_message(
                f"Successfully loaded {labware_text} at position {position_text}")
            self.update_deck_view()
        except Exception as e:
            self.handle_worker_error(f"Failed to load labware: {str(e)}")

    def update_deck_view(self):
        """Update the deck visualization table"""
        for row in range(4):
            for col in range(4):
                position = f"{chr(65+row)}{col+1}"  # A1, A2, etc.
                item = QTableWidgetItem()

                if position in self.loaded_labware:
                    item.setText(self.loaded_labware[position])
                    item.setBackground(QColor(200, 255, 200))
                else:
                    item.setText("Empty")
                    item.setBackground(QColor(240, 240, 240))

                self.deck_table.setItem(row, col, item)

    def browse_protocol_file(self):
        """Browse for protocol file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Protocol File", "", "Python Files (*.py);;JSON Files (*.json)"
        )
        if file_path:
            self.protocol_file_label.setText(file_path)

    def add_custom_labware(self):
        """Add custom labware file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Custom Labware File", "", "JSON Files (*.json)"
        )
        if file_path:
            self.custom_labware_list.addItem(file_path)

    def remove_custom_labware(self):
        """Remove selected custom labware file"""
        current_item = self.custom_labware_list.currentItem()
        if current_item:
            self.custom_labware_list.takeItem(
                self.custom_labware_list.row(current_item))

    def upload_protocol(self):
        """Upload protocol with optional custom labware"""
        if not self.robot:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to robot first")
            return

        protocol_file = self.protocol_file_label.text()
        if protocol_file == "No file selected":
            QMessageBox.warning(
                self, "No File", "Please select a protocol file")
            return

        custom_labware = []
        for i in range(self.custom_labware_list.count()):
            custom_labware.append(self.custom_labware_list.item(i).text())

        try:
            self.log_message("Uploading protocol...")
            if custom_labware:
                self.robot.upload_protocol_custom_labware(
                    protocol_file, *custom_labware)
            else:
                self.robot.upload_protocol(protocol_file)
            self.log_message("Protocol uploaded successfully")
            self.refresh_protocols()
        except Exception as e:
            self.handle_worker_error(f"Failed to upload protocol: {str(e)}")

    def refresh_protocols(self):
        """Refresh the list of available protocols"""
        if not self.robot:
            return

        try:
            self.robot.update_available_protocols()
            protocols = self.robot.available_protocols

            self.protocols_table.setRowCount(len(protocols))
            for i, (name, data) in enumerate(protocols.items()):
                self.protocols_table.setItem(i, 0, QTableWidgetItem(name))
                self.protocols_table.setItem(
                    i, 1, QTableWidgetItem(data["id"]))
                self.protocols_table.setItem(
                    i, 2, QTableWidgetItem(str(data["createdAt"])))

            self.log_message(f"Found {len(protocols)} protocols")
        except Exception as e:
            self.handle_worker_error(f"Failed to refresh protocols: {str(e)}")

    def run_selected_protocol(self):
        """Run the selected protocol"""
        if not self.robot:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to robot first")
            return

        current_row = self.protocols_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "No Selection",
                                "Please select a protocol to run")
            return

        protocol_name = self.protocols_table.item(current_row, 0).text()

        try:
            self.log_message(f"Running protocol: {protocol_name}")
            run_id = self.robot.run_protocol(protocol_name)
            self.current_run_id = run_id
            self.run_id_label.setText(f"Run ID: {run_id}")
            self.run_status_label.setText("Status: Running")
            self.log_message(f"Protocol started with run ID: {run_id}")
        except Exception as e:
            self.handle_worker_error(f"Failed to run protocol: {str(e)}")

    def delete_selected_protocol(self):
        """Delete the selected protocol"""
        if not self.robot:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to robot first")
            return

        current_row = self.protocols_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "No Selection",
                                "Please select a protocol to delete")
            return

        protocol_name = self.protocols_table.item(current_row, 0).text()

        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete protocol '{protocol_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                self.log_message(f"Deleting protocol: {protocol_name}")
                self.robot.delete_protocol(protocol_name)
                self.log_message(f"Protocol deleted successfully")
                self.refresh_protocols()
            except Exception as e:
                self.handle_worker_error(
                    f"Failed to delete protocol: {str(e)}")

    def pickup_tip(self):
        """Pickup tip with selected pipette"""
        if not self.robot:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to robot first")
            return

        # TODO: Implementation would need labware and pipette selection
        self.log_message(
            "Pickup tip function - needs labware/pipette selection dialog")

    def aspirate(self):
        """Aspirate with selected pipette"""
        if not self.robot:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to robot first")
            return

        volume = self.volume_spin.value()
        flow_rate = self.flow_rate_spin.value()

        # TODO: Implementation would need labware and pipette selection
        self.log_message(
            f"Aspirate {volume}µL at {flow_rate}µL/s - needs labware/pipette selection")

    def dispense(self):
        """Dispense with selected pipette"""
        if not self.robot:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to robot first")
            return

        volume = self.volume_spin.value()
        flow_rate = self.flow_rate_spin.value()

        # TODO: Implementation would need labware and pipette selection
        self.log_message(
            f"Dispense {volume}µL at {flow_rate}µL/s - needs labware/pipette selection")

    def blowout(self):
        """Blowout with selected pipette"""
        if not self.robot:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to robot first")
            return

        flow_rate = self.flow_rate_spin.value()

        # TODO: Implementation would need labware and pipette selection
        self.log_message(
            f"Blowout at {flow_rate}µL/s - needs labware/pipette selection")

    def drop_tip(self):
        """Drop tip with selected pipette"""
        if not self.robot:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to robot first")
            return

        # TODO: Implementation would need labware and pipette selection
        self.log_message("Drop tip - needs labware/pipette selection")

    def move_to_well(self):
        """Move to well"""
        if not self.robot:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to robot first")
            return

        # TODO: Implementation would need labware and pipette selection
        self.log_message("Move to well - needs labware/pipette selection")

    def move_to_coordinates(self):
        """Move to specified coordinates"""
        if not self.robot:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to robot first")
            return

        x = self.x_spin.value()
        y = self.y_spin.value()
        z = self.z_spin.value()

        # TODO: Implementation would need pipette selection
        self.log_message(
            f"Move to coordinates X:{x}, Y:{y}, Z:{z} - needs pipette selection")

    def move_relative(self):
        """Move relative distance"""
        if not self.robot:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to robot first")
            return

        distance = self.relative_distance_spin.value()
        axis_text = self.axis_combo.currentText()
        axis = getattr(Axis, axis_text)

        # TODO: Implementation would need pipette selection
        self.log_message(
            f"Move relative {distance}mm on {axis_text} axis - needs pipette selection")

    def play_run(self):
        """Play/start run"""
        if not self.robot or not self.current_run_id:
            QMessageBox.warning(self, "No Run", "No run to play")
            return

        try:
            self.robot.play_run(self.current_run_id)
            self.run_status_label.setText("Status: Running")
            self.log_message("Run started")
        except Exception as e:
            self.handle_worker_error(f"Failed to play run: {str(e)}")

    def pause_run(self):
        """Pause run"""
        if not self.robot or not self.current_run_id:
            QMessageBox.warning(self, "No Run", "No run to pause")
            return

        try:
            self.robot.pause_run(self.current_run_id)
            self.run_status_label.setText("Status: Paused")
            self.log_message("Run paused")
        except Exception as e:
            self.handle_worker_error(f"Failed to pause run: {str(e)}")

    def resume_run(self):
        """Resume run"""
        if not self.robot or not self.current_run_id:
            QMessageBox.warning(self, "No Run", "No run to resume")
            return

        try:
            self.robot.resume_run(self.current_run_id)
            self.run_status_label.setText("Status: Running")
            self.log_message("Run resumed")
        except Exception as e:
            self.handle_worker_error(f"Failed to resume run: {str(e)}")

    def stop_run(self):
        """Stop run"""
        if not self.robot or not self.current_run_id:
            QMessageBox.warning(self, "No Run", "No run to stop")
            return

        try:
            self.robot.stop_run(self.current_run_id)
            self.run_status_label.setText("Status: Stopped")
            self.log_message("Run stopped")
        except Exception as e:
            self.handle_worker_error(f"Failed to stop run: {str(e)}")

    def refresh_runs(self):
        """Refresh run history"""
        if not self.robot:
            return

        try:
            runs = self.robot.get_run_list()
            self.runs_table.setRowCount(len(runs))

            for i, run in enumerate(runs):
                self.runs_table.setItem(
                    i, 0, QTableWidgetItem(run.get("id", "")))
                self.runs_table.setItem(
                    i, 1, QTableWidgetItem(run.get("status", "")))
                self.runs_table.setItem(
                    i, 2, QTableWidgetItem(run.get("createdAt", "")))

            self.log_message(f"Found {len(runs)} runs")
        except Exception as e:
            self.handle_worker_error(f"Failed to refresh runs: {str(e)}")

    def delete_selected_run(self):
        """Delete selected run"""
        if not self.robot:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to robot first")
            return

        current_row = self.runs_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "No Selection",
                                "Please select a run to delete")
            return

        run_id = self.runs_table.item(current_row, 0).text()

        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete run '{run_id}'?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                self.robot.delete_run(run_id)
                self.log_message(f"Run {run_id} deleted")
                self.refresh_runs()
            except Exception as e:
                self.handle_worker_error(f"Failed to delete run: {str(e)}")

    def home_robot(self):
        """Home all axes"""
        if not self.robot:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to robot first")
            return

        try:
            self.log_message("Homing robot...")
            self.robot.home()
            self.log_message("Robot homed successfully")
        except Exception as e:
            self.handle_worker_error(f"Failed to home robot: {str(e)}")

    def lights_on(self):
        """Turn lights on"""
        if not self.robot:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to robot first")
            return

        try:
            self.robot.lights_on()
            self.light_status_label.setText("On")
            self.log_message("Lights turned on")
        except Exception as e:
            self.handle_worker_error(f"Failed to turn lights on: {str(e)}")

    def lights_off(self):
        """Turn lights off"""
        if not self.robot:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to robot first")
            return

        try:
            self.robot.lights_off()
            self.light_status_label.setText("Off")
            self.log_message("Lights turned off")
        except Exception as e:
            self.handle_worker_error(f"Failed to turn lights off: {str(e)}")

    def flash_lights(self):
        """Flash lights"""
        if not self.robot:
            QMessageBox.warning(self, "Not Connected",
                                "Please connect to robot first")
            return

        count = self.flash_count_spin.value()

        try:
            self.log_message(f"Flashing lights {count} times...")
            self.robot.flash_lights(count)
            self.log_message("Lights flashed")
        except Exception as e:
            self.handle_worker_error(f"Failed to flash lights: {str(e)}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Set application icon if available
    # app.setWindowIcon(QIcon('icon.png'))

    window = OpentronFlexControlPanel()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
