import os

import shutil
from xml.dom import minidom
import datetime as dt
import hashlib
from PyQt5.QtCore import (
    QThread,
    pyqtSignal,
)
from typing import Optional, Any

from QATCH.core.constants import Constants
from QATCH.common.architecture import Architecture


class RecoveryWorker(QThread):
    """Handles the asynchronous process of writing run XML and executing recovery.

    This worker parses existing run metadata, updates or creates an XML record
    with machine metrics and audit trails, renames associated data files,
    and moves the final directory to the logged data storage.

    Attributes:
        progress (pyqtSignal): Emits int (0-100) representing completion percentage.
        finished_task (pyqtSignal): Emits (bool, str) representing (success, error_message).
    """

    progress = pyqtSignal(int)
    finished_task = pyqtSignal(bool, str)

    def __init__(
        self,
        run_metadata: Any,
        new_name: str,
        device_name: str,
        initials: str,
        parent: Optional[Any] = None,
    ) -> None:
        """Initializes the RecoveryWorker with necessary run details.

        Args:
            run_metadata: An object containing 'filepath', 'ruling', 'start', 'stop',
                'duration', and 'samples'.
            new_name (str): The new name for the run and its associated XML file.
            device_name (str): The name of the device associated with the run.
            initials (str): The initials of the user performing the recovery.
            parent (QObject, optional): The parent object for the QThread.
        """
        super().__init__(parent)
        self.run_metadata = run_metadata
        self.new_name = new_name
        self.device_name = device_name
        self.initials = initials

        self.logged_data_base = Constants.log_prefer_path  # user's preferred path

    def run(self):
        """Executes the recovery logic in a separate thread.

        Performs XML manipulation, hashing for integrity signatures, file renaming,
        and directory relocation.
        """
        # variables required for history logging:
        old_folder_path = "UNKNOWN"
        final_destination = "UNKNOWN"
        task_result = "UNKNOWN"
        try:
            self.progress.emit(10)
            old_folder_path = os.path.abspath(self.run_metadata.filepath)
            existing_xml_path = None

            # Locate existing XML
            for f in os.listdir(old_folder_path):
                if f.endswith(".xml"):
                    existing_xml_path = os.path.join(old_folder_path, f)
                    break

            if existing_xml_path and os.path.exists(existing_xml_path):
                audit_action = "PARAMS"
                run_doc = minidom.parse(existing_xml_path)
                xml = run_doc.documentElement
                if xml is not None:
                    for old_metrics in xml.getElementsByTagName("metrics"):
                        xml.removeChild(old_metrics)
                else:
                    xml = run_doc.createElement("run_info")
                    run_doc.appendChild(xml)
            else:
                # Logic for a brand new recovery XML
                audit_action = "CAPTURE"
                run_doc = minidom.Document()
                xml = run_doc.createElement("run_info")
                run_doc.appendChild(xml)

            # Set machine, device, name, and ruling field
            try:
                xml.setAttribute("machine", Architecture.get_os_name())
            except NameError:
                xml.setAttribute("machine", os.name)

            xml.setAttribute("device", self.device_name)
            xml.setAttribute("name", self.new_name)

            run_ruling = getattr(self.run_metadata, "ruling", "UNKNOWN")
            xml.setAttribute("ruling", run_ruling)

            # Generate top-level signature
            run_info_sig = hashlib.sha256(
                f"RUNINFO_{self.new_name}_{dt.datetime.now().isoformat()}".encode("utf-8")
            ).hexdigest()
            xml.setAttribute("signature", run_info_sig)

            # Build metrics block
            metrics = run_doc.createElement("metrics")

            def add_metric(name, value, units=None):
                m = run_doc.createElement("metric")
                m.setAttribute("name", name)
                m.setAttribute("value", str(value))
                if units:
                    m.setAttribute("units", units)
                metrics.appendChild(m)

            add_metric("calibrated", "UNKNOWN")
            start_val = getattr(self.run_metadata, "start", "UNKNOWN")
            add_metric("start", start_val)
            stop_val = getattr(self.run_metadata, "stop", "UNKNOWN")
            add_metric("stop", stop_val)

            raw_duration = getattr(self.run_metadata, "duration", 0.0)
            duration_units = "seconds"
            try:
                dur_float = float(raw_duration)
                if dur_float > 60.0:
                    dur_float /= 60.0
                    duration_units = "minutes"
                duration_str = f"{dur_float:2.4f}"
            except (ValueError, TypeError):
                duration_str = str(raw_duration)

            add_metric("duration", duration_str, duration_units)
            samples_val = getattr(self.run_metadata, "samples", "UNKNOWN")
            add_metric("samples", samples_val)

            metrics_sig = hashlib.sha256(
                f"METRICS_{start_val}_{stop_val}_{samples_val}".encode("utf-8")
            ).hexdigest()
            metrics.setAttribute("signature", metrics_sig)
            xml.appendChild(metrics)

            # Build audits block
            existing_audits = xml.getElementsByTagName("audits")
            if existing_audits:
                audits = existing_audits[0]
            else:
                audits = run_doc.createElement("audits")
                audits_root_sig = hashlib.sha256(
                    f"AUDITS_{dt.datetime.now().isoformat()}".encode("utf-8")
                ).hexdigest()
                audits.setAttribute("signature", audits_root_sig)
                xml.appendChild(audits)

            recorded_time = dt.datetime.now().isoformat()
            raw_sig_string = f"{audit_action}{recorded_time}{self.initials}"
            audit_sig = hashlib.sha256(raw_sig_string.encode("utf-8")).hexdigest()

            audit = run_doc.createElement("audit")
            audit.setAttribute("profile", getattr(self, "profile_id", "recovery_default"))
            audit.setAttribute("action", audit_action)
            audit.setAttribute("recorded", recorded_time)

            try:
                machine_name = Architecture.get_os_name()
            except NameError:
                machine_name = os.name

            audit.setAttribute("machine", machine_name)
            audit.setAttribute("username", getattr(self, "username", "System Administrator"))
            audit.setAttribute("initials", self.initials)
            audit.setAttribute("role", "ADMIN")
            audit.setAttribute("signature", audit_sig)
            audits.appendChild(audit)

            self.progress.emit(40)

            # File Operations: Write XML
            target_xml_path = os.path.join(old_folder_path, f"{self.new_name}.xml")
            with open(target_xml_path, "w", encoding="utf-8") as xml_file:
                run_doc.writexml(xml_file, indent="  ", addindent="  ", newl="\n")

            if existing_xml_path and os.path.abspath(existing_xml_path) != os.path.abspath(
                target_xml_path
            ):
                os.remove(existing_xml_path)

            self.progress.emit(60)

            # Rename Zip
            for file_name in os.listdir(old_folder_path):
                file_path = os.path.join(old_folder_path, file_name)
                if file_name.endswith(".zip") and file_name != "capture.zip":
                    os.rename(file_path, os.path.join(old_folder_path, "capture.zip"))

            self.progress.emit(80)

            # Move to destination
            target_device_dir = os.path.join(self.logged_data_base, self.device_name)
            os.makedirs(target_device_dir, exist_ok=True)
            final_destination = os.path.join(target_device_dir, self.new_name)

            if os.path.exists(final_destination):
                shutil.rmtree(final_destination)

            shutil.move(old_folder_path, final_destination)

            self.progress.emit(100)
            self.finished_task.emit(True, "")
            task_result = "SUCCESS"

        except Exception as e:
            self.finished_task.emit(False, str(e))
            task_result = "FAILURE"

        finally:
            history_path = os.path.join(
                os.getcwd(), Constants.log_export_path, "export_history.log"
            )
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    log_lines = f.read()
            else:
                log_lines = ""
            with open(history_path, "w") as f:
                f.write(
                    f"<b>Recovered 1 run(s) at {str(dt.datetime.now()).split('.')[0]}</b><br/>\n"
                )
                f.write(f'<small>from "{old_folder_path}" <br/>\n')
                f.write(f'to "{final_destination}"</small><br/>\n')
                f.write(f"<small>Settings: ")
                f.write(f'new_name = "{self.new_name}", ')
                f.write(f'device_name = "{self.device_name}", ')
                f.write(f'initials = "{self.initials}", ')
                f.write(f'result = "{task_result}"')
                f.write("</small><br/>\n<br/>\n")
                f.write(log_lines)
