
import hashlib
import subprocess
import platform
import json
import os
from typing import Dict, Union
import dropbox
from dropbox.exceptions import ApiError

try:
    from QATCH.common.logger import Logger as Log
except (ModuleNotFoundError, ImportError):
    class Log:
        @staticmethod
        def w(msg: str) -> None:
            print(msg)

        @staticmethod
        def e(msg: str) -> None:
            print(msg)

        @staticmethod
        def i(msg: str) -> None:
            print(msg)


class DeviceFingerprint:
    @staticmethod
    def run_wmic_command(command: str) -> str:
        try:
            output = subprocess.check_output(
                command, shell=True, stderr=subprocess.DEVNULL)
            lines = output.decode('utf-8').strip().split('\n')
            if len(lines) > 1:
                return lines[1].strip()
        except Exception as e:
            Log.w(f"WMIC command failed: {command}, Error: {e}")
        return ""

    @staticmethod
    def get_bios_serial() -> str:
        serial = DeviceFingerprint.run_wmic_command(
            "wmic bios get serialnumber")
        return serial if serial and serial != "SerialNumber" else "UNKNOWN"

    @staticmethod
    def get_motherboard_serial() -> str:
        serial = DeviceFingerprint.run_wmic_command(
            "wmic baseboard get serialnumber")
        return serial if serial and serial != "SerialNumber" else "UNKNOWN"

    @staticmethod
    def get_cpu_id() -> str:
        cpu_id = DeviceFingerprint.run_wmic_command(
            "wmic cpu get processorid")
        return cpu_id if cpu_id and cpu_id != "ProcessorId" else "UNKNOWN"

    @staticmethod
    def get_disk_serial() -> str:
        serial = DeviceFingerprint.run_wmic_command(
            "wmic diskdrive where Index=0 get serialnumber"
        )
        return serial if serial and serial != "SerialNumber" else "UNKNOWN"

    @staticmethod
    def get_uuid() -> str:
        uuid = DeviceFingerprint.run_wmic_command(
            "wmic csproduct get uuid")
        return uuid if uuid and uuid != "UUID" else "UNKNOWN"

    @staticmethod
    def generate_key() -> str:
        """
        Generate a unique license key based on immutable Windows hardware information.
        This key will be consistent for the same device.
        """
        device_info = {
            'bios_serial': DeviceFingerprint.get_bios_serial(),
            'motherboard_serial': DeviceFingerprint.get_motherboard_serial(),
            'cpu_id': DeviceFingerprint.get_cpu_id(),
            'disk_serial': DeviceFingerprint.get_disk_serial(),
            'system_uuid': DeviceFingerprint.get_uuid()
        }
        device_string = json.dumps(device_info, sort_keys=True)
        hash_obj = hashlib.sha256(device_string.encode())
        full_hash = hash_obj.hexdigest().upper()
        key_parts = [full_hash[i:i+4] for i in range(0, 20, 4)]
        license_key = '-'.join(key_parts)

        return license_key

    @staticmethod
    def get_device_summary() -> Dict:
        """Get a summary of device information for display/logging"""
        try:
            computer_name = subprocess.check_output(
                "hostname", shell=True).decode().strip()
        except:
            computer_name = "UNKNOWN"

        return {
            'computer_name': computer_name,
            'os_version': platform.platform(),
            'bios_serial': DeviceFingerprint.get_bios_serial(),
            'motherboard_serial': DeviceFingerprint.get_motherboard_serial(),
            'cpu_id': DeviceFingerprint.get_cpu_id(),
            'disk_serial': DeviceFingerprint.get_disk_serial(),
            'system_uuid': DeviceFingerprint.get_uuid(),
            'license_key': DeviceFingerprint.generate_key()
        }

    @staticmethod
    def get_key() -> Union[str, None]:
        try:
            summary = DeviceFingerprint.get_device_summary()
            key = summary.get("license_key", None)
            return key
        except:
            Log.e("Device fingerprint could not be generated.")
            return None


if __name__ == "__main__":
    DeviceFingerprint.get_key()
