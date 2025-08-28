import hashlib
import subprocess
import platform
import json
from typing import Dict, Union

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
    def run_command(command: str, shell: bool = True, use_powershell: bool = False) -> str:
        """Run a command and return its output, with PowerShell option."""
        try:
            if use_powershell:
                ps_command = ['powershell', '-Command', command]
                output = subprocess.check_output(
                    ps_command, stderr=subprocess.DEVNULL, text=True)
            else:
                output = subprocess.check_output(
                    command, shell=shell, stderr=subprocess.DEVNULL, text=True)

            return output.strip()
        except Exception as e:
            Log.w(f"Command failed: {command}, Error: {e}")
            return ""

    @staticmethod
    def get_bios_serial() -> str:
        # Try PowerShell first (more reliable on modern Windows)
        serial = DeviceFingerprint.run_command(
            "Get-WmiObject -Class Win32_BIOS | Select-Object -ExpandProperty SerialNumber",
            use_powershell=True
        )

        if not serial or serial == "SerialNumber":
            # Fallback to WMIC
            output = DeviceFingerprint.run_command(
                "wmic bios get serialnumber")
            lines = output.split('\n')
            if len(lines) > 1:
                serial = lines[1].strip()

        return serial if serial and serial != "SerialNumber" else "UNKNOWN"

    @staticmethod
    def get_motherboard_serial() -> str:
        # Try PowerShell first
        serial = DeviceFingerprint.run_command(
            "Get-WmiObject -Class Win32_BaseBoard | Select-Object -ExpandProperty SerialNumber",
            use_powershell=True
        )

        if not serial or serial == "SerialNumber":
            # Fallback to WMIC
            output = DeviceFingerprint.run_command(
                "wmic baseboard get serialnumber")
            lines = output.split('\n')
            if len(lines) > 1:
                serial = lines[1].strip()

        return serial if serial and serial != "SerialNumber" else "UNKNOWN"

    @staticmethod
    def get_cpu_id() -> str:
        # Try PowerShell first
        cpu_id = DeviceFingerprint.run_command(
            "Get-WmiObject -Class Win32_Processor | Select-Object -ExpandProperty ProcessorId",
            use_powershell=True
        )

        if not cpu_id or cpu_id == "ProcessorId":
            # Fallback to WMIC
            output = DeviceFingerprint.run_command("wmic cpu get processorid")
            lines = output.split('\n')
            if len(lines) > 1:
                cpu_id = lines[1].strip()

        return cpu_id if cpu_id and cpu_id != "ProcessorId" else "UNKNOWN"

    @staticmethod
    def get_disk_serial() -> str:
        # Try PowerShell first
        serial = DeviceFingerprint.run_command(
            "Get-WmiObject -Class Win32_DiskDrive | Where-Object {$_.Index -eq 0} | Select-Object -ExpandProperty SerialNumber",
            use_powershell=True
        )

        if not serial or serial == "SerialNumber":
            # Fallback to WMIC
            output = DeviceFingerprint.run_command(
                "wmic diskdrive where Index=0 get serialnumber"
            )
            lines = output.split('\n')
            if len(lines) > 1:
                serial = lines[1].strip()

        # Clean up the serial number (remove extra spaces)
        if serial:
            serial = ' '.join(serial.split())

        return serial if serial and serial != "SerialNumber" else "UNKNOWN"

    @staticmethod
    def get_uuid() -> str:
        # Try PowerShell first
        uuid = DeviceFingerprint.run_command(
            "Get-WmiObject -Class Win32_ComputerSystemProduct | Select-Object -ExpandProperty UUID",
            use_powershell=True
        )

        if not uuid or uuid == "UUID":
            # Fallback to WMIC
            output = DeviceFingerprint.run_command("wmic csproduct get uuid")
            lines = output.split('\n')
            if len(lines) > 1:
                uuid = lines[1].strip()

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

        # Log the device info for debugging
        Log.i(f"Device info collected: {json.dumps(device_info, indent=2)}")

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
                "hostname", shell=True, text=True).strip()
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
        except Exception as e:
            Log.e(f"Device fingerprint could not be generated: {e}")
            return None


if __name__ == "__main__":
    key = DeviceFingerprint.get_key()
    if key:
        print(f"Generated license key: {key}")
    else:
        print("Failed to generate license key")
