import hashlib
import subprocess
import platform
import json
import winreg
from typing import Dict, Union, Optional


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
    def query_registry(key_path: str, value_name: str, hive=winreg.HKEY_LOCAL_MACHINE) -> Optional[str]:
        try:
            with winreg.OpenKey(hive, key_path, 0, winreg.KEY_READ) as key:
                value, _ = winreg.QueryValueEx(key, value_name)
                return str(value).strip() if value else None
        except Exception as e:
            Log.w(
                f"Registry query failed: {key_path}\\{value_name}, Error: {e}")
            return None

    @staticmethod
    def get_registry_hardware_id() -> str:
        """Get a unique hardware ID from various registry sources."""
        hardware_ids = []

        machine_guid = DeviceFingerprint.query_registry(
            r"SOFTWARE\Microsoft\Cryptography",
            "MachineGuid"
        )
        if machine_guid:
            hardware_ids.append(f"MachineGuid:{machine_guid}")

        product_id = DeviceFingerprint.query_registry(
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
            "ProductId"
        )
        if product_id:
            hardware_ids.append(f"ProductId:{product_id}")

        install_date = DeviceFingerprint.query_registry(
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
            "InstallDate"
        )
        if install_date:
            hardware_ids.append(f"InstallDate:{install_date}")

        return '|'.join(hardware_ids) if hardware_ids else "UNKNOWN"

    @staticmethod
    def get_bios_serial() -> str:
        serial = DeviceFingerprint.run_command(
            "Get-WmiObject -Class Win32_BIOS | Select-Object -ExpandProperty SerialNumber",
            use_powershell=True
        )

        if not serial or serial == "SerialNumber":
            output = DeviceFingerprint.run_command(
                "wmic bios get serialnumber")
            lines = output.split('\n')
            if len(lines) > 1:
                serial = lines[1].strip()
        if not serial or serial == "SerialNumber":
            serial = DeviceFingerprint.query_registry(
                r"HARDWARE\DESCRIPTION\System\BIOS",
                "SystemSerialNumber"
            )
            if not serial:
                serial = DeviceFingerprint.query_registry(
                    r"SYSTEM\CurrentControlSet\Control\SystemInformation",
                    "SystemSerialNumber"
                )

        return serial if serial and serial != "SerialNumber" else "UNKNOWN"

    @staticmethod
    def get_motherboard_serial() -> str:
        serial = DeviceFingerprint.run_command(
            "Get-WmiObject -Class Win32_BaseBoard | Select-Object -ExpandProperty SerialNumber",
            use_powershell=True
        )

        if not serial or serial == "SerialNumber":
            output = DeviceFingerprint.run_command(
                "wmic baseboard get serialnumber")
            lines = output.split('\n')
            if len(lines) > 1:
                serial = lines[1].strip()

        if not serial or serial == "SerialNumber":
            serial = DeviceFingerprint.query_registry(
                r"SYSTEM\CurrentControlSet\Control\SystemInformation",
                "SystemProductName"
            )
            if not serial:
                manufacturer = DeviceFingerprint.query_registry(
                    r"SYSTEM\CurrentControlSet\Control\SystemInformation",
                    "SystemManufacturer"
                )
                product = DeviceFingerprint.query_registry(
                    r"SYSTEM\CurrentControlSet\Control\SystemInformation",
                    "SystemProductName"
                )
                if manufacturer and product:
                    serial = f"{manufacturer}-{product}"

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

        # If still no CPU ID, try registry
        if not cpu_id or cpu_id == "ProcessorId":
            # Try to get CPU information from registry
            cpu_name = DeviceFingerprint.query_registry(
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                "ProcessorNameString"
            )
            cpu_identifier = DeviceFingerprint.query_registry(
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                "Identifier"
            )
            if cpu_name or cpu_identifier:
                cpu_id = f"{cpu_identifier or ''}-{cpu_name or ''}"

        return cpu_id if cpu_id and cpu_id != "ProcessorId" else "UNKNOWN"

    @staticmethod
    def get_disk_serial() -> str:
        serial = DeviceFingerprint.run_command(
            "Get-WmiObject -Class Win32_DiskDrive | Where-Object {$_.Index -eq 0} | Select-Object -ExpandProperty SerialNumber",
            use_powershell=True
        )

        if not serial or serial == "SerialNumber":
            output = DeviceFingerprint.run_command(
                "wmic diskdrive where Index=0 get serialnumber"
            )
            lines = output.split('\n')
            if len(lines) > 1:
                serial = lines[1].strip()

        if not serial or serial == "SerialNumber":
            serial = DeviceFingerprint.run_command(
                "Get-Disk | Where-Object {$_.Number -eq 0} | Select-Object -ExpandProperty UniqueId",
                use_powershell=True
            )

            if not serial:
                serial = DeviceFingerprint.run_command(
                    "vol C: | findstr Serial",
                    shell=True
                )
                if serial and "Serial" in serial:
                    # Extract just the serial number
                    parts = serial.split()
                    if len(parts) >= 3:
                        serial = parts[-1]

        if serial:
            serial = ' '.join(serial.split())

        return serial if serial and serial != "SerialNumber" else "UNKNOWN"

    @staticmethod
    def get_uuid() -> str:
        uuid = DeviceFingerprint.run_command(
            "Get-WmiObject -Class Win32_ComputerSystemProduct | Select-Object -ExpandProperty UUID",
            use_powershell=True
        )

        if not uuid or uuid == "UUID":
            output = DeviceFingerprint.run_command("wmic csproduct get uuid")
            lines = output.split('\n')
            if len(lines) > 1:
                uuid = lines[1].strip()

        if not uuid or uuid == "UUID":
            uuid = DeviceFingerprint.query_registry(
                r"SYSTEM\CurrentControlSet\Control\SystemInformation",
                "ComputerHardwareId"
            )

            if not uuid:
                uuid = DeviceFingerprint.query_registry(
                    r"SOFTWARE\Microsoft\Cryptography",
                    "MachineGuid"
                )

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

        if all(v == "UNKNOWN" for v in device_info.values()):
            Log.w("All primary hardware queries failed, using registry fallback")
            device_info['registry_hw_id'] = DeviceFingerprint.get_registry_hardware_id()

        # Log.i(f"Device info collected: {json.dumps(device_info, indent=2)}")

        device_string = json.dumps(device_info, sort_keys=True)
        hash_obj = hashlib.sha256(device_string.encode())
        full_hash = hash_obj.hexdigest().upper()
        key_parts = [full_hash[i:i+4] for i in range(0, 20, 4)]
        license_key = '-'.join(key_parts)

        return license_key

    @staticmethod
    def get_device_summary() -> Dict:
        try:
            computer_name = subprocess.check_output(
                "hostname", shell=True, text=True).strip()
        except:
            computer_name = "UNKNOWN"

        summary = {
            'computer_name': computer_name,
            'os_version': platform.platform(),
            'bios_serial': DeviceFingerprint.get_bios_serial(),
            'motherboard_serial': DeviceFingerprint.get_motherboard_serial(),
            'cpu_id': DeviceFingerprint.get_cpu_id(),
            'disk_serial': DeviceFingerprint.get_disk_serial(),
            'system_uuid': DeviceFingerprint.get_uuid(),
        }

        if all(v == "UNKNOWN" for k, v in summary.items() if k not in ['computer_name', 'os_version']):
            summary['registry_hw_id'] = DeviceFingerprint.get_registry_hardware_id()

        summary['license_key'] = DeviceFingerprint.generate_key()

        return summary

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
        # Print summary for debugging
        summary = DeviceFingerprint.get_device_summary()
        print(f"\nDevice Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    else:
        print("Failed to generate license key")
