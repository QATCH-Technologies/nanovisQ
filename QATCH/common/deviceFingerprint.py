"""
deviceFingerprint.py

This module provides functionality to generate unique device fingerprints based on
immutable Windows hardware information such as BIOS serial numbers, motherboard
serials, CPU IDs, disk serials, and system UUIDs. The generated fingerprints can
be used as license keys for software licensing purposes.

Author: 
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date: 
    2025-09-02
"""

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
        """Fallback logger implementation when QATCH logger is not available."""

        @staticmethod
        def w(msg: str) -> None:
            """Log a warning message.

            Args:
                msg: The warning message to log.
            """
            print(msg)

        @staticmethod
        def e(msg: str) -> None:
            """Log an error message.

            Args:
                msg: The error message to log.
            """
            print(msg)

        @staticmethod
        def i(msg: str) -> None:
            """Log an info message.

            Args:
                msg: The info message to log.
            """
            print(msg)


class DeviceFingerprint:
    """A utility class for generating device fingerprints based on hardware information.

    This class provides static methods to collect various hardware identifiers from
    Windows systems and generate unique license keys based on this information.
    All methods are static as this is a utility class that doesn't maintain state.
    """

    # These are class variables, with a shared state across DeviceFingerprint method calls
    no_powershell_cmds = False
    no_wmic_cmds = False

    @staticmethod
    def reset_failure_flags():
        """Provide option to reset failure flags when/if needed"""
        DeviceFingerprint.no_powershell_cmds = False
        DeviceFingerprint.no_wmic_cmds = False

    @staticmethod
    def run_command(command: str, shell: bool = True, use_powershell: bool = False) -> str:
        """Execute an arbitrary given system command string and return its output.

        WARNING: This method has security implications if misused. In its current usage,
        the caller passes in commands that are always hardcoded, with no opportunity for
        injection of malicious commands by users; however the function accepts arbitrary
        command strings which could potentially be a security risk if it is ever misused.

        Args:
            command: The command string to execute.
            shell: Whether to use shell execution (default: True).
            use_powershell: Whether to execute the command via PowerShell (default: False).

        Returns:
            The stripped output of the command, or empty string if the command fails.

        Note:
            Errors are suppressed and logged as warnings. STDERR is redirected to DEVNULL.
        """
        if use_powershell and DeviceFingerprint.no_powershell_cmds:
            Log.d(f"Skipping powershell command: {command}")
            return ""
        if command.lower().startswith("wmic") and DeviceFingerprint.no_wmic_cmds:
            Log.d(f"Skipping wmic utility command: {command}")
            return ""

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
            if use_powershell:
                DeviceFingerprint.no_powershell_cmds = True
            elif command.lower().startswith("wmic"):
                DeviceFingerprint.no_wmic_cmds = True
            return ""

    @staticmethod
    def query_registry(key_path: str, value_name: str, hive=winreg.HKEY_LOCAL_MACHINE) -> Optional[str]:
        """Query a Windows registry value.

        Args:
            key_path: The registry key path to query.
            value_name: The name of the value to retrieve.
            hive: The registry hive to use (default: HKEY_LOCAL_MACHINE).

        Returns:
            The registry value as a string if found, None otherwise.

        Note:
            Registry access errors are logged as warnings and None is returned.
        """
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
        """Get a unique hardware ID from various registry sources.

        This method queries multiple registry locations for hardware identifiers
        including MachineGuid, ProductId, and InstallDate.

        Returns:
            A pipe-separated string of hardware IDs, or "UNKNOWN" if none found.
        """
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
        """Get the BIOS serial number.

        This method tries multiple approaches to retrieve the BIOS serial number:
        1. PowerShell WMI query
        2. WMIC command
        3. Registry queries as fallback

        Returns:
            The BIOS serial number as a string, or "UNKNOWN" if not found.
        """
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
        """Get the motherboard serial number.

        This method attempts to retrieve the motherboard serial number using:
        1. PowerShell WMI query for baseboard
        2. WMIC command for baseboard
        3. Registry queries for system information
        4. Fallback to manufacturer-product combination

        Returns:
            The motherboard serial number as a string, or "UNKNOWN" if not found.
        """
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
        """Get the CPU processor ID.

        This method retrieves the CPU processor ID using multiple approaches:
        1. PowerShell WMI query
        2. WMIC command
        3. Registry queries for CPU information as fallback

        Returns:
            The CPU processor ID as a string, or "UNKNOWN" if not found.
        """
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
        """Get the primary disk serial number.

        This method attempts to retrieve the serial number of the primary disk (index 0)
        using various approaches:
        1. PowerShell WMI query for disk drives
        2. WMIC command for disk drives
        3. Get-Disk PowerShell cmdlet for unique ID
        4. Volume serial number as fallback

        Returns:
            The disk serial number as a string, or "UNKNOWN" if not found.
        """
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
        """Get the system UUID.

        This method retrieves the system UUID using multiple approaches:
        1. PowerShell WMI query for computer system product
        2. WMIC command for computer system product
        3. Registry queries for hardware ID and machine GUID as fallbacks

        Returns:
            The system UUID as a string, or "UNKNOWN" if not found.
        """
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
        """Generate a unique license key based on immutable Windows hardware information.

        This method collects various hardware identifiers and generates a SHA256-based
        license key. The key format is XXXX-XXXX-XXXX-XXXX-XXXX (20 characters + hyphens).

        The hardware information used includes:
        - BIOS serial number
        - Motherboard serial number  
        - CPU processor ID
        - Primary disk serial number
        - System UUID

        If all primary hardware queries fail, registry-based hardware ID is used as fallback.

        Returns:
            A formatted license key string (e.g., "1A2B-3C4D-5E6F-7890-ABCD").

        Note:
            The key will be consistent for the same device across multiple generations,
            making it suitable for software licensing purposes.
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

        device_string = json.dumps(device_info, sort_keys=True)
        hash_obj = hashlib.sha256(device_string.encode())
        full_hash = hash_obj.hexdigest().upper()
        key_parts = [full_hash[i:i+4] for i in range(0, 20, 4)]
        license_key = '-'.join(key_parts)

        return license_key

    @staticmethod
    def get_device_summary() -> Dict:
        """Get a comprehensive summary of device information.

        This method collects all available device information including computer name,
        OS version, and all hardware identifiers. It also generates the license key.

        Returns:
            A dictionary containing:
                - computer_name: The system hostname
                - os_version: Platform information string
                - bios_serial: BIOS serial number
                - motherboard_serial: Motherboard serial number
                - cpu_id: CPU processor ID
                - disk_serial: Primary disk serial number
                - system_uuid: System UUID
                - license_key: Generated license key
                - registry_hw_id: Registry hardware ID (if other methods fail)

        Note:
            If all hardware identification methods return "UNKNOWN", a registry-based
            hardware ID is included as a fallback.
        """
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
        """Get the device license key.

        This is a convenience method that generates and returns just the license key
        without the full device summary.

        Returns:
            The generated license key string, or None if generation fails.
        """
        try:
            summary = DeviceFingerprint.get_device_summary()
            key = summary.get("license_key", None)
            return key
        except Exception as e:
            Log.e(f"Device fingerprint could not be generated: {e}")
            return None
