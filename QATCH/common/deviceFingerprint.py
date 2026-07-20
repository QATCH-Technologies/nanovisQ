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

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid as _uuid_mod
from typing import Dict, Union, Optional

_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    import winreg
else:
    winreg = None


try:
    from QATCH.common.logger import Logger as Log
    from QATCH.core.constants import Constants
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
            """
            Log an informational message.

            Parameters:
                msg (str): The message to log.
            """
            print(msg)

        @staticmethod
        def d(msg: str) -> None:
            """Log a debug message.

            Args:
                msg: The debug message to log.
            """
            print(msg)

    class Constants:
        """Minimal fallback"""

        app_publisher = "QATCH"
        app_name = "nanovisQ"


class DeviceFingerprint:
    """A utility class for generating device fingerprints based on hardware information.

    This class provides static methods to collect various hardware identifiers from
    Windows systems and generate unique license keys based on this information.
    All methods are static as this is a utility class that doesn't maintain state.
    """

    # These are class variables, with a shared state across DeviceFingerprint method calls
    no_powershell_cmds = False
    no_wmic_cmds = False

    # Protects flag reads/writes in concurrent contexts
    _flag_lock = __import__("threading").Lock()

    @staticmethod
    def reset_failure_flags():
        """
        Reset persistent failure flags that disable PowerShell and WMIC command execution.

        Sets DeviceFingerprint.no_powershell_cmds and DeviceFingerprint.no_wmic_cmds to False under the internal lock to ensure a thread-safe update. Call this after environment changes or when retrying command-based probes that were previously disabled.

        No return value.
        """
        with DeviceFingerprint._flag_lock:
            DeviceFingerprint.no_powershell_cmds = False
            DeviceFingerprint.no_wmic_cmds = False

    @staticmethod
    def run_command(
        command: str, shell: bool = True, use_powershell: bool = False, timeout: float = 5.0
    ) -> str:
        """
        Execute a system command and return its trimmed stdout.

        This will run the provided command (optionally via PowerShell) and return the command's
        stdout with surrounding whitespace removed. STDERR is discarded. On failure the function
        returns an empty string and sets class-level failure flags to avoid repeating failing
        invocations: if a PowerShell invocation fails, DeviceFingerprint.no_powershell_cmds is set;
        if a WMIC command (commands starting with "wmic") fails, DeviceFingerprint.no_wmic_cmds is set.
        If the corresponding failure flag is already set, matching commands are skipped and an empty
        string is returned immediately. The use of creationflags with PowerShell commands is required
        to prevent console windows from flashing on the screen when running from a frozen EXE process.

        Security: accepting arbitrary command strings is potentially dangerous. Callers MUST ensure
        commands are not influenced by untrusted input.

        Parameters:
            command (str): Command string to execute.
            shell (bool): If True (default) run the command through the system shell when not using PowerShell.
            use_powershell (bool): If True, invoke PowerShell to run the command; failure will toggle the PowerShell failure flag.
            timeout (float): Timeout in seconds for the command (default 5.0).

        Returns:
            str: Trimmed stdout from the command on success, or an empty string on failure or when skipped.
        """
        with DeviceFingerprint._flag_lock:
            if use_powershell and DeviceFingerprint.no_powershell_cmds:
                Log.d(f"Skipping powershell command: {command}")
                return ""
            if command.lower().startswith("wmic") and DeviceFingerprint.no_wmic_cmds:
                Log.d(f"Skipping wmic utility command: {command}")
                return ""

        try:
            if use_powershell:
                ps_command = ["powershell", "-Command", command]
                if _IS_WINDOWS:
                    CREATE_NO_WINDOW = 0x08000000
                    output = subprocess.check_output(
                        ps_command,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        timeout=timeout,
                        creationflags=CREATE_NO_WINDOW,
                    )
                else:
                    output = subprocess.check_output(
                        ps_command, stderr=subprocess.DEVNULL, text=True, timeout=timeout
                    )
            else:
                output = subprocess.check_output(
                    command, shell=shell, stderr=subprocess.DEVNULL, text=True, timeout=timeout
                )

            return output.strip()

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, ValueError) as e:
            Log.w(f"Command failed: {command}, Error: {e}")
            with DeviceFingerprint._flag_lock:
                if use_powershell:
                    DeviceFingerprint.no_powershell_cmds = True
                elif command.lower().startswith("wmic"):
                    DeviceFingerprint.no_wmic_cmds = True
                return ""

    @staticmethod
    def query_registry(key_path: str, value_name: str, hive=None) -> Optional[str]:
        """Query a Windows registry value.

        Args:
            key_path: The registry key path to query.
            value_name: The name of the value to retrieve.
            hive: The registry hive to use (default: HKEY_LOCAL_MACHINE).

        Returns:
            The registry value as a string if found, None otherwise (also
            None on non-Windows platforms, where there is no registry).

        Note:
            Registry access errors are logged as warnings and None is returned.
        """
        if winreg is None:
            return None
        if hive is None:
            hive = winreg.HKEY_LOCAL_MACHINE
        try:
            with winreg.OpenKey(hive, key_path, 0, winreg.KEY_READ) as key:
                value, _ = winreg.QueryValueEx(key, value_name)
                return str(value).strip() if value else None
        except Exception as e:
            Log.w(f"Registry query failed: {key_path}\\{value_name}, Error: {e}")
            return None

    @staticmethod
    def write_reg_sz_value(root_key, subkey_path, value_name, value_data):
        if winreg is None:
            Log.d(f"Registry write of '{value_name}' skipped (not on Windows).")
            return
        try:
            # Open or create the key with write access
            try:
                key = winreg.OpenKey(root_key, subkey_path, 0, winreg.KEY_SET_VALUE)
            except FileNotFoundError:
                key = winreg.CreateKey(root_key, subkey_path)
                key = winreg.OpenKey(root_key, subkey_path, 0, winreg.KEY_SET_VALUE)

            # Set the REG_SZ value
            winreg.SetValueEx(key, value_name, 0, winreg.REG_SZ, value_data)
            Log.d(f"Successfully wrote REG_SZ value '{value_name}' to '{subkey_path}'")

        except Exception as e:
            Log.e(f"Error writing registry value: {e}")
        finally:
            if "key" in locals() and key:
                winreg.CloseKey(key)

    # ------------------------------------------------------------------
    # POSIX (Linux/macOS) settings store
    # ------------------------------------------------------------------
    # Windows persists the generated license key via the registry
    # (write_reg_sz_value/query_registry above). There is no registry on
    # POSIX platforms, so the same key/created/hash values are persisted to a
    # small JSON file under the user's XDG config directory instead - same
    # role, same three named values, just a different backing store.
    @staticmethod
    def _posix_settings_path() -> str:
        """Path to the per-user settings file used in place of the registry
        on Linux/macOS: `$XDG_CONFIG_HOME/{publisher}/{app}/device_settings.json`,
        falling back to `~/.config/...` when `XDG_CONFIG_HOME` isn't set.
        """
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
            os.path.expanduser("~"), ".config"
        )
        return os.path.join(
            base, Constants.app_publisher, Constants.app_name, "device_settings.json"
        )

    @staticmethod
    def _posix_read_value(value_name: str) -> Optional[str]:
        """POSIX counterpart to `query_registry`: reads one named value from
        the JSON settings file. Returns None if the file, or the value
        within it, doesn't exist."""
        try:
            with open(DeviceFingerprint._posix_settings_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            value = data.get(value_name)
            return str(value).strip() if value else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _posix_write_value(value_name: str, value_data: str) -> None:
        """POSIX counterpart to `write_reg_sz_value`: merges one named value
        into the JSON settings file, creating the file/directory as needed."""
        path = DeviceFingerprint._posix_settings_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {}
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                data = {}
            data[value_name] = value_data
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            Log.d(f"Successfully wrote '{value_name}' to '{path}'")
        except OSError as e:
            Log.e(f"Error writing settings value: {e}")

    # ------------------------------------------------------------------
    # POSIX (Linux) hardware identifiers
    # ------------------------------------------------------------------
    # Counterparts to the WMI/registry-based Windows probes below - read from
    # sysfs/procfs instead, which need no subprocess/root access for the
    # common case (a few DMI fields are root-only on some distros/BIOSes and
    # simply come back empty there, same as an unset WMI field on Windows).
    _DMI_PLACEHOLDERS = frozenset(
        {
            "",
            "none",
            "not specified",
            "to be filled by o.e.m.",
            "default string",
            "system serial number",
            "0123456789",
        }
    )

    @staticmethod
    def _read_dmi(name: str) -> Optional[str]:
        """Reads one DMI attribute from `/sys/class/dmi/id/<name>`."""
        try:
            with open(f"/sys/class/dmi/id/{name}", "r", encoding="utf-8", errors="ignore") as f:
                value = f.read().strip()
        except OSError:
            return None
        if value.lower() in DeviceFingerprint._DMI_PLACEHOLDERS:
            return None
        return value

    @staticmethod
    def _linux_bios_serial() -> str:
        return (
            DeviceFingerprint._read_dmi("bios_serial")
            or DeviceFingerprint._read_dmi("product_serial")
            or "UNKNOWN"
        )

    @staticmethod
    def _linux_motherboard_serial() -> str:
        serial = DeviceFingerprint._read_dmi("board_serial")
        if serial:
            return serial
        vendor = DeviceFingerprint._read_dmi("board_vendor")
        name = DeviceFingerprint._read_dmi("board_name")
        if vendor and name:
            return f"{vendor}-{name}"
        return "UNKNOWN"

    @staticmethod
    def _linux_cpu_id() -> str:
        """No Linux syscall/file exposes the raw CPUID string the way Windows'
        Win32_Processor.ProcessorId does, so this hashes a handful of stable
        identifying fields from /proc/cpuinfo into a deterministic surrogate
        (falls back to a real per-board serial on ARM/Raspberry Pi, which
        /proc/cpuinfo does expose directly)."""
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            return "UNKNOWN"

        fields: Dict[str, str] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()
            if key == "serial" and value and set(value) != {"0"}:
                return value  # ARM/RPi: a real per-board serial
            if key in ("model name", "vendor_id", "cpu family", "model", "stepping"):
                fields.setdefault(key, value)

        if not fields:
            return "UNKNOWN"
        digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest().upper()
        return digest[:24]

    @staticmethod
    def _linux_disk_serial() -> str:
        """Serial of the first real (non-loop/ram/optical) block device."""
        block_dir = "/sys/block"
        try:
            devices = sorted(os.listdir(block_dir))
        except OSError:
            return "UNKNOWN"

        for dev in devices:
            if dev.startswith(("loop", "ram", "sr", "zram")):
                continue
            for rel in ("device/serial", "serial"):
                try:
                    with open(
                        os.path.join(block_dir, dev, rel), "r", encoding="utf-8", errors="ignore"
                    ) as f:
                        serial = f.read().strip()
                    if serial:
                        return serial
                except OSError:
                    continue
        return "UNKNOWN"

    @staticmethod
    def _linux_uuid() -> str:
        """systemd's machine-id is the standard, world-readable, per-install
        unique identifier on Linux (no root needed, unlike DMI product_uuid
        on many distros) - preferred over the DMI UUID for that reason."""
        for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    value = f.read().strip()
                if value:
                    return value
            except OSError:
                continue

        dmi_uuid = DeviceFingerprint._read_dmi("product_uuid")
        if dmi_uuid:
            return dmi_uuid

        # Last resort: MAC-address-derived node ID (stable unless the NIC changes).
        return f"NODE-{_uuid_mod.getnode():012X}"

    @staticmethod
    def _linux_os_hardware_id() -> str:
        """POSIX counterpart to `get_registry_hardware_id`: OS-install
        identity from /etc/machine-id and /etc/os-release, used as the same
        last-resort fallback when every primary hardware probe fails."""
        hardware_ids = []

        machine_id = None
        for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    machine_id = f.read().strip()
                if machine_id:
                    break
            except OSError:
                continue
        if machine_id:
            hardware_ids.append(f"MachineId:{machine_id}")

        os_release = {}
        try:
            with open("/etc/os-release", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "=" in line:
                        key, _, value = line.strip().partition("=")
                        os_release[key] = value.strip('"')
        except OSError:
            pass
        if os_release.get("VERSION_ID"):
            hardware_ids.append(f"VersionId:{os_release['VERSION_ID']}")
        if os_release.get("BUILD_ID"):
            hardware_ids.append(f"BuildId:{os_release['BUILD_ID']}")

        return "|".join(hardware_ids) if hardware_ids else "UNKNOWN"

    @staticmethod
    def get_registry_hardware_id() -> str:
        """Get a unique hardware/OS-install ID, used as a last-resort
        fallback when every primary hardware probe returns "UNKNOWN".

        On Windows this queries multiple registry locations for hardware
        identifiers including MachineGuid, ProductId, and InstallDate. On
        Linux/macOS it delegates to `_linux_os_hardware_id`
        (/etc/machine-id + /etc/os-release) instead - there is no registry.

        Returns:
            A pipe-separated string of hardware IDs, or "UNKNOWN" if none found.
        """
        if winreg is None:
            return DeviceFingerprint._linux_os_hardware_id()

        hardware_ids = []

        machine_guid = DeviceFingerprint.query_registry(
            r"SOFTWARE\Microsoft\Cryptography", "MachineGuid"
        )
        if machine_guid:
            hardware_ids.append(f"MachineGuid:{machine_guid}")

        product_id = DeviceFingerprint.query_registry(
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion", "ProductId"
        )
        if product_id:
            hardware_ids.append(f"ProductId:{product_id}")

        install_date = DeviceFingerprint.query_registry(
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion", "InstallDate"
        )
        if install_date:
            hardware_ids.append(f"InstallDate:{install_date}")

        return "|".join(hardware_ids) if hardware_ids else "UNKNOWN"

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
        if winreg is None:
            return DeviceFingerprint._linux_bios_serial()

        serial = DeviceFingerprint.run_command(
            "Get-WmiObject -Class Win32_BIOS | Select-Object -ExpandProperty SerialNumber",
            use_powershell=True,
        )

        if not serial or serial == "SerialNumber":
            output = DeviceFingerprint.run_command("wmic bios get serialnumber")
            lines = output.split("\n")
            if len(lines) > 1:
                serial = lines[1].strip()
        if not serial or serial == "SerialNumber":
            serial = DeviceFingerprint.query_registry(
                r"HARDWARE\DESCRIPTION\System\BIOS", "SystemSerialNumber"
            )
            if not serial:
                serial = DeviceFingerprint.query_registry(
                    r"SYSTEM\CurrentControlSet\Control\SystemInformation", "SystemSerialNumber"
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
        if winreg is None:
            return DeviceFingerprint._linux_motherboard_serial()

        serial = DeviceFingerprint.run_command(
            "Get-WmiObject -Class Win32_BaseBoard | Select-Object -ExpandProperty SerialNumber",
            use_powershell=True,
        )

        if not serial or serial == "SerialNumber":
            output = DeviceFingerprint.run_command("wmic baseboard get serialnumber")
            lines = output.split("\n")
            if len(lines) > 1:
                serial = lines[1].strip()

        if not serial or serial == "SerialNumber":
            serial = DeviceFingerprint.query_registry(
                r"SYSTEM\CurrentControlSet\Control\SystemInformation", "SystemProductName"
            )
            if not serial:
                manufacturer = DeviceFingerprint.query_registry(
                    r"SYSTEM\CurrentControlSet\Control\SystemInformation", "SystemManufacturer"
                )
                product = DeviceFingerprint.query_registry(
                    r"SYSTEM\CurrentControlSet\Control\SystemInformation", "SystemProductName"
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
        if winreg is None:
            return DeviceFingerprint._linux_cpu_id()

        # Try PowerShell first
        cpu_id = DeviceFingerprint.run_command(
            "Get-WmiObject -Class Win32_Processor | Select-Object -ExpandProperty ProcessorId",
            use_powershell=True,
        )

        if not cpu_id or cpu_id == "ProcessorId":
            # Fallback to WMIC
            output = DeviceFingerprint.run_command("wmic cpu get processorid")
            lines = output.split("\n")
            if len(lines) > 1:
                cpu_id = lines[1].strip()

        if not cpu_id or cpu_id == "ProcessorId":
            # Try to get CPU information from registry
            cpu_name = DeviceFingerprint.query_registry(
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0", "ProcessorNameString"
            )
            cpu_identifier = DeviceFingerprint.query_registry(
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0", "Identifier"
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
        if winreg is None:
            return DeviceFingerprint._linux_disk_serial()

        serial = DeviceFingerprint.run_command(
            "Get-WmiObject -Class Win32_DiskDrive | Where-Object {$_.Index -eq 0} | Select-Object -ExpandProperty SerialNumber",
            use_powershell=True,
        )

        if not serial or serial == "SerialNumber":
            output = DeviceFingerprint.run_command("wmic diskdrive where Index=0 get serialnumber")
            lines = output.split("\n")
            if len(lines) > 1:
                serial = lines[1].strip()

        if not serial or serial == "SerialNumber":
            serial = DeviceFingerprint.run_command(
                "Get-Disk | Where-Object {$_.Number -eq 0} | Select-Object -ExpandProperty UniqueId",
                use_powershell=True,
            )

            if not serial:
                serial = DeviceFingerprint.run_command("vol C: | findstr Serial", shell=True)
                if serial and "Serial" in serial:
                    # Extract just the serial number
                    parts = serial.split()
                    if len(parts) >= 3:
                        serial = parts[-1]

        if serial:
            serial = " ".join(serial.split())

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
        if winreg is None:
            return DeviceFingerprint._linux_uuid()

        uuid = DeviceFingerprint.run_command(
            "Get-WmiObject -Class Win32_ComputerSystemProduct | Select-Object -ExpandProperty UUID",
            use_powershell=True,
        )

        if not uuid or uuid == "UUID":
            output = DeviceFingerprint.run_command("wmic csproduct get uuid")
            lines = output.split("\n")
            if len(lines) > 1:
                uuid = lines[1].strip()

        if not uuid or uuid == "UUID":
            uuid = DeviceFingerprint.query_registry(
                r"SYSTEM\CurrentControlSet\Control\SystemInformation", "ComputerHardwareId"
            )

            if not uuid:
                uuid = DeviceFingerprint.query_registry(
                    r"SOFTWARE\Microsoft\Cryptography", "MachineGuid"
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
            "bios_serial": DeviceFingerprint.get_bios_serial(),
            "motherboard_serial": DeviceFingerprint.get_motherboard_serial(),
            "cpu_id": DeviceFingerprint.get_cpu_id(),
            "disk_serial": DeviceFingerprint.get_disk_serial(),
            "system_uuid": DeviceFingerprint.get_uuid(),
        }

        if all(v == "UNKNOWN" for v in device_info.values()):
            Log.w("All primary hardware queries failed, using registry fallback")
            device_info["registry_hw_id"] = DeviceFingerprint.get_registry_hardware_id()

        device_string = json.dumps(device_info, sort_keys=True)
        hash_obj = hashlib.sha256(device_string.encode())
        full_hash = hash_obj.hexdigest().upper()
        key_parts = [full_hash[i : i + 4] for i in range(0, 20, 4)]
        license_key = "-".join(key_parts)
        license_created = str(int(time.time()))
        license_hash = DeviceFingerprint.get_license_hash(license_key, license_created)

        # Persist the generated key info to AppSettings: the registry on
        # Windows, or the POSIX settings JSON file (see _posix_write_value)
        # everywhere else.
        if winreg is not None:
            subkey_path = r"SOFTWARE\{ORG}\{APP}".format(
                ORG=Constants.app_publisher, APP=Constants.app_name
            )
            DeviceFingerprint.write_reg_sz_value(
                root_key=winreg.HKEY_CURRENT_USER,
                subkey_path=subkey_path,
                value_name="License_Key",
                value_data=license_key,
            )
            DeviceFingerprint.write_reg_sz_value(
                root_key=winreg.HKEY_CURRENT_USER,
                subkey_path=subkey_path,
                value_name="License_Created",
                value_data=license_created,
            )
            DeviceFingerprint.write_reg_sz_value(
                root_key=winreg.HKEY_CURRENT_USER,
                subkey_path=subkey_path,
                value_name="License_Hash",
                value_data=license_hash,
            )
        else:
            DeviceFingerprint._posix_write_value("License_Key", license_key)
            DeviceFingerprint._posix_write_value("License_Created", license_created)
            DeviceFingerprint._posix_write_value("License_Hash", license_hash)

        return license_key

    @staticmethod
    def get_license_hash(key: str, created: str) -> str:
        if isinstance(key, str) and isinstance(created, str) and created.isnumeric():
            try:
                license_bytes = (
                    f"{key}@{datetime.datetime.fromtimestamp(int(created)).isoformat()}Z".encode(
                        "utf-8"
                    )
                )
                hash_str = hashlib.sha256(license_bytes).hexdigest()
                return hash_str

            except ValueError:
                return "ValueError"
            except Exception:
                return "Exception"
        else:
            return "TypeError"

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
            computer_name = subprocess.check_output("hostname", shell=True, text=True).strip()
        except:
            computer_name = "UNKNOWN"

        summary = {
            "computer_name": computer_name,
            "os_version": platform.platform(),
            "bios_serial": DeviceFingerprint.get_bios_serial(),
            "motherboard_serial": DeviceFingerprint.get_motherboard_serial(),
            "cpu_id": DeviceFingerprint.get_cpu_id(),
            "disk_serial": DeviceFingerprint.get_disk_serial(),
            "system_uuid": DeviceFingerprint.get_uuid(),
        }

        if all(
            v == "UNKNOWN" for k, v in summary.items() if k not in ["computer_name", "os_version"]
        ):
            summary["registry_hw_id"] = DeviceFingerprint.get_registry_hardware_id()

        summary["license_key"] = DeviceFingerprint.generate_key()

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
            if winreg is not None:
                subkey_path = r"SOFTWARE\{ORG}\{APP}".format(
                    ORG=Constants.app_publisher, APP=Constants.app_name
                )
                license_key = DeviceFingerprint.query_registry(
                    hive=winreg.HKEY_CURRENT_USER, key_path=subkey_path, value_name="License_Key"
                )
                license_created = DeviceFingerprint.query_registry(
                    hive=winreg.HKEY_CURRENT_USER,
                    key_path=subkey_path,
                    value_name="License_Created",
                )
                license_hash = DeviceFingerprint.query_registry(
                    hive=winreg.HKEY_CURRENT_USER, key_path=subkey_path, value_name="License_Hash"
                )
            else:
                license_key = DeviceFingerprint._posix_read_value("License_Key")
                license_created = DeviceFingerprint._posix_read_value("License_Created")
                license_hash = DeviceFingerprint._posix_read_value("License_Hash")

            if license_key and license_created and license_hash:
                calculated_hash = DeviceFingerprint.get_license_hash(license_key, license_created)
            else:
                # indicate missing (or partial) key info
                calculated_hash = "Missing"  # NOTE: Do NOT use `None` here!
                if license_hash == calculated_hash:
                    # This could occur if user sets "License_Hash"="Missing"
                    # without specifying "License_Created" in AppSettings...
                    # The result, if allowed to proceed, would be a manually
                    # set "License_Key" being used resulting in a spoof hack
                    license_hash = "Invalid"  # Prevent hack attempt
                    license_key = None  # Invalidate key

            key = None  # assume failure by default
            if license_hash == calculated_hash:
                Log.d(f'Pulled a VALID license key "{license_key}" from AppSettings')
                key = license_key  # success
            elif not license_key:
                Log.w("No valid license key found in AppSettings")
                Log.w("Generating a license key for this device...")
            else:
                Log.e("Pulled an INVALID license key from AppSettings")
                Log.e("Generating a license key for this device...")

            if not key:
                # Key not available in registry, generate a new one (with timestamp and signature)
                summary = DeviceFingerprint.get_device_summary()
                key = summary.get("license_key", None)

            return key
        except Exception as e:
            Log.e(f"Device fingerprint could not be generated: {e}")
            return None
