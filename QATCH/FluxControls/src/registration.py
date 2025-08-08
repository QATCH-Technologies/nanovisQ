import json
import re
import os
from typing import Union
try:
    from src.constants import DEFAULT_DEV_CONFIG_PATH

    class Log:
        def d(tag, msg=""): print("DEBUG:", tag, msg)
        def i(tag, msg=""): print("INFO:", tag, msg)
        def w(tag, msg=""): print("WARNING:", tag, msg)
        def e(tag, msg=""): print("ERROR:", tag, msg)
    Log.i(print("Running FluxControls as standalone app"))
except (ModuleNotFoundError, ImportError):
    from QATCH.common.logger import Logger as Log
    from QATCH.FluxControls.src.constants import CONFIG_FILE


class DeviceRegistration:
    """
    A class responsible for managing device registration, including loading and saving 
    device configuration, validating MAC addresses, and registering a new device.

    Attributes:
        device (dict): A dictionary holding the device information (name and MAC address).
    """

    def __init__(self, config_path: str = DEFAULT_DEV_CONFIG_PATH) -> None:
        """
        Initializes the DeviceRegistration class and attempts to load an existing device configuration.
        If no configuration exists, prompts the user to register a device.
        """
        self.device = {}
        self._config_path = config_path
        self.load_device()

    def load_device(self) -> Union[dict, None]:
        """
        Loads the device configuration from a file if it exists. If no configuration is found,
        it prompts the user to register a new device.

        If the device configuration file exists, it is loaded into the device attribute.
        If not, the user is prompted to enter device details.
        """
        if os.path.exists(DEFAULT_DEV_CONFIG_PATH):
            with open(DEFAULT_DEV_CONFIG_PATH, "r") as file:
                self.device = json.load(file)
                return self.device
        else:
            Log.i("No device configuration found!")
            self.register_device()

    def save_device(self) -> None:
        """
        Saves the current device configuration to a file.

        The device information (name and MAC address) is saved in a JSON format to the
        configuration file defined by CONFIG_FILE.
        """
        with open(DEFAULT_DEV_CONFIG_PATH, "w") as file:
            json.dump(self.device, file, indent=4)

    @staticmethod
    def is_valid_mac(mac: str) -> bool:
        """
        Validates the format of a given MAC address.

        Args:
            mac (str): The MAC address to validate.

        Returns:
            bool: True if the MAC address is valid, False otherwise.
        """
        pattern = re.compile(r"^([0-9A-Fa-f]{2}[-:]){5}[0-9A-Fa-f]{2}$")
        return bool(pattern.match(mac))

    def register_device(self, device_name: str, mac_address: str) -> None:
        """
        Registers a new device given its name and MAC address.
        Validates the MAC address format before saving.

        Args:
            device_name (str): The name of the device.
            mac_address (str): The MAC address in format XX:XX:XX:XX:XX:XX or XX-XX-XX-XX-XX-XX.

        Raises:
            ValueError: If the provided MAC address is not valid.
        """
        if not self.is_valid_mac(mac_address):
            raise ValueError(f"Invalid MAC address format: {mac_address}")

        self.device = {"name": device_name, "mac": mac_address}
        self.save_device()

    def get_mac_address(self) -> str:
        """
        Retrieves the MAC address of the registered device.

        Returns:
            str: The MAC address of the registered device, or None if not registered.
        """
        return self.device.get("mac")

    def get_device_name(self) -> str:
        """
        Retrieves the name of the registered device.

        Returns:
            str: The name of the registered device, or None if not registered.
        """
        return self.device.get("name")

    @property
    def config_path(self):
        return self._config_path

    @config_path.setter
    def config_path(self, config_path: str):
        if os.path.exists(config_path):
            self._config_path = config_path
            return

        devices_dir = os.path.join('.', 'devices')
        os.makedirs(devices_dir, exist_ok=True)
        filename = os.path.basename(config_path)
        new_path = os.path.join(devices_dir, filename)
        open(new_path, 'a').close()
        self._config_path = new_path
