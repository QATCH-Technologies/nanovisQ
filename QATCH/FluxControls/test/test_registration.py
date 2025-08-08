import unittest
import tempfile
import os
import json

# Import the module under test
import src.registration as registration_module
from src.registration import DeviceRegistration


class TestDeviceRegistration(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory and switch into it
        self.tmpdir = tempfile.TemporaryDirectory()
        self.orig_cwd = os.getcwd()
        os.chdir(self.tmpdir.name)

        # Backup the module‐level constant and original register_device
        self.orig_default_path = registration_module.DEFAULT_DEV_CONFIG_PATH
        self.orig_register = registration_module.DeviceRegistration.register_device

        # Point DEFAULT_DEV_CONFIG_PATH to a file in our temp dir
        registration_module.DEFAULT_DEV_CONFIG_PATH = "config.json"

    def tearDown(self):
        # Restore cwd and clean up
        os.chdir(self.orig_cwd)
        self.tmpdir.cleanup()

        # Restore patched names
        registration_module.DEFAULT_DEV_CONFIG_PATH = self.orig_default_path
        registration_module.DeviceRegistration.register_device = self.orig_register

    def test_is_valid_mac(self):
        # Valid MACs
        self.assertTrue(DeviceRegistration.is_valid_mac("AA:BB:CC:DD:EE:FF"))
        self.assertTrue(DeviceRegistration.is_valid_mac("aa-bb-cc-dd-ee-ff"))
        # Invalid MACs
        self.assertFalse(DeviceRegistration.is_valid_mac("GG:HH:II:JJ:KK:LL"))
        self.assertFalse(DeviceRegistration.is_valid_mac("1234567890AB"))
        self.assertFalse(DeviceRegistration.is_valid_mac(
            "AA:BB:CC:DD:EE"))  # too short

    def test_register_device_invalid_raises(self):
        # Bypass __init__, so no load_device on instantiation
        dr = DeviceRegistration.__new__(DeviceRegistration)
        dr._config_path = registration_module.DEFAULT_DEV_CONFIG_PATH
        with self.assertRaises(ValueError):
            dr.register_device("Name", "INVALID_MAC")

    def test_register_device_valid_and_persistence(self):
        dr = DeviceRegistration.__new__(DeviceRegistration)
        dr._config_path = registration_module.DEFAULT_DEV_CONFIG_PATH

        name = "MyDevice"
        mac = "01:23:45:67:89:AB"
        dr.register_device(name, mac)

        # Attributes set
        self.assertEqual(dr.get_device_name(), name)
        self.assertEqual(dr.get_mac_address(), mac)

        # File was written correctly
        with open(registration_module.DEFAULT_DEV_CONFIG_PATH, "r") as f:
            data = json.load(f)
        self.assertEqual(data, {"name": name, "mac": mac})

    def test_save_device_only(self):
        # Create a dr with a device dict, then save
        dr = DeviceRegistration.__new__(DeviceRegistration)
        dr.device = {"name": "X", "mac": "AA:AA:AA:AA:AA:AA"}
        dr._config_path = registration_module.DEFAULT_DEV_CONFIG_PATH

        dr.save_device()
        with open("config.json", "r") as f:
            data = json.load(f)
        self.assertEqual(data, dr.device)

    def test_load_device_existing_config(self):
        # Pre-write a config
        existing = {"name": "Loaded", "mac": "FF:EE:DD:CC:BB:AA"}
        with open("config.json", "w") as f:
            json.dump(existing, f)

        # instantiate — __init__ will call load_device()
        dr = DeviceRegistration()
        self.assertEqual(dr.device, existing)

    def test_load_device_missing_invokes_register(self):
        # Ensure file does not exist
        if os.path.exists("config.json"):
            os.remove("config.json")

        # Stub out register_device(self) to set a flag
        def stub_register(self):
            self.did_register = True
        registration_module.DeviceRegistration.register_device = stub_register

        dr = DeviceRegistration()
        self.assertTrue(hasattr(dr, "did_register") and dr.did_register)

    def test_config_path_setter_existing_file(self):
        dr = DeviceRegistration.__new__(DeviceRegistration)
        dr._config_path = None

        # Create an existing file
        open("already_there.json", "a").close()
        dr.config_path = "already_there.json"
        self.assertEqual(dr._config_path, "already_there.json")
        # No ./devices directory should be created
        self.assertFalse(os.path.isdir("devices"))

    def test_config_path_setter_new_file(self):
        dr = DeviceRegistration.__new__(DeviceRegistration)
        dr._config_path = None

        # Point at a non-existent path
        dr.config_path = "new_config.json"

        # Should create ./devices and devices/new_config.json
        self.assertTrue(os.path.isdir("devices"))
        expected = os.path.join(".", "devices", "new_config.json")
        self.assertTrue(os.path.isfile(expected))
        self.assertEqual(dr._config_path, expected)


if __name__ == "__main__":
    unittest.main()
