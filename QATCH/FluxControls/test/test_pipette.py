# tests/test_pipette.py

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.pipette import Pipette
    from src.constants import Pipettes, MountPositions
except (ImportError, ModuleNotFoundError):
    from QATCH.FluxControls.src.pipette import Pipette
    from QATCH.FluxControls.src.constants import Pipettes, MountPositions


class TestPipetteProperties(unittest.TestCase):
    def setUp(self):
        # Use the first enum members for valid initialization
        self.valid_pip = next(iter(Pipettes))
        self.valid_mount = next(iter(MountPositions))
        self.p = Pipette(self.valid_pip, self.valid_mount)

    def test_initial_properties(self):
        """id defaults to None; pipette and mount_position return enum values."""
        self.assertIsNone(self.p.id)
        self.assertEqual(self.p.pipette, self.valid_pip.value)
        self.assertEqual(self.p.mount_position, self.valid_mount.value)

    def test_id_setter(self):
        """Setting id via property works."""
        self.p.id = 'pip123'
        self.assertEqual(self.p.id, 'pip123')
        # Changing again
        self.p.id = 456
        self.assertEqual(self.p.id, 456)

    def test_valid_pipette_property(self):
        """All Pipettes enum members accepted by pipette.setter."""
        for pip in Pipettes:
            self.p.pipette = pip
            self.assertEqual(self.p.pipette, pip.value)

    def test_invalid_pipette_property_raises(self):
        """Setting pipette property to invalid type raises ValueError."""
        invalid = next(iter(MountPositions))
        with self.assertRaises(ValueError) as cm:
            self.p.pipette = invalid
        self.assertEqual(str(cm.exception),
                         f"Invalid pipette tip {invalid.value}.")

    def test_valid_mount_property(self):
        """All MountPositions enum members accepted by mount_position.setter."""
        for mount in MountPositions:
            self.p.mount_position = mount
            self.assertEqual(self.p.mount_position, mount.value)

    def test_invalid_mount_property_raises(self):
        """Setting mount_position property to invalid type raises ValueError."""
        invalid = next(iter(Pipettes))
        with self.assertRaises(ValueError) as cm:
            self.p.mount_position = invalid
        self.assertEqual(str(cm.exception),
                         f"Invalid mount position {invalid.value}.")

    def test_is_valid_helpers(self):
        """_is_valid_pipette and _is_valid_mount_position work correctly."""
        # Valid pipettes
        for pip in Pipettes:
            self.assertTrue(self.p._is_valid_pipette(pip))
        # Invalid pipettes
        for mount in MountPositions:
            self.assertFalse(self.p._is_valid_pipette(mount))

        # Valid mounts
        for mount in MountPositions:
            self.assertTrue(self.p._is_valid_mount_position(mount))
        # Invalid mounts
        for pip in Pipettes:
            self.assertFalse(self.p._is_valid_mount_position(pip))


if __name__ == '__main__':
    unittest.main(verbosity=2)
