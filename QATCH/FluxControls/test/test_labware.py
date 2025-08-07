import unittest
import os

from src.labware import (
    Labware,
    Ordering,
    Brand,
    Metadata,
    Dimensions,
    Well,
    Wells,
    GroupMetadata,
    Group,
    Parameters,
    CornerOffsetFromSlot,
    StackingOffsetWithLabware,
)
from src.constants import DeckLocations
LABWARE_FILE = os.path.join('labware', 'nanovis_flux.json')


class TestLabwareModule(unittest.TestCase):
    def test_simple_classes(self):
        # Ordering
        o = Ordering([[1, 2], [3, 4]])
        self.assertEqual(o.ordering, [[1, 2], [3, 4]])

        # Brand
        b = Brand("b", ["x"])
        self.assertEqual(b.brand, "b")
        self.assertEqual(b.brand_id, ["x"])

        # Metadata
        m = Metadata("n", "c", "u", ["t"])
        self.assertEqual(m.display_name, "n")
        self.assertEqual(m.display_category, "c")
        self.assertEqual(m.display_volume_units, "u")
        self.assertEqual(m.tags, ["t"])

        # Dimensions
        d = Dimensions(1.0, 2.0, 3.0)
        self.assertEqual(d.x_dimension, 1.0)
        self.assertEqual(d.y_dimension, 2.0)
        self.assertEqual(d.z_dimension, 3.0)

        # Well and Wells
        wdict = {"A1": {"depth": 5.0, "totalLiquidVolume": 100.0,
                        "shape": "s", "diameter": 2.0,
                        "x": 1.1, "y": 2.2, "z": 3.3}}
        wells = Wells(wdict)
        self.assertIsInstance(wells.wells["A1"], Well)
        w = wells.wells["A1"]
        self.assertEqual(w.depth, 5.0)
        self.assertEqual(w.total_liquid_volume, 100.0)
        self.assertEqual(w.shape, "s")
        self.assertEqual(w.diameter, 2.0)
        self.assertEqual((w.x, w.y, w.z), (1.1, 2.2, 3.3))

        # GroupMetadata and Group
        gm = GroupMetadata("flat")
        self.assertEqual(gm.well_bottom_shape, "flat")
        g = Group({"wellBottomShape": "round"}, ["A1", "B1"])
        self.assertIsInstance(g.metadata, GroupMetadata)
        self.assertEqual(g.metadata.well_bottom_shape, "round")
        self.assertEqual(g.wells, ["A1", "B1"])

        # Parameters
        p = Parameters("f", ["q"], True, False, "ln")
        self.assertEqual(p.format, "f")
        self.assertEqual(p.quirks, ["q"])
        self.assertTrue(p.is_tiprack)
        self.assertFalse(p.is_magnetic_module_compatible)
        self.assertEqual(p.load_name, "ln")

        # CornerOffsetFromSlot
        cos = CornerOffsetFromSlot(1, 2, 3)
        self.assertEqual(cos.get_offsets(), {"x": 1, "y": 2, "z": 3})

        # StackingOffsetWithLabware
        sawl = StackingOffsetWithLabware({"k": {"x": 0.1, "y": 0.2, "z": 0.3}})
        self.assertEqual(
            sawl.stacking_offset_with_labware,
            {"k": {"x": 0.1, "y": 0.2, "z": 0.3}}
        )

    def test_load_json_real_file(self):
        # Should load a dict from the real labware file
        loaded = Labware.load_json(LABWARE_FILE)
        self.assertIsInstance(loaded, dict)
        # Basic expected keys
        for key in ["ordering", "brand", "metadata", "dimensions", "wells"]:
            self.assertIn(key, loaded)

    def test_labware_from_real_file(self):
        loc = DeckLocations.A1  # pick a valid enum member
        lw = Labware(loc, LABWARE_FILE)

        # Basic properties
        self.assertEqual(lw.location, loc)
        self.assertIsInstance(lw.display_name, str)
        self.assertIsInstance(lw.load_name, str)
        self.assertIsInstance(lw.name_space, str)
        self.assertIsInstance(lw.version, int)
        self.assertIsInstance(lw.is_tiprack, bool)

        # Offsets helper
        offs = lw.get_offsets()
        self.assertIsInstance(offs, dict)
        for axis in ("x", "y", "z"):
            self.assertIn(axis, offs)

        # Wells mapping
        self.assertIsInstance(lw.wells.wells, dict)
        # Ensure at least one well exists
        self.assertGreater(len(lw.wells.wells), 0)

    def test_invalid_definition_type(self):
        loc = list(DeckLocations)[0]
        with self.assertRaises(ValueError):
            Labware(loc, 12345)


if __name__ == "__main__":
    unittest.main()
