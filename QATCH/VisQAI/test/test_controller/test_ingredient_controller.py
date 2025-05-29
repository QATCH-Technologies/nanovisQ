# tests/test_ingredient_controller_integration.py

import unittest
from pathlib import Path

from src.models.ingredient import Protein, Buffer, Stabilizer, Surfactant, Salt
from src.db.db import Database
from src.controller.ingredient_controller import IngredientController


class TestIngredientControllerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # ensure our test_assets directory exists
        cls.assets_dir = Path(__file__).parent / "test_assets"
        cls.assets_dir.mkdir(parents=True, exist_ok=True)
        cls.db_file = cls.assets_dir / "test_app.db"

    def setUp(self):
        # delete any leftover DB from a previous run
        if self.db_file.exists():
            self.db_file.unlink()
        # create a fresh DB and controller
        self.db = Database(self.db_file)
        self.ctrl = IngredientController(self.db)

    def tearDown(self):
        # close the connection and remove the file
        self.db.conn.close()
        if self.db_file.exists():
            self.db_file.unlink()

    def _roundtrip(self, ing):
        """
        Helper: add via controller.add, then fetch back by name and return it.
        """
        self.ctrl.add(ing)
        fetched = self.ctrl.get_by_name(ing.name, ing)
        self.assertIsNotNone(fetched, f"Failed to fetch back {ing}")
        return fetched

    def _fetch_row(self, name: str):
        """Raw SQL helper to grab the (id, name, type) tuple."""
        return self.db.conn.execute(
            "SELECT id, name, type FROM ingredient WHERE name = ?", (name,)
        ).fetchone()

    def test_add_and_fetch_protein(self):
        p = Protein(enc_id=1, name="ProtA", molecular_weight=50.0,
                    pI_mean=6.5, pI_range=0.2)

        # 1) add via controller
        self.ctrl.add_protein(p)

        # 2) raw‐SQL: did it go in?
        row = self._fetch_row("ProtA")
        self.assertIsNotNone(row, "Protein row not found in DB")
        db_id, db_name, db_type = row
        self.assertEqual(db_name, "ProtA")
        self.assertEqual(db_type, "Protein")

        # 3) fetch by id
        fetched_by_id = self.ctrl.get_protein_by_id(db_id)
        self.assertIsInstance(fetched_by_id, Protein)
        self.assertEqual(fetched_by_id.id, db_id)
        self.assertEqual(fetched_by_id.name, "ProtA")
        self.assertAlmostEqual(fetched_by_id.molecular_weight, 50.0)

        # 4) fetch by name
        fetched_by_name = self.ctrl.get_protein_by_name("ProtA")
        self.assertEqual(fetched_by_name.id, db_id)
        self.assertEqual(fetched_by_name.name, "ProtA")

        # 5) duplicate‐name check
        with self.assertRaises(ValueError):
            self.ctrl.add_protein(
                Protein(enc_id=2, name="ProtA", molecular_weight=60.0,
                        pI_mean=7.0, pI_range=0.1)
            )

    def test_add_and_get_buffer(self):
        b = Buffer(enc_id=2, name="BufX", pH=7.4)
        fetched = self._roundtrip(b)
        self.assertIsInstance(fetched, Buffer)
        self.assertEqual(fetched.pH, 7.4)

        # delete by id then ensure it's gone
        self.ctrl.delete_buffer_by_id(fetched.id)
        self.assertIsNone(self.ctrl.get_buffer_by_name("BufX"))

    def test_add_and_get_salt(self):
        s = Salt(enc_id=3, name="SaltY")
        fetched = self._roundtrip(s)
        self.assertIsInstance(fetched, Salt)

        # delete by name then ensure it's gone
        self.ctrl.delete_salt_by_name("SaltY")
        self.assertIsNone(self.ctrl.get_salt_by_name("SaltY"))

    def test_add_and_get_surfactant(self):
        sf = Surfactant(enc_id=4, name="SurfZ")
        fetched = self._roundtrip(sf)
        self.assertIsInstance(fetched, Surfactant)

        # delete all surfactants
        self.ctrl.delete_all_surfactants()
        self.assertEqual(self.ctrl.get_all_surfactants(), [])

    def test_add_and_get_stabilizer(self):
        st = Stabilizer(enc_id=5, name="StabW")
        fetched = self._roundtrip(st)
        self.assertIsInstance(fetched, Stabilizer)

        # delete by id
        self.ctrl.delete_stabilizer_by_id(fetched.id)
        self.assertIsNone(self.ctrl.get_stabilizer_by_name("StabW"))

    def test_get_all_ingredients_and_delete_all(self):
        # add one of each
        types = [
            Protein(enc_id=10, name="P1", molecular_weight=20,
                    pI_mean=5, pI_range=0.1),
            Buffer(enc_id=11, name="B1", pH=7.1),
            Salt(enc_id=12, name="S1"),
            Surfactant(enc_id=13, name="SF1"),
            Stabilizer(enc_id=14, name="ST1"),
        ]
        for ing in types:
            self.ctrl.add(ing)

        all_ings = self.ctrl.get_all_ingredients()
        self.assertEqual({type(i) for i in all_ings},
                         set(type(i) for i in types))

        # now delete everything
        self.ctrl.delete_all_ingredients()
        self.assertEqual(self.ctrl.get_all_ingredients(), [])

    def test_update_protein(self):
        # add original
        original = Protein(enc_id=20, name="Orig",
                           molecular_weight=30, pI_mean=6, pI_range=0.2)
        self.ctrl.add_protein(original)
        fetched = self.ctrl.get_protein_by_name("Orig")
        self.assertIsNotNone(fetched)

        # prepare updated object with different name
        updated = Protein(enc_id=-1, name="Updated",
                          molecular_weight=35, pI_mean=6.3, pI_range=0.3)
        # silent DB name‐clash check: no other "Updated" in DB
        self.ctrl.update_protein(fetched.id, updated)

        # old name gone, new present
        self.assertIsNone(self.ctrl.get_protein_by_name("Orig"))
        up = self.ctrl.get_protein_by_name("Updated")
        self.assertIsNotNone(up)
        self.assertEqual(up.name, "Updated")
        # enc_id preserved
        self.assertEqual(up.enc_id, fetched.enc_id)

    def test_dispatch_methods_raise_on_bad_type(self):
        class Fake:
            type = "Nope"
            name = "x"

        fake = Fake()
        # every top‐level dispatch should blow up
        with self.assertRaises(ValueError):
            self.ctrl.get_by_id(1, fake)
        with self.assertRaises(ValueError):
            self.ctrl.get_by_name("x", fake)
        with self.assertRaises(ValueError):
            self.ctrl.get_by_type(fake)
        with self.assertRaises(ValueError):
            self.ctrl.delete_by_id(1, fake)
        with self.assertRaises(ValueError):
            self.ctrl.delete_by_name("x", fake)
        with self.assertRaises(ValueError):
            self.ctrl.delete_by_type(fake)
        with self.assertRaises(ValueError):
            self.ctrl.add(fake)
        with self.assertRaises(ValueError):
            self.ctrl.update(1, fake)


if __name__ == "__main__":
    unittest.main()
