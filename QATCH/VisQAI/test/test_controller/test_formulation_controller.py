# test/test_controller/test_formulation_controller.py

import unittest
from pathlib import Path

from src.db.db import Database
from src.controller.ingredient_controller import IngredientController
from src.controller.formulation_controller import FormulationController
from src.models.ingredient import Protein, Buffer, Salt, Surfactant, Stabilizer
from src.models.formulation import Formulation, Component, ViscosityProfile


class TestFormulationControllerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets_dir = Path(__file__).parent / "test_assets"
        cls.assets_dir.mkdir(parents=True, exist_ok=True)
        cls.db_file = cls.assets_dir / "test_app.db"

    def setUp(self):
        # remove any leftover DB
        if self.db_file.exists():
            self.db_file.unlink()
        # real database + controllers
        self.db = Database(self.db_file)
        self.ctrl = FormulationController(self.db)

    def tearDown(self):
        self.db.conn.close()
        if self.db_file.exists():
            self.db_file.unlink()

    def make_formulation(self, suffix=""):
        # create one of each ingredient
        p = Protein(
            enc_id=1, name=f"prot{suffix}", molecular_weight=50.0, pI_mean=7.0, pI_range=0.5)
        b = Buffer(enc_id=2, name=f"buf{suffix}", pH=7.4)
        s = Salt(enc_id=3, name=f"salt{suffix}")
        sf = Surfactant(enc_id=4, name=f"surf{suffix}")
        st = Stabilizer(enc_id=5, name=f"stab{suffix}")
        # no viscosity profile for now
        vp = ViscosityProfile([1, 10], [0.5, 0.3], "cP")
        formulation = Formulation()
        formulation.set_buffer(b, concentration=1.0, units="mg/mL")
        formulation.set_protein(p, concentration=2.0, units="mg/mL")
        formulation.set_salt(s, concentration=0.5, units="M")
        formulation.set_surfactant(sf, concentration=0.05, units="%w")
        formulation.set_stabilizer(st, concentration=0.1, units="M")
        formulation.set_temperature(25.0)
        formulation.set_viscosity_profile(vp)
        return formulation

    def test_add_and_retrieve_formulation(self):
        f = self.make_formulation()
        # add
        returned = self.ctrl.add_formulation(f)
        self.assertIs(returned, f)
        self.assertIsNotNone(
            f.id, "Formulation.id should be set by Database.add_formulation")

        # get_all_formulations
        all_forms = self.ctrl.get_all_formulations()
        self.assertEqual(len(all_forms), 1)
        self.assertEqual(all_forms[0], f)

        # get_formulation_by_id
        fetched = self.ctrl.get_formulation_by_id(f.id)
        self.assertEqual(fetched, f)

        # find_id
        found_id = self.ctrl.find_id(f)
        self.assertEqual(found_id, f.id)

    def test_add_duplicate_formulation(self):
        f1 = self.make_formulation()
        f2 = self.make_formulation()  # identical payload

        a1 = self.ctrl.add_formulation(f1)
        a2 = self.ctrl.add_formulation(f2)
        # should short-circuit and return existing, not create a second
        self.assertNotEqual(a1.id, a2.id)
        self.assertEqual(len(self.ctrl.get_all_formulations()), 2)

    def test_delete_formulation(self):
        f = self.make_formulation()
        self.ctrl.add_formulation(f)
        fid = f.id

        deleted = self.ctrl.delete_formulation_by_id(fid)
        self.assertEqual(deleted, f)

        # now gone
        self.assertEqual(self.ctrl.get_all_formulations(), [])
        self.assertIsNone(self.ctrl.get_formulation_by_id(fid))

    def test_delete_nonexistent_raises(self):
        with self.assertRaises(ValueError) as cm:
            self.ctrl.delete_formulation_by_id(999)
        self.assertIn("does not exist", str(cm.exception))

    def test_update_formulation(self):
        # add an original
        old = self.make_formulation()
        self.ctrl.add_formulation(old)
        old_id = old.id

        # make a slightly different one
        new = self.make_formulation(suffix="_new")
        updated = self.ctrl.update_formulation(old_id, new)
        self.assertIs(updated, new)

        all_forms = self.ctrl.get_all_formulations()
        self.assertIn(new, all_forms)
        self.assertNotIn(old, all_forms)

    def test_update_no_change(self):
        f = self.make_formulation()
        self.ctrl.add_formulation(f)
        fid = f.id

        same = self.ctrl.update_formulation(fid, f)
        self.assertIs(same, f)
        # still exactly one
        self.assertEqual(self.ctrl.get_all_formulations(), [f])

    def test_update_nonexistent_raises(self):
        f = self.make_formulation()
        with self.assertRaises(ValueError) as cm:
            self.ctrl.update_formulation(1234, f)
        self.assertIn("does not exist", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
