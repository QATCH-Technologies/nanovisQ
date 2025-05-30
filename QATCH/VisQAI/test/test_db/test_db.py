import unittest
from pathlib import Path

from src.models.ingredient import Protein, Buffer, Stabilizer, Surfactant, Salt
from src.models.formulation import Formulation, ViscosityProfile
from src.db.db import Database


class BaseTestDatabase(unittest.TestCase):
    encryption_key = None  # Default (no encryption)

    @classmethod
    def setUpClass(cls):
        cls.assets_dir = Path(__file__).parent / "test_assets"
        cls.assets_dir.mkdir(parents=True, exist_ok=True)
        cls.db_file = cls.assets_dir / "test_app.db"

    def setUp(self):
        if self.db_file.exists():
            self.db_file.unlink()
        self.db = Database(self.db_file, self.encryption_key)

    def tearDown(self):
        self.db.close()
        self.assertTrue(self.db_file.exists())
        if self.db_file.exists():
            self.db_file.unlink()

    # --- Ingredient tests ---

    def test_add_and_get_each_ingredient_type(self):
        p = Protein(enc_id=10, name="ProtA", molecular_weight=50.0,
                    pI_mean=6.5, pI_range=0.3)
        b = Buffer(enc_id=20, name="BuffB", pH=7.4)
        s1 = Stabilizer(enc_id=30, name="StabC")
        s2 = Surfactant(enc_id=40, name="SurfD")
        salt = Salt(enc_id=50, name="SaltE")

        ids = {}
        for ing in (p, b, s1, s2, salt):
            iid = self.db.add_ingredient(ing)
            ids[type(ing).__name__] = iid

        got_p = self.db.get_ingredient(ids['Protein'])
        self.assertIsInstance(got_p, Protein)
        self.assertEqual(got_p.enc_id, 10)
        self.assertAlmostEqual(got_p.molecular_weight, 50.0)

        got_b = self.db.get_ingredient(ids['Buffer'])
        self.assertIsInstance(got_b, Buffer)
        self.assertAlmostEqual(got_b.pH, 7.4)

        self.assertIsInstance(self.db.get_ingredient(
            ids['Stabilizer']), Stabilizer)
        self.assertIsInstance(self.db.get_ingredient(
            ids['Surfactant']), Surfactant)
        self.assertIsInstance(self.db.get_ingredient(ids['Salt']), Salt)

    def test_get_all_and_delete_ingredient(self):
        p = Protein(enc_id=1, name="X", molecular_weight=10,
                    pI_mean=5, pI_range=0.1)
        self.db.add_ingredient(p)
        self.db.add_ingredient(p)
        all_ings = self.db.get_all_ingredients()
        self.assertEqual(len(all_ings), 2)
        self.assertTrue(self.db.delete_ingredient(1))
        remaining = self.db.get_all_ingredients()
        self.assertEqual(len(remaining), 1)

        # delete all
        self.db.delete_all_ingredients()
        self.assertEqual(self.db.get_all_ingredients(), [])

    def test_update_ingredient(self):
        b = Buffer(enc_id=99, name="OldBuff", pH=6.0)
        iid = self.db.add_ingredient(b)
        newb = Buffer(enc_id=123, name="NewBuff", pH=8.2)
        self.assertTrue(self.db.update_ingredient(iid, newb))
        got = self.db.get_ingredient(iid)
        self.assertIsInstance(got, Buffer)
        self.assertEqual(got.enc_id, 123)
        self.assertAlmostEqual(got.pH, 8.2)
        self.assertFalse(self.db.update_ingredient(9999, newb))

    # --- Formulation tests ---

    def make_sample_formulation(self):
        f = Formulation()
        p = Protein(enc_id=1, name="ProtA", molecular_weight=20,
                    pI_mean=6, pI_range=0.2)
        f.set_protein(p, concentration=1.5, units="mg/mL")
        b = Buffer(enc_id=2, name="BuffB", pH=7.0)
        f.set_buffer(b, concentration=2.0, units="mM")
        vp = ViscosityProfile([1, 10], [0.1, 0.05], units="Pa·s")
        vp.is_measured = True
        f.set_viscosity_profile(vp)
        f.set_temperature(37.0)
        return f

    def test_add_and_get_formulation(self):
        form = self.make_sample_formulation()
        fid = self.db.add_formulation(form)
        got = self.db.get_formulation(fid)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got.temperature, 37.0)

        prot_comp = got._components['protein']
        self.assertEqual(prot_comp.concentration, 1.5)
        self.assertIsInstance(prot_comp.ingredient, Protein)

        buf_comp = got._components['buffer']
        self.assertEqual(buf_comp.concentration, 2.0)
        self.assertIsInstance(buf_comp.ingredient, Buffer)

        self.assertTrue(got.viscosity_profile.is_measured)
        self.assertListEqual(got.viscosity_profile.shear_rates, [1, 10])

    def test_get_all_and_delete_formulation(self):
        self.db.add_formulation(self.make_sample_formulation())
        self.db.add_formulation(self.make_sample_formulation())
        all_forms = self.db.get_all_formulations()
        self.assertEqual(len(all_forms), 2)

        fid = all_forms[0].id
        self.assertTrue(self.db.delete_formulation(fid))
        self.assertEqual(len(self.db.get_all_formulations()), 1)

        self.db.delete_all_formulations()
        self.assertEqual(self.db.get_all_formulations(), [])

    def test_components_linked_to_existing_ingredients(self):
        form = self.make_sample_formulation()
        fid = self.db.add_formulation(form)

        # fetch all component entries for that formulation
        c = self.db.conn.cursor()
        c.execute(
            "SELECT component_type, ingredient_id "
            "FROM formulation_component WHERE formulation_id = ?",
            (fid,)
        )
        comp_rows = c.fetchall()

        # should have exactly one entry per non-None component in form._components
        expected_types = [
            k for k, v in form._components.items() if v is not None]
        self.assertEqual(
            {r[0] for r in comp_rows},
            set(expected_types),
            "Formulation_component rows don't match form._components keys"
        )

        # each ingredient_id must exist in ingredient table and match original enc_id/name
        for comp_type, iid in comp_rows:
            c.execute("SELECT enc_id, name FROM ingredient WHERE id = ?", (iid,))
            row = c.fetchone()
            self.assertIsNotNone(
                row,
                f"Component '{comp_type}' references missing ingredient id {iid}"
            )
            enc_id_db, name_db = row
            orig_ing = form._components[comp_type].ingredient
            self.assertEqual(
                enc_id_db,
                orig_ing.enc_id,
                f"enc_id mismatch for component '{comp_type}'"
            )
            self.assertEqual(
                name_db,
                orig_ing.name,
                f"name mismatch for component '{comp_type}'"
            )

    def test_database_file_reload(self):
        form = self.make_sample_formulation()
        fid = self.db.add_formulation(form)

        # write database to file, load it back again from file
        self.db.close()
        self.assertTrue(self.db_file.exists())
        self.db = Database(self.db_file, self.encryption_key)

        # get the formulation, make sure it matches what was added
        got = self.db.get_formulation(fid)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got.temperature, 37.0)

        prot_comp = got._components['protein']
        self.assertEqual(prot_comp.concentration, 1.5)
        self.assertIsInstance(prot_comp.ingredient, Protein)

        buf_comp = got._components['buffer']
        self.assertEqual(buf_comp.concentration, 2.0)
        self.assertIsInstance(buf_comp.ingredient, Buffer)

        self.assertTrue(got.viscosity_profile.is_measured)
        self.assertListEqual(got.viscosity_profile.shear_rates, [1, 10])

    def test_database_file_write_on_close_if_missing(self):
        self.test_database_file_reload()
        self.db.close()  # release file lock
        self.db_file.unlink()
        self.assertFalse(self.db_file.exists())

        # file was locked, so re-open new blank database
        # without making changes, close the database
        # it should still be written to file again, as the file is missing
        self.db = Database(self.db_file, self.encryption_key)
        self.assertEqual(self.db.conn.total_changes, self.db.init_changes)
        self.db.close()
        self.assertTrue(self.db_file.exists())

        # re-create database, with changes, so file exists on tearDown()
        self.db = Database(self.db_file, self.encryption_key)
        self.db.add_ingredient(Salt(enc_id=1, name="NaCl"))
        self.assertGreater(self.db.conn.total_changes, self.db.init_changes)

    def test_database_file_cannot_unlink_while_open(self):
        permission_error = False
        try:
            # try to delete the file while database open
            self.test_database_file_reload()
            self.db_file.unlink()
        except PermissionError as e:
            permission_error = True
        self.assertTrue(self.db_file.exists())
        self.assertTrue(permission_error)


class TestDatabaseWithoutEncryption(BaseTestDatabase):
    # Subclass for encyption OFF
    encryption_key = None


class TestDatabaseWithEncryption(BaseTestDatabase):
    # Subclass for encyption ON
    encryption_key = "supersecretkey"


if __name__ == '__main__':
    unittest.main()
