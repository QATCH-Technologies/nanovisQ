"""
test_database.py

Integration tests for the Database class, verifying:
    - CRUD operations for each ingredient subclass (Protein, Buffer, Stabilizer, Surfactant, Salt)
    - CRUD operations for formulations, including component linking and viscosity profile persistence
    - Deletion and retrieval of ingredients and formulations
    - Integrity of component-to-ingredient foreign keys
    - Database file behavior on close (write/create, reload, and lock behavior)
    - Behavior with and without encryption key

Author:
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-06-02

Version:
    1.1
"""

import os
import shutil
import subprocess
import unittest
from pathlib import Path

from src.models.ingredient import Protein, Buffer, Stabilizer, Surfactant, Salt, ProteinClass
from src.models.formulation import Formulation, ViscosityProfile
from src.db.db import Database


class BaseTestDatabase(unittest.TestCase):
    """Base test suite for Database functionality, parameterized by encryption_key.

    Subclasses override `encryption_key` to test with and without encryption.
    """

    encryption_key = None  # Default: no encryption
    parse_file_key = False  # Default: no key parsed

    @classmethod
    def setUpClass(cls):
        """Create test_assets directory and define path to the test database file."""
        cls.assets_dir = Path(__file__).parent.parent / "test_assets"
        cls.assets_dir.mkdir(parents=True, exist_ok=True)
        cls.db_file = cls.assets_dir / "test_app.db"

    def setUp(self):
        """Initialize a fresh Database instance before each test."""
        if self.db_file.exists():
            self.db_file.unlink()

        if self.parse_file_key:
            # use existing default baseline database, if available
            if not os.path.isfile("assets/app.db"):
                # create database file with metadata containing `app_key`
                subprocess.run(["py", "../../make_data_db.py"])
            if os.path.isfile("assets/app.db"):
                shutil.copyfile("assets/app.db", self.db_file)
            else:
                raise FileNotFoundError("Unable to create database file!")

        self.db = Database(
            path=self.db_file,
            encryption_key=self.encryption_key,
            parse_file_key=self.parse_file_key)

    def tearDown(self):
        """Close the database, verify file exists, then remove it after each test."""
        self.db.close()
        self.assertTrue(self.db_file.exists(),
                        "Database file should exist after close.")
        if self.db_file.exists():
            self.db_file.unlink()

    # --- Ingredient tests ---

    def test_add_and_get_each_ingredient_type(self):
        """
        Test that adding and retrieving each subclass of Ingredient works correctly.

        - Add one instance of Protein, Buffer, Stabilizer, Surfactant, and Salt
        - Verify that get_ingredient returns correct subclass instance with matching fields
        """
        p = Protein(enc_id=10, name="ProtA", molecular_weight=50.0,
                    pI_mean=6.5, pI_range=0.3, class_type=ProteinClass.MAB_IGG1)
        b = Buffer(enc_id=20, name="BuffB", pH=7.4)
        s1 = Stabilizer(enc_id=30, name="StabC")
        s2 = Surfactant(enc_id=40, name="SurfD")
        salt = Salt(enc_id=50, name="SaltE")

        ids = {}
        for ing in (p, b, s1, s2, salt):
            iid = self.db.add_ingredient(ing)
            ids[type(ing).__name__] = iid

        got_p = self.db.get_ingredient(ids['Protein'])
        self.assertIsInstance(got_p, Protein, "Expected a Protein instance")
        self.assertEqual(got_p.enc_id, 10)
        self.assertAlmostEqual(got_p.molecular_weight, 50.0)

        got_b = self.db.get_ingredient(ids['Buffer'])
        self.assertIsInstance(got_b, Buffer, "Expected a Buffer instance")
        self.assertAlmostEqual(got_b.pH, 7.4)

        self.assertIsInstance(self.db.get_ingredient(
            ids['Stabilizer']), Stabilizer)
        self.assertIsInstance(self.db.get_ingredient(
            ids['Surfactant']), Surfactant)
        self.assertIsInstance(self.db.get_ingredient(ids['Salt']), Salt)

    def test_get_all_and_delete_ingredient(self):
        """
        Test that get_all_ingredients returns all inserted rows, and delete_ingredient removes by ID.

        - Insert the same Protein twice (two distinct rows)
        - Verify get_all_ingredients returns length 2
        - Delete one row by ID
        - Verify only one row remains
        - Delete all ingredients and verify empty list returned
        """
        all_ings = self.db.get_all_ingredients()
        start_count = len(all_ings)
        x = Protein(enc_id=1, name="X", molecular_weight=10,
                    pI_mean=5, pI_range=0.1, class_type=ProteinClass.NONE)
        id1 = self.db.add_ingredient(x)
        y = Protein(enc_id=2, name="Y", molecular_weight=10,
                    pI_mean=5, pI_range=0.1, class_type=ProteinClass.OTHER)
        id2 = self.db.add_ingredient(y)
        all_ings = self.db.get_all_ingredients()
        self.assertEqual(len(all_ings)-start_count, 2,
                         "Expected two ingredient entries")
        self.assertTrue(self.db.delete_ingredient(
            id1), "Deletion by ID should return True")
        remaining = self.db.get_all_ingredients()
        rem_p = [i for i in remaining if isinstance(i, Protein)][0]
        self.assertIn(rem_p.class_type, ProteinClass.all())
        self.assertEqual(len(remaining)-start_count, 1,
                         "One ingredient should remain after deletion")
        self.db.delete_all_ingredients()
        self.assertEqual(self.db.get_all_ingredients(), [],
                         "Expected empty ingredient list")

    def test_update_ingredient(self):
        """
        Test that update_ingredient correctly updates subclass fields and returns False for missing ID.

        - Insert a Buffer
        - Update its enc_id and pH via a new Buffer instance
        - Verify updated fields match
        - Attempt to update non-existent ID and expect False
        """
        b = Buffer(enc_id=99, name="OldBuff", pH=6.0)
        iid = self.db.add_ingredient(b)
        newb = Buffer(enc_id=123, name="NewBuff", pH=8.2)
        self.assertTrue(self.db.update_ingredient(iid, newb),
                        "Expected update to return True")
        got = self.db.get_ingredient(iid)
        self.assertIsInstance(got, Buffer)
        self.assertEqual(got.enc_id, 123)
        self.assertAlmostEqual(got.pH, 8.2)
        self.assertFalse(self.db.update_ingredient(9999, newb),
                         "Expected False when updating non-existent ID")

    def _make_formulation(self):
        """
        Construct a sample Formulation instance with one Protein, one Buffer, a ViscosityProfile, and temperature.

        Returns:
            Formulation: The newly created formulation with components and viscosity profile.
        """
        f = Formulation()
        p = Protein(enc_id=1, name="ProtA", molecular_weight=20,
                    pI_mean=6, pI_range=0.2, class_type=ProteinClass.FC_FUSION)
        f.set_protein(p, concentration=1.5, units="mg/mL")
        b = Buffer(enc_id=2, name="BuffB", pH=7.0)
        f.set_buffer(b, concentration=2.0, units="mM")
        vp = ViscosityProfile([1, 10], [0.1, 0.05], units="Pa·s")
        vp.is_measured = True
        f.set_viscosity_profile(vp)
        f.set_temperature(37.0)
        return f

    def test_add_and_get_formulation(self):
        """
        Test that add_formulation and get_formulation work properly.

        - Create a sample formulation
        - Insert it into the database
        - Retrieve by ID and verify:
            - Temperature matches
            - Protein and Buffer concentrations and ingredient types match
            - ViscosityProfile is_measured flag and shear_rates match
        """
        form = self._make_formulation()
        fid = self.db.add_formulation(form)
        got = self.db.get_formulation(fid)
        self.assertIsNotNone(got, "Expected to retrieve a Formulation")
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
        """
        Test that get_all_formulations returns all rows and delete_formulation removes by ID.

        - Insert the sample formulation twice
        - Verify two entries returned
        - Delete one by ID; verify count decremented
        - Delete all; verify no formulations remain
        """
        self.db.add_formulation(self._make_formulation())
        self.db.add_formulation(self._make_formulation())
        all_forms = self.db.get_all_formulations()
        self.assertEqual(len(all_forms), 2, "Expected two formulations")
        fid = all_forms[0].id

        self.assertTrue(self.db.delete_formulation(
            fid), "Expected delete_formulation to return True")
        self.assertEqual(len(self.db.get_all_formulations()),
                         1, "One formulation should remain")

        self.db.delete_all_formulations()
        self.assertEqual(self.db.get_all_formulations(), [],
                         "Expected no formulations after delete_all")

    def test_components_linked_to_existing_ingredients(self):
        """
        Test that formulation_component table correctly references existing ingredient rows.

        - Insert a sample formulation
        - Query the underlying formulation_component rows
        - Verify that each component_type matches a non-None key from form._components
        - Verify each referenced ingredient_id points to a row in ingredient table with matching enc_id/name
        """
        form = self._make_formulation()
        fid = self.db.add_formulation(form)

        c = self.db.conn.cursor()
        c.execute(
            "SELECT component_type, ingredient_id FROM formulation_component WHERE formulation_id = ?",
            (fid,)
        )
        comp_rows = c.fetchall()

        expected_types = [
            k for k, v in form._components.items() if v is not None]
        self.assertEqual(
            {r[0] for r in comp_rows},
            set(expected_types),
            "Formulation_component rows do not match form._components keys"
        )

        for comp_type, iid in comp_rows:
            c.execute("SELECT enc_id, name FROM ingredient WHERE id = ?", (iid,))
            row = c.fetchone()
            self.assertIsNotNone(
                row, f"Component '{comp_type}' references missing ingredient ID {iid}")
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
                f"Name mismatch for component '{comp_type}'"
            )

    def test_database_file_reload(self):
        """
        Test that closing and reopening the Database from file retains inserted data.

        - Insert a sample formulation
        - Close the database (writes to file)
        - Reopen a new Database instance from the same file
        - Retrieve the same formulation and verify its data matches
        """
        form = self._make_formulation()
        fid = self.db.add_formulation(form)

        self.db.close()
        self.assertTrue(self.db_file.exists(),
                        "Database file should exist after close")
        self.db = Database(self.db_file, self.encryption_key,
                           self.parse_file_key)

        got = self.db.get_formulation(fid)
        self.assertIsNotNone(
            got, "Expected to retrieve a Formulation after reload")
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
        """
        Test that a new Database file is written on close if the file is missing.

        - Use test_database_file_reload to create and reload database
        - Close and then delete the file to simulate missing file
        - Reopen a fresh Database and close without changes; verify file is written
        - Reopen again, make a change, and verify total_changes > init_changes
        """
        self.test_database_file_reload()

        self.db.close()
        self.db_file.unlink()
        self.assertFalse(self.db_file.exists(),
                         "Database file should be removed")

        # Opening without making changes and then closing should recreate the file
        self.db = Database(self.db_file, self.encryption_key,
                           self.parse_file_key)
        self.assertEqual(self.db.conn.total_changes,
                         self.db.init_changes, "No changes yet")
        self.db.close()
        self.assertTrue(self.db_file.exists(),
                        "Database file should be recreated on close")

        # Now reopen, make a change, and ensure total_changes > init_changes
        self.db = Database(self.db_file, self.encryption_key,
                           self.parse_file_key)
        self.db.add_ingredient(Salt(enc_id=1, name="NaCl"))
        self.assertGreater(self.db.conn.total_changes,
                           self.db.init_changes, "Changes should be recorded")

    def test_database_file_cannot_unlink_while_open(self):
        """
        Test that attempting to delete the database file while it is open raises PermissionError.

        - Use test_database_file_reload to create and reload database
        - Attempt to unlink the file while `self.db` is still open
        - Expect PermissionError and verify file still exists
        """
        self.test_database_file_reload()

        permission_error = False
        try:
            self.db_file.unlink()
        except PermissionError:
            permission_error = True

        self.assertTrue(self.db_file.exists(),
                        "Database file should still exist")
        self.assertTrue(permission_error,
                        "Expected PermissionError when deleting open DB file")


class TestDatabaseWithEncryption(BaseTestDatabase):
    """Runs BaseTestDatabase with encryption_key = None (no encryption)."""
    encryption_key = "supersecretkey"


class TestDatabaseWithFileKey(BaseTestDatabase):
    """Runs BaseTestDatabase with a non-empty encryption_key to test encrypted storage."""
    parse_file_key = True


if __name__ == '__main__':
    unittest.main()
