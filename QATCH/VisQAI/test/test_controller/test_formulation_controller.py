"""
test_formulation_controller.py

Integration tests for the FormulationController, verifying end-to-end behavior of:
    - Adding, retrieving, and finding formulations
    - Preventing duplicate entries
    - Deleting and updating formulations, including error conditions
    - Bulk import from pandas DataFrame (`add_all_from_dataframe`)
    - Export to DataFrame (`get_all_as_dataframe`), including empty and multiple-row scenarios

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-06-02

Version:
    1.1
"""
import unittest
from pathlib import Path
import pandas as pd
from src.db.db import Database
from src.controller.formulation_controller import FormulationController
from src.models.ingredient import Protein, Buffer, Salt, Surfactant, Stabilizer
from src.models.formulation import Formulation, ViscosityProfile


class TestFormulationControllerIntegration(unittest.TestCase):
    """Integration tests for the FormulationController.

    Verifies CRUD operations on formulations, including adding, retrieving,
    updating, deleting, and bulk importing/exporting via DataFrame.
    """

    @classmethod
    def setUpClass(cls):
        """Create the test_assets directory and define the test database file path."""
        cls.assets_dir = Path(__file__).parent / "test_assets"
        cls.assets_dir.mkdir(parents=True, exist_ok=True)
        cls.db_file = cls.assets_dir / "test_app.db"

    def setUp(self):
        """Initialize a fresh Database and FormulationController before each test."""
        if self.db_file.exists():
            self.db_file.unlink()
        self.db = Database(self.db_file)
        self.ctrl = FormulationController(self.db)

    def tearDown(self):
        """Close the database connection and remove the test database file after each test."""
        self.db.conn.close()
        if self.db_file.exists():
            self.db_file.unlink()

    def make_formulation(self, suffix=""):
        """Construct a Formulation instance with all components and a viscosity profile.

        Args:
            suffix (str, optional): Suffix appended to each ingredient name to avoid collisions.

        Returns:
            Formulation: A fully populated formulation with protein, buffer, salt, surfactant,
                stabilizer, temperature, and viscosity profile.
        """
        p = Protein(
            enc_id=-1, name=f"prot{suffix}", molecular_weight=50.0, pI_mean=7.0, pI_range=0.5
        )
        b = Buffer(enc_id=-1, name=f"buf{suffix}", pH=7.4)
        s = Salt(enc_id=-1, name=f"salt{suffix}")
        sf = Surfactant(enc_id=-1, name=f"surf{suffix}")
        st = Stabilizer(enc_id=-1, name=f"stab{suffix}")
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
        """Test that a formulation can be added, retrieved, and found by ID."""
        f = self.make_formulation()

        # Add formulation and verify ID is set
        returned = self.ctrl.add_formulation(f)
        self.assertIs(returned, f)
        self.assertIsNotNone(
            f.id, "Formulation.id should be set by Database.add_formulation")
        all_forms = self.ctrl.get_all_formulations()
        self.assertEqual(len(all_forms), 1)
        self.assertEqual(all_forms[0], f)
        fetched = self.ctrl.get_formulation_by_id(f.id)
        self.assertEqual(fetched, f)
        found_id = self.ctrl.find_id(f)
        self.assertEqual(found_id, f.id)

    def test_add_duplicate_formulation(self):
        """Test that adding an identical formulation does not create a duplicate entry."""
        f1 = self.make_formulation()
        f2 = self.make_formulation()
        a1 = self.ctrl.add_formulation(f1)
        a2 = self.ctrl.add_formulation(f2)
        self.assertEqual(a1.id, a2.id)
        self.assertEqual(len(self.ctrl.get_all_formulations()), 1)

    def test_delete_formulation(self):
        """Test that a formulation can be deleted by its ID and is removed from storage."""
        f = self.make_formulation()
        self.ctrl.add_formulation(f)
        fid = f.id

        deleted = self.ctrl.delete_formulation_by_id(fid)
        self.assertEqual(deleted, f)

        # After deletion, no formulations should remain
        self.assertEqual(self.ctrl.get_all_formulations(), [])
        self.assertIsNone(self.ctrl.get_formulation_by_id(fid))

    def test_delete_nonexistent_raises(self):
        """Test that deleting a non-existent formulation raises a ValueError."""
        with self.assertRaises(ValueError) as cm:
            self.ctrl.delete_formulation_by_id(999)
        self.assertIn("does not exist", str(cm.exception))

    def test_update_formulation(self):
        """Test updating a stored formulation replaces the old one with new data."""
        old = self.make_formulation()
        self.ctrl.add_formulation(old)
        old_id = old.id

        new = self.make_formulation(suffix="_new")
        updated = self.ctrl.update_formulation(old_id, new)
        self.assertIs(updated, new)

        all_forms = self.ctrl.get_all_formulations()
        self.assertIn(new, all_forms)
        self.assertNotIn(old, all_forms)

    def test_update_no_change(self):
        """Test updating a formulation to itself returns the same instance and does not duplicate."""
        f = self.make_formulation()
        self.ctrl.add_formulation(f)
        fid = f.id

        same = self.ctrl.update_formulation(fid, f)
        self.assertIs(same, f)
        self.assertEqual(self.ctrl.get_all_formulations(), [f])

    def test_update_nonexistent_raises(self):
        """Test that updating a non-existent formulation raises a ValueError."""
        f = self.make_formulation()
        with self.assertRaises(ValueError) as cm:
            self.ctrl.update_formulation(1234, f)
        self.assertIn("does not exist", str(cm.exception))

    def _make_complete_dataframe(self):
        """Create a DataFrame matchng the columns expected by add_all_from_dataframe."""
        shear_rates = [100, 1000, 10000, 100000, 15000000]
        data = {
            "Protein_type": ["Lysozyme"],
            "MW": [14300.0],
            "PI_mean": [11.35],
            "PI_range": [0.2],
            "Protein_conc": [1.5],
            "Temperature": [25.0],
            "Buffer_type": ["PBS"],
            "Buffer_pH": [7.4],
            "Buffer_conc": [10.0],
            "Salt_type": ["NaCl"],
            "Salt_conc": [150.0],
            "Stabilizer_type": ["Sucrose"],
            "Stabilizer_conc": [0.2],
            "Surfactant_type": ["Tween20"],
            "Surfactant_conc": [0.05],
        }
        for r in shear_rates:
            data[f"Viscosity_{r}"] = [0.89 * (r / 100)]

        df = pd.DataFrame(data)
        return df

    def test_add_all_from_dataframe_success(self):
        """Test that a well-formed DataFrame is imported correctly as a single formulation."""
        df = self._make_complete_dataframe()
        added = self.ctrl.add_all_from_dataframe(df)
        self.assertIsInstance(added, list)
        self.assertEqual(len(added), 1)

        form = added[0]
        self.assertIsNotNone(form.id)
        self.assertIsNotNone(form.protein.ingredient.id)
        self.assertIsNotNone(form.buffer.ingredient.id)
        self.assertIsNotNone(form.salt.ingredient.id)
        self.assertIsNotNone(form.stabilizer.ingredient.id)
        self.assertIsNotNone(form.surfactant.ingredient.id)

        out_df = self.ctrl.get_all_as_dataframe()
        self.assertEqual(len(out_df), 1)

        row = out_df.iloc[0]
        self.assertEqual(row["Protein_type"], 8001)
        self.assertAlmostEqual(row["MW"], 14300.0)
        self.assertAlmostEqual(row["PI_mean"], 11.35)
        self.assertAlmostEqual(row["PI_range"], 0.2)
        self.assertAlmostEqual(row["Protein_conc"], 1.5)
        self.assertAlmostEqual(row["Temperature"], 25.0)
        self.assertEqual(row["Buffer_type"], 8001)
        self.assertAlmostEqual(row["Buffer_pH"], 7.4)
        self.assertAlmostEqual(row["Buffer_conc"], 10.0)
        self.assertEqual(row["Salt_type"], 8001)
        self.assertAlmostEqual(row["Salt_conc"], 150.0)
        self.assertEqual(row["Stabilizer_type"], 8001)
        self.assertAlmostEqual(row["Stabilizer_conc"], 0.2)
        self.assertEqual(row["Surfactant_type"], 8001)
        self.assertAlmostEqual(row["Surfactant_conc"], 0.05)

        self.assertAlmostEqual(row["Viscosity_1000"],
                               df.at[0, "Viscosity_1000"])
        self.assertAlmostEqual(
            row["Viscosity_10000"], df.at[0, "Viscosity_10000"])
        self.assertAlmostEqual(
            row["Viscosity_100000"], df.at[0, "Viscosity_100000"])
        self.assertAlmostEqual(
            row["Viscosity_15000000"], df.at[0, "Viscosity_15000000"])

    def test_add_all_from_dataframe_missing_columns(self):
        """Test that importing a DataFrame missing required columns raises a ValueError."""
        df = pd.DataFrame({
            "Protein_type": ["Lysozyme"],
            "MW": [14300.0],
        })
        with self.assertRaises(ValueError) as context:
            self.ctrl.add_all_from_dataframe(df)
        msg = str(context.exception)
        self.assertIn("DataFrame is missing columns", msg)

    def test_get_all_as_dataframe_empty(self):
        """Test that get_all_as_dataframe returns an empty DataFrame with correct columns when no data exists."""
        out_df = self.ctrl.get_all_as_dataframe()
        expected_columns = [
            "ID",
            "Protein_type", "MW", "PI_mean", "PI_range", "Protein_conc",
            "Temperature",
            "Buffer_type", "Buffer_pH", "Buffer_conc",
            "Salt_type", "Salt_conc",
            "Stabilizer_type", "Stabilizer_conc",
            "Surfactant_type", "Surfactant_conc",
            "Viscosity_100", "Viscosity_1000", "Viscosity_10000",
            "Viscosity_100000", "Viscosity_15000000"
        ]
        self.assertListEqual(list(out_df.columns), expected_columns)
        self.assertEqual(len(out_df), 0)

    def test_add_multiple_rows(self):
        """Test that duplicate rows in the DataFrame import only create one unique formulation entry."""
        df = pd.concat([self._make_complete_dataframe()]
                       * 2, ignore_index=True)
        added = self.ctrl.add_all_from_dataframe(df)
        self.assertEqual(len(added), 2)

        out_df = self.ctrl.get_all_as_dataframe()
        self.assertEqual(len(out_df), 1)
        ids = out_df["ID"].tolist()
        self.assertEqual(len(ids), len(set(ids)))

    def test_get_all_as_dataframe_column_order(self):
        """Test that the DataFrame returned by get_all_as_dataframe has columns in the defined order."""
        df = self._make_complete_dataframe()
        _ = self.ctrl.add_all_from_dataframe(df)
        out_df = self.ctrl.get_all_as_dataframe()
        expected_columns = [
            "ID",
            "Protein_type", "MW", "PI_mean", "PI_range", "Protein_conc",
            "Temperature",
            "Buffer_type", "Buffer_pH", "Buffer_conc",
            "Salt_type", "Salt_conc",
            "Stabilizer_type", "Stabilizer_conc",
            "Surfactant_type", "Surfactant_conc",
            "Viscosity_100", "Viscosity_1000", "Viscosity_10000",
            "Viscosity_100000", "Viscosity_15000000"
        ]
        self.assertListEqual(list(out_df.columns), expected_columns)


if __name__ == "__main__":
    unittest.main()
