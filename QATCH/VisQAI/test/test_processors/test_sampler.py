import os
import unittest
import numpy as np

from src.processors.sampler import Sampler
from src.db.db import Database
from src.utils.constraints import Constraints
from src.models.formulation import Formulation, ViscosityProfile
from src.managers.asset_manager import AssetError
from src.models.ingredient import Protein, Buffer, Salt, Stabilizer, Surfactant
from src.controller.ingredient_controller import IngredientController


class TestSampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protein = Protein(enc_id=-1,
                              name="ProtA", molecular_weight=100, pI_mean=10, pI_range=1)
        cls.buffer = Buffer(enc_id=-1, name="BuffA", pH=10)
        cls.stabilizer = Stabilizer(enc_id=-1, name="StabA")
        cls.salt = Salt(enc_id=-1, name="SaltA")
        cls.surfactant = Surfactant(enc_id=-1, name="SurfA")
        cls.all_ings = [
            cls.protein,
            cls.buffer,
            cls.stabilizer,
            cls.salt,
            cls.surfactant
        ]

    def setUp(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(
            os.path.join(base_dir, os.pardir, os.pardir))
        self.assets_dir = os.path.join(project_root, "assets")
        if not os.path.isdir(self.assets_dir):
            self.skipTest(f"Assets directory not found at {self.assets_dir}")

        zip_files = [f for f in os.listdir(
            self.assets_dir) if f.endswith(".zip")]
        if not zip_files:
            self.skipTest(f"No .zip asset files found in {self.assets_dir}")
        self.asset_name = os.path.splitext(zip_files[0])[0]

        self.db = Database(path=os.path.join(
            "test", "assets", "app.db"), parse_file_key=True)
        self.ing_ctrl = IngredientController(self.db)
        self.ing_ctrl.delete_all_ingredients()

        self.constraints = Constraints(db=self.db)
        for ing in self.all_ings:
            self.ing_ctrl.add(ing)
        self.sampler = Sampler(
            asset_name=self.asset_name,
            database=self.db,
            constraints=self.constraints,
            seed=42
        )

    def test_invalid_asset_raises_asset_error(self):
        with self.assertRaises(AssetError):
            Sampler(asset_name="this_asset_does_not_exist", database=self.db)

    def test_generate_random_samples_count_and_type(self):
        samples = self.sampler._generate_random_samples(10)
        self.assertEqual(len(samples), 10)
        for form in samples:
            self.assertIsInstance(form, Formulation)

    def test_acquisition_ucb_computation(self):
        viscosity = {"8": 1.0, "12": 3.0}
        uncertainty = np.array([0.2, 0.4, np.nan])
        score = self.sampler._acquisition_ucb(
            viscosity,
            uncertainty,
            kappa=1.0,
            reference_shear_rate=10.0
        )
        mu = 1.0
        sigma = np.nanmean(uncertainty)
        expected = mu + 1.0 * sigma
        self.assertAlmostEqual(score, expected, places=7)

    def test_add_sample_updates_internal_state(self):
        sample = self.sampler._generate_random_samples(1)[0]
        self.sampler.add_sample(sample)
        self.assertIs(self.sampler._last_formulation, sample)
        self.assertIsInstance(
            self.sampler._current_viscosity, ViscosityProfile)
        self.assertIsInstance(self.sampler._current_uncertainty, np.ndarray)
        self.assertGreater(self.sampler._current_uncertainty.size, 0)

    def test_get_next_sample_without_and_with_prior(self):
        next_form = self.sampler.get_next_sample(use_ucb=True)
        self.assertIsInstance(next_form, Formulation)

        prior = self.sampler._generate_random_samples(1)[0]
        self.sampler.add_sample(prior)
        next_form2 = self.sampler.get_next_sample(use_ucb=True)
        print(next_form2)
        self.assertIsInstance(next_form2, Formulation)

    def test_generated_samples_respect_numeric_constraints(self):
        constrained = Constraints(db=self.db)
        constrained.add_range("Protein_conc", 50.0, 60.0)
        constrained.add_range("Temperature", 25.0, 26.0)
        sampler_c = Sampler(
            asset_name=self.asset_name,
            database=self.db,
            constraints=constrained,
            seed=123
        )

        samples = sampler_c._generate_random_samples(30)
        self.assertEqual(len(samples), 30)

        for s in samples:
            conc = s.protein.concentration
            self.assertGreaterEqual(conc, 50.0,
                                    f"Protein_conc {conc} < 50.0")
            self.assertLessEqual(conc, 60.0,
                                 f"Protein_conc {conc} > 60.0")

            temp = s.temperature
            self.assertGreaterEqual(temp, 25.0,
                                    f"Temperature {temp} < 25.0")
            self.assertLessEqual(temp, 26.0,
                                 f"Temperature {temp} > 26.0")


if __name__ == "__main__":
    unittest.main()
