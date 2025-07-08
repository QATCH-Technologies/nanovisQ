import os
import unittest
import numpy as np

from src.processors.sampler import Sampler
from src.db.db import Database
from src.utils.constraints import Constraints
from src.models.formulation import Formulation, ViscosityProfile
from src.controller.asset_controller import AssetError
from src.models.ingredient import Protein, Buffer, Salt, Stabilizer, Surfactant
from src.controller.ingredient_controller import IngredientController


class TestSampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create one of each Ingredient subclass with distinct names:
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
        # Locate the project's assets directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(
            os.path.join(base_dir, os.pardir, os.pardir))
        self.assets_dir = os.path.join(project_root, "assets")
        if not os.path.isdir(self.assets_dir):
            self.skipTest(f"Assets directory not found at {self.assets_dir}")

        # Pick the first .zip asset available
        zip_files = [f for f in os.listdir(
            self.assets_dir) if f.endswith(".zip")]
        if not zip_files:
            self.skipTest(f"No .zip asset files found in {self.assets_dir}")
        self.asset_name = os.path.splitext(zip_files[0])[0]

        # Initialize with a real Database; rely on existing seed data
        self.db = Database(path=os.path.join(
            "test", "assets", "app.db"), parse_file_key=True)
        self.ing_ctrl = IngredientController(self.db)
        for ing in self.all_ings:
            self.ing_ctrl.add(ing)
        # Build the Sampler (seeded for reproducibility)
        self.sampler = Sampler(
            asset_name=self.asset_name,
            database=self.db,
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

    def test_make_viscosity_profile(self):
        viscosity = {"5": 1.5, "10": 2.5}
        vp = self.sampler._make_viscosity_profile(viscosity)
        self.assertIsInstance(vp, ViscosityProfile)
        # Order of items follows insertion order
        self.assertEqual(vp.shear_rates, [5.0, 10.0])
        self.assertEqual(vp.viscosities, [1.5, 2.5])

    def test_acquisition_ucb_computation(self):
        viscosity = {"8": 1.0, "12": 3.0}
        uncertainty = np.array([0.2, 0.4, np.nan])
        # Closest to reference_shear_rate=10.0 is 8.0 (first match)
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
        # Use one of the randomly generated formulations as input
        sample = self.sampler._generate_random_samples(1)[0]
        self.sampler.add_sample(sample)

        # After adding, _last_formulation should be exactly our sample
        self.assertIs(self.sampler._last_formulation, sample)
        # And the predictor must have populated viscosity & uncertainty
        self.assertIsInstance(
            self.sampler._current_viscosity, ViscosityProfile)
        self.assertIsInstance(self.sampler._current_uncertainty, np.ndarray)
        self.assertGreater(self.sampler._current_uncertainty.size, 0)

    def test_get_next_sample_without_and_with_prior(self):
        # Without any prior sample
        next_form = self.sampler.get_next_sample(use_ucb=False)
        self.assertIsInstance(next_form, Formulation)

        # After seeding with a sample, still returns a Formulation
        prior = self.sampler._generate_random_samples(1)[0]
        self.sampler.add_sample(prior)
        next_form2 = self.sampler.get_next_sample(use_ucb=False)
        self.assertIsInstance(next_form2, Formulation)

    def test_perturb_formulation_keeps_within_bounds(self):
        # Build a minimal suggestion dict at the lower bounds
        suggestions = {}
        for (low, high), enc in zip(self.sampler._bounds, self.sampler._encoding):
            if enc["type"] == "cat":
                # pick the first choice
                suggestions[enc["feature"]] = enc["choices"][0]
            else:
                suggestions[enc["feature"]] = float(low)

        base_form = self.sampler._build_formulation(suggestions)
        perturbed = self.sampler._perturb_formulation(
            base_form, base_uncertainty=0.5, n=5)

        # Should produce exactly 5 variants, all within [low, high]
        self.assertEqual(len(perturbed), 5)
        for form in perturbed:
            df = form.to_dataframe()
            for (low, high), enc in zip(self.sampler._bounds, self.sampler._encoding):
                feat = enc["feature"]
                val = df[feat].iloc[0]
                self.assertGreaterEqual(val, low, f"{feat} below lower bound")
                self.assertLessEqual(val, high, f"{feat} above upper bound")
                if enc["type"] == "cat":
                    # categorical must remain unchanged
                    self.assertEqual(val, suggestions[feat])


if __name__ == "__main__":
    unittest.main()
