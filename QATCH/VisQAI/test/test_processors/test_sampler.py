# tests/test_sampler.py

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from src.db.db import Database
from src.controller.asset_controller import AssetController, AssetError
from src.controller.ingredient_controller import IngredientController
from src.models.predictor import Predictor
from src.models.formulation import Formulation, ViscosityProfile
from src.processors.sampler import Sampler


class TestSampler(unittest.TestCase):
    def setUp(self):
        self.db = Database(parse_file_key=True)
        self.sampler = Sampler(asset_name="VisQAI-base", database=self.db)

    def test_init_raises_when_asset_missing(self):
        with self.assertRaises(AssetError):
            Sampler('some_other_asset', database=self.db)
        Sampler('VisQAI-base', database=self.db)

    def test_make_viscosity_profile(self):
        s = Sampler('VisQAI-base', database=self.db)
        data = {'0.1': 5, '1.0': 50}
        profile = s._make_viscosity_profile(data)
        self.assertIsInstance(profile, ViscosityProfile)
        self.assertListEqual(profile.shear_rates, [0.1, 1.0])
        self.assertListEqual(profile.viscosities, [5.0, 50.0])

    def test_get_next_sample_no_history(self):
        with self.assertRaises(RuntimeError):
            next = self.sampler.get_next_sample()
            print(next.to_dict())


if __name__ == '__main__':
    unittest.main()
