# tests/test_constraints.py

import unittest
import numpy as np
import os
from unittest.mock import MagicMock
from src.models.formulation import ViscosityProfile, Formulation
from src.models.predictor import Predictor
from src.controller.ingredient_controller import IngredientController
from src.controller.asset_controller import AssetController

from src.models.ingredient import Protein, Buffer, Salt, Stabilizer, Surfactant
from src.db.db import Database
from src.processors.optimizer import Constraints, Optimizer


class TestConstraints(unittest.TestCase):

    def setUp(self):
        self.db = Database(path=os.path.join(
            "test", "assets", "app.db"), parse_file_key=True)
        self.ing_ctrl = IngredientController(self.db)
        self.constraints = Constraints(db=self.db)

    def test_add_range_valid(self):
        self.constraints.add_range("Protein_conc", 0.1, 5.0)
        self.assertEqual(self.constraints._ranges["Protein_conc"], (0.1, 5.0))

        self.constraints.add_range("Temperature", -20.0, 80.0)
        self.assertEqual(
            self.constraints._ranges["Temperature"], (-20.0, 80.0))

    def test_add_range_invalid_feature(self):
        with self.assertRaises(ValueError) as cm:
            self.constraints.add_range("Unknown_feat", 0, 1)
        self.assertIn("Unknown numeric feature", str(cm.exception))

    def test_add_range_negative_non_temperature(self):
        with self.assertRaises(ValueError) as cm_low:
            self.constraints.add_range("Buffer_conc", -1.0, 2.0)
        self.assertIn("Negative values are not allowed", str(cm_low.exception))

        with self.assertRaises(ValueError) as cm_high:
            self.constraints.add_range("Salt_conc", 0.0, -0.5)
        self.assertIn("Negative values are not allowed",
                      str(cm_high.exception))

    def test_add_choices_invalid_feature(self):
        dummy = Protein(enc_id=-1, name="P", molecular_weight=50.0,
                        pI_mean=6.0, pI_range=0.2)
        with self.assertRaises(ValueError) as cm:
            self.constraints.add_choices("Protein_conc", [dummy])
        self.assertIn("Unknown categorical feature", str(cm.exception))

    def test_add_choices_non_ingredient(self):
        with self.assertRaises(TypeError) as cm:
            self.constraints.add_choices("Protein_type", ["not_an_ingredient"])
        self.assertIn("must be Ingredient instances", str(cm.exception))

    def test_add_choices_not_persisted(self):
        prot = Protein(enc_id=-1, name="NewProt", molecular_weight=30.0,
                       pI_mean=5.0, pI_range=0.1)
        with self.assertRaises(ValueError) as cm:
            self.constraints.add_choices("Protein_type", [prot])
        self.assertIn("has not been added to persistent store",
                      str(cm.exception))

    def test_add_choices_valid(self):
        prot = Protein(enc_id=-1, name="GoodProt", molecular_weight=25.0,
                       pI_mean=4.5, pI_range=0.2)
        self.ing_ctrl.add(prot)
        self.constraints.add_choices("Protein_type", [prot])
        self.assertIn("Protein_type", self.constraints._choices)
        self.assertEqual(self.constraints._choices["Protein_type"], [prot])

    def test_build_empty_db_raises(self):
        with self.assertRaises(ValueError) as cm:
            self.constraints.build()
        self.assertIn("No choices available", str(cm.exception))

    def test_build_with_defaults_and_ranges(self):
        ing_list = [
            Protein(enc_id=-1, name="P", molecular_weight=10.0,
                    pI_mean=6.0, pI_range=0.2),
            Buffer(enc_id=-1, name="B", pH=7.4),
            Salt(enc_id=-1, name="Salt"),
            Stabilizer(enc_id=-1, name="Stab"),
            Surfactant(enc_id=-1, name="Surf"),
        ]
        for ing in ing_list:
            self.ing_ctrl.add(ing)

        self.constraints.add_range("Protein_conc", 0.5, 2.5)
        bounds, encoding = self.constraints.build()
        total_feats = len(Constraints._CATEGORICAL) + len(Constraints._NUMERIC)
        self.assertEqual(len(bounds), total_feats)
        self.assertEqual(len(encoding), total_feats)

        numeric_entries = [
            b for enc, b in zip(encoding, bounds) if enc["type"] == "num"
        ]
        self.assertIn((0.5, 2.5), numeric_entries)

        for enc, bnd in zip(encoding, bounds):
            if enc["type"] == "cat":
                names = enc["choices"]
                self.assertEqual(bnd, (0.0, float(len(names) - 1)))
                self.assertIsInstance(names, list)
                self.assertGreater(len(names), 0)


class DummyPredictor(Predictor):
    """Always returns the provided 'return_profile'."""

    def __init__(self, return_profile: ViscosityProfile):
        self.return_profile = return_profile

    def predict(self, formulation: Formulation) -> ViscosityProfile:
        return self.return_profile


class TestOptimizer(unittest.TestCase):
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
        self.db = Database(path=os.path.join(
            "test", "assets", "app.db"), parse_file_key=True)
        self.ing_ctrl = IngredientController(self.db)
        for ing in self.all_ings:
            self.ing_ctrl.add(ing)
        self.constraints = Constraints(db=self.db)
        self.constraints._ingredient_ctrl = self.ing_ctrl

        self.target = ViscosityProfile(
            shear_rates=[100, 1_000, 10_000, 100_000, 15_000_000],
            viscosities=[10, 10, 10, 10, 10],
            units="cP"
        )

        self.predictor = Predictor(zip_path=os.path.join(
            "test", "assets", "VisQAI-base.zip"))

        self.optimizer = Optimizer(
            constraints=self.constraints,
            predictor=self.predictor,
            target=self.target,
            maxiter=10,
            popsize=2,
            seed=42
        )

    def test_constraints_build_shape(self):
        bounds, encoding = self.constraints.build()
        expected = len(Constraints._CATEGORICAL) + len(Constraints._NUMERIC)
        self.assertEqual(len(bounds), expected)
        self.assertEqual(len(encoding), expected)
        for enc in encoding:
            self.assertIn("feature", enc)
            self.assertIn("type", enc)
            self.assertIn(enc["type"], ("cat", "num"))

    def test_decode_categoricals(self):
        x = np.zeros(len(self.optimizer.encoding))
        decoded = self.optimizer._decode(x)
        for enc in self.optimizer.encoding:
            if enc["type"] == "cat":
                feat = enc["feature"]
                self.assertEqual(
                    decoded[feat], self.optimizer.cat_choices[feat][0])

    def test_build_formulation_no_error(self):
        x = np.zeros(len(self.optimizer.encoding))
        feat_dict = self.optimizer._decode(x)
        form = self.optimizer._build_formulation(feat_dict)
        self.assertIsInstance(form, Formulation)

    def test_optimize_returns_formulation(self):
        best = self.optimizer.optimize()
        self.assertIsInstance(best, Formulation)


if __name__ == "__main__":
    unittest.main()
