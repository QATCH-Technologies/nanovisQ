import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.predictor import Predictor

ZIP_PATH = (
    Path(__file__).resolve().parent
    / "test_models" / "assets" / "VisQAI-base.zip"
)


class TestPredictorWithRealModel(unittest.TestCase):
    def setUp(self):
        # ensure the file actually exists
        self.zip_path = ZIP_PATH
        assert self.zip_path.is_file(
        ), f"Test ZIP not found at {self.zip_path!r}"

        # make sure any stale modules are gone
        for mod in ("custom_layers", "data_processor", "predictor"):
            if mod in sys.modules:
                del sys.modules[mod]

    def tearDown(self):
        # clean up loaded modules so other tests aren’t polluted
        for mod in ("custom_layers", "data_processor", "predictor"):
            if mod in sys.modules:
                del sys.modules[mod]

    def test_init_and_cleanup(self):
        p = Predictor(str(self.zip_path))
        # after init, an ensemble must be set
        self.assertIsNotNone(p.ensemble)
        # members should be > 0
        self.assertGreater(len(p.ensemble.members), 0)
        # cleanup should drop ensemble
        p.cleanup()
        self.assertIsNone(p.ensemble)

    def test_predict_and_predict_uncertainty(self):
        p = Predictor(str(self.zip_path))
        # build a minimal DataFrame with the same columns your model expects
        # e.g., replace these with real features
        df = pd.DataFrame(
            {col: [0.0] for col in p.ensemble.members[0].preprocessor.feature_names_in_})
        # point‐estimate
        preds = p.predict(df)
        self.assertIsInstance(preds, np.ndarray)
        # uncertainty
        mean, std = p.predict_uncertainty(df)
        self.assertIsInstance(mean, np.ndarray)
        self.assertIsInstance(std, np.ndarray)
        p.cleanup()

    def test_missing_zip_raises(self):
        missing = Path(str(self.zip_path) + ".does_not_exist")
        with self.assertRaises(FileNotFoundError):
            Predictor(str(missing))


if __name__ == "__main__":
    unittest.main()
