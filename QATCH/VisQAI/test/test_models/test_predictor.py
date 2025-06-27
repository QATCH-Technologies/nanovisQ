import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.predictor import Predictor

ZIP_PATH = (
    Path(__file__).resolve().parent / "assets" / "VisQAI-base.zip"
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
        from io import StringIO
        csv = """ID,Protein_type,MW,PI_mean,PI_range,Protein_conc,Temperature,Buffer_type,Buffer_pH,Buffer_conc,Salt_type,Salt_conc,Stabilizer_type,Stabilizer_conc,Surfactant_type,Surfactant_conc,Viscosity_100,Viscosity_1000,Viscosity_10000,Viscosity_100000,Viscosity_15000000,
        F1,poly-hIgG,150,7.6,1,145,25,PBS,7.4,10,NaCl,140,Sucrose,1,none,0,12.5,11.5,9.8,8.8,6.92,
        """
        df = pd.read_csv(StringIO(csv))
        # point‐estimate
        preds = p.predict(df)
        self.assertIsInstance(preds, np.ndarray)
        # uncertainty
        mean, std = p.predict_uncertainty(df)
        self.assertIsInstance(mean, np.ndarray)
        self.assertIsInstance(std, np.ndarray)
        p.cleanup()

    def test_predict_after_update(self):
        p = Predictor(str(self.zip_path))

        # build the same minimal DataFrame you used for predict()
        from io import StringIO
        csv = """ID,Protein_type,MW,PI_mean,PI_range,Protein_conc,Temperature,Buffer_type,Buffer_pH,Buffer_conc,Salt_type,Salt_conc,Stabilizer_type,Stabilizer_conc,Surfactant_type,Surfactant_conc,Viscosity_100,Viscosity_1000,Viscosity_10000,Viscosity_100000,Viscosity_15000000,
        F1,poly-hIgG,150,7.6,1,145,25,PBS,7.4,10,NaCl,140,Sucrose,1,none,0,12.5,11.5,9.8,8.8,6.92,
        """
        df = pd.read_csv(StringIO(csv))

        # use one of the true columns as a target for update()
        y_true = df['Viscosity_100'].values

        # perform a single-epoch update on our one sample
        p.update(df, y_true, epochs=1,
                 batch_size=1, save=False)

        # after updating, predictions should still run
        preds = p.predict(df)
        self.assertIsInstance(preds, np.ndarray)

        # and uncertainty still works
        mean, std = p.predict_uncertainty(df)
        self.assertIsInstance(mean, np.ndarray)
        self.assertIsInstance(std, np.ndarray)
        self.assertEqual(mean.shape, preds.shape)
        self.assertEqual(std.shape, preds.shape)

        p.cleanup()

    def test_missing_zip_raises(self):
        missing = Path(str(self.zip_path) + ".does_not_exist")
        with self.assertRaises(FileNotFoundError):
            Predictor(str(missing))


if __name__ == "__main__":
    unittest.main()
