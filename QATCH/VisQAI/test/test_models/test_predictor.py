import pandas as pd
import numpy as np
import unittest
import tempfile
import zipfile
from pathlib import Path
from src.models.predictor import Predictor, ComponentLoadError


class TestRealPredictor(unittest.TestCase):

    ASSET_ZIP = Path(__file__).parent / "assets" / "VisQAI-base.zip"

    @classmethod
    def setUpClass(cls):
        # Ensure the test asset bundle exists
        if not cls.ASSET_ZIP.is_file():
            raise FileNotFoundError(f"Bundle not found: {cls.ASSET_ZIP}")
        cls.predictor = Predictor(cls.ASSET_ZIP)
        cls.valid_data = pd.DataFrame({
            'Protein_type': [0],
            'Buffer_type': [0],
            'Salt_type': [0],
            'Stabilizer_type': [0],
            'Surfactant_type': [0],
            'MW': [150],
            'PI_mean': [7.6],
            'PI_range': [1],
            'Protein_conc': [145],
            'Temperature': [25],
            'Buffer_pH': [7.4],
            'Buffer_conc': [10],
            'Salt_conc': [140],
            'Stabilizer_conc': [1],
            'Surfactant_conc': [0],
        })
        # Valid update DataFrame including target viscosities
        cls.valid_update = pd.DataFrame({
            **cls.valid_data.to_dict(orient='list'),
            'Viscosity_100': [10],
            'Viscosity_1000': [9],
            'Viscosity_10000': [9],
            'Viscosity_100000': [8],
            'Viscosity_15000000': [6]
        })

    def test_predict_shape_and_type(self):
        """predict() returns a 2D NumPy array of shape (n_samples, 5)"""
        preds = self.predictor.predict(self.valid_data)
        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(preds.ndim, 2)
        self.assertEqual(preds.shape[1], 5)
        self.assertGreater(preds.shape[0], 0)

    def test_predict_no_nan_or_inf(self):
        """Predictions should not contain NaNs or infinite values"""
        preds = self.predictor.predict(self.valid_data)
        self.assertFalse(np.isnan(preds).any(), "Predictions contain NaNs")
        self.assertFalse(np.isinf(preds).any(),
                         "Predictions contain infinities")

    def test_predict_consistency(self):
        """Repeated calls on the same input should yield identical outputs"""
        preds1 = self.predictor.predict(self.valid_data)
        preds2 = self.predictor.predict(self.valid_data)
        np.testing.assert_array_equal(preds1, preds2)

    def test_predict_different_inputs_produce_different_outputs(self):
        """Slightly altered inputs should change the predictions"""
        altered = self.valid_data.copy()
        altered.loc[0, 'Protein_conc'] += 1
        preds_orig = self.predictor.predict(self.valid_data)
        preds_alt = self.predictor.predict(altered)
        self.assertFalse(np.allclose(preds_orig, preds_alt),
                         "Different inputs should yield different predictions")

    def test_confidence_range(self):
        """Confidence values should be between 0 and 1 inclusive"""
        _, conf, _ = self.predictor.predict_with_uncertainty(
            self.valid_data, n_samples=20)
        self.assertTrue(np.all(conf >= 0) and np.all(conf <= 1),
                        "Confidence must be within [0, 1]")

    def test_predict_missing_columns_raises(self):
        """predict() should error if required feature column is missing"""
        invalid = self.valid_data.drop(columns=['MW'])
        with self.assertRaises(ComponentLoadError):
            self.predictor.predict(invalid)

    def test_predict_with_uncertainty_default(self):
        """predict_with_uncertainty() without specifying n_samples"""
        mean_vals, uncertainty, std = self.predictor.predict_with_uncertainty(
            self.valid_data)
        self.assertIsInstance(mean_vals, np.ndarray)
        self.assertIsInstance(uncertainty, np.ndarray)
        self.assertEqual(mean_vals.shape, uncertainty.shape)
        self.assertEqual(mean_vals.shape[1], 5)
        direct = self.predictor.predict(self.valid_data)
        np.testing.assert_allclose(mean_vals, direct, rtol=1e-5, atol=1e-3)

    def test_predict_with_uncertainty_n_samples_variation(self):
        """Std deviation should reflect different sample sizes"""
        _, uncertainty10, _ = self.predictor.predict_with_uncertainty(
            self.valid_data, n_samples=10)
        _, unceratinty50, _ = self.predictor.predict_with_uncertainty(
            self.valid_data, n_samples=50)
        self.assertFalse(np.allclose(uncertainty10, unceratinty50),
                         "Std dev should vary with different n_samples")

    def test_update_returns_expected_keys_and_methods(self):
        """update() returns a dict with timestamp and component instances"""
        result = self.predictor.update(self.valid_update)
        for key in ('updated_at', 'dataprocessor', 'transformer', 'model', 'predictor'):
            self.assertIn(key, result)
        dp = result['dataprocessor']
        self.assertTrue(hasattr(dp, 'process_train')
                        and callable(dp.process_train))
        tr = result['transformer']
        self.assertTrue(hasattr(tr, 'transform') and callable(tr.transform))
        self.assertTrue(hasattr(tr, 'fit_transform')
                        and callable(tr.fit_transform))

    def test_predict_after_update(self):
        """After an update, predict() still returns valid predictions"""
        # Perform update
        self.predictor.update(self.valid_update)
        preds = self.predictor.predict(self.valid_data)
        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(preds.ndim, 2)
        self.assertEqual(preds.shape[1], 5)

        # self.predictor.update(pd.DataFrame(), train_full=True)
        # preds = self.predictor.predict(self.valid_data)
        # self.assertIsInstance(preds, np.ndarray)
        # self.assertEqual(preds.ndim, 2)
        # self.assertEqual(preds.shape[1], 5)
        # print(preds)

    def test_update_train_full_uses_full_dataset(self):
        """update(train_full=True) should pull data from form_ctrl.get_all_as_dataframe"""
        custom_df = self.valid_update.copy()
        self.predictor.form_ctrl.get_all_as_dataframe = lambda encoded=True: custom_df
        result = self.predictor.update(pd.DataFrame(), train_full=True)
        self.assertIn('updated_at', result)
        self.assertIn('model', result)
        del self.predictor.form_ctrl.get_all_as_dataframe

    def test_update_twice_changes_timestamp(self):
        """Subsequent update() calls should produce increasing timestamps"""
        r1 = self.predictor.update(self.valid_update)
        t1 = r1['updated_at']
        r2 = self.predictor.update(self.valid_update)
        t2 = r2['updated_at']
        self.assertGreater(t2, t1)

    def test_update_invalid_data_raises(self):
        """update() should error if a viscosity column is missing"""
        invalid_update = self.valid_update.drop(columns=['Viscosity_100'])
        with self.assertRaises(ComponentLoadError):
            self.predictor.update(invalid_update)

    def test_missing_bundle_raises_error(self):
        """Initializing with a nonexistent path raises ComponentLoadError"""
        with self.assertRaises(ComponentLoadError):
            Predictor(Path('/nonexistent/path'))


class TestInternalMethods(unittest.TestCase):
    """Tests for private helper methods in Executor."""

    class DummyExecutor(Predictor):
        def __init__(self):
            pass  # Bypass full initialization

    def setUp(self):
        self.dummy = self.DummyExecutor()

    def test_extract_if_archive_directory(self):
        """_extract_if_archive should return the same directory when given a folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            result = self.dummy._extract_if_archive(path)
            self.assertEqual(result, path.resolve())

    def test_extract_if_archive_zip(self):
        """_extract_if_archive should extract a zip file to a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / 'test.zip'
            dummy_dir = Path(tmpdir) / 'pkg'
            nested = dummy_dir / 'file.txt'
            dummy_dir.mkdir()
            nested.write_text('hello')
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.write(nested, arcname='pkg/file.txt')
            result = self.dummy._extract_if_archive(zip_path)
            self.assertTrue((Path(result) / 'pkg' / 'file.txt').is_file())
            self.assertIsNotNone(self.dummy._tempdir)

    def test_extract_if_archive_nonexistent_path_raises(self):
        """_extract_if_archive should raise ComponentLoadError for invalid paths."""
        with self.assertRaises(ComponentLoadError):
            self.dummy._extract_if_archive(Path('/unlikely/to/exist/path'))

    def test_load_pyc_module_missing_pycache_raises(self):
        """_load_pyc_module should error if '__pycache__' is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_dir = Path(tmpdir)
            with self.assertRaises(ComponentLoadError):
                self.dummy._load_pyc_module(pkg_dir, 'anymodule')

    def test_load_pyc_module_no_matching_pyc_raises(self):
        """_load_pyc_module should error if no matching .pyc found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_dir = Path(tmpdir)
            (pkg_dir / '__pycache__').mkdir()
            unrelated = pkg_dir / '__pycache__' / 'othermodule.cpython-38.pyc'
            unrelated.write_bytes(b'')
            with self.assertRaises(ComponentLoadError):
                self.dummy._load_pyc_module(pkg_dir, 'targetmodule')


if __name__ == '__main__':
    unittest.main()
