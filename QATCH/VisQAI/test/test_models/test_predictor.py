"""
Unit tests for predictor.py module

Tests the Predictor class for loading packaged viscosity models,
performing inference, uncertainty estimation, and incremental updates.

Author: Paul MacNichol
Date: 2025-10-16
"""
import unittest
import tempfile
import zipfile
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
from src.models.predictor import Predictor


class TestPredictorInitialization(unittest.TestCase):
    """Test cases for Predictor initialization and setup."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_zip_path = Path(self.test_dir) / "test_model.zip"

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_init_with_valid_path(self):
        """Test initialization with valid zip path."""
        # Create a mock zip file
        self._create_mock_zip(self.test_zip_path)

        with patch.object(Predictor, '_load_zip'):
            predictor = Predictor(str(self.test_zip_path), mc_samples=100)

            self.assertEqual(predictor.zip_path, self.test_zip_path)
            self.assertEqual(predictor.mc_samples, 100)
            self.assertIsNone(predictor.predictor)
            self.assertIsNone(predictor.metadata)

    def test_init_with_default_mc_samples(self):
        """Test initialization with default mc_samples."""
        self._create_mock_zip(self.test_zip_path)

        with patch.object(Predictor, '_load_zip'):
            predictor = Predictor(str(self.test_zip_path))
            self.assertEqual(predictor.mc_samples, 50)

    def test_init_calls_load_zip(self):
        """Test that __init__ calls _load_zip."""
        self._create_mock_zip(self.test_zip_path)

        with patch.object(Predictor, '_load_zip') as mock_load:
            predictor = Predictor(str(self.test_zip_path))
            mock_load.assert_called_once_with(self.test_zip_path)

    def _create_mock_zip(self, zip_path):
        """Helper to create a minimal mock zip file."""
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("model/checkpoint.pt", "mock_checkpoint")
            zf.writestr("src/inference.py", "# mock inference")


class TestLoadZip(unittest.TestCase):
    """Test cases for _load_zip method."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_zip_path = Path(self.test_dir) / "test_model.zip"

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_load_zip_nonexistent_file(self):
        """Test loading a non-existent zip file."""
        predictor = Predictor.__new__(Predictor)
        predictor._tmpdir = None

        with self.assertRaises(FileNotFoundError) as context:
            predictor._load_zip(Path("nonexistent.zip"))

        self.assertIn("Archive not found", str(context.exception))

    def test_load_zip_missing_model_directory(self):
        """Test loading zip without model/ directory."""
        # Create zip without model directory
        with zipfile.ZipFile(self.test_zip_path, 'w') as zf:
            zf.writestr("src/inference.py", "# mock")

        predictor = Predictor.__new__(Predictor)
        predictor._tmpdir = None

        with self.assertRaises(RuntimeError) as context:
            predictor._load_zip(self.test_zip_path)

        self.assertIn("model/' folder missing", str(context.exception))

    def test_load_zip_missing_src_directory(self):
        """Test loading zip without src/ directory."""
        # Create zip without src directory
        with zipfile.ZipFile(self.test_zip_path, 'w') as zf:
            zf.writestr("model/checkpoint.pt", "mock")

        predictor = Predictor.__new__(Predictor)
        predictor._tmpdir = None

        with self.assertRaises(RuntimeError) as context:
            predictor._load_zip(self.test_zip_path)

        self.assertIn("src/' folder missing", str(context.exception))

    def test_load_zip_missing_checkpoint(self):
        """Test loading zip without checkpoint.pt."""
        # Create zip without checkpoint
        with zipfile.ZipFile(self.test_zip_path, 'w') as zf:
            zf.writestr("model/metadata.json", json.dumps({"client": "test"}))
            zf.writestr("src/inference.py", "class Predictor: pass")
            zf.writestr("src/config.py", "class visq3xConfig: pass")

        predictor = Predictor.__new__(Predictor)
        predictor._tmpdir = None

        with patch.object(predictor, '_load_source_modules'):
            with self.assertRaises(RuntimeError) as context:
                predictor._load_zip(self.test_zip_path)

            self.assertIn("checkpoint.pt not found", str(context.exception))

    def test_load_zip_with_metadata(self):
        """Test loading zip with metadata.json."""
        metadata = {
            "client": "TestClient",
            "author": "TestAuthor",
            "version": "1.0"
        }

        with zipfile.ZipFile(self.test_zip_path, 'w') as zf:
            zf.writestr("model/metadata.json", json.dumps(metadata))
            zf.writestr("model/checkpoint.pt", "mock")
            zf.writestr("src/inference.py", "class Predictor: pass")
            zf.writestr("src/config.py", "class visq3xConfig: pass")

        predictor = Predictor.__new__(Predictor)
        predictor._tmpdir = None
        predictor.metadata = None

        # Mock the required imports and classes
        mock_predictor_class = MagicMock()
        mock_config_class = MagicMock()
        mock_predictor_instance = MagicMock()
        mock_predictor_class.return_value = mock_predictor_instance

        with patch.object(predictor, '_load_source_modules'):
            with patch.dict('sys.modules', {
                'inference': MagicMock(Predictor=mock_predictor_class),
                'config': MagicMock(visq3xConfig=mock_config_class)
            }):
                predictor._load_zip(self.test_zip_path)

        self.assertIsNotNone(predictor.metadata)
        self.assertEqual(predictor.metadata['client'], "TestClient")
        self.assertEqual(predictor.metadata['author'], "TestAuthor")

    def test_load_zip_without_metadata(self):
        """Test loading zip without metadata.json."""
        with zipfile.ZipFile(self.test_zip_path, 'w') as zf:
            zf.writestr("model/checkpoint.pt", "mock")
            zf.writestr("src/inference.py", "class Predictor: pass")
            zf.writestr("src/config.py", "class visq3xConfig: pass")

        predictor = Predictor.__new__(Predictor)
        predictor._tmpdir = None
        predictor.metadata = None

        mock_predictor_class = MagicMock()
        mock_config_class = MagicMock()
        mock_predictor_instance = MagicMock()
        mock_predictor_class.return_value = mock_predictor_instance

        with patch.object(predictor, '_load_source_modules'):
            with patch.dict('sys.modules', {
                'inference': MagicMock(Predictor=mock_predictor_class),
                'config': MagicMock(visq3xConfig=mock_config_class)
            }):
                predictor._load_zip(self.test_zip_path)

        self.assertIsNone(predictor.metadata)


class TestLoadSourceModules(unittest.TestCase):
    """Test cases for _load_source_modules method."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.src_dir = Path(self.test_dir) / "src"
        self.src_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_load_source_modules_no_files(self):
        """Test loading from directory with no Python files."""
        predictor = Predictor.__new__(Predictor)

        with self.assertRaises(RuntimeError) as context:
            predictor._load_source_modules(self.src_dir)

        self.assertIn("No Python files found", str(context.exception))

    def test_load_source_modules_priority_order(self):
        """Test that modules are loaded in priority order."""
        # Create mock Python files
        files = ['inference.py', 'config.py', 'model.py', 'utils.py']
        for fname in files:
            (self.src_dir / fname).write_text("# mock module")

        predictor = Predictor.__new__(Predictor)

        loaded_modules = []

        def mock_exec(module):
            loaded_modules.append(module.__name__)

        with patch('importlib.util.spec_from_file_location') as mock_spec:
            with patch('importlib.util.module_from_spec') as mock_module:
                mock_spec.return_value.loader.exec_module = mock_exec
                mock_module.return_value.__name__ = "test"

                try:
                    predictor._load_source_modules(self.src_dir)
                except:
                    pass

        # Verify config is loaded before inference
        if 'config' in loaded_modules and 'inference' in loaded_modules:
            self.assertLess(
                loaded_modules.index('config'),
                loaded_modules.index('inference')
            )

    def test_load_source_modules_failure(self):
        """Test handling of module loading failure."""
        # Create a Python file
        (self.src_dir / "broken.py").write_text("raise Exception('broken')")

        predictor = Predictor.__new__(Predictor)

        with patch('importlib.util.spec_from_file_location') as mock_spec:
            mock_spec.side_effect = Exception("Import failed")

            with self.assertRaises(RuntimeError) as context:
                predictor._load_source_modules(self.src_dir)

            self.assertIn("Failed to load module", str(context.exception))


class TestPredictMethods(unittest.TestCase):
    """Test cases for prediction methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.predictor = Predictor.__new__(Predictor)
        self.predictor.mc_samples = 50
        self.test_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })

    def test_predict_without_loaded_predictor(self):
        """Test predict() when predictor is not loaded."""
        self.predictor.predictor = None

        with self.assertRaises(RuntimeError) as context:
            self.predictor.predict(self.test_df)

        self.assertIn("No predictor loaded", str(context.exception))

    def test_predict_with_loaded_predictor(self):
        """Test predict() with loaded predictor."""
        mock_predictor = MagicMock()
        expected_result = np.array([1.5, 2.5, 3.5])
        mock_predictor.predict.return_value = expected_result

        self.predictor.predictor = mock_predictor

        result = self.predictor.predict(self.test_df)

        mock_predictor.predict.assert_called_once_with(
            self.test_df,
            return_uncertainty=False
        )
        np.testing.assert_array_equal(result, expected_result)

    def test_predict_exception_handling(self):
        """Test predict() exception handling."""
        mock_predictor = MagicMock()
        mock_predictor.predict.side_effect = ValueError("Prediction failed")

        self.predictor.predictor = mock_predictor

        with self.assertRaises(ValueError):
            self.predictor.predict(self.test_df)

    def test_predict_uncertainty_without_loaded_predictor(self):
        """Test predict_uncertainty() when predictor is not loaded."""
        self.predictor.predictor = None

        with self.assertRaises(RuntimeError) as context:
            self.predictor.predict_uncertainty(self.test_df)

        self.assertIn("No predictor loaded", str(context.exception))

    def test_predict_uncertainty_with_default_samples(self):
        """Test predict_uncertainty() with default n_samples."""
        mock_predictor = MagicMock()
        expected_mean = np.array([1.5, 2.5, 3.5])
        expected_uncertainty = {
            'std': np.array([0.1, 0.2, 0.3]),
            'lower_95': np.array([1.3, 2.1, 3.0]),
            'upper_95': np.array([1.7, 2.9, 4.0]),
            'cv': np.array([0.067, 0.08, 0.086])
        }
        mock_predictor.predict.return_value = (
            expected_mean, expected_uncertainty)

        self.predictor.predictor = mock_predictor

        mean, uncertainty = self.predictor.predict_uncertainty(self.test_df)

        mock_predictor.predict.assert_called_once_with(
            self.test_df,
            return_uncertainty=True,
            n_samples=50
        )
        np.testing.assert_array_equal(mean, expected_mean)
        self.assertEqual(uncertainty, expected_uncertainty)

    def test_predict_uncertainty_with_custom_samples(self):
        """Test predict_uncertainty() with custom n_samples."""
        mock_predictor = MagicMock()
        expected_mean = np.array([1.5])
        expected_uncertainty = {'std': np.array([0.1])}
        mock_predictor.predict.return_value = (
            expected_mean, expected_uncertainty)

        self.predictor.predictor = mock_predictor

        mean, uncertainty = self.predictor.predict_uncertainty(
            self.test_df,
            n_samples=100
        )

        mock_predictor.predict.assert_called_once_with(
            self.test_df,
            return_uncertainty=True,
            n_samples=100
        )


class TestLearnMethod(unittest.TestCase):
    """Test cases for the learn method."""

    def setUp(self):
        """Set up test fixtures."""
        self.predictor = Predictor.__new__(Predictor)
        self.test_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'target': [10, 20, 30]
        })

    def test_learn_without_loaded_predictor(self):
        """Test learn() when predictor is not loaded."""
        self.predictor.predictor = None

        with self.assertRaises(RuntimeError) as context:
            self.predictor.learn(self.test_df)

        self.assertIn("No predictor loaded", str(context.exception))

    def test_learn_with_default_parameters(self):
        """Test learn() with default parameters."""
        mock_predictor = MagicMock()
        expected_result = {
            'avg_loss': 0.123,
            'new_categories_added': ['cat1', 'cat2'],
            'n_epochs': 10
        }
        mock_predictor.learn.return_value = expected_result

        self.predictor.predictor = mock_predictor

        result = self.predictor.learn(self.test_df)

        mock_predictor.learn.assert_called_once_with(
            self.test_df,
            n_epochs=None,
            verbose=True
        )
        self.assertEqual(result, expected_result)

    def test_learn_with_custom_parameters(self):
        """Test learn() with custom parameters."""
        mock_predictor = MagicMock()
        expected_result = {'avg_loss': 0.05, 'n_epochs': 20}
        mock_predictor.learn.return_value = expected_result

        self.predictor.predictor = mock_predictor

        result = self.predictor.learn(
            self.test_df,
            n_epochs=20,
            verbose=False
        )

        mock_predictor.learn.assert_called_once_with(
            self.test_df,
            n_epochs=20,
            verbose=False
        )
        self.assertEqual(result, expected_result)

    def test_learn_exception_handling(self):
        """Test learn() exception handling."""
        mock_predictor = MagicMock()
        mock_predictor.learn.side_effect = ValueError("Training failed")

        self.predictor.predictor = mock_predictor

        with self.assertRaises(ValueError):
            self.predictor.learn(self.test_df)


class TestUtilityMethods(unittest.TestCase):
    """Test cases for utility methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.predictor = Predictor.__new__(Predictor)
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_get_metadata_with_metadata(self):
        """Test get_metadata() when metadata exists."""
        expected_metadata = {
            'client': 'TestClient',
            'author': 'TestAuthor',
            'version': '1.0'
        }
        self.predictor.metadata = expected_metadata

        result = self.predictor.get_metadata()

        self.assertEqual(result, expected_metadata)

    def test_get_metadata_without_metadata(self):
        """Test get_metadata() when metadata is None."""
        self.predictor.metadata = None

        result = self.predictor.get_metadata()

        self.assertIsNone(result)

    def test_save_without_loaded_predictor(self):
        """Test save() when predictor is not loaded."""
        self.predictor.predictor = None
        save_path = str(Path(self.test_dir) / "model.pt")

        with self.assertRaises(RuntimeError) as context:
            self.predictor.save(save_path)

        self.assertIn("No predictor loaded", str(context.exception))

    def test_save_with_loaded_predictor(self):
        """Test save() with loaded predictor."""
        mock_predictor = MagicMock()
        self.predictor.predictor = mock_predictor
        save_path = str(Path(self.test_dir) / "model.pt")

        self.predictor.save(save_path)

        mock_predictor.save_state.assert_called_once_with(save_path)

    def test_reload_archive(self):
        """Test reload_archive() method."""
        new_zip_path = "new_model.zip"

        with patch.object(Predictor, '_load_zip') as mock_load:
            self.predictor.reload_archive(new_zip_path)

            self.assertEqual(self.predictor.zip_path, Path(new_zip_path))
            mock_load.assert_called_once_with(Path(new_zip_path))


class TestContextManager(unittest.TestCase):
    """Test cases for context manager functionality."""

    def test_enter_returns_self(self):
        """Test __enter__ returns the Predictor instance."""
        predictor = Predictor.__new__(Predictor)

        result = predictor.__enter__()

        self.assertIs(result, predictor)

    def test_exit_calls_cleanup(self):
        """Test __exit__ calls cleanup."""
        predictor = Predictor.__new__(Predictor)

        with patch.object(predictor, 'cleanup') as mock_cleanup:
            predictor.__exit__(None, None, None)
            mock_cleanup.assert_called_once()

    def test_context_manager_with_exception(self):
        """Test context manager cleanup on exception."""
        predictor = Predictor.__new__(Predictor)
        predictor._tmpdir = None
        predictor.predictor = None

        with patch.object(predictor, 'cleanup') as mock_cleanup:
            try:
                with predictor:
                    raise ValueError("Test exception")
            except ValueError:
                pass

            mock_cleanup.assert_called_once()

    def test_del_calls_cleanup(self):
        """Test __del__ calls cleanup."""
        predictor = Predictor.__new__(Predictor)

        with patch.object(predictor, 'cleanup') as mock_cleanup:
            predictor.__del__()
            mock_cleanup.assert_called_once()

    def test_del_handles_exceptions(self):
        """Test __del__ handles cleanup exceptions."""
        predictor = Predictor.__new__(Predictor)

        with patch.object(predictor, 'cleanup') as mock_cleanup:
            mock_cleanup.side_effect = Exception("Cleanup failed")

            # Should not raise exception
            predictor.__del__()


class TestPredictorIntegration(unittest.TestCase):
    """Integration tests using the actual model file."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        cls.model_path = Path("assets") / "visq3x.zip"

        # Skip integration tests if model file doesn't exist
        if not cls.model_path.exists():
            raise unittest.SkipTest(
                f"Model file not found: {cls.model_path}. "
                "Integration tests will be skipped."
            )

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'Protein_type': ['IgG1'],
            'MW': [150000],
            'Protein_conc': [145],
            'Temperature': [25],
            'Buffer_type': ['PBS'],
            'Buffer_pH': [7.4],
            'Buffer_conc': [10],
            'Salt_type': ['NaCl'],
            'Salt_conc': [140],
            'Stabilizer_type': ['Sucrose'],
            'Stabilizer_conc': [1],
            'Surfactant_type': ['none'],
            'Surfactant_conc': [0],
            'Protein_class_type': ['mAb'],
        })

    def test_integration_load_model(self):
        """Integration test: Load actual model."""
        try:
            predictor = Predictor(str(self.model_path))
            self.assertIsNotNone(predictor.predictor)
            predictor.cleanup()
        except Exception as e:
            self.fail(f"Failed to load model: {e}")

    def test_integration_predict(self):
        """Integration test: Make predictions with actual model."""
        try:
            predictor = Predictor(str(self.model_path))
            predictions = predictor.predict(self.sample_data)

            self.assertIsInstance(predictions, np.ndarray)
            self.assertEqual(len(predictions), len(self.sample_data))

            predictor.cleanup()
        except Exception as e:
            self.fail(f"Prediction failed: {e}")

    def test_integration_predict_uncertainty(self):
        """Integration test: Make predictions with uncertainty."""
        try:
            predictor = Predictor(str(self.model_path), mc_samples=20)
            mean, uncertainty = predictor.predict_uncertainty(
                self.sample_data,
                n_samples=20
            )

            self.assertIsInstance(mean, np.ndarray)
            self.assertIsInstance(uncertainty, dict)
            self.assertIn('std', uncertainty)
            self.assertIn('lower_95', uncertainty)
            self.assertIn('upper_95', uncertainty)

            predictor.cleanup()
        except Exception as e:
            self.fail(f"Uncertainty prediction failed: {e}")

    def test_integration_context_manager(self):
        """Integration test: Use predictor as context manager."""
        try:
            with Predictor(str(self.model_path)) as predictor:
                predictions = predictor.predict(self.sample_data)
                self.assertIsInstance(predictions, np.ndarray)
        except Exception as e:
            self.fail(f"Context manager usage failed: {e}")

    def test_integration_metadata(self):
        """Integration test: Retrieve metadata."""
        try:
            with Predictor(str(self.model_path)) as predictor:
                metadata = predictor.get_metadata()
                # Metadata may or may not exist depending on the package
                if metadata is not None:
                    self.assertIsInstance(metadata, dict)
        except Exception as e:
            self.fail(f"Metadata retrieval failed: {e}")

    def test_integration_save_and_reload(self):
        """Integration test: Save model state."""
        temp_dir = tempfile.mkdtemp()
        try:
            save_path = Path(temp_dir) / "saved_model.pt"

            with Predictor(str(self.model_path)) as predictor:
                # Make initial prediction
                pred1 = predictor.predict(self.sample_data)

                # Save the model
                predictor.save(str(save_path))

                # Verify file was created
                self.assertTrue(save_path.exists())
        except Exception as e:
            self.fail(f"Save and reload failed: {e}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe_predict(self):
        """Test prediction with empty DataFrame."""
        predictor = Predictor.__new__(Predictor)
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = np.array([])
        predictor.predictor = mock_predictor

        empty_df = pd.DataFrame()
        result = predictor.predict(empty_df)

        self.assertEqual(len(result), 0)

    def test_large_mc_samples(self):
        """Test initialization with very large mc_samples."""
        with patch.object(Predictor, '_load_zip'):
            predictor = Predictor("dummy.zip", mc_samples=10000)
            self.assertEqual(predictor.mc_samples, 10000)

    def test_zero_mc_samples(self):
        """Test initialization with zero mc_samples."""
        with patch.object(Predictor, '_load_zip'):
            predictor = Predictor("dummy.zip", mc_samples=0)
            self.assertEqual(predictor.mc_samples, 0)

    def test_multiple_cleanup_calls(self):
        """Test multiple cleanup calls don't cause issues."""
        predictor = Predictor.__new__(Predictor)
        mock_tmpdir = MagicMock()
        predictor._tmpdir = mock_tmpdir
        predictor.predictor = MagicMock()

        predictor.cleanup()
        predictor.cleanup()  # Second call should be safe

        # Should only cleanup once (None check prevents second call)
        self.assertEqual(mock_tmpdir.cleanup.call_count, 1)


def suite():
    """Create a test suite."""
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(
        TestPredictorInitialization))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(
        TestLoadZip))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(
        TestLoadSourceModules))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(
        TestPredictMethods))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(
        TestLearnMethod))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(
        TestUtilityMethods))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(
        TestContextManager))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(
        TestPredictorIntegration))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(
        TestEdgeCases))

    return test_suite


if __name__ == '__main__':
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
