"""
test_predictor.py

This module provides dummy data loaders, transformers, models, and executors to test
the functionality of the Predictor implementation. It uses the unittest framework
to verify that Predictor correctly loads components from pickle or joblib files,
handles missing or invalid components, processes inputs, and raises appropriate
ComponentLoadError exceptions on failure.

The dummy classes emulate minimal behavior for:
- Data loading (process or load)
- Transformation (transform)
- Model prediction (predict)
- Executor behavior (set_model/predict or attribute-based model assignment)

Utilities are included to dump objects to pickle or joblib files for testing.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-06-03

Version:
    1.1
"""

import zipfile
import pickle
import tempfile
import unittest
from pathlib import Path
import datetime
try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

from src.models.predictor import Predictor, ComponentLoadError, _KERAS_AVAILABLE


class DummyDataProcessorProcessOnly:
    """
    Dummy data loader that only implements `process` and `load` by adding 10 to the input.

    This class simulates a data loader whose `process` and `load` methods both take a raw
    input value and return the value incremented by 10.

    Methods:
        process(raw): Increment `raw` by 10.
        load(raw): Call `process(raw)` and return its result.
    """

    def process(self, raw_X, raw_y=None):
        """
        Process the raw input by adding 10.

        Args:
            raw_X (numeric): The raw input data.
            raw_y (numeric): The raw target data.

        Returns:
            numeric: The processed data (raw + 10).
        """
        if raw_y is None:
            return raw_X + 10
        else:
            return raw_X, raw_y

    def load(self, raw):
        """
        Load the raw input by delegating to `process`.

        Args:
            raw (numeric): The raw input data.

        Returns:
            numeric: The loaded data (same as processed result, raw + 10).
        """
        return self.process(raw)


class DummyDataProcessorLoadOnly:
    """
    Dummy data loader that only implements `load` and `process` by doubling the input.

    This class simulates a data loader whose `load` and `process` methods both take a raw
    input value and return the value multiplied by 2.

    Methods:
        load(raw): Multiply `raw` by 2.
        process(raw): Call `load(raw)` and return its result.
    """

    def load(self, raw):
        """
        Load the raw input by multiplying it by 2.

        Args:
            raw (numeric): The raw input data.

        Returns:
            numeric: The loaded data (raw * 2).
        """
        return raw * 2

    def process(self, raw):
        """
        Process the raw input by delegating to `load`.

        Args:
            raw (numeric): The raw input data.

        Returns:
            numeric: The processed data (same as loaded result, raw * 2).
        """
        return self.load(raw)


class DummyDataProcessorBad:
    """
    Dummy data loader with neither `process` nor `load` methods.

    Used to simulate missing-method errors when Predictor tries to load this component.
    """

    def some_other(self):
        """
        Placeholder method that does nothing.

        Returns:
            int: Always returns 0.
        """
        return 0


class DummyTransformer:
    """
    Dummy transformer that subtracts 1 from the input.

    Methods:
        transform(x): Subtract 1 from `x`.
    """

    def transform(self, X):
        """
        Transform the input data by subtracting 1.

        Args:
            x (numeric): The input data to transform.

        Returns:
            numeric: The transformed data (x - 1).
        """
        return X - 1

    def fit_transform(self, X, y):
        """
        Transform and fit the input data x, y.

        Args:
            x (numeric): The input data to transform.
            y (numeric): The target data to transform.

        Returns:
            numeric: The transformed data (x, y).
        """
        return X, y


class DummyTransformerBad:
    """
    Dummy transformer that always raises an exception inside `transform`.

    Used to simulate a failing transformer during Predictor prediction.
    """

    def transform(self, X):
        """
        Attempt to transform the input but always raise RuntimeError.

        Args:
            x (any): The input data.

        Raises:
            RuntimeError: Always raised to indicate transformer failure.
        """
        raise RuntimeError("transformer failure")

    def fit_transform(self, X, y):
        """
        Attempt to transform the input but always raise RuntimeError.

        Args:
            x (any): The input data.
            y (any): The target vector

        Raises:
            RuntimeError: Always raised to indicate transformer failure.
        """
        raise RuntimeError("transformer failure")


class DummyModel:
    """
    Dummy model that multiplies inputs by 3.

    Methods:
        predict(X): Multiply `X` by 3.
    """

    def predict(self, X):
        """
        Predict method that multiplies input data by 3.

        Args:
            X (numeric or array-like): The input features.

        Returns:
            numeric or array-like: The prediction result (X * 3).
        """
        return X * 3

    def fit(self, X, y):
        """
        Fit method that returns input data and the target data.

        Args:
            X (numeric or array-like): The input features.
            X (numeric or array-like): The target features.

        Returns:
            numeric or array-like: The prediction result X, y.
        """
        return X, y


class DummyPredictor:
    """
    Dummy executor that has `set_model(...)` and `predict(...)`.

    The executor stores a `model` attribute when `set_model` is called and then delegates
    its `predict` method to `model.predict(data)`.

    Attributes:
        model: Initially `None`, set via `set_model(...)`.
    """

    def __init__(self):
        """
        Initialize the DummyExecutorWithSetModel.

        Sets `self.model` to None initially.
        """
        self.model: DummyModel = None

    def set_model(self, model):
        """
        Set the internal model used for prediction.

        Args:
            model: An object with a `predict(...)` method.
        """
        self.model = model

    def predict(self, X):
        """
        Predict using the internal model's `predict` method.

        Args:
            X (numeric or array-like): Input data for prediction.

        Returns:
            numeric or array-like: The output of `model.predict(X)`.
        """
        return self.model.predict(X)

    def update(self, X, y, model: DummyModel):
        """
        Update using the internal model's `update` method.

        Args:
            X (numeric or array-like): Input data for fit.
            y (numeric or array-like): Target data for fit.

        Returns:
            numeric or array-like: The output of `model.fit(X, y)`.
        """
        return model.fit(X, y)


class DummyPredictorWithAttr:
    """
    Dummy executor without `set_model`; relies on attribute assignment.

    Predictor will assign `model` as an attribute directly. The `predict(...)` method calls
    `self.model.predict(X)`. Raises `AttributeError` if `model` is not set.
    """

    def predict(self, X):
        """
        Predict by delegating to `self.model.predict`.

        Args:
            X (numeric or array-like): Input data for prediction.

        Returns:
            numeric or array-like: The output of `model.predict(X)`.

        Raises:
            AttributeError: If `self.model` has not been set by Predictor.
        """
        return self.model.predict(X)

    def update(self, X, y):
        """
        Update by delegating to `self.model.update`.

        Args:
            X (numeric or array-like): Input data for fit.
            y (numeric or array-like): Target data for fit.

        Returns:
            numeric or array-like: The output of `model.predict(X)`.

        Raises:
            AttributeError: If `self.model` has not been set by Predictor.
        """
        return self.model.update(X, y)


class DummyExecutorBad:
    """
    Dummy executor missing the `predict` method.

    Used to simulate missing-method errors when Predictor tries to load this component.
    """

    def some_other(self):
        """
        Placeholder method that does nothing.

        Returns:
            int: Always returns 0.
        """
        return 0


class DummyExecutorFailPredict:
    """
    Dummy executor whose `predict(...)` always raises an exception.

    Used to simulate an executor runtime failure during Predictor prediction.
    """

    def __init__(self):
        """
        Initialize the DummyExecutorFailPredict with a placeholder DummyModel.
        """
        self.model = DummyModel()

    def predict(self, X):
        """
        Attempt to predict but always raise ValueError.

        Args:
            X (numeric or array-like): Input data for prediction.

        Raises:
            ValueError: Always raised to simulate executor failure.
        """
        raise ValueError("predictor failure")

    def update(self, X, y):
        """
        Attempt to update but always raise ValueError.
        Args:
            X (numeric or array-like): Input data for fit.
            y (numeric or array-like): Target data for fit.

        Raises:
            ValueError: Always raised to simulate executor failure.
        """
        raise ValueError("predictor failure")


def _dump_pickle(obj, path: Path):
    """
    Serialize `obj` to a file at `path` using pickle.

    Args:
        obj (any): The Python object to serialize.
        path (Path): The destination file path for the pickle.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _dump_joblib(obj, path: Path):
    """
    Serialize `obj` to a file at `path` using joblib.

    Args:
        obj (any): The Python object to serialize.
        path (Path): The destination file path for the joblib dump.

    Raises:
        unittest.SkipTest: If joblib is not installed.
    """
    if not _JOBLIB_AVAILABLE:
        raise unittest.SkipTest("joblib not installed")
    joblib.dump(obj, str(path))


class TestPredictor(unittest.TestCase):
    """
    Unit test suite for the Predictor class.

    Tests cover:
    1. Loading all components (data loader, transformer, model, executor) from pickle files.
    2. Loading components from joblib files when available.
    3. Case-insensitive filename lookups.
    4. Loading from a ZIP archive containing component files.
    5. Missing component scenarios raising ComponentLoadError.
    6. Invalid component methods raising ComponentLoadError.
    7. Runtime failures in transformer, data loader, or executor raising ComponentLoadError.
    8. Keras-specific loading errors for SavedModel directories or .h5 files.
    9. Handling invalid ZIP files or nonexistent paths.

    Uses temporary directories created in `setUp` and cleaned in `tearDown`.
    """

    def setUp(self):
        """
        Set up a temporary directory before each test method.

        Creates a new temporary folder and stores its Path in `self.tempdir`.
        """
        self.tempdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """
        Clean up the temporary directory after each test method.

        Recursively deletes all files and subdirectories under `self.tempdir`.
        """
        for child in self.tempdir.rglob("*"):
            try:
                if child.is_file():
                    child.unlink()
                else:
                    child.rmdir()
            except OSError:
                pass
        try:
            self.tempdir.rmdir()
        except OSError:
            pass

    def _make_component_folder(
        self,
        use_joblib: bool = False,
        include_transformer: bool = True,
        executor_has_set_model: bool = True,
        uppercase_names: bool = False
    ) -> Path:
        """
        Create a folder with serialized dummy components for Predictor tests.

        Args:
            use_joblib (bool): If True, dump components with joblib (.joblib); otherwise pickle (.pkl).
            include_transformer (bool): If False, remove the transformer file after creation.
            executor_has_set_model (bool): If True, use DummyExecutorWithSetModel; otherwise DummyExecutorWithAttr.
            uppercase_names (bool): If True, write component filenames in uppercase (e.g., DATAPROCESSOR.PKL).

        Returns:
            Path: The path to the created components folder.
        """
        folder = self.tempdir / "components"
        folder.mkdir()

        ext = ".joblib" if use_joblib else ".pkl"
        name_map = {
            "dataprocessor": DummyDataProcessorProcessOnly(),
            "preprocessor": DummyTransformer(),
            "model": DummyModel(),
            "predictor": DummyPredictor() if executor_has_set_model else DummyPredictorWithAttr(),
        }

        for base, obj in name_map.items():
            fname = base + ext
            if uppercase_names:
                fname = fname.upper()
            target = folder / fname
            if use_joblib:
                _dump_joblib(obj, target)
            else:
                _dump_pickle(obj, target)

        if not include_transformer:
            for child in folder.iterdir():
                if child.name.lower().startswith("preprocessor"):
                    child.unlink()
                    break

        return folder

    def test_predictor_with_pickle_all_components(self):
        """
        Test loading all components from pickle files.

        Verifies that Predictor correctly loads:
        - DummyDataLoaderProcessOnly as `data_loader`
        - DummyTransformer as `transformer`
        - DummyModel as `model`
        - DummyExecutorWithSetModel as `executor`
        Also checks that executor.model is set to `model` and
        that `predict(5)` returns the expected value 42.
        """
        comp_dir = self._make_component_folder(
            use_joblib=False,
            include_transformer=True,
            executor_has_set_model=True,
            uppercase_names=False
        )

        p = Predictor(comp_dir)

        self.assertIsInstance(p.data_processor, DummyDataProcessorProcessOnly)
        self.assertIsInstance(p.transformer, DummyTransformer)
        self.assertIsInstance(p.model, DummyModel)
        self.assertIsInstance(p.predictor, DummyPredictor)

        self.assertIs(p.predictor.model, p.model)

        result = p.predict(5)
        self.assertEqual(result, 42)

    def test_predictor_with_joblib_no_transformer_attr_executor(self):
        """
        Test loading components from joblib files without a transformer and using attribute-based executor.

        Requires joblib to be installed. If not, the test is skipped.
        Verifies that Predictor sets `transformer` to None,
        uses DummyExecutorWithAttr as `executor`, and that executor.model is set correctly.
        Checks that `predict(7)` returns the expected value 51.
        """
        if not _JOBLIB_AVAILABLE:
            self.skipTest("joblib not installed")

        comp_dir = self._make_component_folder(
            use_joblib=True,
            include_transformer=False,
            executor_has_set_model=False,
            uppercase_names=False
        )

        p = Predictor(comp_dir)

        self.assertIsNone(p.transformer)

        self.assertIsInstance(p.predictor, DummyPredictorWithAttr)
        self.assertIs(p.predictor.model, p.model)
        self.assertEqual(p.predict(7), 51)

    def test_case_insensitive_filenames(self):
        """
        Test that Predictor handles component filenames case-insensitively.

        Creates pickle files with uppercase names and verifies that Predictor
        still loads each component and produces the correct prediction
        `predict(3) == (3 + 10 - 1) * 3`.
        """
        comp_dir = self._make_component_folder(
            use_joblib=False,
            include_transformer=True,
            executor_has_set_model=True,
            uppercase_names=True
        )

        p = Predictor(comp_dir)

        self.assertIsInstance(p.data_processor, DummyDataProcessorProcessOnly)
        self.assertIsInstance(p.transformer, DummyTransformer)
        self.assertIsInstance(p.model, DummyModel)
        self.assertIsInstance(p.predictor, DummyPredictor)

        self.assertEqual(p.predict(3), (3 + 10 - 1) * 3)

    def test_loading_from_zip_archive(self):
        """
        Test that Predictor can load components from a ZIP archive.

        Creates a ZIP containing all component files and then passes the ZIP path
        to Predictor. Verifies that `predict(4)` yields `(4 + 10 - 1) * 3`.
        """
        comp_dir = self._make_component_folder(
            use_joblib=False,
            include_transformer=True,
            executor_has_set_model=True
        )

        zip_path = self.tempdir / "archive.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for child in comp_dir.iterdir():
                zf.write(child, arcname=child.name)

        p = Predictor(zip_path)

        self.assertEqual(p.predict(4), (4 + 10 - 1) * 3)

    def test_missing_data_loader(self):
        """
        Test that missing data loader component raises ComponentLoadError.

        Creates a folder containing only preprocessor, model, and executor files.
        Verifies that Predictor initialization raises ComponentLoadError containing
        the message "Missing data loader".
        """
        folder = self.tempdir / "missing_loader"
        folder.mkdir()
        # Only preprocessor, model, executor, but no dataprocessor
        _dump_pickle(DummyTransformer(), folder / "preprocessor.pkl")
        _dump_pickle(DummyModel(), folder / "model.pkl")
        _dump_pickle(DummyPredictor(), folder / "predictor.pkl")

        with self.assertRaises(ComponentLoadError) as cm:
            Predictor(folder)
        self.assertIn("Missing data processor", str(cm.exception))

    def test_missing_executor(self):
        """
        Test that missing executor component raises ComponentLoadError.

        Creates a folder with dataprocessor and model, but no predictor.
        Verifies that Predictor initialization raises ComponentLoadError containing
        the message "Missing executor".
        """
        folder = self.tempdir / "missing_executor"
        folder.mkdir()
        _dump_pickle(DummyDataProcessorProcessOnly(),
                     folder / "dataprocessor.pkl")
        _dump_pickle(DummyModel(), folder / "model.pkl")
        # No predictor.pkl present

        with self.assertRaises(ComponentLoadError) as cm:
            Predictor(folder)
        self.assertIn("Missing executor", str(cm.exception))

    def test_data_loader_missing_methods(self):
        """
        Test that a data loader missing `process` or `load` raises ComponentLoadError.

        Creates a folder with a DummyDataLoaderBad for dataprocessor.
        Verifies that Predictor initialization raises ComponentLoadError with
        a message mentioning "missing required method".
        """
        folder = self.tempdir / "bad_loader"
        folder.mkdir()
        _dump_pickle(DummyDataProcessorBad(), folder / "dataprocessor.pkl")
        _dump_pickle(DummyTransformer(), folder / "preprocessor.pkl")
        _dump_pickle(DummyModel(), folder / "model.pkl")
        _dump_pickle(DummyPredictor(), folder / "predictor.pkl")

        with self.assertRaises(ComponentLoadError) as cm:
            Predictor(folder)
        # The message should mention “missing required method”
        self.assertIn("missing required method", str(cm.exception).lower())

    def test_executor_missing_predict_method(self):
        """
        Test that an executor missing `predict` raises ComponentLoadError.

        Creates a folder with DummyExecutorBad for predictor.
        Verifies that Predictor initialization raises ComponentLoadError containing
        the message "missing required method 'predict()'".
        """
        folder = self.tempdir / "bad_executor"
        folder.mkdir()
        _dump_pickle(DummyDataProcessorProcessOnly(),
                     folder / "dataprocessor.pkl")
        _dump_pickle(DummyTransformer(), folder / "preprocessor.pkl")
        _dump_pickle(DummyModel(), folder / "model.pkl")
        _dump_pickle(DummyExecutorBad(), folder /
                     "predictor.pkl")  # no .predict()

        with self.assertRaises(ComponentLoadError) as cm:
            Predictor(folder)
        self.assertIn("missing required method 'predict()'", str(cm.exception))

    def test_transformer_raises_exception(self):
        """
        Test that a transformer raising an exception during `predict()` triggers ComponentLoadError.

        Creates a folder with DummyTransformerBad for preprocessor.
        Verifies that p.predict(...) raises ComponentLoadError containing
        the message "Transformer failed".
        """
        folder = self.tempdir / "transformer_fails"
        folder.mkdir()
        _dump_pickle(DummyDataProcessorProcessOnly(),
                     folder / "dataprocessor.pkl")
        _dump_pickle(DummyTransformerBad(), folder / "preprocessor.pkl")
        _dump_pickle(DummyModel(), folder / "model.pkl")
        _dump_pickle(DummyPredictor(), folder / "predictor.pkl")

        p = Predictor(folder)
        with self.assertRaises(ComponentLoadError) as cm:
            p.predict(1)
        self.assertIn("Transformer failed", str(cm.exception))

    class FailDataLoader:
        """
        Nested class that simulates a data loader whose `process` raises an exception.

        Used to simulate a failing data loader during Predictor prediction.
        """

        def process(self, raw):
            """
            Attempt to process data but always raise RuntimeError.

            Args:
                raw (any): The raw input data.

            Raises:
                RuntimeError: Always raised to simulate loader failure.
            """
            raise RuntimeError("loader failure")

        def load(self, raw):
            """
            Delegate to `process(raw)`.

            Args:
                raw (any): The raw input data.

            Returns:
                Any: Never returns, always raises.
            """
            return self.process(raw)

    def test_data_loader_raises_exception(self):
        """
        Test that a data loader raising an exception during `predict()` triggers ComponentLoadError.

        Creates a folder with FailDataLoader for dataprocessor.
        Verifies that p.predict(...) raises ComponentLoadError containing
        the message "Data loader failed".
        """
        folder = self.tempdir / "loader_fails"
        folder.mkdir()
        _dump_pickle(self.FailDataLoader(), folder / "dataprocessor.pkl")
        _dump_pickle(DummyTransformer(), folder / "preprocessor.pkl")
        _dump_pickle(DummyModel(), folder / "model.pkl")
        _dump_pickle(DummyPredictor(), folder / "predictor.pkl")

        p = Predictor(folder)
        with self.assertRaises(ComponentLoadError) as cm:
            p.predict(2)
        self.assertIn("Data loader failed", str(cm.exception))

    def test_executor_predict_raises_exception(self):
        """
        Test that an executor raising an exception during `predict()` triggers ComponentLoadError.

        Creates a folder with DummyExecutorFailPredict for predictor.
        Verifies that p.predict(...) raises ComponentLoadError containing
        the message "Executor.predict(...) failed".
        """
        folder = self.tempdir / "executor_fails"
        folder.mkdir()
        _dump_pickle(DummyDataProcessorProcessOnly(),
                     folder / "dataprocessor.pkl")
        _dump_pickle(DummyTransformer(), folder / "preprocessor.pkl")
        _dump_pickle(DummyModel(), folder / "model.pkl")
        _dump_pickle(DummyExecutorFailPredict(), folder / "predictor.pkl")

        p = Predictor(folder)
        # data_loader.process(3)=13 → transformer.transform(13)=12 → executor.predict(12) raises
        with self.assertRaises(ComponentLoadError) as cm:
            p.predict(3)
        self.assertIn("Executor.predict(...) failed", str(cm.exception))

    @unittest.skipUnless(_KERAS_AVAILABLE, "need Keras installed for SavedModel tests")
    def test_load_keras_savedmodel_directory(self):
        """
        Test loading a Keras SavedModel directory when Keras is installed.

        Creates a folder with a subdirectory named "model" (simulating a SavedModel),
        along with dataprocessor, preprocessor, and executor pickle files.
        Expects Predictor initialization to raise ComponentLoadError containing
        the message "Failed to load Keras SavedModel".
        """
        folder = self.tempdir / "keras_dir"
        folder.mkdir()
        _dump_pickle(DummyDataProcessorProcessOnly(),
                     folder / "dataprocessor.pkl")
        _dump_pickle(DummyTransformer(), folder / "preprocessor.pkl")
        (folder / "model").mkdir()  # a directory named "model"
        _dump_pickle(DummyPredictor(), folder / "predictor.pkl")

        with self.assertRaises(ComponentLoadError) as cm:
            Predictor(folder)
        self.assertIn("Failed to load Keras SavedModel", str(cm.exception))

    @unittest.skipIf(_KERAS_AVAILABLE, "skip .h5 test if Keras is installed")
    def test_h5_model_without_keras_installed(self):
        """
        Test loading a .h5 model file when Keras is not installed.

        Creates a folder with an empty "model.h5" file and dataprocessor, preprocessor,
        and executor pickle files. Expects Predictor initialization to raise ComponentLoadError
        containing the message "Keras is not installed; cannot load '.h5'".
        """
        folder = self.tempdir / "h5_test"
        folder.mkdir()
        _dump_pickle(DummyDataProcessorProcessOnly(),
                     folder / "dataprocessor.pkl")
        _dump_pickle(DummyTransformer(), folder / "preprocessor.pkl")
        # Create an empty model.h5
        open(folder / "model.h5", "wb").close()
        _dump_pickle(DummyPredictor(), folder / "predictor.pkl")

        with self.assertRaises(ComponentLoadError) as cm:
            Predictor(folder)
        self.assertIn("Keras is not installed; cannot load '.h5'",
                      str(cm.exception))

    def test_no_model_artifact_found(self):
        """
        Test that missing model artifact raises ComponentLoadError.

        Creates a folder with dataprocessor, preprocessor, and executor, but no model file
        (.pkl, .joblib, .h5, or SavedModel dir). Expects Predictor initialization to raise
        ComponentLoadError containing the message "No model artifact found under".
        """
        folder = self.tempdir / "no_model"
        folder.mkdir()
        _dump_pickle(DummyDataProcessorProcessOnly(),
                     folder / "dataprocessor.pkl")
        _dump_pickle(DummyTransformer(), folder / "preprocessor.pkl")
        _dump_pickle(DummyPredictor(), folder / "predictor.pkl")
        # No model.* files or directories

        with self.assertRaises(ComponentLoadError) as cm:
            Predictor(folder)
        self.assertIn("No model artifact found under", str(cm.exception))

    def test_invalid_zip_file(self):
        """
        Test that passing an invalid ZIP file path raises ComponentLoadError.

        Creates a file `not_a_real_zip.zip` with non-ZIP content. Expects
        Predictor initialization to raise ComponentLoadError containing the message
        "neither a directory nor a valid ZIP".
        """
        fake_zip = self.tempdir / "not_a_real_zip.zip"
        fake_zip.write_text("not actually a zip")

        with self.assertRaises(ComponentLoadError) as cm:
            Predictor(fake_zip)
        self.assertIn("neither a directory nor a valid ZIP", str(cm.exception))

    def test_missing_path_entirely(self):
        """
        Test that passing a nonexistent path raises ComponentLoadError.

        Expects Predictor initialization to raise ComponentLoadError containing the message
        "neither a directory nor a valid ZIP".
        """
        nonexistent = self.tempdir / "does_not_exist"
        with self.assertRaises(ComponentLoadError) as cm:
            Predictor(nonexistent)
        self.assertIn("neither a directory nor a valid ZIP", str(cm.exception))

    def test_update_call(self):
        """
        Test that Predictor.update(...) returns a dict containing the correct components and a timestamp,
        and that predictor.update() is called with the scaled features, labels, and model.
        """

        # Create component folder with the new dummy classes
        folder = self.tempdir / "update_components"
        folder.mkdir()

        _dump_pickle(DummyDataProcessorProcessOnly(),
                     folder / "dataprocessor.pkl")
        _dump_pickle(DummyTransformer(), folder / "preprocessor.pkl")
        _dump_pickle(DummyModel(), folder / "model.pkl")
        _dump_pickle(DummyPredictor(), folder / "predictor.pkl")

        # Initialize Predictor and call update(...)
        p = Predictor(folder)
        raw_X = [1, 2, 3]
        raw_y = [4, 5, 6]
        ret = p.update(raw_X, raw_y)

        # Check returned dict has expected keys
        self.assertIn("dataprocessor", ret)
        self.assertIn("transformer", ret)
        self.assertIn("model", ret)
        self.assertIn("predictor", ret)
        self.assertIn("updated_at", ret)

        # Verify instances are correct
        self.assertIsInstance(ret["dataprocessor"],
                              DummyDataProcessorProcessOnly)
        self.assertIsInstance(ret["transformer"], DummyTransformer)
        self.assertIsInstance(ret["model"], DummyModel)
        self.assertIsInstance(ret["predictor"], DummyPredictor)


if __name__ == "__main__":
    unittest.main()
