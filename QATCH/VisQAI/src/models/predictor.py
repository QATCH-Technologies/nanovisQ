"""
predictor.py

This module provides a generic `Preidctor` class that dynamically loads
and wires together components for data processing, preprocessing, modeling, and prediction.
Components can be serialized with pickle, joblib, or saved as Keras models (.h5 or SavedModel directories).
The module supports loading from either a directory or a ZIP archive and performs basic validation to
ensure required methods exist on loaded components.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-06-03

Version:
    1.1
"""

import pickle
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Union

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

try:
    from keras.models import load_model as _keras_load_model
    _KERAS_AVAILABLE = True
except ImportError:
    _KERAS_AVAILABLE = False


class ComponentLoadError(Exception):
    """Exception raised when a required pipeline component cannot be loaded or validated."""
    pass


class Predictor:
    """
    Dynamically load and execute a pipeline of components for data processing, transformation,
    and prediction.

    The pipeline consists of four components that must be present in a specified directory or ZIP archive:
        1. dataprocessor: Must implement either `load(data)` or `process(raw)`.
        2. transformer(optional): Must implement `transform(X)`, if present.
        3. model: Can be a pickled sklearn-like estimator(with `predict`) or a Keras model(.h5 or SavedModel directory).
        4. predictor: Must implement `predict(X)`. If it also provides `set_model(model)`, that method will be used to attach
           the model; otherwise, the model is assigned to its `.model` attribute.

    During `predict(raw_input)`, the following steps are performed:
        1. The dataprocessors loads or processes the raw input.
        2. If a transformer is provided, the output of step 1 is transformed.
        3. The predictors's `predict` method is called on the final data.

    Attributes:
        dataprocessor: The loaded data loading component.
        transformer: The loaded transformer component, or None if not provided.
        model: The loaded machine learning model.
        predictor: The loaded predictor component, wired to use `model`.

    Raises:
        ComponentLoadError: If any required component is missing, invalid, or fails during loading or execution.
    """

    def __init__(
        self,
        source: Union[str, Path],
        *,
        data_processor_name: str = "dataprocessor",
        transformer_name: str = "preprocessor",
        model_name: str = "model",
        executor_name: str = "predictor",
    ) -> None:
        """
        Initialize a Predictor that loads components from a directory or ZIP archive.

        This method will:
            1. Extract `source` if it is a ZIP, or treat it as a directory.
            2. Locate and load the data_processor component(pickle or joblib).
            3. Locate and optionally load the transformer component(pickle or joblib).
            4. Locate and load the model artifact(pickle, joblib, .h5, or SavedModel directory).
            5. Locate and load the executor component(pickle or joblib).
            6. Wire the loaded model to the executor via `set_model(...)` or `.model`.

        Args:
            source: Path to a directory or ZIP archive containing serialized components.
            data_processor_name: Basename(without extension) for the data loading component.
            transformer_name: Basename(without extension) for the optional transformer component.
            model_name: Basename(without extension) for the model artifact.
            executor_name: Basename(without extension) for the executor component.

        Raises:
            ComponentLoadError: If any required component is missing, invalid, or fails to load.
        """
        self._tempdir: Optional[tempfile.TemporaryDirectory] = None
        self._base_dir = self._extract_if_archive(Path(source))

        # Load data loader component
        data_processor_path = self._locate_file(
            self._base_dir, data_processor_name, [".pkl", ".joblib"]
        )
        if data_processor_path is None:
            raise ComponentLoadError(
                f"Missing data processor: expected '{data_processor_name}.pkl' or "
                f"'{data_processor_name}.joblib' in '{self._base_dir}'."
            )
        self.data_processor = self._load_binary(
            data_processor_path, required_methods=("process", "load")
        )

        # Load optional transformer component
        transformer_path = self._locate_file(
            self._base_dir, transformer_name, [".pkl", ".joblib"]
        )
        if transformer_path:
            self.transformer = self._load_binary(
                transformer_path, required_methods=(
                    "transform", "fit_transform")
            )
        else:
            self.transformer = None

        # Load model artifact
        self.model = self._load_model(self._base_dir, model_name)

        # Load executor component
        predictor_path = self._locate_file(
            self._base_dir, executor_name, [".pkl", ".joblib"]
        )
        if predictor_path is None:
            raise ComponentLoadError(
                f"Missing executor: expected '{executor_name}.pkl' or "
                f"'{executor_name}.joblib' in '{self._base_dir}'."
            )
        self.predictor = self._load_binary(
            predictor_path, required_methods=("predict",)
        )

        # Attach model to executor
        self._wire_model_to_executor()

    def _extract_if_archive(self, path: Path) -> Path:
        """
        Extract the contents of a ZIP archive if `path` is a ZIP; otherwise, return the directory itself.

        Args:
            path: Path to either a directory or a ZIP archive.

        Returns:
            Path: If `path` is a directory, returns the absolute directory path. If `path` is a valid ZIP file,
                  extracts it into a temporary directory and returns the path to that temporary directory.

        Raises:
            ComponentLoadError: If `path` is neither a directory nor a valid ZIP file, or if extraction fails.
        """
        if path.is_dir():
            return path.resolve()

        if path.is_file() and zipfile.is_zipfile(path):
            try:
                temp_dir = tempfile.TemporaryDirectory()
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(temp_dir.name)
                self._tempdir = temp_dir
                return Path(temp_dir.name).resolve()
            except Exception as e:
                raise ComponentLoadError(
                    f"Failed to extract ZIP '{path}': {e!s}"
                )

        raise ComponentLoadError(
            f"Path '{path}' is neither a directory nor a valid ZIP file."
        )

    def _locate_file(
        self, directory: Path, basename: str, extensions: list[str]
    ) -> Optional[Path]:
        """
        Search `directory` for a file matching `basename + extension` in a case-insensitive manner.

        Args:
            directory: Path to the directory to search.
            basename: Filename without extension(e.g., "dataprocessor").
            extensions: List of file extensions to consider(e.g., [".pkl", ".joblib"]).

        Returns:
            Path or None: The first matching file path if found; otherwise, None.
        """
        for entry in directory.iterdir():
            if not entry.is_file():
                continue
            lower_name = entry.name.lower()
            for ext in extensions:
                if lower_name == f"{basename.lower()}{ext}":
                    return entry
        return None

    def _load_binary(
        self, file_path: Path, required_methods: tuple[str, ...]
    ) -> Any:
        """
        Load a serialized object from a pickle or joblib file, and verify it has the required methods.

        Args:
            file_path: Path to the file ending in .pkl or .joblib.
            required_methods: Tuple of method names that the loaded object must implement.

        Returns:
            Any: The deserialized object.

        Raises:
            ComponentLoadError:
                - If the file extension is unsupported.
                - If deserialization fails.
                - If any of the required methods is missing or not callable.
        """
        ext = file_path.suffix.lower()
        try:
            if ext == ".pkl":
                with open(file_path, "rb") as f:
                    component = pickle.load(f)
            elif ext == ".joblib":
                if not _JOBLIB_AVAILABLE:
                    raise ComponentLoadError(
                        f"joblib is not installed; cannot load '{file_path}'."
                    )
                component = joblib.load(str(file_path))
            else:
                raise ComponentLoadError(
                    f"Unsupported serialization format: '{ext}' for '{file_path}'."
                )
        except Exception as e:
            raise ComponentLoadError(f"Failed to load '{file_path}': {e!s}")

        for method in required_methods:
            if not hasattr(component, method) or not callable(getattr(component, method)):
                raise ComponentLoadError(
                    f"Loaded object from '{file_path}' is missing required method '{method}()'."
                )

        return component

    def _load_model(self, directory: Path, model_basename: str) -> Any:
        """
        Locate and load a model artifact from `directory`. Supports:
            - Pickle(.pkl) or joblib(.joblib) serialized objects with `predict` method.
            - Keras .h5 files(requires Keras).
            - Keras SavedModel directories(requires Keras).

        Args:
            directory: Path to the directory containing model artifacts.
            model_basename: Basename of the model file or directory(e.g., "model").

        Returns:
            Any: The loaded model object.

        Raises:
            ComponentLoadError:
                - If a supported artifact cannot be found.
                - If loading fails.
                - If Keras is not installed when attempting to load a .h5 or SavedModel.
        """
        # Attempt to load pickle or joblib first
        model_file = self._locate_file(
            directory, model_basename, [".pkl", ".joblib"])
        if model_file:
            return self._load_binary(model_file, required_methods=("predict",))

        # Attempt to load Keras .h5
        h5_path = self._locate_file(directory, model_basename, [".h5"])
        if h5_path:
            if not _KERAS_AVAILABLE:
                raise ComponentLoadError(
                    "Keras is not installed; cannot load '.h5' model."
                )
            try:
                return _keras_load_model(str(h5_path))
            except Exception as e:
                raise ComponentLoadError(
                    f"Failed to load Keras model from '{h5_path}': {e!s}"
                )

        # Attempt to load Keras SavedModel directory
        saved_model_dir = directory / model_basename
        if saved_model_dir.is_dir() and _KERAS_AVAILABLE:
            try:
                return _keras_load_model(str(saved_model_dir))
            except Exception as e:
                raise ComponentLoadError(
                    f"Failed to load Keras SavedModel from '{saved_model_dir}': {e!s}"
                )

        raise ComponentLoadError(
            f"No model artifact found under '{directory}'. Expected one of:\n"
            f"  - {model_basename}.pkl  (pickle)\n"
            f"  - {model_basename}.joblib  (joblib)\n"
            f"  - {model_basename}.h5  (Keras .h5)\n"
            f"  - A SavedModel directory named '{model_basename}/' (Keras SavedModel)."
        )

    def _wire_model_to_executor(self) -> None:
        """
        Attach the loaded model to the executor component.

        The method first checks if the executor has a callable `set_model(model)` method.
        If so, it calls `executor.set_model(self.model)`. Otherwise, it attempts to set
        the model via `executor.model = self.model`.

        Raises:
            ComponentLoadError: If attaching the model fails, or if neither approach is supported.
        """
        if hasattr(self.predictor, "set_model") and callable(self.predictor.set_model):
            try:
                self.predictor.set_model(self.model)
            except Exception as e:
                raise ComponentLoadError(
                    f"Executor.set_model(...) failed: {e!s}")
        else:
            try:
                setattr(self.predictor, "model", self.model)
            except Exception:
                raise ComponentLoadError(
                    "Executor does not support `set_model(...)` or setting `.model` attribute."
                )

    def predict(self, raw_input: Any) -> Any:
        """
        Execute the full pipeline on `raw_input` and return the prediction.

        The steps are:
            1. Use `data_loader.process(raw_input)` if available; otherwise, call `data_loader.load(raw_input)`.
            2. If a transformer is present, call `transformer.transform(...)` on the data from step 1.
            3. Call `predictor.predict(...)` on the final data from step 2.

        Args:
            raw_input: The raw input data to be loaded and transformed.

        Returns:
            Any: The output returned by `executor.predict(...)`.

        Raises:
            ComponentLoadError: If any of the component methods(`process`, `load`, `transform`, or `predict`)
                                raises an exception during execution.
        """
        # Data loading/processing
        try:
            if hasattr(self.data_processor, "process"):
                data = self.data_processor.process(raw_input)
            else:
                data = self.data_processor.load(raw_input)
        except Exception as e:
            raise ComponentLoadError(f"Data loader failed: {e!s}")

        # Transformation
        if self.transformer is not None:
            try:
                data = self.transformer.transform(data)
            except Exception as e:
                raise ComponentLoadError(f"Transformer failed: {e!s}")

        # Executor prediction
        try:
            return self.predictor.predict(data)
        except Exception as e:
            raise ComponentLoadError(f"Executor.predict(...) failed: {e!s}")

    def update(
        self,
        raw_X_new: Any,
        raw_y_new: Any,
    ) -> dict:
        """
        Incorporates new data into the existing pipeline by processing, scaling, and updating the model.

        This method takes raw input features and labels, runs them through the data processor
        to obtain features and labels in the expected format, applies the transformer's
        `fit_transform` to scale the features, and then calls the predictor's `update`
        method to perform transfer learning on the underlying model.
        Finally, it returns a dictionary of the updated components and a timestamp.

        Args:
            raw_X_new (Any): New, unprocessed feature data.
            raw_y_new (Any): New, unprocessed label data.

        Returns:
            dict: A dictionary containing the following keys:
                dataprocessor (Any): The data processor instance after processing.
                transformer (Any): The transformer instance after fitting to new features.
                model (Any): The underlying model object (updated in place by the predictor).
                predictor (Any): The predictor instance after performing `update(...)`.
                updated_at (str): ISO-formatted timestamp (YYYY-MM-DDThh:mm:ss.ssssss) 
                    indicating when the update occurred.

        Raises:
            ComponentLoadError: If any of the following conditions occur:
                - `self.data_processor.process(raw_X_new, raw_y_new)` does not return a tuple
                of length 2 (features, labels).
                - The transformer object does not implement a `fit_transform(...)` method.
                - The predictor object does not implement an `update(...)` method.
        """
        processed = self.data_processor.process(raw_X_new, raw_y_new)
        if not (
            isinstance(processed, tuple)
            and len(processed) == 2
        ):
            raise ComponentLoadError(
                "dataprocessor.process(raw_X_new, raw_y_new) must return (features, labels)."
            )
        features, labels = processed

        if not hasattr(self.transformer, "fit_transform"):
            raise ComponentLoadError(
                "The serialized transformer does not implement an fit_transform(...) method.")

        scaled_features = self.transformer.fit_transform(features, raw_y_new)

        if not hasattr(self.predictor, "update"):
            raise ComponentLoadError(
                "The serialized predictor does not implement an update(...) method.")

        self.predictor.update(
            scaled_features,
            labels,
            model=self.model,
        )
        return {"dataprocessor": self.data_processor,
                "transformer":   self.transformer,
                "model":         self.model,
                "predictor":     self.predictor,
                "updated_at":    datetime.now().isoformat()}
