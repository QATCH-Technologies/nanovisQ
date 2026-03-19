"""
Design Space and Constraint Management for Bioformulations.

This module provides the structural definition of the formulation search space.
It acts as a configuration engine that translates high-level laboratory
requirements-such as allowed ingredient types and concentration ranges-into
the mathematical structures required by optimization and sampling algorithms.

Example:
    >>> constraints = Constraints(db=my_database)
    >>> constraints.add_range("Protein_conc", 10.0, 250.0)
    >>> constraints.add_choices("Buffer_type", [citrate, phosphate])
    >>> bounds, encoding = constraints.build()

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    2.1
"""

import copy
import math
from typing import Any, Dict, List, Tuple, Type

try:
    TAG = "[Constraints (HEADLESS)]"
    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database
    from src.models.ingredient import (
        Buffer,
        Excipient,
        Ingredient,
        Protein,
        Salt,
        Stabilizer,
        Surfactant,
    )
    from src.utils.list_utils import ListUtils

    class Log:
        """Fallback logger implementation when QATCH logger is not available."""

        @staticmethod
        def w(msg: str) -> None:
            """Log a warning message.

            Args:
                msg: The warning message to log.
            """
            print(msg)

        @staticmethod
        def e(msg: str) -> None:
            """Log an error message.

            Args:
                msg: The error message to log.
            """
            print(msg)

        @staticmethod
        def i(msg: str) -> None:
            """
            Log an informational message.

            Parameters:
                msg (str): The message to log.
            """
            print(msg)

        @staticmethod
        def d(msg: str) -> None:
            """Log a debug message.

            Args:
                msg: The debug message to log.
            """
            print(msg)

except (ModuleNotFoundError, ImportError):
    TAG = "[Constraints]"

    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.models.ingredient import (
        Buffer,
        Excipient,
        Ingredient,
        Protein,
        Salt,
        Stabilizer,
        Surfactant,
    )
    from QATCH.VisQAI.src.utils.list_utils import ListUtils
    from QATCH.common.logger import Logger as Log


class Constraints:
    """Manages search space boundaries and categorical choices for formulations.

    This class serves as a configuration layer that defines the feasible region
    for optimization and sampling. It maintains a registry of numeric ranges
    (e.g., concentration limits) and categorical pools (e.g., allowed buffer types).
    The `build` method translates these high-level constraints into a
    standardized format used by the Differential Evolution and Active Learning
    engines.

    Attributes:
        _NUMERIC (List[str]): Registry of allowed numerical feature names.
        _CATEGORICAL (List[str]): Registry of allowed categorical feature names.
        _FEATURE_CLASS (Dict[str, Type[Ingredient]]): Mapping of categorical
            features to their corresponding Ingredient subclasses.
        _DEFAULT_RANGES (Dict[str, Tuple[int, int]]): Fallback boundaries
            for numeric features if no range is explicitly provided.
    """

    _NUMERIC: List[str] = [
        "Protein_conc",
        "Temperature",
        "Buffer_conc",
        "Salt_conc",
        "Stabilizer_conc",
        "Surfactant_conc",
        "Excipient_conc",
    ]
    _CATEGORICAL: List[str] = [
        "Protein_type",
        "Buffer_type",
        "Salt_type",
        "Stabilizer_type",
        "Surfactant_type",
        "Excipient_type",
    ]
    _FEATURE_CLASS: Dict[str, Type[Ingredient]] = {
        "Protein_type": Protein,
        "Buffer_type": Buffer,
        "Salt_type": Salt,
        "Stabilizer_type": Stabilizer,
        "Surfactant_type": Surfactant,
        "Excipient_type": Excipient,
    }
    _DEFAULT_RANGES = {
        "Protein_conc": (0, 600),
        "Temperature": (15, 40),
        "Buffer_conc": (0, 50),
        "Salt_conc": (0, 150),
        "Stabilizer_conc": (0, 0.5),
        "Surfactant_conc": (0, 0.3),
        "Excipient_conc": (0, 600),
    }

    def __init__(self, db: Database):
        """Initializes Constraints with a database connection.

        Args:
            db: A Database instance used to resolve ingredient identities
                and retrieve historical ranges.
        """
        self._db = db
        self._ingredient_ctrl = IngredientController(db=self._db)
        self._ranges: Dict[str, Tuple[float, float]] = {}
        self._choices: Dict[str, List[Ingredient]] = {}

    def __deepcopy__(self, memo):
        """Creates a custom deep copy of the constraints.

        This implementation ensures that the database connection and ingredient
        controller are not duplicated, as they are often singleton-like resources
        or contain non-serializable states.

        Args:
            memo: The memoization dictionary for the deepcopy operation.

        Returns:
            Constraints: A new instance with copied ranges and choices,
                but shared database references.
        """
        new_obj = type(self)(self._db)
        for key, value in self.__dict__.items():
            if key in ("_db", "_ingredient_ctrl"):
                new_obj.__dict__[key] = None
            else:
                new_obj.__dict__[key] = copy.deepcopy(value, memo)
        return new_obj

    def add_range(self, feature: str, low: float, high: float) -> None:
        """Sets the lower and upper boundaries for a numeric feature.

        Args:
            feature: The name of the numeric feature (e.g., 'Protein_conc').
            low: The minimum allowed value.
            high: The maximum allowed value.

        Raises:
            ValueError: If the feature is not recognized, or if non-temperature
                values are negative.
        """
        if feature not in self._NUMERIC:
            raise ValueError(
                f"Unknown numeric feature '{feature}'.  Only {self._NUMERIC} are allowed in add_range()."
            )
        if feature != "Temperature" and (low < 0.0 or high < 0.0):
            raise ValueError(f"Negative values are not allowed for numeric feature {feature}")
        self._ranges[feature] = (float(low), float(high))

    def add_choices(self, feature: str, choices: List[Ingredient]) -> None:
        """Sets the pool of allowed ingredients for a categorical feature.

        Args:
            feature: The name of the categorical feature (e.g., 'Salt_type').
            choices: A list of Ingredient instances allowed for this feature.

        Raises:
            ValueError: If the feature is not recognized.
            TypeError: If any choice is not an instance of the Ingredient class.

        Note:
            If an empty list is provided, the `build` method will fallback
            to including all ingredients of the appropriate type from the database.
        """
        if feature not in self._CATEGORICAL:
            raise ValueError(
                f"Unknown categorical feature '{feature}'.  Only {self._CATEGORICAL} are allowed in add_choices()."
            )
        for c in choices:
            if not isinstance(c, Ingredient):
                raise TypeError(
                    f"All choices for '{feature}' must be Ingredient instances; got {c!r} of type {type(c).__name__}"
                )
        if not choices:
            Log.w(
                TAG,
                f"add_choices('{feature}', []) - empty choice list stored. "
                f"build() will fall back to all ingredients of this type.",
            )
        self._choices[feature] = list(choices)

    def set_db(self, db: Database) -> None:
        """Updates the database and refreshes the ingredient controller.

        Args:
            db: The new Database instance to use.

        Raises:
            TypeError: If db is not an instance of the Database class.
        """
        if not isinstance(db, Database):
            raise TypeError("db must be an instance of Database")
        self._db = db
        self._ingredient_ctrl = IngredientController(db=self._db)

    def get_db(self) -> Database:
        """Retrieves the current database instance.

        Returns:
            Database: The active database connection.
        """
        return self._db

    def build(self) -> Tuple[List[Tuple[float, float]], List[Dict[str, Any]]]:
        """Finalizes and encodes the constraints into a mathematical search space.

        This method processes all defined categorical and numeric constraints,
        resolving defaults where necessary. Categorical features are converted
        into index-based bounds (0.0 to N-1), and their names are sorted
        case-insensitively to ensure reproducible encoding across sessions.

        Returns:
            Tuple[List[Tuple[float, float]], List[Dict[str, Any]]]:
                - bounds: A list of (min, max) tuples for the optimizer.
                - encoding: A list of dictionaries describing the metadata
                  for each feature in the search vector.

        Raises:
            ValueError: If a categorical feature has no available choices
        """
        bounds: List[Tuple[float, float]] = []
        encoding: List[Dict[str, Any]] = []

        all_ingredients = self._ingredient_ctrl.get_all_ingredients()
        all_features = self._CATEGORICAL + self._NUMERIC

        for feat in all_features:
            if feat in self._CATEGORICAL:
                chosen = self._choices.get(feat)
                # Treat an explicitly-set empty list the same as "not constrained":
                # add_choices(feature, []) means zero valid choices were found
                # (e.g. "is not" applied to every available ingredient), which should
                # fall back to all ingredients rather than raising ValueError.
                if not chosen:
                    cls = self._FEATURE_CLASS[feat]
                    chosen = [ing for ing in all_ingredients if isinstance(ing, cls)]
                if not chosen:
                    raise ValueError(f"No choices available for '{feat}'.")

                names = ListUtils.unique_case_insensitive_sort([ing.name for ing in chosen])
                encoding.append({"feature": feat, "type": "cat", "choices": names})
                bounds.append((0.0, float(len(names) - 1)))

            elif feat in self._NUMERIC:
                if feat in self._ranges:
                    low, high = self._ranges[feat]
                else:
                    low, high = self._DEFAULT_RANGES[feat]
                if not (math.isfinite(low) and math.isfinite(high)):
                    raise ValueError(f"Bounds for '{feat}' must be finite. Got ({low}, {high}).")
                encoding.append({"feature": feat, "type": "num"})
                bounds.append((low, high))

            else:
                raise ValueError(f"Unknown feature '{feat}' in build()")

        return bounds, encoding
