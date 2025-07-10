from typing import Dict, Any, List, Tuple, Type
import math
import copy

try:
    from src.controller.ingredient_controller import IngredientController
    from src.models.ingredient import Ingredient, Protein, Buffer, Salt, Stabilizer, Surfactant
    from src.db.db import Database
    from src.utils.list_utils import ListUtils
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.models.ingredient import Ingredient, Protein, Buffer, Salt, Stabilizer, Surfactant
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.utils.list_utils import ListUtils


class Constraints:
    _NUMERIC: List[str] = [
        "Protein_conc",
        "Temperature",
        "Buffer_conc",
        "Salt_conc",
        "Stabilizer_conc",
        "Surfactant_conc",
    ]
    _CATEGORICAL: List[str] = [
        "Protein_type",
        "Buffer_type",
        "Salt_type",
        "Stabilizer_type",
        "Surfactant_type",
    ]
    _FEATURE_CLASS: Dict[str, Type[Ingredient]] = {
        "Protein_type": Protein,
        "Buffer_type": Buffer,
        "Salt_type": Salt,
        "Stabilizer_type": Stabilizer,
        "Surfactant_type": Surfactant,
    }
    _DEFAULT_RANGES = {
        "Protein_conc": (0, 1000),
        "Temperature": (24, 25),
        "Buffer_conc": (0, 100),
        "Salt_conc": (0, 200),
        "Stabilizer_conc": (0, 1),
        "Surfactant_conc": (0, 1), }

    def __init__(self, db: Database):
        self._db = db
        self._ingredient_ctrl = IngredientController(db=self._db)
        self._ranges: Dict[str, Tuple[float, float]] = {}
        self._choices: Dict[str, List[Ingredient]] = {}

    def __deepcopy__(self, memo):
        """Custom deep copy to avoid copying the database and ingredient controller."""
        new_obj = type(self)(self._db)
        for key, value in self.__dict__.items():
            if key in ('_db', '_ingredient_ctrl'):
                new_obj.__dict__[key] = None
            else:
                new_obj.__dict__[key] = copy.deepcopy(value, memo)
        return new_obj

    def add_range(self, feature: str, low: float, high: float) -> None:
        if feature not in self._NUMERIC:
            raise ValueError(
                f"Unknown numeric feature '{feature}'.  Only {self._NUMERIC} are allowed in add_range()."
            )
        if feature != 'Temperature' and (low < 0.0 or high < 0.0):
            raise ValueError(
                f"Negative values are not allowed for numeric feature {feature}"
            )
        self._ranges[feature] = (float(low), float(high))

    def add_choices(self, feature: str, choices: List[Ingredient]) -> None:
        if feature not in self._CATEGORICAL:
            raise ValueError(
                f"Unknown categorical feature '{feature}'.  Only {self._CATEGORICAL} are allowed in add_choices()."
            )
        for c in choices:
            if not isinstance(c, Ingredient):
                raise TypeError(
                    f"All choices for '{feature}' must be Ingredient instances; got {c!r} of type {type(c).__name__}"
                )
            if not self._ingredient_ctrl.get_by_name(name=c.name, ingredient=c):
                raise ValueError(
                    f"`{c.name}` has not been added to persistent store yet.")
        self._choices[feature] = list(choices)

    def set_db(self, db: Database) -> None:
        """Set the database to use for ingredient retrieval."""
        if not isinstance(db, Database):
            raise TypeError("db must be an instance of Database")
        self._db = db
        self._ingredient_ctrl = IngredientController(db=self._db)

    def get_db(self) -> Database:
        """Get the current database instance."""
        return self._db

    def build(self) -> Tuple[
        List[Tuple[float, float]],
        List[Dict[str, Any]]
    ]:
        bounds: List[Tuple[float, float]] = []
        encoding: List[Dict[str, Any]] = []

        all_ingredients = self._ingredient_ctrl.get_all_ingredients()
        for ing in all_ingredients:
            print(ing.to_dict())
        all_features = self._CATEGORICAL + self._NUMERIC

        for feat in all_features:
            if feat in self._CATEGORICAL:
                chosen = self._choices.get(feat)
                if chosen is None:
                    cls = self._FEATURE_CLASS[feat]
                    chosen = [
                        ing for ing in all_ingredients if isinstance(ing, cls)]
                if not chosen:
                    raise ValueError(f"No choices available for '{feat}'.")

                names = ListUtils.unique_case_insensitive_sort(
                    [ing.name for ing in chosen])
                encoding.append({
                    "feature": feat,
                    "type": "cat",
                    "choices": names
                })
                bounds.append((0.0, float(len(names) - 1)))

            elif feat in self._NUMERIC:
                if feat in self._ranges:
                    low, high = self._ranges[feat]
                else:
                    low, high = self._DEFAULT_RANGES[feat]
                if not (math.isfinite(low) and math.isfinite(high)):
                    raise ValueError(
                        f"Bounds for '{feat}' must be finite. Got ({low}, {high})."
                    )
                encoding.append({"feature": feat, "type": "num"})
                bounds.append((low, high))

            else:
                raise ValueError(f"Unknown feature '{feat}' in build()")

        return bounds, encoding
