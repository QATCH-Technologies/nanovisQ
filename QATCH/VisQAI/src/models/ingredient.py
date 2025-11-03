"""
ingredient.py

This module defines the abstract base class `Ingredient` and its concrete subclasses
for modeling different types of ingredients used in formulations. Each ingredient
has an encoded identifier (`enc_id`), a human-readable `name`, and an optional
database primary key (`id`). Subclasses include:

- `Protein`: Represents a protein with optional molecular weight and isoelectric point attributes.
- `Buffer`: Represents a buffer with an optional pH value.
- `Stabilizer`: Represents a generic stabilizer ingredient.
- `Surfactant`: Represents a surfactant ingredient.
- `Salt`: Represents a salt ingredient.

Each class includes validation logic for its attributes, conversion to dictionary
representations, and comparison methods to support sorting and equality checks.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-07-23

Version:
    1.6
"""

from enum import StrEnum, unique
from abc import ABC
from typing import Any, Dict, Type, TypeVar, Union, Optional, Tuple
TAG = "[Ingredient]"
T = TypeVar("T", bound="Ingredient")


class Ingredient(ABC):
    """Abstract base class representing a generic ingredient with identification and metadata.

    Attributes:
        _id (Optional[int]): Database-assigned unique identifier for the ingredient.
        _enc_id (int): Encoded identifier used for developer or user distinction.
        _name (str): Name of the ingredient.
        _is_user (bool): Flag indicating whether the ingredient was created by a user.
    """

    def __init__(
        self,
        enc_id: int = -1,
        name: str = None,
        id: Optional[int] = None,
    ) -> None:
        """Initialize an Ingredient instance with encoded ID, name, and optional database ID.

        Args:
            enc_id (int): Developer- or user-assigned encoded identifier. Must be an integer.
            name (str): Human-readable name of the ingredient. Must be a non-empty string.
            id (Optional[int]): Optional database primary key. If provided, must be an integer.

        Raises:
            TypeError: If `id` is not an integer or None, if `enc_id` is not an integer,
                or if `name` is not a string.
            ValueError: If `name` is an empty or whitespace-only string.
        """
        if id is not None and not isinstance(id, int):
            raise TypeError("`id` must be an integer or None")
        self._id: Optional[int] = id
        self._enc_id: int = self._validate_int(enc_id, "enc_id")
        if not isinstance(name, str):
            raise TypeError("`name` must be a string")
        if not name.strip():
            raise ValueError("`name` cannot be empty")
        self._name: str = name.strip()
        self._is_user: bool = True

    @staticmethod
    def _validate_int(value: Any, field: str) -> int:
        """Ensure that a given value is an integer.

        Args:
            value (Any): The value to validate.
            field (str): The name of the field being validated (used in error messages).

        Returns:
            int: The validated integer value.

        Raises:
            TypeError: If `value` is not an integer.
        """
        if not isinstance(value, int):
            raise TypeError(f"{field!r} must be an integer")
        return value

    @staticmethod
    def _validate_number(value: Any, field: str, min_val: float = 0.0) -> float:
        """Ensure that a given value is either None or a numeric type above a minimum threshold.

        Args:
            value (Any): The value to validate, which may be None or numeric.
            field (str): The name of the field being validated (used in error messages).
            min_val (float, optional): Minimum allowable value (inclusive). Defaults to 0.0.

        Returns:
            float: The validated numeric value, or None if `value` is None.

        Raises:
            TypeError: If `value` is not None and not an int or float.
            ValueError: If `value` is not None and is less than `min_val`.
        """
        if value is not None and not isinstance(value, (int, float)):
            raise TypeError(f"{field!r} must be a number or None")
        if value is not None and value < min_val:
            raise ValueError(f"{field!r} must be >= {min_val} or None")
        return value

    @property
    def id(self) -> Optional[int]:
        """Get the database ID of the ingredient.

        Returns:
            Optional[int]: The database primary key, or None if not assigned.
        """
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        """Set the database ID of the ingredient.

        Args:
            value (int): The new database primary key.

        Raises:
            TypeError: If `value` is not an integer.
        """
        self._id = self._validate_int(value, "id")

    @property
    def enc_id(self) -> int:
        """Get the encoded identifier of the ingredient.

        Returns:
            int: The encoded identifier.
        """
        return self._enc_id

    @property
    def type(self) -> str:
        """Get the concrete subclass name of the ingredient.

        Returns:
            str: The class name (e.g., "Protein", "Buffer").
        """
        return str(self.__class__.__name__)

    @enc_id.setter
    def enc_id(self, value: int) -> None:
        """Set the encoded identifier of the ingredient.

        Args:
            value (int): The new encoded identifier.

        Raises:
            TypeError: If `value` is not an integer.
        """
        self._enc_id = self._validate_int(value, "enc_id")

    @property
    def name(self) -> str:
        """Get the name of the ingredient.

        Returns:
            str: The ingredient's name.
        """
        return self._name

    @name.setter
    def name(self, name: Any) -> None:
        """Set the name of the ingredient.

        Args:
            name (Any): The new name to assign. Must be a non-empty string.

        Raises:
            TypeError: If `name` is not a string.
            ValueError: If `name` is an empty or whitespace-only string.
        """
        if not isinstance(name, str):
            raise TypeError("`name` must be a string")
        if not name.strip():
            raise ValueError("`name` cannot be empty")
        self._name = name.strip()

    @property
    def is_user(self) -> bool:
        """Indicate whether the ingredient was created by a user.

        Returns:
            bool: True if user-created, False otherwise.
        """
        return self._is_user

    @is_user.setter
    def is_user(self, flag: bool) -> None:
        """Set the flag indicating whether the ingredient is user-created.

        Args:
            flag (bool): True if user-created, False otherwise.
        """
        self._is_user = flag

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Ingredient to a dictionary representation.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "enc_id" (int): The encoded identifier.
                - "name" (str): The ingredient name.
                - "type" (str): The concrete subclass name.
                - "user?" (bool): Whether the ingredient is user-created.
        """
        return {
            # "id": self._id,
            "enc_id": self._enc_id,
            "name": self._name,
            "type": self.type,
            "user?": self.is_user,
        }

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create an Ingredient instance from a dictionary.

        Args:
            data (Dict[str, Any]): A dictionary containing keys:
                - "enc_id" (int): The encoded identifier.
                - "name" (str): The ingredient name.
                - "id" (Optional[int]): Optional database primary key.

        Returns:
            T: An instance of the concrete Ingredient subclass.

        Raises:
            KeyError: If required keys ("enc_id" or "name") are missing from `data`.
        """
        return cls(
            enc_id=data["enc_id"],
            name=data["name"],
            id=data.get("id"),
        )

    def __repr__(self) -> str:
        """Provide an unambiguous string representation for debugging.

        Returns:
            str: A string in the format
                "ClassName(id=<id>, enc_id=<enc_id>, name='<name>', user?=<is_user>)".
        """
        return (
            f"{self.__class__.__name__}"
            f"(id={self._id!r}, enc_id={self._enc_id!r}, name={self._name!r}, user?={self._is_user})"
        )

    def __eq__(self, other: Any) -> bool:
        """Check equality between two Ingredient instances.

        Two ingredients are equal if they share the same concrete type, database ID,
        encoded ID, and name.

        Args:
            other (Any): The object to compare.

        Returns:
            bool: True if `other` is an Ingredient of the same type with identical IDs and name; False otherwise.
        """
        if not isinstance(other, Ingredient):
            return NotImplemented
        return (
            type(self) is type(other)
            and self._id == other._id
            and self._enc_id == other._enc_id
            and self._name == other._name
        )

    def __lt__(self, other: Any) -> bool:
        """Define a strict ordering between two Ingredient instances of the same subclass.

        If both have assigned database IDs, compare by `id`. Otherwise, compare by `enc_id`.

        Args:
            other (Any): The object to compare.

        Returns:
            bool: True if this instance precedes `other`; False if it follows.

        Raises:
            NotImplementedError: If `other` is not an Ingredient or is a different subclass.
        """
        if not isinstance(other, Ingredient) or type(self) is not type(other):
            return NotImplemented
        # Compare by database ID if both are present, else by encoded ID
        if self._id is not None and other._id is not None:
            return self._id < other._id
        return self._enc_id < other._enc_id


@unique
class ProteinClass(StrEnum):
    NONE = "None"
    ADC = "ADC"
    FC_FUSION = "FC-Fusion"
    MAB_IGG1 = "mAb_IgG1"
    MAB_IGG4 = "mAb_IgG4"
    POLYCLONAL = "Polyclonal"
    BISPECIFIC = "Bispecific"
    TRISPECIFIC = "Trispecific"
    OTHER = "Other"

    @classmethod
    def all(cls) -> Tuple["ProteinClass", ...]:
        return tuple(cls)

    @classmethod
    def all_strings(cls) -> Tuple[str, ...]:
        return tuple(m.value for m in cls)

    @classmethod
    def from_value(cls, value: str) -> "ProteinClass":
        return cls(value)

    @property
    def kP(self) -> float:
        kP_mapping = {
            self.POLYCLONAL: 3.0,
            self.MAB_IGG1: 3.0,
            self.MAB_IGG4: 3.5,
            self.FC_FUSION: 4.5,
            self.BISPECIFIC: 5.0,
            self.TRISPECIFIC: 5.5,
            self.NONE: 2.0,
            self.ADC: 2.5,
            self.OTHER: 2.0,
        }
        return kP_mapping[self]

    @classmethod
    def get_kP_mapping(cls) -> Dict[str, float]:
        return {pc.value: pc.kP for pc in cls}


class Protein(Ingredient):
    """Represents a protein ingredient with optional molecular weight and isoelectric point information.

    Attributes:
        _molecular_weight (Optional[float]): Molecular weight of the protein in suitable units.
        _pI_mean (Optional[float]): Mean isoelectric point of the protein.
        _pI_range (Optional[float]): Range of isoelectric point values for the protein.
    """

    def __init__(
        self,
        enc_id: int,
        name: str,
        molecular_weight: Union[int, float] = None,
        pI_mean: Union[int, float] = None,
        pI_range: Union[int, float] = None,
        class_type: ProteinClass = None,
        id: Optional[int] = None,
    ) -> None:
        """Initialize a Protein instance with encoded ID, name, and optional properties.

        Args:
            enc_id (int): Encoded identifier (must be an integer).
            name (str): Name of the protein (must be a non-empty string).
            molecular_weight (Union[int, float], optional): Molecular weight of the protein. Must be ≥ 0 if provided.
            pI_mean (Union[int, float], optional): Mean isoelectric point of the protein. Must be ≥ 0 if provided.
            pI_range (Union[int, float], optional): Range of isoelectric point values. Must be ≥ 0 if provided.
            id (Optional[int], optional): Database primary key (if already created). Must be an integer or None.

        Raises:
            TypeError: If `enc_id` is not an integer, or if `name` is not a string,
                or if any of `molecular_weight`, `pI_mean`, or `pI_range` are not numeric when provided,
                or if `id` is not an integer or None.
            ValueError: If `name` is empty or whitespace-only,
                or if any numeric property is negative.
        """
        super().__init__(enc_id=enc_id, name=name, id=id)
        self._class_type: ProteinClass = class_type
        self._molecular_weight: float = self._validate_number(
            molecular_weight, "molecular_weight"
        )
        self._pI_mean: float = self._validate_number(pI_mean, "pI_mean")
        self._pI_range: float = self._validate_number(pI_range, "pI_range")

    @property
    def class_type(self) -> Union[ProteinClass, None]:
        """Get the class type of the protein.

        Returns:
            Union[ProteinClass, None]: The ProteinClass type enum if set, otherwise None.
        """
        return self._class_type

    @class_type.setter
    def class_type(self, class_type: str) -> None:
        """Set the class type of the protein.

        Args:
            class_type (str): New class type. Must be a member of ProteinClass enum.

        Raises:
            ValueError: If `class_type` is not a member of ProteinClass enum.
        """
        try:
            ct = ProteinClass.from_value(class_type)
            self._class_type = ct
        except ValueError:
            raise ValueError(
                f"`{class_type}` is not a supported class of protein. Supported classes are: {ProteinClass.all_strings()}.")

    @property
    def molecular_weight(self) -> Union[float, None]:
        """Get the molecular weight of the protein.

        Returns:
            Union[float, None]: The molecular weight if set, otherwise None.
        """
        return self._molecular_weight

    @molecular_weight.setter
    def molecular_weight(self, mw: Any) -> None:
        """Set the molecular weight of the protein.

        Args:
            mw (Any): New molecular weight. Must be a number ≥ 0 or None.

        Raises:
            TypeError: If `mw` is not numeric or None.
            ValueError: If `mw` is negative.
        """
        self._molecular_weight = self._validate_number(mw, "molecular_weight")

    @property
    def pI_mean(self) -> Union[float, None]:
        """Get the mean isoelectric point of the protein.

        Returns:
            Union[float, None]: The mean isoelectric point if set, otherwise None.
        """
        return self._pI_mean

    @pI_mean.setter
    def pI_mean(self, p: Any) -> None:
        """Set the mean isoelectric point of the protein.

        Args:
            p (Any): New mean pI value. Must be a number ≥ 0 or None.

        Raises:
            TypeError: If `p` is not numeric or None.
            ValueError: If `p` is negative.
        """
        self._pI_mean = self._validate_number(p, "pI_mean")

    @property
    def pI_range(self) -> Union[float, None]:
        """Get the isoelectric point range of the protein.

        Returns:
            Union[float, None]: The pI range if set, otherwise None.
        """
        return self._pI_range

    @pI_range.setter
    def pI_range(self, r: Any) -> None:
        """Set the isoelectric point range of the protein.

        Args:
            r (Any): New pI range. Must be a number ≥ 0 or None.

        Raises:
            TypeError: If `r` is not numeric or None.
            ValueError: If `r` is negative.
        """
        self._pI_range = self._validate_number(r, "pI_range")

    def __eq__(self, other: Any) -> bool:
        """Check equality between two Protein instances.

        Two Proteins are equal if they share the same concrete attributes:
        base (see Ingredient), molecular_weight, pI_mean, and pI_range

        Args:
            other (Any): The object to compare.

        Returns:
            bool: True if `other` is a Protein of the same type with identical attributes; False otherwise.
        """
        super().__eq__(other)

        if not isinstance(other, Protein):
            return NotImplemented
        return (
            type(self) is type(other)
            and self.name == other.name
            and self.molecular_weight == other.molecular_weight
            and self.pI_mean == other.pI_mean
            and self.pI_range == other.pI_range
            and self.class_type == other.class_type
        )


class Buffer(Ingredient):
    """Represents a buffer ingredient with an optional pH value.

    Attributes:
        _pH (Optional[float]): pH of the buffer, if provided.
    """

    def __init__(
        self,
        enc_id: int,
        name: str,
        pH: Union[int, float] = None,
        id: Optional[int] = None,
    ) -> None:
        """Initialize a Buffer instance with encoded ID, name, and optional pH.

        Args:
            enc_id (int): Encoded identifier (must be an integer).
            name (str): Name of the buffer (must be a non-empty string).
            pH (Union[int, float], optional): pH value. Must be between 0 and 14, or None.
            id (Optional[int], optional): Database primary key (if already created). Must be an integer or None.

        Raises:
            TypeError: If `enc_id` is not an integer, or if `name` is not a string,
                or if `pH` is not numeric or None, or if `id` is not an integer or None.
            ValueError: If `name` is empty or whitespace-only,
                or if `pH` is outside the range 0 to 14 when provided.
        """
        super().__init__(enc_id=enc_id, name=name, id=id)
        self.pH = pH

    @property
    def pH(self) -> Union[float, None]:
        """Get the pH of the buffer.

        Returns:
            Union[float, None]: The pH value if set, otherwise None.
        """
        return self._pH

    @pH.setter
    def pH(self, value: Any) -> None:
        """Set the pH of the buffer.

        Args:
            value (Any): New pH value. Must be a number between 0 and 14, or None.

        Raises:
            TypeError: If `value` is not numeric or None.
            ValueError: If `value` is not between 0 and 14 when provided.
        """
        if value is not None and not isinstance(value, (int, float)):
            raise TypeError("pH must be a number or None")
        if value is not None and not (0.0 <= value <= 14.0):
            raise ValueError("pH must be between 0 and 14 or None")
        self._pH = value

    def __eq__(self, other: Any) -> bool:
        """Check equality between two Buffer instances.

        Two Buffers are equal if they share the same concrete attributes:
        base (see Ingredient), pH

        Args:
            other (Any): The object to compare.

        Returns:
            bool: True if `other` is a Protein of the same type with identical attributes; False otherwise.
        """
        super().__eq__(other)

        if not isinstance(other, Protein):
            return NotImplemented
        return (
            type(self) is type(other)
            and self.pH == other.pH
        )


class Stabilizer(Ingredient):
    """Represents a stabilizer ingredient without additional properties beyond Ingredient."""
    pass


class Surfactant(Ingredient):
    """Represents a surfactant ingredient without additional properties beyond Ingredient."""
    pass


class Salt(Ingredient):
    """Represents a salt ingredient without additional properties beyond Ingredient."""
    pass


class Excipient(Ingredient):
    """Represents a Excipient ingredient without additional properties beyond Ingredient."""
    pass
