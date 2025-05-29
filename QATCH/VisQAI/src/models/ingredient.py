from abc import ABC
from typing import Any, Dict, Type, TypeVar, Union, Optional
TAG = "[Ingredient]"
T = TypeVar("T", bound="Ingredient")


class Ingredient(ABC):
    def __init__(
        self,
        enc_id: int,
        name: str,
        id: Optional[int] = None,
    ) -> None:
        if id is not None and not isinstance(id, int):
            raise TypeError("`id` must be an integer or None")
        self._id: Optional[int] = id
        self._enc_id: int = self._validate_int(enc_id, "enc_id")
        if not isinstance(name, str):
            raise TypeError("`name` must be a string")
        if not name.strip():
            raise ValueError("`name` cannot be empty")
        self._name: str = name.strip()

    @staticmethod
    def _validate_int(value: Any, field: str) -> int:
        if not isinstance(value, int):
            raise TypeError(f"{field!r} must be an integer")
        return value

    @staticmethod
    def _validate_number(value: Any, field: str, min_val: float = 0.0) -> float:
        if not isinstance(value, (int, float)):
            raise TypeError(f"{field!r} must be a number")
        if value < min_val:
            raise ValueError(f"{field!r} must be >= {min_val}")
        return float(value)

    @property
    def id(self) -> Optional[int]:
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        self._id = self._validate_int(value, "id")

    @property
    def enc_id(self) -> int:
        """The encoding integer you use for modeling."""
        return self._enc_id

    @property
    def type(self) -> str:
        return str(self.__class__.__name__)

    @enc_id.setter
    def enc_id(self, value: int) -> None:
        self._enc_id = self._validate_int(value, "enc_id")

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: Any) -> None:
        if not isinstance(name, str):
            raise TypeError("`name` must be a string")
        if not name.strip():
            raise ValueError("`name` cannot be empty")
        self._name = name.strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "enc_id": self._enc_id,
            "name": self._name,
            "type": self.type,
        }

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        return cls(
            enc_id=data["enc_id"],
            name=data["name"],
            id=data.get("id"),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(id={self._id!r}, enc_id={self._enc_id!r}, name={self._name!r})"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Ingredient):
            return NotImplemented
        return (
            type(self) is type(other)
            and self._id == other._id
            and self._enc_id == other._enc_id
            and self._name == other._name
        )

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Ingredient) or type(self) is not type(other):
            return NotImplemented
        # you can compare either by id or by enc_id
        if self._id is not None and other._id is not None:
            return self._id < other._id
        return self._enc_id < other._enc_id


class Protein(Ingredient):
    def __init__(
        self,
        enc_id: int,
        name: str,
        molecular_weight: Union[int, float],
        pI_mean: Union[int, float],
        pI_range: Union[int, float],
        id: Optional[int] = None,
    ) -> None:
        super().__init__(enc_id=enc_id, name=name, id=id)
        self._molecular_weight = self._validate_number(
            molecular_weight, "molecular_weight")
        self._pI_mean = self._validate_number(pI_mean, "pI_mean")
        self._pI_range = self._validate_number(pI_range, "pI_range")

    @property
    def molecular_weight(self) -> float:
        return self._molecular_weight

    @molecular_weight.setter
    def molecular_weight(self, mw: Any) -> None:
        self._molecular_weight = self._validate_number(mw, "molecular_weight")

    @property
    def pI_mean(self) -> float:
        return self._pI_mean

    @pI_mean.setter
    def pI_mean(self, p: Any) -> None:
        self._pI_mean = self._validate_number(p, "pI_mean")

    @property
    def pI_range(self) -> float:
        return self._pI_range

    @pI_range.setter
    def pI_range(self, r: Any) -> None:
        self._pI_range = self._validate_number(r, "pI_range")


class Buffer(Ingredient):
    def __init__(
        self,
        enc_id: int,
        name: str,
        pH: Union[int, float],
        id: Optional[int] = None,
    ) -> None:
        super().__init__(enc_id=enc_id, name=name, id=id)
        self.pH = pH

    @property
    def pH(self) -> float:
        return self._pH

    @pH.setter
    def pH(self, value: Any) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("pH must be a number")
        if not (0.0 <= value <= 14.0):
            raise ValueError("pH must be between 0 and 14")
        self._pH = float(value)


class Stabilizer(Ingredient):

    pass


class Surfactant(Ingredient):

    pass


class Salt(Ingredient):

    pass
