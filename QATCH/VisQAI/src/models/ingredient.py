from abc import ABC
from typing import Any, Dict, Type, TypeVar, Union

Number = Union[int, float]
T = TypeVar("T", bound="Ingredient")


class Ingredient(ABC):
    def __init__(self, enc_id: int, name: str) -> None:
        self._enc_id = self._validate_int(enc_id, "enc_id")
        self._name = name

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
    def enc_id(self) -> int:
        return self._enc_id

    @enc_id.setter
    def enc_id(self, enc_id) -> None:
        self._enc_id = enc_id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: Any) -> None:
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        if not name.strip():
            raise ValueError("Name cannot be empty")
        self._name = name

    def __repr__(self) -> str:
        attrs = ", ".join(
            f"{k.lstrip('_')}={v!r}" for k, v in self.__dict__.items()
        )
        return f"{self.__class__.__name__}({attrs})"

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} {self.name!r}>"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Ingredient):
            return NotImplemented
        return self.__class__ is other.__class__ and self.__dict__ == other.__dict__

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Ingredient):
            return NotImplemented
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self.enc_id < other.enc_id

    def to_dict(self) -> Dict[str, Any]:
        return {k.lstrip('_'): v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        return cls(**data)


class Protein(Ingredient):
    def __init__(
        self,
        enc_id: int,
        name: str,
        molecular_weight: Number,
        pI_mean: Number,
        pI_range: Number
    ) -> None:
        super().__init__(enc_id, name)
        self._molecular_weight = molecular_weight
        self._pI_mean = pI_mean
        self._pI_range = pI_range

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
    def __init__(self, enc_id: int, name: str, pH: Number) -> None:
        super().__init__(enc_id, name)
        self._pH = pH

    @property
    def pH(self) -> float:
        return self._pH

    @pH.setter
    def pH(self, pH: Any) -> None:
        if not isinstance(pH, (int, float)):
            raise TypeError("pH must be a number")
        if not (0.0 <= pH <= 14.0):
            raise ValueError("pH must be between 0.0 and 14.0")
        self._pH = float(pH)


class Stabilizer(Ingredient):
    pass


class Surfactant(Ingredient):
    pass
