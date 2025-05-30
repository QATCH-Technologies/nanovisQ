import math
from typing import Optional, Dict, Any, List
try:
    from src.models.ingredient import (
        Ingredient, Buffer, Protein, Stabilizer, Surfactant, Salt)
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.ingredient import (
        Ingredient, Buffer, Protein, Stabilizer, Surfactant, Salt)

TAG = "[Formulation]"


class ViscosityProfile:
    def __init__(self, shear_rates: List[float], viscosities: List[float], units: str) -> None:
        if not isinstance(shear_rates, list) or not isinstance(viscosities, list):
            raise TypeError(
                "shear_rates and viscosities must be lists of numeric values")
        if len(shear_rates) != len(viscosities):
            raise ValueError(
                "shear_rates and viscosities must have the same length")
        if any(not isinstance(sr, (int, float)) for sr in shear_rates):
            raise TypeError("all shear_rates must be numeric")
        if any(not isinstance(v, (int, float)) for v in viscosities):
            raise TypeError("all viscosities must be numeric")
        if not isinstance(units, str) or not units.strip():
            raise ValueError("units must be a non-empty string")
        pairs = sorted(
            ((float(sr), float(v)) for sr, v in zip(shear_rates, viscosities)),
            key=lambda x: x[0]
        )
        self.shear_rates: List[float] = [sr for sr, _ in pairs]
        self.viscosities: List[float] = [v for _, v in pairs]
        self.units: str = units.strip()
        self._is_measured: bool = False

    @property
    def is_measured(self) -> bool:
        return self._is_measured

    @is_measured.setter
    def is_measured(self, value: bool) -> None:
        self._is_measured = value

    def get_viscosity(self, shear_rate: float) -> float:
        if not isinstance(shear_rate, (int, float)):
            raise TypeError("shear_rate must be numeric")
        sr = float(shear_rate)
        srs = self.shear_rates
        vs = self.viscosities

        import bisect
        idx = bisect.bisect_left(srs, sr)
        if idx < len(srs) and srs[idx] == sr:
            return vs[idx]
        if idx == 0:
            x0, x1 = srs[0], srs[1]
            y0, y1 = vs[0], vs[1]
        elif idx == len(srs):
            x0, x1 = srs[-2], srs[-1]
            y0, y1 = vs[-2], vs[-1]
        else:
            x0, x1 = srs[idx-1], srs[idx]
            y0, y1 = vs[idx-1], vs[idx]
        return y0 + (sr - x0) * (y1 - y0) / (x1 - x0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shear_rates": self.shear_rates,
            "viscosities": self.viscosities,
            "units": self.units,
            "is_measured": self._is_measured,
        }

    def __repr__(self) -> str:
        return (
            f"ViscosityProfile(shear_rates={self.shear_rates}, "
            f"viscosities={self.viscosities}, units={self.units!r})"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ViscosityProfile):
            return NotImplemented
        return (
            self.shear_rates == other.shear_rates
            and self.viscosities == other.viscosities
            and self.units == other.units
        )


class Component:
    def __init__(self, ingredient: Ingredient, concentration: float, units: str) -> None:
        if not isinstance(ingredient, Ingredient):
            raise TypeError("ingredient must be an Ingredient")
        if not isinstance(concentration, (int, float)):
            raise TypeError("concentration must be numeric")
        if concentration < 0:
            raise ValueError("concentration must be non-negative")
        if not isinstance(units, str) or not units.strip():
            raise ValueError("units must be a non-empty string")

        self.ingredient: Ingredient = ingredient
        self.concentration: float = float(concentration)
        self.units: str = units.strip()

    def __repr__(self) -> str:
        cls = self.ingredient.__class__.__name__
        return (f"Component({cls}={self.ingredient.name!r}, "
                f"conc={self.concentration}, units={self.units!r})")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.ingredient.to_dict(),
            "concentration": self.concentration,
            "units": self.units,
        }


class Formulation:
    def __init__(self, id: Optional[int] = None) -> None:
        if id is not None and not isinstance(id, int):
            raise TypeError("`id` must be an integer or None")
        self._id: Optional[int] = id

        self._components: Dict[str, Optional[Component]] = {
            "protein":    None,
            "buffer":     None,
            "stabilizer": None,
            "surfactant": None,
            "salt":       None,
        }
        self._temperature:       Optional[float] = None
        self._viscosity_profile: Optional[ViscosityProfile] = None

    @property
    def id(self) -> Optional[int]:
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("`id` must be an integer")
        self._id = value

    def set_protein(self, protein: Protein, concentration: float, units: str) -> None:
        self._components["protein"] = Component(protein, concentration, units)

    def set_buffer(self, buffer: Buffer, concentration: float, units: str) -> None:
        self._components["buffer"] = Component(buffer, concentration, units)

    def set_stabilizer(self, stabilizer: Stabilizer, concentration: float, units: str) -> None:
        self._components["stabilizer"] = Component(
            stabilizer, concentration, units)

    def set_surfactant(self, surfactant: Surfactant, concentration: float, units: str) -> None:
        self._components["surfactant"] = Component(
            surfactant, concentration, units)

    def set_salt(self, salt: Salt, concentration: float, units: str) -> None:
        self._components["salt"] = Component(salt, concentration, units)

    @property
    def protein(self) -> Optional[Component]:
        return self._components["protein"]

    @property
    def buffer(self) -> Optional[Component]:
        return self._components["buffer"]

    @property
    def stabilizer(self) -> Optional[Component]:
        return self._components["stabilizer"]

    @property
    def surfactant(self) -> Optional[Component]:
        return self._components["surfactant"]

    @property
    def salt(self) -> Optional[Component]:
        return self._components["salt"]

    @property
    def temperature(self) -> Optional[float]:
        return self._temperature

    def set_temperature(self, temp) -> None:
        if temp is None or math.isnan(temp):
            temp = 25.0
        self._temperature = temp

    def set_viscosity_profile(self, profile: ViscosityProfile) -> None:
        if not isinstance(profile, ViscosityProfile):
            raise TypeError("profile must be a ViscosityProfile")
        self._viscosity_profile = profile

    @property
    def viscosity_profile(self) -> Optional[ViscosityProfile]:
        return self._viscosity_profile

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"id": self.id}
        for key, comp in self._components.items():
            data[key] = comp.to_dict() if comp is not None else None
        data["temperature"] = self.temperature
        data["viscosity_profile"] = (
            self.viscosity_profile.to_dict()
            if self.viscosity_profile is not None
            else None
        )
        return data

    def __repr__(self) -> str:
        parts = [f"{k}={v!r}" for k, v in self._components.items()]
        parts.append(f"temperature={self.temperature!r}")
        parts.append(f"viscosity_profile={self.viscosity_profile!r}")
        return f"Formulation(id={self.id}, " + ", ".join(parts) + ")"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Formulation):
            return NotImplemented
        print(self.to_dict())
        print(other.to_dict())
        return self.to_dict() == other.to_dict()
