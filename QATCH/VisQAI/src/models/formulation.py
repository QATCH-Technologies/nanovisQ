from typing import Optional, Dict, Any
from ingredient import Ingredient, Buffer, Protein, Stabilizer, Surfactant


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
    """A formulation made up of one protein, buffer, stabilizer, and surfactant."""

    def __init__(self, formulation_id: int) -> None:
        if not isinstance(formulation_id, int):
            raise TypeError("formulation_id must be an integer")
        self._id: int = formulation_id
        self._components: Dict[str, Optional[Component]] = {
            "protein": None,
            "buffer": None,
            "stabilizer": None,
            "surfactant": None,
        }
        self._temperature: Optional[float] = None
        self._nacl_concentration: Optional[float] = None

    @property
    def id(self) -> int:
        return self._id

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
    def temperature(self) -> Optional[float]:
        return self._temperature

    def set_temperature(self, temp: float) -> None:
        if not isinstance(temp, (int, float)):
            raise TypeError("temperature must be numeric")
        self._temperature = float(temp)

    @property
    def nacl_concentration(self) -> Optional[float]:
        return self._nacl_concentration

    def set_nacl_concentration(self, conc: float) -> None:
        if not isinstance(conc, (int, float)):
            raise TypeError("nacl_concentration must be numeric")
        if conc < 0:
            raise ValueError("nacl_concentration must be non-negative")
        self._nacl_concentration = float(conc)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"id": self.id}
        for key, comp in self._components.items():
            data[key] = comp.to_dict() if comp is not None else None
        data["temperature"] = self.temperature
        data["nacl_concentration"] = self.nacl_concentration
        return data

    def __repr__(self) -> str:
        parts = [f"{k}={v!r}" for k, v in self._components.items()]
        parts.append(f"temperature={self.temperature!r}")
        parts.append(f"nacl_concentration={self.nacl_concentration!r}")
        return f"Formulation(id={self.id}, " + ", ".join(parts) + ")"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Formulation):
            return NotImplemented
        return self.to_dict() == other.to_dict()


if __name__ == "__main__":
    form = Formulation(1)
    protein = Protein(1, "BSA", 100, 10, 1)
    buff = Buffer(1, "PBS", 7.4)
    stab = Stabilizer(1, 'None')
    surf = Surfactant(1, "None")
    form.set_temperature(25)
    form.set_nacl_concentration(100)
    form.set_protein(protein, 100, "mg/ml")
    form.set_buffer(buff, 10, 'M')
    form.set_stabilizer(stab, 0, "M")
    form.set_surfactant(surf, 0, "%w")
    input(form.protein.ingredient.molecular_weight)
    print(form.to_dict())
