"""
formulation.py

This module defines classes for constructing and representing bioformulation applications.
It includes:

- `ViscosityProfile`: Represents a dynamic `ViscosityProfile` with shear rates and measured
    viscosities.
- `Component`: Encapsulates an `Ingredient` with its concentration and units.
- `Formulation`: Represents a complete formulation composed of optional components
  (protein, buffer, stabilizer, surfactant, salt), along with temperature and
  an associated `ViscosityProfile`.

The classes provide validation for input parameters, conversion to dictionary
representations, and comparison logic for equality checks. Utility methods ensure
that temperature defaults to 25°C if not provided or if NaN, and that viscosity
profiles are properly typed.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.6
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

try:
    TAG = "[Formulation (HEADLESS)]"
    from src.models.ingredient import (
        Buffer,
        Excipient,
        Ingredient,
        Protein,
        Salt,
        Stabilizer,
        Surfactant,
    )
except (ModuleNotFoundError, ImportError):
    TAG = "[Formulation]"
    from QATCH.VisQAI.src.models.ingredient import (
        Buffer,
        Excipient,
        Ingredient,
        Protein,
        Salt,
        Stabilizer,
        Surfactant,
    )


class ViscosityProfile:
    """Represents a viscosity profile as a function of shear rate.

    This class stores matched pairs of shear rates and viscosities, ensuring
    the data is validated, sorted by ascending shear rate, and smoothly interpolated.

    Attributes:
        shear_rates (List[float]): Sorted list of shear rates.
        viscosities (List[float]): Corresponding sorted list of viscosities.
        units (str): The units for viscosity measurement (e.g., "cP").
        _is_measured (bool): Flag indicating whether this profile has been measured.
    """

    def __init__(
        self, shear_rates: List[float], viscosities: List[float], units: str = "cP"
    ) -> None:
        if not isinstance(shear_rates, list) or not isinstance(viscosities, list):
            raise TypeError("shear_rates and viscosities must be lists of numeric values")
        if len(shear_rates) != len(viscosities):
            raise ValueError("shear_rates and viscosities must have the same length")
        if any(not isinstance(sr, (int, float)) for sr in shear_rates):
            raise TypeError("all shear_rates must be numeric")
        if any(not isinstance(v, (int, float)) for v in viscosities):
            raise TypeError("all viscosities must be numeric")
        if not isinstance(units, str) or not units.strip():
            raise ValueError("units must be a non-empty string")

        pairs = sorted(
            ((float(sr), float(v)) for sr, v in zip(shear_rates, viscosities)),
            key=lambda x: x[0],
        )

        self.shear_rates: List[float] = [sr for sr, _ in pairs]
        self.viscosities: List[float] = [v for _, v in pairs]
        self.units: str = units.strip()
        self._is_measured: bool = False
        self._build_interpolator()

    def _build_interpolator(self, std_tol: float = 0.1):
        if not self.shear_rates:
            self._interpolator = None
            return

        vs_monotonic = np.array(self.viscosities)
        if len(vs_monotonic) > 1:
            visc_diffs = np.diff(vs_monotonic)
            std_dev = np.std(visc_diffs) if len(visc_diffs) > 0 else 0.0
            tolerance = std_tol * std_dev

            for i in range(1, len(vs_monotonic)):
                if vs_monotonic[i] > vs_monotonic[i - 1] + tolerance:
                    vs_monotonic[i] = vs_monotonic[i - 1] + tolerance

        self._vs_monotonic = vs_monotonic
        if len(self.shear_rates) > 1:
            log_srs = np.log10(np.maximum(self.shear_rates, 1e-10))
            self._interpolator = PchipInterpolator(log_srs, self._vs_monotonic)
        else:
            self._interpolator = None

    @property
    def is_measured(self) -> bool:
        return self._is_measured

    @is_measured.setter
    def is_measured(self, value: bool) -> None:
        self._is_measured = value

    def get_viscosity(self, shear_rate: float, std_tol: float = 0.1) -> float:
        """Retrieve viscosity at a given shear rate using a PCHIP interpolator.

        Args:
            shear_rate (float): Shear rate at which to compute viscosity.
            std_tol (float): Kept for backward compatibility.
                             (Monotonic adjustment is now handled at instantiation).

        Returns:
            float: The estimated viscosity at the specified shear rate.
        """
        if not isinstance(shear_rate, (int, float)):
            raise TypeError("shear_rate must be numeric")

        if not self.shear_rates:
            return 0.0

        sr = float(shear_rate)
        if sr <= self.shear_rates[0]:
            return float(self._vs_monotonic[0])
        if sr >= self.shear_rates[-1]:
            return float(self._vs_monotonic[-1])
        if self._interpolator is not None:
            log_sr = np.log10(max(sr, 1e-10))
            return float(self._interpolator(log_sr))

        return float(self._vs_monotonic[0])

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
    """Represents a formulation component consisting of an ingredient and its concentration.

    Attributes:
        ingredient (Ingredient): The ingredient object associated with this component.
        concentration (float): The concentration of the ingredient (non-negative).
        units (str): The units for the concentration (e.g., "mg/mL").
    """

    def __init__(self, ingredient: Ingredient, concentration: float, units: str) -> None:
        """Initialize a Component with an ingredient and its concentration.

        Validates that the ingredient is of the correct type, the concentration is a non-negative numeric value,
        and the units string is non-empty.

        Args:
            ingredient (Ingredient): The ingredient for this component.
            concentration (float): The concentration of the ingredient (must be ≥ 0).
            units (str): A non-empty string specifying the units of concentration.

        Raises:
            TypeError: If `ingredient` is not an instance of Ingredient, or if `concentration` is not numeric.
            ValueError: If `concentration` is negative, or if `units` is not a non-empty string.
        """
        if not isinstance(ingredient, Ingredient):
            raise TypeError(f"ingredient must be an Ingredient object found {ingredient}")
        if not isinstance(concentration, (int, float)):
            raise TypeError("concentration must be numeric")
        if concentration < 0:
            raise ValueError("concentration must be non-negative")
        if not isinstance(units, str) or not units.strip():
            raise ValueError("units must be a non-empty string")

        self.ingredient: Ingredient = ingredient
        self.concentration: float = float(concentration)
        self.units: str = units.strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the component to a dictionary representation.

        Returns:
            Dict[str, Any]: A dictionary containing the following keys:
                - "type" (Dict[str, Any]): The dictionary representation of the associated Ingredient.
                - "concentration" (float): The concentration value of the component.
                - "units" (str): The units for the concentration.
        """
        return {
            "type": self.ingredient.to_dict(),
            "concentration": self.concentration,
            "units": self.units,
        }

    def __repr__(self) -> str:
        cls = self.ingredient.__class__.__name__
        return (
            f"Component({cls}={self.ingredient.name!r}, "
            f"conc={self.concentration}, units={self.units!r})"
        )


class Formulation:
    TAU = 1.5
    """Represents a complete formulation consisting of various components, temperature, and viscosity profile.

    Each formulation can include a protein, buffer, stabilizer, surfactant, salt, and excipient as `Component` objects,
    Each formulation can include a protein, buffer, stabilizer, surfactant, salt, and excipient as `Component` objects,
    along with an optional temperature and viscosity profile.

    Attributes:
        _id (Optional[int]): Unique identifier for the formulation.
        _components (Dict[str, Optional[Component]]): Mapping of component types to their `Component` instances.
        _temperature (Optional[float]): Temperature at which the formulation was prepared (default 25.0 if unset or NaN).
        _viscosity_profile (Optional[ViscosityProfile]): Viscosity profile associated with the formulation.
    """

    def __init__(
        self,
        id: Optional[int] = None,
        name: Optional[str] = None,
        signature: Optional[str] = None,
    ) -> None:
        """Initialize a Formulation, optionally with ID, name, and signature.

        Args:
            id (Optional[int]): Integer identifier for the formulation.
            name (Optional[str]): Human-readable name for the formulation.
            signature (Optional[str]): SHA256 hash signature for DB recall.
        """
        if id is not None and not isinstance(id, int):
            raise TypeError("`id` must be an integer or None")
        if name is not None and not isinstance(name, str):
            raise TypeError("`name` must be a string or None")
        if signature is not None and not isinstance(signature, str):
            raise TypeError("`signature` must be a string or None")

        self._id: Optional[int] = id
        self._name: Optional[str] = name
        self._signature: Optional[str] = signature
        self._components: Dict[str, Optional[Component]] = {
            "protein": None,
            "buffer": None,
            "stabilizer": None,
            "surfactant": None,
            "salt": None,
            "excipient": None,
        }
        self._temperature: Optional[float] = None
        self._viscosity_profile: Optional[ViscosityProfile] = None
        self.missing_fields = []
        self.notes = ""
        self._icl: bool = True
        self._last_model: Optional[str] = None

    @property
    def icl(self) -> bool:
        """Get the In-Context Learning inclusion flag."""
        return self._icl

    @icl.setter
    def icl(self, value: bool) -> None:
        """Set the In-Context Learning inclusion flag."""
        if not isinstance(value, bool):
            raise TypeError("icl must be a boolean")
        self._icl = value

    @property
    def last_model(self) -> Optional[str]:
        """Get the name of the last model used for prediction."""
        return self._last_model

    @last_model.setter
    def last_model(self, value: str) -> None:
        """Set the name of the last model used for prediction."""
        if value is not None and not isinstance(value, str):
            raise TypeError("last_model must be a string or None")
        self._last_model = value

    @property
    def id(self) -> Optional[int]:
        """Get the formulation's unique identifier.

        Returns:
            Optional[int]: The formulation ID, or None if not set.
        """
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        """Set the formulation's unique identifier.

        Args:
            value (int): New integer ID for the formulation.

        Raises:
            TypeError: If `value` is not an integer.
        """
        if not isinstance(value, int):
            raise TypeError("`id` must be an integer")
        self._id = value

    @property
    def name(self) -> Optional[str]:
        """Get the formulation name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the formulation name."""
        if not isinstance(value, str):
            raise TypeError("name must be a string")
        self._name = value

    @property
    def signature(self) -> Optional[str]:
        """Get the formulation SHA256 signature."""
        return self._signature

    @signature.setter
    def signature(self, value: str) -> None:
        """Set the formulation SHA256 signature."""
        if not isinstance(value, str):
            raise TypeError("signature must be a string")
        self._signature = value

    def set_protein(self, protein: Protein, concentration: float, units: str) -> None:
        """Assign a protein component to the formulation.

        Args:
            protein (Protein): An instance of `Protein` to include.
            concentration (float): Concentration of the protein (must be ≥ 0).
            units (str): Units for the concentration (non-empty string).

        Raises:
            TypeError: If `protein` is not a `Protein`, or if concentration is not numeric.
            ValueError: If concentration is negative, or if `units` is an empty string.
        """
        self._components["protein"] = Component(protein, concentration, units)

    def set_buffer(self, buffer: Buffer, concentration: float, units: str) -> None:
        """Assign a buffer component to the formulation.

        Args:
            buffer (Buffer): An instance of `Buffer` to include.
            concentration (float): Concentration of the buffer (must be ≥ 0).
            units (str): Units for the concentration (non-empty string).

        Raises:
            TypeError: If `buffer` is not a `Buffer`, or if concentration is not numeric.
            ValueError: If concentration is negative, or if `units` is an empty string.
        """
        self._components["buffer"] = Component(buffer, concentration, units)

    def set_stabilizer(self, stabilizer: Stabilizer, concentration: float, units: str) -> None:
        """Assign a stabilizer component to the formulation.

        Args:
            stabilizer (Stabilizer): An instance of `Stabilizer` to include.
            concentration (float): Concentration of the stabilizer (must be ≥ 0).
            units (str): Units for the concentration (non-empty string).

        Raises:
            TypeError: If `stabilizer` is not a `Stabilizer`, or if concentration is not numeric.
            ValueError: If concentration is negative, or if `units` is an empty string.
        """
        self._components["stabilizer"] = Component(stabilizer, concentration, units)

    def set_surfactant(self, surfactant: Surfactant, concentration: float, units: str) -> None:
        """Assign a surfactant component to the formulation.

        Args:
            surfactant (Surfactant): An instance of `Surfactant` to include.
            concentration (float): Concentration of the surfactant (must be ≥ 0).
            units (str): Units for the concentration (non-empty string).

        Raises:
            TypeError: If `surfactant` is not a `Surfactant`, or if concentration is not numeric.
            ValueError: If concentration is negative, or if `units` is an empty string.
        """
        self._components["surfactant"] = Component(surfactant, concentration, units)

    def set_salt(self, salt: Salt, concentration: float, units: str) -> None:
        """Assign a salt component to the formulation.

        Args:
            salt (Salt): An instance of `Salt` to include.
            concentration (float): Concentration of the salt (must be ≥ 0).
            units (str): Units for the concentration (non-empty string).

        Raises:
            TypeError: If `salt` is not a `Salt`, or if concentration is not numeric.
            ValueError: If concentration is negative, or if `units` is an empty string.
        """
        self._components["salt"] = Component(salt, concentration, units)

    def set_excipient(self, excipient: Excipient, concentration: float, units: str) -> None:
        """Assign a excpient component to the formulation.

        Args:
            excipient (Excipient): An instance of `Excipient` to include.
            concentration (float): Concentration of the excipient (must be ≥ 0).
            excipient (Excipient): An instance of `Excipient` to include.
            concentration (float): Concentration of the excipient (must be ≥ 0).
            units (str): Units for the concentration (non-empty string).

        Raises:
            TypeError: If `excipient` is not an `Excipient`, or if concentration is not numeric.
            TypeError: If `excipient` is not an `Excipient`, or if concentration is not numeric.
            ValueError: If concentration is negative, or if `units` is an empty string.
        """
        self._components["excipient"] = Component(excipient, concentration, units)

    @property
    def protein(self) -> Optional[Component]:
        """Get the protein component of the formulation.

        Returns:
            Optional[Component]: The protein `Component` if set, otherwise None.
        """
        return self._components["protein"]

    @property
    def buffer(self) -> Optional[Component]:
        """Get the buffer component of the formulation.

        Returns:
            Optional[Component]: The buffer `Component` if set, otherwise None.
        """
        return self._components["buffer"]

    @property
    def stabilizer(self) -> Optional[Component]:
        """Get the stabilizer component of the formulation.

        Returns:
            Optional[Component]: The stabilizer `Component` if set, otherwise None.
        """
        return self._components["stabilizer"]

    @property
    def surfactant(self) -> Optional[Component]:
        """Get the surfactant component of the formulation.

        Returns:
            Optional[Component]: The surfactant `Component` if set, otherwise None.
        """
        return self._components["surfactant"]

    @property
    def salt(self) -> Optional[Component]:
        """Get the salt component of the formulation.

        Returns:
            Optional[Component]: The salt `Component` if set, otherwise None.
        """
        return self._components["salt"]

    @property
    def excipient(self) -> Optional[Component]:
        """Get the excipient component of the formulation.

        Returns:
            Optional[Component]: The excipient `Component` if set, otherwise None.
            Optional[Component]: The excipient `Component` if set, otherwise None.
        """
        return self._components["excipient"]

    @property
    def temperature(self) -> Optional[float]:
        """Get the formulation's temperature.

        Returns:
            Optional[float]: The temperature value if set, otherwise None.
        """
        return self._temperature

    def set_temperature(self, temp) -> None:
        """Set the formulation's temperature, defaulting to 25.0 if None or NaN.

        Args:
            temp (Any): The temperature value to set. If None or NaN, sets to 25.0.
        """
        if temp is None or math.isnan(temp):
            temp = 25.0
        self._temperature = temp

    def set_viscosity_profile(self, profile: ViscosityProfile) -> None:
        """Associate a viscosity profile with the formulation.

        Args:
            profile (ViscosityProfile): A `ViscosityProfile` instance.

        Raises:
            TypeError: If `profile` is not an instance of `ViscosityProfile`.
        """
        if not isinstance(profile, ViscosityProfile):
            raise TypeError("profile must be a ViscosityProfile")
        self._viscosity_profile = profile

    @property
    def viscosity_profile(self) -> Optional[ViscosityProfile]:
        """Get the viscosity profile associated with the formulation.

        Returns:
            Optional[ViscosityProfile]: The `ViscosityProfile` if set, otherwise None.
        """
        return self._viscosity_profile

    def to_dict(self) -> Dict[str, Any]:
        """Convert the formulation to a dictionary representation.

        The dictionary includes:
            - "id": Formulation ID (or None).
            - "name": Formulation name (or None).
            - "signature": Formulation signature (or None).
            - Each component (protein, buffer, stabilizer, surfactant, salt) if set, as a nested dict.
            - "temperature": The formulation temperature (or None).
            - "viscosity_profile": Dictionary representation of the `ViscosityProfile` (or None).

        Returns:
            Dict[str, Any]: A dictionary capturing all set attributes of the formulation.
        """
        data: Dict[str, Any] = {
            "name": self.name,
            "signature": self.signature,
        }
        for key, comp in self._components.items():
            if comp is not None:
                comp_dict = comp.to_dict()
                comp_dict.pop("id", None)
                data[key] = comp_dict
        data["temperature"] = self.temperature
        data["viscosity_profile"] = (
            self.viscosity_profile.to_dict() if self.viscosity_profile is not None else None
        )
        return data

    def to_dataframe(self, encoded: bool = True, training: bool = True) -> pd.DataFrame:
        """Convert this Formulation into a one-row pandas DataFrame.

        Args:
            encoded (bool): If encoded is set to true, the enc_id's are returned
                for each categorical feature of the dataframe. If encoded is false, the
                name is returned instead of the enc_id.
            training (bool): If true, viscosity columns are included (if available).

        Returns:
            pd.DataFrame: A DataFrame with exactly one row, containing the following
                columns in this order:
                    - ID
                    - Protein_type, Protein_class_type, kP, MW, PI_mean, PI_range, Protein_conc
                    - Temperature
                    - Buffer_type, Buffer_pH, Buffer_conc
                    - Salt_type, Salt_conc
                    - Stabilizer_type, Stabilizer_conc
                    - Surfactant_type, Surfactant_conc
                    - Excipient_type, Excipient_conc
                    - C_Class, HCI
                    - Viscosity columns (if training=True)
        """
        expected = [
            "ID",
            "Protein_type",
            "Protein_class_type",
            "kP",
            "MW",
            "PI_mean",
            "PI_range",
            "Protein_conc",
            "Temperature",
            "Buffer_type",
            "Buffer_pH",
            "Buffer_conc",
            "Salt_type",
            "Salt_conc",
            "Stabilizer_type",
            "Stabilizer_conc",
            "Surfactant_type",
            "Surfactant_conc",
            "Excipient_type",
            "Excipient_conc",
            "C_Class",
            "HCI",
        ]

        def safe_get(component, attr_path, default=0):
            """Safely retrieves nested attributes, returning default if any step is None."""
            if component is None or component.ingredient is None:
                return default

            obj = component.ingredient
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr, default)
                    if obj is None:
                        return default
            except Exception:
                return default
            return obj

        def get_comp_data(comp_name):
            comp = getattr(self, comp_name, None)
            if comp and comp.ingredient:
                return comp, comp.ingredient
            return None, None

        prot, prot_ing = get_comp_data("protein")
        buff, buff_ing = get_comp_data("buffer")
        salt, salt_ing = get_comp_data("salt")
        stab, stab_ing = get_comp_data("stabilizer")
        surf, surf_ing = get_comp_data("surfactant")
        exc, exc_ing = get_comp_data("excipient")

        row = {
            "ID": self.id,
            "Temperature": (
                getattr(self, "temperature", 25.0) if self.temperature is not None else 25.0
            ),
            # Protein Defaults
            "Protein_class_type": safe_get(prot, "class_type.value", 0),
            "kP": safe_get(prot, "class_type.kP", 0),
            "HCI": safe_get(prot, "class_type.hci", 0),
            "C_Class": safe_get(prot, "class_type.c_class", 0),
            "MW": safe_get(prot, "molecular_weight", 0),
            "PI_mean": safe_get(prot, "pI_mean", 0),
            "PI_range": safe_get(prot, "pI_range", 0),
            "Protein_conc": prot.concentration if prot else 0.0,
            "Buffer_pH": safe_get(buff, "pH", 0),
            "Buffer_conc": buff.concentration if buff else 0.0,
            "Salt_conc": salt.concentration if salt else 0.0,
            "Excipient_conc": exc.concentration if exc else 0.0,
            "Stabilizer_conc": stab.concentration if stab else 0.0,
            "Surfactant_conc": surf.concentration if surf else 0.0,
        }

        if encoded:
            row.update(
                {
                    "Protein_type": getattr(prot_ing, "enc_id", 0) if prot_ing else 0,
                    "Buffer_type": getattr(buff_ing, "enc_id", 0) if buff_ing else 0,
                    "Salt_type": getattr(salt_ing, "enc_id", 0) if salt_ing else 0,
                    "Excipient_type": getattr(exc_ing, "enc_id", 0) if exc_ing else 0,
                    "Stabilizer_type": (getattr(stab_ing, "enc_id", 0) if stab_ing else 0),
                    "Surfactant_type": (getattr(surf_ing, "enc_id", 0) if surf_ing else 0),
                }
            )
        else:
            row.update(
                {
                    "Protein_type": prot_ing.name if prot_ing else "None",
                    "Buffer_type": buff_ing.name if buff_ing else "None",
                    "Salt_type": salt_ing.name if salt_ing else "None",
                    "Excipient_type": exc_ing.name if exc_ing else "None",
                    "Stabilizer_type": stab_ing.name if stab_ing else "None",
                    "Surfactant_type": surf_ing.name if surf_ing else "None",
                }
            )
        if training:
            shear_rates = [100, 1000, 10000, 100000, 15000000]
            visc_cols = [f"Viscosity_{r}" for r in shear_rates]
            expected.extend(visc_cols)

            if self.viscosity_profile is not None:
                for r in shear_rates:
                    row[f"Viscosity_{r}"] = self.viscosity_profile.get_viscosity(r)
            else:
                for col in visc_cols:
                    row[col] = pd.NA

        df = pd.DataFrame([row])
        for col in expected:
            if col not in df.columns:
                df[col] = pd.NA

        return df[expected]

    def __repr__(self) -> str:
        """Provide a string representation of the formulation for debugging.

        Returns:
            str: String describing the formulation ID, components, temperature, and viscosity profile.
        """
        parts = [f"{k}={v!r}" for k, v in self._components.items()]
        parts.append(f"temperature={self.temperature!r}")
        parts.append(f"viscosity_profile={self.viscosity_profile!r}")

        return (
            f"Formulation(id={self.id}, name={self.name!r}, "
            f"signature={self.signature!r}, " + ", ".join(parts) + ")"
        )

    def __eq__(self, other: Any) -> bool:
        """Compare two formulations for equality, ignoring their IDs.

        Two formulations are considered equal if all components, temperature,
        and viscosity profile match, regardless of their `id` values.

        Args:
            other (Any): Object to compare against.

        Returns:
            bool: True if `other` is a `Formulation` with identical data (excluding IDs), False otherwise.
        """
        if not isinstance(other, Formulation):
            return NotImplemented

        d1 = self.to_dict()
        d2 = other.to_dict()

        d1.pop("id", None)
        d2.pop("id", None)
        return d1 == d2
