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
    2025-06-03

Version:
    1.4
"""
import math
import pandas as pd
from typing import Optional, Dict, Any, List
import numpy as np
from scipy.optimize import curve_fit
import bisect
try:
    from src.models.ingredient import (
        Ingredient, Buffer, Protein, Stabilizer, Surfactant, Salt, Excipient)
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.ingredient import (
        Ingredient, Buffer, Protein, Stabilizer, Surfactant, Salt, Excipient)

TAG = "[Formulation]"


class ViscosityProfile:
    """Represents a viscosity profile as a function of shear rate.

    This class stores matched pairs of shear rates and viscosities, ensuring
    the data is validated and sorted by ascending shear rate.

    Attributes:
        shear_rates (List[float]): Sorted list of shear rates.
        viscosities (List[float]): Corresponding sorted list of viscosities.
        units (str): The units for viscosity measurement (e.g., "cP").
        _is_measured (bool): Flag indicating whether this profile has been measured.
    """

    def __init__(self, shear_rates: List[float], viscosities: List[float], units: str = "cP") -> None:
        """Initialize a ViscosityProfile with shear rate–viscosity pairs.

        Validates and stores shear rates and viscosities in ascending order of shear rate.

        Args:
            shear_rates (List[float]): A list of numeric shear rates.
            viscosities (List[float]): A list of numeric viscosities corresponding to each shear rate.
            units (str): The units of the viscosity values (e.g., "cP").

        Raises:
            TypeError: If `shear_rates` or `viscosities` is not a list, or contains non-numeric values.
            ValueError: If the lengths of `shear_rates` and `viscosities` differ, or if `units` is an empty string.
        """
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
        """Indicates whether the viscosity profile has been flagged as measured.

        Returns:
            bool: True if the profile has been measured, False otherwise.
        """
        return self._is_measured

    @is_measured.setter
    def is_measured(self, value: bool) -> None:
        """Set the measurement status of the viscosity profile.

        Args:
            value (bool): True to mark the profile as measured, False to mark it as unmeasured.
        """
        self._is_measured = value

    def get_viscosity(self, shear_rate: float, std_tol: float = 0.0) -> float:
        """Retrieve viscosity at a given shear rate using logarithmic interpolation,
        enforcing a monotonic (non-increasing) relationship with tolerance.

        The logarithmic model is fitted to (shear_rate, viscosity) data as:
            η = a * log10(y - c) + b

        Args:
            shear_rate (float): Shear rate at which to compute viscosity.
            std_tol (float): Allowed standard deviation tolerance for monotonic deviation.

        Returns:
            float: The estimated viscosity at the specified shear rate.
        """
        if not isinstance(shear_rate, (int, float)):
            raise TypeError("shear_rate must be numeric")

        sr = float(shear_rate)
        srs = np.asarray(self.shear_rates, dtype=float)
        vs = np.asarray(self.viscosities, dtype=float)
        vs_monotonic = vs.copy()
        for i in range(1, len(vs_monotonic)):
            if vs_monotonic[i] > vs_monotonic[i - 1] + std_tol:
                vs_monotonic[i] = vs_monotonic[i - 1] + std_tol

        def log_func(x, a, b, c):
            return a * np.log10(np.maximum(x - c, 1e-8)) + b  # avoid log(0)

        try:
            # Reasonable bounds: prevent weird large offsets
            bounds = ([-np.inf, -np.inf, -np.inf],
                      [np.inf, np.inf, np.min(srs) * 0.9])
            popt, _ = curve_fit(log_func, srs, vs_monotonic,
                                p0=(2, 1, 0.1), bounds=bounds)
            a_fit, b_fit, c_fit = popt
            fitted_viscosity = log_func(sr, a_fit, b_fit, c_fit)
        except Exception:

            idx = bisect.bisect_left(srs, sr)
            if idx == 0:
                x0, x1, y0, y1 = srs[0], srs[1], vs_monotonic[0], vs_monotonic[1]
            elif idx == len(srs):
                x0, x1, y0, y1 = srs[-2], srs[-1], vs_monotonic[-2], vs_monotonic[-1]
            else:
                x0, x1, y0, y1 = srs[idx -
                                     1], srs[idx], vs_monotonic[idx - 1], vs_monotonic[idx]
            fitted_viscosity = y0 + (sr - x0) * (y1 - y0) / (x1 - x0)
        fitted_viscosity = max(fitted_viscosity, 0.0)
        if sr <= srs[0]:
            fitted_viscosity = vs_monotonic[0]
        elif sr >= srs[-1]:
            fitted_viscosity = vs_monotonic[-1]

        return fitted_viscosity

    def to_dict(self) -> Dict[str, Any]:
        """Convert the viscosity profile to a dictionary representation.

        Returns:
            Dict[str, Any]: A dictionary containing the following keys:
                - "shear_rates" (List[float]): The list of shear rates.
                - "viscosities" (List[float]): The list of viscosities.
                - "units" (str): The units of measurement for the viscosities.
                - "is_measured" (bool): Flag indicating whether this profile has been marked as measured.
        """
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
            raise TypeError(
                f"ingredient must be an Ingredient object found {ingredient}")
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
        return (f"Component({cls}={self.ingredient.name!r}, "
                f"conc={self.concentration}, units={self.units!r})")


class Formulation:
    """Represents a complete formulation consisting of various components, temperature, and viscosity profile.

    Each formulation can include a protein, buffer, stabilizer, surfactant, and salt as `Component` objects,
    along with an optional temperature and viscosity profile.

    Attributes:
        _id (Optional[int]): Unique identifier for the formulation.
        _components (Dict[str, Optional[Component]]): Mapping of component types to their `Component` instances.
        _temperature (Optional[float]): Temperature at which the formulation was prepared (default 25.0 if unset or NaN).
        _viscosity_profile (Optional[ViscosityProfile]): Viscosity profile associated with the formulation.
    """

    def __init__(self, id: Optional[int] = None) -> None:
        """Initialize a Formulation, optionally with a preset ID.

        Args:
            id (Optional[int]): Integer identifier for the formulation. If None, ID is unset.

        Raises:
            TypeError: If `id` is not an integer or None.
        """
        if id is not None and not isinstance(id, int):
            raise TypeError("`id` must be an integer or None")
        self._id: Optional[int] = id

        self._components: Dict[str, Optional[Component]] = {
            "protein":    None,
            "buffer":     None,
            "stabilizer": None,
            "surfactant": None,
            "salt":       None,
            "excipient":  None,
        }
        self._temperature: Optional[float] = None
        self._viscosity_profile: Optional[ViscosityProfile] = None

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
        self._components["stabilizer"] = Component(
            stabilizer, concentration, units)

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
        self._components["surfactant"] = Component(
            surfactant, concentration, units)

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
            salt (Salt): An instance of `Excipient` to include.
            excpient (float): Concentration of the excpient (must be ≥ 0).
            units (str): Units for the concentration (non-empty string).

        Raises:
            TypeError: If `salt` is not a `Excipient`, or if concentration is not numeric.
            ValueError: If concentration is negative, or if `units` is an empty string.
        """
        self._components["excipient"] = Component(
            excipient, concentration, units)

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
            Optional[Component]: The salt `Component` if set, otherwise None.
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
            - Each component (protein, buffer, stabilizer, surfactant, salt) if set, as a nested dict.
            - "temperature": The formulation temperature (or None).
            - "viscosity_profile": Dictionary representation of the `ViscosityProfile` (or None).

        Returns:
            Dict[str, Any]: A dictionary capturing all set attributes of the formulation.
        """
        data: Dict[str, Any] = {"id": self.id}
        for key, comp in self._components.items():
            if comp is not None:
                comp_dict = comp.to_dict()
                # Remove any internal 'id' field if present in component dict
                comp_dict.pop('id', None)
                data[key] = comp_dict
        data["temperature"] = self.temperature
        data["viscosity_profile"] = (
            self.viscosity_profile.to_dict()
            if self.viscosity_profile is not None
            else None
        )
        return data

    def to_dataframe(self, encoded: bool = True, training: bool = True) -> pd.DataFrame:
        """Convert this Formulation into a one‐row pandas DataFrame.

        Args:
            encoded (bool): If endocded is set to true, the enc_id's are returned
                for each categorical feature of the dataframe.  If encoded is false, the
                name is returned instead of the enc_id.

        Returns:
            pd.DataFrame: A DataFrame with exactly one row, containing the following
                columns in this order:
                    - "ID"
                    - "Protein_type"
                    - "Protein_class_type"
                    - "MW"
                    - "PI_mean"
                    - "PI_range"
                    - "Protein_conc"
                    - "Temperature"
                    - "Buffer_type"
                    - "Buffer_pH"
                    - "Buffer_conc"
                    - "Salt_type"
                    - "Salt_conc"
                    - "Excipient_type"
                    - "Excipient_conc
                    - "Stabilizer_type"
                    - "Stabilizer_conc"
                    - "Surfactant_type"
                    - "Surfactant_conc"
                    - "Viscosity_100"
                    - "Viscosity_1000"
                    - "Viscosity_10000"
                    - "Viscosity_100000"
                    - "Viscosity_15000000"
        """
        if encoded:
            row = {
                "ID":              self.id,
                "Protein_type":    self.protein.ingredient.enc_id,
                "Protein_class_type":   self.protein.ingredient.class_type,
                "MW":              self.protein.ingredient.molecular_weight,
                "PI_mean":         self.protein.ingredient.pI_mean,
                "PI_range":        self.protein.ingredient.pI_range,
                "Protein_conc":    self.protein.concentration,
                "Temperature":     getattr(self, "temperature", pd.NA),
                "Buffer_type":     self.buffer.ingredient.enc_id,
                "Buffer_pH":       self.buffer.ingredient.pH,
                "Buffer_conc":     self.buffer.concentration,
                "Salt_type":       self.salt.ingredient.enc_id,
                "Salt_conc":       self.salt.concentration,
                "Excipient_type":  self.excipient.ingredient.enc_id,
                "Excipient_conc":  self.excipient.concentration,
                "Stabilizer_type": self.stabilizer.ingredient.enc_id,
                "Stabilizer_conc": self.stabilizer.concentration,
                "Surfactant_type": self.surfactant.ingredient.enc_id,
                "Surfactant_conc": self.surfactant.concentration,
            }
        else:
            row = {
                "ID":              self.id,
                "Protein_type":    self.protein.ingredient.name,
                "Protein_class_type":   self.protein.ingredient.class_type,
                "MW":              self.protein.ingredient.molecular_weight,
                "PI_mean":         self.protein.ingredient.pI_mean,
                "PI_range":        self.protein.ingredient.pI_range,
                "Protein_conc":    self.protein.concentration,
                "Temperature":     getattr(self, "temperature", pd.NA),
                "Buffer_type":     self.buffer.ingredient.name,
                "Buffer_pH":       self.buffer.ingredient.pH,
                "Buffer_conc":     self.buffer.concentration,
                "Salt_type":       self.salt.ingredient.name,
                "Salt_conc":       self.salt.concentration,
                "Excipient_type":  self.excipient.ingredient.name,
                "Excipient_conc":  self.excipient.concentration,
                "Stabilizer_type": self.stabilizer.ingredient.name,
                "Stabilizer_conc": self.stabilizer.concentration,
                "Surfactant_type": self.surfactant.ingredient.name,
                "Surfactant_conc": self.surfactant.concentration,
            }

        if training:
            shear_rates = [100, 1000, 10000, 100000, 15000000]
            if self.viscosity_profile is not None:
                for r in shear_rates:
                    row[f"Viscosity_{int(r)}"] = self.viscosity_profile.get_viscosity(
                        r)

        df = pd.DataFrame([row])

        expected = [
            "ID",
            "Protein_type", "MW", "PI_mean", "PI_range", "Protein_conc", "Protein_class_type",
            "Temperature",
            "Buffer_type", "Buffer_pH", "Buffer_conc",
            "Salt_type", "Salt_conc",
            "Excipient_type", "Excipient_conc",
            "Stabilizer_type", "Stabilizer_conc",
            "Surfactant_type", "Surfactant_conc"]
        if training:
            expected.extend([
                "Viscosity_100", "Viscosity_1000", "Viscosity_10000",
                "Viscosity_100000", "Viscosity_15000000"
            ])
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
        return f"Formulation(id={self.id}, " + ", ".join(parts) + ")"

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
