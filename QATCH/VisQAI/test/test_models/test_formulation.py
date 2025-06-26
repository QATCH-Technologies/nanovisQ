"""
test_models.py

Unit tests for the Formulation, Component, and ViscosityProfile model classes under `src/models`.
Verifies:
    - Type and value validation in constructors
    - Sorting and unit normalization in ViscosityProfile
    - Interpolation logic in get_viscosity, including boundary cases
    - Setter/getter behavior for is_measured
    - to_dict(), __repr__(), and __eq__() methods for ViscosityProfile
    - Component initialization, repr, and to_dict validation
    - Formulation initialization, component setters, temperature logic, and viscosity_profile assignment
    - to_dict(), __repr__(), and __eq__() for Formulation, including id handling

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Date:
    2025-06-03

Version:
    1.4
"""

from src.models.formulation import (
    ViscosityProfile, Component, Formulation
)
from src.models.ingredient import Buffer, Protein, Stabilizer, Surfactant, Salt
import pandas as pd
import unittest


class TestViscosityProfile(unittest.TestCase):
    """Unit tests for the ViscosityProfile class."""

    def test_init_type_checks(self):
        """
        Test that constructor enforces type and length checks for shear_rates, viscosities, and units.

        - Passing non-list for shear_rates or viscosities raises TypeError.
        - Mismatched list lengths raises ValueError.
        - Non-numeric entries in shear_rates or viscosities raise TypeError.
        - Empty or whitespace-only units string raises ValueError.
        """
        with self.assertRaises(TypeError):
            ViscosityProfile("not-a-list", [1, 2], "u")
        with self.assertRaises(TypeError):
            ViscosityProfile([1, 2], "not-a-list", "u")
        with self.assertRaises(ValueError):
            ViscosityProfile([1, 2, 3], [1, 2], "u")
        with self.assertRaises(TypeError):
            ViscosityProfile([1, "two"], [1.0, 2.0], "u")
        with self.assertRaises(TypeError):
            ViscosityProfile([1.0, 2.0], [1, None], "u")
        with self.assertRaises(ValueError):
            ViscosityProfile([1.0], [2.0], "")

    def test_sorts_and_strips_units(self):
        """
        Test that shear_rates and viscosities are sorted ascending by shear_rate
        and that units string is stripped of whitespace.

        - Given unsorted lists, verify sorted state in object.
        - Confirm units string is trimmed.
        - Confirm default is_measured flag is False.
        """
        vp = ViscosityProfile([10, 1, 5], [100, 10, 50], "  cp ")
        self.assertEqual(vp.shear_rates, [1.0, 5.0, 10.0])
        self.assertEqual(vp.viscosities, [10.0, 50.0, 100.0])
        self.assertEqual(vp.units, "cp")
        self.assertFalse(vp.is_measured)

    def test_is_measured_setter(self):
        """
        Test setter and getter for is_measured flag.

        - Can set True and False interchangeably.
        """
        vp = ViscosityProfile([1], [1], "u")
        vp.is_measured = True
        self.assertTrue(vp.is_measured)
        vp.is_measured = False
        self.assertFalse(vp.is_measured)

    def test_get_viscosity_exact(self):
        """
        Test that get_viscosity returns exact viscosity when shear_rate matches an entry.

        - Verify both int and float input return correct float viscosity.
        """
        vp = ViscosityProfile([1, 2, 3], [10, 20, 30], "u")
        self.assertEqual(vp.get_viscosity(2), 20.0)
        self.assertEqual(vp.get_viscosity(2.0), 20.0)

    def test_get_viscosity_interpolation(self):
        """
        Test linear interpolation behavior of get_viscosity for values between, below, and above known shear_rates.

        - Midpoint: should return average.
        - Below lowest: extrapolate using first two points.
        - Above highest: extrapolate using last two points.
        """
        vp = ViscosityProfile([1, 3], [10, 30], "u")
        # midpoint
        self.assertEqual(vp.get_viscosity(2), 20.0)
        # below lowest (0 -> extrapolate from (1,10) and (3,30))
        expected_below = 10.0 + (0 - 1) * (30 - 10) / (3 - 1)
        self.assertEqual(vp.get_viscosity(0), expected_below)
        # above highest (4 -> extrapolate)
        expected_above = 10.0 + (4 - 1) * (30 - 10) / (3 - 1)
        self.assertEqual(vp.get_viscosity(4), expected_above)

    def test_get_viscosity_type_error(self):
        """
        Test that get_viscosity raises TypeError when shear_rate argument is non-numeric.
        """
        vp = ViscosityProfile([1], [1], "u")
        with self.assertRaises(TypeError):
            vp.get_viscosity("fast")

    def test_to_dict_and_repr_and_eq(self):
        """
        Test to_dict, __repr__, and __eq__ methods.

        - to_dict returns correct dictionary with sorted values and is_measured flag.
        - repr contains class name.
        - __eq__ compares only shear_rates, viscosities, and units (ignores is_measured).
        """
        vp = ViscosityProfile([2, 1], [20, 10], "cP")
        vp.is_measured = True
        d = vp.to_dict()
        self.assertEqual(d, {
            "shear_rates": [1.0, 2.0],
            "viscosities": [10.0, 20.0],
            "units": "cP",
            "is_measured": True
        })
        r = repr(vp)
        self.assertIn("ViscosityProfile", r)
        # eq ignores is_measured
        vp2 = ViscosityProfile([1, 2], [10, 20], "cP")
        self.assertEqual(vp, vp2)
        vp3 = ViscosityProfile([1, 2], [10, 21], "cP")
        self.assertNotEqual(vp, vp3)


class TestComponent(unittest.TestCase):
    """Unit tests for the Component class."""

    def setUp(self):
        """Create a sample Protein to use in Component tests."""
        self.prot = Protein(1, "P", 10, 7, 1)

    def test_init_type_and_value(self):
        """
        Test constructor type and value validations.

        - ingredient must be Ingredient subclass; invalid type raises TypeError.
        - concentration must be numeric; invalid type raises TypeError.
        - negative concentration raises ValueError.
        - units must be non-empty string; empty raises ValueError.
        - Valid input should store correct floating concentration and trimmed units.
        """
        with self.assertRaises(TypeError):
            Component("not-ing", 1, "u")
        with self.assertRaises(TypeError):
            Component(self.prot, "conc", "u")
        with self.assertRaises(ValueError):
            Component(self.prot, -1, "u")
        with self.assertRaises(ValueError):
            Component(self.prot, 1, "")
        # valid
        comp = Component(self.prot, 5, " mg/mL ")
        self.assertIs(comp.ingredient, self.prot)
        self.assertEqual(comp.concentration, 5.0)
        self.assertEqual(comp.units, "mg/mL")

    def test_repr_and_to_dict(self):
        """
        Test __repr__ and to_dict methods.

        - repr should contain "Component(Protein='P'...)".
        - to_dict returns a dictionary containing 'type', 'concentration', and 'units'.
        """
        comp = Component(self.prot, 2, "X")
        r = repr(comp)
        self.assertIn("Component(Protein='P'", r)
        d = comp.to_dict()
        self.assertIn("type", d)
        self.assertEqual(d["concentration"], 2.0)
        self.assertEqual(d["units"], "X")


class TestFormulation(unittest.TestCase):
    """Unit tests for the Formulation class."""

    def setUp(self):
        """Create a Formulation and sample Ingredient and ViscosityProfile instances."""
        self.form = Formulation()
        self.prot = Protein(1, "BSA", 100, 10, 1)
        self.buf = Buffer(2, "PBS", 7.4)
        self.stab = Stabilizer(3, "None")
        self.surf = Surfactant(4, "None")
        self.salt = Salt(5, "NaCl")
        self.vp = ViscosityProfile([1, 10], [1, 2], "u")

    def test_init_and_id_property(self):
        """
        Test id parameter and id setter/getter.

        - Passing integer id in constructor should set .id.
        - Passing non-int raises TypeError.
        - Setting .id to non-int raises TypeError.
        """
        f = Formulation(id=42)
        self.assertEqual(f.id, 42)
        with self.assertRaises(TypeError):
            Formulation(id="nope")
        # setter
        f.id = 100
        self.assertEqual(f.id, 100)
        with self.assertRaises(TypeError):
            f.id = None  # must be int

    def test_default_components_none(self):
        """
        Test that all component properties default to None.

        - protein, buffer, stabilizer, surfactant, and salt should be None initially.
        """
        for name in ("protein", "buffer", "stabilizer", "surfactant", "salt"):
            self.assertIsNone(getattr(self.form, name))

    def test_setting_components(self):
        """
        Test that set_<component> methods populate component slots correctly.

        - Call each setter with valid Ingredient, concentration, and units.
        - After each, verify that component is not None with correct concentration/units.
        """
        self.form.set_protein(self.prot, 50, "mg")
        self.assertIsNotNone(self.form.protein)
        self.assertEqual(self.form.protein.concentration, 50.0)

        self.form.set_buffer(self.buf, 1, "L")
        self.assertEqual(self.form.buffer.units, "L")

        self.form.set_stabilizer(self.stab, 0, "M")
        self.form.set_surfactant(self.surf, 0, "%w")
        self.form.set_salt(self.salt, 5, "g")

        # confirm all five slots filled
        for comp in (
                self.form.protein,
                self.form.buffer,
                self.form.stabilizer,
                self.form.surfactant,
                self.form.salt):
            self.assertIsNotNone(comp)

    def test_temperature(self):
        """
        Test set_temperature behavior.

        - Numeric input sets temperature attribute.
        - NaN input resets temperature to default 25.0.
        - Non-numeric input raises TypeError.
        """
        self.form.set_temperature(37)
        self.assertEqual(self.form.temperature, 37.0)

        self.form.set_temperature(float("nan"))
        self.assertEqual(self.form.temperature, 25.0)

        with self.assertRaises(TypeError):
            self.form.set_temperature("hot")

    def test_viscosity_profile(self):
        """
        Test set_viscosity_profile enforces type check and correctly assigns profile.

        - Passing non-ViscosityProfile raises TypeError.
        - Valid profile is stored and returned by .viscosity_profile.
        """
        with self.assertRaises(TypeError):
            self.form.set_viscosity_profile("not-profile")
        self.form.set_viscosity_profile(self.vp)
        self.assertIs(self.form.viscosity_profile, self.vp)

    def test_to_dict_and_repr_and_eq(self):
        """
        Test to_dict, __repr__, and __eq__ methods for Formulation.

        - Build a full formulation with all five components, temperature, and viscosity_profile.
        - Verify to_dict returns dictionary with keys: 'id', component fields, 'temperature', 'viscosity_profile'.
        - __repr__ contains "Formulation(".
        - __eq__ compares based on to_dict excluding id; differing id yields inequality.
        - Comparing to non-Formulation returns False.
        """
        # build full formulation
        self.form.set_protein(self.prot, 10, "u")
        self.form.set_buffer(self.buf, 1, "u")
        self.form.set_stabilizer(self.stab, 0, "u")
        self.form.set_surfactant(self.surf, 0, "u")
        self.form.set_salt(self.salt, 5, "u")
        self.form.set_temperature(20)
        self.form.set_viscosity_profile(self.vp)

        d = self.form.to_dict()
        # id should be None
        self.assertIsNone(d["id"])
        # each component should produce a dict with 'concentration' key
        for key in ("protein", "buffer", "stabilizer", "surfactant", "salt"):
            self.assertIsInstance(d[key], dict)
            self.assertIn("concentration", d[key])
        self.assertEqual(d["temperature"], 20.0)
        self.assertIsInstance(d["viscosity_profile"], dict)

        r = repr(self.form)
        self.assertIn("Formulation(", r)

        # eq comparison
        f2 = Formulation()
        for setter in (
            lambda f: f.set_protein(self.prot, 10, "u"),
            lambda f: f.set_buffer(self.buf, 1, "u"),
            lambda f: f.set_stabilizer(self.stab, 0, "u"),
            lambda f: f.set_surfactant(self.surf, 0, "u"),
            lambda f: f.set_salt(self.salt, 5, "u"),
            lambda f: f.set_temperature(20),
            lambda f: f.set_viscosity_profile(self.vp),
        ):
            setter(f2)
        self.assertEqual(self.form, f2)

        # differ-by-id yields inequality
        f3 = Formulation(id=1)
        self.assertNotEqual(self.form, f3)
        # comparing to non-Formulation returns False
        self.assertFalse(self.form == 123)

    def test_to_dataframe(self):
        """
        Test that to_dataframe() returns a one-row DataFrame with all expected columns,
        correctly populated from the Formulation’s components, temperature, and viscosity_profile.
        """
        self.form.set_protein(self.prot, concentration=5.0, units="mg/mL")
        self.form.set_buffer(self.buf, concentration=1.0, units="mM")
        self.form.set_stabilizer(self.stab, concentration=0.1, units="w/v")
        self.form.set_surfactant(self.surf, concentration=0.01, units="w/v")
        self.form.set_salt(self.salt, concentration=0.05, units="mM")
        self.form.set_temperature(37.0)
        self.form.set_viscosity_profile(self.vp)
        df = self.form.to_dataframe()

        expected_columns = [
            "ID",
            "Protein_type", "MW", "PI_mean", "PI_range", "Protein_conc",
            "Temperature",
            "Buffer_type", "Buffer_pH", "Buffer_conc",
            "Salt_type", "Salt_conc",
            "Stabilizer_type", "Stabilizer_conc",
            "Surfactant_type", "Surfactant_conc",
            "Viscosity_100", "Viscosity_1000", "Viscosity_10000",
            "Viscosity_100000", "Viscosity_15000000"
        ]
        self.assertListEqual(list(df.columns), expected_columns)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]

        self.assertEqual(row["Protein_type"], self.prot.enc_id)
        self.assertEqual(row["MW"], self.prot.molecular_weight)
        self.assertEqual(row["PI_mean"], self.prot.pI_mean)
        self.assertEqual(row["PI_range"], self.prot.pI_range)
        self.assertEqual(row["Protein_conc"], 5.0)
        self.assertEqual(row["Buffer_type"], self.buf.enc_id)
        self.assertEqual(row["Buffer_pH"], self.buf.pH)
        self.assertEqual(row["Buffer_conc"], 1.0)
        self.assertEqual(row["Salt_type"], self.salt.enc_id)
        self.assertEqual(row["Salt_conc"], 0.05)
        self.assertEqual(row["Stabilizer_type"], self.stab.enc_id)
        self.assertEqual(row["Stabilizer_conc"], 0.1)
        self.assertEqual(row["Surfactant_type"], self.surf.enc_id)
        self.assertEqual(row["Surfactant_conc"], 0.01)
        self.assertEqual(row["Temperature"], 37.0)
        self.assertEqual(row["Viscosity_100"], 12)
        self.assertEqual(row["Viscosity_1000"], 112)


if __name__ == "__main__":
    unittest.main()
