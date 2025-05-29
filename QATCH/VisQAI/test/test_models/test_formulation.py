from src.models.formulation import (
    ViscosityProfile, Component, Formulation
)
from src.models.ingredient import (
    Ingredient, Buffer, Protein, Stabilizer, Surfactant, Salt
)
import os
import sys
import math
import unittest

# — adjust this path so "src/models" is on sys.path —
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../src")))


class TestViscosityProfile(unittest.TestCase):
    def test_init_type_checks(self):
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
        vp = ViscosityProfile([10, 1, 5], [100, 10, 50], "  cp ")
        # sorted by shear_rates
        self.assertEqual(vp.shear_rates, [1.0, 5.0, 10.0])
        self.assertEqual(vp.viscosities, [10.0, 50.0, 100.0])
        self.assertEqual(vp.units, "cp")
        self.assertFalse(vp.is_measured)

    def test_is_measured_setter(self):
        vp = ViscosityProfile([1], [1], "u")
        vp.is_measured = True
        self.assertTrue(vp.is_measured)
        vp.is_measured = False
        self.assertFalse(vp.is_measured)

    def test_get_viscosity_exact(self):
        vp = ViscosityProfile([1, 2, 3], [10, 20, 30], "u")
        self.assertEqual(vp.get_viscosity(2), 20.0)
        self.assertEqual(vp.get_viscosity(2.0), 20.0)

    def test_get_viscosity_interpolation(self):
        vp = ViscosityProfile([1, 3], [10, 30], "u")
        # mid-point
        self.assertEqual(vp.get_viscosity(2), 20.0)
        # below lowest -> linear between 1 and 3
        self.assertEqual(vp.get_viscosity(0), 10.0 + (0 - 1)*(30-10)/(3-1))
        # above highest
        self.assertEqual(vp.get_viscosity(4), 10.0 + (4 - 1)*(30-10)/(3-1))

    def test_get_viscosity_type_error(self):
        vp = ViscosityProfile([1], [1], "u")
        with self.assertRaises(TypeError):
            vp.get_viscosity("fast")

    def test_to_dict_and_repr_and_eq(self):
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
        # eq only checks data, not is_measured
        vp2 = ViscosityProfile([1, 2], [10, 20], "cP")
        self.assertEqual(vp, vp2)
        vp3 = ViscosityProfile([1, 2], [10, 21], "cP")
        self.assertNotEqual(vp, vp3)


class TestComponent(unittest.TestCase):
    def setUp(self):
        self.prot = Protein(1, "P", 10, 7, 1)

    def test_init_type_and_value(self):
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
        comp = Component(self.prot, 2, "X")
        r = repr(comp)
        self.assertIn("Component(Protein='P'", r)
        d = comp.to_dict()
        self.assertIn("type", d)
        self.assertEqual(d["concentration"], 2.0)
        self.assertEqual(d["units"], "X")


class TestFormulation(unittest.TestCase):
    def setUp(self):
        self.form = Formulation()
        self.prot = Protein(1, "BSA", 100, 10, 1)
        self.buf = Buffer(2, "PBS", 7.4)
        self.stab = Stabilizer(3, "None")
        self.surf = Surfactant(4, "None")
        self.salt = Salt(5, "NaCl")
        self.vp = ViscosityProfile([1, 10], [1, 2], "u")

    def test_init_and_id_property(self):
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
        for name in ("protein", "buffer", "stabilizer", "surfactant", "salt"):
            self.assertIsNone(getattr(self.form, name))

    def test_setting_components(self):
        self.form.set_protein(self.prot, 50, "mg")
        self.assertIsNotNone(self.form.protein)
        self.assertEqual(self.form.protein.concentration, 50.0)
        self.form.set_buffer(self.buf, 1, "L")
        self.assertEqual(self.form.buffer.units, "L")
        self.form.set_stabilizer(self.stab, 0, "M")
        self.form.set_surfactant(self.surf, 0, "%w")
        self.form.set_salt(self.salt, 5, "g")
        # confirm all five slots filled
        for comp in (self.form.protein, self.form.buffer, self.form.stabilizer,
                     self.form.surfactant, self.form.salt):
            self.assertIsNotNone(comp)

    def test_temperature(self):
        # valid numeric
        self.form.set_temperature(37)
        self.assertEqual(self.form.temperature, 37.0)
        # nan resets to 25.0
        self.form.set_temperature(float("nan"))
        self.assertEqual(self.form.temperature, 25.0)
        # string->TypeError
        with self.assertRaises(TypeError):
            self.form.set_temperature("hot")

    def test_viscosity_profile(self):
        with self.assertRaises(TypeError):
            self.form.set_viscosity_profile("not-profile")
        self.form.set_viscosity_profile(self.vp)
        self.assertIs(self.form.viscosity_profile, self.vp)

    def test_to_dict_and_repr_and_eq(self):
        # build a full formulation
        self.form.set_protein(self.prot, 10, "u")
        self.form.set_buffer(self.buf, 1, "u")
        self.form.set_stabilizer(self.stab, 0, "u")
        self.form.set_surfactant(self.surf, 0, "u")
        self.form.set_salt(self.salt, 5, "u")
        self.form.set_temperature(20)
        self.form.set_viscosity_profile(self.vp)

        d = self.form.to_dict()
        # id
        self.assertIsNone(d["id"])
        # each component present as dict
        for key in ("protein", "buffer", "stabilizer", "surfactant", "salt"):
            self.assertIsInstance(d[key], dict)
            self.assertIn("concentration", d[key])
        self.assertEqual(d["temperature"], 20.0)
        self.assertIsInstance(d["viscosity_profile"], dict)

        r = repr(self.form)
        self.assertIn("Formulation(", r)
        # equality via to_dict
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

        # differ-by-id
        f3 = Formulation(id=1)
        self.assertNotEqual(self.form, f3)
        # compare to non-Formulation
        self.assertFalse(self.form == 123)


if __name__ == "__main__":
    unittest.main()
