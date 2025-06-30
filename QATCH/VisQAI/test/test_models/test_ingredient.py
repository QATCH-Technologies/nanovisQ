"""
test_models_ingredient.py

Unit tests for the Ingredient model classes, verifying:
    - Base Ingredient validation (id, enc_id, name)
    - Protein subclass validation (molecular_weight, pI_mean, pI_range)
    - Buffer subclass validation (pH range and type)
    - Behavior of simple subclasses (Stabilizer, Surfactant, Salt) for type, repr, ordering, and equality

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-06-02

Version:
    1.0
"""

import unittest
from src.models.ingredient import (
    Ingredient, Protein, Buffer, Stabilizer, Surfactant, Salt
)


class TestIngredientBase(unittest.TestCase):
    """Unit tests for the base Ingredient class."""

    def test_valid_initialization(self):
        """
        Test that an Ingredient can be initialized with valid enc_id and name.

        - Leading/trailing whitespace in name is trimmed.
        - id defaults to None when not provided.
        """
        ing = Ingredient(enc_id=5, name=" Water ")
        self.assertIsNone(ing.id)
        self.assertEqual(ing.enc_id, 5)
        self.assertEqual(ing.name, "Water")

    def test_invalid_id_type(self):
        """
        Test that passing a non-integer to the id parameter raises TypeError.
        """
        with self.assertRaises(TypeError):
            Ingredient(enc_id=1, name="X", id="not-an-int")

    def test_invalid_enc_id_type(self):
        """
        Test that passing a non-integer to enc_id raises TypeError.
        """
        with self.assertRaises(TypeError):
            Ingredient(enc_id="nope", name="X")

    def test_invalid_name_type(self):
        """
        Test that passing a non-string name raises TypeError.
        """
        with self.assertRaises(TypeError):
            Ingredient(enc_id=1, name=123)

    def test_invalid_name_empty(self):
        """
        Test that passing an empty or whitespace-only name raises ValueError.
        """
        with self.assertRaises(ValueError):
            Ingredient(enc_id=1, name="   ")

    def test_id_setter_valid(self):
        """
        Test that setting id to a valid integer updates the attribute.
        """
        ing = Ingredient(1, "A")
        ing.id = 10
        self.assertEqual(ing.id, 10)

    def test_id_setter_invalid(self):
        """
        Test that setting id to a non-integer raises TypeError.
        """
        ing = Ingredient(1, "A")
        with self.assertRaises(TypeError):
            ing.id = "ten"

    def test_enc_id_setter_valid(self):
        """
        Test that setting enc_id to a valid integer updates the attribute.
        """
        ing = Ingredient(1, "A")
        ing.enc_id = 99
        self.assertEqual(ing.enc_id, 99)

    def test_enc_id_setter_invalid(self):
        """
        Test that setting enc_id to a non-integer raises TypeError.
        """
        ing = Ingredient(1, "A")
        with self.assertRaises(TypeError):
            ing.enc_id = 3.14

    def test_name_setter_valid(self):
        """
        Test that setting name to a non-empty string (with whitespace) is trimmed correctly.
        """
        ing = Ingredient(1, "Foo")
        ing.name = "  Bar  "
        self.assertEqual(ing.name, "Bar")

    def test_name_setter_invalid_type(self):
        """
        Test that setting name to a non-string raises TypeError.
        """
        ing = Ingredient(1, "Foo")
        with self.assertRaises(TypeError):
            ing.name = 42

    def test_name_setter_invalid_empty(self):
        """
        Test that setting name to an empty or whitespace-only string raises ValueError.
        """
        ing = Ingredient(1, "Foo")
        with self.assertRaises(ValueError):
            ing.name = "   "

    def test_to_dict_and_from_dict(self):
        """
        Test to_dict and from_dict functionality.

        - Verify to_dict returns a dict with enc_id, name, type, and user flag.
        - Verify from_dict reconstructs an Ingredient with matching fields.
        - Check that from_dict preserves a None id when id not provided.
        """
        ing = Ingredient(enc_id=7, name="Z", id=3)
        d = ing.to_dict()
        self.assertEqual(
            d, {"enc_id": 7, "name": "Z", "type": "Ingredient", "user?": True}
        )
        ing2 = Ingredient.from_dict(d)
        ing2.id = 3
        self.assertEqual(ing2, ing)

        # from_dict should preserve None id if id not in dict
        d2 = {"enc_id": 8, "name": "Y"}
        ing3 = Ingredient.from_dict(d2)
        self.assertIsNone(ing3.id)
        self.assertEqual(ing3.enc_id, 8)
        self.assertEqual(ing3.name, "Y")

    def test_repr(self):
        """
        Test that __repr__ contains class name and key attributes.
        """
        ing = Ingredient(enc_id=2, name="R", id=5)
        r = repr(ing)
        self.assertIn("Ingredient", r)
        self.assertIn("id=5", r)
        self.assertIn("enc_id=2", r)
        self.assertIn("name='R'", r)

    def test_equality_and_inequality(self):
        """
        Test __eq__ and __ne__ behavior.

        - Two Ingredients with same id, enc_id, and name are equal.
        - Different id yields inequality.
        - Comparing to non-Ingredient returns False.
        """
        a = Ingredient(1, "One", id=1)
        b = Ingredient(1, "One", id=1)
        c = Ingredient(1, "One", id=2)
        self.assertTrue(a == b)
        self.assertFalse(a != b)
        self.assertFalse(a == c)
        # comparing to non‐Ingredient returns False
        self.assertFalse(a == 42)

    def test_ordering_by_id(self):
        """
        Test __lt__ orders by id when both ids are not None.
        """
        a = Ingredient(1, "A", id=1)
        b = Ingredient(1, "B", id=2)
        self.assertTrue(a < b)
        self.assertFalse(b < a)

    def test_ordering_by_enc_id_when_id_none(self):
        """
        Test __lt__ orders by enc_id when id is None for both Ingredients.
        """
        a = Ingredient(5, "A")
        b = Ingredient(7, "B")
        self.assertTrue(a < b)
        self.assertFalse(b < a)

    def test_type_property(self):
        """
        Test that the type property returns the class name.
        """
        ing = Ingredient(1, "Test")
        self.assertEqual(ing.type, "Ingredient")


class TestProtein(unittest.TestCase):
    """Unit tests for the Protein subclass of Ingredient."""

    def test_valid_protein(self):
        """
        Test that Protein initializes correctly with valid molecular_weight, pI_mean, and pI_range.

        - Verify id, enc_id, name, and numeric properties match inputs.
        """
        p = Protein(enc_id=10, name="MyProt", molecular_weight=50.5,
                    pI_mean=6.8, pI_range=1.2, id=99)
        self.assertEqual(p.id, 99)
        self.assertEqual(p.enc_id, 10)
        self.assertEqual(p.name, "MyProt")
        self.assertAlmostEqual(p.molecular_weight, 50.5)
        self.assertAlmostEqual(p.pI_mean, 6.8)
        self.assertAlmostEqual(p.pI_range, 1.2)

    def test_invalid_molecular_weight_type(self):
        """
        Test that passing a non-numeric molecular_weight raises TypeError.
        """
        with self.assertRaises(TypeError):
            Protein(1, "P", molecular_weight="heavy", pI_mean=7, pI_range=1)

    def test_invalid_molecular_weight_negative(self):
        """
        Test that passing a negative molecular_weight raises ValueError.
        """
        with self.assertRaises(ValueError):
            Protein(1, "P", molecular_weight=-1, pI_mean=7, pI_range=1)

    def test_property_setters(self):
        """
        Test setters for molecular_weight, pI_mean, and pI_range.

        - Setting valid numeric values updates attributes.
        - Setting non-numeric or out-of-range values raises appropriate exceptions.
        """
        p = Protein(1, "P", 10, 7, 1)
        p.molecular_weight = 20
        p.pI_mean = 8.1
        p.pI_range = 0.5
        self.assertEqual(p.molecular_weight, 20.0)
        self.assertEqual(p.pI_mean, 8.1)
        self.assertEqual(p.pI_range, 0.5)

        with self.assertRaises(TypeError):
            p.molecular_weight = "big"
        with self.assertRaises(ValueError):
            p.pI_range = -0.1


class TestBuffer(unittest.TestCase):
    """Unit tests for the Buffer subclass of Ingredient."""

    def test_valid_buffer(self):
        """
        Test that Buffer initializes correctly with a valid pH value.

        - Verify id, enc_id, name, and pH match inputs.
        """
        buf = Buffer(enc_id=3, name="PBS", pH=7.4, id=11)
        self.assertEqual(buf.id, 11)
        self.assertEqual(buf.enc_id, 3)
        self.assertEqual(buf.name, "PBS")
        self.assertAlmostEqual(buf.pH, 7.4)

    def test_invalid_pH_type(self):
        """
        Test that passing a non-numeric pH raises TypeError.
        """
        with self.assertRaises(TypeError):
            Buffer(1, "B", pH="acidic")

    def test_invalid_pH_low(self):
        """
        Test that passing a pH below 0.0 raises ValueError.
        """
        with self.assertRaises(ValueError):
            Buffer(1, "B", pH=-0.1)

    def test_invalid_pH_high(self):
        """
        Test that passing a pH above 14.0 raises ValueError.
        """
        with self.assertRaises(ValueError):
            Buffer(1, "B", pH=14.1)

    def test_pH_setter_valid_and_invalid(self):
        """
        Test the pH setter for valid and invalid values.

        - Setting to a valid numeric pH updates attribute.
        - Setting to a value outside [0.0, 14.0] raises ValueError.
        """
        buf = Buffer(1, "B", pH=7)
        buf.pH = 3.2
        self.assertEqual(buf.pH, 3.2)

        with self.assertRaises(ValueError):
            buf.pH = 15


class TestSimpleSubclasses(unittest.TestCase):
    """Unit tests for Stabilizer, Surfactant, and Salt subclasses of Ingredient."""

    def test_stabilizer_behaves_like_ingredient(self):
        """
        Test that Stabilizer inherits from Ingredient correctly.

        - type property returns "Stabilizer".
        - repr contains class name.
        - Equality behaves as base class.
        """
        s = Stabilizer(2, "Stab", id=4)
        self.assertEqual(s.type, "Stabilizer")
        self.assertEqual(repr(s).split("(")[0], "Stabilizer")
        self.assertEqual(s, Stabilizer(2, "Stab", id=4))

    def test_surfactant_behaves_like_ingredient(self):
        """
        Test that Surfactant inherits from Ingredient correctly.

        - type property returns "Surfactant".
        - repr contains the name.
        """
        s = Surfactant(5, "Surf")
        self.assertEqual(s.type, "Surfactant")
        self.assertIn("Surf", repr(s))

    def test_salt_behaves_like_ingredient(self):
        """
        Test that Salt inherits from Ingredient correctly.

        - type property returns "Salt".
        - id defaults to None when not provided.
        - Equality and ordering by enc_id behave as expected.
        """
        s = Salt(7, "NaCl", id=None)
        self.assertEqual(s.type, "Salt")
        self.assertIsNone(s.id)
        other = Salt(7, "NaCl")
        self.assertTrue(s == other)
        # equal -> not less-than
        self.assertFalse(s < other)


if __name__ == "__main__":
    unittest.main()
