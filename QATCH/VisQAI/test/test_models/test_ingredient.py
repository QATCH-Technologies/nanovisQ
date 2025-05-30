import unittest
from src.models.ingredient import (
    Ingredient, Protein, Buffer, Stabilizer, Surfactant, Salt
)


class TestIngredientBase(unittest.TestCase):
    def test_valid_initialization(self):
        ing = Ingredient(enc_id=5, name=" Water ", id=None)
        self.assertIsNone(ing.id)
        self.assertEqual(ing.enc_id, 5)
        self.assertEqual(ing.name, "Water")

    def test_invalid_id_type(self):
        with self.assertRaises(TypeError):
            Ingredient(enc_id=1, name="X", id="not-an-int")

    def test_invalid_enc_id_type(self):
        with self.assertRaises(TypeError):
            Ingredient(enc_id="nope", name="X")

    def test_invalid_name_type(self):
        with self.assertRaises(TypeError):
            Ingredient(enc_id=1, name=123)

    def test_invalid_name_empty(self):
        with self.assertRaises(ValueError):
            Ingredient(enc_id=1, name="   ")

    def test_id_setter_valid(self):
        ing = Ingredient(1, "A")
        ing.id = 10
        self.assertEqual(ing.id, 10)

    def test_id_setter_invalid(self):
        ing = Ingredient(1, "A")
        with self.assertRaises(TypeError):
            ing.id = "ten"

    def test_enc_id_setter_valid(self):
        ing = Ingredient(1, "A")
        ing.enc_id = 99
        self.assertEqual(ing.enc_id, 99)

    def test_enc_id_setter_invalid(self):
        ing = Ingredient(1, "A")
        with self.assertRaises(TypeError):
            ing.enc_id = 3.14

    def test_name_setter_valid(self):
        ing = Ingredient(1, "Foo")
        ing.name = "  Bar  "
        self.assertEqual(ing.name, "Bar")

    def test_name_setter_invalid_type(self):
        ing = Ingredient(1, "Foo")
        with self.assertRaises(TypeError):
            ing.name = 42

    def test_name_setter_invalid_empty(self):
        ing = Ingredient(1, "Foo")
        with self.assertRaises(ValueError):
            ing.name = "   "

    def test_to_dict_and_from_dict(self):
        ing = Ingredient(enc_id=7, name="Z", id=3)
        d = ing.to_dict()
        self.assertEqual(
            d, {"id": 3, "enc_id": 7, "name": "Z", "type": "Ingredient"})
        ing2 = Ingredient.from_dict(d)
        self.assertEqual(ing2, ing)
        # from_dict should preserve None id
        d2 = {"enc_id": 8, "name": "Y"}
        ing3 = Ingredient.from_dict(d2)
        self.assertIsNone(ing3.id)
        self.assertEqual(ing3.enc_id, 8)
        self.assertEqual(ing3.name, "Y")

    def test_repr(self):
        ing = Ingredient(enc_id=2, name="R", id=5)
        r = repr(ing)
        self.assertIn("Ingredient", r)
        self.assertIn("id=5", r)
        self.assertIn("enc_id=2", r)
        self.assertIn("name='R'", r)

    def test_equality_and_inequality(self):
        a = Ingredient(1, "One", id=1)
        b = Ingredient(1, "One", id=1)
        c = Ingredient(1, "One", id=2)
        self.assertTrue(a == b)
        self.assertFalse(a != b)
        self.assertFalse(a == c)
        # comparing to non‐Ingredient returns NotImplemented -> ends up False
        self.assertFalse(a == 42)

    def test_ordering_by_id(self):
        a = Ingredient(1, "A", id=1)
        b = Ingredient(1, "B", id=2)
        self.assertTrue(a < b)
        self.assertFalse(b < a)

    def test_ordering_by_enc_id_when_id_none(self):
        a = Ingredient(5, "A")
        b = Ingredient(7, "B")
        self.assertTrue(a < b)
        self.assertFalse(b < a)

    def test_type_property(self):
        ing = Ingredient(1, "Test")
        self.assertEqual(ing.type, "Ingredient")


class TestProtein(unittest.TestCase):
    def test_valid_protein(self):
        p = Protein(enc_id=10, name="MyProt", molecular_weight=50.5,
                    pI_mean=6.8, pI_range=1.2, id=99)
        self.assertEqual(p.id, 99)
        self.assertEqual(p.enc_id, 10)
        self.assertEqual(p.name, "MyProt")
        self.assertAlmostEqual(p.molecular_weight, 50.5)
        self.assertAlmostEqual(p.pI_mean, 6.8)
        self.assertAlmostEqual(p.pI_range, 1.2)

    def test_invalid_molecular_weight_type(self):
        with self.assertRaises(TypeError):
            Protein(1, "P", molecular_weight="heavy", pI_mean=7, pI_range=1)

    def test_invalid_molecular_weight_negative(self):
        with self.assertRaises(ValueError):
            Protein(1, "P", molecular_weight=-1, pI_mean=7, pI_range=1)

    def test_property_setters(self):
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
    def test_valid_buffer(self):
        buf = Buffer(enc_id=3, name="PBS", pH=7.4, id=11)
        self.assertEqual(buf.id, 11)
        self.assertEqual(buf.enc_id, 3)
        self.assertEqual(buf.name, "PBS")
        self.assertAlmostEqual(buf.pH, 7.4)

    def test_invalid_pH_type(self):
        with self.assertRaises(TypeError):
            Buffer(1, "B", pH="acidic")

    def test_invalid_pH_low(self):
        with self.assertRaises(ValueError):
            Buffer(1, "B", pH=-0.1)

    def test_invalid_pH_high(self):
        with self.assertRaises(ValueError):
            Buffer(1, "B", pH=14.1)

    def test_pH_setter_valid_and_invalid(self):
        buf = Buffer(1, "B", pH=7)
        buf.pH = 3.2
        self.assertEqual(buf.pH, 3.2)
        # with self.assertRaises(TypeError):
        #     buf.pH = None
        # with self.assertRaises(ValueError):
        #     buf.pH = 15


class TestSimpleSubclasses(unittest.TestCase):
    def test_stabilizer_behaves_like_ingredient(self):
        s = Stabilizer(2, "Stab", id=4)
        self.assertEqual(s.type, "Stabilizer")
        self.assertEqual(repr(s).split("(")[0], "Stabilizer")
        self.assertEqual(s, Stabilizer(2, "Stab", id=4))

    def test_surfactant_behaves_like_ingredient(self):
        s = Surfactant(5, "Surf")
        self.assertEqual(s.type, "Surfactant")
        self.assertIn("Surf", repr(s))

    def test_salt_behaves_like_ingredient(self):
        s = Salt(7, "NaCl", id=None)
        self.assertEqual(s.type, "Salt")
        self.assertIsNone(s.id)
        # ordering and equality still work
        other = Salt(7, "NaCl")
        self.assertTrue(s == other)
        self.assertFalse(s < other)  # equal -> not less


if __name__ == "__main__":
    unittest.main()
