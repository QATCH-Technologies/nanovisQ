"""
test_ingredient_controller.py

Integration tests for the IngredientController, verifying CRUD operations on:
    - Adding and fetching each ingredient subclass (Protein, Buffer, Salt, Surfactant, Stabilizer)
    - Auto-assignment of developer and user enc_id values
    - Name trimming and duplicate name handling
    - Deleting by ID, name, and deleting all for each type, including error conditions
    - Updating existing records while preserving enc_id and is_user, including identical-object short-circuit
    - Dispatch methods that should raise on unsupported types

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-06-02

Version:
    1.0
"""

import unittest
from pathlib import Path

from src.models.ingredient import Protein, Buffer, Stabilizer, Surfactant, Salt
from src.db.db import Database
from src.controller.ingredient_controller import IngredientController


class TestIngredientController(unittest.TestCase):
    """Integration tests for IngredientController CRUD functionality and enc_id logic."""

    @classmethod
    def setUpClass(cls):
        """Create the test_assets directory and define the test database file path."""
        cls.assets_dir = Path(__file__).parent / "test_assets"
        cls.assets_dir.mkdir(parents=True, exist_ok=True)
        cls.db_file = cls.assets_dir / "test_app.db"

    def setUp(self):
        """Initialize a fresh Database and IngredientController before each test."""
        if self.db_file.exists():
            self.db_file.unlink()
        self.db = Database(self.db_file)
        self.ctrl = IngredientController(self.db)

    def tearDown(self):
        """Close the database connection and remove the test database file after each test."""
        self.db.conn.close()
        if self.db_file.exists():
            self.db_file.unlink()

    def _roundtrip(self, ing):
        """
        Helper method to add an ingredient and fetch it back by name.

        Args:
            ing (Ingredient): The ingredient instance to add and fetch.

        Returns:
            Ingredient: The fetched ingredient instance from the database.

        Raises:
            AssertionError: If the fetched ingredient is None.
        """
        self.ctrl.add(ing)
        fetched = self.ctrl.get_by_name(ing.name, ing)
        self.assertIsNotNone(fetched, f"Failed to fetch back {ing}")
        return fetched

    def _fetch_row(self, name: str):
        """
        Helper method to query the raw ingredient table for a given name.

        Args:
            name (str): The ingredient name to query.

        Returns:
            tuple or None: The database row for the ingredient, or None if not found.
        """
        return self.db.conn.execute(
            "SELECT id, name, type, enc_id, is_user FROM ingredient WHERE name = ?", (
                name,)
        ).fetchone()

    def test_add_and_fetch_protein(self):
        """
        Test that adding a Protein persists correctly and can be retrieved by ID and name.

        Verifies:
            - The raw database row has correct fields (name, type, enc_id, is_user).
            - get_protein_by_id returns a Protein with matching attributes.
            - get_protein_by_name returns the same Protein.
            - Adding a duplicate name does not create a second entry.
        """
        p = Protein(enc_id=1, name="ProtA", molecular_weight=50.0,
                    pI_mean=6.5, pI_range=0.2)
        p.is_user = False
        self.ctrl.add_protein(p)

        row = self._fetch_row("ProtA")
        self.assertIsNotNone(row, "Protein row not found in DB")
        db_id, db_name, db_type, db_enc_id, db_is_user = row
        self.assertEqual(db_name, "ProtA")
        self.assertEqual(db_type, "Protein")
        self.assertEqual(db_enc_id, 1)
        self.assertFalse(db_is_user)

        fetched_by_id = self.ctrl.get_protein_by_id(db_id)
        self.assertIsInstance(fetched_by_id, Protein)
        self.assertEqual(fetched_by_id.id, db_id)
        self.assertEqual(fetched_by_id.name, "ProtA")
        self.assertAlmostEqual(fetched_by_id.molecular_weight, 50.0)

        fetched_by_name = self.ctrl.get_protein_by_name("ProtA")
        self.assertEqual(fetched_by_name.id, db_id)
        self.assertEqual(fetched_by_name.name, "ProtA")

        dup = Protein(enc_id=-1, name="ProtA",
                      molecular_weight=60.0, pI_mean=7.0, pI_range=0.1)
        dup.is_user = False
        ret = self.ctrl.add_protein(dup)
        self.assertEqual(dup.name, ret.name)
        self.assertEqual(len(self.ctrl.get_all_proteins()), 1)

    def test_add_and_get_buffer(self):
        """
        Test that adding a Buffer persists correctly and can be fetched, then deleted.

        Verifies:
            - _roundtrip successfully adds and returns a Buffer.
            - delete_buffer_by_id removes the buffer, so get_buffer_by_name returns None.
        """
        b = Buffer(enc_id=2, name="BufX", pH=7.4)
        b.is_user = False
        fetched = self._roundtrip(b)
        self.assertIsInstance(fetched, Buffer)
        self.assertEqual(fetched.pH, 7.4)

        self.ctrl.delete_buffer_by_id(fetched.id)
        self.assertIsNone(self.ctrl.get_buffer_by_name("BufX"))

    def test_add_and_get_salt(self):
        """
        Test that adding a Salt persists correctly and can be fetched, then deleted by name.

        Verifies:
            - _roundtrip successfully adds and returns a Salt.
            - delete_salt_by_name removes the salt, so get_salt_by_name returns None.
        """
        s = Salt(enc_id=3, name="SaltY")
        s.is_user = False
        fetched = self._roundtrip(s)
        self.assertIsInstance(fetched, Salt)

        self.ctrl.delete_salt_by_name("SaltY")
        self.assertIsNone(self.ctrl.get_salt_by_name("SaltY"))

    def test_add_and_get_surfactant(self):
        """
        Test that adding a Surfactant persists correctly and can be fetched, then all surfactants deleted.

        Verifies:
            - _roundtrip successfully adds and returns a Surfactant.
            - delete_all_surfactants removes all entries, so get_all_surfactants returns an empty list.
        """
        sf = Surfactant(enc_id=4, name="SurfZ")
        sf.is_user = False
        fetched = self._roundtrip(sf)
        self.assertIsInstance(fetched, Surfactant)

        self.ctrl.delete_all_surfactants()
        self.assertEqual(self.ctrl.get_all_surfactants(), [])

    def test_add_and_get_stabilizer(self):
        """
        Test that adding a Stabilizer persists correctly and can be fetched, then deleted by ID.

        Verifies:
            - _roundtrip successfully adds and returns a Stabilizer.
            - delete_stabilizer_by_id removes the stabilizer, so get_stabilizer_by_name returns None.
        """
        st = Stabilizer(enc_id=5, name="StabW")
        st.is_user = False
        fetched = self._roundtrip(st)
        self.assertIsInstance(fetched, Stabilizer)

        self.ctrl.delete_stabilizer_by_id(fetched.id)
        self.assertIsNone(self.ctrl.get_stabilizer_by_name("StabW"))

    def test_get_all_ingredients_and_delete_all(self):
        """
        Test that multiple ingredient types can be added, all retrieved, and then deleted.

        Verifies:
            - get_all_ingredients returns all five types.
            - delete_all_ingredients clears the database.
        """
        types = [
            Protein(enc_id=10, name="P1", molecular_weight=20,
                    pI_mean=5, pI_range=0.1),
            Buffer(enc_id=11, name="B1", pH=7.1),
            Salt(enc_id=12, name="S1"),
            Surfactant(enc_id=13, name="SF1"),
            Stabilizer(enc_id=14, name="ST1"),
        ]
        for ing in types:
            ing.is_user = False
            self.ctrl.add(ing)

        all_ings = self.ctrl.get_all_ingredients()
        self.assertEqual({type(i) for i in all_ings},
                         set(type(i) for i in types))

        self.ctrl.delete_all_ingredients()
        self.assertEqual(self.ctrl.get_all_ingredients(), [])

    def test_update_protein(self):
        """
        Test updating a Protein record preserves enc_id and is_user, and changes name.

        Verifies:
            - Original protein is fetched, then updated.
            - Old name no longer exists; new name returns the updated object with preserved enc_id/is_user.
        """
        original = Protein(enc_id=20, name="Orig",
                           molecular_weight=30, pI_mean=6, pI_range=0.2)
        original.is_user = False
        self.ctrl.add_protein(original)
        fetched = self.ctrl.get_protein_by_name("Orig")
        self.assertIsNotNone(fetched)

        updated = Protein(enc_id=-1, name="Updated",
                          molecular_weight=35, pI_mean=6.3, pI_range=0.3)
        updated.is_user = True  # should be ignored, original's is_user preserved
        self.ctrl.update_protein(fetched.id, updated)
        self.assertIsNone(self.ctrl.get_protein_by_name("Orig"))
        up = self.ctrl.get_protein_by_name("Updated")
        self.assertIsNotNone(up)
        self.assertEqual(up.name, "Updated")
        self.assertEqual(up.enc_id, fetched.enc_id)
        self.assertFalse(up.is_user)

    def test_dispatch_methods_raise_on_bad_type(self):
        """
        Test that generic dispatch methods raise ValueError when given an unsupported type.

        Verifies:
            - get_by_id, get_by_name, get_by_type, delete_by_id, delete_by_name, delete_by_type,
              add, update all raise when passed an object with type attribute not matching any subclass.
        """
        class Fake:
            type = "Nope"
            name = "x"

        fake = Fake()
        with self.assertRaises(ValueError):
            self.ctrl.get_by_id(1, fake)
        with self.assertRaises(ValueError):
            self.ctrl.get_by_name("x", fake)
        with self.assertRaises(ValueError):
            self.ctrl.get_by_type(fake)
        with self.assertRaises(ValueError):
            self.ctrl.delete_by_id(1, fake)
        with self.assertRaises(ValueError):
            self.ctrl.delete_by_name("x", fake)
        with self.assertRaises(ValueError):
            self.ctrl.delete_by_type(fake)
        with self.assertRaises(ValueError):
            self.ctrl.add(fake)
        with self.assertRaises(ValueError):
            self.ctrl.update(1, fake)

    def test_get_all_empty_initially(self):
        """
        Test that get_all_ingredients initially returns an empty list when DB is empty.
        """
        self.assertEqual(self.ctrl.get_all_ingredients(), [])

    def test_fetch_nonexistent_returns_none(self):
        """
        Test that fetching by ID for each subclass returns None for non-existent IDs.
        """
        self.assertIsNone(self.ctrl.get_protein_by_id(999))
        self.assertIsNone(self.ctrl.get_buffer_by_id(999))
        self.assertIsNone(self.ctrl.get_salt_by_id(999))
        self.assertIsNone(self.ctrl.get_surfactant_by_id(999))
        self.assertIsNone(self.ctrl.get_stabilizer_by_id(999))

    def test_delete_nonexistent_raises(self):
        """
        Test that attempting to delete non-existent IDs for each subclass raises ValueError.
        """
        with self.assertRaises(ValueError):
            self.ctrl.delete_protein_by_id(999)
        with self.assertRaises(ValueError):
            self.ctrl.delete_buffer_by_id(999)
        with self.assertRaises(ValueError):
            self.ctrl.delete_salt_by_id(999)
        with self.assertRaises(ValueError):
            self.ctrl.delete_surfactant_by_id(999)
        with self.assertRaises(ValueError):
            self.ctrl.delete_stabilizer_by_id(999)

    def test_cross_type_same_name_allowed(self):
        """
        Test that different subclasses can share the same name without conflict.

        Verifies:
            - Adding a Protein and a Buffer both named "Common" persists both.
            - get_all_ingredients returns two entries with name "Common".
        """
        p = Protein(enc_id=1, name="Common", molecular_weight=10.0,
                    pI_mean=5.0, pI_range=0.1)
        p.is_user = False
        self.ctrl.add(p)
        b = Buffer(enc_id=2, name="Common", pH=7.0)
        b.is_user = False
        self.ctrl.add(b)

        self.assertIsInstance(self.ctrl.get_protein_by_name("Common"), Protein)
        self.assertIsInstance(self.ctrl.get_buffer_by_name("Common"), Buffer)

        commons = [ing for ing in self.ctrl.get_all_ingredients()
                   if ing.name == "Common"]
        self.assertEqual(len(commons), 2)

    def test_trimmed_name(self):
        """
        Test that whitespace around the name is trimmed on insert and fetch.

        Verifies:
            - Adding a Buffer with name "  TrimTest  " is stored as "TrimTest".
        """
        b = Buffer(enc_id=3, name="  TrimTest  ", pH=7.2)
        b.is_user = False
        self.ctrl.add(b)
        fetched = self.ctrl.get_buffer_by_name("TrimTest")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "TrimTest")

    def test_duplicate_buffer_name_raises(self):
        """
        Test that adding two Buffers with the same name does not create a duplicate.

        Verifies:
            - Second insert returns the existing object and get_all_buffers has length 1.
        """
        b1 = Buffer(enc_id=-1, name="BufDup", pH=6.5)
        b1.is_user = False
        b2 = Buffer(enc_id=-1, name="BufDup", pH=6.8)
        b2.is_user = False
        self.ctrl.add(b1)
        self.ctrl.add(b2)
        self.assertEqual(b1.name, b2.name)
        self.assertEqual(len(self.ctrl.get_all_buffers()), 1)

    def test_update_protein_nonexistent(self):
        """
        Test that updating a non-existent Protein raises ValueError.
        """
        updated = Protein(enc_id=6, name="NoExist",
                          molecular_weight=20.0, pI_mean=6.0, pI_range=0.2)
        updated.is_user = False
        with self.assertRaises(ValueError):
            self.ctrl.update_protein(999, updated)

    def test_enc_id_auto_assign_dev(self):
        """
        Test that developer-created ingredients receive sequential enc_id starting at 1.

        Verifies:
            - First developer Protein -> enc_id = 1
            - Second developer Protein -> enc_id = 2
        """
        p1 = Protein(enc_id=-1, name="DevProt1",
                     molecular_weight=10.0, pI_mean=5.0, pI_range=0.1)
        p1.is_user = False
        self.ctrl.add_protein(p1)
        fetched1 = self.ctrl.get_protein_by_name("DevProt1")
        self.assertEqual(fetched1.enc_id, 1)

        p2 = Protein(enc_id=-1, name="DevProt2",
                     molecular_weight=12.0, pI_mean=5.5, pI_range=0.2)
        p2.is_user = False
        self.ctrl.add_protein(p2)
        fetched2 = self.ctrl.get_protein_by_name("DevProt2")
        self.assertEqual(fetched2.enc_id, 2)

    def test_enc_id_auto_assign_user(self):
        """
        Test that user-created ingredients receive enc_id starting at USER_START_ID.

        Verifies:
            - First user Buffer -> enc_id = USER_START_ID
            - Second user Buffer -> enc_id = USER_START_ID + 1
        """
        b1 = Buffer(enc_id=-1, name="UserBuf1", pH=7.0)
        b1.is_user = True
        self.ctrl.add_buffer(b1)
        fetched1 = self.ctrl.get_buffer_by_name("UserBuf1")
        self.assertEqual(fetched1.enc_id, IngredientController.USER_START_ID)

        b2 = Buffer(enc_id=-1, name="UserBuf2", pH=7.2)
        b2.is_user = True
        self.ctrl.add_buffer(b2)
        fetched2 = self.ctrl.get_buffer_by_name("UserBuf2")
        self.assertEqual(
            fetched2.enc_id, IngredientController.USER_START_ID + 1)

    def test_no_overlap_dev_and_user(self):
        """
        Test that developer and user enc_id ranges do not overlap across subclasses.

        Verifies:
            - Developer Salt -> enc_id = 1
            - User Surfactant -> enc_id = USER_START_ID
            - Developer Stabilizer -> enc_id = 1
            - User Stabilizer -> enc_id = USER_START_ID
        """
        s_dev = Salt(enc_id=-1, name="DevSalt")
        s_dev.is_user = False
        self.ctrl.add_salt(s_dev)
        fetched_dev = self.ctrl.get_salt_by_name("DevSalt")
        self.assertEqual(fetched_dev.enc_id, 1)

        sf_user = Surfactant(enc_id=-1, name="UserSurf")
        sf_user.is_user = True
        self.ctrl.add_surfactant(sf_user)
        fetched_user = self.ctrl.get_surfactant_by_name("UserSurf")
        self.assertEqual(fetched_user.enc_id,
                         IngredientController.USER_START_ID)

        stab_dev = Stabilizer(enc_id=-1, name="DevStab")
        stab_dev.is_user = False
        self.ctrl.add_stabilizer(stab_dev)
        fetched_stab = self.ctrl.get_stabilizer_by_name("DevStab")
        self.assertEqual(fetched_stab.enc_id, 1)

        stab_user = Stabilizer(enc_id=-1, name="UserStab")
        stab_user.is_user = True
        self.ctrl.add_stabilizer(stab_user)
        fetched_stab_user = self.ctrl.get_stabilizer_by_name("UserStab")
        self.assertEqual(fetched_stab_user.enc_id,
                         IngredientController.USER_START_ID)

    def test_dev_id_exhaustion_raises(self):
        """
        Test that when DEV_MAX_ID is exhausted for a subclass, adding another developer ingredient raises RuntimeError.
        """
        original_dev_max = IngredientController.DEV_MAX_ID
        IngredientController.DEV_MAX_ID = 2

        try:
            p1 = Protein(enc_id=-1, name="DExhaust1",
                         molecular_weight=10.0, pI_mean=5.0, pI_range=0.1)
            p1.is_user = False
            self.ctrl.add_protein(p1)
            fetched1 = self.ctrl.get_protein_by_name("DExhaust1")
            self.assertEqual(fetched1.enc_id, 1)

            p2 = Protein(enc_id=-1, name="DExhaust2",
                         molecular_weight=11.0, pI_mean=5.1, pI_range=0.2)
            p2.is_user = False
            self.ctrl.add_protein(p2)
            fetched2 = self.ctrl.get_protein_by_name("DExhaust2")
            self.assertEqual(fetched2.enc_id, 2)

            p3 = Protein(enc_id=-1, name="DExhaust3",
                         molecular_weight=12.0, pI_mean=5.2, pI_range=0.3)
            p3.is_user = False
            with self.assertRaises(RuntimeError):
                self.ctrl.add_protein(p3)
        finally:
            IngredientController.DEV_MAX_ID = original_dev_max

    def _make_protein(self, name="prot1", is_user=True):
        """
        Helper to create a Protein with default values.

        Args:
            name (str): Name of the protein.
            is_user (bool): Whether to assign user-created enc_id range.

        Returns:
            Protein: A new Protein instance with those attributes.
        """
        p = Protein(enc_id=-1, name=name, molecular_weight=50.0,
                    pI_mean=6.5, pI_range=1.0)
        p.is_user = is_user
        return p

    def _make_buffer(self, name="buff1", is_user=True):
        """
        Helper to create a Buffer with default values.

        Args:
            name (str): Name of the buffer.
            is_user (bool): Whether to assign user-created enc_id range.

        Returns:
            Buffer: A new Buffer instance with those attributes.
        """
        b = Buffer(enc_id=-1, name=name, pH=7.4)
        b.is_user = is_user
        return b

    def _make_salt(self, name="salt1", is_user=True):
        """
        Helper to create a Salt with default values.

        Args:
            name (str): Name of the salt.
            is_user (bool): Whether to assign user-created enc_id range.

        Returns:
            Salt: A new Salt instance with those attributes.
        """
        s = Salt(enc_id=-1, name=name)
        s.is_user = is_user
        return s

    def _make_stabilizer(self, name="stab1", is_user=True):
        """
        Helper to create a Stabilizer with default values.

        Args:
            name (str): Name of the stabilizer.
            is_user (bool): Whether to assign user-created enc_id range.

        Returns:
            Stabilizer: A new Stabilizer instance with those attributes.
        """
        s = Stabilizer(enc_id=-1, name=name)
        s.is_user = is_user
        return s

    def _make_surfactant(self, name="surf1", is_user=True):
        """
        Helper to create a Surfactant with default values.

        Args:
            name (str): Name of the surfactant.
            is_user (bool): Whether to assign user-created enc_id range.

        Returns:
            Surfactant: A new Surfactant instance with those attributes.
        """
        sf = Surfactant(enc_id=-1, name=name)
        sf.is_user = is_user
        return sf

    def test_delete_protein_by_id_and_name(self):
        """
        Test that deleting a Protein by ID removes it, then deleting by name after re-adding also removes it.

        Verifies:
            - delete_protein_by_id removes the record, so get_protein_by_id returns None.
            - delete_protein_by_name removes by name, so get_protein_by_name returns None.
        """
        p = self._make_protein(name="P_A")
        added = self.ctrl.add(p)
        pid = added.id

        self.ctrl.delete_protein_by_id(pid)
        self.assertIsNone(self.ctrl.get_protein_by_id(pid))

        p2 = self._make_protein(name="P_B")
        added2 = self.ctrl.add(p2)
        self.ctrl.delete_protein_by_name("P_B")
        self.assertIsNone(self.ctrl.get_protein_by_name("P_B"))

    def test_delete_protein_by_id_errors_if_not_exist(self):
        """
        Test that deleting a non-existent Protein by ID raises ValueError.
        """
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_protein_by_id(9999)
        self.assertIn("Protein with id 9999 does not exist",
                      str(ctx.exception))

    def test_delete_protein_by_name_errors_if_not_exist(self):
        """
        Test that deleting a non-existent Protein by name raises ValueError.
        """
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_protein_by_name("NON_EXIST")
        self.assertIn(
            "Protein with name 'NON_EXIST' does not exist", str(ctx.exception))

    def test_delete_all_proteins_empty_errors(self):
        """
        Test that delete_all_proteins on an empty database raises ValueError.
        """
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_all_proteins()
        self.assertIn("No items of type 'Protein' found", str(ctx.exception))

    def test_delete_all_proteins_success(self):
        """
        Test that delete_all_proteins removes all Protein entries when present.
        """
        p1 = self.ctrl.add(self._make_protein(name="Px1"))
        p2 = self.ctrl.add(self._make_protein(name="Px2"))
        all_prots = self.ctrl.get_all_proteins()
        self.assertEqual({p.name for p in all_prots}, {"Px1", "Px2"})
        self.ctrl.delete_all_proteins()
        self.assertEqual(len(self.ctrl.get_all_proteins()), 0)

    def test_delete_buffer_by_id_and_name(self):
        """
        Test deleting a Buffer by ID and by name, ensuring removal in both cases.
        """
        b = self._make_buffer(name="B_A")
        added = self.ctrl.add(b)
        bid = added.id
        self.ctrl.delete_buffer_by_id(bid)
        self.assertIsNone(self.ctrl.get_buffer_by_id(bid))

        b2 = self._make_buffer(name="B_B")
        added2 = self.ctrl.add(b2)
        self.ctrl.delete_buffer_by_name("B_B")
        self.assertIsNone(self.ctrl.get_buffer_by_name("B_B"))

    def test_delete_buffer_not_exist_errors(self):
        """
        Test that deleting non-existent Buffer by ID or name raises ValueError.
        """
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_buffer_by_id(12345)
        self.assertIn("Buffer with id 12345 does not exist",
                      str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            self.ctrl.delete_buffer_by_name("NO_BUFF")
        self.assertIn("Buffer with name 'NO_BUFF' does not exist",
                      str(ctx2.exception))

    def test_delete_all_buffers_empty_errors(self):
        """
        Test that delete_all_buffers on an empty database raises ValueError.
        """
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_all_buffers()
        self.assertIn("No items of type 'Buffer' found", str(ctx.exception))

    def test_delete_all_buffers_success(self):
        """
        Test that delete_all_buffers removes all Buffer entries when present.
        """
        b1 = self.ctrl.add(self._make_buffer(name="B1"))
        b2 = self.ctrl.add(self._make_buffer(name="B2"))
        self.assertEqual(len(self.ctrl.get_all_buffers()), 2)
        self.ctrl.delete_all_buffers()
        self.assertEqual(len(self.ctrl.get_all_buffers()), 0)

    def test_delete_salt_by_id_and_name(self):
        """
        Test deleting a Salt by ID and by name, ensuring removal in both cases.
        """
        s = self._make_salt(name="S_A")
        added = self.ctrl.add(s)
        sid = added.id
        self.ctrl.delete_salt_by_id(sid)
        self.assertIsNone(self.ctrl.get_salt_by_id(sid))

        s2 = self._make_salt(name="S_B")
        added2 = self.ctrl.add(s2)
        self.ctrl.delete_salt_by_name("S_B")
        self.assertIsNone(self.ctrl.get_salt_by_name("S_B"))

    def test_delete_salt_not_exist_errors(self):
        """
        Test that deleting non-existent Salt by ID or name raises ValueError.
        """
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_salt_by_id(2222)
        self.assertIn("Salt with id 2222 does not exist", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            self.ctrl.delete_salt_by_name("NO_SALT")
        self.assertIn("Salt with name 'NO_SALT' does not exist",
                      str(ctx2.exception))

    def test_delete_all_salts_empty_errors(self):
        """
        Test that delete_all_salts on an empty database raises ValueError.
        """
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_all_salts()
        self.assertIn("No items of type 'Salt' found", str(ctx.exception))

    def test_delete_all_salts_success(self):
        """
        Test that delete_all_salts removes all Salt entries when present.
        """
        s1 = self.ctrl.add(self._make_salt(name="Sal1"))
        s2 = self.ctrl.add(self._make_salt(name="Sal2"))
        self.assertEqual(len(self.ctrl.get_all_salts()), 2)
        self.ctrl.delete_all_salts()
        self.assertEqual(len(self.ctrl.get_all_salts()), 0)

    def test_delete_surfactant_by_id_and_name(self):
        """
        Test deleting a Surfactant by ID and by name, ensuring removal in both cases.
        """
        sf = self._make_surfactant(name="SF_A")
        added = self.ctrl.add(sf)
        sfid = added.id
        self.ctrl.delete_surfactant_by_id(sfid)
        self.assertIsNone(self.ctrl.get_surfactant_by_id(sfid))

        sf2 = self._make_surfactant(name="SF_B")
        added2 = self.ctrl.add(sf2)
        self.ctrl.delete_surfactant_by_name("SF_B")
        self.assertIsNone(self.ctrl.get_surfactant_by_name("SF_B"))

    def test_delete_surfactant_not_exist_errors(self):
        """
        Test that deleting non-existent Surfactant by ID or name raises ValueError.
        """
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_surfactant_by_id(3333)
        self.assertIn("Surfactant with id 3333 does not exist",
                      str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            self.ctrl.delete_surfactant_by_name("NO_SF")
        self.assertIn(
            "Surfactant with name 'NO_SF' does not exist", str(ctx2.exception))

    def test_delete_all_surfactants_empty_errors(self):
        """
        Test that delete_all_surfactants on an empty database raises ValueError.
        """
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_all_surfactants()
        self.assertIn("No items of type 'Surfactant' found",
                      str(ctx.exception))

    def test_delete_all_surfactants_success(self):
        """
        Test that delete_all_surfactants removes all Surfactant entries when present.
        """
        sf1 = self.ctrl.add(self._make_surfactant(name="SF1"))
        sf2 = self.ctrl.add(self._make_surfactant(name="SF2"))
        self.assertEqual(len(self.ctrl.get_all_surfactants()), 2)
        self.ctrl.delete_all_surfactants()
        self.assertEqual(len(self.ctrl.get_all_surfactants()), 0)

    def test_delete_stabilizer_by_id_and_name(self):
        """
        Test deleting a Stabilizer by ID and by name, ensuring removal in both cases.
        """
        st = self._make_stabilizer(name="ST_A")
        added = self.ctrl.add(st)
        stid = added.id
        self.ctrl.delete_stabilizer_by_id(stid)
        self.assertIsNone(self.ctrl.get_stabilizer_by_id(stid))

        st2 = self._make_stabilizer(name="ST_B")
        added2 = self.ctrl.add(st2)
        self.ctrl.delete_stabilizer_by_name("ST_B")
        self.assertIsNone(self.ctrl.get_stabilizer_by_name("ST_B"))

    def test_delete_stabilizer_not_exist_errors(self):
        """
        Test that deleting non-existent Stabilizer by ID or name raises ValueError.
        """
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_stabilizer_by_id(4444)
        self.assertIn("Stabilizer with id 4444 does not exist",
                      str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            self.ctrl.delete_stabilizer_by_name("NO_STAB")
        self.assertIn(
            "Stabilizer with name 'NO_STAB' does not exist", str(ctx2.exception))

    def test_delete_all_stabilizers_empty_errors(self):
        """
        Test that delete_all_stabilizers on an empty database raises ValueError.
        """
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_all_stabilizers()
        self.assertIn("No items of type 'Stabilizer' found",
                      str(ctx.exception))

    def test_delete_all_stabilizers_success(self):
        """
        Test that delete_all_stabilizers removes all Stabilizer entries when present.
        """
        st1 = self.ctrl.add(self._make_stabilizer(name="ST1"))
        st2 = self.ctrl.add(self._make_stabilizer(name="ST2"))
        self.assertEqual(len(self.ctrl.get_all_stabilizers()), 2)
        self.ctrl.delete_all_stabilizers()
        self.assertEqual(len(self.ctrl.get_all_stabilizers()), 0)

    def test_update_protein_successful(self):
        """
        Test that update_protein replaces existing record, preserves enc_id and is_user, and renames.
        """
        orig = self.ctrl.add(self._make_protein(name="OrigProt"))
        orig_id = orig.id
        orig_enc = orig.enc_id
        orig_is_user = orig.is_user

        newp = self._make_protein(name="UpdatedProt", is_user=False)
        updated = self.ctrl.update_protein(orig_id, newp)
        self.assertEqual(updated.enc_id, orig_enc)
        self.assertEqual(updated.is_user, orig_is_user)
        self.assertIsNone(self.ctrl.get_protein_by_name("OrigProt"))
        fetched = self.ctrl.get_protein_by_name("UpdatedProt")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "UpdatedProt")

    def test_update_protein_identical_returns_same(self):
        """
        Test that update_protein returns the same instance when input equals existing record.
        """
        orig = self.ctrl.add(self._make_protein(name="SameProt"))
        orig_id = orig.id
        clone = self._make_protein(name="SameProt", is_user=orig.is_user)
        clone.enc_id = orig.enc_id
        clone.id = orig.id

        result = self.ctrl.update_protein(orig_id, clone)
        self.assertIs(result, clone)

    def test_update_protein_nonexistent_raises(self):
        """
        Test that update_protein on a non-existent ID raises ValueError.
        """
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.update_protein(99999, self._make_protein(name="Nope"))
        self.assertIn("Protein with id '99999' does not exist",
                      str(ctx.exception))

    def test_update_buffer_successful(self):
        """
        Test that update_buffer replaces existing record, preserves enc_id and is_user, and renames.
        """
        orig = self.ctrl.add(self._make_buffer(name="OrigBuff"))
        orig_id = orig.id
        orig_enc = orig.enc_id
        orig_is_user = orig.is_user

        newb = self._make_buffer(name="UpdatedBuff", is_user=False)
        updated = self.ctrl.update_buffer(orig_id, newb)
        self.assertEqual(updated.enc_id, orig_enc)
        self.assertEqual(updated.is_user, orig_is_user)
        self.assertIsNone(self.ctrl.get_buffer_by_name("OrigBuff"))
        self.assertIsNotNone(self.ctrl.get_buffer_by_name("UpdatedBuff"))

    def test_update_buffer_identical_returns_same(self):
        """
        Test that update_buffer returns the same instance when input equals existing record.
        """
        orig = self.ctrl.add(self._make_buffer(name="SameBuff"))
        orig_id = orig.id
        clone = self._make_buffer(name="SameBuff", is_user=orig.is_user)
        clone.enc_id = orig.enc_id
        clone.id = orig.id

        result = self.ctrl.update_buffer(orig_id, clone)
        self.assertIs(result, clone)

    def test_update_buffer_nonexistent_raises(self):
        """
        Test that update_buffer on a non-existent ID raises ValueError.
        """
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.update_buffer(88888, self._make_buffer(name="NoBuff"))
        self.assertIn("Buffer with id '88888' does not exist",
                      str(ctx.exception))

    def test_update_salt_successful(self):
        """
        Test that update_salt replaces existing record, preserves enc_id and is_user, and renames.
        """
        orig = self.ctrl.add(self._make_salt(name="OrigSalt"))
        orig_id = orig.id
        orig_enc = orig.enc_id
        orig_is_user = orig.is_user

        news = self._make_salt(name="UpdatedSalt", is_user=False)
        updated = self.ctrl.update_salt(orig_id, news)
        self.assertEqual(updated.enc_id, orig_enc)
        self.assertEqual(updated.is_user, orig_is_user)
        self.assertIsNone(self.ctrl.get_salt_by_name("OrigSalt"))
        self.assertIsNotNone(self.ctrl.get_salt_by_name("UpdatedSalt"))

    def test_update_salt_identical_returns_same(self):
        """
        Test that update_salt returns the same instance when input equals existing record.
        """
        orig = self.ctrl.add(self._make_salt(name="SameSalt"))
        orig_id = orig.id
        clone = self._make_salt(name="SameSalt", is_user=orig.is_user)
        clone.enc_id = orig.enc_id
        clone.id = orig.id

        result = self.ctrl.update_salt(orig_id, clone)
        self.assertIs(result, clone)

    def test_update_salt_nonexistent_raises(self):
        """
        Test that update_salt on a non-existent ID raises ValueError.
        """
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.update_salt(77777, self._make_salt(name="NoSalt"))
        self.assertIn("Salt with id '77777' does not exist",
                      str(ctx.exception))

    def test_update_surfactant_successful(self):
        """
        Test that update_surfactant replaces existing record, preserves enc_id and is_user, and renames.
        """
        orig = self.ctrl.add(self._make_surfactant(name="OrigSurf"))
        orig_id = orig.id
        orig_enc = orig.enc_id
        orig_is_user = orig.is_user

        newsf = self._make_surfactant(name="UpdatedSurf", is_user=False)
        updated = self.ctrl.update_surfactant(orig_id, newsf)
        self.assertEqual(updated.enc_id, orig_enc)
        self.assertEqual(updated.is_user, orig_is_user)
        self.assertIsNone(self.ctrl.get_surfactant_by_name("OrigSurf"))
        self.assertIsNotNone(self.ctrl.get_surfactant_by_name("UpdatedSurf"))

    def test_update_surfactant_identical_returns_same(self):
        """
        Test that update_surfactant returns the same instance when input equals existing record.
        """
        orig = self.ctrl.add(self._make_surfactant(name="SameSurf"))
        orig_id = orig.id
        clone = self._make_surfactant(name="SameSurf", is_user=orig.is_user)
        clone.enc_id = orig.enc_id
        clone.id = orig.id

        result = self.ctrl.update_surfactant(orig_id, clone)
        self.assertIs(result, clone)

    def test_update_surfactant_nonexistent_raises(self):
        """
        Test that update_surfactant on a non-existent ID raises ValueError.
        """
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.update_surfactant(
                66666, self._make_surfactant(name="NoSurf"))
        self.assertIn("Surfactant with id '66666' does not exist",
                      str(ctx.exception))

    def test_update_stabilizer_successful(self):
        """
        Test that update_stabilizer replaces existing record, preserves enc_id and is_user, and renames.
        """
        orig = self.ctrl.add(self._make_stabilizer(name="OrigStab"))
        orig_id = orig.id
        orig_enc = orig.enc_id
        orig_is_user = orig.is_user

        newstab = self._make_stabilizer(name="UpdatedStab", is_user=False)
        updated = self.ctrl.update_stabilizer(orig_id, newstab)
        self.assertEqual(updated.enc_id, orig_enc)
        self.assertEqual(updated.is_user, orig_is_user)
        self.assertIsNone(self.ctrl.get_stabilizer_by_name("OrigStab"))
        self.assertIsNotNone(self.ctrl.get_stabilizer_by_name("UpdatedStab"))

    def test_update_stabilizer_identical_returns_same(self):
        """
        Test that update_stabilizer returns the same instance when input equals existing record.
        """
        orig = self.ctrl.add(self._make_stabilizer(name="SameStab"))
        orig_id = orig.id
        clone = self._make_stabilizer(name="SameStab", is_user=orig.is_user)
        clone.enc_id = orig.enc_id
        clone.id = orig.id

        result = self.ctrl.update_stabilizer(orig_id, clone)
        self.assertIs(result, clone)

    def test_update_stabilizer_nonexistent_raises(self):
        """
        Test that update_stabilizer on a non-existent ID raises ValueError.
        """
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.update_stabilizer(
                55555, self._make_stabilizer(name="NoStab"))
        self.assertIn("Stabilizer with id '55555' does not exist",
                      str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
