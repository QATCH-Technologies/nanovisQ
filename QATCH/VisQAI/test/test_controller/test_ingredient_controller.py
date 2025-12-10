"""
test_ingredient_controller.py

Comprehensive integration tests for the IngredientController, verifying CRUD operations on:
    - Adding and fetching each ingredient subclass (Protein, Buffer, Salt, Surfactant, Stabilizer, Excipient)
    - Auto-assignment of developer and user enc_id values
    - Name trimming and duplicate name handling
    - Deleting by ID, name, and deleting all for each type, including error conditions
    - Updating existing records while preserving enc_id and is_user, including identical-object short-circuit
    - Dispatch methods that should raise on unsupported types
    - Fuzzy matching functionality
    - User mode filtering

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-10-21

Version:
    1.3
"""

import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from src.models.ingredient import Protein, Buffer, Stabilizer, Surfactant, Salt, Excipient, ProteinClass
    from src.db.db import Database
    from src.controller.ingredient_controller import IngredientController
except (ModuleNotFoundError, ImportError):
    # Fallback imports for different project structures
    try:
        from QATCH.VisQAI.src.models.ingredient import Protein, Buffer, Stabilizer, Surfactant, Salt, Excipient, ProteinClass
        from QATCH.VisQAI.src.db.db import Database
        from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    except (ModuleNotFoundError, ImportError):
        # For standalone testing
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models.ingredient import Protein, Buffer, Stabilizer, Surfactant, Salt, Excipient, ProteinClass
        from db.db import Database
        from controller.ingredient_controller import IngredientController


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
        self.ctrl._user_mode = False

    def tearDown(self):
        """Close the database connection and remove the test database file after each test."""
        self.db.conn.close()
        if self.db_file.exists():
            self.db_file.unlink()

    # ----- Helper Methods ----- #

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
            "SELECT id, name, type, enc_id, is_user FROM ingredient WHERE name = ?",
            (name,)
        ).fetchone()

    def _make_protein(self, name="prot1", is_user=True, enc_id=-1):
        """Helper to create a Protein with default values."""
        p = Protein(
            enc_id=enc_id,
            name=name,
            molecular_weight=50.0,
            pI_mean=6.5,
            pI_range=1.0,
            class_type=ProteinClass.MAB_IGG1
        )
        p.is_user = is_user
        return p

    def _make_buffer(self, name="buff1", is_user=True, enc_id=-1, ph=7.4):
        """Helper to create a Buffer with default values."""
        b = Buffer(enc_id=enc_id, name=name, pH=ph)
        b.is_user = is_user
        return b

    def _make_salt(self, name="salt1", is_user=True, enc_id=-1):
        """Helper to create a Salt with default values."""
        s = Salt(enc_id=enc_id, name=name)
        s.is_user = is_user
        return s

    def _make_stabilizer(self, name="stab1", is_user=True, enc_id=-1):
        """Helper to create a Stabilizer with default values."""
        st = Stabilizer(enc_id=enc_id, name=name)
        st.is_user = is_user
        return st

    def _make_surfactant(self, name="surf1", is_user=True, enc_id=-1):
        """Helper to create a Surfactant with default values."""
        sf = Surfactant(enc_id=enc_id, name=name)
        sf.is_user = is_user
        return sf

    def _make_excipient(self, name="excip1", is_user=True, enc_id=-1):
        """Helper to create an Excipient with default values."""
        ex = Excipient(enc_id=enc_id, name=name)
        ex.is_user = is_user
        return ex

    # ----- Initialization Tests ----- #

    def test_controller_initialization_default(self):
        """Test that controller initializes with default user_mode=True."""
        self.assertFalse(self.ctrl._user_mode)
        self.assertEqual(self.ctrl.db, self.db)

    def test_controller_initialization_dev_mode(self):
        """Test that controller initializes with user_mode=False."""
        dev_ctrl = IngredientController(self.db, user_mode=False)
        self.assertFalse(dev_ctrl._user_mode)

    def test_class_constants(self):
        """Test that class constants are correctly defined."""
        self.assertEqual(IngredientController.DEV_MAX_ID, 8000)
        self.assertEqual(IngredientController.USER_START_ID, 8001)

    # ----- Add and Fetch Tests ----- #

    def test_add_and_fetch_protein(self):
        """Test adding a Protein persists correctly and can be retrieved by ID and name."""
        p = self._make_protein(name="ProtA", is_user=False, enc_id=1)
        self.ctrl.add_protein(p)

        # Verify raw database row
        row = self._fetch_row("ProtA")
        self.assertIsNotNone(row, "Protein row not found in DB")
        db_id, db_name, db_type, db_enc_id, db_is_user = row
        self.assertEqual(db_name, "ProtA")
        self.assertEqual(db_type, "Protein")
        self.assertEqual(db_enc_id, 1)
        self.assertEqual(db_is_user, 0)  # False stored as 0

        # Fetch by ID
        fetched_by_id = self.ctrl.get_protein_by_id(db_id)
        self.assertIsInstance(fetched_by_id, Protein)
        self.assertEqual(fetched_by_id.id, db_id)
        self.assertEqual(fetched_by_id.name, "ProtA")
        self.assertAlmostEqual(fetched_by_id.molecular_weight, 50.0)

        # Fetch by name
        fetched_by_name = self.ctrl.get_protein_by_name("ProtA")
        self.assertEqual(fetched_by_name.id, db_id)
        self.assertEqual(fetched_by_name.name, "ProtA")

    def test_add_duplicate_protein_returns_existing(self):
        """Test that adding a duplicate Protein name returns existing instance."""
        p1 = self._make_protein(name="ProtA", is_user=False)
        self.ctrl.add_protein(p1)

        p2 = self._make_protein(name="ProtA", is_user=False)
        ret = self.ctrl.add_protein(p2)

        self.assertEqual(ret.name, "ProtA")
        self.assertEqual(len(self.ctrl.get_all_proteins()), 1)

    def test_add_and_get_buffer(self):
        """Test adding a Buffer persists correctly and can be fetched."""
        b = self._make_buffer(name="BufX", is_user=False, enc_id=2, ph=7.4)
        fetched = self._roundtrip(b)

        self.assertIsInstance(fetched, Buffer)
        self.assertEqual(fetched.pH, 7.4)
        self.assertEqual(fetched.name, "BufX")

    def test_add_and_get_salt(self):
        """Test adding a Salt persists correctly and can be fetched."""
        s = self._make_salt(name="SaltY", is_user=False, enc_id=3)
        fetched = self._roundtrip(s)

        self.assertIsInstance(fetched, Salt)
        self.assertEqual(fetched.name, "SaltY")

    def test_add_and_get_surfactant(self):
        """Test adding a Surfactant persists correctly and can be fetched."""
        sf = self._make_surfactant(name="SurfZ", is_user=False, enc_id=4)
        fetched = self._roundtrip(sf)

        self.assertIsInstance(fetched, Surfactant)
        self.assertEqual(fetched.name, "SurfZ")

    def test_add_and_get_stabilizer(self):
        """Test adding a Stabilizer persists correctly and can be fetched."""
        st = self._make_stabilizer(name="StabW", is_user=False, enc_id=5)
        fetched = self._roundtrip(st)

        self.assertIsInstance(fetched, Stabilizer)
        self.assertEqual(fetched.name, "StabW")

    def test_add_and_get_excipient(self):
        """Test adding an Excipient persists correctly and can be fetched."""
        ex = self._make_excipient(name="ExcipA", is_user=False, enc_id=6)
        fetched = self._roundtrip(ex)

        self.assertIsInstance(fetched, Excipient)
        self.assertEqual(fetched.name, "ExcipA")

    # ----- Get All and Delete All Tests ----- #

    def test_get_all_ingredients_and_delete_all(self):
        """Test that multiple ingredient types can be added, all retrieved, and then deleted."""
        types = [
            self._make_protein(name="P1", is_user=False),
            self._make_buffer(name="B1", is_user=False),
            self._make_salt(name="S1", is_user=False),
            self._make_surfactant(name="SF1", is_user=False),
            self._make_stabilizer(name="ST1", is_user=False),
            self._make_excipient(name="EX1", is_user=False),
        ]

        for ing in types:
            self.ctrl.add(ing)

        all_ings = self.ctrl.get_all_ingredients()
        self.assertEqual(len(all_ings), 6)
        self.assertEqual(
            {type(i) for i in all_ings},
            {Protein, Buffer, Salt, Surfactant, Stabilizer, Excipient}
        )

        self.ctrl.delete_all_ingredients()
        self.assertEqual(self.ctrl.get_all_ingredients(), [])

    def test_get_all_empty_initially(self):
        """Test that get_all_ingredients initially returns an empty list when DB is empty."""
        self.assertEqual(self.ctrl.get_all_ingredients(), [])

    def test_fetch_nonexistent_returns_none(self):
        """Test that fetching by ID for each subclass returns None for non-existent IDs."""
        self.assertIsNone(self.ctrl.get_protein_by_id(999))
        self.assertIsNone(self.ctrl.get_buffer_by_id(999))
        self.assertIsNone(self.ctrl.get_salt_by_id(999))
        self.assertIsNone(self.ctrl.get_surfactant_by_id(999))
        self.assertIsNone(self.ctrl.get_stabilizer_by_id(999))
        self.assertIsNone(self.ctrl.get_excipient_by_id(999))

    # ----- Deletion Tests by ID ----- #

    def test_delete_protein_by_id(self):
        """Test deleting a Protein by ID removes it from the database."""
        p = self._make_protein(name="P_A")
        added = self.ctrl.add(p)
        pid = added.id

        self.ctrl.delete_protein_by_id(pid)
        self.assertIsNone(self.ctrl.get_protein_by_id(pid))

    def test_delete_protein_by_id_errors_if_not_exist(self):
        """Test that deleting a non-existent Protein by ID raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_protein_by_id(9999)
        self.assertIn("Protein with id 9999 does not exist",
                      str(ctx.exception))

    def test_delete_buffer_by_id(self):
        """Test deleting a Buffer by ID removes it from the database."""
        b = self._make_buffer(name="B_A")
        added = self.ctrl.add(b)
        bid = added.id

        self.ctrl.delete_buffer_by_id(bid)
        self.assertIsNone(self.ctrl.get_buffer_by_id(bid))

    def test_delete_buffer_by_id_errors_if_not_exist(self):
        """Test that deleting a non-existent Buffer by ID raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_buffer_by_id(12345)
        self.assertIn("Buffer with id 12345 does not exist",
                      str(ctx.exception))

    def test_delete_salt_by_id(self):
        """Test deleting a Salt by ID removes it from the database."""
        s = self._make_salt(name="S_A")
        added = self.ctrl.add(s)
        sid = added.id

        self.ctrl.delete_salt_by_id(sid)
        self.assertIsNone(self.ctrl.get_salt_by_id(sid))

    def test_delete_salt_by_id_errors_if_not_exist(self):
        """Test that deleting a non-existent Salt by ID raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_salt_by_id(2222)
        self.assertIn("Salt with id 2222 does not exist", str(ctx.exception))

    def test_delete_surfactant_by_id(self):
        """Test deleting a Surfactant by ID removes it from the database."""
        sf = self._make_surfactant(name="SF_A")
        added = self.ctrl.add(sf)
        sfid = added.id

        self.ctrl.delete_surfactant_by_id(sfid)
        self.assertIsNone(self.ctrl.get_surfactant_by_id(sfid))

    def test_delete_surfactant_by_id_errors_if_not_exist(self):
        """Test that deleting a non-existent Surfactant by ID raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_surfactant_by_id(3333)
        self.assertIn("Surfactant with id 3333 does not exist",
                      str(ctx.exception))

    def test_delete_stabilizer_by_id(self):
        """Test deleting a Stabilizer by ID removes it from the database."""
        st = self._make_stabilizer(name="ST_A")
        added = self.ctrl.add(st)
        stid = added.id

        self.ctrl.delete_stabilizer_by_id(stid)
        self.assertIsNone(self.ctrl.get_stabilizer_by_id(stid))

    def test_delete_stabilizer_by_id_errors_if_not_exist(self):
        """Test that deleting a non-existent Stabilizer by ID raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_stabilizer_by_id(4444)
        self.assertIn("Stabilizer with id 4444 does not exist",
                      str(ctx.exception))

    def test_delete_excipient_by_id(self):
        """Test deleting an Excipient by ID removes it from the database."""
        ex = self._make_excipient(name="EX_A")
        added = self.ctrl.add(ex)
        exid = added.id

        self.ctrl.delete_excipient_by_id(exid)
        self.assertIsNone(self.ctrl.get_excipient_by_id(exid))

    def test_delete_excipient_by_id_errors_if_not_exist(self):
        """Test that deleting a non-existent Excipient by ID raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_excipient_by_id(5555)
        self.assertIn("Excipient with id 5555 does not exist",
                      str(ctx.exception))

    # ----- Deletion Tests by Name ----- #

    def test_delete_protein_by_name(self):
        """Test deleting a Protein by name removes it from the database."""
        p = self._make_protein(name="P_B")
        self.ctrl.add(p)

        self.ctrl.delete_protein_by_name("P_B")
        self.assertIsNone(self.ctrl.get_protein_by_name("P_B"))

    def test_delete_protein_by_name_errors_if_not_exist(self):
        """Test that deleting a non-existent Protein by name raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_protein_by_name("NON_EXIST")
        self.assertIn(
            "Protein with name 'NON_EXIST' does not exist", str(ctx.exception))

    def test_delete_buffer_by_name(self):
        """Test deleting a Buffer by name removes it from the database."""
        b = self._make_buffer(name="B_B")
        self.ctrl.add(b)

        self.ctrl.delete_buffer_by_name("B_B")
        self.assertIsNone(self.ctrl.get_buffer_by_name("B_B"))

    def test_delete_buffer_by_name_errors_if_not_exist(self):
        """Test that deleting a non-existent Buffer by name raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_buffer_by_name("NO_BUFF")
        self.assertIn("Buffer with name 'NO_BUFF' does not exist",
                      str(ctx.exception))

    def test_delete_salt_by_name(self):
        """Test deleting a Salt by name removes it from the database."""
        s = self._make_salt(name="S_B")
        self.ctrl.add(s)

        self.ctrl.delete_salt_by_name("S_B")
        self.assertIsNone(self.ctrl.get_salt_by_name("S_B"))

    def test_delete_salt_by_name_errors_if_not_exist(self):
        """Test that deleting a non-existent Salt by name raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_salt_by_name("NO_SALT")
        self.assertIn("Salt with name 'NO_SALT' does not exist",
                      str(ctx.exception))

    def test_delete_surfactant_by_name(self):
        """Test deleting a Surfactant by name removes it from the database."""
        sf = self._make_surfactant(name="SF_B")
        self.ctrl.add(sf)

        self.ctrl.delete_surfactant_by_name("SF_B")
        self.assertIsNone(self.ctrl.get_surfactant_by_name("SF_B"))

    def test_delete_surfactant_by_name_errors_if_not_exist(self):
        """Test that deleting a non-existent Surfactant by name raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_surfactant_by_name("NO_SF")
        self.assertIn(
            "Surfactant with name 'NO_SF' does not exist", str(ctx.exception))

    def test_delete_stabilizer_by_name(self):
        """Test deleting a Stabilizer by name removes it from the database."""
        st = self._make_stabilizer(name="ST_B")
        self.ctrl.add(st)

        self.ctrl.delete_stabilizer_by_name("ST_B")
        self.assertIsNone(self.ctrl.get_stabilizer_by_name("ST_B"))

    def test_delete_stabilizer_by_name_errors_if_not_exist(self):
        """Test that deleting a non-existent Stabilizer by name raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_stabilizer_by_name("NO_STAB")
        self.assertIn(
            "Stabilizer with name 'NO_STAB' does not exist", str(ctx.exception))

    def test_delete_excipient_by_name(self):
        """Test deleting an Excipient by name removes it from the database."""
        ex = self._make_excipient(name="EX_B")
        self.ctrl.add(ex)

        self.ctrl.delete_excipient_by_name("EX_B")
        self.assertIsNone(self.ctrl.get_excipient_by_name("EX_B"))

    def test_delete_excipient_by_name_errors_if_not_exist(self):
        """Test that deleting a non-existent Excipient by name raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_excipient_by_name("NO_EXCIP")
        self.assertIn(
            "Excipient with name 'NO_EXCIP' does not exist", str(ctx.exception))

    # ----- Delete All Tests ----- #

    def test_delete_all_proteins_empty_errors(self):
        """Test that delete_all_proteins on an empty database raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_all_proteins()
        self.assertIn("No items of type 'Protein' found", str(ctx.exception))

    def test_delete_all_proteins_success(self):
        """Test that delete_all_proteins removes all Protein entries when present."""
        self.ctrl.add(self._make_protein(name="Px1"))
        self.ctrl.add(self._make_protein(name="Px2"))

        all_prots = self.ctrl.get_all_proteins()
        self.assertEqual(len(all_prots), 2)

        self.ctrl.delete_all_proteins()
        self.assertEqual(len(self.ctrl.get_all_proteins()), 0)

    def test_delete_all_buffers_empty_errors(self):
        """Test that delete_all_buffers on an empty database raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_all_buffers()
        self.assertIn("No items of type 'Buffer' found", str(ctx.exception))

    def test_delete_all_buffers_success(self):
        """Test that delete_all_buffers removes all Buffer entries when present."""
        self.ctrl.add(self._make_buffer(name="B1"))
        self.ctrl.add(self._make_buffer(name="B2"))

        self.assertEqual(len(self.ctrl.get_all_buffers()), 2)

        self.ctrl.delete_all_buffers()
        self.assertEqual(len(self.ctrl.get_all_buffers()), 0)

    def test_delete_all_salts_empty_errors(self):
        """Test that delete_all_salts on an empty database raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_all_salts()
        self.assertIn("No items of type 'Salt' found", str(ctx.exception))

    def test_delete_all_salts_success(self):
        """Test that delete_all_salts removes all Salt entries when present."""
        self.ctrl.add(self._make_salt(name="Sal1"))
        self.ctrl.add(self._make_salt(name="Sal2"))

        self.assertEqual(len(self.ctrl.get_all_salts()), 2)

        self.ctrl.delete_all_salts()
        self.assertEqual(len(self.ctrl.get_all_salts()), 0)

    def test_delete_all_surfactants_empty_errors(self):
        """Test that delete_all_surfactants on an empty database raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_all_surfactants()
        self.assertIn("No items of type 'Surfactant' found",
                      str(ctx.exception))

    def test_delete_all_surfactants_success(self):
        """Test that delete_all_surfactants removes all Surfactant entries when present."""
        self.ctrl.add(self._make_surfactant(name="SF1"))
        self.ctrl.add(self._make_surfactant(name="SF2"))

        self.assertEqual(len(self.ctrl.get_all_surfactants()), 2)

        self.ctrl.delete_all_surfactants()
        self.assertEqual(len(self.ctrl.get_all_surfactants()), 0)

    def test_delete_all_stabilizers_empty_errors(self):
        """Test that delete_all_stabilizers on an empty database raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_all_stabilizers()
        self.assertIn("No items of type 'Stabilizer' found",
                      str(ctx.exception))

    def test_delete_all_stabilizers_success(self):
        """Test that delete_all_stabilizers removes all Stabilizer entries when present."""
        self.ctrl.add(self._make_stabilizer(name="ST1"))
        self.ctrl.add(self._make_stabilizer(name="ST2"))

        self.assertEqual(len(self.ctrl.get_all_stabilizers()), 2)

        self.ctrl.delete_all_stabilizers()
        self.assertEqual(len(self.ctrl.get_all_stabilizers()), 0)

    def test_delete_all_excipients_empty_errors(self):
        """Test that delete_all_excipients on an empty database raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.delete_all_excipients()
        self.assertIn("No items of type 'Excipient' found", str(ctx.exception))

    def test_delete_all_excipients_success(self):
        """Test that delete_all_excipients removes all Excipient entries when present."""
        self.ctrl.add(self._make_excipient(name="EX1"))
        self.ctrl.add(self._make_excipient(name="EX2"))

        self.assertEqual(len(self.ctrl.get_all_excipients()), 2)

        self.ctrl.delete_all_excipients()
        self.assertEqual(len(self.ctrl.get_all_excipients()), 0)

    # ----- Update Tests ----- #

    def test_update_protein_successful(self):
        """Test that update_protein replaces existing record and preserves enc_id and is_user."""
        self.ctrl._user_mode = True
        orig = self.ctrl.add(self._make_protein(name="Original", is_user=True))
        orig_id = orig.id
        orig_enc = orig.enc_id
        orig_is_user = orig.is_user

        newp = self._make_protein(name="Updated", is_user=True)
        updated = self.ctrl.update_protein(orig_id, newp)

        # Should preserve enc_id and is_user from existing
        self.assertEqual(updated.enc_id, orig_enc)
        self.assertEqual(updated.is_user, orig_is_user)

        # Old name should not exist
        self.assertIsNone(self.ctrl.get_protein_by_name("Original"))

        # New name should exist
        fetched = self.ctrl.get_protein_by_name("Updated")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "Updated")

    def test_update_protein_identical_returns_same(self):
        """Test that update_protein returns the same instance when input equals existing record."""
        orig = self.ctrl.add(self._make_protein(
            name="SameProt", is_user=False))
        orig_id = orig.id

        clone = self._make_protein(name="SameProt", is_user=orig.is_user)
        clone.enc_id = orig.enc_id
        clone.id = orig.id

        result = self.ctrl.update_protein(orig_id, clone)
        self.assertEqual(result, clone)

    def test_update_protein_nonexistent_raises(self):
        """Test that update_protein on a non-existent ID raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.ctrl.update_protein(99999, self._make_protein(name="Nope"))
        self.assertIn("Protein with id '99999' does not exist",
                      str(ctx.exception))

    def test_update_buffer_successful(self):
        """Test that update_buffer replaces existing record and preserves enc_id and is_user."""
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

    def test_update_salt_successful(self):
        """Test that update_salt replaces existing record and preserves enc_id and is_user."""
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

    def test_update_surfactant_successful(self):
        """Test that update_surfactant replaces existing record and preserves enc_id and is_user."""
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

    def test_update_stabilizer_successful(self):
        """Test that update_stabilizer replaces existing record and preserves enc_id and is_user."""
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

    def test_update_excipient_successful(self):
        """Test that update_excipient replaces existing record and preserves enc_id and is_user."""
        orig = self.ctrl.add(self._make_excipient(name="OrigExcip"))
        orig_id = orig.id
        orig_enc = orig.enc_id
        orig_is_user = orig.is_user

        newex = self._make_excipient(name="UpdatedExcip", is_user=False)
        updated = self.ctrl.update_excipient(orig_id, newex)

        self.assertEqual(updated.enc_id, orig_enc)
        self.assertEqual(updated.is_user, orig_is_user)
        self.assertIsNone(self.ctrl.get_excipient_by_name("OrigExcip"))
        self.assertIsNotNone(self.ctrl.get_excipient_by_name("UpdatedExcip"))

    # ----- Dispatcher Method Tests ----- #

    def test_dispatch_methods_raise_on_bad_type(self):
        """Test that generic dispatch methods raise ValueError when given an unsupported type."""
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

    # ----- enc_id Auto-Assignment Tests ----- #

    def test_enc_id_auto_assign_dev(self):
        """Test that developer-created ingredients receive sequential enc_id starting at 1."""
        self.ctrl._user_mode = False
        p1 = self._make_protein(name="Protein", is_user=False)
        self.ctrl.add_protein(p1)
        fetched1 = self.ctrl.get_protein_by_name("Protein")
        self.assertEqual(fetched1.enc_id, 1)

        p2 = self._make_protein(name="Other", is_user=False)
        self.ctrl.add_protein(p2)
        fetched2 = self.ctrl.get_protein_by_name("Other")
        self.assertEqual(fetched2.enc_id, 2)

    def test_enc_id_auto_assign_user(self):
        """Test that user-created ingredients receive enc_id starting at USER_START_ID."""
        self.ctrl._user_mode = True
        b1 = self._make_buffer(name="Buffer", is_user=True)
        self.ctrl.add_buffer(b1)
        fetched1 = self.ctrl.get_buffer_by_name("Buffer")
        self.assertEqual(fetched1.enc_id, IngredientController.USER_START_ID)

        b2 = self._make_buffer(name="Other", is_user=True)
        self.ctrl.add_buffer(b2)
        fetched2 = self.ctrl.get_buffer_by_name("Other")
        self.assertEqual(
            fetched2.enc_id, IngredientController.USER_START_ID + 1)

    def test_no_overlap_dev_and_user(self):
        """Test that developer and user enc_id ranges do not overlap across subclasses."""
        # Developer Salt
        s_dev = self._make_salt(name="DevSalt", is_user=False)
        self.ctrl.add_salt(s_dev)
        fetched_dev = self.ctrl.get_salt_by_name("DevSalt")
        self.assertEqual(fetched_dev.enc_id, 1)

        # User Surfactant
        sf_user = self._make_surfactant(name="UserSurf", is_user=True)
        self.ctrl.add_surfactant(sf_user)
        fetched_user = self.ctrl.get_surfactant_by_name("UserSurf")
        self.assertEqual(fetched_user.enc_id,
                         IngredientController.USER_START_ID)

        # Developer Stabilizer
        stab_dev = self._make_stabilizer(name="DevStab", is_user=False)
        self.ctrl.add_stabilizer(stab_dev)
        fetched_stab = self.ctrl.get_stabilizer_by_name("DevStab")
        self.assertEqual(fetched_stab.enc_id, 1)

        # User Stabilizer
        stab_user = self._make_stabilizer(name="UserStab", is_user=True)
        self.ctrl.add_stabilizer(stab_user)
        fetched_stab_user = self.ctrl.get_stabilizer_by_name("UserStab")
        self.assertEqual(fetched_stab_user.enc_id,
                         IngredientController.USER_START_ID)

    def test_dev_id_exhaustion_raises(self):
        """Test that when DEV_MAX_ID is exhausted for a subclass, adding another raises RuntimeError."""
        original_dev_max = IngredientController.DEV_MAX_ID
        IngredientController.DEV_MAX_ID = 2

        try:
            p1 = self._make_protein(name="DExhaust1", is_user=False)
            self.ctrl.add_protein(p1)
            fetched1 = self.ctrl.get_protein_by_name("DExhaust1")
            self.assertEqual(fetched1.enc_id, 1)

            p2 = self._make_protein(name="Other Protein", is_user=False)
            self.ctrl.add_protein(p2)
            fetched2 = self.ctrl.get_protein_by_name("Other Protein")
            self.assertEqual(fetched2.enc_id, 2)

            p3 = self._make_protein(name="Exhausted", is_user=False)
            with self.assertRaises(RuntimeError) as ctx:
                self.ctrl.add_protein(p3)
            self.assertIn("No developer enc_id available", str(ctx.exception))
        finally:
            IngredientController.DEV_MAX_ID = original_dev_max

    # ----- Name Handling Tests ----- #

    def test_trimmed_name(self):
        """Test that whitespace around the name is trimmed on insert and fetch."""
        b = self._make_buffer(name="  TrimTest  ", is_user=False)
        self.ctrl.add(b)

        fetched = self.ctrl.get_buffer_by_name("TrimTest")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "TrimTest")

    def test_cross_type_same_name_allowed(self):
        """Test that different subclasses can share the same name without conflict."""
        p = self._make_protein(name="Common", is_user=False)
        self.ctrl.add(p)

        b = self._make_buffer(name="Common", is_user=False)
        self.ctrl.add(b)

        self.assertIsInstance(self.ctrl.get_protein_by_name("Common"), Protein)
        self.assertIsInstance(self.ctrl.get_buffer_by_name("Common"), Buffer)

        commons = [ing for ing in self.ctrl.get_all_ingredients()
                   if ing.name == "Common"]
        self.assertEqual(len(commons), 2)

    # ----- Fuzzy Matching Tests ----- #

    def test_fuzzy_fetch_exact_match(self):
        """Test fuzzy_fetch with exact match."""
        p = self._make_protein(name="ExactMatch")
        self.ctrl.add(p)

        matches = self.ctrl.fuzzy_fetch("ExactMatch")
        self.assertIn("ExactMatch", matches)

    def test_fuzzy_fetch_close_match(self):
        """Test fuzzy_fetch with close but not exact match."""
        p = self._make_protein(name="Albumin")
        self.ctrl.add(p)

        matches = self.ctrl.fuzzy_fetch("Albumen", score_cutoff=70)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0], "Albumin")

    def test_fuzzy_fetch_no_match(self):
        """Test fuzzy_fetch with no close matches."""
        p = self._make_protein(name="Albumin")
        self.ctrl.add(p)

        matches = self.ctrl.fuzzy_fetch("XYZ", score_cutoff=75)
        self.assertEqual(len(matches), 0)

    def test_fuzzy_fetch_multiple_matches(self):
        """Test fuzzy_fetch returns multiple matches when available."""
        self.ctrl.add(self._make_protein(name="Buffer1"))
        self.ctrl.add(self._make_protein(name="Buffer2"))
        self.ctrl.add(self._make_protein(name="Buffer3"))

        matches = self.ctrl.fuzzy_fetch("Buff", max_results=5, score_cutoff=60)
        self.assertGreaterEqual(len(matches), 1)

    def test_fuzzy_fetch_max_results_limit(self):
        """Test that fuzzy_fetch respects max_results parameter."""
        for i in range(5):
            self.ctrl.add(self._make_protein(name=f"Test{i}"))

        matches = self.ctrl.fuzzy_fetch("Test", max_results=2, score_cutoff=50)
        self.assertLessEqual(len(matches), 2)

    # ----- User Mode Tests ----- #

    def test_get_all_ingredient_names_filters_in_user_mode(self):
        """Test that get_all_ingredient_names filters out dev ingredients in user mode."""
        self.ctrl._user_mode = True
        dev_protein = self._make_protein(name="DevProtein", is_user=False)
        user_protein = self._make_protein(name="UserProtein", is_user=True)

        self.ctrl.add(dev_protein)
        self.ctrl.add(user_protein)

        names = self.ctrl.get_all_ingredient_names()
        self.assertNotIn("DevProtein", names)
        self.assertIn("UserProtein", names)

    def test_get_all_ingredient_names_returns_all_in_dev_mode(self):
        """Test that get_all_ingredient_names returns all ingredients in dev mode."""
        dev_ctrl = IngredientController(self.db, user_mode=False)

        dev_protein = self._make_protein(name="DevProtein", is_user=False)
        user_protein = self._make_protein(name="UserProtein", is_user=True)

        dev_ctrl.add(dev_protein)
        dev_ctrl.add(user_protein)

        names = dev_ctrl.get_all_ingredient_names()
        self.assertIn("UserProtein", names)
        self.assertIn("DevProtein", names)

    def test_fetch_by_name_respects_user_mode(self):
        """Test that _fetch_by_name respects user_mode setting."""
        self.ctrl._user_mode = True
        dev_protein = self._make_protein(name="DevProtein", is_user=False)
        self.ctrl.add(dev_protein)

        # In user mode, should not find dev ingredient
        result = self.ctrl.get_protein_by_name("DevProtein")
        self.assertIsNone(result)

        # In dev mode, should find it
        dev_ctrl = IngredientController(self.db, user_mode=False)
        result = dev_ctrl.get_protein_by_name("DevProtein")
        self.assertIsNotNone(result)

    # ----- Additional Edge Cases ----- #

    def test_duplicate_buffer_name_does_not_create_duplicate(self):
        """Test that adding two Buffers with the same name does not create a duplicate."""
        b1 = self._make_buffer(name="BufDup", is_user=False, ph=6.5)
        b2 = self._make_buffer(name="BufDup", is_user=False, ph=6.5)

        self.ctrl.add(b1)
        self.ctrl.add(b2)

        self.assertEqual(len(self.ctrl.get_all_buffers()), 1)

    def test_empty_database_operations(self):
        """Test various operations on empty database."""
        self.assertEqual(self.ctrl.get_all_ingredients(), [])
        self.assertEqual(self.ctrl.get_all_proteins(), [])
        self.assertEqual(self.ctrl.get_all_buffers(), [])
        self.assertIsNone(self.ctrl.get_protein_by_name("NonExistent"))

    def test_add_with_dispatcher_works_correctly(self):
        """Test that the add dispatcher correctly routes to type-specific methods."""
        p = self._make_protein(name="DispatchProtein")
        result = self.ctrl.add(p)

        self.assertIsNotNone(result.id)
        self.assertEqual(result.name, "DispatchProtein")

        fetched = self.ctrl.get_protein_by_name("DispatchProtein")
        self.assertIsNotNone(fetched)

    def test_get_by_type_dispatcher_works_correctly(self):
        """Test that get_by_type dispatcher returns correct type list."""
        self.ctrl.add(self._make_protein(name="P1"))
        self.ctrl.add(self._make_protein(name="P2"))
        self.ctrl.add(self._make_buffer(name="B1"))

        proteins = self.ctrl.get_by_type(self._make_protein())
        self.assertEqual(len(proteins), 2)
        self.assertTrue(all(isinstance(p, Protein) for p in proteins))


if __name__ == "__main__":
    unittest.main()
