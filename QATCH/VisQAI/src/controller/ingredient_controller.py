"""
ingredient_controller.py

This module defines the `IngredientController` class for managing CRUD operations
on `Ingredient` objects and their subclasses (Protein, Buffer, Stabilizer, Surfactant, Salt)
in the database. The controller handles adding new ingredients with auto-assigned `enc_id`,
retrieving by ID or name, updating, and deleting both single and all instances of a given type.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-10-22

Version:
    1.7
"""

from typing import List, Optional
from rapidfuzz import process, fuzz

try:
    from src.db.db import Database
    from src.models.ingredient import Protein, Salt, Stabilizer, Surfactant, Buffer, Ingredient, Excipient
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.models.ingredient import Protein, Salt, Stabilizer, Surfactant, Buffer, Ingredient, Excipient


class IngredientController:
    """Controller for managing `Ingredient` persistence and retrieval.

    This class wraps a `Database` instance to handle CRUD operations on `Ingredient`
    and its subclasses. It supports adding new ingredients with auto-incremented `enc_id`
    values, retrieving ingredients by ID or name, updating existing records, and deleting
    single or all records of a given ingredient type.

    Attributes:
        DEV_MAX_ID (int): Maximum `enc_id` reserved for developer-created ingredients.
        USER_START_ID (int): Starting `enc_id` for user-created ingredients.
        db (Database): The database instance used for storing and retrieving ingredients.
    """

    # Reserve 1..DEV_MAX_ID for developer ingredients of each TYPE
    DEV_MAX_ID = 8000
    # User IDs start at DEV_MAX_ID+1 for each TYPE
    USER_START_ID = DEV_MAX_ID + 1

    def __init__(self, db: Database, user_mode: bool = True) -> None:
        """Initialize the IngredientController with a database instance.

        Args:
            db (Database): An initialized `Database` instance for persisting ingredients.
        """
        self.db: Database = db
        self._user_mode: bool = user_mode

    def get_all_ingredients(self) -> List[Ingredient]:
        """Retrieve all ingredients stored in the database.

        Returns:
            List[Ingredient]: A list of all `Ingredient` instances in the database.
        """
        return self.db.get_all_ingredients()

    def get_all_ingredient_names(self) -> List[str]:
        """Retrieve the names of every ingredient in the database.

        Returns:
            List[str]: A list of all stored ingredient names.
        """

        names = [ing.name for ing in self.get_all_ingredients()
                 if self._user_mode and ing.is_user or not self._user_mode]

        return list(set(names))

    def delete_all_ingredients(self) -> None:
        """Delete all ingredients from the database."""
        self.db.delete_all_ingredients()

    def get_by_id(self, id: int, ingredient: Ingredient) -> Ingredient:
        """Retrieve a specific ingredient by ID, dispatching to the correct subclass method.

        Args:
            id (int): The primary key of the ingredient to fetch.
            ingredient (Ingredient): An instance of the ingredient subclass type to determine lookup.

        Returns:
            Ingredient: The retrieved `Ingredient` subclass instance, or None if not found.

        Raises:
            ValueError: If the provided `ingredient` type is not supported.
        """
        t = ingredient.type
        if t == "Protein":
            return self.get_protein_by_id(id)
        elif t == "Buffer":
            return self.get_buffer_by_id(id)
        elif t == "Stabilizer":
            return self.get_stabilizer_by_id(id)
        elif t == "Salt":
            return self.get_salt_by_id(id)
        elif t == "Surfactant":
            return self.get_surfactant_by_id(id)
        elif t == "Excipient":
            return self.get_excipient_by_id(id)
        else:
            raise ValueError(f"Ingredient type '{t}' not supported.")

    def get_by_name(self, name: str, ingredient: Ingredient) -> Ingredient:
        """Retrieve a specific ingredient by name, dispatching to the correct subclass method.

        Args:
            name (str): The name of the ingredient to fetch.
            ingredient (Ingredient): An instance of the ingredient subclass type to determine lookup.

        Returns:
            Ingredient: The retrieved `Ingredient` subclass instance, or None if not found.

        Raises:
            ValueError: If the provided `ingredient` type is not supported.
        """
        t = ingredient.type
        if t == "Protein":
            return self.get_protein_by_name(name)
        elif t == "Buffer":
            return self.get_buffer_by_name(name)
        elif t == "Stabilizer":
            return self.get_stabilizer_by_name(name)
        elif t == "Salt":
            return self.get_salt_by_name(name)
        elif t == "Surfactant":
            return self.get_surfactant_by_name(name)
        elif t == "Excipient":
            return self.get_excipient_by_name(name)
        else:
            raise ValueError(f"Ingredient type '{t}' not supported.")

    def get_by_type(self, ingredient: Ingredient) -> List[Ingredient]:
        """Retrieve all ingredients of a given subclass type.

        Args:
            ingredient (Ingredient): An instance of the ingredient subclass type to determine lookup.

        Returns:
            List[Ingredient]: A list of all ingredients of the specified type.

        Raises:
            ValueError: If the provided `ingredient` type is not supported.
        """
        t = ingredient.type
        if t == "Protein":
            return self.get_all_proteins()
        elif t == "Buffer":
            return self.get_all_buffers()
        elif t == "Stabilizer":
            return self.get_all_stabilizers()
        elif t == "Salt":
            return self.get_all_salts()
        elif t == "Surfactant":
            return self.get_all_surfactants()
        elif t == "Excipient":
            return self.get_all_excipients()
        else:
            raise ValueError(f"Ingredient type '{t}' not supported.")

    def delete_by_id(self, id: int, ingredient: Ingredient) -> None:
        """Delete a specific ingredient by ID, dispatching to the correct subclass method.

        Args:
            id (int): The primary key of the ingredient to delete.
            ingredient (Ingredient): An instance of the ingredient subclass type to determine deletion method.

        Raises:
            ValueError: If the provided `ingredient` type is not supported.
        """
        t = ingredient.type
        if t == "Protein":
            return self.delete_protein_by_id(id)
        elif t == "Buffer":
            return self.delete_buffer_by_id(id)
        elif t == "Stabilizer":
            return self.delete_stabilizer_by_id(id)
        elif t == "Salt":
            return self.delete_salt_by_id(id)
        elif t == "Surfactant":
            return self.delete_surfactant_by_id(id)
        elif t == "Excipient":
            return self.delete_excipient_by_id(id)
        else:
            raise ValueError(f"Ingredient type '{t}' not supported.")

    def delete_by_name(self, name: str, ingredient: Ingredient) -> None:
        """Delete a specific ingredient by name, dispatching to the correct subclass method.

        Args:
            name (str): The name of the ingredient to delete.
            ingredient (Ingredient): An instance of the ingredient subclass type to determine deletion method.

        Raises:
            ValueError: If the provided `ingredient` type is not supported.
        """
        t = ingredient.type
        if t == "Protein":
            return self.delete_protein_by_name(name)
        elif t == "Buffer":
            return self.delete_buffer_by_name(name)
        elif t == "Stabilizer":
            return self.delete_stabilizer_by_name(name)
        elif t == "Salt":
            return self.delete_salt_by_name(name)
        elif t == "Surfactant":
            return self.delete_surfactant_by_name(name)
        elif t == "Excipient":
            return self.delete_excipient_by_name(name)
        else:
            raise ValueError(f"Ingredient type '{t}' not supported.")

    def delete_by_type(self, ingredient: Ingredient) -> None:
        """Delete all ingredients of a given subclass type.

        Args:
            ingredient (Ingredient): An instance of the ingredient subclass type to determine deletion method.

        Raises:
            ValueError: If the provided `ingredient` type is not supported.
        """
        t = ingredient.type
        if t == "Protein":
            return self.delete_all_proteins()
        elif t == "Buffer":
            return self.delete_all_buffers()
        elif t == "Stabilizer":
            return self.delete_all_stabilizers()
        elif t == "Salt":
            return self.delete_all_salts()
        elif t == "Surfactant":
            return self.delete_all_surfactants()
        elif t == "Excipient":
            return self.delete_all_excipients()
        else:
            raise ValueError(f"Ingredient type '{t}' not supported.")

    def add(self, ingredient: Ingredient) -> Ingredient:
        """Add a new ingredient to the database, dispatching to the correct subclass method.

        If an ingredient with the same name already exists, returns the existing instance.

        Args:
            ingredient (Ingredient): The `Ingredient` subclass instance to add.

        Returns:
            Ingredient: The newly added or existing `Ingredient` instance.

        Raises:
            ValueError: If the provided `ingredient` type is not supported.
        """
        t = ingredient.type
        if t == "Protein":
            return self.add_protein(ingredient)
        elif t == "Buffer":
            return self.add_buffer(ingredient)
        elif t == "Salt":
            return self.add_salt(ingredient)
        elif t == "Stabilizer":
            return self.add_stabilizer(ingredient)
        elif t == "Surfactant":
            return self.add_surfactant(ingredient)
        elif t == "Excipient":
            return self.add_excipient(ingredient)
        else:
            raise ValueError(f"Ingredient type '{t}' not supported.")

    def update(self, id: int, ingredient: Ingredient) -> None:
        """Update an existing ingredient by ID, dispatching to the correct subclass method.

        Args:
            id (int): The primary key of the ingredient to update.
            ingredient (Ingredient): The new `Ingredient` subclass instance containing updated data.

        Raises:
            ValueError: If the provided `ingredient` type is not supported.
        """
        t = ingredient.type
        if t == "Protein":
            return self.update_protein(id, ingredient)
        elif t == "Buffer":
            return self.update_buffer(id, ingredient)
        elif t == "Salt":
            return self.update_salt(id, ingredient)
        elif t == "Stabilizer":
            return self.update_stabilizer(id, ingredient)
        elif t == "Surfactant":
            return self.update_surfactant(id, ingredient)
        elif t == "Excipient":
            return self.update_excipient(id, ingredient)
        else:
            raise ValueError(f"Ingredient type '{t}' not supported.")

    # ----- Accessors ----- #

    def get_protein_by_id(self, id: int) -> Optional[Protein]:
        """Retrieve a `Protein` by its database ID.

        Args:
            id (int): The primary key of the protein to fetch.

        Returns:
            Optional[Protein]: The `Protein` instance if found, otherwise None.
        """
        return self.db.get_ingredient(id)

    def get_protein_by_name(self, name: str) -> Optional[Protein]:
        """Retrieve a `Protein` by its name.

        Args:
            name (str): Name of the protein to fetch.

        Returns:
            Optional[Protein]: The `Protein` instance if found, otherwise None.
        """
        return self._fetch_by_name(name, type="Protein")

    def get_all_proteins(self) -> List[Protein]:
        """Retrieve all `Protein` instances from the database.

        Returns:
            List[Protein]: A list of all proteins in the database.
        """
        return self._fetch_by_type("Protein")

    def get_buffer_by_id(self, id: int) -> Optional[Buffer]:
        """Retrieve a `Buffer` by its database ID.

        Args:
            id (int): The primary key of the buffer to fetch.

        Returns:
            Optional[Buffer]: The `Buffer` instance if found, otherwise None.
        """
        return self.db.get_ingredient(id)

    def get_buffer_by_name(self, name: str) -> Optional[Buffer]:
        """Retrieve a `Buffer` by its name.

        Args:
            name (str): Name of the buffer to fetch.

        Returns:
            Optional[Buffer]: The `Buffer` instance if found, otherwise None.
        """
        return self._fetch_by_name(name, type="Buffer")

    def get_all_buffers(self) -> List[Buffer]:
        """Retrieve all `Buffer` instances from the database.

        Returns:
            List[Buffer]: A list of all buffers in the database.
        """
        return self._fetch_by_type("Buffer")

    def get_salt_by_id(self, id: int) -> Optional[Salt]:
        """Retrieve a `Salt` by its database ID.

        Args:
            id (int): The primary key of the salt to fetch.

        Returns:
            Optional[Salt]: The `Salt` instance if found, otherwise None.
        """
        return self.db.get_ingredient(id)

    def get_salt_by_name(self, name: str) -> Optional[Salt]:
        """Retrieve a `Salt` by its name.

        Args:
            name (str): Name of the salt to fetch.

        Returns:
            Optional[Salt]: The `Salt` instance if found, otherwise None.
        """
        return self._fetch_by_name(name, type="Salt")

    def get_all_salts(self) -> List[Salt]:
        """Retrieve all `Salt` instances from the database.

        Returns:
            List[Salt]: A list of all salts in the database.
        """
        return self._fetch_by_type("Salt")

    def get_excipient_by_id(self, id: int) -> Optional[Excipient]:
        """Retrieve a `Excipient` by its database ID.

        Args:
            id (int): The primary key of the surfactant to fetch.

        Returns:
            Optional[Excipient]: The `Excipient` instance if found, otherwise None.
        """
        return self.db.get_ingredient(id)

    def get_excipient_by_name(self, name: str) -> Optional[Excipient]:
        """Retrieve a `Excipient` by its name.

        Args:
            name (str): Name of the excipient to fetch.

        Returns:
            Optional[Excipient]: The `Excipient` instance if found, otherwise None.
        """
        return self._fetch_by_name(name, type="Excipient")

    def get_all_excipients(self) -> List[Excipient]:
        """Retrieve all `Excipients` instances from the database.

        Returns:
            List[Excipient]: A list of all excipients in the database.
        """
        return self._fetch_by_type("Excipient")

    def get_surfactant_by_id(self, id: int) -> Optional[Surfactant]:
        """Retrieve a `Surfactant` by its database ID.

        Args:
            id (int): The primary key of the surfactant to fetch.

        Returns:
            Optional[Surfactant]: The `Surfactant` instance if found, otherwise None.
        """
        return self.db.get_ingredient(id)

    def get_surfactant_by_name(self, name: str) -> Optional[Surfactant]:
        """Retrieve a `Surfactant` by its name.

        Args:
            name (str): Name of the surfactant to fetch.

        Returns:
            Optional[Surfactant]: The `Surfactant` instance if found, otherwise None.
        """
        return self._fetch_by_name(name, type="Surfactant")

    def get_all_surfactants(self) -> List[Surfactant]:
        """Retrieve all `Surfactant` instances from the database.

        Returns:
            List[Surfactant]: A list of all surfactants in the database.
        """
        return self._fetch_by_type("Surfactant")

    def get_stabilizer_by_id(self, id: int) -> Optional[Stabilizer]:
        """Retrieve a `Stabilizer` by its database ID.

        Args:
            id (int): The primary key of the stabilizer to fetch.

        Returns:
            Optional[Stabilizer]: The `Stabilizer` instance if found, otherwise None.
        """
        return self.db.get_ingredient(id)

    def get_stabilizer_by_name(self, name: str) -> Optional[Stabilizer]:
        """Retrieve a `Stabilizer` by its name.

        Args:
            name (str): Name of the stabilizer to fetch.

        Returns:
            Optional[Stabilizer]: The `Stabilizer` instance if found, otherwise None.
        """
        return self._fetch_by_name(name, type="Stabilizer")

    def get_all_stabilizers(self) -> List[Stabilizer]:
        """Retrieve all `Stabilizer` instances from the database.

        Returns:
            List[Stabilizer]: A list of all stabilizers in the database.
        """
        return self._fetch_by_type("Stabilizer")

    # ----- Creators (per‐type, with auto‐assignment of enc_id) ----- #

    def add_protein(self, protein: Protein) -> Protein:
        """Add a new `Protein` to the database, assigning a unique `enc_id` if needed.

        If a protein with the same name already exists, returns the existing instance.

        Args:
            protein (Protein): A `Protein` instance with `name` and optional properties.

        Returns:
            Protein: The newly added or existing `Protein` instance.
        """
        existing = self.get_protein_by_name(protein.name)
        if existing is not None:
            if existing != protein:
                return self.update_protein(existing.id, protein)
            return existing

        protein.enc_id = self._get_next_enc_id(
            is_user=protein.is_user, ing_type="Protein")
        db_id = self.db.add_ingredient(protein)
        protein.id = db_id

        return protein

    def add_buffer(self, buffer: Buffer) -> Buffer:
        """Add a new `Buffer` to the database, assigning a unique `enc_id` if needed.

        If a buffer with the same name already exists, returns the existing instance.

        Args:
            buffer (Buffer): A `Buffer` instance with `name` and optional pH.

        Returns:
            Buffer: The newly added or existing `Buffer` instance.
        """
        existing = self.get_buffer_by_name(buffer.name)
        if existing is not None:
            if existing != buffer:
                return self.update_buffer(existing.id, buffer)
            return existing

        buffer.enc_id = self._get_next_enc_id(
            is_user=buffer.is_user, ing_type="Buffer")
        db_id = self.db.add_ingredient(buffer)
        buffer.id = db_id

        return buffer

    def add_salt(self, salt: Salt) -> Salt:
        """Add a new `Salt` to the database, assigning a unique `enc_id` if needed.

        If a salt with the same name already exists, returns the existing instance.

        Args:
            salt (Salt): A `Salt` instance with `name`.

        Returns:
            Salt: The newly added or existing `Salt` instance.
        """
        existing = self.get_salt_by_name(salt.name)
        if existing is not None:
            if existing != salt:
                return self.update_salt(existing.id, salt)
            return existing

        salt.enc_id = self._get_next_enc_id(
            is_user=salt.is_user, ing_type="Salt")
        db_id = self.db.add_ingredient(salt)
        salt.id = db_id

        return salt

    def add_stabilizer(self, stabilizer: Stabilizer) -> Stabilizer:
        """Add a new `Stabilizer` to the database, assigning a unique `enc_id` if needed.

        If a stabilizer with the same name already exists, returns the existing instance.

        Args:
            stabilizer (Stabilizer): A `Stabilizer` instance with `name`.

        Returns:
            Stabilizer: The newly added or existing `Stabilizer` instance.
        """
        existing = self.get_stabilizer_by_name(stabilizer.name)
        if existing is not None:
            if existing != stabilizer:
                return self.update_stabilizer(existing.id, stabilizer)
            return existing

        stabilizer.enc_id = self._get_next_enc_id(
            is_user=stabilizer.is_user, ing_type="Stabilizer")
        db_id = self.db.add_ingredient(stabilizer)
        stabilizer.id = db_id

        return stabilizer

    def add_surfactant(self, surfactant: Surfactant) -> Surfactant:
        """Add a new `Surfactant` to the database, assigning a unique `enc_id` if needed.

        If a surfactant with the same name already exists, returns the existing instance.

        Args:
            surfactant (Surfactant): A `Surfactant` instance with `name`.

        Returns:
            Surfactant: The newly added or existing `Surfactant` instance.
        """
        existing = self.get_surfactant_by_name(surfactant.name)
        if existing is not None:
            if existing != surfactant:
                return self.update_surfactant(existing.id, surfactant)
            return existing

        surfactant.enc_id = self._get_next_enc_id(
            is_user=surfactant.is_user, ing_type="Surfactant")
        db_id = self.db.add_ingredient(surfactant)
        surfactant.id = db_id

        return surfactant

    def add_excipient(self, excipient: Excipient) -> Excipient:
        """Add a new `Excipient` to the database, assigning a unique `enc_id` if needed.

        If a Excipient with the same name already exists, returns the existing instance.

        Args:
            excipient (Excipient): A `Excipient` instance with `name`.

        Returns:
            Excipient: The newly added or existing `Excipient` instance.
        """
        existing = self.get_excipient_by_name(excipient.name)
        if existing is not None:
            if existing != excipient:
                return self.update_excipient(existing.id, excipient)
            return existing

        excipient.enc_id = self._get_next_enc_id(
            is_user=excipient.is_user, ing_type="Excipient")
        db_id = self.db.add_ingredient(excipient)
        excipient.id = db_id

        return excipient

    # ----- Deletion ----- #

    def delete_protein_by_id(self, id: int) -> None:
        """Delete a `Protein` by its database ID.

        Args:
            id (int): The primary key of the protein to delete.

        Raises:
            ValueError: If no protein exists with the given ID.
        """
        if self.get_protein_by_id(id) is None:
            raise ValueError(f"Protein with id {id} does not exist.")
        self.db.delete_ingredient(id)

    def delete_protein_by_name(self, name: str) -> None:
        """Delete a `Protein` by its name.

        Args:
            name (str): The name of the protein to delete.

        Raises:
            ValueError: If no protein exists with the given name.
        """
        protein = self.get_protein_by_name(name)
        if protein is None:
            raise ValueError(f"Protein with name '{name}' does not exist.")
        self.db.delete_ingredient(protein.id)

    def delete_all_proteins(self) -> None:
        """Delete all `Protein` instances from the database.

        Raises:
            ValueError: If no proteins are found.
        """
        proteins = self.get_all_proteins()
        if not proteins:
            raise ValueError("No items of type 'Protein' found.")
        for p in proteins:
            self.db.delete_ingredient(p.id)

    def delete_buffer_by_id(self, id: int) -> None:
        """Delete a `Buffer` by its database ID.

        Args:
            id (int): The primary key of the buffer to delete.

        Raises:
            ValueError: If no buffer exists with the given ID.
        """
        if self.get_buffer_by_id(id) is None:
            raise ValueError(f"Buffer with id {id} does not exist.")
        self.db.delete_ingredient(id)

    def delete_buffer_by_name(self, name: str) -> None:
        """Delete a `Buffer` by its name.

        Args:
            name (str): The name of the buffer to delete.

        Raises:
            ValueError: If no buffer exists with the given name.
        """
        buffer = self.get_buffer_by_name(name)
        if buffer is None:
            raise ValueError(f"Buffer with name '{name}' does not exist.")
        self.db.delete_ingredient(buffer.id)

    def delete_all_buffers(self) -> None:
        """Delete all `Buffer` instances from the database.

        Raises:
            ValueError: If no buffers are found.
        """
        buffers = self.get_all_buffers()
        if not buffers:
            raise ValueError("No items of type 'Buffer' found.")
        for b in buffers:
            self.db.delete_ingredient(b.id)

    def delete_salt_by_id(self, id: int) -> None:
        """Delete a `Salt` by its database ID.

        Args:
            id (int): The primary key of the salt to delete.

        Raises:
            ValueError: If no salt exists with the given ID.
        """
        if self.get_salt_by_id(id) is None:
            raise ValueError(f"Salt with id {id} does not exist.")
        self.db.delete_ingredient(id)

    def delete_salt_by_name(self, name: str) -> None:
        """Delete a `Salt` by its name.

        Args:
            name (str): The name of the salt to delete.

        Raises:
            ValueError: If no salt exists with the given name.
        """
        salt = self.get_salt_by_name(name)
        if salt is None:
            raise ValueError(f"Salt with name '{name}' does not exist.")
        self.db.delete_ingredient(salt.id)

    def delete_all_salts(self) -> None:
        """Delete all `Salt` instances from the database.

        Raises:
            ValueError: If no salts are found.
        """
        salts = self.get_all_salts()
        if not salts:
            raise ValueError("No items of type 'Salt' found.")
        for s in salts:
            self.db.delete_ingredient(s.id)

    def delete_surfactant_by_id(self, id: int) -> None:
        """Delete a `Surfactant` by its database ID.

        Args:
            id (int): The primary key of the surfactant to delete.

        Raises:
            ValueError: If no surfactant exists with the given ID.
        """
        if self.get_surfactant_by_id(id) is None:
            raise ValueError(f"Surfactant with id {id} does not exist.")
        self.db.delete_ingredient(id)

    def delete_surfactant_by_name(self, name: str) -> None:
        """Delete a `Surfactant` by its name.

        Args:
            name (str): The name of the surfactant to delete.

        Raises:
            ValueError: If no surfactant exists with the given name.
        """
        surf = self.get_surfactant_by_name(name)
        if surf is None:
            raise ValueError(f"Surfactant with name '{name}' does not exist.")
        self.db.delete_ingredient(surf.id)

    def delete_all_surfactants(self) -> None:
        """Delete all `Surfactant` instances from the database.

        Raises:
            ValueError: If no surfactants are found.
        """
        surfs = self.get_all_surfactants()
        if not surfs:
            raise ValueError("No items of type 'Surfactant' found.")
        for s in surfs:
            self.db.delete_ingredient(s.id)

    def delete_stabilizer_by_id(self, id: int) -> None:
        """Delete a `Stabilizer` by its database ID.

        Args:
            id (int): The primary key of the stabilizer to delete.

        Raises:
            ValueError: If no stabilizer exists with the given ID.
        """
        if self.get_stabilizer_by_id(id) is None:
            raise ValueError(f"Stabilizer with id {id} does not exist.")
        self.db.delete_ingredient(id)

    def delete_stabilizer_by_name(self, name: str) -> None:
        """Delete a `Stabilizer` by its name.

        Args:
            name (str): The name of the stabilizer to delete.

        Raises:
            ValueError: If no stabilizer exists with the given name.
        """
        stab = self.get_stabilizer_by_name(name)
        if stab is None:
            raise ValueError(f"Stabilizer with name '{name}' does not exist.")
        self.db.delete_ingredient(stab.id)

    def delete_all_stabilizers(self) -> None:
        """Delete all `Stabilizer` instances from the database.

        Raises:
            ValueError: If no stabilizers are found.
        """
        stabs = self.get_all_stabilizers()
        if not stabs:
            raise ValueError("No items of type 'Stabilizer' found.")
        for s in stabs:
            self.db.delete_ingredient(s.id)

    def delete_excipient_by_id(self, id: int) -> None:
        """Delete a `Excipient` by its database ID.

        Args:
            id (int): The primary key of the excipient to delete.

        Raises:
            ValueError: If no excipient exists with the given ID.
        """
        if self.get_excipient_by_id(id) is None:
            raise ValueError(f"Excipient with id {id} does not exist.")
        self.db.delete_ingredient(id)

    def delete_excipient_by_name(self, name: str) -> None:
        """Delete a `Excipient` by its name.

        Args:
            name (str): The name of the Excipient to delete.

        Raises:
            ValueError: If no Excipient exists with the given name.
        """
        excip = self.get_excipient_by_name(name)
        if excip is None:
            raise ValueError(f"Excipient with name '{name}' does not exist.")
        self.db.delete_ingredient(excip.id)

    def delete_all_excipients(self) -> None:
        """Delete all `Excipient` instances from the database.

        Raises:
            ValueError: If no Excipients are found.
        """
        excips = self.get_all_excipients()
        if not excips:
            raise ValueError("No items of type 'Excipient' found.")
        for e in excips:
            self.db.delete_ingredient(e.id)

    # ----- Mutators (update) ----- #

    def update_protein(self, id: int, p_new: Protein) -> Protein:
        """Update an existing `Protein` record by replacing it.

        Preserves the original `enc_id` and `is_user` fields.

        Args:
            id (int): The primary key of the existing protein to update.
            p_new (Protein): The new `Protein` instance containing updated data.

        Returns:
            Protein: The updated `Protein` instance.

        Raises:
            ValueError: If no protein exists with the given ID.
        """
        p_fetch = self.get_protein_by_id(id)
        if p_fetch is None:
            raise ValueError(f"Protein with id '{id}' does not exist.")
        if p_fetch == p_new:
            return p_fetch

        # Preserve enc_id and is_user
        p_new.enc_id = p_fetch.enc_id
        p_new.is_user = p_fetch.is_user

        self.db.update_ingredient(p_fetch.id, p_new)
        return p_new

    def update_buffer(self, id: int, b_new: Buffer) -> Buffer:
        """Update an existing `Buffer` record by replacing it.

        Preserves the original `enc_id` and `is_user` fields.

        Args:
            id (int): The primary key of the existing buffer to update.
            b_new (Buffer): The new `Buffer` instance containing updated data.

        Returns:
            Buffer: The updated `Buffer` instance.

        Raises:
            ValueError: If no buffer exists with the given ID.
        """
        b_fetch = self.get_buffer_by_id(id)
        if b_fetch is None:
            raise ValueError(f"Buffer with id '{id}' does not exist.")
        if b_fetch == b_new:
            return b_new

        b_new.enc_id = b_fetch.enc_id
        b_new.is_user = b_fetch.is_user

        self.db.update_ingredient(b_fetch.id, b_new)
        return b_new

    def update_salt(self, id: int, s_new: Salt) -> Salt:
        """Update an existing `Salt` record by replacing it.

        Preserves the original `enc_id` and `is_user` fields.

        Args:
            id (int): The primary key of the existing salt to update.
            s_new (Salt): The new `Salt` instance containing updated data.

        Returns:
            Salt: The updated `Salt` instance.

        Raises:
            ValueError: If no salt exists with the given ID.
        """
        s_fetch = self.get_salt_by_id(id)
        if s_fetch is None:
            raise ValueError(f"Salt with id '{id}' does not exist.")
        if s_fetch == s_new:
            return s_new

        s_new.enc_id = s_fetch.enc_id
        s_new.is_user = s_fetch.is_user

        self.db.update_ingredient(s_fetch.id, s_new)
        return s_new

    def update_surfactant(self, id: int, s_new: Surfactant) -> Surfactant:
        """Update an existing `Surfactant` record by replacing it.

        Preserves the original `enc_id` and `is_user` fields.

        Args:
            id (int): The primary key of the existing surfactant to update.
            s_new (Surfactant): The new `Surfactant` instance containing updated data.

        Returns:
            Surfactant: The updated `Surfactant` instance.

        Raises:
            ValueError: If no surfactant exists with the given ID.
        """
        s_fetch = self.get_surfactant_by_id(id)
        if s_fetch is None:
            raise ValueError(f"Surfactant with id '{id}' does not exist.")
        if s_fetch == s_new:
            return s_new

        s_new.enc_id = s_fetch.enc_id
        s_new.is_user = s_fetch.is_user

        self.db.update_ingredient(s_fetch.id, s_new)
        return s_new

    def update_stabilizer(self, id: int, s_new: Stabilizer) -> Stabilizer:
        """Update an existing `Stabilizer` record by replacing it.

        Preserves the original `enc_id` and `is_user` fields.

        Args:
            id (int): The primary key of the existing stabilizer to update.
            s_new (Stabilizer): The new `Stabilizer` instance containing updated data.

        Returns:
            Stabilizer: The updated `Stabilizer` instance.

        Raises:
            ValueError: If no stabilizer exists with the given ID.
        """
        s_fetch = self.get_stabilizer_by_id(id)
        if s_fetch is None:
            raise ValueError(f"Stabilizer with id '{id}' does not exist.")
        if s_fetch == s_new:
            return s_new

        s_new.enc_id = s_fetch.enc_id
        s_new.is_user = s_fetch.is_user

        self.db.update_ingredient(s_fetch.id, s_new)
        return s_new

    def update_excipient(self, id: int, e_new: Excipient) -> Excipient:
        """Update an existing `Excipient` record by replacing it.

        Preserves the original `enc_id` and `is_user` fields.

        Args:
            id (int): The primary key of the existing excipient to update.
            e_new (Excipient): The new `Excipient` instance containing updated data.

        Returns:
            Stabilizer: The updated `Excipient` instance.

        Raises:
            ValueError: If no excipient exists with the given ID.
        """
        e_fetch = self.get_excipient_by_id(id)
        if e_fetch is None:
            raise ValueError(f"Excipients with id '{id}' does not exist.")
        if e_fetch == e_new:
            return e_new

        e_new.enc_id = e_fetch.enc_id
        e_new.is_user = e_fetch.is_user

        self.db.update_ingredient(e_fetch.id, e_new)
        return e_new

    def fuzzy_fetch(self,
                    name: str,
                    max_results: int = 5,
                    score_cutoff: int = 90) -> list[str]:
        """
        Utility to perform fuzzy matching between ingredient names and persistent names
        stored in the database.  This method operates by fetching all persistent ingredient names
        and then fuzzily matching them against the parameterized name str.  The best match(es) above the
        score_cutoff is returned to the caller as a string list containing ingredient name(s).

        Args:
            name (str): the name of the ingredient to fetch.
            max_results (int): the number of fuzzily matched Ingredient objects to return.
            score_cutoff (int): confidence measure for RapidFuzz to determine accuracy.

        Returns:
            list[str] a list of fuzzily matched ingredient names or a list of None * [max_results]
            if no matches are found.
        """
        all_names = self.get_all_ingredient_names()
        matches = process.extract(
            query=name.lower(),
            choices=all_names,
            scorer=fuzz.WRatio,
            limit=max_results,
            score_cutoff=score_cutoff
        )
        return [match_name for match_name, score, idx in matches]

    def _fetch_by_type(self, type: str) -> List[Ingredient]:
        """Helper method to retrieve all ingredients of a given subclass type.

        Args:
            type (str): The subclass name (e.g., "Protein", "Buffer").

        Returns:
            List[Ingredient]: A list of ingredients matching the specified type.
        """
        ingredients = self.db.get_all_ingredients()
        return [ing for ing in ingredients if ing.type == type]

    def _fetch_by_name(self, name: str, type: str) -> Optional[Ingredient]:
        """Helper method to retrieve a single ingredient by name and subclass type.
        This method performs a fuzzy match on the ingredient name to allow for minor discrepancies
        in user input. If a fuzzy match is found, it uses that name to search for the ingredient.

        Args:
            name (str): The ingredient name to match.
            type (str): The subclass name (e.g., "Protein", "Buffer").

        Returns:
            Optional[Ingredient]: The matching ingredient instance if found, otherwise None.
        """
        ingredients = self.db.get_all_ingredients()
        # Perform a fuzzy matching search on the name.
        fuzzy_name = self.fuzzy_fetch(name=str(name), max_results=1)
        if fuzzy_name:
            name = fuzzy_name[0]
        for ing in ingredients:
            # Skip non-user ingredients while in user mode.
            if self._user_mode and not ing.is_user:
                continue
            # Return first matching ingredient by type and name.
            if ing.type == type and ing.name == name:
                return ing
        return None

    def _get_next_enc_id(self, *, is_user: bool, ing_type: str) -> int:
        """Compute the next `enc_id` for a new ingredient of a given subclass.

        Args:
            is_user (bool): Whether the new ingredient is user-created (`True`) or developer-created (`False`).
            ing_type (str): The subclass name (e.g., "Protein", "Buffer").

        Returns:
            int: The next available `enc_id` for this subclass and user/dev category.

        Raises:
            RuntimeError: If no developer `enc_id` slots remain (for developer-created ingredients).
        """
        same_type = self._fetch_by_type(type=ing_type)
        if is_user:
            # User-created enc_id must start at USER_START_ID
            used = [ing.enc_id for ing in same_type if ing.enc_id >=
                    self.USER_START_ID]
            if not used:
                return self.USER_START_ID
            else:
                return max(used) + 1
        else:
            # Developer-created enc_id must be in range [1..DEV_MAX_ID]
            used = [ing.enc_id for ing in same_type if 1 <=
                    ing.enc_id <= self.DEV_MAX_ID]
            if not used:
                return 1
            next_id = max(used) + 1
            if next_id > self.DEV_MAX_ID:
                raise RuntimeError(
                    f"No developer enc_id available for type '{ing_type}' (1..{self.DEV_MAX_ID} exhausted)."
                )
            return next_id
