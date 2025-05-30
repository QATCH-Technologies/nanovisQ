from src.db.db import Database
from src.models.ingredient import Protein, Salt, Stabilizer, Surfactant, Buffer, Ingredient
from typing import List, Union


class IngredientController:
    def __init__(self, db: Database) -> None:
        self.db: Database = db

    # ----- Top-Level Abstraction ----- #
    def get_all_ingredients(self) -> List[Ingredient]:
        return self.db.get_all_ingredients()

    def delete_all_ingredients(self):
        self.db.delete_all_ingredients()

    def get_by_id(self, id: int, ingredient: Ingredient) -> Ingredient:
        if ingredient.type == "Protein":
            return self.get_protein_by_id(id)
        elif ingredient.type == "Buffer":
            return self.get_buffer_by_id(id)
        elif ingredient.type == "Stabilizer":
            return self.get_stabilizer_by_id(id)
        elif ingredient.type == "Salt":
            return self.get_salt_by_id(id)
        elif ingredient.type == "Surfactant":
            return self.get_surfactant_by_id(id)
        else:
            raise ValueError(
                f"Ingredient type of '{ingredient.type}' not supported.")

    def get_by_name(self, name: str, ingredient: Ingredient) -> Ingredient:
        if ingredient.type == "Protein":
            return self.get_protein_by_name(name)
        elif ingredient.type == "Buffer":
            return self.get_buffer_by_name(name)
        elif ingredient.type == "Stabilizer":
            return self.get_stabilizer_by_name(name)
        elif ingredient.type == "Salt":
            return self.get_salt_by_name(name)
        elif ingredient.type == "Surfactant":
            return self.get_surfactant_by_name(name)
        else:
            raise ValueError(
                f"Ingredient type of '{ingredient.type}' not supported.")

    def get_by_type(self, ingredient: Ingredient) -> Ingredient:
        if ingredient.type == "Protein":
            return self.get_all_proteins()
        elif ingredient.type == "Buffer":
            return self.get_all_buffers()
        elif ingredient.type == "Stabilizer":
            return self.get_all_stabilizers()
        elif ingredient.type == "Salt":
            return self.get_all_salts()
        elif ingredient.type == "Surfactant":
            return self.get_all_surfactants()
        else:
            raise ValueError(
                f"Ingredient type of '{ingredient.type}' not supported.")

    def delete_by_id(self, id: int, ingredient: Ingredient):
        if ingredient.type == "Protein":
            return self.delete_protein_by_id(id)
        elif ingredient.type == "Buffer":
            return self.delete_buffer_by_id(id)
        elif ingredient.type == "Stabilizer":
            return self.delete_stabilizer_by_id(id)
        elif ingredient.type == "Salt":
            return self.delete_salt_by_id(id)
        elif ingredient.type == "Surfactant":
            return self.delete_surfactant_by_id(id)
        else:
            raise ValueError(
                f"Ingredient type of '{ingredient.type}' not supported.")

    def delete_by_name(self, name: str, ingredient: Ingredient):
        if ingredient.type == "Protein":
            return self.delete_protein_by_name(name)
        elif ingredient.type == "Buffer":
            return self.delete_buffer_by_name(name)
        elif ingredient.type == "Stabilizer":
            return self.delete_stabilizer_by_name(name)
        elif ingredient.type == "Salt":
            return self.delete_salt_by_name(name)
        elif ingredient.type == "Surfactant":
            return self.delete_surfactant_by_name(name)
        else:
            raise ValueError(
                f"Ingredient type of '{ingredient.type}' not supported.")

    def delete_by_type(self, ingredient: Ingredient):
        if ingredient.type == "Protein":
            return self.delete_all_proteins()
        elif ingredient.type == "Buffer":
            return self.delete_all_buffers()
        elif ingredient.type == "Stabilizer":
            return self.delete_all_stabilizers()
        elif ingredient.type == "Salt":
            return self.delete_all_salts()
        elif ingredient.type == "Surfactant":
            return self.delete_all_surfactants()
        else:
            raise ValueError(
                f"Ingredient type of '{ingredient.type}' not supported.")

    def add(self, ingredient: Ingredient) -> None:
        if ingredient.type == "Protein":
            return self.add_protein(ingredient)
        elif ingredient.type == "Buffer":
            return self.add_buffer(ingredient)
        elif ingredient.type == "Stabilizer":
            return self.add_stabilizer(ingredient)
        elif ingredient.type == "Salt":
            return self.add_salt(ingredient)
        elif ingredient.type == "Surfactant":
            return self.add_surfactant(ingredient)
        else:
            raise ValueError(
                f"Ingredient type of '{ingredient.type}' not supported.")

    def update(self, id: int, ingredient: Ingredient) -> None:
        if ingredient.type == "Protein":
            return self.update_protein(id, ingredient)
        elif ingredient.type == "Buffer":
            return self.update_buffer(id, ingredient)
        elif ingredient.type == "Stabilizer":
            return self.update_stabilizer(id, ingredient)
        elif ingredient.type == "Salt":
            return self.update_salt(id, ingredient)
        elif ingredient.type == "Surfactant":
            return self.update_surfactant(id, ingredient)
        else:
            raise ValueError(
                f"Ingredient type of '{ingredient.type}' not supported.")
    # ----- Acessors ----- #

    def get_protein_by_id(self, id: int) -> Protein:
        return self.db.get_ingredient(id)

    def get_protein_by_name(self, name: str) -> Protein:
        return self._fetch_by_name(name, type="Protein")

    def get_all_proteins(self) -> List[Protein]:
        return self._fetch_by_type("Protein")

    def get_buffer_by_id(self, id: int) -> Buffer:
        return self.db.get_ingredient(id)

    def get_buffer_by_name(self, name: str) -> Buffer:
        return self._fetch_by_name(name, type="Buffer")

    def get_all_buffers(self) -> List[Buffer]:
        return self._fetch_by_type("Buffer")

    def get_salt_by_id(self, id: int) -> Salt:
        return self.db.get_ingredient(id)

    def get_salt_by_name(self, name: str) -> Salt:
        return self._fetch_by_name(name, type="Salt")

    def get_all_salts(self) -> List[Salt]:
        return self._fetch_by_type("Salt")

    def get_surfactant_by_id(self, id: int) -> Surfactant:
        return self.db.get_ingredient(id)

    def get_surfactant_by_name(self, name: str) -> Surfactant:
        return self._fetch_by_name(name, type="Surfactant")

    def get_all_surfactants(self) -> List[Surfactant]:
        return self._fetch_by_type("Surfactant")

    def get_stabilizer_by_id(self, id: int) -> Stabilizer:
        return self.db.get_ingredient(id)

    def get_stabilizer_by_name(self, name: str) -> Stabilizer:
        return self._fetch_by_name(name, type="Stabilizer")

    def get_all_stabilizers(self) -> List[Stabilizer]:
        return self._fetch_by_type("Stabilizer")

    # ----- Creators ----- #

    def add_protein(self, protein: Protein) -> None:
        if not self.get_protein_by_name(protein.name) is None:
            raise ValueError(
                f"Protein with name '{protein.name}' already exists.")
        self.db.add_ingredient(protein)

    def add_buffer(self, buffer: Buffer) -> None:
        if not self.get_buffer_by_name(buffer.name) is None:
            raise ValueError(
                f"Buffer with name '{buffer.name}' already exists.")
        self.db.add_ingredient(buffer)

    def add_salt(self, salt: Salt) -> None:
        if not self.get_salt_by_name(salt.name) is None:
            raise ValueError(
                f"Salt with name '{salt.name}' already exists.")

        self.db.add_ingredient(salt)

    def add_stabilizer(self, stabilizer: Stabilizer) -> None:
        if not self.get_stabilizer_by_name(stabilizer.name) is None:
            raise ValueError(
                f"Stabilizer with name '{stabilizer.name}' already exists.")
        self.db.add_ingredient(stabilizer)

    def add_surfactant(self, surfactant: Surfactant) -> None:
        if not self.get_surfactant_by_name(surfactant.name) is None:
            raise ValueError(
                f"Surfactant with name '{surfactant.name}' already exists.")
        self.db.add_ingredient(surfactant)
    # ----- Deletion ----- #

    def delete_protein_by_id(self, id: int):
        if self.get_protein_by_id(id) is None:
            raise ValueError(f"Protein with id {id} does not exist.")
        self.db.delete_ingredient(id)

    def delete_protein_by_name(self, name: str):
        protein = self.get_protein_by_name(name)
        if protein is None:
            raise ValueError(f"Protein with name {name} does not exist.")
        self.db.delete_ingredient(protein.id)

    def delete_all_proteins(self):
        proteins = self.get_all_proteins()
        if proteins is None:
            raise ValueError(f"No items of type 'Protein' found.")
        for p in proteins:
            self.db.delete_ingredient(p.id)

    def delete_buffer_by_id(self, id: int):
        if self.get_buffer_by_id(id) is None:
            raise ValueError(f"Buffer with id {id} does not exist.")
        self.db.delete_ingredient(id)

    def delete_buffer_by_name(self, name: str):
        buffer = self.get_buffer_by_name(name)
        if buffer is None:
            raise ValueError(f"Buffer with name {name} does not exist.")
        self.db.delete_ingredient(buffer.id)

    def delete_all_buffers(self):
        buffers = self.get_all_buffers()
        if buffers is None:
            raise ValueError(f"No items of type 'Buffer' found.")
        for b in buffers:
            self.db.delete_ingredient(b.id)

    def delete_salt_by_id(self, id: int):
        if self.get_salt_by_id(id) is None:
            raise ValueError(f"Salt with id {id} does not exist.")
        self.db.delete_ingredient(id)

    def delete_salt_by_name(self, name: str):
        salt = self.get_salt_by_name(name)
        if salt is None:
            raise ValueError(f"Salt with name {name} does not exist.")
        self.db.delete_ingredient(salt.id)

    def delete_all_salts(self):
        salts = self.get_all_salts()
        if salts is None:
            raise ValueError(f"No items of type 'Salt' found.")
        for s in salts:
            self.db.delete_ingredient(s.id)

    def delete_surfactant_by_id(self, id: int):
        if self.get_surfactant_by_id(id) is None:
            raise ValueError(f"Surfactant with id {id} does not exist.")
        self.db.delete_ingredient(id)

    def delete_surfactant_by_name(self, name: str):
        surfactant = self.get_surfactant_by_name(name)
        if surfactant is None:
            raise ValueError(f"Surfactant with name {name} does not exist.")
        self.db.delete_ingredient(surfactant.id)

    def delete_all_surfactants(self):
        surfactants = self.get_all_surfactants()
        if surfactants is None:
            raise ValueError(f"No items of type 'Surfactant' found.")
        for s in surfactants:
            self.db.delete_ingredient(s.id)

    def delete_stabilizer_by_id(self, id: int):
        if self.get_stabilizer_by_id(id) is None:
            raise ValueError(f"Stabilizer with id {id} does not exist.")
        self.db.delete_ingredient(id)

    def delete_stabilizer_by_name(self, name: str):
        stabilizer = self.get_stabilizer_by_name(name)
        if stabilizer is None:
            raise ValueError(f"Stabilizer with name {name} does not exist.")
        self.db.delete_ingredient(stabilizer.id)

    def delete_all_stabilizers(self):
        stabilizers = self.get_all_stabilizers()
        if stabilizers is None:
            raise ValueError(f"No items of type 'Stabilizer' found.")
        for s in stabilizers:
            self.db.delete_ingredient(s.id)
    # ----- Mutators ----- #

    def update_protein(self, id: int, p_new: Protein) -> Protein:
        p_fetch = self.get_protein_by_id(id)
        if p_fetch is None:
            raise ValueError(f"Protein with id '{id}' does not exist.")
        if p_fetch == p_new:
            return p_new

        self.delete_protein_by_id(p_fetch.id)
        p_new.enc_id = p_fetch.enc_id
        self.add_protein(p_new)

    def update_buffer(self, id: int, b_new: Buffer):
        b_fetch = self.get_protein_by_id(id)
        if b_fetch is None:
            raise ValueError(f"Buffer with id '{id}' does not exist.")
        if b_fetch == b_new:
            return b_new

        self.delete_protein_by_id(b_fetch.id)
        b_new.enc_id = b_fetch.enc_id
        self.add_protein(b_new)

    def update_salt(self, id: int, s_new: Salt):
        s_fetch = self.get_protein_by_id(id)
        if s_fetch is None:
            raise ValueError(f"Salt with id '{id}' does not exist.")
        if s_fetch == s_new:
            return s_new

        self.delete_protein_by_id(s_fetch.id)
        s_new.enc_id = s_fetch.enc_id
        self.add_protein(s_new)

    def update_surfactant(self, id: int, s_new: Surfactant):
        s_fetch = self.get_protein_by_id(id)
        if s_fetch is None:
            raise ValueError(f"Surfactant with id '{id}' does not exist.")
        if s_fetch == s_new:
            return s_new

        self.delete_protein_by_id(s_fetch.id)
        s_new.enc_id = s_fetch.enc_id
        self.add_protein(s_new)

    def update_stabilizer(self, id: int, s_new: Stabilizer):
        s_fetch = self.get_protein_by_id(id)
        if s_fetch is None:
            raise ValueError(f"Stabilizer with id '{id}' does not exist.")
        if s_fetch == s_new:
            return s_new

        self.delete_protein_by_id(s_fetch.id)
        s_new.enc_id = s_fetch.enc_id
        self.add_protein(s_new)
    # ----- Helpers ----- #

    def _fetch_by_type(self, type: str) -> Union[List[Ingredient], None]:
        ingredients = self.db.get_all_ingredients()
        ingredients_of_type = []
        for ingredient in ingredients:
            if ingredient.type == type:
                ingredients_of_type.append(ingredient)
        if len(ingredients) == 0:
            return []
        return ingredients_of_type

    def _fetch_by_name(self, name: str, type: str) -> Ingredient:
        ingredients = self.db.get_all_ingredients()
        for ingredient in ingredients:
            if ingredient.type == type and ingredient.name == name:
                return ingredient
        return None
