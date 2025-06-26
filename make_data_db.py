import os
import uuid
import json
import time
import random

from typing import List, Optional, Union

from QATCH.core.constants import Constants
from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
from QATCH.VisQAI.src.db.db import Database
from QATCH.VisQAI.src.models.ingredient import Protein, Surfactant, Salt, Buffer, Stabilizer, Ingredient

app_date = Constants.app_date
app_encoding = Constants.app_encoding
app_key = uuid.uuid4().hex
app_name = "VisQ.AI"
app_publisher = "QATCH"
app_title = f"{app_publisher} {app_name} Encrypted Database"
app_version = Constants.app_version

metadata = {}
for var in sorted(locals()):
    if var.startswith("app"):
        metadata[var] = eval(var)
print("Gathered metadata:", metadata)

QATCH_ROOT = os.getcwd()
if os.path.basename(QATCH_ROOT) == "VisQAI":
    QATCH_ROOT = os.path.dirname(QATCH_ROOT)
elif os.path.isdir(os.path.join(QATCH_ROOT, "QATCH")):
    QATCH_ROOT = os.path.join(QATCH_ROOT, "QATCH")
DB_PATH = os.path.join(QATCH_ROOT, "VisQAI/assets/app.db")

print("Creating database file at:", DB_PATH)

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

database = Database(
    path=DB_PATH,
    encryption_key=metadata.get("app_key", None))
ing_ctrl = IngredientController(db=database)

core_ingredients = [
    # Buffers:
    Buffer(enc_id=0, name="None", pH=0.0),
    Buffer(enc_id=1, name="Accetate", pH=5.0),
    Buffer(enc_id=2, name="Histidine", pH=6.0),
    Buffer(enc_id=3, name="PBS", pH=7.4),
    # Surfactants:
    Surfactant(enc_id=0, name="None"),
    Surfactant(enc_id=1, name="Tween-20"),
    Surfactant(enc_id=2, name="Tween-80"),
    # Stabilizers:
    Stabilizer(enc_id=0, name="None"),
    Stabilizer(enc_id=1, name="Sucrose"),
    Stabilizer(enc_id=2, name="Trehalose"),
    # Salts:
    Salt(enc_id=0, name="None"),
    Salt(enc_id=1, name="NaCl")
]

# Set `is_user = False` and add each ingredient
for core in core_ingredients:
    ing: Ingredient = core
    ing.is_user = False
    ing_ctrl.add(ing)

database.close()  # creates file


def _shuffle_text(text: str, seed: Union[int, None] = None) -> tuple[str, int]:
    """Complementary method to `Database()._shuffle_text` return value.

    Args:
        text (str): The string to be shuffled (or unshuffled, perhaps).

    Returns:
        str: The character shuffled string.
        int: The seed for `random` to use (for repeatability).
    """
    if seed is None:
        milliseconds = int(round(time.time() * 1000))
        seed = milliseconds % 255
    if seed == ord('\n'):
        seed += 1  # seed 10 cannot be used, it breaks metadata parsing
    random.seed(seed)
    indices = list(range(len(text)))
    random.shuffle(indices)
    shuffled = ''.join(text[i] for i in indices)
    return shuffled, seed


with open(DB_PATH, "rb") as f:
    DB_METADATA = f.readline()  # trash
    DB_CONTENT = f.read()

with open(DB_PATH, "wb") as f:
    print("Encrypting metadata...")
    dumpster = json.dumps(metadata)
    print("Plaintext metadata:", dumpster)
    shuffled, seed = _shuffle_text(database._caesar_cipher(dumpster))
    # print(f"shuffled: '{shuffled}'")
    print("seed is", seed)
    DB_METADATA = "".join([chr(seed), shuffled])
    print("metadata length = ", len(dumpster))
    lines = b"\n".join([DB_METADATA.encode(), DB_CONTENT])
    f.write(lines)

with open(DB_PATH, "rb") as f:
    print("Decrypting metadata...")
    enc_metadata = f.readline().decode(app_encoding).rsplit('\n', 1)[0]
    print(f"Reading encrypted metadata: '{enc_metadata}'")
    seed = ord(enc_metadata[0])
    shuffled = enc_metadata[1:]
    # print("shuffled bytes:", shuffled)
    print("seed is", seed)
    enc_metadata = database._shuffle_text(shuffled, seed)
    print("metadata length = ", len(enc_metadata))
    str_metadata = database._caesar_cipher(enc_metadata, -len(enc_metadata))
    # print("str_metadata = ", str_metadata)
    loadster = json.loads(str_metadata)
    print("Decrypted metadata:", loadster)

# Re-open database, and confirm metadata can be loaded properly
database = Database(path=DB_PATH, parse_file_key=True)
print("Loaded metadata:", database.metadata)
database.close()
