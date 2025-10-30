import os
import uuid
import json
import time
import random

import pandas as pd

from typing import List, Optional, Union
from rapidfuzz import process, fuzz

from QATCH.core.constants import Constants
from QATCH.VisQAI.src.controller.formulation_controller import FormulationController
from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
from QATCH.VisQAI.src.db.db import Database
from QATCH.VisQAI.src.models.ingredient import ProteinClass

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
ing_ctrl._user_mode = False

# Populate DB with core training samples
print(f"Adding core training samples to newly created app.db...")
form_ctrl = FormulationController(db=database)
csv_path = os.path.join(QATCH_ROOT,  # DO NOT COMMIT THIS CSV FILE
                        "VisQAI", "assets", "formulation_data_10282025.csv")
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")


def _read_and_normalize_csv(path: str) -> pd.DataFrame:
    """Massage dataframe to an expected parsable format by:
        1. Removing undesired columns from the dataset
        2. Removing any blank/partial rows at end of file
        3. Enforcing that columns match a valid type (None vs NaN)
        4. Normalize protein class types to values from enum
    """
    df = pd.read_csv(csv_path)
    # Step 1: Removing undesired columns from the dataset
    # Step 2: Removing any blank/partial rows at end of file
    # Step 3: Enforcing that columns match a valid type (None vs NaN)
    float_cols = ["MW", "PI_mean", "PI_range"]
    for col in float_cols:
        # Convert to numeric, invalid values become NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Step 4: Normalize protein class types to values from enum
    valid_classes = ProteinClass.all_strings()
    given_classes = df["Protein_class_type"].tolist()
    for i in range(len(given_classes)):
        if given_classes[i] not in valid_classes:
            matches = process.extract(
                query=given_classes[i],
                choices=valid_classes,
                scorer=fuzz.WRatio,
                limit=1,
                score_cutoff=0.5
            )
            best_match_class = [
                match_name for match_name, score, idx in matches][0]
            if best_match_class not in valid_classes:
                best_match_class = ProteinClass.OTHER.value
            given_classes[i] = best_match_class
    df["Protein_class_type"] = given_classes
    return df


df = _read_and_normalize_csv(csv_path)
added_forms = form_ctrl.add_all_from_dataframe(df, verbose_print=True)
print(f"Added {len(added_forms)} core training samples!")

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
