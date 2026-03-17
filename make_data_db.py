import json
import os
import random
import time
import uuid
from typing import List, Optional, Union

import pandas as pd
from rapidfuzz import fuzz, process

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

database = Database(path=DB_PATH, encryption_key=metadata.get("app_key", None))
ing_ctrl = IngredientController(db=database)
ing_ctrl._user_mode = False

# Populate DB with core training samples
print(f"Adding core training samples to newly created app.db...")
form_ctrl = FormulationController(db=database)
csv_path = os.path.join(
    QATCH_ROOT,  # DO NOT COMMIT THIS CSV FILE
    "VisQAI",
    "assets",
    "formulation_data_02052026.csv",
)
<<<<<<< HEAD
logger = logging.getLogger("DB_MAKER")

ADMIN_SPACE_LIMIT = IngredientController.DEV_MAX_ID
SHEAR_RATES = [100, 1000, 10000, 100000, 15000000]
SOURCE_CSV = "formulation_data_03042026.csv"
=======
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")
>>>>>>> c8b8db9a73c06821c07e683989b6114d95b0f143


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
        df[col] = pd.to_numeric(df[col], errors="coerce")
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
                score_cutoff=0.5,
            )
            best_match_class = [match_name for match_name, score, idx in matches][0]
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
    if seed == ord("\n"):
        seed += 1  # seed 10 cannot be used, it breaks metadata parsing
    random.seed(seed)
    indices = list(range(len(text)))
    random.shuffle(indices)
    shuffled = "".join(text[i] for i in indices)
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
    enc_metadata = f.readline().decode(app_encoding).rsplit("\n", 1)[0]
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

<<<<<<< HEAD
def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-processes the source DataFrame to enforce data types and valid enumerations.

    Ensures numeric columns are floats and fuzzy-matches 'Protein_class_type'
    values to valid ProteinClass enum members.

    Args:
        df (pd.DataFrame): The raw source DataFrame.

    Returns:
        pd.DataFrame: The normalized DataFrame ready for import.
    """
    float_cols = ["MW", "PI_mean", "PI_range"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_classes = ProteinClass.all_strings()
    if "Protein_class_type" in df.columns:
        normalized_classes = []
        for val in df["Protein_class_type"]:
            # Preserve NaN/null values as-is — these represent rows with no protein
            # and must not be fuzzy-matched (str(NaN) == "nan" scores above cutoff
            # against enum members like "Polyclonal", producing incorrect assignments).
            if pd.isna(val):
                normalized_classes.append(None)
                continue
            val_str = str(val)
            if val_str not in valid_classes:
                match = process.extractOne(
                    query=val_str,
                    choices=valid_classes,
                    scorer=fuzz.WRatio,
                    score_cutoff=50,
                )
                if match:
                    normalized_classes.append(match[0])
                else:
                    normalized_classes.append(ProteinClass.OTHER.value)
            else:
                normalized_classes.append(val_str)
        df["Protein_class_type"] = normalized_classes

    # Fill NaN optional ingredient type columns with "none" to prevent the string
    # "nan" (produced by str(float('nan'))) from being stored as an ingredient name.
    # Affects poly-hIgG rows that have empty cells for unused ingredient slots.
    optional_ing_cols = [
        "Salt_type",
        "Stabilizer_type",
        "Surfactant_type",
        "Excipient_type",
    ]
    for col in optional_ing_cols:
        if col in df.columns:
            df[col] = df[col].fillna("none")

    if "icl" not in df.columns:
        df["icl"] = True  # Default ICL behavior is True

    if "last_model" not in df.columns:
        df["last_model"] = None  # Default is empty/None
    return df


def verify_database_integrity(
    db_path: Path, original_df: pd.DataFrame, expected_key: str
):
    """Performs rigorous integrity checks on the generated database.

    Verifies metadata keys, viscosity profile validity, ingredient uniqueness,
    ID range compliance, and ensures the stored data matches the source CSV.

    Args:
        db_path (Path): Path to the generated database file.
        original_df (pd.DataFrame): The original source DataFrame for comparison.
        expected_key (str): The expected application key for metadata verification.

    Raises:
        ValueError: If any integrity check fails.
    """
    logger.info("Running post-generation verification checks...")

    db = Database(path=db_path, parse_file_key=True)

    try:
        if db.metadata.get("app_key") != expected_key:
            raise ValueError(
                f"Metadata key mismatch! Expected {expected_key}, got {db.metadata.get('app_key')}"
            )

        form_ctrl = FormulationController(db)

        # Verify Viscosity Profiles
        all_forms = form_ctrl.get_all_formulations()
        for f in all_forms:
            if not f.viscosity_profile:
                raise ValueError(f"Formulation {f.id} has no viscosity profile.")

            viscs = f.viscosity_profile.viscosities
            if any(v <= 0 for v in viscs):
                logger.warning(
                    f"Formulation {f.id} has non-positive viscosities: {viscs}"
                )

        logger.info("Viscosity verification passed.")

        # Verify Ingredient Integrity
        ingredients = db.get_all_ingredients()
        seen = set()
        duplicates = []

        for ing in ingredients:
            if ing.enc_id >= ADMIN_SPACE_LIMIT:
                raise ValueError(
                    f"Ingredient '{ing.name}' has enc_id {ing.enc_id} which is outside admin space (<{ADMIN_SPACE_LIMIT})."
                )

            key = (ing.name, type(ing).__name__)
            if key in seen:
                duplicates.append(key)
            seen.add(key)

        if duplicates:
            raise ValueError(f"Duplicate ingredients found: {duplicates}")
        logger.info(
            f"Ingredient verification passed ({len(ingredients)} unique ingredients)."
        )

        # Verify DataFrame Equality
        db_df = form_ctrl.get_all_as_dataframe(encoded=False)

        dump_path = db_path.parent / "db_verification_export.csv"
        db_df.to_csv(dump_path, index=False)
        logger.info(f"Database dump saved to: {dump_path}")

        common_cols = [c for c in original_df.columns if c in db_df.columns]

        if "ID" in common_cols:
            common_cols.remove("ID")

        df_src = original_df[common_cols].copy()
        df_db = db_df[common_cols].copy()

        # Normalize types (force float) and strings (lowercase/strip) for comparison
        for col in common_cols:
            try:
                # Attempt numeric conversion first
                src_numeric = pd.to_numeric(df_src[col])
                db_numeric = pd.to_numeric(df_db[col])

                # Force float to ensure 100 == 100.0, handling NaNs as 0.0
                df_src[col] = src_numeric.fillna(0.0).astype(float)
                df_db[col] = db_numeric.fillna(0.0).astype(float)

            except (ValueError, TypeError):
                # Fallback to normalized strings
                df_src[col] = (
                    df_src[col]
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    .replace({"nan": "none", "none": "none"})
                )
                df_db[col] = (
                    df_db[col]
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    .replace({"nan": "none", "none": "none"})
                )

        # Deduplicate source data (DB handles duplicates on insertion)
        df_src_dedup = df_src.drop_duplicates()
        dedup_count = len(df_src) - len(df_src_dedup)

        if dedup_count > 0:
            logger.warning(
                f"Found {dedup_count} duplicate rows in source CSV (DB has unique). Using unique rows for check."
            )
            df_src = df_src_dedup

        if len(df_src) != len(df_db):
            logger.error(
                f"Row count mismatch: Source (Unique) {len(df_src)} vs DB {len(df_db)}"
            )
            raise ValueError(
                f"Row count mismatch: Source {len(df_src)} vs DB {len(df_db)}"
            )

        try:
            # Sort by stable columns to ensure alignment
            sort_candidates = [
                "Protein_type",
                "Buffer_type",
                "Protein_conc",
                "Temperature",
                "Viscosity_100",
            ]
            sort_cols = [c for c in sort_candidates if c in df_src.columns]

            df_src_sorted = df_src.sort_values(by=sort_cols).reset_index(drop=True)
            df_db_sorted = df_db.sort_values(by=sort_cols).reset_index(drop=True)

            pd.testing.assert_frame_equal(
                df_src_sorted,
                df_db_sorted,
                check_dtype=False,
                check_like=True,
                atol=1e-4,
            )
            logger.info("DataFrame verification passed: DB export matches source CSV.")
        except AssertionError as e:
            logger.error(f"DataFrame mismatch details: {e}")
            raise ValueError("DataFrame verification failed (content mismatch).")

    finally:
        db.close()


def main():
    """Main execution function for database generation."""
    db_path, csv_path = get_project_paths()
    logger.info(f"DB Path:  {db_path}")
    logger.info(f"CSV Path: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    app_key = uuid.uuid4().hex
    metadata = {
        "app_date": Constants.app_date,
        "app_encoding": Constants.app_encoding,
        "app_key": app_key,
        "app_name": "VisQ.AI",
        "app_publisher": "QATCH",
        "app_title": f"QATCH VisQ.AI Encrypted Database",
        "app_version": Constants.app_version,
    }

    if db_path.exists():
        logger.info("Removing existing database...")
        db_path.unlink()

    database = Database(path=db_path, encryption_key=app_key)
    ing_ctrl = IngredientController(db=database)
    ing_ctrl._user_mode = False
    form_ctrl = FormulationController(db=database)
    form_ctrl.ingredient_controller._user_mode = False
    logger.info("Reading and normalizing CSV...")
    df_normalized = normalize_dataframe(pd.read_csv(csv_path))

    logger.info(f"Adding {len(df_normalized)} samples to database...")
    form_ctrl.add_all_from_dataframe(df_normalized, verbose_print=True)

    database.close()

    # Inject Encrypted Metadata Header
    temp_db = Database(path=":memory:")
    temp_db.metadata = {"app_encoding": Constants.app_encoding}

    header_bytes = encrypt_metadata_header(metadata, temp_db)

    with open(db_path, "rb") as f:
        f.readline()  # Skip default header
        content = f.read()

    with open(db_path, "wb") as f:
        f.write(header_bytes)
        f.write(content)

    logger.info(f"Encrypted metadata injected (Seed: {header_bytes[0]}).")

    verify_database_integrity(db_path, df_normalized, app_key)

    logger.info("SUCCESS: Database generated and verified.")


if __name__ == "__main__":
    main()
=======
# Re-open database, and confirm metadata can be loaded properly
database = Database(path=DB_PATH, parse_file_key=True)
print("Loaded metadata:", database.metadata)
database.close()
>>>>>>> c8b8db9a73c06821c07e683989b6114d95b0f143
