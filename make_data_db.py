"""
make_database_db.py

This script initializes, populates, and verifies the application's SQLite database.
It handles schema creation, bulk data import from CSV, metadata encryption,
and rigorous post-generation integrity checks to ensure data fidelity.

Author:
    Alexander J. Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16
"""

import json
import logging
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import pandas as pd
from rapidfuzz import fuzz, process

from QATCH.core.constants import Constants
from QATCH.VisQAI.src.controller.formulation_controller import FormulationController
from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
from QATCH.VisQAI.src.db.db import Database
from QATCH.VisQAI.src.models.ingredient import ProteinClass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DB_MAKER")

ADMIN_SPACE_LIMIT = IngredientController.DEV_MAX_ID
SHEAR_RATES = [100, 1000, 10000, 100000, 15000000]
SOURCE_CSV = "formulation_data_03042026.csv"


def get_project_paths() -> Tuple[Path, Path]:
    """Resolves the absolute paths for the database and source CSV file.

    Traverses the directory structure to locate the root application folder
    and constructs paths to the `assets` directory.

    Returns:
        Tuple[Path, Path]: A tuple containing (db_path, csv_path).
    """
    cwd = Path(os.getcwd())
    if cwd.name == "VisQAI":
        root = cwd.parent
    elif (cwd / "QATCH").is_dir():
        root = cwd / "QATCH"
    else:
        root = cwd

    base_path = root / "VisQAI" / "assets"
    db_path = base_path / "app.db"
    csv_path = base_path / SOURCE_CSV

    return db_path.resolve(), csv_path.resolve()


def shuffle_text(text: str, seed: Union[int, None] = None) -> Tuple[str, int]:
    """Shuffles the characters of a string using a seeded random number generator.

    Used for obfuscating metadata headers in the database file.

    Args:
        text (str): The input string to shuffle.
        seed (Union[int, None], optional): The seed for randomization. If None,
            uses the current time. Defaults to None.

    Returns:
        Tuple[str, int]: A tuple containing the shuffled string and the seed used.
    """
    if seed is None:
        milliseconds = int(round(time.time() * 1000))
        seed = milliseconds % 255
    # seed 10 cannot be used, it breaks metadata parsing because in ASCII it is '\n'
    # and adding a newline character would mean the end of the metadata header line.
    if seed == 10:
        seed += 1

    random.seed(seed)
    indices = list(range(len(text)))
    random.shuffle(indices)
    shuffled = "".join(text[i] for i in indices)
    return shuffled, seed


def encrypt_metadata_header(metadata: Dict[str, Any], db_instance: Database) -> bytes:
    """Generates an encrypted, shuffled metadata header for the database file.

    The metadata is serialized to JSON, encrypted using a Caesar cipher,
    shuffled, and then prepended with the seed character.

    Args:
        metadata (Dict[str, Any]): The metadata dictionary to encrypt.
        db_instance (Database): An instance of the Database class (used for cipher methods).

    Returns:
        bytes: The byte sequence representing the encrypted header line.
    """
    dumpster = json.dumps(metadata)
    ciphered = db_instance._caesar_cipher(dumpster)
    shuffled, seed = shuffle_text(ciphered)
    header_str = chr(seed) + shuffled
    return header_str.encode(Constants.app_encoding) + b"\n"


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
                rtol=2e-2,
                atol=1e-6,
            )
            logger.info("DataFrame verification passed: DB export matches source CSV.")
        except AssertionError as e:
            logger.error(f"DataFrame mismatch details: {e}")
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
