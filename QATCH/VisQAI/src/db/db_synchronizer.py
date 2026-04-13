"""
db_synchronizer.py

Database synchronization utilities for NanovisQ.

This module provides the ``DatabaseSynchronizer`` class, which compares a
bundled (reference) SQLite database against a local user database and
forward-migrates the local copy to match the bundled schema and seed data.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-25

Version:
    1.0
"""

import sqlite3
from pathlib import Path


try:
    TAG = "[DBSync (HEADLESS)]"
    from db import Database

    class Log:
        """Minimal logging shim used when running in headless mode."""

        @staticmethod
        def d(TAG, msg=""):
            """Print a DEBUG-level message.

            Args:
                TAG: Identifier string prepended to the message.
                msg: The message body to print.
            """
            print("DEBUG:", TAG, msg)

        @staticmethod
        def i(TAG, msg=""):
            """Print an INFO-level message.

            Args:
                TAG: Identifier string prepended to the message.
                msg: The message body to print.
            """
            print("INFO:", TAG, msg)

        @staticmethod
        def w(TAG, msg=""):
            """Print a WARNING-level message.

            Args:
                TAG: Identifier string prepended to the message.
                msg: The message body to print.
            """
            print("WARNING:", TAG, msg)

        @staticmethod
        def e(TAG, msg=""):
            """Print an ERROR-level message.

            Args:
                TAG: Identifier string prepended to the message.
                msg: The message body to print.
            """
            print("ERROR:", TAG, msg)

except (ModuleNotFoundError, ImportError):
    TAG = "[DBSync]"
    from QATCH.common.logger import Logger as Log
    from QATCH.VisQAI.src.db.db import Database


class DatabaseSynchronizer:
    """Synchronizes a local SQLite database against a bundled reference copy.

    All methods are static; this class is used as a namespace rather than
    instantiated directly.
    """

    @staticmethod
    def sync_on_launch(
        local_db: Database,
        bundled_db_path: Path,
        dry_run: bool = False,
    ) -> bool:
        """Synchronize the local database with the bundled reference database.

        Compares ``db_version`` metadata between the two databases.  When the
        bundled version is newer the local schema is migrated (new tables /
        columns / indexes added) and seed data is upserted via
        ``INSERT OR IGNORE``.

        Args:
            local_db: The open local ``Database`` instance to be updated.
            bundled_db_path: Filesystem path to the bundled (reference)
                database file.
            dry_run: When ``True``, schema and seed changes are previewed in
                the log but the transaction is rolled back without committing.

        Returns:
            ``True`` if a synchronization was committed, ``False`` if the
            database was already up to date, a dry run was performed, or an
            error occurred.
        """
        if not bundled_db_path.exists():
            Log.w(TAG, "Bundled database not found. Skipping sync.")
            return False
        bundled_db = Database(path=bundled_db_path, parse_file_key=True)
        temp_bundled_path = bundled_db.create_temp_decrypt()
        if not temp_bundled_path:
            Log.e(TAG, "Failed to create temporary decrypted bundled database.")
            return False

        c = local_db.conn.cursor()
        sync_occurred = False

        try:
            c.execute(f"ATTACH DATABASE '{temp_bundled_path}' AS bundled;")
            local_version = local_db.metadata.get("db_version", 0)
            bundled_version = bundled_db.metadata.get("db_version", 0)

            if bundled_version <= local_version:
                Log.i(TAG, f"Database is up to date (Version {local_version}).")
                return False

            Log.i(TAG, f"Updating database from version {local_version} to {bundled_version}...")
            DatabaseSynchronizer._sync_schema(c)

            local_db.begin_bulk()
            DatabaseSynchronizer._sync_seed_data(c)
            local_db.end_bulk()

            local_db.update_metadata_version(bundled_version)
            if dry_run:
                Log.i(TAG, "DRY RUN Schema and seed changes previewed - no commit performed.")
                local_db.conn.rollback()
                return False

            local_db._commit()

            # If the local database uses encryption, force a disk backup
            if local_db.use_encryption:
                local_db.backup()

            sync_occurred = True
            Log.i(TAG, "Database synchronization complete.")

        except Exception as e:
            Log.e(TAG, f"Error during database synchronization: {e}")
            local_db.conn.rollback()
        finally:
            # 9. Cleanup
            try:
                c.execute("DETACH DATABASE bundled;")
            except sqlite3.OperationalError:
                pass  # Already detached or failed to attach
            bundled_db.cleanup_temp_decrypt(temp_bundled_path)
            bundled_db.close()

        return sync_occurred

    @staticmethod
    def _sync_schema(c: sqlite3.Cursor):
        """Apply DDL migrations from the bundled schema to the local database.

        Iterates over every non-system table in the bundled database and:

        * Creates the table in ``main`` if it does not exist.
        * Adds any columns that are present in ``bundled`` but missing from
          ``main`` via ``ALTER TABLE … ADD COLUMN``.
        * Creates any non-system indexes that exist in ``bundled`` but are
          absent from ``main``.

        All ``CREATE TABLE`` and ``ALTER TABLE`` statements are collected first
        and executed in a second pass so that the cursor is not advanced while
        iterating over the result set.  Index synchronization runs after all
        table and column migrations have been applied.

        Args:
            c: An active SQLite cursor connected to ``main`` with the bundled
                database attached as ``bundled``.
        """
        c.execute(
            "SELECT name, sql FROM bundled.sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        bundled_tables = c.fetchall()
        migrations = []  # Collect all DDL first

        for table_name, table_sql in bundled_tables:
            c.execute(
                "SELECT name FROM main.sqlite_master WHERE type='table' AND name=?;",
                (table_name,),
            )
            if not c.fetchone():
                migrations.append((f"Creating table {table_name}", table_sql))
                continue

            c.execute(f"PRAGMA main.table_info('{table_name}');")
            main_cols = {row[1] for row in c.fetchall()}
            c.execute(f"PRAGMA bundled.table_info('{table_name}');")
            bundled_cols = {row[1]: row for row in c.fetchall()}

            for col in set(bundled_cols.keys()) - main_cols:
                col_data = bundled_cols[col]
                col_name = col_data[1]
                col_type = col_data[2]
                notnull = col_data[3]
                dflt_value = col_data[4]
                pk = col_data[5]

                if pk:
                    Log.w(
                        TAG,
                        f"Column '{col_name}' in '{table_name}' is a PRIMARY KEY; "
                        "SQLite cannot add PRIMARY KEY via ALTER TABLE — skipping PK constraint.",
                    )

                col_def = f"{col_name} {col_type}"
                if dflt_value is not None:
                    # Quote string literals; leave numeric/keyword literals bare
                    if isinstance(dflt_value, str) and not dflt_value.upper() in (
                        "NULL", "TRUE", "FALSE", "CURRENT_TIME", "CURRENT_DATE",
                        "CURRENT_TIMESTAMP",
                    ) and not dflt_value.lstrip("-").replace(".", "", 1).isdigit():
                        escaped = dflt_value.replace("'", "''")
                        col_def += f" DEFAULT '{escaped}'"
                    else:
                        col_def += f" DEFAULT {dflt_value}"
                elif notnull:
                    Log.w(
                        TAG,
                        f"Column '{col_name}' in '{table_name}' is NOT NULL but has no "
                        "DEFAULT; SQLite disallows adding NOT NULL columns without a DEFAULT — "
                        "omitting NOT NULL constraint.",
                    )

                if notnull and dflt_value is not None:
                    col_def += " NOT NULL"

                migrations.append(
                    (
                        f"Adding column '{col_name}' to '{table_name}'",
                        f"ALTER TABLE main.{table_name} ADD COLUMN {col_def};",
                    )
                )
        for description, sql in migrations:
            Log.i(TAG, description)
            c.execute(sql)

        c.execute(
            "SELECT name, sql FROM bundled.sqlite_master "
            "WHERE type='index' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%';"
        )
        bundled_indexes = c.fetchall()

        for index_name, index_sql in bundled_indexes:
            c.execute(
                "SELECT name FROM main.sqlite_master WHERE type='index' AND name=?;",
                (index_name,),
            )
            if not c.fetchone():
                Log.i(TAG, f"Creating missing index: {index_name}")
                c.execute(index_sql)

    @staticmethod
    def _sync_seed_data(c: sqlite3.Cursor):
        """Upsert bundled seed data into the local database.

        Uses ``INSERT OR IGNORE`` so that existing local rows are preserved.
        The sync order is:

        1. ``ingredient`` rows with ``enc_id < 8000`` (core / system
           ingredients only; user-created ingredients are skipped).
        2. All other non-system, non-formulation tables that exist in the
           bundled database (derived dynamically).
        3. ``formulation``, ``formulation_component``, and
           ``viscosity_profile`` tables, in that order.

        Args:
            c: An active SQLite cursor connected to ``main`` with the bundled
                database attached as ``bundled``.
        """
        # Sync core ingredients
        c.execute(
            "INSERT OR IGNORE INTO main.ingredient "
            "SELECT * FROM bundled.ingredient WHERE enc_id < 8000;"
        )

        # Derive subclass tables dynamically — any table with an ingredient_id FK
        c.execute(
            "SELECT name FROM bundled.sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' AND name != 'ingredient' "
            "AND name NOT IN ('formulation', 'formulation_component', 'viscosity_profile');"
        )
        subclass_tables = [row[0] for row in c.fetchall()]
        for table in subclass_tables:
            c.execute(f"INSERT OR IGNORE INTO main.{table} SELECT * FROM bundled.{table};")

        # Sync formulation data
        for table in ("formulation", "formulation_component", "viscosity_profile"):
            c.execute(f"INSERT OR IGNORE INTO main.{table} SELECT * FROM bundled.{table};")
