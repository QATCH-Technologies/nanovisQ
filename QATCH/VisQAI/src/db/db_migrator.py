"""
database_migration.py

A comprehensive database migration system for handling version upgrades,
data preservation, and automatic field population for the QATCH database.

Author: Paul MacNichol
Date: 2025-08-27
Version: 1.1
"""

import sqlite3
import json
import shutil
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from pathlib import Path
from datetime import datetime
import hashlib
import traceback
from dataclasses import dataclass
from enum import Enum


class MigrationStatus(Enum):
    """Enumeration of possible migration operation states.

    This enum represents the lifecycle of a database migration,
    from initialization to completion or rollback.

    Attributes:
        PENDING (str): Migration has been registered but not yet started.
        IN_PROGRESS (str): Migration is currently being applied.
        COMPLETED (str): Migration was successfully applied.
        FAILED (str): Migration encountered an error and did not complete.
        ROLLED_BACK (str): Migration changes were reverted after failure or manual rollback.
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class MigrationVersion:
    """Represents a semantic version for database migrations.

    A version is composed of three components: major, minor, and patch.
    Comparison operators are implemented to allow version ordering and equality
    checks, making it easier to determine migration paths.

    Attributes:
        major (int): The major version number, incremented for incompatible changes.
        minor (int): The minor version number, incremented for backward-compatible changes.
        patch (int): The patch version number, incremented for bug fixes or small updates.
    """

    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        """Return the version as a string in `major.minor.patch` format.

        Returns:
            str: String representation of the version.
        """
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other) -> bool:
        """Check if this version is less than another.

        Args:
            other (MigrationVersion): The version to compare against.

        Returns:
            bool: True if this version is less than ``other``, False otherwise.
        """
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __gt__(self, other) -> bool:
        """Check if this version is greater than another.

        Args:
            other (MigrationVersion): The version to compare against.

        Returns:
            bool: True if this version is greater than ``other``, False otherwise.
        """
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)

    def __eq__(self, other) -> bool:
        """Check if this version is equal to another.

        Args:
            other (MigrationVersion): The version to compare against.

        Returns:
            bool: True if the versions are equal, False otherwise.
        """
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __le__(self, other) -> bool:
        """Check if this version is less than or equal to another.

        Args:
            other (MigrationVersion): The version to compare against.

        Returns:
            bool: True if this version is less than or equal to ``other``,
            False otherwise.
        """
        return self < other or self == other

    def __ge__(self, other) -> bool:
        """Check if this version is greater than or equal to another.

        Args:
            other (MigrationVersion): The version to compare against.

        Returns:
            bool: True if this version is greater than or equal to ``other``,
            False otherwise.
        """
        return self > other or self == other

    def __hash__(self) -> int:
        """Compute a hash value for this version.

        Returns:
            int: Hash value derived from the version components.
        """
        return hash((self.major, self.minor, self.patch))

    @classmethod
    def from_string(cls, version_str: str) -> 'MigrationVersion':
        """Create a ``MigrationVersion`` from a string.

        The string should follow semantic versioning format:
        ``"major.minor.patch"``. Missing components default to 0.

        Args:
            version_str (str): Version string, e.g., ``"1.2.3"``.

        Returns:
            MigrationVersion: Parsed version instance.

        Raises:
            ValueError: If the version string contains non-numeric components.

        Examples:
            >>> MigrationVersion.from_string("1.2.3")
            MigrationVersion(major=1, minor=2, patch=3)
            >>> MigrationVersion.from_string("2.0")
            MigrationVersion(major=2, minor=0, patch=0)
        """
        parts = version_str.split('.')
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0
        )


@dataclass
class Migration:
    """Represents a single database migration step.

    Each migration defines a transition from one version to another, including
    the SQL statements to apply (upgrade) or revert (rollback) the changes.
    Optional data transformations and default value autofills can also be provided.

    Attributes:
        from_version (MigrationVersion): The starting version of the database.
        to_version (MigrationVersion): The target version after applying this migration.
        up_sql (List[str]): SQL statements to apply the migration (upgrade).
        down_sql (List[str]): SQL statements to revert the migration (rollback).
        data_transform (Optional[Callable]): Optional function to transform data during migration.
        autofill_defaults (Dict[str, Any]): Default values for new fields added during migration.
        description (str): Human-readable description of the migration step.
    """

    from_version: MigrationVersion
    to_version: MigrationVersion
    up_sql: List[str]  # SQL statements for upgrade
    down_sql: List[str]  # SQL statements for rollback
    data_transform: Optional[Callable] = None
    autofill_defaults: Dict[str, Any] = None
    description: str = ""

    def __post_init__(self):
        """Initialize default values for optional attributes after object creation.

        Specifically, ensures that `autofill_defaults` is an empty dictionary
        if not provided by the user.
        """
        if self.autofill_defaults is None:
            self.autofill_defaults = {}


class DatabaseMigrator:
    """Manage database schema migrations for SQLite databases.

    The `DatabaseMigrator` class provides a full-featured system for
    managing, applying, and rolling back database migrations. It
    tracks schema versions, maintains migration history, supports
    data transformations, and validates database integrity.

    Attributes:
        db_path (Path): Path to the SQLite database file.
        backup_dir (Path): Directory where database backups are stored.
        migrations (Dict[Tuple[MigrationVersion, MigrationVersion], Migration]):
            Mapping of migration transitions to `Migration` objects.
        migration_graph (Dict[MigrationVersion, List[MigrationVersion]]):
            Adjacency list representing migration paths between versions.

    Methods:
        __init__(db_path, backup_dir=None):
            Initialize the migrator, setup backup directory, and register built-in migrations.
        _init_migration_table():
            Create tracking tables `schema_migrations` and `database_metadata`.
        register_migration(migration):
            Register a migration and update the migration graph.
        get_current_version() -> Optional[MigrationVersion]:
            Get the current database version.
        find_migration_path(from_version, to_version) -> List[Migration]:
            Find the optimal migration path between two versions.
        create_backup(suffix="") -> Path:
            Create a timestamped backup of the current database.
        apply_migration(migration, conn):
            Apply a single migration, including SQL, defaults, and transformations.
        migrate(target_version=None, dry_run=False, create_backup=True) -> Tuple[bool, List[str]]:
            Migrate the database to a target version, optionally dry-run or backup.
        rollback(target_version) -> Tuple[bool, List[str]]:
            Rollback the database to a previous version.
        get_migration_history() -> List[Dict[str, Any]]:
            Retrieve the migration history of the database.
        validate_database() -> Tuple[bool, List[str]]:
            Validate database schema and integrity, reporting missing tables or orphaned records.

    Example:
        >>> migrator = DatabaseMigrator("mydb.sqlite")
        >>> success, logs = migrator.migrate()
        >>> if success:
        ...     print("Migration completed successfully")
        >>> else:
        ...     print("Migration failed")
        >>> for line in logs:
        ...     print(line)
    """

    _REQUIRED_TABLES = {
        'ingredient', 'protein', 'buffer', 'stabilizer',
        'surfactant', 'salt', 'formulation',
        'formulation_component', 'viscosity_profile'
    }

    def __init__(self, db_path: Union[str, Path], backup_dir: Optional[Path] = None):
        """Initialize the database migration system.

        Sets up the paths, creates backup directories if needed, initializes
        the migration tracking table, and registers any built-in migrations.

        Args:
            db_path (Union[str, Path]): Path to the SQLite database file.
            backup_dir (Optional[Path]): Directory where database backups will
                be stored. If not provided, defaults to a "backups" folder
                in the same directory as `db_path`.

        Attributes:
            db_path (Path): Absolute path to the database file.
            backup_dir (Path): Absolute path to the backup directory.
            migrations (Dict[Tuple[MigrationVersion, MigrationVersion], Migration]):
                Mapping of migration transitions to Migration objects.
            migration_graph (Dict[MigrationVersion, List[MigrationVersion]]):
                Adjacency list representing possible migration paths.

        """
        self.db_path = Path(db_path)
        self.backup_dir = backup_dir or self.db_path.parent / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.migrations: Dict[Tuple[MigrationVersion,
                                    MigrationVersion], Migration] = {}
        self.migration_graph: Dict[MigrationVersion,
                                   List[MigrationVersion]] = {}
        self._init_migration_table()
        self._register_builtin_migrations()

    def _init_migration_table(self) -> None:
        """Initialize the database migration tracking tables.

        Creates the necessary tables to track applied migrations and
        store database metadata if they do not already exist. Specifically:

        - `schema_migrations`: Records each migration applied, including
        version, timestamp, status, rollback SQL, and any error messages.
        - `database_metadata`: Stores key-value metadata about the database,
        such as current version, with timestamps for updates.

        This method is typically called during the initialization of
        `DatabaseMigrator` to ensure migration infrastructure exists.

        Raises:
            sqlite3.DatabaseError: If there is an error creating the tables.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    checksum TEXT,
                    status TEXT DEFAULT 'completed',
                    rollback_sql TEXT,
                    error_message TEXT
                )
            """)

            # Add metadata table for version info
            conn.execute("""
                CREATE TABLE IF NOT EXISTS database_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _register_builtin_migrations(self) -> None:
        """Register known migration paths for the database schema.

        Currently no built-in migrations are registered.

        To add a migration:
        1. Define a Migration object with:
            - from_version (MigrationVersion)
            - to_version (MigrationVersion)
            - up_sql (list of SQL statements to apply schema changes)
            - down_sql (list of SQL statements to revert changes)
            - autofill_defaults (dict of default values for new columns)
            - optional: data_transform(conn) for data fixes/transformations
            - description (human-readable explanation)

        2. Register it with:
            self.register_migration(Migration(...))

        Example skeleton:
            self.register_migration(Migration(
                from_version=MigrationVersion(1, 0, 0),
                to_version=MigrationVersion(1, 1, 0),
                up_sql=["ALTER TABLE example ADD COLUMN notes TEXT"],
                down_sql=["ALTER TABLE example DROP COLUMN notes"],
                autofill_defaults={'notes': ''},
                description="Add notes field to example table"
            ))
        """
        pass

    def register_migration(self, migration: Migration) -> None:
        """Register a database migration in the system.

        Adds a `Migration` object to the internal registry and updates
        the migration graph to track available migration paths.

        Args:
            migration (Migration): The migration object to register.

        Side Effects:
            - Adds the migration to `self.migrations` keyed by
            `(from_version, to_version)`.
            - Updates `self.migration_graph` to include the new
            path from `from_version` to `to_version`.

        Example:
            >>> migrator = DatabaseMigrator("mydb.sqlite")
            >>> migration = Migration(
            ...     from_version=MigrationVersion(1, 0, 0),
            ...     to_version=MigrationVersion(1, 1, 0),
            ...     up_sql=["ALTER TABLE users ADD COLUMN email TEXT;"],
            ...     down_sql=["ALTER TABLE users DROP COLUMN email;"]
            ... )
            >>> migrator.register_migration(migration)
        """
        key = (migration.from_version, migration.to_version)
        self.migrations[key] = migration

        # Build migration graph
        if migration.from_version not in self.migration_graph:
            self.migration_graph[migration.from_version] = []
        self.migration_graph[migration.from_version].append(
            migration.to_version)

    def get_current_version(self) -> Optional[MigrationVersion]:
        """Retrieve the current version of the database.

        Attempts to determine the current database version by checking:
        1. The `database_metadata` table for a key named `'version'`.
        2. The most recently completed migration in the `schema_migrations` table.
        3. Defaults to version `1.0.0` if no version information is available.

        Returns:
            Optional[MigrationVersion]: The current database version, or
            `MigrationVersion(1, 0, 0)` if no version info exists.

        Notes:
            - Handles `sqlite3.OperationalError` in case tables do not exist yet.
            - Ensures that the database always has a version to work from
            even if migrations have not been applied.

        Example:
            >>> migrator = DatabaseMigrator("mydb.sqlite")
            >>> current_version = migrator.get_current_version()
            >>> print(current_version)
            1.0.0
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()

            # Try to get version from metadata table
            try:
                cursor.execute(
                    "SELECT value FROM database_metadata WHERE key = 'version'"
                )
                result = cursor.fetchone()
                if result:
                    return MigrationVersion.from_string(result[0])
            except sqlite3.OperationalError:
                pass

            # Fallback: Check migration history
            try:
                cursor.execute(
                    "SELECT version FROM schema_migrations "
                    "WHERE status = 'completed' "
                    "ORDER BY applied_at DESC LIMIT 1"
                )
                result = cursor.fetchone()
                if result:
                    return MigrationVersion.from_string(result[0])
            except sqlite3.OperationalError:
                pass

            # If no version info, assume base version
            return MigrationVersion(1, 0, 0)

        finally:
            conn.close()

    def find_migration_path(
        self,
        from_version: MigrationVersion,
        to_version: MigrationVersion
    ) -> List[Migration]:
        """Determine the optimal sequence of migrations from one version to another.

        Uses a breadth-first search (BFS) on the migration graph to find the
        shortest path of migrations needed to upgrade or downgrade the database
        from `from_version` to `to_version`.

        Args:
            from_version (MigrationVersion): The starting version of the database.
            to_version (MigrationVersion): The target version to migrate to.

        Returns:
            List[Migration]: Ordered list of `Migration` objects to apply
            sequentially to reach the target version. Returns an empty list
            if `from_version` is the same as `to_version`.

        Raises:
            ValueError: If no valid migration path exists between the two versions.

        Example:
            >>> migrator = DatabaseMigrator("mydb.sqlite")
            >>> path = migrator.find_migration_path(
            ...     MigrationVersion(1, 0, 0),
            ...     MigrationVersion(1, 2, 0)
            ... )
            >>> for migration in path:
            ...     print(migration.from_version, "->", migration.to_version)
            1.0.0 -> 1.1.0
            1.1.0 -> 1.2.0
        """
        if from_version == to_version:
            return []

        # BFS to find path
        from collections import deque
        queue = deque([(from_version, [])])
        visited = {from_version}

        while queue:
            current_version, path = queue.popleft()

            if current_version in self.migration_graph:
                for next_version in self.migration_graph[current_version]:
                    if next_version == to_version:
                        # Found the target
                        migration = self.migrations[(
                            current_version, next_version)]
                        return path + [migration]

                    if next_version not in visited:
                        visited.add(next_version)
                        migration = self.migrations[(
                            current_version, next_version)]
                        queue.append((next_version, path + [migration]))

        raise ValueError(
            f"No migration path found from {from_version} to {to_version}")

    def create_backup(self, suffix: str = "") -> Path:
        """Create a timestamped backup of the current database.

        Copies the database file to the backup directory with an optional suffix,
        generates a checksum for verification, and stores backup metadata in a JSON file.

        Args:
            suffix (str): Optional suffix to append to the backup filename
                (default: empty string).

        Returns:
            Path: The full path to the created backup database file.

        Side Effects:
            - Copies the database file to `self.backup_dir`.
            - Creates a JSON metadata file alongside the backup containing:
            original path, backup timestamp, checksum, and current database version.

        Raises:
            FileNotFoundError: If the database file does not exist.
            IOError: If the file cannot be copied or metadata cannot be written.

        Example:
            >>> migrator = DatabaseMigrator("mydb.sqlite")
            >>> backup_path = migrator.create_backup(suffix="_pre_migration")
            >>> print(backup_path)
            Path("backups/mydb_20250827_121530_pre_migration.db")
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{self.db_path.stem}_{timestamp}{suffix}.db"
        backup_path = self.backup_dir / backup_name

        shutil.copy2(self.db_path, backup_path)

        # Create checksum for verification
        with open(backup_path, 'rb') as f:
            checksum = hashlib.md5(f.read()).hexdigest()

        # Store backup metadata
        metadata_path = backup_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'original_path': str(self.db_path),
                'backup_time': timestamp,
                'checksum': checksum,
                'version': str(self.get_current_version())
            }, f, indent=2)

        return backup_path

    def apply_migration(self, migration: Migration, conn: sqlite3.Connection) -> None:
        """Apply a single migration step to the database.

        Executes the SQL statements for upgrading the database schema, applies
        default values to new fields, runs optional data transformations, and
        records the migration in the tracking tables.

        Args:
            migration (Migration): The migration object containing upgrade SQL,
                rollback SQL, autofill defaults, and optional data transformation.
            conn (sqlite3.Connection): An open SQLite connection to use for applying
                the migration.

        Raises:
            sqlite3.DatabaseError: If executing any SQL statement fails.
            sqlite3.IntegrityError: If data constraints are violated during update.

        Notes:
            - This method assumes `conn` is already open and will commit changes.
            - Rollback handling is not performed here; it must be managed externally.

        Example:
            >>> conn = sqlite3.connect("mydb.sqlite")
            >>> migration = Migration(
            ...     from_version=MigrationVersion(1, 0, 0),
            ...     to_version=MigrationVersion(1, 1, 0),
            ...     up_sql=["ALTER TABLE users ADD COLUMN email TEXT;"],
            ...     down_sql=["ALTER TABLE users DROP COLUMN email;"]
            ... )
            >>> migrator.apply_migration(migration, conn)
        """
        cursor = conn.cursor()

        # Execute upgrade SQL
        for sql in migration.up_sql:
            cursor.execute(sql)

        # Apply autofill defaults for new fields
        if migration.autofill_defaults:
            for table_field, default_value in migration.autofill_defaults.items():
                if '.' in table_field:
                    table, field = table_field.split('.')
                else:
                    # Try to infer table from ALTER statements
                    for sql in migration.up_sql:
                        if 'ALTER TABLE' in sql and f'ADD COLUMN {table_field}' in sql:
                            table = sql.split('ALTER TABLE')[1].split()[0]
                            field = table_field
                            break
                    else:
                        continue

                if default_value is not None and default_value != 'CURRENT_TIMESTAMP':
                    # Update existing rows with default value
                    if isinstance(default_value, str):
                        cursor.execute(
                            f"UPDATE {table} SET {field} = ?", (default_value,))
                    else:
                        cursor.execute(
                            f"UPDATE {table} SET {field} = {default_value}")

        # Apply custom data transformation if provided
        if migration.data_transform:
            migration.data_transform(conn)

        # Record migration
        cursor.execute(
            "INSERT INTO schema_migrations (version, status, rollback_sql) VALUES (?, ?, ?)",
            (str(migration.to_version), MigrationStatus.COMPLETED.value,
             json.dumps(migration.down_sql))
        )

        # Update version in metadata
        cursor.execute(
            "INSERT OR REPLACE INTO database_metadata (key, value) VALUES ('version', ?)",
            (str(migration.to_version),)
        )

        conn.commit()

    def migrate(
        self,
        target_version: Optional[MigrationVersion] = None,
        dry_run: bool = False,
        create_backup: bool = True
    ) -> Tuple[bool, List[str]]:
        """Migrate the database to a target version.

        Determines the migration path from the current database version to the
        specified `target_version` (or the latest available version if None),
        optionally creates a backup, and applies the migrations in order.
        Supports dry-run mode for simulation without making changes.

        Args:
            target_version (Optional[MigrationVersion]): The desired version to migrate to.
                If None, migrates to the latest available version.
            dry_run (bool): If True, only simulate the migration without applying changes.
            create_backup (bool): If True, creates a backup of the database before applying
                any migrations.

        Returns:
            Tuple[bool, List[str]]: 
                - bool: True if migration succeeded (or dry-run completed), False on failure.
                - List[str]: Log messages describing migration steps, backups, and errors.

        Raises:
            sqlite3.DatabaseError: If any database operation fails outside of migration execution.
            ValueError: If no migration path exists between current and target versions.

        Notes:
            - Dry-run mode does not modify the database or create backups.
            - Migration failures are logged and stop further migration steps.
            - Foreign key constraints are enabled during migration.

        Example:
            >>> migrator = DatabaseMigrator("mydb.sqlite")
            >>> success, logs = migrator.migrate()
            >>> if success:
            ...     print("Migration completed successfully")
            >>> else:
            ...     print("Migration failed")
            >>> for line in logs:
            ...     print(line)
        """
        messages = []

        # Get current version
        current_version = self.get_current_version()
        messages.append(f"Current database version: {current_version}")

        # Determine target version
        if target_version is None:
            # Find latest version
            all_versions = set()
            for from_v, to_v in self.migrations.keys():
                all_versions.add(from_v)
                all_versions.add(to_v)
            if all_versions:
                target_version = max(all_versions)
            else:
                messages.append("No migrations available")
                return True, messages

        messages.append(f"Target version: {target_version}")

        # Find migration path
        try:
            migration_path = self.find_migration_path(
                current_version, target_version)
        except ValueError as e:
            messages.append(f"Error: {e}")
            return False, messages

        if not migration_path:
            messages.append("Database is already at target version")
            return True, messages

        messages.append(f"Migration path: {len(migration_path)} steps")
        for m in migration_path:
            messages.append(
                f"  - {m.from_version} -> {m.to_version}: {m.description}")

        if dry_run:
            messages.append("Dry run completed - no changes made")
            return True, messages

        # Create backup
        if create_backup:
            backup_path = self.create_backup(suffix=f"_pre_{target_version}")
            messages.append(f"Backup created: {backup_path}")

        # Apply migrations
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")

            for i, migration in enumerate(migration_path, 1):
                messages.append(f"Applying migration {i}/{len(migration_path)}: "
                                f"{migration.from_version} -> {migration.to_version}")
                try:
                    self.apply_migration(migration, conn)
                    messages.append(f"Migration completed successfully")
                except Exception as e:
                    messages.append(f" Migration failed: {e}")
                    conn.rollback()

                    # Record failure
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO schema_migrations (version, status, error_message) "
                        "VALUES (?, ?, ?)",
                        (str(migration.to_version),
                         MigrationStatus.FAILED.value, str(e))
                    )
                    conn.commit()

                    return False, messages

            messages.append(
                f"Successfully migrated to version {target_version}")
            return True, messages

        finally:
            conn.close()

    def rollback(self, target_version: MigrationVersion) -> Tuple[bool, List[str]]:
        """Rollback the database to a previous version.

        Executes the rollback SQL statements of migrations that are newer than
        the specified `target_version`, updates migration statuses, and sets
        the database version accordingly. Creates a backup before performing
        the rollback.

        Args:
            target_version (MigrationVersion): The version to rollback to.

        Returns:
            Tuple[bool, List[str]]: 
                - bool: True if rollback succeeded, False if an error occurred
                or rollback is not possible.
                - List[str]: Log messages detailing rollback steps, warnings,
                and backup information.

        Side Effects:
            - Creates a backup of the database prior to rollback in `self.backup_dir`.
            - Executes rollback SQL for all migrations newer than `target_version`.
            - Updates `schema_migrations` status to `ROLLED_BACK`.
            - Updates `database_metadata` to reflect the new current version.
            - Commits changes if successful, otherwise rolls back the transaction.

        Notes:
            - Rollback is only possible if the current version is higher than
            the `target_version`.
            - Warnings during execution of individual rollback statements
            are logged but do not stop the overall rollback process.

        Example:
            >>> migrator = DatabaseMigrator("mydb.sqlite")
            >>> success, logs = migrator.rollback(MigrationVersion(1, 1, 0))
            >>> if success:
            ...     print("Rollback completed successfully")
            >>> else:
            ...     print("Rollback failed")
            >>> for line in logs:
            ...     print(line)
        """
        messages = []
        current_version = self.get_current_version()

        if current_version <= target_version:
            messages.append(f"Cannot rollback: current version {current_version} "
                            f"is not newer than target {target_version}")
            return False, messages

        # Create backup before rollback
        backup_path = self.create_backup(
            suffix=f"_pre_rollback_{target_version}")
        messages.append(f"Backup created: {backup_path}")

        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()

            # Get rollback SQL from migration history
            cursor.execute(
                "SELECT version, rollback_sql FROM schema_migrations "
                "WHERE status = 'completed' AND version > ? "
                "ORDER BY applied_at DESC",
                (str(target_version),)
            )

            rollbacks = cursor.fetchall()

            for version, rollback_sql_json in rollbacks:
                messages.append(f"Rolling back version {version}")
                rollback_sql = json.loads(rollback_sql_json)

                for sql in rollback_sql:
                    try:
                        cursor.execute(sql)
                    except Exception as e:
                        messages.append(f"  Warning: {e}")

                # Update migration status
                cursor.execute(
                    "UPDATE schema_migrations SET status = ? WHERE version = ?",
                    (MigrationStatus.ROLLED_BACK.value, version)
                )

            # Update current version
            cursor.execute(
                "INSERT OR REPLACE INTO database_metadata (key, value) VALUES ('version', ?)",
                (str(target_version),)
            )

            conn.commit()
            messages.append(
                f"Successfully rolled back to version {target_version}")
            return True, messages

        except Exception as e:
            conn.rollback()
            messages.append(f"Rollback failed: {e}")
            return False, messages
        finally:
            conn.close()

    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Retrieve the migration history of the database.

        Returns a chronological list of all migrations that have been applied,
        including their version, timestamp, status, and any error messages.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing:
                - 'version' (str): The version of the migration.
                - 'applied_at' (str): Timestamp when the migration was applied.
                - 'status' (str): Status of the migration (e.g., COMPLETED, FAILED, ROLLED_BACK).
                - 'error_message' (Optional[str]): Error message if the migration failed.

        Example:
            >>> migrator = DatabaseMigrator("mydb.sqlite")
            >>> history = migrator.get_migration_history()
            >>> for record in history:
            ...     print(record['version'], record['status'], record['applied_at'])
            1.0.0 COMPLETED 2025-08-27 12:15:30
            1.1.0 FAILED 2025-08-27 12:20:12
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT version, applied_at, status, error_message "
                "FROM schema_migrations ORDER BY applied_at DESC"
            )

            history = []
            for row in cursor.fetchall():
                history.append({
                    'version': row[0],
                    'applied_at': row[1],
                    'status': row[2],
                    'error_message': row[3]
                })

            return history

        finally:
            conn.close()

    def validate_database(self) -> Tuple[bool, List[str]]:
        """Validate the database schema and integrity against expected standards.

        Performs a series of checks on the database, including the presence
        of required tables, foreign key integrity, and detection of orphaned
        records in critical tables.

        Returns:
            Tuple[bool, List[str]]:
                - bool: True if the database is valid, False if issues were found.
                - List[str]: A list of descriptive messages for any issues detected,
                such as missing tables, foreign key violations, or orphaned records.

        Notes:
            - Checks foreign key constraints and counts orphaned `formulation_component` entries.

        Example:
            >>> migrator = DatabaseMigrator("mydb.sqlite")
            >>> is_valid, issues = migrator.validate_database()
            >>> if is_valid:
            ...     print("Database schema is valid")
            >>> else:
            ...     print("Database issues found:", issues)
        """
        issues = []
        conn = sqlite3.connect(str(self.db_path))

        try:
            cursor = conn.cursor()

            # Check for required tables
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}

            required_tables = self._REQUIRED_TABLES

            missing_tables = required_tables - tables
            if missing_tables:
                issues.append(f"Missing tables: {', '.join(missing_tables)}")

            # Check foreign key integrity
            cursor.execute("PRAGMA foreign_key_check")
            fk_errors = cursor.fetchall()
            if fk_errors:
                issues.append(f"Foreign key violations: {len(fk_errors)}")

            # Check for orphaned records
            cursor.execute("""
                SELECT COUNT(*) FROM formulation_component fc
                LEFT JOIN ingredient i ON fc.ingredient_id = i.id
                WHERE i.id IS NULL
            """)
            orphaned = cursor.fetchone()[0]
            if orphaned > 0:
                issues.append(f"Orphaned component records: {orphaned}")

            return len(issues) == 0, issues

        finally:
            conn.close()


# Example usage
def example_usage():
    import os
    from db import Database

    db_path = os.path.join('assets', 'app.db')
    db = Database(db_path, parse_file_key=True)
    temp_path = db.create_temp_decrypt()
    migrator = DatabaseMigrator(temp_path)

    # Check current version
    current = migrator.get_current_version()
    print(f"Current version: {current}")
    new_migration = Migration(
        from_version=current,
        to_version=MigrationVersion(1, 1, 0),
        up_sql=[
            "ALTER TABLE users ADD COLUMN email TEXT;"
        ],
        down_sql=[
            "ALTER TABLE users DROP COLUMN email;"
        ],
        data_transform=lambda conn: conn.execute(
            "UPDATE users SET email = 'example@example.com' WHERE email IS NULL;"
        ),
        autofill_defaults={
            "users.email": "example@example.com"
        },
        description="Add email column to users table with default values"
    )
    migrator.register_migration(new_migration)
    # Try a dry-run
    success, messages = migrator.migrate(
        target_version=MigrationVersion(1, 1, 0), dry_run=True)
    for msg in messages:
        print(msg)

    migrator.register_migration()
    # Perform actual migration
    success, messages = migrator.migrate(
        target_version=MigrationVersion(1, 1, 0),
        create_backup=True
    )
    for msg in messages:
        print(msg)

    # Check migration history
    history = migrator.get_migration_history()
    for entry in history:
        print(f"{entry['version']}: {entry['status']} at {entry['applied_at']}")

    # Validate database
    is_valid, issues = migrator.validate_database()
    if is_valid:
        print("Database validation passed")
    else:
        print("Database validation issues:", issues)
    db.cleanup_temp_decrypt()


if __name__ == "__main__":
    example_usage()
