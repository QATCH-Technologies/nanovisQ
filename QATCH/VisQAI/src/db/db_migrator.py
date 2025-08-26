"""
database_migration.py

A comprehensive database migration system for handling version upgrades,
data preservation, and automatic field population for the QATCH database.

Author: Paul MacNichol
Date: 2025-08-22
Version: 1.0
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
    """Status of a migration operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class MigrationVersion:
    """Represents a database version."""
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other) -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __gt__(self, other) -> bool:
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)

    def __eq__(self, other) -> bool:
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __le__(self, other) -> bool:
        return self < other or self == other

    def __ge__(self, other) -> bool:
        return self > other or self == other

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))

    @classmethod
    def from_string(cls, version_str: str) -> 'MigrationVersion':
        """Parse version string like '1.2.3' into MigrationVersion."""
        parts = version_str.split('.')
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0
        )


@dataclass
class Migration:
    """Represents a single migration step."""
    from_version: MigrationVersion
    to_version: MigrationVersion
    up_sql: List[str]  # SQL statements for upgrade
    down_sql: List[str]  # SQL statements for rollback
    # Custom data transformation function
    data_transform: Optional[Callable] = None
    autofill_defaults: Dict[str, Any] = None  # Default values for new fields
    description: str = ""

    def __post_init__(self):
        if self.autofill_defaults is None:
            self.autofill_defaults = {}


class DatabaseMigrator:
    """
    Manages database migrations with version tracking, rollback support,
    and automatic field population.
    """

    def __init__(self, db_path: Union[str, Path], backup_dir: Optional[Path] = None):
        """
        Initialize the migration system.

        Args:
            db_path: Path to the database file
            backup_dir: Directory for storing backups (default: db_path.parent/backups)
        """
        self.db_path = Path(db_path)
        self.backup_dir = backup_dir or self.db_path.parent / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.migrations: Dict[Tuple[MigrationVersion,
                                    MigrationVersion], Migration] = {}
        self.migration_graph: Dict[MigrationVersion,
                                   List[MigrationVersion]] = {}

        # Initialize migration tracking table
        self._init_migration_table()

        # Register built-in migrations
        self._register_builtin_migrations()

    def _init_migration_table(self) -> None:
        """Create migration tracking table if it doesn't exist."""
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
        """
        Register a migration in the system.

        Args:
            migration: Migration object to register
        """
        key = (migration.from_version, migration.to_version)
        self.migrations[key] = migration

        # Build migration graph
        if migration.from_version not in self.migration_graph:
            self.migration_graph[migration.from_version] = []
        self.migration_graph[migration.from_version].append(
            migration.to_version)

    def get_current_version(self) -> Optional[MigrationVersion]:
        """Get the current database version."""
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
        """
        Find the optimal migration path between two versions.
        Uses BFS to find shortest path in migration graph.

        Args:
            from_version: Starting version
            to_version: Target version

        Returns:
            List of migrations to apply in order
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
        """
        Create a backup of the current database.

        Args:
            suffix: Optional suffix for backup filename

        Returns:
            Path to backup file
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
        """
        Apply a single migration to the database.

        Args:
            migration: Migration to apply
            conn: Database connection to use
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
        """
        Migrate database to target version.

        Args:
            target_version: Target version (None = latest available)
            dry_run: If True, only simulate migration without applying
            create_backup: If True, create backup before migration

        Returns:
            Tuple of (success, list of messages)
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
        """
        Rollback database to a previous version.

        Args:
            target_version: Version to rollback to

        Returns:
            Tuple of (success, list of messages)
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
        """Get the migration history of the database."""
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
        """
        Validate database schema against expected version.

        Returns:
            Tuple of (is_valid, list of issues)
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

            required_tables = {
                'ingredient', 'protein', 'buffer', 'stabilizer',
                'surfactant', 'salt', 'formulation',
                'formulation_component', 'viscosity_profile'
            }

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


# Example usage function
def example_usage():
    """Example of how to use the DatabaseMigrationSystem."""

    # Initialize migration system
    migrator = DatabaseMigrator("path/to/database.db")

    # Check current version
    current = migrator.get_current_version()
    print(f"Current version: {current}")

    # Perform dry run
    success, messages = migrator.migrate(dry_run=True)
    for msg in messages:
        print(msg)

    # Perform actual migration
    success, messages = migrator.migrate(
        target_version=MigrationVersion(1, 4, 0),
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
