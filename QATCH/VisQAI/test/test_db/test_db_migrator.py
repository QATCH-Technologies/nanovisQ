"""
test_database_migration.py

Comprehensive unit tests for the DatabaseMigrator system.

Author: Test Suite
Date: 2025-08-22
Version: 1.0

Note: If the MigrationVersion class is missing comparison operators, add these to the class:
    def __le__(self, other) -> bool:
        return (self.major, self.minor, self.patch) <= (other.major, other.minor, other.patch)
    
    def __gt__(self, other) -> bool:
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)
    
    def __ge__(self, other) -> bool:
        return (self.major, self.minor, self.patch) >= (other.major, other.minor, other.patch)
"""

import unittest
import sqlite3
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, call
import hashlib

# Import the module to test
from src.db.db_migrator import (
    DatabaseMigrator,
    Migration,
    MigrationVersion,
    MigrationStatus
)


class TestMigrationVersion(unittest.TestCase):
    """Test the MigrationVersion dataclass."""

    def test_version_creation(self):
        """Test creating MigrationVersion instances."""
        version = MigrationVersion(1, 2, 3)
        self.assertEqual(version.major, 1)
        self.assertEqual(version.minor, 2)
        self.assertEqual(version.patch, 3)

    def test_version_string_representation(self):
        """Test string representation of version."""
        version = MigrationVersion(2, 5, 1)
        self.assertEqual(str(version), "2.5.1")

    def test_version_from_string(self):
        """Test parsing version from string."""
        version = MigrationVersion.from_string("1.2.3")
        self.assertEqual(version.major, 1)
        self.assertEqual(version.minor, 2)
        self.assertEqual(version.patch, 3)

        # Test with partial version strings
        version = MigrationVersion.from_string("2.1")
        self.assertEqual(version.major, 2)
        self.assertEqual(version.minor, 1)
        self.assertEqual(version.patch, 0)

        version = MigrationVersion.from_string("3")
        self.assertEqual(version.major, 3)
        self.assertEqual(version.minor, 0)
        self.assertEqual(version.patch, 0)

    def test_version_comparison(self):
        """Test version comparison operators."""
        v1 = MigrationVersion(1, 0, 0)
        v2 = MigrationVersion(1, 1, 0)
        v3 = MigrationVersion(2, 0, 0)
        v4 = MigrationVersion(1, 0, 0)

        # Test less than
        self.assertTrue(v1 < v2)
        self.assertTrue(v2 < v3)
        self.assertFalse(v3 < v1)

        # Test equality
        self.assertTrue(v1 == v4)
        self.assertFalse(v1 == v2)

        # Test less than or equal (which the rollback method uses)
        self.assertTrue(v1 <= v2)
        self.assertTrue(v1 <= v4)  # Equal case
        self.assertFalse(v3 <= v1)

        # Test greater than
        self.assertTrue(v3 > v1)
        self.assertTrue(v2 > v1)
        self.assertFalse(v1 > v3)

        # Test greater than or equal
        self.assertTrue(v3 >= v1)
        self.assertTrue(v1 >= v1)  # Equal case
        self.assertFalse(v1 >= v3)

    def test_version_hash(self):
        """Test version hashing for use in dictionaries."""
        v1 = MigrationVersion(1, 2, 3)
        v2 = MigrationVersion(1, 2, 3)
        v3 = MigrationVersion(3, 2, 1)

        self.assertEqual(hash(v1), hash(v2))
        self.assertNotEqual(hash(v1), hash(v3))

        # Test in dictionary
        version_dict = {v1: "test"}
        self.assertEqual(version_dict[v2], "test")


class TestDatabaseMigrator(unittest.TestCase):
    """Test the DatabaseMigrator class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.backup_dir = Path(self.temp_dir) / "backups"

        # Create a test database with initial schema
        self._create_test_database()

        # Initialize migrator
        self.migrator = DatabaseMigrator(self.db_path, self.backup_dir)

    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_database(self):
        """Create a test database with initial schema."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    value INTEGER
                )
            """)
            conn.execute("""
                INSERT INTO test_table (name, value) VALUES 
                ('item1', 100),
                ('item2', 200),
                ('item3', 300)
            """)
            conn.commit()
        finally:
            conn.close()

    def test_initialization(self):
        """Test migrator initialization."""
        self.assertTrue(self.db_path.exists())
        self.assertTrue(self.backup_dir.exists())

        # Check migration tracking tables were created
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
            self.assertIn('schema_migrations', tables)
            self.assertIn('database_metadata', tables)
        finally:
            conn.close()

    def test_register_migration(self):
        """Test registering migrations."""
        migration = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=["ALTER TABLE test_table ADD COLUMN status TEXT"],
            down_sql=["ALTER TABLE test_table DROP COLUMN status"]
        )

        self.migrator.register_migration(migration)

        key = (MigrationVersion(1, 0, 0), MigrationVersion(1, 1, 0))
        self.assertIn(key, self.migrator.migrations)
        self.assertEqual(self.migrator.migrations[key], migration)

        # Check migration graph
        self.assertIn(MigrationVersion(1, 0, 0), self.migrator.migration_graph)
        self.assertIn(
            MigrationVersion(1, 1, 0),
            self.migrator.migration_graph[MigrationVersion(1, 0, 0)]
        )

    def test_get_current_version(self):
        """Test getting current database version."""
        # Initial state - should return base version
        version = self.migrator.get_current_version()
        self.assertEqual(version, MigrationVersion(1, 0, 0))

        # Set a version in metadata
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                "INSERT INTO database_metadata (key, value) VALUES ('version', '2.3.1')"
            )
            conn.commit()
        finally:
            conn.close()

        version = self.migrator.get_current_version()
        self.assertEqual(version, MigrationVersion(2, 3, 1))

    def test_find_migration_path_simple(self):
        """Test finding a simple migration path."""
        # Register a linear migration path
        m1 = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=["SQL1"], down_sql=["SQL1_DOWN"]
        )
        m2 = Migration(
            from_version=MigrationVersion(1, 1, 0),
            to_version=MigrationVersion(1, 2, 0),
            up_sql=["SQL2"], down_sql=["SQL2_DOWN"]
        )

        self.migrator.register_migration(m1)
        self.migrator.register_migration(m2)

        path = self.migrator.find_migration_path(
            MigrationVersion(1, 0, 0),
            MigrationVersion(1, 2, 0)
        )

        self.assertEqual(len(path), 2)
        self.assertEqual(path[0], m1)
        self.assertEqual(path[1], m2)

    def test_find_migration_path_no_path(self):
        """Test when no migration path exists."""
        m1 = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=["SQL1"], down_sql=["SQL1_DOWN"]
        )
        self.migrator.register_migration(m1)

        with self.assertRaises(ValueError) as context:
            self.migrator.find_migration_path(
                MigrationVersion(1, 0, 0),
                MigrationVersion(2, 0, 0)
            )
        self.assertIn("No migration path found", str(context.exception))

    def test_find_migration_path_same_version(self):
        """Test migration path when already at target version."""
        path = self.migrator.find_migration_path(
            MigrationVersion(1, 0, 0),
            MigrationVersion(1, 0, 0)
        )
        self.assertEqual(path, [])

    def test_create_backup(self):
        """Test creating database backups."""
        backup_path = self.migrator.create_backup(suffix="_test")

        self.assertTrue(backup_path.exists())
        self.assertIn("_test.db", str(backup_path))

        # Check backup metadata file
        metadata_path = backup_path.with_suffix('.json')
        self.assertTrue(metadata_path.exists())

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            self.assertEqual(metadata['original_path'], str(self.db_path))
            self.assertIn('checksum', metadata)
            self.assertIn('backup_time', metadata)
            self.assertIn('version', metadata)

    def test_apply_migration_basic(self):
        """Test applying a basic migration."""
        migration = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=[
                "ALTER TABLE test_table ADD COLUMN status TEXT",
                "ALTER TABLE test_table ADD COLUMN created_at TIMESTAMP"
            ],
            down_sql=[
                "ALTER TABLE test_table DROP COLUMN status",
                "ALTER TABLE test_table DROP COLUMN created_at"
            ],
            description="Add status and timestamp fields"
        )

        conn = sqlite3.connect(str(self.db_path))
        try:
            self.migrator.apply_migration(migration, conn)

            # Check that columns were added
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(test_table)")
            columns = {row[1] for row in cursor.fetchall()}
            self.assertIn('status', columns)
            self.assertIn('created_at', columns)

            # Check migration was recorded
            cursor.execute(
                "SELECT version, status FROM schema_migrations WHERE version = ?",
                (str(migration.to_version),)
            )
            result = cursor.fetchone()
            self.assertIsNotNone(result)
            self.assertEqual(result[1], MigrationStatus.COMPLETED.value)

        finally:
            conn.close()

    def test_apply_migration_with_defaults(self):
        """Test applying migration with autofill defaults."""
        migration = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=["ALTER TABLE test_table ADD COLUMN status TEXT"],
            down_sql=["ALTER TABLE test_table DROP COLUMN status"],
            autofill_defaults={'status': 'active'}
        )

        conn = sqlite3.connect(str(self.db_path))
        try:
            self.migrator.apply_migration(migration, conn)

            # Check that default values were applied
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM test_table")
            statuses = [row[0] for row in cursor.fetchall()]
            self.assertTrue(all(s == 'active' for s in statuses))

        finally:
            conn.close()

    def test_apply_migration_with_transform(self):
        """Test applying migration with data transformation."""
        def double_values(conn):
            conn.execute("UPDATE test_table SET value = value * 2")

        migration = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=["ALTER TABLE test_table ADD COLUMN doubled BOOLEAN"],
            down_sql=["ALTER TABLE test_table DROP COLUMN doubled"],
            data_transform=double_values
        )

        conn = sqlite3.connect(str(self.db_path))
        try:
            self.migrator.apply_migration(migration, conn)

            # Check that values were doubled
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM test_table ORDER BY id")
            values = [row[0] for row in cursor.fetchall()]
            self.assertEqual(values, [200, 400, 600])

        finally:
            conn.close()

    def test_migrate_success(self):
        """Test successful migration to target version."""
        # Register migrations
        m1 = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=["ALTER TABLE test_table ADD COLUMN field1 TEXT"],
            down_sql=["ALTER TABLE test_table DROP COLUMN field1"],
            description="Add field1"
        )
        m2 = Migration(
            from_version=MigrationVersion(1, 1, 0),
            to_version=MigrationVersion(1, 2, 0),
            up_sql=["ALTER TABLE test_table ADD COLUMN field2 TEXT"],
            down_sql=["ALTER TABLE test_table DROP COLUMN field2"],
            description="Add field2"
        )

        self.migrator.register_migration(m1)
        self.migrator.register_migration(m2)

        # Perform migration
        success, messages = self.migrator.migrate(
            target_version=MigrationVersion(1, 2, 0),
            create_backup=True
        )

        self.assertTrue(success)
        self.assertIn("Successfully migrated to version 1.2.0", messages[-1])

        # Verify current version
        self.assertEqual(
            self.migrator.get_current_version(),
            MigrationVersion(1, 2, 0)
        )

        # Verify backup was created
        backup_files = list(self.backup_dir.glob("*.db"))
        self.assertGreater(len(backup_files), 0)

    def test_migrate_dry_run(self):
        """Test dry run migration."""
        m1 = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=["ALTER TABLE test_table ADD COLUMN field1 TEXT"],
            down_sql=["ALTER TABLE test_table DROP COLUMN field1"]
        )

        self.migrator.register_migration(m1)

        # Perform dry run
        success, messages = self.migrator.migrate(
            target_version=MigrationVersion(1, 1, 0),
            dry_run=True
        )

        self.assertTrue(success)
        self.assertIn("Dry run completed", ' '.join(messages))

        # Verify version hasn't changed
        self.assertEqual(
            self.migrator.get_current_version(),
            MigrationVersion(1, 0, 0)
        )

    def test_migrate_already_at_target(self):
        """Test migration when already at target version."""
        success, messages = self.migrator.migrate(
            target_version=MigrationVersion(1, 0, 0)
        )

        self.assertTrue(success)
        self.assertIn("already at target version", ' '.join(messages))

    def test_migrate_no_path(self):
        """Test migration when no path exists."""
        success, messages = self.migrator.migrate(
            target_version=MigrationVersion(5, 0, 0)
        )

        self.assertFalse(success)
        self.assertIn("No migration path found", ' '.join(messages))

    def test_migrate_failure(self):
        """Test handling migration failure."""
        # Create a migration that will fail
        m1 = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=["INVALID SQL STATEMENT"],  # This will fail
            down_sql=["SELECT 1"]
        )

        self.migrator.register_migration(m1)

        success, messages = self.migrator.migrate(
            target_version=MigrationVersion(1, 1, 0),
            create_backup=False
        )

        self.assertFalse(success)
        self.assertIn("Migration failed", ' '.join(messages))

        # Check that failure was recorded
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT status, error_message FROM schema_migrations WHERE version = '1.1.0'"
            )
            result = cursor.fetchone()
            if result:  # Migration record might exist
                self.assertEqual(result[0], MigrationStatus.FAILED.value)
                self.assertIsNotNone(result[1])
        finally:
            conn.close()

    def test_rollback_success(self):
        """Test successful rollback."""
        # First, apply a migration
        m1 = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=["ALTER TABLE test_table ADD COLUMN temp_field TEXT"],
            down_sql=["ALTER TABLE test_table DROP COLUMN temp_field"]
        )

        self.migrator.register_migration(m1)
        self.migrator.migrate(target_version=MigrationVersion(1, 1, 0))

        # Now rollback
        success, messages = self.migrator.rollback(MigrationVersion(1, 0, 0))

        self.assertTrue(success)
        self.assertIn("Successfully rolled back", ' '.join(messages))

        # Verify version
        self.assertEqual(
            self.migrator.get_current_version(),
            MigrationVersion(1, 0, 0)
        )

    def test_rollback_invalid_target(self):
        """Test rollback with invalid target version."""
        success, messages = self.migrator.rollback(MigrationVersion(2, 0, 0))

        self.assertFalse(success)
        self.assertIn("Cannot rollback", ' '.join(messages))

    def test_get_migration_history(self):
        """Test retrieving migration history."""
        # Apply a migration to create history
        m1 = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=["SELECT 1"],
            down_sql=["SELECT 1"]
        )

        self.migrator.register_migration(m1)
        self.migrator.migrate(target_version=MigrationVersion(1, 1, 0))

        history = self.migrator.get_migration_history()

        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)

        # Check history entry structure
        entry = history[0]
        self.assertIn('version', entry)
        self.assertIn('applied_at', entry)
        self.assertIn('status', entry)
        self.assertIn('error_message', entry)

    def test_validate_database_orphaned_records(self):
        """Test database validation with orphaned records."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Create minimal schema with orphaned records
            conn.execute("CREATE TABLE ingredient (id INTEGER PRIMARY KEY)")
            conn.execute("""
                CREATE TABLE formulation_component (
                    id INTEGER PRIMARY KEY,
                    ingredient_id INTEGER
                )
            """)
            # Insert orphaned record
            conn.execute(
                "INSERT INTO formulation_component (ingredient_id) VALUES (999)")
            conn.commit()
        finally:
            conn.close()

        is_valid, issues = self.migrator.validate_database()

        self.assertFalse(is_valid)
        # Will have issues for both missing tables and orphaned records
        self.assertGreater(len(issues), 0)

    def test_migration_with_complex_autofill(self):
        """Test migration with complex autofill scenarios."""
        migration = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=[
                "ALTER TABLE test_table ADD COLUMN status TEXT",
                "ALTER TABLE test_table ADD COLUMN priority INTEGER"
            ],
            down_sql=[
                "ALTER TABLE test_table DROP COLUMN status",
                "ALTER TABLE test_table DROP COLUMN priority"
            ],
            autofill_defaults={
                'test_table.status': 'pending',
                'test_table.priority': 5
            }
        )

        conn = sqlite3.connect(str(self.db_path))
        try:
            self.migrator.apply_migration(migration, conn)

            cursor = conn.cursor()
            cursor.execute("SELECT status, priority FROM test_table")
            results = cursor.fetchall()

            for status, priority in results:
                self.assertEqual(status, 'pending')
                self.assertEqual(priority, 5)

        finally:
            conn.close()

    def test_migration_graph_branching(self):
        """Test migration path finding with branching paths."""

        migrations = [
            Migration(MigrationVersion(1, 0, 0), MigrationVersion(1, 1, 0),
                      ["SQL1"], ["SQL1_DOWN"]),
            Migration(MigrationVersion(1, 0, 0), MigrationVersion(1, 1, 1),
                      ["SQL2"], ["SQL2_DOWN"]),
            Migration(MigrationVersion(1, 1, 0), MigrationVersion(1, 2, 0),
                      ["SQL3"], ["SQL3_DOWN"]),
            Migration(MigrationVersion(1, 1, 1), MigrationVersion(1, 2, 1),
                      ["SQL4"], ["SQL4_DOWN"]),
            Migration(MigrationVersion(1, 2, 0), MigrationVersion(2, 0, 0),
                      ["SQL5"], ["SQL5_DOWN"]),
            Migration(MigrationVersion(1, 2, 1), MigrationVersion(2, 0, 0),
                      ["SQL6"], ["SQL6_DOWN"])
        ]

        for m in migrations:
            self.migrator.register_migration(m)

        path = self.migrator.find_migration_path(
            MigrationVersion(1, 0, 0),
            MigrationVersion(2, 0, 0)
        )

        # Should be 3 steps through one of the branches
        self.assertEqual(len(path), 3)

    def test_concurrent_migration_safety(self):
        """Test that migrations handle concurrent access safely."""
        # This is a simplified test - in production you'd want more robust testing

        migration = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=["ALTER TABLE test_table ADD COLUMN lock_field TEXT"],
            down_sql=["ALTER TABLE test_table DROP COLUMN lock_field"]
        )

        self.migrator.register_migration(migration)

        # Create two connections to simulate concurrent access
        conn1 = sqlite3.connect(str(self.db_path))
        conn2 = sqlite3.connect(str(self.db_path))

        try:
            # Try to apply migration from both connections
            # One should succeed, one should fail or wait
            try:
                self.migrator.apply_migration(migration, conn1)
                success1 = True
            except:
                success1 = False

            try:
                self.migrator.apply_migration(migration, conn2)
                success2 = True
            except:
                success2 = False

            # At least one should succeed
            self.assertTrue(success1 or success2)

        finally:
            conn1.close()
            conn2.close()

    def test_backup_integrity(self):
        """Test that backups maintain data integrity."""
        # Add some test data
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                "INSERT INTO test_table (name, value) VALUES ('test', 999)")
            conn.commit()
        finally:
            conn.close()

        # Create backup
        backup_path = self.migrator.create_backup()

        # Verify backup has same data
        backup_conn = sqlite3.connect(str(backup_path))
        try:
            cursor = backup_conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM test_table WHERE name='test' AND value=999")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
        finally:
            backup_conn.close()

        # Verify checksum in metadata
        metadata_path = backup_path.with_suffix('.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Calculate actual checksum
        with open(backup_path, 'rb') as f:
            actual_checksum = hashlib.md5(f.read()).hexdigest()

        self.assertEqual(metadata['checksum'], actual_checksum)

    def test_edge_case_empty_migration(self):
        """Test handling of empty migrations."""
        migration = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 0, 1),
            up_sql=[],
            down_sql=[]
        )

        conn = sqlite3.connect(str(self.db_path))
        try:
            # Should not raise error
            self.migrator.apply_migration(migration, conn)

            # Should still record migration
            cursor = conn.cursor()
            cursor.execute(
                "SELECT version FROM schema_migrations WHERE version = '1.0.1'")
            result = cursor.fetchone()
            self.assertIsNotNone(result)
        finally:
            conn.close()

    def test_special_characters_in_migration(self):
        """Test migrations with special characters and SQL injection attempts."""
        migration = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=["ALTER TABLE test_table ADD COLUMN \"special'field\" TEXT"],
            down_sql=["ALTER TABLE test_table DROP COLUMN \"special'field\""],
            autofill_defaults={
                "special'field": "test'; DROP TABLE test_table; --"}
        )

        conn = sqlite3.connect(str(self.db_path))
        try:
            # Should handle special characters safely
            self.migrator.apply_migration(migration, conn)

            # Verify table still exists and wasn't dropped
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'")
            result = cursor.fetchone()
            self.assertIsNotNone(result)

        finally:
            conn.close()

    def test_large_migration_batch(self):
        """Test handling of large batch migrations."""
        migrations = []
        for i in range(10):
            migrations.append(Migration(
                from_version=MigrationVersion(1, i, 0),
                to_version=MigrationVersion(1, i+1, 0),
                up_sql=[f"ALTER TABLE test_table ADD COLUMN field_{i} TEXT"],
                down_sql=[f"ALTER TABLE test_table DROP COLUMN field_{i}"],
                description=f"Add field_{i}"
            ))

        for m in migrations:
            self.migrator.register_migration(m)

        # Migrate through all versions
        success, messages = self.migrator.migrate(
            target_version=MigrationVersion(1, 10, 0),
            create_backup=False  # Skip backup for speed
        )

        self.assertTrue(success)
        self.assertEqual(self.migrator.get_current_version(),
                         MigrationVersion(1, 10, 0))

        # Verify all fields were added
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(test_table)")
            columns = {row[1] for row in cursor.fetchall()}
            for i in range(10):
                self.assertIn(f'field_{i}', columns)
        finally:
            conn.close()

    def test_migration_version_edge_cases(self):
        """Test version handling edge cases."""
        # Test version with zeros
        v1 = MigrationVersion(0, 0, 0)
        v2 = MigrationVersion(0, 0, 1)
        self.assertTrue(v1 < v2)

        # Test large version numbers
        v3 = MigrationVersion(999, 999, 999)
        self.assertEqual(str(v3), "999.999.999")

        # Test version from incomplete string
        v4 = MigrationVersion.from_string("2")
        self.assertEqual(v4, MigrationVersion(2, 0, 0))

    def test_rollback_with_failed_migration(self):
        """Test rollback behavior when migrations have failed."""
        # Apply successful migration
        m1 = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=["ALTER TABLE test_table ADD COLUMN field1 TEXT"],
            down_sql=["ALTER TABLE test_table DROP COLUMN field1"]
        )
        self.migrator.register_migration(m1)
        self.migrator.migrate(target_version=MigrationVersion(1, 1, 0))

        # Attempt failed migration
        m2 = Migration(
            from_version=MigrationVersion(1, 1, 0),
            to_version=MigrationVersion(1, 2, 0),
            up_sql=["INVALID SQL"],
            down_sql=["SELECT 1"]
        )
        self.migrator.register_migration(m2)
        self.migrator.migrate(target_version=MigrationVersion(1, 2, 0))

        # Should be able to rollback to 1.0.0
        success, messages = self.migrator.rollback(MigrationVersion(1, 0, 0))
        self.assertTrue(success)

    def test_autofill_with_null_values(self):
        """Test autofill behavior with NULL and special values."""
        migration = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=[
                "ALTER TABLE test_table ADD COLUMN nullable_field TEXT",
                "ALTER TABLE test_table ADD COLUMN timestamp_field TIMESTAMP"
            ],
            down_sql=[
                "ALTER TABLE test_table DROP COLUMN nullable_field",
                "ALTER TABLE test_table DROP COLUMN timestamp_field"
            ],
            autofill_defaults={
                'nullable_field': None,
                'timestamp_field': 'CURRENT_TIMESTAMP'
            }
        )

        conn = sqlite3.connect(str(self.db_path))
        try:
            self.migrator.apply_migration(migration, conn)

            cursor = conn.cursor()
            cursor.execute("SELECT nullable_field FROM test_table LIMIT 1")
            result = cursor.fetchone()
            # Should remain NULL, not be updated
            self.assertIsNone(result[0])

        finally:
            conn.close()

    def test_migration_with_foreign_keys(self):
        """Test migrations with foreign key constraints."""
        # Create related tables
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("""
                CREATE TABLE parent_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT
                )
            """)
            conn.execute("INSERT INTO parent_table (name) VALUES ('parent1')")
            conn.commit()
        finally:
            conn.close()

        migration = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=[
                "ALTER TABLE test_table ADD COLUMN parent_id INTEGER REFERENCES parent_table(id)"
            ],
            down_sql=[
                "ALTER TABLE test_table DROP COLUMN parent_id"
            ],
            autofill_defaults={'parent_id': 1}
        )

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            self.migrator.apply_migration(migration, conn)

            # Verify foreign key constraint is enforced
            cursor = conn.cursor()
            with self.assertRaises(sqlite3.IntegrityError):
                cursor.execute("UPDATE test_table SET parent_id = 999")

        finally:
            conn.close()

    def test_data_transform_error_handling(self):
        """Test error handling in data transformation functions."""
        def failing_transform(conn):
            raise ValueError("Transform failed!")

        migration = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=["ALTER TABLE test_table ADD COLUMN transformed BOOLEAN"],
            down_sql=["ALTER TABLE test_table DROP COLUMN transformed"],
            data_transform=failing_transform
        )

        conn = sqlite3.connect(str(self.db_path))
        try:
            with self.assertRaises(ValueError) as context:
                self.migrator.apply_migration(migration, conn)
            self.assertEqual(str(context.exception), "Transform failed!")

        finally:
            conn.close()

    def test_migration_with_complex_schema(self):
        """Test migrations with complex schema changes."""
        # Create a more complex initial schema
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                CREATE TABLE complex_table (
                    id INTEGER PRIMARY KEY,
                    data JSON,
                    metadata TEXT,
                    UNIQUE(data, metadata)
                )
            """)
            conn.execute("""
                CREATE INDEX idx_complex_metadata ON complex_table(metadata)
            """)
            conn.commit()
        finally:
            conn.close()

        migration = Migration(
            from_version=MigrationVersion(1, 0, 0),
            to_version=MigrationVersion(1, 1, 0),
            up_sql=[
                "DROP INDEX idx_complex_metadata",
                "ALTER TABLE complex_table ADD COLUMN version INTEGER DEFAULT 1",
                "CREATE INDEX idx_complex_version ON complex_table(version)",
                "CREATE TRIGGER update_version AFTER UPDATE ON complex_table "
                "BEGIN UPDATE complex_table SET version = version + 1 WHERE id = NEW.id; END"
            ],
            down_sql=[
                "DROP TRIGGER update_version",
                "DROP INDEX idx_complex_version",
                "ALTER TABLE complex_table DROP COLUMN version",
                "CREATE INDEX idx_complex_metadata ON complex_table(metadata)"
            ]
        )

        conn = sqlite3.connect(str(self.db_path))
        try:
            self.migrator.apply_migration(migration, conn)

            # Verify new schema
            cursor = conn.cursor()
            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_complex_version'")
            result = cursor.fetchone()
            self.assertIsNotNone(result)

            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='trigger' AND name='update_version'")
            result = cursor.fetchone()
            self.assertIsNotNone(result)

        finally:
            conn.close()

    def test_get_migration_history_empty(self):
        """Test getting migration history when empty."""
        history = self.migrator.get_migration_history()
        self.assertEqual(history, [])

    def test_cleanup_old_backups(self):
        """Test that old backups can be identified (manual cleanup)."""
        # Create multiple backups
        for i in range(5):
            self.migrator.create_backup(suffix=f"_test{i}")

        backup_files = list(self.backup_dir.glob("*.db"))
        self.assertEqual(len(backup_files), 5)

        # Get backup metadata
        backups_with_dates = []
        for backup in backup_files:
            metadata_path = backup.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    backups_with_dates.append(
                        (backup, metadata['backup_time']))

        # Should be able to sort by date
        backups_with_dates.sort(key=lambda x: x[1])
        self.assertEqual(len(backups_with_dates), 5)


if __name__ == '__main__':
    unittest.main()
