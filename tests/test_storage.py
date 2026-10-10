import csv
import json
import tempfile
import unittest
from pathlib import Path

import bananacheck_storage as storage


class SQLiteStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.database = str(self.root / "test.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_legacy_files_migrate_once_without_being_removed(self):
        source = self.root / "datos_fruta.csv"
        with source.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=["fecha_hora", "usuario", "diagnostico"])
            writer.writeheader()
            writer.writerow({"fecha_hora": "2026-10-10 10:00:00", "usuario": "ana", "diagnostico": "Apta"})
        (self.root / ".bananacheck_cuenta.json").write_text(
            json.dumps({"usuarios": [{"usuario": "ana", "rol": "USUARIO", "sal": "01", "hash": "02"}]}),
            encoding="utf-8",
        )

        storage.initialize(str(self.root), self.database)
        storage.initialize(str(self.root), self.database)

        self.assertEqual(len(storage.list_analyses(path=self.database)), 1)
        self.assertEqual(storage.list_accounts(self.database)[0]["usuario"], "ana")
        self.assertTrue(source.exists())

    def test_analysis_save_is_atomic_and_filters_apply(self):
        storage.initialize(str(self.root), self.database)
        data = {"fecha_hora": "2026-10-10 10:00:00", "finca": "Las Palmas", "diagnostico": "Apta"}

        analysis_id, total = storage.save_analysis("fruta", data, "ana", 80, "Apta", self.database)

        self.assertEqual((analysis_id, total), (1, 1))
        self.assertEqual(len(storage.list_history("ANA", self.database)), 1)
        self.assertEqual(len(storage.list_comparisons(self.database)), 1)
        self.assertEqual(len(storage.list_analyses({"farm": "las palmas", "desde": "2026-10-01"}, self.database)), 1)
        self.assertEqual(len(storage.list_analyses({"type": "terreno"}, self.database)), 0)

    def test_farm_uniqueness_is_case_insensitive(self):
        storage.initialize(str(self.root), self.database)
        storage.add_farm("La Esperanza", "Urabá", "staff", self.database)
        with self.assertRaises(Exception):
            storage.add_farm("la esperanza", "urabá", "staff", self.database)

    def test_backup_is_a_consistent_sqlite_database(self):
        storage.initialize(str(self.root), self.database)
        storage.add_farm("La Esperanza", "Urabá", "staff", self.database)
        backup_path = str(self.root / "backup.db")

        storage.backup(backup_path, self.database)

        with storage.connect(backup_path) as database:
            self.assertEqual(database.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(database.execute("SELECT COUNT(*) FROM farms").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()