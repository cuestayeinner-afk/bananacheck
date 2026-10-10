"""SQLite persistence and one-time migration from BananaCheck's CSV/JSON files."""

import csv
import json
import os
import sqlite3
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, ".bananacheck_data")
DATABASE_PATH = os.path.join(DATA_DIR, "bananacheck.db")

ANALYSIS_FILES = {
    "fruta": "datos_fruta.csv",
    "terreno": "datos_terreno.csv",
}


class ManagedConnection(sqlite3.Connection):
    def __exit__(self, exception_type, exception, traceback):
        try:
            return super().__exit__(exception_type, exception, traceback)
        finally:
            self.close()


def connect(path=None):
    connection = sqlite3.connect(path or DATABASE_PATH, timeout=10, factory=ManagedConnection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _read_csv(directory, filename):
    file_path = os.path.join(directory, filename)
    if not os.path.isfile(file_path):
        return []
    with open(file_path, "r", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def initialize(directory=None, path=None):
    directory = directory or BASE_DIR
    path = path or DATABASE_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with connect(path) as database:
        database.executescript(
            """
            CREATE TABLE IF NOT EXISTS migrations (
                name TEXT PRIMARY KEY,
                completed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS accounts (
                username TEXT PRIMARY KEY COLLATE NOCASE,
                role TEXT NOT NULL,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL CHECK(type IN ('fruta', 'terreno')),
                created_at TEXT NOT NULL,
                username TEXT NOT NULL DEFAULT '',
                farm TEXT NOT NULL DEFAULT '',
                health_score INTEGER,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS analyses_created_at_idx ON analyses(created_at);
            CREATE INDEX IF NOT EXISTS analyses_username_idx ON analyses(username COLLATE NOCASE);
            CREATE TABLE IF NOT EXISTS comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                type TEXT NOT NULL,
                health_score TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                username TEXT NOT NULL DEFAULT '',
                farm TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                type TEXT NOT NULL,
                username TEXT NOT NULL,
                health_score TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS history_username_idx ON history(username COLLATE NOCASE);
            CREATE TABLE IF NOT EXISTS farms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                registered_at TEXT NOT NULL,
                name TEXT NOT NULL,
                location TEXT NOT NULL,
                registered_by TEXT NOT NULL DEFAULT '',
                UNIQUE(name COLLATE NOCASE, location COLLATE NOCASE)
            );
            """
        )
        if database.execute("SELECT 1 FROM migrations WHERE name = 'legacy_import_v1'").fetchone():
            return

        for analysis_type, filename in ANALYSIS_FILES.items():
            for record in _read_csv(directory, filename):
                created_at = record.get("fecha_hora") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                database.execute(
                    "INSERT INTO analyses(type, created_at, username, farm, payload) VALUES (?, ?, ?, ?, ?)",
                    (analysis_type, created_at, record.get("usuario", ""), record.get("finca", ""), json.dumps(record, ensure_ascii=False)),
                )
        for record in _read_csv(directory, "analisis_guardados.csv"):
            database.execute(
                "INSERT INTO comparisons(created_at, type, health_score, summary) VALUES (?, ?, ?, ?)",
                (record.get("fecha_hora", ""), record.get("tipo", ""), record.get("salud_pct", ""), record.get("resumen", "")),
            )
        for record in _read_csv(directory, "historial_analisis.csv"):
            database.execute(
                "INSERT INTO history(created_at, type, username, health_score, summary) VALUES (?, ?, ?, ?, ?)",
                (record.get("fecha_hora", ""), record.get("tipo", ""), record.get("usuario", ""), record.get("salud_pct", ""), record.get("resumen", "")),
            )
        for record in _read_csv(directory, "fincas.csv"):
            if record.get("nombre") and record.get("ubicacion"):
                database.execute(
                    "INSERT OR IGNORE INTO farms(registered_at, name, location, registered_by) VALUES (?, ?, ?, ?)",
                    (record.get("fecha_registro", ""), record["nombre"], record["ubicacion"], record.get("registrada_por", "")),
                )

        accounts_path = os.path.join(directory, ".bananacheck_cuenta.json")
        if os.path.isfile(accounts_path):
            with open(accounts_path, "r", encoding="utf-8") as source:
                raw_accounts = json.load(source)
            accounts = raw_accounts.get("usuarios", [])
            if not accounts and raw_accounts.get("usuario"):
                raw_accounts.setdefault("rol", "ADMIN")
                accounts = [raw_accounts]
            for account in accounts:
                if all(account.get(key) for key in ("usuario", "rol", "sal", "hash")):
                    database.execute(
                        "INSERT OR IGNORE INTO accounts(username, role, salt, password_hash) VALUES (?, ?, ?, ?)",
                        (account["usuario"], account["rol"], account["sal"], account["hash"]),
                    )
        database.execute(
            "INSERT INTO migrations(name, completed_at) VALUES (?, ?)",
            ("legacy_import_v1", datetime.now().isoformat(timespec="seconds")),
        )


def list_accounts(path=None):
    with connect(path) as database:
        return [
            {"usuario": row["username"], "rol": row["role"], "sal": row["salt"], "hash": row["password_hash"]}
            for row in database.execute("SELECT * FROM accounts ORDER BY rowid")
        ]


def save_accounts(accounts, path=None, create=False):
    with connect(path) as database:
        if create and database.execute("SELECT 1 FROM accounts LIMIT 1").fetchone():
            raise FileExistsError("La cuenta ya está configurada")
        if create:
            account = accounts[0]
            database.execute(
                "INSERT INTO accounts(username, role, salt, password_hash) VALUES (?, ?, ?, ?)",
                (account["usuario"], account["rol"], account["sal"], account["hash"]),
            )
            return
        database.execute("DELETE FROM accounts")
        database.executemany(
            "INSERT INTO accounts(username, role, salt, password_hash) VALUES (?, ?, ?, ?)",
            [(account["usuario"], account["rol"], account["sal"], account["hash"]) for account in accounts],
        )


def has_accounts(path=None):
    with connect(path) as database:
        return database.execute("SELECT 1 FROM accounts LIMIT 1").fetchone() is not None


def save_analysis(analysis_type, data, username, score, summary, path=None):
    created_at = data["fecha_hora"]
    farm = str(data.get("finca") or "").strip()
    with connect(path) as database:
        cursor = database.execute(
            "INSERT INTO analyses(type, created_at, username, farm, health_score, payload) VALUES (?, ?, ?, ?, ?, ?)",
            (analysis_type, created_at, username or "", farm, score, json.dumps(data, ensure_ascii=False)),
        )
        analysis_id = cursor.lastrowid
        database.execute(
            "INSERT INTO comparisons(created_at, type, health_score, summary, username, farm) VALUES (?, ?, ?, ?, ?, ?)",
            (created_at, analysis_type, "" if score is None else str(score), summary or "", username or "", farm),
        )
        if username:
            database.execute(
                "INSERT INTO history(created_at, type, username, health_score, summary) VALUES (?, ?, ?, ?, ?)",
                (created_at, analysis_type, username, "" if score is None else str(score), summary or ""),
            )
        total = database.execute("SELECT COUNT(*) FROM analyses WHERE type = ?", (analysis_type,)).fetchone()[0]
        return analysis_id, total


def get_analysis(analysis_id, path=None):
    with connect(path) as database:
        row = database.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        return result


def list_analyses(filters=None, path=None):
    filters = filters or {}
    clauses = []
    parameters = []
    for key, column in (("type", "type"), ("username", "username"), ("farm", "farm")):
        value = str(filters.get(key) or "").strip()
        if value:
            clauses.append(f"{column} = ? COLLATE NOCASE")
            parameters.append(value)
    for key, operator in (("desde", ">="), ("hasta", "<=")):
        value = str(filters.get(key) or "").strip()
        if value:
            clauses.append(f"substr(created_at, 1, 10) {operator} ?")
            parameters.append(value)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect(path) as database:
        rows = database.execute(f"SELECT * FROM analyses {where} ORDER BY id DESC", parameters).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item.update(json.loads(item.pop("payload")))
            results.append(item)
        return results


def list_comparisons(path=None):
    with connect(path) as database:
        rows = database.execute("SELECT * FROM comparisons ORDER BY id").fetchall()
        return [dict(row) | {"fecha_hora": row["created_at"], "tipo": row["type"], "salud_pct": row["health_score"], "resumen": row["summary"]} for row in rows]


def list_history(username=None, path=None):
    with connect(path) as database:
        if username:
            rows = database.execute("SELECT * FROM history WHERE username = ? COLLATE NOCASE ORDER BY id", (username,)).fetchall()
        else:
            rows = database.execute("SELECT * FROM history ORDER BY id").fetchall()
        return [dict(row) | {"fecha_hora": row["created_at"], "tipo": row["type"], "usuario": row["username"], "salud_pct": row["health_score"], "resumen": row["summary"]} for row in rows]


def list_farms(path=None):
    with connect(path) as database:
        rows = database.execute("SELECT * FROM farms ORDER BY id").fetchall()
        return [dict(row) | {"fecha_registro": row["registered_at"], "nombre": row["name"], "ubicacion": row["location"], "registrada_por": row["registered_by"]} for row in rows]


def add_farm(name, location, registered_by, path=None):
    with connect(path) as database:
        cursor = database.execute(
            "INSERT INTO farms(registered_at, name, location, registered_by) VALUES (?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, location, registered_by),
        )
        return cursor.lastrowid


def count_analyses(analysis_type, path=None):
    with connect(path) as database:
        return database.execute("SELECT COUNT(*) FROM analyses WHERE type = ?", (analysis_type,)).fetchone()[0]


def backup(destination, path=None):
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    with connect(path) as source, sqlite3.connect(destination) as target:
        source.backup(target)
    return destination