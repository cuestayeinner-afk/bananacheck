#!/usr/bin/env python3
"""Create and verify consistent snapshots of BananaCheck's SQLite database."""

import argparse
import os
import sqlite3
from datetime import datetime

import bananacheck_storage as storage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Ruta para el archivo .db de respaldo")
    parser.add_argument("--verify", help="Comprueba la integridad de un respaldo existente")
    args = parser.parse_args()

    if args.verify:
        with sqlite3.connect(args.verify) as database:
            resultado = database.execute("PRAGMA integrity_check").fetchone()[0]
        print(resultado)
        return 0 if resultado == "ok" else 1

    storage.initialize()
    destino = args.output or os.path.join(
        storage.DATA_DIR,
        "backups",
        f"bananacheck-{datetime.now().strftime('%Y%m%d-%H%M%S')}.db",
    )
    storage.backup(destino)
    print(f"Respaldo creado: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())