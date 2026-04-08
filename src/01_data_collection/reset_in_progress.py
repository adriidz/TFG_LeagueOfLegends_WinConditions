import sqlite3
import shutil
from pathlib import Path

db_path = Path("data/clean/raw_state.sqlite")
backup_path = Path("data/clean/raw_state_backup.sqlite")

if not db_path.exists():
    raise FileNotFoundError(f"No existe la base de datos: {db_path}")

shutil.copy2(db_path, backup_path)
print(f"Copia creada: {backup_path}")

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM matches WHERE status = 'IN_PROGRESS'")
before = cur.fetchone()[0]

cur.execute("DELETE FROM matches WHERE status = 'IN_PROGRESS'")
deleted = cur.rowcount

conn.commit()

cur.execute("SELECT COUNT(*) FROM matches WHERE status = 'IN_PROGRESS'")
after = cur.fetchone()[0]

conn.close()

print(f"IN_PROGRESS antes: {before}")
print(f"Filas borradas: {deleted}")
print(f"IN_PROGRESS después: {after}")