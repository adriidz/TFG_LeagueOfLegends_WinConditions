# raw_store.py
import os
import json
import csv
import sqlite3
from datetime import datetime

RAW_ROOT_DEFAULT = "Data_raw/raw"
INDEX_CSV_DEFAULT = "Data_clean/raw_index.csv"
STATE_DB_DEFAULT = "Data_clean/raw_state.sqlite"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _match_dir(raw_root: str, routing_region: str, match_id: str) -> str:
    return os.path.join(raw_root, routing_region, match_id)


def save_raw_match(raw_root: str, routing_region: str, match_id: str, match_json: dict, timeline_json: dict):
    mdir = _match_dir(raw_root, routing_region, match_id)
    ensure_dir(mdir)

    match_path = os.path.join(mdir, "match.json")
    tl_path = os.path.join(mdir, "timeline.json")

    # Escritura atómica simple: escribe y reemplaza
    tmp_match = match_path + ".tmp"
    tmp_tl = tl_path + ".tmp"

    with open(tmp_match, "w", encoding="utf-8") as f:
        json.dump(match_json, f, ensure_ascii=False)
    with open(tmp_tl, "w", encoding="utf-8") as f:
        json.dump(timeline_json, f, ensure_ascii=False)

    os.replace(tmp_match, match_path)
    os.replace(tmp_tl, tl_path)

    return match_path, tl_path


def open_state_db(db_path: str):
    ensure_dir(os.path.dirname(db_path) or ".")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id TEXT PRIMARY KEY,
            routing_region TEXT,
            status TEXT,
            last_error TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    return conn


def state_get(conn, match_id: str):
    cur = conn.execute("SELECT match_id, routing_region, status, last_error, updated_at FROM matches WHERE match_id=?",
                       (match_id,))
    return cur.fetchone()


def state_set(conn, match_id: str, routing_region: str, status: str, last_error: str = None):
    now = datetime.utcnow().isoformat()
    conn.execute("""
        INSERT INTO matches(match_id, routing_region, status, last_error, updated_at)
        VALUES(?,?,?,?,?)
        ON CONFLICT(match_id) DO UPDATE SET
          routing_region=excluded.routing_region,
          status=excluded.status,
          last_error=excluded.last_error,
          updated_at=excluded.updated_at
    """, (match_id, routing_region, status, last_error, now))
    conn.commit()


def append_index_row(index_csv: str, row: dict, fieldnames: list):
    ensure_dir(os.path.dirname(index_csv) or ".")
    file_exists = os.path.exists(index_csv)

    with open(index_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fieldnames})