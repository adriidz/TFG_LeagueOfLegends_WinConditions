import csv
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

RAW_ROOT = Path("Data_raw/raw")
ROUTING_REGION = "europe"

OUT_INDEX = Path("Data_clean/raw_index_rebuilt.csv")
OUT_STATE = Path("Data_clean/raw_state_rebuilt.sqlite")

def patch_major_minor(game_version: str):
    try:
        parts = str(game_version).split(".")
        return f"{int(parts[0])}.{int(parts[1])}"
    except Exception:
        return None

def utc_now():
    return datetime.utcnow().isoformat()

def open_state_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
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

def state_set(conn, match_id: str, routing_region: str, status: str, last_error: str = None):
    now = utc_now()
    conn.execute("""
        INSERT INTO matches(match_id, routing_region, status, last_error, updated_at)
        VALUES(?,?,?,?,?)
        ON CONFLICT(match_id) DO UPDATE SET
          routing_region=excluded.routing_region,
          status=excluded.status,
          last_error=excluded.last_error,
          updated_at=excluded.updated_at
    """, (match_id, routing_region, status, last_error, now))

def main():
    t0 = time.perf_counter()

    region_dir = RAW_ROOT / ROUTING_REGION
    if not region_dir.exists():
        raise SystemExit(f"No existe: {region_dir}")

    # Campos del índice (igual que raw_collector suele escribir)
    fieldnames = [
        "matchId",
        "routingRegion",
        "platformRegion",
        "queueId",
        "gameVersion",
        "gameDuration",
        "gameCreation",
        "gameEndTimestamp",
        "gameMode",
        "mapId",
        "patchMajorMinor",
        "sourceTier",
    ]

    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)

    conn = open_state_db(OUT_STATE)

    total_dirs = 0
    ok = 0
    missing_tl = 0
    bad_json = 0
    missing_match = 0

    with open(OUT_INDEX, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for match_dir in region_dir.iterdir():
            if not match_dir.is_dir():
                continue

            total_dirs += 1
            match_id = match_dir.name
            match_path = match_dir / "match.json"
            tl_path = match_dir / "timeline.json"

            if not match_path.exists():
                missing_match += 1
                state_set(conn, match_id, ROUTING_REGION, "MISSING_MATCH", "match.json not found")
                continue

            try:
                match = json.loads(match_path.read_text(encoding="utf-8"))
            except Exception as e:
                bad_json += 1
                state_set(conn, match_id, ROUTING_REGION, "BAD_JSON", f"{type(e).__name__}: {e}")
                continue

            info = (match.get("info") or {})
            row = {
                "matchId": match_id,
                "routingRegion": ROUTING_REGION,
                "platformRegion": info.get("platformId"),
                "queueId": info.get("queueId"),
                "gameVersion": info.get("gameVersion"),
                "gameDuration": info.get("gameDuration"),
                "gameCreation": info.get("gameCreation"),
                "gameEndTimestamp": info.get("gameEndTimestamp"),
                "gameMode": info.get("gameMode"),
                "mapId": info.get("mapId"),
                "patchMajorMinor": patch_major_minor(info.get("gameVersion")),
                "sourceTier": None,
            }
            w.writerow(row)

            if tl_path.exists():
                ok += 1
                state_set(conn, match_id, ROUTING_REGION, "OK", None)
            else:
                missing_tl += 1
                state_set(conn, match_id, ROUTING_REGION, "MISSING_TIMELINE", "timeline.json not found")

            if total_dirs % 5000 == 0:
                elapsed = time.perf_counter() - t0
                rate = total_dirs / elapsed if elapsed > 0 else 0.0
                print(f"[PROGRESS] {total_dirs} carpetas | {elapsed:.1f}s | {rate:.1f} carpetas/s")

    conn.commit()
    conn.close()

    dt = time.perf_counter() - t0
    print("\n=== REBUILD DONE ===")
    print(f"Carpetas vistas:        {total_dirs}")
    print(f"OK:                    {ok}")
    print(f"MISSING_TIMELINE:      {missing_tl}")
    print(f"BAD_JSON:              {bad_json}")
    print(f"MISSING_MATCH:         {missing_match}")
    print(f"\nIndex rebuilt:         {OUT_INDEX}")
    print(f"State rebuilt:         {OUT_STATE}")
    print(f"\nTiempo total:          {dt:.2f} s")

if __name__ == "__main__":
    main()