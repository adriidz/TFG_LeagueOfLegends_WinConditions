import os
import csv
import json
import sqlite3
from collections import Counter
from typing import Dict, List, Set, Tuple

RAW_ROOT = os.getenv("RAW_ROOT", "data/raw/raw")
REGION = os.getenv("REGION", "europe")
INDEX_CSV = os.getenv("RAW_INDEX_CSV", "data/clean/raw_index.csv")
STATE_DB = os.getenv("RAW_STATE_DB", "data/clean/raw_state.sqlite")
SAMPLE_JSON = int(os.getenv("RAW_CHECK_SAMPLE_JSON", "200"))


def list_raw_matches(base: str) -> Tuple[Set[str], List[str], List[str]]:
    raw_ids: Set[str] = set()
    missing_files: List[str] = []
    bad_json: List[str] = []

    if not os.path.isdir(base):
        return raw_ids, missing_files, bad_json

    dirs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    raw_ids = set(dirs)

    # Validación estructural completa
    for match_id in dirs:
        md = os.path.join(base, match_id)
        mp = os.path.join(md, "match.json")
        tp = os.path.join(md, "timeline.json")
        if not (os.path.exists(mp) and os.path.exists(tp)):
            missing_files.append(match_id)

    # Validación de JSON sobre muestra limitada para que no tarde siglos
    sample = dirs[: min(SAMPLE_JSON, len(dirs))]
    for match_id in sample:
        md = os.path.join(base, match_id)
        mp = os.path.join(md, "match.json")
        tp = os.path.join(md, "timeline.json")
        if not (os.path.exists(mp) and os.path.exists(tp)):
            continue
        try:
            with open(mp, "r", encoding="utf-8") as f:
                m = json.load(f)
            with open(tp, "r", encoding="utf-8") as f:
                t = json.load(f)
            _ = (((m or {}).get("info") or {}).get("gameVersion"))
            _ = ((((t or {}).get("info") or {}).get("frames")) or [])
        except Exception:
            bad_json.append(match_id)

    return raw_ids, missing_files, bad_json


def load_index(index_csv: str) -> Tuple[Set[str], List[str], Counter]:
    index_ids: Set[str] = set()
    dup_ids: List[str] = []
    tiers = Counter()

    if not os.path.exists(index_csv):
        return index_ids, dup_ids, tiers

    seen = Counter()
    with open(index_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            match_id = (row.get("matchId") or "").strip()
            if not match_id:
                continue
            seen[match_id] += 1
            index_ids.add(match_id)
            tiers[(row.get("sourceTier") or "").strip() or "<empty>"] += 1

    dup_ids = [mid for mid, n in seen.items() if n > 1]
    return index_ids, dup_ids, tiers


def load_state(state_db: str) -> Tuple[Dict[str, str], Counter]:
    states: Dict[str, str] = {}
    counts = Counter()

    if not os.path.exists(state_db):
        return states, counts

    conn = sqlite3.connect(state_db)
    try:
        cur = conn.execute("SELECT match_id, status FROM matches")
        for match_id, status in cur.fetchall():
            states[match_id] = status
            counts[status] += 1
    finally:
        conn.close()
    return states, counts


def preview(items: List[str], n: int = 10) -> str:
    if not items:
        return "[]"
    head = items[:n]
    suffix = "" if len(items) <= n else f" ... (+{len(items)-n})"
    return str(head) + suffix


if __name__ == "__main__":
    base = os.path.join(RAW_ROOT, REGION)
    raw_ids, missing_files, bad_json = list_raw_matches(base)
    index_ids, dup_ids, tiers = load_index(INDEX_CSV)
    states, state_counts = load_state(STATE_DB)

    ok_ids = {mid for mid, st in states.items() if st == "OK"}
    in_progress_ids = {mid for mid, st in states.items() if st == "IN_PROGRESS"}

    raw_not_in_index = sorted(raw_ids - index_ids)
    index_not_in_raw = sorted(index_ids - raw_ids)
    ok_not_in_index = sorted(ok_ids - index_ids)
    ok_not_in_raw = sorted(ok_ids - raw_ids)
    index_not_ok = sorted(index_ids - ok_ids)
    in_progress_with_raw = sorted(in_progress_ids & raw_ids)

    print("=== RAW CONSISTENCY CHECK ===")
    print(f"RAW base:   {base}")
    print(f"INDEX CSV:  {INDEX_CSV}")
    print(f"STATE DB:   {STATE_DB}")
    print()
    print(f"raw_dirs            = {len(raw_ids)}")
    print(f"index_unique_ids    = {len(index_ids)}")
    print(f"state_rows          = {len(states)}")
    print(f"state_status_counts = {dict(state_counts)}")
    print(f"source_tier_counts  = {dict(tiers)}")
    print()

    print(f"missing_raw_files   = {len(missing_files)} {preview(sorted(missing_files))}")
    print(f"bad_json_sample     = {len(bad_json)} {preview(sorted(bad_json))}")
    print(f"duplicate_index_ids = {len(dup_ids)} {preview(sorted(dup_ids))}")
    print()

    print(f"raw_not_in_index    = {len(raw_not_in_index)} {preview(raw_not_in_index)}")
    print(f"index_not_in_raw    = {len(index_not_in_raw)} {preview(index_not_in_raw)}")
    print(f"ok_not_in_index     = {len(ok_not_in_index)} {preview(ok_not_in_index)}")
    print(f"ok_not_in_raw       = {len(ok_not_in_raw)} {preview(ok_not_in_raw)}")
    print(f"index_not_ok        = {len(index_not_ok)} {preview(index_not_ok)}")
    print(f"inprog_with_raw     = {len(in_progress_with_raw)} {preview(in_progress_with_raw)}")
    print()

    problems = []
    if missing_files:
        problems.append("Hay directorios RAW sin match.json o timeline.json")
    if bad_json:
        problems.append("Hay JSON corrupto o ilegible en la muestra")
    if dup_ids:
        problems.append("Hay matchId duplicados en el índice")
    if index_not_in_raw:
        problems.append("Hay filas en índice sin RAW correspondiente")
    if ok_not_in_raw:
        problems.append("Hay estados OK sin RAW")
    if ok_not_in_index:
        problems.append("Hay estados OK sin fila en índice")
    if index_not_ok:
        problems.append("Hay filas en índice cuyo estado no es OK")

    if problems:
        print("RESULTADO: INCONSISTENCIAS DETECTADAS")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)

    print("RESULTADO: CONSISTENTE")
