import csv
import json
import os
import sqlite3
from typing import Dict, List, Set

RAW_ROOT = os.getenv('RAW_ROOT', 'data/raw/raw')
REGION = os.getenv('REGION', 'europe')
INDEX_CSV = os.getenv('RAW_INDEX_CSV', 'data/clean/raw_index.csv')
STATE_DB = os.getenv('RAW_STATE_DB', 'data/clean/raw_state.sqlite')
DRY_RUN = os.getenv('DRY_RUN', '1') not in {'0', 'false', 'False'}
DRY_RUN = False # Override para evitar accidentes, quitar para permitir reparaciones reales

INDEX_FIELDS = [
    'matchId',
    'routingRegion',
    'platformRegion',
    'queueId',
    'gameVersion',
    'gameDuration',
    'gameCreation',
    'gameEndTimestamp',
    'gameMode',
    'mapId',
    'patchMajorMinor',
    'sourceTier',
]


def patch_major_minor(game_version):
    if not game_version or not isinstance(game_version, str):
        return None
    parts = game_version.split('.')
    return '.'.join(parts[:2]) if len(parts) >= 2 else game_version


def load_indexed_ids(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, 'r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            mid = row.get('matchId')
            if mid:
                out.add(mid)
    return out


def load_ok_ids(db_path: str) -> Set[str]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT match_id FROM matches WHERE status='OK'")
        return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def append_rows(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, 'a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if not file_exists:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in INDEX_FIELDS})


def build_row_from_match_json(match_id: str) -> Dict:
    mp = os.path.join(RAW_ROOT, REGION, match_id, 'match.json')
    with open(mp, 'r', encoding='utf-8') as f:
        match = json.load(f)
    info = (match or {}).get('info', {}) or {}
    return {
        'matchId': match_id,
        'routingRegion': REGION,
        'platformRegion': info.get('platformId'),
        'queueId': info.get('queueId'),
        'gameVersion': info.get('gameVersion'),
        'gameDuration': info.get('gameDuration'),
        'gameCreation': info.get('gameCreation'),
        'gameEndTimestamp': info.get('gameEndTimestamp'),
        'gameMode': info.get('gameMode'),
        'mapId': info.get('mapId'),
        'patchMajorMinor': patch_major_minor(info.get('gameVersion')),
        'sourceTier': None,
    }


def main():
    base = os.path.join(RAW_ROOT, REGION)
    raw_ids = {
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    }
    idx_ids = load_indexed_ids(INDEX_CSV)
    ok_ids = load_ok_ids(STATE_DB)

    repair_ids = sorted((raw_ids & ok_ids) - idx_ids)

    print('=== RAW INDEX REPAIR ===')
    print(f'RAW dirs:      {len(raw_ids)}')
    print(f'INDEX ids:     {len(idx_ids)}')
    print(f'OK state ids:  {len(ok_ids)}')
    print(f'To repair:     {len(repair_ids)}')

    if repair_ids:
        preview = repair_ids[:10]
        print('Sample:', preview, ('... (+%d)' % (len(repair_ids)-10)) if len(repair_ids) > 10 else '')

    if DRY_RUN or not repair_ids:
        print('DRY_RUN active; no changes written.' if DRY_RUN else 'Nothing to repair.')
        return

    rows = []
    missing_raw = []
    bad_json = []
    for mid in repair_ids:
        mp = os.path.join(RAW_ROOT, REGION, mid, 'match.json')
        if not os.path.exists(mp):
            missing_raw.append(mid)
            continue
        try:
            rows.append(build_row_from_match_json(mid))
        except Exception as e:
            bad_json.append((mid, f'{type(e).__name__}: {e}'))

    print(f'Rows ready:    {len(rows)}')
    print(f'Missing raw:   {len(missing_raw)}')
    print(f'Bad json:      {len(bad_json)}')

    if rows:
        append_rows(INDEX_CSV, rows)
        print(f'Appended {len(rows)} rows to {INDEX_CSV}')

    if missing_raw:
        print('Missing raw sample:', missing_raw[:10])
    if bad_json:
        print('Bad json sample:', bad_json[:5])


if __name__ == '__main__':
    main()
