# raw_collector.py
import os
import time
import random
from typing import List, Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from riotwatcher import LolWatcher, RiotWatcher, ApiError

from raw_store import (
    RAW_ROOT_DEFAULT, INDEX_CSV_DEFAULT, STATE_DB_DEFAULT,
    open_state_db, state_get, state_set,
    save_raw_match, append_index_row
)

# Reusa tu wrapper robusto de api.py (429/5xx), que ya tienes bien montado :contentReference[oaicite:3]{index=3}
from api import riot_call_with_retry, API_KEY, REGION, MATCH_REGION

load_dotenv("TFG.env")

RAW_ROOT = os.getenv("RAW_ROOT", RAW_ROOT_DEFAULT)
INDEX_CSV = os.getenv("RAW_INDEX_CSV", INDEX_CSV_DEFAULT)
STATE_DB = os.getenv("RAW_STATE_DB", STATE_DB_DEFAULT)

QUEUE_RANKED_SOLO = int(os.getenv("RAW_QUEUE_ID", "420"))
MATCHES_PER_PLAYER = int(os.getenv("RAW_MATCHES_PER_PLAYER", "300"))
TOTAL_MATCHES_TARGET = int(os.getenv("RAW_TOTAL_MATCHES_TARGET", "200000"))

MIN_PATCH_MAJOR = 16
MIN_PATCH_MINOR = 2

SAVE_EVERY_N = int(os.getenv("RAW_SAVE_EVERY_N", "50"))

watcher = LolWatcher(API_KEY)
account_watcher = RiotWatcher(API_KEY)

INDEX_FIELDS = [
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


def patch_major_minor(game_version: Optional[str]) -> Optional[str]:
    # "14.3.123.456" -> "14.3"
    if not game_version or not isinstance(game_version, str):
        return None
    parts = game_version.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else game_version

def parse_patch_mm(game_version: Optional[str]):
    if not game_version or not isinstance(game_version, str):
        return None
    try:
        parts = game_version.split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return None

def is_patch_at_least(mm, min_major: int, min_minor: int) -> bool:
    if mm is None:
        return False
    major, minor = mm
    return (major > min_major) or (major == min_major and minor >= min_minor)

def get_high_elo_players() -> List[Dict]:
    """Muy parecido a tu crawler: challenger/gm/master mezclados."""
    queue_type = "RANKED_SOLO_5x5"
    tiers = []

    for name, func in [
        ("CHALLENGER", watcher.league.challenger_by_queue),
        ("GRANDMASTER", watcher.league.grandmaster_by_queue),
        # ("MASTER", watcher.league.masters_by_queue),
    ]:
        try:
            league = riot_call_with_retry(func, MATCH_REGION, queue_type)
            entries = league.get("entries", []) or []
            for e in entries:
                e["tier"] = league.get("tier", name)
            random.shuffle(entries)
            tiers.append(entries)
            print(f"{name}: {len(entries)}")
        except ApiError as err:
            print(f"{name}: error {err}")
            tiers.append([])

    # round-robin para balancear
    balanced = []
    while any(tiers):
        for t in tiers:
            if t:
                balanced.append(t.pop())
    print(f"Players total: {len(balanced)}")
    return balanced


def get_match_ids_paged(puuid: str, count: int, queue: Optional[int]) -> List[str]:
    """Riot limita count<=100, paginamos con start."""
    out = []
    start = 0
    remaining = int(count)

    while remaining > 0:
        batch = 100 if remaining > 100 else remaining
        try:
            ids = riot_call_with_retry(
                watcher.match.matchlist_by_puuid,
                REGION,
                puuid,
                start=start,
                count=batch,
                queue=queue
            )
        except ApiError:
            break

        if not ids:
            break
        out.extend(ids)
        got = len(ids)
        start += got
        remaining -= got
        if got < batch:
            break
        time.sleep(0.15)
    return out


def collect_one_match(conn, match_id: str) -> bool:
    # Dedupe por estado
    st = state_get(conn, match_id)
    if st and st[2] == "OK":
        return None

    try:
        match = riot_call_with_retry(watcher.match.by_id, REGION, match_id)

        info = (match or {}).get("info", {}) or {}
        mm = parse_patch_mm(info.get("gameVersion"))
        if not is_patch_at_least(mm, MIN_PATCH_MAJOR, MIN_PATCH_MINOR):
            # no descargamos timeline, no guardamos en disco
            state_set(conn, match_id, REGION, "SKIP_PATCH", f"patch<{MIN_PATCH_MAJOR}.{MIN_PATCH_MINOR}")
            return None

        timeline = riot_call_with_retry(watcher.match.timeline_by_match, REGION, match_id)
    except ApiError as err:
        code = getattr(err.response, "status_code", None)
        # 404/403/etc: marcamos FAIL para no insistir infinito
        state_set(conn, match_id, REGION, "FAIL", f"ApiError status={code}")
        return None
    except Exception as e:
        state_set(conn, match_id, REGION, "FAIL", f"Exception {type(e).__name__}: {e}")
        return None

    # Guardar RAW
    save_raw_match(RAW_ROOT, REGION, match_id, match, timeline)
    state_set(conn, match_id, REGION, "OK", None)

    # Index row (solo meta estable, para filtrar sin re-leer JSON)
    info = (match or {}).get("info", {}) or {}
    row = {
        "matchId": match_id,
        "routingRegion": REGION,
        "platformRegion": info.get("platformId"),
        "queueId": info.get("queueId"),
        "gameVersion": info.get("gameVersion"),
        "gameDuration": info.get("gameDuration"),
        "gameCreation": info.get("gameCreation"),
        "gameEndTimestamp": info.get("gameEndTimestamp"),
        "gameMode": info.get("gameMode"),
        "mapId": info.get("mapId"),
        "patchMajorMinor": patch_major_minor(info.get("gameVersion")),
        "sourceTier": None,  # lo rellenamos desde el loop de jugadores
    }
    # OJO: lo añadimos desde fuera porque depende del jugador origen
    return row


def run():
    os.makedirs("Data_clean", exist_ok=True)
    conn = open_state_db(STATE_DB)

    print("=== INICIO COLECCIÓN RAW ===")

    print("Parche mínimo:", f"{MIN_PATCH_MAJOR}.{MIN_PATCH_MINOR}\n")

    players = get_high_elo_players()
    total_ok = 0
    buffered_rows = []
    buffered_n = 0

    # Si quieres reanudar conteo real:
    try:
        if os.path.exists(INDEX_CSV):
            df = pd.read_csv(INDEX_CSV, usecols=["matchId"])
            total_ok = int(df["matchId"].nunique())
            print(f"Reanudando: {total_ok} matches ya indexados.")
    except Exception:
        pass

    for i, entry in enumerate(players):
        if total_ok >= TOTAL_MATCHES_TARGET:
            break

        puuid = entry.get("puuid")
        tier = entry.get("tier")

        if not puuid:
            continue

        print(f"\n[{i+1}/{len(players)}] tier={tier} total_ok={total_ok}")
        ids = get_match_ids_paged(puuid, MATCHES_PER_PLAYER, queue=QUEUE_RANKED_SOLO)
        if not ids:
            continue

        for mid in ids:
            if total_ok >= TOTAL_MATCHES_TARGET:
                break

            got = collect_one_match(conn, mid)
            if not got:
                continue

            got["sourceTier"] = tier
            buffered_rows.append(got)
            buffered_n += 1
            total_ok += 1
            print(f"  OK {mid} (total_ok={total_ok})")

            # flush index cada X
            if buffered_n >= SAVE_EVERY_N:
                for r in buffered_rows:
                    append_index_row(INDEX_CSV, r, INDEX_FIELDS)
                buffered_rows = []
                buffered_n = 0

            time.sleep(0.25)

        # flush al acabar jugador
        if buffered_rows:
            for r in buffered_rows:
                append_index_row(INDEX_CSV, r, INDEX_FIELDS)
            buffered_rows = []
            buffered_n = 0

    conn.close()
    print("Done.")


if __name__ == "__main__":
    run()