import csv
import os
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from dotenv import load_dotenv
from riotwatcher import LolWatcher, RiotWatcher, ApiError
import concurrent.futures

from raw_store import (
    RAW_ROOT_DEFAULT,
    INDEX_CSV_DEFAULT,
    STATE_DB_DEFAULT,
    open_state_db,
    state_get,
    state_set,
    save_raw_match,
    append_index_row,
)
from api import riot_call_with_retry, API_KEY, REGION, MATCH_REGION

load_dotenv("TFG.env")

RAW_ROOT = os.getenv("RAW_ROOT", RAW_ROOT_DEFAULT)
INDEX_CSV = os.getenv("RAW_INDEX_CSV", INDEX_CSV_DEFAULT)
STATE_DB = os.getenv("RAW_STATE_DB", STATE_DB_DEFAULT)

QUEUE_RANKED_SOLO = int(os.getenv("RAW_QUEUE_ID", "420"))
MATCHES_PER_PLAYER = int(os.getenv("RAW_MATCHES_PER_PLAYER", "300"))
TOTAL_MATCHES_TARGET = int(os.getenv("RAW_TOTAL_MATCHES_TARGET", "200000"))
MAX_WORKERS = int(os.getenv("RAW_MAX_WORKERS", "8"))
SAVE_EVERY_N = int(os.getenv("RAW_SAVE_EVERY_N", "50"))
REQUEUE_STALE_MINUTES = int(os.getenv("RAW_REQUEUE_STALE_MINUTES", "120"))

MIN_PATCH_MAJOR = 16
MIN_PATCH_MINOR = 2

TERMINAL_STATUSES = {"OK", "SKIP_PATCH", "FAIL"}
IN_PROGRESS_STATUS = "IN_PROGRESS"

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
    if not game_version or not isinstance(game_version, str):
        return None
    parts = game_version.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else game_version


def parse_patch_mm(game_version: Optional[str]) -> Optional[Tuple[int, int]]:
    if not game_version or not isinstance(game_version, str):
        return None
    try:
        parts = game_version.split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def is_patch_at_least(mm: Optional[Tuple[int, int]], min_major: int, min_minor: int) -> bool:
    if mm is None:
        return False
    major, minor = mm
    return (major > min_major) or (major == min_major and minor >= min_minor)


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def raw_files_exist(raw_root: str, routing_region: str, match_id: str) -> bool:
    match_path = os.path.join(raw_root, routing_region, match_id, "match.json")
    timeline_path = os.path.join(raw_root, routing_region, match_id, "timeline.json")
    return os.path.exists(match_path) and os.path.exists(timeline_path)


def load_indexed_ids(index_csv: str) -> Set[str]:
    if not os.path.exists(index_csv):
        return set()

    ids: Set[str] = set()
    with open(index_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            match_id = row.get("matchId")
            if match_id:
                ids.add(match_id)
    return ids


def flush_success_buffer(
    conn,
    buffered_rows: List[Dict[str, Any]],
    buffered_ok_ids: List[str],
    indexed_ids: Set[str],
) -> int:
    if not buffered_rows and not buffered_ok_ids:
        return 0

    new_ok = 0

    for row in buffered_rows:
        match_id = row["matchId"]
        if match_id in indexed_ids:
            continue
        append_index_row(INDEX_CSV, row, INDEX_FIELDS)
        indexed_ids.add(match_id)
        new_ok += 1

    for match_id in buffered_ok_ids:
        state_set(conn, match_id, REGION, "OK", None)

    buffered_rows.clear()
    buffered_ok_ids.clear()
    return new_ok


def try_claim_match(conn, match_id: str, indexed_ids: Set[str]) -> bool:
    st = state_get(conn, match_id)
    if st:
        status = st[2]
        updated_at = parse_iso_datetime(st[4])

        if status == "OK":
            return False

        if status in {"SKIP_PATCH", "FAIL"}:
            return False

        if status == IN_PROGRESS_STATUS:
            if match_id in indexed_ids and raw_files_exist(RAW_ROOT, REGION, match_id):
                state_set(conn, match_id, REGION, "OK", None)
                return False

            if updated_at is not None:
                stale_after = datetime.utcnow() - timedelta(minutes=REQUEUE_STALE_MINUTES)
                if updated_at >= stale_after:
                    return False
        # Si está en un estado desconocido o el IN_PROGRESS está stale, lo reclamamos.

    state_set(conn, match_id, REGION, IN_PROGRESS_STATUS, None)
    return True


def get_high_elo_players() -> List[Dict[str, Any]]:
    """Muy parecido a tu crawler: challenger/gm/master mezclados."""
    queue_type = "RANKED_SOLO_5x5"
    tiers: List[List[Dict[str, Any]]] = []

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

    balanced: List[Dict[str, Any]] = []
    while any(tiers):
        for tier_entries in tiers:
            if tier_entries:
                balanced.append(tier_entries.pop())
    print(f"Players total: {len(balanced)}")
    return balanced


def get_match_ids_paged(puuid: str, count: int, queue: Optional[int]) -> List[str]:
    """Riot limita count<=100, paginamos con start."""
    out: List[str] = []
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
                queue=queue,
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


def iter_unique(values: List[str]) -> Iterator[str]:
    seen: Set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        yield value


def fetch_one_match(match_id: str) -> Dict[str, Any]:
    try:
        match = riot_call_with_retry(watcher.match.by_id, REGION, match_id)
        info = (match or {}).get("info", {}) or {}

        mm = parse_patch_mm(info.get("gameVersion"))
        if not is_patch_at_least(mm, MIN_PATCH_MAJOR, MIN_PATCH_MINOR):
            return {
                "status": "SKIP_PATCH",
                "match_id": match_id,
                "error": f"patch<{MIN_PATCH_MAJOR}.{MIN_PATCH_MINOR}",
            }

        timeline = riot_call_with_retry(watcher.match.timeline_by_match, REGION, match_id)
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
            "sourceTier": None,
        }
        return {
            "status": "SUCCESS",
            "match_id": match_id,
            "match": match,
            "timeline": timeline,
            "row": row,
        }
    except ApiError as err:
        code = getattr(err.response, "status_code", None)
        return {
            "status": "FAIL",
            "match_id": match_id,
            "error": f"ApiError status={code}",
        }
    except Exception as exc:
        return {
            "status": "FAIL",
            "match_id": match_id,
            "error": f"Exception {type(exc).__name__}: {exc}",
        }


def process_completed_future(
    conn,
    future: concurrent.futures.Future,
    match_id: str,
    tier: Optional[str],
    indexed_ids: Set[str],
    buffered_rows: List[Dict[str, Any]],
    buffered_ok_ids: List[str],
) -> int:
    try:
        result = future.result()
    except Exception as exc:
        state_set(conn, match_id, REGION, "FAIL", f"Worker exception: {type(exc).__name__}: {exc}")
        return 0

    status = result.get("status")

    if status == "SKIP_PATCH":
        state_set(conn, match_id, REGION, "SKIP_PATCH", result.get("error"))
        return 0

    if status == "FAIL":
        state_set(conn, match_id, REGION, "FAIL", result.get("error"))
        return 0

    if status != "SUCCESS":
        state_set(conn, match_id, REGION, "FAIL", f"Unknown worker status: {status}")
        return 0

    try:
        save_raw_match(
            RAW_ROOT,
            REGION,
            match_id,
            result["match"],
            result["timeline"],
        )
    except Exception as exc:
        state_set(conn, match_id, REGION, "FAIL", f"Save raw failed: {type(exc).__name__}: {exc}")
        return 0

    if match_id in indexed_ids:
        state_set(conn, match_id, REGION, "OK", None)
        return 0

    row = result["row"]
    row["sourceTier"] = tier
    buffered_rows.append(row)
    buffered_ok_ids.append(match_id)
    print(f"  OK {match_id} (pending_flush={len(buffered_ok_ids)})")

    if len(buffered_ok_ids) >= SAVE_EVERY_N:
        return flush_success_buffer(conn, buffered_rows, buffered_ok_ids, indexed_ids)
    return 0


def run() -> None:
    os.makedirs("Data_clean", exist_ok=True)
    conn = open_state_db(STATE_DB)

    print("=== INICIO COLECCIÓN RAW ===")
    print("Parche mínimo:", f"{MIN_PATCH_MAJOR}.{MIN_PATCH_MINOR}\n")

    indexed_ids = load_indexed_ids(INDEX_CSV)
    total_ok = len(indexed_ids)
    if total_ok:
        print(f"Reanudando: {total_ok} matches ya indexados.")

    players = get_high_elo_players()
    buffered_rows: List[Dict[str, Any]] = []
    buffered_ok_ids: List[str] = []

    try:
        for i, entry in enumerate(players):
            if total_ok >= TOTAL_MATCHES_TARGET:
                break

            puuid = entry.get("puuid")
            tier = entry.get("tier")
            if not puuid:
                continue

            print(f"\n[{i + 1}/{len(players)}] tier={tier} total_ok={total_ok}")
            ids = list(iter_unique(get_match_ids_paged(puuid, MATCHES_PER_PLAYER, queue=QUEUE_RANKED_SOLO)))
            if not ids:
                continue

            ids_iter = iter(ids)
            futures: Dict[concurrent.futures.Future, Tuple[str, Optional[str]]] = {}
            no_more_ids = False

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                while futures or not no_more_ids:
                    while (
                        not no_more_ids
                        and len(futures) < MAX_WORKERS
                        and (total_ok + len(buffered_ok_ids) + len(futures)) < TOTAL_MATCHES_TARGET
                    ):
                        try:
                            match_id = next(ids_iter)
                        except StopIteration:
                            no_more_ids = True
                            break

                        if not try_claim_match(conn, match_id, indexed_ids):
                            continue

                        future = executor.submit(fetch_one_match, match_id)
                        futures[future] = (match_id, tier)

                    if not futures:
                        break

                    done, _ = concurrent.futures.wait(
                        futures,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    for future in done:
                        match_id, source_tier = futures.pop(future)
                        total_ok += process_completed_future(
                            conn,
                            future,
                            match_id,
                            source_tier,
                            indexed_ids,
                            buffered_rows,
                            buffered_ok_ids,
                        )

                # Vacía lo que haya quedado de este jugador antes de seguir.
                total_ok += flush_success_buffer(conn, buffered_rows, buffered_ok_ids, indexed_ids)

            print(f"  Cerrado bloque jugador. total_ok={total_ok}")

    finally:
        total_ok += flush_success_buffer(conn, buffered_rows, buffered_ok_ids, indexed_ids)
        conn.close()

    print(f"Done. total_ok={total_ok}")


if __name__ == "__main__":
    run()
