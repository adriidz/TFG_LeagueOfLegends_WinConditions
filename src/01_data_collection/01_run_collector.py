#!/usr/bin/env python3
"""
raw_collector_hardened.py

Recolector raw de partidas con estado persistente y endurecimientos adicionales:
- FAIL reintetable hasta un máximo configurable, en vez de terminal inmediato;
- opción para incluir MASTER en el pool inicial;
- flush del índice más robusto ante errores parciales de escritura;
- utilidades para reintento de IN_PROGRESS stale y estados FAIL antiguos.

Mantiene la misma interfaz de imports que el script previo:
- raw_store
- api
"""

import csv
import os
import random
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import concurrent.futures
from dotenv import load_dotenv
from riotwatcher import ApiError, LolWatcher, RiotWatcher

from raw_store import (
    INDEX_CSV_DEFAULT,
    RAW_ROOT_DEFAULT,
    STATE_DB_DEFAULT,
    append_index_row,
    open_state_db,
    raw_files_exist,
    save_raw_match,
    state_get,
    state_set,
)
from api import API_KEY, MATCH_REGION, REGION, riot_call_with_retry, watcher, account_watcher

load_dotenv("TFG.env")

# Variables configurables vía TFG.env
TOTAL_MATCHES_TARGET = int(os.getenv("RAW_TOTAL_MATCHES_TARGET", "200000"))
MAX_WORKERS = int(os.getenv("RAW_MAX_WORKERS", "8"))
# Por defecto True: Master aporta diversidad de jugadores esencial para reducir autocorrelación
INCLUDE_MASTER = os.getenv("RAW_INCLUDE_MASTER", "1").strip().lower() not in {"0", "false", "no"}
MIN_PATCH_MAJOR = int(os.getenv("RAW_MIN_PATCH_MAJOR", "16"))
MIN_PATCH_MINOR = int(os.getenv("RAW_MIN_PATCH_MINOR", "2"))
# Partidas por jugador: reducido a 40 para maximizar diversidad de jugadores únicos.
# Con 300 partidas/jugador y ~850 Challengers, los mismos jugadores se repiten constantemente
# entre partidas → autocorrelación alta. Con 40, el pool de jugadores únicos es mucho mayor.
MATCHES_PER_PLAYER = int(os.getenv("RAW_MATCHES_PER_PLAYER", "20"))
MATCHLIST_START_OFFSET = int(os.getenv("RAW_MATCHLIST_START_OFFSET", "0"))
MATCHLIST_RANDOMIZE_OFFSET = os.getenv("RAW_MATCHLIST_RANDOMIZE_OFFSET", "0").strip().lower() not in {"0", "false", "no"}
MATCHLIST_MAX_OFFSET = int(os.getenv("RAW_MATCHLIST_MAX_OFFSET", "0"))
# Pesos por tier para controlar proporciones en el dataset final.
# Peso 0 = excluir ese tier completamente (no se llama a su API).
# Default actual: solo Masters, para corregir la sobrerepresentación de GM y Challenger
# ya existente (~75k GM y ~16k Chall vs ~41k Master en las primeras 152k partidas).
# Cambiar en TFG.env: RAW_TIER_WEIGHTS=MASTER:15,GRANDMASTER:5,CHALLENGER:1
_tier_weights_raw = os.getenv("RAW_TIER_WEIGHTS", "MASTER:8,GRANDMASTER:3,CHALLENGER:1")
TIER_WEIGHTS: Dict[str, int] = {}
for _part in _tier_weights_raw.split(","):
    _kv = _part.strip().split(":")
    if len(_kv) == 2:
        try:
            TIER_WEIGHTS[_kv[0].strip().upper()] = max(0, int(_kv[1].strip()))  # 0 permitido
        except ValueError:
            pass
if not TIER_WEIGHTS:
    TIER_WEIGHTS = {"MASTER": 1, "GRANDMASTER": 0, "CHALLENGER": 0}

# Constantes internas del recolector (no recomendadas modificar)
RAW_ROOT = RAW_ROOT_DEFAULT
INDEX_CSV = INDEX_CSV_DEFAULT
STATE_DB = STATE_DB_DEFAULT

QUEUE_RANKED_SOLO = 420
SAVE_EVERY_N = 50
REQUEUE_STALE_MINUTES = 10
MAX_FAIL_RETRIES = 3

TERMINAL_STATUSES = {"OK", "SKIP_PATCH"}
IN_PROGRESS_STATUS = "IN_PROGRESS"
FAIL_STATUS = "FAIL"

# watcher y account_watcher importados de api.py (fuente única)

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

FAIL_ATTEMPT_RE = re.compile(r"(?:^|\|)attempt=(\d+)(?:\||$)")

def get_match_ids_paged(puuid: str, count: int, queue: Optional[int], start_offset: int = 0) -> List[str]:
    out: List[str] = []
    start = max(0, int(start_offset))
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


def explain_skip_reason(conn, match_id: str, indexed_ids: Set[str]) -> str:
    st = state_get(conn, match_id)
    if match_id in indexed_ids:
        return "already_indexed"

    if not st:
        return "not_claimable_unknown"

    status = state_status(st)
    updated_at = state_updated_at(st)
    error_text = state_error(st)

    if status == "OK":
        return "state_OK"
    if status == "SKIP_PATCH":
        return f"state_SKIP_PATCH ({error_text})"
    if status == IN_PROGRESS_STATUS:
        return f"state_IN_PROGRESS updated_at={updated_at}"
    if status == FAIL_STATUS:
        attempts = parse_fail_attempts(error_text)
        return f"state_FAIL attempts={attempts} error={error_text}"

    return f"state_{status} error={error_text}"


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


# raw_files_exist importado de raw_store.py


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


def state_status(st: Any) -> Optional[str]:
    try:
        return st[2]
    except Exception:
        return None


def state_error(st: Any) -> Optional[str]:
    try:
        return st[3]
    except Exception:
        return None


def state_updated_at(st: Any) -> Optional[datetime]:
    try:
        return parse_iso_datetime(st[4])
    except Exception:
        return None


def parse_fail_attempts(error_text: Optional[str]) -> int:
    if not error_text:
        return 0
    m = FAIL_ATTEMPT_RE.search(str(error_text))
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def format_fail_error(attempt: int, message: Optional[str]) -> str:
    clean = (message or "unknown_error").replace("\n", " ").strip()
    return f"attempt={attempt}|{clean}"


def flush_success_buffer(
    conn,
    buffered_rows: List[Dict[str, Any]],
    buffered_ok_ids: List[str],
    indexed_ids: Set[str],
) -> int:
    if not buffered_rows and not buffered_ok_ids:
        return 0

    new_ok = 0
    remaining_rows: List[Dict[str, Any]] = []
    remaining_ok_ids: List[str] = []

    for row, match_id in zip(buffered_rows, buffered_ok_ids):
        if match_id in indexed_ids:
            state_set(conn, match_id, REGION, "OK", None)
            continue
        try:
            append_index_row(INDEX_CSV, row, INDEX_FIELDS)
            indexed_ids.add(match_id)
            state_set(conn, match_id, REGION, "OK", None)
            new_ok += 1
        except Exception as exc:
            remaining_rows.append(row)
            remaining_ok_ids.append(match_id)
            state_set(conn, match_id, REGION, IN_PROGRESS_STATUS, f"flush_pending|{type(exc).__name__}: {exc}")

    buffered_rows[:] = remaining_rows
    buffered_ok_ids[:] = remaining_ok_ids
    return new_ok


def try_claim_match(conn, match_id: str, indexed_ids: Set[str]) -> bool:
    """Reclama una partida para procesamiento con transacción atómica.

    Usa BEGIN IMMEDIATE para obtener un write-lock antes de leer,
    evitando race conditions con múltiples workers.
    """
    try:
        conn.execute("BEGIN IMMEDIATE")
    except Exception:
        return False
    try:
        st = state_get(conn, match_id)
        if st:
            status = state_status(st)
            updated_at = state_updated_at(st)
            error_text = state_error(st)

            if status == "OK":
                conn.execute("COMMIT")
                return False
            if status == "SKIP_PATCH":
                conn.execute("COMMIT")
                return False
            if status == IN_PROGRESS_STATUS:
                if match_id in indexed_ids and raw_files_exist(RAW_ROOT, REGION, match_id):
                    state_set(conn, match_id, REGION, "OK", None, commit=False)
                    conn.execute("COMMIT")
                    return False
                if updated_at is not None:
                    stale_after = datetime.utcnow() - timedelta(minutes=REQUEUE_STALE_MINUTES)
                    if updated_at >= stale_after:
                        conn.execute("COMMIT")
                        return False
            elif status == FAIL_STATUS:
                attempts = parse_fail_attempts(error_text)
                if attempts >= MAX_FAIL_RETRIES:
                    conn.execute("COMMIT")
                    return False
                if updated_at is not None:
                    stale_after = datetime.utcnow() - timedelta(minutes=REQUEUE_STALE_MINUTES)
                    if updated_at >= stale_after:
                        conn.execute("COMMIT")
                        return False

        state_set(conn, match_id, REGION, IN_PROGRESS_STATUS, None, commit=False)
        conn.execute("COMMIT")
        return True
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        return False


def get_high_elo_players() -> List[Dict[str, Any]]:
    """
    Recoge jugadores de cada tier y los mezcla respetando TIER_WEIGHTS.

    En lugar de un intercalado 1:1 (que ignora el tamaño del pool de cada tier),
    se crea una lista ponderada: por cada jugador de Challenger se añaden
    TIER_WEIGHTS[MASTER] jugadores de Master y TIER_WEIGHTS[GRANDMASTER] de GM.
    Esto garantiza que la proporción de partidas por tier en el dataset refleje
    los pesos configurados, reduciendo la autocorrelación entre partidas.
    """
    queue_type = "RANKED_SOLO_5x5"
    tier_entries_map: Dict[str, List[Dict[str, Any]]] = {}

    sources = [
        ("CHALLENGER", watcher.league.challenger_by_queue),
        ("GRANDMASTER", watcher.league.grandmaster_by_queue),
    ]
    if INCLUDE_MASTER:
        sources.append(("MASTER", watcher.league.masters_by_queue))

    for name, func in sources:
        weight = TIER_WEIGHTS.get(name, 1)
        if weight == 0:
            print(f"{name}: EXCLUIDO (peso=0)")
            tier_entries_map[name] = []
            continue
        try:
            league = riot_call_with_retry(func, MATCH_REGION, queue_type)
            entries = league.get("entries", []) or []
            for e in entries:
                e["tier"] = league.get("tier", name)
            random.shuffle(entries)
            tier_entries_map[name] = entries
            print(f"{name}: {len(entries)} jugadores (peso={weight})")
        except ApiError as err:
            print(f"{name}: error {err}")
            tier_entries_map[name] = []

    # Construir lista ponderada: cada tier contribuye según su peso relativo.
    # Tiers con peso 0 ya tienen lista vacía, así que no aportan jugadores.
    weighted: List[Dict[str, Any]] = []
    queues = [
        (name, tier_entries_map.get(name, []), TIER_WEIGHTS.get(name, 0))
        for name, _ in sources
        if TIER_WEIGHTS.get(name, 0) > 0  # omitir tiers excluidos del bucle de mezcla
    ]
    exhausted = [False] * len(queues)
    indices = [0] * len(queues)

    while not all(exhausted):
        for i, (name, entries, weight) in enumerate(queues):
            if exhausted[i]:
                continue
            added = 0
            while added < weight and indices[i] < len(entries):
                weighted.append(entries[indices[i]])
                indices[i] += 1
                added += 1
            if indices[i] >= len(entries):
                exhausted[i] = True

    print(f"Players total en lista ponderada: {len(weighted)}")
    print(f"Pesos aplicados: { {n: w for n, _, w in queues} }")
    return weighted


# def get_match_ids_paged(puuid: str, count: int, queue: Optional[int]) -> List[str]:
#     out: List[str] = []
#     start = 0
#     remaining = int(count)

#     while remaining > 0:
#         batch = 100 if remaining > 100 else remaining
#         try:
#             ids = riot_call_with_retry(
#                 watcher.match.matchlist_by_puuid,
#                 REGION,
#                 puuid,
#                 start=start,
#                 count=batch,
#                 queue=queue,
#             )
#         except ApiError:
#             break

#         if not ids:
#             break
#         out.extend(ids)
#         got = len(ids)
#         start += got
#         remaining -= got
#         if got < batch:
#             break
#         time.sleep(0.15)
#     return out


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
        return {"status": FAIL_STATUS, "match_id": match_id, "error": f"ApiError status={code}"}
    except Exception as exc:
        return {"status": FAIL_STATUS, "match_id": match_id, "error": f"Exception {type(exc).__name__}: {exc}"}


def mark_fail(conn, match_id: str, message: str) -> None:
    st = state_get(conn, match_id)
    attempts = parse_fail_attempts(state_error(st)) + 1
    state_set(conn, match_id, REGION, FAIL_STATUS, format_fail_error(attempts, message))


def should_abandon_player(
    claimed_count: int,
    player_skip_patch_count: int,
    seen_results: int,
    window: int = 12,
    max_skip_patch_ratio: float = 0.8,
) -> bool:
    """Corta pronto jugadores cuyo tramo reciente es mayoritariamente antiguo.

    Evita gastar demasiadas llamadas en historiales donde el bloque inspeccionado
    cae fuera del parche objetivo.
    """
    if claimed_count < window or seen_results < window:
        return False
    return (player_skip_patch_count / max(1, seen_results)) >= max_skip_patch_ratio


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
        mark_fail(conn, match_id, f"Worker exception: {type(exc).__name__}: {exc}")
        return 0

    status = result.get("status")
    if status == "SKIP_PATCH":
        state_set(conn, match_id, REGION, "SKIP_PATCH", result.get("error"))
        return 0
    if status == FAIL_STATUS:
        mark_fail(conn, match_id, str(result.get("error")))
        return 0
    if status != "SUCCESS":
        mark_fail(conn, match_id, f"Unknown worker status: {status}")
        return 0

    try:
        save_raw_match(RAW_ROOT, REGION, match_id, result["match"], result["timeline"])
    except Exception as exc:
        mark_fail(conn, match_id, f"Save raw failed: {type(exc).__name__}: {exc}")
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
    os.makedirs("data/clean", exist_ok=True)
    conn = open_state_db(STATE_DB)

    print("=== INICIO COLECCIÓN RAW ===")
    print("Parche mínimo:", f"{MIN_PATCH_MAJOR}.{MIN_PATCH_MINOR}")
    print("Include MASTER:", INCLUDE_MASTER)
    print("Matches por jugador:", MATCHES_PER_PLAYER)
    print("Pesos por tier:", TIER_WEIGHTS)
    print("Max fail retries:", MAX_FAIL_RETRIES)
    print(f"[Rutas] Estado persistente de partidas db: {os.path.abspath(STATE_DB)}")
    print(f"[Rutas] Índice de partidas csv (Entrada/Salida): {os.path.abspath(INDEX_CSV)}")
    print(f"[Rutas] Guardando raw matches (JSON) en: {os.path.abspath(RAW_ROOT)}")
    print()

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

            start_offset = MATCHLIST_START_OFFSET
            if MATCHLIST_RANDOMIZE_OFFSET and MATCHLIST_MAX_OFFSET > 0:
                start_offset = random.randint(0, MATCHLIST_MAX_OFFSET)

            print(
                f"\n[{i + 1}/{len(players)}] tier={tier} "
                f"total_ok={total_ok} pending_flush={len(buffered_ok_ids)} "
                f"visible_total={total_ok + len(buffered_ok_ids)} start_offset={start_offset}"
            )

            ids = list(
                iter_unique(
                    get_match_ids_paged(
                        puuid,
                        MATCHES_PER_PLAYER,
                        queue=QUEUE_RANKED_SOLO,
                        start_offset=start_offset,
                    )
                )
            )

            # Heurística anti-solape: prioriza IDs que aún no están en índice/estado terminal
            # para gastar antes las llamadas en partidas potencialmente nuevas.
            ids = [
                mid for mid in ids
                if mid not in indexed_ids and state_status(state_get(conn, mid)) not in TERMINAL_STATUSES
            ]
            if not ids:
                print("  Sin ids para este jugador.")
                continue

            print(f"  IDs obtenidos: {len(ids)}")

            ids_iter = iter(ids)
            futures: Dict[concurrent.futures.Future, Tuple[str, Optional[str]]] = {}
            no_more_ids = False

            claimed_count = 0
            player_skip_patch_count = 0
            player_seen_results = 0
            skip_reasons: Dict[str, int] = {}

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
                            reason = explain_skip_reason(conn, match_id, indexed_ids)
                            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                            print(f"  SKIP {match_id} -> {reason}")
                            continue

                        claimed_count += 1
                        future = executor.submit(fetch_one_match, match_id)
                        futures[future] = (match_id, tier)

                    if not futures:
                        break

                    done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        match_id, source_tier = futures.pop(future)
                        result = future.result()
                        player_seen_results += 1
                        if result.get("status") == "SKIP_PATCH":
                            player_skip_patch_count += 1

                        class _ResolvedFuture:
                            def __init__(self, value):
                                self._value = value
                            def result(self):
                                return self._value

                        total_ok += process_completed_future(
                            conn,
                            _ResolvedFuture(result),
                            match_id,
                            source_tier,
                            indexed_ids,
                            buffered_rows,
                            buffered_ok_ids,
                        )

                        if should_abandon_player(claimed_count, player_skip_patch_count, player_seen_results):
                            print(
                                "  Corte temprano: demasiadas partidas antiguas en el tramo reciente "
                                f"({player_skip_patch_count}/{player_seen_results} SKIP_PATCH)."
                            )
                            no_more_ids = True

                total_ok += flush_success_buffer(conn, buffered_rows, buffered_ok_ids, indexed_ids)

            print(f"  Resumen jugador: claimed={claimed_count} total_ok={total_ok}")
            print(f"  Motivos skip: {skip_reasons}")
            print(f"  Cerrado bloque jugador. total_ok={total_ok}")
    finally:
        total_ok += flush_success_buffer(conn, buffered_rows, buffered_ok_ids, indexed_ids)
        conn.close()

    print(f"Done. total_ok={total_ok}")


if __name__ == "__main__":
    run()
