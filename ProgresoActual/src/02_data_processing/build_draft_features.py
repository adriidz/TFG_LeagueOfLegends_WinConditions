#!/usr/bin/env python3
"""
Build draft features for the ProgresoActual support-only pipeline.

This script reads only `match.json` files and writes pre-game/team-composition
features inside ProgresoActual. It intentionally does not build labels and does
not read timelines.

Output unit
-----------
One row per (match_id, team_id), from the perspective of that team:
- ally_* columns describe the selected team draft
- enemy_* columns describe the opposing team draft

Typical usage
-------------
python ProgresoActual/src/02_data_processing/build_draft_features.py \
  --raw-root data/raw/raw \
  --region europe \
  --sample-frac 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from shared_utils import (
    BLUE_TEAM_ID,
    RED_TEAM_ID,
    CANONICAL_ROLES,
    ROLE_KEYS_LOWER,
    DEFAULT_MIN_DURATION_MINUTES,
    extract_runes,
    extract_summoner_spells,
    extract_team_bans,
    extract_team_role_map,
    game_duration_minutes,
    get_match_id,
    get_match_info,
    get_target_frac,
    infer_patch,
    list_match_dirs,
    load_json,
    participant_lookup,
    safe_game_duration_seconds,
    side_from_team_id,
    validate_no_duplicate_keys,
)

DEFAULT_RAW_ROOT = os.path.join("data", "raw", "raw")
DEFAULT_REGION = "europe"
DEFAULT_OUT_DIR = os.path.join("ProgresoActual", "data", "clean", "features")
DEFAULT_OUT_NAME = "draft_features"


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def format_sample_suffix(sample_frac: Optional[float]) -> str:
    if sample_frac is None or sample_frac <= 0.0 or sample_frac >= 1.0:
        return ""
    return f"_sample{int(round(sample_frac * 100))}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ProgresoActual draft_features from match.json files.")
    p.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--outdir", default=DEFAULT_OUT_DIR)
    p.add_argument("--out-name", default=DEFAULT_OUT_NAME)
    p.add_argument("--min-duration-minutes", type=float, default=DEFAULT_MIN_DURATION_MINUTES)
    p.add_argument("--max-matches", type=int, default=0)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--shuffle-match-dirs", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def extract_draft_row(
    info: dict,
    match_id: str,
    team_id: int,
    role_map: Dict[int, Dict[str, int]],
    p_lookup: Dict[int, dict],
    bans: Dict[int, List[Optional[int]]],
) -> dict:
    enemy_id = RED_TEAM_ID if team_id == BLUE_TEAM_ID else BLUE_TEAM_ID
    own_roles = role_map[team_id]
    enemy_roles = role_map[enemy_id]
    game_version = info.get("gameVersion")

    row: dict = {
        "match_id": match_id,
        "team_id": team_id,
        "side": side_from_team_id(team_id),
        "patch": infer_patch(game_version),
        "game_version": game_version,
        "game_start_timestamp": info.get("gameStartTimestamp") or info.get("gameCreation"),
        "platform_id": str(info.get("platformId")) if info.get("platformId") else None,
        "queue_id": info.get("queueId") if isinstance(info.get("queueId"), int) else None,
        "game_duration_seconds": safe_game_duration_seconds(info),
    }

    for prefix, roles in (("ally", own_roles), ("enemy", enemy_roles)):
        for canonical_role, lower_role in zip(CANONICAL_ROLES, ROLE_KEYS_LOWER):
            participant_id = roles[canonical_role]
            participant = p_lookup.get(participant_id, {})
            row[f"{prefix}_{lower_role}_participant_id"] = participant_id
            row[f"{prefix}_{lower_role}_champion_id"] = participant.get("championId")
            row[f"{prefix}_{lower_role}_champion_name"] = participant.get("championName")
            summoner1_id, summoner2_id = extract_summoner_spells(participant)
            row[f"{prefix}_{lower_role}_summoner1_id"] = summoner1_id
            row[f"{prefix}_{lower_role}_summoner2_id"] = summoner2_id
            keystone_id, primary_style_id, sub_style_id = extract_runes(participant)
            row[f"{prefix}_{lower_role}_keystone_id"] = keystone_id
            row[f"{prefix}_{lower_role}_primary_style_id"] = primary_style_id
            row[f"{prefix}_{lower_role}_sub_style_id"] = sub_style_id

    own_bans = bans.get(team_id, [None] * 5)
    enemy_bans = bans.get(enemy_id, [None] * 5)
    for idx in range(5):
        row[f"ally_ban_{idx + 1}_champion_id"] = own_bans[idx] if idx < len(own_bans) else None
        row[f"enemy_ban_{idx + 1}_champion_id"] = enemy_bans[idx] if idx < len(enemy_bans) else None

    return row


def build_overall_summary(df: pd.DataFrame, counters: Dict[str, int], elapsed_seconds: float) -> pd.DataFrame:
    return pd.DataFrame([{
        **counters,
        "rows": int(len(df)),
        "unique_matches": int(df["match_id"].nunique()) if "match_id" in df.columns else 0,
        "unique_match_team_keys": int(df[["match_id", "team_id"]].drop_duplicates().shape[0]) if not df.empty else 0,
        "elapsed_seconds": float(elapsed_seconds),
        "rows_per_second": float(len(df) / elapsed_seconds) if elapsed_seconds > 0 else None,
    }])


def save_analysis_tables(df: pd.DataFrame, analysis_dir: str, counters: Dict[str, int], elapsed_seconds: float) -> None:
    ensure_dir(analysis_dir)
    build_overall_summary(df, counters, elapsed_seconds).to_csv(
        os.path.join(analysis_dir, "overall_summary.csv"),
        index=False,
    )
    if df.empty:
        return
    if "patch" in df.columns:
        patch_counts = df.groupby("patch", dropna=False).size().reset_index(name="n").sort_values("n", ascending=False)
        patch_counts.to_csv(os.path.join(analysis_dir, "patch_counts.csv"), index=False)
    if "side" in df.columns:
        side_counts = df.groupby("side", dropna=False).size().reset_index(name="n").sort_values("side")
        side_counts.to_csv(os.path.join(analysis_dir, "side_counts.csv"), index=False)
    support_counts = (
        df.groupby("ally_utility_champion_name", dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    support_counts.to_csv(os.path.join(analysis_dir, "support_champion_counts.csv"), index=False)


def main() -> None:
    args = parse_args()
    raw_base = os.path.join(args.raw_root, args.region)
    match_dirs = list_match_dirs(raw_base)
    print(f"Directorios de partida detectados: {len(match_dirs)}")

    if args.shuffle_match_dirs:
        rng = random.Random(args.seed)
        rng.shuffle(match_dirs)
        print(f"Partidas barajadas con seed={args.seed}.")

    target_frac = get_target_frac(args.sample_frac)
    if target_frac is not None and 0.0 < target_frac < 1.0:
        limit = max(1, int(len(match_dirs) * target_frac))
        match_dirs = match_dirs[:limit]
        print(f"Muestreo ({target_frac}): {limit} partidas.")

    if args.max_matches and args.max_matches > 0:
        match_dirs = match_dirs[:args.max_matches]
        print(f"Limitado a: {len(match_dirs)} partidas.")

    suffix = format_sample_suffix(target_frac)
    ensure_dir(args.outdir)
    out_path = os.path.join(args.outdir, f"{args.out_name}{suffix}.parquet")
    analysis_dir = os.path.splitext(out_path)[0] + "_analysis"

    print(f"\n[Rutas] RAW: {os.path.abspath(raw_base)}")
    print(f"[Rutas] Output parquet: {os.path.abspath(out_path)}")
    print(f"[Rutas] Analysis dir: {os.path.abspath(analysis_dir)}")

    rows: List[dict] = []
    counters = {
        "matches_seen": 0,
        "matches_kept": 0,
        "bad_match_json": 0,
        "missing_info": 0,
        "short_matches": 0,
        "bad_roles": 0,
        "missing_team_role": 0,
    }
    t0 = time.time()

    for match_dir in match_dirs:
        counters["matches_seen"] += 1
        if counters["matches_seen"] % 1000 == 0:
            elapsed = time.time() - t0
            rate = counters["matches_seen"] / elapsed if elapsed > 0 else 0.0
            print(
                f"[{counters['matches_seen']}/{len(match_dirs)}] "
                f"kept={counters['matches_kept']} rows={len(rows)} rate={rate:.1f}/s"
            )

        match_path = os.path.join(match_dir, "match.json")
        try:
            match = load_json(match_path)
        except (OSError, json.JSONDecodeError):
            counters["bad_match_json"] += 1
            continue

        info = get_match_info(match)
        if not info:
            counters["missing_info"] += 1
            continue

        duration_minutes = game_duration_minutes(info)
        if duration_minutes is None or duration_minutes < args.min_duration_minutes:
            counters["short_matches"] += 1
            continue

        role_map = extract_team_role_map(info)
        if not (BLUE_TEAM_ID in role_map and RED_TEAM_ID in role_map):
            counters["bad_roles"] += 1
            continue

        counters["matches_kept"] += 1
        match_id = get_match_id(match, match_dir)
        p_lookup = participant_lookup(info)
        bans = extract_team_bans(info)

        for team_id in (BLUE_TEAM_ID, RED_TEAM_ID):
            if team_id not in role_map:
                counters["missing_team_role"] += 1
                continue
            rows.append(extract_draft_row(info, match_id, team_id, role_map, p_lookup, bans))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No se han generado filas para draft_features.")

    validate_no_duplicate_keys(df)
    df = df.sort_values(["match_id", "team_id"]).reset_index(drop=True)
    df.to_parquet(out_path, index=False)

    elapsed_seconds = time.time() - t0
    save_analysis_tables(df, analysis_dir, counters, elapsed_seconds)

    print(f"\nHecho en {elapsed_seconds:.1f}s")
    print(f"Matches seen: {counters['matches_seen']}")
    print(f"Matches kept: {counters['matches_kept']}")
    print(f"Rows draft_features: {len(df)}")
    print(f"bad_match_json={counters['bad_match_json']} | missing_info={counters['missing_info']} | "
          f"short={counters['short_matches']} | bad_roles={counters['bad_roles']}")
    print(f"Parquet guardado en: {os.path.abspath(out_path)}")
    print(f"Analisis guardado en: {os.path.abspath(analysis_dir)}")


if __name__ == "__main__":
    main()
