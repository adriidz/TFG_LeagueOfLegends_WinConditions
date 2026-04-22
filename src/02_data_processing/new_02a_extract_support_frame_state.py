#!/usr/bin/env python3
"""
Extract a reusable support frame-state table from raw match/timeline data.

Purpose
-------
Read each raw match only once and store a per-frame support/ADC state that can be
rescored many times later with different windows, start_minutes, thresholds, and
formula weights.

Output
------
A parquet with one row per (match_id, team_id, frame_idx) containing positional,
zone, and resource state for support and ADC.

Typical usage
-------------
python new_02a_extract_support_frame_state.py \
  --raw-root data/raw/raw \
  --region europe \
  --sample-frac 0.05 \
  --outdir data_new/clean/frame_state
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from shared_utils import (
    BASE_ZONES,
    BOT_SIDE_ZONES,
    BLUE_TEAM_ID,
    RED_TEAM_ID,
    DEFAULT_MIN_DURATION_MINUTES,
    classify_map_zone,
    extract_team_role_map,
    game_duration_minutes,
    get_match_id,
    get_match_info,
    get_participant_frame,
    get_timeline_frames,
    get_target_frac,
    infer_patch,
    list_match_dirs,
    load_json,
    participant_is_alive,
    extract_position,
    participant_lookup,
    safe_game_duration_seconds,
    side_from_team_id,
)

DEFAULT_RAW_ROOT = os.path.join("data", "raw", "raw")
DEFAULT_REGION = "europe"
DEFAULT_OUT_DIR = os.path.join("data_new", "clean", "frame_state")
DEFAULT_OUT_NAME = "support_frame_state"

JOIN_KEYS = ["match_id", "team_id", "frame_idx"]


def format_sample_suffix(sample_frac: Optional[float]) -> str:
    if sample_frac is None or sample_frac <= 0.0 or sample_frac >= 1.0:
        return ""
    return f"_sample{int(round(sample_frac * 100))}"


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_distance(a: Optional[tuple], b: Optional[tuple]) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        return float(math.hypot(a[0] - b[0], a[1] - b[1]))
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract support frame-state parquet from raw matches.")
    p.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--outdir", default=DEFAULT_OUT_DIR)
    p.add_argument("--out-name", default=DEFAULT_OUT_NAME)
    p.add_argument("--min-duration-minutes", type=float, default=DEFAULT_MIN_DURATION_MINUTES)
    p.add_argument("--max-matches", type=int, default=0)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--shuffle-match-dirs", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-only-post-minute", type=float, default=None,
                   help="Optional: only keep frames from this minute onward in the cached parquet.")
    return p.parse_args()


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
    suffix = format_sample_suffix(target_frac)
    if target_frac is not None and 0.0 < target_frac < 1.0:
        limit = max(1, int(len(match_dirs) * target_frac))
        match_dirs = match_dirs[:limit]
        print(f"Muestreo ({target_frac}): {limit} partidas.")

    if args.max_matches and args.max_matches > 0:
        match_dirs = match_dirs[:args.max_matches]
        print(f"Limitado a: {len(match_dirs)} partidas.")

    out_path = os.path.join(args.outdir, f"{args.out_name}{suffix}.parquet")
    ensure_dir(args.outdir)
    print(f"\n[Rutas] RAW: {os.path.abspath(raw_base)}")
    print(f"[Rutas] Output parquet: {os.path.abspath(out_path)}")

    rows: List[dict] = []
    total_seen = kept_matches = 0
    short = bad_match = bad_tl = missing_info = bad_roles = 0
    t0 = time.time()

    for mdir in match_dirs:
        total_seen += 1
        if total_seen % 1000 == 0:
            elapsed = time.time() - t0
            rate = total_seen / elapsed if elapsed > 0 else 0.0
            print(f"[{total_seen}/{len(match_dirs)}] kept_matches={kept_matches} rows={len(rows)} rate={rate:.1f}/s")

        match_path = os.path.join(mdir, "match.json")
        tl_path = os.path.join(mdir, "timeline.json")
        try:
            match = load_json(match_path)
        except Exception:
            bad_match += 1
            continue
        try:
            timeline = load_json(tl_path)
        except Exception:
            bad_tl += 1
            continue

        info = get_match_info(match)
        if not info:
            missing_info += 1
            continue

        dur = game_duration_minutes(info)
        if dur is None or dur < args.min_duration_minutes:
            short += 1
            continue

        role_map = extract_team_role_map(info)
        if not (BLUE_TEAM_ID in role_map and RED_TEAM_ID in role_map):
            bad_roles += 1
            continue

        frames = get_timeline_frames(timeline)
        if not frames:
            continue

        kept_matches += 1
        match_id = get_match_id(match, mdir)
        p_lookup = participant_lookup(info)
        patch = infer_patch(info.get("gameVersion"))
        game_ts = info.get("gameStartTimestamp") or info.get("gameCreation")
        platform_id = str(info.get("platformId")) if info.get("platformId") else None
        queue_id = info.get("queueId") if isinstance(info.get("queueId"), int) else None
        game_duration_seconds = safe_game_duration_seconds(info)

        for team_id in (BLUE_TEAM_ID, RED_TEAM_ID):
            rm = role_map.get(team_id)
            if not rm:
                continue
            support_pid = rm["UTILITY"]
            adc_pid = rm["BOTTOM"]
            support_meta = p_lookup.get(support_pid, {})
            adc_meta = p_lookup.get(adc_pid, {})
            side = side_from_team_id(team_id)

            for frame_idx, frame in enumerate(frames):
                minute = float(frame.get("timestamp", 0)) / 60000.0
                if args.keep_only_post_minute is not None and minute < args.keep_only_post_minute:
                    continue

                supp_pf = get_participant_frame(frame, support_pid)
                adc_pf = get_participant_frame(frame, adc_pid)

                supp_alive = bool(participant_is_alive(supp_pf))
                adc_alive = bool(participant_is_alive(adc_pf))

                supp_pos = extract_position(supp_pf) if supp_pf is not None else None
                adc_pos = extract_position(adc_pf) if adc_pf is not None else None

                supp_zone = classify_map_zone(supp_pos[0], supp_pos[1], team_id) if supp_pos is not None else None
                adc_zone = classify_map_zone(adc_pos[0], adc_pos[1], team_id) if adc_pos is not None else None

                dist_to_adc = safe_distance(supp_pos, adc_pos)

                rows.append({
                    "match_id": match_id,
                    "team_id": team_id,
                    "side": side,
                    "patch": patch,
                    "game_start_timestamp": game_ts,
                    "platform_id": platform_id,
                    "queue_id": queue_id,
                    "game_duration_seconds": game_duration_seconds,
                    "frame_idx": frame_idx,
                    "minute": minute,
                    "support_participant_id": support_pid,
                    "support_champion_id": support_meta.get("championId"),
                    "support_champion_name": support_meta.get("championName"),
                    "adc_participant_id": adc_pid,
                    "adc_champion_id": adc_meta.get("championId"),
                    "adc_champion_name": adc_meta.get("championName"),
                    "support_alive": supp_alive,
                    "adc_alive": adc_alive,
                    "support_x": safe_float(supp_pos[0]) if supp_pos is not None else None,
                    "support_y": safe_float(supp_pos[1]) if supp_pos is not None else None,
                    "adc_x": safe_float(adc_pos[0]) if adc_pos is not None else None,
                    "adc_y": safe_float(adc_pos[1]) if adc_pos is not None else None,
                    "support_zone": supp_zone,
                    "adc_zone": adc_zone,
                    "support_in_base": bool(supp_zone in BASE_ZONES) if supp_zone is not None else None,
                    "adc_in_base": bool(adc_zone in BASE_ZONES) if adc_zone is not None else None,
                    "support_in_bot_extended": bool((supp_zone in BOT_SIDE_ZONES) or (supp_zone == "RIVER_BOT")) if supp_zone is not None else None,
                    "dist_to_adc": dist_to_adc,
                    "support_xp": safe_float(supp_pf.get("xp")) if supp_pf else None,
                    "adc_xp": safe_float(adc_pf.get("xp")) if adc_pf else None,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No se han generado filas para support_frame_state.")

    dup_mask = df.duplicated(subset=JOIN_KEYS, keep=False)
    if dup_mask.any():
        preview = df.loc[dup_mask, JOIN_KEYS].head(10)
        raise SystemExit(
            "Duplicados por ['match_id', 'team_id', 'frame_idx'] en support_frame_state. "
            f"Primeros ejemplos:\n{preview.to_string(index=False)}"
        )

    df = df.sort_values(JOIN_KEYS).reset_index(drop=True)
    df.to_parquet(out_path, index=False)

    elapsed = time.time() - t0
    print(f"\nHecho en {elapsed:.1f}s")
    print(f"Matches procesados: {total_seen}")
    print(f"Matches kept: {kept_matches}")
    print(f"Rows frame-state: {len(df)}")
    print(f"bad_match={bad_match} | bad_tl={bad_tl} | missing_info={missing_info} | short={short} | bad_roles={bad_roles}")
    print(f"Parquet guardado en: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
